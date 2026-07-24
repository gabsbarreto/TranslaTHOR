from __future__ import annotations

import json
import shutil
import sys
import time
import types
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    AVAILABLE_TRANSLATION_MODELS,
    AVAILABLE_OCR_ENGINES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EXTRACTION_MODE,
    DEFAULT_LLM_MIN_P,
    DEFAULT_LLM_PRESENCE_PENALTY,
    DEFAULT_LLM_REPETITION_PENALTY,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_TOP_P,
    DEFAULT_OUTPUT_MODE,
    DEFAULT_OCR_ENGINE,
    DEFAULT_QWEN_OCR_BASE_SIZE,
    DEFAULT_QWEN_OCR_BATCH_SIZE,
    DEFAULT_QWEN_OCR_CROP_MODE,
    DEFAULT_QWEN_OCR_DPI,
    DEFAULT_QWEN_OCR_IMAGE_SIZE,
    DEFAULT_QWEN_OCR_MAX_CROPS,
    DEFAULT_QWEN_OCR_MAX_TOKENS,
    DEFAULT_QWEN_OCR_MIN_CROPS,
    DEFAULT_QWEN_OCR_MIN_P,
    DEFAULT_QWEN_OCR_MODEL,
    DEFAULT_QWEN_OCR_NGRAM_SIZE,
    DEFAULT_QWEN_OCR_NGRAM_WINDOW,
    DEFAULT_QWEN_OCR_PRESENCE_PENALTY,
    DEFAULT_QWEN_OCR_PROMPT,
    DEFAULT_QWEN_OCR_REPETITION_PENALTY,
    DEFAULT_QWEN_OCR_SKIP_REPEAT,
    DEFAULT_QWEN_OCR_TEMPERATURE,
    DEFAULT_QWEN_OCR_TOP_K,
    DEFAULT_QWEN_OCR_TOP_P,
    DEFAULT_TRANSLATION_CHUNK_GROUP_SIZE,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_SURYA2_DPI,
    DEFAULT_SURYA2_STRATEGY,
    ENABLE_LOCAL_VLM_REPAIR,
    ENABLE_QWEN_OCR_FALLBACK,
    FRONTEND_DIR,
    KEEP_EXTRACTION_DEBUG_ARTIFACTS,
    MARKER_TIMEOUT_SECONDS,
)
from app.models.schema import JobQueueState, JobStage
from app.services.job_queue import JobQueue
from app.services.job_store import TERMINAL_JOB_STAGES, JobStore
from app.services.markdown_builder import MarkdownBuilder
from app.services.original_layout_reconstructor import OriginalLayoutReconstructor
from app.services.pipeline import TranslationPipeline
from app.services.reconstructor import Reconstructor
from app.utils.logging import configure_logging

configure_logging()

# Keep imports and test collection working in lightweight environments.
try:
    from python_multipart.multipart import parse_options_header as _parse_options_header  # type: ignore

    _ = _parse_options_header
except Exception:
    multipart_pkg = types.ModuleType("multipart")
    multipart_pkg.__dict__["__version__"] = "0.0"
    multipart_submodule = types.ModuleType("multipart.multipart")

    def parse_options_header(value: str) -> tuple[str, dict]:
        return value, {}

    multipart_submodule.parse_options_header = parse_options_header  # type: ignore[attr-defined]
    sys.modules.setdefault("multipart", multipart_pkg)
    sys.modules.setdefault("multipart.multipart", multipart_submodule)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    job_store.reconcile_stale_jobs()
    try:
        yield
    finally:
        pipeline.shutdown()


app = FastAPI(title="Local PDF Translation App", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

job_store = JobStore()
pipeline = TranslationPipeline(job_store)
job_queue = JobQueue(job_store, pipeline)
reconstructor = Reconstructor()
markdown_builder = MarkdownBuilder()
original_layout_reconstructor = OriginalLayoutReconstructor()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs(include_archived: bool = False) -> list[dict]:
    return [
        status.model_dump() for status in job_store.list_jobs(include_archived=include_archived)
    ]


@app.delete("/api/jobs")
def clear_jobs() -> dict[str, int]:
    job_queue.stop_all()
    return {"removed": job_store.clear_jobs()}


@app.delete("/api/jobs/cleanup-terminal")
def clear_terminal_jobs() -> dict[str, int]:
    removed = job_store.clear_jobs_by_stage({JobStage.CANCELLED, JobStage.FAILED})
    return {"removed": removed}


@app.post("/api/jobs/stop-all")
def stop_all_jobs() -> dict[str, int]:
    result = job_queue.stop_all()
    return {**result, "interrupted_cancelled": _mark_interrupted_processing_jobs_cancelled()}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    result = job_queue.cancel_job(job_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Job is not queued or active.")
    return result


@app.post("/api/jobs/{job_id}/archive")
def archive_job(job_id: str) -> dict:
    try:
        return job_store.archive_job(job_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/unarchive")
def unarchive_job(job_id: str) -> dict:
    try:
        return job_store.unarchive_job(job_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, int | str]:
    try:
        status = job_store.load_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if status.stage not in TERMINAL_JOB_STAGES:
        raise HTTPException(
            status_code=409,
            detail="Only completed, cancelled, or failed jobs can be permanently deleted.",
        )
    if job_queue.contains(job_id):
        raise HTTPException(
            status_code=409,
            detail="Job resources are still shutting down; try again shortly.",
        )
    if not job_store.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"removed": 1, "job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return job_store.load_status(job_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/api/jobs")
async def create_job(
    files: list[UploadFile] = File(...),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    model: str = Form(DEFAULT_TRANSLATION_MODEL),
    temperature: float = Form(DEFAULT_LLM_TEMPERATURE),
    top_p: float = Form(DEFAULT_LLM_TOP_P),
    top_k: int = Form(DEFAULT_LLM_TOP_K),
    min_p: float = Form(DEFAULT_LLM_MIN_P),
    presence_penalty: float = Form(DEFAULT_LLM_PRESENCE_PENALTY),
    repetition_penalty: float = Form(DEFAULT_LLM_REPETITION_PENALTY),
    max_tokens: int = Form(2048),
    output_mode: str = Form(DEFAULT_OUTPUT_MODE),
    profile_pipeline: bool = Form(False),
    extraction_mode: str = Form(DEFAULT_EXTRACTION_MODE),
    ocr_engine: str = Form(DEFAULT_OCR_ENGINE),
    use_local_vlm_repair: bool = Form(ENABLE_LOCAL_VLM_REPAIR),
    keep_debug_artifacts: bool = Form(KEEP_EXTRACTION_DEBUG_ARTIFACTS),
) -> dict:
    created: list[dict] = []
    for upload in files:
        settings = _build_job_settings(
            chunk_size=chunk_size,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            output_mode=output_mode,
            profile_pipeline=profile_pipeline,
            extraction_mode=extraction_mode,
            ocr_engine=ocr_engine,
            use_local_vlm_repair=use_local_vlm_repair,
            keep_debug_artifacts=keep_debug_artifacts,
        )
        job_id, job_dir = job_store.create_job(upload.filename, settings=settings)
        pdf_path = job_dir / "input.pdf"
        with pdf_path.open("wb") as file:
            shutil.copyfileobj(upload.file, file)
        job_queue.enqueue(
            job_id,
            pdf_path,
            settings,
        )
        created.append({"job_id": job_id, "filename": upload.filename})
    return {"jobs": created}


@app.get("/api/jobs/{job_id}/artifacts/{artifact_type}")
def get_artifact(job_id: str, artifact_type: str) -> FileResponse:
    try:
        status = job_store.load_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if artifact_type in {"pdf", "pdf_readable", "pdf_faithful", "pdf_original_layout"}:
        path = _ensure_pdf_artifact(job_id, artifact_type)
        return FileResponse(path, media_type="application/pdf", filename=path.name)
    if artifact_type == "source_markdown":
        path = _source_markdown_path(job_id)
        return FileResponse(path, media_type="text/markdown", filename=path.name)

    path_str = status.artifacts.get(artifact_type)
    if not path_str:
        raise HTTPException(status_code=404, detail="Artifact not available")
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    media_type = {
        "markdown": "text/markdown",
        "json": "application/json",
        "profile_json": "application/json",
        "profile_csv": "text/csv",
        "profile_summary": "text/plain",
        "extraction_result": "application/json",
        "marker_detection": "application/json",
        "logical_translation_chunks": "application/json",
        "reconstruction_report": "application/json",
    }.get(artifact_type, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/jobs/{job_id}/pdf/{mode}")
def get_pdf(job_id: str, mode: str) -> FileResponse:
    normalized_mode = mode.replace("-", "_")
    if normalized_mode not in {"readable", "faithful", "original_layout"}:
        raise HTTPException(status_code=400, detail="Unsupported PDF mode")
    path = _ensure_pdf_artifact(job_id, f"pdf_{normalized_mode}")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.get("/api/jobs/{job_id}/ocr-pdf/{mode}")
def get_ocr_pdf(job_id: str, mode: str) -> FileResponse:
    if mode not in {"readable", "faithful"}:
        raise HTTPException(status_code=400, detail="Unsupported PDF mode")
    path = _ensure_source_pdf_artifact(job_id, mode)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


def _mark_interrupted_processing_jobs_cancelled() -> int:
    processing = {
        JobStage.EXTRACTION,
        JobStage.OCR_LAYOUT,
        JobStage.STRUCTURE,
        JobStage.TRANSLATION,
        JobStage.PDF,
    }
    cancelled = 0
    for status in job_store.list_jobs():
        if status.stage not in processing:
            continue
        try:
            job_store.update_status(
                status.job_id,
                stage=JobStage.CANCELLED,
                progress=1.0,
                message="Cancelled by Stop All Processes.",
                error=None,
                queue_state=JobQueueState.NONE,
                queue_position=None,
                jobs_ahead=None,
                completed_at=job_store.utc_now(),
            )
            cancelled += 1
        except FileNotFoundError:
            continue
    return cancelled


def _ensure_pdf_artifact(job_id: str, artifact_type: str) -> Path:
    with job_store.artifact_generation_lock(job_id):
        return _ensure_pdf_artifact_locked(job_id, artifact_type)


def _ensure_pdf_artifact_locked(job_id: str, artifact_type: str) -> Path:
    # Read only after acquiring the generation lock. A preceding request may
    # have refreshed figure assets or registered another PDF mode while this
    # request was waiting.
    status = job_store.load_status(job_id)
    if status.stage != JobStage.COMPLETE:
        raise HTTPException(
            status_code=409,
            detail="Translated PDFs are available only after translation is complete.",
        )
    if artifact_type == "pdf_original_layout":
        mode = "original_layout"
    elif artifact_type == "pdf_faithful":
        mode = "faithful"
    else:
        mode = "readable"
    key = f"pdf_{mode}"
    artifacts_dir = job_store.get_job_dir(job_id) / "artifacts"
    markdown_path = Path(status.artifacts.get("markdown", artifacts_dir / "translated.md"))
    json_path = Path(status.artifacts.get("json", artifacts_dir / "structured.json"))
    pdf_path = _translated_pdf_path(status, artifacts_dir, mode)

    if mode == "original_layout":
        from app.models.schema import DocumentModel

        if not json_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Translated structured JSON is required for original-layout reconstruction.",
            )
        source_pdf = job_store.get_job_dir(job_id) / "input.pdf"
        if not source_pdf.exists():
            raise HTTPException(status_code=404, detail="Original uploaded PDF is missing.")
        document = DocumentModel.model_validate_json(json_path.read_text(encoding="utf-8"))
        if _figure_assets_need_refresh(document):
            extraction_metadata: dict = {}
            extraction_result_path = Path(
                status.artifacts.get(
                    "extraction_result",
                    artifacts_dir / "extraction_result.json",
                )
            )
            if extraction_result_path.exists():
                try:
                    extraction_payload = json.loads(
                        extraction_result_path.read_text(encoding="utf-8")
                    )
                    extraction_metadata = dict(extraction_payload.get("metadata") or {})
                except (OSError, ValueError, TypeError):
                    extraction_metadata = {}
            document = pipeline.figure_extractor.extract(
                pdf_path=source_pdf,
                document=document,
                artifact_dir=artifacts_dir / "figures",
                extraction_metadata=extraction_metadata,
            )
            _write_text_atomically(json_path, document.model_dump_json(indent=2))
        report_path = artifacts_dir / "reconstruction_report_original_layout.json"
        staged_pdf_path = _temporary_artifact_path(pdf_path)
        staged_report_path = _temporary_artifact_path(report_path)
        try:
            report = original_layout_reconstructor.reconstruct(
                source_pdf_path=source_pdf,
                output_pdf_path=staged_pdf_path,
                document=document,
                report_path=staged_report_path,
            )
            staged_pdf_path.replace(pdf_path)
            report["output_pdf"] = str(pdf_path.resolve())
            _write_text_atomically(
                report_path,
                json.dumps(report, ensure_ascii=False, indent=2),
            )
        finally:
            staged_pdf_path.unlink(missing_ok=True)
            staged_report_path.unlink(missing_ok=True)
        reconstruction_metadata = {
            "status": report["status"],
            "pages_successfully_reconstructed": report["pages_successfully_reconstructed"],
            "pages_using_fallback_behavior": report["pages_using_fallback_behavior"],
            "warning_count": len(report["warnings"]),
        }
        warning = None
        if report["status"] != "complete":
            warning = (
                "Original-layout reconstruction is partial; unchanged pages and skipped regions are "
                "listed in the reconstruction report. Use the readable PDF as the safe fallback."
            )
        job_store.merge_status(
            job_id,
            artifacts={
                key: str(pdf_path),
                "reconstruction_report": str(report_path),
            },
            translation={"original_layout_reconstruction": reconstruction_metadata},
            translation_warnings=[warning] if warning else None,
        )
        return pdf_path

    started = time.perf_counter()
    if json_path.exists():
        from app.models.schema import DocumentModel

        document = DocumentModel.model_validate_json(json_path.read_text(encoding="utf-8"))
        markdown_text = markdown_builder.build(document)
    elif markdown_path.exists():
        markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise HTTPException(
            status_code=404, detail="Translated JSON or Markdown is required before PDF generation"
        )
    markdown_loaded = time.perf_counter()
    html = reconstructor.markdown_to_html(markdown_text, title=status.filename, output_mode=mode)
    html_built = time.perf_counter()
    staged_pdf_path = _temporary_artifact_path(pdf_path)
    try:
        reconstructor.html_to_pdf(html, staged_pdf_path)
        staged_pdf_path.replace(pdf_path)
    finally:
        staged_pdf_path.unlink(missing_ok=True)
    completed = time.perf_counter()

    profile_path = artifacts_dir / f"pdf_generation_profile_{mode}.json"
    _write_text_atomically(
        profile_path,
        (
            "{\n"
            f'  "mode": "{mode}",\n'
            f'  "markdown_read_s": {markdown_loaded - started:.6f},\n'
            f'  "html_reconstruction_s": {html_built - markdown_loaded:.6f},\n'
            f'  "pdf_export_s": {completed - html_built:.6f},\n'
            f'  "total_s": {completed - started:.6f}\n'
            "}\n"
        ),
    )
    artifact_updates = {
        key: str(pdf_path),
        f"pdf_profile_{mode}": str(profile_path),
    }
    if artifact_type == "pdf":
        artifact_updates["pdf"] = str(pdf_path)
    job_store.merge_status(job_id, artifacts=artifact_updates)
    return pdf_path


def _temporary_artifact_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def _write_text_atomically(path: Path, value: str) -> None:
    temporary_path = _temporary_artifact_path(path)
    try:
        temporary_path.write_text(value, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _figure_assets_need_refresh(document) -> bool:
    has_figure_candidates = bool(document.figures) or any(
        block.block_type.value == "figure" for block in document.blocks
    )
    if not has_figure_candidates:
        return False
    summary = document.metadata.translation.get("figure_extraction")
    if not isinstance(summary, dict):
        return True
    for figure in document.figures:
        if figure.bbox is None:
            continue
        if not figure.source_block_ids or not figure.image_path:
            return True
        if not Path(figure.image_path).exists():
            return True
    return False


def _translated_pdf_path(status, artifacts_dir: Path, mode: str) -> Path:
    return artifacts_dir / _translated_pdf_filename(status, mode)


def _translated_pdf_filename(status, mode: str) -> str:
    original = Path(status.source_filename or status.filename)
    stem = original.stem or "translated"
    if mode == "readable":
        return f"{stem}_translated.pdf"
    if mode == "original_layout":
        return f"{stem}_translated_original_layout.pdf"
    return f"{stem}_translated_{mode}.pdf"


def _source_markdown_path(job_id: str) -> Path:
    try:
        status = job_store.load_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    path = Path(
        status.artifacts.get(
            "source_markdown",
            job_store.get_job_dir(job_id) / "artifacts" / "source.md",
        )
    )
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="Original extraction Markdown is not available for this job."
        )
    return path


def _ensure_source_pdf_artifact(job_id: str, mode: str) -> Path:
    source_markdown = _source_markdown_path(job_id)
    status = job_store.load_status(job_id)
    artifacts_dir = job_store.get_job_dir(job_id) / "artifacts"
    key = f"source_pdf_{mode}"
    pdf_path = Path(status.artifacts.get(key, artifacts_dir / f"source_ocr_{mode}.pdf"))
    if pdf_path.exists():
        return pdf_path
    markdown_text = source_markdown.read_text(encoding="utf-8", errors="ignore")
    html = reconstructor.markdown_to_html(
        markdown_text, title=f"OCR source - {status.filename}", output_mode=mode
    )
    reconstructor.html_to_pdf(html, pdf_path)
    artifacts = dict(status.artifacts)
    artifacts[key] = str(pdf_path)
    job_store.update_status(job_id, artifacts=artifacts)
    return pdf_path


def _build_job_settings(
    *,
    chunk_size: int,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    max_tokens: int,
    output_mode: str,
    profile_pipeline: bool,
    extraction_mode: str,
    ocr_engine: str,
    use_local_vlm_repair: bool,
    keep_debug_artifacts: bool,
) -> dict:
    selected_model = model if model in AVAILABLE_TRANSLATION_MODELS else DEFAULT_TRANSLATION_MODEL
    selected_mode = (
        extraction_mode
        if extraction_mode in {"auto", "digital", "scanned", "strip_and_force_ocr", "auto_repair"}
        else DEFAULT_EXTRACTION_MODE
    )
    selected_ocr_engine = ocr_engine if ocr_engine in AVAILABLE_OCR_ENGINES else DEFAULT_OCR_ENGINE
    return {
        "chunk_size": chunk_size,
        "translation_chunk_group_size": DEFAULT_TRANSLATION_CHUNK_GROUP_SIZE,
        "model": selected_model,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "output_mode": output_mode,
        "profile_pipeline": profile_pipeline,
        "extraction_mode": selected_mode,
        "ocr_engine": selected_ocr_engine,
        "use_local_vlm_repair": bool(use_local_vlm_repair),
        "keep_debug_artifacts": bool(keep_debug_artifacts),
        "marker_timeout_seconds": MARKER_TIMEOUT_SECONDS,
        "surya2_dpi": DEFAULT_SURYA2_DPI,
        "surya2_strategy": DEFAULT_SURYA2_STRATEGY,
        "qwen_ocr_fallback": ENABLE_QWEN_OCR_FALLBACK,
        "qwen_ocr_model": DEFAULT_QWEN_OCR_MODEL,
        "qwen_ocr_max_tokens": DEFAULT_QWEN_OCR_MAX_TOKENS,
        "qwen_ocr_temperature": DEFAULT_QWEN_OCR_TEMPERATURE,
        "qwen_ocr_top_p": DEFAULT_QWEN_OCR_TOP_P,
        "qwen_ocr_top_k": DEFAULT_QWEN_OCR_TOP_K,
        "qwen_ocr_min_p": DEFAULT_QWEN_OCR_MIN_P,
        "qwen_ocr_presence_penalty": DEFAULT_QWEN_OCR_PRESENCE_PENALTY,
        "qwen_ocr_repetition_penalty": DEFAULT_QWEN_OCR_REPETITION_PENALTY,
        "qwen_ocr_prompt": DEFAULT_QWEN_OCR_PROMPT,
        "qwen_ocr_dpi": DEFAULT_QWEN_OCR_DPI,
        "qwen_ocr_batch_size": DEFAULT_QWEN_OCR_BATCH_SIZE,
        "qwen_ocr_crop_mode": DEFAULT_QWEN_OCR_CROP_MODE,
        "qwen_ocr_min_crops": DEFAULT_QWEN_OCR_MIN_CROPS,
        "qwen_ocr_max_crops": DEFAULT_QWEN_OCR_MAX_CROPS,
        "qwen_ocr_base_size": DEFAULT_QWEN_OCR_BASE_SIZE,
        "qwen_ocr_image_size": DEFAULT_QWEN_OCR_IMAGE_SIZE,
        "qwen_ocr_skip_repeat": DEFAULT_QWEN_OCR_SKIP_REPEAT,
        "qwen_ocr_ngram_size": DEFAULT_QWEN_OCR_NGRAM_SIZE,
        "qwen_ocr_ngram_window": DEFAULT_QWEN_OCR_NGRAM_WINDOW,
        "translation_model": {
            "provider": "mlx",
            "model_id": selected_model,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
        },
    }


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
