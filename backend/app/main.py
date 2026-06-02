from __future__ import annotations

import shutil
import sys
import time
import types
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    AVAILABLE_TRANSLATION_MODELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EXTRACTION_MODE,
    DEFAULT_LLM_MIN_P,
    DEFAULT_LLM_PRESENCE_PENALTY,
    DEFAULT_LLM_REPETITION_PENALTY,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_TOP_P,
    DEFAULT_OUTPUT_MODE,
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
    ENABLE_LOCAL_VLM_REPAIR,
    ENABLE_QWEN_OCR_FALLBACK,
    FRONTEND_DIR,
    KEEP_EXTRACTION_DEBUG_ARTIFACTS,
    MARKER_TIMEOUT_SECONDS,
)
from app.models.schema import JobStage
from app.services.job_queue import JobQueue
from app.services.job_store import JobStore
from app.services.markdown_builder import MarkdownBuilder
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

app = FastAPI(title="Local PDF Translation App")
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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [status.model_dump() for status in job_store.list_jobs()]


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
    use_local_vlm_repair: bool = Form(ENABLE_LOCAL_VLM_REPAIR),
    keep_debug_artifacts: bool = Form(KEEP_EXTRACTION_DEBUG_ARTIFACTS),
) -> dict:
    created: list[dict] = []
    for upload in files:
        job_id, job_dir = job_store.create_job(upload.filename)
        pdf_path = job_dir / "input.pdf"
        with pdf_path.open("wb") as file:
            shutil.copyfileobj(upload.file, file)
        job_queue.enqueue(
            job_id,
            pdf_path,
            _build_job_settings(
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
                use_local_vlm_repair=use_local_vlm_repair,
                keep_debug_artifacts=keep_debug_artifacts,
            ),
        )
        created.append({"job_id": job_id, "filename": upload.filename})
    return {"jobs": created}


@app.get("/api/jobs/{job_id}/artifacts/{artifact_type}")
def get_artifact(job_id: str, artifact_type: str) -> FileResponse:
    try:
        status = job_store.load_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if artifact_type in {"pdf", "pdf_readable", "pdf_faithful"}:
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
    }.get(artifact_type, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/jobs/{job_id}/pdf/{mode}")
def get_pdf(job_id: str, mode: str) -> FileResponse:
    if mode not in {"readable", "faithful"}:
        raise HTTPException(status_code=400, detail="Unsupported PDF mode")
    path = _ensure_pdf_artifact(job_id, f"pdf_{mode}")
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
            )
            cancelled += 1
        except FileNotFoundError:
            continue
    return cancelled


def _ensure_pdf_artifact(job_id: str, artifact_type: str) -> Path:
    status = job_store.load_status(job_id)
    mode = "faithful" if artifact_type == "pdf_faithful" else "readable"
    key = f"pdf_{mode}"
    artifacts_dir = job_store.get_job_dir(job_id) / "artifacts"
    markdown_path = Path(status.artifacts.get("markdown", artifacts_dir / "translated.md"))
    json_path = Path(status.artifacts.get("json", artifacts_dir / "structured.json"))
    pdf_path = Path(status.artifacts.get(key, artifacts_dir / f"translated_{mode}.pdf"))
    if pdf_path.exists():
        return pdf_path

    started = time.perf_counter()
    if json_path.exists():
        from app.models.schema import DocumentModel

        document = DocumentModel.model_validate_json(json_path.read_text(encoding="utf-8"))
        markdown_text = markdown_builder.build(document)
    elif markdown_path.exists():
        markdown_text = markdown_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=404, detail="Translated JSON or Markdown is required before PDF generation")
    markdown_loaded = time.perf_counter()
    html = reconstructor.markdown_to_html(markdown_text, title=status.filename, output_mode=mode)
    html_built = time.perf_counter()
    reconstructor.html_to_pdf(html, pdf_path)
    completed = time.perf_counter()

    profile_path = artifacts_dir / f"pdf_generation_profile_{mode}.json"
    profile_path.write_text(
        (
            "{\n"
            f'  "mode": "{mode}",\n'
            f'  "markdown_read_s": {markdown_loaded - started:.6f},\n'
            f'  "html_reconstruction_s": {html_built - markdown_loaded:.6f},\n'
            f'  "pdf_export_s": {completed - html_built:.6f},\n'
            f'  "total_s": {completed - started:.6f}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    artifacts = dict(status.artifacts)
    artifacts[key] = str(pdf_path)
    artifacts[f"pdf_profile_{mode}"] = str(profile_path)
    if artifact_type == "pdf":
        artifacts["pdf"] = str(pdf_path)
    job_store.update_status(job_id, artifacts=artifacts)
    return pdf_path


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
        raise HTTPException(status_code=404, detail="Original extraction Markdown is not available for this job.")
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
    html = reconstructor.markdown_to_html(markdown_text, title=f"OCR source - {status.filename}", output_mode=mode)
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
    use_local_vlm_repair: bool,
    keep_debug_artifacts: bool,
) -> dict:
    selected_model = model if model in AVAILABLE_TRANSLATION_MODELS else DEFAULT_TRANSLATION_MODEL
    selected_mode = (
        extraction_mode
        if extraction_mode in {"auto", "digital", "scanned", "strip_and_force_ocr", "auto_repair"}
        else DEFAULT_EXTRACTION_MODE
    )
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
        "use_local_vlm_repair": bool(use_local_vlm_repair),
        "keep_debug_artifacts": bool(keep_debug_artifacts),
        "marker_timeout_seconds": MARKER_TIMEOUT_SECONDS,
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
