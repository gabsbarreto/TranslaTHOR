from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    AVAILABLE_TRANSLATION_MODELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DEEPSEEK_OCR_BASE_SIZE,
    DEFAULT_DEEPSEEK_OCR_CROP_MODE,
    DEFAULT_DEEPSEEK_OCR_IMAGE_SIZE,
    DEFAULT_DEEPSEEK_OCR_MAX_TOKENS,
    DEFAULT_DEEPSEEK_OCR_MAX_CROPS,
    DEFAULT_DEEPSEEK_OCR_MODEL,
    DEFAULT_DEEPSEEK_OCR_MIN_CROPS,
    DEFAULT_DEEPSEEK_OCR_NGRAM_SIZE,
    DEFAULT_DEEPSEEK_OCR_NGRAM_WINDOW,
    DEFAULT_DEEPSEEK_OCR_PROMPT,
    DEFAULT_DEEPSEEK_OCR_SKIP_REPEAT,
    DEFAULT_OUTPUT_MODE,
    DEFAULT_RENDER_STRATEGY,
    DEFAULT_TRANSLATION_MODEL,
    FRONTEND_DIR,
)
from app.models.schema import JobStage
from app.services.job_store import JobStore
from app.services.job_queue import JobQueue
from app.services.markdown_builder import MarkdownBuilder
from app.services.pipeline import TranslationPipeline
from app.services.reconstructor import Reconstructor
from app.utils.logging import configure_logging

configure_logging()

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


class RetranslateRequest(BaseModel):
    chunk_size: int = DEFAULT_CHUNK_SIZE
    model: str = DEFAULT_TRANSLATION_MODEL
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    output_mode: str = DEFAULT_OUTPUT_MODE
    profile_pipeline: bool = False


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.model_dump() for j in job_store.list_jobs()]


@app.delete("/api/jobs")
def clear_jobs() -> dict[str, int]:
    job_queue.stop_all()
    removed = job_store.clear_jobs()
    return {"removed": removed}


@app.delete("/api/jobs/cleanup-terminal")
def clear_terminal_jobs() -> dict[str, int]:
    return _clear_terminal_jobs_impl()


@app.post("/api/jobs/cleanup-terminal")
def clear_terminal_jobs_compat() -> dict[str, int]:
    return _clear_terminal_jobs_impl()


def _clear_terminal_jobs_impl() -> dict[str, int]:
    removed = job_store.clear_jobs_by_stage({JobStage.CANCELLED, JobStage.FAILED})
    return {"removed": removed}


@app.post("/api/jobs/stop-all")
def stop_all_jobs() -> dict[str, int]:
    return job_queue.stop_all()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    return _cancel_job_impl(job_id)


@app.get("/api/jobs/{job_id}/cancel")
def cancel_job_compat(job_id: str) -> dict[str, str]:
    return _cancel_job_impl(job_id)


def _cancel_job_impl(job_id: str) -> dict[str, str]:
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
    chunk_size: int = Form(1800),
    model: str = Form(DEFAULT_TRANSLATION_MODEL),
    temperature: float = Form(0.2),
    top_p: float = Form(0.9),
    max_tokens: int = Form(2048),
    output_mode: str = Form(DEFAULT_OUTPUT_MODE),
    profile_pipeline: bool = Form(False),
) -> dict:
    created: list[dict] = []
    for upload in files:
        job_id, job_dir = job_store.create_job(upload.filename)
        in_pdf = job_dir / "input.pdf"
        with in_pdf.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

        settings = _build_job_settings(
            chunk_size=chunk_size,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            output_mode=output_mode,
            profile_pipeline=profile_pipeline,
            reuse_ocr_cache=False,
        )
        job_queue.enqueue(job_id, in_pdf, settings)
        created.append({"job_id": job_id, "filename": upload.filename})

    return {"jobs": created}


@app.post("/api/jobs/{job_id}/retranslate")
def retranslate_job(job_id: str, request: RetranslateRequest) -> dict[str, dict[str, str]]:
    return _create_retranslation_job(job_id, request)


@app.get("/api/jobs/{job_id}/retranslate")
def retranslate_job_compat(
    job_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model: str = DEFAULT_TRANSLATION_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    profile_pipeline: bool = False,
) -> dict[str, dict[str, str]]:
    request = RetranslateRequest(
        chunk_size=chunk_size,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        output_mode=output_mode,
        profile_pipeline=profile_pipeline,
    )
    return _create_retranslation_job(job_id, request)


def _create_retranslation_job(job_id: str, request: RetranslateRequest) -> dict[str, dict[str, str]]:
    try:
        source_status = job_store.load_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    source_job_dir = job_store.get_job_dir(job_id)
    source_pdf = source_job_dir / "input.pdf"
    source_ocr_dir = source_job_dir / "deepseek_ocr"

    if not source_pdf.exists():
        raise HTTPException(status_code=404, detail="Source PDF is missing for this job.")
    if not source_ocr_dir.exists():
        raise HTTPException(status_code=400, detail="OCR cache is unavailable for this job.")
    if not any(source_ocr_dir.glob("page_*.md")):
        raise HTTPException(status_code=400, detail="OCR cache is incomplete for this job.")

    source_name = source_status.source_filename or source_status.filename
    new_job_id, new_job_dir = job_store.create_job(source_name)
    try:
        new_pdf = new_job_dir / "input.pdf"
        shutil.copy2(source_pdf, new_pdf)
        _copy_cached_ocr_data(source_ocr_dir, new_job_dir / "deepseek_ocr")
        settings = _build_job_settings(
            chunk_size=request.chunk_size,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            output_mode=request.output_mode,
            profile_pipeline=request.profile_pipeline,
            reuse_ocr_cache=True,
        )
        job_queue.enqueue(new_job_id, new_pdf, settings)
    except Exception as exc:
        shutil.rmtree(new_job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Unable to create retranslation job: {exc}") from exc

    new_status = job_store.load_status(new_job_id)
    return {"job": {"job_id": new_job_id, "filename": new_status.filename, "source_job_id": job_id}}


@app.get("/api/jobs/{job_id}/artifacts/{artifact_type}")
def get_artifact(job_id: str, artifact_type: str) -> FileResponse:
    try:
        status = job_store.load_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if artifact_type in {"pdf", "pdf_readable", "pdf_faithful"}:
        path = _ensure_pdf_artifact(job_id, artifact_type)
        filename = path.name
        return FileResponse(path, media_type="application/pdf", filename=filename)

    path_str = status.artifacts.get(artifact_type)
    if not path_str:
        raise HTTPException(status_code=404, detail="Artifact not available")

    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    media = {
        "pdf": "application/pdf",
        "markdown": "text/markdown",
        "json": "application/json",
        "profile_json": "application/json",
        "profile_csv": "text/csv",
        "profile_summary": "text/plain",
    }.get(artifact_type, "application/octet-stream")

    filename = path.name
    return FileResponse(path, media_type=media, filename=filename)


@app.get("/api/jobs/{job_id}/pdf/{mode}")
def get_pdf(job_id: str, mode: str) -> FileResponse:
    valid_modes = {"readable", "faithful"}
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Unsupported PDF mode")
    artifact_type = "pdf_faithful" if mode in {"faithful", "visual_sandwich_pdf"} else "pdf_readable"
    path = _ensure_pdf_artifact(job_id, artifact_type)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


def _ensure_pdf_artifact(job_id: str, artifact_type: str) -> Path:
    status = job_store.load_status(job_id)
    mode = "faithful" if artifact_type == "pdf_faithful" else "readable"
    key = "pdf_faithful" if mode == "faithful" else "pdf_readable"

    artifacts_dir = job_store.get_job_dir(job_id) / "artifacts"
    md_path = Path(status.artifacts.get("markdown", artifacts_dir / "translated.md"))
    json_path = Path(status.artifacts.get("json", artifacts_dir / "structured.json"))

    pdf_path = Path(status.artifacts.get(key, artifacts_dir / f"translated_{mode}.pdf"))
    if not pdf_path.exists():
        t0 = time.perf_counter()
        if json_path.exists():
            from app.models.schema import DocumentModel

            document = DocumentModel.model_validate_json(json_path.read_text(encoding="utf-8"))
            markdown_text = markdown_builder.build(document)
        elif md_path.exists():
            markdown_text = md_path.read_text(encoding="utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=404, detail="Translated structured JSON or Markdown is required before PDF generation")
        t1 = time.perf_counter()
        html = reconstructor.markdown_to_html(markdown_text, title=status.filename, output_mode=mode)
        t2 = time.perf_counter()
        reconstructor.html_to_pdf(html, pdf_path)
        t3 = time.perf_counter()

        profile_path = artifacts_dir / f"pdf_generation_profile_{mode}.json"
        profile_path.write_text(
            (
                "{\n"
                f'  "mode": "{mode}",\n'
                f'  "markdown_read_s": {t1 - t0:.6f},\n'
                f'  "html_reconstruction_s": {t2 - t1:.6f},\n'
                f'  "pdf_export_s": {t3 - t2:.6f},\n'
                f'  "total_s": {t3 - t0:.6f}\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        updated_artifacts = dict(status.artifacts)
        updated_artifacts[key] = str(pdf_path)
        updated_artifacts[f"pdf_profile_{mode}"] = str(profile_path)
        if artifact_type == "pdf":
            updated_artifacts["pdf"] = str(pdf_path)
        job_store.update_status(job_id, artifacts=updated_artifacts)

    return pdf_path


def _build_job_settings(
    *,
    chunk_size: int,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    output_mode: str,
    profile_pipeline: bool,
    reuse_ocr_cache: bool,
) -> dict:
    selected_model = model if model in AVAILABLE_TRANSLATION_MODELS else DEFAULT_TRANSLATION_MODEL
    return {
        "chunk_size": chunk_size,
        "model": selected_model,
        "available_models": AVAILABLE_TRANSLATION_MODELS,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "output_mode": output_mode,
        "render_strategy": DEFAULT_RENDER_STRATEGY,
        "profile_pipeline": profile_pipeline,
        "reuse_ocr_cache": reuse_ocr_cache,
        "deepseek_ocr_model": DEFAULT_DEEPSEEK_OCR_MODEL,
        "deepseek_ocr_max_tokens": DEFAULT_DEEPSEEK_OCR_MAX_TOKENS,
        "deepseek_ocr_prompt": DEFAULT_DEEPSEEK_OCR_PROMPT,
        "deepseek_ocr_crop_mode": DEFAULT_DEEPSEEK_OCR_CROP_MODE,
        "deepseek_ocr_min_crops": DEFAULT_DEEPSEEK_OCR_MIN_CROPS,
        "deepseek_ocr_max_crops": DEFAULT_DEEPSEEK_OCR_MAX_CROPS,
        "deepseek_ocr_base_size": DEFAULT_DEEPSEEK_OCR_BASE_SIZE,
        "deepseek_ocr_image_size": DEFAULT_DEEPSEEK_OCR_IMAGE_SIZE,
        "deepseek_ocr_skip_repeat": DEFAULT_DEEPSEEK_OCR_SKIP_REPEAT,
        "deepseek_ocr_ngram_size": DEFAULT_DEEPSEEK_OCR_NGRAM_SIZE,
        "deepseek_ocr_ngram_window": DEFAULT_DEEPSEEK_OCR_NGRAM_WINDOW,
        "translation_model": {
            "provider": "mlx",
            "model_id": selected_model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
    }


def _copy_cached_ocr_data(source_ocr_dir: Path, target_ocr_dir: Path) -> None:
    target_ocr_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in ("page_*.md", "page_*.json", "metadata*.json"):
        for source_file in source_ocr_dir.glob(pattern):
            shutil.copy2(source_file, target_ocr_dir / source_file.name)
            copied += 1
    if copied == 0:
        raise RuntimeError("No OCR cache files were copied.")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
