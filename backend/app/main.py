from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
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
from app.models.regions import OcrResultsPayload, PageRegionPayload
from app.services.job_store import JobStore
from app.services.job_queue import JobQueue
from app.services.markdown_builder import MarkdownBuilder
from app.services.ocr_region_service import OcrRegionService
from app.services.pipeline import TranslationPipeline
from app.services.reconstructor import Reconstructor
from app.utils.logging import configure_logging

configure_logging()

# Lightweight fallback to keep module imports/test collection working in environments
# where python-multipart is intentionally not installed.
try:
    from multipart.multipart import parse_options_header as _parse_options_header  # type: ignore

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
ocr_region_service = OcrRegionService()
_selected_ocr_lock = threading.RLock()
_selected_ocr_cancelled_jobs: set[str] = set()
_selected_ocr_processes: dict[str, list[subprocess.Popen]] = {}


class RetranslateRequest(BaseModel):
    chunk_size: int = DEFAULT_CHUNK_SIZE
    model: str = DEFAULT_TRANSLATION_MODEL
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    output_mode: str = DEFAULT_OUTPUT_MODE
    profile_pipeline: bool = False
    translation_input_mode: str = "continuous_document"


class StartJobRequest(BaseModel):
    chunk_size: int = DEFAULT_CHUNK_SIZE
    model: str = DEFAULT_TRANSLATION_MODEL
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    output_mode: str = DEFAULT_OUTPUT_MODE
    profile_pipeline: bool = False
    ocr_input_mode: str = "selected_regions"
    ocr_full_page_fallback: bool = True
    translation_input_mode: str = "continuous_document"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [_status_dump_with_region_artifacts(j) for j in job_store.list_jobs()]


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
    result = job_queue.stop_all()
    selected_result = _cancel_selected_ocr_jobs()
    interrupted_result = _mark_interrupted_processing_jobs_cancelled()
    draft_result = _mark_uploaded_draft_jobs_cancelled()
    return {**result, **selected_result, **interrupted_result, **draft_result}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    return _cancel_job_impl(job_id)


@app.get("/api/jobs/{job_id}/cancel")
def cancel_job_compat(job_id: str) -> dict[str, str]:
    return _cancel_job_impl(job_id)


def _cancel_job_impl(job_id: str) -> dict[str, str]:
    result = job_queue.cancel_job(job_id)
    if result["status"] == "not_found":
        if _cancel_selected_ocr_job(job_id):
            return {"status": "selected_ocr_cancelled"}
        if _mark_uploaded_draft_job_cancelled(job_id):
            return {"status": "draft_cancelled"}
        raise HTTPException(status_code=404, detail="Job is not queued or active.")
    return result


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return _status_dump_with_region_artifacts(job_store.load_status(job_id))
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
    defer_ocr_selection: bool = Form(False),
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
            ocr_input_mode="full_page",
            translation_input_mode="continuous_document",
        )
        if not defer_ocr_selection:
            job_queue.enqueue(job_id, in_pdf, settings)
        created.append({"job_id": job_id, "filename": upload.filename})

    return {"jobs": created}


@app.post("/api/jobs/draft")
async def create_draft_job(file: UploadFile = File(...)) -> dict[str, str]:
    job_id, job_dir = job_store.create_job(file.filename)
    in_pdf = job_dir / "input.pdf"
    with in_pdf.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"job_id": job_id, "filename": file.filename}


@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, request: StartJobRequest) -> dict[str, str]:
    pdf_path = _job_pdf_path(job_id)
    ocr_input_mode = request.ocr_input_mode if request.ocr_input_mode in {"selected_regions", "full_page"} else "selected_regions"

    if ocr_input_mode == "selected_regions":
        ocr_payload = ocr_region_service.region_store.load_ocr_results(job_store.get_job_dir(job_id))
        if (ocr_payload is None or not ocr_payload.results) and request.ocr_full_page_fallback:
            ocr_input_mode = "full_page"
        elif ocr_payload is None or not ocr_payload.results:
            raise HTTPException(
                status_code=400,
                detail="No selected-region OCR results found. Run OCR on selected boxes first, or enable full-page fallback.",
            )
        elif _summarize_ocr_results(ocr_payload)["nonempty_region_count"] == 0:
            raise HTTPException(
                status_code=400,
                detail="Selected-region OCR completed, but no selected region produced text. Re-run OCR or adjust the regions before translation.",
            )

    settings = _build_job_settings(
        chunk_size=request.chunk_size,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        output_mode=request.output_mode,
        profile_pipeline=request.profile_pipeline,
        reuse_ocr_cache=False,
        ocr_input_mode=ocr_input_mode,
        translation_input_mode=request.translation_input_mode,
    )
    job_queue.enqueue(job_id, pdf_path, settings)
    return {"status": "queued", "job_id": job_id, "ocr_input_mode": ocr_input_mode}


@app.get("/api/jobs/{job_id}/pages/{page_number}/image")
def get_page_image(job_id: str, page_number: int, dpi: int = Query(150, ge=72, le=400)) -> FileResponse:
    job_dir = job_store.get_job_dir(job_id)
    pdf_path = _job_pdf_path(job_id)
    try:
        image_path, _, _ = ocr_region_service.render_page_image(
            pdf_path=pdf_path,
            job_dir=job_dir,
            page_number=page_number,
            dpi=dpi,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to render page image: {exc}") from exc
    return FileResponse(image_path, media_type="image/png", filename=image_path.name)


@app.get("/api/jobs/{job_id}/pdf-metadata")
def get_pdf_metadata(job_id: str) -> dict:
    pdf_path = _job_pdf_path(job_id)
    inspection = ocr_region_service.inspect_pdf(pdf_path)
    return {
        "job_id": job_id,
        "page_count": inspection.page_count,
        "pages": [
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "has_embedded_text": page.has_embedded_text,
                "embedded_text_quality": page.embedded_text_quality,
            }
            for page in inspection.pages
        ],
    }


@app.get("/api/jobs/{job_id}/pages/{page_number}/boxes")
def get_detected_boxes(
    job_id: str,
    page_number: int,
    refresh: bool = Query(False),
    detailed: bool = Query(False),
    replace_saved: bool = Query(False),
    dpi: int = Query(150, ge=72, le=400),
) -> dict:
    job_dir = job_store.get_job_dir(job_id)
    pdf_path = _job_pdf_path(job_id)
    try:
        payload = ocr_region_service.get_or_detect_regions(
            pdf_file_id=job_id,
            pdf_path=pdf_path,
            job_dir=job_dir,
            page_number=page_number,
            dpi=dpi,
            refresh=refresh,
            detailed=detailed,
            replace_saved=replace_saved,
        )
    except ValueError as exc:
        if "Saved boxes already exist" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to detect regions: {exc}") from exc
    return payload.model_dump(mode="json")


@app.post("/api/jobs/{job_id}/pages/{page_number}/boxes")
def update_boxes(job_id: str, page_number: int, payload: PageRegionPayload) -> dict:
    if payload.page_number != page_number:
        raise HTTPException(status_code=400, detail="Page number mismatch.")
    job_dir = job_store.get_job_dir(job_id)
    try:
        saved = ocr_region_service.save_regions(job_dir=job_dir, payload=payload)
        status = job_store.load_status(job_id)
        artifacts = dict(status.artifacts)
        artifacts["ocr_regions"] = str(ocr_region_service.region_store.all_regions_path(job_dir))
        job_store.update_status(job_id, artifacts=artifacts, message="OCR boxes saved.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to save regions: {exc}") from exc
    return saved.model_dump(mode="json")


@app.get("/api/jobs/{job_id}/boxes/summary")
def get_boxes_summary(job_id: str) -> dict:
    job_dir = job_store.get_job_dir(job_id)
    pages = ocr_region_service.region_store.list_page_regions(job_dir)
    return {
        "job_id": job_id,
        "has_saved_boxes": bool(pages),
        "saved_page_count": len(pages),
        "saved_pages": [page.page_number for page in pages],
        "total_region_count": sum(len(page.regions) for page in pages),
        "selected_region_count": sum(1 for page in pages for region in page.regions if region.selected),
        "all_regions_path": str(ocr_region_service.region_store.all_regions_path(job_dir)),
    }


@app.post("/api/jobs/{job_id}/duplicate-for-ocr-rerun")
@app.post("/api/jobs/{job_id}/duplicate-for-ocr-rerun/")
def duplicate_job_for_ocr_rerun(job_id: str) -> dict[str, str]:
    try:
        source_status = job_store.load_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if source_status.stage in {JobStage.EXTRACTION, JobStage.OCR_LAYOUT, JobStage.STRUCTURE, JobStage.TRANSLATION, JobStage.PDF}:
        raise HTTPException(status_code=409, detail="This job is still processing. Wait for it to finish before creating an OCR rerun.")

    source_job_dir = job_store.get_job_dir(job_id)
    source_pdf = source_job_dir / "input.pdf"
    if not source_pdf.exists():
        raise HTTPException(status_code=404, detail="Source PDF is missing for this job.")

    source_name = source_status.source_filename or source_status.filename
    rerun_name = _ocr_rerun_filename(source_name)
    new_job_id, new_job_dir = job_store.create_job(rerun_name)
    try:
        shutil.copy2(source_pdf, new_job_dir / "input.pdf")
        copied_pages = _copy_saved_regions_for_rerun(source_job_dir, new_job_dir, new_job_id)
        artifacts: dict[str, str] = {
            "parent_job_id": job_id,
            "rerun_type": "ocr_rerun",
        }
        if copied_pages:
            artifacts["ocr_regions"] = str(ocr_region_service.region_store.all_regions_path(new_job_dir))
        job_store.update_status(
            new_job_id,
            stage=JobStage.UPLOADED,
            progress=0.0,
            message="Created editable OCR rerun from previous job.",
            error=None,
            artifacts=artifacts,
            translation={},
        )
    except Exception as exc:
        shutil.rmtree(new_job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Unable to create OCR rerun: {exc}") from exc

    new_status = job_store.load_status(new_job_id)
    return {
        "new_job_id": new_job_id,
        "source_job_id": job_id,
        "filename": new_status.filename,
        "message": "Created editable OCR rerun from previous job.",
    }


@app.post("/api/jobs/{job_id}/ocr/selected")
def run_ocr_for_selected_boxes(
    job_id: str,
    dpi: int = Query(300, ge=100, le=600),
    page_number: int | None = Query(default=None, ge=1),
    box_id: str | None = Query(default=None, min_length=1),
) -> dict:
    job_dir = job_store.get_job_dir(job_id)
    pdf_path = _job_pdf_path(job_id)
    try:
        job_store.update_status(
            job_id,
            stage=JobStage.OCR_LAYOUT,
            progress=0.35,
            message="Running OCR on selected regions",
            error=None,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        _start_selected_ocr_tracking(job_id)

        def on_ocr_progress(event: dict) -> None:
            update = _selected_ocr_progress_from_event(event)
            if update is None:
                return
            progress, message = update
            try:
                job_store.update_status(
                    job_id,
                    stage=JobStage.OCR_LAYOUT,
                    progress=progress,
                    message=message,
                    error=None,
                )
            except FileNotFoundError:
                pass

        results = ocr_region_service.run_selected_ocr(
            pdf_file_id=job_id,
            pdf_path=pdf_path,
            job_dir=job_dir,
            dpi=dpi,
            model_name=DEFAULT_DEEPSEEK_OCR_MODEL,
            max_tokens=DEFAULT_DEEPSEEK_OCR_MAX_TOKENS,
            prompt=DEFAULT_DEEPSEEK_OCR_PROMPT,
            crop_mode=DEFAULT_DEEPSEEK_OCR_CROP_MODE,
            min_crops=DEFAULT_DEEPSEEK_OCR_MIN_CROPS,
            max_crops=DEFAULT_DEEPSEEK_OCR_MAX_CROPS,
            base_size=DEFAULT_DEEPSEEK_OCR_BASE_SIZE,
            image_size=DEFAULT_DEEPSEEK_OCR_IMAGE_SIZE,
            skip_repeat=DEFAULT_DEEPSEEK_OCR_SKIP_REPEAT,
            ngram_size=DEFAULT_DEEPSEEK_OCR_NGRAM_SIZE,
            ngram_window=DEFAULT_DEEPSEEK_OCR_NGRAM_WINDOW,
            page_number=page_number,
            box_id=box_id,
            on_ocr_progress=on_ocr_progress,
            cancel_requested=lambda: _is_selected_ocr_cancelled(job_id),
            on_process_started=lambda process: _register_selected_ocr_process(job_id, process),
            on_process_finished=lambda process: _unregister_selected_ocr_process(job_id, process),
        )
    except ValueError as exc:
        job_store.update_status(
            job_id,
            stage=JobStage.UPLOADED,
            progress=0.0,
            message="Selected-region OCR did not run",
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if _is_selected_ocr_cancelled(job_id) or str(exc) == "Cancelled by user":
            try:
                job_store.update_status(
                    job_id,
                    stage=JobStage.CANCELLED,
                    progress=1.0,
                    message="Selected-region OCR cancelled by user.",
                    error=None,
                )
            except FileNotFoundError:
                pass
            raise HTTPException(status_code=409, detail="Selected-region OCR cancelled by user.") from exc
        job_store.update_status(
            job_id,
            stage=JobStage.UPLOADED,
            progress=0.0,
            message="Selected-region OCR failed",
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=f"Selected-region OCR failed: {exc}") from exc
    finally:
        _finish_selected_ocr_tracking(job_id)
    status = job_store.load_status(job_id)
    updated_artifacts = dict(status.artifacts)
    updated_artifacts["ocr_results"] = str(ocr_region_service.region_store.ocr_results_path(job_dir))
    source_md_path = _write_selected_ocr_source_markdown(job_id, results)
    updated_artifacts["source_markdown"] = str(source_md_path)
    summary = _summarize_ocr_results(results)
    empty_note = ""
    if summary["pages_without_text_count"]:
        empty_note = f" {summary['pages_without_text_count']} selected page(s) returned empty OCR."
    job_store.update_status(
        job_id,
        stage=JobStage.UPLOADED,
        progress=0.0,
        message=(
            f"Selected-region OCR completed for {summary['total_region_count']} region(s) "
            f"across {summary['selected_page_count']} page(s); text found on "
            f"{summary['pages_with_text_count']} page(s).{empty_note}"
        ),
        error=None,
        artifacts=updated_artifacts,
    )
    return {"pdf_file_id": results.pdf_file_id, "count": len(results.results), **summary}


@app.get("/api/jobs/{job_id}/ocr-results")
def get_ocr_results(job_id: str) -> dict:
    job_dir = job_store.get_job_dir(job_id)
    payload = ocr_region_service.region_store.load_ocr_results(job_dir)
    if payload is None:
        return OcrResultsPayload(pdf_file_id=job_id, results=[]).model_dump(mode="json")
    return payload.model_dump(mode="json")


def _summarize_ocr_results(payload: OcrResultsPayload) -> dict:
    all_pages = {item.page_number for item in payload.results}
    pages_with_text = {item.page_number for item in payload.results if item.ocr_text.strip()}
    pages_without_text = all_pages - pages_with_text
    nonempty_region_count = sum(1 for item in payload.results if item.ocr_text.strip())
    return {
        "total_region_count": len(payload.results),
        "nonempty_region_count": nonempty_region_count,
        "empty_region_count": len(payload.results) - nonempty_region_count,
        "selected_page_count": len(all_pages),
        "pages_with_text_count": len(pages_with_text),
        "pages_without_text_count": len(pages_without_text),
        "pages_with_text": sorted(pages_with_text),
        "pages_without_text": sorted(pages_without_text),
    }


def _start_selected_ocr_tracking(job_id: str) -> None:
    with _selected_ocr_lock:
        _selected_ocr_cancelled_jobs.discard(job_id)
        _selected_ocr_processes[job_id] = []


def _finish_selected_ocr_tracking(job_id: str) -> None:
    with _selected_ocr_lock:
        _selected_ocr_cancelled_jobs.discard(job_id)
        _selected_ocr_processes.pop(job_id, None)


def _is_selected_ocr_cancelled(job_id: str) -> bool:
    with _selected_ocr_lock:
        return job_id in _selected_ocr_cancelled_jobs


def _register_selected_ocr_process(job_id: str, process: subprocess.Popen) -> None:
    with _selected_ocr_lock:
        _selected_ocr_processes.setdefault(job_id, []).append(process)
        cancelled = job_id in _selected_ocr_cancelled_jobs
    if cancelled:
        _terminate_process_tree(process)


def _unregister_selected_ocr_process(job_id: str, process: subprocess.Popen) -> None:
    with _selected_ocr_lock:
        processes = _selected_ocr_processes.get(job_id, [])
        _selected_ocr_processes[job_id] = [item for item in processes if item.pid != process.pid]


def _cancel_selected_ocr_job(job_id: str) -> bool:
    with _selected_ocr_lock:
        processes = list(_selected_ocr_processes.get(job_id, []))
        is_active = job_id in _selected_ocr_processes
        if is_active:
            _selected_ocr_cancelled_jobs.add(job_id)

    if not is_active:
        return False

    for process in processes:
        _terminate_process_tree(process)

    try:
        job_store.update_status(
            job_id,
            stage=JobStage.CANCELLED,
            progress=1.0,
            message="Selected-region OCR cancelled by user.",
            error=None,
        )
    except FileNotFoundError:
        pass
    return True


def _cancel_selected_ocr_jobs() -> dict[str, int]:
    with _selected_ocr_lock:
        job_ids = list(_selected_ocr_processes)
    cancelled = 0
    for job_id in job_ids:
        if _cancel_selected_ocr_job(job_id):
            cancelled += 1
    return {"selected_ocr_cancelled": cancelled}


def _mark_interrupted_processing_jobs_cancelled() -> dict[str, int]:
    processing_stages = {
        JobStage.EXTRACTION,
        JobStage.OCR_LAYOUT,
        JobStage.STRUCTURE,
        JobStage.TRANSLATION,
        JobStage.PDF,
    }
    cancelled = 0
    for status in job_store.list_jobs():
        if status.stage not in processing_stages:
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
    return {"interrupted_cancelled": cancelled}


def _mark_uploaded_draft_job_cancelled(job_id: str) -> bool:
    try:
        status = job_store.load_status(job_id)
    except FileNotFoundError:
        return False
    if status.stage != JobStage.UPLOADED:
        return False
    job_store.update_status(
        job_id,
        stage=JobStage.CANCELLED,
        progress=1.0,
        message="Cancelled before OCR started.",
        error=None,
    )
    return True


def _mark_uploaded_draft_jobs_cancelled() -> dict[str, int]:
    cancelled = 0
    for status in job_store.list_jobs():
        if status.stage != JobStage.UPLOADED:
            continue
        if _mark_uploaded_draft_job_cancelled(status.job_id):
            cancelled += 1
    return {"draft_cancelled": cancelled}


def _terminate_process_tree(process: subprocess.Popen) -> None:
    try:
        if process.poll() is not None:
            return
        try:
            pgid = os.getpgid(process.pid)
        except Exception:
            pgid = None
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
    except Exception:
        try:
            if process.poll() is None:
                process.kill()
        except Exception:
            pass


def _selected_ocr_progress_from_event(event: dict) -> tuple[float, str] | None:
    """Map DeepSeek OCR worker events onto the existing job progress bar.

    Selected-region OCR sends one image per selected OCR box. The worker reports
    per-image progress, so the UI can display useful counters like 1/15 without
    changing the OCR endpoint contract.
    """
    event_name = event.get("event")
    phase = str(event.get("phase") or "primary")
    phase_label = "OCR selected regions" if phase == "primary" else "Retrying empty OCR regions"

    if event_name == "model_loading":
        return 0.35, "Loading OCR model for selected regions"
    if event_name == "model_loaded":
        total = _positive_int(event.get("pages"))
        if total is None:
            return 0.36, "OCR model loaded; processing selected regions"
        return 0.36, f"OCR model loaded; processing {total} selected region(s)"
    if event_name not in {"page_started", "page_done"}:
        return None

    index = _positive_int(event.get("index"))
    total = _positive_int(event.get("total"))
    if index is None or total is None:
        return None

    if event_name == "page_started":
        fraction = max(0.0, min((index - 1) / total, 1.0))
        return round(0.36 + 0.48 * fraction, 3), f"{phase_label}: {index}/{total}"

    fraction = max(0.0, min(index / total, 1.0))
    chars = _positive_int(event.get("chars"))
    chars_note = f"; {chars} characters" if chars is not None else ""
    return round(0.36 + 0.48 * fraction, 3), f"{phase_label}: {index}/{total} complete{chars_note}"


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _status_dump_with_region_artifacts(status) -> dict:
    data = status.model_dump()
    job_dir = job_store.get_job_dir(status.job_id)
    all_regions = ocr_region_service.region_store.all_regions_path(job_dir)
    ocr_results = ocr_region_service.region_store.ocr_results_path(job_dir)
    source_markdown = job_dir / "artifacts" / "source.md"
    if all_regions.exists():
        artifacts = dict(data.get("artifacts") or {})
        artifacts.setdefault("ocr_regions", str(all_regions))
        data["artifacts"] = artifacts
    if ocr_results.exists() or source_markdown.exists():
        artifacts = dict(data.get("artifacts") or {})
        if ocr_results.exists():
            artifacts.setdefault("ocr_results", str(ocr_results))
        if source_markdown.exists():
            artifacts.setdefault("source_markdown", str(source_markdown))
        data["artifacts"] = artifacts
    return data


def _ocr_rerun_filename(source_name: str) -> str:
    path = Path(source_name)
    suffix = path.suffix
    stem = path.stem or source_name
    if suffix:
        return f"{stem} OCR rerun{suffix}"
    return f"{source_name} OCR rerun"


def _copy_saved_regions_for_rerun(source_job_dir: Path, new_job_dir: Path, new_job_id: str) -> int:
    pages = ocr_region_service.region_store.list_page_regions(source_job_dir)
    for page in pages:
        copied = page.model_copy(update={"pdf_file_id": new_job_id})
        ocr_region_service.region_store.save_page_regions(new_job_dir, copied)
    if pages:
        ocr_region_service.region_store.save_all_regions(new_job_dir)
    return len(pages)


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
    source_region_ocr_dir = source_job_dir / "ocr_regions"
    source_region_results = source_region_ocr_dir / "results.json"

    if not source_pdf.exists():
        raise HTTPException(status_code=404, detail="Source PDF is missing for this job.")
    has_page_cache = source_ocr_dir.exists() and any(source_ocr_dir.glob("page_*.md"))
    has_region_cache = source_region_results.exists()
    if not has_page_cache and not has_region_cache:
        raise HTTPException(status_code=400, detail="OCR cache is unavailable for this job.")

    source_name = source_status.source_filename or source_status.filename
    new_job_id, new_job_dir = job_store.create_job(source_name)
    try:
        new_pdf = new_job_dir / "input.pdf"
        shutil.copy2(source_pdf, new_pdf)
        ocr_input_mode = "full_page"
        reuse_ocr_cache = False
        if has_page_cache:
            _copy_cached_ocr_data(source_ocr_dir, new_job_dir / "deepseek_ocr")
            ocr_input_mode = "full_page"
            reuse_ocr_cache = True
        elif has_region_cache:
            _copy_selected_region_ocr_data(source_region_ocr_dir, new_job_dir / "ocr_regions")
            ocr_input_mode = "selected_regions"
            reuse_ocr_cache = False
        settings = _build_job_settings(
            chunk_size=request.chunk_size,
            model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            output_mode=request.output_mode,
            profile_pipeline=request.profile_pipeline,
            reuse_ocr_cache=reuse_ocr_cache,
            ocr_input_mode=ocr_input_mode,
            translation_input_mode=request.translation_input_mode,
        )
        job_queue.enqueue(new_job_id, new_pdf, settings)
    except Exception as exc:
        shutil.rmtree(new_job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Unable to create retranslation job: {exc}") from exc

    new_status = job_store.load_status(new_job_id)
    return {"job": {"job_id": new_job_id, "filename": new_status.filename, "source_job_id": job_id}}


def _job_pdf_path(job_id: str) -> Path:
    job_dir = job_store.get_job_dir(job_id)
    pdf_path = job_dir / "input.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Input PDF not found for this job.")
    return pdf_path


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
    if artifact_type == "source_markdown":
        path = _ensure_ocr_source_markdown(job_id)
        return FileResponse(path, media_type="text/markdown", filename=path.name)

    path_str = status.artifacts.get(artifact_type)
    if not path_str:
        raise HTTPException(status_code=404, detail="Artifact not available")

    path = Path(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    media = {
        "pdf": "application/pdf",
        "markdown": "text/markdown",
        "source_markdown": "text/markdown",
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


@app.get("/api/jobs/{job_id}/ocr-pdf/{mode}")
def get_ocr_pdf(job_id: str, mode: str) -> FileResponse:
    valid_modes = {"readable", "faithful"}
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Unsupported PDF mode")
    path = _ensure_ocr_source_pdf_artifact(job_id, mode)
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


def _ensure_ocr_source_markdown(job_id: str) -> Path:
    try:
        status = job_store.load_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    job_dir = job_store.get_job_dir(job_id)
    artifacts_dir = job_dir / "artifacts"
    source_md_path = Path(status.artifacts.get("source_markdown", artifacts_dir / "source.md"))
    if source_md_path.exists():
        return source_md_path

    full_page_dir = job_dir / "deepseek_ocr"
    page_markdown = sorted(full_page_dir.glob("page_*.md")) if full_page_dir.exists() else []
    if page_markdown:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        text = "\n\n".join(path.read_text(encoding="utf-8", errors="ignore").strip() for path in page_markdown)
        source_md_path.write_text(text.strip() + "\n", encoding="utf-8")
        _record_source_markdown_artifact(job_id, source_md_path)
        return source_md_path

    payload = ocr_region_service.region_store.load_ocr_results(job_dir)
    if payload is not None and payload.results:
        source_md_path = _write_selected_ocr_source_markdown(job_id, payload)
        _record_source_markdown_artifact(job_id, source_md_path)
        return source_md_path

    raise HTTPException(status_code=404, detail="Original OCR markdown is not available for this job.")


def _ensure_ocr_source_pdf_artifact(job_id: str, mode: str) -> Path:
    source_md_path = _ensure_ocr_source_markdown(job_id)
    status = job_store.load_status(job_id)
    artifacts_dir = job_store.get_job_dir(job_id) / "artifacts"
    key = f"source_pdf_{mode}"
    pdf_path = Path(status.artifacts.get(key, artifacts_dir / f"source_ocr_{mode}.pdf"))
    if not pdf_path.exists():
        markdown_text = source_md_path.read_text(encoding="utf-8", errors="ignore")
        html = reconstructor.markdown_to_html(markdown_text, title=f"OCR source - {status.filename}", output_mode=mode)
        reconstructor.html_to_pdf(html, pdf_path)
        updated_artifacts = dict(status.artifacts)
        updated_artifacts["source_markdown"] = str(source_md_path)
        updated_artifacts[key] = str(pdf_path)
        job_store.update_status(job_id, artifacts=updated_artifacts)
    return pdf_path


def _write_selected_ocr_source_markdown(job_id: str, payload: OcrResultsPayload) -> Path:
    job_dir = job_store.get_job_dir(job_id)
    out_path = job_dir / "artifacts" / "source.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_order: dict[int, list] = {}
    for item in sorted(payload.results, key=lambda result: (result.page_number, result.reading_order, result.box_id)):
        if not item.ocr_text.strip():
            continue
        page_order.setdefault(item.page_number, []).append(item)

    lines: list[str] = []
    for page_number in sorted(page_order):
        lines.append(f"<!-- page: {page_number} -->")
        for item in page_order[page_number]:
            lines.append(item.ocr_text.strip())
            lines.append("")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_path


def _record_source_markdown_artifact(job_id: str, source_md_path: Path) -> None:
    try:
        status = job_store.load_status(job_id)
    except FileNotFoundError:
        return
    artifacts = dict(status.artifacts)
    artifacts["source_markdown"] = str(source_md_path)
    job_store.update_status(job_id, artifacts=artifacts)


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
    ocr_input_mode: str = "full_page",
    translation_input_mode: str = "continuous_document",
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
        "ocr_input_mode": ocr_input_mode,
        "translation_input_mode": translation_input_mode
        if translation_input_mode in {"continuous_document", "page_by_page"}
        else "continuous_document",
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


def _copy_selected_region_ocr_data(source_ocr_dir: Path, target_ocr_dir: Path) -> None:
    if not source_ocr_dir.exists():
        raise RuntimeError("Selected-region OCR directory does not exist.")
    if target_ocr_dir.exists():
        shutil.rmtree(target_ocr_dir)
    shutil.copytree(source_ocr_dir, target_ocr_dir)
    results_path = target_ocr_dir / "results.json"
    if not results_path.exists():
        raise RuntimeError("Selected-region OCR results.json is missing.")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
