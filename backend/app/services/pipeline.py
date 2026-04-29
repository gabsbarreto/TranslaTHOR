from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from app.config import (
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
    DEFAULT_RENDER_STRATEGY,
)
from app.models.schema import JobStage
from app.services.deepseek_ocr_pipeline import DeepSeekOcrPipeline
from app.services.job_store import JobStore
from app.services.markdown_builder import MarkdownBuilder
from app.services.pdf_inspector import PdfInspector
from app.services.profiler import PipelineProfiler
from app.services.region_store import RegionStore
from app.services.translation_debug import write_translation_comparison_report
from app.services.translation_subprocess import run_translation_subprocess

logger = logging.getLogger(__name__)


class TranslationPipeline:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store
        self.inspector = PdfInspector()
        self.deepseek_parser = DeepSeekOcrPipeline()
        self.md_builder = MarkdownBuilder()
        self.region_store = RegionStore()
        self._lock = threading.RLock()
        self._cancelled_jobs: set[str] = set()
        self._active_processes: dict[str, list[subprocess.Popen]] = {}

    def cancel_job(self, job_id: str) -> None:
        with self._lock:
            self._cancelled_jobs.add(job_id)
            processes = list(self._active_processes.get(job_id, []))
        for process in processes:
            self._terminate_process(process)
        try:
            self.job_store.update_status(
                job_id,
                stage=JobStage.CANCELLED,
                progress=1.0,
                message="Cancelled by user.",
                error=None,
            )
        except FileNotFoundError:
            logger.info("Cancelled job %s disappeared before status update", job_id)

    def run(self, job_id: str, pdf_path: Path, settings: dict) -> None:
        profile_enabled = bool(settings.get("profile_pipeline", False))
        profiler = PipelineProfiler(enabled=profile_enabled)
        artifacts_dir = self.job_store.get_job_dir(job_id) / "artifacts"
        render_strategy = str(settings.get("render_strategy", DEFAULT_RENDER_STRATEGY))
        reuse_ocr_cache = bool(settings.get("reuse_ocr_cache", False))
        ocr_input_mode = str(settings.get("ocr_input_mode", "full_page"))
        translation_input_mode = str(settings.get("translation_input_mode", "continuous_document"))
        translation_metadata = {
            "model": str(settings.get("model", "")),
            "temperature": float(settings.get("temperature", 0.2)),
            "top_p": float(settings.get("top_p", 0.9)),
        }

        try:
            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id)
                return
            if not self._update_status(job_id, stage=JobStage.EXTRACTION, progress=0.1, message="Inspecting PDF"):
                return
            with profiler.step("pdf_loading"):
                inspection = self.inspector.inspect(pdf_path)
            self._write_decision_debug(
                job_id,
                {
                    page.page_number: (
                        "ocr: selected_regions" if ocr_input_mode == "selected_regions" else "ocr: deepseek_ocr"
                    )
                    for page in inspection.pages
                },
            )

            if ocr_input_mode == "selected_regions":
                if not self._update_status(
                    job_id,
                    stage=JobStage.OCR_LAYOUT,
                    progress=0.32,
                    message="Building OCR document from selected regions",
                ):
                    return
                with profiler.step("ocr_selected_region_load"):
                    ocr_results = self.region_store.load_ocr_results(self.job_store.get_job_dir(job_id))
                    if ocr_results is None or not ocr_results.results:
                        raise RuntimeError(
                            "Selected-region OCR results were not found. Run OCR on selected boxes before translation."
                        )
                    doc, marker_md = self.deepseek_parser.parse_selected_regions_document(
                        inspection=inspection,
                        ocr_results=ocr_results,
                        profiler=profiler,
                        translation_input_mode=translation_input_mode,
                    )
            elif reuse_ocr_cache:
                if not self._update_status(
                    job_id,
                    stage=JobStage.OCR_LAYOUT,
                    progress=0.32,
                    message="Reusing cached OCR markdown",
                ):
                    return
                with profiler.step("ocr_cache_load"):
                    doc, marker_md = self.deepseek_parser.parse_cached_document(
                        inspection=inspection,
                        job_dir=self.job_store.get_job_dir(job_id),
                        profiler=profiler,
                    )
            else:
                if not self._update_status(
                    job_id,
                    stage=JobStage.OCR_LAYOUT,
                    progress=0.3,
                    message="Running DeepSeek-OCR-2 MLX parser",
                ):
                    return
                preprocess = {
                    "grayscale": bool(settings.get("grayscale", False)),
                    "binarize": bool(settings.get("binarize", False)),
                    "denoise": bool(settings.get("denoise", False)),
                    "contrast": float(settings.get("contrast", 1.0)),
                }
                if not self._update_status(
                    job_id,
                    stage=JobStage.OCR_LAYOUT,
                    progress=0.32,
                    message="Running DeepSeek-OCR-2 MLX parser",
                ):
                    return
                with profiler.step("ocr_layout_detection"):
                    doc, marker_md = self.deepseek_parser.parse_document(
                        pdf_path=pdf_path,
                        inspection=inspection,
                        job_dir=self.job_store.get_job_dir(job_id),
                        dpi=int(settings.get("dpi", 300)),
                        preprocess=preprocess,
                        model_name=str(settings.get("deepseek_ocr_model", DEFAULT_DEEPSEEK_OCR_MODEL)),
                        max_tokens=int(settings.get("deepseek_ocr_max_tokens", DEFAULT_DEEPSEEK_OCR_MAX_TOKENS)),
                        prompt=str(settings.get("deepseek_ocr_prompt", DEFAULT_DEEPSEEK_OCR_PROMPT)),
                        crop_mode=bool(settings.get("deepseek_ocr_crop_mode", DEFAULT_DEEPSEEK_OCR_CROP_MODE)),
                        min_crops=int(settings.get("deepseek_ocr_min_crops", DEFAULT_DEEPSEEK_OCR_MIN_CROPS)),
                        max_crops=int(settings.get("deepseek_ocr_max_crops", DEFAULT_DEEPSEEK_OCR_MAX_CROPS)),
                        base_size=int(settings.get("deepseek_ocr_base_size", DEFAULT_DEEPSEEK_OCR_BASE_SIZE)),
                        image_size=int(settings.get("deepseek_ocr_image_size", DEFAULT_DEEPSEEK_OCR_IMAGE_SIZE)),
                        skip_repeat=bool(settings.get("deepseek_ocr_skip_repeat", DEFAULT_DEEPSEEK_OCR_SKIP_REPEAT)),
                        ngram_size=int(settings.get("deepseek_ocr_ngram_size", DEFAULT_DEEPSEEK_OCR_NGRAM_SIZE)),
                        ngram_window=int(settings.get("deepseek_ocr_ngram_window", DEFAULT_DEEPSEEK_OCR_NGRAM_WINDOW)),
                        profiler=profiler,
                        render_strategy=render_strategy,
                        cancel_requested=lambda: self._is_cancelled(job_id),
                        on_process_started=lambda process: self._register_process(job_id, process),
                        on_process_finished=lambda process: self._unregister_process(job_id, process),
                    )
            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id)
                return

            if not self._update_status(
                job_id,
                stage=JobStage.STRUCTURE,
                progress=0.5,
                message="Building structured JSON and Markdown",
            ):
                return
            with profiler.step("structure_markdown_generation"):
                markdown_text = self.md_builder.build(doc, marker_md)

            json_path = artifacts_dir / "structured.json"
            md_path = artifacts_dir / "translated.md"
            source_md_path = artifacts_dir / "source.md"
            pdf_readable = artifacts_dir / "translated_readable.pdf"
            pdf_faithful = artifacts_dir / "translated_faithful.pdf"

            with profiler.step("structured_json_write"):
                self.deepseek_parser.save_document_json(doc, json_path)
            source_md_path.write_text(markdown_text, encoding="utf-8")

            if not self._update_status(
                job_id,
                stage=JobStage.TRANSLATION,
                progress=0.7,
                message="Starting isolated MLX translation subprocess",
            ):
                return

            def on_chunk_translated(index: int, total: int, preview: str) -> None:
                # Keep the UI responsive during long local model inference.
                partial_progress = 0.7 + (0.18 * (index / max(total, 1)))
                safe_preview = preview if preview else "(empty output)"
                self._update_status(
                    job_id,
                    stage=JobStage.TRANSLATION,
                    progress=round(min(partial_progress, 0.88), 3),
                    message=f"Chunk {index}/{total}: {safe_preview}",
                )

            def on_chunk_started(index: int, total: int) -> None:
                self._update_status(
                    job_id,
                    stage=JobStage.TRANSLATION,
                    progress=0.7,
                    message=f"Chunk {index}/{total}: translating...",
                )

            def on_table_progress(index: int, total: int, label: str) -> None:
                partial_progress = 0.88 + (0.08 * (index / max(total, 1)))
                self._update_status(
                    job_id,
                    stage=JobStage.TRANSLATION,
                    progress=round(min(partial_progress, 0.96), 3),
                    message=f"Tables {index}/{total}: {label}",
                )

            run_translation_subprocess(
                document_path=json_path,
                markdown_path=source_md_path,
                output_document_path=json_path,
                output_markdown_path=md_path,
                settings=settings,
                on_chunk_started=on_chunk_started,
                on_chunk_translated=on_chunk_translated,
                on_table_progress=on_table_progress,
                on_process_started=lambda process: self._register_process(job_id, process),
                on_process_finished=lambda process: self._unregister_process(job_id, process),
                profiler=profiler,
            )
            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id)
                return

            translated_document = self.deepseek_parser.load_document_json(json_path)
            comparison_json, comparison_md = write_translation_comparison_report(
                source_path=source_md_path,
                translated_path=md_path,
                document=translated_document,
                output_dir=artifacts_dir / "debug",
            )

            profile_json = artifacts_dir / "timing_profile.json"
            profile_csv = artifacts_dir / "timing_profile.csv"
            profile_summary = artifacts_dir / "timing_summary.txt"
            if profile_enabled:
                json_path_prof, csv_path_prof, summary_path_prof = profiler.dump(artifacts_dir)
                profile_json = json_path_prof
                profile_csv = csv_path_prof
                profile_summary = summary_path_prof
                logger.info("Pipeline timing summary for %s\n%s", job_id, "\n".join(profiler.summary_lines()))

            self._update_status(
                job_id,
                stage=JobStage.COMPLETE,
                progress=1.0,
                message=f"Done at {datetime.utcnow().isoformat()}Z. PDFs are generated on demand when downloaded.",
                translation=translation_metadata,
                artifacts={
                    "json": str(json_path),
                    "markdown": str(md_path),
                    "debug_markdown": str(md_path),
                    "pdf_readable": str(pdf_readable),
                    "pdf_faithful": str(pdf_faithful),
                    "profile_json": str(profile_json) if profile_enabled else "",
                    "profile_csv": str(profile_csv) if profile_enabled else "",
                    "profile_summary": str(profile_summary) if profile_enabled else "",
                    "translation_comparison_json": str(comparison_json),
                    "translation_comparison_md": str(comparison_md),
                },
            )
        except Exception as exc:
            if self._is_cancelled(job_id) or str(exc) == "Cancelled by user":
                self._mark_cancelled(job_id)
                return
            logger.exception("Pipeline failed for job %s", job_id)
            self._update_status(
                job_id,
                stage=JobStage.FAILED,
                progress=1.0,
                message="Pipeline failed",
                error=str(exc),
            )
        finally:
            with self._lock:
                self._active_processes.pop(job_id, None)
                self._cancelled_jobs.discard(job_id)

    def _update_status(self, job_id: str, **updates: object) -> bool:
        try:
            self.job_store.update_status(job_id, **updates)
            return True
        except FileNotFoundError:
            logger.info("Job %s no longer exists; stopping background work", job_id)
            return False

    def _write_decision_debug(self, job_id: str, reasons: dict[int, str]) -> None:
        job_dir = self.job_store.get_job_dir(job_id)
        debug_path = job_dir / "artifacts" / "extraction_decisions.json"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(page): reason for page, reason in sorted(reasons.items())}
        debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _register_process(self, job_id: str, process: subprocess.Popen) -> None:
        with self._lock:
            self._active_processes.setdefault(job_id, []).append(process)

    def _unregister_process(self, job_id: str, process: subprocess.Popen) -> None:
        with self._lock:
            items = self._active_processes.get(job_id, [])
            self._active_processes[job_id] = [item for item in items if item.pid != process.pid]

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled_jobs

    def _mark_cancelled(self, job_id: str) -> None:
        self._update_status(
            job_id,
            stage=JobStage.CANCELLED,
            progress=1.0,
            message="Cancelled by user.",
            error=None,
        )

    def _terminate_process(self, process: subprocess.Popen) -> None:
        try:
            if process.poll() is None:
                try:
                    pgid = os.getpgid(process.pid)
                except Exception:
                    pgid = None

                if pgid is not None:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=2)
        except Exception:
            try:
                if process.poll() is None:
                    try:
                        pgid = os.getpgid(process.pid)
                    except Exception:
                        pgid = None
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        process.kill()
            except Exception:
                logger.debug("Unable to terminate process %s", getattr(process, "pid", "unknown"))
