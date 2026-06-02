from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.config import (
    DEFAULT_LLM_MIN_P,
    DEFAULT_LLM_PRESENCE_PENALTY,
    DEFAULT_LLM_REPETITION_PENALTY,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_TOP_P,
    ENABLE_LOCAL_VLM_REPAIR,
    ENABLE_QWEN_OCR_FALLBACK,
    KEEP_EXTRACTION_DEBUG_ARTIFACTS,
    MARKER_TIMEOUT_SECONDS,
)
from app.models.schema import DocumentModel
from app.models.schema import JobStage
from app.services.job_store import JobStore
from app.services.pdf_extraction import PDFExtractor
from app.services.pdf_extraction.models import PDFTypeDetectionResult
from app.services.pdf_extraction.qwen_ocr_fallback import QwenFullPageOCRFallback
from app.services.profiler import PipelineProfiler
from app.services.translation_debug import write_translation_comparison_report
from app.services.translation_subprocess import run_translation_subprocess

logger = logging.getLogger(__name__)


class TranslationPipeline:
    def __init__(self, job_store: JobStore) -> None:
        self.job_store = job_store
        self.pdf_extractor = PDFExtractor()
        self.qwen_ocr_fallback = QwenFullPageOCRFallback()
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
        self._run_pipeline(job_id, pdf_path, settings)

    def _run_pipeline(self, job_id: str, pdf_path: Path, settings: dict) -> None:
        profile_enabled = bool(settings.get("profile_pipeline", False))
        profiler = PipelineProfiler(enabled=profile_enabled)
        job_dir = self.job_store.get_job_dir(job_id)
        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "structured.json"
        md_path = artifacts_dir / "translated.md"
        source_md_path = artifacts_dir / "source.md"
        extraction_json_path = artifacts_dir / "extraction_result.json"
        marker_detection_path = artifacts_dir / "marker_detection.json"
        pdf_readable = artifacts_dir / "translated_readable.pdf"
        pdf_faithful = artifacts_dir / "translated_faithful.pdf"
        extraction_mode = str(settings.get("extraction_mode", "auto"))
        keep_debug_artifacts = bool(settings.get("keep_debug_artifacts", KEEP_EXTRACTION_DEBUG_ARTIFACTS))
        translation_metadata = {
            "model": str(settings.get("model", "")),
            "temperature": float(settings.get("temperature", DEFAULT_LLM_TEMPERATURE)),
            "top_p": float(settings.get("top_p", DEFAULT_LLM_TOP_P)),
            "top_k": int(settings.get("top_k", DEFAULT_LLM_TOP_K)),
            "min_p": float(settings.get("min_p", DEFAULT_LLM_MIN_P)),
            "presence_penalty": float(
                settings.get("presence_penalty", DEFAULT_LLM_PRESENCE_PENALTY)
            ),
            "repetition_penalty": float(
                settings.get("repetition_penalty", DEFAULT_LLM_REPETITION_PENALTY)
            ),
        }

        try:
            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id)
                return
            if not self._update_status(
                job_id,
                stage=JobStage.EXTRACTION,
                progress=0.08,
                message="Detecting PDF text quality",
            ):
                return

            with profiler.step("pdf_text_quality_detection"):
                detection = self.pdf_extractor.detector.detect(pdf_path)

            if self._should_bypass_marker_for_qwen(detection, settings):
                if not self._update_status(
                    job_id,
                    stage=JobStage.OCR_LAYOUT,
                    progress=0.18,
                    message=(
                        f"PDF classified as {detection.classification}; "
                        "skipping Marker and running Surya layout before Qwen full-page OCR"
                    ),
                    translation={
                        **translation_metadata,
                        "pdf_classification": detection.classification,
                        "marker_mode": "skipped_for_surya_qwen_full_page_ocr",
                        "fallback_engine": "surya_layout_qwen_full_page_ocr",
                        "ocr_used": True,
                        "force_ocr": False,
                        "strip_existing_ocr": False,
                        "warnings": detection.warnings,
                    },
                ):
                    return

                def on_qwen_progress(event: dict) -> None:
                    update = self._qwen_ocr_progress_from_event(event)
                    if update is None:
                        return
                    progress, message = update
                    self._update_status(
                        job_id,
                        stage=JobStage.OCR_LAYOUT,
                        progress=progress,
                        message=message,
                    )

                with profiler.step("surya_qwen_full_page_ocr_direct"):
                    result = self.qwen_ocr_fallback.extract(
                        pdf_path=pdf_path,
                        job_dir=job_dir,
                        pdf_classification=detection.classification,
                        marker_warnings=detection.warnings,
                        marker_metadata={
                            "pdf_classification": detection.classification,
                            "extraction_mode": extraction_mode,
                            "marker_mode": "skipped_for_surya_qwen_full_page_ocr",
                            "marker_skipped": True,
                            "marker_skip_reason": "poor_text_quality",
                            "fallback_engine": "surya_layout_qwen_full_page_ocr",
                            "detection": self._detection_metadata(detection),
                        },
                        settings=settings,
                        profiler=profiler,
                        cancel_requested=lambda: self._is_cancelled(job_id),
                        on_process_started=lambda process: self._register_process(job_id, process),
                        on_process_finished=lambda process: self._unregister_process(job_id, process),
                        on_ocr_progress=on_qwen_progress,
                    )
            else:
                with profiler.step("marker_extraction"):
                    result = self.pdf_extractor.extract(
                        pdf_path=pdf_path,
                        mode=extraction_mode,
                        use_local_vlm_repair=bool(settings.get("use_local_vlm_repair", ENABLE_LOCAL_VLM_REPAIR)),
                        keep_debug_artifacts=keep_debug_artifacts,
                        job_dir=job_dir,
                        timeout=int(settings.get("marker_timeout_seconds", MARKER_TIMEOUT_SECONDS)),
                        cancel_requested=lambda: self._is_cancelled(job_id),
                        on_process_started=lambda process: self._register_process(job_id, process),
                        on_process_finished=lambda process: self._unregister_process(job_id, process),
                        on_detection_complete=lambda detection, marker_mode: self._update_status(
                            job_id,
                            stage=JobStage.EXTRACTION,
                            progress=0.14,
                            message=(
                                f"PDF classified as {detection.classification}; "
                                f"running Marker mode {marker_mode}"
                            ),
                            translation={
                                **translation_metadata,
                                "pdf_classification": detection.classification,
                                "marker_mode": marker_mode,
                                "ocr_used": marker_mode in {"force_ocr", "strip_existing_ocr_force_ocr"},
                                "force_ocr": marker_mode in {"force_ocr", "strip_existing_ocr_force_ocr"},
                                "strip_existing_ocr": marker_mode == "strip_existing_ocr_force_ocr",
                                "warnings": detection.warnings,
                            },
                        ),
                    )
                if self._is_cancelled(job_id):
                    self._mark_cancelled(job_id)
                    return
                if self._should_run_qwen_ocr_fallback(result, settings):
                    if not self._update_status(
                        job_id,
                        stage=JobStage.OCR_LAYOUT,
                        progress=0.24,
                        message=(
                            f"Marker first pass classified as {result.pdf_classification}; "
                            "running Qwen full-page OCR fallback"
                        ),
                        translation={
                            **translation_metadata,
                            "pdf_classification": result.pdf_classification,
                            "marker_mode": str(result.metadata.get("marker_mode", "")),
                            "fallback_engine": "qwen_full_page_ocr",
                            "ocr_used": True,
                            "warnings": result.warnings,
                        },
                    ):
                        return

                    def on_qwen_progress(event: dict) -> None:
                        update = self._qwen_ocr_progress_from_event(event)
                        if update is None:
                            return
                        progress, message = update
                        self._update_status(
                            job_id,
                            stage=JobStage.OCR_LAYOUT,
                            progress=progress,
                            message=message,
                        )

                    with profiler.step("qwen_full_page_ocr_fallback"):
                        result = self.qwen_ocr_fallback.extract(
                            pdf_path=pdf_path,
                            job_dir=job_dir,
                            pdf_classification=result.pdf_classification,
                            marker_warnings=result.warnings,
                            marker_metadata=result.metadata,
                            settings=settings,
                            profiler=profiler,
                            cancel_requested=lambda: self._is_cancelled(job_id),
                            on_process_started=lambda process: self._register_process(job_id, process),
                            on_process_finished=lambda process: self._unregister_process(job_id, process),
                            on_ocr_progress=on_qwen_progress,
                        )

            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id)
                return
            if result.document is None:
                raise RuntimeError("Marker extraction did not return a structured document.")

            if not self._update_status(
                job_id,
                stage=JobStage.STRUCTURE,
                progress=0.52,
                message=(
                    f"Extraction produced {len(result.blocks)} block(s); "
                    f"classified as {result.pdf_classification}"
                ),
            ):
                return

            with profiler.step("marker_artifact_write"):
                json_path.write_text(result.document.model_dump_json(indent=2), encoding="utf-8")
                source_md_path.write_text(result.markdown.strip() + "\n", encoding="utf-8")
                marker_detection = result.metadata.get("detection", {})
                marker_detection_path.write_text(json.dumps(marker_detection, indent=2), encoding="utf-8")
                extraction_payload = {
                    "markdown": result.markdown,
                    "chunks": [asdict(chunk) for chunk in result.chunks],
                    "pages": result.pages,
                    "blocks": result.blocks,
                    "metadata": result.metadata,
                    "extraction_mode": result.extraction_mode,
                    "pdf_classification": result.pdf_classification,
                    "used_ocr": result.used_ocr,
                    "used_force_ocr": result.used_force_ocr,
                    "stripped_existing_ocr": result.stripped_existing_ocr,
                    "used_local_vlm_repair": result.used_local_vlm_repair,
                    "warnings": result.warnings,
                }
                extraction_json_path.write_text(json.dumps(extraction_payload, indent=2), encoding="utf-8")

            artifacts = {
                "json": str(json_path),
                "markdown": str(md_path),
                "source_markdown": str(source_md_path),
                "debug_markdown": str(md_path),
                "pdf_readable": str(pdf_readable),
                "pdf_faithful": str(pdf_faithful),
                "extraction_result": str(extraction_json_path),
                "marker_detection": str(marker_detection_path),
                "logical_translation_chunks": str(result.metadata.get("ocr_logical_chunks_path", "")),
            }
            self._update_status(
                job_id,
                stage=JobStage.TRANSLATION,
                progress=0.66,
                message=(
                    f"Translation started. Marker mode: {result.metadata.get('marker_mode', 'unknown')}; "
                    f"OCR used: {str(result.used_ocr).lower()}"
                ),
                translation={
                    **translation_metadata,
                    "pdf_classification": result.pdf_classification,
                    "marker_mode": str(result.metadata.get("marker_mode", "")),
                    "fallback_engine": str(result.metadata.get("fallback_engine", "")),
                    "ocr_used": result.used_ocr,
                    "force_ocr": result.used_force_ocr,
                    "strip_existing_ocr": result.stripped_existing_ocr,
                    "local_vlm_repair_used": result.used_local_vlm_repair,
                    "extraction_time_seconds": result.metadata.get("extraction_time_seconds"),
                    "warnings": result.warnings,
                },
                artifacts=artifacts,
            )

            logger.info("Translation started")
            logger.info("Translation chunks: %s", len(result.chunks))
            translation_started = time.perf_counter()

            def on_chunk_translated(index: int, total: int, preview: str) -> None:
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
            logger.info("Translation completed in %.2f seconds", time.perf_counter() - translation_started)

            if self._is_cancelled(job_id):
                self._mark_cancelled(job_id)
                return

            translated_document = DocumentModel.model_validate_json(json_path.read_text(encoding="utf-8"))
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

            artifacts.update(
                {
                    "profile_json": str(profile_json) if profile_enabled else "",
                    "profile_csv": str(profile_csv) if profile_enabled else "",
                    "profile_summary": str(profile_summary) if profile_enabled else "",
                    "translation_comparison_json": str(comparison_json),
                    "translation_comparison_md": str(comparison_md),
                }
            )
            self._update_status(
                job_id,
                stage=JobStage.COMPLETE,
                progress=1.0,
                message=f"Done at {datetime.utcnow().isoformat()}Z. PDFs are generated on demand when downloaded.",
                translation={
                    **translation_metadata,
                    "pdf_classification": result.pdf_classification,
                    "marker_mode": str(result.metadata.get("marker_mode", "")),
                    "fallback_engine": str(result.metadata.get("fallback_engine", "")),
                    "ocr_used": result.used_ocr,
                    "force_ocr": result.used_force_ocr,
                    "strip_existing_ocr": result.stripped_existing_ocr,
                    "local_vlm_repair_used": result.used_local_vlm_repair,
                    "extraction_time_seconds": result.metadata.get("extraction_time_seconds"),
                    "warnings": result.warnings,
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            if self._is_cancelled(job_id) or str(exc) == "Cancelled by user":
                self._mark_cancelled(job_id)
                return
            logger.exception("Marker pipeline failed for job %s", job_id)
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

    def _should_run_qwen_ocr_fallback(self, result, settings: dict) -> bool:
        enabled = bool(settings.get("enable_qwen_ocr_fallback", settings.get("qwen_ocr_fallback", ENABLE_QWEN_OCR_FALLBACK)))
        if not enabled:
            return False
        if str(settings.get("extraction_mode", "auto")) not in {"auto", "auto_repair"}:
            return False
        return result.pdf_classification != "digital_good_text"

    def _should_bypass_marker_for_qwen(self, detection: PDFTypeDetectionResult, settings: dict) -> bool:
        enabled = bool(settings.get("enable_qwen_ocr_fallback", settings.get("qwen_ocr_fallback", ENABLE_QWEN_OCR_FALLBACK)))
        if not enabled:
            return False
        if str(settings.get("extraction_mode", "auto")) not in {"auto", "auto_repair"}:
            return False
        return detection.classification in {"scanned_no_text", "bad_hidden_ocr"}

    def _detection_metadata(self, detection: PDFTypeDetectionResult) -> dict:
        return {
            "embedded_text_chars": detection.embedded_text_chars,
            "embedded_text_words": detection.embedded_text_words,
            "meaningful_page_count": detection.meaningful_page_count,
            "garbled_page_count": detection.garbled_page_count,
            "image_dominant_page_count": detection.image_dominant_page_count,
            "scanned_page_count": detection.scanned_page_count,
            "metadata": detection.metadata,
            "pages": [asdict(page) for page in detection.pages],
        }

    def _qwen_ocr_progress_from_event(self, event: dict) -> tuple[float, str] | None:
        kind = str(event.get("event", ""))
        if kind == "layout_detection_started":
            pages = int(event.get("pages") or 0)
            return 0.2, f"Detecting Surya layout for {pages} page image(s)"
        if kind == "layout_detection_complete":
            pages = int(event.get("pages") or 0)
            regions = int(event.get("reconciled_regions") or event.get("regions") or 0)
            return 0.24, f"Surya layout detected {regions} region(s) across {pages} page(s)"
        if kind == "model_loading":
            return 0.26, "Loading Qwen OCR model"
        if kind == "model_loaded":
            pages = int(event.get("pages") or 0)
            return 0.3, f"Qwen OCR model loaded; processing {pages} full page image(s)"
        if kind == "batch_started":
            batch_index = int(event.get("batch_index") or 1)
            batch_size = int(event.get("batch_size") or 1)
            return 0.32, f"Qwen OCR batch {batch_index}: {batch_size} page image(s)"
        if kind == "page_done":
            index = int(event.get("index") or 0)
            total = int(event.get("total") or 1)
            progress = 0.34 + (0.16 * (index / max(total, 1)))
            chars = int(event.get("chars") or 0)
            return round(min(progress, 0.5), 3), f"Qwen OCR page {index}/{total}: {chars} chars"
        if kind == "complete":
            pages = int(event.get("pages") or 0)
            return 0.5, f"Qwen OCR completed for {pages} page(s)"
        return None

    def _update_status(self, job_id: str, **updates: object) -> bool:
        try:
            self.job_store.update_status(job_id, **updates)
            return True
        except FileNotFoundError:
            logger.info("Job %s no longer exists; stopping background work", job_id)
            return False

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
