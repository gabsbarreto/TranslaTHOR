from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from app.models.schema import SourceType
from app.config import BASE_DIR
from app.services.markdown_builder import MarkdownBuilder as AppMarkdownBuilder
from app.services.pdf_extraction.deepseek_fallback import DeepSeekFallbackOCR
from app.services.pdf_extraction.local_vlm_service import LocalVLMRepairService
from app.services.pdf_extraction.markdown_builder import MarkerDocumentBuilder
from app.services.pdf_extraction.models import (
    ExtractionMode,
    MarkerMode,
    PDFExtractionResult,
    PDFTypeDetectionResult,
)
from app.services.pdf_extraction.pdf_type_detector import PDFTypeDetector

logger = logging.getLogger(__name__)


class MarkerExecutionError(RuntimeError):
    def __init__(self, message: str, return_code: int, stdout: str, stderr: str) -> None:
        super().__init__(message)
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class PDFExtractor:
    def __init__(
        self,
        detector: PDFTypeDetector | None = None,
        document_builder: MarkerDocumentBuilder | None = None,
        local_vlm_service: LocalVLMRepairService | None = None,
        deepseek_fallback: DeepSeekFallbackOCR | None = None,
    ) -> None:
        self.detector = detector or PDFTypeDetector()
        self.document_builder = document_builder or MarkerDocumentBuilder()
        self.local_vlm_service = local_vlm_service or LocalVLMRepairService()
        self.deepseek_fallback = deepseek_fallback or DeepSeekFallbackOCR()

    def extract(
        self,
        pdf_path: Path,
        mode: str = "auto",
        use_local_vlm_repair: bool = False,
        use_deepseek_fallback: bool = False,
        keep_debug_artifacts: bool = False,
        job_dir: Path | None = None,
        timeout: int | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
        on_process_finished: Callable[[subprocess.Popen], None] | None = None,
        on_detection_complete: Callable[[PDFTypeDetectionResult, MarkerMode], None] | None = None,
    ) -> PDFExtractionResult:
        started = time.perf_counter()
        mode = self._normalize_mode(mode)
        use_local_vlm_repair = use_local_vlm_repair or mode == "auto_repair"

        logger.info("PDF extraction started")
        logger.info("PDF path: %s", pdf_path)
        detection = self.detector.detect(pdf_path)
        logger.info("PDF pages: %s", detection.page_count)
        logger.info("PDF classified as: %s", detection.classification)
        logger.info("Extraction mode requested: %s", mode)

        marker_mode = self._select_marker_mode(mode, detection.classification)
        requested_marker_mode = marker_mode
        output_format = "json"
        logger.info("Marker mode selected: %s", marker_mode)
        logger.info("Marker output format: %s", output_format)
        if on_detection_complete is not None:
            on_detection_complete(detection, marker_mode)
        warnings = list(detection.warnings)

        temp_context = (
            tempfile.TemporaryDirectory(prefix="marker_extract_")
            if job_dir is None or not keep_debug_artifacts
            else None
        )
        output_root = (
            Path(temp_context.name)
            if temp_context is not None
            else (job_dir / "marker")
        )
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            marker_payload, marker_mode, marker_retry_metadata = self._run_marker_with_recovery(
                pdf_path=pdf_path,
                output_dir=output_root,
                output_format=output_format,
                marker_mode=marker_mode,
                classification=detection.classification,
                mode=mode,
                keep_debug_artifacts=keep_debug_artifacts,
                timeout=timeout,
                cancel_requested=cancel_requested,
                on_process_started=on_process_started,
                on_process_finished=on_process_finished,
                warnings=warnings,
            )
            used_force_ocr = marker_mode in {"force_ocr", "strip_existing_ocr_force_ocr"}
            stripped_existing_ocr = marker_mode == "strip_existing_ocr_force_ocr"
            used_ocr = used_force_ocr or (
                marker_mode == "normal"
                and detection.classification in {"scanned_no_text", "bad_hidden_ocr", "mixed", "unknown"}
            )
            parser_metadata = {
                "pdf_classification": detection.classification,
                "extraction_mode": mode,
                "marker_mode": marker_mode,
                "marker_requested_mode": requested_marker_mode,
                "ocr_used": used_ocr,
                "force_ocr": used_force_ocr,
                "strip_existing_ocr": stripped_existing_ocr,
                **marker_retry_metadata,
                "detection": {
                    "embedded_text_chars": detection.embedded_text_chars,
                    "embedded_text_words": detection.embedded_text_words,
                    "meaningful_page_count": detection.meaningful_page_count,
                    "garbled_page_count": detection.garbled_page_count,
                    "image_dominant_page_count": detection.image_dominant_page_count,
                    "scanned_page_count": detection.scanned_page_count,
                    "metadata": detection.metadata,
                    "pages": [asdict(page) for page in detection.pages],
                },
            }
            source_type = SourceType.OCR if used_ocr else SourceType.EMBEDDED
            document, markdown, chunks = self.document_builder.build_document(
                marker_payload=marker_payload,
                detection=detection,
                filename=pdf_path.name,
                source_type=source_type,
                parser_metadata=parser_metadata,
                warnings=warnings,
            )

            repaired_count = 0
            if use_local_vlm_repair:
                logger.info("Local VLM repair enabled: true")
                debug_dir = (job_dir / "artifacts" / "debug") if keep_debug_artifacts and job_dir is not None else None
                repair_context = {
                    "pdf_classification": detection.classification,
                    "marker_mode": marker_mode,
                    "marker_requested_mode": requested_marker_mode,
                    "detected_language": document.metadata.detected_language,
                    "suspicious_hidden_ocr": bool(detection.metadata.get("suspicious_hidden_ocr")),
                    "marker_fallback_to_normal": bool(marker_retry_metadata.get("marker_fallback_to_normal")),
                }
                repaired_count, repair_warnings = self.local_vlm_service.repair_blocks(
                    document.blocks,
                    debug_dir,
                    extraction_context=repair_context,
                )
                warnings.extend(repair_warnings)
                if repaired_count:
                    document.warnings = warnings
                    markdown = AppMarkdownBuilder().build(document)
            else:
                logger.info("Local VLM repair enabled: false")

            deepseek_used = False
            if use_deepseek_fallback or mode == "deepseek_fallback":
                fallback_result = self.deepseek_fallback.repair_selected_blocks(document)
                deepseek_used = fallback_result.used
                warnings.extend(fallback_result.warnings)

            elapsed = time.perf_counter() - started
            logger.info("OCR used: %s", used_ocr)
            logger.info("Force OCR used: %s", used_force_ocr)
            logger.info("Existing OCR stripped: %s", stripped_existing_ocr)
            logger.info("Local VLM repair used: %s", repaired_count > 0)
            logger.info("DeepSeek fallback enabled: %s", use_deepseek_fallback or mode == "deepseek_fallback")
            logger.info("DeepSeek fallback used: %s", deepseek_used)
            logger.info("Pages processed: %s", len(document.pages))
            logger.info("Blocks extracted: %s", len(document.blocks))
            logger.info("Chunks created: %s", len(chunks))
            logger.info("Extraction completed in %.2f seconds", elapsed)

            return PDFExtractionResult(
                markdown=markdown,
                chunks=chunks,
                pages=[page.model_dump() for page in document.pages],
                blocks=[block.model_dump() for block in document.blocks],
                metadata={
                    **parser_metadata,
                    "marker_output_dir": str(output_root) if keep_debug_artifacts else "",
                    "extraction_time_seconds": round(elapsed, 3),
                },
                extraction_mode=mode,
                pdf_classification=detection.classification,
                used_ocr=used_ocr,
                used_force_ocr=used_force_ocr,
                stripped_existing_ocr=stripped_existing_ocr,
                used_local_vlm_repair=repaired_count > 0,
                used_deepseek_fallback=deepseek_used,
                warnings=warnings,
                document=document,
            )
        finally:
            if temp_context is not None:
                temp_context.cleanup()

    def _normalize_mode(self, mode: str) -> ExtractionMode:
        allowed = {"auto", "digital", "scanned", "strip_and_force_ocr", "auto_repair", "deepseek_fallback"}
        return mode if mode in allowed else "auto"  # type: ignore[return-value]

    def _select_marker_mode(self, mode: ExtractionMode, classification: str) -> MarkerMode:
        if mode == "digital":
            return "text_only"
        if mode in {"auto", "auto_repair"}:
            return "text_only"
        if mode == "scanned":
            return "force_ocr"
        if mode in {"strip_and_force_ocr", "deepseek_fallback"}:
            return "strip_existing_ocr_force_ocr"
        if classification == "digital_good_text":
            return "normal"
        if classification == "bad_hidden_ocr":
            return "strip_existing_ocr_force_ocr"
        if classification in {"scanned_no_text", "mixed", "unknown"}:
            return "force_ocr"
        return "normal"

    def _run_marker(
        self,
        *,
        pdf_path: Path,
        output_dir: Path,
        output_format: str,
        marker_mode: MarkerMode,
        keep_debug_artifacts: bool,
        timeout: int | None,
        cancel_requested: Callable[[], bool] | None,
        on_process_started: Callable[[subprocess.Popen], None] | None,
        on_process_finished: Callable[[subprocess.Popen], None] | None,
        env_overrides: dict[str, str] | None = None,
    ):
        marker_bin = os.getenv("MARKER_BIN") or self._default_marker_bin()
        cmd = [
            marker_bin,
            str(pdf_path),
            "--output_dir",
            str(output_dir),
            "--output_format",
            output_format,
        ]
        if marker_mode in {"force_ocr", "strip_existing_ocr_force_ocr"}:
            cmd.append("--force_ocr")
        if marker_mode == "strip_existing_ocr_force_ocr":
            cmd.append("--strip_existing_ocr")
        if marker_mode == "text_only":
            cmd.append("--disable_ocr")
        if keep_debug_artifacts:
            cmd.append("--debug")

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        if on_process_started is not None:
            on_process_started(process)
        try:
            deadline = time.monotonic() + timeout if timeout else None
            while process.poll() is None:
                if cancel_requested is not None and cancel_requested():
                    process.terminate()
                    raise RuntimeError("Cancelled by user")
                if deadline is not None and time.monotonic() > deadline:
                    process.terminate()
                    try:
                        stdout, stderr = process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                    raise RuntimeError(f"Marker timed out after {timeout} seconds: {stderr[-2000:] or stdout[-2000:]}")
                time.sleep(0.2)
            stdout, stderr = process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise RuntimeError(f"Marker timed out after {timeout} seconds: {stderr[-2000:]}")
        finally:
            if on_process_finished is not None:
                on_process_finished(process)

        if process.returncode != 0:
            self._write_marker_failure(output_dir, cmd, process.returncode, stdout, stderr)
            message = f"Marker failed with exit code {process.returncode}: {stderr[-4000:] or stdout[-4000:]}"
            raise MarkerExecutionError(message, process.returncode, stdout, stderr)

        payload_path = self._find_marker_payload(output_dir, output_format)
        if payload_path is None:
            self._write_marker_failure(output_dir, cmd, process.returncode or 0, stdout, stderr)
            raise RuntimeError(f"Marker completed but no {output_format} output was found in {output_dir}")
        return json.loads(payload_path.read_text(encoding="utf-8"))

    def _run_marker_with_recovery(
        self,
        *,
        pdf_path: Path,
        output_dir: Path,
        output_format: str,
        marker_mode: MarkerMode,
        classification: str,
        mode: ExtractionMode,
        keep_debug_artifacts: bool,
        timeout: int | None,
        cancel_requested: Callable[[], bool] | None,
        on_process_started: Callable[[subprocess.Popen], None] | None,
        on_process_finished: Callable[[subprocess.Popen], None] | None,
        warnings: list[str],
    ):
        retry_metadata = {
            "marker_retried_on_cpu": False,
            "marker_fallback_to_normal": False,
        }
        try:
            return (
                self._run_marker(
                    pdf_path=pdf_path,
                    output_dir=output_dir,
                    output_format=output_format,
                    marker_mode=marker_mode,
                    keep_debug_artifacts=keep_debug_artifacts,
                    timeout=timeout,
                    cancel_requested=cancel_requested,
                    on_process_started=on_process_started,
                    on_process_finished=on_process_finished,
                ),
                marker_mode,
                retry_metadata,
            )
        except MarkerExecutionError as exc:
            if marker_mode not in {"force_ocr", "strip_existing_ocr_force_ocr"}:
                raise
            if self._looks_like_accelerator_failure(exc):
                warnings.append(
                    "Marker/Surya failed on the current accelerator while OCR was enabled; retrying the same Marker OCR mode on CPU."
                )
                cpu_output_dir = output_dir / "retry_cpu"
                try:
                    payload = self._run_marker(
                        pdf_path=pdf_path,
                        output_dir=cpu_output_dir,
                        output_format=output_format,
                        marker_mode=marker_mode,
                        keep_debug_artifacts=keep_debug_artifacts,
                        timeout=timeout,
                        cancel_requested=cancel_requested,
                        on_process_started=on_process_started,
                        on_process_finished=on_process_finished,
                        env_overrides={"TORCH_DEVICE": "cpu", "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                    )
                    retry_metadata["marker_retried_on_cpu"] = True
                    return payload, marker_mode, retry_metadata
                except MarkerExecutionError as cpu_exc:
                    warnings.append(f"Marker CPU OCR retry also failed: {self._short_error(cpu_exc)}")
                    exc = cpu_exc

            if classification == "bad_hidden_ocr" and mode in {"auto", "auto_repair", "strip_and_force_ocr"}:
                warnings.append(
                    "Falling back to Marker normal mode using the existing hidden OCR layer because forced OCR failed. "
                    "Extraction can continue, but OCR quality may need local LLM repair."
                )
                fallback_output_dir = output_dir / "fallback_normal"
                payload = self._run_marker(
                    pdf_path=pdf_path,
                    output_dir=fallback_output_dir,
                    output_format=output_format,
                    marker_mode="normal",
                    keep_debug_artifacts=keep_debug_artifacts,
                    timeout=timeout,
                    cancel_requested=cancel_requested,
                    on_process_started=on_process_started,
                    on_process_finished=on_process_finished,
                )
                retry_metadata["marker_fallback_to_normal"] = True
                return payload, "normal", retry_metadata
            raise exc

    def _looks_like_accelerator_failure(self, exc: MarkerExecutionError) -> bool:
        text = f"{exc.stderr}\n{exc.stdout}".lower()
        return "torch.acceleratorerror" in text or "mps" in text or "accelerator" in text

    def _short_error(self, exc: MarkerExecutionError) -> str:
        detail = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
        return detail[-1][-500:] if detail else str(exc)[-500:]

    def _find_marker_payload(self, output_dir: Path, output_format: str) -> Path | None:
        suffix = ".json" if output_format in {"json", "chunks"} else ".md"
        candidates = sorted(output_dir.rglob(f"*{suffix}"), key=lambda path: (len(path.parts), path.name))
        if output_format in {"json", "chunks"}:
            candidates = [
                path for path in candidates
                if path.name not in {"metadata.json", "debug_data.json"} and "debug" not in path.parts
            ] or candidates
        return candidates[0] if candidates else None

    def _default_marker_bin(self) -> str:
        isolated_marker = BASE_DIR / ".venv-marker" / "bin" / "marker_single"
        if isolated_marker.exists():
            return str(isolated_marker)
        return "marker_single"

    def _write_marker_failure(self, output_dir: Path, cmd: list[str], return_code: int, stdout: str, stderr: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cmd": cmd,
            "return_code": return_code,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
        (output_dir / "marker_failure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
