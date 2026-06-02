from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from app.config import (
    BASE_DIR,
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
)
from app.models.schema import Block
from app.services.pdf_extraction.models import ExtractionChunk, PDFExtractionResult
from app.services.pdf_inspector import PdfInspector
from app.services.profiler import PipelineProfiler
from app.services.qwen_markdown_parser import QwenMarkdownParser
from app.services.renderer import PageRenderer

logger = logging.getLogger(__name__)


class QwenFullPageOCRFallback:
    """Render full PDF pages and preserve the Markdown emitted by Qwen OCR."""

    def __init__(self) -> None:
        self.inspector = PdfInspector()
        self.renderer = PageRenderer()
        self.markdown_parser = QwenMarkdownParser()

    def extract(
        self,
        *,
        pdf_path: Path,
        job_dir: Path,
        pdf_classification: str,
        marker_warnings: list[str],
        marker_metadata: dict,
        settings: dict,
        profiler: PipelineProfiler | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
        on_process_finished: Callable[[subprocess.Popen], None] | None = None,
        on_ocr_progress: Callable[[dict], None] | None = None,
    ) -> PDFExtractionResult:
        started = time.perf_counter()
        inspection = self.inspector.inspect(pdf_path)
        qwen_dir = job_dir / "qwen_ocr"
        render_dir = qwen_dir / "rendered_pages"
        markdown_dir = qwen_dir / "markdown"
        render_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        dpi = int(settings.get("qwen_ocr_dpi", DEFAULT_QWEN_OCR_DPI))
        image_paths: list[Path] = []
        output_names: list[str] = []
        image_metadata: list[dict] = []
        for page in inspection.pages:
            if cancel_requested is not None and cancel_requested():
                raise RuntimeError("Cancelled by user")
            context = (
                profiler.step("qwen_page_rendering", page=page.page_number)
                if profiler is not None
                else _nullcontext()
            )
            with context:
                rendered = self.renderer.render_page(
                    pdf_path,
                    page.page_number,
                    render_dir / f"page_{page.page_number:04d}.png",
                    dpi=dpi,
                    profiler=profiler,
                    stage_prefix="qwen_page_rendering",
                )
            metadata = self._rendered_page_metadata(rendered)
            metadata.update({"page_number": page.page_number, "render_dpi": dpi})
            image_paths.append(rendered)
            image_metadata.append(metadata)
            output_names.append(f"page_{page.page_number:04d}")

        self._run_qwen_ocr(
            image_paths=image_paths,
            output_dir=markdown_dir,
            output_names=output_names,
            settings=settings,
            cancel_requested=cancel_requested,
            on_process_started=on_process_started,
            on_process_finished=on_process_finished,
            on_ocr_progress=on_ocr_progress,
        )
        document, markdown = self.markdown_parser.build_document_from_markdown_dir(
            inspection=inspection,
            markdown_dir=markdown_dir,
            profiler=profiler,
            strict_page_files=True,
        )
        chunks = self._chunks_from_blocks(document.blocks)
        marker_skipped = bool(marker_metadata.get("marker_skipped"))
        qwen_reason = (
            "PDF text-quality detection classified the document as poor text; Marker was skipped and Qwen full-page OCR was used."
            if marker_skipped
            else "Marker first pass did not classify the document as good digital text; Qwen full-page OCR fallback was used."
        )
        warnings = [*marker_warnings, qwen_reason, *document.warnings]
        document.warnings = warnings
        elapsed = time.perf_counter() - started

        return PDFExtractionResult(
            markdown=markdown,
            chunks=chunks,
            pages=[page.model_dump() for page in document.pages],
            blocks=[block.model_dump() for block in document.blocks],
            metadata={
                "pdf_classification": pdf_classification,
                "extraction_mode": "qwen_full_page_ocr_fallback",
                "marker_mode": "qwen_full_page_ocr_fallback",
                "fallback_engine": "qwen_full_page_ocr",
                "ocr_used": True,
                "force_ocr": False,
                "strip_existing_ocr": False,
                "qwen_ocr_model": str(settings.get("qwen_ocr_model", DEFAULT_QWEN_OCR_MODEL)),
                "qwen_ocr_max_tokens": int(settings.get("qwen_ocr_max_tokens", DEFAULT_QWEN_OCR_MAX_TOKENS)),
                "qwen_ocr_temperature": float(settings.get("qwen_ocr_temperature", DEFAULT_QWEN_OCR_TEMPERATURE)),
                "qwen_ocr_top_p": float(settings.get("qwen_ocr_top_p", DEFAULT_QWEN_OCR_TOP_P)),
                "qwen_ocr_top_k": int(settings.get("qwen_ocr_top_k", DEFAULT_QWEN_OCR_TOP_K)),
                "qwen_ocr_min_p": float(settings.get("qwen_ocr_min_p", DEFAULT_QWEN_OCR_MIN_P)),
                "qwen_ocr_presence_penalty": float(
                    settings.get("qwen_ocr_presence_penalty", DEFAULT_QWEN_OCR_PRESENCE_PENALTY)
                ),
                "qwen_ocr_repetition_penalty": float(
                    settings.get("qwen_ocr_repetition_penalty", DEFAULT_QWEN_OCR_REPETITION_PENALTY)
                ),
                "qwen_ocr_prompt": str(settings.get("qwen_ocr_prompt", DEFAULT_QWEN_OCR_PROMPT)),
                "qwen_ocr_dpi": dpi,
                "qwen_ocr_batch_size": int(settings.get("qwen_ocr_batch_size", DEFAULT_QWEN_OCR_BATCH_SIZE)),
                "qwen_ocr_crop_mode": bool(settings.get("qwen_ocr_crop_mode", DEFAULT_QWEN_OCR_CROP_MODE)),
                "qwen_ocr_min_crops": int(settings.get("qwen_ocr_min_crops", DEFAULT_QWEN_OCR_MIN_CROPS)),
                "qwen_ocr_max_crops": int(settings.get("qwen_ocr_max_crops", DEFAULT_QWEN_OCR_MAX_CROPS)),
                "qwen_ocr_base_size": int(settings.get("qwen_ocr_base_size", DEFAULT_QWEN_OCR_BASE_SIZE)),
                "qwen_ocr_image_size": int(settings.get("qwen_ocr_image_size", DEFAULT_QWEN_OCR_IMAGE_SIZE)),
                "qwen_ocr_skip_repeat": bool(settings.get("qwen_ocr_skip_repeat", DEFAULT_QWEN_OCR_SKIP_REPEAT)),
                "qwen_ocr_ngram_size": int(settings.get("qwen_ocr_ngram_size", DEFAULT_QWEN_OCR_NGRAM_SIZE)),
                "qwen_ocr_ngram_window": int(settings.get("qwen_ocr_ngram_window", DEFAULT_QWEN_OCR_NGRAM_WINDOW)),
                "qwen_ocr_image_mode": "rendered_page_png",
                "qwen_ocr_image_metadata": image_metadata,
                "qwen_ocr_output_dir": str(qwen_dir),
                "marker_first_pass": marker_metadata,
                "detection": marker_metadata.get("detection", {}),
                "extraction_time_seconds": round(elapsed, 3),
            },
            extraction_mode="qwen_full_page_ocr_fallback",
            pdf_classification=pdf_classification,
            used_ocr=True,
            used_force_ocr=False,
            stripped_existing_ocr=False,
            used_local_vlm_repair=False,
            warnings=warnings,
            document=document,
        )

    def _run_qwen_ocr(
        self,
        *,
        image_paths: list[Path],
        output_dir: Path,
        output_names: list[str],
        settings: dict,
        cancel_requested: Callable[[], bool] | None,
        on_process_started: Callable[[subprocess.Popen], None] | None,
        on_process_finished: Callable[[subprocess.Popen], None] | None,
        on_ocr_progress: Callable[[dict], None] | None,
    ) -> None:
        worker = Path(os.getenv("QWEN_OCR_WORKER", str(BASE_DIR / "scripts" / "qwen_ocr_worker.py")))
        cmd = [
            self._resolve_worker_python_executable(),
            str(worker),
            "--model",
            str(settings.get("qwen_ocr_model", DEFAULT_QWEN_OCR_MODEL)),
            "--images-json",
            json.dumps([str(path) for path in image_paths]),
            "--output-dir",
            str(output_dir),
            "--max-tokens",
            str(int(settings.get("qwen_ocr_max_tokens", DEFAULT_QWEN_OCR_MAX_TOKENS))),
            "--temperature",
            str(float(settings.get("qwen_ocr_temperature", DEFAULT_QWEN_OCR_TEMPERATURE))),
            "--top-p",
            str(float(settings.get("qwen_ocr_top_p", DEFAULT_QWEN_OCR_TOP_P))),
            "--top-k",
            str(int(settings.get("qwen_ocr_top_k", DEFAULT_QWEN_OCR_TOP_K))),
            "--min-p",
            str(float(settings.get("qwen_ocr_min_p", DEFAULT_QWEN_OCR_MIN_P))),
            "--presence-penalty",
            str(float(settings.get("qwen_ocr_presence_penalty", DEFAULT_QWEN_OCR_PRESENCE_PENALTY))),
            "--repetition-penalty",
            str(float(settings.get("qwen_ocr_repetition_penalty", DEFAULT_QWEN_OCR_REPETITION_PENALTY))),
            "--prompt",
            str(settings.get("qwen_ocr_prompt", DEFAULT_QWEN_OCR_PROMPT)),
            "--crop-mode",
            "true" if bool(settings.get("qwen_ocr_crop_mode", DEFAULT_QWEN_OCR_CROP_MODE)) else "false",
            "--min-crops",
            str(int(settings.get("qwen_ocr_min_crops", DEFAULT_QWEN_OCR_MIN_CROPS))),
            "--max-crops",
            str(int(settings.get("qwen_ocr_max_crops", DEFAULT_QWEN_OCR_MAX_CROPS))),
            "--base-size",
            str(int(settings.get("qwen_ocr_base_size", DEFAULT_QWEN_OCR_BASE_SIZE))),
            "--image-size",
            str(int(settings.get("qwen_ocr_image_size", DEFAULT_QWEN_OCR_IMAGE_SIZE))),
            "--skip-repeat",
            "true" if bool(settings.get("qwen_ocr_skip_repeat", DEFAULT_QWEN_OCR_SKIP_REPEAT)) else "false",
            "--ngram-size",
            str(int(settings.get("qwen_ocr_ngram_size", DEFAULT_QWEN_OCR_NGRAM_SIZE))),
            "--ngram-window",
            str(int(settings.get("qwen_ocr_ngram_window", DEFAULT_QWEN_OCR_NGRAM_WINDOW))),
            "--batch-size",
            str(int(settings.get("qwen_ocr_batch_size", DEFAULT_QWEN_OCR_BATCH_SIZE))),
            "--enable-thinking",
            "false",
            "--verbose",
            "true",
            "--names-json",
            json.dumps(output_names),
        ]
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Qwen OCR requires a Python environment with mlx-vlm. Set QWEN_OCR_PYTHON.") from exc

        if on_process_started is not None:
            on_process_started(process)
        stdout, stderr = self._communicate_with_cancel(
            process,
            cancel_requested=cancel_requested,
            on_stdout_line=self._ocr_worker_line_handler(on_ocr_progress),
        )
        if on_process_finished is not None:
            on_process_finished(process)
        if process.returncode != 0:
            if process.returncode == -15 and cancel_requested is not None and cancel_requested():
                raise RuntimeError("Cancelled by user")
            raise RuntimeError(f"Qwen OCR failed: {(stderr or stdout)[-2000:]}")

    def _resolve_worker_python_executable(self) -> str:
        configured = os.getenv("QWEN_OCR_PYTHON")
        if not configured:
            return sys.executable
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)
        logger.warning("Ignoring QWEN_OCR_PYTHON because it does not exist: %s", configured)
        return sys.executable

    def _ocr_worker_line_handler(self, callback: Callable[[dict], None] | None) -> Callable[[str], None]:
        def handle(line: str) -> None:
            stripped = line.strip()
            if not stripped:
                return
            logger.info("Qwen OCR worker: %s", stripped)
            if callback is None or '"event"' not in line:
                return
            try:
                callback(json.loads(line))
            except json.JSONDecodeError:
                return

        return handle

    def _communicate_with_cancel(
        self,
        process: subprocess.Popen,
        *,
        cancel_requested: Callable[[], bool] | None,
        on_stdout_line: Callable[[str], None],
    ) -> tuple[str, str]:
        result = {"stdout": "", "stderr": ""}

        def target() -> None:
            stdout_chunks: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                stdout_chunks.append(line)
                on_stdout_line(line)
            result["stderr"] = process.stderr.read() if process.stderr is not None else ""
            process.wait()
            result["stdout"] = "".join(stdout_chunks)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0.2)
            if cancel_requested is not None and cancel_requested():
                process.terminate()
        thread.join()
        return result["stdout"], result["stderr"]

    def _rendered_page_metadata(self, rendered_path: Path) -> dict:
        with Image.open(rendered_path) as image:
            return {
                "input_path": str(rendered_path),
                "ocr_image_path": str(rendered_path),
                "ocr_image_mode": "rendered_page_png",
                "original_width": image.width,
                "original_height": image.height,
                "ocr_image_width": image.width,
                "ocr_image_height": image.height,
            }

    def _chunks_from_blocks(self, blocks: list[Block]) -> list[ExtractionChunk]:
        chunks: list[ExtractionChunk] = []
        for index, block in enumerate(blocks, start=1):
            if not block.text.strip():
                continue
            chunks.append(
                ExtractionChunk(
                    chunk_id=f"qwen-ocr-{index}",
                    page_number=block.page_number,
                    block_ids=[block.id],
                    block_type=block.block_type.value,
                    bbox=block.bbox.model_dump() if block.bbox else None,
                    polygon=None,
                    original_text=block.text.strip(),
                )
            )
        return chunks


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_args) -> None:
        return None
