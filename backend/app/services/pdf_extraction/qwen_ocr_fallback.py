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

from PIL import Image, ImageDraw

from app.config import (
    BASE_DIR,
    DEFAULT_QWEN_OCR_BASE_SIZE,
    DEFAULT_QWEN_OCR_BATCH_SIZE,
    DEFAULT_QWEN_OCR_CROP_MODE,
    DEFAULT_QWEN_OCR_DPI,
    DEFAULT_QWEN_OCR_FIRST_PAGE_BOTTOM_MASK_RATIO,
    DEFAULT_QWEN_OCR_FIRST_PAGE_TOP_MASK_RATIO,
    DEFAULT_QWEN_OCR_IMAGE_SCALE,
    DEFAULT_QWEN_OCR_IMAGE_SIZE,
    DEFAULT_QWEN_OCR_JPEG_QUALITY,
    DEFAULT_QWEN_OCR_LEFT_MASK_RATIO,
    DEFAULT_QWEN_OCR_MASK_MARGINS,
    DEFAULT_QWEN_OCR_MAX_CROPS,
    DEFAULT_QWEN_OCR_MAX_TOKENS,
    DEFAULT_QWEN_OCR_MIN_CROPS,
    DEFAULT_QWEN_OCR_MIN_P,
    DEFAULT_QWEN_OCR_MODEL,
    DEFAULT_QWEN_OCR_NGRAM_SIZE,
    DEFAULT_QWEN_OCR_NGRAM_WINDOW,
    DEFAULT_QWEN_OCR_OTHER_PAGE_BOTTOM_MASK_RATIO,
    DEFAULT_QWEN_OCR_OTHER_PAGE_TOP_MASK_RATIO,
    DEFAULT_QWEN_OCR_PRESENCE_PENALTY,
    DEFAULT_QWEN_OCR_PROMPT,
    DEFAULT_QWEN_OCR_REPETITION_PENALTY,
    DEFAULT_QWEN_OCR_RIGHT_MASK_RATIO,
    DEFAULT_QWEN_OCR_SKIP_REPEAT,
    DEFAULT_QWEN_OCR_TEMPERATURE,
    DEFAULT_QWEN_OCR_TOP_K,
    DEFAULT_QWEN_OCR_TOP_P,
)
from app.models.schema import Block
from app.services.deepseek_ocr_pipeline import DeepSeekOcrPipeline
from app.services.markdown_builder import MarkdownBuilder as AppMarkdownBuilder
from app.services.pdf_extraction.models import ExtractionChunk, PDFExtractionResult
from app.services.pdf_inspector import PdfInspector
from app.services.profiler import PipelineProfiler
from app.services.renderer import PageRenderer

logger = logging.getLogger(__name__)


class QwenFullPageOCRFallback:
    """Full-page Qwen VLM OCR fallback using the qwen-ocr branch image profile."""

    def __init__(self) -> None:
        self.inspector = PdfInspector()
        self.renderer = PageRenderer()
        self.markdown_parser = DeepSeekOcrPipeline()
        self.markdown_builder = AppMarkdownBuilder()

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
        image_dir = qwen_dir / "compressed_pages"
        markdown_dir = qwen_dir / "markdown"
        for path in (render_dir, image_dir, markdown_dir):
            path.mkdir(parents=True, exist_ok=True)

        dpi = int(settings.get("qwen_ocr_dpi", DEFAULT_QWEN_OCR_DPI))
        scale = float(settings.get("qwen_ocr_image_scale", DEFAULT_QWEN_OCR_IMAGE_SCALE))
        quality = int(settings.get("qwen_ocr_jpeg_quality", DEFAULT_QWEN_OCR_JPEG_QUALITY))
        mask_margins = bool(settings.get("qwen_ocr_mask_margins", DEFAULT_QWEN_OCR_MASK_MARGINS))
        mask_config = self._margin_mask_config(settings)

        image_paths: list[Path] = []
        output_names: list[str] = []
        image_metadata: list[dict] = []
        for page in inspection.pages:
            if cancel_requested is not None and cancel_requested():
                raise RuntimeError("Cancelled by user")
            with profiler.step("qwen_page_rendering", page=page.page_number) if profiler is not None else _nullcontext():
                rendered = self.renderer.render_page(
                    pdf_path,
                    page.page_number,
                    render_dir / f"page_{page.page_number:04d}.png",
                    dpi=dpi,
                    profiler=profiler,
                    stage_prefix="qwen_page_rendering",
                )
            compressed = image_dir / f"page_{page.page_number:04d}.jpg"
            metadata = self._save_compressed_full_page(
                input_path=rendered,
                output_path=compressed,
                scale=scale,
                quality=quality,
                page_number=page.page_number,
                mask_margins=mask_margins,
                mask_config=mask_config,
            )
            metadata.update({"page_number": page.page_number, "render_dpi": dpi})
            image_metadata.append(metadata)
            image_paths.append(compressed)
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

        document, marker_md = self.markdown_parser.build_document_from_markdown_dir(
            inspection=inspection,
            markdown_dir=markdown_dir,
            profiler=profiler,
            strict_page_files=False,
            warning_message=(
                "Parsed with Qwen 3.5 4B MLX-VLM full-page OCR fallback. "
                "Page images were JPEG compressed at 75 percent scale and quality by default."
            ),
            include_page_markers=False,
            sanitize_ocr_markdown=True,
            merge_page_continuations=True,
        )
        document.metadata.translation.update(
            {
                "suppress_page_markers": True,
                "ocr_markdown_sanitized": True,
                "page_continuation_merge": True,
            }
        )
        for block in document.blocks:
            block.metadata.setdefault("parser", "qwen_full_page_ocr")
        markdown = self.markdown_builder.build(document, marker_md)
        chunks = self._chunks_from_blocks(document.blocks)
        elapsed = time.perf_counter() - started
        warnings = [
            *marker_warnings,
            "Marker first pass did not classify the document as good digital text; Qwen full-page OCR fallback was used.",
            *document.warnings,
        ]
        document.warnings = warnings

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
                "qwen_ocr_temperature": float(
                    settings.get("qwen_ocr_temperature", DEFAULT_QWEN_OCR_TEMPERATURE)
                ),
                "qwen_ocr_top_p": float(settings.get("qwen_ocr_top_p", DEFAULT_QWEN_OCR_TOP_P)),
                "qwen_ocr_top_k": int(settings.get("qwen_ocr_top_k", DEFAULT_QWEN_OCR_TOP_K)),
                "qwen_ocr_min_p": float(settings.get("qwen_ocr_min_p", DEFAULT_QWEN_OCR_MIN_P)),
                "qwen_ocr_presence_penalty": float(
                    settings.get("qwen_ocr_presence_penalty", DEFAULT_QWEN_OCR_PRESENCE_PENALTY)
                ),
                "qwen_ocr_repetition_penalty": float(
                    settings.get("qwen_ocr_repetition_penalty", DEFAULT_QWEN_OCR_REPETITION_PENALTY)
                ),
                "qwen_ocr_dpi": dpi,
                "qwen_ocr_image_scale": scale,
                "qwen_ocr_jpeg_quality": quality,
                "qwen_ocr_mask_margins": mask_margins,
                "qwen_ocr_margin_mask_config": mask_config,
                "qwen_ocr_batch_size": int(settings.get("qwen_ocr_batch_size", DEFAULT_QWEN_OCR_BATCH_SIZE)),
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
            used_deepseek_fallback=False,
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
        python_executable = os.getenv("QWEN_OCR_PYTHON") or os.getenv("DEEPSEEK_OCR_PYTHON") or sys.executable
        worker = Path(os.getenv("QWEN_OCR_WORKER", str(BASE_DIR / "scripts" / "qwen_ocr_worker.py")))
        cmd = [
            python_executable,
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
            str(
                float(
                    settings.get("qwen_ocr_presence_penalty", DEFAULT_QWEN_OCR_PRESENCE_PENALTY)
                )
            ),
            "--repetition-penalty",
            str(
                float(
                    settings.get(
                        "qwen_ocr_repetition_penalty",
                        DEFAULT_QWEN_OCR_REPETITION_PENALTY,
                    )
                )
            ),
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
            raise RuntimeError(
                "Qwen OCR requires a Python environment with mlx-vlm. Set QWEN_OCR_PYTHON or DEEPSEEK_OCR_PYTHON."
            ) from exc

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

    def _ocr_worker_line_handler(self, on_ocr_progress: Callable[[dict], None] | None) -> Callable[[str], None]:
        def handle(line: str) -> None:
            if '"event"' not in line:
                return
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Unable to parse Qwen OCR progress event: %s", line.strip())
                return
            logger.info("Qwen OCR worker: %s", line.strip())
            if on_ocr_progress is not None:
                on_ocr_progress(event)

        return handle

    def _communicate_with_cancel(
        self,
        process: subprocess.Popen,
        cancel_requested: Callable[[], bool] | None,
        on_stdout_line: Callable[[str], None],
    ) -> tuple[str, str]:
        result: dict[str, str] = {"stdout": "", "stderr": ""}

        def target() -> None:
            stdout_chunks: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                stdout_chunks.append(line)
                on_stdout_line(line)
            stderr = process.stderr.read() if process.stderr is not None else ""
            process.wait()
            result["stdout"] = "".join(stdout_chunks)
            result["stderr"] = stderr

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0.2)
            if cancel_requested is not None and cancel_requested():
                process.terminate()
        thread.join()
        return result["stdout"], result["stderr"]

    def _margin_mask_config(self, settings: dict) -> dict[str, float]:
        return {
            "first_page_top_ratio": self._ratio(
                settings.get("qwen_ocr_first_page_top_mask_ratio", DEFAULT_QWEN_OCR_FIRST_PAGE_TOP_MASK_RATIO)
            ),
            "first_page_bottom_ratio": self._ratio(
                settings.get("qwen_ocr_first_page_bottom_mask_ratio", DEFAULT_QWEN_OCR_FIRST_PAGE_BOTTOM_MASK_RATIO)
            ),
            "other_page_top_ratio": self._ratio(
                settings.get("qwen_ocr_other_page_top_mask_ratio", DEFAULT_QWEN_OCR_OTHER_PAGE_TOP_MASK_RATIO)
            ),
            "other_page_bottom_ratio": self._ratio(
                settings.get("qwen_ocr_other_page_bottom_mask_ratio", DEFAULT_QWEN_OCR_OTHER_PAGE_BOTTOM_MASK_RATIO)
            ),
            "left_ratio": self._ratio(settings.get("qwen_ocr_left_mask_ratio", DEFAULT_QWEN_OCR_LEFT_MASK_RATIO)),
            "right_ratio": self._ratio(settings.get("qwen_ocr_right_mask_ratio", DEFAULT_QWEN_OCR_RIGHT_MASK_RATIO)),
        }

    def _ratio(self, value: object) -> float:
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(0.45, ratio))

    def _save_compressed_full_page(
        self,
        *,
        input_path: Path,
        output_path: Path,
        scale: float,
        quality: int,
        page_number: int = 1,
        mask_margins: bool = False,
        mask_config: dict[str, float] | None = None,
    ) -> dict:
        with Image.open(input_path) as image:
            source = image.convert("RGB")
            original_size = {"width": source.width, "height": source.height}
            mask_metadata = self._mask_headers_footers(source, page_number, mask_config or {}) if mask_margins else None
            if scale != 1.0:
                width = max(1, int(source.width * scale))
                height = max(1, int(source.height * scale))
                source = source.resize((width, height), Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            source.save(output_path, format="JPEG", quality=quality, optimize=True)
            metadata = {
                "input_path": str(input_path),
                "ocr_image_path": str(output_path),
                "ocr_image_mode": "full_page_masked_margins_jpeg_s075_q75"
                if mask_metadata is not None
                else "full_page_jpeg_s075_q75",
                "original_width": original_size["width"],
                "original_height": original_size["height"],
                "ocr_image_width": source.width,
                "ocr_image_height": source.height,
                "ocr_raw_page_scale": scale,
                "ocr_jpeg_quality": quality,
                "ocr_margin_mask_enabled": mask_metadata is not None,
            }
            if mask_metadata is not None:
                metadata["ocr_margin_mask"] = mask_metadata
            return metadata

    def _mask_headers_footers(
        self,
        image: Image.Image,
        page_number: int,
        config: dict[str, float],
    ) -> dict:
        width, height = image.size
        top_ratio = config.get("first_page_top_ratio", 0.0) if page_number == 1 else config.get("other_page_top_ratio", 0.07)
        bottom_ratio = (
            config.get("first_page_bottom_ratio", 0.035)
            if page_number == 1
            else config.get("other_page_bottom_ratio", 0.06)
        )
        left_ratio = config.get("left_ratio", 0.06)
        right_ratio = config.get("right_ratio", 0.03)

        top_px = int(height * self._ratio(top_ratio))
        bottom_px = int(height * self._ratio(bottom_ratio))
        left_px = int(width * self._ratio(left_ratio))
        right_px = int(width * self._ratio(right_ratio))

        draw = ImageDraw.Draw(image)
        if top_px > 0:
            draw.rectangle((0, 0, width, top_px), fill="white")
        if bottom_px > 0:
            draw.rectangle((0, max(0, height - bottom_px), width, height), fill="white")
        if left_px > 0:
            draw.rectangle((0, 0, left_px, height), fill="white")
        if right_px > 0:
            draw.rectangle((max(0, width - right_px), 0, width, height), fill="white")

        return {
            "page_number": page_number,
            "top_px": top_px,
            "bottom_px": bottom_px,
            "left_px": left_px,
            "right_px": right_px,
            "first_page_top_ratio": self._ratio(config.get("first_page_top_ratio", 0.0)),
            "first_page_bottom_ratio": self._ratio(config.get("first_page_bottom_ratio", 0.035)),
            "other_page_top_ratio": self._ratio(config.get("other_page_top_ratio", 0.07)),
            "other_page_bottom_ratio": self._ratio(config.get("other_page_bottom_ratio", 0.06)),
            "left_ratio": self._ratio(left_ratio),
            "right_ratio": self._ratio(right_ratio),
        }

    def _chunks_from_blocks(self, blocks: list[Block]) -> list[ExtractionChunk]:
        chunks: list[ExtractionChunk] = []
        for index, block in enumerate(blocks, start=1):
            text = block.text.strip()
            if not text:
                continue
            chunks.append(
                ExtractionChunk(
                    chunk_id=f"qwen-ocr-{index}",
                    page_number=block.page_number,
                    block_ids=[block.id],
                    block_type=block.block_type.value,
                    bbox=block.bbox.model_dump() if block.bbox else None,
                    polygon=None,
                    original_text=text,
                )
            )
        return chunks


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_args) -> None:
        return None
