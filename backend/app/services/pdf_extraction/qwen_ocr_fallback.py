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
from app.services.markdown_builder import MarkdownBuilder
from app.services.ocr_to_translation_parser import OCRToTranslationParser
from app.services.pdf_extraction.models import ExtractionChunk, PDFExtractionResult
from app.services.pdf_inspector import PdfInspector
from app.services.profiler import PipelineProfiler
from app.services.qwen_markdown_parser import QwenMarkdownParser
from app.services.renderer import PageRenderer

logger = logging.getLogger(__name__)

BAD_SCAN_CLASSIFICATIONS = {"scanned_no_text", "bad_hidden_ocr"}
SURYA_OVERLAY_PROMPT_SUFFIX = """

The page image contains document text plus visible layout annotations added by Surya.
Each layout rectangle has a label immediately above it in this exact visual form:
SURYA <number>: <type>

Transcribe the visible source document text inside every Surya rectangle exactly once, in
ascending SURYA number order. Use the Surya type as structure metadata. Do not transcribe the
annotation label itself as document text. Do not invent text outside the rectangles.
Wrap every region exactly like this:
<region index="<number>" type="<type>">
Markdown transcription of the document text inside that rectangle
</region>

Preserve headings, paragraphs, list items, footnotes, page headers, and page footers.
Return only the region elements and their Markdown content.
""".strip()


class QwenFullPageOCRFallback:
    """Render full PDF pages and preserve the Markdown emitted by Qwen OCR."""

    def __init__(self) -> None:
        self.inspector = PdfInspector()
        self.renderer = PageRenderer()
        self.markdown_parser = QwenMarkdownParser()
        self.ocr_to_translation_parser = OCRToTranslationParser()

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

        base_prompt = str(settings.get("qwen_ocr_prompt", DEFAULT_QWEN_OCR_PROMPT))
        ocr_prompt = base_prompt
        ocr_image_paths = image_paths
        surya_layout_manifest: dict = {}
        use_surya_layout = self._should_use_surya_layout(pdf_classification)
        if use_surya_layout:
            surya_layout_dir = qwen_dir / "surya_layout"
            surya_layout_manifest = self._run_surya_layout(
                render_dir=render_dir,
                output_dir=surya_layout_dir,
                settings=settings,
                cancel_requested=cancel_requested,
                on_process_started=on_process_started,
                on_process_finished=on_process_finished,
                on_ocr_progress=on_ocr_progress,
            )
            ocr_image_paths = self._surya_boxed_page_paths(
                surya_layout_manifest,
                expected_pages=len(image_paths),
            )
            image_metadata = self._surya_page_metadata(
                image_metadata,
                surya_layout_manifest,
                ocr_image_paths,
            )
            ocr_prompt = self._surya_overlay_prompt(base_prompt)

        self._run_qwen_ocr(
            image_paths=ocr_image_paths,
            output_dir=markdown_dir,
            output_names=output_names,
            settings=settings,
            prompt=ocr_prompt,
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
            surya_layout_manifest=surya_layout_manifest if use_surya_layout else None,
        )
        logical_chunks_path = qwen_dir / "logical_translation_chunks.json"
        logical_warnings: list[str] = []
        excluded_regions: list[dict] = []
        if use_surya_layout:
            logical_result = self.ocr_to_translation_parser.prepare(
                document,
                document_id=job_dir.name,
            )
            document = logical_result.document
            logical_warnings = logical_result.warnings
            excluded_regions = logical_result.excluded_regions
            logical_chunks_path.write_text(
                json.dumps(
                    {
                        "document_id": job_dir.name,
                        "chunks": [chunk.model_dump() for chunk in document.translation_chunks],
                        "excluded_regions": excluded_regions,
                        "warnings": logical_warnings,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            markdown = MarkdownBuilder().build(document)
            chunks = self._chunks_from_logical_translation_chunks(document.translation_chunks)
        else:
            chunks = self._chunks_from_blocks(document.blocks)
        marker_skipped = bool(marker_metadata.get("marker_skipped"))
        qwen_reason = (
            "PDF text-quality detection classified the document as a poor scan; Marker was skipped and Surya layout detection plus Qwen full-page OCR were used."
            if marker_skipped and use_surya_layout
            else "PDF text-quality detection classified the document as poor text; Marker was skipped and Qwen full-page OCR was used."
            if marker_skipped
            else "Marker first pass did not classify the document as good digital text; Qwen full-page OCR fallback was used."
        )
        warnings = [*marker_warnings, qwen_reason, *document.warnings, *logical_warnings]
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
                "marker_mode": (
                    "skipped_for_surya_qwen_full_page_ocr"
                    if marker_skipped and use_surya_layout
                    else "qwen_full_page_ocr_fallback"
                ),
                "fallback_engine": (
                    "surya_layout_qwen_full_page_ocr" if use_surya_layout else "qwen_full_page_ocr"
                ),
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
                "qwen_ocr_prompt": ocr_prompt,
                "qwen_ocr_base_prompt": base_prompt,
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
                "qwen_ocr_image_mode": (
                    "surya_boxed_page_png" if use_surya_layout else "rendered_page_png"
                ),
                "qwen_ocr_image_metadata": image_metadata,
                "qwen_ocr_output_dir": str(qwen_dir),
                "surya_layout_used": use_surya_layout,
                "surya_layout_manifest": (
                    str(qwen_dir / "surya_layout" / "layout.json") if use_surya_layout else ""
                ),
                "surya_layout_region_count": int(surya_layout_manifest.get("region_count", 0)),
                "surya_layout_reconciled_region_count": int(
                    surya_layout_manifest.get("reconciled_region_count", 0)
                ),
                "surya_layout_merged_region_count": int(
                    surya_layout_manifest.get("merged_region_count", 0)
                ),
                "surya_layout_overlap_count": int(surya_layout_manifest.get("overlap_count", 0)),
                "ocr_logical_chunks_used": use_surya_layout,
                "ocr_logical_chunks_path": str(logical_chunks_path) if use_surya_layout else "",
                "ocr_logical_chunk_count": len(document.translation_chunks) if use_surya_layout else 0,
                "ocr_excluded_region_count": len(excluded_regions),
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
        prompt: str,
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
            prompt,
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
            on_stdout_line=self._worker_line_handler("Qwen OCR", on_ocr_progress),
        )
        if on_process_finished is not None:
            on_process_finished(process)
        if process.returncode != 0:
            if process.returncode == -15 and cancel_requested is not None and cancel_requested():
                raise RuntimeError("Cancelled by user")
            raise RuntimeError(f"Qwen OCR failed: {(stderr or stdout)[-2000:]}")

    def _run_surya_layout(
        self,
        *,
        render_dir: Path,
        output_dir: Path,
        settings: dict,
        cancel_requested: Callable[[], bool] | None,
        on_process_started: Callable[[subprocess.Popen], None] | None,
        on_process_finished: Callable[[subprocess.Popen], None] | None,
        on_ocr_progress: Callable[[dict], None] | None,
    ) -> dict:
        worker = Path(
            os.getenv("SURYA_LAYOUT_WORKER", str(BASE_DIR / "scripts" / "surya_layout_worker.py"))
        )
        cmd = [
            self._resolve_surya_python_executable(),
            str(worker),
            "--input-dir",
            str(render_dir),
            "--output-dir",
            str(output_dir),
            "--padding",
            str(max(0, int(settings.get("surya_layout_padding", 16)))),
        ]
        batch_size = settings.get("surya_layout_batch_size")
        if batch_size is not None:
            cmd.extend(["--batch-size", str(max(1, int(batch_size)))])
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
                "Surya layout detection requires the Marker Python environment. "
                "Set SURYA_LAYOUT_PYTHON."
            ) from exc

        if on_process_started is not None:
            on_process_started(process)
        stdout, stderr = self._communicate_with_cancel(
            process,
            cancel_requested=cancel_requested,
            on_stdout_line=self._worker_line_handler("Surya layout", on_ocr_progress),
        )
        if on_process_finished is not None:
            on_process_finished(process)
        if process.returncode != 0:
            if process.returncode == -15 and cancel_requested is not None and cancel_requested():
                raise RuntimeError("Cancelled by user")
            raise RuntimeError(f"Surya layout detection failed: {(stderr or stdout)[-2000:]}")

        manifest_path = output_dir / "layout.json"
        if not manifest_path.exists():
            raise RuntimeError("Surya layout detection did not write layout.json.")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _resolve_worker_python_executable(self) -> str:
        configured = os.getenv("QWEN_OCR_PYTHON")
        if not configured:
            return sys.executable
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)
        logger.warning("Ignoring QWEN_OCR_PYTHON because it does not exist: %s", configured)
        return sys.executable

    def _resolve_surya_python_executable(self) -> str:
        configured = os.getenv("SURYA_LAYOUT_PYTHON")
        if configured:
            path = Path(configured).expanduser()
            if path.exists():
                return str(path)
            logger.warning("Ignoring SURYA_LAYOUT_PYTHON because it does not exist: %s", configured)
        isolated_marker_python = BASE_DIR / ".venv-marker" / "bin" / "python"
        if isolated_marker_python.exists():
            return str(isolated_marker_python)
        return sys.executable

    def _worker_line_handler(
        self,
        worker_name: str,
        callback: Callable[[dict], None] | None,
    ) -> Callable[[str], None]:
        def handle(line: str) -> None:
            stripped = line.strip()
            if not stripped:
                return
            logger.info("%s worker: %s", worker_name, stripped)
            if callback is None or '"event"' not in line:
                return
            try:
                callback(json.loads(line))
            except json.JSONDecodeError:
                return

        return handle

    def _should_use_surya_layout(self, pdf_classification: str) -> bool:
        return pdf_classification in BAD_SCAN_CLASSIFICATIONS

    def _surya_overlay_prompt(self, base_prompt: str) -> str:
        return f"{base_prompt.strip()}\n\n{SURYA_OVERLAY_PROMPT_SUFFIX}"

    def _surya_boxed_page_paths(self, manifest: dict, *, expected_pages: int) -> list[Path]:
        paths = [Path(str(page["boxed_page_path"])) for page in manifest.get("pages", [])]
        if len(paths) != expected_pages:
            raise RuntimeError(
                f"Surya layout detection wrote {len(paths)} boxed page(s); expected {expected_pages}."
            )
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise RuntimeError(f"Surya layout boxed page is missing: {missing[0]}")
        return paths

    def _surya_page_metadata(
        self,
        rendered_metadata: list[dict],
        manifest: dict,
        boxed_page_paths: list[Path],
    ) -> list[dict]:
        pages = manifest.get("pages", [])
        metadata: list[dict] = []
        for rendered, page, boxed_page_path in zip(rendered_metadata, pages, boxed_page_paths):
            item = dict(rendered)
            with Image.open(boxed_page_path) as image:
                item.update(
                    {
                        "ocr_image_path": str(boxed_page_path),
                        "ocr_image_mode": "surya_boxed_page_png",
                        "ocr_image_width": image.width,
                        "ocr_image_height": image.height,
                        "surya_region_count": len(page.get("regions", [])),
                        "surya_reconciled_region_count": len(page.get("reconciled_regions", [])),
                    }
                )
            metadata.append(item)
        return metadata

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

    def _chunks_from_logical_translation_chunks(self, chunks) -> list[ExtractionChunk]:
        return [
            ExtractionChunk(
                chunk_id=chunk.id,
                page_number=int(chunk.page_start or 1),
                page_end=chunk.page_end,
                block_ids=list(chunk.block_ids),
                block_type=chunk.chunk_type,
                bbox=None,
                polygon=None,
                original_text=chunk.source_text,
                source_region_ids=list(chunk.source_region_ids),
                source_region_indexes=list(chunk.source_region_indexes),
                source_region_types=list(chunk.source_region_types),
                section_path=list(chunk.section_path),
                source_text_before_cleaning=chunk.source_text_before_cleaning,
                status=chunk.status,
                warnings=list(chunk.warnings),
            )
            for chunk in chunks
            if chunk.status == "ready_for_translation"
        ]


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_args) -> None:
        return None
