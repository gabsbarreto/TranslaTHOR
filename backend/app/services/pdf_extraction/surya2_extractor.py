from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from app.config import DEFAULT_SURYA2_DPI, DEFAULT_SURYA2_STRATEGY
from app.models.schema import BlockType, DocumentModel
from app.services.markdown_builder import MarkdownBuilder
from app.services.pdf_extraction.models import PDFExtractionResult
from app.services.pdf_extraction.surya2_adapter import Surya2DocumentAdapter
from app.services.pdf_extraction.surya2_runtime import Surya2Runtime
from app.services.pdf_inspector import PdfInspector
from app.services.profiler import PipelineProfiler
from app.services.renderer import PageRenderer

logger = logging.getLogger(__name__)


class Surya2LlamaCppExtractor:
    """Direct Surya 2 extractor using a persistent llama.cpp-backed worker."""

    def __init__(self, runtime: Surya2Runtime | None = None) -> None:
        self.runtime = runtime or Surya2Runtime()
        self.inspector = PdfInspector()
        self.renderer = PageRenderer()
        self.adapter = Surya2DocumentAdapter()

    def extract(
        self,
        *,
        pdf_path: Path,
        job_dir: Path,
        pdf_classification: str,
        detection_metadata: dict,
        warnings: list[str],
        settings: dict,
        profiler: PipelineProfiler | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
        on_process_finished: Callable[[subprocess.Popen], None] | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> PDFExtractionResult:
        started = time.perf_counter()
        strategy = str(settings.get("surya2_strategy", DEFAULT_SURYA2_STRATEGY))
        if strategy not in {"full_page", "layout_then_block"}:
            raise ValueError(f"Unsupported Surya 2 strategy: {strategy}")
        dpi = int(settings.get("surya2_dpi", DEFAULT_SURYA2_DPI))
        if dpi <= 0:
            raise ValueError("surya2_dpi must be positive.")

        inspection = self.inspector.inspect(pdf_path)
        surya_dir = job_dir / "surya2"
        render_dir = surya_dir / f"rendered_{dpi}dpi"
        raw_path = surya_dir / f"raw_{strategy}.json"
        overlay_dir = surya_dir / f"overlays_{strategy}"
        figure_dir = surya_dir / f"figures_{strategy}"
        logical_chunks_path = surya_dir / f"logical_translation_chunks_{strategy}.json"
        render_dir.mkdir(parents=True, exist_ok=True)

        image_paths: list[Path] = []
        render_started = time.perf_counter()
        for page in inspection.pages:
            if cancel_requested is not None and cancel_requested():
                raise RuntimeError("Cancelled by user")
            image_path = self.renderer.render_page(
                pdf_path,
                page.page_number,
                render_dir / f"page_{page.page_number:04d}.png",
                dpi=dpi,
                profiler=profiler,
                stage_prefix="surya2_page_rendering",
            )
            image_paths.append(image_path)
            if on_progress is not None:
                on_progress(
                    {
                        "event": "render_page_done",
                        "page_number": page.page_number,
                        "total": inspection.page_count,
                    }
                )
        render_seconds = time.perf_counter() - render_started

        raw_payload = self.runtime.run(
            image_paths=image_paths,
            output_path=raw_path,
            strategy=strategy,
            cancel_requested=cancel_requested,
            on_process_started=on_process_started,
            on_process_finished=on_process_finished,
            on_event=on_progress,
        )
        raw_pages = list(raw_payload.get("pages") or [])
        if len(raw_pages) != inspection.page_count:
            raise RuntimeError(
                f"Surya 2 returned {len(raw_pages)} page(s); expected {inspection.page_count}."
            )

        document, markdown, chunks = self.adapter.build_document(
            raw_pages=raw_pages,
            inspection=inspection,
            strategy=strategy,
            document_id=job_dir.name,
            warnings=warnings,
        )
        self._write_figure_crops(document, image_paths, figure_dir)
        markdown = MarkdownBuilder().build(document)
        self._write_overlays(image_paths, raw_pages, overlay_dir)
        logical_chunks_path.write_text(
            json.dumps(
                {
                    "document_id": job_dir.name,
                    "engine": "surya2_llamacpp",
                    "strategy": strategy,
                    "chunks": [
                        chunk.model_dump(mode="json") for chunk in document.translation_chunks
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        elapsed = time.perf_counter() - started
        error_count = sum(
            1 for page in raw_pages for block in page.get("blocks", []) if block.get("error")
        )
        skipped_count = sum(
            1 for page in raw_pages for block in page.get("blocks", []) if block.get("skipped")
        )
        metadata = {
            "pdf_classification": pdf_classification,
            "extraction_mode": "surya2_llamacpp",
            "marker_mode": "skipped_for_surya2_llamacpp",
            "fallback_engine": "surya2_llamacpp",
            "ocr_engine": "surya2_llamacpp",
            "ocr_used": True,
            "force_ocr": True,
            "strip_existing_ocr": False,
            "surya2_version": str(raw_payload.get("surya_version", "")),
            "surya2_strategy": strategy,
            "surya2_dpi": dpi,
            "surya2_backend": "llamacpp",
            "surya2_raw_path": str(raw_path),
            "surya2_overlay_dir": str(overlay_dir),
            "surya2_figure_dir": str(figure_dir),
            "surya2_render_dir": str(render_dir),
            "surya2_error_block_count": error_count,
            "surya2_skipped_block_count": skipped_count,
            "surya2_batching": raw_payload.get("batching", {}),
            "surya2_worker_timing": raw_payload.get("timing", {}),
            "surya2_render_seconds": round(render_seconds, 6),
            "llama_cpp_version": self._llama_cpp_version(),
            "ocr_logical_chunks_used": True,
            "ocr_logical_chunks_path": str(logical_chunks_path),
            "ocr_logical_chunk_count": len(document.translation_chunks),
            "detection": detection_metadata,
            "extraction_time_seconds": round(elapsed, 3),
        }
        document.metadata.translation = {
            **document.metadata.translation,
            **metadata,
        }
        return PDFExtractionResult(
            markdown=markdown,
            chunks=chunks,
            pages=[page.model_dump(mode="json") for page in document.pages],
            blocks=[block.model_dump(mode="json") for block in document.blocks],
            metadata=metadata,
            extraction_mode="surya2_llamacpp",
            pdf_classification=pdf_classification,
            used_ocr=True,
            used_force_ocr=True,
            stripped_existing_ocr=False,
            used_local_vlm_repair=False,
            warnings=document.warnings,
            document=document,
        )

    def close(self) -> None:
        self.runtime.close()

    def _write_overlays(
        self,
        image_paths: list[Path],
        raw_pages: list[dict],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path, page in zip(image_paths, raw_pages):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            for block in page.get("blocks", []):
                bbox = block.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                color = (
                    "#dc2626"
                    if block.get("error")
                    else ("#d97706" if block.get("skipped") else "#2563eb")
                )
                draw.rectangle(tuple(float(value) for value in bbox), outline=color, width=3)
                label = f"{int(block.get('reading_order', 0))}: {block.get('label', 'Unknown')}"
                draw.text((float(bbox[0]) + 3, max(0.0, float(bbox[1]) - 14)), label, fill=color)
            image.save(output_dir / image_path.name)
            image.close()

    def _write_figure_crops(
        self,
        document: DocumentModel,
        image_paths: list[Path],
        output_dir: Path,
    ) -> None:
        page_paths = {page_number: path for page_number, path in enumerate(image_paths, start=1)}
        figure_blocks = [block for block in document.blocks if block.block_type == BlockType.FIGURE]
        if not figure_blocks:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        for figure, block in zip(document.figures, figure_blocks):
            image_path = page_paths.get(block.page_number)
            raw_bbox = block.metadata.get("surya_image_bbox")
            if image_path is None or not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
                continue
            with Image.open(image_path) as source:
                x0 = max(0, min(source.width, int(float(raw_bbox[0]))))
                y0 = max(0, min(source.height, int(float(raw_bbox[1]))))
                x1 = max(0, min(source.width, int(float(raw_bbox[2]))))
                y1 = max(0, min(source.height, int(float(raw_bbox[3]))))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = source.crop((x0, y0, x1, y1)).convert("RGB")
            crop_path = output_dir / f"{figure.id}.png"
            crop.save(crop_path)
            crop.close()
            resolved_path = str(crop_path.resolve())
            figure.image_path = resolved_path
            block.metadata["figure_asset_id"] = figure.id
            block.metadata["figure_crop_path"] = resolved_path

    def _llama_cpp_version(self) -> str:
        executable = shutil.which("llama-server")
        if executable is None:
            return ""
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            first_line = (result.stdout or result.stderr).strip().splitlines()
            return first_line[0] if first_line else ""
        except Exception:
            return ""
