from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Callable

from langdetect import detect

from app.models.inspection import PdfInspection
from app.models.schema import (
    Block,
    BlockType,
    BoundingBox,
    DocumentMetadata,
    DocumentModel,
    FigureAsset,
    PageMetadata,
    SourceType,
    TableModel,
)
from app.models.regions import OcrRegionResult, OcrResultsPayload, RegionType
from app.services.renderer import PageRenderer
from app.services.profiler import PipelineProfiler
from app.services.ocr_text_compiler import compile_ocr_results_to_document_text

logger = logging.getLogger(__name__)


class DeepSeekOcrPipeline:
    """Optional MLX-VLM OCR backend for scanned or visually complex PDFs.

    DeepSeek-OCR-2 returns markdown from page images. It does not currently provide the same
    rich block-coordinate structure as Marker JSON in this app, so we preserve page mapping and
    semantic markdown blocks while leaving bbox empty for generated text blocks.
    """

    def __init__(self) -> None:
        self.renderer = PageRenderer()

    def parse_document(
        self,
        pdf_path: Path,
        inspection: PdfInspection,
        job_dir: Path,
        dpi: int,
        preprocess: dict[str, bool | float],
        model_name: str = "mlx-community/DeepSeek-OCR-2-8bit",
        max_tokens: int = 4096,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        crop_mode: bool = True,
        min_crops: int = 2,
        max_crops: int = 6,
        base_size: int = 1024,
        image_size: int = 768,
        skip_repeat: bool = True,
        ngram_size: int = 20,
        ngram_window: int = 90,
        profiler: PipelineProfiler | None = None,
        render_strategy: str = "pre_render_all",
        cancel_requested: Callable[[], bool] | None = None,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
        on_process_finished: Callable[[subprocess.Popen], None] | None = None,
    ) -> tuple[DocumentModel, str]:
        out_dir = job_dir / "deepseek_ocr"
        out_dir.mkdir(parents=True, exist_ok=True)

        image_paths: list[Path] = []
        if render_strategy == "pre_render_all":
            for page in inspection.pages:
                if cancel_requested is not None and cancel_requested():
                    raise RuntimeError("Cancelled by user")
                with profiler.step("page_rendering", page=page.page_number) if profiler is not None else _nullcontext():
                    image_path = self.renderer.render_page(
                        pdf_path,
                        page.page_number,
                        out_dir / f"page_{page.page_number:04d}.png",
                        dpi=dpi,
                        grayscale=bool(preprocess.get("grayscale", False)),
                        binarize=bool(preprocess.get("binarize", False)),
                        denoise=bool(preprocess.get("denoise", False)),
                        #contrast=float(preprocess.get("contrast", 1.0)),
                        profiler=profiler,
                        stage_prefix="page_rendering",
                    )
                image_paths.append(image_path)
        else:
            for page in inspection.pages:
                image_paths.append(
                    self.renderer.render_page(
                        pdf_path,
                        page.page_number,
                        out_dir / f"page_{page.page_number:04d}.png",
                        dpi=dpi,
                        grayscale=bool(preprocess.get("grayscale", False)),
                        binarize=bool(preprocess.get("binarize", False)),
                        denoise=bool(preprocess.get("denoise", False)),
                        #contrast=float(preprocess.get("contrast", 1.0)),
                        profiler=profiler,
                        stage_prefix="page_rendering",
                    )
                )

        with profiler.step("ocr_recognition") if profiler is not None else _nullcontext():
            self._run_pdf_ocr(
                image_paths,
                out_dir,
                model_name,
                max_tokens,
                prompt=prompt,
                crop_mode=crop_mode,
                min_crops=min_crops,
                max_crops=max_crops,
                base_size=base_size,
                image_size=image_size,
                skip_repeat=skip_repeat,
                ngram_size=ngram_size,
                ngram_window=ngram_window,
                cancel_requested=cancel_requested,
                on_process_started=on_process_started,
                on_process_finished=on_process_finished,
            )
        return self._build_document_from_markdown_files(
            inspection=inspection,
            ocr_output_dir=out_dir,
            profiler=profiler,
            strict_page_files=False,
            warning_message=(
                "Parsed with DeepSeek-OCR-2 MLX backend. Page text was generated from rendered images; exact source "
                "bounding boxes are unavailable."
            ),
        )

    def parse_cached_document(
        self,
        inspection: PdfInspection,
        job_dir: Path,
        profiler: PipelineProfiler | None = None,
    ) -> tuple[DocumentModel, str]:
        out_dir = job_dir / "deepseek_ocr"
        if not out_dir.exists():
            raise RuntimeError("Cached OCR directory missing for this job.")
        return self._build_document_from_markdown_files(
            inspection=inspection,
            ocr_output_dir=out_dir,
            profiler=profiler,
            strict_page_files=True,
            warning_message=(
                "Parsed from cached DeepSeek-OCR markdown. OCR inference was skipped and page markdown was reused."
            ),
        )

    def parse_selected_regions_document(
        self,
        inspection: PdfInspection,
        ocr_results: OcrResultsPayload,
        profiler: PipelineProfiler | None = None,
        translation_input_mode: str = "continuous_document",
    ) -> tuple[DocumentModel, str]:
        if translation_input_mode == "continuous_document":
            return self._parse_selected_regions_continuous_document(
                inspection=inspection,
                ocr_results=ocr_results,
                profiler=profiler,
            )
        return self._parse_selected_regions_page_by_page(
            inspection=inspection,
            ocr_results=ocr_results,
            profiler=profiler,
        )

    def _parse_selected_regions_continuous_document(
        self,
        inspection: PdfInspection,
        ocr_results: OcrResultsPayload,
        profiler: PipelineProfiler | None = None,
    ) -> tuple[DocumentModel, str]:
        compiled = compile_ocr_results_to_document_text(ocr_results)
        first_page = compiled.spans[0].page_number if compiled.spans else 1
        blocks = self._blocks_from_markdown(compiled.text, first_page, 0)
        tables, figures = self._extract_structures_from_markdown(compiled.text, first_page)
        span_payload = [
            {
                "page_number": span.page_number,
                "box_id": span.box_id,
                "reading_order": span.reading_order,
                "source_start": span.start,
                "source_end": span.end,
            }
            for span in compiled.spans
        ]
        for block in blocks:
            block.metadata.update(
                {
                    "parser": "deepseek_ocr_2_regions",
                    "translation_input_mode": "continuous_document",
                }
            )
        if blocks:
            # Keep source page/region offsets available for debugging and future
            # layout-aware rebuilds while giving translation one continuous text.
            blocks[0].metadata["ocr_region_spans"] = span_payload

        with profiler.step("language_detection") if profiler is not None else _nullcontext():
            language = self._detect_language(blocks)

        doc = DocumentModel(
            metadata=DocumentMetadata(
                filename=inspection.filename,
                title=inspection.title,
                author=inspection.author,
                page_count=inspection.page_count,
                detected_language=language,
                translation={"translation_input_mode": "continuous_document"},
            ),
            pages=[
                PageMetadata(
                    page_number=p.page_number,
                    width=p.width,
                    height=p.height,
                    has_embedded_text=p.has_embedded_text,
                    embedded_text_quality=p.embedded_text_quality,
                    extraction_mode=SourceType.OCR,
                )
                for p in inspection.pages
            ],
            blocks=blocks,
            tables=tables,
            figures=figures,
            warnings=[
                "Parsed from selected OCR regions as one continuous document before translation chunking.",
                "Original page/region offsets are preserved in block metadata for debugging.",
            ],
        )
        # Return no marker markdown here so source.md and translated.md are both
        # rendered from the same parsed block model. This keeps paragraph/list/
        # heading boundaries aligned for PDF reconstruction.
        return doc, ""

    def _parse_selected_regions_page_by_page(
        self,
        inspection: PdfInspection,
        ocr_results: OcrResultsPayload,
        profiler: PipelineProfiler | None = None,
    ) -> tuple[DocumentModel, str]:
        region_items = sorted(
            ocr_results.results,
            key=lambda item: (item.page_number, item.reading_order, item.box_id),
        )

        blocks: list[Block] = []
        pages_md: list[str] = []
        page_order: dict[int, list[OcrRegionResult]] = {}
        for item in region_items:
            page_order.setdefault(item.page_number, []).append(item)

        for page in inspection.pages:
            page_items = page_order.get(page.page_number, [])
            lines: list[str] = []
            for item in page_items:
                text = item.ocr_text.strip()
                if not text:
                    continue
                block_type = self._region_type_to_block_type(item.box_type)
                block = Block(
                    id=f"deepseek-region-{item.page_number}-{item.box_id}",
                    page_number=item.page_number,
                    block_type=block_type,
                    text=text,
                    bbox=BoundingBox(x0=item.x0, y0=item.y0, x1=item.x1, y1=item.y1),
                    confidence=item.ocr_confidence,
                    reading_order_index=len(blocks),
                    source_type=SourceType.OCR,
                    metadata={
                        "parser": "deepseek_ocr_2_regions",
                        "region_id": item.box_id,
                        "region_type": item.box_type.value,
                    },
                )
                blocks.append(block)
                lines.append(text)
            pages_md.append(f"\n<!-- page: {page.page_number} -->\n\n" + "\n\n".join(lines).strip() + "\n")

        with profiler.step("language_detection") if profiler is not None else _nullcontext():
            language = self._detect_language(blocks)

        doc = DocumentModel(
            metadata=DocumentMetadata(
                filename=inspection.filename,
                title=inspection.title,
                author=inspection.author,
                page_count=inspection.page_count,
                detected_language=language,
            ),
            pages=[
                PageMetadata(
                    page_number=p.page_number,
                    width=p.width,
                    height=p.height,
                    has_embedded_text=p.has_embedded_text,
                    embedded_text_quality=p.embedded_text_quality,
                    extraction_mode=SourceType.OCR,
                )
                for p in inspection.pages
            ],
            blocks=blocks,
            tables=[],
            figures=[],
            warnings=[
                "Parsed from selected OCR regions. Reading order and box metadata were preserved from user-edited regions."
            ],
        )
        return doc, "\n".join(pages_md)

    def _run_pdf_ocr(
        self,
        image_paths: list[Path],
        output_dir: Path,
        model_name: str,
        max_tokens: int,
        prompt: str,
        crop_mode: bool,
        min_crops: int,
        max_crops: int,
        base_size: int,
        image_size: int,
        skip_repeat: bool,
        ngram_size: int,
        ngram_window: int,
        output_names: list[str] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
        on_process_finished: Callable[[subprocess.Popen], None] | None = None,
        on_ocr_progress: Callable[[dict], None] | None = None,
    ) -> None:
        python_executable = os.getenv("DEEPSEEK_OCR_PYTHON", sys.executable)
        cmd = [
            python_executable,
            "scripts/deepseek_ocr_worker.py",
            "--model",
            model_name,
            "--images-json",
            json_dumps_paths(image_paths),
            "--output-dir",
            str(output_dir),
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            "0.0",
            "--prompt",
            prompt,
            "--crop-mode",
            "true" if crop_mode else "false",
            "--min-crops",
            str(min_crops),
            "--max-crops",
            str(max_crops),
            "--base-size",
            str(base_size),
            "--image-size",
            str(image_size),
            "--skip-repeat",
            "true" if skip_repeat else "false",
            "--ngram-size",
            str(ngram_size),
            "--ngram-window",
            str(ngram_window),
        ]
        if output_names:
            cmd.extend(["--names-json", json.dumps(output_names)])
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            if on_process_started is not None:
                on_process_started(process)
            stdout, stderr = self._communicate_with_cancel(
                process,
                cancel_requested,
                on_stdout_line=self._ocr_worker_line_handler(on_ocr_progress) if on_ocr_progress is not None else None,
            )
            if on_process_finished is not None:
                on_process_finished(process)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
            if on_ocr_progress is None:
                for line in stdout.splitlines():
                    if '"event"' in line:
                        logger.info("DeepSeek OCR worker: %s", line)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "DeepSeek OCR requires a Python environment with mlx-vlm. "
                "Run scripts/setup_deepseek_ocr_env.sh and set DEEPSEEK_OCR_PYTHON=.venv-deepseek-ocr/bin/python."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            if exc.returncode == -15 and cancel_requested is not None and cancel_requested():
                raise RuntimeError("Cancelled by user") from exc
            raise RuntimeError(
                f"DeepSeek OCR failed. If this mentions transformers/mlx-vlm dependency "
                f"conflicts, use a separate DeepSeek OCR env via DEEPSEEK_OCR_PYTHON. Details: {stderr[-1000:]}"
            ) from exc

    def run_ocr_on_images(
        self,
        *,
        image_paths: list[Path],
        output_dir: Path,
        output_names: list[str],
        model_name: str = "mlx-community/DeepSeek-OCR-2-8bit",
        max_tokens: int = 4096,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        crop_mode: bool = True,
        min_crops: int = 2,
        max_crops: int = 6,
        base_size: int = 1024,
        image_size: int = 768,
        skip_repeat: bool = True,
        ngram_size: int = 20,
        ngram_window: int = 90,
        on_ocr_progress: Callable[[dict], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
        on_process_finished: Callable[[subprocess.Popen], None] | None = None,
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._run_pdf_ocr(
            image_paths=image_paths,
            output_dir=output_dir,
            model_name=model_name,
            max_tokens=max_tokens,
            prompt=prompt,
            crop_mode=crop_mode,
            min_crops=min_crops,
            max_crops=max_crops,
            base_size=base_size,
            image_size=image_size,
            skip_repeat=skip_repeat,
            ngram_size=ngram_size,
            ngram_window=ngram_window,
            output_names=output_names,
            on_ocr_progress=on_ocr_progress,
            cancel_requested=cancel_requested,
            on_process_started=on_process_started,
            on_process_finished=on_process_finished,
        )
        results: dict[str, str] = {}
        for name in output_names:
            md_path = output_dir / f"{name}.md"
            if md_path.exists():
                results[name] = md_path.read_text(encoding="utf-8", errors="ignore")
        return results

    def _communicate_with_cancel(
        self,
        process: subprocess.Popen,
        cancel_requested: Callable[[], bool] | None = None,
        on_stdout_line: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        if cancel_requested is None and on_stdout_line is None:
            stdout, stderr = process.communicate()
            return stdout, stderr

        result: dict[str, str] = {"stdout": "", "stderr": ""}

        def target() -> None:
            if on_stdout_line is None:
                stdout, stderr = process.communicate()
                result["stdout"] = stdout
                result["stderr"] = stderr
                return

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

    def _ocr_worker_line_handler(self, on_ocr_progress: Callable[[dict], None] | None) -> Callable[[str], None]:
        def handle(line: str) -> None:
            if '"event"' not in line:
                return
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Unable to parse OCR worker progress event: %s", line.strip())
                return
            logger.info("DeepSeek OCR worker: %s", line.strip())
            if on_ocr_progress is not None:
                on_ocr_progress(event)

        return handle

    def _clean_mlx_vlm_output(self, output: str) -> str:
        text = output.strip()
        if not text:
            return ""
        match = re.search(
            r"Files:\s*\[.*?\]\s*\n\s*Prompt:\s*(?P<body>.*?)(?:\n=+\nPrompt:\s+\d+\s+tokens|\Z)",
            text,
            flags=re.DOTALL,
        )
        if match:
            text = match.group("body").strip()

        lines = text.splitlines()
        if lines and lines[0].strip() == "<image>":
            lines = lines[1:]
        if lines and self._looks_like_echoed_prompt(lines[0]):
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
        text = "\n".join(lines).strip()

        text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        text = re.sub(r"\[\[\d+\s*,\s*\d+\s*,\s*\d+\s*,?\s*$", "", text)
        return text.strip()

    def _looks_like_echoed_prompt(self, line: str) -> bool:
        lowered = line.lower()
        return "convert" in lowered and ("markdown" in lowered or "document" in lowered)

    def _blocks_from_markdown(self, markdown: str, page_number: int, start_order: int) -> list[Block]:
        blocks: list[Block] = []
        paragraph_lines: list[str] = []
        in_table = False

        def flush_paragraph() -> None:
            if not paragraph_lines:
                return
            text = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
            paragraph_lines.clear()
            if text:
                blocks.append(self._block(page_number, start_order + len(blocks), BlockType.PARAGRAPH, text))

        def flush_table() -> None:
            nonlocal in_table
            if in_table:
                blocks.append(self._block(page_number, start_order + len(blocks), BlockType.TABLE, "[TABLE]"))
                in_table = False

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                flush_table()
                continue
            if line.startswith("|") and line.endswith("|"):
                flush_paragraph()
                in_table = True
            elif match := re.match(r"^(#{1,6})\s+(.+)$", line):
                flush_paragraph()
                flush_table()
                blocks.append(self._block(page_number, start_order + len(blocks), BlockType.HEADING, match.group(2).strip()))
            elif re.match(r"^[-*+]\s+", line):
                flush_paragraph()
                flush_table()
                blocks.append(self._block(page_number, start_order + len(blocks), BlockType.LIST, re.sub(r"^[-*+]\s+", "", line)))
            elif re.match(r"^(Table|Figure)\s+\d+", line, flags=re.IGNORECASE):
                flush_paragraph()
                flush_table()
                blocks.append(self._block(page_number, start_order + len(blocks), BlockType.CAPTION, line))
            else:
                flush_table()
                paragraph_lines.append(line)

        flush_paragraph()
        flush_table()
        return blocks

    def _block(self, page_number: int, order: int, block_type: BlockType, text: str) -> Block:
        return Block(
            id=f"deepseek-p{page_number}-b{order}",
            page_number=page_number,
            block_type=block_type,
            text=text,
            bbox=None if block_type != BlockType.TABLE else BoundingBox(x0=0, y0=0, x1=0, y1=0),
            reading_order_index=order,
            source_type=SourceType.OCR,
            metadata={"parser": "deepseek_ocr_2"},
        )

    def _region_type_to_block_type(self, region_type: RegionType) -> BlockType:
        mapping = {
            RegionType.PAGE: BlockType.PARAGRAPH,
            RegionType.TEXT: BlockType.PARAGRAPH,
            RegionType.TABLE: BlockType.TABLE,
            RegionType.FIGURE: BlockType.FIGURE,
            RegionType.CAPTION: BlockType.CAPTION,
            RegionType.HEADER: BlockType.HEADER,
            RegionType.FOOTER: BlockType.FOOTER,
            RegionType.REFERENCE: BlockType.REFERENCE,
            RegionType.OTHER: BlockType.UNKNOWN,
        }
        return mapping.get(region_type, BlockType.UNKNOWN)

    def _extract_structures_from_markdown(self, markdown: str, page_number: int) -> tuple[list[TableModel], list[FigureAsset]]:
        tables: list[TableModel] = []
        figures: list[FigureAsset] = []
        table_lines: list[str] = []
        caption_text: str | None = None

        def flush_table() -> None:
            nonlocal table_lines, caption_text
            if not table_lines:
                return
            rows = [line.strip("|").split("|") for line in table_lines if line.strip("|")]
            rows = [[cell.strip() for cell in row] for row in rows]
            headers = rows[0] if rows else []
            body = rows[2:] if len(rows) > 2 and all(c.strip("-: ") == "" for c in rows[1]) else rows[1:]
            cells = [[TableModel.TableCell(text=cell) for cell in row] for row in body]
            tables.append(
                TableModel(
                    id=f"deepseek-table-p{page_number}-{len(tables)}",
                    page_numbers=[page_number],
                    page=page_number,
                    headers=headers,
                    rows=body,
                    cells=cells,
                    caption=caption_text,
                    parse_mode="markdown_table",
                )
            )
            table_lines = []
            caption_text = None

        for raw in markdown.splitlines():
            line = raw.strip()
            if line.startswith("|") and line.endswith("|"):
                table_lines.append(line)
                continue
            flush_table()
            if re.match(r"^Figure\s+\d+", line, flags=re.IGNORECASE):
                figures.append(FigureAsset(id=f"deepseek-fig-p{page_number}-{len(figures)}", page_number=page_number))
            if re.match(r"^Table\s+\d+", line, flags=re.IGNORECASE):
                caption_text = line
        flush_table()
        return tables, figures

    def _detect_language(self, blocks: list[Block]) -> str | None:
        text = "\n".join(b.text for b in blocks if b.block_type in {BlockType.PARAGRAPH, BlockType.HEADING})
        text = text[:4000].strip()
        if len(text) < 40:
            return None
        try:
            return detect(text)
        except Exception:
            return None

    @staticmethod
    def save_document_json(document: DocumentModel, out_path: Path) -> None:
        out_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def load_document_json(path: Path) -> DocumentModel:
        return DocumentModel.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _build_document_from_markdown_files(
        self,
        inspection: PdfInspection,
        ocr_output_dir: Path,
        profiler: PipelineProfiler | None,
        strict_page_files: bool,
        warning_message: str,
    ) -> tuple[DocumentModel, str]:
        pages_md: list[str] = []
        blocks: list[Block] = []
        tables: list[TableModel] = []
        figures: list[FigureAsset] = []
        missing_pages: list[int] = []

        for page in inspection.pages:
            markdown_path = ocr_output_dir / f"page_{page.page_number:04d}.md"
            if not markdown_path.exists():
                missing_pages.append(page.page_number)
                markdown = ""
            else:
                markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")
            pages_md.append(f"\n<!-- page: {page.page_number} -->\n\n{markdown.strip()}\n")
            blocks.extend(self._blocks_from_markdown(markdown, page.page_number, len(blocks)))
            table_objs, figure_objs = self._extract_structures_from_markdown(markdown, page.page_number)
            tables.extend(table_objs)
            figures.extend(figure_objs)

        if strict_page_files and missing_pages:
            sample = ", ".join(str(page) for page in missing_pages[:10])
            raise RuntimeError(f"Cached OCR markdown is incomplete; missing page files for pages: {sample}.")

        with profiler.step("language_detection") if profiler is not None else _nullcontext():
            language = self._detect_language(blocks)

        warnings = [warning_message]
        if missing_pages:
            warnings.append(
                "Some OCR page markdown files were missing. Missing page numbers: "
                + ", ".join(str(page) for page in missing_pages[:20])
            )

        doc = DocumentModel(
            metadata=DocumentMetadata(
                filename=inspection.filename,
                title=inspection.title,
                author=inspection.author,
                page_count=inspection.page_count,
                detected_language=language,
            ),
            pages=[
                PageMetadata(
                    page_number=p.page_number,
                    width=p.width,
                    height=p.height,
                    has_embedded_text=p.has_embedded_text,
                    embedded_text_quality=p.embedded_text_quality,
                    extraction_mode=SourceType.OCR,
                )
                for p in inspection.pages
            ],
            blocks=blocks,
            tables=tables,
            figures=figures,
            warnings=warnings,
        )
        return doc, "\n".join(pages_md)


def json_dumps_paths(paths: list[Path]) -> str:
    import json

    return json.dumps([str(path) for path in paths])
