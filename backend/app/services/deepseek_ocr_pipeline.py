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
from app.services.renderer import PageRenderer
from app.services.profiler import PipelineProfiler

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
        model_name: str = "mlx-community/DeepSeek-OCR-2-bf16",
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
                        contrast=float(preprocess.get("contrast", 1.0)),
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
                        contrast=float(preprocess.get("contrast", 1.0)),
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
        cancel_requested: Callable[[], bool] | None = None,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
        on_process_finished: Callable[[subprocess.Popen], None] | None = None,
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
            stdout, stderr = self._communicate_with_cancel(process, cancel_requested)
            if on_process_finished is not None:
                on_process_finished(process)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
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

    def _communicate_with_cancel(
        self,
        process: subprocess.Popen,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> tuple[str, str]:
        if cancel_requested is None:
            stdout, stderr = process.communicate()
            return stdout, stderr

        result: dict[str, str] = {"stdout": "", "stderr": ""}

        def target() -> None:
            stdout, stderr = process.communicate()
            result["stdout"] = stdout
            result["stderr"] = stderr

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0.2)
            if cancel_requested():
                process.terminate()
        thread.join()
        return result["stdout"], result["stderr"]

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
