from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path

try:
    from langdetect import detect
except Exception:  # pragma: no cover - lightweight test environments may omit langdetect
    detect = None

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
from app.services.profiler import PipelineProfiler


class QwenMarkdownParser:
    """Convert Qwen page Markdown into the shared document model without filtering OCR text."""

    def build_document_from_markdown_dir(
        self,
        *,
        inspection: PdfInspection,
        markdown_dir: Path,
        profiler: PipelineProfiler | None = None,
        strict_page_files: bool = False,
    ) -> tuple[DocumentModel, str]:
        page_items: list[tuple[int, str]] = []
        missing_pages: list[int] = []
        for page in inspection.pages:
            path = markdown_dir / f"page_{page.page_number:04d}.md"
            if path.exists():
                markdown = path.read_text(encoding="utf-8", errors="ignore")
            else:
                markdown = ""
                missing_pages.append(page.page_number)
            page_items.append((page.page_number, markdown))

        if strict_page_files and missing_pages:
            sample = ", ".join(str(page) for page in missing_pages[:10])
            raise RuntimeError(f"Qwen OCR Markdown is incomplete; missing page files: {sample}.")

        blocks: list[Block] = []
        tables: list[TableModel] = []
        figures: list[FigureAsset] = []
        for page_number, markdown in page_items:
            blocks.extend(self._blocks_from_markdown(markdown, page_number, len(blocks)))
            page_tables, page_figures = self._extract_structures_from_markdown(markdown, page_number)
            tables.extend(page_tables)
            figures.extend(page_figures)

        with profiler.step("language_detection") if profiler is not None else nullcontext():
            language = self._detect_language(blocks)

        warnings = [
            "Parsed from Qwen full-page OCR Markdown. OCR text was preserved without header, footer, or page-number filtering."
        ]
        if missing_pages:
            warnings.append(
                "Some Qwen OCR page Markdown files were missing: "
                + ", ".join(str(page) for page in missing_pages[:20])
            )

        document = DocumentModel(
            metadata=DocumentMetadata(
                filename=inspection.filename,
                title=inspection.title,
                author=inspection.author,
                page_count=inspection.page_count,
                detected_language=language,
                translation={"ocr_markdown_preserved": True},
            ),
            pages=[
                PageMetadata(
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    has_embedded_text=page.has_embedded_text,
                    embedded_text_quality=page.embedded_text_quality,
                    extraction_mode=SourceType.OCR,
                )
                for page in inspection.pages
            ],
            blocks=blocks,
            tables=tables,
            figures=figures,
            warnings=warnings,
        )
        source_markdown = "\n\n".join(markdown for _page_number, markdown in page_items)
        return document, source_markdown

    def _blocks_from_markdown(self, markdown: str, page_number: int, start_order: int) -> list[Block]:
        blocks: list[Block] = []
        paragraph_lines: list[str] = []
        table_lines: list[str] = []

        def append(block_type: BlockType, text: str) -> None:
            if text.strip():
                blocks.append(self._block(page_number, start_order + len(blocks), block_type, text.strip()))

        def flush_paragraph() -> None:
            if paragraph_lines:
                append(BlockType.PARAGRAPH, " ".join(paragraph_lines))
                paragraph_lines.clear()

        def flush_table() -> None:
            if table_lines:
                append(BlockType.TABLE, "[TABLE]")
                table_lines.clear()

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                flush_table()
                continue
            if line.startswith("|") and line.endswith("|"):
                flush_paragraph()
                table_lines.append(line)
                continue

            flush_table()
            if match := re.match(r"^(#{1,6})\s+(.+)$", line):
                flush_paragraph()
                append(BlockType.HEADING, match.group(2))
            elif re.match(r"^[-*+]\s+", line):
                flush_paragraph()
                append(BlockType.LIST, re.sub(r"^[-*+]\s+", "", line))
            elif re.match(r"^(Table|Figure)\s+\d+", line, flags=re.IGNORECASE):
                flush_paragraph()
                append(BlockType.CAPTION, line)
            else:
                paragraph_lines.append(line)

        flush_paragraph()
        flush_table()
        return blocks

    def _block(self, page_number: int, order: int, block_type: BlockType, text: str) -> Block:
        bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0) if block_type == BlockType.TABLE else None
        return Block(
            id=f"qwen-p{page_number}-b{order}",
            page_number=page_number,
            block_type=block_type,
            text=text,
            bbox=bbox,
            reading_order_index=order,
            source_type=SourceType.OCR,
            metadata={"parser": "qwen_full_page_ocr"},
        )

    def _extract_structures_from_markdown(
        self,
        markdown: str,
        page_number: int,
    ) -> tuple[list[TableModel], list[FigureAsset]]:
        tables: list[TableModel] = []
        figures: list[FigureAsset] = []
        table_lines: list[str] = []
        caption_text: str | None = None

        def flush_table() -> None:
            nonlocal caption_text
            if not table_lines:
                return
            rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table_lines]
            headers = rows[0] if rows else []
            body = rows[2:] if len(rows) > 2 and all(cell.strip("-: ") == "" for cell in rows[1]) else rows[1:]
            cells = [[TableModel.TableCell(text=cell) for cell in row] for row in body]
            tables.append(
                TableModel(
                    id=f"qwen-table-p{page_number}-{len(tables)}",
                    page_numbers=[page_number],
                    page=page_number,
                    headers=headers,
                    rows=body,
                    cells=cells,
                    caption=caption_text,
                    parse_mode="markdown_table",
                )
            )
            table_lines.clear()
            caption_text = None

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if line.startswith("|") and line.endswith("|"):
                table_lines.append(line)
                continue
            flush_table()
            if re.match(r"^Figure\s+\d+", line, flags=re.IGNORECASE):
                figures.append(FigureAsset(id=f"qwen-fig-p{page_number}-{len(figures)}", page_number=page_number))
            if re.match(r"^Table\s+\d+", line, flags=re.IGNORECASE):
                caption_text = line
        flush_table()
        return tables, figures

    def _detect_language(self, blocks: list[Block]) -> str | None:
        text = "\n".join(block.text for block in blocks if block.text.strip())[:4000].strip()
        if detect is None or len(text) < 40:
            return None
        try:
            return detect(text)
        except Exception:
            return None
