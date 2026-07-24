from __future__ import annotations

import html as html_lib
import math
import re
from html.parser import HTMLParser
from typing import Any, Iterable

try:
    from langdetect import detect  # type: ignore[import-untyped]
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
from app.services.markdown_builder import MarkdownBuilder
from app.services.ocr_to_translation_parser import OCRToTranslationParser
from app.services.pdf_extraction.models import ExtractionChunk


VISUAL_LABELS = {"diagram", "figure", "picture"}
LABEL_TO_BLOCK_TYPE = {
    "bibliography": BlockType.REFERENCE,
    "caption": BlockType.CAPTION,
    "chemicalblock": BlockType.EQUATION,
    "code": BlockType.PARAGRAPH,
    "diagram": BlockType.FIGURE,
    "equation": BlockType.EQUATION,
    "figure": BlockType.FIGURE,
    "footnote": BlockType.FOOTNOTE,
    "form": BlockType.TABLE,
    "listgroup": BlockType.LIST,
    "listitem": BlockType.LIST,
    "pagefooter": BlockType.FOOTER,
    "pageheader": BlockType.HEADER,
    "pagenumber": BlockType.PAGE_NUMBER,
    "picture": BlockType.FIGURE,
    "sectionheader": BlockType.HEADING,
    "table": BlockType.TABLE,
    "tableofcontents": BlockType.REFERENCE,
    "text": BlockType.PARAGRAPH,
    "title": BlockType.HEADING,
}


def normalized_label(value: str) -> str:
    return re.sub(r"[^a-z]", "", str(value).lower())


def image_polygon_to_pdf(
    polygon: Iterable[Iterable[float]],
    *,
    image_width: float,
    image_height: float,
    pdf_width: float,
    pdf_height: float,
) -> list[list[float]]:
    """Scale top-left image coordinates to TranslaTHOR's top-left PDF-point space."""
    if image_width <= 0 or image_height <= 0 or pdf_width <= 0 or pdf_height <= 0:
        raise ValueError("Image and PDF dimensions must be positive.")
    scale_x = pdf_width / image_width
    scale_y = pdf_height / image_height
    converted: list[list[float]] = []
    for point in polygon:
        values = list(point)
        if len(values) < 2:
            raise ValueError("Every polygon point must contain x and y coordinates.")
        image_x = min(max(float(values[0]), 0.0), image_width)
        image_y = min(max(float(values[1]), 0.0), image_height)
        converted.append(
            [
                min(max(image_x * scale_x, 0.0), pdf_width),
                min(max(image_y * scale_y, 0.0), pdf_height),
            ]
        )
    if len(converted) < 4:
        raise ValueError("A Surya polygon must contain at least four points.")
    return converted


def bbox_from_polygon(polygon: list[list[float]]) -> BoundingBox:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))


class Surya2DocumentAdapter:
    """Map the Surya 2 page/block schema into TranslaTHOR's shared model."""

    def __init__(self) -> None:
        self.logical_parser = OCRToTranslationParser()

    def build_document(
        self,
        *,
        raw_pages: list[dict[str, Any]],
        inspection: PdfInspection,
        strategy: str,
        document_id: str | None = None,
        warnings: list[str] | None = None,
    ) -> tuple[DocumentModel, str, list[ExtractionChunk]]:
        inspection_by_page = {page.page_number: page for page in inspection.pages}
        blocks: list[Block] = []
        tables: list[TableModel] = []
        figures: list[FigureAsset] = []
        adapter_warnings = list(warnings or [])
        global_order = 0

        for fallback_page_number, raw_page in enumerate(raw_pages, start=1):
            page_number = int(raw_page.get("page_number") or fallback_page_number)
            page = inspection_by_page.get(page_number)
            if page is None:
                adapter_warnings.append(
                    f"Surya 2 returned page {page_number}, which is absent from PDF inspection."
                )
                continue
            image_bbox = list(raw_page.get("image_bbox") or [])
            if len(image_bbox) != 4:
                raise ValueError(f"Surya 2 page {page_number} has no valid image_bbox.")
            image_width = float(image_bbox[2]) - float(image_bbox[0])
            image_height = float(image_bbox[3]) - float(image_bbox[1])

            ordered_raw_blocks = sorted(
                (item for item in raw_page.get("blocks", []) if isinstance(item, dict)),
                key=lambda item: int(item.get("reading_order", 0)),
            )
            for page_order, raw_block in enumerate(ordered_raw_blocks):
                label = str(raw_block.get("label") or "Text")
                raw_label = str(raw_block.get("raw_label") or label)
                label_key = normalized_label(label)
                block_type = LABEL_TO_BLOCK_TYPE.get(label_key, BlockType.UNKNOWN)
                image_polygon = raw_block.get("polygon") or self._polygon_from_bbox(
                    raw_block.get("bbox")
                )
                try:
                    pdf_polygon = image_polygon_to_pdf(
                        image_polygon,
                        image_width=image_width,
                        image_height=image_height,
                        pdf_width=page.width,
                        pdf_height=page.height,
                    )
                    bbox = bbox_from_polygon(pdf_polygon)
                except (TypeError, ValueError) as exc:
                    pdf_polygon = None
                    bbox = None
                    adapter_warnings.append(
                        f"Surya 2 block {page_number}:{page_order} has invalid coordinates: {exc}"
                    )

                raw_html = str(raw_block.get("html") or "").strip()
                skipped = bool(raw_block.get("skipped", False))
                error = bool(raw_block.get("error", False))
                text = self._content_for_block(block_type, raw_html)
                if label_key in VISUAL_LABELS:
                    text = ""
                    skipped = True
                block_id = f"surya2-p{page_number:04d}-b{page_order:04d}"
                metadata = {
                    "parser": "surya2_llamacpp",
                    "ocr_engine": "surya2_llamacpp",
                    "surya2_strategy": strategy,
                    "surya_label": label,
                    "surya_raw_label": raw_label,
                    "surya_reading_order": int(raw_block.get("reading_order", page_order)),
                    "surya_skipped": skipped,
                    "surya_error": error,
                    "surya_image_polygon": image_polygon,
                    "surya_image_bbox": raw_block.get("bbox"),
                    "surya_image_width": image_width,
                    "surya_image_height": image_height,
                    "coordinate_space": "pdf_points_top_left",
                    "source_region_ids": [block_id],
                }
                if pdf_polygon is not None:
                    assert bbox is not None
                    metadata["polygon"] = pdf_polygon
                    metadata["surya_bbox"] = [
                        bbox.x0,
                        bbox.y0,
                        bbox.x1,
                        bbox.y1,
                    ]
                metadata["surya_page_width"] = page.width
                metadata["surya_page_height"] = page.height

                confidence = self._optional_float(raw_block.get("confidence"))
                block = Block(
                    id=block_id,
                    page_number=page_number,
                    block_type=block_type,
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    reading_order_index=global_order,
                    source_type=SourceType.OCR,
                    raw_label=raw_label,
                    html=raw_html,
                    polygon=pdf_polygon,
                    skipped=skipped,
                    error=error,
                    metadata=metadata,
                )
                blocks.append(block)
                if block_type == BlockType.TABLE:
                    tables.append(self._table_from_block(block))
                if block_type == BlockType.FIGURE:
                    figure_id = f"surya2-figure-{len(figures) + 1}"
                    block.metadata["figure_asset_id"] = figure_id
                    figures.append(
                        FigureAsset(
                            id=figure_id,
                            page_number=page_number,
                            bbox=bbox,
                            source_block_ids=[block_id],
                            source_region_ids=[block_id],
                            extraction_metadata={
                                "parser": "surya2_llamacpp",
                                "reading_order_index": global_order,
                            },
                        )
                    )
                global_order += 1

        self._associate_captions(blocks, figures, tables)
        error_count = sum(1 for block in blocks if block.error)
        if error_count:
            adapter_warnings.append(f"Surya 2 reported {error_count} block OCR error(s).")

        document = DocumentModel(
            metadata=DocumentMetadata(
                filename=inspection.filename,
                title=inspection.title,
                author=inspection.author,
                page_count=inspection.page_count,
                detected_language=self._detect_language(blocks),
                translation={
                    "extraction_engine": "surya2_llamacpp",
                    "surya2_strategy": strategy,
                    "coordinate_space": "pdf_points_top_left",
                },
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
            warnings=adapter_warnings,
        )
        logical_result = self.logical_parser.prepare(document, document_id=document_id)
        document = logical_result.document
        document.warnings.extend(logical_result.warnings)
        markdown = MarkdownBuilder().build(document)
        chunks = self._chunks_from_document(document)
        return document, markdown, chunks

    def _content_for_block(self, block_type: BlockType, raw_html: str) -> str:
        if not raw_html:
            return ""
        if block_type in {BlockType.TABLE, BlockType.EQUATION}:
            return raw_html
        return _HTMLToText.convert(raw_html)

    def _table_from_block(self, block: Block) -> TableModel:
        parsed = _SuryaTableHTMLParser.extract(block.html or "")
        return TableModel(
            id=f"surya2-table-{block.page_number}-{block.reading_order_index}",
            page_numbers=[block.page_number],
            page=block.page_number,
            bbox=block.bbox,
            headers=parsed.headers,
            rows=parsed.rows,
            cells=parsed.cells,
            parse_mode="surya2_html" if parsed.rows or parsed.headers else "surya2_raw_html",
            debug={
                "source_block_id": block.id,
                "surya2_block_id": block.id,
                "html": block.html or "",
                "render_from_block_text": True,
            },
        )

    def _associate_captions(
        self,
        blocks: list[Block],
        figures: list[FigureAsset],
        tables: list[TableModel],
    ) -> None:
        captions = [block for block in blocks if block.block_type == BlockType.CAPTION]
        used_caption_ids: set[str] = set()
        for figure in figures:
            caption = self._nearest_caption(
                figure.page_number,
                figure.bbox,
                captions,
                used_caption_ids,
                preferred_pattern=r"^\s*(fig(?:ure)?|abbildung|figura)\b",
            )
            if caption is None:
                continue
            figure.caption_block_id = caption.id
            caption.metadata["caption_for_figure_id"] = figure.id
            used_caption_ids.add(caption.id)
        for table in tables:
            caption = self._nearest_caption(
                int(table.page or table.page_numbers[0]),
                table.bbox,
                captions,
                used_caption_ids,
                preferred_pattern=r"^\s*(table|tableau|tabla|tabelle|tabela)\b",
            )
            if caption is None:
                continue
            table.caption_block_id = caption.id
            table.caption = caption.text
            caption.metadata["caption_for_table_id"] = table.id
            used_caption_ids.add(caption.id)

    def _nearest_caption(
        self,
        page_number: int,
        bbox: BoundingBox | None,
        captions: list[Block],
        used_caption_ids: set[str],
        *,
        preferred_pattern: str,
    ) -> Block | None:
        candidates = [
            item
            for item in captions
            if item.page_number == page_number and item.id not in used_caption_ids
        ]
        if not candidates:
            return None
        preferred = [
            item for item in candidates if re.match(preferred_pattern, item.text, re.IGNORECASE)
        ]
        pool = preferred or candidates
        if bbox is None:
            return min(pool, key=lambda item: item.reading_order_index)

        def distance(item: Block) -> float:
            if item.bbox is None:
                return math.inf
            vertical_gap = min(
                abs(item.bbox.y0 - bbox.y1),
                abs(bbox.y0 - item.bbox.y1),
            )
            horizontal_gap = max(
                0.0,
                max(item.bbox.x0, bbox.x0) - min(item.bbox.x1, bbox.x1),
            )
            return vertical_gap + horizontal_gap

        selected = min(pool, key=distance)
        return selected if math.isfinite(distance(selected)) else None

    def _chunks_from_document(self, document: DocumentModel) -> list[ExtractionChunk]:
        chunks: list[ExtractionChunk] = []
        for chunk in document.translation_chunks:
            if chunk.status != "ready_for_translation" or not chunk.source_text.strip():
                continue
            block_lookup = {block.id: block for block in document.blocks}
            chunk_blocks = [
                block_lookup[block_id] for block_id in chunk.block_ids if block_id in block_lookup
            ]
            bbox = self._merged_bbox(chunk_blocks)
            polygons = [block.polygon for block in chunk_blocks if block.polygon]
            chunks.append(
                ExtractionChunk(
                    chunk_id=chunk.id,
                    page_number=int(chunk.page_start or 1),
                    page_end=chunk.page_end,
                    block_ids=list(chunk.block_ids),
                    block_type=chunk.chunk_type,
                    bbox=bbox.model_dump() if bbox else None,
                    polygon=polygons[0] if len(polygons) == 1 else None,
                    original_text=chunk.source_text,
                    source_region_ids=list(chunk.source_region_ids),
                    source_region_indexes=list(chunk.source_region_indexes),
                    source_region_types=list(chunk.source_region_types),
                    section_path=list(chunk.section_path),
                    source_text_before_cleaning=chunk.source_text_before_cleaning,
                    status=chunk.status,
                    warnings=list(chunk.warnings),
                )
            )
        return chunks

    def _merged_bbox(self, blocks: list[Block]) -> BoundingBox | None:
        boxes = [block.bbox for block in blocks if block.bbox is not None]
        if not boxes:
            return None
        return BoundingBox(
            x0=min(box.x0 for box in boxes),
            y0=min(box.y0 for box in boxes),
            x1=max(box.x1 for box in boxes),
            y1=max(box.y1 for box in boxes),
        )

    def _polygon_from_bbox(self, bbox: Any) -> list[list[float]]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("Surya block has neither polygon nor a valid bbox.")
        x0, y0, x1, y1 = [float(value) for value in bbox]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def _optional_float(self, value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _detect_language(self, blocks: list[Block]) -> str | None:
        sample = "\n".join(
            block.text for block in blocks if block.text and not block.skipped and not block.error
        )[:5000]
        if detect is None or len(sample.strip()) < 40:
            return None
        try:
            return str(detect(sample))
        except Exception:
            return None


class _HTMLToText(HTMLParser):
    BREAK_TAGS = {
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    @classmethod
    def convert(cls, value: str) -> str:
        parser = cls()
        parser.feed(value)
        parser.close()
        text = html_lib.unescape("".join(parser.parts))
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line).strip()

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BREAK_TAGS:
            self.parts.append("\n")


class _SuryaTableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows_with_header_flags: list[list[tuple[str, bool, int, int]]] = []
        self.current_row: list[tuple[str, bool, int, int]] | None = None
        self.current_cell: list[str] | None = None
        self.current_is_header = False
        self.current_rowspan = 1
        self.current_colspan = 1

    @classmethod
    def extract(cls, value: str) -> "_ParsedTable":
        parser = cls()
        parser.feed(value)
        parser.close()
        headers: list[str] = []
        body_rows: list[list[str]] = []
        cells: list[list[TableModel.TableCell]] = []
        for row in parser.rows_with_header_flags:
            texts = [cell[0] for cell in row]
            if row and all(cell[1] for cell in row) and not headers:
                headers = texts
            else:
                body_rows.append(texts)
                cells.append(
                    [
                        TableModel.TableCell(
                            text=text,
                            rowspan=rowspan,
                            colspan=colspan,
                        )
                        for text, _is_header, rowspan, colspan in row
                    ]
                )
        return _ParsedTable(headers=headers, rows=body_rows, cells=cells)

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            values = {str(name).lower(): value for name, value in attrs}
            self.current_cell = []
            self.current_is_header = tag == "th"
            self.current_rowspan = self._positive_int(values.get("rowspan"))
            self.current_colspan = self._positive_int(values.get("colspan"))
        elif tag == "br" and self.current_cell is not None:
            self.current_cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current_cell is not None:
            if self.current_row is not None:
                self.current_row.append(
                    (
                        re.sub(r"\s+", " ", "".join(self.current_cell)).strip(),
                        self.current_is_header,
                        self.current_rowspan,
                        self.current_colspan,
                    )
                )
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if self.current_row:
                self.rows_with_header_flags.append(self.current_row)
            self.current_row = None

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def _positive_int(self, value: Any) -> int:
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            return 1


class _ParsedTable:
    def __init__(
        self,
        *,
        headers: list[str],
        rows: list[list[str]],
        cells: list[list[TableModel.TableCell]],
    ) -> None:
        self.headers = headers
        self.rows = rows
        self.cells = cells
