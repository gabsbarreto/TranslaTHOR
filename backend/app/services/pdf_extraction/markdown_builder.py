from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

try:
    from langdetect import detect
except Exception:  # pragma: no cover - lightweight test environments may omit langdetect
    detect = None

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
from app.services.markdown_builder import MarkdownBuilder as AppMarkdownBuilder
from app.services.pdf_extraction.models import ExtractionChunk, PDFTypeDetectionResult


class MarkerDocumentBuilder:
    def build_document(
        self,
        *,
        marker_payload: Any,
        detection: PDFTypeDetectionResult,
        filename: str,
        source_type: SourceType,
        parser_metadata: dict[str, Any],
        warnings: list[str],
    ) -> tuple[DocumentModel, str, list[ExtractionChunk]]:
        pages_payload = self._page_payloads(marker_payload)
        blocks: list[Block] = []
        figures: list[FigureAsset] = []
        tables: list[TableModel] = []
        chunks: list[ExtractionChunk] = []
        reading_order = 0

        for fallback_index, page_payload in enumerate(pages_payload, start=1):
            page_number = self._page_number(page_payload, fallback_index)
            page_width, page_height = self._marker_page_dimensions(
                page_payload,
                detection,
                page_number,
            )
            for node in self._iter_content_blocks(page_payload):
                block_type = self._map_block_type(str(node.get("block_type", "")))
                text = self._node_text(node)
                caption_text = self._caption_text_from_table_html(text) if block_type == BlockType.TABLE else None
                if caption_text is not None:
                    block_type = BlockType.CAPTION
                    text = caption_text
                if block_type not in {BlockType.FIGURE, BlockType.TABLE} and not text.strip():
                    continue
                block_id = str(node.get("id") or f"marker-{page_number}-{reading_order}")
                polygon = self._polygon(node)
                bbox = self._bbox_from_polygon(polygon)
                metadata = {
                    "parser": "marker",
                    "marker_block_type": node.get("block_type"),
                    "marker_id": node.get("id"),
                    "section_hierarchy": node.get("section_hierarchy") or {},
                    "marker_page_width": page_width,
                    "marker_page_height": page_height,
                    "coordinate_space": {
                        "name": "marker_page_coordinates",
                        "width": page_width,
                        "height": page_height,
                    },
                }
                if polygon:
                    metadata["polygon"] = polygon
                if caption_text is not None:
                    metadata["marker_table_caption_normalized"] = True

                block = Block(
                    id=block_id,
                    page_number=page_number,
                    block_type=block_type,
                    text=text.strip(),
                    bbox=bbox,
                    confidence=self._confidence(node),
                    reading_order_index=reading_order,
                    source_type=source_type,
                    metadata=metadata,
                )
                blocks.append(block)

                if block_type == BlockType.TABLE:
                    table = self._table_from_block(block, node)
                    if table is not None:
                        tables.append(table)
                elif block_type == BlockType.FIGURE:
                    figures.append(
                        FigureAsset(
                            id=f"figure-{len(figures)}",
                            page_number=page_number,
                            bbox=bbox,
                            caption_block_id=None,
                            image_path=None,
                            detection_confidence=block.confidence,
                            source_block_ids=[block.id],
                            source_region_ids=[
                                str(value)
                                for value in block.metadata.get("source_region_ids", [])
                            ],
                            extraction_metadata={
                                "marker_block_type": node.get("block_type"),
                                "marker_id": node.get("id"),
                                "reading_order_index": reading_order,
                            },
                        )
                    )

                if text.strip():
                    chunks.append(
                        ExtractionChunk(
                            chunk_id=f"extract-{len(chunks)}",
                            page_number=page_number,
                            block_ids=[block_id],
                            block_type=block_type.value,
                            bbox=bbox.model_dump() if bbox is not None else None,
                            polygon=polygon,
                            original_text=text.strip(),
                        )
                    )
                reading_order += 1

        if not blocks and isinstance(marker_payload, str):
            blocks, chunks = self._fallback_blocks_from_markdown(marker_payload, detection, source_type)
        else:
            chunks = self._chunks_from_blocks(blocks)

        pages = [
            PageMetadata(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                has_embedded_text=page.has_selectable_text,
                embedded_text_quality=1.0 if page.looks_meaningful else 0.0,
                extraction_mode=source_type,
            )
            for page in detection.pages
        ]
        if not pages and pages_payload:
            for index, page_payload in enumerate(pages_payload, start=1):
                polygon = self._polygon(page_payload)
                bbox = self._bbox_from_polygon(polygon)
                width = bbox.x1 if bbox is not None else 0.0
                height = bbox.y1 if bbox is not None else 0.0
                pages.append(
                    PageMetadata(
                        page_number=index,
                        width=width,
                        height=height,
                        has_embedded_text=False,
                        embedded_text_quality=0.0,
                        extraction_mode=source_type,
                    )
                )

        language = self._detect_language(blocks)
        metadata = dict(parser_metadata)
        metadata.setdefault("extraction_engine", "marker")
        doc = DocumentModel(
            metadata=DocumentMetadata(
                filename=filename,
                title=detection.metadata.get("title"),
                author=detection.metadata.get("author"),
                page_count=detection.page_count or len(pages),
                detected_language=language,
                translation=metadata,
            ),
            pages=pages,
            blocks=blocks,
            tables=tables,
            figures=figures,
            warnings=warnings,
        )
        markdown = AppMarkdownBuilder().build(doc)
        return doc, markdown, chunks

    def _marker_page_dimensions(
        self,
        page_payload: dict[str, Any],
        detection: PDFTypeDetectionResult,
        page_number: int,
    ) -> tuple[float, float]:
        polygon = self._polygon(page_payload)
        bbox = self._bbox_from_polygon(polygon)
        if bbox is not None and bbox.x1 > bbox.x0 and bbox.y1 > bbox.y0:
            return bbox.x1 - min(0.0, bbox.x0), bbox.y1 - min(0.0, bbox.y0)
        for page in detection.pages:
            if page.page_number == page_number:
                return float(page.width), float(page.height)
        return 0.0, 0.0

    def _chunks_from_blocks(self, blocks: list[Block]) -> list[ExtractionChunk]:
        chunks: list[ExtractionChunk] = []
        for block in blocks:
            if not block.text.strip():
                continue
            chunks.append(
                ExtractionChunk(
                    chunk_id=f"extract-{len(chunks)}",
                    page_number=block.page_number,
                    block_ids=[block.id],
                    block_type=block.block_type.value,
                    bbox=block.bbox.model_dump() if block.bbox is not None else None,
                    polygon=block.metadata.get("polygon"),
                    original_text=block.text.strip(),
                )
            )
        return chunks

    def _page_payloads(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            if str(payload.get("block_type", "")).lower() == "document":
                children = payload.get("children") or []
                return [item for item in children if isinstance(item, dict)]
            if "children" in payload:
                return [payload]
            if "pages" in payload and isinstance(payload["pages"], list):
                return [item for item in payload["pages"] if isinstance(item, dict)]
        if isinstance(payload, list):
            pages = [item for item in payload if isinstance(item, dict) and str(item.get("block_type", "")).lower() == "page"]
            return pages or [item for item in payload if isinstance(item, dict)]
        return []

    def _iter_content_blocks(self, node: dict[str, Any]):
        children = node.get("children")
        if not isinstance(children, list):
            return
        for child in children:
            if not isinstance(child, dict):
                continue
            child_type = str(child.get("block_type", ""))
            if child_type.lower() in {"tablegroup", "formgroup"}:
                yield from self._iter_content_blocks(child)
                continue
            if child_type and child_type.lower() not in {"line", "span"}:
                yield child
            if child_type.lower() in {"table", "form"}:
                # Marker emits a complete table block and then every TableCell as descendants.
                # The table block is the canonical extraction unit; descending into cells
                # duplicates the same text in Markdown and translation chunks.
                continue
            yield from self._iter_content_blocks(child)

    def _page_number(self, page_payload: dict[str, Any], fallback: int) -> int:
        for key in ("page_number", "page", "page_id"):
            if key in page_payload:
                try:
                    value = int(page_payload[key])
                    return value + 1 if key == "page_id" and value == fallback - 1 else max(1, value)
                except Exception:
                    pass
        block_id = str(page_payload.get("id") or "")
        match = re.search(r"/page/(\d+)", block_id)
        if match:
            return int(match.group(1)) + 1
        return fallback

    def _map_block_type(self, marker_type: str) -> BlockType:
        normalized = marker_type.lower()
        if normalized in {"sectionheader", "title"}:
            return BlockType.HEADING
        if normalized in {"listitem", "listgroup"}:
            return BlockType.LIST
        if normalized in {"table", "tablegroup", "form"}:
            return BlockType.TABLE
        if normalized in {"figure", "figuregroup", "picture", "picturegroup", "handwriting"}:
            return BlockType.FIGURE
        if normalized == "caption":
            return BlockType.CAPTION
        if normalized == "footnote":
            return BlockType.FOOTNOTE
        if normalized == "pageheader":
            return BlockType.HEADER
        if normalized == "pagefooter":
            return BlockType.FOOTER
        if normalized in {"equation", "textinlinemath"}:
            return BlockType.EQUATION
        if normalized == "tableofcontents":
            return BlockType.REFERENCE
        return BlockType.PARAGRAPH

    def _node_text(self, node: dict[str, Any]) -> str:
        for key in ("markdown", "text", "plain_text"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        html = node.get("html")
        if isinstance(html, str) and html.strip():
            if "<table" in html.lower():
                return html.strip()
            return _HTMLTextExtractor.to_text(html)
        return ""

    def _polygon(self, node: dict[str, Any]) -> list[list[float]] | None:
        raw = node.get("polygon") or node.get("bbox")
        if not raw:
            return None
        if isinstance(raw, list) and len(raw) == 4 and all(isinstance(item, (int, float)) for item in raw):
            x0, y0, x1, y1 = [float(item) for item in raw]
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        if isinstance(raw, list):
            points: list[list[float]] = []
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        points.append([float(item[0]), float(item[1])])
                    except Exception:
                        continue
            return points or None
        return None

    def _bbox_from_polygon(self, polygon: list[list[float]] | None) -> BoundingBox | None:
        if not polygon:
            return None
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))

    def _confidence(self, node: dict[str, Any]) -> float | None:
        for key in ("confidence", "score", "ocr_confidence"):
            if key in node:
                try:
                    return float(node[key])
                except Exception:
                    return None
        return None

    def _table_from_block(self, block: Block, node: dict[str, Any]) -> TableModel | None:
        if str(block.metadata.get("marker_block_type", "")).lower() in {"tablegroup", "formgroup"}:
            return None
        html = block.text
        if not html.strip():
            return None
        if "<table" not in html.lower():
            return TableModel(
                id=f"table-{len(block.id)}-{block.reading_order_index}",
                page_numbers=[block.page_number],
                page=block.page_number,
                bbox=block.bbox,
                rows=[[block.text]],
                parse_mode="marker_text",
            )
        rows = _TableExtractor.extract(html)
        if not rows:
            return None
        headers = rows[0] if len(rows) > 1 else []
        body_rows = rows[1:] if headers else rows
        return TableModel(
            id=f"table-{block.reading_order_index}",
            page_numbers=[block.page_number],
            page=block.page_number,
            bbox=block.bbox,
            headers=headers,
            rows=body_rows,
            parse_mode="marker_html",
            debug={"marker_block_id": block.id, "render_from_block_text": True},
        )

    def _caption_text_from_table_html(self, text: str) -> str | None:
        if "<table" not in text.lower():
            return None
        rows = _TableExtractor.extract(text)
        if not rows:
            return None
        nonempty_cells = [cell.strip() for row in rows for cell in row if cell.strip()]
        if not nonempty_cells or len(nonempty_cells) > 3:
            return None
        first_cell = nonempty_cells[0]
        if not re.match(r"^(tablo|table)\s+\d+[.:]?\s+", first_cell, flags=re.IGNORECASE):
            return None
        if any(len(row) > 1 for row in rows):
            return None
        caption = " ".join(nonempty_cells)
        caption = re.sub(r"\s+", " ", caption).strip()
        return caption or None

    def _fallback_blocks_from_markdown(
        self,
        markdown: str,
        detection: PDFTypeDetectionResult,
        source_type: SourceType,
    ) -> tuple[list[Block], list[ExtractionChunk]]:
        blocks: list[Block] = []
        chunks: list[ExtractionChunk] = []
        page_number = 1
        for part in re.split(r"\n{2,}", markdown):
            text = part.strip()
            if not text:
                continue
            page_match = re.match(r"<!--\s*page:\s*(\d+)\s*-->", text)
            if page_match:
                page_number = int(page_match.group(1))
                continue
            block_type = BlockType.HEADING if text.startswith("#") else BlockType.PARAGRAPH
            block = Block(
                id=f"marker-md-{len(blocks)}",
                page_number=page_number,
                block_type=block_type,
                text=text.lstrip("# ").strip(),
                reading_order_index=len(blocks),
                source_type=source_type,
                metadata={"parser": "marker_markdown_fallback"},
            )
            blocks.append(block)
            chunks.append(
                ExtractionChunk(
                    chunk_id=f"extract-{len(chunks)}",
                    page_number=page_number,
                    block_ids=[block.id],
                    block_type=block.block_type.value,
                    bbox=None,
                    polygon=None,
                    original_text=block.text,
                )
            )
        _ = detection
        return blocks, chunks

    def _detect_language(self, blocks: list[Block]) -> str | None:
        text = "\n".join(block.text for block in blocks if block.text.strip())[:5000]
        if len(text.strip()) < 20:
            return None
        try:
            if detect is None:
                return None
            return detect(text)
        except Exception:
            return None


class _HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    @classmethod
    def to_text(cls, html: str) -> str:
        parser = cls()
        parser.feed(html)
        text = "".join(parser.parts)
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


class _TableExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    @classmethod
    def extract(cls, html: str) -> list[list[str]]:
        parser = cls()
        parser.feed(html)
        return [[cell.strip() for cell in row] for row in parser.rows if any(cell.strip() for cell in row)]

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current_cell is not None:
            if self.current_row is not None:
                self.current_row.append("".join(self.current_cell).strip())
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)
