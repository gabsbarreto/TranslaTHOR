from __future__ import annotations

import math
import re
from html.parser import HTMLParser
from typing import Any

try:
    from langdetect import detect
except Exception:  # pragma: no cover - lightweight test environments may omit langdetect
    detect = None

try:
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0
except (ImportError, AttributeError):  # pragma: no cover - lightweight compatibility
    pass

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
from app.services.table_markup import MAX_TABLE_CELL_SPAN, ParsedTableCell, parse_table_rows


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
        if not pages_payload and not isinstance(marker_payload, str):
            raise ValueError(
                "Marker JSON does not contain canonical symbolic Document/Page blocks; "
                "refusing to interpret debug spans as document content."
            )
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
                caption_text = (
                    self._caption_text_from_table_html(text)
                    if block_type == BlockType.TABLE
                    else None
                )
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
                                str(value) for value in block.metadata.get("source_region_ids", [])
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
            blocks, chunks = self._fallback_blocks_from_markdown(
                marker_payload, detection, source_type
            )
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

    def chunks_from_blocks(self, blocks: list[Block]) -> list[ExtractionChunk]:
        """Rebuild extraction chunks after a source-backed structural repair."""

        return self._chunks_from_blocks(blocks)

    def _page_payloads(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            if str(payload.get("block_type", "")).lower() == "document":
                children = payload.get("children") or []
                return [
                    item
                    for item in children
                    if isinstance(item, dict) and str(item.get("block_type", "")).lower() == "page"
                ]
            if str(payload.get("block_type", "")).lower() == "page" and isinstance(
                payload.get("children"), list
            ):
                return [payload]
            if "pages" in payload and isinstance(payload["pages"], list):
                return [
                    item
                    for item in payload["pages"]
                    if isinstance(item, dict) and str(item.get("block_type", "")).lower() == "page"
                ]
        if isinstance(payload, list):
            return [
                item
                for item in payload
                if isinstance(item, dict)
                and str(item.get("block_type", "")).lower() == "page"
                and isinstance(item.get("children"), list)
            ]
        return []

    def _iter_content_blocks(self, node: dict[str, Any]):
        children = self._ordered_children(node)
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

    def _ordered_children(self, node: dict[str, Any]) -> list[Any] | None:
        """Keep Marker order, except for unambiguous table/figure notes.

        Marker normally emits a sound multi-column reading order, so sorting all
        blocks by coordinates would be destructive. Some versions nevertheless
        leave a direct Caption or Footnote at the end of a page even when its box
        is immediately beside a TableGroup or FigureGroup. Move only those
        strongly anchored notes and leave every other child in Marker order.
        """

        children = node.get("children")
        if not isinstance(children, list):
            return None
        if str(node.get("block_type", "")).casefold() != "page":
            return children

        anchor_types = {
            "tablegroup",
            "formgroup",
            "figuregroup",
            "picturegroup",
            "table",
            "form",
            "figure",
            "picture",
            "handwriting",
        }
        note_types = {"caption", "footnote"}
        anchors: list[tuple[int, BoundingBox]] = []
        for index, child in enumerate(children):
            if not isinstance(child, dict):
                continue
            if str(child.get("block_type", "")).casefold() not in anchor_types:
                continue
            bbox = self._bbox_from_polygon(self._polygon(child))
            if bbox is not None and bbox.x1 > bbox.x0 and bbox.y1 > bbox.y0:
                anchors.append((index, bbox))

        if not anchors:
            return children

        attached: dict[int, dict[str, list[tuple[float, int, Any]]]] = {}
        claimed: set[int] = set()
        for note_index, child in enumerate(children):
            if not isinstance(child, dict):
                continue
            if str(child.get("block_type", "")).casefold() not in note_types:
                continue
            note_bbox = self._bbox_from_polygon(self._polygon(child))
            if note_bbox is None or note_bbox.x1 <= note_bbox.x0 or note_bbox.y1 <= note_bbox.y0:
                continue

            note_width = note_bbox.x1 - note_bbox.x0
            note_height = note_bbox.y1 - note_bbox.y0
            max_gap = max(24.0, note_height * 2.5)
            best: tuple[float, int, str] | None = None
            for anchor_index, anchor_bbox in anchors:
                if anchor_index == note_index:
                    continue
                overlap = max(
                    0.0,
                    min(note_bbox.x1, anchor_bbox.x1) - max(note_bbox.x0, anchor_bbox.x0),
                )
                anchor_width = anchor_bbox.x1 - anchor_bbox.x0
                if overlap / max(1.0, min(note_width, anchor_width)) < 0.5:
                    continue

                if note_bbox.y0 >= anchor_bbox.y1:
                    gap = note_bbox.y0 - anchor_bbox.y1
                    position = "after"
                elif anchor_bbox.y0 >= note_bbox.y1:
                    gap = anchor_bbox.y0 - note_bbox.y1
                    position = "before"
                else:
                    continue
                if gap > max_gap:
                    continue

                horizontal_offset = abs(
                    (note_bbox.x0 + note_bbox.x1) / 2 - (anchor_bbox.x0 + anchor_bbox.x1) / 2
                ) / max(1.0, anchor_width)
                score = gap + horizontal_offset
                if best is None or score < best[0]:
                    best = (score, anchor_index, position)

            if best is None:
                continue
            _, anchor_index, position = best
            attached.setdefault(anchor_index, {"before": [], "after": []})[position].append(
                (note_bbox.y0, note_index, child)
            )
            claimed.add(note_index)

        if not claimed:
            return children

        ordered: list[Any] = []
        for index, child in enumerate(children):
            if index in claimed:
                continue
            placement = attached.get(index)
            if placement:
                ordered.extend(item[2] for item in sorted(placement["before"]))
            ordered.append(child)
            if placement:
                ordered.extend(item[2] for item in sorted(placement["after"]))
        return ordered

    def _page_number(self, page_payload: dict[str, Any], fallback: int) -> int:
        for key in ("page_number", "page", "page_id"):
            if key in page_payload:
                try:
                    value = int(page_payload[key])
                    return value + 1 if key == "page_id" else max(1, value)
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
        if (
            isinstance(raw, list)
            and len(raw) == 4
            and all(isinstance(item, (int, float)) for item in raw)
        ):
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
        parsed_rows = parse_table_rows(html)
        if not parsed_rows:
            return None
        rows = [[cell.text for cell in row] for row in parsed_rows]
        headers = rows[0] if len(rows) > 1 else []
        body_rows = rows[1:] if headers else rows
        cell_rows, cell_debug = self._table_cells_from_marker(
            node=node,
            parsed_rows=parsed_rows,
            coordinate_space=block.metadata["coordinate_space"],
        )
        header_cells = cell_rows[0] if headers and cell_rows else []
        body_cells = cell_rows[1:] if headers else cell_rows
        return TableModel(
            id=f"table-{block.reading_order_index}",
            page_numbers=[block.page_number],
            page=block.page_number,
            bbox=block.bbox,
            headers=headers,
            header_cells=header_cells,
            rows=body_rows,
            cells=body_cells,
            parse_mode="marker_html",
            debug={
                "marker_block_id": block.id,
                "render_from_block_text": True,
                "cell_coordinate_space": block.metadata["coordinate_space"],
                **cell_debug,
            },
        )

    def _table_cells_from_marker(
        self,
        *,
        node: dict[str, Any],
        parsed_rows: list[list[ParsedTableCell]],
        coordinate_space: dict[str, Any],
    ) -> tuple[list[list[TableModel.TableCell]], dict[str, Any]]:
        """Build logical cells and attach trustworthy Marker child geometry.

        The table HTML owns the logical structure. TableCell descendants are a
        geometry/provenance source only: an incomplete child list must never
        shift the geometry onto a different logical cell.
        """

        logical_rows: list[list[TableModel.TableCell]] = []
        descriptors: list[tuple[int, int, ParsedTableCell, TableModel.TableCell]] = []
        occupied_until: dict[int, int] = {}
        for row_index, parsed_row in enumerate(parsed_rows):
            logical_row: list[TableModel.TableCell] = []
            next_column = 0
            for parsed_cell in parsed_row:
                while any(
                    occupied_until.get(column, 0) > row_index
                    for column in range(next_column, next_column + parsed_cell.colspan)
                ):
                    next_column += 1
                cell = TableModel.TableCell(
                    text=parsed_cell.text,
                    rowspan=parsed_cell.rowspan,
                    colspan=parsed_cell.colspan,
                    row_index=row_index,
                    column_index=next_column,
                    extraction_metadata={
                        "parser": "marker_table_html",
                        "coordinate_space": coordinate_space,
                    },
                )
                logical_row.append(cell)
                descriptors.append((row_index, next_column, parsed_cell, cell))
                if parsed_cell.rowspan > 1:
                    for column in range(next_column, next_column + parsed_cell.colspan):
                        occupied_until[column] = row_index + parsed_cell.rowspan
                next_column += parsed_cell.colspan
            logical_rows.append(logical_row)

        children = list(self._marker_table_cell_nodes(node))
        unmatched = set(range(len(descriptors)))
        matched = 0
        valid_geometry = 0
        unmatched_children = 0
        for child_index, child in enumerate(children):
            marker_row = self._nonnegative_index(child, ("row_index", "row", "row_id"))
            marker_column = self._nonnegative_index(
                child,
                ("column_index", "col_index", "column", "col", "col_id"),
            )
            match_index: int | None = None
            if marker_row is not None and marker_column is not None:
                match_index = next(
                    (
                        index
                        for index in unmatched
                        if descriptors[index][0] == marker_row
                        and descriptors[index][1] == marker_column
                    ),
                    None,
                )

            child_text = self._node_text(child).strip()
            if (
                match_index is None
                and len(children) == len(descriptors)
                and child_index in unmatched
            ):
                descriptor_text = descriptors[child_index][2].text
                if self._same_cell_text(child_text, descriptor_text):
                    match_index = child_index

            if match_index is None:
                text_matches = [
                    index
                    for index in unmatched
                    if self._same_cell_text(child_text, descriptors[index][2].text)
                ]
                if len(text_matches) == 1:
                    match_index = text_matches[0]

            if match_index is None:
                unmatched_children += 1
                continue

            row_index, column_index, parsed_cell, current = descriptors[match_index]
            polygon = self._polygon(child)
            bbox = self._bbox_from_polygon(polygon)
            if not self._valid_cell_bbox(bbox, coordinate_space):
                polygon = None
                bbox = None
            marker_rowspan = self._positive_int(child, ("rowspan", "row_span"))
            marker_colspan = self._positive_int(child, ("colspan", "col_span", "column_span"))
            metadata = dict(current.extraction_metadata)
            metadata.update(
                {
                    "parser": "marker_table_cell",
                    "marker_id": child.get("id"),
                    "marker_block_type": child.get("block_type"),
                    "marker_child_order": child_index,
                    "marker_row_index": marker_row,
                    "marker_column_index": marker_column,
                    "coordinate_space": coordinate_space,
                }
            )
            enriched = current.model_copy(
                update={
                    "text": child_text if child_text or not parsed_cell.text else parsed_cell.text,
                    "rowspan": marker_rowspan or parsed_cell.rowspan,
                    "colspan": marker_colspan or parsed_cell.colspan,
                    "bbox": bbox,
                    "confidence": self._confidence(child),
                    "row_index": row_index,
                    "column_index": column_index,
                    "source_id": str(child.get("id")) if child.get("id") is not None else None,
                    "polygon": polygon or [],
                    "extraction_metadata": metadata,
                }
            )
            row_cell_index = next(
                index
                for index, existing in enumerate(logical_rows[row_index])
                if existing is current
            )
            logical_rows[row_index][row_cell_index] = enriched
            descriptors[match_index] = (row_index, column_index, parsed_cell, enriched)
            unmatched.remove(match_index)
            matched += 1
            if bbox is not None:
                valid_geometry += 1

        logical_count = len(descriptors)
        missing_geometry = logical_count - valid_geometry
        if logical_count and valid_geometry == logical_count:
            geometry_status = "complete"
            geometry_source = "marker_table_cell_polygons"
        elif valid_geometry:
            geometry_status = "partial"
            geometry_source = "marker_table_cell_polygons"
        else:
            geometry_status = "unavailable"
            geometry_source = "unavailable"

        return logical_rows, {
            "cell_geometry_source": geometry_source,
            "cell_geometry_status": geometry_status,
            "marker_table_cell_count": len(children),
            "matched_marker_table_cell_count": matched,
            "unmatched_marker_table_cell_count": unmatched_children,
            "logical_table_cell_count": logical_count,
            "unmatched_logical_table_cell_count": len(unmatched),
            "valid_marker_table_cell_geometry_count": valid_geometry,
            "missing_logical_table_cell_geometry_count": missing_geometry,
            "invalid_marker_table_cell_geometry_count": matched - valid_geometry,
        }

    def _marker_table_cell_nodes(self, node: dict[str, Any]):
        children = node.get("children")
        if not isinstance(children, list):
            return
        for child in children:
            if not isinstance(child, dict):
                continue
            child_type = str(child.get("block_type", "")).lower()
            if child_type == "tablecell":
                yield child
                continue
            if child_type not in {"table", "form"}:
                yield from self._marker_table_cell_nodes(child)

    def _nonnegative_index(self, node: dict[str, Any], keys: tuple[str, ...]) -> int | None:
        for source in (node, node.get("metadata")):
            if not isinstance(source, dict):
                continue
            for key in keys:
                if key not in source:
                    continue
                try:
                    value = int(source[key])
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    return value
        return None

    def _positive_int(self, node: dict[str, Any], keys: tuple[str, ...]) -> int | None:
        for source in (node, node.get("metadata")):
            if not isinstance(source, dict):
                continue
            for key in keys:
                if key not in source:
                    continue
                try:
                    value = int(source[key])
                except (TypeError, ValueError):
                    continue
                if 0 < value <= MAX_TABLE_CELL_SPAN:
                    return value
        return None

    def _same_cell_text(self, first: str, second: str) -> bool:
        def normalize(value: str) -> str:
            return re.sub(r"\s+", " ", value).strip().casefold()

        return normalize(first) == normalize(second)

    def _valid_cell_bbox(
        self,
        bbox: BoundingBox | None,
        coordinate_space: dict[str, Any],
    ) -> bool:
        if bbox is None:
            return False
        values = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        if not all(math.isfinite(value) for value in values):
            return False
        if bbox.x1 <= bbox.x0 or bbox.y1 <= bbox.y0 or bbox.x0 < 0 or bbox.y0 < 0:
            return False
        try:
            width = float(coordinate_space.get("width") or 0)
            height = float(coordinate_space.get("height") or 0)
        except (TypeError, ValueError):
            return False
        if width > 0 and bbox.x1 > width:
            return False
        if height > 0 and bbox.y1 > height:
            return False
        return True

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
        text = self._document_language_sample(blocks)
        if len(text.strip()) < 20:
            return None
        try:
            if detect is None:
                return None
            return detect(text)
        except Exception:
            return None

    def _document_language_sample(self, blocks: list[Block]) -> str:
        texts = [" ".join(block.text.split()) for block in blocks if block.text.strip()]
        if not texts:
            return ""

        # Sample the whole reading order so a long bilingual abstract on page 1
        # cannot determine the language for the rest of the paper. Cap both the
        # number of regions and each region's contribution for predictable cost.
        max_blocks = 500
        if len(texts) > max_blocks:
            last_index = len(texts) - 1
            texts = [
                texts[round(index * last_index / (max_blocks - 1))] for index in range(max_blocks)
            ]
        budget = 50_000
        per_block = max(1, (budget - len(texts)) // len(texts))
        sampled: list[str] = []
        for text in texts:
            if len(text) <= per_block:
                sampled.append(text)
                continue
            head = per_block // 2
            sampled.append(f"{text[:head]} {text[-(per_block - head - 1) :]}")
        return "\n".join(sampled)[:budget]


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
        return [
            [cell.strip() for cell in row]
            for row in parser.rows
            if any(cell.strip() for cell in row)
        ]

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
