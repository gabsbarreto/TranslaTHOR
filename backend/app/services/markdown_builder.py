from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from app.models.schema import Block, BlockType, DocumentModel, FigureAsset, TableModel
from app.services.table_markup import parse_table_rows, rows_have_consistent_shape


QWEN_REMOTE_MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(\s*<?(?:https?|ftp)://[^)>\s]+>?"
    r"(?:\s+[\"'][^\"']*[\"'])?\s*\)",
    flags=re.IGNORECASE,
)


@dataclass
class _CaptionFlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: tuple[float, ...]
    assignment: tuple[str, str] | None = None


class MarkdownBuilder:
    LONG_TABLE_ROW_THRESHOLD = 18

    def build(self, document: DocumentModel, marker_markdown: str | None = None) -> str:
        if marker_markdown and marker_markdown.strip() and not document.blocks:
            return marker_markdown.strip()

        lines: list[str] = []
        page = 0
        caption_by_id = {
            block.id: block
            for block in document.blocks
            if block.block_type == BlockType.CAPTION
        }
        table_caption_by_id = self._table_caption_map(document, caption_by_id)
        associated_caption_ids = {
            figure.caption_block_id
            for figure in document.figures
            if figure.caption_block_id
            and (
                self._available_asset(figure.vector_path)
                or self._available_asset(figure.image_path)
            )
        }
        associated_caption_ids.update(
            caption.id for caption in table_caption_by_id.values()
        )
        figure_anchors = self._figure_anchors(document, caption_by_id)
        translation_flow_text = self._translation_flow_text(document)
        rendered_figures: set[str] = set()
        tables_by_page: dict[int, list[TableModel]] = {}
        for table in document.tables:
            for page_number in table.page_numbers:
                tables_by_page.setdefault(page_number, []).append(table)
        rendered_tables: set[str] = set()
        for block in document.blocks:
            if self._is_marker_table_cell(block):
                continue
            block_text = translation_flow_text.get(
                block.id,
                self._readable_block_text(block, document.figures),
            )
            anchored_figures = figure_anchors.get(block.id, [])
            if (
                block.block_type not in {BlockType.TABLE, BlockType.FIGURE}
                and not block_text.strip()
                and not anchored_figures
            ):
                continue
            if block.page_number != page:
                page = block.page_number
                lines.append(f"\n<!-- page: {page} -->\n")

            if block.block_type == BlockType.FIGURE:
                # Text detected inside a graph remains part of the captured visual and
                # is deliberately not emitted as translatable body text.
                lines.extend(
                    f"{fallback}\n"
                    for fallback in self._missing_qwen_image_fallbacks(
                        block,
                        document.figures,
                    )
                )
            elif block.block_type == BlockType.HEADING:
                lines.append(f"## {block_text}\n")
            elif block.block_type == BlockType.LIST:
                lines.append(self._list_markdown(block_text))
            elif block.block_type == BlockType.TABLE:
                matched_table = self._table_for_block(
                    document,
                    block,
                    tables_by_page.get(block.page_number, []),
                    rendered_tables,
                )
                if matched_table is not None:
                    if matched_table.id in rendered_tables:
                        continue
                    lines.append(
                        self._table_figure_html(
                            matched_table,
                            block=block,
                            caption_by_id=caption_by_id,
                            caption_block=table_caption_by_id.get(matched_table.id),
                            ordinal=len(rendered_tables) + 1,
                        )
                    )
                    rendered_tables.add(matched_table.id)
                elif block_text.strip():
                    lines.append(block_text.strip() + "\n")
            elif block.block_type == BlockType.CAPTION:
                if block.id not in associated_caption_ids:
                    lines.append(f"*{block_text}*\n")
            elif block.block_type == BlockType.FOOTNOTE:
                lines.append(f"<small>[Footnote] {block_text}</small>\n")
            elif block.block_type == BlockType.REFERENCE:
                lines.append(f"- {block_text}\n")
            elif block.block_type in {BlockType.HEADER, BlockType.FOOTER}:
                lines.append(f"<small>{block_text}</small>\n")
            else:
                if block_text:
                    lines.append(block_text + "\n")

            for figure in anchored_figures:
                if figure.id in rendered_figures:
                    continue
                figure_html = self._figure_html(figure, caption_by_id)
                if figure_html:
                    lines.append(figure_html)
                    rendered_figures.add(figure.id)

        for table in document.tables:
            if table.id in rendered_tables:
                continue
            lines.append(
                self._table_figure_html(
                    table,
                    block=None,
                    caption_by_id=caption_by_id,
                    caption_block=table_caption_by_id.get(table.id),
                    ordinal=len(rendered_tables) + 1,
                )
            )
            rendered_tables.add(table.id)

        return "\n".join(lines)

    def _translation_flow_text(self, document: DocumentModel) -> dict[str, str]:
        groups: dict[str, list[Block]] = {}
        for block in document.blocks:
            group_id = str(
                (block.metadata or {}).get("translation_placement_group_id") or ""
            ).strip()
            if group_id and block.block_type == BlockType.PARAGRAPH:
                groups.setdefault(group_id, []).append(block)

        rendered: dict[str, str] = {}
        for blocks in groups.values():
            try:
                ordered = sorted(
                    blocks,
                    key=lambda block: int(
                        block.metadata.get("translation_placement_index")
                    ),
                )
                expected_count = int(
                    ordered[0].metadata.get("translation_placement_count")
                )
                indexes = [
                    int(block.metadata.get("translation_placement_index"))
                    for block in ordered
                ]
            except (TypeError, ValueError):
                continue
            if expected_count != len(ordered) or indexes != list(range(expected_count)):
                continue
            parts = [
                self._readable_block_text(block, document.figures).strip()
                for block in ordered
            ]
            if not all(parts):
                continue
            text = ""
            for part in parts:
                if text.endswith("-"):
                    text = f"{text[:-1]}{part}"
                else:
                    text = f"{text} {part}".strip()
            rendered[ordered[0].id] = text
            rendered.update({block.id: "" for block in ordered[1:]})
        return rendered

    def _readable_block_text(
        self,
        block: Block,
        figures: list[FigureAsset],
    ) -> str:
        # Qwen can emit a generated remote image URL instead of source-page
        # geometry. Suppress it only when each token can be replaced by a
        # materialised job-local FigureAsset from the same source region.
        # Otherwise retain its alt text as an explicit missing-asset warning.
        matches = list(QWEN_REMOTE_MARKDOWN_IMAGE_PATTERN.finditer(block.text))
        if not matches or not self._is_qwen_ocr_block(block):
            return block.text

        local_figures = iter(self._matching_materialized_figures(block, figures))

        def replace(match: re.Match[str]) -> str:
            if next(local_figures, None) is not None:
                return ""
            return self._missing_image_fallback(match)

        return QWEN_REMOTE_MARKDOWN_IMAGE_PATTERN.sub(replace, block.text).strip()

    def _missing_qwen_image_fallbacks(
        self,
        block: Block,
        figures: list[FigureAsset],
    ) -> list[str]:
        if not self._is_qwen_ocr_block(block):
            return []
        remote_images = list(QWEN_REMOTE_MARKDOWN_IMAGE_PATTERN.finditer(block.text))
        local_asset_count = len(self._matching_materialized_figures(block, figures))
        return [
            self._missing_image_fallback(match)
            for match in remote_images[local_asset_count:]
        ]

    def _missing_image_fallback(self, match: re.Match[str]) -> str:
        alt = re.sub(r"\s+", " ", match.group("alt")).strip()
        return f"[Image unavailable: {alt}]" if alt else "[Image unavailable]"

    def _is_qwen_ocr_block(self, block: Block) -> bool:
        parser = str((block.metadata or {}).get("parser", ""))
        return "qwen" in parser.casefold()

    def _matching_materialized_figures(
        self,
        block: Block,
        figures: list[FigureAsset],
    ) -> list[FigureAsset]:
        block_region_ids = {
            str(region_id)
            for region_id in (block.metadata or {}).get("source_region_ids", [])
            if region_id is not None
        }
        matched: list[FigureAsset] = []
        for figure in figures:
            if figure.page_number != block.page_number:
                continue
            if not self._has_materialized_local_asset(figure):
                continue
            source_block_match = block.id in figure.source_block_ids
            source_region_match = bool(
                block_region_ids
                and block_region_ids.intersection(map(str, figure.source_region_ids))
            )
            if source_block_match or source_region_match:
                matched.append(figure)
        return matched

    def _has_materialized_local_asset(self, figure: FigureAsset) -> bool:
        for value in (figure.vector_path, figure.image_path):
            if not value:
                continue
            if re.match(r"^[a-z][a-z0-9+.-]*://", value, flags=re.IGNORECASE):
                continue
            if Path(value).expanduser().is_file():
                return True
        return False

    def _list_markdown(self, text: str) -> str:
        stripped = self._clean_list_text(text)
        if self._is_explicit_numbered_item(stripped):
            return f"{stripped}\n"
        return f"- {stripped}\n"

    def _clean_list_text(self, text: str) -> str:
        stripped = text.strip()
        # Marker can merge a repeated section heading into the first list item
        # at a page boundary, for example "REFERENCES 11. Johns ...".
        return re.sub(
            r"^(references|referencias|bibliography|bibliograf[ií]a)\s+(\d+[.)]\s+)",
            r"\2",
            stripped,
            flags=re.IGNORECASE,
        )

    def _is_explicit_numbered_item(self, text: str) -> bool:
        return bool(re.match(r"^\d+[.)]\s+\S", text.strip()))

    def _table_for_block(
        self,
        document: DocumentModel,
        block: Block,
        page_tables: list[TableModel],
        rendered_table_ids: set[str],
    ) -> TableModel | None:
        for table in page_tables:
            debug = getattr(table, "debug", {})
            if block.id in {
                debug.get("marker_block_id"),
                debug.get("source_block_id"),
                debug.get("surya2_block_id"),
            }:
                return table
        return next(
            (table for table in page_tables if table.id not in rendered_table_ids),
            None,
        )

    def _render_table_from_block_text(self, table: TableModel, block_text: str) -> bool:
        debug = getattr(table, "debug", {})
        return bool(
            (debug.get("render_from_block_text") or debug.get("marker_block_id"))
            and "<table" in block_text.lower()
        )

    def _is_marker_table_cell(self, block: Block) -> bool:
        metadata = getattr(block, "metadata", {}) or {}
        return str(metadata.get("marker_block_type", "")).lower() == "tablecell"

    def _escape_table_cell(self, text: str) -> str:
        return html.escape(str(text).strip()).replace("\n", "<br>")

    def _table_figure_html(
        self,
        table: TableModel,
        *,
        block: Block | None,
        caption_by_id: dict[str, Block],
        caption_block: Block | None,
        ordinal: int,
    ) -> str:
        rendered_table = table
        if block is not None and self._render_table_from_block_text(table, block.text):
            table_markup = block.text.strip()
        else:
            if block is not None:
                rendered_table = self._table_with_block_translation(table, block)
            table_markup = (
                self._table_html(rendered_table)
                if rendered_table.headers or rendered_table.rows or rendered_table.cells
                else ""
            )

        if caption_block is None and table.caption_block_id:
            caption_block = caption_by_id.get(table.caption_block_id)
        caption = (
            caption_block.text.strip()
            if caption_block is not None and caption_block.text.strip()
            else str(table.caption or "").strip()
        )
        if not caption:
            caption = f"Table {ordinal}"

        parsed_rows = parse_table_rows(table_markup)
        row_count = len(parsed_rows) or len(rendered_table.rows) + bool(rendered_table.headers)
        figure_class = "document-table"
        if row_count >= self.LONG_TABLE_ROW_THRESHOLD:
            figure_class += " document-table--long"
        parts = [f'<figure class="{figure_class}">']
        if table_markup.strip():
            parts.append(table_markup)
        elif table.fallback_image_path:
            path = html.escape(str(Path(table.fallback_image_path).resolve()), quote=True)
            parts.append(f'<img src="{path}" alt="{html.escape(caption, quote=True)}">')
        if table.notes:
            parts.append(f'<small class="table-notes">{html.escape(table.notes)}</small>')
        parts.append(f"<figcaption>{html.escape(caption)}</figcaption>")
        parts.append("</figure>\n")
        return "\n".join(parts)

    def _table_caption_map(
        self,
        document: DocumentModel,
        caption_by_id: dict[str, Block],
    ) -> dict[str, Block]:
        """Resolve captions without mutating older persisted structured JSON.

        Newly extracted tables carry ``caption_block_id``. Jobs created before
        that link existed still contain a table and caption on the same page,
        but OCR reading order can put the caption on either side of the table.
        Recover those relationships from bounded spatial and reading-order
        evidence, then suppress the standalone duplicate caption.
        """

        resolved: dict[str, Block] = {}
        claimed_caption_ids: set[str] = set()
        figure_caption_ids = {
            figure.caption_block_id
            for figure in document.figures
            if figure.caption_block_id in caption_by_id
        }
        for table in document.tables:
            if not table.caption_block_id:
                continue
            caption = caption_by_id.get(table.caption_block_id)
            if caption is None or caption.id in claimed_caption_ids:
                continue
            resolved[table.id] = caption
            claimed_caption_ids.add(caption.id)

        table_blocks_by_page: dict[int, list[Block]] = {}
        for block in document.blocks:
            if block.block_type == BlockType.TABLE:
                table_blocks_by_page.setdefault(block.page_number, []).append(block)
        for blocks in table_blocks_by_page.values():
            blocks.sort(key=lambda item: item.reading_order_index)

        claimed_table_block_ids: set[str] = set()
        table_block_by_table_id: dict[str, Block] = {}
        for table in document.tables:
            page_number = table.page_numbers[0] if table.page_numbers else table.page
            if page_number is None:
                continue
            candidates = table_blocks_by_page.get(page_number, [])
            source_id = str(
                table.debug.get("source_block_id")
                or table.debug.get("marker_block_id")
                or ""
            )
            table_block = next(
                (candidate for candidate in candidates if candidate.id == source_id),
                None,
            )
            if table_block is None:
                table_block = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.id not in claimed_table_block_ids
                    ),
                    None,
                )
            if table_block is None:
                continue
            claimed_table_block_ids.add(table_block.id)
            table_block_by_table_id[table.id] = table_block

        page_heights = {page.page_number: page.height for page in document.pages}
        ranked_candidates: list[tuple[tuple[float, ...], str, Block]] = []
        for table in document.tables:
            if table.id in resolved:
                continue
            table_block = table_block_by_table_id.get(table.id)
            if table_block is None:
                continue
            table_bbox = table_block.bbox or table.bbox
            for caption in caption_by_id.values():
                if (
                    caption.page_number != table_block.page_number
                    or caption.id in claimed_caption_ids
                    or caption.id in figure_caption_ids
                ):
                    continue
                score = self._table_caption_score(
                    table_block,
                    table_bbox,
                    caption,
                    page_height=page_heights.get(table_block.page_number),
                )
                if score is not None:
                    ranked_candidates.append((score, table.id, caption))

        inferred = self._minimum_cost_caption_assignment(ranked_candidates)
        resolved.update(inferred)
        return resolved

    def _minimum_cost_caption_assignment(
        self,
        candidates: list[tuple[tuple[float, ...], str, Block]],
    ) -> dict[str, Block]:
        """Choose the largest reliable one-to-one assignment at minimum cost.

        A locally closest pair is not always the best page-level assignment: a
        flexible table can otherwise consume the only caption available to a
        neighbouring table. Successive shortest augmenting paths give maximum
        cardinality first and minimum aggregate evidence cost second. The
        residual edges allow an earlier choice to be reassigned when that
        produces a better complete matching.
        """

        if not candidates:
            return {}
        caption_by_id = {caption.id: caption for _score, _table_id, caption in candidates}
        table_ids = sorted({table_id for _score, table_id, _caption in candidates})
        caption_ids = sorted(
            caption_by_id,
            key=lambda caption_id: (
                caption_by_id[caption_id].page_number,
                caption_by_id[caption_id].reading_order_index,
                caption_id,
            ),
        )
        source = 0
        table_offset = 1
        caption_offset = table_offset + len(table_ids)
        sink = caption_offset + len(caption_ids)
        graph: list[list[_CaptionFlowEdge]] = [[] for _ in range(sink + 1)]
        table_node = {table_id: table_offset + index for index, table_id in enumerate(table_ids)}
        caption_node = {
            caption_id: caption_offset + index
            for index, caption_id in enumerate(caption_ids)
        }
        zero = tuple(0.0 for _ in candidates[0][0])

        def add_edge(
            start: int,
            end: int,
            cost: tuple[float, ...],
            *,
            assignment: tuple[str, str] | None = None,
        ) -> None:
            forward = _CaptionFlowEdge(
                to=end,
                reverse=len(graph[end]),
                capacity=1,
                cost=cost,
                assignment=assignment,
            )
            reverse = _CaptionFlowEdge(
                to=start,
                reverse=len(graph[start]),
                capacity=0,
                cost=tuple(-value for value in cost),
            )
            graph[start].append(forward)
            graph[end].append(reverse)

        for table_id in table_ids:
            add_edge(source, table_node[table_id], zero)
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (
                item[1],
                item[2].page_number,
                item[2].reading_order_index,
                item[2].id,
            ),
        )
        for score, table_id, caption in ordered_candidates:
            add_edge(
                table_node[table_id],
                caption_node[caption.id],
                score,
                assignment=(table_id, caption.id),
            )
        for caption_id in caption_ids:
            add_edge(caption_node[caption_id], sink, zero)

        # Bellman-Ford is intentionally used instead of a greedy augmenting
        # path: residual edges have negative tuple costs after the first match.
        while True:
            distances: list[tuple[float, ...] | None] = [None] * len(graph)
            previous: list[tuple[int, int] | None] = [None] * len(graph)
            distances[source] = zero
            for _ in range(len(graph) - 1):
                changed = False
                for node, edges in enumerate(graph):
                    distance = distances[node]
                    if distance is None:
                        continue
                    for edge_index, edge in enumerate(edges):
                        if edge.capacity <= 0:
                            continue
                        candidate_distance = tuple(
                            left + right for left, right in zip(distance, edge.cost, strict=True)
                        )
                        if distances[edge.to] is None or candidate_distance < distances[edge.to]:
                            distances[edge.to] = candidate_distance
                            previous[edge.to] = (node, edge_index)
                            changed = True
                if not changed:
                    break
            if previous[sink] is None:
                break
            node = sink
            while node != source:
                prior = previous[node]
                if prior is None:  # pragma: no cover - guarded by the complete path above
                    break
                start, edge_index = prior
                edge = graph[start][edge_index]
                edge.capacity -= 1
                graph[node][edge.reverse].capacity += 1
                node = start

        resolved: dict[str, Block] = {}
        for table_id in table_ids:
            for edge in graph[table_node[table_id]]:
                if edge.assignment is None or edge.capacity != 0:
                    continue
                assigned_table_id, caption_id = edge.assignment
                resolved[assigned_table_id] = caption_by_id[caption_id]
                break
        return resolved

    def _table_caption_score(
        self,
        table_block: Block,
        table_bbox,
        caption: Block,
        *,
        page_height: float | None,
    ) -> tuple[float, ...] | None:
        """Return a lower-is-better score for a reliable legacy association."""

        order_delta = caption.reading_order_index - table_block.reading_order_index
        order_distance = abs(order_delta)
        if table_bbox is not None and caption.bbox is not None:
            # A generous but finite order window tolerates column-order errors
            # without associating unrelated captions elsewhere on the page.
            if order_distance > 6:
                return None
            overlap = self._horizontal_overlap_ratio(table_bbox, caption.bbox)
            if overlap < 0.2:
                return None

            caption_above = caption.bbox.y1 <= table_bbox.y0 + 4.0
            caption_below = caption.bbox.y0 >= table_bbox.y1 - 4.0
            if not (caption_above or caption_below):
                return None
            edge_gap = (
                max(0.0, table_bbox.y0 - caption.bbox.y1)
                if caption_above
                else max(0.0, caption.bbox.y0 - table_bbox.y1)
            )
            height = max(1.0, float(page_height or 800.0))
            max_edge_gap = max(36.0, min(120.0, height * 0.12))
            if edge_gap > max_edge_gap:
                return None

            # Spatial evidence ranks ahead of the legacy no-bbox fallback.
            # Edge proximity is strongest, with order distance and overlap as
            # deterministic tie breakers.
            return (0.0, edge_gap / height, float(order_distance), -overlap)

        # Older structured JSON may have no usable boxes. Preserve the narrow
        # historical fallback only for an immediately following caption; an
        # above-caption relationship is not safe to infer without geometry.
        if not 0 < order_delta <= 2:
            return None
        return (1.0, float(order_delta), 0.0, 0.0)

    def _horizontal_overlap_ratio(
        self,
        first,
        second,
    ) -> float:
        overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
        return overlap / max(1.0, min(first.x1 - first.x0, second.x1 - second.x0))

    def _table_with_block_translation(self, table: TableModel, block: Block) -> TableModel:
        source_hint = str(
            block.metadata.get("source_text_before_cleaning")
            or block.metadata.get("source_text")
            or ""
        )
        rows = parse_table_rows(block.text, source_hint=source_hint)
        if not rows_have_consistent_shape(rows):
            return table

        expected_rows = (1 if table.headers else 0) + len(table.rows)
        expected_columns = len(table.headers) or (len(table.rows[0]) if table.rows else 0)
        if expected_rows and len(rows) != expected_rows:
            return table
        if expected_columns and any(len(row) != expected_columns for row in rows):
            return table

        rendered = table.model_copy(deep=True)
        body_start = 0
        if table.headers:
            rendered.headers = [cell.text for cell in rows[0]]
            body_start = 1
        body = rows[body_start:]
        rendered.rows = [[cell.text for cell in row] for row in body]
        rendered.cells = [
            [
                (
                    table.cells[row_index][column_index].model_copy(update={"text": cell.text})
                    if row_index < len(table.cells)
                    and column_index < len(table.cells[row_index])
                    else TableModel.TableCell(text=cell.text)
                )
                for column_index, cell in enumerate(row)
            ]
            for row_index, row in enumerate(body)
        ]
        return rendered

    def _table_html(self, table: TableModel) -> str:
        rows = table.cells or []
        if not rows and table.rows:
            rows = [
                [type("Cell", (), {"text": c, "rowspan": 1, "colspan": 1})() for c in r]
                for r in table.rows
            ]

        lines: list[str] = ['<table class="structured-table">']
        if table.headers:
            lines.append("<thead><tr>")
            for h in table.headers:
                lines.append(f"<th>{self._escape_table_cell(h)}</th>")
            lines.append("</tr></thead>")
        lines.append("<tbody>")
        for row in rows:
            lines.append("<tr>")
            for cell in row:
                rowspan = int(getattr(cell, "rowspan", 1) or 1)
                colspan = int(getattr(cell, "colspan", 1) or 1)
                attrs = []
                if rowspan > 1:
                    attrs.append(f'rowspan="{rowspan}"')
                if colspan > 1:
                    attrs.append(f'colspan="{colspan}"')
                attr = (" " + " ".join(attrs)) if attrs else ""
                lines.append(f"<td{attr}>{self._escape_table_cell(getattr(cell, 'text', ''))}</td>")
            lines.append("</tr>")
        lines.append("</tbody></table>")
        return "\n".join(lines)

    def _figure_anchors(
        self,
        document: DocumentModel,
        caption_by_id: dict[str, Block],
    ) -> dict[str, list[FigureAsset]]:
        block_positions = {block.id: index for index, block in enumerate(document.blocks)}
        anchors: dict[str, list[FigureAsset]] = {}
        for figure in document.figures:
            if not (figure.vector_path or figure.image_path):
                continue
            source_positions = [
                (block_positions[block_id], block_id)
                for block_id in figure.source_block_ids
                if block_id in block_positions
            ]
            if not source_positions:
                continue
            original_position, anchor_id = min(source_positions)
            caption = (
                caption_by_id.get(figure.caption_block_id)
                if figure.caption_block_id
                else None
            )
            mention = self._first_figure_mention(document, caption)
            if mention is not None:
                mention_position = block_positions.get(mention.id, -1)
                if mention_position > original_position:
                    anchor_id = mention.id
            anchors.setdefault(anchor_id, []).append(figure)
        return anchors

    def _first_figure_mention(
        self,
        document: DocumentModel,
        caption: Block | None,
    ) -> Block | None:
        if caption is None:
            return None
        label = re.search(
            r"\b(?:figure|fig\.?)\s*(?P<number>\d+[a-z]?)\b",
            caption.text,
            flags=re.IGNORECASE,
        )
        if label is None:
            return None
        number = re.escape(label.group("number"))
        reference = re.compile(
            rf"\b(?:figure|fig\.?)\s*{number}\b",
            flags=re.IGNORECASE,
        )
        for block in document.blocks:
            if block.block_type in {BlockType.FIGURE, BlockType.CAPTION}:
                continue
            if reference.search(block.text):
                return block
        return None

    def _figure_html(
        self,
        figure: FigureAsset,
        caption_by_id: dict[str, Block],
    ) -> str:
        asset_path = self._available_asset(figure.vector_path) or self._available_asset(
            figure.image_path
        )
        if not asset_path:
            return ""
        source = self._asset_uri(asset_path)
        caption = (
            caption_by_id.get(figure.caption_block_id)
            if figure.caption_block_id
            else None
        )
        caption_html = ""
        if caption is not None and caption.text.strip():
            caption_html = f"<figcaption>{html.escape(caption.text.strip())}</figcaption>"
        alt = html.escape(caption.text.strip() if caption is not None else figure.id)
        return (
            f'<figure class="document-figure" data-figure-id="{html.escape(figure.id)}">'
            f'<img src="{html.escape(source, quote=True)}" alt="{alt}" />'
            f"{caption_html}</figure>\n"
        )

    def _asset_uri(self, value: str) -> str:
        if re.match(r"^[a-z][a-z0-9+.-]*://", value, flags=re.IGNORECASE):
            return value
        return Path(value).expanduser().resolve().as_uri()

    def _available_asset(self, value: str | None) -> str | None:
        if not value:
            return None
        if re.match(r"^[a-z][a-z0-9+.-]*://", value, flags=re.IGNORECASE):
            return value
        return value if Path(value).expanduser().is_file() else None
