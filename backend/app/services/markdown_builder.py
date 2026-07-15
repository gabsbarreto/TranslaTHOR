from __future__ import annotations

import html
import re
from pathlib import Path

from app.models.schema import Block, BlockType, DocumentModel, FigureAsset, TableModel
from app.services.table_markup import parse_table_rows, rows_have_consistent_shape


class MarkdownBuilder:
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
        }
        associated_caption_ids.update(
            caption.id for caption in table_caption_by_id.values()
        )
        figure_anchors = self._figure_anchors(document, caption_by_id)
        rendered_figures: set[str] = set()
        tables_by_page: dict[int, list[TableModel]] = {}
        for table in document.tables:
            for page_number in table.page_numbers:
                tables_by_page.setdefault(page_number, []).append(table)
        rendered_tables: set[str] = set()
        for block in document.blocks:
            if self._is_marker_table_cell(block):
                continue
            anchored_figures = figure_anchors.get(block.id, [])
            if (
                block.block_type not in {BlockType.TABLE, BlockType.FIGURE}
                and not block.text.strip()
                and not anchored_figures
            ):
                continue
            if block.page_number != page:
                page = block.page_number
                lines.append(f"\n<!-- page: {page} -->\n")

            if block.block_type == BlockType.FIGURE:
                # Text detected inside a graph remains part of the captured visual and
                # is deliberately not emitted as translatable body text.
                pass
            elif block.block_type == BlockType.HEADING:
                lines.append(f"## {block.text}\n")
            elif block.block_type == BlockType.LIST:
                lines.append(self._list_markdown(block.text))
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
                elif block.text.strip():
                    lines.append(block.text.strip() + "\n")
            elif block.block_type == BlockType.CAPTION:
                if block.id not in associated_caption_ids:
                    lines.append(f"*{block.text}*\n")
            elif block.block_type == BlockType.FOOTNOTE:
                lines.append(f"<small>[Footnote] {block.text}</small>\n")
            elif block.block_type == BlockType.REFERENCE:
                lines.append(f"- {block.text}\n")
            elif block.block_type in {BlockType.HEADER, BlockType.FOOTER}:
                lines.append(f"<small>{block.text}</small>\n")
            else:
                lines.append(block.text + "\n")

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
            if debug.get("marker_block_id") == block.id or debug.get("source_block_id") == block.id:
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

        parts = ['<figure class="document-table">']
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
        that link existed still contain a reliable table block followed by its
        caption, so reconstruction recovers that narrow relationship at render
        time and suppresses the standalone duplicate caption.
        """

        resolved: dict[str, Block] = {}
        claimed_caption_ids: set[str] = set()
        for table in document.tables:
            if not table.caption_block_id:
                continue
            caption = caption_by_id.get(table.caption_block_id)
            if caption is None:
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

        for table in document.tables:
            if table.id in resolved:
                continue
            table_block = table_block_by_table_id.get(table.id)
            if table_block is None:
                continue
            page_number = table_block.page_number

            caption_candidates = [
                caption
                for caption in caption_by_id.values()
                if caption.page_number == page_number
                and caption.id not in claimed_caption_ids
                and 0 < caption.reading_order_index - table_block.reading_order_index <= 2
            ]
            if table_block.bbox is not None:
                caption_candidates = [
                    caption
                    for caption in caption_candidates
                    if caption.bbox is None
                    or (
                        caption.bbox.y0 >= table_block.bbox.y1 - 4.0
                        and self._horizontal_overlap_ratio(
                            table_block.bbox,
                            caption.bbox,
                        )
                        >= 0.2
                    )
                ]
            if not caption_candidates:
                continue
            caption = min(
                caption_candidates,
                key=lambda item: item.reading_order_index,
            )
            resolved[table.id] = caption
            claimed_caption_ids.add(caption.id)
        return resolved

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
            rows = [[type("Cell", (), {"text": c, "rowspan": 1, "colspan": 1})() for c in r] for r in table.rows]

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
