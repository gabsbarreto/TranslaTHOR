from __future__ import annotations

import html
import re
from pathlib import Path

from app.models.schema import Block, BlockType, DocumentModel, FigureAsset, TableModel


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
        associated_caption_ids = {
            figure.caption_block_id
            for figure in document.figures
            if figure.caption_block_id
        }
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
                )
                if matched_table is not None:
                    if matched_table.id in rendered_tables:
                        continue
                    if self._render_table_from_block_text(matched_table, block.text):
                        lines.append(block.text.strip() + "\n")
                    else:
                        title = matched_table.caption or f"Table {len(rendered_tables) + 1}"
                        lines.append(f"\n### {title}\n")
                        lines.append(self._table_html(matched_table))
                        if matched_table.notes:
                            lines.append(f"\n<small>{matched_table.notes}</small>\n")
                        if matched_table.fallback_image_path:
                            lines.append(f"\n![{title}]({matched_table.fallback_image_path})\n")
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
            title = table.caption or f"Table {len(rendered_tables) + 1}"
            lines.append(f"\n### {title}\n")
            lines.append(self._table_html(table))
            if table.notes:
                lines.append(f"\n<small>{table.notes}</small>\n")
            if table.fallback_image_path:
                lines.append(f"\n![{title}]({table.fallback_image_path})\n")
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
    ) -> TableModel | None:
        for table in page_tables:
            if getattr(table, "debug", {}).get("marker_block_id") == block.id:
                return table
        if page_tables:
            return page_tables[0]
        return None

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
        return str(text).replace("\n", "<br>").replace("|", "\\|").strip()

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
