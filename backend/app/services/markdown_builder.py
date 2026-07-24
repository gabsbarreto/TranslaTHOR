from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from app.models.schema import BlockType, DocumentModel, TableModel


class MarkdownBuilder:
    def build(self, document: DocumentModel, marker_markdown: str | None = None) -> str:
        if marker_markdown and marker_markdown.strip():
            base = marker_markdown.strip()
            extras = self._build_figure_extras(document)
            return base + ("\n\n" + extras if extras else "")

        lines: list[str] = []
        page = 0
        tables_by_page: dict[int, list[TableModel]] = {}
        for table in document.tables:
            for page_number in table.page_numbers:
                tables_by_page.setdefault(page_number, []).append(table)
        rendered_tables: set[str] = set()
        rendered_figures: set[str] = set()
        figures_by_id = {figure.id: figure for figure in document.figures}
        blocks_by_id = {block.id: block for block in document.blocks}
        for block in document.blocks:
            if self._is_marker_table_cell(block):
                continue
            figure = None
            if block.block_type == BlockType.FIGURE:
                figure = figures_by_id.get(str(block.metadata.get("figure_asset_id", "")))
            if (
                block.block_type not in {BlockType.TABLE, BlockType.FIGURE}
                and not block.text.strip()
            ):
                continue
            if block.page_number != page:
                page = block.page_number
                lines.append(f"\n<!-- page: {page} -->\n")

            if block.block_type == BlockType.FIGURE:
                if figure is not None:
                    caption = blocks_by_id.get(figure.caption_block_id or "")
                    alt = caption.text if caption is not None else figure.id
                    if figure.image_path:
                        destination = self._image_destination(figure.image_path)
                        lines.append(f"![{alt}]({destination})\n")
                    else:
                        lines.append("_Figure preserved as placeholder._\n")
                    rendered_figures.add(figure.id)
            elif block.block_type == BlockType.HEADING:
                lines.append(f"## {block.text}\n")
            elif block.block_type == BlockType.LIST:
                lines.append(self._list_markdown(block.text))
            elif block.block_type == BlockType.TABLE:
                matched_table = self._table_for_block(
                    document, block, tables_by_page.get(block.page_number, [])
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
                            destination = self._image_destination(matched_table.fallback_image_path)
                            lines.append(f"\n![{title}]({destination})\n")
                    rendered_tables.add(matched_table.id)
                elif block.text.strip():
                    lines.append(block.text.strip() + "\n")
            elif block.block_type == BlockType.CAPTION:
                lines.append(f"*{block.text}*\n")
            elif block.block_type == BlockType.FOOTNOTE:
                lines.append(f"<small>[Footnote] {block.text}</small>\n")
            elif block.block_type == BlockType.REFERENCE:
                lines.append(f"- {block.text}\n")
            elif block.block_type in {BlockType.HEADER, BlockType.FOOTER}:
                lines.append(f"<small>{block.text}</small>\n")
            else:
                lines.append(block.text + "\n")

        for table in document.tables:
            if table.id in rendered_tables:
                continue
            title = table.caption or f"Table {len(rendered_tables) + 1}"
            lines.append(f"\n### {title}\n")
            lines.append(self._table_html(table))
            if table.notes:
                lines.append(f"\n<small>{table.notes}</small>\n")
            if table.fallback_image_path:
                destination = self._image_destination(table.fallback_image_path)
                lines.append(f"\n![{title}]({destination})\n")
            rendered_tables.add(table.id)

        extra = self._build_figure_extras(document, exclude_ids=rendered_figures)
        if extra:
            lines.append(extra)

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
        block,
        page_tables: list[TableModel],
    ) -> TableModel | None:
        for table in page_tables:
            if getattr(table, "debug", {}).get("marker_block_id") == block.id:
                return table
        if page_tables:
            return page_tables[0]
        return None

    def _render_table_from_block_text(self, table, block_text: str) -> bool:
        debug = getattr(table, "debug", {})
        return bool(
            (debug.get("render_from_block_text") or debug.get("marker_block_id"))
            and "<table" in block_text.lower()
        )

    def _is_marker_table_cell(self, block) -> bool:
        metadata = getattr(block, "metadata", {}) or {}
        return str(metadata.get("marker_block_type", "")).lower() == "tablecell"

    def _escape_table_cell(self, text: str) -> str:
        return str(text).replace("\n", "<br>").replace("|", "\\|").strip()

    def _table_html(self, table) -> str:
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

    def _build_figure_extras(
        self,
        document: DocumentModel,
        *,
        exclude_ids: set[str] | None = None,
    ) -> str:
        lines: list[str] = []
        figures = [figure for figure in document.figures if figure.id not in (exclude_ids or set())]
        if figures:
            lines.append("\n## Figures\n")
            for idx, figure in enumerate(figures, start=1):
                lines.append(f"\n### Figure {idx}\n")
                if figure.image_path:
                    destination = self._image_destination(figure.image_path)
                    lines.append(f"![Figure {idx}]({destination})\n")
                else:
                    lines.append("_Figure preserved as placeholder._\n")
        return "\n".join(lines).strip()

    def _image_destination(self, value: str) -> str:
        if urlparse(value).scheme:
            return value
        path = Path(value).expanduser()
        return path.as_uri() if path.is_absolute() else value
