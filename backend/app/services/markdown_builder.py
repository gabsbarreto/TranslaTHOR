from __future__ import annotations

from app.models.schema import BlockType, DocumentModel


class MarkdownBuilder:
    def build(self, document: DocumentModel, marker_markdown: str | None = None) -> str:
        if marker_markdown and marker_markdown.strip():
            base = marker_markdown.strip()
            extras = self._build_figure_extras(document)
            return base + ("\n\n" + extras if extras else "")

        lines: list[str] = []
        page = 0
        tables_by_page: dict[int, list] = {}
        for table in document.tables:
            for page_number in table.page_numbers:
                tables_by_page.setdefault(page_number, []).append(table)
        rendered_tables: set[str] = set()
        for block in document.blocks:
            if block.block_type != BlockType.TABLE and not block.text.strip():
                continue
            if block.page_number != page:
                page = block.page_number
                lines.append(f"\n<!-- page: {page} -->\n")

            if block.block_type == BlockType.HEADING:
                lines.append(f"## {block.text}\n")
            elif block.block_type == BlockType.LIST:
                lines.append(f"- {block.text}\n")
            elif block.block_type == BlockType.TABLE:
                for table in tables_by_page.get(block.page_number, []):
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
                lines.append(f"\n![{title}]({table.fallback_image_path})\n")
            rendered_tables.add(table.id)

        extra = self._build_figure_extras(document)
        if extra:
            lines.append(extra)

        return "\n".join(lines)

    def _table_block_markdown(self, text: str) -> str:
        # Marker table blocks can arrive as plain cell text; keep them visually separate even when
        # the full row/column geometry is unavailable.
        return f"\n<div class=\"table-block\">\n\n{text}\n\n</div>\n"

    def _escape_table_cell(self, text: str) -> str:
        return str(text).replace("\n", "<br>").replace("|", "\\|").strip()

    def _table_html(self, table) -> str:
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

    def _build_figure_extras(self, document: DocumentModel) -> str:
        lines: list[str] = []
        if document.figures:
            lines.append("\n## Figures\n")
            for idx, figure in enumerate(document.figures, start=1):
                lines.append(f"\n### Figure {idx}\n")
                if figure.image_path:
                    lines.append(f"![Figure {idx}]({figure.image_path})\n")
                else:
                    lines.append("_Figure preserved as placeholder._\n")
        return "\n".join(lines).strip()
