from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass(frozen=True)
class ParsedTableCell:
    tag: str
    text: str


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[ParsedTableCell]] = []
        self._row: list[ParsedTableCell] | None = None
        self._cell_tag: str | None = None
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        normalized = tag.lower()
        if normalized == "tr":
            self._row = []
        elif normalized in {"td", "th"} and self._row is not None:
            self._cell_tag = normalized
            self._cell_parts = []
        elif normalized == "br" and self._cell_tag is not None:
            self._cell_parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if self._cell_tag is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"td", "th"} and self._row is not None and self._cell_tag is not None:
            text = "\n".join(
                line.strip()
                for line in "".join(self._cell_parts).splitlines()
                if line.strip()
            )
            self._row.append(ParsedTableCell(tag=self._cell_tag, text=text))
            self._cell_tag = None
            self._cell_parts = []
        elif normalized == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def parse_table_rows(
    markup: str,
    *,
    source_hint: str | None = None,
) -> list[list[ParsedTableCell]]:
    """Parse HTML or pipe-Markdown tables, including flattened OCR Markdown.

    OCR logical chunks intentionally collapse whitespace. A table such as
    ``| A | B |\n|---|---|\n| 1 | 2 |`` therefore reaches reconstruction as
    one line. The separator row still gives us a deterministic column count;
    ``source_hint`` can additionally preserve the original multiline shape.
    """

    text = str(markup or "").strip()
    if not text:
        return []
    if re.search(r"<table\b", text, flags=re.IGNORECASE):
        parser = _TableHTMLParser()
        try:
            parser.feed(text)
            parser.close()
        except Exception:
            return []
        return parser.rows
    if "|" not in text:
        return []
    return _parse_markdown_rows(text, source_hint=source_hint)


def rows_have_consistent_shape(rows: list[list[ParsedTableCell]]) -> bool:
    return bool(rows) and bool(rows[0]) and all(len(row) == len(rows[0]) for row in rows)


def _parse_markdown_rows(
    markup: str,
    *,
    source_hint: str | None,
) -> list[list[ParsedTableCell]]:
    multiline = _markdown_lines(markup)
    if len(multiline) >= 2:
        return _cells_from_markdown_rows(multiline)

    hint_lines = _markdown_lines(source_hint or "")
    width = _markdown_width(hint_lines) or _separator_width(markup)
    if width is None:
        return []
    flat_rows = _consume_flat_rows(markup, width)
    if not flat_rows:
        return []
    return _cells_from_markdown_rows(flat_rows)


def _markdown_lines(markup: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in str(markup or "").splitlines():
        line = raw_line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        rows.append([part.strip() for part in _split_pipe_cells(line[1:-1])])
    return rows


def _markdown_width(rows: list[list[str]]) -> int | None:
    widths = {len(row) for row in rows if row}
    return next(iter(widths)) if len(widths) == 1 else None


def _separator_width(markup: str) -> int | None:
    parts = [part.strip() for part in _split_pipe_cells(markup)]
    longest = 0
    current = 0
    for part in parts:
        if _is_separator_cell(part):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest or None


def _consume_flat_rows(markup: str, width: int) -> list[list[str]]:
    if width <= 0:
        return []
    parts = [part.strip() for part in _split_pipe_cells(markup.strip())]
    if parts and not parts[0]:
        parts = parts[1:]
    if parts and not parts[-1]:
        parts = parts[:-1]

    rows: list[list[str]] = []
    cursor = 0
    while cursor + width <= len(parts):
        rows.append(parts[cursor : cursor + width])
        cursor += width
        # Adjacent Markdown rows contribute ``| |`` after whitespace is
        # collapsed, leaving one structural empty token between rows.
        if cursor < len(parts) and not parts[cursor]:
            cursor += 1
    if cursor != len(parts):
        return []
    return rows


def _cells_from_markdown_rows(rows: list[list[str]]) -> list[list[ParsedTableCell]]:
    data_rows = [row for row in rows if not row or not all(_is_separator_cell(cell) for cell in row)]
    if not data_rows or not all(len(row) == len(data_rows[0]) for row in data_rows):
        return []
    parsed: list[list[ParsedTableCell]] = []
    for row_index, row in enumerate(data_rows):
        tag = "th" if row_index == 0 else "td"
        parsed.append([ParsedTableCell(tag=tag, text=cell.strip()) for cell in row])
    return parsed


def _is_separator_cell(text: str) -> bool:
    return bool(re.fullmatch(r":?-{1,}:?", text.strip()))


def _split_pipe_cells(text: str) -> list[str]:
    r"""Split Markdown cells while treating ``\|`` as literal cell content."""

    parts: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\" and index + 1 < len(text) and text[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if character == "|":
            parts.append("".join(current))
            current = []
        else:
            current.append(character)
        index += 1
    parts.append("".join(current))
    return parts
