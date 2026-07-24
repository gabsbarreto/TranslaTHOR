from __future__ import annotations

import html
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import fitz

from app.models.schema import Block, BlockType, BoundingBox, DocumentModel, TableModel
from app.services.pdf_coordinates import bbox_area, bbox_intersection_area, convert_bbox_to_pdf

logger = logging.getLogger(__name__)


@dataclass
class MarkerTableRepairSummary:
    suspicious_count: int = 0
    repaired_count: int = 0
    failed_count: int = 0
    repairs: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "suspicious_count": self.suspicious_count,
            "repaired_count": self.repaired_count,
            "failed_count": self.failed_count,
            "repairs": self.repairs,
        }


@dataclass(frozen=True)
class _TableShape:
    rows: list[list[str]]
    row_count: int
    column_count: int
    cell_count: int
    nonempty_cell_count: int
    nonempty_row_count: int
    total_characters: int
    largest_cell_characters: int
    largest_cell_share: float


@dataclass(frozen=True)
class _Candidate:
    rows: list[list[str]]
    bbox: tuple[float, float, float, float]
    strategy: str
    token_recall: float
    token_precision: float
    overlap_ratio: float


class MarkerTableRepairService:
    """Repair demonstrably collapsed Marker tables from the source PDF grid.

    Marker normally supplies the canonical table markup. Some ruled digital tables
    are emitted with almost every value inside one cell followed by empty rows. For
    those cases only, PyMuPDF is used inside the exact Marker bounding box and the
    replacement is accepted only when its text strongly agrees with Marker.
    """

    minimum_token_overlap = 0.88

    def repair(self, pdf_path: Path, document: DocumentModel) -> MarkerTableRepairSummary:
        summary = MarkerTableRepairSummary()
        tables_by_block_id = {
            str(table.debug.get("marker_block_id")): table
            for table in document.tables
            if table.debug.get("marker_block_id")
        }
        suspicious: list[tuple[Block, TableModel, _TableShape]] = []
        for block in document.blocks:
            if block.block_type != BlockType.TABLE or "<table" not in block.text.lower():
                continue
            table = tables_by_block_id.get(block.id)
            if table is None:
                continue
            shape = _table_shape(block.text)
            if _is_structurally_suspicious(shape):
                suspicious.append((block, table, shape))

        summary.suspicious_count = len(suspicious)
        if not suspicious:
            return summary

        try:
            pdf = fitz.open(pdf_path)
        except Exception as exc:
            summary.failed_count = len(suspicious)
            summary.warnings.append(
                f"Could not inspect {len(suspicious)} suspicious Marker table(s) in the source PDF: {exc}"
            )
            return summary

        with pdf:
            for block, table, marker_shape in suspicious:
                repair = self._repair_block(pdf, block, table, marker_shape)
                if repair is None:
                    summary.failed_count += 1
                    summary.warnings.append(
                        f"Table {block.id} on page {block.page_number} has unreliable Marker structure; "
                        "no matching PDF table grid was found, so the source markup was retained."
                    )
                    continue
                summary.repaired_count += 1
                summary.repairs.append(repair)

        return summary

    def _repair_block(
        self,
        pdf: fitz.Document,
        block: Block,
        table: TableModel,
        marker_shape: _TableShape,
    ) -> dict[str, Any] | None:
        if block.page_number < 1 or block.page_number > pdf.page_count:
            return None
        page = pdf[block.page_number - 1]
        conversion = convert_bbox_to_pdf(
            block.bbox,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            metadata=block.metadata,
        )
        if conversion.bbox is None:
            return None

        clip = fitz.Rect(
            conversion.bbox.x0,
            conversion.bbox.y0,
            conversion.bbox.x1,
            conversion.bbox.y1,
        )
        candidate = self._best_candidate(page, clip, marker_shape)
        if candidate is None:
            return None

        repaired_html = _rows_to_html(candidate.rows)
        repaired_shape = _table_shape(repaired_html)
        if repaired_shape.nonempty_row_count <= marker_shape.nonempty_row_count:
            return None

        block.text = repaired_html
        block.metadata["marker_table_repair"] = {
            "engine": "pymupdf_find_tables",
            "strategy": candidate.strategy,
            "reason": "collapsed_marker_cell_structure",
            "marker_row_count": marker_shape.row_count,
            "marker_column_count": marker_shape.column_count,
            "marker_nonempty_cell_count": marker_shape.nonempty_cell_count,
            "marker_largest_cell_share": round(marker_shape.largest_cell_share, 4),
            "repaired_row_count": repaired_shape.row_count,
            "repaired_column_count": repaired_shape.column_count,
            "repaired_nonempty_cell_count": repaired_shape.nonempty_cell_count,
            "token_recall": round(candidate.token_recall, 4),
            "token_precision": round(candidate.token_precision, 4),
            "pdf_table_bbox": [round(value, 4) for value in candidate.bbox],
            "coordinate_conversion": conversion.metadata,
        }
        table.headers = list(candidate.rows[0])
        table.rows = [list(row) for row in candidate.rows[1:]]
        table.cells = []
        table.parse_mode = "pymupdf_repaired_marker"
        table.debug.update(
            {
                "render_from_block_text": True,
                "table_repair_engine": "pymupdf_find_tables",
                "table_repair_strategy": candidate.strategy,
                "pdf_table_bbox": [round(value, 4) for value in candidate.bbox],
                "token_recall": round(candidate.token_recall, 4),
                "token_precision": round(candidate.token_precision, 4),
            }
        )
        logger.info(
            "Repaired collapsed Marker table %s on page %s: %sx%s -> %sx%s",
            block.id,
            block.page_number,
            marker_shape.row_count,
            marker_shape.column_count,
            repaired_shape.row_count,
            repaired_shape.column_count,
        )
        return {
            "block_id": block.id,
            "page_number": block.page_number,
            "marker_rows": marker_shape.row_count,
            "marker_columns": marker_shape.column_count,
            "repaired_rows": repaired_shape.row_count,
            "repaired_columns": repaired_shape.column_count,
            "token_recall": round(candidate.token_recall, 4),
            "token_precision": round(candidate.token_precision, 4),
        }

    def _best_candidate(
        self,
        page: fitz.Page,
        clip: fitz.Rect,
        marker_shape: _TableShape,
    ) -> _Candidate | None:
        candidates: list[_Candidate] = []
        marker_tokens = _table_tokens(marker_shape.rows)
        seen: set[tuple[float, float, float, float]] = set()
        for strategy in ("lines_strict", "lines"):
            try:
                finder = page.find_tables(clip=clip, strategy=strategy)
            except Exception as exc:
                logger.debug("PDF table detection with %s failed: %s", strategy, exc)
                continue
            for detected in finder.tables:
                rows = _normalize_detected_rows(detected.extract())
                if len(rows) < 2:
                    continue
                column_count = len(rows[0])
                if column_count < 2 or any(len(row) != column_count for row in rows):
                    continue
                candidate_tokens = _table_tokens(rows)
                token_recall, token_precision = _token_overlap(marker_tokens, candidate_tokens)
                if (
                    token_recall < self.minimum_token_overlap
                    or token_precision < self.minimum_token_overlap
                ):
                    continue
                bbox = tuple(float(value) for value in detected.bbox)
                rounded_bbox = tuple(round(value, 2) for value in bbox)
                if rounded_bbox in seen:
                    continue
                seen.add(rounded_bbox)
                candidate_bbox = _bbox_from_tuple(bbox)
                overlap = bbox_intersection_area(
                    candidate_bbox,
                    _bbox_from_tuple(tuple(float(value) for value in clip)),
                ) / max(1.0, bbox_area(candidate_bbox))
                candidates.append(
                    _Candidate(
                        rows=rows,
                        bbox=bbox,
                        strategy=strategy,
                        token_recall=token_recall,
                        token_precision=token_precision,
                        overlap_ratio=overlap,
                    )
                )

        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                min(item.token_recall, item.token_precision),
                len(item.rows),
                item.overlap_ratio,
            ),
        )


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        normalized = tag.lower()
        if normalized == "tr":
            self._row = []
        elif normalized in {"td", "th"}:
            self._cell = []
        elif normalized == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"td", "th"} and self._cell is not None:
            if self._row is not None:
                self._row.append(_normalize_cell_text("".join(self._cell)))
            self._cell = None
        elif normalized == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _table_shape(table_html: str) -> _TableShape:
    parser = _TableHTMLParser()
    try:
        parser.feed(table_html)
        parser.close()
    except Exception:
        parser.rows = []
    rows = parser.rows
    cells = [cell for row in rows for cell in row]
    lengths = [len(re.sub(r"\s+", "", cell)) for cell in cells]
    total = sum(lengths)
    largest = max(lengths, default=0)
    return _TableShape(
        rows=rows,
        row_count=len(rows),
        column_count=max((len(row) for row in rows), default=0),
        cell_count=len(cells),
        nonempty_cell_count=sum(bool(cell.strip()) for cell in cells),
        nonempty_row_count=sum(any(cell.strip() for cell in row) for row in rows),
        total_characters=total,
        largest_cell_characters=largest,
        largest_cell_share=largest / max(1, total),
    )


def _is_structurally_suspicious(shape: _TableShape) -> bool:
    if shape.row_count < 3 or shape.column_count < 2 or shape.cell_count < 6:
        return False
    empty_cell_ratio = 1.0 - (shape.nonempty_cell_count / shape.cell_count)
    mostly_empty_rows = shape.nonempty_row_count <= max(2, int(shape.row_count * 0.25))
    dominant_cell = shape.largest_cell_share >= 0.72
    return dominant_cell and (empty_cell_ratio >= 0.55 or mostly_empty_rows)


def _normalize_detected_rows(raw_rows: list[list[str | None]]) -> list[list[str]]:
    rows = [
        [_normalize_cell_text("" if cell is None else str(cell)) for cell in row]
        for row in raw_rows
    ]
    while rows and not any(cell for cell in rows[0]):
        rows.pop(0)
    while rows and not any(cell for cell in rows[-1]):
        rows.pop()
    return rows


def _normalize_cell_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _rows_to_html(rows: list[list[str]]) -> str:
    rendered = ["<table><thead>", _row_to_html(rows[0], "th"), "</thead><tbody>"]
    rendered.extend(_row_to_html(row, "td") for row in rows[1:])
    rendered.append("</tbody></table>")
    return "".join(rendered)


def _row_to_html(row: list[str], tag: str) -> str:
    cells = "".join(
        f"<{tag}>{html.escape(cell).replace(chr(10), '<br>')}</{tag}>" for cell in row
    )
    return f"<tr>{cells}</tr>"


def _table_tokens(rows: list[list[str]]) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for row in rows:
        for cell in row:
            normalized = unicodedata.normalize("NFKD", cell.casefold())
            normalized = "".join(character for character in normalized if not unicodedata.combining(character))
            tokens.update(re.findall(r"[a-z0-9]+", normalized))
    return tokens


def _token_overlap(source: Counter[str], candidate: Counter[str]) -> tuple[float, float]:
    matched = sum((source & candidate).values())
    return matched / max(1, sum(source.values())), matched / max(1, sum(candidate.values()))


def _bbox_from_tuple(values: tuple[float, float, float, float]) -> BoundingBox:
    return BoundingBox(x0=values[0], y0=values[1], x1=values[2], y1=values[3])
