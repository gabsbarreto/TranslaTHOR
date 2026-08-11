from __future__ import annotations

import html
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]

from app.models.schema import Block, BlockType, BoundingBox, DocumentModel, TableModel
from app.services.pdf_coordinates import bbox_area, bbox_intersection_area, convert_bbox_to_pdf
from app.services.table_markup import ParsedTableCell, parse_table_rows, rows_have_consistent_shape

logger = logging.getLogger(__name__)


@dataclass
class MarkerTableRepairSummary:
    suspicious_count: int = 0
    repaired_count: int = 0
    failed_count: int = 0
    repairs: list[dict[str, Any]] = field(default_factory=list)
    source_validated_count: int = 0
    source_incomplete_count: int = 0
    source_unavailable_count: int = 0
    incomplete_block_ids: list[str] = field(default_factory=list)
    validations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "suspicious_count": self.suspicious_count,
            "repaired_count": self.repaired_count,
            "failed_count": self.failed_count,
            "repairs": self.repairs,
            "source_validated_count": self.source_validated_count,
            "source_incomplete_count": self.source_incomplete_count,
            "source_unavailable_count": self.source_unavailable_count,
            "incomplete_block_ids": self.incomplete_block_ids,
            "validations": self.validations,
        }


@dataclass
class MarkerTableOCRMergeSummary:
    attempted_count: int = 0
    merged_count: int = 0
    failed_count: int = 0
    merges: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "attempted_count": self.attempted_count,
            "merged_count": self.merged_count,
            "failed_count": self.failed_count,
            "merges": self.merges,
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
    minimum_source_numeric_tokens = 3

    def repair(self, pdf_path: Path, document: DocumentModel) -> MarkerTableRepairSummary:
        summary = MarkerTableRepairSummary()
        tables_by_block_id = _tables_by_block_id(document)
        table_entries: list[tuple[Block, TableModel]] = []
        suspicious: list[tuple[Block, TableModel, _TableShape]] = []
        for block in document.blocks:
            if block.block_type != BlockType.TABLE or "<table" not in block.text.lower():
                continue
            table = tables_by_block_id.get(block.id)
            if table is None:
                continue
            table_entries.append((block, table))
            shape = _table_shape(block.text)
            if _is_structurally_suspicious(shape):
                suspicious.append((block, table, shape))

        summary.suspicious_count = len(suspicious)
        if not table_entries:
            return summary

        try:
            pdf = fitz.open(pdf_path)
        except Exception as exc:
            summary.failed_count = len(suspicious)
            summary.source_unavailable_count = len(table_entries)
            if suspicious:
                summary.warnings.append(
                    f"Could not inspect {len(suspicious)} suspicious Marker table(s) in the source PDF: {exc}"
                )
            return summary

        structurally_failed: set[str] = set()
        with pdf:
            for block, table, marker_shape in suspicious:
                repair = self._repair_block(pdf, block, table, marker_shape)
                if repair is None:
                    summary.failed_count += 1
                    structurally_failed.add(block.id)
                    _mark_table_incomplete(block, "collapsed_marker_cell_structure")
                    summary.warnings.append(
                        f"Table {block.id} on page {block.page_number} has unreliable Marker structure; "
                        "no matching PDF table grid was found, so the source markup was retained."
                    )
                    continue
                summary.repaired_count += 1
                summary.repairs.append(repair)

            for block, table in table_entries:
                validation = self._source_number_validation(pdf, block)
                block.metadata["marker_table_source_validation"] = validation
                table.debug["source_number_validation"] = validation
                summary.validations.append(
                    {
                        "block_id": block.id,
                        "page_number": block.page_number,
                        **validation,
                    }
                )
                status = validation["status"]
                if status == "unavailable":
                    summary.source_unavailable_count += 1
                    continue
                if status == "not_applicable":
                    continue
                summary.source_validated_count += 1
                if status == "incomplete":
                    summary.source_incomplete_count += 1
                    _mark_table_incomplete(block, "source_numeric_values_missing")
                    missing = validation.get("missing_numeric_tokens") or []
                    summary.warnings.append(
                        f"Table {block.id} on page {block.page_number} is missing "
                        f"{len(missing)} numeric source value(s) in Marker output."
                    )
                elif block.id not in structurally_failed:
                    block.metadata.pop("marker_table_incomplete", None)
                    block.metadata.pop("marker_table_incomplete_reasons", None)

        summary.incomplete_block_ids = sorted(
            block.id
            for block, _table in table_entries
            if block.metadata.get("marker_table_incomplete")
        )

        return summary

    def merge_incomplete_from_ocr_retry(
        self,
        primary: DocumentModel,
        retry: DocumentModel,
        block_ids: list[str],
    ) -> MarkerTableOCRMergeSummary:
        """Fill proven omissions from a forced-OCR retry without replacing good rows.

        Marker balanced mode generally preserves row association better than a
        forced-OCR pass, while forced OCR can recover values absent from a PDF text
        layer. Only empty primary cells are filled, and only with numeric tokens that
        the source-PDF validation proved are missing. This prevents a retry value from
        being moved to a different row merely because OCR shifted its vertical box.
        """

        summary = MarkerTableOCRMergeSummary(attempted_count=len(block_ids))
        primary_blocks = {block.id: block for block in primary.blocks}
        primary_tables = _tables_by_block_id(primary)
        retry_tables = _tables_by_block_id(retry)
        retry_table_blocks = [
            block for block in retry.blocks if block.block_type == BlockType.TABLE
        ]

        for block_id in block_ids:
            block = primary_blocks.get(block_id)
            table = primary_tables.get(block_id)
            if block is None or table is None:
                summary.failed_count += 1
                continue
            retry_block = _matching_retry_table(block, retry_table_blocks)
            if retry_block is None:
                summary.failed_count += 1
                summary.warnings.append(
                    f"Forced-OCR retry did not return a matching table for {block.id}."
                )
                continue

            primary_rows = parse_table_rows(block.text)
            retry_rows = parse_table_rows(retry_block.text)
            validation = block.metadata.get("marker_table_source_validation")
            source_numbers = (
                validation.get("source_numeric_tokens", []) if isinstance(validation, dict) else []
            )
            deficits = Counter(str(value) for value in source_numbers) - _numeric_table_tokens(
                primary_rows
            )
            merged_rows, filled_cells, normalized_labels = _merge_retry_rows(
                primary_rows,
                retry_rows,
                deficits,
            )
            if not filled_cells:
                summary.failed_count += 1
                summary.warnings.append(
                    f"Forced-OCR retry for table {block.id} could not be mapped to empty primary cells safely."
                )
                continue

            block.text = _parsed_rows_to_html(merged_rows)
            block.metadata["marker_table_ocr_retry"] = {
                "retry_block_id": retry_block.id,
                "filled_cells": filled_cells,
                "normalized_labels": normalized_labels,
                "policy": "fill_source_verified_numeric_omissions_only",
            }
            rendered_rows = [[cell.text for cell in row] for row in merged_rows]
            had_headers = bool(table.headers)
            table.headers = list(rendered_rows[0]) if had_headers and rendered_rows else []
            table.rows = [list(row) for row in rendered_rows[1:]] if had_headers else rendered_rows
            table.header_cells = []
            table.cells = []
            table.parse_mode = "marker_balanced_ocr_retry_merged"
            table.debug.update(
                {
                    "render_from_block_text": True,
                    "marker_ocr_retry_block_id": retry_block.id,
                    "marker_ocr_retry_filled_cells": filled_cells,
                }
            )
            retry_table = retry_tables.get(retry_block.id)
            if retry_table is not None:
                table.debug["marker_ocr_retry_parse_mode"] = retry_table.parse_mode
            summary.merged_count += 1
            summary.merges.append(
                {
                    "block_id": block.id,
                    "page_number": block.page_number,
                    "retry_block_id": retry_block.id,
                    "filled_cells": filled_cells,
                    "normalized_labels": normalized_labels,
                }
            )

        return summary

    def _source_number_validation(
        self,
        pdf: fitz.Document,
        block: Block,
    ) -> dict[str, Any]:
        if block.page_number < 1 or block.page_number > pdf.page_count:
            return {"status": "unavailable", "reason": "page_out_of_range"}
        page = pdf[block.page_number - 1]
        conversion = convert_bbox_to_pdf(
            block.bbox,
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
            metadata=block.metadata,
        )
        if conversion.bbox is None:
            return {"status": "unavailable", "reason": "table_bbox_unavailable"}
        clip = fitz.Rect(
            conversion.bbox.x0,
            conversion.bbox.y0,
            conversion.bbox.x1,
            conversion.bbox.y1,
        )
        try:
            source_text = page.get_text("text", clip=clip)
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason": "source_text_extraction_failed",
                "error": str(exc),
            }
        source_tokens = _numeric_text_tokens(source_text)
        extracted_tokens = _numeric_table_tokens(parse_table_rows(block.text))
        source_total = sum(source_tokens.values())
        if source_total < self.minimum_source_numeric_tokens:
            return {
                "status": "not_applicable",
                "reason": "too_few_source_numeric_tokens",
                "source_numeric_count": source_total,
                "extracted_numeric_count": sum(extracted_tokens.values()),
                "coordinate_conversion": conversion.metadata,
            }
        matched = sum((source_tokens & extracted_tokens).values())
        missing = source_tokens - extracted_tokens
        missing_tokens = sorted(missing.elements(), key=_numeric_sort_key)
        return {
            "status": "incomplete" if missing_tokens else "complete",
            "source_numeric_count": source_total,
            "extracted_numeric_count": sum(extracted_tokens.values()),
            "matched_numeric_count": matched,
            "numeric_recall": round(matched / source_total, 4),
            "source_numeric_tokens": sorted(source_tokens.elements(), key=_numeric_sort_key),
            "missing_numeric_tokens": missing_tokens,
            "coordinate_conversion": conversion.metadata,
        }

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
                raw_bbox = detected.bbox
                if len(raw_bbox) != 4:
                    continue
                bbox = (
                    float(raw_bbox[0]),
                    float(raw_bbox[1]),
                    float(raw_bbox[2]),
                    float(raw_bbox[3]),
                )
                rounded_bbox = (
                    round(bbox[0], 2),
                    round(bbox[1], 2),
                    round(bbox[2], 2),
                    round(bbox[3], 2),
                )
                if rounded_bbox in seen:
                    continue
                seen.add(rounded_bbox)
                candidate_bbox = _bbox_from_tuple(bbox)
                overlap = bbox_intersection_area(
                    candidate_bbox,
                    _bbox_from_tuple(
                        (float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1))
                    ),
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
    cells = "".join(f"<{tag}>{html.escape(cell).replace(chr(10), '<br>')}</{tag}>" for cell in row)
    return f"<tr>{cells}</tr>"


def _table_tokens(rows: list[list[str]]) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for row in rows:
        for cell in row:
            normalized = unicodedata.normalize("NFKD", cell.casefold())
            normalized = "".join(
                character for character in normalized if not unicodedata.combining(character)
            )
            tokens.update(re.findall(r"[a-z0-9]+", normalized))
    return tokens


def _token_overlap(source: Counter[str], candidate: Counter[str]) -> tuple[float, float]:
    matched = sum((source & candidate).values())
    return matched / max(1, sum(source.values())), matched / max(1, sum(candidate.values()))


def _tables_by_block_id(document: DocumentModel) -> dict[str, TableModel]:
    tables: dict[str, TableModel] = {}
    for table in document.tables:
        block_id = str(
            table.debug.get("source_block_id") or table.debug.get("marker_block_id") or ""
        )
        if block_id:
            tables[block_id] = table
    return tables


def _mark_table_incomplete(block: Block, reason: str) -> None:
    block.metadata["marker_table_incomplete"] = True
    raw_reasons = block.metadata.get("marker_table_incomplete_reasons")
    reasons = [str(value) for value in raw_reasons] if isinstance(raw_reasons, list) else []
    if reason not in reasons:
        reasons.append(reason)
    block.metadata["marker_table_incomplete_reasons"] = reasons


def _numeric_text_tokens(text: str) -> Counter[str]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    tokens: Counter[str] = Counter()
    for match in re.finditer(r"(?<![\w])\d+(?:[.,]\d+)*(?![\w])", normalized):
        value = match.group(0).replace(",", ".")
        parts = value.split(".")
        if len(parts) > 2:
            # Multiple separators are normally thousands grouping. Decimal values
            # in scientific tables have only one separator.
            value = "".join(parts)
        value = value.lstrip("0") or "0"
        if value.startswith("."):
            value = f"0{value}"
        tokens[value] += 1
    return tokens


def _numeric_table_tokens(rows: list[list[ParsedTableCell]]) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for row in rows:
        for cell in row:
            tokens.update(_numeric_text_tokens(cell.text))
    return tokens


def _numeric_sort_key(value: str) -> tuple[float, str]:
    try:
        return float(value), value
    except ValueError:
        return float("inf"), value


def _matching_retry_table(primary: Block, retry_blocks: list[Block]) -> Block | None:
    exact = next(
        (
            block
            for block in retry_blocks
            if block.id == primary.id and block.page_number == primary.page_number
        ),
        None,
    )
    if exact is not None:
        return exact
    candidates = [block for block in retry_blocks if block.page_number == primary.page_number]
    if primary.bbox is None:
        return candidates[0] if len(candidates) == 1 else None
    scored: list[tuple[float, Block]] = []
    for candidate in candidates:
        if candidate.bbox is None:
            continue
        overlap = bbox_intersection_area(primary.bbox, candidate.bbox)
        overlap_ratio = overlap / max(1.0, min(bbox_area(primary.bbox), bbox_area(candidate.bbox)))
        if overlap_ratio >= 0.5:
            scored.append((overlap_ratio, candidate))
    return max(scored, key=lambda item: item[0])[1] if scored else None


def _merge_retry_rows(
    primary_rows: list[list[ParsedTableCell]],
    retry_rows: list[list[ParsedTableCell]],
    deficits: Counter[str],
) -> tuple[list[list[ParsedTableCell]], int, int]:
    if (
        not rows_have_consistent_shape(primary_rows)
        or not rows_have_consistent_shape(retry_rows)
        or len(primary_rows[0]) != len(retry_rows[0])
    ):
        return primary_rows, 0, 0
    alignments = _align_retry_rows(primary_rows, retry_rows)
    if not alignments:
        return primary_rows, 0, 0

    merged = [list(row) for row in primary_rows]
    remaining = Counter(deficits)
    filled_cells = 0
    normalized_labels = 0
    for primary_index, retry_index in alignments:
        primary_row = merged[primary_index]
        retry_row = retry_rows[retry_index]
        primary_anchor = primary_row[0]
        retry_anchor = retry_row[0]
        if (
            primary_anchor.text.strip()
            and retry_anchor.text.strip()
            and primary_anchor.text != retry_anchor.text
            and _row_anchor(primary_row) == _row_anchor(retry_row)
        ):
            primary_row[0] = replace(primary_anchor, text=retry_anchor.text)
            normalized_labels += 1

        for column_index in range(1, len(primary_row)):
            primary_cell = primary_row[column_index]
            retry_cell = retry_row[column_index]
            if primary_cell.text.strip() or not retry_cell.text.strip():
                continue
            candidate_tokens = _numeric_text_tokens(retry_cell.text)
            if not candidate_tokens:
                continue
            if any(count > remaining[token] for token, count in candidate_tokens.items()):
                continue
            primary_row[column_index] = replace(primary_cell, text=retry_cell.text)
            remaining.subtract(candidate_tokens)
            remaining += Counter()
            filled_cells += 1
    return merged, filled_cells, normalized_labels


def _align_retry_rows(
    primary_rows: list[list[ParsedTableCell]],
    retry_rows: list[list[ParsedTableCell]],
) -> list[tuple[int, int]]:
    primary_anchors = [_row_anchor(row) for row in primary_rows]
    retry_anchors = [_row_anchor(row) for row in retry_rows]
    if len(primary_rows) == len(retry_rows) and all(
        _anchor_similarity(primary, retry) >= 0.72
        for primary, retry in zip(primary_anchors, retry_anchors)
    ):
        return list(enumerate(range(len(retry_rows))))

    aligned: list[tuple[int, int]] = []
    retry_cursor = 0
    for primary_index, primary_anchor in enumerate(primary_anchors):
        candidates = [
            (retry_index, _anchor_similarity(primary_anchor, retry_anchors[retry_index]))
            for retry_index in range(retry_cursor, len(retry_rows))
        ]
        if not candidates:
            break
        retry_index, score = max(candidates, key=lambda item: (item[1], -item[0]))
        if score < 0.82:
            continue
        aligned.append((primary_index, retry_index))
        retry_cursor = retry_index + 1
    return aligned


def _row_anchor(row: list[ParsedTableCell]) -> str:
    text = next((cell.text for cell in row if cell.text.strip()), "")
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return "".join(character for character in normalized if character.isalnum())


def _anchor_similarity(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0
    return SequenceMatcher(None, first, second, autojunk=False).ratio()


def _parsed_rows_to_html(rows: list[list[ParsedTableCell]]) -> str:
    rendered = ["<table><tbody>"]
    for row in rows:
        rendered.append("<tr>")
        for cell in row:
            attributes = ""
            if cell.rowspan > 1:
                attributes += f' rowspan="{cell.rowspan}"'
            if cell.colspan > 1:
                attributes += f' colspan="{cell.colspan}"'
            text = html.escape(cell.text).replace("\n", "<br>")
            rendered.append(f"<{cell.tag}{attributes}>{text}</{cell.tag}>")
        rendered.append("</tr>")
    rendered.append("</tbody></table>")
    return "".join(rendered)


def _bbox_from_tuple(values: tuple[float, float, float, float]) -> BoundingBox:
    return BoundingBox(x0=values[0], y0=values[1], x1=values[2], y1=values[3])
