from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import fitz

from app.models.schema import Block, BlockType, BoundingBox, DocumentModel, SourceType
from app.services.pdf_coordinates import (
    bbox_area,
    bbox_intersection_area,
    convert_bbox_to_pdf,
)

logger = logging.getLogger(__name__)


@dataclass
class _ReplacementRegion:
    page_number: int
    block_ids: list[str]
    block_type: BlockType
    bbox: BoundingBox
    translated_text: str
    source_text: str
    style_hints: dict[str, Any]
    coordinate_metadata: list[dict[str, Any]]


@dataclass
class _ParsedTableCell:
    tag: str
    text: str


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_ParsedTableCell]] = []
        self._row: list[_ParsedTableCell] | None = None
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
            self._row.append(_ParsedTableCell(tag=self._cell_tag, text=text))
            self._cell_tag = None
            self._cell_parts = []
        elif normalized == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


class OriginalLayoutReconstructor:
    """Conservatively replace translated text while retaining the source PDF page art."""

    minimum_scale = 0.6
    locked_block_types = {BlockType.FIGURE, BlockType.EQUATION}
    conservative_skip_types = {BlockType.TABLE}

    def reconstruct(
        self,
        *,
        source_pdf_path: Path,
        output_pdf_path: Path,
        document: DocumentModel,
        report_path: Path,
    ) -> dict[str, Any]:
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = self._initial_report(document)
        page_metadata = {page.page_number: page for page in document.pages}
        blocks_by_page: dict[int, list[Block]] = {}
        for block in document.blocks:
            blocks_by_page.setdefault(block.page_number, []).append(block)
        recovered_translations = self._recover_per_block_translations(document)

        temporary_path = output_pdf_path.with_name(f".{output_pdf_path.stem}.tmp.pdf")
        if temporary_path.exists():
            temporary_path.unlink()

        with fitz.open(source_pdf_path) as pdf:
            report["total_pages"] = pdf.page_count
            if document.metadata.page_count != pdf.page_count:
                self._warning(
                    report,
                    code="page_count_mismatch",
                    reason=(
                        f"Structured JSON reports {document.metadata.page_count} pages but the source PDF "
                        f"contains {pdf.page_count}; only matching pages were considered."
                    ),
                )

            locked_by_page = self._locked_regions(document, pdf, report)
            for page_number in range(1, pdf.page_count + 1):
                page = pdf[page_number - 1]
                page_report: dict[str, Any] = {
                    "page_number": page_number,
                    "status": "unchanged",
                    "regions_replaced": 0,
                    "regions_skipped": 0,
                    "fallback_required": False,
                    "warnings": [],
                }
                report["pages"].append(page_report)
                supported, reason = self._page_is_supported(
                    page,
                    page_metadata.get(page_number),
                )
                if not supported:
                    page_report["status"] = "fallback_original_page"
                    page_report["warnings"].append(reason)
                    report["pages_using_fallback_behavior"] += 1
                    self._warning(
                        report,
                        page_number=page_number,
                        code="page_not_safely_replaceable",
                        reason=reason,
                    )
                    continue

                replacements = self._replacement_regions(
                    page=page,
                    page_number=page_number,
                    blocks=blocks_by_page.get(page_number, []),
                    all_blocks=document.blocks,
                    locked_regions=locked_by_page.get(page_number, []),
                    recovered_translations=recovered_translations,
                    report=report,
                    page_report=page_report,
                )
                approved: list[tuple[_ReplacementRegion, str, str, float]] = []
                for region in replacements:
                    html_text, css = self._region_html_and_css(region, page)
                    spare_height, scale = self._preflight(
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                        region=region,
                        html_text=html_text,
                        css=css,
                    )
                    if spare_height < 0 or scale < self.minimum_scale:
                        report["text_boxes_did_not_fit"] += 1
                        page_report["regions_skipped"] += 1
                        page_report["fallback_required"] = True
                        self._skipped_region(
                            report,
                            region,
                            reason="translated_text_did_not_fit_minimum_scale",
                            scale=scale,
                        )
                        self._warning(
                            report,
                            page_number=page_number,
                            code="text_box_overflow",
                            reason=(
                                f"Translated text for {', '.join(region.block_ids)} requires scale "
                                f"{scale:.3f}, below the minimum {self.minimum_scale:.3f}; source text was retained."
                            ),
                        )
                        continue
                    approved.append((region, html_text, css, scale))

                if not approved:
                    if page_report["fallback_required"]:
                        page_report["status"] = "fallback_original_page"
                        report["pages_using_fallback_behavior"] += 1
                    else:
                        page_report["status"] = "success_no_replacement_needed"
                        report["pages_successfully_reconstructed"] += 1
                    continue

                original_links = page.get_links()
                for region, _html_text, _css, _scale in approved:
                    page.add_redact_annot(
                        self._fitz_rect(region.bbox),
                        fill=None,
                        cross_out=False,
                    )
                page.apply_redactions(images=0, graphics=0, text=0)

                page_failed = False
                for region, html_text, css, _preflight_scale in approved:
                    spare_height, scale = page.insert_htmlbox(
                        self._fitz_rect(region.bbox),
                        html_text,
                        css=css,
                        scale_low=self.minimum_scale,
                        overlay=True,
                    )
                    entry = {
                        "page_number": page_number,
                        "block_ids": region.block_ids,
                        "block_type": region.block_type.value,
                        "bbox": region.bbox.model_dump(),
                        "scale": round(float(scale), 6),
                        "spare_height": round(float(spare_height), 6),
                    }
                    report["scaling_applied"].append(entry)
                    if spare_height < 0 or scale < self.minimum_scale:
                        page_failed = True
                        report["text_boxes_did_not_fit"] += 1
                        page_report["regions_skipped"] += 1
                        page_report["fallback_required"] = True
                        self._skipped_region(
                            report,
                            region,
                            reason="unexpected_post_redaction_overflow",
                            scale=scale,
                        )
                        self._warning(
                            report,
                            page_number=page_number,
                            code="unexpected_post_redaction_overflow",
                            reason=(
                                f"Text insertion unexpectedly failed for {', '.join(region.block_ids)} "
                                "after a successful preflight."
                            ),
                        )
                        continue
                    report["regions_replaced"] += 1
                    page_report["regions_replaced"] += 1
                    report["regions"].append({**entry, "status": "replaced"})

                self._restore_missing_links(page, original_links, report, page_number)
                if page_failed or page_report["fallback_required"]:
                    page_report["status"] = "partial"
                    report["pages_using_fallback_behavior"] += 1
                else:
                    page_report["status"] = "success"
                    report["pages_successfully_reconstructed"] += 1

            pdf.save(temporary_path, garbage=3, deflate=True)

        temporary_path.replace(output_pdf_path)
        report["status"] = (
            "partial" if report["pages_using_fallback_behavior"] else "complete"
        )
        report["output_pdf"] = str(output_pdf_path.resolve())
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _initial_report(self, document: DocumentModel) -> dict[str, Any]:
        valid_figures = [figure for figure in document.figures if figure.bbox is not None]
        raster_fallbacks = [
            {
                "figure_id": figure.id,
                "page_number": figure.page_number,
                "reason": figure.extraction_metadata.get(
                    "raster_fallback_reason",
                    "vector_asset_unavailable",
                ),
            }
            for figure in valid_figures
            if figure.image_path and not figure.vector_path
        ]
        low_confidence = []
        for figure in valid_figures:
            caption_details = figure.extraction_metadata.get("caption_association") or {}
            caption_confidence = caption_details.get("confidence")
            detection_low = (
                figure.detection_confidence is not None
                and figure.detection_confidence < 0.6
            )
            caption_low = (
                figure.caption_block_id
                and caption_confidence is not None
                and float(caption_confidence) < 0.55
            )
            if detection_low or caption_low:
                low_confidence.append(
                    {
                        "figure_id": figure.id,
                        "page_number": figure.page_number,
                        "detection_confidence": figure.detection_confidence,
                        "caption_block_id": figure.caption_block_id,
                        "caption_confidence": caption_confidence,
                    }
                )
        return {
            "mode": "original_layout",
            "status": "pending",
            "total_pages": 0,
            "pages_successfully_reconstructed": 0,
            "pages_using_fallback_behavior": 0,
            "figures_preserved": len(valid_figures),
            "regions_replaced": 0,
            "regions_skipped": 0,
            "regions_missing_or_invalid_bboxes": 0,
            "text_boxes_did_not_fit": 0,
            "scaling_applied": [],
            "raster_figure_fallbacks": raster_fallbacks,
            "low_confidence_figure_or_caption_associations": low_confidence,
            "warnings": [],
            "pages": [],
            "regions": [],
            "safe_fallback": "readable_pdf",
            "minimum_text_scale": self.minimum_scale,
            "internal_figure_text_policy": "preserve_source_language",
        }

    def _page_is_supported(self, page: fitz.Page, metadata) -> tuple[bool, str]:
        if page.rotation:
            return False, "Rotated pages are retained unchanged in this first original-layout implementation."
        if metadata is None:
            return False, "Structured page metadata is missing; the source page was retained unchanged."
        if metadata.extraction_mode == SourceType.OCR:
            return (
                False,
                "The page was extracted through OCR, so visible source text cannot be safely removed without inpainting.",
            )
        if not metadata.has_embedded_text or metadata.embedded_text_quality < 0.35:
            return (
                False,
                "The page is scanned, image-only, hidden-OCR, or has unreliable embedded text; it was retained unchanged.",
            )
        if len(page.get_text("text").strip()) < 5:
            return False, "The page has no reliable removable PDF text and was retained unchanged."
        return True, ""

    def _locked_regions(
        self,
        document: DocumentModel,
        pdf: fitz.Document,
        report: dict[str, Any],
    ) -> dict[int, list[BoundingBox]]:
        locked: dict[int, list[BoundingBox]] = {}
        represented_figure_blocks: set[str] = set()
        for figure in document.figures:
            if figure.bbox is None or not (1 <= figure.page_number <= pdf.page_count):
                continue
            represented_figure_blocks.update(figure.source_block_ids)
            page = pdf[figure.page_number - 1]
            conversion = convert_bbox_to_pdf(
                figure.bbox,
                page_width=page.rect.width,
                page_height=page.rect.height,
                metadata={"source_page_width": page.rect.width, "source_page_height": page.rect.height},
            )
            if conversion.bbox is not None:
                locked.setdefault(figure.page_number, []).append(conversion.bbox)
        for block in document.blocks:
            if (
                block.block_type not in self.locked_block_types
                or not (1 <= block.page_number <= pdf.page_count)
            ):
                continue
            if (
                block.block_type == BlockType.FIGURE
                and block.id in represented_figure_blocks
            ):
                # Canonical figure assets have already deduplicated nested Marker
                # FigureGroup/Figure boxes and excluded any associated caption.
                continue
            page = pdf[block.page_number - 1]
            conversion = convert_bbox_to_pdf(
                block.bbox,
                page_width=page.rect.width,
                page_height=page.rect.height,
                metadata=block.metadata,
            )
            if conversion.bbox is not None:
                locked.setdefault(block.page_number, []).append(conversion.bbox)
            else:
                report["regions_missing_or_invalid_bboxes"] += 1
        return locked

    def _replacement_regions(
        self,
        *,
        page: fitz.Page,
        page_number: int,
        blocks: list[Block],
        all_blocks: list[Block],
        locked_regions: list[BoundingBox],
        recovered_translations: dict[str, str],
        report: dict[str, Any],
        page_report: dict[str, Any],
    ) -> list[_ReplacementRegion]:
        block_by_id = {block.id: block for block in all_blocks}
        consumed: set[str] = set()
        replacements: list[_ReplacementRegion] = []
        for block in blocks:
            if block.id in consumed:
                continue
            recovered_text = recovered_translations.get(block.id)
            if recovered_text is None and not block.text.strip():
                if block.metadata.get("merged_into_block_id"):
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="merged_translation_could_not_be_mapped_to_source_region",
                    )
                continue
            if block.block_type in self.locked_block_types:
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="locked_visual_region",
                    fallback_required=False,
                )
                continue
            if block.block_type == BlockType.TABLE:
                replacements.extend(
                    self._table_replacement_regions(
                        page=page,
                        block=block,
                        locked_regions=locked_regions,
                        report=report,
                        page_report=page_report,
                    )
                )
                continue
            if block.block_type in self.conservative_skip_types:
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="table_layout_preserved_conservatively",
                )
                continue

            if recovered_text is not None:
                translated_from = [block.id]
                source_blocks = [block]
                translated_text = recovered_text
            else:
                translated_from = block.metadata.get("translated_from_block_ids")
                if not isinstance(translated_from, list) or not translated_from:
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="translation_not_confirmed_for_region",
                    )
                    continue
                source_blocks = [
                    block_by_id[block_id]
                    for block_id in translated_from
                    if block_id in block_by_id
                ]
                if not source_blocks or any(item.page_number != page_number for item in source_blocks):
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="translated_group_spans_pages_or_is_missing",
                    )
                    consumed.update(str(value) for value in translated_from)
                    continue
                translated_text = block.text.strip()

            converted: list[BoundingBox] = []
            conversions: list[dict[str, Any]] = []
            for source_block in source_blocks:
                conversion = convert_bbox_to_pdf(
                    source_block.bbox,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                    metadata=source_block.metadata,
                )
                conversions.append(conversion.metadata)
                if conversion.bbox is not None:
                    converted.append(conversion.bbox)
            consumed.update(item.id for item in source_blocks)
            if len(converted) != len(source_blocks):
                report["regions_missing_or_invalid_bboxes"] += 1
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="missing_or_invalid_source_bbox",
                )
                continue

            bbox = self._union_bbox(converted)
            if any(self._overlaps_locked_region(bbox, locked) for locked in locked_regions):
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="overlaps_figure_graph_or_equation",
                    bbox=bbox,
                )
                continue

            source_text = " ".join(
                str(item.metadata.get("source_text", "")).strip()
                for item in source_blocks
                if str(item.metadata.get("source_text", "")).strip()
            )
            if source_text and self._normalized_text(source_text) == self._normalized_text(translated_text):
                # English or otherwise unchanged text already matches the visual base.
                continue
            replacements.append(
                _ReplacementRegion(
                    page_number=page_number,
                    block_ids=[item.id for item in source_blocks],
                    block_type=block.block_type,
                    bbox=bbox,
                    translated_text=translated_text,
                    source_text=source_text,
                    style_hints=dict(block.style_hints or {}),
                    coordinate_metadata=conversions,
                )
            )
        return replacements

    def _recover_per_block_translations(self, document: DocumentModel) -> dict[str, str]:
        block_by_id = {block.id: block for block in document.blocks}
        recovered: dict[str, str] = {}
        for chunk in document.translation_chunks:
            if len(chunk.block_ids) <= 1 or not chunk.translated_text.strip():
                continue
            source_parts = self._paragraph_parts(chunk.source_text)
            translated_parts = self._paragraph_parts(chunk.translated_text)
            if not (
                len(source_parts) == len(translated_parts) == len(chunk.block_ids)
            ):
                continue
            source_blocks = [block_by_id.get(block_id) for block_id in chunk.block_ids]
            if any(block is None for block in source_blocks):
                continue
            if not all(
                self._normalized_text(
                    str(block.metadata.get("source_text", block.text))
                )
                == self._normalized_text(source_part)
                for block, source_part in zip(source_blocks, source_parts, strict=True)
                if block is not None
            ):
                continue
            recovered.update(
                {
                    block_id: translated_part
                    for block_id, translated_part in zip(
                        chunk.block_ids,
                        translated_parts,
                        strict=True,
                    )
                }
            )
        return recovered

    def _paragraph_parts(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]

    def _table_replacement_regions(
        self,
        *,
        page: fitz.Page,
        block: Block,
        locked_regions: list[BoundingBox],
        report: dict[str, Any],
        page_report: dict[str, Any],
    ) -> list[_ReplacementRegion]:
        translated_from = block.metadata.get("translated_from_block_ids")
        source_markup = str(block.metadata.get("source_text", ""))
        if (
            not isinstance(translated_from, list)
            or block.id not in translated_from
            or not source_markup.strip()
        ):
            self._skip_block(
                report,
                page_report,
                block,
                reason="table_translation_not_confirmed",
            )
            return []

        source_rows = self._parse_table_rows(source_markup)
        translated_rows = self._parse_table_rows(block.text)
        if (
            not source_rows
            or len(source_rows) != len(translated_rows)
            or any(
                len(source_row) != len(translated_row)
                for source_row, translated_row in zip(
                    source_rows,
                    translated_rows,
                    strict=True,
                )
            )
            or self._table_markup_is_suspicious(source_rows)
            or self._table_markup_is_suspicious(translated_rows)
        ):
            self._skip_block(
                report,
                page_report,
                block,
                reason="table_translation_structure_unreliable",
            )
            return []

        conversion = convert_bbox_to_pdf(
            block.bbox,
            page_width=page.rect.width,
            page_height=page.rect.height,
            metadata=block.metadata,
        )
        if conversion.bbox is None:
            report["regions_missing_or_invalid_bboxes"] += 1
            self._skip_block(
                report,
                page_report,
                block,
                reason="missing_or_invalid_source_bbox",
            )
            return []

        grid_rows = self._table_grid_rows(page, conversion.bbox)
        aligned = self._align_table_rows(
            page,
            source_rows,
            translated_rows,
            grid_rows,
        )
        if aligned is None:
            self._skip_block(
                report,
                page_report,
                block,
                reason="table_cell_geometry_unreliable",
                bbox=conversion.bbox,
            )
            return []

        replacements: list[_ReplacementRegion] = []
        for row_index, (source_row, translated_row, rectangles) in enumerate(aligned):
            for column_index, (source_cell, translated_cell, rectangle) in enumerate(
                zip(source_row, translated_row, rectangles, strict=True)
            ):
                source_text = source_cell.text.strip()
                translated_text = translated_cell.text.strip()
                if self._normalized_text(source_text) == self._normalized_text(translated_text):
                    continue
                if source_text and not translated_text:
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="table_translation_contains_empty_target_cell",
                        bbox=conversion.bbox,
                    )
                    return []
                bbox = BoundingBox(
                    x0=float(rectangle.x0),
                    y0=float(rectangle.y0),
                    x1=float(rectangle.x1),
                    y1=float(rectangle.y1),
                )
                if any(
                    self._overlaps_locked_region(bbox, locked)
                    for locked in locked_regions
                ):
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="table_overlaps_figure_graph_or_equation",
                        bbox=conversion.bbox,
                    )
                    return []
                style_hints = self._source_style_hints(
                    page,
                    bbox,
                    infer_alignment=True,
                )
                style_hints.update(
                    {
                        "line_height": 0.9,
                        "text_align": style_hints.get(
                            "text_align",
                            "left" if column_index == 0 else "center",
                        ),
                    }
                )
                replacements.append(
                    _ReplacementRegion(
                        page_number=block.page_number,
                        block_ids=[
                            f"{block.id}#cell-r{row_index + 1:03d}-c{column_index + 1:03d}"
                        ],
                        block_type=BlockType.TABLE,
                        bbox=bbox,
                        translated_text=translated_text,
                        source_text=source_text,
                        style_hints=style_hints,
                        coordinate_metadata=[
                            {
                                **conversion.metadata,
                                "table_grid_detection": "pdf_vector_cell_rectangles",
                                "table_block_id": block.id,
                                "row_index": row_index,
                                "column_index": column_index,
                            }
                        ],
                    )
                )
        return replacements

    def _parse_table_rows(self, markup: str) -> list[list[_ParsedTableCell]]:
        parser = _TableHTMLParser()
        try:
            parser.feed(markup)
            parser.close()
        except Exception:
            return []
        return parser.rows

    def _table_markup_is_suspicious(self, rows: list[list[_ParsedTableCell]]) -> bool:
        lengths = [len(cell.text) for row in rows for cell in row]
        if not lengths:
            return True
        largest = max(lengths)
        remainder = max(1, sum(lengths) - largest)
        return largest > 400 and largest > remainder * 0.45

    def _table_grid_rows(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
    ) -> list[list[fitz.Rect]]:
        table_rect = self._fitz_rect(bbox)
        rectangles: dict[tuple[float, float, float, float], fitz.Rect] = {}
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        for drawing in drawings:
            for item in drawing.get("items", []):
                if not item or item[0] != "re":
                    continue
                rectangle = fitz.Rect(item[1])
                if rectangle.width < 5 or rectangle.height < 3:
                    continue
                if (
                    rectangle.x0 < table_rect.x0 - 3
                    or rectangle.y0 < table_rect.y0 - 3
                    or rectangle.x1 > table_rect.x1 + 3
                    or rectangle.y1 > table_rect.y1 + 3
                ):
                    continue
                if (
                    rectangle.width > table_rect.width * 0.95
                    and rectangle.height > table_rect.height * 0.8
                ):
                    continue
                key = tuple(round(float(value), 2) for value in rectangle)
                rectangles[key] = rectangle

        rows: list[list[fitz.Rect]] = []
        for rectangle in sorted(
            rectangles.values(),
            key=lambda value: (round(value.y0, 1), value.x0),
        ):
            if (
                not rows
                or abs(rows[-1][0].y0 - rectangle.y0) > 1.0
                or abs(rows[-1][0].y1 - rectangle.y1) > 1.0
            ):
                rows.append([rectangle])
            else:
                rows[-1].append(rectangle)
        for row in rows:
            row.sort(key=lambda value: value.x0)
        return rows

    def _align_table_rows(
        self,
        page: fitz.Page,
        source_rows: list[list[_ParsedTableCell]],
        translated_rows: list[list[_ParsedTableCell]],
        grid_rows: list[list[fitz.Rect]],
    ) -> list[tuple[list[_ParsedTableCell], list[_ParsedTableCell], list[fitz.Rect]]] | None:
        if not grid_rows or len(grid_rows) > len(source_rows):
            return None

        @lru_cache(maxsize=None)
        def solve(
            grid_index: int,
            logical_index: int,
        ) -> tuple[
            float,
            tuple[tuple[list[_ParsedTableCell], list[_ParsedTableCell], list[fitz.Rect]], ...],
        ] | None:
            if grid_index == len(grid_rows) and logical_index == len(source_rows):
                return 0.0, ()
            if grid_index >= len(grid_rows) or logical_index >= len(source_rows):
                return None

            remaining_grid = len(grid_rows) - grid_index
            remaining_logical = len(source_rows) - logical_index
            max_take = min(3, remaining_logical - (remaining_grid - 1))
            best = None
            for take in range(1, max_take + 1):
                source_group = source_rows[logical_index : logical_index + take]
                translated_group = translated_rows[logical_index : logical_index + take]
                column_count = len(grid_rows[grid_index])
                if any(len(row) != column_count for row in [*source_group, *translated_group]):
                    continue
                combined_source = self._combine_table_rows(source_group)
                combined_translated = self._combine_table_rows(translated_group)
                similarity = self._table_row_similarity(
                    page,
                    combined_source,
                    grid_rows[grid_index],
                )
                if similarity < 0.68:
                    continue
                tail = solve(grid_index + 1, logical_index + take)
                if tail is None:
                    continue
                score = similarity + tail[0]
                entry = (
                    combined_source,
                    combined_translated,
                    grid_rows[grid_index],
                )
                candidate = (score, (entry, *tail[1]))
                if best is None or candidate[0] > best[0]:
                    best = candidate
            return best

        result = solve(0, 0)
        return list(result[1]) if result is not None else None

    def _combine_table_rows(
        self,
        rows: list[list[_ParsedTableCell]],
    ) -> list[_ParsedTableCell]:
        combined: list[_ParsedTableCell] = []
        for column_index in range(len(rows[0])):
            cells = [row[column_index] for row in rows]
            combined.append(
                _ParsedTableCell(
                    tag="th" if any(cell.tag == "th" for cell in cells) else "td",
                    text="\n".join(cell.text for cell in cells if cell.text),
                )
            )
        return combined

    def _table_row_similarity(
        self,
        page: fitz.Page,
        source_row: list[_ParsedTableCell],
        rectangles: list[fitz.Rect],
    ) -> float:
        weighted_score = 0.0
        total_weight = 0.0
        for cell, rectangle in zip(source_row, rectangles, strict=True):
            expected = self._comparison_text(cell.text)
            actual = self._comparison_text(page.get_text("text", clip=rectangle))
            weight = float(max(2, len(expected), len(actual)))
            total_weight += weight
            if not expected and not actual:
                score = 1.0
            elif not expected or not actual:
                score = 0.0
            else:
                score = SequenceMatcher(None, expected, actual).ratio()
                if expected in actual or actual in expected:
                    score = max(score, min(len(expected), len(actual)) / max(len(expected), len(actual)))
            weighted_score += score * weight
        return weighted_score / max(1.0, total_weight)

    def _comparison_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return "".join(character for character in normalized if character.isalnum())

    def _region_html_and_css(
        self,
        region: _ReplacementRegion,
        page: fitz.Page,
    ) -> tuple[str, str]:
        escaped = html.escape(region.translated_text)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", escaped) if part.strip()]
        rendered = "".join(f"<p>{part.replace(chr(10), '<br>')}</p>" for part in paragraphs)
        if not rendered:
            rendered = f"<p>{escaped}</p>"
        style_hints = {
            **self._source_style_hints(page, region.bbox),
            **region.style_hints,
        }
        font_size = self._font_size(region, style_hints)
        family = str(
            style_hints.get("font_family")
            or ("sans-serif" if region.block_type == BlockType.HEADING else "serif")
        )
        weight = str(
            style_hints.get("font_weight")
            or ("bold" if region.block_type == BlockType.HEADING else "normal")
        )
        style = str(
            style_hints.get("font_style")
            or ("italic" if region.block_type in {BlockType.CAPTION, BlockType.FOOTNOTE} else "normal")
        )
        default_align = (
            "center"
            if region.block_type == BlockType.CAPTION
            else "justify"
            if region.block_type == BlockType.PARAGRAPH and len(region.translated_text) > 120
            else "left"
        )
        align = str(style_hints.get("text_align") or default_align).lower()
        if align not in {"left", "right", "center", "justify"}:
            align = "left"
        try:
            line_height = float(style_hints.get("line_height", 1.15))
        except (TypeError, ValueError):
            line_height = 1.15
        line_height = max(0.8, min(1.5, line_height))
        css = (
            f"* {{ font-family: {family}; font-size: {font_size:.2f}pt; "
            f"font-weight: {weight}; font-style: {style}; color: #111; }} "
            f"p {{ margin: 0; padding: 0; line-height: {line_height:.2f}; text-align: {align}; }}"
        )
        return f"<div>{rendered}</div>", css

    def _font_size(
        self,
        region: _ReplacementRegion,
        style_hints: dict[str, Any],
    ) -> float:
        hint = style_hints.get("font_size")
        try:
            hinted = float(hint) if hint is not None else 0.0
        except (TypeError, ValueError):
            hinted = 0.0
        if hinted > 0:
            return max(6.0, min(24.0, hinted))
        height = max(1.0, region.bbox.y1 - region.bbox.y0)
        lines = max(1, region.translated_text.count("\n") + 1)
        return max(7.0, min(12.0, height / lines * 0.72))

    def _source_style_hints(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
        *,
        infer_alignment: bool = False,
    ) -> dict[str, Any]:
        rectangle = self._fitz_rect(bbox)
        spans: list[dict[str, Any]] = []
        try:
            payload = page.get_text("dict", clip=rectangle)
        except Exception:
            return {}
        for text_block in payload.get("blocks", []):
            for line in text_block.get("lines", []):
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        spans.append(span)
        if not spans:
            return {}

        weighted_sizes: list[tuple[float, int]] = []
        total_weight = 0
        bold_weight = 0
        italic_weight = 0
        sans_weight = 0
        text_rectangles: list[fitz.Rect] = []
        for span in spans:
            text = str(span.get("text", ""))
            weight = max(1, len(text.strip()))
            total_weight += weight
            try:
                size = float(span.get("size", 0.0))
            except (TypeError, ValueError):
                size = 0.0
            if size > 0:
                weighted_sizes.append((size, weight))
            font_name = str(span.get("font", "")).lower()
            flags = int(span.get("flags", 0) or 0)
            if "bold" in font_name or flags & 16:
                bold_weight += weight
            if "italic" in font_name or "oblique" in font_name or flags & 2:
                italic_weight += weight
            if any(name in font_name for name in ("arial", "helv", "sans")):
                sans_weight += weight
            try:
                text_rectangles.append(fitz.Rect(span["bbox"]))
            except Exception:
                continue

        hints: dict[str, Any] = {}
        if weighted_sizes:
            threshold = sum(weight for _size, weight in weighted_sizes) / 2
            running = 0
            for size, weight in sorted(weighted_sizes):
                running += weight
                if running >= threshold:
                    hints["font_size"] = size
                    break
        hints["font_weight"] = "bold" if bold_weight > total_weight / 2 else "normal"
        hints["font_style"] = "italic" if italic_weight > total_weight / 2 else "normal"
        hints["font_family"] = "sans-serif" if sans_weight > total_weight / 2 else "serif"
        if infer_alignment and text_rectangles:
            text_x0 = min(value.x0 for value in text_rectangles)
            text_x1 = max(value.x1 for value in text_rectangles)
            left_gap = max(0.0, text_x0 - rectangle.x0)
            right_gap = max(0.0, rectangle.x1 - text_x1)
            if abs(left_gap - right_gap) <= max(2.0, rectangle.width * 0.08):
                hints["text_align"] = "center"
            elif right_gap + 2.0 < left_gap:
                hints["text_align"] = "right"
            else:
                hints["text_align"] = "left"
        return hints

    def _preflight(
        self,
        *,
        page_width: float,
        page_height: float,
        region: _ReplacementRegion,
        html_text: str,
        css: str,
    ) -> tuple[float, float]:
        probe = fitz.open()
        try:
            page = probe.new_page(width=page_width, height=page_height)
            spare_height, scale = page.insert_htmlbox(
                self._fitz_rect(region.bbox),
                html_text,
                css=css,
                scale_low=self.minimum_scale,
            )
            return float(spare_height), float(scale)
        finally:
            probe.close()

    def _restore_missing_links(
        self,
        page: fitz.Page,
        original_links: list[dict[str, Any]],
        report: dict[str, Any],
        page_number: int,
    ) -> None:
        remaining = {self._link_signature(link) for link in page.get_links()}
        for link in original_links:
            if self._link_signature(link) in remaining:
                continue
            payload = {key: value for key, value in link.items() if key not in {"xref", "id"}}
            try:
                page.insert_link(payload)
            except Exception as exc:
                self._warning(
                    report,
                    page_number=page_number,
                    code="link_restore_failed",
                    reason=f"A link removed during text replacement could not be restored: {exc}",
                )

    def _link_signature(self, link: dict[str, Any]) -> tuple:
        rectangle = fitz.Rect(link.get("from", fitz.Rect()))
        return (
            link.get("kind"),
            round(rectangle.x0, 2),
            round(rectangle.y0, 2),
            round(rectangle.x1, 2),
            round(rectangle.y1, 2),
            link.get("uri"),
            link.get("page"),
        )

    def _skip_block(
        self,
        report: dict[str, Any],
        page_report: dict[str, Any],
        block: Block,
        *,
        reason: str,
        bbox: BoundingBox | None = None,
        fallback_required: bool = True,
    ) -> None:
        report["regions_skipped"] += 1
        page_report["regions_skipped"] += 1
        if fallback_required:
            page_report["fallback_required"] = True
        report["regions"].append(
            {
                "page_number": block.page_number,
                "block_ids": [block.id],
                "block_type": block.block_type.value,
                "bbox": bbox.model_dump() if bbox is not None else None,
                "status": "skipped",
                "reason": reason,
            }
        )
        self._warning(
            report,
            page_number=block.page_number,
            code="region_skipped",
            reason=f"Region {block.id} was skipped: {reason}.",
        )

    def _skipped_region(
        self,
        report: dict[str, Any],
        region: _ReplacementRegion,
        *,
        reason: str,
        scale: float,
    ) -> None:
        report["regions_skipped"] += 1
        report["regions"].append(
            {
                "page_number": region.page_number,
                "block_ids": region.block_ids,
                "block_type": region.block_type.value,
                "bbox": region.bbox.model_dump(),
                "status": "skipped",
                "reason": reason,
                "scale": round(float(scale), 6),
            }
        )

    def _warning(
        self,
        report: dict[str, Any],
        *,
        code: str,
        reason: str,
        page_number: int | None = None,
    ) -> None:
        report["warnings"].append(
            {
                "page_number": page_number,
                "code": code,
                "reason": reason,
            }
        )

    def _overlaps_locked_region(self, region: BoundingBox, locked: BoundingBox) -> bool:
        intersection = bbox_intersection_area(region, locked)
        if intersection <= 0:
            return False
        return intersection / max(1.0, bbox_area(region)) > 0.08 or intersection > 36.0

    def _union_bbox(self, bboxes: list[BoundingBox]) -> BoundingBox:
        return BoundingBox(
            x0=min(bbox.x0 for bbox in bboxes),
            y0=min(bbox.y0 for bbox in bboxes),
            x1=max(bbox.x1 for bbox in bboxes),
            y1=max(bbox.y1 for bbox in bboxes),
        )

    def _fitz_rect(self, bbox: BoundingBox) -> fitz.Rect:
        return fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

    def _normalized_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
