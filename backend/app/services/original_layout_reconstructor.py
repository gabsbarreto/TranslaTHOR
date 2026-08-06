from __future__ import annotations

import html
import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]

from app.models.schema import (
    Block,
    BlockType,
    BoundingBox,
    DocumentModel,
    SourceType,
    TableModel,
)
from app.services.pdf_coordinates import (
    bbox_area,
    bbox_intersection_area,
    convert_bbox_to_pdf,
)
from app.services.table_markup import ParsedTableCell, parse_table_rows

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
    redaction_bboxes: list[BoundingBox] | None = None
    redaction_fill: tuple[float, float, float] | None = None
    reconstruction_strategy: str = "embedded_text_replacement"
    authoritative_bbox: bool = False


@dataclass(frozen=True)
class _PhysicalTableLine:
    y0: float
    y1: float
    cells: tuple[str, ...]


@dataclass(frozen=True)
class _SemanticTableGrid:
    rows: tuple[tuple[fitz.Rect, ...], ...]
    score: float
    signature: tuple[float, ...]
    assignment_signature: tuple[str, ...]


@dataclass(frozen=True)
class _HiddenOCRLine:
    block_index: int
    line_index: int
    bbox: BoundingBox
    text: str

    @property
    def key(self) -> tuple[int, int]:
        return self.block_index, self.line_index


@dataclass(frozen=True)
class _HiddenOCRMatch:
    lines: tuple[_HiddenOCRLine, ...]
    bbox: BoundingBox
    text: str
    score: float
    competing_score: float
    minimum_score: float

    @property
    def keys(self) -> frozenset[tuple[int, int]]:
        return frozenset(line.key for line in self.lines)


class OriginalLayoutReconstructor:
    """Replace translated text while retaining the source PDF page art."""

    # A zero lower bound asks PyMuPDF to find whatever scale is required to
    # contain the complete target. Placement fidelity, rather than an arbitrary
    # readability floor, is the only acceptance rule for an extracted text box.
    minimum_scale = 0.0
    maximum_incidental_table_cell_overlap = 0.25
    locked_redaction_guard = 1.0
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
        tables_by_block_id: dict[str, TableModel] = {}
        for table in document.tables:
            source_block_id = str(
                table.debug.get("source_block_id") or table.debug.get("marker_block_id") or ""
            )
            if source_block_id and source_block_id not in tables_by_block_id:
                tables_by_block_id[source_block_id] = table
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
                    "reconstruction_strategy": "none",
                    "regions_replaced": 0,
                    "regions_skipped": 0,
                    "regions_retained": 0,
                    "fallback_required": False,
                    "warnings": [],
                }
                report["pages"].append(page_report)
                strategy, reason = self._page_strategy(
                    page,
                    page_metadata.get(page_number),
                    blocks_by_page.get(page_number, []),
                )
                if strategy == "unsupported":
                    page_report["status"] = "fallback_original_page"
                    page_report["fallback_required"] = True
                    page_report["warnings"].append(reason)
                    report["pages_using_fallback_behavior"] += 1
                    self._warning(
                        report,
                        page_number=page_number,
                        code="page_not_safely_replaceable",
                        reason=reason,
                    )
                    continue

                authoritative_bbox_overlay = strategy in {
                    "authoritative_bbox_overlay",
                    # Backward-compatible handling for an unusual Surya page
                    # that reaches the legacy strategy branch below.
                    "surya2_image_overlay",
                }
                scan_overlay = strategy in {"ocr_text_overlay", "ocr_table_overlay"}
                scan_table_only = strategy == "ocr_table_overlay"
                page_report["reconstruction_strategy"] = strategy
                page_is_ocr = bool(
                    page_metadata.get(page_number)
                    and page_metadata[page_number].extraction_mode == SourceType.OCR
                )
                if scan_overlay or (authoritative_bbox_overlay and page_is_ocr):
                    report["scan_overlay_pages"] += 1
                if authoritative_bbox_overlay and any(
                    self._is_surya2_region(block) for block in blocks_by_page.get(page_number, [])
                ):
                    report["surya2_image_overlay_pages"] += 1
                if authoritative_bbox_overlay:
                    report["authoritative_bbox_overlay_pages"] += 1

                replacements = self._replacement_regions(
                    page=page,
                    page_number=page_number,
                    blocks=blocks_by_page.get(page_number, []),
                    all_blocks=document.blocks,
                    locked_regions=locked_by_page.get(page_number, []),
                    tables_by_block_id=tables_by_block_id,
                    recovered_translations=recovered_translations,
                    report=report,
                    page_report=page_report,
                    scan_overlay=scan_overlay,
                    scan_table_only=scan_table_only,
                    authoritative_bbox_overlay=authoritative_bbox_overlay,
                )
                guarded_replacements: list[_ReplacementRegion] = []
                for region in replacements:
                    if region.authoritative_bbox:
                        # Extracted text regions are authoritative. Do not trim
                        # or reject their boxes using a second geometry opinion.
                        guarded_replacements.append(region)
                        continue
                    region.redaction_bboxes = self._redaction_bboxes_avoiding_locked_regions(
                        self._redaction_bboxes(region),
                        locked_by_page.get(page_number, []),
                        page_width=float(page.rect.width),
                        page_height=float(page.rect.height),
                    )
                    if region.redaction_bboxes:
                        guarded_replacements.append(region)
                        continue
                    page_report["regions_skipped"] += 1
                    page_report["fallback_required"] = True
                    self._skipped_region(
                        report,
                        region,
                        reason="redaction_region_consumed_by_locked_visual_guard",
                        scale=1.0,
                    )
                    self._warning(
                        report,
                        page_number=page_number,
                        code="locked_visual_redaction_guard",
                        reason=(
                            f"Region {', '.join(region.block_ids)} could not be safely redacted "
                            "without touching a locked figure or equation."
                        ),
                    )
                replacements = guarded_replacements
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
                            reason="translated_text_did_not_fit_box",
                            scale=scale,
                        )
                        self._warning(
                            report,
                            page_number=page_number,
                            code="text_box_overflow",
                            reason=(
                                f"Translated text for {', '.join(region.block_ids)} could not fit "
                                "entirely inside its extracted box, even with automatic scaling; "
                                "source text was retained."
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

                backup = fitz.open()
                backup.insert_pdf(
                    pdf,
                    from_page=page_number - 1,
                    to_page=page_number - 1,
                )
                original_links = page.get_links()
                transaction_entries: list[dict[str, Any]] = []
                failed_region: _ReplacementRegion | None = None
                failed_scale = 0.0
                transaction_error: str | None = None
                try:
                    for region, _html_text, _css, _scale in approved:
                        redaction_bboxes = self._redaction_bboxes(region)
                        for redaction_bbox in redaction_bboxes:
                            page.add_redact_annot(
                                self._fitz_rect(redaction_bbox),
                                fill=region.redaction_fill,
                                cross_out=False,
                            )
                    if not page.apply_redactions(images=0, graphics=0, text=0):
                        raise RuntimeError("redactions_not_applied")

                    for region, html_text, css, _preflight_scale in approved:
                        try:
                            spare_height, scale = page.insert_htmlbox(
                                self._fitz_rect(region.bbox),
                                html_text,
                                css=css,
                                scale_low=self.minimum_scale,
                                overlay=True,
                            )
                        except Exception as exc:
                            failed_region = region
                            transaction_error = f"text_insertion_error:{type(exc).__name__}"
                            break
                        entry = {
                            "page_number": page_number,
                            "block_ids": region.block_ids,
                            "block_type": region.block_type.value,
                            "bbox": region.bbox.model_dump(),
                            "reconstruction_strategy": region.reconstruction_strategy,
                            "source_character_count": self._source_character_count(
                                region.source_text
                            ),
                            "source_text_mask_count": len(region.redaction_bboxes or []),
                            "source_text_masks": [
                                bbox.model_dump() for bbox in (region.redaction_bboxes or [])
                            ],
                            "applied_redaction_bboxes": [
                                bbox.model_dump() for bbox in self._redaction_bboxes(region)
                            ],
                            "coordinate_metadata": region.coordinate_metadata,
                            "scale": round(float(scale), 6),
                            "spare_height": round(float(spare_height), 6),
                        }
                        transaction_entries.append(entry)
                        if spare_height < 0 or scale < self.minimum_scale:
                            failed_region = region
                            failed_scale = float(scale)
                            transaction_error = "unexpected_post_redaction_overflow"
                            report["text_boxes_did_not_fit"] += 1
                            break
                except Exception as exc:
                    failed_region = failed_region or approved[0][0]
                    transaction_error = f"page_reconstruction_error:{type(exc).__name__}"

                if failed_region is not None:
                    for entry in transaction_entries:
                        report["scaling_applied"].append({**entry, "status": "rolled_back"})
                    page = None
                    pdf.delete_page(page_number - 1)
                    pdf.insert_pdf(backup, from_page=0, to_page=0, start_at=page_number - 1)
                    backup.close()
                    page_report["regions_skipped"] += len(approved)
                    page_report["fallback_required"] = True
                    page_report["status"] = "fallback_original_page"
                    failed_reason = (
                        "unexpected_post_redaction_overflow"
                        if transaction_error == "unexpected_post_redaction_overflow"
                        else "page_reconstruction_transaction_failed"
                    )
                    for region, _html_text, _css, preflight_scale in approved:
                        self._skipped_region(
                            report,
                            region,
                            reason=(
                                failed_reason
                                if region is failed_region
                                else "page_transaction_rolled_back_after_insertion_failure"
                            ),
                            scale=failed_scale or preflight_scale,
                        )
                    self._warning(
                        report,
                        page_number=page_number,
                        code="page_reconstruction_rolled_back",
                        reason=(
                            f"Page reconstruction was rolled back after {transaction_error}; "
                            "the complete original page was retained."
                        ),
                    )
                    report["pages_using_fallback_behavior"] += 1
                    continue

                backup.close()
                self._restore_missing_links(page, original_links, report, page_number)
                for entry, (region, _html_text, _css, _preflight_scale) in zip(
                    transaction_entries,
                    approved,
                    strict=True,
                ):
                    report["scaling_applied"].append({**entry, "status": "committed"})
                    report["regions_replaced"] += 1
                    page_report["regions_replaced"] += 1
                    report["scan_text_masks"] += len(region.redaction_bboxes or [])
                    if region.authoritative_bbox:
                        report["authoritative_bbox_text_masks"] += len(
                            region.redaction_bboxes or []
                        )
                    if region.reconstruction_strategy == "surya2_authoritative_bbox_overlay":
                        report["surya2_image_text_masks"] += len(region.redaction_bboxes or [])
                    report["regions"].append({**entry, "status": "replaced"})
                raster_table_ids = {
                    str(metadata["table_block_id"])
                    for region, _html_text, _css, _preflight_scale in approved
                    if region.reconstruction_strategy == "ocr_table_cell_overlay"
                    for metadata in region.coordinate_metadata
                    if metadata.get("table_block_id")
                }
                report["raster_tables_reconstructed"] += len(raster_table_ids)
                if page_report["fallback_required"]:
                    page_report["status"] = "partial"
                    report["pages_using_fallback_behavior"] += 1
                else:
                    page_report["status"] = "success"
                    report["pages_successfully_reconstructed"] += 1

            pdf.save(temporary_path, garbage=3, deflate=True)

        temporary_path.replace(output_pdf_path)
        report["status"] = "partial" if report["pages_using_fallback_behavior"] else "complete"
        report["output_pdf"] = str(output_pdf_path.resolve())
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _initial_report(self, document: DocumentModel) -> dict[str, Any]:
        page_dimensions = {
            page.page_number: (float(page.width), float(page.height)) for page in document.pages
        }
        valid_figures = [
            figure
            for figure in document.figures
            if self._figure_bbox_is_valid(
                figure.bbox,
                page_dimensions.get(figure.page_number),
            )
        ]
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
                figure.detection_confidence is not None and figure.detection_confidence < 0.6
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
            "figures_preserved": 0,
            "regions_replaced": 0,
            "regions_skipped": 0,
            "regions_retained": 0,
            "regions_missing_or_invalid_bboxes": 0,
            "text_boxes_did_not_fit": 0,
            "scaling_applied": [],
            "raster_figure_fallbacks": raster_fallbacks,
            "low_confidence_figure_or_caption_associations": low_confidence,
            "warnings": [],
            "pages": [],
            "regions": [],
            "scan_overlay_pages": 0,
            "scan_text_masks": 0,
            "scan_text_regions_aligned": 0,
            "scan_text_regions_alignment_failed": 0,
            "surya2_image_overlay_pages": 0,
            "surya2_image_text_masks": 0,
            "authoritative_bbox_overlay_pages": 0,
            "authoritative_bbox_text_masks": 0,
            "region_policy": "authoritative_extracted_bbox_with_full_text_fit",
            "fit_policy": "automatic_downscale_without_readability_floor",
            "surya2_region_policy": "authoritative_bbox_with_full_text_fit",
            "raster_tables_reconstructed": 0,
            "safe_fallback": "readable_pdf",
            "minimum_text_scale": self.minimum_scale,
            "internal_figure_text_policy": "preserve_source_language",
        }

    def _figure_bbox_is_valid(
        self,
        bbox: BoundingBox | None,
        page_dimensions: tuple[float, float] | None,
    ) -> bool:
        if bbox is None or page_dimensions is None or bbox_area(bbox) < 1.0:
            return False
        width, height = page_dimensions
        values = (bbox.x0, bbox.y0, bbox.x1, bbox.y1, width, height)
        if not all(math.isfinite(float(value)) for value in values):
            return False
        tolerance = 0.5
        return (
            bbox.x0 >= -tolerance
            and bbox.y0 >= -tolerance
            and bbox.x1 <= width + tolerance
            and bbox.y1 <= height + tolerance
        )

    def _page_strategy(
        self,
        page: fitz.Page,
        metadata,
        blocks: list[Block],
    ) -> tuple[str, str]:
        if page.rotation:
            return (
                "unsupported",
                "Rotated pages are retained unchanged in this first original-layout implementation.",
            )
        if any(
            block.bbox is not None
            and block.block_type not in self.locked_block_types
            and (
                bool(block.text.strip())
                or isinstance(block.metadata.get("translated_from_block_ids"), list)
            )
            for block in blocks
        ):
            # Every extractor's text-region geometry is authoritative. Source
            # PDF text, hidden OCR, table-cell recovery, and visual overlap
            # heuristics are not allowed to veto a valid extracted box.
            return "authoritative_bbox_overlay", ""
        if metadata is None:
            return (
                "unsupported",
                "Structured page metadata is missing; the source page was retained unchanged.",
            )
        if metadata.extraction_mode == SourceType.OCR:
            surya2_blocks = [block for block in blocks if self._is_surya2_region(block)]
            if any(
                block.bbox is not None and block.block_type not in self.locked_block_types
                for block in surya2_blocks
            ):
                # Surya's region geometry is authoritative even when the scan
                # also contains a selectable hidden-OCR layer. Hidden OCR is a
                # competing transcription and must not veto a Surya box.
                return "surya2_image_overlay", ""
            if metadata.has_embedded_text and len(page.get_text("words")) >= 5:
                translated_tables = {
                    block.id
                    for block in blocks
                    if block.block_type == BlockType.TABLE
                    and isinstance(block.metadata.get("translated_from_block_ids"), list)
                    and block.id in block.metadata.get("translated_from_block_ids", [])
                }
                translated_body = any(
                    block.block_type not in {BlockType.TABLE, BlockType.CAPTION}
                    and isinstance(block.metadata.get("translated_from_block_ids"), list)
                    for block in blocks
                )
                if translated_tables and not translated_body:
                    return "ocr_table_overlay", ""
                return "ocr_text_overlay", ""
            return (
                "unsupported",
                "The OCR page has neither usable hidden text geometry nor Surya 2 image-region "
                "geometry for a safe translated overlay; it was retained unchanged.",
            )
        if not metadata.has_embedded_text or metadata.embedded_text_quality < 0.35:
            return (
                "unsupported",
                "The page is scanned, image-only, hidden-OCR, or has unreliable embedded text; it was retained unchanged.",
            )
        if len(page.get_text("text").strip()) < 5:
            return (
                "unsupported",
                "The page has no reliable removable PDF text and was retained unchanged.",
            )
        return "embedded_text_replacement", ""

    def _is_surya2_region(self, block: Block) -> bool:
        return (
            str(block.metadata.get("ocr_engine", "")).casefold() == "surya2_llamacpp"
            or str(block.metadata.get("parser", "")).casefold() == "surya2_llamacpp"
        )

    def _locked_regions(
        self,
        document: DocumentModel,
        pdf: fitz.Document,
        report: dict[str, Any],
    ) -> dict[int, list[BoundingBox]]:
        locked: dict[int, list[BoundingBox]] = {}
        represented_figure_blocks: set[str] = set()
        for figure in document.figures:
            if not (1 <= figure.page_number <= pdf.page_count):
                report["regions_missing_or_invalid_bboxes"] += 1
                self._warning(
                    report,
                    page_number=figure.page_number,
                    code="figure_lock_region_invalid",
                    reason=f"Figure {figure.id} refers to a source page that does not exist.",
                )
                continue
            page = pdf[figure.page_number - 1]
            bbox_valid = self._figure_bbox_is_valid(
                figure.bbox,
                (float(page.rect.width), float(page.rect.height)),
            )
            conversion = convert_bbox_to_pdf(
                figure.bbox,
                page_width=page.rect.width,
                page_height=page.rect.height,
                metadata={
                    "source_page_width": page.rect.width,
                    "source_page_height": page.rect.height,
                },
            )
            if conversion.bbox is not None:
                represented_figure_blocks.update(figure.source_block_ids)
                locked.setdefault(figure.page_number, []).append(conversion.bbox)
                if bbox_valid:
                    report["figures_preserved"] += 1
            if not bbox_valid or conversion.bbox is None:
                report["regions_missing_or_invalid_bboxes"] += 1
                self._warning(
                    report,
                    page_number=figure.page_number,
                    code="figure_lock_region_invalid",
                    reason=(
                        f"Figure {figure.id} has a missing, empty, or out-of-page bounding box; "
                        "it was not counted as a validated preserved figure."
                    ),
                )
        for block in document.blocks:
            if block.block_type not in self.locked_block_types or not (
                1 <= block.page_number <= pdf.page_count
            ):
                continue
            if block.block_type == BlockType.FIGURE and block.id in represented_figure_blocks:
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
        tables_by_block_id: dict[str, TableModel],
        recovered_translations: dict[str, str],
        report: dict[str, Any],
        page_report: dict[str, Any],
        scan_overlay: bool = False,
        scan_table_only: bool = False,
        authoritative_bbox_overlay: bool = False,
    ) -> list[_ReplacementRegion]:
        block_by_id = {block.id: block for block in all_blocks}
        consumed: set[str] = set()
        replacements: list[_ReplacementRegion] = []
        claimed_scan_lines: set[tuple[int, int]] = set()
        scan_text_sequences = (
            self._hidden_ocr_text_sequences(self._hidden_ocr_text_blocks(page))
            if scan_overlay
            else []
        )
        scan_caption_ids: set[str] = set()
        scan_table_blocks = {
            block.id: block for block in blocks if block.block_type == BlockType.TABLE
        }
        successful_scan_table_regions: dict[str, list[_ReplacementRegion]] = {}
        failed_scan_table_ids: set[str] = set()
        if scan_table_only:
            ordered = sorted(blocks, key=lambda item: item.reading_order_index)
            for index, candidate in enumerate(ordered[:-1]):
                following = ordered[index + 1]
                if (
                    candidate.block_type == BlockType.TABLE
                    and following.block_type == BlockType.CAPTION
                    and following.reading_order_index - candidate.reading_order_index <= 2
                ):
                    scan_caption_ids.add(following.id)
        for block in blocks:
            if block.id in consumed:
                continue
            if block.metadata.get("excluded_from_translation"):
                exclusion_reason = str(
                    block.metadata.get("translation_exclusion_reason")
                    or "excluded_from_translation"
                )
                self._retain_block(
                    report,
                    page_report,
                    block,
                    reason=exclusion_reason,
                    bbox=self._block_pdf_bbox(page, block),
                )
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
                self._retain_block(
                    report,
                    page_report,
                    block,
                    reason="locked_visual_region",
                    bbox=self._block_pdf_bbox(page, block),
                )
                continue
            authoritative_block = authoritative_bbox_overlay
            if block.block_type == BlockType.TABLE and not authoritative_block:
                validation = block.metadata.get("translation_validation")
                if (
                    isinstance(validation, dict)
                    and validation.get("status") == "translation_failed"
                ):
                    if scan_overlay:
                        failed_scan_table_ids.add(block.id)
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason=str(
                            validation.get("reason")
                            or "translation_target_language_validation_failed"
                        ),
                    )
                    continue
                skipped_before = page_report["regions_skipped"]
                table_regions = self._table_replacement_regions(
                    page=page,
                    block=block,
                    table_model=tables_by_block_id.get(block.id),
                    locked_regions=locked_regions,
                    report=report,
                    page_report=page_report,
                    scan_overlay=scan_overlay,
                )
                replacements.extend(table_regions)
                if scan_overlay and table_regions:
                    successful_scan_table_regions[block.id] = table_regions
                elif scan_overlay and page_report["regions_skipped"] > skipped_before:
                    failed_scan_table_ids.add(block.id)
                continue
            if block.block_type in self.conservative_skip_types and not authoritative_block:
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="table_layout_preserved_conservatively",
                )
                continue

            if recovered_text is not None:
                translated_from: list[str] = [block.id]
                source_blocks = [block]
                translated_text = recovered_text
            else:
                translated_from_value = block.metadata.get("translated_from_block_ids")
                if not isinstance(translated_from_value, list) or not translated_from_value:
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason=(
                            "scan_table_only_non_table_translation_unavailable"
                            if scan_table_only
                            and block.block_type != BlockType.TABLE
                            and block.id not in scan_caption_ids
                            else "translation_not_confirmed_for_region"
                        ),
                    )
                    continue
                translated_from = [str(value) for value in translated_from_value]
                source_blocks = [
                    block_by_id[block_id] for block_id in translated_from if block_id in block_by_id
                ]
                if not source_blocks or any(
                    item.page_number != page_number for item in source_blocks
                ):
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="translated_group_spans_pages_or_is_missing",
                    )
                    consumed.update(str(value) for value in translated_from)
                    continue
                translated_text = block.text.strip()

            authoritative_region = authoritative_block

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
            source_bbox_missing = len(converted) != len(source_blocks)
            recover_bbox_from_hidden_ocr = (
                scan_overlay and not authoritative_region and source_bbox_missing
            )
            if source_bbox_missing and not recover_bbox_from_hidden_ocr:
                report["regions_missing_or_invalid_bboxes"] += 1
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="missing_or_invalid_source_bbox",
                )
                continue

            global_search_bbox = BoundingBox(
                x0=0.0,
                y0=0.0,
                x1=float(page.rect.width),
                y1=float(page.rect.height),
            )
            bbox = (
                global_search_bbox if recover_bbox_from_hidden_ocr else self._union_bbox(converted)
            )

            source_text = " ".join(
                str(item.metadata.get("source_text", "")).strip()
                for item in source_blocks
                if str(item.metadata.get("source_text", "")).strip()
            )
            validation = block.metadata.get("translation_validation")
            if (
                not authoritative_region
                and isinstance(validation, dict)
                and validation.get("status") == "translation_failed"
            ):
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason=str(
                        validation.get("reason") or "translation_target_language_validation_failed"
                    ),
                    bbox=None if recover_bbox_from_hidden_ocr else bbox,
                )
                continue
            if (
                not authoritative_region
                and source_text
                and self._normalized_text(source_text) == self._normalized_text(translated_text)
            ):
                # English or otherwise unchanged text already matches the visual base.
                continue
            if not scan_overlay and not authoritative_region:
                source_validation = self._embedded_source_text_validation(
                    page,
                    bbox,
                    source_text,
                )
                if not source_validation["safe"]:
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason=str(source_validation["reason"]),
                        bbox=bbox,
                        alignment_diagnostics={
                            "source_text_validation": source_validation,
                        },
                    )
                    continue
                conversions.append(
                    {
                        "geometry_source": "embedded_pdf_text_validation",
                        "coordinate_space": "pdf_points",
                        "source_text_validation": source_validation,
                    }
                )
            if scan_overlay and not authoritative_region and not source_text:
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="scan_source_text_missing",
                    bbox=None if recover_bbox_from_hidden_ocr else bbox,
                )
                continue
            scan_match: _HiddenOCRMatch | None = None
            if scan_overlay and not authoritative_region:
                if self._source_text_is_probably_english(source_text):
                    # The application targets English. Re-typesetting an
                    # already-English passage only introduces scan artefacts
                    # when a translator has lightly paraphrased it.
                    continue
                if self._translation_script_is_suspicious(source_text, translated_text):
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="translated_text_script_incompatible_with_english_output",
                        bbox=None if recover_bbox_from_hidden_ocr else bbox,
                    )
                    continue
            if scan_overlay and not authoritative_region:
                scan_match, match_metadata = self._match_hidden_ocr_lines(
                    page,
                    source_text,
                    preferred_bbox=bbox,
                    unavailable_line_keys=claimed_scan_lines,
                    text_sequences=scan_text_sequences,
                )
                if recover_bbox_from_hidden_ocr:
                    match_metadata = {
                        **match_metadata,
                        "geometry_source": "global_hidden_ocr_alignment_recovered_bbox",
                        "preferred_search_extent_pdf": global_search_bbox.model_dump(),
                    }
                if scan_match is None:
                    if recover_bbox_from_hidden_ocr:
                        report["regions_missing_or_invalid_bboxes"] += 1
                    report["scan_text_regions_alignment_failed"] += 1
                    failure_reason = str(
                        match_metadata.get("reason", "hidden_ocr_text_alignment_failed")
                    )
                    if scan_table_only and block.block_type == BlockType.CAPTION:
                        failure_reason = "caption_hidden_ocr_text_mismatch"
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason=failure_reason,
                        bbox=None if recover_bbox_from_hidden_ocr else bbox,
                        alignment_diagnostics=match_metadata,
                    )
                    continue
                if block.block_type not in {
                    BlockType.TABLE,
                    BlockType.CAPTION,
                } and self._scan_match_has_multicolumn_text(page, scan_match.lines):
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="scan_region_multicolumn_layout_requires_table_geometry",
                        bbox=scan_match.bbox,
                    )
                    continue
                bbox = self._scan_match_envelope(page, scan_match.bbox)

            if not authoritative_region and any(
                self._overlaps_locked_region(bbox, locked) for locked in locked_regions
            ):
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="overlaps_figure_graph_or_equation",
                    bbox=bbox,
                )
                continue
            replacement = _ReplacementRegion(
                page_number=page_number,
                block_ids=[item.id for item in source_blocks],
                block_type=block.block_type,
                bbox=bbox,
                translated_text=translated_text,
                source_text=source_text,
                style_hints=dict(block.style_hints or {}),
                coordinate_metadata=conversions,
                authoritative_bbox=authoritative_region,
            )
            if scan_overlay and not authoritative_region and scan_match is not None:
                masks = self._scan_match_masks(page, scan_match.lines)
                fill, background_metadata = self._scan_background_fill(page, bbox)
                if fill is None or not masks:
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="scan_text_mask_or_background_not_reliable",
                        bbox=bbox,
                    )
                    continue
                replacement.redaction_bboxes = masks
                replacement.redaction_fill = fill
                replacement.reconstruction_strategy = "ocr_hidden_text_overlay"
                alignment_metadata: dict[str, Any] = {
                    "geometry_source": "hidden_ocr_contiguous_line_alignment",
                    "coordinate_space": "pdf_points",
                    "matched_hidden_ocr_bbox_pdf": scan_match.bbox.model_dump(),
                    "matched_hidden_ocr_text": scan_match.text,
                    "source_text_similarity": {
                        "score": round(scan_match.score, 6),
                        "competing_score": round(scan_match.competing_score, 6),
                        "minimum_score": scan_match.minimum_score,
                    },
                    "mask_source": "matched_hidden_ocr_line_geometry",
                    "scan_background": background_metadata,
                }
                if recover_bbox_from_hidden_ocr:
                    alignment_metadata.update(
                        {
                            "geometry_source": "global_hidden_ocr_alignment_recovered_bbox",
                            "bbox_recovered_by_global_hidden_ocr_alignment": True,
                            "preferred_search_extent_pdf": global_search_bbox.model_dump(),
                        }
                    )
                else:
                    alignment_metadata["surya_region_bbox_pdf"] = (
                        converted[0].model_dump()
                        if len(converted) == 1
                        else self._union_bbox(converted).model_dump()
                    )
                replacement.coordinate_metadata.append(alignment_metadata)
                claimed_scan_lines.update(scan_match.keys)
                report["scan_text_regions_aligned"] += 1
            elif authoritative_region:
                fill, background_metadata = self._authoritative_background_fill(page, bbox)
                replacement.redaction_bboxes = [bbox]
                replacement.redaction_fill = fill
                is_surya2_region = all(
                    self._is_surya2_region(source_block) for source_block in source_blocks
                )
                replacement.reconstruction_strategy = (
                    "surya2_authoritative_bbox_overlay"
                    if is_surya2_region
                    else "authoritative_bbox_overlay"
                )
                replacement.coordinate_metadata.append(
                    {
                        "geometry_source": "extracted_region_bbox",
                        "coordinate_space": "pdf_points_top_left",
                        "extracted_region_bbox_pdf": bbox.model_dump(),
                        "mask_source": "full_region_bbox",
                        "region_policy": "authoritative_extracted_bbox_with_full_text_fit",
                        "extractor": (
                            "surya2_llamacpp"
                            if is_surya2_region
                            else str(block.metadata.get("parser") or "unknown")
                        ),
                        "scan_background": background_metadata,
                    }
                )
            replacements.append(replacement)

        if (
            scan_overlay
            and len(scan_table_blocks) > 1
            and successful_scan_table_regions
            and failed_scan_table_ids
        ):
            # OCR table segmentation does not currently expose a trustworthy
            # higher-level group ID. Treat all reconstructed tables on the same
            # scan page as one conservative visual group: a failed sibling must
            # not leave the page with a mixture of translated and source tables.
            # Non-table replacements remain eligible and are committed normally.
            retained_region_ids = {
                id(region)
                for regions in successful_scan_table_regions.values()
                for region in regions
            }
            replacements = [
                region for region in replacements if id(region) not in retained_region_ids
            ]
            retained_table_ids = sorted(successful_scan_table_regions)
            for table_id in retained_table_ids:
                table_block = scan_table_blocks[table_id]
                self._skip_block(
                    report,
                    page_report,
                    table_block,
                    reason="scan_table_group_retained_after_sibling_failure",
                    bbox=self._block_pdf_bbox(page, table_block),
                )
            failed_ids = sorted(failed_scan_table_ids)
            self._warning(
                report,
                page_number=page_number,
                code="scan_table_group_atomic_fallback",
                reason=(
                    "Multiple table regions form a conservative scan-page table group. "
                    f"Table overlay failed for {', '.join(failed_ids)}, so translated overlays "
                    f"for {', '.join(retained_table_ids)} were also retained in the source "
                    "language; non-table regions remain eligible for reconstruction."
                ),
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
            if not (len(source_parts) == len(translated_parts) == len(chunk.block_ids)):
                continue
            source_blocks = [block_by_id.get(block_id) for block_id in chunk.block_ids]
            if any(block is None for block in source_blocks):
                continue
            if not all(
                self._normalized_text(str(block.metadata.get("source_text", block.text)))
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
        table_model: TableModel | None,
        locked_regions: list[BoundingBox],
        report: dict[str, Any],
        page_report: dict[str, Any],
        scan_overlay: bool = False,
    ) -> list[_ReplacementRegion]:
        translated_from = block.metadata.get("translated_from_block_ids")
        source_markup = str(block.metadata.get("source_text", ""))
        source_hint = str(block.metadata.get("source_text_before_cleaning", ""))
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

        source_rows = self._parse_table_rows(source_markup, source_hint=source_hint)
        translated_rows = self._parse_table_rows(block.text, source_hint=source_hint)
        if (
            not source_rows
            or len(source_rows) != len(translated_rows)
            or any(
                len(source_row) != len(translated_row)
                or any(
                    source_cell.rowspan != translated_cell.rowspan
                    or source_cell.colspan != translated_cell.colspan
                    for source_cell, translated_cell in zip(
                        source_row,
                        translated_row,
                        strict=True,
                    )
                )
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

        expected_columns = max(
            (sum(max(1, int(cell.colspan or 1)) for cell in row) for row in source_rows),
            default=0,
        )
        grid_rows, grid_strategy = self._stored_table_grid_rows(
            page=page,
            block=block,
            table=table_model,
            source_rows=source_rows,
            table_bbox=conversion.bbox,
        )
        if not grid_rows:
            grid_rows, grid_strategy = self._table_grid_rows(
                page,
                conversion.bbox,
                source_rows=source_rows,
                expected_rows=len(source_rows),
                expected_columns=expected_columns,
            )
        aligned = self._align_table_rows(
            page,
            source_rows,
            translated_rows,
            grid_rows,
        )
        if aligned is None or not self._table_alignment_has_full_cell_coverage(page, aligned):
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
            insertion_rectangles = self._nonoverlapping_table_row_rectangles(rectangles)
            for column_index, (
                source_cell,
                translated_cell,
                rectangle,
                insertion_rectangle,
            ) in enumerate(
                zip(
                    source_row,
                    translated_row,
                    rectangles,
                    insertion_rectangles,
                    strict=True,
                )
            ):
                source_text = source_cell.text.strip()
                translated_text = translated_cell.text.strip()
                if self._normalized_text(source_text) == self._normalized_text(translated_text):
                    continue
                if not source_text and translated_text:
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="table_translation_added_content_to_empty_source_cell",
                        bbox=conversion.bbox,
                    )
                    return []
                if source_text and not translated_text:
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="table_translation_contains_empty_target_cell",
                        bbox=conversion.bbox,
                    )
                    return []
                cell_bbox = BoundingBox(
                    x0=float(rectangle.x0),
                    y0=float(rectangle.y0),
                    x1=float(rectangle.x1),
                    y1=float(rectangle.y1),
                )
                if not translated_text:
                    # Empty cells can contain arrows or other non-text artwork.
                    # Leaving them untouched is part of the atomic table policy.
                    continue
                if any(
                    self._overlaps_locked_region(cell_bbox, locked) for locked in locked_regions
                ):
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="table_overlaps_figure_graph_or_equation",
                        bbox=conversion.bbox,
                    )
                    return []
                horizontal_inset = (
                    4.0 if grid_strategy == "pymupdf_text_lattice_semantic_alignment" else 2.0
                )
                bbox = self._inset_bbox(
                    BoundingBox(
                        x0=float(insertion_rectangle.x0),
                        y0=float(insertion_rectangle.y0),
                        x1=float(insertion_rectangle.x1),
                        y1=float(insertion_rectangle.y1),
                    ),
                    horizontal=horizontal_inset,
                    vertical=1.0,
                )
                redaction_bboxes = [
                    self._inset_bbox(
                        cell_bbox,
                        horizontal=horizontal_inset,
                        vertical=1.0,
                    )
                ]
                redaction_fill: tuple[float, float, float] | None = None
                background_metadata: dict[str, Any] = {}
                if scan_overlay:
                    redaction_bboxes = self._scan_cell_text_masks(page, cell_bbox)
                    if source_text and not redaction_bboxes:
                        self._skip_block(
                            report,
                            page_report,
                            block,
                            reason="table_source_text_masks_missing",
                            bbox=conversion.bbox,
                        )
                        return []
                    redaction_fill, background_metadata = self._scan_background_fill(
                        page,
                        cell_bbox,
                    )
                    if redaction_fill is None:
                        self._skip_block(
                            report,
                            page_report,
                            block,
                            reason="table_scan_cell_background_not_uniform_or_light",
                            bbox=conversion.bbox,
                        )
                        return []
                style_hints = self._source_style_hints(
                    page,
                    cell_bbox,
                    infer_alignment=True,
                )
                inferred_alignment = style_hints.get(
                    "text_align",
                    "left" if column_index == 0 else "center",
                )
                if column_index > 0 and len(self._comparison_text(source_text)) <= 24:
                    inferred_alignment = "center"
                style_hints.update(
                    {
                        "line_height": 0.9,
                        "text_align": inferred_alignment,
                    }
                )
                replacements.append(
                    _ReplacementRegion(
                        page_number=block.page_number,
                        block_ids=[f"{block.id}#cell-r{row_index + 1:03d}-c{column_index + 1:03d}"],
                        block_type=BlockType.TABLE,
                        bbox=bbox,
                        translated_text=translated_text,
                        source_text=source_text,
                        style_hints=style_hints,
                        coordinate_metadata=[
                            {
                                **conversion.metadata,
                                "table_grid_detection": grid_strategy,
                                "table_block_id": block.id,
                                "row_index": row_index,
                                "column_index": column_index,
                                "scan_background": background_metadata,
                            }
                        ],
                        redaction_bboxes=redaction_bboxes,
                        redaction_fill=redaction_fill,
                        reconstruction_strategy=(
                            "ocr_table_cell_overlay"
                            if scan_overlay
                            else "embedded_table_cell_replacement"
                        ),
                    )
                )
        if replacements:
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
                    self._skip_block(
                        report,
                        page_report,
                        block,
                        reason="table_atomic_reconstruction_overflow",
                        bbox=conversion.bbox,
                    )
                    self._warning(
                        report,
                        page_number=block.page_number,
                        code="table_atomic_reconstruction_overflow",
                        reason=(
                            "At least one translated table cell did not fit at the minimum scale; "
                            "the entire source table was retained."
                        ),
                    )
                    return []
        return replacements

    def _nonoverlapping_table_row_rectangles(
        self,
        rectangles: list[fitz.Rect],
    ) -> list[fitz.Rect]:
        """Create disjoint insertion boxes while retaining source redaction boxes."""

        normalized = [fitz.Rect(rectangle) for rectangle in rectangles]
        for left, right in zip(normalized, normalized[1:]):
            if left.x1 <= right.x0:
                continue
            boundary = (left.x1 + right.x0) / 2
            if boundary - left.x0 >= 1.0 and right.x1 - boundary >= 1.0:
                left.x1 = boundary
                right.x0 = boundary
        return normalized

    def _parse_table_rows(
        self,
        markup: str,
        *,
        source_hint: str | None = None,
    ) -> list[list[ParsedTableCell]]:
        return parse_table_rows(markup, source_hint=source_hint)

    def _table_markup_is_suspicious(self, rows: list[list[ParsedTableCell]]) -> bool:
        lengths = [len(cell.text) for row in rows for cell in row]
        if not lengths:
            return True
        largest = max(lengths)
        remainder = max(1, sum(lengths) - largest)
        return largest > 400 and largest > remainder * 0.45

    def _stored_table_grid_rows(
        self,
        *,
        page: fitz.Page,
        block: Block,
        table: TableModel | None,
        source_rows: list[list[ParsedTableCell]],
        table_bbox: BoundingBox,
    ) -> tuple[list[list[fitz.Rect]], str]:
        """Return extraction-time cell polygons after validating their topology.

        Marker already predicts complete table-cell polygons, including cells
        in whitespace-separated and partially ruled tables. Persisted geometry
        is therefore the preferred source for new jobs. It remains advisory:
        malformed, incomplete, overlapping, or out-of-table cells are rejected
        and reconstruction falls back to source-PDF geometry recovery.
        """

        if table is None:
            return [], "unavailable"
        header_cells = list(getattr(table, "header_cells", []) or [])
        body_cells = list(table.cells or [])
        model_rows = ([header_cells] if header_cells else []) + body_cells
        if (
            not model_rows
            or len(model_rows) != len(source_rows)
            or any(
                len(model_row) != len(source_row)
                for model_row, source_row in zip(model_rows, source_rows, strict=True)
            )
            or any(cell.bbox is None for row in model_rows for cell in row)
        ):
            return [], "unavailable"

        expected_rows: list[list[tuple[int, int, int, int]]] = []
        occupied_until: dict[int, int] = {}
        for row_index, source_row in enumerate(source_rows):
            expected_row: list[tuple[int, int, int, int]] = []
            next_column = 0
            for source_cell in source_row:
                while any(
                    occupied_until.get(column, 0) > row_index
                    for column in range(next_column, next_column + source_cell.colspan)
                ):
                    next_column += 1
                if row_index + source_cell.rowspan > len(source_rows):
                    return [], "unavailable"
                expected_row.append(
                    (
                        row_index,
                        next_column,
                        source_cell.rowspan,
                        source_cell.colspan,
                    )
                )
                if source_cell.rowspan > 1:
                    for column in range(next_column, next_column + source_cell.colspan):
                        occupied_until[column] = row_index + source_cell.rowspan
                next_column += source_cell.colspan
            expected_rows.append(expected_row)

        for model_row, expected_row in zip(model_rows, expected_rows, strict=True):
            for cell, (row_index, column_index, rowspan, colspan) in zip(
                model_row,
                expected_row,
                strict=True,
            ):
                if (
                    cell.row_index != row_index
                    or cell.column_index != column_index
                    or cell.rowspan != rowspan
                    or cell.colspan != colspan
                ):
                    return [], "unavailable"

        coordinate_metadata = dict(block.metadata)
        stored_space = table.debug.get("cell_coordinate_space")
        if isinstance(stored_space, dict):
            coordinate_metadata["coordinate_space"] = stored_space
            width = stored_space.get("width")
            height = stored_space.get("height")
            if width is not None:
                coordinate_metadata["source_page_width"] = width
                coordinate_metadata["marker_page_width"] = width
            if height is not None:
                coordinate_metadata["source_page_height"] = height
                coordinate_metadata["marker_page_height"] = height

        table_rect = self._fitz_rect(table_bbox)
        converted_rows: list[list[fitz.Rect]] = []
        placements: list[tuple[int, int, int, int, fitz.Rect, bool]] = []
        for model_row, source_row, expected_row in zip(
            model_rows,
            source_rows,
            expected_rows,
            strict=True,
        ):
            converted_row: list[fitz.Rect] = []
            for cell, source_cell, (row_index, column_index, rowspan, colspan) in zip(
                model_row,
                source_row,
                expected_row,
                strict=True,
            ):
                conversion = convert_bbox_to_pdf(
                    cell.bbox,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                    metadata=coordinate_metadata,
                )
                if conversion.bbox is None:
                    return [], "unavailable"
                rectangle = self._fitz_rect(conversion.bbox)
                if (
                    rectangle.width < 1
                    or rectangle.height < 1
                    or self._rect_outside_with_tolerance(
                        rectangle,
                        page.rect,
                        tolerance=0.05,
                    )
                    or self._rect_outside_with_tolerance(rectangle, table_rect, tolerance=4.0)
                ):
                    return [], "unavailable"
                converted_row.append(rectangle)
                placements.append(
                    (
                        row_index,
                        column_index,
                        rowspan,
                        colspan,
                        rectangle,
                        bool(self._comparison_text(source_cell.text)),
                    )
                )
            converted_rows.append(converted_row)

        for index, placement in enumerate(placements):
            row_index, column_index, rowspan, colspan, rectangle, has_text = placement
            for other_placement in placements[index + 1 :]:
                (
                    other_row,
                    other_column,
                    other_rowspan,
                    other_colspan,
                    other,
                    other_has_text,
                ) = other_placement
                intersection = fitz.Rect(rectangle & other)
                smaller_cell_area = min(rectangle.get_area(), other.get_area())
                overlap_ratio = (
                    intersection.get_area() / smaller_cell_area if smaller_cell_area > 0 else 1.0
                )
                if (
                    intersection.get_area() > 0.75
                    and intersection.width > 1.5
                    and intersection.height > 1.5
                    and has_text
                    and other_has_text
                    and overlap_ratio > self.maximum_incidental_table_cell_overlap
                ):
                    return [], "unavailable"
                rectangle_mid_x = (rectangle.x0 + rectangle.x1) / 2
                other_mid_x = (other.x0 + other.x1) / 2
                rectangle_mid_y = (rectangle.y0 + rectangle.y1) / 2
                other_mid_y = (other.y0 + other.y1) / 2
                if (column_index + colspan <= other_column and rectangle_mid_x >= other_mid_x) or (
                    other_column + other_colspan <= column_index and other_mid_x >= rectangle_mid_x
                ):
                    return [], "unavailable"
                if (row_index + rowspan <= other_row and rectangle_mid_y >= other_mid_y) or (
                    other_row + other_rowspan <= row_index and other_mid_y >= rectangle_mid_y
                ):
                    return [], "unavailable"

        try:
            page_words = page.get_text("words", sort=False)
        except Exception:
            page_words = []
        expanded_rows = [
            [
                self._expand_table_cell_to_pdf_words(
                    page,
                    rectangle,
                    table_rect,
                    source_cell.text,
                    words=page_words,
                )
                if self._comparison_text(source_cell.text)
                else rectangle
                for source_cell, rectangle in zip(source_row, converted_row, strict=True)
            ]
            for source_row, converted_row in zip(source_rows, converted_rows, strict=True)
        ]
        return expanded_rows, "marker_table_cell_polygons"

    def _expand_table_cell_to_pdf_words(
        self,
        page: fitz.Page,
        rectangle: fitz.Rect,
        table_rect: fitz.Rect,
        source_text: str = "",
        *,
        words: list[tuple[Any, ...]] | None = None,
    ) -> fitz.Rect:
        """Include complete PDF words whose centres fall inside a predicted cell.

        Marker polygons may end part-way through a glyph or word. Expanding to
        the source PDF's complete word boxes prevents both clipped validation
        text and untranslated glyph fragments after redaction.
        """

        expanded = fitz.Rect(rectangle)
        if words is None:
            try:
                words = page.get_text("words", sort=False)
            except Exception:
                return expanded
        for word in words:
            if len(word) < 5:
                continue
            word_rect = fitz.Rect(float(word[0]), float(word[1]), float(word[2]), float(word[3]))
            center_x = (word_rect.x0 + word_rect.x1) / 2
            center_y = (word_rect.y0 + word_rect.y1) / 2
            if (
                rectangle.x0 <= center_x <= rectangle.x1
                and rectangle.y0 <= center_y <= rectangle.y1
            ):
                expanded.include_rect(word_rect)
        matching_rect = self._nearby_exact_table_cell_text_rect(
            words,
            rectangle=rectangle,
            table_rect=table_rect,
            source_text=source_text,
        )
        if matching_rect is not None:
            expanded.include_rect(matching_rect)
        return fitz.Rect(expanded & table_rect)

    def _nearby_exact_table_cell_text_rect(
        self,
        words: list[tuple[Any, ...]],
        *,
        rectangle: fitz.Rect,
        table_rect: fitz.Rect,
        source_text: str,
    ) -> fitz.Rect | None:
        """Find an exact source-text word run close to a predicted cell.

        Empty placeholder columns in a table model can shift the following
        polygon slightly. Only an exact normalized match on one PDF line is
        accepted, bounded to a local horizontal search area and the predicted
        vertical cell band.
        """

        expected = self._comparison_text(source_text)
        if not expected:
            return None
        horizontal_margin = max(12.0, min(36.0, rectangle.width * 0.65))
        search_rect = fitz.Rect(
            max(table_rect.x0, rectangle.x0 - horizontal_margin),
            rectangle.y0 - 1.0,
            min(table_rect.x1, rectangle.x1 + horizontal_margin),
            rectangle.y1 + 1.0,
        )
        lines: dict[tuple[int, int], list[tuple[int, fitz.Rect, str]]] = {}
        for word in words:
            if len(word) < 8:
                continue
            word_rect = fitz.Rect(float(word[0]), float(word[1]), float(word[2]), float(word[3]))
            center_x = (word_rect.x0 + word_rect.x1) / 2
            center_y = (word_rect.y0 + word_rect.y1) / 2
            if not (
                search_rect.x0 <= center_x <= search_rect.x1
                and search_rect.y0 <= center_y <= search_rect.y1
            ):
                continue
            lines.setdefault((int(word[5]), int(word[6])), []).append(
                (int(word[7]), word_rect, str(word[4]))
            )

        matches: list[fitz.Rect] = []
        for line in lines.values():
            ordered = sorted(line)
            for start in range(len(ordered)):
                candidate = ""
                candidate_rect: fitz.Rect | None = None
                for _word_index, word_rect, text in ordered[start:]:
                    candidate += text
                    candidate_rect = (
                        fitz.Rect(word_rect)
                        if candidate_rect is None
                        else fitz.Rect(candidate_rect | word_rect)
                    )
                    normalized = self._comparison_text(candidate)
                    if normalized == expected and candidate_rect is not None:
                        matches.append(candidate_rect)
                        break
                    if len(normalized) > len(expected):
                        break
        if not matches:
            return None
        rectangle_center = (rectangle.x0 + rectangle.x1) / 2
        return min(
            matches,
            key=lambda match: abs(((match.x0 + match.x1) / 2) - rectangle_center),
        )

    def _rect_outside_with_tolerance(
        self,
        rectangle: fitz.Rect,
        container: fitz.Rect,
        *,
        tolerance: float,
    ) -> bool:
        return bool(
            rectangle.x0 < container.x0 - tolerance
            or rectangle.y0 < container.y0 - tolerance
            or rectangle.x1 > container.x1 + tolerance
            or rectangle.y1 > container.y1 + tolerance
        )

    def _table_grid_rows(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
        *,
        source_rows: list[list[ParsedTableCell]],
        expected_rows: int,
        expected_columns: int,
    ) -> tuple[list[list[fitz.Rect]], str]:
        table_rect = self._fitz_rect(bbox)
        search_rect = (
            fitz.Rect(
                table_rect.x0 - 3.0,
                table_rect.y0 - 3.0,
                table_rect.x1 + 3.0,
                table_rect.y1 + 3.0,
            )
            & page.rect
        )
        try:
            finder = page.find_tables(clip=search_rect, strategy="lines_strict")
            candidates = []
            for table in finder.tables:
                candidate_rect = fitz.Rect(table.bbox)
                overlap = fitz.Rect(candidate_rect & table_rect).get_area()
                union_area = candidate_rect.get_area() + table_rect.get_area() - overlap
                candidate_rows = [
                    [fitz.Rect(cell) for cell in row.cells if cell is not None]
                    for row in table.rows
                ]
                alignment = self._align_table_rows(
                    page,
                    source_rows,
                    source_rows,
                    candidate_rows,
                )
                if (
                    not candidate_rows
                    or int(table.col_count) != expected_columns
                    or len(candidate_rows) > expected_rows
                    or overlap / max(1.0, union_area) < 0.65
                    or alignment is None
                    or not self._table_alignment_has_full_cell_coverage(page, alignment)
                ):
                    continue
                candidates.append((overlap, candidate_rows))
            if candidates:
                _overlap, detected_rows = max(candidates, key=lambda item: item[0])
                return detected_rows, "pymupdf_find_tables_lines_strict"
        except Exception as exc:
            logger.debug("PyMuPDF table grid detection failed: %s", exc)

        rectangles: dict[tuple[float, float, float, float], fitz.Rect] = {}
        verticals: list[float] = []
        horizontals: list[float] = []
        try:
            drawings = page.get_drawings()
        except Exception:
            return [], "unavailable"
        for drawing in drawings:
            for item in drawing.get("items", []):
                if not item:
                    continue
                if item[0] == "l":
                    start = fitz.Point(item[1])
                    end = fitz.Point(item[2])
                    if (
                        abs(start.x - end.x) <= 1.2
                        and min(start.y, end.y) <= table_rect.y0 + table_rect.height * 0.5
                        and max(start.y, end.y) >= table_rect.y1 - table_rect.height * 0.5
                        and table_rect.x0 - 3 <= start.x <= table_rect.x1 + 3
                    ):
                        verticals.append(float((start.x + end.x) / 2))
                    if (
                        abs(start.y - end.y) <= 1.2
                        and min(start.x, end.x) <= table_rect.x0 + table_rect.width * 0.5
                        and max(start.x, end.x) >= table_rect.x1 - table_rect.width * 0.5
                        and table_rect.y0 - 3 <= start.y <= table_rect.y1 + 3
                    ):
                        horizontals.append(float((start.y + end.y) / 2))
                    continue
                if item[0] != "re":
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
                key = (
                    round(float(rectangle.x0), 2),
                    round(float(rectangle.y0), 2),
                    round(float(rectangle.x1), 2),
                    round(float(rectangle.y1), 2),
                )
                rectangles[key] = rectangle

        detected_rows = []
        for rectangle in sorted(
            rectangles.values(),
            key=lambda value: (round(value.y0, 1), value.x0),
        ):
            if (
                not detected_rows
                or abs(detected_rows[-1][0].y0 - rectangle.y0) > 1.0
                or abs(detected_rows[-1][0].y1 - rectangle.y1) > 1.0
            ):
                detected_rows.append([rectangle])
            else:
                detected_rows[-1].append(rectangle)
        for row in detected_rows:
            row.sort(key=lambda value: value.x0)
        rectangle_alignment = self._align_table_rows(
            page,
            source_rows,
            source_rows,
            detected_rows,
        )
        if (
            detected_rows
            and len(detected_rows) <= expected_rows
            and rectangle_alignment is not None
            and self._table_alignment_has_full_cell_coverage(page, rectangle_alignment)
        ):
            return detected_rows, "pdf_vector_cell_rectangles"

        xs = self._cluster_coordinates(verticals)
        ys = self._cluster_coordinates(horizontals)
        if len(xs) == expected_columns + 1 and len(ys) == expected_rows + 1:
            line_rows = [
                [
                    fitz.Rect(xs[column], ys[row], xs[column + 1], ys[row + 1])
                    for column in range(expected_columns)
                ]
                for row in range(expected_rows)
            ]
            line_alignment = self._align_table_rows(
                page,
                source_rows,
                source_rows,
                line_rows,
            )
            if line_alignment is not None and self._table_alignment_has_full_cell_coverage(
                page,
                line_alignment,
            ):
                return line_rows, "pdf_vector_line_grid"

        semantic_rows = self._semantic_text_table_grid_rows(
            page=page,
            table_rect=table_rect,
            search_rect=search_rect,
            source_rows=source_rows,
        )
        if semantic_rows:
            return semantic_rows, "pymupdf_text_lattice_semantic_alignment"
        return [], "unavailable"

    def _semantic_text_table_grid_rows(
        self,
        *,
        page: fitz.Page,
        table_rect: fitz.Rect,
        search_rect: fitz.Rect,
        source_rows: list[list[ParsedTableCell]],
    ) -> list[list[fitz.Rect]]:
        """Coarsen a physical text lattice to the known semantic table shape.

        Borderless and partially ruled tables often contain excellent PDF text
        geometry but no closed vector cells. PyMuPDF deliberately reports the
        physical baselines and whitespace bands in that case, which can be much
        finer than the logical HTML table. We treat those bands only as column
        boundary candidates, then accept a layout solely when the source cell
        text uniquely aligns to every PDF word in monotonic row order.
        """

        if not source_rows or not source_rows[0]:
            return []
        logical_columns = max(
            sum(max(1, int(cell.colspan or 1)) for cell in row) for row in source_rows
        )
        if logical_columns <= 0 or any(
            sum(max(1, int(cell.colspan or 1)) for cell in row) != logical_columns
            or any(int(cell.rowspan or 1) != 1 for cell in row)
            for row in source_rows
        ):
            return []
        boundary_candidates = self._horizontal_rule_column_candidates(
            page,
            table_rect,
            logical_columns,
        )
        try:
            finder = page.find_tables(clip=search_rect, strategy="text")
        except Exception as exc:
            logger.debug("PyMuPDF text-table lattice detection failed: %s", exc)
            finder = None

        if finder is not None:
            for table in finder.tables:
                candidate_rect = fitz.Rect(table.bbox)
                intersection = fitz.Rect(candidate_rect & table_rect).get_area()
                if (
                    intersection / max(1.0, min(candidate_rect.get_area(), table_rect.get_area()))
                    < 0.55
                ):
                    continue
                physical_columns = int(table.col_count)
                if physical_columns < logical_columns or physical_columns > min(
                    12, logical_columns + 6
                ):
                    continue
                x_values = [
                    float(value)
                    for row in table.rows
                    for cell in row.cells
                    if cell is not None
                    for value in (cell[0], cell[2])
                ]
                physical_edges = self._cluster_coordinates(x_values, tolerance=1.0)
                if len(physical_edges) != physical_columns + 1:
                    continue
                internal_edges = physical_edges[1:-1]
                if len(internal_edges) < logical_columns - 1:
                    continue
                candidate_count = math.comb(len(internal_edges), logical_columns - 1)
                if candidate_count > 256:
                    continue
                for selected in combinations(internal_edges, logical_columns - 1):
                    boundary_candidates.append(
                        (float(table_rect.x0), *selected, float(table_rect.x1))
                    )

        unique_boundaries: dict[tuple[float, ...], tuple[float, ...]] = {}
        for boundaries in boundary_candidates:
            if len(boundaries) != logical_columns + 1 or any(
                right - left < 3.0 for left, right in zip(boundaries, boundaries[1:])
            ):
                continue
            key = tuple(round(value, 2) for value in boundaries)
            unique_boundaries[key] = tuple(float(value) for value in boundaries)

        candidates: dict[tuple[float, ...], _SemanticTableGrid] = {}
        for boundaries in unique_boundaries.values():
            candidate = self._semantic_grid_candidate(
                page=page,
                table_rect=table_rect,
                source_rows=source_rows,
                column_edges=boundaries,
            )
            if candidate is not None:
                current = candidates.get(candidate.signature)
                if current is None or candidate.score > current.score:
                    candidates[candidate.signature] = candidate
        ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        if not ranked:
            return []
        assignments: dict[tuple[str, ...], _SemanticTableGrid] = {}
        for candidate in ranked:
            assignments.setdefault(candidate.assignment_signature, candidate)
        ranked_assignments = sorted(
            assignments.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        if len(ranked_assignments) > 1 and self._semantic_scores_are_ambiguous(
            ranked_assignments[0].score,
            ranked_assignments[1].score,
        ):
            logger.debug("Rejected ambiguous semantic table geometry with near-tied layouts")
            return []
        return [list(row) for row in ranked_assignments[0].rows]

    def _horizontal_rule_column_candidates(
        self,
        page: fitz.Page,
        table_rect: fitz.Rect,
        logical_columns: int,
    ) -> list[tuple[float, ...]]:
        """Recover column separators from per-column horizontal rule segments."""

        y_groups: list[tuple[float, list[float]]] = []
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        for drawing in drawings:
            for item in drawing.get("items", []):
                if not item or item[0] != "l":
                    continue
                start = fitz.Point(item[1])
                end = fitz.Point(item[2])
                if (
                    abs(start.y - end.y) > 1.2
                    or max(start.x, end.x) < table_rect.x0 - 4.0
                    or min(start.x, end.x) > table_rect.x1 + 4.0
                    or start.y < table_rect.y0 - 4.0
                    or start.y > table_rect.y1 + 4.0
                    or abs(start.x - end.x) < 3.0
                ):
                    continue
                y = float((start.y + end.y) / 2)
                group = next((entry for entry in y_groups if abs(entry[0] - y) <= 1.2), None)
                if group is None:
                    group = (y, [])
                    y_groups.append(group)
                group[1].extend([float(start.x), float(end.x)])

        candidates: list[tuple[float, ...]] = []
        for _y, endpoints in y_groups:
            edges = self._cluster_coordinates(endpoints, tolerance=1.2)
            if len(edges) != logical_columns + 1:
                continue
            if edges[-1] - edges[0] < table_rect.width * 0.65:
                continue
            candidates.append((float(table_rect.x0), *edges[1:-1], float(table_rect.x1)))
        return candidates

    def _semantic_grid_candidate(
        self,
        *,
        page: fitz.Page,
        table_rect: fitz.Rect,
        source_rows: list[list[ParsedTableCell]],
        column_edges: tuple[float, ...],
    ) -> _SemanticTableGrid | None:
        physical_lines = self._physical_table_lines(page, table_rect, column_edges)
        if not physical_lines or len(physical_lines) < len(source_rows):
            return None
        logical_columns = len(column_edges) - 1

        @lru_cache(maxsize=None)
        def combined_cells(start: int, end: int) -> tuple[str, ...]:
            return tuple(
                " ".join(
                    physical_lines[index].cells[column]
                    for index in range(start, end)
                    if physical_lines[index].cells[column]
                )
                for column in range(logical_columns)
            )

        @lru_cache(maxsize=None)
        def solve(
            logical_index: int,
            physical_index: int,
        ) -> tuple[tuple[float, tuple[tuple[int, int], ...]], ...]:
            if logical_index == len(source_rows) and physical_index == len(physical_lines):
                return ((0.0, ()),)
            if logical_index >= len(source_rows) or physical_index >= len(physical_lines):
                return ()
            remaining_rows = len(source_rows) - logical_index - 1
            max_take = len(physical_lines) - physical_index - remaining_rows
            candidates: list[tuple[float, tuple[tuple[int, int], ...]]] = []
            for take in range(1, max_take + 1):
                end = physical_index + take
                actual_cells = self._collapse_physical_cells_for_colspans(
                    source_rows[logical_index],
                    combined_cells(physical_index, end),
                )
                if actual_cells is None:
                    continue
                score = self._semantic_table_row_similarity(
                    source_rows[logical_index],
                    actual_cells,
                )
                if score < 0.68:
                    continue
                for tail_score, tail_ranges in solve(logical_index + 1, end):
                    candidates.append((score + tail_score, ((physical_index, end), *tail_ranges)))
            candidates.sort(key=lambda item: item[0], reverse=True)
            unique: dict[tuple[tuple[int, int], ...], float] = {}
            for score, ranges in candidates:
                unique.setdefault(ranges, score)
                if len(unique) >= 2:
                    break
            return tuple((score, ranges) for ranges, score in unique.items())

        solutions = solve(0, 0)
        if not solutions:
            return None
        best_score, row_ranges = solutions[0]
        average_score = best_score / len(source_rows)
        if average_score < 0.68:
            return None
        if len(solutions) > 1:
            second_average = solutions[1][0] / len(source_rows)
            if self._semantic_scores_are_ambiguous(average_score, second_average):
                return None

        horizontal_rules = self._horizontal_table_rule_positions(page, table_rect)
        first_text_top = min(
            physical_lines[index].y0 for index in range(row_ranges[0][0], row_ranges[0][1])
        )
        top_rules = [
            value
            for value in horizontal_rules
            if table_rect.y0 - 1.0 <= value <= first_text_top + 2.0
        ]
        row_edges = [
            float(
                min(top_rules, key=lambda value: abs(value - first_text_top))
                if top_rules
                else table_rect.y0
            )
        ]
        for current, following in zip(row_ranges, row_ranges[1:]):
            current_bottom = max(
                physical_lines[index].y1 for index in range(current[0], current[1])
            )
            following_top = min(
                physical_lines[index].y0 for index in range(following[0], following[1])
            )
            midpoint = (current_bottom + following_top) / 2
            rules_between = [
                value
                for value in horizontal_rules
                if current_bottom - 2.0 <= value <= following_top + 2.0
            ]
            boundary = (
                min(rules_between, key=lambda value: abs(value - midpoint))
                if rules_between
                else midpoint
            )
            if boundary <= row_edges[-1] or boundary >= table_rect.y1:
                return None
            row_edges.append(float(boundary))
        last_text_bottom = max(
            physical_lines[index].y1 for index in range(row_ranges[-1][0], row_ranges[-1][1])
        )
        bottom_rules = [
            value
            for value in horizontal_rules
            if last_text_bottom - 2.0 <= value <= table_rect.y1 + 1.0
        ]
        bottom = (
            min(bottom_rules, key=lambda value: abs(value - last_text_bottom))
            if bottom_rules
            else table_rect.y1
        )
        if bottom <= row_edges[-1]:
            return None
        row_edges.append(float(bottom))

        rows: tuple[tuple[fitz.Rect, ...], ...] = tuple(
            self._semantic_row_rectangles(
                source_row,
                column_edges=column_edges,
                y0=row_edges[row_index],
                y1=row_edges[row_index + 1],
            )
            for row_index, source_row in enumerate(source_rows)
        )
        if any(
            len(row) != len(source_row) for row, source_row in zip(rows, source_rows, strict=True)
        ):
            return None
        signature = tuple(
            round(value, 2)
            for value in (
                *column_edges,
                *row_edges,
            )
        )
        assignment_signature = tuple(
            self._comparison_text(cell)
            for source_row, (start, end) in zip(source_rows, row_ranges, strict=True)
            for cell in (
                self._collapse_physical_cells_for_colspans(
                    source_row,
                    combined_cells(start, end),
                )
                or ()
            )
        )
        return _SemanticTableGrid(
            rows=rows,
            score=average_score,
            signature=signature,
            assignment_signature=assignment_signature,
        )

    def _collapse_physical_cells_for_colspans(
        self,
        source_row: list[ParsedTableCell],
        physical_cells: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        collapsed: list[str] = []
        column_index = 0
        for source_cell in source_row:
            colspan = max(1, int(source_cell.colspan or 1))
            end = column_index + colspan
            if end > len(physical_cells):
                return None
            collapsed.append(
                " ".join(cell for cell in physical_cells[column_index:end] if cell).strip()
            )
            column_index = end
        if column_index != len(physical_cells):
            return None
        return tuple(collapsed)

    def _semantic_row_rectangles(
        self,
        source_row: list[ParsedTableCell],
        *,
        column_edges: tuple[float, ...],
        y0: float,
        y1: float,
    ) -> tuple[fitz.Rect, ...]:
        rectangles: list[fitz.Rect] = []
        column_index = 0
        for source_cell in source_row:
            colspan = max(1, int(source_cell.colspan or 1))
            end = column_index + colspan
            if end >= len(column_edges):
                return ()
            rectangles.append(
                fitz.Rect(
                    column_edges[column_index],
                    y0,
                    column_edges[end],
                    y1,
                )
            )
            column_index = end
        if column_index != len(column_edges) - 1:
            return ()
        return tuple(rectangles)

    def _semantic_scores_are_ambiguous(self, best: float, second: float) -> bool:
        difference = best - second
        if difference <= 1e-6:
            return True
        # An exact or near-exact source-text alignment is itself decisive. For
        # noisier text, require a wider margin between competing partitions.
        return best < 0.985 and difference < 0.025

    def _horizontal_table_rule_positions(
        self,
        page: fitz.Page,
        table_rect: fitz.Rect,
    ) -> list[float]:
        groups: list[dict[str, Any]] = []
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        for drawing in drawings:
            for item in drawing.get("items", []):
                if not item or item[0] != "l":
                    continue
                start = fitz.Point(item[1])
                end = fitz.Point(item[2])
                if (
                    abs(start.y - end.y) > 1.2
                    or start.y < table_rect.y0 - 4.0
                    or start.y > table_rect.y1 + 4.0
                ):
                    continue
                clipped_start = max(table_rect.x0, min(start.x, end.x))
                clipped_end = min(table_rect.x1, max(start.x, end.x))
                if clipped_end - clipped_start < 3.0:
                    continue
                y = float((start.y + end.y) / 2)
                group = next(
                    (candidate for candidate in groups if abs(float(candidate["y"]) - y) <= 1.2),
                    None,
                )
                if group is None:
                    group = {"y": y, "segments": []}
                    groups.append(group)
                group["segments"].append((float(clipped_start), float(clipped_end)))

        positions: list[float] = []
        for group in groups:
            segments = sorted(group["segments"])
            merged: list[list[float]] = []
            for start, end in segments:
                if not merged or start > merged[-1][1] + 1.5:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            coverage = sum(end - start for start, end in merged)
            if coverage >= table_rect.width * 0.5:
                positions.append(float(group["y"]))
        return sorted(positions)

    def _physical_table_lines(
        self,
        page: fitz.Page,
        table_rect: fitz.Rect,
        column_edges: tuple[float, ...],
    ) -> tuple[_PhysicalTableLine, ...]:
        try:
            words = page.get_text("words", clip=table_rect, sort=True)
        except Exception:
            return ()
        words = [word for word in words if len(word) >= 5 and self._comparison_text(str(word[4]))]
        if not words:
            return ()

        heights = sorted(float(word[3]) - float(word[1]) for word in words)
        median_height = heights[len(heights) // 2]
        y_tolerance = min(3.5, max(1.0, median_height * 0.25))
        grouped: list[list[Any]] = []
        for word in sorted(words, key=lambda value: (float(value[1]), float(value[0]))):
            y0 = float(word[1])
            if (
                not grouped
                or abs(y0 - sum(float(item[1]) for item in grouped[-1]) / len(grouped[-1]))
                > y_tolerance
            ):
                grouped.append([word])
            else:
                grouped[-1].append(word)

        lines: list[_PhysicalTableLine] = []
        for group in grouped:
            cell_words: list[list[Any]] = [[] for _ in range(len(column_edges) - 1)]
            for word in sorted(group, key=lambda value: float(value[0])):
                center = (float(word[0]) + float(word[2])) / 2
                column = next(
                    (
                        index
                        for index, (left, right) in enumerate(zip(column_edges, column_edges[1:]))
                        if left <= center < right
                        or (index == len(column_edges) - 2 and center == right)
                    ),
                    None,
                )
                if column is None:
                    return ()
                cell_words[column].append(word)
            lines.append(
                _PhysicalTableLine(
                    y0=min(float(word[1]) for word in group),
                    y1=max(float(word[3]) for word in group),
                    cells=tuple(
                        " ".join(str(word[4]) for word in words_in_cell)
                        for words_in_cell in cell_words
                    ),
                )
            )
        return tuple(lines)

    def _semantic_table_row_similarity(
        self,
        source_row: list[ParsedTableCell],
        actual_cells: tuple[str, ...],
    ) -> float:
        weighted_score = 0.0
        total_weight = 0.0
        nonempty_scores: list[float] = []
        for source_cell, actual_text in zip(source_row, actual_cells, strict=True):
            expected = self._comparison_text(source_cell.text)
            actual = self._comparison_text(actual_text)
            weight = float(max(2, len(expected), len(actual)))
            total_weight += weight
            score = self._normalized_text_similarity(expected, actual)
            if expected or actual:
                nonempty_scores.append(score)
            weighted_score += score * weight
        if any(score < 0.42 for score in nonempty_scores):
            return 0.0
        return weighted_score / max(1.0, total_weight)

    def _normalized_text_similarity(self, expected: str, actual: str) -> float:
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0
        score = SequenceMatcher(None, expected, actual).ratio()
        if expected in actual or actual in expected:
            score = max(score, min(len(expected), len(actual)) / max(len(expected), len(actual)))
        return score

    def _cluster_coordinates(self, values: list[float], tolerance: float = 1.2) -> list[float]:
        clusters: list[list[float]] = []
        for value in sorted(values):
            if not clusters or abs(value - sum(clusters[-1]) / len(clusters[-1])) > tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        return [sum(cluster) / len(cluster) for cluster in clusters]

    def _match_hidden_ocr_lines(
        self,
        page: fitz.Page,
        expected_text: str,
        *,
        preferred_bbox: BoundingBox,
        unavailable_line_keys: set[tuple[int, int]],
        text_sequences: list[tuple[_HiddenOCRLine, ...]],
    ) -> tuple[_HiddenOCRMatch | None, dict[str, Any]]:
        """Align OCR source text to its actual PDF line geometry.

        Surya provides the document structure and a useful initial position, but
        a missed or merged layout region can shift later Qwen-to-Surya pairings.
        The hidden OCR layer is therefore used as a second, independent geometry
        check. Only contiguous lines are considered. Adjacent native PDF text
        blocks may be joined only when their
        geometry is continuous, which supports split headings and paragraphs
        without jumping across columns or through an intervening figure.
        """

        expected = self._hidden_ocr_alignment_text(expected_text)
        if len(expected) < 5:
            return None, {
                "reason": "hidden_ocr_source_text_too_short_for_unique_alignment",
                "expected_characters": len(expected),
            }
        if not text_sequences:
            return None, {"reason": "hidden_ocr_line_geometry_unavailable"}

        minimum_characters = max(3, int(len(expected) * 0.35))
        maximum_characters = max(24, int(len(expected) * 1.8) + 8)
        preferred_center = (
            (preferred_bbox.x0 + preferred_bbox.x1) / 2,
            (preferred_bbox.y0 + preferred_bbox.y1) / 2,
        )
        candidates: list[
            tuple[
                float,
                int,
                float,
                tuple[_HiddenOCRLine, ...],
                str,
                float,
                float,
                float,
            ]
        ] = []
        for lines in text_sequences:
            for start in range(len(lines)):
                selected: list[_HiddenOCRLine] = []
                for line in lines[start:]:
                    if line.key in unavailable_line_keys:
                        break
                    selected.append(line)
                    actual_text = " ".join(item.text for item in selected)
                    actual = self._hidden_ocr_alignment_text(actual_text)
                    if len(actual) > maximum_characters:
                        break
                    if len(actual) < minimum_characters:
                        continue
                    score = self._normalized_text_similarity(expected, actual)
                    coverage = min(len(expected), len(actual)) / max(len(expected), len(actual))
                    edge_length = min(36, len(expected), len(actual))
                    prefix_score = SequenceMatcher(
                        None,
                        expected[:edge_length],
                        actual[:edge_length],
                    ).ratio()
                    suffix_score = SequenceMatcher(
                        None,
                        expected[-edge_length:],
                        actual[-edge_length:],
                    ).ratio()
                    candidate_bbox = self._union_bbox([item.bbox for item in selected])
                    candidate_center = (
                        (candidate_bbox.x0 + candidate_bbox.x1) / 2,
                        (candidate_bbox.y0 + candidate_bbox.y1) / 2,
                    )
                    distance = math.hypot(
                        candidate_center[0] - preferred_center[0],
                        candidate_center[1] - preferred_center[1],
                    )
                    candidates.append(
                        (
                            score,
                            abs(len(actual) - len(expected)),
                            distance,
                            tuple(selected),
                            actual_text,
                            coverage,
                            prefix_score,
                            suffix_score,
                        )
                    )

        if not candidates:
            return None, {
                "reason": "hidden_ocr_text_alignment_no_candidate",
                "expected_characters": len(expected),
            }
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        best = candidates[0]
        best_keys = {line.key for line in best[3]}
        competing_score = max(
            (
                candidate[0]
                for candidate in candidates[1:]
                if best_keys.isdisjoint(line.key for line in candidate[3])
            ),
            default=0.0,
        )
        minimum_score = 0.96 if len(expected) < 8 else 0.9 if len(expected) < 20 else 0.84
        if best[0] < minimum_score or best[5] < 0.88 or best[6] < 0.55 or best[7] < 0.55:
            return None, {
                "reason": "hidden_ocr_text_alignment_low_confidence",
                "score": round(best[0], 6),
                "competing_score": round(competing_score, 6),
                "minimum_score": minimum_score,
                "length_coverage": round(best[5], 6),
                "prefix_score": round(best[6], 6),
                "suffix_score": round(best[7], 6),
                "expected_characters": len(expected),
            }
        minimum_margin = 0.012 if best[0] >= 0.985 else 0.035
        if competing_score and best[0] - competing_score < minimum_margin:
            return None, {
                "reason": "hidden_ocr_text_alignment_ambiguous",
                "score": round(best[0], 6),
                "competing_score": round(competing_score, 6),
                "minimum_margin": minimum_margin,
                "expected_characters": len(expected),
            }

        matched_lines = best[3]
        return (
            _HiddenOCRMatch(
                lines=matched_lines,
                bbox=self._union_bbox([line.bbox for line in matched_lines]),
                text=best[4],
                score=float(best[0]),
                competing_score=float(competing_score),
                minimum_score=minimum_score,
            ),
            {
                "reason": "matched",
                "score": round(best[0], 6),
                "competing_score": round(competing_score, 6),
            },
        )

    def _hidden_ocr_alignment_text(self, text: str) -> str:
        """Normalize a narrow OCR confusion in structural Roman-numeral labels."""

        compared = self._comparison_text(text)
        structural_label = re.fullmatch(
            r"(tabla|table|figura|figure)([ivxlcdm1]+)",
            compared,
        )
        if structural_label is None:
            return compared
        numeral = structural_label.group(2).replace("1", "i")
        return f"{structural_label.group(1)}{numeral}"

    def _hidden_ocr_text_blocks(
        self,
        page: fitz.Page,
    ) -> list[tuple[_HiddenOCRLine, ...]]:
        try:
            payload = page.get_text("dict")
        except Exception:
            return []
        text_blocks: list[tuple[_HiddenOCRLine, ...]] = []
        for block_index, text_block in enumerate(payload.get("blocks", [])):
            lines: list[_HiddenOCRLine] = []
            for line_index, line in enumerate(text_block.get("lines", [])):
                spans = [
                    span
                    for span in line.get("spans", [])
                    if self._comparison_text(str(span.get("text", "")))
                ]
                if not spans:
                    continue
                text = "".join(str(span.get("text", "")) for span in spans).strip()
                rectangles: list[fitz.Rect] = []
                for span in spans:
                    try:
                        rectangle = fitz.Rect(span["bbox"])
                    except Exception:
                        continue
                    if rectangle.is_valid and not rectangle.is_empty:
                        rectangles.append(rectangle)
                if not rectangles:
                    continue
                rectangle = fitz.Rect(rectangles[0])
                for span_rectangle in rectangles[1:]:
                    rectangle |= span_rectangle
                lines.append(
                    _HiddenOCRLine(
                        block_index=block_index,
                        line_index=line_index,
                        bbox=BoundingBox(
                            x0=float(rectangle.x0),
                            y0=float(rectangle.y0),
                            x1=float(rectangle.x1),
                            y1=float(rectangle.y1),
                        ),
                        text=text,
                    )
                )
            if lines:
                text_blocks.append(tuple(lines))
        return text_blocks

    def _hidden_ocr_text_sequences(
        self,
        text_blocks: list[tuple[_HiddenOCRLine, ...]],
    ) -> list[tuple[_HiddenOCRLine, ...]]:
        sequences = list(text_blocks)
        all_lines = [line for block in text_blocks for line in block]
        if len(all_lines) < 2:
            return sequences

        # Native OCR text blocks are not reliable paragraph containers. A
        # continuation can be placed near the end of a later block after lines
        # from the opposite column. Build additional reading lanes from line
        # geometry so that the matcher can take that continuation without also
        # consuming the unrelated lines.
        candidate_edges: list[
            tuple[tuple[int, float, float, float], _HiddenOCRLine, _HiddenOCRLine]
        ] = []
        for first in all_lines:
            for second in all_lines:
                if not self._hidden_ocr_lines_are_continuous(first, second):
                    continue
                same_native_sequence = (
                    first.block_index == second.block_index
                    and second.line_index == first.line_index + 1
                )
                first_height = max(1.0, first.bbox.y1 - first.bbox.y0)
                top_advance = second.bbox.y0 - first.bbox.y0
                horizontal_offset = abs(first.bbox.x0 - second.bbox.x0)
                candidate_edges.append(
                    (
                        (
                            0 if same_native_sequence else 1,
                            top_advance / first_height,
                            horizontal_offset,
                            second.bbox.x0,
                        ),
                        first,
                        second,
                    )
                )

        successors: dict[tuple[int, int], _HiddenOCRLine] = {}
        predecessors: dict[tuple[int, int], _HiddenOCRLine] = {}
        for _score, first, second in sorted(candidate_edges, key=lambda item: item[0]):
            if first.key in successors or second.key in predecessors:
                continue
            successors[first.key] = second
            predecessors[second.key] = first

        known_sequences = {tuple(line.key for line in sequence) for sequence in sequences}
        for first in all_lines:
            if first.key in predecessors:
                continue
            lane = [first]
            seen = {first.key}
            while lane[-1].key in successors:
                following = successors[lane[-1].key]
                if following.key in seen:
                    break
                lane.append(following)
                seen.add(following.key)
            signature = tuple(line.key for line in lane)
            if len(lane) > 1 and signature not in known_sequences:
                sequences.append(tuple(lane))
                known_sequences.add(signature)
        return sequences

    def _hidden_ocr_lines_are_continuous(
        self,
        first: _HiddenOCRLine,
        second: _HiddenOCRLine,
    ) -> bool:
        first_height = max(1.0, first.bbox.y1 - first.bbox.y0)
        second_height = max(1.0, second.bbox.y1 - second.bbox.y0)
        top_advance = second.bbox.y0 - first.bbox.y0
        vertical_gap = second.bbox.y0 - first.bbox.y1
        if top_advance < min(2.0, first_height * 0.25):
            return False
        if vertical_gap < -min(first_height, second_height) * 0.25:
            return False
        if vertical_gap > max(18.0, (first_height + second_height) * 1.4):
            return False
        horizontal_overlap = max(
            0.0,
            min(first.bbox.x1, second.bbox.x1) - max(first.bbox.x0, second.bbox.x0),
        )
        narrower_width = max(
            1.0,
            min(first.bbox.x1 - first.bbox.x0, second.bbox.x1 - second.bbox.x0),
        )
        return (
            horizontal_overlap / narrower_width >= 0.5
            or abs(first.bbox.x0 - second.bbox.x0) <= 18.0
        )

    def _hidden_ocr_blocks_are_continuous(
        self,
        first: tuple[_HiddenOCRLine, ...],
        second: tuple[_HiddenOCRLine, ...],
    ) -> bool:
        if not first or not second:
            return False
        return self._hidden_ocr_lines_are_continuous(first[-1], second[0])

    def _scan_match_envelope(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
    ) -> BoundingBox:
        return BoundingBox(
            x0=max(0.0, bbox.x0 - 2.5),
            y0=max(0.0, bbox.y0 - 1.5),
            x1=min(float(page.rect.width), bbox.x1 + 2.5),
            y1=min(float(page.rect.height), bbox.y1 + 2.5),
        )

    def _scan_match_masks(
        self,
        page: fitz.Page,
        lines: tuple[_HiddenOCRLine, ...],
    ) -> list[BoundingBox]:
        masks: list[BoundingBox] = []
        for line in lines:
            masks.append(
                BoundingBox(
                    x0=max(0.0, line.bbox.x0 - 2.5),
                    y0=max(0.0, line.bbox.y0 - 2.8),
                    x1=min(float(page.rect.width), line.bbox.x1 + 2.5),
                    y1=min(float(page.rect.height), line.bbox.y1 + 2.8),
                )
            )
        return masks

    def _scan_match_has_multicolumn_text(
        self,
        page: fitz.Page,
        lines: tuple[_HiddenOCRLine, ...],
    ) -> bool:
        try:
            page_words = page.get_text("words", sort=False)
        except Exception:
            return False

        gaps_by_line: list[list[tuple[float, float]]] = []
        eligible_lines = 0
        for line in lines:
            line_rectangle = self._fitz_rect(line.bbox)
            word_rectangles = sorted(
                (
                    fitz.Rect(float(word[0]), float(word[1]), float(word[2]), float(word[3]))
                    for word in page_words
                    if len(word) >= 5
                    and self._comparison_text(str(word[4]))
                    and self._word_belongs_to_hidden_ocr_line(word, line_rectangle)
                ),
                key=lambda rectangle: rectangle.x0,
            )
            if len(word_rectangles) < 2:
                continue
            eligible_lines += 1
            gap_threshold = max(28.0, (line.bbox.x1 - line.bbox.x0) * 0.18)
            line_gaps: list[tuple[float, float]] = []
            occupied_right = word_rectangles[0].x1
            for rectangle in word_rectangles[1:]:
                gap = rectangle.x0 - occupied_right
                if gap >= gap_threshold:
                    line_gaps.append((occupied_right, rectangle.x0))
                # Clipped OCR glyphs can be nested inside a genuine word. Use
                # the union of occupied intervals rather than treating the
                # nested glyph as the new right edge.
                occupied_right = max(occupied_right, rectangle.x1)
            gaps_by_line.append(line_gaps)

        if eligible_lines < 3:
            return False
        split_lines = sum(bool(gaps) for gaps in gaps_by_line)
        required_lines = max(2, math.ceil(eligible_lines * 0.3))
        if split_lines < required_lines:
            return False

        # A real column division leaves a gutter at a stable horizontal
        # position. Isolated large inter-word gaps (or clipped OCR debris) do
        # not justify flattening the region as a table.
        for candidate_line, candidate_gaps in enumerate(gaps_by_line):
            for candidate_left, candidate_right in candidate_gaps:
                supporting_lines = 1
                for other_line, other_gaps in enumerate(gaps_by_line):
                    if other_line == candidate_line:
                        continue
                    if any(
                        min(candidate_right, other_right) - max(candidate_left, other_left) >= 8.0
                        for other_left, other_right in other_gaps
                    ):
                        supporting_lines += 1
                if supporting_lines >= required_lines:
                    return True
        return False

    def _word_belongs_to_hidden_ocr_line(
        self,
        word: tuple[Any, ...],
        line_rectangle: fitz.Rect,
    ) -> bool:
        try:
            word_rectangle = fitz.Rect(
                float(word[0]),
                float(word[1]),
                float(word[2]),
                float(word[3]),
            )
        except (TypeError, ValueError):
            return False
        horizontal_overlap = max(
            0.0,
            min(word_rectangle.x1, line_rectangle.x1) - max(word_rectangle.x0, line_rectangle.x0),
        )
        vertical_overlap = max(
            0.0,
            min(word_rectangle.y1, line_rectangle.y1) - max(word_rectangle.y0, line_rectangle.y0),
        )
        if horizontal_overlap <= 0.0 or vertical_overlap <= 0.0:
            return False
        overlap_height = max(
            1.0,
            min(word_rectangle.height, line_rectangle.height),
        )
        return bool(vertical_overlap / overlap_height >= 0.6)

    def _translation_script_is_suspicious(
        self,
        source_text: str,
        translated_text: str,
    ) -> bool:
        """Reject obvious non-English script drift in the English-only workflow."""

        def east_asian_count(value: str) -> int:
            return sum(
                any(
                    marker in unicodedata.name(character, "")
                    for marker in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")
                )
                for character in value
            )

        source_letters = sum(character.isalpha() for character in source_text)
        target_letters = sum(character.isalpha() for character in translated_text)
        if source_letters < 4 or target_letters < 1:
            return False
        return (
            east_asian_count(source_text) / source_letters < 0.05
            and east_asian_count(translated_text) / target_letters >= 0.2
        )

    def _source_text_is_probably_english(self, text: str) -> bool:
        words = re.findall(r"[A-Za-z]+", text.casefold())
        if len(words) < 5:
            return False
        english_function_words = {
            "after",
            "and",
            "are",
            "at",
            "before",
            "between",
            "by",
            "during",
            "for",
            "from",
            "has",
            "have",
            "in",
            "into",
            "is",
            "of",
            "on",
            "or",
            "that",
            "the",
            "their",
            "these",
            "this",
            "to",
            "was",
            "were",
            "with",
        }
        hits = sum(word in english_function_words for word in words)
        return hits >= 2 and hits / len(words) >= 0.08

    def _scan_background_fill(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
    ) -> tuple[tuple[float, float, float] | None, dict[str, Any]]:
        rectangle = self._fitz_rect(bbox)
        try:
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(1.5, 1.5),
                clip=rectangle,
                colorspace=fitz.csRGB,
                alpha=False,
            )
        except Exception as exc:
            return None, {"reason": f"render_failed:{exc}"}
        samples = pixmap.samples
        if not samples or len(samples) % 3:
            return None, {"reason": "empty_render"}
        pixels = [tuple(samples[index : index + 3]) for index in range(0, len(samples), 3)]
        total = len(pixels)
        pixel_luminances = [
            round(0.2126 * red + 0.7152 * green + 0.0722 * blue) for red, green, blue in pixels
        ]
        luminances = sorted(pixel_luminances)
        median = luminances[len(luminances) // 2]
        light_ratio = sum(value >= 225 for value in luminances) / total

        # Quantisation absorbs ordinary scan noise while retaining background
        # colour. A single table-wide grayscale fill would visibly stripe a
        # shaded or coloured header, so every changed cell is sampled itself.
        background_candidates = [
            pixel
            for pixel, luminance in zip(pixels, pixel_luminances, strict=True)
            if luminance >= 200
        ]
        background_candidate_ratio = len(background_candidates) / total
        histogram: dict[tuple[int, int, int], int] = {}
        for red, green, blue in background_candidates:
            key = (
                min(255, int(round(red / 8.0) * 8)),
                min(255, int(round(green / 8.0) * 8)),
                min(255, int(round(blue / 8.0) * 8)),
            )
            histogram[key] = histogram.get(key, 0) + 1
        if not histogram:
            return None, {"reason": "no_light_background_candidates"}
        mode_bucket = max(histogram, key=lambda bucket: histogram[bucket])
        bucket_pixels = [
            pixel
            for pixel in background_candidates
            if tuple(min(255, int(round(channel / 8.0) * 8)) for channel in pixel) == mode_bucket
        ]
        mode = tuple(
            sorted(pixel[channel] for pixel in bucket_pixels)[len(bucket_pixels) // 2]
            for channel in range(3)
        )
        uniform_ratio = sum(
            max(abs(red - mode[0]), abs(green - mode[1]), abs(blue - mode[2])) <= 18
            for red, green, blue in background_candidates
        ) / len(background_candidates)
        mode_luminance = round(0.2126 * mode[0] + 0.7152 * mode[1] + 0.0722 * mode[2])
        metadata = {
            "light_pixel_ratio": round(light_ratio, 6),
            "median_luminance": median,
            "background_candidate_ratio": round(background_candidate_ratio, 6),
            "background_uniform_ratio": round(uniform_ratio, 6),
            "sampled_background_rgb": list(mode),
            "sampled_background_luminance": mode_luminance,
            "threshold": {
                "minimum_light_ratio": 0.45,
                "minimum_median_luminance": 225,
                "minimum_background_candidate_ratio": 0.55,
                "minimum_background_uniform_ratio": 0.8,
                "minimum_background_luminance": 225,
            },
        }
        if (
            light_ratio < 0.45
            or median < 225
            or background_candidate_ratio < 0.55
            or uniform_ratio < 0.8
            or mode_luminance < 225
        ):
            metadata["reason"] = "background_not_uniform_or_light"
            return None, metadata
        return tuple(channel / 255.0 for channel in mode), metadata

    def _authoritative_background_fill(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
    ) -> tuple[tuple[float, float, float], dict[str, Any]]:
        """Choose a fill colour without using background analysis as a veto.

        An extracted text-region box is painted in full. Background sampling is
        used only to make that paint blend into the source page; an inconclusive
        or failed sample falls back to white and never rejects the region.
        """

        fill, metadata = self._scan_background_fill(page, bbox)
        fill_source = "validated_background_sample"
        if fill is None:
            sampled = metadata.get("sampled_background_rgb")
            if (
                isinstance(sampled, list)
                and len(sampled) == 3
                and all(isinstance(channel, (int, float)) for channel in sampled)
            ):
                fill = (
                    min(255.0, max(0.0, float(sampled[0]))) / 255.0,
                    min(255.0, max(0.0, float(sampled[1]))) / 255.0,
                    min(255.0, max(0.0, float(sampled[2]))) / 255.0,
                )
                fill_source = "unvalidated_background_sample"
            else:
                fill = (1.0, 1.0, 1.0)
                fill_source = "white_fallback"
        assert fill is not None
        return fill, {
            **metadata,
            "accepted_without_background_gate": True,
            "fill_source": fill_source,
            "applied_fill_rgb": [round(channel * 255.0) for channel in fill],
        }

    def _scan_cell_text_masks(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
    ) -> list[BoundingBox]:
        # Hidden OCR word boxes localise source glyphs. Expand them enough to
        # absorb scan/OCR baseline drift, but never across the full cell: an
        # arrow, checkbox, or icon beside a label must remain untouched.
        masks = self._scan_text_line_masks(
            page,
            bbox,
            horizontal_expansion=2.5,
            vertical_expansion=3.0,
        )
        return self._clamp_scan_masks(masks, bbox, guard=1.2)

    def _scan_source_text_similarity(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
        expected_text: str,
    ) -> dict[str, Any]:
        expected = self._comparison_text(expected_text)
        try:
            actual_text = page.get_text("text", clip=self._fitz_rect(bbox))
        except Exception as exc:
            return {"score": 0.0, "reason": f"hidden_text_read_failed:{type(exc).__name__}"}
        actual = self._comparison_text(actual_text)
        if not expected or not actual:
            return {
                "score": 0.0,
                "expected_characters": len(expected),
                "actual_characters": len(actual),
                "reason": "empty_expected_or_hidden_text",
            }
        score = SequenceMatcher(None, expected, actual).ratio()
        if expected in actual or actual in expected:
            score = max(score, min(len(expected), len(actual)) / max(len(expected), len(actual)))
        return {
            "score": round(float(score), 6),
            "expected_characters": len(expected),
            "actual_characters": len(actual),
            "minimum_score": 0.62,
        }

    def _embedded_source_text_validation(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
        expected_text: str,
    ) -> dict[str, Any]:
        """Verify that an ordinary digital-text box contains only its source text.

        Marker and Surya boxes are useful placement hints, but a stale or merged
        box can include text from another paragraph or column. Redacting that
        complete rectangle would silently erase the neighbour. Normalising away
        markup, whitespace, and punctuation handles line wrapping and PDF
        hyphenation; the bounded character comparison then permits only tiny
        extraction differences and rejects every substantive unexplained run in
        the visible PDF text.
        """

        visible_expected = re.sub(
            r"(?is)<[^>]+>",
            " ",
            html.unescape(expected_text),
        )
        expected = self._comparison_text(visible_expected)
        if not expected:
            return {
                "safe": False,
                "reason": "embedded_source_text_missing",
                "expected_characters": 0,
                "actual_characters": 0,
            }

        try:
            actual_text = page.get_text("text", clip=self._fitz_rect(bbox))
        except Exception as exc:
            return {
                "safe": False,
                "reason": "embedded_source_text_read_failed",
                "error": type(exc).__name__,
                "expected_characters": len(expected),
                "actual_characters": 0,
            }
        actual = self._comparison_text(actual_text)
        if not actual:
            return {
                "safe": False,
                "reason": "embedded_source_text_not_found_in_bbox",
                "expected_characters": len(expected),
                "actual_characters": 0,
            }

        matcher = SequenceMatcher(None, expected, actual, autojunk=False)
        matching_characters = sum(match.size for match in matcher.get_matching_blocks())
        opcodes = matcher.get_opcodes()
        actual_unexplained_runs = [
            j2 - j1 for tag, _i1, _i2, j1, j2 in opcodes if tag in {"insert", "replace"} and j2 > j1
        ]
        expected_unmatched_runs = [
            i2 - i1 for tag, i1, i2, _j1, _j2 in opcodes if tag in {"delete", "replace"} and i2 > i1
        ]
        actual_unexplained = len(actual) - matching_characters
        expected_unmatched = len(expected) - matching_characters
        maximum_actual_unexplained = max(2, min(4, math.ceil(len(actual) * 0.01)))
        maximum_expected_unmatched = max(2, min(6, math.ceil(len(expected) * 0.02)))
        ratio = matcher.ratio()
        diagnostics: dict[str, Any] = {
            "safe": False,
            "expected_characters": len(expected),
            "actual_characters": len(actual),
            "matching_characters": matching_characters,
            "similarity": round(float(ratio), 6),
            "unexplained_actual_characters": actual_unexplained,
            "unmatched_expected_characters": expected_unmatched,
            "largest_unexplained_actual_run": max(actual_unexplained_runs, default=0),
            "largest_unmatched_expected_run": max(expected_unmatched_runs, default=0),
            "thresholds": {
                "minimum_similarity": 0.92,
                "maximum_unexplained_actual_characters": maximum_actual_unexplained,
                "maximum_unmatched_expected_characters": maximum_expected_unmatched,
                "maximum_unexplained_actual_run": 2,
            },
        }
        if (
            actual_unexplained > maximum_actual_unexplained
            or max(actual_unexplained_runs, default=0) > 2
        ):
            diagnostics["reason"] = "embedded_source_bbox_contains_unexplained_text"
            return diagnostics
        if expected_unmatched > maximum_expected_unmatched or ratio < 0.92:
            diagnostics["reason"] = "embedded_source_text_does_not_match_bbox"
            return diagnostics

        diagnostics["safe"] = True
        diagnostics["reason"] = "embedded_source_text_verified"
        return diagnostics

    def _scan_text_line_masks(
        self,
        page: fitz.Page,
        bbox: BoundingBox,
        *,
        padding: float = 0.0,
        horizontal_expansion: float = 2.0,
        vertical_expansion: float = 1.8,
    ) -> list[BoundingBox]:
        rectangle = self._fitz_rect(bbox)
        search_rectangle = (
            fitz.Rect(
                rectangle.x0 - padding,
                rectangle.y0 - padding,
                rectangle.x1 + padding,
                rectangle.y1 + padding,
            )
            & page.rect
        )
        try:
            words = page.get_text("words", clip=search_rectangle, sort=True)
        except Exception:
            return []
        groups: dict[tuple[int, int], list[fitz.Rect]] = {}
        for word in words:
            if len(word) < 8 or not self._comparison_text(str(word[4])):
                continue
            center_x = (float(word[0]) + float(word[2])) / 2
            center_y = (float(word[1]) + float(word[3])) / 2
            if not (
                search_rectangle.x0 <= center_x <= search_rectangle.x1
                and search_rectangle.y0 <= center_y <= search_rectangle.y1
            ):
                continue
            groups.setdefault((int(word[5]), int(word[6])), []).append(
                fitz.Rect(float(word[0]), float(word[1]), float(word[2]), float(word[3]))
            )

        masks: list[BoundingBox] = []
        inset = fitz.Rect(
            search_rectangle.x0 + 0.25,
            search_rectangle.y0 + 0.15,
            search_rectangle.x1 - 0.25,
            search_rectangle.y1 - 0.15,
        )
        for line_rectangles in groups.values():
            line = fitz.Rect(line_rectangles[0])
            for word_rect in line_rectangles[1:]:
                line |= word_rect
            line = fitz.Rect(
                line.x0 - horizontal_expansion,
                line.y0 - vertical_expansion,
                line.x1 + horizontal_expansion,
                line.y1 + vertical_expansion,
            )
            line &= inset
            if line.width < 1 or line.height < 1:
                continue
            masks.append(
                BoundingBox(
                    x0=float(line.x0),
                    y0=float(line.y0),
                    x1=float(line.x1),
                    y1=float(line.y1),
                )
            )
        return masks

    def _clamp_scan_masks(
        self,
        masks: list[BoundingBox],
        bbox: BoundingBox,
        *,
        guard: float = 0.0,
    ) -> list[BoundingBox]:
        x0 = bbox.x0 + guard
        y0 = bbox.y0 + guard
        x1 = bbox.x1 - guard
        y1 = bbox.y1 - guard
        if x1 <= x0 or y1 <= y0:
            return []
        clamped: list[BoundingBox] = []
        for mask in masks:
            candidate = BoundingBox(
                x0=max(x0, mask.x0),
                y0=max(y0, mask.y0),
                x1=min(x1, mask.x1),
                y1=min(y1, mask.y1),
            )
            if candidate.x1 - candidate.x0 >= 1.0 and candidate.y1 - candidate.y0 >= 1.0:
                clamped.append(candidate)
        return clamped

    def _inset_bbox(
        self,
        bbox: BoundingBox,
        *,
        horizontal: float,
        vertical: float,
    ) -> BoundingBox:
        x0 = bbox.x0 + horizontal
        y0 = bbox.y0 + vertical
        x1 = bbox.x1 - horizontal
        y1 = bbox.y1 - vertical
        if x1 - x0 < 2 or y1 - y0 < 2:
            return bbox.model_copy()
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def _align_table_rows(
        self,
        page: fitz.Page,
        source_rows: list[list[ParsedTableCell]],
        translated_rows: list[list[ParsedTableCell]],
        grid_rows: list[list[fitz.Rect]],
    ) -> list[tuple[list[ParsedTableCell], list[ParsedTableCell], list[fitz.Rect]]] | None:
        if not grid_rows or len(grid_rows) > len(source_rows):
            return None

        @lru_cache(maxsize=None)
        def solve(
            grid_index: int,
            logical_index: int,
        ) -> (
            tuple[
                float,
                tuple[tuple[list[ParsedTableCell], list[ParsedTableCell], list[fitz.Rect]], ...],
            ]
            | None
        ):
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

    def _table_alignment_has_full_cell_coverage(
        self,
        page: fitz.Page,
        alignment: list[tuple[list[ParsedTableCell], list[ParsedTableCell], list[fitz.Rect]]],
    ) -> bool:
        """Reject grids whose cells contain source text absent from the table model.

        The ordinary row scorer deliberately tolerates a neighbouring line that
        bleeds across a PDF clipping boundary. That tolerance is useful while
        locating tables, but is not sufficient before redacting a complete cell:
        an exact expected line must not make unrelated text in the same rectangle
        disappear. Full-cell equality is preferred; small OCR differences and
        boundary bleed are accepted only when every visible line can be explained
        by one of the logical cells in the aligned row.
        """

        for source_row, _translated_row, rectangles in alignment:
            source_lines = [
                normalized
                for cell in source_row
                for line in cell.text.splitlines()
                if (normalized := self._comparison_text(line))
            ]
            for source_cell, rectangle in zip(source_row, rectangles, strict=True):
                expected = self._comparison_text(source_cell.text)
                actual_text = self._table_cell_text(page, rectangle)
                actual = self._comparison_text(actual_text)
                if expected == actual:
                    continue
                if not expected:
                    # Empty logical placeholders are never redacted. Marker
                    # commonly overlaps them with a spanning neighbour, so
                    # visible text inside one cannot make replacement unsafe.
                    continue

                expected_lines = [
                    normalized
                    for line in source_cell.text.splitlines()
                    if (normalized := self._comparison_text(line))
                ]
                actual_lines = [
                    normalized
                    for line in actual_text.splitlines()
                    if (normalized := self._comparison_text(line))
                ]
                if (
                    expected
                    and len(actual_lines) <= max(1, len(expected_lines))
                    and self._text_matches_with_bounded_variation(expected, actual)
                ):
                    continue
                if expected_lines and not all(
                    any(
                        self._text_matches_with_bounded_variation(expected_line, actual_line)
                        for actual_line in actual_lines
                    )
                    for expected_line in expected_lines
                ):
                    return False
                if any(
                    not any(
                        self._text_matches_with_bounded_variation(actual_line, source_line)
                        for source_line in source_lines
                    )
                    for actual_line in actual_lines
                ):
                    return False
        return True

    def _text_matches_with_bounded_variation(self, expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        if not expected or not actual:
            return False
        if self._short_ocr_tokens_are_confusable(expected, actual):
            return True
        longest = max(len(expected), len(actual))
        shortest = min(len(expected), len(actual))
        if shortest <= 3:
            return False
        allowed_length_delta = max(2, math.ceil(longest * 0.08))
        if longest - shortest > allowed_length_delta:
            return False
        minimum_similarity = 0.9 if longest < 12 else 0.84
        return SequenceMatcher(None, expected, actual).ratio() >= minimum_similarity

    def _combine_table_rows(
        self,
        rows: list[list[ParsedTableCell]],
    ) -> list[ParsedTableCell]:
        combined: list[ParsedTableCell] = []
        for column_index in range(len(rows[0])):
            cells = [row[column_index] for row in rows]
            combined.append(
                ParsedTableCell(
                    tag="th" if any(cell.tag == "th" for cell in cells) else "td",
                    text="\n".join(cell.text for cell in cells if cell.text),
                )
            )
        return combined

    def _table_row_similarity(
        self,
        page: fitz.Page,
        source_row: list[ParsedTableCell],
        rectangles: list[fitz.Rect],
    ) -> float:
        weighted_score = 0.0
        total_weight = 0.0
        nonempty_cell_scores: list[float] = []
        for cell, rectangle in zip(source_row, rectangles, strict=True):
            expected = self._comparison_text(cell.text)
            actual_text = self._table_cell_text(page, rectangle)
            actual = self._comparison_text(actual_text)
            if not expected:
                # Empty logical cells are never redacted. PDF glyph boxes from a
                # neighbouring row often cross a predicted cell boundary, so
                # incidental clipped text here is not evidence of unsafe
                # geometry and must not reject an otherwise aligned table.
                continue
            weight = float(max(2, len(expected)))
            total_weight += weight
            if not actual:
                score = 0.0
            else:
                score = self._table_cell_text_similarity(expected, actual_text)
            nonempty_cell_scores.append(score)
            weighted_score += score * weight
        # A long correct cell must not outweigh a short cell mapped to the
        # wrong box: every nonempty source/visual cell needs plausible text.
        if any(score < 0.42 for score in nonempty_cell_scores):
            return 0.0
        if not nonempty_cell_scores:
            return 1.0
        return weighted_score / max(1.0, total_weight)

    def _table_cell_text(self, page: fitz.Page, rectangle: fitz.Rect) -> str:
        """Read only words whose centres belong to this semantic table cell."""

        try:
            # Padding prevents PyMuPDF from returning a truncated word when a
            # detector polygon ends inside its glyph box. Centres are still
            # filtered against the unpadded semantic cell below.
            word_clip = (
                fitz.Rect(
                    rectangle.x0 - 36.0,
                    rectangle.y0 - 4.0,
                    rectangle.x1 + 36.0,
                    rectangle.y1 + 4.0,
                )
                & page.rect
            )
            words = page.get_text("words", clip=word_clip, sort=True)
        except Exception:
            return ""
        groups: dict[tuple[int, int], list[tuple[float, str]]] = {}
        for word in words:
            if len(word) < 8:
                continue
            center_x = (float(word[0]) + float(word[2])) / 2
            center_y = (float(word[1]) + float(word[3])) / 2
            if not (
                rectangle.x0 <= center_x <= rectangle.x1
                and rectangle.y0 <= center_y <= rectangle.y1
            ):
                continue
            text = str(word[4]).strip()
            if not text:
                continue
            groups.setdefault((int(word[5]), int(word[6])), []).append((float(word[0]), text))
        return "\n".join(
            " ".join(text for _x, text in sorted(line)) for _key, line in groups.items()
        )

    def _table_cell_text_similarity(self, expected: str, actual_text: str) -> float:
        """Compare a logical cell with clipped PDF text without trusting substrings.

        PyMuPDF can return a neighbouring line when a glyph box touches the cell
        boundary.  Comparing individual lines lets the intended cell still match,
        while a short header such as ``N`` cannot validate an unrelated paragraph
        merely because that character occurs somewhere inside it.
        """

        candidates = [
            candidate
            for candidate in {
                self._comparison_text(actual_text),
                *(self._comparison_text(line) for line in actual_text.splitlines()),
            }
            if candidate
        ]
        if not candidates:
            return 0.0
        if expected in candidates:
            return 1.0
        if len(expected) <= 3:
            tokens = {
                self._comparison_text(token)
                for token in re.findall(r"\w+", actual_text, flags=re.UNICODE)
            }
            if expected in tokens:
                return 1.0
            return (
                0.9
                if any(self._short_ocr_tokens_are_confusable(expected, token) for token in tokens)
                else 0.0
            )

        best = 0.0
        for actual in candidates:
            score = SequenceMatcher(None, expected, actual).ratio()
            if actual in expected:
                score = max(score, len(actual) / len(expected))
            elif expected in actual:
                extra = len(actual) - len(expected)
                allowed_extra = max(2, math.ceil(len(expected) * 0.35))
                if extra <= allowed_extra:
                    score = max(score, len(expected) / len(actual))
            best = max(best, score)
        return best

    def _short_ocr_tokens_are_confusable(self, expected: str, actual: str) -> bool:
        if len(expected) != len(actual) or not 2 <= len(expected) <= 4:
            return False
        confusable_groups = (
            frozenset({"0", "o"}),
            frozenset({"1", "i", "l"}),
            frozenset({"5", "s"}),
        )
        differences = 0
        for expected_character, actual_character in zip(expected, actual, strict=True):
            if expected_character == actual_character:
                continue
            if not any(
                {expected_character, actual_character} <= group for group in confusable_groups
            ):
                return False
            differences += 1
        return differences == 1

    def _comparison_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return "".join(character for character in normalized if character.isalnum())

    def _region_html_and_css(
        self,
        region: _ReplacementRegion,
        page: fitz.Page,
    ) -> tuple[str, str]:
        table_css = ""
        table_rows = (
            self._parse_table_rows(region.translated_text)
            if region.block_type == BlockType.TABLE
            else []
        )
        if table_rows:
            rendered_rows = []
            for row in table_rows:
                rendered_cells = []
                for cell in row:
                    cell_text = html.escape(cell.text).replace("\n", "<br>")
                    spans = ""
                    if cell.rowspan > 1:
                        spans += f' rowspan="{cell.rowspan}"'
                    if cell.colspan > 1:
                        spans += f' colspan="{cell.colspan}"'
                    rendered_cells.append(f"<{cell.tag}{spans}>{cell_text}</{cell.tag}>")
                rendered_rows.append(f"<tr>{''.join(rendered_cells)}</tr>")
            rendered = f"<table>{''.join(rendered_rows)}</table>"
            table_css = (
                " table { width: 100%; border-collapse: collapse; table-layout: fixed; }"
                " th, td { border: 0.5pt solid #555; padding: 1pt; vertical-align: top; }"
            )
        else:
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
            or (
                "italic"
                if region.block_type in {BlockType.CAPTION, BlockType.FOOTNOTE}
                else "normal"
            )
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
            f"{table_css}"
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
        line_rectangles: list[fitz.Rect] = []
        for text_block in payload.get("blocks", []):
            for line in text_block.get("lines", []):
                line_rectangle: fitz.Rect | None = None
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        spans.append(span)
                        try:
                            span_rectangle = fitz.Rect(span["bbox"])
                            line_rectangle = (
                                span_rectangle
                                if line_rectangle is None
                                else fitz.Rect(line_rectangle | span_rectangle)
                            )
                        except Exception:
                            continue
                if line_rectangle is not None:
                    line_rectangles.append(line_rectangle)
        if not spans:
            return {}

        weighted_sizes: list[tuple[float, int]] = []
        total_weight = 0
        bold_weight = 0
        italic_weight = 0
        sans_weight = 0
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
        if infer_alignment and line_rectangles:
            gaps = [
                (
                    max(0.0, line.x0 - rectangle.x0),
                    max(0.0, rectangle.x1 - line.x1),
                )
                for line in line_rectangles
            ]
            tolerance = max(2.0, rectangle.width * 0.08)
            centered_lines = sum(
                abs(left_gap - right_gap) <= tolerance for left_gap, right_gap in gaps
            )
            sorted_left = sorted(left_gap for left_gap, _right_gap in gaps)
            sorted_right = sorted(right_gap for _left_gap, right_gap in gaps)
            median_left = sorted_left[len(sorted_left) // 2]
            median_right = sorted_right[len(sorted_right) // 2]
            if centered_lines / len(gaps) >= 0.6:
                hints["text_align"] = "center"
            elif median_right + 2.0 < median_left:
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
        alignment_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        report["regions_skipped"] += 1
        page_report["regions_skipped"] += 1
        if fallback_required:
            page_report["fallback_required"] = True
        region = {
            "page_number": block.page_number,
            "block_ids": [block.id],
            "block_type": block.block_type.value,
            "bbox": bbox.model_dump() if bbox is not None else None,
            "source_character_count": self._source_character_count(self._block_source_text(block)),
            "status": "skipped",
            "reason": reason,
        }
        if alignment_diagnostics:
            region["alignment_diagnostics"] = dict(alignment_diagnostics)
        report["regions"].append(region)
        self._warning(
            report,
            page_number=block.page_number,
            code="region_skipped",
            reason=f"Region {block.id} was skipped: {reason}.",
        )

    def _retain_block(
        self,
        report: dict[str, Any],
        page_report: dict[str, Any],
        block: Block,
        *,
        reason: str,
        bbox: BoundingBox | None = None,
    ) -> None:
        """Record an intentional, successful source-region preservation."""

        report["regions_retained"] += 1
        page_report["regions_retained"] += 1
        report["regions"].append(
            {
                "page_number": block.page_number,
                "block_ids": [block.id],
                "block_type": block.block_type.value,
                "bbox": bbox.model_dump() if bbox is not None else None,
                "source_character_count": self._source_character_count(
                    self._block_source_text(block)
                ),
                "status": "retained",
                "reason": reason,
            }
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
                "source_character_count": self._source_character_count(region.source_text),
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
        return bbox_intersection_area(region, locked) > 0

    def _redaction_bboxes_avoiding_locked_regions(
        self,
        bboxes: list[BoundingBox],
        locked_regions: list[BoundingBox],
        *,
        page_width: float,
        page_height: float,
    ) -> list[BoundingBox]:
        """Keep redaction fills and antialiasing away from locked visuals."""

        guarded = list(bboxes)
        for locked in locked_regions:
            obstacle = BoundingBox(
                x0=max(0.0, locked.x0 - self.locked_redaction_guard),
                y0=max(0.0, locked.y0 - self.locked_redaction_guard),
                x1=min(page_width, locked.x1 + self.locked_redaction_guard),
                y1=min(page_height, locked.y1 + self.locked_redaction_guard),
            )
            next_guarded: list[BoundingBox] = []
            for bbox in guarded:
                next_guarded.extend(self._subtract_bbox(bbox, obstacle))
            guarded = next_guarded
            if not guarded:
                break
        return guarded

    def _subtract_bbox(
        self,
        source: BoundingBox,
        obstacle: BoundingBox,
    ) -> list[BoundingBox]:
        x0 = max(source.x0, obstacle.x0)
        y0 = max(source.y0, obstacle.y0)
        x1 = min(source.x1, obstacle.x1)
        y1 = min(source.y1, obstacle.y1)
        if x0 >= x1 or y0 >= y1:
            return [source]

        candidates = (
            (source.x0, source.y0, source.x1, y0),
            (source.x0, y1, source.x1, source.y1),
            (source.x0, y0, x0, y1),
            (x1, y0, source.x1, y1),
        )
        return [
            BoundingBox(x0=left, y0=top, x1=right, y1=bottom)
            for left, top, right, bottom in candidates
            if right - left >= 0.25 and bottom - top >= 0.25
        ]

    def _union_bbox(self, bboxes: list[BoundingBox]) -> BoundingBox:
        return BoundingBox(
            x0=min(bbox.x0 for bbox in bboxes),
            y0=min(bbox.y0 for bbox in bboxes),
            x1=max(bbox.x1 for bbox in bboxes),
            y1=max(bbox.y1 for bbox in bboxes),
        )

    def _fitz_rect(self, bbox: BoundingBox) -> fitz.Rect:
        return fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

    def _redaction_bboxes(self, region: _ReplacementRegion) -> list[BoundingBox]:
        """Return the exact rectangles passed to PyMuPDF redaction."""

        return region.redaction_bboxes if region.redaction_bboxes is not None else [region.bbox]

    def _block_pdf_bbox(self, page: fitz.Page, block: Block) -> BoundingBox | None:
        """Return a validated block rectangle in PDF-page coordinates for reporting."""

        return convert_bbox_to_pdf(
            block.bbox,
            page_width=page.rect.width,
            page_height=page.rect.height,
            metadata=block.metadata,
        ).bbox

    def _block_source_text(self, block: Block) -> str:
        source_text = block.metadata.get("source_text")
        if source_text is None:
            source_text = block.text
        return str(source_text)

    def _source_character_count(self, source_text: str) -> int:
        visible = re.sub(r"(?is)<[^>]+>", " ", html.unescape(source_text))
        return len(self._normalized_text(visible))

    def _normalized_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
