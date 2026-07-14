from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
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
                    report=report,
                    page_report=page_report,
                )
                approved: list[tuple[_ReplacementRegion, str, str, float]] = []
                for region in replacements:
                    html_text, css = self._region_html_and_css(region)
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
        for figure in document.figures:
            if figure.bbox is None or not (1 <= figure.page_number <= pdf.page_count):
                continue
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
        report: dict[str, Any],
        page_report: dict[str, Any],
    ) -> list[_ReplacementRegion]:
        block_by_id = {block.id: block for block in all_blocks}
        consumed: set[str] = set()
        replacements: list[_ReplacementRegion] = []
        for block in blocks:
            if block.id in consumed or not block.text.strip():
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
            if block.block_type in self.conservative_skip_types:
                self._skip_block(
                    report,
                    page_report,
                    block,
                    reason="table_layout_preserved_conservatively",
                )
                continue

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
            if source_text and self._normalized_text(source_text) == self._normalized_text(block.text):
                # English or otherwise unchanged text already matches the visual base.
                continue
            replacements.append(
                _ReplacementRegion(
                    page_number=page_number,
                    block_ids=[item.id for item in source_blocks],
                    block_type=block.block_type,
                    bbox=bbox,
                    translated_text=block.text.strip(),
                    source_text=source_text,
                    style_hints=dict(block.style_hints or {}),
                    coordinate_metadata=conversions,
                )
            )
        return replacements

    def _region_html_and_css(self, region: _ReplacementRegion) -> tuple[str, str]:
        escaped = html.escape(region.translated_text)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", escaped) if part.strip()]
        rendered = "".join(f"<p>{part.replace(chr(10), '<br>')}</p>" for part in paragraphs)
        if not rendered:
            rendered = f"<p>{escaped}</p>"
        font_size = self._font_size(region)
        family = "sans-serif" if region.block_type == BlockType.HEADING else "serif"
        weight = "bold" if region.block_type == BlockType.HEADING else "normal"
        style = "italic" if region.block_type in {BlockType.CAPTION, BlockType.FOOTNOTE} else "normal"
        align = str(region.style_hints.get("text_align") or "left").lower()
        if align not in {"left", "right", "center", "justify"}:
            align = "left"
        css = (
            f"* {{ font-family: {family}; font-size: {font_size:.2f}pt; "
            f"font-weight: {weight}; font-style: {style}; color: #111; }} "
            f"p {{ margin: 0; padding: 0; line-height: 1.15; text-align: {align}; }}"
        )
        return f"<div>{rendered}</div>", css

    def _font_size(self, region: _ReplacementRegion) -> float:
        hint = region.style_hints.get("font_size")
        try:
            hinted = float(hint) if hint is not None else 0.0
        except (TypeError, ValueError):
            hinted = 0.0
        if hinted > 0:
            return max(6.0, min(24.0, hinted))
        height = max(1.0, region.bbox.y1 - region.bbox.y0)
        lines = max(1, region.translated_text.count("\n") + 1)
        return max(7.0, min(12.0, height / lines * 0.72))

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
