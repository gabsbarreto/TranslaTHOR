from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz

from app.models.schema import (
    Block,
    BlockType,
    BoundingBox,
    DocumentModel,
    FigureAsset,
    FigureAssetType,
)
from app.services.pdf_coordinates import (
    bbox_area,
    bbox_intersection_area,
    bbox_iou,
    convert_bbox_to_pdf,
)

logger = logging.getLogger(__name__)

FIGURE_CAPTION_PATTERN = re.compile(
    r"^[\s▶►•·\-–—]*(?:figure|fig\.?|figura|abbildung|abb\.?|graphique|grafik)\s*\d+",
    flags=re.IGNORECASE,
)
TABLE_CAPTION_PATTERN = re.compile(
    r"^[\s▶►•·\-–—]*(?:table|tableau|tab\.?|tablo|tabelle|tabela|cuadro)\s*\d+",
    flags=re.IGNORECASE,
)


@dataclass
class _FigureCandidate:
    page_number: int
    bbox: BoundingBox | None
    source_block_ids: list[str] = field(default_factory=list)
    source_region_ids: list[str] = field(default_factory=list)
    reading_order_index: int | None = None
    confidence: float | None = None
    marker_type: str = ""
    caption_block_id: str | None = None
    priority: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class FigureExtractionService:
    """Identify figure regions and materialise reusable job-local assets."""

    preview_scale = 3.0

    def extract(
        self,
        *,
        pdf_path: Path,
        document: DocumentModel,
        artifact_dir: Path,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> DocumentModel:
        populated = document.model_copy(deep=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        surya_regions = self._surya_regions(extraction_metadata or {})
        candidates = self._collect_candidates(populated, surya_regions)

        with fitz.open(pdf_path) as source_pdf:
            normalized = [
                self._normalize_candidate(candidate, populated, source_pdf)
                for candidate in candidates
            ]
            canonical = self._deduplicate_candidates(normalized)
            caption_bboxes = self._caption_bboxes(populated, source_pdf)
            self._associate_captions(canonical, populated.blocks, caption_bboxes)

            assets: list[FigureAsset] = []
            valid_index_by_page: dict[int, int] = {}
            for candidate in sorted(canonical, key=self._candidate_sort_key):
                if candidate.bbox is None or not (1 <= candidate.page_number <= source_pdf.page_count):
                    asset = self._unmaterialized_asset(candidate, len(assets) + 1)
                    assets.append(asset)
                    warnings.append(
                        f"Figure {asset.id} could not be materialised because its page or bounding box is invalid."
                    )
                    continue

                page_index = candidate.page_number - 1
                valid_index_by_page[candidate.page_number] = valid_index_by_page.get(candidate.page_number, 0) + 1
                sequence = valid_index_by_page[candidate.page_number]
                basename = f"figure-p{candidate.page_number:04d}-{sequence:03d}"
                asset, asset_warnings = self._materialize_candidate(
                    source_pdf=source_pdf,
                    page_index=page_index,
                    candidate=candidate,
                    artifact_dir=artifact_dir,
                    basename=basename,
                )
                assets.append(asset)
                warnings.extend(asset_warnings)

        populated.figures = assets
        self._link_blocks_to_assets(populated, assets)
        populated.metadata.translation = {
            **populated.metadata.translation,
            "figure_extraction": {
                "figure_count": len(assets),
                "materialized_count": sum(1 for asset in assets if asset.image_path),
                "vector_asset_count": sum(1 for asset in assets if asset.vector_path),
                "raster_fallback_count": sum(
                    1 for asset in assets if asset.image_path and not asset.vector_path
                ),
                "internal_figure_text_policy": "preserve_source_language",
            },
        }
        for warning in warnings:
            if warning not in populated.warnings:
                populated.warnings.append(warning)
        return populated

    def _collect_candidates(
        self,
        document: DocumentModel,
        surya_regions: dict[str, dict[str, Any]],
    ) -> list[_FigureCandidate]:
        candidates: list[_FigureCandidate] = []
        for block in document.blocks:
            if block.block_type != BlockType.FIGURE:
                continue
            metadata = dict(block.metadata or {})
            source_region_ids = [str(value) for value in metadata.get("source_region_ids", [])]
            region_metadata = [surya_regions[value] for value in source_region_ids if value in surya_regions]
            if region_metadata:
                metadata["surya_source_regions"] = region_metadata
            confidence = self._candidate_confidence(block, region_metadata)
            marker_type = str(
                metadata.get("marker_block_type")
                or metadata.get("surya_region_type")
                or "figure"
            )
            candidates.append(
                _FigureCandidate(
                    page_number=block.page_number,
                    bbox=block.bbox,
                    source_block_ids=[block.id],
                    source_region_ids=source_region_ids,
                    reading_order_index=block.reading_order_index,
                    confidence=confidence,
                    marker_type=marker_type,
                    priority=self._candidate_priority(marker_type),
                    metadata=metadata,
                )
            )

        for figure in document.figures:
            metadata = dict(figure.extraction_metadata or {})
            candidates.append(
                _FigureCandidate(
                    page_number=figure.page_number,
                    bbox=figure.bbox,
                    source_block_ids=list(figure.source_block_ids),
                    source_region_ids=list(figure.source_region_ids),
                    reading_order_index=self._optional_int(metadata.get("reading_order_index")),
                    confidence=figure.detection_confidence,
                    marker_type=str(metadata.get("marker_block_type") or "existing_figure_record"),
                    caption_block_id=figure.caption_block_id,
                    priority=1,
                    metadata={**metadata, "existing_figure_id": figure.id},
                )
            )
        return candidates

    def _normalize_candidate(
        self,
        candidate: _FigureCandidate,
        document: DocumentModel,
        source_pdf: fitz.Document,
    ) -> _FigureCandidate:
        if not (1 <= candidate.page_number <= source_pdf.page_count):
            candidate.bbox = None
            candidate.metadata["coordinate_conversion"] = {
                "conversion": "invalid",
                "reason": "page_outside_document",
            }
            return candidate

        page = source_pdf[candidate.page_number - 1]
        conversion = convert_bbox_to_pdf(
            candidate.bbox,
            page_width=page.rect.width,
            page_height=page.rect.height,
            metadata=candidate.metadata,
        )
        candidate.bbox = conversion.bbox
        candidate.metadata["coordinate_conversion"] = conversion.metadata
        candidate.metadata["source_page_rotation"] = int(page.rotation)
        candidate.metadata["source_page_cropbox"] = list(page.cropbox)
        if candidate.confidence is None and candidate.bbox is not None:
            candidate.confidence = 0.85 if candidate.source_block_ids else 0.65
        return candidate

    def _deduplicate_candidates(self, candidates: list[_FigureCandidate]) -> list[_FigureCandidate]:
        valid = [candidate for candidate in candidates if candidate.bbox is not None]
        groups = [candidate for candidate in valid if "group" in candidate.marker_type.lower()]
        leaves = [
            candidate
            for candidate in valid
            if "group" not in candidate.marker_type.lower()
            and candidate.marker_type != "existing_figure_record"
        ]
        for group in groups:
            contained = [leaf for leaf in leaves if self._contains(group, leaf)]
            if len(contained) >= 2:
                group.priority = max(group.priority, 6)

        canonical: list[_FigureCandidate] = []
        for candidate in sorted(valid, key=self._candidate_sort_key):
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(canonical)
                    if self._same_visual_region(existing, candidate)
                ),
                None,
            )
            if duplicate_index is None:
                canonical.append(candidate)
            else:
                canonical[duplicate_index] = self._merge_candidates(
                    canonical[duplicate_index], candidate
                )

        missing = [candidate for candidate in candidates if candidate.bbox is None]
        for candidate in missing:
            if candidate.source_block_ids and any(
                set(candidate.source_block_ids) & set(existing.source_block_ids)
                for existing in canonical
            ):
                continue
            canonical.append(candidate)
        return canonical

    def _same_visual_region(
        self,
        first: _FigureCandidate,
        second: _FigureCandidate,
    ) -> bool:
        if first.page_number != second.page_number or first.bbox is None or second.bbox is None:
            return False
        intersection = bbox_intersection_area(first.bbox, second.bbox)
        smaller = min(bbox_area(first.bbox), bbox_area(second.bbox))
        overlap_of_smaller = intersection / smaller if smaller > 0 else 0.0
        return bbox_iou(first.bbox, second.bbox) >= 0.72 or overlap_of_smaller >= 0.94

    def _contains(self, outer: _FigureCandidate, inner: _FigureCandidate) -> bool:
        if outer.page_number != inner.page_number or outer.bbox is None or inner.bbox is None:
            return False
        inner_area = bbox_area(inner.bbox)
        return inner_area > 0 and bbox_intersection_area(outer.bbox, inner.bbox) / inner_area >= 0.94

    def _merge_candidates(
        self,
        first: _FigureCandidate,
        second: _FigureCandidate,
    ) -> _FigureCandidate:
        if second.priority > first.priority:
            primary, secondary = second, first
        elif second.priority == first.priority and second.bbox and first.bbox:
            primary, secondary = (
                (second, first)
                if bbox_area(second.bbox) < bbox_area(first.bbox)
                else (first, second)
            )
        else:
            primary, secondary = first, second
        primary.source_block_ids = self._unique(
            [*primary.source_block_ids, *secondary.source_block_ids]
        )
        primary.source_region_ids = self._unique(
            [*primary.source_region_ids, *secondary.source_region_ids]
        )
        primary.caption_block_id = primary.caption_block_id or secondary.caption_block_id
        primary.confidence = self._max_optional(primary.confidence, secondary.confidence)
        orders = [
            value
            for value in (primary.reading_order_index, secondary.reading_order_index)
            if value is not None
        ]
        primary.reading_order_index = min(orders) if orders else None
        primary.metadata = {
            **secondary.metadata,
            **primary.metadata,
            "merged_candidate_sources": self._unique(
                [
                    str(secondary.metadata.get("existing_figure_id") or secondary.marker_type),
                    str(primary.metadata.get("existing_figure_id") or primary.marker_type),
                ]
            ),
        }
        return primary

    def _caption_bboxes(
        self,
        document: DocumentModel,
        source_pdf: fitz.Document,
    ) -> dict[str, BoundingBox | None]:
        result: dict[str, BoundingBox | None] = {}
        for block in document.blocks:
            if block.block_type != BlockType.CAPTION or not (1 <= block.page_number <= source_pdf.page_count):
                continue
            page = source_pdf[block.page_number - 1]
            result[block.id] = convert_bbox_to_pdf(
                block.bbox,
                page_width=page.rect.width,
                page_height=page.rect.height,
                metadata=block.metadata,
            ).bbox
        return result

    def _associate_captions(
        self,
        candidates: list[_FigureCandidate],
        blocks: list[Block],
        caption_bboxes: dict[str, BoundingBox | None],
    ) -> None:
        captions = [block for block in blocks if block.block_type == BlockType.CAPTION]
        caption_by_id = {caption.id: caption for caption in captions}
        used: set[str] = set()
        for candidate in sorted(candidates, key=self._candidate_sort_key):
            if candidate.caption_block_id in caption_by_id:
                used.add(str(candidate.caption_block_id))
                candidate.metadata["caption_association"] = {
                    "confidence": 1.0,
                    "method": "existing_relationship",
                }
                self._exclude_caption_from_bbox(
                    candidate,
                    caption_bboxes.get(str(candidate.caption_block_id)),
                )
                continue

            scored: list[tuple[float, Block]] = []
            for caption in captions:
                if caption.id in used or caption.page_number != candidate.page_number:
                    continue
                score = self._caption_score(
                    candidate,
                    caption,
                    caption_bboxes.get(caption.id),
                )
                if score >= 0.32:
                    scored.append((score, caption))
            if not scored:
                continue
            score, caption = max(scored, key=lambda item: item[0])
            candidate.caption_block_id = caption.id
            used.add(caption.id)
            candidate.metadata["caption_association"] = {
                "confidence": round(min(1.0, score), 3),
                "method": "relationship_reading_order_position_distance",
                "caption_block_id": caption.id,
            }
            self._exclude_caption_from_bbox(candidate, caption_bboxes.get(caption.id))

    def _caption_score(
        self,
        candidate: _FigureCandidate,
        caption: Block,
        caption_bbox: BoundingBox | None,
    ) -> float:
        if TABLE_CAPTION_PATTERN.match(caption.text):
            return -1.0
        score = 0.25 if FIGURE_CAPTION_PATTERN.match(caption.text) else 0.0
        if candidate.reading_order_index is not None:
            delta = caption.reading_order_index - candidate.reading_order_index
            if delta in {0, 1}:
                score += 0.34
            elif 1 < delta <= 3:
                score += 0.2
            elif -2 <= delta < 0:
                score += 0.1

        if candidate.bbox is not None and caption_bbox is not None:
            figure = candidate.bbox
            horizontal_overlap = max(
                0.0,
                min(figure.x1, caption_bbox.x1) - max(figure.x0, caption_bbox.x0),
            )
            overlap_ratio = horizontal_overlap / max(
                1.0,
                min(figure.x1 - figure.x0, caption_bbox.x1 - caption_bbox.x0),
            )
            score += min(0.2, overlap_ratio * 0.2)
            if caption_bbox.y0 >= figure.y1 - 2.0:
                gap = max(0.0, caption_bbox.y0 - figure.y1)
                score += max(0.0, 0.36 - gap / max(40.0, figure.y1 - figure.y0))
            elif caption_bbox.y1 <= figure.y0 + 2.0:
                gap = max(0.0, figure.y0 - caption_bbox.y1)
                score += max(0.0, 0.16 - gap / max(60.0, figure.y1 - figure.y0))
            elif bbox_intersection_area(figure, caption_bbox) > 0:
                score += 0.18

        figure_section = candidate.metadata.get("section_hierarchy")
        caption_section = (caption.metadata or {}).get("section_hierarchy")
        if figure_section and figure_section == caption_section:
            score += 0.08
        return score

    def _exclude_caption_from_bbox(
        self,
        candidate: _FigureCandidate,
        caption_bbox: BoundingBox | None,
    ) -> None:
        figure = candidate.bbox
        if figure is None or caption_bbox is None:
            return
        intersection = bbox_intersection_area(figure, caption_bbox)
        if intersection <= 0:
            return
        height = figure.y1 - figure.y0
        caption_height = caption_bbox.y1 - caption_bbox.y0
        if height <= 0 or caption_height > height * 0.35:
            return
        adjusted: BoundingBox | None = None
        if caption_bbox.y0 <= figure.y0 + height * 0.3 and caption_bbox.y1 < figure.y1 - 8:
            adjusted = figure.model_copy(update={"y0": min(figure.y1 - 1, caption_bbox.y1)})
        elif caption_bbox.y1 >= figure.y1 - height * 0.3 and caption_bbox.y0 > figure.y0 + 8:
            adjusted = figure.model_copy(update={"y1": max(figure.y0 + 1, caption_bbox.y0)})
        if adjusted is not None and bbox_area(adjusted) >= max(100.0, bbox_area(figure) * 0.5):
            candidate.metadata["coordinate_conversion"]["caption_excluded_from_capture"] = {
                "original_pdf_bbox": figure.model_dump(),
                "caption_pdf_bbox": caption_bbox.model_dump(),
                "adjusted_pdf_bbox": adjusted.model_dump(),
            }
            candidate.bbox = adjusted

    def _materialize_candidate(
        self,
        *,
        source_pdf: fitz.Document,
        page_index: int,
        candidate: _FigureCandidate,
        artifact_dir: Path,
        basename: str,
    ) -> tuple[FigureAsset, list[str]]:
        assert candidate.bbox is not None
        warnings: list[str] = []
        page = source_pdf[page_index]
        clip = fitz.Rect(
            candidate.bbox.x0,
            candidate.bbox.y0,
            candidate.bbox.x1,
            candidate.bbox.y1,
        )
        has_raster = self._has_raster_content(page, clip)
        has_vector = self._has_vector_content(page, clip)
        has_internal_text = bool(page.get_text("text", clip=clip).strip())
        asset_type = self._asset_type(has_raster, has_vector)
        preview_path = artifact_dir / f"{basename}.png"
        vector_path = artifact_dir / f"{basename}.svg"

        preview_written = False
        try:
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(self.preview_scale, self.preview_scale),
                clip=clip,
                alpha=False,
            )
            pixmap.save(str(preview_path))
            preview_written = True
        except Exception as exc:
            warnings.append(f"Figure {basename} raster preview failed: {exc}")

        vector_written = False
        if asset_type != FigureAssetType.RASTER:
            try:
                vector_document = fitz.open()
                try:
                    vector_page = vector_document.new_page(width=clip.width, height=clip.height)
                    vector_page.show_pdf_page(
                        vector_page.rect,
                        source_pdf,
                        page_index,
                        clip=clip,
                        keep_proportion=True,
                    )
                    svg = vector_page.get_svg_image(text_as_path=1)
                    vector_path.write_text(svg, encoding="utf-8")
                    vector_written = True
                finally:
                    vector_document.close()
            except Exception as exc:
                warnings.append(f"Figure {basename} vector capture failed; raster preview will be used: {exc}")

        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "reading_order_index": candidate.reading_order_index,
                "preview_scale": self.preview_scale,
                "preview_dpi": int(round(72 * self.preview_scale)),
                "vector_format": "svg" if vector_written else None,
                "readable_reconstruction_asset": "vector" if vector_written else "raster",
            }
        )
        if not vector_written:
            metadata["raster_fallback_reason"] = (
                "source_region_is_raster"
                if asset_type == FigureAssetType.RASTER
                else "vector_capture_unavailable"
            )

        width = clip.width
        height = clip.height
        asset = FigureAsset(
            id=basename,
            page_number=candidate.page_number,
            bbox=candidate.bbox,
            caption_block_id=candidate.caption_block_id,
            image_path=str(preview_path.resolve()) if preview_written else None,
            original_width=width,
            original_height=height,
            aspect_ratio=(width / height) if height > 0 else None,
            asset_type=asset_type,
            vector_path=str(vector_path.resolve()) if vector_written else None,
            detection_confidence=candidate.confidence,
            has_internal_text=has_internal_text,
            source_block_ids=list(candidate.source_block_ids),
            source_region_ids=list(candidate.source_region_ids),
            extraction_metadata=metadata,
        )
        return asset, warnings

    def _unmaterialized_asset(self, candidate: _FigureCandidate, index: int) -> FigureAsset:
        return FigureAsset(
            id=f"figure-unresolved-{index:03d}",
            page_number=candidate.page_number,
            bbox=candidate.bbox,
            caption_block_id=candidate.caption_block_id,
            detection_confidence=candidate.confidence,
            source_block_ids=list(candidate.source_block_ids),
            source_region_ids=list(candidate.source_region_ids),
            extraction_metadata={
                **candidate.metadata,
                "materialization_status": "skipped_invalid_bbox_or_page",
            },
        )

    def _has_raster_content(self, page: fitz.Page, clip: fitz.Rect) -> bool:
        for image in page.get_images(full=True):
            try:
                rectangles = page.get_image_rects(image[0])
            except Exception:
                continue
            if any(self._fitz_intersection_area(rectangle, clip) > 1.0 for rectangle in rectangles):
                return True
        return False

    def _has_vector_content(self, page: fitz.Page, clip: fitz.Rect) -> bool:
        try:
            drawings = page.get_drawings()
        except Exception:
            return False
        return any(
            self._fitz_intersection_area(fitz.Rect(item.get("rect", fitz.Rect())), clip) > 1.0
            for item in drawings
        )

    def _asset_type(self, has_raster: bool, has_vector: bool) -> FigureAssetType:
        if has_raster and has_vector:
            return FigureAssetType.MIXED
        if has_raster:
            return FigureAssetType.RASTER
        if has_vector:
            return FigureAssetType.VECTOR
        return FigureAssetType.UNKNOWN

    def _link_blocks_to_assets(self, document: DocumentModel, assets: list[FigureAsset]) -> None:
        block_by_id = {block.id: block for block in document.blocks}
        for asset in assets:
            for index, block_id in enumerate(asset.source_block_ids):
                block = block_by_id.get(block_id)
                if block is None:
                    continue
                block.metadata["figure_asset_id"] = asset.id
                block.metadata["figure_asset_duplicate"] = index > 0
                block.metadata["excluded_from_translation"] = True
                block.metadata["translation_exclusion_reason"] = "figure_internal_text_preserved"

    def _surya_regions(self, extraction_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        manifest_value = extraction_metadata.get("surya_layout_manifest")
        if not manifest_value:
            return {}
        manifest_path = Path(str(manifest_value))
        if not manifest_path.exists():
            return {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Unable to read Surya figure metadata from %s: %s", manifest_path, exc)
            return {}
        regions: dict[str, dict[str, Any]] = {}
        for page in manifest.get("pages", []):
            for region in page.get("regions", []):
                region_id = str(region.get("id") or "")
                if region_id:
                    regions[region_id] = {
                        "id": region_id,
                        "label": region.get("label"),
                        "confidence": region.get("confidence"),
                        "bbox": region.get("bbox"),
                        "polygon": region.get("polygon"),
                        "top_k": region.get("top_k"),
                        "page_index": page.get("page_index"),
                        "page_width": page.get("width"),
                        "page_height": page.get("height"),
                    }
        return regions

    def _candidate_confidence(
        self,
        block: Block,
        region_metadata: list[dict[str, Any]],
    ) -> float | None:
        values: list[float] = []
        if block.confidence is not None:
            values.append(float(block.confidence))
        for region in region_metadata:
            confidence = region.get("confidence")
            if confidence is None:
                continue
            try:
                values.append(float(confidence))
            except (TypeError, ValueError):
                continue
        return max(values) if values else None

    def _candidate_priority(self, marker_type: str) -> int:
        normalized = re.sub(r"[^a-z]", "", marker_type.lower())
        if normalized in {"figure", "picture"}:
            return 4
        if "group" in normalized:
            return 2
        return 3

    def _candidate_sort_key(self, candidate: _FigureCandidate) -> tuple:
        bbox = candidate.bbox
        return (
            candidate.page_number,
            candidate.reading_order_index if candidate.reading_order_index is not None else 10**9,
            bbox.y0 if bbox is not None else 10**9,
            bbox.x0 if bbox is not None else 10**9,
        )

    def _fitz_intersection_area(self, first: fitz.Rect, second: fitz.Rect) -> float:
        intersection = first & second
        return float(max(0.0, intersection.width) * max(0.0, intersection.height))

    def _optional_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _max_optional(self, first: float | None, second: float | None) -> float | None:
        values = [value for value in (first, second) if value is not None]
        return max(values) if values else None

    def _unique(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))
