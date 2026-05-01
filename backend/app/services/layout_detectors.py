from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.models.regions import CoordinateSpace, Region, RegionSource, RegionType
from app.services.coordinate_utils import normalize_region

logger = logging.getLogger(__name__)


class LayoutDetector(Protocol):
    """Layout detector abstraction.

    Implementations can be swapped without changing OCR routing or API contracts.
    Surya, DocLayout-YOLO, LayoutParser, Marker, or custom detectors can implement
    the same `detect` method and return normalized regions.
    """

    name: str

    def detect(self, pdf_path: Path, page_number: int, page_width: float, page_height: float) -> list[Region]:
        ...


@dataclass
class DetectorDecision:
    detector_name: str
    regions: list[Region]


class PyMuPDFLayoutDetector:
    name = "pymupdf"

    def detect(self, pdf_path: Path, page_number: int, page_width: float, page_height: float) -> list[Region]:
        try:
            import fitz
        except Exception:
            logger.warning("PyMuPDF is not installed; returning no candidate boxes")
            return []

        regions: list[Region] = []
        with fitz.open(str(pdf_path)) as document:
            page = document[page_number - 1]
            blocks = page.get_text("blocks") or []
            for idx, block in enumerate(blocks):
                if len(block) < 5:
                    continue
                x0, y0, x1, y1, text = block[:5]
                clean = (text or "").strip()
                if not clean:
                    continue
                if (x1 - x0) < 3 or (y1 - y0) < 3:
                    continue

                region_type = _guess_region_type(clean, float(y0), float(y1), float(page.rect.height))
                region = Region(
                    id=f"r-{page_number}-{idx}-{uuid.uuid4().hex[:6]}",
                    page_number=page_number,
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    coordinate_space=CoordinateSpace.PDF,
                    type=region_type,
                    selected=True,
                    reading_order=idx + 1,
                    source=RegionSource.PYMUPDF,
                )
                regions.append(normalize_region(region, page_width=page_width, page_height=page_height))

            # Table detection support depends on the local PyMuPDF build.
            try:
                finder = getattr(page, "find_tables", None)
                if callable(finder):
                    tables = finder()  # type: ignore[operator]
                    table_items = getattr(tables, "tables", tables)
                    for t_idx, table in enumerate(table_items or [], start=1):
                        bbox = getattr(table, "bbox", None)
                        if not bbox or len(bbox) < 4:
                            continue
                        region = Region(
                            id=f"rt-{page_number}-{t_idx}-{uuid.uuid4().hex[:6]}",
                            page_number=page_number,
                            x0=float(bbox[0]),
                            y0=float(bbox[1]),
                            x1=float(bbox[2]),
                            y1=float(bbox[3]),
                            coordinate_space=CoordinateSpace.PDF,
                            type=RegionType.TABLE,
                            selected=True,
                            reading_order=len(regions) + t_idx,
                            source=RegionSource.PYMUPDF,
                        )
                        regions.append(normalize_region(region, page_width=page_width, page_height=page_height))
            except Exception:
                logger.debug("PyMuPDF table detection failed on page %s", page_number, exc_info=True)

        return _sort_and_reindex(regions)


class SuryaLayoutDetector:
    """Placeholder for Surya integration.

    Swap-in point: replace `detect` with real Surya calls and keep the same output
    schema. This keeps API handlers and OCR pipeline unchanged.
    """

    name = "surya"

    def detect(self, pdf_path: Path, page_number: int, page_width: float, page_height: float) -> list[Region]:
        _ = (pdf_path, page_number, page_width, page_height)
        # TODO: integrate Surya layout + reading order when dependency/runtime is available.
        return []


class HybridLayoutDetector:
    """Prefer embedded text regions; fallback to model detector when quality is poor."""

    def __init__(self, fallback_detector: LayoutDetector | None = None) -> None:
        self.pymupdf = PyMuPDFLayoutDetector()
        self.fallback = fallback_detector or SuryaLayoutDetector()

    def detect(
        self,
        *,
        pdf_path: Path,
        page_number: int,
        page_width: float,
        page_height: float,
        has_embedded_text: bool,
        embedded_text_quality: float,
    ) -> DetectorDecision:
        primary_regions = self.pymupdf.detect(pdf_path, page_number, page_width, page_height)
        if has_embedded_text and embedded_text_quality >= 0.3 and len(primary_regions) >= 2:
            return DetectorDecision(detector_name=self.pymupdf.name, regions=primary_regions)

        fallback_regions = self.fallback.detect(pdf_path, page_number, page_width, page_height)
        if fallback_regions:
            return DetectorDecision(detector_name=self.fallback.name, regions=_sort_and_reindex(fallback_regions))

        if primary_regions:
            return DetectorDecision(detector_name=self.pymupdf.name, regions=primary_regions)

        full_page = Region(
            id=f"manual-full-{page_number}",
            page_number=page_number,
            x0=0.02,
            y0=0.02,
            x1=0.98,
            y1=0.98,
            coordinate_space=CoordinateSpace.NORMALIZED,
            type=RegionType.TEXT,
            selected=True,
            reading_order=1,
            source=RegionSource.MANUAL,
        )
        return DetectorDecision(detector_name="fallback_full_page", regions=[full_page])


def _sort_and_reindex(regions: list[Region]) -> list[Region]:
    ordered = _column_aware_order(regions)
    out: list[Region] = []
    for idx, region in enumerate(ordered, start=1):
        out.append(region.model_copy(update={"reading_order": idx}))
    return out


def _column_aware_order(regions: list[Region]) -> list[Region]:
    if len(regions) < 4:
        return sorted(regions, key=lambda item: (round(item.y0, 4), round(item.x0, 4), item.reading_order, item.id))

    columns = _split_into_columns(regions)
    if len(columns) <= 1:
        return sorted(regions, key=lambda item: (round(item.y0, 4), round(item.x0, 4), item.reading_order, item.id))

    ordered: list[Region] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda item: (round(item.y0, 4), round(item.x0, 4), item.reading_order, item.id)))
    return ordered


def _split_into_columns(regions: list[Region]) -> list[list[Region]]:
    sorted_by_x = sorted(regions, key=lambda item: ((item.x0 + item.x1) / 2.0, item.y0, item.id))
    centers = [(item.x0 + item.x1) / 2.0 for item in sorted_by_x]
    gaps = [(centers[idx + 1] - centers[idx], idx) for idx in range(len(centers) - 1)]
    if not gaps:
        return [regions]

    max_gap, split_idx = max(gaps, key=lambda item: item[0])
    page_span = max(item.x1 for item in regions) - min(item.x0 for item in regions)
    if page_span <= 0 or max_gap < max(0.12, page_span * 0.22):
        return [regions]

    left = sorted_by_x[: split_idx + 1]
    right = sorted_by_x[split_idx + 1 :]
    if min(len(left), len(right)) < 2:
        return [regions]

    return [left, right]


def _guess_region_type(text: str, y0: float, y1: float, page_height: float) -> RegionType:
    lowered = text.strip().lower()
    if not lowered:
        return RegionType.OTHER

    if re.match(r"^references?$", lowered) or re.match(r"^\[\d+\]", lowered):
        return RegionType.REFERENCE
    if re.match(r"^(figure|fig\.)\s*\d+", lowered):
        return RegionType.CAPTION
    if re.match(r"^table\s*\d+", lowered):
        return RegionType.CAPTION

    top_cutoff = page_height * 0.1
    bottom_cutoff = page_height * 0.9
    if y1 <= top_cutoff:
        return RegionType.HEADER
    if y0 >= bottom_cutoff:
        return RegionType.FOOTER

    if "|" in text and text.count("|") >= 3:
        return RegionType.TABLE
    return RegionType.TEXT
