from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.models.schema import BoundingBox


@dataclass(frozen=True)
class BBoxConversion:
    bbox: BoundingBox | None
    metadata: dict[str, Any]


def convert_bbox_to_pdf(
    bbox: BoundingBox | None,
    *,
    page_width: float,
    page_height: float,
    metadata: dict[str, Any] | None = None,
) -> BBoxConversion:
    """Convert an extraction bbox into top-left-origin PDF page coordinates.

    Marker normally emits PDF points. Surya emits coordinates in rendered-page
    pixels, with the source image dimensions recorded on each block. The returned
    metadata records every assumption and scale so persisted figure assets remain
    auditable.
    """

    block_metadata = metadata or {}
    audit: dict[str, Any] = {
        "pdf_page_width": float(page_width),
        "pdf_page_height": float(page_height),
        "conversion": "invalid",
        "clipped_to_page": False,
    }
    if bbox is None:
        audit["reason"] = "missing_bbox"
        return BBoxConversion(None, audit)
    if page_width <= 0 or page_height <= 0:
        audit["reason"] = "invalid_page_dimensions"
        return BBoxConversion(None, audit)

    values = [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]
    audit["source_bbox"] = values
    if not all(math.isfinite(value) for value in values):
        audit["reason"] = "non_finite_bbox"
        return BBoxConversion(None, audit)

    x0, y0, x1, y1 = values
    if x1 < x0:
        x0, x1 = x1, x0
        audit["normalized_inverted_x"] = True
    if y1 < y0:
        y0, y1 = y1, y0
        audit["normalized_inverted_y"] = True

    source_width, source_height, source_space = _source_dimensions(block_metadata)
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.01:
        source_width = 1.0
        source_height = 1.0
        source_space = "normalized_page_fraction"

    if source_width and source_height:
        scale_x = page_width / source_width
        scale_y = page_height / source_height
        x0 *= scale_x
        x1 *= scale_x
        y0 *= scale_y
        y1 *= scale_y
        audit.update(
            {
                "source_space": source_space,
                "source_page_width": source_width,
                "source_page_height": source_height,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "conversion": "scaled_to_pdf_points",
            }
        )
    else:
        audit.update(
            {
                "source_space": source_space,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "conversion": "pdf_points_assumed",
            }
        )

    if x1 <= 0 or y1 <= 0 or x0 >= page_width or y0 >= page_height:
        audit["reason"] = "bbox_outside_page"
        return BBoxConversion(None, audit)

    clipped = (
        max(0.0, x0),
        max(0.0, y0),
        min(float(page_width), x1),
        min(float(page_height), y1),
    )
    if clipped != (x0, y0, x1, y1):
        audit["clipped_to_page"] = True
    x0, y0, x1, y1 = clipped
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        audit["reason"] = "empty_or_tiny_bbox"
        return BBoxConversion(None, audit)

    converted = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    audit["pdf_bbox"] = converted.model_dump()
    audit.pop("reason", None)
    return BBoxConversion(converted, audit)


def bbox_area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.x1 - bbox.x0) * max(0.0, bbox.y1 - bbox.y0)


def bbox_intersection_area(first: BoundingBox, second: BoundingBox) -> float:
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection = bbox_intersection_area(first, second)
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _source_dimensions(metadata: dict[str, Any]) -> tuple[float | None, float | None, str]:
    candidates = (
        ("surya_page_width", "surya_page_height", "surya_rendered_pixels"),
        ("marker_page_width", "marker_page_height", "marker_page_coordinates"),
        ("source_page_width", "source_page_height", "source_page_coordinates"),
    )
    for width_key, height_key, label in candidates:
        width = _positive_float(metadata.get(width_key))
        height = _positive_float(metadata.get(height_key))
        if width and height:
            return width, height, label

    coordinate_space = metadata.get("coordinate_space")
    if isinstance(coordinate_space, dict):
        width = _positive_float(coordinate_space.get("width"))
        height = _positive_float(coordinate_space.get("height"))
        if width and height:
            return width, height, str(coordinate_space.get("name") or "declared_coordinates")
    return None, None, "pdf_points_assumed"


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None
