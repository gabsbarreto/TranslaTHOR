from __future__ import annotations

from app.models.regions import CoordinateSpace, Region


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def normalize_region(region: Region, page_width: float, page_height: float) -> Region:
    if region.coordinate_space == CoordinateSpace.NORMALIZED:
        return region.model_copy(
            update={
                "x0": clamp01(region.x0),
                "y0": clamp01(region.y0),
                "x1": clamp01(region.x1),
                "y1": clamp01(region.y1),
                "coordinate_space": CoordinateSpace.NORMALIZED,
            }
        )

    width = max(float(page_width), 1.0)
    height = max(float(page_height), 1.0)
    x0 = clamp01(region.x0 / width)
    y0 = clamp01(region.y0 / height)
    x1 = clamp01(region.x1 / width)
    y1 = clamp01(region.y1 / height)
    return region.model_copy(
        update={
            "x0": min(x0, x1),
            "y0": min(y0, y1),
            "x1": max(x0, x1),
            "y1": max(y0, y1),
            "coordinate_space": CoordinateSpace.NORMALIZED,
        }
    )


def denormalize_region(region: Region, page_width: float, page_height: float) -> Region:
    if region.coordinate_space == CoordinateSpace.PDF:
        return region

    width = max(float(page_width), 1.0)
    height = max(float(page_height), 1.0)
    x0 = clamp01(region.x0) * width
    y0 = clamp01(region.y0) * height
    x1 = clamp01(region.x1) * width
    y1 = clamp01(region.y1) * height
    return region.model_copy(
        update={
            "x0": min(x0, x1),
            "y0": min(y0, y1),
            "x1": max(x0, x1),
            "y1": max(y0, y1),
            "coordinate_space": CoordinateSpace.PDF,
        }
    )


def normalized_to_image_bbox(
    *, x0: float, y0: float, x1: float, y1: float, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    ix0 = int(round(clamp01(x0) * image_width))
    iy0 = int(round(clamp01(y0) * image_height))
    ix1 = int(round(clamp01(x1) * image_width))
    iy1 = int(round(clamp01(y1) * image_height))
    left = max(0, min(ix0, ix1))
    top = max(0, min(iy0, iy1))
    right = min(image_width, max(ix0, ix1))
    bottom = min(image_height, max(iy0, iy1))

    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)

    return left, top, right, bottom


def pad_normalized_bbox(
    *, x0: float, y0: float, x1: float, y1: float, padding: float
) -> tuple[float, float, float, float]:
    left = min(clamp01(x0), clamp01(x1))
    top = min(clamp01(y0), clamp01(y1))
    right = max(clamp01(x0), clamp01(x1))
    bottom = max(clamp01(y0), clamp01(y1))
    pad = max(0.0, float(padding))
    width = max(right - left, 0.0)
    height = max(bottom - top, 0.0)
    return (
        clamp01(left - width * pad),
        clamp01(top - height * pad),
        clamp01(right + width * pad),
        clamp01(bottom + height * pad),
    )


def image_to_normalized_bbox(
    *, left: float, top: float, right: float, bottom: float, image_width: float, image_height: float
) -> tuple[float, float, float, float]:
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    x0 = clamp01(left / width)
    y0 = clamp01(top / height)
    x1 = clamp01(right / width)
    y1 = clamp01(bottom / height)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def canvas_to_image_bbox(
    *, left: float, top: float, right: float, bottom: float, canvas_width: float, canvas_height: float, image_width: float, image_height: float
) -> tuple[float, float, float, float]:
    cw = max(float(canvas_width), 1.0)
    ch = max(float(canvas_height), 1.0)
    scale_x = float(image_width) / cw
    scale_y = float(image_height) / ch
    return (
        left * scale_x,
        top * scale_y,
        right * scale_x,
        bottom * scale_y,
    )


def image_to_canvas_bbox(
    *, left: float, top: float, right: float, bottom: float, canvas_width: float, canvas_height: float, image_width: float, image_height: float
) -> tuple[float, float, float, float]:
    iw = max(float(image_width), 1.0)
    ih = max(float(image_height), 1.0)
    scale_x = float(canvas_width) / iw
    scale_y = float(canvas_height) / ih
    return (
        left * scale_x,
        top * scale_y,
        right * scale_x,
        bottom * scale_y,
    )
