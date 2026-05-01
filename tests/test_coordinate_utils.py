from __future__ import annotations

from app.models.regions import CoordinateSpace, Region, RegionSource, RegionType
from app.services.coordinate_utils import (
    canvas_to_image_bbox,
    denormalize_region,
    image_to_canvas_bbox,
    image_to_normalized_bbox,
    normalize_region,
    normalized_to_image_bbox,
    pad_normalized_bbox,
)


def _region_pdf() -> Region:
    return Region(
        id="r1",
        page_number=1,
        x0=72.0,
        y0=144.0,
        x1=216.0,
        y1=288.0,
        coordinate_space=CoordinateSpace.PDF,
        type=RegionType.TEXT,
        selected=True,
        reading_order=1,
        source=RegionSource.MANUAL,
    )


def test_normalize_denormalize_roundtrip() -> None:
    page_w, page_h = 612.0, 792.0
    region = _region_pdf()

    normalized = normalize_region(region, page_w, page_h)
    assert normalized.coordinate_space == CoordinateSpace.NORMALIZED

    restored = denormalize_region(normalized, page_w, page_h)
    assert restored.coordinate_space == CoordinateSpace.PDF
    assert abs(restored.x0 - region.x0) < 1e-6
    assert abs(restored.y0 - region.y0) < 1e-6
    assert abs(restored.x1 - region.x1) < 1e-6
    assert abs(restored.y1 - region.y1) < 1e-6


def test_normalized_to_image_and_back() -> None:
    bbox = normalized_to_image_bbox(
        x0=0.125,
        y0=0.25,
        x1=0.625,
        y1=0.75,
        image_width=800,
        image_height=1000,
    )
    assert bbox == (100, 250, 500, 750)

    x0, y0, x1, y1 = image_to_normalized_bbox(
        left=bbox[0],
        top=bbox[1],
        right=bbox[2],
        bottom=bbox[3],
        image_width=800,
        image_height=1000,
    )
    assert (x0, y0, x1, y1) == (0.125, 0.25, 0.625, 0.75)


def test_pad_normalized_bbox_expands_and_clamps() -> None:
    bbox = pad_normalized_bbox(x0=0.1, y0=0.2, x1=0.6, y1=0.7, padding=0.1)
    assert bbox == (0.05, 0.15000000000000002, 0.65, 0.75)

    clamped = pad_normalized_bbox(x0=0.0, y0=0.0, x1=0.2, y1=0.2, padding=1.0)
    assert clamped == (0.0, 0.0, 0.4, 0.4)


def test_canvas_image_coordinate_conversion() -> None:
    image_bbox = canvas_to_image_bbox(
        left=10,
        top=20,
        right=110,
        bottom=220,
        canvas_width=200,
        canvas_height=400,
        image_width=1000,
        image_height=2000,
    )
    assert image_bbox == (50.0, 100.0, 550.0, 1100.0)

    canvas_bbox = image_to_canvas_bbox(
        left=image_bbox[0],
        top=image_bbox[1],
        right=image_bbox[2],
        bottom=image_bbox[3],
        canvas_width=200,
        canvas_height=400,
        image_width=1000,
        image_height=2000,
    )
    assert canvas_bbox == (10.0, 20.0, 110.0, 220.0)
