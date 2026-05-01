from __future__ import annotations

from app.models.regions import CoordinateSpace, Region, RegionSource, RegionType
from app.services.layout_detectors import _sort_and_reindex


def _region(region_id: str, x0: float, y0: float, x1: float, y1: float) -> Region:
    return Region(
        id=region_id,
        page_number=1,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        coordinate_space=CoordinateSpace.NORMALIZED,
        type=RegionType.TEXT,
        selected=True,
        reading_order=0,
        source=RegionSource.PYMUPDF,
    )


def test_reading_order_prefers_vertical_columns_over_horizontal_rows() -> None:
    regions = [
        _region("left-top", 0.08, 0.10, 0.42, 0.20),
        _region("right-top", 0.58, 0.11, 0.92, 0.21),
        _region("left-mid", 0.08, 0.30, 0.42, 0.40),
        _region("right-mid", 0.58, 0.31, 0.92, 0.41),
        _region("left-bottom", 0.08, 0.55, 0.42, 0.65),
        _region("right-bottom", 0.58, 0.56, 0.92, 0.66),
    ]

    ordered = _sort_and_reindex(regions)

    assert [item.id for item in ordered] == [
        "left-top",
        "left-mid",
        "left-bottom",
        "right-top",
        "right-mid",
        "right-bottom",
    ]
    assert [item.reading_order for item in ordered] == [1, 2, 3, 4, 5, 6]
