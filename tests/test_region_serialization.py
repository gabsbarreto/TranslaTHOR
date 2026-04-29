from __future__ import annotations

from app.models.regions import CoordinateSpace, PageRegionPayload, Region, RegionSource, RegionType


def test_page_region_payload_roundtrip_json() -> None:
    payload = PageRegionPayload(
        pdf_file_id="job-123",
        page_number=2,
        page_width=595.0,
        page_height=842.0,
        image_width=1200,
        image_height=1700,
        coordinate_space=CoordinateSpace.NORMALIZED,
        detector="pymupdf",
        regions=[
            Region(
                id="r-2-1",
                page_number=2,
                x0=0.1,
                y0=0.1,
                x1=0.9,
                y1=0.2,
                coordinate_space=CoordinateSpace.NORMALIZED,
                type=RegionType.HEADER,
                selected=False,
                reading_order=1,
                source=RegionSource.PYMUPDF,
            )
        ],
    )

    encoded = payload.model_dump_json()
    decoded = PageRegionPayload.model_validate_json(encoded)

    assert decoded.pdf_file_id == "job-123"
    assert decoded.page_number == 2
    assert decoded.coordinate_space == CoordinateSpace.NORMALIZED
    assert len(decoded.regions) == 1
    assert decoded.regions[0].type == RegionType.HEADER
    assert decoded.regions[0].source == RegionSource.PYMUPDF


def test_default_full_page_region_values_are_serializable() -> None:
    region = Region(
        id="default-full-page-1",
        page_number=1,
        x0=0.0,
        y0=0.0,
        x1=1.0,
        y1=1.0,
        coordinate_space=CoordinateSpace.NORMALIZED,
        type=RegionType.PAGE,
        selected=True,
        reading_order=1,
        source=RegionSource.DEFAULT_FULL_PAGE,
    )

    decoded = Region.model_validate_json(region.model_dump_json())

    assert decoded.type == RegionType.PAGE
    assert decoded.source == RegionSource.DEFAULT_FULL_PAGE
