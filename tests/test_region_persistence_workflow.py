from __future__ import annotations

from pathlib import Path

import pytest

from app.models.inspection import PageInspection, PdfInspection
from app.models.regions import CoordinateSpace, PageRegionPayload, Region, RegionSource, RegionType
from app.services.ocr_region_service import DEFAULT_BOX_INSET_RATIO, OcrRegionService


def _payload(job_id: str, page: int) -> PageRegionPayload:
    return PageRegionPayload(
        pdf_file_id=job_id,
        page_number=page,
        page_width=612,
        page_height=792,
        image_width=1224,
        image_height=1584,
        coordinate_space=CoordinateSpace.NORMALIZED,
        detector="manual",
        regions=[
            Region(
                id="saved-box",
                page_number=page,
                x0=0.1,
                y0=0.2,
                x1=0.8,
                y1=0.9,
                coordinate_space=CoordinateSpace.NORMALIZED,
                type=RegionType.TEXT,
                selected=True,
                reading_order=1,
                source=RegionSource.MANUAL,
            )
        ],
    )


def test_saved_regions_reload_without_screen_pixel_dependency(tmp_path: Path) -> None:
    service = OcrRegionService()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    service.save_regions(job_dir=job_dir, payload=_payload("job-1", 1))

    reloaded = service.region_store.load_page_regions(job_dir, 1)

    assert reloaded is not None
    assert reloaded.regions[0].id == "saved-box"
    assert reloaded.regions[0].coordinate_space == CoordinateSpace.NORMALIZED
    assert reloaded.regions[0].x0 == 0.1
    assert reloaded.regions[0].x1 == 0.8


def test_get_or_detect_returns_saved_regions_even_with_different_render_size(tmp_path: Path) -> None:
    service = OcrRegionService()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    service.save_regions(job_dir=job_dir, payload=_payload("job-1", 1))

    def fail_render(**_kwargs):
        raise AssertionError("saved region reload should not rerender or regenerate boxes")

    service.render_page_image = fail_render  # type: ignore[assignment]

    payload = service.get_or_detect_regions(
        pdf_file_id="job-1",
        pdf_path=tmp_path / "input.pdf",
        job_dir=job_dir,
        page_number=1,
        dpi=300,
        refresh=False,
    )

    assert payload.regions[0].id == "saved-box"
    assert payload.image_width == 1224


def test_refresh_cannot_overwrite_saved_regions_without_confirmation(tmp_path: Path) -> None:
    service = OcrRegionService()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    service.save_regions(job_dir=job_dir, payload=_payload("job-1", 1))

    with pytest.raises(ValueError, match="Saved boxes already exist"):
        service.get_or_detect_regions(
            pdf_file_id="job-1",
            pdf_path=tmp_path / "input.pdf",
            job_dir=job_dir,
            page_number=1,
            refresh=True,
            replace_saved=False,
        )


def test_confirmed_refresh_replaces_saved_regions(tmp_path: Path) -> None:
    service = OcrRegionService()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    service.save_regions(job_dir=job_dir, payload=_payload("job-1", 1))

    service.inspect_pdf = lambda _pdf_path: PdfInspection(  # type: ignore[assignment]
        filename="input.pdf",
        title=None,
        author=None,
        page_count=1,
        pages=[
            PageInspection(
                page_number=1,
                width=612,
                height=792,
                text_length=0,
                embedded_text_quality=0,
                has_embedded_text=False,
            )
        ],
    )
    service.render_page_image = lambda **_kwargs: (tmp_path / "page.png", 2000, 2600)  # type: ignore[assignment]

    payload = service.get_or_detect_regions(
        pdf_file_id="job-1",
        pdf_path=tmp_path / "input.pdf",
        job_dir=job_dir,
        page_number=1,
        refresh=True,
        detailed=False,
        replace_saved=True,
    )

    assert payload.regions[0].id == "default-full-page-1"
    assert payload.regions[0].x0 == DEFAULT_BOX_INSET_RATIO
    assert payload.regions[0].y0 == DEFAULT_BOX_INSET_RATIO
    assert payload.regions[0].x1 == 1.0 - DEFAULT_BOX_INSET_RATIO
    assert payload.regions[0].y1 == 1.0 - DEFAULT_BOX_INSET_RATIO
    assert payload.image_width == 2000


def test_default_regions_are_slightly_inset_for_easy_resizing() -> None:
    region = OcrRegionService()._default_full_page_region(1)

    assert region.x0 == DEFAULT_BOX_INSET_RATIO
    assert region.y0 == DEFAULT_BOX_INSET_RATIO
    assert region.x1 == 1.0 - DEFAULT_BOX_INSET_RATIO
    assert region.y1 == 1.0 - DEFAULT_BOX_INSET_RATIO
    assert region.selected is True
