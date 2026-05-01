from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.models.regions import CoordinateSpace, PageRegionPayload, Region, RegionSource, RegionType
from app.services.ocr_region_service import OcrRegionService


def test_run_selected_ocr_masks_unselected_page_content(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)

    page_image = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "black").save(page_image)

    service = OcrRegionService()

    payload = PageRegionPayload(
        pdf_file_id="job-1",
        page_number=1,
        page_width=612,
        page_height=792,
        image_width=100,
        image_height=100,
        coordinate_space=CoordinateSpace.NORMALIZED,
        detector="manual",
        regions=[
            Region(
                id="box-1",
                page_number=1,
                x0=0.1,
                y0=0.2,
                x1=0.6,
                y1=0.7,
                coordinate_space=CoordinateSpace.NORMALIZED,
                type=RegionType.TEXT,
                selected=True,
                reading_order=1,
                source=RegionSource.MANUAL,
            )
        ],
    )
    service.region_store.save_page_regions(job_dir, payload)

    def fake_render_page_image(*, pdf_path, job_dir, page_number, dpi):
        _ = (pdf_path, job_dir, page_number, dpi)
        return page_image, 100, 100

    captured_images: list[Image.Image] = []

    def fake_run_ocr_on_images(**kwargs):
        image_paths = kwargs["image_paths"]
        output_names = kwargs["output_names"]
        for image_path in image_paths:
            with Image.open(image_path) as masked:
                captured_images.append(masked.convert("RGB").copy())
        return {name: "example text" for name in output_names}

    service.render_page_image = fake_render_page_image  # type: ignore[assignment]
    service.deepseek.run_ocr_on_images = fake_run_ocr_on_images  # type: ignore[assignment]

    results = service.run_selected_ocr(
        pdf_file_id="job-1",
        pdf_path=tmp_path / "input.pdf",
        job_dir=job_dir,
        dpi=300,
        model_name="model",
        max_tokens=1024,
        prompt="<image>\nconvert",
        crop_mode=True,
        min_crops=1,
        max_crops=2,
        base_size=1024,
        image_size=768,
        skip_repeat=True,
        ngram_size=10,
        ngram_window=50,
    )

    assert [image.size for image in captured_images] == [(100, 100)]
    assert captured_images[0].getpixel((5, 5)) == (255, 255, 255)
    assert captured_images[0].getpixel((20, 30)) == (0, 0, 0)
    assert len(results.results) == 1
    result = results.results[0]
    assert result.box_id == "box-1"
    assert result.ocr_text == "example text"
    assert result.coordinate_space == CoordinateSpace.NORMALIZED
    assert result.metadata["ocr_image_mode"] == "masked_page_padded"
    assert str(result.metadata["ocr_image_path"]).endswith("ocr_regions/masked_pages/p0001_box-1.png")
    assert result.metadata["ocr_image_width"] == 100
    assert result.metadata["ocr_image_height"] == 100
    assert result.metadata["ocr_padding"] == 0.1
    assert result.metadata["raw_crop_width"] == 50
    assert result.metadata["raw_crop_height"] == 50
    assert result.metadata["padded_crop_width"] == 60
    assert result.metadata["padded_crop_height"] == 60


def test_selected_ocr_masked_page_keeps_rendered_size(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)

    page_image = tmp_path / "large-page.png"
    Image.new("RGB", (2000, 2666), "black").save(page_image)

    service = OcrRegionService()
    payload = PageRegionPayload(
        pdf_file_id="job-1",
        page_number=1,
        page_width=612,
        page_height=792,
        image_width=2000,
        image_height=2666,
        coordinate_space=CoordinateSpace.NORMALIZED,
        detector="manual",
        regions=[
            Region(
                id="box-1",
                page_number=1,
                x0=0.25,
                y0=0.25,
                x1=0.75,
                y1=0.75,
                coordinate_space=CoordinateSpace.NORMALIZED,
                type=RegionType.TEXT,
                selected=True,
                reading_order=1,
                source=RegionSource.MANUAL,
            )
        ],
    )
    service.region_store.save_page_regions(job_dir, payload)

    def fake_render_page_image(*, pdf_path, job_dir, page_number, dpi):
        _ = (pdf_path, job_dir, page_number, dpi)
        return page_image, 2000, 2666

    captured_images: list[Image.Image] = []

    def fake_run_ocr_on_images(**kwargs):
        image_paths = kwargs["image_paths"]
        output_names = kwargs["output_names"]
        for image_path in image_paths:
            with Image.open(image_path) as masked:
                captured_images.append(masked.convert("RGB").copy())
        return {name: "example text" for name in output_names}

    service.render_page_image = fake_render_page_image  # type: ignore[assignment]
    service.deepseek.run_ocr_on_images = fake_run_ocr_on_images  # type: ignore[assignment]

    results = service.run_selected_ocr(
        pdf_file_id="job-1",
        pdf_path=tmp_path / "input.pdf",
        job_dir=job_dir,
        dpi=300,
        model_name="model",
        max_tokens=1024,
        prompt="<image>\nconvert",
        crop_mode=True,
        min_crops=1,
        max_crops=2,
        base_size=1024,
        image_size=768,
        skip_repeat=True,
        ngram_size=10,
        ngram_window=50,
    )

    assert [image.size for image in captured_images] == [(2000, 2666)]
    assert captured_images[0].getpixel((20, 20)) == (255, 255, 255)
    assert captured_images[0].getpixel((1000, 1333)) == (0, 0, 0)
    assert results.results[0].metadata["ocr_image_width"] == 2000
    assert results.results[0].metadata["ocr_image_height"] == 2666


def test_selected_ocr_retries_empty_primary_with_contrast_crop(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)

    page_image = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "black").save(page_image)

    service = OcrRegionService()
    payload = PageRegionPayload(
        pdf_file_id="job-1",
        page_number=1,
        page_width=612,
        page_height=792,
        image_width=100,
        image_height=100,
        coordinate_space=CoordinateSpace.NORMALIZED,
        detector="manual",
        regions=[
            Region(
                id="box-1",
                page_number=1,
                x0=0.1,
                y0=0.2,
                x1=0.6,
                y1=0.7,
                coordinate_space=CoordinateSpace.NORMALIZED,
                type=RegionType.TEXT,
                selected=True,
                reading_order=1,
                source=RegionSource.MANUAL,
            )
        ],
    )
    service.region_store.save_page_regions(job_dir, payload)

    def fake_render_page_image(*, pdf_path, job_dir, page_number, dpi):
        _ = (pdf_path, job_dir, page_number, dpi)
        return page_image, 100, 100

    calls: list[list[str]] = []

    def fake_run_ocr_on_images(**kwargs):
        output_names = kwargs["output_names"]
        calls.append(output_names)
        if len(calls) == 1:
            return {name: "" for name in output_names}
        return {name: "retry text" for name in output_names}

    service.render_page_image = fake_render_page_image  # type: ignore[assignment]
    service.deepseek.run_ocr_on_images = fake_run_ocr_on_images  # type: ignore[assignment]

    results = service.run_selected_ocr(
        pdf_file_id="job-1",
        pdf_path=tmp_path / "input.pdf",
        job_dir=job_dir,
        dpi=300,
        model_name="model",
        max_tokens=1024,
        prompt="<image>\nconvert",
        crop_mode=True,
        min_crops=1,
        max_crops=2,
        base_size=1024,
        image_size=768,
        skip_repeat=True,
        ngram_size=10,
        ngram_window=50,
    )

    assert len(calls) == 2
    assert calls[1] == ["p0001_box-1_retry_gray_contrast"]
    result = results.results[0]
    assert result.ocr_text == "retry text"
    assert result.metadata["ocr_retry_used"] is True
    assert result.metadata["ocr_image_mode"] == "snapped_crop_grayscale_contrast_retry"
    assert result.metadata["ocr_retry_mode"] == "snapped_crop_grayscale_contrast"
    assert result.metadata["ocr_retry_coordinate_snap"] == 0.001


def test_selected_ocr_processes_multiple_regions_in_one_model_session(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)

    page_image = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "black").save(page_image)

    service = OcrRegionService()
    payload = PageRegionPayload(
        pdf_file_id="job-1",
        page_number=1,
        page_width=612,
        page_height=792,
        image_width=100,
        image_height=100,
        coordinate_space=CoordinateSpace.NORMALIZED,
        detector="manual",
        regions=[
            Region(
                id="box-1",
                page_number=1,
                x0=0.1,
                y0=0.1,
                x1=0.4,
                y1=0.4,
                coordinate_space=CoordinateSpace.NORMALIZED,
                type=RegionType.TEXT,
                selected=True,
                reading_order=1,
                source=RegionSource.MANUAL,
            ),
            Region(
                id="box-2",
                page_number=1,
                x0=0.5,
                y0=0.5,
                x1=0.9,
                y1=0.9,
                coordinate_space=CoordinateSpace.NORMALIZED,
                type=RegionType.TEXT,
                selected=True,
                reading_order=2,
                source=RegionSource.MANUAL,
            ),
        ],
    )
    service.region_store.save_page_regions(job_dir, payload)

    def fake_render_page_image(*, pdf_path, job_dir, page_number, dpi):
        _ = (pdf_path, job_dir, page_number, dpi)
        return page_image, 100, 100

    calls: list[tuple[list[Path], list[str]]] = []

    def fake_run_ocr_on_images(**kwargs):
        image_paths = kwargs["image_paths"]
        output_names = kwargs["output_names"]
        calls.append((image_paths, output_names))
        return {name: f"text for {name}" for name in output_names}

    service.render_page_image = fake_render_page_image  # type: ignore[assignment]
    service.deepseek.run_ocr_on_images = fake_run_ocr_on_images  # type: ignore[assignment]

    results = service.run_selected_ocr(
        pdf_file_id="job-1",
        pdf_path=tmp_path / "input.pdf",
        job_dir=job_dir,
        dpi=300,
        model_name="model",
        max_tokens=1024,
        prompt="<image>\nconvert",
        crop_mode=True,
        min_crops=1,
        max_crops=2,
        base_size=1024,
        image_size=768,
        skip_repeat=True,
        ngram_size=10,
        ngram_window=50,
    )

    assert [names for _paths, names in calls] == [["p0001_box-1", "p0001_box-2"]]
    assert [len(paths) for paths, _names in calls] == [2]
    assert [item.ocr_text for item in results.results] == ["text for p0001_box-1", "text for p0001_box-2"]
