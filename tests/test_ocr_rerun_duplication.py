from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

import app.main as main
import app.services.job_store as job_store_module
from app.models.regions import CoordinateSpace, OcrRegionResult, OcrResultsPayload, PageRegionPayload, Region, RegionSource, RegionType
from app.models.schema import JobStage
from app.services.job_store import JobStore
from app.services.ocr_region_service import OcrRegionService


def _install_temp_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> JobStore:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(job_store_module, "JOBS_DIR", jobs_dir)
    store = JobStore()
    monkeypatch.setattr(main, "job_store", store)
    monkeypatch.setattr(main, "ocr_region_service", OcrRegionService())
    return store


def _save_region(job_dir: Path, job_id: str) -> None:
    payload = PageRegionPayload(
        pdf_file_id=job_id,
        page_number=1,
        page_width=612,
        page_height=792,
        image_width=1224,
        image_height=1584,
        coordinate_space=CoordinateSpace.NORMALIZED,
        detector="manual",
        regions=[
            Region(
                id="saved-box",
                page_number=1,
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
    main.ocr_region_service.region_store.save_page_regions(job_dir, payload)
    main.ocr_region_service.region_store.save_all_regions(job_dir)


def test_duplicate_for_ocr_rerun_copies_pdf_and_boxes_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _install_temp_store(monkeypatch, tmp_path)
    source_job_id, source_job_dir = store.create_job("paper.pdf")
    (source_job_dir / "input.pdf").write_bytes(b"%PDF source")
    _save_region(source_job_dir, source_job_id)
    main.ocr_region_service.region_store.save_ocr_results(
        source_job_dir,
        OcrResultsPayload(
            pdf_file_id=source_job_id,
            results=[
                OcrRegionResult(
                    pdf_file_id=source_job_id,
                    page_number=1,
                    box_id="saved-box",
                    x0=0.1,
                    y0=0.2,
                    x1=0.8,
                    y1=0.9,
                    box_type=RegionType.TEXT,
                    reading_order=1,
                    ocr_text="old OCR text",
                )
            ],
        ),
    )
    artifacts_dir = source_job_dir / "artifacts"
    (artifacts_dir / "translated.md").write_text("old translation", encoding="utf-8")
    (artifacts_dir / "translated_readable.pdf").write_bytes(b"old pdf")
    store.update_status(
        source_job_id,
        stage=JobStage.COMPLETE,
        progress=1.0,
        artifacts={
            "ocr_regions": str(source_job_dir / "regions" / "all_pages.json"),
            "ocr_results": str(source_job_dir / "ocr_regions" / "results.json"),
            "markdown": str(artifacts_dir / "translated.md"),
            "pdf_readable": str(artifacts_dir / "translated_readable.pdf"),
        },
    )

    response = main.duplicate_job_for_ocr_rerun(source_job_id)

    new_job_id = response["new_job_id"]
    new_job_dir = store.get_job_dir(new_job_id)
    new_status = store.load_status(new_job_id)
    copied_regions = main.ocr_region_service.region_store.load_page_regions(new_job_dir, 1)

    assert new_job_id != source_job_id
    assert response["source_job_id"] == source_job_id
    assert response["message"] == "Created editable OCR rerun from previous job."
    assert (new_job_dir / "input.pdf").read_bytes() == b"%PDF source"
    assert copied_regions is not None
    assert copied_regions.pdf_file_id == new_job_id
    assert copied_regions.regions[0].id == "saved-box"
    assert copied_regions.regions[0].coordinate_space == CoordinateSpace.NORMALIZED
    assert new_status.stage == JobStage.UPLOADED
    assert new_status.progress == 0.0
    assert new_status.error is None
    assert new_status.artifacts["parent_job_id"] == source_job_id
    assert "ocr_regions" in new_status.artifacts
    assert "ocr_results" not in new_status.artifacts
    assert "markdown" not in new_status.artifacts
    assert "pdf_readable" not in new_status.artifacts
    assert not (new_job_dir / "ocr_regions" / "results.json").exists()
    assert not (new_job_dir / "artifacts" / "translated.md").exists()
    assert store.load_status(source_job_id).stage == JobStage.COMPLETE
    assert (source_job_dir / "artifacts" / "translated_readable.pdf").read_bytes() == b"old pdf"


def test_duplicate_for_ocr_rerun_allows_no_saved_boxes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _install_temp_store(monkeypatch, tmp_path)
    source_job_id, source_job_dir = store.create_job("paper.pdf")
    (source_job_dir / "input.pdf").write_bytes(b"%PDF source")

    response = main.duplicate_job_for_ocr_rerun(source_job_id)

    new_job_dir = store.get_job_dir(response["new_job_id"])
    new_status = store.load_status(response["new_job_id"])
    assert (new_job_dir / "input.pdf").exists()
    assert main.ocr_region_service.region_store.list_page_regions(new_job_dir) == []
    assert "ocr_regions" not in new_status.artifacts


def test_duplicate_for_ocr_rerun_blocks_active_processing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _install_temp_store(monkeypatch, tmp_path)
    source_job_id, source_job_dir = store.create_job("paper.pdf")
    (source_job_dir / "input.pdf").write_bytes(b"%PDF source")
    store.update_status(source_job_id, stage=JobStage.TRANSLATION)

    with pytest.raises(HTTPException) as exc:
        main.duplicate_job_for_ocr_rerun(source_job_id)

    assert exc.value.status_code == 409


def test_duplicate_for_ocr_rerun_requires_source_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _install_temp_store(monkeypatch, tmp_path)
    source_job_id, _source_job_dir = store.create_job("paper.pdf")

    with pytest.raises(HTTPException) as exc:
        main.duplicate_job_for_ocr_rerun(source_job_id)

    assert exc.value.status_code == 404
