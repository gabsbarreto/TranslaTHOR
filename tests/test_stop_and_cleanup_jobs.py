from __future__ import annotations

from pathlib import Path

import pytest

import app.main as main
import app.services.job_store as job_store_module
from app.models.schema import JobStage
from app.services.job_store import JobStore


def _install_temp_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> JobStore:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(job_store_module, "JOBS_DIR", jobs_dir)
    store = JobStore()
    monkeypatch.setattr(main, "job_store", store)
    return store


def test_stop_all_marks_interrupted_processing_jobs_cancelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _install_temp_store(monkeypatch, tmp_path)
    ocr_job_id, _ = store.create_job("ocr.pdf")
    uploaded_job_id, _ = store.create_job("draft.pdf")
    complete_job_id, _ = store.create_job("done.pdf")
    store.update_status(ocr_job_id, stage=JobStage.OCR_LAYOUT, progress=0.4)
    store.update_status(uploaded_job_id, stage=JobStage.UPLOADED, progress=0.0)
    store.update_status(complete_job_id, stage=JobStage.COMPLETE, progress=1.0)

    interrupted = main._mark_interrupted_processing_jobs_cancelled()
    drafts = main._mark_uploaded_draft_jobs_cancelled()

    assert interrupted == {"interrupted_cancelled": 1}
    assert drafts == {"draft_cancelled": 1}
    assert store.load_status(ocr_job_id).stage == JobStage.CANCELLED
    assert store.load_status(uploaded_job_id).stage == JobStage.CANCELLED
    assert store.load_status(complete_job_id).stage == JobStage.COMPLETE


def test_cancel_job_marks_draft_upload_cancelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _install_temp_store(monkeypatch, tmp_path)
    job_id, _job_dir = store.create_job("draft.pdf")
    monkeypatch.setattr(main.job_queue, "cancel_job", lambda _job_id: {"status": "not_found"})

    result = main._cancel_job_impl(job_id)

    assert result == {"status": "draft_cancelled"}
    status = store.load_status(job_id)
    assert status.stage == JobStage.CANCELLED
    assert status.message == "Cancelled before OCR started."


def test_cleanup_terminal_removes_cancelled_and_failed_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = _install_temp_store(monkeypatch, tmp_path)
    cancelled_job_id, cancelled_dir = store.create_job("cancelled.pdf")
    failed_job_id, failed_dir = store.create_job("failed.pdf")
    active_job_id, active_dir = store.create_job("active.pdf")
    store.update_status(cancelled_job_id, stage=JobStage.CANCELLED, progress=1.0)
    store.update_status(failed_job_id, stage=JobStage.FAILED, progress=1.0)
    store.update_status(active_job_id, stage=JobStage.OCR_LAYOUT, progress=0.5)

    result = main._clear_terminal_jobs_impl()

    assert result == {"removed": 2}
    assert not cancelled_dir.exists()
    assert not failed_dir.exists()
    assert active_dir.exists()
