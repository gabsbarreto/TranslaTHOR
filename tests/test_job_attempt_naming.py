from __future__ import annotations

from app.services import job_store as job_store_module
from app.services.job_store import JobStore


def test_create_job_uses_explorer_style_attempt_suffix(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(job_store_module, "JOBS_DIR", tmp_path)
    store = JobStore()

    job1, _ = store.create_job("file.pdf")
    job2, _ = store.create_job("file.pdf")
    job3, _ = store.create_job("file.pdf")
    other, _ = store.create_job("other.pdf")

    status1 = store.load_status(job1)
    status2 = store.load_status(job2)
    status3 = store.load_status(job3)
    other_status = store.load_status(other)

    assert status1.filename == "file.pdf"
    assert status2.filename == "file (1).pdf"
    assert status3.filename == "file (2).pdf"
    assert status1.source_filename == "file.pdf"
    assert status2.source_filename == "file.pdf"
    assert status3.source_filename == "file.pdf"
    assert status1.attempt == 0
    assert status2.attempt == 1
    assert status3.attempt == 2
    assert other_status.filename == "other.pdf"
