from __future__ import annotations

from app.models.schema import JobStage, JobStatus
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
    assert status1.created_at is not None
    assert other_status.filename == "other.pdf"


def test_list_jobs_uses_chronological_creation_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(job_store_module, "JOBS_DIR", tmp_path)
    store = JobStore()

    for job_id, filename, created_at in [
        ("job-newest", "newest.pdf", "2026-01-01T12:00:03Z"),
        ("job-oldest", "oldest.pdf", "2026-01-01T12:00:01Z"),
        ("job-middle", "middle.pdf", "2026-01-01T12:00:02Z"),
    ]:
        (tmp_path / job_id).mkdir()
        store.save_status(
            job_id,
            JobStatus(
                job_id=job_id,
                filename=filename,
                source_filename=filename,
                created_at=created_at,
                stage=JobStage.UPLOADED,
            ),
        )

    assert [status.job_id for status in store.list_jobs()] == [
        "job-oldest",
        "job-middle",
        "job-newest",
    ]
