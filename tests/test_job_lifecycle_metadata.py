from __future__ import annotations

import asyncio
import io
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

import app.main as main
import app.services.job_store as job_store_module
from app.models.schema import JobQueueState, JobStage, JobStatus
from app.services.job_queue import JobQueue
from app.services.job_store import JobStore


def _store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> JobStore:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(job_store_module, "JOBS_DIR", jobs_dir)
    return JobStore()


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for background queue state")


def test_old_job_statuses_get_backward_compatible_lifecycle_defaults() -> None:
    status = JobStatus.model_validate(
        {
            "job_id": "legacy",
            "filename": "legacy.pdf",
            "stage": "complete",
        }
    )

    assert status.settings == {}
    assert status.queue_state == JobQueueState.NONE
    assert status.queue_position is None
    assert status.jobs_ahead is None
    assert status.queued_at is None
    assert status.started_at is None
    assert status.completed_at is None
    assert status.archived_at is None


def test_archived_jobs_are_filterable_without_resetting_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _store(monkeypatch, tmp_path)
    first_id, _ = store.create_job("paper.pdf", settings={"chunk_size": 900})
    store.update_status(first_id, stage=JobStage.COMPLETE)

    archived = store.archive_job(first_id)

    assert archived.archived_at is not None
    assert store.list_jobs(include_archived=False) == []
    assert [job.job_id for job in store.list_jobs(include_archived=True)] == [first_id]
    second_id, _ = store.create_job("paper.pdf")
    assert store.load_status(second_id).attempt == 1
    assert store.unarchive_job(first_id).archived_at is None


def test_reconcile_stale_jobs_cancels_nonterminal_work_and_clears_queue_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _store(monkeypatch, tmp_path)
    queued_id, _ = store.create_job("queued.pdf")
    running_id, _ = store.create_job("running.pdf")
    complete_id, _ = store.create_job("complete.pdf")
    store.update_status(
        queued_id,
        queue_state=JobQueueState.QUEUED,
        queue_position=1,
        jobs_ahead=0,
    )
    store.update_status(
        running_id,
        stage=JobStage.TRANSLATION,
        queue_state=JobQueueState.RUNNING,
    )
    store.update_status(
        complete_id,
        stage=JobStage.COMPLETE,
        queue_state=JobQueueState.RUNNING,
    )

    assert store.reconcile_stale_jobs() == 3

    queued = store.load_status(queued_id)
    running = store.load_status(running_id)
    complete = store.load_status(complete_id)
    assert queued.stage == JobStage.CANCELLED
    assert "before processing" in queued.message
    assert queued.completed_at is not None
    assert running.stage == JobStage.CANCELLED
    assert "during processing" in running.message
    assert running.completed_at is not None
    assert complete.stage == JobStage.COMPLETE
    for status in (queued, running, complete):
        assert status.queue_state == JobQueueState.NONE
        assert status.queue_position is None
        assert status.jobs_ahead is None


def test_application_startup_reconciles_stale_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _store(monkeypatch, tmp_path)
    job_id, _ = store.create_job("interrupted.pdf")
    store.update_status(
        job_id,
        stage=JobStage.TRANSLATION,
        queue_state=JobQueueState.RUNNING,
    )
    monkeypatch.setattr(main, "job_store", store)

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(run_lifespan())

    status = store.load_status(job_id)
    assert status.stage == JobStage.CANCELLED
    assert status.queue_state == JobQueueState.NONE
    assert "application restarted" in status.message


def test_queue_persists_structured_position_and_terminal_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _store(monkeypatch, tmp_path)
    first_id, first_dir = store.create_job("first.pdf")
    second_id, second_dir = store.create_job("second.pdf")
    first_started = threading.Event()
    release_first = threading.Event()

    class ControlledPipeline:
        def run(self, job_id: str, _pdf_path: Path, _settings: dict) -> None:
            if job_id == first_id:
                first_started.set()
                assert release_first.wait(timeout=2)
            store.update_status(
                job_id,
                stage=JobStage.COMPLETE,
                progress=1.0,
                message="Done",
            )

        def cancel_job(self, job_id: str) -> None:
            store.update_status(job_id, stage=JobStage.CANCELLED, progress=1.0)

    queue = JobQueue(store, ControlledPipeline())  # type: ignore[arg-type]
    queue.enqueue(first_id, first_dir / "input.pdf", {"chunk_size": 100})
    assert first_started.wait(timeout=2)
    queue.enqueue(second_id, second_dir / "input.pdf", {"chunk_size": 100})

    second = store.load_status(second_id)
    assert second.queue_state == JobQueueState.QUEUED
    assert second.queue_position == 1
    assert second.jobs_ahead == 1
    assert second.queued_at is not None
    assert queue.cancel_job(second_id) == {"status": "queued_cancelled"}
    cancelled = store.load_status(second_id)
    assert cancelled.stage == JobStage.CANCELLED
    assert cancelled.queue_state == JobQueueState.NONE
    assert cancelled.completed_at is not None

    release_first.set()
    _wait_for(lambda: store.load_status(first_id).completed_at is not None)
    completed = store.load_status(first_id)
    assert completed.stage == JobStage.COMPLETE
    assert completed.queue_state == JobQueueState.NONE
    assert completed.queued_at is not None
    assert completed.started_at is not None
    assert completed.completed_at is not None


def test_archive_routes_filter_and_individual_delete_is_guarded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _store(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "job_store", store)
    job_id, job_dir = store.create_job("result.pdf")

    with pytest.raises(HTTPException) as exc_info:
        main.archive_job(job_id)
    assert exc_info.value.status_code == 409

    store.update_status(job_id, stage=JobStage.COMPLETE)
    assert main.archive_job(job_id)["archived_at"] is not None
    assert main.list_jobs() == []
    assert [job["job_id"] for job in main.list_jobs(include_archived=True)] == [job_id]
    assert main.unarchive_job(job_id)["archived_at"] is None

    class QueueGuard:
        def __init__(self, contains: bool) -> None:
            self._contains = contains

        def contains(self, _job_id: str) -> bool:
            return self._contains

    monkeypatch.setattr(main, "job_queue", QueueGuard(True))
    with pytest.raises(HTTPException) as exc_info:
        main.delete_job(job_id)
    assert exc_info.value.status_code == 409
    assert job_dir.exists()

    monkeypatch.setattr(main, "job_queue", QueueGuard(False))
    assert main.delete_job(job_id) == {"removed": 1, "job_id": job_id}
    assert not job_dir.exists()


def test_create_job_persists_the_submitted_settings_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _store(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "job_store", store)

    class CapturingQueue:
        def __init__(self) -> None:
            self.settings: dict | None = None

        def enqueue(self, _job_id: str, _pdf_path: Path, settings: dict) -> None:
            self.settings = settings

    queue = CapturingQueue()
    monkeypatch.setattr(main, "job_queue", queue)
    upload = UploadFile(file=io.BytesIO(b"%PDF-test"), filename="settings.pdf")

    result = asyncio.run(
        main.create_job(
            files=[upload],
            chunk_size=777,
            model="mlx-community/Qwen3.5-9B-MLX-4bit",
            temperature=0.2,
            top_p=0.8,
            top_k=12,
            min_p=0.05,
            presence_penalty=1.2,
            repetition_penalty=1.1,
            max_tokens=3072,
            output_mode="readable",
            profile_pipeline=True,
            extraction_mode="digital",
            use_local_vlm_repair=True,
            keep_debug_artifacts=True,
        )
    )

    status = store.load_status(result["jobs"][0]["job_id"])
    assert status.settings["chunk_size"] == 777
    assert status.settings["max_tokens"] == 3072
    assert status.settings["extraction_mode"] == "digital"
    assert status.settings["profile_pipeline"] is True
    assert status.settings == queue.settings
