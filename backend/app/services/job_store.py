from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import JOBS_DIR
from app.models.schema import JobQueueState, JobStage, JobStatus


TERMINAL_JOB_STAGES = {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}


@dataclass
class _JobLockEntry:
    lock: threading.RLock
    users: int = 0
    retired: bool = False


class JobStore:
    def __init__(self) -> None:
        self._locks_guard = threading.Lock()
        self._status_locks: dict[str, _JobLockEntry] = {}
        self._artifact_generation_locks: dict[str, _JobLockEntry] = {}

    def create_job(
        self,
        filename: str,
        *,
        settings: dict | None = None,
    ) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex
        job_dir = JOBS_DIR / job_id
        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        source_filename = filename
        attempt = self._next_attempt_for_source_filename(source_filename)
        display_filename = self._format_attempt_filename(source_filename, attempt)
        status = JobStatus(
            job_id=job_id,
            filename=display_filename,
            source_filename=source_filename,
            attempt=attempt,
            created_at=self.utc_now(),
            stage=JobStage.UPLOADED,
            progress=0.0,
            settings=dict(settings or {}),
        )
        self.save_status(job_id, status)
        return job_id, job_dir

    def status_path(self, job_id: str) -> Path:
        return JOBS_DIR / job_id / "status.json"

    def get_job_dir(self, job_id: str) -> Path:
        return JOBS_DIR / job_id

    def save_status(self, job_id: str, status: JobStatus) -> None:
        with self._status_lock(job_id):
            self._save_status_unlocked(job_id, status)

    def load_status(self, job_id: str) -> JobStatus:
        with self._status_lock(job_id):
            return self._load_status_unlocked(job_id)

    def update_status(self, job_id: str, **updates: object) -> JobStatus:
        with self._status_lock(job_id):
            current = self._load_status_unlocked(job_id)
            updated = current.model_copy(update=updates)
            self._save_status_unlocked(job_id, updated)
            return updated

    def merge_status(
        self,
        job_id: str,
        *,
        artifacts: dict[str, str] | None = None,
        translation: dict[str, Any] | None = None,
        translation_warnings: list[str] | None = None,
        **updates: object,
    ) -> JobStatus:
        """Atomically merge generated metadata into the latest persisted status.

        Artifact generation can take long enough for another status update to
        land after a caller first reads the job.  Accepting patches instead of
        complete dictionaries prevents that caller from replacing newer
        artifact or translation metadata with a stale snapshot.
        """

        with self._status_lock(job_id):
            current = self._load_status_unlocked(job_id)
            merged_updates = dict(updates)
            if artifacts is not None:
                merged_artifacts = dict(current.artifacts)
                merged_artifacts.update(artifacts)
                merged_updates["artifacts"] = merged_artifacts
            if translation is not None or translation_warnings:
                merged_translation = dict(current.translation)
                if translation is not None:
                    merged_translation.update(translation)
                if translation_warnings:
                    existing_warnings = merged_translation.get("warnings")
                    warnings = list(existing_warnings) if isinstance(existing_warnings, list) else []
                    for warning in translation_warnings:
                        if warning not in warnings:
                            warnings.append(warning)
                    merged_translation["warnings"] = warnings
                merged_updates["translation"] = merged_translation
            updated = current.model_copy(update=merged_updates)
            self._save_status_unlocked(job_id, updated)
            return updated

    @contextmanager
    def artifact_generation_lock(self, job_id: str) -> Iterator[None]:
        """Serialize all on-demand PDF generation for one job."""

        with self._artifact_generation_lock(job_id):
            yield

    @contextmanager
    def _status_lock(self, job_id: str) -> Iterator[None]:
        with self._managed_job_lock(self._status_locks, job_id):
            yield

    @contextmanager
    def _artifact_generation_lock(self, job_id: str) -> Iterator[None]:
        with self._managed_job_lock(self._artifact_generation_locks, job_id):
            yield

    @contextmanager
    def _managed_job_lock(
        self,
        entries: dict[str, _JobLockEntry],
        job_id: str,
    ) -> Iterator[None]:
        """Borrow a per-job lock without pruning it while callers are queued."""

        with self._locks_guard:
            entry = entries.setdefault(job_id, _JobLockEntry(lock=threading.RLock()))
            entry.users += 1
        try:
            with entry.lock:
                yield
        finally:
            with self._locks_guard:
                entry.users -= 1
                if entry.retired and entry.users == 0 and entries.get(job_id) is entry:
                    entries.pop(job_id, None)

    def _retire_job_locks(self, job_id: str) -> None:
        """Remove idle lock entries after deletion, once queued borrowers finish."""

        with self._locks_guard:
            for entries in (self._status_locks, self._artifact_generation_locks):
                entry = entries.get(job_id)
                if entry is None:
                    continue
                entry.retired = True
                if entry.users == 0:
                    entries.pop(job_id, None)

    def _load_status_unlocked(self, job_id: str) -> JobStatus:
        return JobStatus.model_validate_json(
            self.status_path(job_id).read_text(encoding="utf-8")
        )

    def _save_status_unlocked(self, job_id: str, status: JobStatus) -> None:
        status_path = self.status_path(job_id)
        temporary_path = status_path.with_name(f".{status_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary_path.write_text(status.model_dump_json(indent=2), encoding="utf-8")
            temporary_path.replace(status_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def list_jobs(self, *, include_archived: bool = True) -> list[JobStatus]:
        items: list[tuple[float, str, JobStatus]] = []
        for status_file in JOBS_DIR.glob("*/status.json"):
            try:
                status = JobStatus.model_validate_json(status_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if status.archived_at and not include_archived:
                continue
            items.append((self._created_sort_key(status, status_file), status.job_id, status))
        return [status for _created_at, _job_id, status in sorted(items)]

    def archive_job(self, job_id: str) -> JobStatus:
        status = self.load_status(job_id)
        if status.stage not in TERMINAL_JOB_STAGES:
            raise ValueError("Only terminal jobs can be archived.")
        if status.archived_at:
            return status
        return self.update_status(job_id, archived_at=self.utc_now())

    def unarchive_job(self, job_id: str) -> JobStatus:
        status = self.load_status(job_id)
        if status.stage not in TERMINAL_JOB_STAGES:
            raise ValueError("Only terminal jobs can be unarchived.")
        if status.archived_at is None:
            return status
        return self.update_status(job_id, archived_at=None)

    def delete_job(self, job_id: str) -> bool:
        removed = False
        # Keep the established artifact -> status lock order used by artifact
        # generation, then retire both entries while every queued borrower still
        # references the same objects. This avoids replacing a live lock with a
        # second lock for the same job.
        with self._artifact_generation_lock(job_id):
            with self._status_lock(job_id):
                job_dir = self.get_job_dir(job_id)
                if job_dir.exists() and job_dir.is_dir():
                    shutil.rmtree(job_dir)
                    removed = True
                self._retire_job_locks(job_id)
        return removed

    def reconcile_stale_jobs(self) -> int:
        """Mark persisted nonterminal jobs as interrupted after an application restart."""
        reconciled = 0
        now = self.utc_now()
        for status in self.list_jobs(include_archived=True):
            queue_updates = {
                "queue_state": JobQueueState.NONE,
                "queue_position": None,
                "jobs_ahead": None,
            }
            if status.stage in TERMINAL_JOB_STAGES:
                if (
                    status.queue_state != JobQueueState.NONE
                    or status.queue_position is not None
                    or status.jobs_ahead is not None
                ):
                    self.update_status(status.job_id, **queue_updates)
                    reconciled += 1
                continue

            if status.stage == JobStage.UPLOADED:
                message = "Cancelled because the application restarted before processing began."
            else:
                message = "Cancelled because the application restarted during processing."
            self.update_status(
                status.job_id,
                stage=JobStage.CANCELLED,
                progress=1.0,
                message=message,
                error=None,
                completed_at=now,
                **queue_updates,
            )
            reconciled += 1
        return reconciled

    def clear_jobs(self) -> int:
        removed = 0
        for job_dir in list(JOBS_DIR.iterdir()):
            if not job_dir.is_dir():
                continue
            if self.delete_job(job_dir.name):
                removed += 1
        return removed

    def clear_jobs_by_stage(self, stages: set[JobStage]) -> int:
        removed = 0
        for status in self.list_jobs():
            if status.stage not in stages:
                continue
            job_dir = self.get_job_dir(status.job_id)
            if not job_dir.exists() or not job_dir.is_dir():
                continue
            if self.delete_job(status.job_id):
                removed += 1
        return removed

    def _next_attempt_for_source_filename(self, source_filename: str) -> int:
        attempts: list[int] = []
        for status in self.list_jobs(include_archived=True):
            candidate = status.source_filename or self._normalize_source_filename(status.filename)
            if candidate != source_filename:
                continue
            attempts.append(int(status.attempt or 0))
        if not attempts:
            return 0
        return max(attempts) + 1

    def _normalize_source_filename(self, filename: str) -> str:
        path = Path(filename)
        suffix = path.suffix
        stem = path.stem
        match = re.match(r"^(?P<base>.*)\s\((?P<n>\d+)\)$", stem)
        if not match:
            return filename
        base = match.group("base")
        if not base:
            return filename
        return f"{base}{suffix}"

    def _format_attempt_filename(self, source_filename: str, attempt: int) -> str:
        if attempt <= 0:
            return source_filename
        path = Path(source_filename)
        suffix = path.suffix
        stem = path.stem
        return f"{stem} ({attempt}){suffix}"

    def utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _created_sort_key(self, status: JobStatus, status_file: Path) -> float:
        if status.created_at:
            try:
                return datetime.fromisoformat(status.created_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        job_dir = status_file.parent
        try:
            return float(getattr(job_dir.stat(), "st_birthtime", status_file.stat().st_mtime))
        except OSError:
            return 0.0
