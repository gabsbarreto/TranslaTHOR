from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import JOBS_DIR
from app.models.schema import JobQueueState, JobStage, JobStatus


TERMINAL_JOB_STAGES = {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}


class JobStore:
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
        self.status_path(job_id).write_text(status.model_dump_json(indent=2), encoding="utf-8")

    def load_status(self, job_id: str) -> JobStatus:
        return JobStatus.model_validate_json(self.status_path(job_id).read_text(encoding="utf-8"))

    def update_status(self, job_id: str, **updates: object) -> JobStatus:
        current = self.load_status(job_id)
        updated = current.model_copy(update=updates)
        self.save_status(job_id, updated)
        return updated

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
        job_dir = self.get_job_dir(job_id)
        if not job_dir.exists() or not job_dir.is_dir():
            return False
        shutil.rmtree(job_dir)
        return True

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
        for job_dir in JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            shutil.rmtree(job_dir)
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
            shutil.rmtree(job_dir)
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
