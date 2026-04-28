from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

from app.config import JOBS_DIR
from app.models.schema import JobStage, JobStatus


class JobStore:
    def create_job(self, filename: str) -> tuple[str, Path]:
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
            stage=JobStage.UPLOADED,
            progress=0.0,
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

    def list_jobs(self) -> list[JobStatus]:
        items: list[JobStatus] = []
        for status_file in JOBS_DIR.glob("*/status.json"):
            try:
                items.append(JobStatus.model_validate_json(status_file.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return sorted(items, key=lambda x: x.job_id, reverse=True)

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
        for status in self.list_jobs():
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
