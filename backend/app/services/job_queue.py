from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from app.models.schema import JobStage
from app.services.job_store import JobStore
from app.services.pipeline import TranslationPipeline


@dataclass
class QueueItem:
    job_id: str
    pdf_path: Path
    settings: dict


class JobQueue:
    def __init__(self, job_store: JobStore, pipeline: TranslationPipeline) -> None:
        self.job_store = job_store
        self.pipeline = pipeline
        self._queue: deque[QueueItem] = deque()
        self._condition = threading.Condition()
        self._active_job_id: str | None = None
        self._worker = threading.Thread(target=self._run, name="job-queue-worker", daemon=True)
        self._worker.start()

    def enqueue(self, job_id: str, pdf_path: Path, settings: dict) -> None:
        with self._condition:
            self._queue.append(QueueItem(job_id=job_id, pdf_path=pdf_path, settings=settings))
            self._refresh_queue_messages_locked()
            self._condition.notify()

    def cancel_job(self, job_id: str) -> dict[str, str]:
        removed_from_queue = False
        is_active = False
        with self._condition:
            if self._active_job_id == job_id:
                is_active = True
            else:
                remaining: deque[QueueItem] = deque()
                for item in self._queue:
                    if item.job_id == job_id and not removed_from_queue:
                        removed_from_queue = True
                        continue
                    remaining.append(item)
                if removed_from_queue:
                    self._queue = remaining
                    self._refresh_queue_messages_locked()
                    self._condition.notify_all()

        if removed_from_queue:
            try:
                self.job_store.update_status(
                    job_id,
                    stage=JobStage.CANCELLED,
                    progress=1.0,
                    message="Cancelled before processing started.",
                    error=None,
                )
            except FileNotFoundError:
                pass
            return {"status": "queued_cancelled"}

        if is_active:
            self.pipeline.cancel_job(job_id)
            return {"status": "active_cancelled"}

        return {"status": "not_found"}

    def stop_all(self) -> dict[str, int]:
        with self._condition:
            queued = list(self._queue)
            self._queue.clear()
            active_job_id = self._active_job_id

        cancelled_queued = 0
        for item in queued:
            try:
                self.job_store.update_status(
                    item.job_id,
                    stage=JobStage.CANCELLED,
                    progress=1.0,
                    message="Cancelled before processing started.",
                    error=None,
                )
                cancelled_queued += 1
            except FileNotFoundError:
                continue

        cancelled_active = 0
        if active_job_id:
            self.pipeline.cancel_job(active_job_id)
            cancelled_active = 1

        return {"queued_cancelled": cancelled_queued, "active_cancelled": cancelled_active}

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue:
                    self._condition.wait()
                item = self._queue.popleft()
                self._active_job_id = item.job_id
                self._refresh_queue_messages_locked()

            try:
                self.pipeline.run(item.job_id, item.pdf_path, item.settings)
            finally:
                with self._condition:
                    self._active_job_id = None
                    self._refresh_queue_messages_locked()
                    self._condition.notify_all()

    def _refresh_queue_messages_locked(self) -> None:
        active_ahead = 1 if self._active_job_id else 0
        for idx, item in enumerate(self._queue):
            queued_ahead = idx + active_ahead
            if queued_ahead > 0:
                message = f"Queued. Waiting for {queued_ahead} job(s) ahead."
            else:
                message = "Queued. Waiting for processing to start."
            try:
                self.job_store.update_status(
                    item.job_id,
                    stage=JobStage.UPLOADED,
                    progress=0.0,
                    message=message,
                    error=None,
                )
            except FileNotFoundError:
                continue
