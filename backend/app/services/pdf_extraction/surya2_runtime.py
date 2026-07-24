from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, cast

from app.config import (
    BASE_DIR,
    DEFAULT_SURYA2_PYTHON,
    DEFAULT_SURYA2_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


class Surya2Runtime:
    """Own one persistent Surya worker and its shared llama-server child."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        worker_path: Path | None = None,
        request_timeout: int = DEFAULT_SURYA2_REQUEST_TIMEOUT,
    ) -> None:
        self.python_executable = python_executable or DEFAULT_SURYA2_PYTHON
        self.worker_path = worker_path or BASE_DIR / "scripts" / "surya2_worker.py"
        self.request_timeout = request_timeout
        self._process: subprocess.Popen[str] | None = None
        self._event_queue: queue.Queue[dict] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self.worker_metadata: dict = {}
        atexit.register(self.close)

    @property
    def process(self) -> subprocess.Popen[str] | None:
        process = self._process
        return process if process is not None and process.poll() is None else None

    def run(
        self,
        *,
        image_paths: list[Path],
        output_path: Path,
        strategy: str,
        cancel_requested: Callable[[], bool] | None = None,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
        on_process_finished: Callable[[subprocess.Popen], None] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> dict:
        with self._lock:
            process = self._ensure_started()
            if on_process_started is not None:
                on_process_started(process)
            request_id = uuid.uuid4().hex
            request = {
                "action": "run",
                "request_id": request_id,
                "image_paths": [str(path) for path in image_paths],
                "output_path": str(output_path),
                "strategy": strategy,
            }
            assert process.stdin is not None
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            deadline = time.monotonic() + self.request_timeout
            try:
                while True:
                    if cancel_requested is not None and cancel_requested():
                        self._stop_locked()
                        raise RuntimeError("Cancelled by user")
                    if process.poll() is not None:
                        self._process = None
                        raise RuntimeError(
                            f"Surya 2 worker exited unexpectedly with code {process.returncode}."
                        )
                    if time.monotonic() > deadline:
                        self._stop_locked()
                        raise RuntimeError(
                            f"Surya 2 timed out after {self.request_timeout} seconds."
                        )
                    event = self._read_event(timeout=0.2)
                    if event is None:
                        continue
                    if on_event is not None:
                        on_event(event)
                    if str(event.get("request_id", "")) != request_id:
                        continue
                    if event.get("event") == "request_error":
                        raise RuntimeError(
                            f"Surya 2 worker failed: {event.get('error', 'unknown error')}"
                        )
                    if event.get("event") == "request_complete":
                        if not output_path.exists():
                            raise RuntimeError(
                                "Surya 2 worker completed without writing its result file."
                            )
                        return cast(
                            dict[str, Any],
                            json.loads(output_path.read_text(encoding="utf-8")),
                        )
            finally:
                if on_process_finished is not None:
                    on_process_finished(process)

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        python_path = Path(self.python_executable).expanduser()
        if not python_path.exists():
            raise RuntimeError(
                "Surya 2 requires its isolated Python environment. "
                "Run `bash scripts/setup_surya2_runtime.sh` or set SURYA2_PYTHON."
            )
        if shutil.which("llama-server") is None:
            raise RuntimeError(
                "Surya 2 requires llama-server. Install it with `brew install llama.cpp`."
            )
        if not self.worker_path.exists():
            raise RuntimeError(f"Surya 2 worker is missing: {self.worker_path}")

        env = os.environ.copy()
        env["SURYA_INFERENCE_BACKEND"] = "llamacpp"
        env["SURYA_INFERENCE_KEEP_ALIVE"] = "0"
        # Surya 0.22.1's generated layout grammar contains ``\d`` escapes that
        # llama.cpp 10090 rejects. Unguided layout output is still parsed and
        # validated by Surya's LayoutPredictor.
        env["SURYA_GUIDED_LAYOUT"] = "false"
        process = subprocess.Popen(
            [str(python_path), str(self.worker_path), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        self._process = process
        event_queue: queue.Queue[dict] = queue.Queue()
        self._event_queue = event_queue
        self._reader_thread = threading.Thread(
            target=self._pump_events,
            args=(process, event_queue),
            name=f"surya2-events-{process.pid}",
            daemon=True,
        )
        self._reader_thread.start()
        deadline = time.monotonic() + 120
        while time.monotonic() <= deadline:
            if process.poll() is not None:
                self._process = None
                raise RuntimeError(
                    f"Surya 2 worker failed to start (exit code {process.returncode})."
                )
            event = self._read_event(timeout=0.2)
            if event is None:
                continue
            logger.info("Surya 2 worker startup: %s", event)
            if event.get("event") == "worker_ready":
                self.worker_metadata = event
                return process
        self._stop_locked()
        raise RuntimeError("Surya 2 worker did not become ready within 120 seconds.")

    def _read_event(
        self,
        *,
        timeout: float,
    ) -> dict | None:
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @staticmethod
    def _pump_events(
        process: subprocess.Popen[str],
        event_queue: queue.Queue[dict],
    ) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event_queue.put(cast(dict[str, Any], json.loads(stripped)))
            except json.JSONDecodeError:
                logger.info("Surya 2 worker: %s", stripped)
                event_queue.put({"event": "worker_log", "message": stripped})

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write('{"action":"shutdown"}\n')
                process.stdin.flush()
            process.wait(timeout=10)
            return
        except Exception:
            pass
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
