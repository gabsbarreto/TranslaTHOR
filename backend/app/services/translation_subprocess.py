from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from app.services.profiler import PipelineProfiler


def run_translation_subprocess(
    *,
    document_path: Path,
    markdown_path: Path,
    output_document_path: Path,
    output_markdown_path: Path,
    settings: dict,
    on_chunk_started: Callable[[int, int], None] | None = None,
    on_chunk_translated: Callable[[int, int, str], None] | None = None,
    on_table_progress: Callable[[int, int, str], None] | None = None,
    on_process_started: Callable[[subprocess.Popen], None] | None = None,
    on_process_finished: Callable[[subprocess.Popen], None] | None = None,
    profiler: PipelineProfiler | None = None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "app.services.translation_worker",
        "--document",
        str(document_path),
        "--markdown",
        str(markdown_path),
        "--output-document",
        str(output_document_path),
        "--output-markdown",
        str(output_markdown_path),
        "--settings-json",
        json.dumps(settings),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"backend:{env.get('PYTHONPATH', '')}".rstrip(":")

    if profiler is not None:
        with profiler.step("translation_subprocess_total"):
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
                start_new_session=True,
            )
            if on_process_started is not None:
                on_process_started(process)
            try:
                _stream_events(
                    process,
                    on_chunk_started,
                    on_chunk_translated,
                    on_table_progress,
                    output_document_path=output_document_path,
                    output_markdown_path=output_markdown_path,
                    profiler=profiler,
                )
            finally:
                if on_process_finished is not None:
                    on_process_finished(process)
            return

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
        start_new_session=True,
    )
    if on_process_started is not None:
        on_process_started(process)
    try:
        _stream_events(
            process,
            on_chunk_started,
            on_chunk_translated,
            on_table_progress,
            output_document_path=output_document_path,
            output_markdown_path=output_markdown_path,
            profiler=profiler,
        )
    finally:
        if on_process_finished is not None:
            on_process_finished(process)


def _stream_events(
    process: subprocess.Popen,
    on_chunk_started: Callable[[int, int], None] | None,
    on_chunk_translated: Callable[[int, int, str], None] | None,
    on_table_progress: Callable[[int, int, str], None] | None,
    output_document_path: Path,
    output_markdown_path: Path,
    profiler: PipelineProfiler | None,
) -> None:

    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if profiler is not None and event.get("event") == "chunk_translated":
            profiler.record("translation_chunk", 0.0)
        if event.get("event") == "chunk_started" and on_chunk_started is not None:
            on_chunk_started(int(event["index"]), int(event["total"]))
        elif event.get("event") == "chunk_translated" and on_chunk_translated is not None:
            on_chunk_translated(int(event["index"]), int(event["total"]), str(event.get("preview", "")))
        elif event.get("event") == "table_progress" and on_table_progress is not None:
            on_table_progress(int(event["index"]), int(event["total"]), str(event.get("label", "")))

    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Translation subprocess failed with exit code {return_code}: {stderr[-4000:]}")

    if not output_document_path.exists() or not output_markdown_path.exists():
        raise RuntimeError("Translation subprocess finished without writing translated artifacts")
