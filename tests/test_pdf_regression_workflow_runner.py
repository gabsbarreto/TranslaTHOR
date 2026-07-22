from __future__ import annotations

import hashlib
import json
import importlib.util
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import fitz
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_pdf_regression_workflows.py"
SPEC = importlib.util.spec_from_file_location("run_pdf_regression_workflows", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)
CorpusCase = RUNNER.CorpusCase
TranslaTHORClient = RUNNER.TranslaTHORClient
load_corpus_cases = RUNNER.load_corpus_cases
run_regression_workflows = RUNNER.run_regression_workflows
select_cases = RUNNER.select_cases


def _synthetic_pdf_bytes() -> bytes:
    document = fitz.open()
    document.new_page(width=320, height=240)
    try:
        return document.tobytes()
    finally:
        document.close()


def _write_test_corpus(tmp_path: Path) -> tuple[Path, Path]:
    spec_path = tmp_path / "corpus_spec.json"
    corpus_root = tmp_path / "corpus"
    fixture_path = corpus_root / "digital" / "case-one.pdf"
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_bytes(b"%PDF-1.4\nfixture")
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "digital_cases": [
                    {
                        "id": "case-one",
                        "language": "fr",
                        "features": ["table", "figure"],
                    }
                ],
                "scanned_cases": [],
            }
        ),
        encoding="utf-8",
    )
    return spec_path, corpus_root


def _write_corpus_manifest(
    spec_path: Path,
    corpus_root: Path,
    *,
    fixture_sha256: str,
    page_count: int,
) -> None:
    (corpus_root / "manifest.json").write_text(
        json.dumps(
            {
                "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                "digital_cases": [
                    {
                        "id": "case-one",
                        "fixture": {
                            "path": "digital/case-one.pdf",
                            "sha256": fixture_sha256,
                            "page_count": page_count,
                        },
                    }
                ],
                "scanned_cases": [],
            }
        ),
        encoding="utf-8",
    )


def test_runner_rejects_fixture_that_no_longer_matches_manifest_sha256(
    tmp_path: Path,
) -> None:
    spec_path, corpus_root = _write_test_corpus(tmp_path)
    fixture = corpus_root / "digital" / "case-one.pdf"
    fixture.write_bytes(_synthetic_pdf_bytes())
    _write_corpus_manifest(
        spec_path,
        corpus_root,
        fixture_sha256="0" * 64,
        page_count=1,
    )

    with pytest.raises(RUNNER.RegressionRunError, match="manifest sha256"):
        load_corpus_cases(spec_path, corpus_root)


def test_runner_rejects_fixture_page_count_that_differs_from_manifest(
    tmp_path: Path,
) -> None:
    spec_path, corpus_root = _write_test_corpus(tmp_path)
    fixture = corpus_root / "digital" / "case-one.pdf"
    fixture.write_bytes(_synthetic_pdf_bytes())
    _write_corpus_manifest(
        spec_path,
        corpus_root,
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        page_count=2,
    )

    with pytest.raises(RUNNER.RegressionRunError, match="manifest records 2"):
        load_corpus_cases(spec_path, corpus_root)


class _SuccessfulAPIHandler(BaseHTTPRequestHandler):
    events: list[str] = []
    upload_body = b""
    status_requests = 0
    pdf_body = _synthetic_pdf_bytes()

    def log_message(self, _format: str, *args: Any) -> None:
        del args

    def _respond(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        type(self).events.append(f"POST {self.path}")
        content_length = int(self.headers["Content-Length"])
        type(self).upload_body = self.rfile.read(content_length)
        self._respond(
            200,
            json.dumps({"jobs": [{"job_id": "job-123", "filename": "case-one.pdf"}]}).encode(),
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        type(self).events.append(f"GET {self.path}")
        if self.path == "/api/health":
            self._respond(200, b'{"status":"ok"}')
            return
        if self.path == "/api/jobs/job-123":
            type(self).status_requests += 1
            stage = "translation" if self.status_requests == 1 else "complete"
            self._respond(
                200,
                json.dumps(
                    {
                        "job_id": "job-123",
                        "stage": stage,
                        "progress": 0.8 if stage == "translation" else 1.0,
                        "message": stage,
                        "error": None,
                    }
                ).encode(),
            )
            return
        if self.path in {
            "/api/jobs/job-123/pdf/readable",
            "/api/jobs/job-123/pdf/original-layout",
        }:
            assert self.status_requests >= 2, "PDF requested before the job completed"
            self._respond(200, self.pdf_body, "application/pdf")
            return
        self._respond(404, b'{"detail":"not found"}')


def test_runner_uploads_only_the_pdf_and_downloads_modes_after_completion(
    tmp_path: Path,
) -> None:
    spec_path, corpus_root = _write_test_corpus(tmp_path)
    cases = select_cases(load_corpus_cases(spec_path, corpus_root), ["case-one"])
    output_dir = tmp_path / "run"
    _SuccessfulAPIHandler.events = []
    _SuccessfulAPIHandler.upload_body = b""
    _SuccessfulAPIHandler.status_requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SuccessfulAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = TranslaTHORClient(f"http://127.0.0.1:{server.server_port}")
        manifest = run_regression_workflows(
            client=client,
            cases=cases,
            corpus_root=corpus_root,
            spec_path=spec_path,
            output_dir=output_dir,
            timeout_seconds=2,
            poll_interval_seconds=0.001,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    multipart_fields = re.findall(rb'; name="([^"]+)"', _SuccessfulAPIHandler.upload_body)
    assert multipart_fields == [b"files"]
    assert b'name="model"' not in _SuccessfulAPIHandler.upload_body
    assert b'name="extraction_mode"' not in _SuccessfulAPIHandler.upload_body
    assert _SuccessfulAPIHandler.events == [
        "GET /api/health",
        "POST /api/jobs",
        "GET /api/jobs/job-123",
        "GET /api/jobs/job-123",
        "GET /api/jobs/job-123/pdf/readable",
        "GET /api/jobs/job-123/pdf/original-layout",
    ]
    assert manifest["result"] == "success"
    assert manifest["cases"]["case-one"]["job_id"] == "job-123"
    assert manifest["cases"]["case-one"]["outcome"] == "complete"
    assert (output_dir / "artifacts/case-one/readable.pdf").read_bytes().startswith(b"%PDF-")
    assert (output_dir / "artifacts/case-one/original_layout.pdf").is_file()
    saved_manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert saved_manifest["cases"]["case-one"]["artifact_downloads"]["readable"]["success"]
    assert saved_manifest["cases"]["case-one"]["artifact_downloads"]["readable"]["page_count"] == 1


class _MalformedPDFClient:
    def download_pdf(self, _job_id: str, _mode: str):
        return RUNNER.HTTPResult(
            status=200,
            headers={"content-type": "application/pdf"},
            body=b"%PDF-1.4\nnot a real PDF",
        )


def test_download_rejects_pdf_header_without_a_valid_document(tmp_path: Path) -> None:
    result = RUNNER._download_artifact(
        _MalformedPDFClient(),
        job_id="malformed-job",
        mode="readable",
        case_dir=tmp_path / "artifacts",
    )

    assert result["success"] is False
    assert result["page_count"] is None
    assert "could not be opened by PyMuPDF" in result["error"]
    assert not (tmp_path / "artifacts" / "readable.pdf").exists()


class _FailedJobClient:
    base_url = "http://test.invalid"

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def upload(self, _pdf_path: Path) -> str:
        return "failed-job"

    def get_job(self, _job_id: str) -> dict[str, Any]:
        return {
            "stage": "failed",
            "progress": 1.0,
            "message": "Translation failed",
            "error": "model unavailable",
        }

    def download_pdf(self, _job_id: str, _mode: str) -> None:
        raise AssertionError("Artifacts must not be requested for a failed job")


def test_failed_job_is_recorded_and_artifacts_are_not_requested(tmp_path: Path) -> None:
    fixture = tmp_path / "case.pdf"
    fixture.write_bytes(b"%PDF-1.4\nfixture")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    case = CorpusCase("failed-case", "digital", "es", (), fixture)

    manifest = run_regression_workflows(
        client=_FailedJobClient(),  # type: ignore[arg-type]
        cases=[case],
        corpus_root=tmp_path,
        spec_path=spec_path,
        output_dir=tmp_path / "run",
        timeout_seconds=2,
        poll_interval_seconds=0.001,
    )

    record = manifest["cases"]["failed-case"]
    assert manifest["result"] == "failure"
    assert record["outcome"] == "job_failed"
    assert "model unavailable" in record["errors"][0]
    assert all(not artifact["attempted"] for artifact in record["artifact_downloads"].values())


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _RunningJobClient(_FailedJobClient):
    def upload(self, _pdf_path: Path) -> str:
        return "running-job"

    def get_job(self, _job_id: str) -> dict[str, Any]:
        return {
            "stage": "translation",
            "progress": 0.5,
            "message": "Still translating",
            "error": None,
        }


def test_timeout_leaves_remote_job_running_and_records_reason(tmp_path: Path) -> None:
    fixture = tmp_path / "case.pdf"
    fixture.write_bytes(b"%PDF-1.4\nfixture")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    case = CorpusCase("slow-case", "scanned", "de", (), fixture)
    clock = _Clock()

    manifest = run_regression_workflows(
        client=_RunningJobClient(),  # type: ignore[arg-type]
        cases=[case],
        corpus_root=tmp_path,
        spec_path=spec_path,
        output_dir=tmp_path / "run",
        timeout_seconds=1,
        poll_interval_seconds=0.4,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    record = manifest["cases"]["slow-case"]
    assert record["outcome"] == "timeout"
    assert "left running and was not cancelled" in record["errors"][0]
    assert manifest["polling"]["timed_out_jobs_are_cancelled"] is False
    assert all(not artifact["attempted"] for artifact in record["artifact_downloads"].values())
