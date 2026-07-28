#!/usr/bin/env python3
"""Run the permanent PDF corpus against an already-running TranslaTHOR API.

This script is intentionally only an API client. It does not start or stop the
application, change server settings, cancel timed-out jobs, or delete jobs.

Example:

    .venv/bin/python scripts/run_pdf_regression_workflows.py \
        --base-url http://127.0.0.1:8000 \
        --case fr-digital-gender-psychiatry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = ROOT / "tests" / "regression_corpus" / "corpus_spec.json"
DEFAULT_CORPUS_ROOT = ROOT / "workspace" / "regression_corpus"
DEFAULT_RUNS_ROOT = ROOT / "workspace" / "regression_runs"
TERMINAL_STAGES = frozenset({"complete", "cancelled", "failed"})
PDF_MODES = ("readable", "original-layout")
MAX_CONSECUTIVE_POLL_ERRORS = 3
SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RegressionRunError(RuntimeError):
    """Raised when the corpus or API does not satisfy the runner contract."""


class APIError(RegressionRunError):
    """An HTTP or response-contract error returned by the TranslaTHOR API."""


@dataclass(frozen=True)
class CorpusCase:
    case_id: str
    category: str
    language: str | None
    features: tuple[str, ...]
    pdf_path: Path


@dataclass(frozen=True)
class HTTPResult:
    status: int
    headers: Mapping[str, str]
    body: bytes


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegressionRunError(f"Could not read JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegressionRunError(f"Expected a JSON object in {path}.")
    return payload


def _manifest_fixture_paths(
    corpus_root: Path,
    *,
    spec_path: Path,
) -> dict[str, Path]:
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.is_file():
        return {}

    manifest = _load_json_object(manifest_path)
    expected_spec_hash = manifest.get("spec_sha256")
    if isinstance(expected_spec_hash, str) and expected_spec_hash != _sha256(spec_path):
        raise RegressionRunError(
            f"Corpus manifest {manifest_path} was built from a different corpus specification. "
            "Rebuild the corpus before running it."
        )

    corpus_root_resolved = corpus_root.resolve()
    fixtures: dict[str, Path] = {}
    for group_name in ("digital_cases", "scanned_cases"):
        group = manifest.get(group_name, [])
        if not isinstance(group, list):
            raise RegressionRunError(f"Invalid {group_name} in {manifest_path}.")
        for record in group:
            if not isinstance(record, dict):
                continue
            fixture = record.get("fixture")
            case_id = record.get("id")
            relative_path = fixture.get("path") if isinstance(fixture, dict) else None
            if not isinstance(case_id, str) or not isinstance(relative_path, str):
                continue
            path = (corpus_root / relative_path).resolve()
            try:
                path.relative_to(corpus_root_resolved)
            except ValueError as exc:
                raise RegressionRunError(
                    f"Fixture path for {case_id} escapes the corpus directory: {relative_path}"
                ) from exc
            if path.is_file():
                expected_sha256 = fixture.get("sha256")
                if expected_sha256 is not None:
                    if not isinstance(expected_sha256, str) or not expected_sha256.strip():
                        raise RegressionRunError(
                            f"Fixture {case_id} has an invalid sha256 value in {manifest_path}."
                        )
                    actual_sha256 = _sha256(path)
                    if actual_sha256.casefold() != expected_sha256.strip().casefold():
                        raise RegressionRunError(
                            f"Corpus fixture {case_id} no longer matches its manifest sha256. "
                            "Rebuild the corpus before running it."
                        )

                expected_page_count = fixture.get("page_count")
                if expected_page_count is not None:
                    if (
                        isinstance(expected_page_count, bool)
                        or not isinstance(expected_page_count, int)
                        or expected_page_count < 1
                    ):
                        raise RegressionRunError(
                            f"Fixture {case_id} has an invalid page_count in {manifest_path}."
                        )
                    try:
                        with fitz.open(path) as document:
                            if document.needs_pass:
                                raise ValueError("PDF is password protected")
                            actual_page_count = document.page_count
                    except Exception as exc:  # PyMuPDF has format-specific exception classes.
                        raise RegressionRunError(
                            f"Could not validate corpus fixture {case_id}: {exc}"
                        ) from exc
                    if actual_page_count != expected_page_count:
                        raise RegressionRunError(
                            f"Corpus fixture {case_id} has {actual_page_count} page(s), but its "
                            f"manifest records {expected_page_count}. Rebuild the corpus before "
                            "running it."
                        )
            fixtures[case_id] = path
    return fixtures


def load_corpus_cases(spec_path: Path, corpus_root: Path) -> list[CorpusCase]:
    """Resolve corpus cases from the specification and optional build manifest."""

    spec_path = spec_path.resolve()
    corpus_root = corpus_root.resolve()
    spec = _load_json_object(spec_path)
    manifest_paths = _manifest_fixture_paths(corpus_root, spec_path=spec_path)
    cases: list[CorpusCase] = []
    seen_ids: set[str] = set()

    for spec_key, directory_name, category in (
        ("digital_cases", "digital", "digital"),
        ("scanned_cases", "scanned", "scanned"),
    ):
        records = spec.get(spec_key)
        if not isinstance(records, list):
            raise RegressionRunError(f"Corpus specification has no valid {spec_key} list.")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise RegressionRunError(f"Every {spec_key} entry must have a string ID.")
            case_id = record["id"]
            if case_id in seen_ids:
                raise RegressionRunError(f"Duplicate corpus case ID: {case_id}")
            if not SAFE_CASE_ID.fullmatch(case_id):
                raise RegressionRunError(
                    f"Corpus case ID may contain only safe filename characters: {case_id!r}"
                )
            seen_ids.add(case_id)
            pdf_path = manifest_paths.get(case_id, corpus_root / directory_name / f"{case_id}.pdf")
            if not pdf_path.is_file():
                raise RegressionRunError(
                    f"Corpus PDF for {case_id} is missing: {pdf_path}. "
                    "Run scripts/build_pdf_regression_corpus.py first."
                )
            features = record.get("features")
            if features is not None and not isinstance(features, list):
                raise RegressionRunError(f"Corpus case {case_id} has an invalid features list.")
            cases.append(
                CorpusCase(
                    case_id=case_id,
                    category=category,
                    language=(
                        record.get("language") if isinstance(record.get("language"), str) else None
                    ),
                    features=tuple(
                        feature for feature in features or [] if isinstance(feature, str)
                    ),
                    pdf_path=pdf_path.resolve(),
                )
            )
    return cases


def select_cases(cases: list[CorpusCase], requested_ids: list[str] | None) -> list[CorpusCase]:
    if not requested_ids:
        return list(cases)
    if len(requested_ids) != len(set(requested_ids)):
        raise RegressionRunError("Each --case value may be supplied only once.")
    by_id = {case.case_id: case for case in cases}
    unknown = [case_id for case_id in requested_ids if case_id not in by_id]
    if unknown:
        available = ", ".join(sorted(by_id))
        raise RegressionRunError(
            f"Unknown corpus case(s): {', '.join(unknown)}. Available cases: {available}"
        )
    return [by_id[case_id] for case_id in requested_ids]


def _multipart_file_body(field_name: str, path: Path) -> tuple[bytes, str]:
    boundary = f"----TranslaTHORRegression{uuid.uuid4().hex}"
    safe_filename = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    return prefix + path.read_bytes() + suffix, boundary


class TranslaTHORClient:
    def __init__(self, base_url: str, *, request_timeout_seconds: float = 300.0) -> None:
        normalized_url = base_url.rstrip("/")
        if not normalized_url.startswith(("http://", "https://")):
            raise RegressionRunError("--base-url must start with http:// or https://")
        if request_timeout_seconds <= 0:
            raise RegressionRunError("--request-timeout-seconds must be greater than zero.")
        self.base_url = normalized_url
        self.request_timeout_seconds = request_timeout_seconds

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HTTPResult:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=dict(headers or {}),
            method=method,
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - URL is an explicit CLI argument.
                request,
                timeout=self.request_timeout_seconds,
            ) as response:
                return HTTPResult(
                    status=response.status,
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except OSError:
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise APIError(f"{method} {path} returned HTTP {exc.code}{suffix}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise APIError(f"{method} {path} failed: {exc}") from exc

    @staticmethod
    def _json_object(result: HTTPResult, operation: str) -> dict[str, Any]:
        try:
            payload = json.loads(result.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIError(f"{operation} returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise APIError(f"{operation} returned JSON that was not an object.")
        return payload

    def health(self) -> dict[str, Any]:
        return self._json_object(self._request("/api/health"), "Health check")

    def upload(self, pdf_path: Path) -> str:
        body, boundary = _multipart_file_body("files", pdf_path)
        result = self._request(
            "/api/jobs",
            method="POST",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        payload = self._json_object(result, "Job upload")
        jobs = payload.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(jobs[0], dict):
            raise APIError("Job upload did not return exactly one job record.")
        job_id = jobs[0].get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise APIError("Job upload response has no valid job_id.")
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any]:
        encoded_job_id = quote(job_id, safe="")
        return self._json_object(
            self._request(f"/api/jobs/{encoded_job_id}"),
            f"Status request for job {job_id}",
        )

    def download_pdf(self, job_id: str, mode: str) -> HTTPResult:
        if mode not in PDF_MODES:
            raise ValueError(f"Unsupported regression PDF mode: {mode}")
        encoded_job_id = quote(job_id, safe="")
        return self._request(f"/api/jobs/{encoded_job_id}/pdf/{mode}")


def _status_snapshot(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: status.get(key)
        for key in (
            "stage",
            "progress",
            "message",
            "error",
            "queue_state",
            "queue_position",
            "jobs_ahead",
        )
    }


def poll_job(
    client: TranslaTHORClient,
    job_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str], bool]:
    """Poll a job, returning terminal status, history, errors, and timeout state."""

    if timeout_seconds <= 0:
        raise RegressionRunError("--timeout-seconds must be greater than zero.")
    if poll_interval_seconds <= 0:
        raise RegressionRunError("--poll-interval-seconds must be greater than zero.")

    deadline = monotonic() + timeout_seconds
    history: list[dict[str, Any]] = []
    poll_errors: list[str] = []
    consecutive_errors = 0
    last_signature: tuple[Any, ...] | None = None

    while True:
        try:
            status = client.get_job(job_id)
            consecutive_errors = 0
        except APIError as exc:
            consecutive_errors += 1
            poll_errors.append(str(exc))
            if consecutive_errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                return None, history, poll_errors, False
        else:
            stage = status.get("stage")
            if not isinstance(stage, str):
                poll_errors.append(f"Job {job_id} returned a status without a string stage.")
                return None, history, poll_errors, False
            snapshot = _status_snapshot(status)
            signature = tuple(snapshot.values())
            if signature != last_signature:
                history.append({"observed_at": _utc_now(), **snapshot})
                last_signature = signature
            if stage in TERMINAL_STAGES:
                return status, history, poll_errors, False

        now = monotonic()
        if now >= deadline:
            return None, history, poll_errors, True
        sleep(min(poll_interval_seconds, max(deadline - now, 0.0)))


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _new_case_record(case: CorpusCase, corpus_root: Path) -> dict[str, Any]:
    try:
        display_path = str(case.pdf_path.relative_to(corpus_root.resolve()))
    except ValueError:
        display_path = str(case.pdf_path)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "language": case.language,
        "features": list(case.features),
        "fixture": {
            "path": display_path,
            "size_bytes": case.pdf_path.stat().st_size,
            "sha256": _sha256(case.pdf_path),
        },
        "started_at": _utc_now(),
        "completed_at": None,
        "outcome": "pending",
        "job_id": None,
        "upload": {
            "success": False,
            "filename": case.pdf_path.name,
            "multipart_fields": ["files"],
            "error": None,
        },
        "status_history": [],
        "terminal_status": None,
        "poll_errors": [],
        "artifact_downloads": {
            mode: {
                "attempted": False,
                "success": False,
                "path": None,
                "size_bytes": None,
                "sha256": None,
                "content_type": None,
                "page_count": None,
                "error": None,
            }
            for mode in PDF_MODES
        },
        "errors": [],
    }


def _download_artifact(
    client: TranslaTHORClient,
    *,
    job_id: str,
    mode: str,
    case_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": True,
        "success": False,
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "content_type": None,
        "page_count": None,
        "error": None,
    }
    try:
        response = client.download_pdf(job_id, mode)
    except APIError as exc:
        result["error"] = str(exc)
        return result

    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    result["content_type"] = content_type or None
    result["size_bytes"] = len(response.body)
    result["sha256"] = hashlib.sha256(response.body).hexdigest()
    if content_type != "application/pdf":
        result["error"] = f"Expected application/pdf, received {content_type or 'no content type'}."
        return result
    if not response.body.startswith(b"%PDF-"):
        result["error"] = "Downloaded response does not have a PDF header."
        return result
    page_count, validation_error = _validate_pdf_bytes(response.body)
    if validation_error:
        result["error"] = validation_error
        return result
    result["page_count"] = page_count

    case_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{mode.replace('-', '_')}.pdf"
    output_path = case_dir / filename
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary_path.write_bytes(response.body)
    temporary_path.replace(output_path)
    result["success"] = True
    result["path"] = str(output_path)
    return result


def _validate_pdf_bytes(data: bytes) -> tuple[int | None, str | None]:
    """Require PyMuPDF to open at least one page with usable geometry."""

    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                return None, "Downloaded PDF is password protected."
            valid_pages = 0
            for page_number in range(document.page_count):
                page = document.load_page(page_number)
                rectangle = page.rect
                if (
                    rectangle.is_valid
                    and not rectangle.is_empty
                    and rectangle.width > 0
                    and rectangle.height > 0
                ):
                    valid_pages += 1
            if valid_pages < 1:
                return None, "Downloaded PDF contains no valid pages."
            return document.page_count, None
    except Exception as exc:  # PyMuPDF exposes format-specific exception classes.
        return None, f"Downloaded PDF could not be opened by PyMuPDF: {type(exc).__name__}: {exc}"


def _summary(records: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    outcomes = [record.get("outcome") for record in records.values()]
    failed_outcomes = {
        "job_failed",
        "job_cancelled",
        "timeout",
        "upload_failed",
        "poll_failed",
        "artifact_failed",
    }
    return {
        "selected": len(records),
        "succeeded": outcomes.count("complete"),
        "failed": sum(outcome in failed_outcomes for outcome in outcomes),
        "pending": outcomes.count("pending"),
        "job_failed": outcomes.count("job_failed"),
        "job_cancelled": outcomes.count("job_cancelled"),
        "timed_out": outcomes.count("timeout"),
        "upload_failed": outcomes.count("upload_failed"),
        "poll_failed": outcomes.count("poll_failed"),
        "artifact_failed": outcomes.count("artifact_failed"),
    }


def run_regression_workflows(
    *,
    client: TranslaTHORClient,
    cases: list[CorpusCase],
    corpus_root: Path,
    spec_path: Path,
    output_dir: Path,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Upload, await, and download selected cases without changing server state."""

    if not cases:
        raise RegressionRunError("No corpus cases were selected.")
    if timeout_seconds <= 0:
        raise RegressionRunError("--timeout-seconds must be greater than zero.")
    if poll_interval_seconds <= 0:
        raise RegressionRunError("--poll-interval-seconds must be greater than zero.")
    health = client.health()
    if health.get("status") != "ok":
        raise APIError(f"TranslaTHOR health check did not return status=ok: {health}")

    output_dir = output_dir.resolve()
    manifest_path = output_dir / "run_manifest.json"
    records: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "runner": "scripts/run_pdf_regression_workflows.py",
        "started_at": _utc_now(),
        "completed_at": None,
        "base_url": client.base_url,
        "server_health": health,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": _sha256(spec_path.resolve()),
        "corpus_root": str(corpus_root.resolve()),
        "output_dir": str(output_dir),
        "selected_case_ids": [case.case_id for case in cases],
        "polling": {
            "per_job_timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "timed_out_jobs_are_cancelled": False,
        },
        "cases": records,
        "summary": _summary(records),
        "result": "running",
    }
    _write_manifest(manifest_path, manifest)

    for case in cases:
        record = _new_case_record(case, corpus_root)
        records[case.case_id] = record
        manifest["summary"] = _summary(records)
        _write_manifest(manifest_path, manifest)
        try:
            job_id = client.upload(case.pdf_path)
        except APIError as exc:
            message = str(exc)
            record["upload"]["error"] = message
            record["errors"].append(message)
            record["outcome"] = "upload_failed"
            record["completed_at"] = _utc_now()
            manifest["summary"] = _summary(records)
            _write_manifest(manifest_path, manifest)
            continue

        record["job_id"] = job_id
        record["upload"]["success"] = True
        _write_manifest(manifest_path, manifest)
        terminal, history, poll_errors, timed_out = poll_job(
            client,
            job_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
        record["status_history"] = history
        record["poll_errors"] = poll_errors
        if timed_out:
            message = (
                f"Job {job_id} did not reach a terminal stage within {timeout_seconds:g} seconds; "
                "it was left running and was not cancelled."
            )
            record["errors"].append(message)
            record["outcome"] = "timeout"
        elif terminal is None:
            message = f"Could not obtain a terminal status for job {job_id}."
            record["errors"].append(message)
            record["outcome"] = "poll_failed"
        else:
            record["terminal_status"] = _status_snapshot(terminal)
            terminal_stage = str(terminal["stage"])
            if terminal_stage != "complete":
                record["outcome"] = f"job_{terminal_stage}"
                terminal_reason = terminal.get("error") or terminal.get("message")
                record["errors"].append(
                    f"Job {job_id} reached {terminal_stage}: "
                    f"{terminal_reason or 'no reason supplied'}"
                )
            else:
                case_dir = output_dir / "artifacts" / case.case_id
                for mode in PDF_MODES:
                    artifact = _download_artifact(
                        client,
                        job_id=job_id,
                        mode=mode,
                        case_dir=case_dir,
                    )
                    if artifact["path"]:
                        artifact["path"] = str(Path(artifact["path"]).relative_to(output_dir))
                    record["artifact_downloads"][mode] = artifact
                    if artifact["error"]:
                        record["errors"].append(f"{mode}: {artifact['error']}")
                    _write_manifest(manifest_path, manifest)
                record["outcome"] = (
                    "complete"
                    if all(
                        artifact["success"] for artifact in record["artifact_downloads"].values()
                    )
                    else "artifact_failed"
                )
        record["completed_at"] = _utc_now()
        manifest["summary"] = _summary(records)
        _write_manifest(manifest_path, manifest)

    manifest["completed_at"] = _utc_now()
    manifest["summary"] = _summary(records)
    manifest["result"] = "success" if manifest["summary"]["failed"] == 0 else "failure"
    _write_manifest(manifest_path, manifest)
    return manifest


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return DEFAULT_RUNS_ROOT / timestamp


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of an already-running TranslaTHOR API (default: %(default)s).",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        help="Corpus case ID to run; repeat for multiple cases. Defaults to all cases.",
    )
    parser.add_argument("--spec-path", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cases = select_cases(
            load_corpus_cases(args.spec_path, args.corpus_root),
            args.case_ids,
        )
        client = TranslaTHORClient(
            args.base_url,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        manifest = run_regression_workflows(
            client=client,
            cases=cases,
            corpus_root=args.corpus_root,
            spec_path=args.spec_path,
            output_dir=args.output_dir or _default_output_dir(),
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except RegressionRunError as exc:
        print(f"Regression run failed: {exc}", file=sys.stderr)
        return 2

    summary = manifest["summary"]
    print(
        f"Regression run {manifest['result']}: {summary['succeeded']}/{summary['selected']} "
        f"cases completed with both PDF artifacts."
    )
    print(f"Manifest: {Path(manifest['output_dir']) / 'run_manifest.json'}")
    return 0 if manifest["result"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
