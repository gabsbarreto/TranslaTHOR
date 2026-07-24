from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    match = re.search(rf"(?:async\s+)?function\s+{re.escape(name)}\s*\(", APP_JS)
    assert match is not None, f"Missing JavaScript function: {name}"
    next_function = re.search(r"\n(?:async\s+)?function\s+\w+\s*\(", APP_JS[match.end() :])
    end = match.end() + next_function.start() if next_function is not None else len(APP_JS)
    return APP_JS[match.start() : end]


def test_upload_remains_a_single_automatic_action() -> None:
    upload_body = _function_body("handleUploadedFiles")

    assert "submitUploadBatch(pdfFiles)" in upload_body


def test_advanced_translation_and_extraction_settings_are_not_visible() -> None:
    removed_control_ids = {
        "extractionMode",
        "chunkSize",
        "temp",
        "maxTokens",
        "model",
        "topP",
        "useLocalVlmRepair",
        "keepDebugArtifacts",
    }

    for control_id in removed_control_ids:
        assert f'id="{control_id}"' not in INDEX_HTML

    assert "Extraction Mode" not in INDEX_HTML
    assert "Translation LLM" not in INDEX_HTML
    assert "LLM Temperature" not in INDEX_HTML


def test_page_uses_current_waiting_and_recent_result_sections() -> None:
    for label in ("Current activity", "Waiting", "Recent results"):
        assert label in INDEX_HTML

    assert "Show excluded" in INDEX_HTML
    assert "Automated Extraction / Translation" not in INDEX_HTML
    assert "Document Queue" not in INDEX_HTML


def test_normal_job_actions_offer_only_the_two_primary_pdf_modes() -> None:
    primary_actions = _function_body("primaryPdfActions")

    assert 'mode: "readable"' in primary_actions
    assert 'label: "Readable PDF"' in primary_actions
    assert 'mode: "original-layout"' in primary_actions
    assert 'label: "Original layout PDF"' in primary_actions
    for unwanted_label in (
        "Faithful PDF",
        "OCR Markdown",
        "OCR PDF",
        "Translated Markdown",
        "Structured JSON",
        "Extraction JSON",
        "Timing summary",
        "Timing JSON",
        "Timing CSV",
        "Extraction report",
        "Detection JSON",
        "Detection report",
    ):
        assert unwanted_label not in APP_JS


def test_primary_pdf_actions_are_complete_only() -> None:
    result_card = _function_body("resultCard")
    prepare_pdf = _function_body("preparePdf")

    assert 'const complete = job.stage === "complete";' in result_card
    assert "complete ? primaryPdfActions(job)" in result_card
    assert "/pdf/${encodeURIComponent(mode)}" in prepare_pdf
    assert "Preparing readable PDF" in APP_JS
    assert "Preparing original layout" in APP_JS


def test_job_details_hold_warnings_configuration_and_reconstruction_information() -> None:
    assert "View details" in APP_JS
    assert "reconstruction_report" in APP_JS
    assert "original_layout_reconstruction" in APP_JS
    assert "warnings" in APP_JS
    assert "Translation model" in APP_JS


def test_archive_unarchive_delete_and_cancel_use_scoped_job_routes() -> None:
    actions = _function_body("handleJobAction")
    permanent_delete = _function_body("permanentDeleteSection")

    assert "/api/jobs/${encodeURIComponent(jobId)}/archive" in actions
    assert "/api/jobs/${encodeURIComponent(jobId)}/unarchive" in actions
    assert "/api/jobs/${encodeURIComponent(jobId)}/cancel" in actions
    assert 'changeJob(job, "delete", `/api/jobs/${encodeURIComponent(jobId)}`, "DELETE"' in actions
    assert actions.count('"POST"') >= 4
    assert 'if (!TERMINAL_STAGES.has(job.stage)) return "";' in permanent_delete


def test_destructive_per_job_actions_require_confirmation() -> None:
    actions = _function_body("handleJobAction")

    assert "Remove “${job.filename}” from the waiting queue" in actions
    assert "Stop processing" in actions
    assert "Permanently delete" in actions
    assert actions.count("window.confirm") == 3
    assert (
        "Delete this job, its uploaded PDF, and all generated files. This cannot be undone."
        in APP_JS
    )


def test_bulk_actions_are_disclosed_under_manage_instead_of_the_primary_header() -> None:
    assert '<details id="managePanel"' in INDEX_HTML
    manage_panel = INDEX_HTML.split('<details id="managePanel"', 1)[1].split("</details>", 1)[0]

    assert "<summary>Manage</summary>" in manage_panel
    for control_id in ("cleanTerminalBtn", "stopAllBtn", "clearResultsBtn"):
        assert f'id="{control_id}"' in manage_panel


def test_inline_status_and_progress_are_accessible() -> None:
    assert 'role="status"' in INDEX_HTML
    assert 'aria-live="polite"' in INDEX_HTML
    assert 'role="progressbar"' in APP_JS
    assert "aria-valuenow" in APP_JS
    assert "aria-valuemin" in APP_JS
    assert "aria-valuemax" in APP_JS


def test_links_do_not_nest_buttons() -> None:
    assert not re.search(r"<a\b[^>]*>\s*<button\b", APP_JS)


def test_dropzone_is_a_keyboard_accessible_file_label() -> None:
    assert '<label id="dropzone" class="dropzone" for="fileInput">' in INDEX_HTML
    file_input = re.search(
        r"<input\b(?=[^>]*\bid=\"fileInput\")[^>]*>", INDEX_HTML, flags=re.DOTALL
    )
    assert file_input is not None
    assert 'class="visually-hidden"' in file_input.group(0)
    assert re.search(r"(?:^|\s)hidden(?:\s|/|>)", file_input.group(0)) is None


def test_frontend_uses_structured_queue_metadata_and_loads_archived_jobs() -> None:
    for field in ("queue_state", "queue_position", "jobs_ahead", "archived_at", "settings"):
        assert field in APP_JS

    poll_jobs = _function_body("pollJobs")
    assert "/api/jobs?include_archived=true" in poll_jobs
