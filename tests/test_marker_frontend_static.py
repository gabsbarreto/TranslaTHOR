from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_upload_defaults_to_automatic_marker_submission() -> None:
    upload_body = APP_JS.split("async function handleUploadedFiles", 1)[1].split("async function submitUploadBatch", 1)[0]

    assert "submitUploadBatch(pdfFiles)" in upload_body


def test_marker_extraction_controls_are_available() -> None:
    assert 'id="extractionMode"' in INDEX_HTML
    assert 'value="strip_and_force_ocr"' in INDEX_HTML
    assert 'id="useLocalVlmRepair"' in INDEX_HTML
    assert 'id="keepDebugArtifacts"' in INDEX_HTML
    assert "Legacy Manual OCR Pipeline" not in INDEX_HTML
    assert "Konva" not in INDEX_HTML


def test_marker_settings_are_sent_to_backend() -> None:
    assert 'form.append("extraction_mode", getInputValue("extractionMode", "auto"));' in APP_JS
    assert 'form.append("use_local_vlm_repair", checkboxValue("useLocalVlmRepair"));' in APP_JS
    assert 'form.append("keep_debug_artifacts", checkboxValue("keepDebugArtifacts"));' in APP_JS
