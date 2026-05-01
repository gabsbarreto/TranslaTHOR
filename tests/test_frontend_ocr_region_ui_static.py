from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_box_click_focuses_instead_of_toggling_ocr_inclusion() -> None:
    toggle_body = APP_JS.split("function toggleBoxSelection", 1)[1].split("function updateBoxStyles", 1)[0]

    assert "focusBox(boxId)" in toggle_body
    assert "box.selected = !box.selected" not in toggle_body


def test_box_delete_control_exists_on_canvas() -> None:
    assert "deleteGroup" in APP_JS
    assert "deleteBoxesById([box.id])" in APP_JS
    assert "No OCR regions on this page. Add a region or use full page." in APP_JS


def test_type_and_selected_controls_are_hidden_from_region_list() -> None:
    render_body = APP_JS.split("function renderBoxList", 1)[1].split("function percentValue", 1)[0]

    assert "data-type-box" not in render_body
    assert "data-select-box" not in render_body
    assert "Selected for OCR" not in render_body
    assert "Type" not in render_body


def test_toolbar_no_longer_exposes_select_or_deselect_all() -> None:
    assert "selectAllBoxesBtn" not in INDEX_HTML
    assert "deselectAllBoxesBtn" not in INDEX_HTML


def test_save_payload_keeps_internal_defaults_for_backend_compatibility() -> None:
    save_body = APP_JS.split("async function savePageBoxes", 1)[1].split("function setAllBoxesSelected", 1)[0]

    assert "selected: true" in save_body
    assert 'type: box.type || "page"' in save_body


def test_ocr_counter_is_rendered_inside_progress_bar() -> None:
    assert "function progressBarLabel" in APP_JS
    assert 'return `OCR ${counter[0].replace(/\\s+/g, "")}`;' in APP_JS
    assert "<small>${escapeHtml(progressText)}</small>" in APP_JS


def test_canvas_drag_drawing_creates_manual_drawn_regions() -> None:
    assert 'id="drawRegionModeBtn"' in INDEX_HTML
    assert "function toggleDrawRegionMode" in APP_JS
    assert "function installCanvasDrawingHandlers" in APP_JS
    assert 'stage.on("pointerdown"' in APP_JS
    assert 'stage.on("pointermove"' in APP_JS
    assert 'stage.on("pointerup pointercancel"' in APP_JS
    assert "if (!canvasState.drawMode && event.target !== stage) return;" in APP_JS
    assert 'source: "manual_drawn"' in APP_JS
    assert "selected: true" in APP_JS
    assert 'coordinate_space: "normalized"' in APP_JS
    assert "item.rect.draggable(!canvasState.drawMode)" in APP_JS


def test_canvas_drag_drawing_ignores_tiny_regions_and_normalizes_direction() -> None:
    assert "MIN_DRAW_REGION_PIXELS" in APP_JS
    assert "MIN_DRAW_REGION_RATIO" in APP_JS
    assert "rect.width < minWidth || rect.height < minHeight" in APP_JS
    assert "Math.min(a.x, b.x)" in APP_JS
    assert "Math.abs(b.x - a.x)" in APP_JS


def test_add_region_button_still_uses_default_manual_box() -> None:
    assert 'addBoxBtn?.addEventListener("click", addManualBox);' in APP_JS
    assert "function buildManualBox" in APP_JS


def test_cancelled_region_job_clears_visual_editor() -> None:
    assert "function resetRegionEditor" in APP_JS
    assert 'job.stage === "cancelled"' in APP_JS
    assert "resetRegionEditor();" in APP_JS


def test_original_ocr_download_buttons_are_available() -> None:
    assert "function sourceMarkdownDownloadLink" in APP_JS
    assert "function sourcePdfDownloadLink" in APP_JS
    assert "/artifacts/source_markdown" in APP_JS
    assert "/ocr-pdf/" in APP_JS
    assert "OCR Markdown" in APP_JS
    assert "OCR PDF" in APP_JS


def test_cancel_stop_and_cleanup_force_page_refresh() -> None:
    assert "function forcePageRefresh" in APP_JS
    assert 'url.searchParams.set("_refresh", Date.now().toString());' in APP_JS
    assert "window.location.replace(url.toString());" in APP_JS
    assert APP_JS.count("forcePageRefresh();") >= 3
    assert "20260501-force-cancel-refresh" in INDEX_HTML
