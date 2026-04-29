from app.main import _selected_ocr_progress_from_event


def test_selected_ocr_progress_reports_box_counter() -> None:
    progress, message = _selected_ocr_progress_from_event(
        {"event": "page_started", "phase": "primary", "index": 2, "total": 15}
    )

    assert 0.35 < progress < 0.84
    assert "OCR selected regions: 2/15" == message


def test_selected_ocr_progress_reports_completed_box_counter_and_chars() -> None:
    progress, message = _selected_ocr_progress_from_event(
        {"event": "page_done", "phase": "primary", "index": 15, "total": 15, "chars": 1234}
    )

    assert progress == 0.84
    assert message == "OCR selected regions: 15/15 complete; 1234 characters"


def test_selected_ocr_progress_labels_retry_phase() -> None:
    progress, message = _selected_ocr_progress_from_event(
        {"event": "page_started", "phase": "retry", "index": 1, "total": 3}
    )

    assert progress == 0.36
    assert message == "Retrying empty OCR regions: 1/3"


def test_selected_ocr_progress_ignores_unrelated_events() -> None:
    assert _selected_ocr_progress_from_event({"event": "unknown"}) is None
