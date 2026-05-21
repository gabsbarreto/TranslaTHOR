from pathlib import Path

from app.models.regions import OcrRegionResult, OcrResultsPayload, RegionType
from app.services.ocr_results import (
    selected_ocr_progress_from_event,
    summarize_ocr_results,
    write_selected_ocr_source_markdown,
)


def _result(page: int, order: int, box_id: str, text: str) -> OcrRegionResult:
    return OcrRegionResult(
        pdf_file_id="job-1",
        page_number=page,
        box_id=box_id,
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        box_type=RegionType.TEXT,
        reading_order=order,
        ocr_text=text,
    )


def test_summarize_ocr_results_counts_empty_pages() -> None:
    payload = OcrResultsPayload(
        pdf_file_id="job-1",
        results=[
            _result(1, 1, "a", ""),
            _result(2, 1, "b", "Text"),
            _result(3, 1, "c", "   "),
        ],
    )

    summary = summarize_ocr_results(payload)

    assert summary["total_region_count"] == 3
    assert summary["nonempty_region_count"] == 1
    assert summary["pages_with_text"] == [2]
    assert summary["pages_without_text"] == [1, 3]


def test_selected_ocr_progress_from_event_reports_retry_phase() -> None:
    result = selected_ocr_progress_from_event(
        {"event": "page_done", "phase": "retry", "index": 2, "total": 4, "chars": 50}
    )

    assert result == (0.6, "Retrying empty OCR regions: 2/4 complete; 50 characters")


def test_write_selected_ocr_source_markdown_orders_nonempty_regions(tmp_path: Path) -> None:
    payload = OcrResultsPayload(
        pdf_file_id="job-1",
        results=[
            _result(2, 1, "b", "Second page."),
            _result(1, 2, "c", "Second block."),
            _result(1, 1, "a", "First block."),
            _result(3, 1, "d", ""),
        ],
    )

    path = write_selected_ocr_source_markdown(tmp_path, payload)

    assert path == tmp_path / "artifacts" / "source.md"
    assert path.read_text(encoding="utf-8") == (
        "<!-- page: 1 -->\n"
        "First block.\n\n"
        "Second block.\n\n"
        "<!-- page: 2 -->\n"
        "Second page.\n"
    )
