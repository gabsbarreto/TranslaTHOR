from app.main import _summarize_ocr_results
from app.models.regions import OcrRegionResult, OcrResultsPayload, RegionType


def _result(page: int, box_id: str, text: str) -> OcrRegionResult:
    return OcrRegionResult(
        pdf_file_id="job-1",
        page_number=page,
        box_id=box_id,
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        box_type=RegionType.PAGE,
        reading_order=1,
        ocr_text=text,
    )


def test_ocr_summary_counts_pages_without_text() -> None:
    payload = OcrResultsPayload(
        pdf_file_id="job-1",
        results=[
            _result(1, "page-1", ""),
            _result(2, "page-2", "translated source"),
            _result(2, "manual-2", ""),
            _result(3, "page-3", "   "),
        ],
    )

    summary = _summarize_ocr_results(payload)

    assert summary["total_region_count"] == 4
    assert summary["nonempty_region_count"] == 1
    assert summary["empty_region_count"] == 3
    assert summary["selected_page_count"] == 3
    assert summary["pages_with_text_count"] == 1
    assert summary["pages_without_text_count"] == 2
    assert summary["pages_with_text"] == [2]
    assert summary["pages_without_text"] == [1, 3]
