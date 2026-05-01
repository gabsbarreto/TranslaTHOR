from __future__ import annotations

from app.models.regions import OcrRegionResult, OcrResultsPayload, RegionType
from app.services.ocr_text_compiler import compile_ocr_results_to_document_text


def _result(page: int, order: int, text: str, box_id: str | None = None) -> OcrRegionResult:
    return OcrRegionResult(
        pdf_file_id="job-1",
        page_number=page,
        box_id=box_id or f"box-{page}-{order}",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        box_type=RegionType.TEXT,
        reading_order=order,
        ocr_text=text,
    )


def test_compile_sorts_by_page_and_reading_order() -> None:
    payload = OcrResultsPayload(
        pdf_file_id="job-1",
        results=[
            _result(2, 1, "Third"),
            _result(1, 2, "Second."),
            _result(1, 1, "First."),
        ],
    )

    compiled = compile_ocr_results_to_document_text(payload)

    assert compiled.text == "First.\n\nSecond.\n\nThird"
    assert [(span.page_number, span.reading_order) for span in compiled.spans] == [(1, 1), (1, 2), (2, 1)]


def test_compile_joins_mid_sentence_across_boundary() -> None:
    compiled = compile_ocr_results_to_document_text(
        [
            _result(1, 1, "The intervention was associated with"),
            _result(2, 1, "a significant result."),
        ]
    )

    assert compiled.text == "The intervention was associated with a significant result."


def test_compile_preserves_paragraph_break_after_sentence() -> None:
    compiled = compile_ocr_results_to_document_text(
        [
            _result(1, 1, "The results were statistically significant."),
            _result(2, 1, "Discussion"),
        ]
    )

    assert compiled.text == "The results were statistically significant.\n\nDiscussion"


def test_compile_repairs_cross_boundary_hyphenation() -> None:
    compiled = compile_ocr_results_to_document_text(
        [
            _result(1, 1, "The intervention was associated with a significant improve-"),
            _result(2, 1, "ment in quality of life among participants."),
        ]
    )

    assert compiled.text == "The intervention was associated with a significant improvement in quality of life among participants."
