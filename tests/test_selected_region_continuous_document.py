from __future__ import annotations

from app.models.inspection import PageInspection, PdfInspection
from app.models.regions import OcrRegionResult, OcrResultsPayload, RegionType
from app.services.deepseek_ocr_pipeline import DeepSeekOcrPipeline
from app.services.markdown_builder import MarkdownBuilder


def _inspection() -> PdfInspection:
    return PdfInspection(
        filename="paper.pdf",
        title=None,
        author=None,
        page_count=2,
        pages=[
            PageInspection(1, 612, 792, 0, 0, False),
            PageInspection(2, 612, 792, 0, 0, False),
        ],
    )


def _result(page: int, text: str) -> OcrRegionResult:
    return OcrRegionResult(
        pdf_file_id="job-1",
        page_number=page,
        box_id=f"box-{page}",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        box_type=RegionType.PAGE,
        reading_order=1,
        ocr_text=text,
    )


def test_selected_region_parser_builds_continuous_document_by_default() -> None:
    doc, marker_md = DeepSeekOcrPipeline().parse_selected_regions_document(
        inspection=_inspection(),
        ocr_results=OcrResultsPayload(
            pdf_file_id="job-1",
            results=[
                _result(1, "The intervention improved qual-"),
                _result(2, "ity of life."),
            ],
        ),
    )

    assert marker_md == ""
    assert doc.metadata.translation["translation_input_mode"] == "continuous_document"
    assert doc.blocks[0].metadata["translation_input_mode"] == "continuous_document"
    assert doc.blocks[0].metadata["ocr_region_spans"][0]["page_number"] == 1
    assert doc.blocks[0].metadata["ocr_region_spans"][1]["page_number"] == 2

    source_md = MarkdownBuilder().build(doc, marker_md)
    assert "The intervention improved quality of life." in source_md


def test_selected_region_parser_keeps_page_by_page_fallback() -> None:
    _doc, marker_md = DeepSeekOcrPipeline().parse_selected_regions_document(
        inspection=_inspection(),
        ocr_results=OcrResultsPayload(
            pdf_file_id="job-1",
            results=[_result(1, "First page."), _result(2, "Second page.")],
        ),
        translation_input_mode="page_by_page",
    )

    assert "<!-- page: 1 -->" in marker_md
    assert "<!-- page: 2 -->" in marker_md
