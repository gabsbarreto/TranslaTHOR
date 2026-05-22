from __future__ import annotations

from app.services.job_store import JobStore
from app.services.pdf_extraction.models import PDFTypeDetectionResult
from app.services.pipeline import TranslationPipeline


def _detection(classification: str) -> PDFTypeDetectionResult:
    return PDFTypeDetectionResult(
        classification=classification,  # type: ignore[arg-type]
        page_count=1,
        pages=[],
        embedded_text_chars=0,
        embedded_text_words=0,
        meaningful_page_count=0,
        garbled_page_count=0,
        image_dominant_page_count=1,
        scanned_page_count=1,
        mixed=False,
        warnings=[],
        metadata={},
    )


def test_pipeline_bypasses_marker_for_clear_poor_text_when_qwen_enabled() -> None:
    pipeline = TranslationPipeline(JobStore())
    settings = {"extraction_mode": "auto", "qwen_ocr_fallback": True}

    assert pipeline._should_bypass_marker_for_qwen(_detection("scanned_no_text"), settings) is True
    assert pipeline._should_bypass_marker_for_qwen(_detection("bad_hidden_ocr"), settings) is True


def test_pipeline_keeps_marker_first_for_uncertain_or_good_text() -> None:
    pipeline = TranslationPipeline(JobStore())
    settings = {"extraction_mode": "auto", "qwen_ocr_fallback": True}

    assert pipeline._should_bypass_marker_for_qwen(_detection("digital_good_text"), settings) is False
    assert pipeline._should_bypass_marker_for_qwen(_detection("mixed"), settings) is False
    assert pipeline._should_bypass_marker_for_qwen(_detection("unknown"), settings) is False


def test_pipeline_does_not_bypass_marker_when_qwen_disabled_or_mode_forced() -> None:
    pipeline = TranslationPipeline(JobStore())

    assert pipeline._should_bypass_marker_for_qwen(
        _detection("scanned_no_text"),
        {"extraction_mode": "auto", "qwen_ocr_fallback": False},
    ) is False
    assert pipeline._should_bypass_marker_for_qwen(
        _detection("scanned_no_text"),
        {"extraction_mode": "digital", "qwen_ocr_fallback": True},
    ) is False
