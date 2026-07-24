from __future__ import annotations

from pathlib import Path

from app.config import DEFAULT_OCR_ENGINE
from app.main import _build_job_settings
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
    )


def test_surya2_is_branch_default_and_job_settings_preserve_engine() -> None:
    assert DEFAULT_OCR_ENGINE == "surya2_llamacpp"
    settings = _build_job_settings(
        chunk_size=1000,
        model="mlx-community/Qwen3.5-9B-MLX-4bit",
        temperature=0.4,
        top_p=0.7,
        top_k=10,
        min_p=0,
        presence_penalty=1.5,
        repetition_penalty=1,
        max_tokens=2048,
        output_mode="readable",
        profile_pipeline=False,
        extraction_mode="auto",
        ocr_engine="surya2_llamacpp",
        use_local_vlm_repair=False,
        keep_debug_artifacts=False,
    )

    assert settings["ocr_engine"] == "surya2_llamacpp"
    assert settings["surya2_strategy"] == "full_page"
    assert settings["surya2_dpi"] == 192


def test_pipeline_routes_only_poor_text_or_forced_jobs_to_surya2() -> None:
    pipeline = TranslationPipeline(JobStore())
    try:
        assert pipeline._should_use_surya2(
            _detection("scanned_no_text"),
            {"ocr_engine": "surya2_llamacpp", "extraction_mode": "auto"},
        )
        assert not pipeline._should_use_surya2(
            _detection("digital_good_text"),
            {"ocr_engine": "surya2_llamacpp", "extraction_mode": "auto"},
        )
        assert pipeline._should_use_surya2(
            _detection("digital_good_text"),
            {"ocr_engine": "surya2_llamacpp", "extraction_mode": "scanned"},
        )
        assert not pipeline._should_use_surya2(
            _detection("scanned_no_text"),
            {"ocr_engine": "marker_surya", "extraction_mode": "auto"},
        )
    finally:
        pipeline.shutdown()


def test_frontend_lists_all_ocr_engines() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend/index.html").read_text(encoding="utf-8")
    javascript = (root / "frontend/app.js").read_text(encoding="utf-8")

    for engine in ("surya2_llamacpp", "surya_qwen_mlx", "marker_surya"):
        assert f'value="{engine}"' in html
    assert 'form.append("ocr_engine"' in javascript
