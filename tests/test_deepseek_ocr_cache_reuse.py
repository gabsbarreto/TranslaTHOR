from __future__ import annotations

import sys
from types import ModuleType

import pytest

from app.models.inspection import PageInspection, PdfInspection
from app.models.schema import BlockType

if "langdetect" not in sys.modules:
    langdetect_stub = ModuleType("langdetect")
    langdetect_stub.detect = lambda _text: "en"  # type: ignore[attr-defined]
    sys.modules["langdetect"] = langdetect_stub

if "pypdfium2" not in sys.modules:
    pypdfium_stub = ModuleType("pypdfium2")

    class _PdfDocument:  # pragma: no cover - guard stub for import only
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("pypdfium2 stub should not be used in this test")

    pypdfium_stub.PdfDocument = _PdfDocument  # type: ignore[attr-defined]
    sys.modules["pypdfium2"] = pypdfium_stub

from app.services.deepseek_ocr_pipeline import DeepSeekOcrPipeline


def _inspection() -> PdfInspection:
    return PdfInspection(
        filename="sample.pdf",
        title="Sample",
        author="Tester",
        page_count=2,
        pages=[
            PageInspection(
                page_number=1,
                width=100.0,
                height=200.0,
                text_length=0,
                embedded_text_quality=0.0,
                has_embedded_text=False,
            ),
            PageInspection(
                page_number=2,
                width=100.0,
                height=200.0,
                text_length=0,
                embedded_text_quality=0.0,
                has_embedded_text=False,
            ),
        ],
    )


def test_parse_cached_document_reuses_markdown(tmp_path) -> None:
    job_dir = tmp_path / "job"
    ocr_dir = job_dir / "deepseek_ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "page_0001.md").write_text("# Intro\n\nAlpha beta gamma delta epsilon zeta eta theta.", encoding="utf-8")
    (ocr_dir / "page_0002.md").write_text("| Col A | Col B |\n|---|---|\n| 1 | 2 |", encoding="utf-8")

    pipeline = DeepSeekOcrPipeline()
    document, marker_markdown = pipeline.parse_cached_document(_inspection(), job_dir)

    assert "cached DeepSeek-OCR markdown" in document.warnings[0]
    assert "<!-- page: 1 -->" in marker_markdown
    assert "<!-- page: 2 -->" in marker_markdown
    assert any(block.block_type == BlockType.HEADING for block in document.blocks)
    assert any(block.block_type == BlockType.TABLE for block in document.blocks)


def test_parse_cached_document_requires_complete_page_files(tmp_path) -> None:
    job_dir = tmp_path / "job"
    ocr_dir = job_dir / "deepseek_ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    (ocr_dir / "page_0001.md").write_text("Only page one.", encoding="utf-8")

    pipeline = DeepSeekOcrPipeline()
    with pytest.raises(RuntimeError, match="missing page files"):
        pipeline.parse_cached_document(_inspection(), job_dir)
