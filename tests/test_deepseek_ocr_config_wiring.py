from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

if "langdetect" not in sys.modules:
    langdetect_stub = ModuleType("langdetect")
    langdetect_stub.detect = lambda _text: "en"  # type: ignore[attr-defined]
    sys.modules["langdetect"] = langdetect_stub

if "pypdfium2" not in sys.modules:
    pypdfium_stub = ModuleType("pypdfium2")

    class _PdfDocument:  # pragma: no cover - import guard only
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("pypdfium2 stub should not be used in this test")

    pypdfium_stub.PdfDocument = _PdfDocument  # type: ignore[attr-defined]
    sys.modules["pypdfium2"] = pypdfium_stub

from app import config
from app.main import _build_job_settings
from app.models.inspection import PageInspection, PdfInspection
from app.models.schema import Block, BlockType, DocumentMetadata, DocumentModel, PageMetadata, SourceType
import app.services.pipeline as pipeline_module
from app.services.pipeline import TranslationPipeline


def _settings() -> dict:
    return _build_job_settings(
        chunk_size=1800,
        model=config.DEFAULT_TRANSLATION_MODEL,
        temperature=0.2,
        top_p=0.9,
        max_tokens=2048,
        output_mode=config.DEFAULT_OUTPUT_MODE,
        profile_pipeline=False,
        reuse_ocr_cache=False,
    )


def test_job_settings_include_all_deepseek_ocr_defaults() -> None:
    settings = _settings()

    assert settings["deepseek_ocr_model"] == config.DEFAULT_DEEPSEEK_OCR_MODEL
    assert settings["deepseek_ocr_max_tokens"] == config.DEFAULT_DEEPSEEK_OCR_MAX_TOKENS
    assert settings["deepseek_ocr_prompt"] == config.DEFAULT_DEEPSEEK_OCR_PROMPT
    assert settings["deepseek_ocr_crop_mode"] == config.DEFAULT_DEEPSEEK_OCR_CROP_MODE
    assert settings["deepseek_ocr_min_crops"] == config.DEFAULT_DEEPSEEK_OCR_MIN_CROPS
    assert settings["deepseek_ocr_max_crops"] == config.DEFAULT_DEEPSEEK_OCR_MAX_CROPS
    assert settings["deepseek_ocr_base_size"] == config.DEFAULT_DEEPSEEK_OCR_BASE_SIZE
    assert settings["deepseek_ocr_image_size"] == config.DEFAULT_DEEPSEEK_OCR_IMAGE_SIZE
    assert settings["deepseek_ocr_skip_repeat"] == config.DEFAULT_DEEPSEEK_OCR_SKIP_REPEAT
    assert settings["deepseek_ocr_ngram_size"] == config.DEFAULT_DEEPSEEK_OCR_NGRAM_SIZE
    assert settings["deepseek_ocr_ngram_window"] == config.DEFAULT_DEEPSEEK_OCR_NGRAM_WINDOW


def test_pipeline_passes_deepseek_ocr_settings_to_parser(tmp_path, monkeypatch) -> None:
    job_dir = tmp_path / "job"
    (job_dir / "artifacts").mkdir(parents=True)
    parser = _RecordingDeepSeekParser()
    pipeline = TranslationPipeline(_FakeJobStore(job_dir))
    pipeline.inspector = _FakeInspector()  # type: ignore[assignment]
    pipeline.deepseek_parser = parser  # type: ignore[assignment]
    monkeypatch.setattr(pipeline_module, "run_translation_subprocess", _fake_translation_subprocess)

    pipeline.run("job-1", tmp_path / "input.pdf", _settings())

    assert parser.kwargs["model_name"] == config.DEFAULT_DEEPSEEK_OCR_MODEL
    assert parser.kwargs["max_tokens"] == config.DEFAULT_DEEPSEEK_OCR_MAX_TOKENS
    assert parser.kwargs["prompt"] == config.DEFAULT_DEEPSEEK_OCR_PROMPT
    assert parser.kwargs["crop_mode"] == config.DEFAULT_DEEPSEEK_OCR_CROP_MODE
    assert parser.kwargs["min_crops"] == config.DEFAULT_DEEPSEEK_OCR_MIN_CROPS
    assert parser.kwargs["max_crops"] == config.DEFAULT_DEEPSEEK_OCR_MAX_CROPS
    assert parser.kwargs["base_size"] == config.DEFAULT_DEEPSEEK_OCR_BASE_SIZE
    assert parser.kwargs["image_size"] == config.DEFAULT_DEEPSEEK_OCR_IMAGE_SIZE
    assert parser.kwargs["skip_repeat"] == config.DEFAULT_DEEPSEEK_OCR_SKIP_REPEAT
    assert parser.kwargs["ngram_size"] == config.DEFAULT_DEEPSEEK_OCR_NGRAM_SIZE
    assert parser.kwargs["ngram_window"] == config.DEFAULT_DEEPSEEK_OCR_NGRAM_WINDOW


class _FakeJobStore:
    def __init__(self, job_dir: Path) -> None:
        self.job_dir = job_dir
        self.updates: list[dict] = []

    def get_job_dir(self, _job_id: str) -> Path:
        return self.job_dir

    def update_status(self, _job_id: str, **updates) -> None:
        self.updates.append(updates)


class _FakeInspector:
    def inspect(self, _pdf_path: Path) -> PdfInspection:
        return PdfInspection(
            filename="input.pdf",
            title=None,
            author=None,
            page_count=1,
            pages=[
                PageInspection(
                    page_number=1,
                    width=100.0,
                    height=200.0,
                    text_length=0,
                    embedded_text_quality=0.0,
                    has_embedded_text=False,
                )
            ],
        )


class _RecordingDeepSeekParser:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def parse_document(self, **kwargs):
        self.kwargs = kwargs
        return _document(), "OCR markdown"

    def save_document_json(self, document: DocumentModel, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def _document() -> DocumentModel:
    return DocumentModel(
        metadata=DocumentMetadata(filename="input.pdf", page_count=1, detected_language="en"),
        pages=[
            PageMetadata(
                page_number=1,
                width=100.0,
                height=200.0,
                has_embedded_text=False,
                embedded_text_quality=0.0,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[
            Block(
                id="b1",
                page_number=1,
                block_type=BlockType.PARAGRAPH,
                text="Example text.",
                bbox=None,
                reading_order_index=0,
                source_type=SourceType.OCR,
            )
        ],
    )


def _fake_translation_subprocess(**kwargs) -> None:
    output_markdown_path = Path(kwargs["output_markdown_path"])
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.write_text("Translated markdown", encoding="utf-8")
