import sys
from types import ModuleType

from app.models.schema import BlockType, TranslationChunk

if "langdetect" not in sys.modules:
    langdetect_stub = ModuleType("langdetect")
    langdetect_stub.detect = lambda _text: "en"  # type: ignore[attr-defined]
    sys.modules["langdetect"] = langdetect_stub

from app.services.translator_mlx import MlxTranslator, TranslationSettings


def test_augment_context_for_heading_adds_instruction() -> None:
    translator = MlxTranslator(TranslationSettings())
    context = translator._augment_context_for_block_type("Prior section", BlockType.HEADING)
    assert "section heading/title" in context
    assert "Prior section" in context


def test_augment_context_for_non_heading_is_unchanged() -> None:
    translator = MlxTranslator(TranslationSettings())
    context = translator._augment_context_for_block_type("Prior section", BlockType.PARAGRAPH)
    assert context == "Prior section"


def test_postprocess_strips_end_of_turn_tokens() -> None:
    translator = MlxTranslator(TranslationSettings())
    text = "INTRODUCTION<end_of_turn><end_of_turn>"
    assert translator._postprocess_translated_text(text) == "INTRODUCTION"


def test_already_english_heuristic_does_not_skip_polish_heading_marked_en() -> None:
    translator = MlxTranslator(TranslationSettings())
    chunk = TranslationChunk(id="c1", block_ids=["b1"], source_text="Metoda", source_language="en")
    assert translator._is_already_english(chunk) is False


def test_already_english_heuristic_does_not_skip_markup_table_marked_en() -> None:
    translator = MlxTranslator(TranslationSettings())
    table_text = "<table><tr><td>Tożsamość płciowa</td><td>Ogółem</td></tr></table>"
    chunk = TranslationChunk(id="c2", block_ids=["b2"], source_text=table_text, source_language="en")
    assert translator._is_already_english(chunk) is False


def test_already_english_heuristic_keeps_english_sentence_skip() -> None:
    translator = MlxTranslator(TranslationSettings())
    text = "The study presents key demographic characteristics of the sample."
    chunk = TranslationChunk(id="c3", block_ids=["b3"], source_text=text, source_language="en")
    assert translator._is_already_english(chunk) is True


def test_chunk_source_language_avoids_document_fallback_for_heading() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._document_language = "en"
    translator._detect_text_language = lambda text: None  # type: ignore[method-assign]
    translator._detect_text_language_relaxed = lambda text: None  # type: ignore[method-assign]
    assert translator._chunk_source_language("Wstęp", BlockType.HEADING) is None
