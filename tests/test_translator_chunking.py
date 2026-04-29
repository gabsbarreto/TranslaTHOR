import sys
from types import ModuleType

from app.models.schema import Block, BlockType, BoundingBox, DocumentMetadata, DocumentModel, PageMetadata, SourceType

if "langdetect" not in sys.modules:
    langdetect_stub = ModuleType("langdetect")
    langdetect_stub.detect = lambda _text: "en"  # type: ignore[attr-defined]
    sys.modules["langdetect"] = langdetect_stub

from app.services.translator_mlx import MlxTranslator, TranslationSettings


def _block(block_id: str, text: str, y0: float, y1: float, x0: float = 50.0) -> Block:
    return Block(
        id=block_id,
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text=text,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x0 + 230, y1=y1),
        reading_order_index=int(y0),
        source_type=SourceType.EMBEDDED,
        style_hints={"font_size": 10},
    )


def test_translation_chunks_follow_paragraph_boundaries() -> None:
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[
            _block("a", "Esta e a primeira linha de um para-", 100, 110),
            _block("b", "grafo que continua na linha seguinte", 112, 122),
            _block("c", "Este e outro paragrafo.", 145, 155),
        ],
    )

    chunks = MlxTranslator(TranslationSettings()).build_chunks(document)

    assert len(chunks) == 2
    assert chunks[0].block_ids == ["a", "b"]
    assert chunks[0].source_text == "Esta e a primeira linha de um paragrafo que continua na linha seguinte"
    assert chunks[1].block_ids == ["c"]


def test_sentence_chunking_respects_end_of_sentence_boundaries() -> None:
    translator = MlxTranslator(TranslationSettings(chunk_size=140))
    translator._token_count = lambda text: len(text.split())  # type: ignore[method-assign]
    sentence_1 = " ".join(["Alpha"] + (["alpha"] * 59)) + "."
    sentence_2 = " ".join(["Beta"] + (["beta"] * 59)) + "."
    sentence_3 = " ".join(["Gamma"] + (["gamma"] * 59)) + "."
    text = f"{sentence_1} {sentence_2} {sentence_3}"

    parts = translator._split_to_token_budget(text)

    assert len(parts) >= 2
    assert all(part.endswith(".") for part in parts)
    assert sentence_1 in " ".join(parts)
    assert sentence_2 in " ".join(parts)
    assert sentence_3 in " ".join(parts)


def test_split_chunks_for_same_block_are_appended_back_to_target() -> None:
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1, detected_language="pt"),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=False,
                embedded_text_quality=0.0,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[
            _block(
                "page-block",
                " ".join(
                    [
                        "Alpha starts a long OCR page.",
                        " ".join(["Beta continues the same OCR page."] * 90),
                        "Gamma closes the page.",
                    ]
                ),
                100,
                500,
            )
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_size=8))
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._token_count = lambda text: len(text.split())  # type: ignore[method-assign]
    translator._is_already_english = lambda chunk: False  # type: ignore[method-assign]
    translator._translate_chunk = lambda text, context="", source_language=None, force_max_tokens=None: f"[{text}]"  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    text = translated_doc.blocks[0].text
    assert "Alpha starts a long OCR page." in text
    assert "Beta continues the same OCR page." in text
    assert "Gamma closes the page." in text
    assert text.count("[") >= 2


def test_chunk_structure_validator_retries_collapsed_translation() -> None:
    translator = MlxTranslator(TranslationSettings())
    calls: list[str] = []

    def fake_translate(text: str, context: str = "", source_language=None, force_max_tokens=None) -> str:
        _ = (text, source_language, force_max_tokens)
        calls.append(context)
        if len(calls) == 1:
            return "short"
        return "First translated paragraph.\n\nSecond translated paragraph.\n\nThird translated paragraph."

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]
    source = "First source paragraph has enough content.\n\nSecond source paragraph has enough content.\n\nThird source paragraph has enough content."

    translated = translator._translate_chunk_with_validation(source, "", "pt", BlockType.PARAGRAPH)

    assert len(calls) == 2
    assert "Preserve the source structure exactly" in calls[1]
    assert translated.count("\n\n") == 2


def test_chunk_structure_validator_rejects_bad_retry_to_source_text() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._translate_chunk = lambda *args, **kwargs: "short"  # type: ignore[method-assign]
    source = "A" * 300

    assert translator._translate_chunk_with_validation(source, "", "pt", BlockType.PARAGRAPH) == source


def test_sentence_splitting_avoids_decimals_and_abbreviations() -> None:
    translator = MlxTranslator(TranslationSettings())
    text = "Dr. Smith measured 3.14 units in the trial. The values were stable."

    sentences = translator._split_into_sentences(text)

    assert sentences == ["Dr. Smith measured 3.14 units in the trial.", "The values were stable."]


def test_table_markup_validation_detects_truncation() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = "<table><tr><td>A</td></tr></table>"
    good = "<table><tr><td>Alpha</td></tr></table>"
    bad = "<table><tr><td>Alpha</td></tr>"

    assert translator._is_valid_table_markup_translation(source, good) is True
    assert translator._is_valid_table_markup_translation(source, bad) is False


def test_table_normalization_repairs_escaped_row_openers_and_missing_row_closes() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = (
        "<table><tr><td>Diagnosis</td></tr>"
        "&lt; tr&gt;<td>Trastorno de uso de sustancias</td><td>10</td>"
        "<tr><td>Insomnio no organico</td><td>5</td></tr></table>"
    )

    normalized = translator._normalize_table_markup_for_translation(source)

    assert "&lt; tr&gt;" not in normalized
    assert "<tr><td>Trastorno de uso de sustancias</td><td>10</td></tr>" in normalized
    assert translator._count_tag_pair(normalized, "tr") == (3, 3)


def test_table_chunks_are_normalized_before_translation() -> None:
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[
            _block(
                "table-block",
                (
                    "<table><tr><td>A</td></tr>&lt; tr&gt;<td>B</td>"
                    "<tr><td>C</td></tr></table>"
                ),
                100,
                110,
            )
        ],
    )

    chunks = MlxTranslator(TranslationSettings()).build_chunks(document)

    assert len(chunks) == 1
    assert chunks[0].source_text == (
        "<table><tr><td>A</td></tr><tr><td>B</td></tr><tr><td>C</td></tr></table>"
    )


def test_table_translation_uses_row_group_fallback_when_whole_table_is_invalid() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = (
        "<table><tr><td>A</td></tr><tr><td>B</td></tr><tr><td>C</td></tr></table>"
    )
    calls = {"count": 0}

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        calls["count"] += 1
        if calls["count"] <= 2:
            return "<table><tr><td>BROKEN</td></tr>"
        return text.replace("A", "Alpha")

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_table_markup_chunk(source, "ctx", "pl")

    assert calls["count"] >= 3
    assert translated.endswith("</table>")
    assert "Alpha" in translated
