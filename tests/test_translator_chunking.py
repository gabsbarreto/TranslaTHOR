import re
import sys
from types import ModuleType

from app.models.schema import (
    Block,
    BlockType,
    BoundingBox,
    DocumentMetadata,
    DocumentModel,
    PageMetadata,
    SourceType,
    TranslationChunk,
)
from app.services.markdown_builder import MarkdownBuilder

if "langdetect" not in sys.modules:
    langdetect_stub = ModuleType("langdetect")
    langdetect_stub.detect = lambda _text: "en"  # type: ignore[attr-defined]
    sys.modules["langdetect"] = langdetect_stub

from app.services.translator_mlx import MlxTranslator, TranslationSettings
from app.services.qwen_markdown_parser import QwenMarkdownParser


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


def _list_block(block_id: str, text: str, y0: float) -> Block:
    block = _block(block_id, text, y0, y0 + 10)
    return block.model_copy(update={"block_type": BlockType.LIST})


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
    assert (
        chunks[0].source_text
        == "Esta e a primeira linha de um paragrafo que continua na linha seguinte"
    )
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
    translator._translate_chunk = (
        lambda text, context="", source_language=None, force_max_tokens=None: f"[{text}]"
    )  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    text = translated_doc.blocks[0].text
    assert "Alpha starts a long OCR page." in text
    assert "Beta continues the same OCR page." in text
    assert "Gamma closes the page." in text
    assert text.count("[") == 1


def test_chunk_structure_validator_retries_collapsed_translation() -> None:
    translator = MlxTranslator(TranslationSettings())
    calls: list[str] = []

    def fake_translate(
        text: str, context: str = "", source_language=None, force_max_tokens=None
    ) -> str:
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

    assert (
        translator._translate_chunk_with_validation(source, "", "pt", BlockType.PARAGRAPH) == source
    )


def test_confident_english_keyword_line_is_not_reverse_translated() -> None:
    source = "Gender Dysphoria – Adolescent Healthcare Clinical Outcomes"
    block = _block("keywords", source, 100, 130)
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="paper.pdf",
            page_count=1,
            detected_language="es",
            translation={"ocr_logical_chunks_prepared": True},
        ),
        pages=[],
        blocks=[block],
        translation_chunks=[
            TranslationChunk(
                id="keywords-chunk",
                block_ids=[block.id],
                source_text=source,
                source_language="es",
            )
        ],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._detect_language_with_confidence = lambda _text: ("en", 0.99)  # type: ignore[method-assign]
    calls: list[str] = []
    translator._translate_chunk = lambda text, *args, **kwargs: calls.append(text) or "BAD"  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert calls == []
    assert translated_doc.blocks[0].text == source
    assert translated_doc.translation_chunks[0].status == "ready_for_translation"


def test_non_english_identity_output_retries_with_english_only_instruction() -> None:
    source = "Los pacientes reciben tratamiento hormonal con seguimiento clinico continuo."
    target = "Patients receive hormonal treatment with continuous clinical follow-up."
    translator = MlxTranslator(TranslationSettings())
    calls: list[str] = []

    def fake_detect(text: str) -> tuple[str, float]:
        lowered = text.lower()
        return (
            ("en", 0.99)
            if "patients receive" in lowered or "continuous clinical" in lowered
            else ("es", 0.99)
        )

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (text, source_language, force_max_tokens)
        calls.append(context)
        return source if len(calls) == 1 else target

    translator._detect_language_with_confidence = fake_detect  # type: ignore[method-assign]
    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_chunk_with_validation(
        source,
        "Use consistent clinical terminology.",
        "es",
        BlockType.PARAGRAPH,
    )

    assert translated == target
    assert len(calls) == 2
    assert "Return English only" in calls[1]


def test_short_non_english_list_item_retries_in_english() -> None:
    source = "Tener 18 años de edad"
    target = "Be at least 18 years of age"
    translator = MlxTranslator(TranslationSettings())
    calls: list[str] = []

    def fake_detect(text: str) -> tuple[str, float]:
        return ("en", 0.99) if "at least" in text.lower() else ("es", 0.99)

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (text, context, source_language, force_max_tokens)
        calls.append(text)
        return source if len(calls) == 1 else target

    translator._detect_language_with_confidence = fake_detect  # type: ignore[method-assign]
    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_chunk_with_validation(
        source,
        "Eligibility criteria",
        "es",
        BlockType.LIST,
    )

    assert translated == target
    assert len(calls) == 2


def test_short_name_and_formula_remain_exempt_from_language_rejection() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )

    assert (
        translator._translation_acceptance_issue(
            "García, Pérez, López",
            "García, Pérez, López",
            "es",
            BlockType.PARAGRAPH,
        )
        is None
    )
    assert (
        translator._translation_acceptance_issue(
            "FSH = 12.4 mIU/mL",
            "FSH = 12.4 mIU/mL",
            "es",
            BlockType.EQUATION,
        )
        is None
    )


def test_failed_non_english_translation_is_detectable_on_chunk() -> None:
    source = "Los pacientes reciben tratamiento hormonal con seguimiento clinico continuo."
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1, detected_language="es"),
        pages=[],
        blocks=[_block("body", source, 100, 140)],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._detect_language_with_confidence = lambda _text: ("es", 0.99)  # type: ignore[method-assign]
    calls: list[str] = []
    translator._translate_chunk = lambda text, *args, **kwargs: calls.append(text) or text  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert len(calls) == 2
    assert translated_doc.blocks[0].text == source
    chunk = translated_doc.translation_chunks[0]
    assert chunk.status == "translation_failed"
    assert chunk.reason == "translation_output_matches_source"
    assert chunk.warnings == [
        "Translation output matched substantive non-English source text after retry."
    ]
    assert translated_doc.blocks[0].metadata["translation_validation"] == {
        "status": "translation_failed",
        "reason": "translation_output_matches_source",
        "warnings": chunk.warnings,
    }
    assert translated_doc.metadata.translation["target_language_validation"] == {
        "status": "warning",
        "failed_chunk_count": 1,
        "failed_chunk_ids": [chunk.id],
        "policy": "retry_english_then_preserve_source",
    }
    assert "failed English-output validation" in translated_doc.warnings[-1]


def test_short_heading_is_validated_while_names_and_formulas_remain_exempt() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: ("es", 0.99)  # type: ignore[method-assign]

    assert (
        translator._translation_acceptance_issue(
            "Resultados",
            "Resultados",
            "es",
            BlockType.HEADING,
        )
        == "translation_output_matches_source"
    )
    assert (
        translator._translation_acceptance_issue(
            "García, Pérez, López",
            "García, Pérez, López",
            "es",
            BlockType.PARAGRAPH,
        )
        is None
    )
    assert (
        translator._translation_acceptance_issue(
            "p < 0.05; n = 24; x = y + 2",
            "p < 0.05; n = 24; x = y + 2",
            "es",
            BlockType.EQUATION,
        )
        is None
    )
    assert (
        translator._translation_acceptance_issue(
            "https://shs.cairn.info/revue-adolescence-2019-1-page-111?lang=fr",
            "https://shs.cairn.info/revue-adolescence-2019-1-page-111?lang=fr",
            "fr",
            BlockType.PARAGRAPH,
        )
        is None
    )


def test_journal_locator_line_is_exempt_from_translation_failure() -> None:
    source = (
        "Cir. Plást. Iberlatinamer. • Vol. 27 - N°4 "
        "Octubre - Noviembre - Diciembre 2001 / Pag 273-280"
    )
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )

    assert (
        translator._translation_acceptance_issue(
            source,
            source,
            "es",
            BlockType.PARAGRAPH,
        )
        is None
    )


def test_repeated_initial_multi_author_line_is_exempt_from_translation_failure() -> None:
    source = (
        "Esteva de Antonio I.*, Giraldo F.**, Bergero de Miguel T.***, "
        "Cano Oncala G.***, Crespillo Gómez C. *, Ruiz de Adana S. *, "
        "Rojo Martínez G. *, Soriguer Escofet F. *"
    )
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )

    assert (
        translator._translation_acceptance_issue(
            source,
            source,
            "es",
            BlockType.PARAGRAPH,
        )
        is None
    )


def test_compact_journal_contact_block_is_exempt_from_translation_failure() -> None:
    source = (
        "CORRESPONDENCIA Dra. María Fernández Rodríguez. C/ Valdés Salas, nº 4. "
        "CP 33400. Avilés. Asturias, España; maria.fernandezr@example.es"
    )
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )

    assert (
        translator._translation_acceptance_issue(
            source,
            source,
            "es",
            BlockType.PARAGRAPH,
        )
        is None
    )


def test_prose_containing_an_email_is_not_mistaken_for_contact_metadata() -> None:
    source = (
        "Los participantes enviaron sus respuestas a research@example.es antes "
        "de completar la entrevista clínica posterior."
    )
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )

    assert (
        translator._translation_acceptance_issue(
            source,
            source,
            "es",
            BlockType.PARAGRAPH,
        )
        == "translation_output_matches_source"
    )


def test_dated_prose_is_not_mistaken_for_bibliographic_metadata() -> None:
    source = (
        "Los pacientes recibieron tratamiento hormonal en 2001 y mantuvieron "
        "seguimiento clínico continuado."
    )
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )

    assert (
        translator._translation_acceptance_issue(
            source,
            source,
            "es",
            BlockType.PARAGRAPH,
        )
        == "translation_output_matches_source"
    )


def test_url_only_chunk_is_not_sent_to_the_translation_model() -> None:
    translator = MlxTranslator(TranslationSettings())
    chunk = TranslationChunk(
        id="link",
        block_ids=["link"],
        source_text="https://example.es/documento?lang=es",
        source_language="es",
    )

    assert translator._is_already_english(chunk) is True


def test_standard_english_heading_in_non_english_document_is_not_reverse_translated() -> None:
    translator = MlxTranslator(TranslationSettings())
    chunk = TranslationChunk(
        id="abstract",
        block_ids=["abstract"],
        source_text="Abstract",
        source_language="es",
    )

    assert translator._is_already_english(chunk) is True


def test_short_heading_language_uses_following_same_page_context() -> None:
    institute = _block("institute", "Institute", 100, 115).model_copy(
        update={"block_type": BlockType.HEADING}
    )
    affiliation_one = _list_block(
        "affiliation-1",
        "Universitätsklinikum Hamburg, Institut für Sexualforschung",
        120,
    )
    affiliation_two = _list_block(
        "affiliation-2",
        "Universitätsklinikum Münster, Klinik für Psychiatrie",
        140,
    )
    keywords = _block("keywords", "Keywords", 180, 195).model_copy(
        update={"block_type": BlockType.HEADING}
    )
    keyword_list = _block(
        "keyword-list",
        "Gender dysphoria, clinical assessment, hormone therapy",
        200,
        220,
    )
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="bilingual-paper.pdf",
            page_count=1,
            detected_language="en",
        ),
        pages=[],
        blocks=[
            institute,
            affiliation_one,
            affiliation_two,
            keywords,
            keyword_list,
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_group_size=1))

    def fake_detect(text: str) -> tuple[str, float]:
        lowered = " ".join(text.casefold().split())
        if lowered.startswith("institute") and "universitätsklinikum" in lowered:
            return "de", 0.999
        if lowered == "institute":
            return "fr", 0.91
        if lowered.startswith("keywords") and "gender dysphoria" in lowered:
            return "en", 0.999
        if lowered == "keywords":
            return "af", 0.99
        if "universitätsklinikum" in lowered:
            return "de", 0.999
        return "en", 0.999

    translator._detect_language_with_confidence = fake_detect  # type: ignore[method-assign]

    chunks = {chunk.block_ids[0]: chunk for chunk in translator.build_chunks(document)}
    institute_chunk = chunks["institute"]
    keywords_chunk = chunks["keywords"]

    assert institute_chunk.source_language == "de"
    assert institute_chunk.source_language_origin == "nearby_context"
    assert institute_chunk.source_language_confidence == 0.999
    assert "Universitätsklinikum Hamburg" in institute_chunk.context
    assert "do not translate, reproduce, or summarize" in institute_chunk.context
    assert translator._is_already_english(institute_chunk) is False
    assert (
        translator._translation_acceptance_issue(
            "Institute",
            "Institute",
            institute_chunk.source_language,
            BlockType.HEADING,
        )
        == "translation_output_matches_source"
    )

    assert keywords_chunk.source_language == "en"
    assert keywords_chunk.source_language_origin == "nearby_context"
    assert keywords_chunk.source_language_confidence == 0.999
    assert translator._is_already_english(keywords_chunk) is True


def test_short_non_english_heading_and_footnote_require_translation() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )

    for source, block_type in (
        ("Bibliografía", BlockType.HEADING),
        ("Código numérico 68", BlockType.PARAGRAPH),
        ("Servicio de Endocrinología y Nutrición", BlockType.FOOTNOTE),
    ):
        assert (
            translator._translation_acceptance_issue(
                source,
                source,
                "es",
                block_type,
            )
            == "translation_output_matches_source"
        )


def test_unchanged_short_cross_language_heading_gets_contextual_disambiguation_retry() -> None:
    translator = MlxTranslator(TranslationSettings())
    contexts: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (text, source_language, force_max_tokens)
        contexts.append(context)
        if "character-for-character copy" in context:
            return "Affiliations"
        return "Institute"

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_chunk_with_validation(
        "Institute",
        "Nearby source text: three university clinic addresses.",
        "de",
        BlockType.HEADING,
        source_language_authoritative=True,
    )

    assert translated == "Affiliations"
    assert len(contexts) == 3
    assert "stated source language" in contexts[-1]
    assert "nearby source context" in contexts[-1]


def test_repeated_homograph_uses_numbered_affiliation_structure_fallback() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._translate_chunk = lambda *args, **kwargs: "Institute"  # type: ignore[method-assign]
    context = (
        "Nearby source text for language and terminology disambiguation only:\n"
        "1 Universitätsklinikum Hamburg, Klinik für Psychiatrie\n"
        "2 Universitätsklinikum Münster, Institut für Sexualforschung"
    )

    translated = translator._translate_chunk_with_validation(
        "Institute",
        context,
        "de",
        BlockType.HEADING,
        source_language_authoritative=True,
    )

    assert translated == "Affiliations"


def test_translation_validation_preserves_numbers_but_allows_acronym_changes() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = "La relación TMF/TFM fue 2,4/1 en 2020."

    assert (
        translator._chunk_translation_issue(
            source,
            "The TFM/TMF ratio was 2.4/1 in 2020.",
            "es",
            BlockType.PARAGRAPH,
        )
        is None
    )
    assert (
        translator._chunk_translation_issue(
            source,
            "The TMF/TFM ratio was 2.4/2 in 2020.",
            "es",
            BlockType.PARAGRAPH,
        )
        == "translation_numbers_changed"
    )
    assert (
        translator._chunk_translation_issue(
            source,
            "The TMF/TFM ratio was 2.4/1 in 2020.",
            "es",
            BlockType.PARAGRAPH,
        )
        is None
    )


def test_translation_validation_allows_natural_numeric_reordering() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = (
        "En ambas series, por criterios diagnósticos, han sido excluidos el 14% "
        "(la mitad de ellos habían realizado autotratamiento previo: Grupo 3) "
        "y sólo se ha excluido de tratamiento hormonal por criterios médicos "
        "2 pacientes hombre-a-mujer."
    )
    translated = (
        "In both series, only two male-to-female patients were excluded from "
        "hormonal treatment for medical reasons, while 14% were excluded based "
        "on diagnostic criteria, half of whom had previously self-treated: Group 3."
    )

    assert (
        translator._chunk_translation_issue(
            source,
            translated,
            "es",
            BlockType.PARAGRAPH,
        )
        is None
    )


def test_translation_validation_rejects_missing_changed_or_duplicated_numbers() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = "Fueron excluidos el 14%, Grupo 3, y 2 pacientes."

    for translated in (
        "14% were excluded in Group 3.",
        "15% were excluded in Group 3, and two patients.",
        "14% were excluded in Group 3, Group 3, and 2 patients.",
    ):
        assert (
            translator._chunk_translation_issue(
                source,
                translated,
                "es",
                BlockType.PARAGRAPH,
            )
            == "translation_numbers_changed"
        )


def test_table_abbreviations_may_use_equally_compact_target_language_forms() -> None:
    translator = MlxTranslator(TranslationSettings())

    assert (
        translator._chunk_translation_issue(
            "<table><tr><td>ACV</td><td>HTA</td></tr></table>",
            "<table><tr><td>CVA</td><td>HTN</td></tr></table>",
            "es",
            BlockType.TABLE,
        )
        is None
    )


def test_known_short_english_target_is_not_rejected_by_language_detector() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "ca",
        0.99,
    )

    assert (
        translator._translation_acceptance_issue(
            "Resumen",
            "Abstract",
            "es",
            BlockType.HEADING,
        )
        is None
    )


def test_changed_short_target_does_not_rely_on_unstable_language_detection() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "fr",
        0.99,
    )

    assert (
        translator._translation_acceptance_issue(
            "épisode dépressif moyen",
            "mean depressive episode",
            "fr",
            BlockType.PARAGRAPH,
        )
        is None
    )


def test_acronym_invariant_ignores_new_uppercase_words_and_statistical_labels() -> None:
    translator = MlxTranslator(TranslationSettings())

    assert (
        translator._translation_invariant_issue(
            "MATÉRIEL ET MÉTHODE",
            "MATERIAL AND METHODS",
            BlockType.HEADING,
        )
        is None
    )
    assert (
        translator._translation_invariant_issue(
            "La media fue 32.74 (DT=11.198).",
            "The mean was 32.74 (SD=11.198).",
            BlockType.PARAGRAPH,
        )
        is None
    )
    assert (
        translator._translation_invariant_issue(
            "HOMBRES Y MUJERES",
            "MEN AND WOMEN",
            BlockType.HEADING,
        )
        is None
    )
    assert (
        translator._translation_invariant_issue(
            "HOMMES ET FEMMES",
            "MEN AND WOMEN",
            BlockType.HEADING,
        )
        is None
    )


def test_parenthesized_document_acronym_is_preserved() -> None:
    translator = MlxTranslator(TranslationSettings())

    assert (
        translator._translation_invariant_issue(
            "Terapia hormonal (TMF) continuada.",
            "Continued hormonal therapy (MTF).",
            BlockType.PARAGRAPH,
        )
        is None
    )


def test_numeric_invariant_normalizes_spaced_ocr_decimal_percentages() -> None:
    translator = MlxTranslator(TranslationSettings())

    assert translator._ordered_numeric_tokens("La tasa fue 12. 6% y p=.008.") == [
        "12.6%",
        ".008",
    ]
    assert translator._ordered_numeric_tokens("The rate was 12.6% and p=.008.") == [
        "12.6%",
        ".008",
    ]


def test_translation_validation_rejects_invented_paragraph_heading() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = "Los pacientes recibieron seguimiento clínico durante seis meses."
    translated = "Materials and Methods\n\nPatients received clinical follow-up for six months."

    assert (
        translator._chunk_translation_issue(
            source,
            translated,
            "es",
            BlockType.PARAGRAPH,
        )
        == "translation_structure_invalid"
    )


def test_reference_blocks_are_preserved_verbatim() -> None:
    source = "García A. Título original. Revista Médica. 2020;12:1-8."
    block = _block("reference", source, 100, 130).model_copy(
        update={"block_type": BlockType.REFERENCE}
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1, detected_language="es"),
        pages=[],
        blocks=[block],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    calls: list[str] = []
    translator._translate_chunk = lambda text, *args, **kwargs: calls.append(text) or "changed"  # type: ignore[method-assign]

    translated_document, _ = translator.translate_document(document, "")

    assert calls == []
    assert translated_document.blocks[0].text == source


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


def test_table_markup_validation_preserves_each_row_width() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = (
        "<table><tr><th>A</th><th>B</th></tr>"
        "<tr><td>C</td><td>D</td></tr></table>"
    )
    redistributed = (
        "<table><tr><th>Alpha</th><th>Beta</th><td>Extra</td></tr>"
        "<tr><td>Delta</td></tr></table>"
    )

    assert translator._is_valid_table_markup_translation(source, redistributed) is False


def test_table_markup_validation_rejects_crossed_cell_tags() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = "<table><tr><td>A</td><th>B</th></tr></table>"
    crossed = "<table><tr><td>Alpha</th><th>Beta</td></tr></table>"

    assert translator._is_valid_table_markup_translation(source, crossed) is False


def test_table_markup_validation_preserves_cell_tags_spans_and_sections() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = (
        '<table><thead><tr><th rowspan="2">Grupo</th>'
        '<th colspan="2">Medidas</th></tr></thead>'
        "<tbody><tr><td>A</td><td>B</td></tr></tbody></table>"
    )
    good = (
        '<table><thead><tr><th rowspan="2">Group</th>'
        '<th colspan="2">Measures</th></tr></thead>'
        "<tbody><tr><td>Alpha</td><td>Beta</td></tr></tbody></table>"
    )
    dropped_span = good.replace(' colspan="2"', "")
    changed_rowspan = good.replace(' rowspan="2"', ' rowspan="3"')
    changed_cell_tag = good.replace("<td>Alpha</td>", "<th>Alpha</th>")
    changed_section = good.replace("<thead>", "<tbody>").replace("</thead>", "</tbody>")

    assert translator._is_valid_table_markup_translation(source, good) is True
    assert translator._is_valid_table_markup_translation(source, dropped_span) is False
    assert translator._is_valid_table_markup_translation(source, changed_rowspan) is False
    assert translator._is_valid_table_markup_translation(source, changed_cell_tag) is False
    assert translator._is_valid_table_markup_translation(source, changed_section) is False


def test_table_markup_validation_preserves_all_layout_attributes() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = (
        '<table class="clinical striped" align="CENTER" '
        'style="width: 100%; border-collapse: collapse" data-layout="fixed">'
        '<thead class="heading" style="background: #eee">'
        '<tr class="labels primary" align="LEFT">'
        '<th class="measure key" colspan="2" '
        'style="font-weight: bold; text-align: left">Grupo</th>'
        "</tr></thead></table>"
    )
    reordered_attributes = (
        '<table data-layout="fixed" style="width:100%;border-collapse:  collapse" '
        'align="center" class="striped clinical">'
        '<thead style="background:#eee;" class="heading">'
        '<tr align="left" class="primary labels">'
        '<th style="font-weight:bold;text-align:left" colspan="2" '
        'class="key measure">Group</th>'
        "</tr></thead></table>"
    )
    dropped_table_class = reordered_attributes.replace(' class="striped clinical"', "")
    changed_section_style = reordered_attributes.replace("background:#eee", "background:#fff")
    changed_row_alignment = reordered_attributes.replace(
        '<tr align="left"',
        '<tr align="right"',
    )
    dropped_cell_class = reordered_attributes.replace(' class="key measure"', "")
    changed_cell_style = reordered_attributes.replace("text-align:left", "text-align:center")

    assert (
        translator._is_valid_table_markup_translation(source, reordered_attributes) is True
    )
    assert translator._is_valid_table_markup_translation(source, dropped_table_class) is False
    assert translator._is_valid_table_markup_translation(source, changed_section_style) is False
    assert translator._is_valid_table_markup_translation(source, changed_row_alignment) is False
    assert translator._is_valid_table_markup_translation(source, dropped_cell_class) is False
    assert translator._is_valid_table_markup_translation(source, changed_cell_style) is False


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
                ("<table><tr><td>A</td></tr>&lt; tr&gt;<td>B</td><tr><td>C</td></tr></table>"),
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
    source = "<table><tr><td>A</td></tr><tr><td>B</td></tr><tr><td>C</td></tr></table>"
    calls = {"count": 0}
    contexts: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        calls["count"] += 1
        contexts.append(context)
        if calls["count"] <= 2:
            return "<table><tr><td>BROKEN</td></tr>"
        return text.replace("A", "Alpha")

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_table_markup_chunk(source, "ctx", "pl")

    assert calls["count"] >= 3
    assert translated.endswith("</table>")
    assert "Alpha" in translated
    assert all("Do not expand source abbreviations" in context for context in contexts)


def test_table_row_group_fallback_preserves_sections_and_translates_every_row() -> None:
    source = (
        '<table class="clinical"><thead data-kind="labels">'
        "<tr><th>Tratamiento hormonal continuo</th><th>Edad media pacientes</th></tr>"
        "</thead><tbody class=\"records\">"
        "<tr><td>Grupo de control</td><td>Seguimiento clínico continuado</td></tr>"
        "<tr><td>Grupo de tratamiento</td><td>Respuesta clínica favorable</td></tr>"
        "</tbody><tfoot>"
        "<tr><td>Total de pacientes</td><td>Cuatro pacientes evaluados</td></tr>"
        "</tfoot></table>"
    )
    replacements = {
        "Tratamiento hormonal continuo": "Continuous hormone treatment",
        "Edad media pacientes": "Mean patient age",
        "Grupo de control": "Control group",
        "Seguimiento clínico continuado": "Continuous clinical follow-up",
        "Grupo de tratamiento": "Treatment group",
        "Respuesta clínica favorable": "Favourable clinical response",
        "Total de pacientes": "Total patients",
        "Cuatro pacientes evaluados": "Four patients assessed",
    }
    translator = MlxTranslator(TranslationSettings())
    translator.TABLE_ROW_GROUP_SIZE = 1
    calls: list[str] = []

    def fake_detect(text: str) -> tuple[str, float]:
        source_words = {
            "tratamiento",
            "edad",
            "media",
            "pacientes",
            "grupo",
            "seguimiento",
            "clínico",
            "continuado",
            "respuesta",
            "clínica",
            "favorable",
            "cuatro",
            "evaluados",
        }
        words = {word.casefold() for word in re.findall(r"[^\W\d_]+", text)}
        return ("es", 0.99) if words & source_words else ("en", 0.99)

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls.append(text)
        if len(calls) <= 2:
            return "<table><tr><td>BROKEN</td></tr>"
        translated = text
        for source_text, target_text in replacements.items():
            translated = translated.replace(source_text, target_text)
        return translated

    translator._detect_language_with_confidence = fake_detect  # type: ignore[method-assign]
    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_table_markup_chunk(source, "Table 1", "es")

    assert len(calls) == 6
    assert all(call.count("<tr") == 1 for call in calls[2:])
    assert '<thead data-kind="labels">' in translated
    assert '<tbody class="records">' in translated
    assert "<tfoot>" in translated
    assert "Continuous hormone treatment" in translated
    assert "Continuous clinical follow-up" in translated
    assert "Four patients assessed" in translated
    assert "Tratamiento" not in translated
    assert "Seguimiento" not in translated
    assert translator._table_markup_topology(translated) == translator._table_markup_topology(
        source
    )


def test_table_row_group_fallback_bisects_an_invalid_group() -> None:
    source = (
        "<table><tbody>"
        "<tr><td>Grupo uno</td></tr><tr><td>Grupo dos</td></tr>"
        "<tr><td>Grupo tres</td></tr><tr><td>Grupo cuatro</td></tr>"
        "</tbody></table>"
    )
    replacements = {
        "Grupo uno": "Group one",
        "Grupo dos": "Group two",
        "Grupo tres": "Group three",
        "Grupo cuatro": "Group four",
    }
    translator = MlxTranslator(TranslationSettings())
    translator.TABLE_ROW_GROUP_SIZE = 4
    calls: list[str] = []

    def fake_detect(text: str) -> tuple[str, float]:
        return ("es", 0.99) if "grupo" in text.casefold() else ("en", 0.99)

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls.append(text)
        if len(calls) <= 3:
            return "<table><tr><td>BROKEN</td></tr></table>"
        translated = text
        for source_text, target_text in replacements.items():
            translated = translated.replace(source_text, target_text)
        return translated

    translator._detect_language_with_confidence = fake_detect  # type: ignore[method-assign]
    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_table_markup_chunk(source, "Table 1", "es")

    assert len(calls) == 5
    assert calls[2].count("<tr") == 4
    assert calls[3].count("<tr") == 2
    assert calls[4].count("<tr") == 2
    assert "Group one" in translated
    assert "Group four" in translated
    assert translator._table_markup_topology(translated) == translator._table_markup_topology(
        source
    )


def test_table_row_group_fallback_rejects_changed_section_topology() -> None:
    source = (
        "<table><thead><tr><th>Tratamiento hormonal continuo</th></tr></thead>"
        "<tbody><tr><td>Seguimiento clínico continuado</td></tr></tbody></table>"
    )
    translator = MlxTranslator(TranslationSettings())
    translator.TABLE_ROW_GROUP_SIZE = 1
    calls = {"count": 0}

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls["count"] += 1
        if calls["count"] <= 2:
            return "<table><tr><td>BROKEN</td></tr>"
        return (
            text.replace("<thead>", "<tbody>")
            .replace("</thead>", "</tbody>")
            .replace("Tratamiento hormonal continuo", "Continuous hormone treatment")
        )

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_table_markup_chunk(source, "Table 1", "es")

    assert translated == source
    assert calls["count"] == 3


def test_structurally_valid_untranslated_table_retries_in_english() -> None:
    source = (
        "<table><tr><td>Los pacientes reciben tratamiento hormonal</td>"
        "<td>El seguimiento clinico continuo sigue siendo necesario</td></tr></table>"
    )
    target = (
        "<table><tr><td>Patients receive hormonal treatment</td>"
        "<td>Continuous clinical follow-up remains necessary</td></tr></table>"
    )
    translator = MlxTranslator(TranslationSettings())
    calls: list[str] = []

    def fake_detect(text: str) -> tuple[str, float]:
        lowered = text.lower()
        return (
            ("en", 0.99)
            if "patients receive" in lowered or "continuous clinical" in lowered
            else ("es", 0.99)
        )

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (text, source_language, force_max_tokens)
        calls.append(context)
        return source if len(calls) == 1 else target

    translator._detect_language_with_confidence = fake_detect  # type: ignore[method-assign]
    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_table_markup_chunk(source, "Table 1", "es")

    assert translated == target
    assert len(calls) == 2
    assert "Return English only" in calls[1]


def test_short_non_english_table_labels_trigger_retry() -> None:
    source = (
        "<table><tr><th>Tratamiento</th><th>Edad media</th></tr>"
        "<tr><td>12</td><td>18</td></tr></table>"
    )
    target = (
        "<table><tr><th>Treatment</th><th>Mean age</th></tr>"
        "<tr><td>12</td><td>18</td></tr></table>"
    )
    translator = MlxTranslator(TranslationSettings())
    calls: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (text, context, source_language, force_max_tokens)
        calls.append(text)
        return source if len(calls) == 1 else target

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated = translator._translate_table_markup_chunk(source, "Table 1", "es")

    assert translated == target
    assert len(calls) == 2


def test_numeric_and_abbreviation_only_table_may_remain_unchanged() -> None:
    source = (
        "<table><tr><th>TMF</th><th>TFM</th></tr>"
        "<tr><td>12</td><td>18</td></tr></table>"
    )

    assert (
        MlxTranslator(TranslationSettings())._is_acceptable_table_translation(
            source,
            source,
            "es",
        )
        is True
    )


def test_mixed_case_study_abbreviations_may_remain_unchanged_in_translated_table() -> None:
    source = (
        "<table><tr><th>Alter</th><th>Gruppe</th></tr>"
        "<tr><td>M (SD)<br>MzFa M (SD)<br>FzMb M (SD)</td><td>Behandlung</td></tr></table>"
    )
    target = (
        "<table><tr><th>Age</th><th>Group</th></tr>"
        "<tr><td>M (SD)<br>MzFa M (SD)<br>FzMb M (SD)</td><td>Treatment</td></tr></table>"
    )
    translator = MlxTranslator(TranslationSettings())

    assert translator._looks_like_table_abbreviation("MzFa") is True
    assert translator._looks_like_table_abbreviation("FzMb") is True
    assert translator._looks_like_table_abbreviation("Alter") is False
    assert translator._table_translation_issue(source, target, "de") is None


def test_numbered_affiliations_and_doi_blocks_may_remain_verbatim() -> None:
    translator = MlxTranslator(TranslationSettings())

    for source in (
        "1 Kurparkklinik Dr. Lauterbach-Klinik GmbH, Bad Liebenstein",
        (
            "Z Sexualforsch 2024; 37: 142–150 DOI 10.1055/a-2368-9352 "
            "ISSN 0932-8114 © 2024. Thieme. All rights reserved."
        ),
    ):
        assert (
            translator._translation_acceptance_issue(
                source,
                source,
                "de",
                BlockType.PARAGRAPH,
            )
            is None
        )


def test_preserved_short_non_english_table_is_reported_as_translation_failure() -> None:
    source = (
        "<table><tr><th>Tratamiento</th><th>Edad media</th></tr>"
        "<tr><td>12</td><td>18</td></tr></table>"
    )
    table_block = _block("table", source, 100, 160).model_copy(
        update={"block_type": BlockType.TABLE}
    )
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="paper.pdf",
            page_count=1,
            detected_language="es",
        ),
        pages=[],
        blocks=[table_block],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda _chunk: False  # type: ignore[method-assign]
    translator._detect_text_language = lambda _text: "es"  # type: ignore[method-assign]
    translator._translate_chunk = lambda text, *args, **kwargs: text  # type: ignore[method-assign]

    translated_document, _ = translator.translate_document(document, "")

    chunk = translated_document.translation_chunks[0]
    assert chunk.status == "translation_failed"
    assert chunk.reason == "translation_output_matches_source"
    assert translated_document.blocks[0].metadata["translation_validation"]["status"] == (
        "translation_failed"
    )


def test_delimited_table_translation_preserves_cell_boundaries() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = f"Primera celda{translator.TABLE_DELIMITER}Segunda celda"

    assert (
        translator._is_valid_chunk_translation_structure(
            source,
            "First cell Second cell",
            BlockType.TABLE,
        )
        is False
    )


def test_table_validation_rejects_a_significant_untranslated_cell() -> None:
    source = (
        "<table><tr><td>Los pacientes reciben tratamiento hormonal continuado</td>"
        "<td>El seguimiento clinico sigue siendo necesario</td></tr></table>"
    )
    partially_translated = (
        "<table><tr><td>Los pacientes reciben tratamiento hormonal continuado</td>"
        "<td>Clinical follow-up remains necessary</td></tr></table>"
    )
    translator = MlxTranslator(TranslationSettings())

    def fake_detect(text: str) -> tuple[str, float]:
        return ("en", 0.99) if "clinical follow" in text.lower() else ("es", 0.99)

    translator._detect_language_with_confidence = fake_detect  # type: ignore[method-assign]

    assert (
        translator._is_acceptable_table_translation(
            source,
            partially_translated,
            "es",
        )
        is False
    )


def test_table_validation_rejects_short_untranslated_label_in_mixed_table() -> None:
    source = (
        "<table><thead><tr><th>Treatment</th><th>Edad media</th></tr></thead>"
        "<tbody><tr><td>Terapia hormonal</td><td>32</td></tr></tbody></table>"
    )
    partially_translated = (
        "<table><thead><tr><th>Treatment</th><th>Edad media</th></tr></thead>"
        "<tbody><tr><td>Hormonal therapy</td><td>32</td></tr></tbody></table>"
    )
    translator = MlxTranslator(TranslationSettings())

    def fake_detect(text: str) -> tuple[str, float]:
        lowered = text.lower()
        if "hormonal therapy" in lowered and len(lowered.split()) > 3:
            return ("en", 0.99)
        return ("es", 0.99) if "edad media" in lowered else ("en", 0.99)

    translator._detect_language_with_confidence = fake_detect  # type: ignore[method-assign]

    assert (
        translator._table_translation_issue(source, partially_translated, "es")
        == "translation_output_matches_source"
    )


def test_table_validation_accepts_translated_cells_despite_html_and_abbreviation_overlap() -> None:
    source = (
        "<table> <thead> <tr> <th>Absolutas</th> <th>Relativas</th> </tr> </thead> "
        "<tbody> <tr> <td>HTA severa</td> <td>HTA</td> </tr> "
        "<tr> <td>C.Isquémica</td> <td>Migraña refractaria</td> </tr> "
        "<tr> <td>Hepatopatía</td> <td>Poliglobulia</td> </tr> "
        "<tr> <td>I.Renal</td> <td>Dislipemia</td> </tr> "
        "<tr> <td></td> <td>Tromboflebitis</td> </tr> </tbody> </table>"
    )
    translated = (
        "<table> <thead> <tr> <th>Absolute</th> <th>Relative</th> </tr> </thead> "
        "<tbody> <tr> <td>Serious HTA</td> <td>HTA</td> </tr> "
        "<tr> <td>C. Ischemic</td> <td>Intractable migraine</td> </tr> "
        "<tr> <td>Hepatopathy</td> <td>Polycythemia</td> </tr> "
        "<tr> <td>Renal I.R.</td> <td>Dyslipidemia</td> </tr> "
        "<tr> <td></td> <td>Thrombophlebitis</td> </tr> </tbody> </table>"
    )
    translator = MlxTranslator(TranslationSettings())

    assert translator._has_valid_table_markup_structure(source, translated) is True
    assert translator._table_translation_issue(source, translated, "es") is None


def test_table_cell_validation_allows_close_medical_cognates_when_changed() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = "<table><tr><td>Hiperprolactinemia</td><td>Edad media</td></tr></table>"
    translated = "<table><tr><td>Hyperprolactinemia</td><td>Mean age</td></tr></table>"
    copied = "<table><tr><td>Hyperprolactinemia</td><td>Edad media</td></tr></table>"

    assert translator._table_translation_issue(source, translated, "es") is None
    assert (
        translator._table_translation_issue(source, copied, "es")
        == "translation_output_matches_source"
    )


def test_table_validation_rejects_cell_tag_role_changes() -> None:
    source = (
        "<table><thead><tr><th>Absolutas</th><th>Relativas</th></tr></thead>"
        "<tbody><tr><td>HTA severa</td><td>Migraña refractaria</td></tr></tbody></table>"
    )
    changed_cell_role = (
        "<table><thead><tr><th>Absolute</th><th>Relative</th></tr></thead>"
        "<tbody><tr><th>Severe HTN</th><td>Intractable migraine</td></tr></tbody></table>"
    )
    translator = MlxTranslator(TranslationSettings())

    assert (
        translator._table_translation_issue(source, changed_cell_role, "es")
        == "translation_table_structure_invalid"
    )


def test_numeric_and_acronym_table_cells_do_not_require_translation() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )

    for value in ("12", "12.5%", "TMF", "TFM"):
        assert (
            translator._translation_acceptance_issue(
                value,
                value,
                "es",
                BlockType.TABLE,
            )
            is None
        )


def test_unicode_table_cells_cannot_be_silently_erased() -> None:
    source = (
        "<table><tr><th>العلاج الهرموني المستمر</th>"
        "<th>المتابعة السريرية الضرورية</th></tr></table>"
    )
    erased = "<table><tr><th></th><th></th></tr></table>"
    translator = MlxTranslator(TranslationSettings())

    assert translator._is_significant_table_cell("العلاج") is True
    assert translator._is_significant_table_cell("Группа лечения") is True
    assert translator._is_significant_table_cell("治疗组") is True
    assert (
        translator._table_translation_issue(source, erased, "ar")
        == "translation_table_cell_missing"
    )
    assert translator._is_acceptable_table_translation(source, erased, "ar") is False


def test_numeric_markdown_table_requires_english_and_exact_pipe_topology() -> None:
    translator = MlxTranslator(TranslationSettings())
    translator._detect_language_with_confidence = lambda _text: (  # type: ignore[method-assign]
        "es",
        0.99,
    )
    source = (
        "| Período de la demanda | Grupo | Intervalos de edad | Total | "
        "|---|---|---|---| | 2007-2009 | HM | 23 | 36 |"
    )
    spanish_output = (
        "| Periodo de la solicitud | Grupo | Intervalos de edad | Total | "
        "|---|---|---|---| | 2007-2009 | HM | 23 | 36 |"
    )
    damaged_english_output = (
        "| Request period | Group | Age intervals | Total "
        "|---|---|---|---| | 2007-2009 | HM | 23 | 36 |"
    )

    assert (
        translator._translation_acceptance_issue(
            source,
            spanish_output,
            "es",
            BlockType.TABLE,
        )
        == "translation_output_not_english"
    )
    assert (
        translator._chunk_translation_issue(
            source,
            damaged_english_output,
            "es",
            BlockType.TABLE,
        )
        == "translation_table_structure_invalid"
    )


def test_adjacent_prose_chunks_are_merged_before_translation() -> None:
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
            _block("a", "Primeiro paragrafo com contexto.", 100, 110),
            _block("b", "Segundo paragrafo com separador ----- no meio.", 140, 150),
            _block("c", "Terceiro paragrafo para manter continuidade.", 180, 190),
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_group_size=4))
    translator._token_count = lambda text: len(text.split())  # type: ignore[method-assign]

    chunks = translator.build_chunks(document)

    assert len(chunks) == 1
    assert chunks[0].block_ids == ["a", "b", "c"]
    assert "Primeiro paragrafo com contexto." in chunks[0].source_text
    assert "Segundo paragrafo com separador ----- no meio." in chunks[0].source_text
    assert "Terceiro paragrafo para manter continuidade." in chunks[0].source_text


def test_adjacent_prose_chunks_do_not_merge_across_pages() -> None:
    first = _block("page-1", "Primeiro paragrafo.", 100, 110)
    second = _block("page-2", "Segundo paragrafo.", 100, 110).model_copy(update={"page_number": 2})
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=2, detected_language="pt"),
        pages=[
            PageMetadata(
                page_number=page_number,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
            for page_number in (1, 2)
        ],
        blocks=[first, second],
    )
    translator = MlxTranslator(TranslationSettings(chunk_group_size=5))
    translator._token_count = lambda text: len(text.split())  # type: ignore[method-assign]

    chunks = translator.build_chunks(document)

    assert [chunk.block_ids for chunk in chunks] == [["page-1"], ["page-2"]]
    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(1, 1), (2, 2)]


def test_adjacent_prose_chunks_do_not_merge_across_figure() -> None:
    figure = Block(
        id="figure",
        page_number=1,
        block_type=BlockType.FIGURE,
        text="",
        bbox=BoundingBox(x0=50, y0=120, x1=280, y1=260),
        reading_order_index=120,
        source_type=SourceType.EMBEDDED,
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1, detected_language="pt"),
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
            _block("before", "Texto antes da figura.", 90, 100),
            figure,
            _block("after", "Texto depois da figura.", 280, 290),
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_group_size=5))
    translator._token_count = lambda text: len(text.split())  # type: ignore[method-assign]

    chunks = translator.build_chunks(document)

    assert [chunk.block_ids for chunk in chunks] == [["before"], ["after"]]


def test_separator_only_blocks_are_not_sent_to_translation() -> None:
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
            _block("a", "---", 100, 110),
            _block("b", "Texto real para traduzir.", 140, 150),
            _block("c", "-----", 180, 190),
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_group_size=5))
    translator._token_count = lambda text: len(text.split())  # type: ignore[method-assign]

    chunks = translator.build_chunks(document)

    assert len(chunks) == 1
    assert chunks[0].block_ids == ["b"]
    assert chunks[0].source_text == "Texto real para traduzir."


def test_separator_only_blocks_do_not_call_llm() -> None:
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
        blocks=[_block("a", "---", 100, 110)],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    calls: list[str] = []
    translator._translate_chunk_with_validation = lambda *args: calls.append(str(args[0])) or "BAD"  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert calls == []
    assert translated_doc.blocks[0].text == "---"


def test_list_items_are_not_merged_so_markdown_bullets_are_preserved() -> None:
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
            _list_block("a", "Criterio A: texto uno.", 100),
            _list_block("b", "Criterio B: texto dos.", 140),
            _list_block("c", "Criterio C: texto tres.", 180),
            _list_block("d", "Criterio D: texto cuatro.", 220),
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_group_size=5))
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda chunk: False  # type: ignore[method-assign]
    translator._translate_chunk_with_validation = (  # type: ignore[method-assign]
        lambda text, context, source_language, block_type, **kwargs: text.replace(
            "Criterio", "Criterion"
        )
    )

    translated_doc, _ = translator.translate_document(document, "")
    translated_markdown = MarkdownBuilder().build(translated_doc)

    assert len(translator.build_chunks(document)) == 4
    assert translated_markdown.count("- Criterion") == 4


def test_translated_list_text_strips_accidental_markdown_marker_before_rendering() -> None:
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
        blocks=[_list_block("a", "Criterio A: texto uno.", 100)],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda chunk: False  # type: ignore[method-assign]
    translator._translate_chunk_with_validation = (  # type: ignore[method-assign]
        lambda *args, **kwargs: "- Criterion A: text one."
    )

    translated_doc, _ = translator.translate_document(document, "")
    translated_markdown = MarkdownBuilder().build(translated_doc)

    assert "- Criterion A: text one." in translated_markdown
    assert "- - Criterion" not in translated_markdown


def test_batched_prose_chunk_is_sent_once_and_mapped_back_to_each_block() -> None:
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
            _block("a", "Primeiro bloco.", 100, 110),
            _block("b", "Segundo bloco.", 140, 150),
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_group_size=5))
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda chunk: False  # type: ignore[method-assign]
    translator._token_count = lambda text: len(text.split())  # type: ignore[method-assign]
    calls: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls.append(text)
        return text.replace("Primeiro bloco.", "First block.").replace(
            "Segundo bloco.",
            "Second block.",
        )

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert len(calls) == 1
    assert "Primeiro bloco." in calls[0]
    assert "Segundo bloco." in calls[0]
    assert translated_doc.blocks[0].text == "First block."
    assert translated_doc.blocks[1].text == "Second block."


def test_prepared_context_group_preserves_each_physical_block_target() -> None:
    left = _block("qwen-p2-b27", "Al no haber", 700, 730, x0=50)
    right = _block(
        "qwen-p2-b28",
        "existido cobertura sanitaria en la sanidad publica hasta 1999.",
        100,
        160,
        x0=330,
    )
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="scan.pdf",
            page_count=1,
            detected_language="es",
            translation={"ocr_logical_chunks_prepared": True},
        ),
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
        blocks=[left, right],
        translation_chunks=[
            TranslationChunk(
                id="p0001-c001",
                block_ids=[left.id, right.id],
                source_text=(
                    "Al no haber existido cobertura sanitaria en la sanidad publica hasta 1999."
                ),
                context="Introduccion",
                source_language="es",
                chunk_type="paragraph",
                page_start=1,
                page_end=1,
            )
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_group_size=5))
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda chunk: False  # type: ignore[method-assign]
    calls: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls.append(text)
        return text.replace("Al no haber", "Since there was no").replace(
            "existido cobertura sanitaria en la sanidad publica hasta 1999.",
            "public healthcare coverage until 1999.",
        )

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert len(calls) == 1
    assert translated_doc.blocks[0].text == "Since there was no"
    assert translated_doc.blocks[1].text == "public healthcare coverage until 1999."
    assert translated_doc.blocks[0].metadata["translated_from_block_ids"] == [left.id]
    assert translated_doc.blocks[1].metadata["translated_from_block_ids"] == [right.id]
    assert "merged_into_block_id" not in translated_doc.blocks[1].metadata


def test_long_prepared_context_group_translates_each_physical_block_once() -> None:
    first_text = "BloqueUno " + "contenido " * 78 + "final."
    second_text = "BloqueDos " + "contenido " * 78 + "final."
    first = _block("first", first_text, 100, 300)
    second = _block("second", second_text, 320, 520)
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="scan.pdf",
            page_count=1,
            detected_language="es",
            translation={"ocr_logical_chunks_prepared": True},
        ),
        pages=[],
        blocks=[first, second],
        translation_chunks=[
            TranslationChunk(
                id="p0001-c001",
                block_ids=[first.id, second.id],
                source_text=f"{first_text} {second_text}",
                context="Seccion",
                source_language="es",
                page_start=1,
                page_end=1,
            )
        ],
    )
    translator = MlxTranslator(TranslationSettings(chunk_size=128, max_tokens=2048))
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda chunk: False  # type: ignore[method-assign]
    calls: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls.append(text)
        return text.replace("BloqueUno", "BlockOne").replace("BloqueDos", "BlockTwo")

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert sum("BloqueUno" in call for call in calls) == 1
    assert sum("BloqueDos" in call for call in calls) == 1
    assert translated_doc.blocks[0].text.startswith("BlockOne")
    assert translated_doc.blocks[1].text.startswith("BlockTwo")
    assert [chunk.block_ids for chunk in translated_doc.translation_chunks] == [
        [first.id],
        [second.id],
    ]


def test_malformed_grouped_translation_falls_back_to_each_physical_block() -> None:
    first = _block("first", "Primer bloque fisico.", 100, 130)
    second = _block("second", "Segundo bloque fisico.", 150, 180)
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="scan.pdf",
            page_count=1,
            detected_language="es",
            translation={"ocr_logical_chunks_prepared": True},
        ),
        pages=[],
        blocks=[first, second],
        translation_chunks=[
            TranslationChunk(
                id="p0001-c001",
                block_ids=[first.id, second.id],
                source_text="Primer bloque fisico. Segundo bloque fisico.",
                context="Seccion",
                source_language="es",
                page_start=1,
                page_end=1,
            )
        ],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda chunk: False  # type: ignore[method-assign]
    calls: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls.append(text)
        if "<translathor-segment" in text:
            return "One merged target without segment tags."
        return {
            "Primer bloque fisico.": "First physical block.",
            "Segundo bloque fisico.": "Second physical block.",
        }[text]

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert len(calls) == 4
    assert translated_doc.blocks[0].text == "First physical block."
    assert translated_doc.blocks[1].text == "Second physical block."
    assert [chunk.block_ids for chunk in translated_doc.translation_chunks] == [
        [first.id],
        [second.id],
    ]


def test_grouped_translation_cannot_collapse_a_substantive_region_to_punctuation() -> None:
    first_text = (
        "Zudem berichteten die Befragten, im Rahmen der Behandlung die vor Beginn "
        "erhoffte Selbstwertsteigerung und Akzeptanz der eigenen Person erreicht zu haben. "
        "Andere machten in ihren Schilderungen wiederum deutlich, vorherige "
        "Zielvorstellungen im Rah-"
    )
    second_text = (
        "men der Therapie nicht oder nur in unbefriedigendem Ausmaß erreicht zu haben."
    )
    first = Block(
        id="/page/2/Text/12",
        page_number=3,
        block_type=BlockType.PARAGRAPH,
        text=first_text,
        bbox=BoundingBox(x0=41.32, y0=697.08, x1=285.50, y1=742.78),
        reading_order_index=44,
        source_type=SourceType.EMBEDDED,
        metadata={"section_hierarchy": {"2": "/page/1/SectionHeader/6"}},
    )
    second = Block(
        id="/page/2/Text/13",
        page_number=3,
        block_type=BlockType.PARAGRAPH,
        text=second_text,
        bbox=BoundingBox(x0=297.64, y0=409.41, x1=538.96, y1=430.78),
        reading_order_index=45,
        source_type=SourceType.EMBEDDED,
        metadata={"section_hierarchy": {"2": "/page/1/SectionHeader/6"}},
    )
    logical_source = first_text[:-1] + second_text
    logical_target = (
        "In addition, respondents reported that they had achieved the anticipated increase "
        "in self-esteem and acceptance of themselves during treatment. Others made clear in "
        "their accounts that earlier treatment goals had not been achieved satisfactorily."
    )
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="paper.pdf",
            page_count=3,
            detected_language="de",
            translation={"ocr_logical_chunks_prepared": True},
        ),
        pages=[],
        blocks=[first, second],
        translation_chunks=[
            TranslationChunk(
                id="p0001-c001",
                block_ids=[first.id, second.id],
                source_text=logical_source,
                source_language="de",
                page_start=3,
                page_end=3,
            )
        ],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda _chunk: False  # type: ignore[method-assign]
    calls: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls.append(text)
        if "<translathor-segment" in text:
            return (
                '<translathor-segment index="0">'
                f"{logical_target}"
                "</translathor-segment>"
                '<translathor-segment index="1">.</translathor-segment>'
            )
        if text == logical_source:
            return logical_target
        raise AssertionError(f"Unexpected unsafe per-block fallback: {text}")

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert len(calls) == 3
    first_target = translated_doc.blocks[0].text
    second_target = translated_doc.blocks[1].text
    assert len(first_target.split()) >= 8
    assert len(second_target.split()) >= 3
    assert " ".join(f"{first_target} {second_target}".split()) == logical_target
    assert (first_target + " " + second_target).count("earlier treatment goals") == 1
    assert all(chunk.translated_text != "." for chunk in translated_doc.translation_chunks)
    assert translated_doc.blocks[0].metadata["translation_placement_index"] == 0
    assert translated_doc.blocks[1].metadata["translation_placement_index"] == 1
    readable_markdown = MarkdownBuilder().build(translated_doc)
    assert logical_target in readable_markdown


def test_collapsed_adjacent_paragraphs_fall_back_without_redistribution() -> None:
    first_text = "Das therapeutische Ziel wurde nicht zufriedenstellend erreicht."
    second_text = "Dieser Befund erfordert eine weitere klinische Beurteilung."
    first = _block("first", first_text, 100, 130)
    second = _block("second", second_text, 150, 180)
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="paper.pdf",
            page_count=1,
            detected_language="de",
            translation={"ocr_logical_chunks_prepared": True},
        ),
        pages=[],
        blocks=[first, second],
        translation_chunks=[
            TranslationChunk(
                id="p0001-c001",
                block_ids=[first.id, second.id],
                source_text=f"{first_text} {second_text}",
                source_language="de",
                page_start=1,
                page_end=1,
            )
        ],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda _chunk: False  # type: ignore[method-assign]
    calls: list[str] = []
    contexts: list[str] = []

    def fake_translate(
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        _ = (context, source_language, force_max_tokens)
        calls.append(text)
        contexts.append(context)
        if "<translathor-segment" in text:
            return (
                '<translathor-segment index="0">The therapeutic goal was not reached '
                "satisfactorily and this finding requires further clinical assessment."
                "</translathor-segment>"
                '<translathor-segment index="1">.</translathor-segment>'
            )
        return {
            first_text: "The therapeutic goal was not reached satisfactorily.",
            second_text: "This finding requires further clinical assessment.",
        }[text]

    translator._translate_chunk = fake_translate  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert len(calls) == 4
    assert f"{first_text} {second_text}" not in calls
    assert "not proven to be one paragraph" in contexts[0]
    assert "Do not shift wording or meaning" in contexts[1]
    assert translated_doc.blocks[0].text == (
        "The therapeutic goal was not reached satisfactorily."
    )
    assert translated_doc.blocks[1].text == (
        "This finding requires further clinical assessment."
    )


def test_continuous_redistribution_aligns_asymmetric_sentence_expansion() -> None:
    translator = MlxTranslator(TranslationSettings())
    segments = [
        (
            "first",
            "Im Jahr 2020 wurde die erste Phase beschrieben. Der zweite Ab-",
            BlockType.PARAGRAPH,
        ),
        ("second", "schnitt blieb kurz.", BlockType.PARAGRAPH),
    ]
    logical_source = (
        "Im Jahr 2020 wurde die erste Phase beschrieben. Der zweite Abschnitt blieb kurz."
    )
    first_target_sentence = (
        "In 2020, the unusually complex first phase was documented in comprehensive detail "
        "for every participating clinical centre."
    )
    logical_target = f"{first_target_sentence} The second section remained brief."

    redistributed = translator._redistribute_logical_translation(
        logical_source,
        logical_target,
        segments,
        "de",
        continuity_proven=True,
    )

    assert redistributed is not None
    assert first_target_sentence in redistributed[0]
    assert redistributed[0].endswith("The second")
    assert "2020" in redistributed[0]
    assert "2020" not in redistributed[1]
    assert " ".join(" ".join(redistributed).split()) == logical_target


def test_continuous_redistribution_rejects_reordered_or_duplicated_numbers() -> None:
    translator = MlxTranslator(TranslationSettings())
    segments = [
        ("first", "Die Studie begann 2020 und dau-", BlockType.PARAGRAPH),
        ("second", "erte bis 2021 an.", BlockType.PARAGRAPH),
    ]
    logical_source = "Die Studie begann 2020 und dauerte bis 2021 an."

    for unsafe_target in (
        "The study continued until 2021 after beginning in 2020.",
        "The study began in 2020 and continued until 2021 and 2021.",
    ):
        assert (
            translator._redistribute_logical_translation(
                logical_source,
                unsafe_target,
                segments,
                "de",
                continuity_proven=True,
            )
            is None
        )


def test_continuous_redistribution_keeps_numeric_markers_in_their_source_region() -> None:
    translator = MlxTranslator(TranslationSettings())
    numeric_segments = [
        ("first", "Die ausführliche Untersuchung ende-", BlockType.PARAGRAPH),
        ("second", "te 2021.", BlockType.PARAGRAPH),
    ]
    assert (
        translator._redistribute_logical_translation(
            "Die ausführliche Untersuchung endete 2021.",
            "In 2021, the extensive study ended.",
            numeric_segments,
            "de",
            continuity_proven=True,
        )
        is None
    )


def test_continuous_redistribution_requires_explicit_continuity_proof() -> None:
    translator = MlxTranslator(TranslationSettings())
    segments = [
        ("first", "Primer párrafo independiente.", BlockType.PARAGRAPH),
        ("second", "Segundo párrafo independiente.", BlockType.PARAGRAPH),
    ]

    assert (
        translator._redistribute_logical_translation(
            "Primer párrafo independiente. Segundo párrafo independiente.",
            "First independent paragraph. Second independent paragraph.",
            segments,
            "es",
            continuity_proven=False,
        )
        is None
    )


def test_tagged_segments_validate_the_complete_passage_language() -> None:
    first_text = "Los pacientes reciben seguimiento clinico."
    second_text = "El tratamiento mantiene resultados terapeuticos."
    first = _block("first", first_text, 100, 130)
    second = _block("second", second_text, 150, 180)
    document = DocumentModel(
        metadata=DocumentMetadata(
            filename="scan.pdf",
            page_count=1,
            detected_language="es",
            translation={"ocr_logical_chunks_prepared": True},
        ),
        pages=[],
        blocks=[first, second],
        translation_chunks=[
            TranslationChunk(
                id="p0001-c001",
                block_ids=[first.id, second.id],
                source_text=f"{first_text} {second_text}",
                source_language="es",
                page_start=1,
                page_end=1,
            )
        ],
    )
    translator = MlxTranslator(TranslationSettings())
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._detect_language_with_confidence = lambda _text: ("es", 0.99)  # type: ignore[method-assign]
    calls: list[str] = []
    translator._translate_chunk = lambda text, *args, **kwargs: calls.append(text) or text  # type: ignore[method-assign]

    translated_doc, _ = translator.translate_document(document, "")

    assert len(calls) == 6
    assert translated_doc.blocks[0].text == first_text
    assert translated_doc.blocks[1].text == second_text
    assert [chunk.status for chunk in translated_doc.translation_chunks] == [
        "translation_failed",
        "translation_failed",
    ]
    assert all(
        chunk.reason == "translation_output_matches_source"
        for chunk in translated_doc.translation_chunks
    )


def test_translation_prompt_includes_context_and_source_language() -> None:
    prompt = MlxTranslator(TranslationSettings())._build_prompt(
        "Texto que traducir.",
        "Use the prior section's terminology.",
        "es",
    )

    assert "SOURCE LANGUAGE: es" in prompt
    assert "Use the prior section's terminology." in prompt
    assert "TEXT:\nTexto que traducir." in prompt


def test_qwen_markdown_parser_preserves_header_and_footer_text() -> None:
    blocks = QwenMarkdownParser()._blocks_from_markdown(
        "Repeated Article Title\n\nBody text.\n\nPage 2",
        page_number=2,
        start_order=0,
    )

    assert blocks[0].text == "Repeated Article Title"
    assert blocks[1].text == "Body text."
    assert blocks[2].text == "Page 2"


def test_translator_includes_header_and_footer_blocks() -> None:
    header = _block("header", "Repeated Article Title", 20, 30).model_copy(
        update={"block_type": BlockType.HEADER}
    )
    footer = _block("footer", "Page 2", 760, 770).model_copy(
        update={"block_type": BlockType.FOOTER}
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1),
        pages=[],
        blocks=[header, footer],
    )

    chunks = MlxTranslator(TranslationSettings()).build_chunks(document)

    assert [chunk.source_text for chunk in chunks] == ["Repeated Article Title", "Page 2"]
