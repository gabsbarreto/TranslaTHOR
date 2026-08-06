from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
import pytest

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
from app.services.cross_page_continuation import (
    CONTINUATION_GROUP_ID,
    CONTINUATION_VISIBLE_INTERVENING_IDS,
    CrossPageContinuationResolver,
)
from app.services.markdown_builder import MarkdownBuilder
from app.services.ocr_to_translation_parser import OCRToTranslationParser
from app.services.original_layout_reconstructor import OriginalLayoutReconstructor
from app.services.translator_mlx import MlxTranslator, TranslationSettings


def test_nonterminal_paragraph_is_linked_across_consecutive_pages() -> None:
    document = _document(
        [
            _block(
                "first", 1, "Die Behandlung wird über einen längeren Zeitraum", 0, y0=690, y1=780
            ),
            _block("second", 2, "fortgeführt und regelmäßig kontrolliert.", 1, y0=40, y1=110),
        ]
    )
    resolution = CrossPageContinuationResolver().resolve(document)

    assert len(resolution.groups) == 1
    group = resolution.groups[0]
    assert group.block_ids == ("first", "second")
    assert group.decision_level == "proven"
    assert "previous_nonterminal" in group.evidence
    assert document.blocks[0].metadata[CONTINUATION_GROUP_ID] == group.id
    assert document.blocks[0].metadata["continues_to_next_page"] is True
    assert document.blocks[1].metadata["continues_from_previous_page"] is True


def test_split_word_is_joined_across_page_boundary_without_geometry() -> None:
    document = _document(
        [
            _block("first", 1, "Die internationale Behand-", 0, bbox=False),
            _block("second", 2, "lung wurde fortgesetzt.", 1, bbox=False),
        ]
    )

    group = CrossPageContinuationResolver().resolve(document).groups[0]

    assert group.block_ids == ("first", "second")
    assert "split_word_hyphen" in group.evidence
    assert "strong_text_without_geometry" in group.evidence


def test_headers_footers_and_page_numbers_are_transparent() -> None:
    document = _document(
        [
            _block("first", 1, "Der Bericht beschreibt unter anderem", 0, y0=690, y1=760),
            _block("footer", 1, "Journal", 1, BlockType.FOOTER, y0=775, y1=790),
            _block("page-number", 1, "12", 2, BlockType.PAGE_NUMBER, y0=780, y1=798),
            _block("header", 2, "Article title", 3, BlockType.HEADER, y0=5, y1=20),
            _block("second", 2, "die klinischen Ergebnisse der Untersuchung.", 4, y0=35, y1=100),
        ]
    )

    group = CrossPageContinuationResolver().resolve(document).groups[0]

    assert group.intervening_block_ids == ("footer", "page-number", "header")
    assert group.visible_intervening_block_ids == ()
    assert "transparent_margin_blocks" in group.evidence


def test_repeated_margin_text_misclassified_as_paragraph_is_transparent() -> None:
    document = _document(
        [
            _block("first", 1, "Die Untersuchung berücksichtigt", 0, y0=680, y1=750),
            _block("margin-1", 1, "Clinical review 12", 1, y0=780, y1=798),
            _block("margin-2", 2, "Clinical review 13", 2, y0=4, y1=20),
            _block("second", 2, "auch die langfristigen Auswirkungen.", 3, y0=38, y1=105),
        ]
    )

    group = CrossPageContinuationResolver().resolve(document).groups[0]

    assert group.intervening_block_ids == ("margin-1", "margin-2")
    assert document.blocks[1].metadata["cross_page_transparent_reason"] == "repeated_margin_text"
    assert document.blocks[2].metadata["cross_page_transparent_reason"] == "repeated_margin_text"


def test_table_and_caption_are_intervening_but_not_prose_or_duplicate_units() -> None:
    document = _document(
        [
            _block("first", 1, "El seguimiento clínico incluye", 0, y0=690, y1=775),
            _block(
                "table",
                2,
                "<table><tr><td>Dato</td></tr></table>",
                1,
                BlockType.TABLE,
                y0=35,
                y1=150,
            ),
            _block("caption", 2, "TABLA I. Resultados", 2, BlockType.CAPTION, y0=155, y1=175),
            _block("second", 2, "también la respuesta terapéutica.", 3, y0=185, y1=260),
        ],
        detected_language="es",
    )

    resolution = CrossPageContinuationResolver().resolve(document)
    chunks = MlxTranslator(TranslationSettings(chunk_group_size=1)).build_chunks(document)

    assert resolution.groups[0].block_ids == ("first", "second")
    assert resolution.groups[0].visible_intervening_block_ids == ("table", "caption")
    prose_chunk = next(chunk for chunk in chunks if chunk.continuation_group_id)
    assert prose_chunk.block_ids == ["first", "second"]
    assert "table" not in prose_chunk.block_ids
    assert sum(chunk.block_ids == ["table"] for chunk in chunks) == 1


@pytest.mark.parametrize("object_type", [BlockType.FIGURE, BlockType.EQUATION])
def test_figure_or_equation_can_intervene_without_entering_prose(
    object_type: BlockType,
) -> None:
    document = _document(
        [
            _block("first", 1, "Die Auswertung zeigt", 0, y0=700, y1=775),
            _block(
                "object",
                2,
                "x = y" if object_type == BlockType.EQUATION else "",
                1,
                object_type,
                y0=30,
                y1=145,
            ),
            _block("second", 2, "eine deutliche Veränderung der Werte.", 2, y0=155, y1=230),
        ]
    )

    group = CrossPageContinuationResolver().resolve(document).groups[0]

    assert group.block_ids == ("first", "second")
    assert group.visible_intervening_block_ids == ("object",)


@pytest.mark.parametrize(
    "ending",
    [
        "Ein vollständiger Satz.",
        "Ein vollständiger Satz!",
        "Ein vollständiger Satz?",
        "Ein unvollständiger Gedanke…",
        "Ein vollständiger Satz. [12]”",
    ],
)
def test_genuine_terminal_punctuation_prevents_continuation(ending: str) -> None:
    document = _document(
        [
            _block("first", 1, ending, 0, y0=700, y1=780),
            _block("second", 2, "der nächste Absatz beginnt hier.", 1, y0=35, y1=105),
        ]
    )

    assert CrossPageContinuationResolver().resolve(document).groups == ()


def test_heading_or_changed_section_hierarchy_prevents_continuation() -> None:
    heading_document = _document(
        [
            _block("first", 1, "Der vorherige Text endet ohne Punkt", 0, y0=700, y1=780),
            _block("heading", 2, "Ergebnisse", 1, BlockType.HEADING, y0=30, y1=60),
            _block("second", 2, "der neue Abschnitt beginnt.", 2, y0=70, y1=130),
        ]
    )
    first = _block("first", 1, "Der vorherige Text endet ohne Punkt", 0, y0=700, y1=780)
    second = _block("second", 2, "der nächste Text beginnt.", 1, y0=35, y1=105)
    first.metadata["section_hierarchy"] = {"1": "methods"}
    second.metadata["section_hierarchy"] = {"1": "results"}

    assert CrossPageContinuationResolver().resolve(heading_document).groups == ()
    assert CrossPageContinuationResolver().resolve(_document([first, second])).groups == ()


@pytest.mark.parametrize(
    "boundary_type,boundary_y",
    [
        (BlockType.LIST, 760),
        (BlockType.REFERENCE, 760),
        (BlockType.FOOTNOTE, 600),
    ],
)
def test_list_reference_or_inline_footnote_blocks_body_merge(
    boundary_type: BlockType,
    boundary_y: int,
) -> None:
    document = _document(
        [
            _block("first", 1, "Der Absatz setzt sich vermutlich fort", 0, y0=680, y1=740),
            _block(
                "boundary",
                1,
                "1. Separater Inhalt",
                1,
                boundary_type,
                y0=boundary_y,
                y1=boundary_y + 25,
            ),
            _block("second", 2, "auf der folgenden Seite.", 2, y0=35, y1=100),
        ]
    )

    assert CrossPageContinuationResolver().resolve(document).groups == ()


@pytest.mark.parametrize(
    "start",
    [
        "NASA liefert weitere Daten.",
        "Müller beschreibt das Ergebnis.",
        "Behandlung bleibt erforderlich.",
    ],
)
def test_uppercase_acronym_proper_name_or_german_noun_is_not_rejected(start: str) -> None:
    document = _document(
        [
            _block("first", 1, "Die Analyse berücksichtigt zusätzlich", 0, y0=700, y1=780),
            _block("second", 2, start, 1, y0=35, y1=105),
        ]
    )

    group = CrossPageContinuationResolver().resolve(document).groups[0]

    assert "uppercase_start_layout_supported" in group.evidence


@pytest.mark.parametrize("ending", ["Der Text geht weiter [12]”", "Der Text geht weiter)"])
def test_trailing_citation_or_bare_closing_bracket_is_not_a_full_stop(ending: str) -> None:
    document = _document(
        [
            _block("first", 1, ending, 0, y0=700, y1=780),
            _block("second", 2, "und endet erst auf der nächsten Seite.", 1, y0=35, y1=105),
        ]
    )

    assert len(CrossPageContinuationResolver().resolve(document).groups) == 1


def test_et_al_abbreviation_can_continue_with_strong_layout_evidence() -> None:
    document = _document(
        [
            _block("first", 1, "Wie bereits gezeigt von Müller et al.", 0, y0=700, y1=780),
            _block("second", 2, "ist dieser Effekt langfristig stabil.", 1, y0=35, y1=105),
        ]
    )

    group = CrossPageContinuationResolver().resolve(document).groups[0]

    assert "abbreviation_terminal_overridden" in group.evidence


def test_object_only_page_can_bridge_nonconsecutive_prose_pages() -> None:
    document = _document(
        [
            _block(
                "first",
                1,
                "At the Amsterdam clinic, the ratio was 1.77/1 in favor of the",
                0,
                y0=700,
                y1=780,
            ),
            _block(
                "table",
                2,
                "<table><tr><td>Amsterdam</td></tr></table>",
                1,
                BlockType.TABLE,
                y0=35,
                y1=690,
            ),
            _block(
                "caption",
                2,
                "Table 1. International cohorts",
                2,
                BlockType.CAPTION,
                y0=700,
                y1=735,
            ),
            _block(
                "second",
                3,
                "MT; in the second period the ratio favored HT.",
                3,
                y0=35,
                y1=105,
            ),
        ]
    )
    document.blocks[0].style_hints = {}
    document.blocks[3].style_hints = {}

    resolution = CrossPageContinuationResolver().resolve(document)

    assert len(resolution.groups) == 1
    group = resolution.groups[0]
    assert group.block_ids == ("first", "second")
    assert group.links[0].previous_page == 1
    assert group.links[0].current_page == 3
    assert group.visible_intervening_block_ids == ("table", "caption")
    assert "consecutive_physical_page_span" in group.evidence
    assert "object_only_intervening_pages" in group.evidence
    assert "uppercase_start_layout_supported" in group.evidence
    seam = document.blocks[0].metadata["cross_page_continuation_seams"][0]
    assert seam["intermediate_page_numbers"] == [2]
    segments = [
        ("first", document.blocks[0].text, BlockType.PARAGRAPH),
        ("second", document.blocks[3].text, BlockType.PARAGRAPH),
    ]
    assert MlxTranslator(TranslationSettings())._physical_segments_form_continuous_paragraph(
        segments,
        {block.id: block for block in document.blocks},
    )


def test_multiple_object_only_pages_can_bridge_one_proven_prose_seam() -> None:
    document = _document(
        [
            _block("first", 1, "Die Auswertung berücksichtigt die", 0, y0=700, y1=780),
            _block(
                "table",
                2,
                "<table><tr><td>Daten</td></tr></table>",
                1,
                BlockType.TABLE,
                y0=35,
                y1=735,
            ),
            _block("figure", 3, "", 2, BlockType.FIGURE, y0=35, y1=735),
            _block("second", 4, "NASA-Ergebnisse der Folgestudie.", 3, y0=35, y1=105),
        ]
    )

    group = CrossPageContinuationResolver().resolve(document).groups[0]

    assert group.block_ids == ("first", "second")
    assert group.visible_intervening_block_ids == ("table", "figure")
    seam = document.blocks[0].metadata["cross_page_continuation_seams"][0]
    assert seam["intermediate_page_numbers"] == [2, 3]


def test_object_only_page_bridge_is_shared_by_ocr_and_marker_units() -> None:
    blocks = [
        _block(
            "first",
            1,
            "La proporción fue favorable a las",
            0,
            y0=700,
            y1=780,
            source=SourceType.OCR,
        ),
        _block(
            "table-1",
            2,
            "<table><tr><td>Datos</td></tr></table>",
            1,
            BlockType.TABLE,
            y0=35,
            y1=350,
            source=SourceType.OCR,
        ),
        _block(
            "table-2",
            2,
            "<table><tr><td>Más datos</td></tr></table>",
            2,
            BlockType.TABLE,
            y0=360,
            y1=735,
            source=SourceType.OCR,
        ),
        _block(
            "second",
            3,
            "MT; durante el segundo período cambió la proporción.",
            3,
            y0=35,
            y1=105,
            source=SourceType.OCR,
        ),
    ]
    base = _document(blocks, detected_language="es")
    prepared = OCRToTranslationParser().prepare(base.model_copy(deep=True)).document
    marker = base.model_copy(deep=True)
    for block in marker.blocks:
        block.source_type = SourceType.EMBEDDED
    marker_chunks = MlxTranslator(TranslationSettings(chunk_group_size=1)).build_chunks(marker)

    prepared_chunk = next(
        chunk for chunk in prepared.translation_chunks if chunk.continuation_group_id
    )
    marker_chunk = next(chunk for chunk in marker_chunks if chunk.continuation_group_id)
    assert prepared_chunk.block_ids == marker_chunk.block_ids == ["first", "second"]
    assert "object_only_intervening_pages" in prepared_chunk.continuation_evidence
    assert prepared_chunk.continuation_evidence == marker_chunk.continuation_evidence
    for table_id in ("table-1", "table-2"):
        assert sum(chunk.block_ids == [table_id] for chunk in prepared.translation_chunks) == 1
        assert sum(chunk.block_ids == [table_id] for chunk in marker_chunks) == 1


def test_blank_intervening_page_does_not_bridge_nonconsecutive_prose() -> None:
    document = _document(
        [
            _block("first", 1, "Der Absatz setzt sich fort", 0, y0=700, y1=780),
            _block("second", 3, "auf einer nicht angrenzenden Seite.", 1, y0=35, y1=105),
        ]
    )

    assert CrossPageContinuationResolver().resolve(document).groups == ()


@pytest.mark.parametrize(
    "blocker_type,blocker_text",
    [
        (BlockType.HEADING, "Neue Ergebnisse"),
        (BlockType.PARAGRAPH, "Ein eigenständiger Absatz."),
        (BlockType.LIST, "1. Eigenständiger Listenpunkt"),
        (BlockType.REFERENCE, "[1] Eigenständige Referenz"),
        (BlockType.FOOTNOTE, "1 Eigenständige Fußnote"),
    ],
)
def test_structural_content_on_intermediate_page_blocks_object_page_bridge(
    blocker_type: BlockType,
    blocker_text: str,
) -> None:
    document = _document(
        [
            _block("first", 1, "Der Absatz setzt sich fort", 0, y0=700, y1=780),
            _block(
                "table",
                2,
                "<table><tr><td>Daten</td></tr></table>",
                1,
                BlockType.TABLE,
                y0=35,
                y1=560,
            ),
            _block("blocker", 2, blocker_text, 2, blocker_type, y0=590, y1=630),
            _block("second", 3, "auf der dritten Seite.", 3, y0=35, y1=105),
        ]
    )

    assert CrossPageContinuationResolver().resolve(document).groups == ()


def test_paragraph_can_span_three_consecutive_pages() -> None:
    document = _document(
        [
            _block("first", 1, "Der lange Absatz beginnt und", 0, y0=700, y1=780),
            _block(
                "middle",
                2,
                "setzt sich über die gesamte zweite Seite fort und",
                1,
                y0=35,
                y1=780,
            ),
            _block("last", 3, "endet schließlich auf der dritten Seite.", 2, y0=35, y1=110),
        ]
    )

    resolution = CrossPageContinuationResolver().resolve(document)

    assert len(resolution.groups) == 1
    assert resolution.groups[0].block_ids == ("first", "middle", "last")
    assert len(resolution.groups[0].links) == 2


def test_ambiguous_seam_without_geometry_remains_separate() -> None:
    document = _document(
        [
            _block("first", 1, "Ein Absatz ohne eindeutiges Ende", 0, bbox=False),
            _block("second", 2, "ein möglicherweise neuer Absatz.", 1, bbox=False),
        ]
    )

    assert CrossPageContinuationResolver().resolve(document).groups == ()


def test_surya_decomposed_caption_bound_table_is_one_intervening_object() -> None:
    document = _document(
        [
            _block("first", 1, "En adolescentes se seleccionan casos muy", 0, y0=690, y1=750),
            _block("footnote", 1, "1 Nota de página.", 1, BlockType.FOOTNOTE, y0=760, y1=790),
            _block("table-heading", 2, "A.- Criterios de elegibilidad", 2, y0=40, y1=70),
            _block("table-intro", 2, "Existen los criterios siguientes:", 3, y0=70, y1=90),
            _block("table-list", 2, "1.- Primero. 2.- Segundo.", 4, BlockType.LIST, y0=90, y1=180),
            _block("table-caption", 2, "TABLA II. Criterios", 5, BlockType.CAPTION, y0=185, y1=205),
            _block("second", 2, "seleccionados para retrasar la pubertad.", 6, y0=235, y1=310),
        ],
        detected_language="es",
    )

    group = CrossPageContinuationResolver().resolve(document).groups[0]

    assert group.block_ids == ("first", "second")
    assert set(group.visible_intervening_block_ids) == {
        "footnote",
        "table-heading",
        "table-intro",
        "table-list",
        "table-caption",
    }
    assert "caption_bound_table_region" in group.evidence
    assert "dedicated_page_footnote_lane" in group.evidence


def test_prepared_ocr_and_marker_units_use_the_same_resolver_group() -> None:
    blocks = [
        _block(
            "first", 1, "El tratamiento continúa durante", 0, y0=700, y1=780, source=SourceType.OCR
        ),
        _block("header", 2, "Título", 1, BlockType.HEADER, y0=5, y1=20, source=SourceType.OCR),
        _block("second", 2, "los meses siguientes.", 2, y0=35, y1=105, source=SourceType.OCR),
    ]
    prepared = OCRToTranslationParser().prepare(_document(blocks, detected_language="es")).document

    marker_blocks = [
        block.model_copy(update={"source_type": SourceType.EMBEDDED, "metadata": {}})
        for block in blocks
    ]
    marker = _document(marker_blocks, detected_language="es")
    marker_chunks = MlxTranslator(TranslationSettings(chunk_group_size=1)).build_chunks(marker)

    prepared_chunk = next(
        chunk for chunk in prepared.translation_chunks if chunk.continuation_group_id
    )
    marker_chunk = next(chunk for chunk in marker_chunks if chunk.continuation_group_id)
    assert prepared_chunk.block_ids == marker_chunk.block_ids == ["first", "second"]
    assert prepared_chunk.continuation_evidence == marker_chunk.continuation_evidence


def test_proven_passage_across_object_page_is_translated_once_and_mapped_locally() -> None:
    document = _document(
        [
            _block(
                "first",
                1,
                "El tratamiento clínico continúa con una evaluación detallada de la",
                0,
                y0=690,
                y1=780,
            ),
            _block(
                "figure",
                2,
                "",
                1,
                BlockType.FIGURE,
                y0=35,
                y1=735,
            ),
            _block(
                "second",
                3,
                "respuesta terapéutica durante los meses siguientes.",
                2,
                y0=35,
                y1=110,
            ),
        ],
        detected_language="es",
    )
    logical_source = (
        "El tratamiento clínico continúa con una evaluación detallada de la respuesta "
        "terapéutica durante los meses siguientes."
    )
    logical_target = (
        "The clinical treatment continues with a detailed evaluation of the therapeutic "
        "response during the following months."
    )
    calls: list[str] = []
    translator = MlxTranslator(TranslationSettings(chunk_group_size=1))
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda _chunk: False  # type: ignore[method-assign]
    translator._is_acceptable_chunk_translation = lambda *args, **kwargs: True  # type: ignore[method-assign]
    translator._is_valid_chunk_translation_structure = lambda *args, **kwargs: True  # type: ignore[method-assign]
    translator._translation_acceptance_issue = lambda *args, **kwargs: None  # type: ignore[method-assign]

    def translate(text: str, *args, **kwargs) -> str:
        _ = (args, kwargs)
        calls.append(text)
        assert text == logical_source
        return logical_target

    translator._translate_chunk = translate  # type: ignore[method-assign]

    translated, _ = translator.translate_document(document, "")

    assert calls == [logical_source]
    paragraphs = [block for block in translated.blocks if block.block_type == BlockType.PARAGRAPH]
    assert " ".join(block.text for block in paragraphs) == logical_target
    assert [chunk.block_ids for chunk in translated.translation_chunks] == [["first"], ["second"]]
    assert [(chunk.page_start, chunk.page_end) for chunk in translated.translation_chunks] == [
        (1, 1),
        (3, 3),
    ]
    for block in paragraphs:
        assert block.metadata["translated_from_block_ids"] == [block.id]
        assert "merged_into_block_id" not in block.metadata
        assert all(
            next(item for item in translated.blocks if item.id == source_id).page_number
            == block.page_number
            for source_id in block.metadata["translated_from_block_ids"]
        )
    figure = next(block for block in translated.blocks if block.id == "figure")
    assert figure.metadata["translation_exclusion_reason"] == "figure_internal_text_preserved"


def test_page_six_to_seven_margin_interruption_does_not_duplicate_continuation() -> None:
    first_id = "surya2-p0006-b0017"
    second_id = "surya2-p0007-b0001"
    document = _document(
        [
            _block(
                first_id,
                6,
                "Siempre se justifica por el",
                0,
                y0=630,
                y1=745,
                x0=304,
                x1=562,
                source=SourceType.OCR,
            ),
            _block(
                "surya2-p0006-b0018",
                6,
                "278",
                1,
                BlockType.PAGE_NUMBER,
                y0=755,
                y1=775,
                source=SourceType.OCR,
            ),
            _block(
                "surya2-p0006-b0019",
                6,
                "Revista clínica",
                2,
                BlockType.FOOTER,
                y0=778,
                y1=798,
                source=SourceType.OCR,
            ),
            _block(
                "surya2-p0007-b0000",
                7,
                "Título del artículo",
                3,
                BlockType.HEADER,
                y0=5,
                y1=25,
                source=SourceType.OCR,
            ),
            _block(
                second_id,
                7,
                "elevado coste, pero no hay diferencias relevantes en cuanto al precio.",
                4,
                y0=40,
                y1=205,
                x0=35,
                x1=287,
                source=SourceType.OCR,
            ),
        ],
        detected_language="es",
    )
    prepared = OCRToTranslationParser().prepare(document).document
    logical_source = (
        "Siempre se justifica por el elevado coste, pero no hay diferencias relevantes "
        "en cuanto al precio."
    )
    logical_target = (
        "It is always justified by the high cost, but there are no relevant price differences."
    )
    calls: list[str] = []
    translator = MlxTranslator(TranslationSettings(chunk_group_size=1))
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator._is_already_english = lambda _chunk: False  # type: ignore[method-assign]
    translator._is_acceptable_chunk_translation = lambda *args, **kwargs: True  # type: ignore[method-assign]
    translator._is_valid_chunk_translation_structure = lambda *args, **kwargs: True  # type: ignore[method-assign]
    translator._translation_acceptance_issue = lambda *args, **kwargs: None  # type: ignore[method-assign]

    def translate(text: str, *args: Any, **kwargs: Any) -> str:
        _ = (args, kwargs)
        calls.append(text)
        assert text == logical_source
        return logical_target

    translator._translate_chunk = translate  # type: ignore[method-assign]

    translated, _ = translator.translate_document(prepared, "")

    first = next(block for block in translated.blocks if block.id == first_id)
    second = next(block for block in translated.blocks if block.id == second_id)
    combined_target = " ".join((first.text, second.text))
    assert calls == [logical_source]
    assert combined_target == logical_target
    assert combined_target.casefold().count("high cost") == 1
    assert first.text and second.text
    assert first.page_number == 6
    assert second.page_number == 7
    assert first.metadata["translated_from_block_ids"] == [first_id]
    assert second.metadata["translated_from_block_ids"] == [second_id]
    assert [chunk.block_ids for chunk in translated.translation_chunks] == [
        [first_id],
        [second_id],
    ]


def test_tagged_boundary_failure_uses_complete_context_then_safe_source_fallback() -> None:
    first = _block("first", 1, "Primer fragmento que continúa", 0, y0=700, y1=780)
    second = _block("second", 2, "en la página siguiente.", 1, y0=35, y1=105)
    document = _document([first, second], detected_language="es")
    CrossPageContinuationResolver().resolve(document)
    segments = [
        (first.id, first.text, first.block_type),
        (second.id, second.text, second.block_type),
    ]
    translator = MlxTranslator(TranslationSettings())
    logical_calls: list[str] = []
    tagged_calls: list[str] = []
    fallback_contexts: list[str] = []
    translator._translate_chunk_with_validation = (  # type: ignore[method-assign]
        lambda text, *args, **kwargs: logical_calls.append(text) or "A complete English passage."
    )
    translator._redistribute_logical_translation = lambda *args, **kwargs: None  # type: ignore[method-assign]

    def malformed_tags(text: str, *args, **kwargs) -> str:
        _ = (args, kwargs)
        tagged_calls.append(text)
        return "mapping tags were lost"

    translator._translate_chunk = malformed_tags  # type: ignore[method-assign]

    def invalid_separate(text: str, context: str, *args, **kwargs) -> str:
        _ = (args, kwargs)
        fallback_contexts.append(context)
        return text

    translator._translate_single_physical_segment = invalid_separate  # type: ignore[method-assign]
    translator._translation_acceptance_issue = lambda *args, **kwargs: "source_preserved"  # type: ignore[method-assign]

    targets = translator._translate_tagged_physical_segments(
        segments,
        "",
        "es",
        f"Complete passage: {first.text} {second.text}",
        {first.id: first, second.id: second},
        continuity_proven=True,
    )

    assert logical_calls == [f"{first.text} {second.text}"]
    assert len(tagged_calls) == 2
    assert len(fallback_contexts) == 2
    assert all(first.text in context and second.text in context for context in fallback_contexts)
    assert targets == [first.text, second.text]


def test_oversized_continuation_uses_seam_aware_bounded_context() -> None:
    segments = [
        ("first", "FIRST " + "uno " * 75, BlockType.PARAGRAPH),
        ("middle", "MIDDLE " + "dos " * 75, BlockType.PARAGRAPH),
        ("last", "LAST " + "tres " * 75, BlockType.PARAGRAPH),
    ]
    translator = MlxTranslator(TranslationSettings(chunk_size=128, max_tokens=1024))
    translator._token_count = lambda text: len(text.split())  # type: ignore[method-assign]
    contexts: list[str] = []

    def translate_segment(text: str, context: str, *args, **kwargs) -> str:
        _ = (args, kwargs)
        contexts.append(context)
        return text

    translator._translate_single_physical_segment = translate_segment  # type: ignore[method-assign]

    targets = translator._translate_physical_segments(
        segments,
        "Section",
        "es",
        " ".join(segment[1] for segment in segments),
        {},
    )

    assert targets == [segment[1] for segment in segments]
    assert len(contexts) == 3
    assert all("adjacent seam context" in context for context in contexts)
    assert "[following fragment]" in contexts[0]
    assert "[preceding fragment]" in contexts[1]
    assert "[following fragment]" in contexts[1]
    assert "[preceding fragment]" in contexts[2]
    assert all(len(context.split()) < 128 for context in contexts)


def test_readable_output_keeps_object_only_page_between_translated_fragments() -> None:
    document = _document(
        [
            _block("first", 1, "Source first", 0, y0=700, y1=780),
            _block(
                "table",
                2,
                "<table><tr><td>OBJECT</td></tr></table>",
                1,
                BlockType.TABLE,
                y0=30,
                y1=150,
            ),
            _block("second", 3, "source second.", 2, y0=35, y1=105),
        ]
    )
    CrossPageContinuationResolver().resolve(document)
    document.blocks[0].text = "Translated first"
    document.blocks[2].text = "translated second."
    for index, block in enumerate((document.blocks[0], document.blocks[2])):
        block.metadata.update(
            {
                "translation_placement_group_id": "placement",
                "translation_placement_index": index,
                "translation_placement_count": 2,
            }
        )

    markdown = MarkdownBuilder().build(document)

    assert markdown.index("Translated first") < markdown.index("OBJECT")
    assert markdown.index("OBJECT") < markdown.index("translated second.")
    assert document.blocks[0].metadata[CONTINUATION_VISIBLE_INTERVENING_IDS] == ["table"]


def test_original_layout_reconstructor_receives_page_local_regions_across_object_page(
    tmp_path: Path,
) -> None:
    source_pdf = tmp_path / "source.pdf"
    pdf = fitz.open()
    page_one = pdf.new_page(width=600, height=800)
    page_one.insert_textbox(fitz.Rect(40, 690, 280, 780), "QUELLTEXT EINS", fontsize=10)
    object_page = pdf.new_page(width=600, height=800)
    object_page.draw_rect(fitz.Rect(40, 35, 560, 735), color=(0, 0, 0))
    page_three = pdf.new_page(width=600, height=800)
    page_three.insert_textbox(fitz.Rect(40, 35, 280, 110), "QUELLTEXT ZWEI", fontsize=10)
    pdf.save(source_pdf)
    pdf.close()

    first = _block("first", 1, "QUELLTEXT EINS", 0, y0=690, y1=780)
    figure = _block("figure", 2, "", 1, BlockType.FIGURE, y0=35, y1=735)
    second = _block("second", 3, "QUELLTEXT ZWEI", 2, y0=35, y1=110)
    document = _document([first, figure, second])
    resolution = CrossPageContinuationResolver().resolve(document)
    assert resolution.groups[0].block_ids == ("first", "second")
    first.metadata.update({"source_text": first.text, "translated_from_block_ids": [first.id]})
    second.metadata.update({"source_text": second.text, "translated_from_block_ids": [second.id]})
    first.text = "TRANSLATED ONE"
    second.text = "TRANSLATED TWO"
    captured_regions: list[Any] = []

    class CapturingReconstructor(OriginalLayoutReconstructor):
        def _replacement_regions(self, *args: Any, **kwargs: Any) -> list[Any]:
            regions = super()._replacement_regions(*args, **kwargs)
            captured_regions.extend(regions)
            return regions

    CapturingReconstructor().reconstruct(
        source_pdf_path=source_pdf,
        output_pdf_path=tmp_path / "translated.pdf",
        document=document,
        report_path=tmp_path / "report.json",
    )

    assert len(captured_regions) == 2
    assert all(len(region.block_ids) == 1 for region in captured_regions)
    assert all(
        next(block for block in document.blocks if block.id == region.block_ids[0]).page_number
        == region.page_number
        for region in captured_regions
    )


def test_legacy_translation_chunk_loads_with_continuation_defaults() -> None:
    chunk = TranslationChunk.model_validate(
        {
            "id": "legacy",
            "block_ids": ["one"],
            "source_text": "Legacy artifact text.",
        }
    )

    assert chunk.continuation_group_id is None
    assert chunk.continuation_decision_level is None
    assert chunk.continuation_confidence is None
    assert chunk.continuation_evidence == []
    assert chunk.continuation_intervening_block_ids == []


def _document(
    blocks: list[Block],
    *,
    detected_language: str = "de",
) -> DocumentModel:
    page_count = max((block.page_number for block in blocks), default=1)
    return DocumentModel(
        metadata=DocumentMetadata(
            filename="paper.pdf",
            page_count=page_count,
            detected_language=detected_language,
        ),
        pages=[
            PageMetadata(
                page_number=page_number,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
            for page_number in range(1, page_count + 1)
        ],
        blocks=blocks,
    )


def _block(
    block_id: str,
    page_number: int,
    text: str,
    order: int,
    block_type: BlockType = BlockType.PARAGRAPH,
    *,
    y0: float = 100,
    y1: float = 160,
    x0: float = 40,
    x1: float = 280,
    bbox: bool = True,
    source: SourceType = SourceType.EMBEDDED,
) -> Block:
    return Block(
        id=block_id,
        page_number=page_number,
        block_type=block_type,
        text=text,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1) if bbox else None,
        reading_order_index=order,
        source_type=source,
        style_hints={"font_size": 10},
    )
