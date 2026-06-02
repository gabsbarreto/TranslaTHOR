from __future__ import annotations

from app.models.schema import Block, BlockType, DocumentMetadata, DocumentModel, PageMetadata, SourceType
from app.services.ocr_to_translation_parser import OCRToTranslationParser
from app.services.translator_mlx import MlxTranslator, TranslationSettings


def test_ocr_cleaning_repairs_line_break_hyphenation_without_removing_real_hyphens() -> None:
    parser = OCRToTranslationParser()

    cleaned = parser.clean_ocr_text(
        "Die Zu-\nweisung erfolgt im follow-up mit cost-effectiveness und well-being.\n"
        "Geschlechtsidentitätser-\nleben bleibt erhalten."
    )

    assert cleaned == (
        "Die Zuweisung erfolgt im follow-up mit cost-effectiveness und well-being. "
        "Geschlechtsidentitätserleben bleibt erhalten."
    )


def test_ocr_parser_merges_body_across_page_headers_and_excludes_page_elements() -> None:
    document = _document(
        [
            _block("body-1", 1, BlockType.PARAGRAPH, "Kinder stellen sowohl", 1),
            _block("footer", 1, BlockType.FOOTER, "Journal 1", 2),
            _block("header", 2, BlockType.HEADER, "Repeated title 2", 1),
            _block("body-2", 2, BlockType.PARAGRAPH, "aus klinischer als auch ethischer Sicht Fragen.", 2),
        ]
    )

    result = OCRToTranslationParser().prepare(document, document_id="doc-1")

    assert len(result.document.translation_chunks) == 1
    chunk = result.document.translation_chunks[0]
    assert chunk.id == "p0001-p0002-c001"
    assert chunk.page_start == 1
    assert chunk.page_end == 2
    assert chunk.source_region_ids == ["page_0001-r001", "page_0002-r002"]
    assert chunk.source_region_indexes == [1, 2]
    assert chunk.source_text == "Kinder stellen sowohl aus klinischer als auch ethischer Sicht Fragen."
    assert [item["block_id"] for item in result.excluded_regions] == ["footer", "header"]


def test_ocr_parser_reports_original_region_indexes_after_surya_reconciliation() -> None:
    block = _block("merged", 1, BlockType.PARAGRAPH, "Vollständiger Absatz.", 4)
    block.metadata["source_region_ids"] = ["page_0001-r004", "page_0001-r005"]
    block.metadata["surya_region_index"] = 3

    chunk = OCRToTranslationParser().prepare(_document([block])).document.translation_chunks[0]

    assert chunk.source_region_ids == ["page_0001-r004", "page_0001-r005"]
    assert chunk.source_region_indexes == [4, 5]


def test_ocr_parser_groups_keywords_and_preserves_section_path() -> None:
    document = _document(
        [
            _block("section", 1, BlockType.HEADING, "1 Hintergrund", 1),
            _block("keywords-heading", 1, BlockType.HEADING, "Schlagwörter", 2),
            _block("keywords", 1, BlockType.PARAGRAPH, "Transgender – Behandlung", 3),
            _block("body", 1, BlockType.PARAGRAPH, "Ein vollständiger Absatz.", 4),
        ]
    )

    chunks = OCRToTranslationParser().prepare(document).document.translation_chunks

    assert [chunk.chunk_type for chunk in chunks] == ["heading", "keywords", "paragraph"]
    assert chunks[1].source_text == "Schlagwörter\nTransgender – Behandlung"
    assert chunks[1].section_path == ["1 Hintergrund"]
    assert chunks[2].section_path == ["1 Hintergrund"]


def test_ocr_parser_keeps_footnote_continuation_separate_from_body() -> None:
    document = _document(
        [
            _block("footnote-1", 1, BlockType.FOOTNOTE, "2 Der Begriff endet zuletzt", 1, y0=750, y1=800),
            _block("body", 2, BlockType.PARAGRAPH, "normaler Haupttext.", 1, y0=100, y1=200),
            _block("footnote-2", 2, BlockType.PARAGRAPH, "ist fraglich und wird erklärt.", 2, y0=760, y1=900),
        ]
    )

    chunks = OCRToTranslationParser().prepare(document).document.translation_chunks

    assert [chunk.chunk_type for chunk in chunks] == ["paragraph", "footnote"]
    assert chunks[0].source_text == "normaler Haupttext."
    assert chunks[1].page_start == 1
    assert chunks[1].page_end == 2
    assert chunks[1].source_text == "2 Der Begriff endet zuletzt ist fraglich und wird erklärt."


def test_ocr_parser_filters_repeated_margin_text_misclassified_as_body() -> None:
    repeated = "Repeated article title"
    document = _document(
        [
            _block("margin-1", 1, BlockType.PARAGRAPH, repeated, 1, y0=10, y1=40),
            _block("body-1", 1, BlockType.PARAGRAPH, "Erster Absatz.", 2, y0=200, y1=300),
            _block("margin-2", 2, BlockType.PARAGRAPH, repeated, 1, y0=10, y1=40),
            _block("body-2", 2, BlockType.PARAGRAPH, "Zweiter Absatz.", 2, y0=200, y1=300),
        ]
    )

    result = OCRToTranslationParser().prepare(document)

    assert [item["reason"] for item in result.excluded_regions] == [
        "repeated_margin_text",
        "repeated_margin_text",
    ]
    assert [chunk.source_text for chunk in result.document.translation_chunks] == [
        "Erster Absatz.",
        "Zweiter Absatz.",
    ]


def test_translator_uses_prepared_logical_chunks_instead_of_excluded_blocks() -> None:
    document = _document(
        [
            _block("header", 1, BlockType.HEADER, "Repeated title", 1),
            _block("body", 1, BlockType.PARAGRAPH, "Texto principal.", 2),
        ]
    )
    prepared = OCRToTranslationParser().prepare(document).document

    chunks = MlxTranslator(TranslationSettings()).build_chunks(prepared)

    assert [chunk.block_ids for chunk in chunks] == [["body"]]
    assert chunks[0].source_region_ids == ["page_0001-r002"]


def _document(blocks: list[Block]) -> DocumentModel:
    return DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=2),
        pages=[
            PageMetadata(
                page_number=page_number,
                width=600,
                height=1000,
                has_embedded_text=False,
                embedded_text_quality=0.0,
                extraction_mode=SourceType.OCR,
            )
            for page_number in (1, 2)
        ],
        blocks=blocks,
    )


def _block(
    block_id: str,
    page_number: int,
    block_type: BlockType,
    text: str,
    region_index: int,
    *,
    y0: int = 100,
    y1: int = 200,
) -> Block:
    return Block(
        id=block_id,
        page_number=page_number,
        block_type=block_type,
        text=text,
        reading_order_index=region_index,
        source_type=SourceType.OCR,
        metadata={
            "surya_region_index": region_index,
            "surya_region_type": block_type.value,
            "source_region_ids": [f"page_{page_number:04d}-r{region_index:03d}"],
            "surya_bbox": [10, y0, 500, y1],
            "surya_page_height": 1000,
        },
    )
