from __future__ import annotations

import sys
from types import ModuleType

from app.models.schema import Block, BlockType, DocumentMetadata, DocumentModel, PageMetadata, SourceType
from app.services.markdown_builder import MarkdownBuilder

if "langdetect" not in sys.modules:
    langdetect_stub = ModuleType("langdetect")
    langdetect_stub.detect = lambda _text: "en"  # type: ignore[attr-defined]
    sys.modules["langdetect"] = langdetect_stub

from app.services.translator_mlx import MlxTranslator, TranslationSettings


def test_numbered_reference_items_render_as_ordered_list_items() -> None:
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=1,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[
            _block("h", BlockType.HEADING, "REFERENCIAS"),
            _block("r1", BlockType.LIST, "1. First reference"),
            _block("r2", BlockType.LIST, "REFERENCES 11. Continued reference"),
        ],
    )

    markdown = MarkdownBuilder().build(document)

    assert "\n1. First reference\n" in markdown
    assert "\n11. Continued reference\n" in markdown
    assert "- 1. First reference" not in markdown
    assert "REFERENCES 11." not in markdown


def test_marker_html_tables_are_not_translated_again_as_table_model_rows() -> None:
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=1,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[
            _block(
                "table-block",
                BlockType.TABLE,
                "<table><tr><td>Diagnóstico</td><td>Prevalencia</td></tr></table>",
            )
        ],
        tables=[
            {
                "id": "table-1",
                "page_numbers": [1],
                "rows": [["Diagnóstico", "Prevalencia"]],
                "debug": {"marker_block_id": "table-block", "render_from_block_text": True},
            }
        ],
    )

    chunks = MlxTranslator(TranslationSettings()).build_chunks(document)

    assert len(chunks) == 1
    assert chunks[0].block_ids == ["table-block"]
    assert chunks[0].source_text.startswith("<table")


def test_table_markup_validation_rejects_dropped_nonempty_cells() -> None:
    translator = MlxTranslator(TranslationSettings())
    source = "<table><tbody><tr><th>Tablo 4. Regression</th></tr><tr><td>modeli</td></tr></tbody></table>"
    translated = "<table><tbody><tr><th>Table 4. Regression</th></tr><tr><td></td></tr></tbody></table>"

    assert translator._is_valid_table_markup_translation(source, translated) is False


def _block(block_id: str, block_type: BlockType, text: str) -> Block:
    return Block(
        id=block_id,
        page_number=1,
        block_type=block_type,
        text=text,
        bbox=None,
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
    )
