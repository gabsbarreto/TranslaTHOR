from __future__ import annotations

from pathlib import Path

from app.models.inspection import PageInspection, PdfInspection
from app.models.schema import Block, BlockType, BoundingBox, SourceType, TableModel
from app.services.qwen_markdown_parser import QwenMarkdownParser


def test_qwen_parser_preserves_raw_page_markdown(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    page_one = "Article Header\n\nBody text.\n\nPage 1\n"
    page_two = "Article Header\n\n| A | B |\n| - | - |\n| 1 | 2 |\n\nPage 2\n"
    (markdown_dir / "page_0001.md").write_text(page_one, encoding="utf-8")
    (markdown_dir / "page_0002.md").write_text(page_two, encoding="utf-8")

    document, source_markdown = QwenMarkdownParser().build_document_from_markdown_dir(
        inspection=_inspection(),
        markdown_dir=markdown_dir,
        strict_page_files=True,
    )

    assert source_markdown == f"{page_one}\n\n{page_two}"
    assert [block.text for block in document.blocks if block.block_type == BlockType.PARAGRAPH] == [
        "Article Header",
        "Body text.",
        "Page 1",
        "Article Header",
        "Page 2",
    ]
    assert [block.text for block in document.blocks if block.block_type == BlockType.TABLE] == ["[TABLE]"]
    assert document.metadata.translation["ocr_markdown_preserved"] is True


def test_qwen_parser_normalizes_surya_regions_without_leaking_wrappers(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    page_one = """<region index="1" type="PageHeader">
Journal title 10
</region>
<region index="2" type="SectionHeader">
Results
</region>
<region index="3" type="Text">
The paragraph continues.
</region>
<region index="4" type="Footnote">
1 Source note.
</region>
<region index="5" type="PageFooter">
10
</region>
"""
    (markdown_dir / "page_0001.md").write_text(page_one, encoding="utf-8")
    (markdown_dir / "page_0002.md").write_text("", encoding="utf-8")

    document, source_markdown = QwenMarkdownParser().build_document_from_markdown_dir(
        inspection=_inspection(),
        markdown_dir=markdown_dir,
        strict_page_files=True,
    )

    assert [block.block_type for block in document.blocks] == [
        BlockType.HEADER,
        BlockType.HEADING,
        BlockType.PARAGRAPH,
        BlockType.FOOTNOTE,
        BlockType.FOOTER,
    ]
    assert [block.text for block in document.blocks] == [
        "Journal title 10",
        "Results",
        "The paragraph continues.",
        "1 Source note.",
        "10",
    ]
    assert document.blocks[0].metadata["parser"] == "qwen_surya_full_page_ocr"
    assert document.blocks[0].metadata["surya_region_index"] == 1
    assert document.blocks[0].metadata["surya_region_type"] == "PageHeader"
    assert "<region" not in source_markdown
    assert "Journal title 10" in source_markdown
    assert any("Surya layout region tags" in warning for warning in document.warnings)


def test_qwen_parser_removes_surya_wrappers_when_output_contains_extra_text() -> None:
    blocks = QwenMarkdownParser()._blocks_from_markdown(
        """Preface added by model.
<region index="1" type="Text">
Visible content.
</region>
""",
        page_number=1,
        start_order=0,
    )

    assert [block.text for block in blocks] == ["Preface added by model.", "Visible content."]
    assert all("<region" not in block.text for block in blocks)


def test_qwen_parser_accepts_shortened_surya_list_label() -> None:
    blocks = QwenMarkdownParser()._blocks_from_markdown(
        """<region index="1" type="list">
First item.
</region>
""",
        page_number=1,
        start_order=0,
    )

    assert [block.block_type for block in blocks] == [BlockType.LIST]


def test_qwen_parser_aligns_after_qwen_omits_a_surya_region() -> None:
    blocks = QwenMarkdownParser()._blocks_from_markdown(
        """<region index="1" type="SectionHeader">
Summary
</region>
<region index="2" type="SectionHeader">
Keywords
</region>
<region index="3" type="Text">
gender variance – treatment
</region>
""",
        page_number=1,
        start_order=0,
        surya_page={
            "width": 600,
            "height": 800,
            "reconciled_regions": [
                {"index": 1, "label": "SectionHeader", "bbox": [10, 10, 300, 40], "source_region_ids": ["r1"]},
                {"index": 2, "label": "Text", "bbox": [10, 50, 500, 100], "source_region_ids": ["r2"]},
                {"index": 3, "label": "SectionHeader", "bbox": [10, 110, 300, 140], "source_region_ids": ["r3"]},
                {"index": 4, "label": "Text", "bbox": [10, 150, 500, 200], "source_region_ids": ["r4"]},
            ],
        },
    )

    assert [block.metadata["source_region_ids"] for block in blocks] == [["r1"], ["r3"], ["r4"]]
    assert blocks[1].metadata["surya_region_mapping"] == "after_omitted_region"


def test_qwen_parser_partitions_raw_ids_when_qwen_splits_reconciled_footnote() -> None:
    blocks = QwenMarkdownParser()._blocks_from_markdown(
        """<region index="1" type="Footnote">
1 First note.
</region>
<region index="2" type="Footnote">
2 Second note.
</region>
""",
        page_number=1,
        start_order=0,
        surya_page={
            "width": 600,
            "height": 800,
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "Footnote",
                    "bbox": [10, 600, 500, 760],
                    "source_region_ids": ["r1", "r2"],
                },
            ],
        },
    )

    assert [block.metadata["source_region_ids"] for block in blocks] == [["r1"], ["r2"]]


def test_qwen_parser_links_table_block_geometry_and_following_caption(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "page_0001.md").write_text(
        """<region index="1" type="Table">
| A | B |
|---|---|
| uno | dos |
</region>
<region index="2" type="Caption">
TABLA I. Ejemplo
</region>
""",
        encoding="utf-8",
    )
    (markdown_dir / "page_0002.md").write_text("", encoding="utf-8")
    manifest = {
        "pages": [
            {
                "page_index": 1,
                "width": 1200,
                "height": 1600,
                "reconciled_regions": [
                    {
                        "index": 1,
                        "label": "Table",
                        "bbox": [100, 200, 1100, 700],
                        "source_region_ids": ["table-r1"],
                    },
                    {
                        "index": 2,
                        "label": "Caption",
                        "bbox": [100, 710, 900, 760],
                        "source_region_ids": ["caption-r2"],
                    },
                ],
            }
        ]
    }

    document, _source_markdown = QwenMarkdownParser().build_document_from_markdown_dir(
        inspection=_inspection(),
        markdown_dir=markdown_dir,
        strict_page_files=True,
        surya_layout_manifest=manifest,
    )

    table_block = next(block for block in document.blocks if block.block_type == BlockType.TABLE)
    caption_block = next(block for block in document.blocks if block.block_type == BlockType.CAPTION)
    table = document.tables[0]
    assert table.debug["source_block_id"] == table_block.id
    assert table.debug["source_region_ids"] == ["table-r1"]
    assert table.bbox == table_block.bbox
    assert table.caption_block_id == caption_block.id
    assert table.caption == "TABLA I. Ejemplo"


def test_qwen_parser_matches_tables_by_content_when_surya_misclassifies_one(
    tmp_path: Path,
) -> None:
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "page_0001.md").write_text(
        """<region index="1" type="text">
| First | Value |
|---|---|
| uno | dos |
</region>
<region index="2" type="Table">
| Second | Value |
|---|---|
| tres | cuatro |
</region>
""",
        encoding="utf-8",
    )
    (markdown_dir / "page_0002.md").write_text("", encoding="utf-8")
    manifest = {
        "pages": [
            {
                "page_index": 1,
                "width": 1200,
                "height": 1600,
                "reconciled_regions": [
                    {
                        "index": 1,
                        "label": "Text",
                        "bbox": [100, 200, 1100, 500],
                        "source_region_ids": ["misclassified-r1"],
                    },
                    {
                        "index": 2,
                        "label": "Table",
                        "bbox": [100, 600, 1100, 900],
                        "source_region_ids": ["table-r2"],
                    },
                ],
            }
        ]
    }

    document, _source_markdown = QwenMarkdownParser().build_document_from_markdown_dir(
        inspection=_inspection(),
        markdown_dir=markdown_dir,
        strict_page_files=True,
        surya_layout_manifest=manifest,
    )

    first, second = document.tables
    first_block = next(block for block in document.blocks if block.id == first.debug["source_block_id"])
    second_block = next(block for block in document.blocks if block.id == second.debug["source_block_id"])
    assert first_block.text.startswith("| First")
    assert first_block.block_type == BlockType.TABLE
    assert first.debug["geometry_reliable"] is False
    assert first.bbox is None
    assert first_block.bbox is None
    assert second_block.text.startswith("| Second")
    assert second.debug["geometry_reliable"] is True
    assert second.debug["source_region_ids"] == ["table-r2"]
    assert second.bbox == BoundingBox(x0=100, y0=600, x1=1100, y1=900)


def test_qwen_parser_does_not_attach_caption_from_other_column() -> None:
    table = TableModel(
        id="table",
        page_numbers=[1],
        headers=["A", "B"],
        rows=[["uno", "dos"]],
    )
    table_block = Block(
        id="table-block",
        page_number=1,
        block_type=BlockType.TABLE,
        text="| A | B |\n|---|---|\n| uno | dos |",
        bbox=BoundingBox(x0=100, y0=200, x1=550, y1=700),
        reading_order_index=0,
        source_type=SourceType.OCR,
        metadata={
            "surya_region_type": "Table",
            "qwen_region_type": "Table",
            "surya_page_width": 1200,
            "surya_page_height": 1600,
        },
    )
    other_column_caption = Block(
        id="other-caption",
        page_number=1,
        block_type=BlockType.CAPTION,
        text="Figure 2. Unrelated caption",
        bbox=BoundingBox(x0=650, y0=710, x1=1100, y1=760),
        reading_order_index=1,
        source_type=SourceType.OCR,
        metadata={"qwen_region_type": "Caption"},
    )

    QwenMarkdownParser()._link_page_tables(
        [table],
        [table_block, other_column_caption],
    )

    assert table.caption_block_id is None
    assert table.caption is None


def _inspection() -> PdfInspection:
    return PdfInspection(
        filename="paper.pdf",
        title=None,
        author=None,
        page_count=2,
        pages=[
            PageInspection(
                page_number=page_number,
                width=600,
                height=800,
                text_length=0,
                embedded_text_quality=0.0,
                has_embedded_text=False,
            )
            for page_number in (1, 2)
        ],
    )
