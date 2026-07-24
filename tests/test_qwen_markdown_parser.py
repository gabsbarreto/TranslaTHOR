from __future__ import annotations

import json
from pathlib import Path

import fitz

from app.models.inspection import PageInspection, PdfInspection
from app.models.schema import Block, BlockType, BoundingBox, SourceType, TableModel
from app.services.ocr_to_translation_parser import OCRToTranslationParser
from app.services.pdf_extraction.qwen_ocr_fallback import QwenFullPageOCRFallback
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


def test_qwen_parser_anchors_markdown_image_to_nearby_surya_figure() -> None:
    blocks = QwenMarkdownParser()._blocks_from_markdown(
        """<region index="1" type="Text">
The paragraph spans two detected text columns.
</region>
<region index="2" type="Figure">
![Generated chart description](https://example.invalid/generated.png)
</region>
<region index="3" type="Caption">
Figura 2. Incidencia por año
</region>
<region index="4" type="PageFooter">
Journal 2025; 1:1-5
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
                    "label": "Text",
                    "bbox": [10, 10, 280, 100],
                    "source_region_ids": ["left-text"],
                },
                {
                    "index": 2,
                    "label": "Text",
                    "bbox": [320, 10, 590, 100],
                    "source_region_ids": ["right-text"],
                },
                {
                    "index": 3,
                    "label": "Figure",
                    "bbox": [80, 140, 520, 560],
                    "source_region_ids": ["actual-figure"],
                },
                {
                    "index": 4,
                    "label": "Caption",
                    "bbox": [80, 570, 520, 610],
                    "source_region_ids": ["actual-caption"],
                },
                {
                    "index": 5,
                    "label": "PageFooter",
                    "bbox": [180, 740, 420, 770],
                    "source_region_ids": ["footer"],
                },
            ],
        },
    )

    assert [block.block_type for block in blocks] == [
        BlockType.PARAGRAPH,
        BlockType.FIGURE,
        BlockType.CAPTION,
        BlockType.FOOTER,
    ]
    assert [block.metadata["source_region_ids"] for block in blocks] == [
        ["left-text"],
        ["actual-figure"],
        ["actual-caption"],
        ["footer"],
    ]
    assert blocks[1].metadata["surya_region_mapping"] == "after_omitted_text_to_visual"
    assert blocks[2].text == "Figura 2. Incidencia por año"
    assert blocks[2].bbox == BoundingBox(x0=80, y0=570, x1=520, y1=610)


def test_qwen_parser_does_not_jump_across_structural_region_for_figure() -> None:
    parser = QwenMarkdownParser()
    regions = [
        {"index": 1, "label": "Caption"},
        {"index": 2, "label": "Figure"},
    ]

    assert (
        parser._nearby_visual_region_match(
            regions,
            cursor=0,
            output_type="Figure",
            text="![Chart](https://example.invalid/chart.png)",
        )
        is None
    )


def test_qwen_parser_does_not_jump_long_prose_to_a_later_figure() -> None:
    text = (
        "Los pacientes recibieron tratamiento hormonal y seguimiento clínico durante "
        "todo el periodo del estudio. Los resultados se revisaron en cada visita y "
        "se registraron de forma independiente para el análisis posterior."
    )

    block = QwenMarkdownParser()._blocks_from_markdown(
        f'''<region index="1" type="Figure">
{text}
</region>
''',
        page_number=1,
        start_order=0,
        surya_page={
            "width": 600,
            "height": 800,
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "Text",
                    "bbox": [40, 80, 560, 220],
                    "source_region_ids": ["body-text"],
                },
                {
                    "index": 2,
                    "label": "Figure",
                    "bbox": [80, 260, 520, 650],
                    "source_region_ids": ["later-figure"],
                },
            ],
        },
    )[0]

    assert block.block_type == BlockType.PARAGRAPH
    assert block.bbox is None
    assert block.metadata["source_region_ids"] == []
    assert block.metadata["surya_region_mapping"] == "reading_order"
    guard = block.metadata["structural_type_guard"]
    assert guard["reason"] == "natural_language_prose_conflicts_with_qwen_visual_label"
    assert guard["rejected_source_region_ids"] == ["body-text"]
    assert guard["rejected_bbox"] == [40, 80, 560, 220]


def test_qwen_parser_downgrades_text_mismatched_with_locked_visual_regions() -> None:
    text = (
        "Durante el seguimiento clínico, cada paciente recibió una evaluación completa "
        "y el equipo registró los cambios terapéuticos antes de la siguiente consulta."
    )
    parser = QwenMarkdownParser()

    for layout_label in ("Figure", "Picture", "Table"):
        block = parser._blocks_from_markdown(
            f'''<region index="1" type="Text">
{text}
</region>
''',
            page_number=1,
            start_order=0,
            surya_page={
                "width": 600,
                "height": 800,
                "reconciled_regions": [
                    {
                        "index": 1,
                        "label": layout_label,
                        "bbox": [60, 120, 540, 620],
                        "source_region_ids": [f"wrong-{layout_label.casefold()}"],
                    }
                ],
            },
        )[0]

        assert block.block_type == BlockType.PARAGRAPH
        assert block.bbox is None
        assert block.metadata["source_region_ids"] == []
        guard = block.metadata["structural_type_guard"]
        assert guard["reason"] == "natural_language_prose_conflicts_with_layout_visual_label"
        assert guard["rejected_source_region_ids"] == [
            f"wrong-{layout_label.casefold()}"
        ]
        assert guard["rejected_bbox"] == [60, 120, 540, 620]


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
    assert first_block.text.startswith("<table>")
    assert "<th>First</th>" in first_block.text
    assert first_block.block_type == BlockType.TABLE
    assert first.debug["geometry_reliable"] is False
    assert first.bbox is None
    assert first_block.bbox is None
    assert second_block.text.startswith("<table>")
    assert "<th>Second</th>" in second_block.text
    assert second.debug["geometry_reliable"] is True
    assert second.debug["source_region_ids"] == ["table-r2"]
    assert second.bbox == BoundingBox(x0=100, y0=600, x1=1100, y1=900)


def test_qwen_parser_repairs_ragged_markdown_table_with_empty_cells_only(
    tmp_path: Path,
) -> None:
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "page_0001.md").write_text(
        """<region index="1" type="Table">
| Period | Group | Ages | | | | | Total | |
|---|---|---|---|---|---|---|---|---|
| | | Under 18 | % | 18-29 | % | 30-45 | % | 46+ | % |
| 2007 | A | 1 | | 2 | | 3 | | 4 | | 10 |
| | Total | 1 | 10 | 2 | 20 | 3 | 30 | 4 | 40 | 10 | 100 |
| 2008 | A | 2 | | 3 | | 4 | | 5 | | 14 |
| | Total | 2 | 14 | 3 | 21 | 4 | 29 | 5 | 36 | 14 | 100 |
</region>
""",
        encoding="utf-8",
    )
    (markdown_dir / "page_0002.md").write_text("", encoding="utf-8")
    manifest = _single_region_manifest("Table")

    document, _ = QwenMarkdownParser().build_document_from_markdown_dir(
        inspection=_inspection(),
        markdown_dir=markdown_dir,
        strict_page_files=True,
        surya_layout_manifest=manifest,
    )

    assert len(document.tables) == 1
    table = document.tables[0]
    assert table.parse_mode == "qwen_markdown_table_repaired"
    assert len(table.headers) == 12
    assert table.headers[10] == "Total"
    assert all(len(row) == 12 for row in table.rows)
    assert table.debug["topology_repair"]["strategy"] == "empty_cell_padding_only"
    assert table.debug["topology_repair"]["placement_assumed"] is True
    assert table.debug["geometry_reliable"] is False
    assert table.debug["cell_geometry_reliable"] is False
    assert table.debug["reconstruction_scope"] == "readable_reflow_only"
    assert table.bbox is None
    block = next(block for block in document.blocks if block.block_type == BlockType.TABLE)
    assert block.text.startswith("<table>")
    assert block.text.count("<tr>") == 6
    assert block.metadata["qwen_table_topology_repair"]["canonical_width"] == 12
    assert block.metadata["table_geometry_reliable"] is False
    assert block.metadata["table_cell_geometry_reliable"] is False
    assert block.metadata["table_reconstruction_scope"] == "readable_reflow_only"
    assert block.bbox is None
    prepared = OCRToTranslationParser().prepare(document, document_id="ragged-table").document
    table_chunk = next(chunk for chunk in prepared.translation_chunks if chunk.chunk_type == "table")
    assert table_chunk.source_text.startswith("<table>")
    assert table_chunk.source_text.count("<tr>") == 6
    assert table_chunk.source_text.count("<th>") == 12
    assert table_chunk.source_text.count("<td>") == 60


def test_qwen_parser_does_not_claim_geometry_for_ambiguous_internal_missing_cell(
    tmp_path: Path,
) -> None:
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "page_0001.md").write_text(
        """<region index="1" type="Table">
| A | B | C | D |
|---|---|---|---|
| one | two | four |
| five | six | seven | eight |
| nine | ten | eleven | twelve |
</region>
""",
        encoding="utf-8",
    )
    (markdown_dir / "page_0002.md").write_text("", encoding="utf-8")

    document, _ = QwenMarkdownParser().build_document_from_markdown_dir(
        inspection=_inspection(),
        markdown_dir=markdown_dir,
        strict_page_files=True,
        surya_layout_manifest=_single_region_manifest("Table"),
    )

    assert len(document.tables) == 1
    table = document.tables[0]
    assert len(table.headers) == 4
    assert all(len(row) == 4 for row in table.rows)
    repair = table.debug["topology_repair"]
    assert repair["placement_assumed"] is True
    assert repair["operations"] == [
        {
            "row_index": 1,
            "original_width": 3,
            "inserted_empty_cells": 1,
            "insert_at": 3,
            "placement_assumed": True,
            "placement_basis": "trailing_padding_default",
        }
    ]
    assert table.rows[0] == ["one", "two", "four", ""]
    assert table.debug["geometry_reliable"] is False
    assert table.debug["cell_geometry_reliable"] is False
    assert table.bbox is None
    block = next(block for block in document.blocks if block.block_type == BlockType.TABLE)
    assert block.bbox is None
    assert block.metadata["table_geometry_reliable"] is False
    assert block.metadata["table_reconstruction_scope"] == "readable_reflow_only"


def test_qwen_parser_rejects_ambiguous_ragged_table_topology(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "page_0001.md").write_text(
        """<region index="1" type="Table">
| A | B |
|---|---|
| one | two | three | four | five |
| six | seven | eight |
</region>
""",
        encoding="utf-8",
    )
    (markdown_dir / "page_0002.md").write_text("", encoding="utf-8")

    document, _ = QwenMarkdownParser().build_document_from_markdown_dir(
        inspection=_inspection(),
        markdown_dir=markdown_dir,
        strict_page_files=True,
        surya_layout_manifest=_single_region_manifest("Table"),
    )

    assert document.tables == []
    block = document.blocks[0]
    assert block.block_type == BlockType.TABLE
    assert block.text.startswith("| A | B |")
    assert not block.metadata.get("qwen_table_markup_normalized")


def test_qwen_parser_infers_concise_two_column_table_from_independent_geometry(
    tmp_path: Path,
) -> None:
    document = _parse_geometry_fixture(
        tmp_path,
        lines=[
            ("Absolute", "Relative"),
            ("Stroke", "Migraine"),
            ("Diabetes", "Dyslipidaemia"),
            ("Renal", "Smoking"),
        ],
        table_score=0.31,
    )

    assert len(document.tables) == 1
    table = document.tables[0]
    assert table.parse_mode == "hidden_ocr_geometry_inferred"
    assert table.headers == ["Absolute", "Relative"]
    assert table.rows[-1] == ["Renal", "Smoking"]
    assert table.header_cells[0].bbox is not None
    assert all(cell.bbox is not None for row in table.cells for cell in row)
    block = next(block for block in document.blocks if block.block_type == BlockType.TABLE)
    assert block.metadata["inferred_table_evidence"]["gutter_supporting_rows"] == 4


def test_qwen_parser_requires_table_model_evidence_for_concise_columns(tmp_path: Path) -> None:
    document = _parse_geometry_fixture(
        tmp_path,
        lines=[
            ("Absolute", "Relative"),
            ("Stroke", "Migraine"),
            ("Diabetes", "Dyslipidaemia"),
            ("Renal", "Smoking"),
        ],
        table_score=0.19,
    )

    assert document.tables == []
    assert all(block.block_type != BlockType.TABLE for block in document.blocks)


def test_qwen_parser_aligns_one_output_to_contiguous_surya_fragments() -> None:
    parser = QwenMarkdownParser()
    blocks = parser._blocks_from_markdown(
        """<region index="1" type="text">
First fragment continued fragment
</region>
""",
        page_number=1,
        start_order=0,
        surya_page={
            "width": 1200,
            "height": 1600,
            "regions": [
                {"id": "r1", "label": "Text", "top_k": {"Text": 0.99}},
                {"id": "r2", "label": "Text", "top_k": {"Text": 0.99}},
            ],
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "Text",
                    "bbox": [100, 200, 500, 260],
                    "source_region_ids": ["r1"],
                },
                {
                    "index": 2,
                    "label": "Text",
                    "bbox": [100, 270, 500, 330],
                    "source_region_ids": ["r2"],
                },
            ],
            "embedded_text_geometry": {
                "available": True,
                "words": [
                    {"text": "First", "bbox": [110, 210, 170, 240]},
                    {"text": "fragment", "bbox": [180, 210, 270, 240]},
                    {"text": "continued", "bbox": [110, 280, 210, 310]},
                    {"text": "fragment", "bbox": [220, 280, 310, 310]},
                ],
            },
        },
    )

    assert len(blocks) == 1
    assert blocks[0].metadata["source_region_ids"] == ["r1", "r2"]
    assert blocks[0].metadata["surya_region_mapping"] == "embedded_text_geometry"
    assert blocks[0].metadata["embedded_text_alignment"]["matched_region_indexes"] == [1, 2]
    assert blocks[0].bbox == BoundingBox(x0=100, y0=200, x1=500, y1=330)


def test_qwen_parser_globally_aligns_body_text_when_column_orders_disagree() -> None:
    left_text = (
        "During treatment, patients are reviewed every three months and their "
        "clinical progress is recorded for the complete follow-up period."
    )
    right_text = (
        "The comparison group was evaluated first because the visual reading "
        "order begins in the right-hand column on this page."
    )
    blocks = QwenMarkdownParser()._blocks_from_markdown(
        f'''<region index="1" type="Text">
{right_text}
</region>
<region index="2" type="Text">
{left_text}
</region>
''',
        page_number=1,
        start_order=0,
        surya_page={
            "width": 1200,
            "height": 1600,
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "Text",
                    "bbox": [100, 300, 550, 520],
                    "source_region_ids": ["left-body"],
                },
                {
                    "index": 2,
                    "label": "PageFooter",
                    "bbox": [40, 1500, 80, 1570],
                    "source_region_ids": ["page-number"],
                },
                {
                    "index": 3,
                    "label": "Text",
                    "bbox": [650, 200, 1100, 430],
                    "source_region_ids": ["right-body"],
                },
            ],
            "embedded_text_geometry": {
                "available": True,
                "words": [
                    {"text": right_text, "bbox": [670, 220, 1080, 410]},
                    {"text": left_text, "bbox": [120, 320, 530, 500]},
                    {"text": "27", "bbox": [50, 1510, 70, 1560]},
                ],
            },
        },
    )

    assert blocks[0].metadata["source_region_ids"] == ["right-body"]
    assert blocks[1].metadata["source_region_ids"] == ["left-body"]
    assert blocks[1].metadata["surya_region_mapping"] == "embedded_text_geometry_global"
    assert blocks[1].metadata["embedded_text_alignment"]["search_scope"] == "global"
    assert blocks[1].bbox == BoundingBox(x0=100, y0=300, x1=550, y1=520)
    assert blocks[1].block_type == BlockType.PARAGRAPH


def test_qwen_parser_downgrades_long_footer_prose_but_keeps_short_footer() -> None:
    parser = QwenMarkdownParser()
    long_text = (
        "During medical treatment, patients are reviewed every three months. "
        "Their clinical progress, laboratory results, and satisfaction are assessed "
        "throughout follow-up so that treatment can be adjusted safely."
    )
    long_block = parser._blocks_from_markdown(
        f'''<region index="1" type="PageFooter">
{long_text}
</region>
''',
        page_number=1,
        start_order=0,
        surya_page={
            "width": 600,
            "height": 800,
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "PageFooter",
                    "bbox": [40, 650, 560, 770],
                    "source_region_ids": ["mislabelled-body"],
                }
            ],
        },
    )[0]
    short_block = parser._blocks_from_markdown(
        '''<region index="1" type="PageFooter">
Journal Name - Vol. 4 - 2001
</region>
''',
        page_number=1,
        start_order=0,
        surya_page={
            "width": 600,
            "height": 800,
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "PageFooter",
                    "bbox": [80, 760, 520, 790],
                    "source_region_ids": ["running-footer"],
                }
            ],
        },
    )[0]

    assert long_block.block_type == BlockType.PARAGRAPH
    assert long_block.metadata["structural_type_guard"]["reason"] == (
        "long_prose_is_not_running_matter"
    )
    assert long_block.bbox is None
    assert long_block.metadata["source_region_ids"] == []
    assert long_block.metadata["structural_type_guard"]["rejected_bbox"] == [
        40,
        650,
        560,
        770,
    ]
    assert long_block.metadata["structural_type_guard"][
        "rejected_source_region_ids"
    ] == ["mislabelled-body"]
    assert short_block.block_type == BlockType.FOOTER
    assert short_block.metadata["structural_type_guard"]["status"] == "accepted"


def test_qwen_parser_preserves_semantic_footnote_over_footer_position_label() -> None:
    block = QwenMarkdownParser()._blocks_from_markdown(
        '''<region index="1" type="Footnote">
Documents used in the study may be requested from the authors.
</region>
''',
        page_number=1,
        start_order=0,
        surya_page={
            "width": 600,
            "height": 800,
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "PageFooter",
                    "bbox": [50, 720, 550, 775],
                    "source_region_ids": ["source-note"],
                }
            ],
        },
    )[0]

    assert block.block_type == BlockType.FOOTNOTE
    assert block.metadata["structural_type_guard"]["reason"] == (
        "semantic_qwen_label_overrides_marginal_layout_label"
    )


def test_qwen_parser_does_not_promote_long_prose_to_heading() -> None:
    text = (
        "Of the participants who had not previously received treatment, most "
        "continued clinical evaluation while the remaining group began supervised "
        "therapy at another centre after the baseline assessment was complete."
    )
    block = QwenMarkdownParser()._blocks_from_markdown(
        f'''<region index="1" type="Text">
{text}
</region>
''',
        page_number=1,
        start_order=0,
        surya_page={
            "width": 600,
            "height": 800,
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "SectionHeader",
                    "bbox": [240, 680, 380, 705],
                    "source_region_ids": ["wrong-heading"],
                }
            ],
        },
    )[0]

    assert block.block_type == BlockType.PARAGRAPH
    assert block.metadata["structural_type_guard"]["reason"] == (
        "long_prose_is_not_a_heading"
    )
    assert block.bbox is None
    assert block.metadata["source_region_ids"] == []


def test_qwen_parser_can_split_one_surya_text_region_by_hidden_text_content() -> None:
    first_text = (
        "The first paragraph occupies the upper part of a single merged layout "
        "region and contains complete clinical observations."
    )
    second_text = (
        "The second paragraph occupies the lower part of that same merged layout "
        "region and reports the follow-up outcome."
    )
    blocks = QwenMarkdownParser()._blocks_from_markdown(
        f'''<region index="1" type="Text">
{first_text}
</region>
<region index="2" type="Text">
{second_text}
</region>
''',
        page_number=1,
        start_order=0,
        surya_page={
            "width": 600,
            "height": 800,
            "reconciled_regions": [
                {
                    "index": 1,
                    "label": "Text",
                    "bbox": [50, 150, 550, 500],
                    "source_region_ids": ["upper-source", "lower-source"],
                }
            ],
            "embedded_text_geometry": {
                "available": True,
                "words": [
                    {"text": first_text, "bbox": [70, 180, 530, 300]},
                    {"text": second_text, "bbox": [70, 340, 530, 470]},
                ],
            },
        },
    )

    assert len(blocks) == 2
    assert [block.metadata["source_region_ids"] for block in blocks] == [
        ["upper-source"],
        ["lower-source"],
    ]
    assert all(
        block.metadata["embedded_text_alignment"]["partial_containment"]
        for block in blocks
    )


def test_qwen_parser_comparison_text_preserves_non_latin_alphanumerics() -> None:
    parser = QwenMarkdownParser()

    assert parser._comparison_text("École déjà") == "ecoledeja"
    assert parser._comparison_text("Привет мир") == "приветмир"
    assert parser._comparison_text("العَرَبِيَّة") == "العربية"
    assert parser._comparison_text("研究 方法 2025") == "研究方法2025"


def test_qwen_parser_rejects_nonfirst_best_with_ambiguous_runner_up() -> None:
    parser = QwenMarkdownParser()
    expected = "a" * 50
    near_match = f"{'a' * 49}b"
    regions = [
        {
            "index": 1,
            "label": "Text",
            "bbox": [0, 0, 100, 20],
            "source_region_ids": ["r1"],
        },
        {
            "index": 2,
            "label": "Text",
            "bbox": [0, 30, 100, 50],
            "source_region_ids": ["r2"],
        },
    ]
    surya_page = {
        "embedded_text_geometry": {
            "available": True,
            "words": [
                {"text": near_match, "bbox": [1, 1, 99, 19]},
                {"text": expected, "bbox": [1, 31, 99, 49]},
            ],
        }
    }

    assert (
        parser._embedded_text_region_match(
            expected,
            regions,
            cursor=0,
            surya_page=surya_page,
        )
        is None
    )


def test_qwen_parser_prefers_exact_short_heading_over_word_inside_prose() -> None:
    parser = QwenMarkdownParser()
    regions = [
        {
            "index": 1,
            "label": "Text",
            "bbox": [0, 0, 500, 100],
            "source_region_ids": ["body"],
        },
        {
            "index": 2,
            "label": "SectionHeader",
            "bbox": [0, 120, 200, 150],
            "source_region_ids": ["heading"],
        },
    ]
    result = parser._embedded_text_region_match(
        "Results",
        regions,
        cursor=0,
        search_all=True,
        minimum_expected_length=6,
        surya_page={
            "embedded_text_geometry": {
                "available": True,
                "words": [
                    {
                        "text": "The results show a clinically meaningful difference in follow-up.",
                        "bbox": [10, 10, 480, 80],
                    },
                    {"text": "Results", "bbox": [10, 125, 100, 145]},
                ],
            }
        },
    )

    assert result is not None
    assert result[0:2] == (1, 1)
    assert result[2]["score"] == 1.0
    assert result[2]["partial_containment"] is False


def test_qwen_parser_does_not_promote_boxed_two_column_prose(tmp_path: Path) -> None:
    document = _parse_geometry_fixture(
        tmp_path,
        lines=[
            (
                "This is a complete sentence with many ordinary prose words",
                "This is another complete sentence with many ordinary prose words",
            ),
            (
                "The paragraph continues across a normal line in its column",
                "The other paragraph also continues normally in its column",
            ),
            (
                "Nothing in these aligned lines represents concise table cells",
                "The visual box alone must not turn prose into a table",
            ),
            (
                "Both columns contain flowing language and sentence structure",
                "Both columns should remain ordinary paragraph content here",
            ),
        ],
        table_score=0.31,
    )

    assert document.tables == []
    assert all(block.block_type != BlockType.TABLE for block in document.blocks)


def test_qwen_fallback_records_hidden_ocr_geometry_for_alignment(tmp_path: Path) -> None:
    pdf_path = tmp_path / "hidden.pdf"
    source = fitz.open()
    page = source.new_page(width=600, height=800)
    page.insert_text((50, 80), "Visible hidden OCR geometry")
    source.save(pdf_path)
    source.close()
    manifest_path = tmp_path / "layout.json"
    manifest = {
        "pages": [{"page_index": 1, "width": 1200, "height": 1600}],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    inspection = PdfInspection(
        filename="hidden.pdf",
        title=None,
        author=None,
        page_count=1,
        pages=[
            PageInspection(
                page_number=1,
                width=600,
                height=800,
                text_length=27,
                embedded_text_quality=0.9,
                has_embedded_text=True,
            )
        ],
    )
    QwenFullPageOCRFallback()._attach_embedded_text_geometry(
        pdf_path=pdf_path,
        inspection=inspection,
        manifest=manifest,
        manifest_path=manifest_path,
    )

    geometry = manifest["pages"][0]["embedded_text_geometry"]
    assert geometry["available"] is True
    assert geometry["usage"] == "alignment_only"
    assert geometry["coordinate_space"]["scale_x"] == 2.0
    assert [word["text"] for word in geometry["words"]] == [
        "Visible",
        "hidden",
        "OCR",
        "geometry",
    ]
    assert json.loads(manifest_path.read_text())["embedded_text_geometry_page_count"] == 1


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


def test_qwen_parser_rejects_missing_zero_and_outside_table_geometry() -> None:
    parser = QwenMarkdownParser()
    source_markup = "| A | B |\n|---|---|\n| uno | dos |"
    cases = (
        ("missing", None, "missing_bbox"),
        ("zero", BoundingBox(x0=0, y0=0, x1=0, y1=0), "non_positive_bbox_area"),
        (
            "outside",
            BoundingBox(x0=100, y0=200, x1=1250, y1=900),
            "bbox_outside_page",
        ),
    )

    for case_name, bbox, expected_reason in cases:
        table = TableModel(
            id=f"table-{case_name}",
            page_numbers=[1],
            headers=["A", "B"],
            rows=[["uno", "dos"]],
        )
        block = parser._block(
            page_number=1,
            order=0,
            block_type=BlockType.TABLE,
            text=source_markup,
            bbox=bbox,
            metadata={
                "surya_region_type": "Table",
                "qwen_region_type": "Table",
                "surya_page_width": 1200,
                "surya_page_height": 1600,
                "source_region_ids": [f"region-{case_name}"],
            },
        )

        parser._link_page_tables([table], [block])

        assert block.bbox is None
        assert table.bbox is None
        assert block.metadata["table_geometry_reliable"] is False
        assert block.metadata["table_cell_geometry_reliable"] is False
        assert block.metadata["table_reconstruction_scope"] == "readable_reflow_only"
        assert block.metadata["table_geometry_validation"]["reason"] == expected_reason
        assert table.debug["geometry_reliable"] is False
        assert table.debug["reconstruction_scope"] == "readable_reflow_only"
        assert table.debug["geometry_validation"]["reason"] == expected_reason


def _single_region_manifest(label: str) -> dict:
    return {
        "pages": [
            {
                "page_index": 1,
                "width": 1200,
                "height": 1600,
                "regions": [
                    {
                        "id": "r1",
                        "label": label,
                        "top_k": {label: 0.9, "Table": 0.31},
                    }
                ],
                "reconciled_regions": [
                    {
                        "index": 1,
                        "label": label,
                        "bbox": [100, 200, 1100, 900],
                        "source_region_ids": ["r1"],
                    }
                ],
            }
        ]
    }


def _parse_geometry_fixture(
    tmp_path: Path,
    *,
    lines: list[tuple[str, str]],
    table_score: float,
):
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    body = "\n".join(f"{left} {right}" for left, right in lines)
    (markdown_dir / "page_0001.md").write_text(
        f'<region index="1" type="text">\n{body}\n</region>\n',
        encoding="utf-8",
    )
    (markdown_dir / "page_0002.md").write_text("", encoding="utf-8")
    words: list[dict] = []
    for row_index, (left, right) in enumerate(lines):
        y0 = 220 + row_index * 100
        for column_x, text in ((120, left), (700, right)):
            cursor = column_x
            for word_index, word in enumerate(text.split()):
                width = max(30, len(word) * 9)
                words.append(
                    {
                        "text": word,
                        "bbox": [cursor, y0, cursor + width, y0 + 40],
                        "block": 0 if column_x < 500 else 1,
                        "line": row_index,
                        "word": word_index,
                    }
                )
                cursor += width + 12
    manifest = {
        "pages": [
            {
                "page_index": 1,
                "width": 1200,
                "height": 1600,
                "regions": [
                    {
                        "id": "r1",
                        "label": "Text",
                        "top_k": {"Text": 0.45, "Table": table_score},
                    }
                ],
                "reconciled_regions": [
                    {
                        "index": 1,
                        "label": "Text",
                        "bbox": [100, 200, 1100, 650],
                        "source_region_ids": ["r1"],
                    }
                ],
                "embedded_text_geometry": {
                    "available": True,
                    "usage": "alignment_only",
                    "words": words,
                },
            }
        ]
    }
    document, _ = QwenMarkdownParser().build_document_from_markdown_dir(
        inspection=_inspection(),
        markdown_dir=markdown_dir,
        strict_page_files=True,
        surya_layout_manifest=manifest,
    )
    return document


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
