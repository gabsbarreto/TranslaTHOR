from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.schema import (
    Block,
    BlockType,
    BoundingBox,
    DocumentMetadata,
    DocumentModel,
    PageMetadata,
    SourceType,
    TableModel,
)
from app.services.markdown_builder import MarkdownBuilder
from app.services.pdf_extraction.table_repair import MarkerTableRepairService
from app.services.original_layout_reconstructor import OriginalLayoutReconstructor
from app.services.translator_mlx import MlxTranslator, TranslationSettings


fitz = pytest.importorskip("fitz")


def test_collapsed_marker_table_is_repaired_from_clipped_pdf_grid(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ruled-table.pdf"
    rows = [
        ["Variable", "Group", "P"],
        ["Age", "42", "0.05"],
        ["Status", "Active", ""],
        ["Region", "North", ""],
    ]
    _write_ruled_table_pdf(pdf_path, rows)
    document = _collapsed_document(rows)

    summary = MarkerTableRepairService().repair(pdf_path, document)

    assert summary.suspicious_count == 1
    assert summary.repaired_count == 1
    assert summary.failed_count == 0
    table = document.tables[0]
    block = document.blocks[0]
    assert table.parse_mode == "pymupdf_repaired_marker"
    assert table.headers == rows[0]
    assert table.rows == rows[1:]
    assert block.text.count("<tr>") == 4
    assert block.text.count("<th>") == 3
    assert block.text.count("<td>") == 9
    assert "Variable<br>Group<br>P<br>Age" not in block.text
    assert block.metadata["marker_table_repair"]["token_recall"] == 1.0
    assert block.metadata["marker_table_repair"]["token_precision"] == 1.0

    markdown = MarkdownBuilder().build(document)
    assert markdown.count("<table>") == 1
    assert "<tr><td>Age</td><td>42</td><td>0.05</td></tr>" in markdown

    chunks = MlxTranslator(TranslationSettings()).build_chunks(document)
    assert len(chunks) == 1
    assert chunks[0].block_ids == ["/page/0/Table/1"]
    assert chunks[0].source_text.count("<tr>") == 4


def test_collapsed_marker_table_is_retained_when_pdf_text_does_not_match(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ruled-table.pdf"
    pdf_rows = [
        ["Variable", "Group", "P"],
        ["Age", "42", "0.05"],
        ["Status", "Active", ""],
        ["Region", "North", ""],
    ]
    marker_rows = [
        ["Different", "Headings", "Here"],
        ["Unrelated", "17", "0.9"],
        ["Other", "Values", ""],
        ["Final", "Entry", ""],
    ]
    _write_ruled_table_pdf(pdf_path, pdf_rows)
    document = _collapsed_document(marker_rows)
    original_html = document.blocks[0].text

    summary = MarkerTableRepairService().repair(pdf_path, document)

    assert summary.suspicious_count == 1
    assert summary.repaired_count == 0
    assert summary.failed_count == 1
    assert document.blocks[0].text == original_html
    assert document.tables[0].parse_mode == "marker_html"
    assert "no matching PDF table grid was found" in summary.warnings[0]


def test_repaired_table_is_reconstructed_once_in_authoritative_box(tmp_path: Path) -> None:
    pdf_path = tmp_path / "ruled-table.pdf"
    output_path = tmp_path / "translated.pdf"
    report_path = tmp_path / "report.json"
    rows = [
        ["Variable", "Group", "P"],
        ["Age", "42", "0.05"],
        ["Status", "Active", ""],
        ["Region", "North", ""],
    ]
    _write_ruled_table_pdf(pdf_path, rows)
    document = _collapsed_document(rows)
    summary = MarkerTableRepairService().repair(pdf_path, document)
    assert summary.repaired_count == 1

    block = document.blocks[0]
    source_html = block.text
    block.metadata["source_text"] = source_html
    block.metadata["translated_from_block_ids"] = [block.id]
    block.text = (
        source_html.replace("Variable", "Measure")
        .replace("Status", "State")
        .replace("Active", "Enabled")
        .replace("Region", "Area")
    )

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=pdf_path,
        output_pdf_path=output_path,
        document=document,
        report_path=report_path,
    )

    assert output_path.exists()
    assert report["regions_replaced"] == 1
    assert report["regions_skipped"] == 0
    replaced = next(region for region in report["regions"] if region["status"] == "replaced")
    assert replaced["bbox"] == block.bbox.model_dump()
    assert replaced["source_text_masks"] == [block.bbox.model_dump()]
    with fitz.open(output_path) as translated:
        output_text = translated[0].get_text("text")
        assert "Measure" in output_text
        assert "Enabled" in output_text
        assert "Area" in output_text
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert not any(
        region.get("reason") == "table_translation_structure_unreliable"
        for region in persisted["regions"]
    )


def _write_ruled_table_pdf(pdf_path: Path, rows: list[list[str]]) -> None:
    document = fitz.open()
    page = document.new_page(width=400, height=300)
    x_positions = [40, 190, 270, 350]
    y_positions = [70, 100, 130, 160, 190]
    for row_index, row in enumerate(rows):
        for column_index, text in enumerate(row):
            page.draw_rect(
                fitz.Rect(
                    x_positions[column_index],
                    y_positions[row_index],
                    x_positions[column_index + 1],
                    y_positions[row_index + 1],
                ),
                width=0.5,
            )
            if text:
                page.insert_text(
                    (x_positions[column_index] + 3, y_positions[row_index] + 18),
                    text,
                    fontsize=8,
                )
    document.save(pdf_path)
    document.close()


def _collapsed_document(rows: list[list[str]]) -> DocumentModel:
    flattened = "<br>".join(cell for row in rows for cell in row if cell)
    empty_rows = "".join("<tr><td></td><td></td><td></td></tr>" for _ in rows[1:])
    malformed = (
        f"<table><tbody><tr><th>{flattened}</th><th></th><th></th></tr>{empty_rows}</tbody></table>"
    )
    block = Block(
        id="/page/0/Table/1",
        page_number=1,
        block_type=BlockType.TABLE,
        text=malformed,
        bbox=BoundingBox(x0=40, y0=70, x1=350, y1=190),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "parser": "marker",
            "marker_block_type": "Table",
            "marker_page_width": 400,
            "marker_page_height": 300,
        },
    )
    return DocumentModel(
        metadata=DocumentMetadata(filename="ruled-table.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=400,
                height=300,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[block],
        tables=[
            TableModel(
                id="table-0",
                page_numbers=[1],
                page=1,
                bbox=block.bbox,
                headers=[flattened, "", ""],
                rows=[["", "", ""] for _ in rows[1:]],
                parse_mode="marker_html",
                debug={
                    "marker_block_id": block.id,
                    "render_from_block_text": True,
                },
            )
        ],
    )
