from app.models.schema import (
    Block,
    BlockType,
    DocumentMetadata,
    DocumentModel,
    PageMetadata,
    SourceType,
    TableModel,
)
from app.services.markdown_builder import MarkdownBuilder
from app.services.table_markup import parse_table_rows


def test_markdown_builder_renders_structured_table_html() -> None:
    table = TableModel(
        id="t1",
        page_numbers=[1],
        headers=["A", "B"],
        cells=[
            [TableModel.TableCell(text="x", rowspan=1, colspan=1), TableModel.TableCell(text="y", rowspan=1, colspan=1)]
        ],
        rows=[["x", "y"]],
    )
    doc = DocumentModel(
        metadata=DocumentMetadata(filename="f.pdf", page_count=1),
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
        blocks=[],
        tables=[table],
    )
    markdown = MarkdownBuilder().build(doc)
    assert "<table class=\"structured-table\">" in markdown
    assert "<th>A</th>" in markdown
    assert "<td>x</td>" in markdown


def test_flattened_ocr_table_translation_replaces_stale_table_model_once() -> None:
    source = (
        "| | 0-3 MESES | 3-6 MESES |\n"
        "|---|---|---|\n"
        "| ENDOCRINOLOGÍA | Evaluación basal | Tratamiento Hormonal |"
    )
    translated = (
        "| | 0-3 MONTHS | 3-6 MONTHS | "
        "|---|---|---| "
        "| ENDOCRINOLOGY | Baseline evaluation | Hormonal treatment |"
    )
    table = TableModel(
        id="t1",
        page_numbers=[1],
        headers=["", "0-3 MESES", "3-6 MESES"],
        rows=[["ENDOCRINOLOGÍA", "Evaluación basal", "Tratamiento Hormonal"]],
        cells=[
            [
                TableModel.TableCell(text="ENDOCRINOLOGÍA"),
                TableModel.TableCell(text="Evaluación basal"),
                TableModel.TableCell(text="Tratamiento Hormonal"),
            ]
        ],
        caption="TABLA I. Secuencia básica",
        caption_block_id="caption",
        debug={"source_block_id": "table-block"},
    )
    blocks = [
        Block(
            id="table-block",
            page_number=1,
            block_type=BlockType.TABLE,
            text=translated,
            reading_order_index=0,
            source_type=SourceType.OCR,
            metadata={
                "source_text_before_cleaning": source,
                "translated_from_block_ids": ["table-block"],
            },
        ),
        Block(
            id="caption",
            page_number=1,
            block_type=BlockType.CAPTION,
            text="Table I. Basic sequence",
            reading_order_index=1,
            source_type=SourceType.OCR,
        ),
    ]
    doc = DocumentModel(
        metadata=DocumentMetadata(filename="scan.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=blocks,
        tables=[table],
    )

    markdown = MarkdownBuilder().build(doc)

    assert '<figure class="document-table">' in markdown
    assert "ENDOCRINOLOGY" in markdown
    assert "Baseline evaluation" in markdown
    assert "ENDOCRINOLOGÍA" not in markdown
    assert markdown.count("Table I. Basic sequence") == 1
    assert "### Table 1" not in markdown


def test_flattened_markdown_parser_preserves_empty_cells_and_shape() -> None:
    source = (
        "| | A | B | C |\n"
        "|---|---|---|---|\n"
        "| first | | | last |\n"
        "| second | x | | y |"
    )
    translated = (
        "| | A translated | B translated | C translated | "
        "|---|---|---|---| "
        "| first translated | | | last translated | "
        "| second translated | x | | y |"
    )

    rows = parse_table_rows(translated, source_hint=source)

    assert len(rows) == 3
    assert all(len(row) == 4 for row in rows)
    assert [cell.text for cell in rows[1]] == ["first translated", "", "", "last translated"]


def test_table_parser_preserves_escaped_pipe_inside_cell() -> None:
    source = (
        "| Measure | Value |\n"
        "|---|---|\n"
        r"| Risk \| benefit | high |"
    )
    flattened_translation = (
        r"| Measure translated | Value translated | "
        r"|---|---| "
        r"| Risk \| benefit translated | high |"
    )

    multiline_rows = parse_table_rows(source)
    flattened_rows = parse_table_rows(flattened_translation, source_hint=source)

    assert multiline_rows[1][0].text == "Risk | benefit"
    assert flattened_rows[1][0].text == "Risk | benefit translated"
    assert all(len(row) == 2 for row in flattened_rows)


def test_markdown_builder_recovers_unlinked_caption_for_existing_job() -> None:
    table = TableModel(
        id="legacy-table",
        page_numbers=[1],
        headers=["Spanish heading"],
        rows=[["Spanish cell"]],
        cells=[[TableModel.TableCell(text="Spanish cell")]],
    )
    blocks = [
        Block(
            id="legacy-table-block",
            page_number=1,
            block_type=BlockType.TABLE,
            text="| English heading |\n|---|\n| English cell |",
            reading_order_index=10,
            source_type=SourceType.OCR,
            metadata={
                "source_text_before_cleaning": (
                    "| Spanish heading |\n|---|\n| Spanish cell |"
                ),
                "translated_from_block_ids": ["legacy-table-block"],
            },
        ),
        Block(
            id="legacy-caption",
            page_number=1,
            block_type=BlockType.CAPTION,
            text="Table I. Translated legacy caption",
            reading_order_index=11,
            source_type=SourceType.OCR,
        ),
    ]
    doc = DocumentModel(
        metadata=DocumentMetadata(filename="legacy.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=0.1,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=blocks,
        tables=[table],
    )

    markdown = MarkdownBuilder().build(doc)

    assert "English heading" in markdown
    assert "Spanish heading" not in markdown
    assert markdown.count("Table I. Translated legacy caption") == 1
    assert "Table 1" not in markdown


def test_markdown_builder_matches_multiple_legacy_tables_in_page_order() -> None:
    tables = [
        TableModel(
            id="legacy-table-1",
            page_numbers=[1],
            headers=["Spanish H1"],
            rows=[["Spanish C1"]],
            cells=[[TableModel.TableCell(text="Spanish C1")]],
        ),
        TableModel(
            id="legacy-table-2",
            page_numbers=[1],
            headers=["Spanish H2"],
            rows=[["Spanish C2"]],
            cells=[[TableModel.TableCell(text="Spanish C2")]],
        ),
    ]
    blocks: list[Block] = []
    for index in range(2):
        number = index + 1
        blocks.extend(
            [
                Block(
                    id=f"table-block-{number}",
                    page_number=1,
                    block_type=BlockType.TABLE,
                    text=(
                        f"| English H{number} |\n"
                        "|---|\n"
                        f"| English C{number} |"
                    ),
                    reading_order_index=index * 2,
                    source_type=SourceType.OCR,
                    metadata={
                        "source_text_before_cleaning": (
                            f"| Spanish H{number} |\n"
                            "|---|\n"
                            f"| Spanish C{number} |"
                        ),
                        "translated_from_block_ids": [f"table-block-{number}"],
                    },
                ),
                Block(
                    id=f"caption-{number}",
                    page_number=1,
                    block_type=BlockType.CAPTION,
                    text=f"Table {number}. English caption",
                    reading_order_index=index * 2 + 1,
                    source_type=SourceType.OCR,
                ),
            ]
        )
    document = DocumentModel(
        metadata=DocumentMetadata(filename="legacy.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=0.1,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=blocks,
        tables=tables,
    )

    markdown = MarkdownBuilder().build(document)

    for number in (1, 2):
        assert f"English H{number}" in markdown
        assert f"English C{number}" in markdown
        assert f"Spanish H{number}" not in markdown
        assert f"Spanish C{number}" not in markdown
        assert markdown.count(f"Table {number}. English caption") == 1
    assert markdown.count('<figure class="document-table">') == 2
