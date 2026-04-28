from app.models.schema import (
    DocumentMetadata,
    DocumentModel,
    PageMetadata,
    SourceType,
    TableModel,
)
from app.services.markdown_builder import MarkdownBuilder


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
