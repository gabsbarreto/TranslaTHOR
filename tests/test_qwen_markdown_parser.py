from __future__ import annotations

from pathlib import Path

from app.models.inspection import PageInspection, PdfInspection
from app.models.schema import BlockType
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
