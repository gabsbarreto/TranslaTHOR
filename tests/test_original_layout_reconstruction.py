from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageChops, ImageDraw

from app.models.schema import (
    Block,
    BlockType,
    BoundingBox,
    DocumentMetadata,
    DocumentModel,
    FigureAsset,
    PageMetadata,
    SourceType,
    TableModel,
    TranslationChunk,
)
from app.services.original_layout_reconstructor import (
    OriginalLayoutReconstructor,
    _SemanticTableGrid,
)


def _translated_block(
    block_id: str,
    block_type: BlockType,
    source: str,
    translated: str,
    bbox: BoundingBox,
    order: int,
) -> Block:
    return Block(
        id=block_id,
        page_number=1,
        block_type=block_type,
        text=translated,
        bbox=bbox,
        reading_order_index=order,
        source_type=SourceType.EMBEDDED,
        style_hints={"font_size": 10},
        metadata={
            "source_text": source,
            "translated_from_block_ids": [block_id],
        },
    )


def _create_original_layout_source(path: Path) -> None:
    pdf = fitz.open()
    page = pdf.new_page(width=400, height=500)
    page.draw_rect(page.rect, fill=(0.96, 0.97, 0.99), color=None)
    page.insert_textbox(
        fitz.Rect(40, 35, 360, 65),
        "QUELLE EINLEITUNG",
        fontsize=10,
    )
    graph = fitz.Rect(60, 100, 340, 280)
    page.draw_rect(graph, color=(0.05, 0.2, 0.65), width=2)
    page.draw_line((80, 250), (315, 250), color=(0, 0, 0), width=1)
    page.draw_line((80, 250), (80, 125), color=(0, 0, 0), width=1)
    page.draw_polyline(
        [(80, 230), (130, 205), (180, 215), (230, 155), (310, 135)],
        color=(0.8, 0.1, 0.1),
        width=2,
    )
    page.insert_text((105, 145), "ACHSE QUELLE", fontsize=9)
    page.insert_textbox(
        fitz.Rect(60, 290, 340, 325),
        "Figure 1. QUELLBESCHRIFTUNG",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(40, 350, 360, 405),
        "QUELLE HAUPTTEXT",
        fontsize=10,
    )
    page.insert_textbox(fitz.Rect(40, 430, 140, 455), "E = mc2", fontsize=10)
    pdf.save(path)
    pdf.close()


def _original_layout_document() -> DocumentModel:
    figure_bbox = BoundingBox(x0=60, y0=100, x1=340, y1=280)
    blocks = [
        _translated_block(
            "intro",
            BlockType.PARAGRAPH,
            "QUELLE EINLEITUNG",
            "Translated introduction",
            BoundingBox(x0=40, y0=35, x1=360, y1=65),
            0,
        ),
        Block(
            id="figure",
            page_number=1,
            block_type=BlockType.FIGURE,
            text="ACHSE QUELLE",
            bbox=figure_bbox,
            reading_order_index=1,
            source_type=SourceType.EMBEDDED,
        ),
        _translated_block(
            "caption",
            BlockType.CAPTION,
            "Figure 1. QUELLBESCHRIFTUNG",
            "Figure 1. Translated external caption",
            BoundingBox(x0=60, y0=290, x1=340, y1=325),
            2,
        ),
        _translated_block(
            "body",
            BlockType.PARAGRAPH,
            "QUELLE HAUPTTEXT",
            "Translated body text",
            BoundingBox(x0=40, y0=350, x1=360, y1=405),
            3,
        ),
        Block(
            id="equation",
            page_number=1,
            block_type=BlockType.EQUATION,
            text="E = mc2",
            bbox=BoundingBox(x0=40, y0=430, x1=140, y1=455),
            reading_order_index=4,
            source_type=SourceType.EMBEDDED,
        ),
    ]
    return DocumentModel(
        metadata=DocumentMetadata(filename="source.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=400,
                height=500,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=blocks,
        figures=[
            FigureAsset(
                id="figure-p0001-001",
                page_number=1,
                bbox=figure_bbox,
                caption_block_id="caption",
                image_path="preview.png",
                source_block_ids=["figure"],
                detection_confidence=0.9,
            )
        ],
    )


def _render_rgb(path: Path, scale: float = 2.0) -> Image.Image:
    with fitz.open(path) as pdf:
        pixmap = pdf[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def test_original_layout_preserves_page_and_pixels_outside_text_regions(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "translated_original_layout.pdf"
    report_path = tmp_path / "report.json"
    _create_original_layout_source(source)

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_original_layout_document(),
        report_path=report_path,
    )

    with fitz.open(source) as source_pdf, fitz.open(output) as output_pdf:
        assert output_pdf.page_count == source_pdf.page_count == 1
        assert output_pdf[0].rect == source_pdf[0].rect
        output_text = output_pdf[0].get_text("text")
        assert "ACHSE QUELLE" in output_text
        assert "Translated external caption" in output_text
        assert "QUELLE HAUPTTEXT" not in output_text

    assert report["status"] == "complete"
    assert report["pages_successfully_reconstructed"] == 1
    assert report["figures_preserved"] == 1
    assert report["regions_replaced"] == 3
    assert report["text_boxes_did_not_fit"] == 0
    assert len(report["scaling_applied"]) == 3
    assert len(report["raster_figure_fallbacks"]) == 1
    assert "low_confidence_figure_or_caption_associations" in report
    assert report_path.is_file()
    assert any(
        region.get("reason") == "locked_visual_region"
        for region in report["regions"]
    )

    source_image = _render_rgb(source)
    output_image = _render_rgb(output)
    graph_box = (120, 200, 680, 560)
    assert source_image.crop(graph_box).tobytes() == output_image.crop(graph_box).tobytes()

    difference = ImageChops.difference(source_image, output_image)
    outside_mask = Image.new("L", difference.size, 255)
    mask_draw = ImageDraw.Draw(outside_mask)
    for region in report["regions"]:
        if region.get("status") != "replaced":
            continue
        bbox = region["bbox"]
        mask_draw.rectangle(
            (
                int(bbox["x0"] * 2) - 3,
                int(bbox["y0"] * 2) - 3,
                int(bbox["x1"] * 2) + 3,
                int(bbox["y1"] * 2) + 3,
            ),
            fill=0,
        )
    outside = Image.composite(difference, Image.new("RGB", difference.size), outside_mask)
    assert outside.getbbox() is None


def test_overflow_is_reported_and_source_text_is_retained(tmp_path: Path) -> None:
    source = tmp_path / "overflow-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=200)
    page.insert_textbox(fitz.Rect(30, 30, 100, 43), "SOURCE", fontsize=7)
    pdf.save(source)
    pdf.close()

    block = _translated_block(
        "tiny",
        BlockType.PARAGRAPH,
        "SOURCE",
        "This translated sentence is far too long to fit inside the tiny source rectangle " * 8,
        BoundingBox(x0=30, y0=30, x1=100, y1=43),
        0,
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename="overflow-source.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=300,
                height=200,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[block],
    )
    output = tmp_path / "overflow-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "overflow-report.json",
    )

    assert report["text_boxes_did_not_fit"] == 1
    assert report["regions_replaced"] == 0
    assert report["pages_using_fallback_behavior"] == 1
    assert any(
        region.get("reason") == "translated_text_did_not_fit_minimum_scale"
        for region in report["regions"]
    )
    with fitz.open(output) as output_pdf:
        assert "SOURCE" in output_pdf[0].get_text("text")


def test_scanned_page_is_retained_with_safe_warning(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.png"
    scan = Image.new("RGB", (400, 500), "white")
    drawing = ImageDraw.Draw(scan)
    drawing.rectangle((25, 25, 375, 475), outline="black", width=3)
    drawing.text((80, 220), "VISIBLE SOURCE SCAN", fill="black")
    scan.save(scan_path)

    source = tmp_path / "scan.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=400, height=500)
    page.insert_image(page.rect, filename=str(scan_path))
    pdf.save(source)
    pdf.close()

    block = _translated_block(
        "ocr-body",
        BlockType.PARAGRAPH,
        "VISIBLE SOURCE SCAN",
        "Translated scan text",
        BoundingBox(x0=70, y0=200, x1=330, y1=260),
        0,
    )
    block.source_type = SourceType.OCR
    document = DocumentModel(
        metadata=DocumentMetadata(filename="scan.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=400,
                height=500,
                has_embedded_text=False,
                embedded_text_quality=0.0,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[block],
    )
    output = tmp_path / "scan-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "scan-report.json",
    )

    assert report["status"] == "partial"
    assert report["pages_using_fallback_behavior"] == 1
    assert report["regions_replaced"] == 0
    assert any(warning["code"] == "page_not_safely_replaceable" for warning in report["warnings"])
    assert _render_rgb(source).tobytes() == _render_rgb(output).tobytes()


def test_cross_page_translation_batch_is_recovered_per_source_block(tmp_path: Path) -> None:
    source = tmp_path / "cross-page-source.pdf"
    pdf = fitz.open()
    for text in ("SOURCE PAGE ONE", "SOURCE PAGE TWO"):
        page = pdf.new_page(width=300, height=200)
        page.insert_textbox(fitz.Rect(30, 40, 270, 75), text, fontsize=10)
    pdf.save(source)
    pdf.close()

    first = Block(
        id="page-one",
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text="Translated page one\n\nTranslated page two",
        bbox=BoundingBox(x0=30, y0=40, x1=270, y1=75),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        style_hints={"font_size": 10},
        metadata={
            "source_text": "SOURCE PAGE ONE",
            "translated_from_block_ids": ["page-one", "page-two"],
        },
    )
    second = Block(
        id="page-two",
        page_number=2,
        block_type=BlockType.PARAGRAPH,
        text="",
        bbox=BoundingBox(x0=30, y0=40, x1=270, y1=75),
        reading_order_index=1,
        source_type=SourceType.EMBEDDED,
        style_hints={"font_size": 10},
        metadata={
            "source_text": "SOURCE PAGE TWO",
            "merged_into_block_id": "page-one",
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=2),
        pages=[
            PageMetadata(
                page_number=page_number,
                width=300,
                height=200,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
            for page_number in (1, 2)
        ],
        blocks=[first, second],
        translation_chunks=[
            TranslationChunk(
                id="chunk-cross-page",
                block_ids=["page-one", "page-two"],
                source_text="SOURCE PAGE ONE\n\nSOURCE PAGE TWO",
                translated_text="Translated page one\n\nTranslated page two",
            )
        ],
    )

    output = tmp_path / "cross-page-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "cross-page-report.json",
    )

    with fitz.open(output) as translated:
        assert "Translated page one" in translated[0].get_text("text")
        assert "Translated page two" in translated[1].get_text("text")
        assert "SOURCE PAGE" not in "".join(page.get_text("text") for page in translated)
    assert report["pages_successfully_reconstructed"] == 2
    assert report["pages_using_fallback_behavior"] == 0


def test_canonical_figure_asset_prevents_group_bbox_from_locking_caption(tmp_path: Path) -> None:
    source = tmp_path / "nested-figure-source.pdf"
    output = tmp_path / "nested-figure-output.pdf"
    _create_original_layout_source(source)
    document = _original_layout_document()
    group = Block(
        id="figure-group",
        page_number=1,
        block_type=BlockType.FIGURE,
        text="",
        bbox=BoundingBox(x0=60, y0=100, x1=340, y1=325),
        reading_order_index=1,
        source_type=SourceType.EMBEDDED,
    )
    document.blocks.insert(1, group)
    document.figures[0].source_block_ids = ["figure-group", "figure"]

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "nested-figure-report.json",
    )

    with fitz.open(output) as translated:
        text = translated[0].get_text("text")
        assert "Translated external caption" in text
        assert "QUELLBESCHRIFTUNG" not in text
        assert "ACHSE QUELLE" in text
    assert not any(
        region.get("reason") == "overlaps_figure_graph_or_equation"
        and region.get("block_ids") == ["caption"]
        for region in report["regions"]
    )


def test_source_pdf_font_size_allows_tight_two_line_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source-font-size.pdf"
    source_text = (
        "Variable situation professionnelle en étudiant pour collège, lycée et études "
        "supérieures, emploi et autre situation"
    )
    translated_text = (
        "Employment status as student for middle school, high school and higher education, "
        "employment and other status"
    )
    pdf = fitz.open()
    page = pdf.new_page(width=425, height=150)
    page.insert_text(
        (45, 47),
        "Variable situation professionnelle en étudiant pour collège, lycée et études",
        fontsize=7,
    )
    page.insert_text(
        (45, 55),
        "supérieures, emploi et autre situation",
        fontsize=7,
    )
    pdf.save(source)
    pdf.close()

    block = Block(
        id="tight",
        page_number=1,
        block_type=BlockType.LIST,
        text=translated_text,
        bbox=BoundingBox(x0=45, y0=40, x1=379, y1=58),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "source_text": source_text,
            "translated_from_block_ids": ["tight"],
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=425,
                height=150,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[block],
    )
    output = tmp_path / "source-font-size-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "source-font-size-report.json",
    )

    assert report["text_boxes_did_not_fit"] == 0
    assert report["regions_replaced"] == 1
    with fitz.open(output) as translated:
        assert "Employment status" in translated[0].get_text("text")


def test_vector_grid_table_is_reconstructed_cell_by_cell(tmp_path: Path) -> None:
    source = tmp_path / "table-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=180)
    cells = [
        (fitz.Rect(30, 40, 200, 70), "Diagnostic", True),
        (fitz.Rect(200, 40, 270, 70), "N", True),
        (fitz.Rect(30, 70, 200, 100), "Dépression", False),
        (fitz.Rect(200, 70, 270, 100), "15", False),
    ]
    for rectangle, text, _bold in cells:
        page.draw_rect(rectangle, color=(0, 0, 0), width=0.5)
        page.insert_textbox(rectangle, text, fontsize=8, align=fitz.TEXT_ALIGN_CENTER)
    pdf.save(source)
    pdf.close()

    source_table = (
        "<table><tr><th>Diagnostic</th><th>N</th></tr>"
        "<tr><td>Dépression</td><td>15</td></tr></table>"
    )
    translated_table = (
        "<table><tr><th>Diagnosis</th><th>N</th></tr>"
        "<tr><td>Depression</td><td>15</td></tr></table>"
    )
    block = Block(
        id="table",
        page_number=1,
        block_type=BlockType.TABLE,
        text=translated_table,
        bbox=BoundingBox(x0=30, y0=40, x1=270, y1=100),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "source_text": source_table,
            "translated_from_block_ids": ["table"],
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=300,
                height=180,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[block],
    )
    output = tmp_path / "table-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "table-report.json",
    )

    with fitz.open(source) as source_pdf, fitz.open(output) as translated:
        text = translated[0].get_text("text")
        assert "Diagnosis" in text
        assert "Depression" in text
        assert "Dépression" not in text
        assert len(translated[0].get_drawings()) == len(source_pdf[0].get_drawings())
    assert report["status"] == "complete"
    assert report["regions_replaced"] == 2


def _horizontal_rule_table_source(path: Path) -> tuple[str, str, Block, list[float], list[float]]:
    x_edges = [30.0, 135.0, 310.0, 390.0]
    y_edges = [30.0, 55.0, 115.0, 180.0]
    source_cells = [
        ["Kategorie", "Beschreibung", "N"],
        ["Lange Quelle", "Erste Beschreibung\nüber zwei Zeilen", "10"],
        ["Zweite Quelle", "Weitere Beschreibung", "20"],
    ]
    pdf = fitz.open()
    page = pdf.new_page(width=420, height=230)
    for y in y_edges:
        for left, right in zip(x_edges, x_edges[1:]):
            page.draw_line((left, y), (right, y), color=(0, 0, 0), width=0.6)
    for row_index, row in enumerate(source_cells):
        for column_index, text in enumerate(row):
            page.insert_textbox(
                fitz.Rect(
                    x_edges[column_index] + 3,
                    y_edges[row_index] + 3,
                    x_edges[column_index + 1] - 3,
                    y_edges[row_index + 1] - 3,
                ),
                text,
                fontsize=7.5,
                align=fitz.TEXT_ALIGN_CENTER if column_index == 2 else fitz.TEXT_ALIGN_LEFT,
            )
    page.insert_text((30, 210), "OUTSIDE TABLE", fontsize=8)
    pdf.save(path)
    pdf.close()

    source_markup = (
        "<table><tr><th>Kategorie</th><th>Beschreibung</th><th>N</th></tr>"
        "<tr><td>Lange Quelle</td><td>Erste Beschreibung<br>über zwei Zeilen</td><td>10</td></tr>"
        "<tr><td>Zweite Quelle</td><td>Weitere Beschreibung</td><td>20</td></tr></table>"
    )
    translated_markup = (
        "<table><tr><th>Category</th><th>Description</th><th>N</th></tr>"
        "<tr><td>Long source</td><td>First description over two lines</td><td>10</td></tr>"
        "<tr><td>Second source</td><td>Additional description</td><td>20</td></tr></table>"
    )
    block = Block(
        id="horizontal-table",
        page_number=1,
        block_type=BlockType.TABLE,
        text=translated_markup,
        bbox=BoundingBox(x0=x_edges[0], y0=y_edges[0], x1=x_edges[-1], y1=y_edges[-1]),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "source_text": source_markup,
            "translated_from_block_ids": ["horizontal-table"],
            "marker_page_width": 420,
            "marker_page_height": 230,
        },
    )
    return source_markup, translated_markup, block, x_edges, y_edges


def _single_page_table_document(block: Block, *, tables: list[TableModel] | None = None) -> DocumentModel:
    return DocumentModel(
        metadata=DocumentMetadata(filename="horizontal-table.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=420,
                height=230,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[block],
        tables=tables or [],
    )


def _stored_horizontal_table(
    *,
    block: Block,
    source_markup: str,
    x_edges: list[float],
    y_edges: list[float],
) -> TableModel:
    parsed_source = OriginalLayoutReconstructor()._parse_table_rows(source_markup)

    def table_cell(row: int, column: int) -> TableModel.TableCell:
        return TableModel.TableCell(
            text=parsed_source[row][column].text,
            bbox=BoundingBox(
                x0=x_edges[column],
                y0=y_edges[row],
                x1=x_edges[column + 1],
                y1=y_edges[row + 1],
            ),
            row_index=row,
            column_index=column,
        )

    return TableModel(
        id="stored-horizontal-table",
        page_numbers=[1],
        page=1,
        bbox=block.bbox,
        headers=[cell.text for cell in parsed_source[0]],
        header_cells=[table_cell(0, column) for column in range(3)],
        rows=[[cell.text for cell in row] for row in parsed_source[1:]],
        cells=[
            [table_cell(row, column) for column in range(3)]
            for row in range(1, 3)
        ],
        debug={
            "marker_block_id": block.id,
            "cell_geometry_source": "marker_table_cell_polygons",
            "cell_coordinate_space": {
                "name": "marker_page_coordinates",
                "width": 420,
                "height": 230,
            },
        },
    )


def test_horizontal_rule_table_is_semantically_coalesced(tmp_path: Path) -> None:
    source = tmp_path / "horizontal-table-source.pdf"
    _source_markup, _translated_markup, block, _x_edges, y_edges = _horizontal_rule_table_source(
        source
    )
    output = tmp_path / "horizontal-table-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_single_page_table_document(block),
        report_path=tmp_path / "horizontal-table-report.json",
    )

    with fitz.open(source) as source_pdf, fitz.open(output) as translated:
        text = translated[0].get_text("text")
        assert "Category" in text
        assert "Additional description" in text
        assert "Kategorie" not in text
        assert "Beschreibung" not in text
        assert "OUTSIDE TABLE" in text
        assert len(translated[0].get_drawings()) == len(source_pdf[0].get_drawings())
    strategies = {
        metadata.get("table_grid_detection")
        for region in report["regions"]
        for metadata in region.get("coordinate_metadata", [])
    }
    assert "pymupdf_text_lattice_semantic_alignment" in strategies
    assert not any(
        region.get("reason") == "table_cell_geometry_unreliable"
        for region in report["regions"]
    )

    source_image = _render_rgb(source, scale=2)
    output_image = _render_rgb(output, scale=2)
    for y in y_edges:
        rule_strip = (60, round(y * 2) - 1, 780, round(y * 2) + 2)
        assert source_image.crop(rule_strip).tobytes() == output_image.crop(rule_strip).tobytes()
    assert source_image.crop((60, 400, 780, 440)).tobytes() == output_image.crop(
        (60, 400, 780, 440)
    ).tobytes()


def test_marker_cell_polygons_are_preferred_for_table_geometry(tmp_path: Path) -> None:
    source = tmp_path / "stored-table-source.pdf"
    source_markup, _translated_markup, block, x_edges, y_edges = _horizontal_rule_table_source(
        source
    )
    table = _stored_horizontal_table(
        block=block,
        source_markup=source_markup,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=tmp_path / "stored-table-output.pdf",
        document=_single_page_table_document(block, tables=[table]),
        report_path=tmp_path / "stored-table-report.json",
    )

    strategies = {
        metadata.get("table_grid_detection")
        for region in report["regions"]
        for metadata in region.get("coordinate_metadata", [])
    }
    assert strategies == {"marker_table_cell_polygons"}
    assert report["regions_skipped"] == 0


def test_marker_cell_polygons_require_matching_source_topology(tmp_path: Path) -> None:
    source = tmp_path / "stored-table-topology-source.pdf"
    source_markup, _translated_markup, block, x_edges, y_edges = _horizontal_rule_table_source(
        source
    )
    source_rows = OriginalLayoutReconstructor()._parse_table_rows(source_markup)
    table = _stored_horizontal_table(
        block=block,
        source_markup=source_markup,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    assert block.bbox is not None

    invalid_tables: list[TableModel] = []
    invalid_row = table.model_copy(deep=True)
    invalid_row.header_cells[0].row_index = 1
    invalid_tables.append(invalid_row)
    invalid_column = table.model_copy(deep=True)
    invalid_column.header_cells[1].column_index = 0
    invalid_tables.append(invalid_column)
    invalid_rowspan = table.model_copy(deep=True)
    invalid_rowspan.cells[0][0].rowspan = 2
    invalid_tables.append(invalid_rowspan)
    invalid_colspan = table.model_copy(deep=True)
    invalid_colspan.cells[0][1].colspan = 2
    invalid_tables.append(invalid_colspan)

    with fitz.open(source) as pdf:
        reconstructor = OriginalLayoutReconstructor()
        valid_rows, valid_strategy = reconstructor._stored_table_grid_rows(
            page=pdf[0],
            block=block,
            table=table,
            source_rows=source_rows,
            table_bbox=block.bbox,
        )
        assert valid_rows
        assert valid_strategy == "marker_table_cell_polygons"
        for invalid_table in invalid_tables:
            rows, strategy = reconstructor._stored_table_grid_rows(
                page=pdf[0],
                block=block,
                table=invalid_table,
                source_rows=source_rows,
                table_bbox=block.bbox,
            )
            assert rows == []
            assert strategy == "unavailable"


def test_marker_cell_polygons_reject_swapped_and_repeated_regions(tmp_path: Path) -> None:
    source = tmp_path / "stored-table-placement-source.pdf"
    source_markup, _translated_markup, block, x_edges, y_edges = _horizontal_rule_table_source(
        source
    )
    source_rows = OriginalLayoutReconstructor()._parse_table_rows(source_markup)
    table = _stored_horizontal_table(
        block=block,
        source_markup=source_markup,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    assert block.bbox is not None

    swapped = table.model_copy(deep=True)
    first_bbox = swapped.header_cells[0].bbox
    second_bbox = swapped.header_cells[1].bbox
    assert first_bbox is not None
    assert second_bbox is not None
    swapped.header_cells[0].bbox = second_bbox
    swapped.header_cells[1].bbox = first_bbox

    repeated = table.model_copy(deep=True)
    repeated_bbox = repeated.header_cells[0].bbox
    assert repeated_bbox is not None
    repeated.header_cells[1].bbox = repeated_bbox.model_copy(deep=True)

    with fitz.open(source) as pdf:
        reconstructor = OriginalLayoutReconstructor()
        for invalid_table in (swapped, repeated):
            rows, strategy = reconstructor._stored_table_grid_rows(
                page=pdf[0],
                block=block,
                table=invalid_table,
                source_rows=source_rows,
                table_bbox=block.bbox,
            )
            assert rows == []
            assert strategy == "unavailable"


def test_semantic_table_geometry_rejects_source_text_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "mismatched-horizontal-table.pdf"
    _source_markup, _translated_markup, block, _x_edges, _y_edges = _horizontal_rule_table_source(
        source
    )
    block.metadata["source_text"] = str(block.metadata["source_text"]).replace(
        "Kategorie",
        "Unrelated source heading",
    )
    output = tmp_path / "mismatched-horizontal-table-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_single_page_table_document(block),
        report_path=tmp_path / "mismatched-horizontal-table-report.json",
    )

    assert report["regions_replaced"] == 0
    assert any(
        region.get("reason") == "table_cell_geometry_unreliable"
        for region in report["regions"]
    )
    assert _render_rgb(source).tobytes() == _render_rgb(output).tobytes()


def test_semantic_table_geometry_rejects_near_tied_partitions(monkeypatch) -> None:
    reconstructor = OriginalLayoutReconstructor()
    source_rows = reconstructor._parse_table_rows(
        "<table><tr><td>Alpha</td><td>Beta</td></tr></table>"
    )
    boundary_candidates = [(10.0, 100.0, 210.0), (10.0, 120.0, 210.0)]
    monkeypatch.setattr(
        reconstructor,
        "_horizontal_rule_column_candidates",
        lambda _page, _rect, _columns: boundary_candidates,
    )

    def candidate(*, column_edges, **_kwargs):
        first = column_edges == boundary_candidates[0]
        return _SemanticTableGrid(
            rows=((fitz.Rect(10, 10, 100, 30), fitz.Rect(100, 10, 210, 30)),),
            score=0.80 if first else 0.79,
            signature=tuple(column_edges),
            assignment_signature=("alpha", "beta" if first else "beta-shifted"),
        )

    monkeypatch.setattr(reconstructor, "_semantic_grid_candidate", candidate)
    pdf = fitz.open()
    page = pdf.new_page(width=220, height=100)
    try:
        rows = reconstructor._semantic_text_table_grid_rows(
            page=page,
            table_rect=fitz.Rect(10, 10, 210, 30),
            search_rect=fitz.Rect(7, 7, 213, 33),
            source_rows=source_rows,
        )
    finally:
        pdf.close()

    assert rows == []


def test_hidden_ocr_table_masks_text_and_preserves_grid_and_figure(tmp_path: Path) -> None:
    page_width, page_height = 320, 240
    scan_path = tmp_path / "table-scan.png"
    scan = Image.new("RGB", (page_width, page_height), "white")
    drawing = ImageDraw.Draw(scan)
    xs = [30, 150, 290]
    ys = [35, 65, 95]
    for x in xs:
        drawing.line((x, ys[0], x, ys[-1]), fill="black", width=2)
    for y in ys:
        drawing.line((xs[0], y, xs[-1], y), fill="black", width=2)
    visible_cells = [
        ((42, 44), "Diagnostico"),
        ((205, 44), "N"),
        ((42, 74), "Depresion"),
        ((205, 74), "15"),
    ]
    for position, text in visible_cells:
        drawing.text(position, text, fill="black")
    drawing.text((30, 105), "TABLA I. Resultados", fill="black")
    drawing.rectangle((220, 155, 295, 225), fill=(30, 90, 180), outline="black", width=2)
    scan.save(scan_path)

    source = tmp_path / "hidden-ocr-table.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=page_width, height=page_height)
    page.insert_image(page.rect, filename=str(scan_path))
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), color=(0, 0, 0), width=0.5)
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), color=(0, 0, 0), width=0.5)
    hidden_cells = [
        (fitz.Rect(32, 37, 148, 63), "Diagnostico"),
        (fitz.Rect(152, 37, 288, 63), "N"),
        (fitz.Rect(32, 67, 148, 93), "Depresion"),
        (fitz.Rect(152, 67, 288, 93), "15"),
    ]
    for rectangle, text in hidden_cells:
        page.insert_textbox(
            rectangle,
            text,
            fontsize=9,
            align=fitz.TEXT_ALIGN_CENTER,
            render_mode=3,
        )
    page.insert_textbox(
        fitz.Rect(30, 101, 250, 120),
        "TABLA I. Resultados",
        fontsize=8,
        render_mode=3,
    )
    pdf.save(source)
    pdf.close()

    source_table = (
        "| Diagnostico | N |\n"
        "|---|---|\n"
        "| Depresion | 15 |"
    )
    translated_table = (
        "| Diagnosis | N | "
        "|---|---| "
        "| Depression | 15 |"
    )
    table = Block(
        id="ocr-table",
        page_number=1,
        block_type=BlockType.TABLE,
        text=translated_table,
        bbox=BoundingBox(x0=30, y0=35, x1=290, y1=95),
        reading_order_index=0,
        source_type=SourceType.OCR,
        metadata={
            "source_text": source_table.replace("\n", " "),
            "source_text_before_cleaning": source_table,
            "translated_from_block_ids": ["ocr-table"],
        },
    )
    caption = Block(
        id="ocr-caption",
        page_number=1,
        block_type=BlockType.CAPTION,
        text="Table I. Results",
        bbox=BoundingBox(x0=30, y0=101, x1=250, y1=120),
        reading_order_index=1,
        source_type=SourceType.OCR,
        metadata={
            "source_text": "TABLA I. Resultados",
            "translated_from_block_ids": ["ocr-caption"],
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=page_width,
                height=page_height,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[table, caption],
        figures=[
            FigureAsset(
                id="figure",
                page_number=1,
                bbox=BoundingBox(x0=220, y0=155, x1=295, y1=225),
            )
        ],
    )

    output = tmp_path / "hidden-ocr-table-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "hidden-ocr-table-report.json",
    )

    with fitz.open(output) as translated:
        output_text = translated[0].get_text("text")
        assert "Diagnosis" in output_text
        assert "Depression" in output_text
        assert "Table I. Results" in output_text
        assert "Diagnostico" not in output_text
        assert "Depresion" not in output_text
    assert report["raster_tables_reconstructed"] == 1
    assert report["scan_overlay_pages"] == 1
    # Only changed text cells plus the caption are masked. Unchanged numeric
    # and symbol cells stay pixel-identical so arrows and annotations survive.
    assert report["scan_text_masks"] == 3
    assert report["pages"][0]["reconstruction_strategy"] == "ocr_table_overlay"

    source_image = _render_rgb(source, scale=2)
    output_image = _render_rgb(output, scale=2)
    figure_box = (440, 310, 590, 450)
    assert source_image.crop(figure_box).tobytes() == output_image.crop(figure_box).tobytes()
    for coordinate in ((60, 70), (300, 70), (580, 70), (60, 130), (60, 190)):
        assert source_image.getpixel(coordinate) == output_image.getpixel(coordinate)

    approved = Image.new("L", source_image.size, 0)
    approved_draw = ImageDraw.Draw(approved)
    for region in report["regions"]:
        assert region["source_text_masks"]
        for box in [region["bbox"], *region["source_text_masks"]]:
            approved_draw.rectangle(
                (
                    round(box["x0"] * 2) - 2,
                    round(box["y0"] * 2) - 2,
                    round(box["x1"] * 2) + 2,
                    round(box["y1"] * 2) + 2,
                ),
                fill=255,
            )
    difference = ImageChops.difference(source_image, output_image)
    outside_difference = Image.composite(
        Image.new("RGB", difference.size, "black"),
        difference,
        approved,
    )
    assert outside_difference.getbbox() is None


def test_scan_table_rejects_content_added_to_empty_visual_cell(tmp_path: Path) -> None:
    source = tmp_path / "empty-cell-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=150)
    xs = [30, 150, 270]
    ys = [30, 60, 90]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), color=(0, 0, 0), width=0.6)
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), color=(0, 0, 0), width=0.6)
    page.insert_textbox(
        fitz.Rect(34, 34, 146, 56),
        "Source header words",
        fontsize=7,
        align=fitz.TEXT_ALIGN_CENTER,
    )
    page.draw_line((175, 45), (240, 45), color=(0.1, 0.2, 0.8), width=1.2)
    page.draw_polyline(
        [(240, 45), (234, 41), (234, 49)],
        color=(0.1, 0.2, 0.8),
        width=1.2,
    )
    page.insert_textbox(
        fitz.Rect(34, 64, 146, 86),
        "Second source label",
        fontsize=7,
        align=fitz.TEXT_ALIGN_CENTER,
    )
    page.insert_textbox(
        fitz.Rect(154, 64, 266, 86),
        "value here",
        fontsize=7,
        align=fitz.TEXT_ALIGN_CENTER,
    )
    page.insert_text((40, 115), "UNRELATED BODY TEXT", fontsize=8)
    pdf.save(source)
    pdf.close()

    source_table = (
        "| Source header words | |\n"
        "|---|---|\n"
        "| Second source label | value here |"
    )
    translated_table = (
        "| Source header words | Invented text |\n"
        "|---|---|\n"
        "| Second source label | value here |"
    )
    table = Block(
        id="unsafe-empty-cell",
        page_number=1,
        block_type=BlockType.TABLE,
        text=translated_table,
        bbox=BoundingBox(x0=30, y0=30, x1=270, y1=90),
        reading_order_index=0,
        source_type=SourceType.OCR,
        metadata={
            "source_text": source_table,
            "source_text_before_cleaning": source_table,
            "translated_from_block_ids": ["unsafe-empty-cell"],
        },
    )
    drifted_caption = Block(
        id="drifted-caption",
        page_number=1,
        block_type=BlockType.CAPTION,
        text="Table I. Translated caption",
        bbox=BoundingBox(x0=30, y0=100, x1=270, y1=125),
        reading_order_index=1,
        source_type=SourceType.OCR,
        metadata={
            "source_text": "TABLA I. Leyenda original",
            "translated_from_block_ids": ["drifted-caption"],
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=300,
                height=150,
                has_embedded_text=True,
                embedded_text_quality=0.1,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[table, drifted_caption],
    )
    output = tmp_path / "empty-cell-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "empty-cell-report.json",
    )

    assert report["regions_replaced"] == 0
    assert report["raster_tables_reconstructed"] == 0
    assert any(
        region.get("reason") == "table_translation_added_content_to_empty_source_cell"
        for region in report["regions"]
    )
    assert any(
        region.get("reason") == "caption_hidden_ocr_text_mismatch"
        for region in report["regions"]
    )
    assert ImageChops.difference(
        _render_rgb(source, scale=2),
        _render_rgb(output, scale=2),
    ).getbbox() is None


def test_scan_background_sampling_preserves_light_cell_colour(tmp_path: Path) -> None:
    source = tmp_path / "coloured-cell.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=220, height=120)
    page.draw_rect(
        fitz.Rect(20, 20, 200, 100),
        color=None,
        fill=(0.92, 0.96, 1.0),
    )
    page.insert_text((60, 62), "SOURCE LABEL", fontsize=9)
    pdf.save(source)
    pdf.close()

    with fitz.open(source) as opened:
        fill, metadata = OriginalLayoutReconstructor()._scan_background_fill(
            opened[0],
            BoundingBox(x0=20, y0=20, x1=200, y1=100),
        )

    assert fill is not None
    assert fill[2] > fill[1] > fill[0]
    assert metadata["background_uniform_ratio"] > 0.8
    assert metadata["sampled_background_rgb"][2] >= 250


def test_scan_background_sampling_rejects_mixed_cell_background(tmp_path: Path) -> None:
    source = tmp_path / "mixed-cell.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=220, height=120)
    page.draw_rect(fitz.Rect(20, 20, 110, 100), color=None, fill=(1.0, 1.0, 1.0))
    page.draw_rect(fitz.Rect(110, 20, 200, 100), color=None, fill=(0.86, 0.94, 1.0))
    pdf.save(source)
    pdf.close()

    with fitz.open(source) as opened:
        fill, metadata = OriginalLayoutReconstructor()._scan_background_fill(
            opened[0],
            BoundingBox(x0=20, y0=20, x1=200, y1=100),
        )

    assert fill is None
    assert metadata["reason"] == "background_not_uniform_or_light"
    assert metadata["background_uniform_ratio"] < 0.8


def test_table_overflow_is_atomic_across_all_cells(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "atomic-table-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=150)
    xs = [30, 150, 270]
    ys = [30, 60, 90]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]), color=(0, 0, 0), width=0.6)
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y), color=(0, 0, 0), width=0.6)
    for rectangle, text in (
        (fitz.Rect(34, 34, 146, 56), "Uno"),
        (fitz.Rect(154, 34, 266, 56), "Dos"),
        (fitz.Rect(34, 64, 146, 86), "Tres"),
        (fitz.Rect(154, 64, 266, 86), "Cuatro"),
    ):
        page.insert_textbox(rectangle, text, fontsize=8, align=fitz.TEXT_ALIGN_CENTER)
    pdf.save(source)
    pdf.close()

    source_table = "| Uno | Dos |\n|---|---|\n| Tres | Cuatro |"
    translated_table = "| One | Two |\n|---|---|\n| Three | Four |"
    table = Block(
        id="atomic-table",
        page_number=1,
        block_type=BlockType.TABLE,
        text=translated_table,
        bbox=BoundingBox(x0=30, y0=30, x1=270, y1=90),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "source_text": source_table,
            "source_text_before_cleaning": source_table,
            "translated_from_block_ids": ["atomic-table"],
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=300,
                height=150,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[table],
    )
    reconstructor = OriginalLayoutReconstructor()
    preflight_calls = 0

    def fail_second_preflight(**_kwargs) -> tuple[float, float]:
        nonlocal preflight_calls
        preflight_calls += 1
        return (-1.0, 0.5) if preflight_calls == 2 else (1.0, 1.0)

    reconstructor._preflight = fail_second_preflight  # type: ignore[method-assign]
    output = tmp_path / "atomic-table-output.pdf"
    report = reconstructor.reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "atomic-table-report.json",
    )

    assert report["regions_replaced"] == 0
    assert report["text_boxes_did_not_fit"] == 1
    assert any(
        region.get("reason") == "table_atomic_reconstruction_overflow"
        for region in report["regions"]
    )
    assert ImageChops.difference(
        _render_rgb(source, scale=2),
        _render_rgb(output, scale=2),
    ).getbbox() is None

    postflight_reconstructor = OriginalLayoutReconstructor()
    postflight_reconstructor._preflight = (  # type: ignore[method-assign]
        lambda **_kwargs: (1.0, 1.0)
    )
    original_insert_htmlbox = fitz.Page.insert_htmlbox
    insert_calls = 0

    def fail_second_real_insert(page, *args, **kwargs):
        nonlocal insert_calls
        insert_calls += 1
        if insert_calls == 2:
            return -1.0, 0.5
        return original_insert_htmlbox(page, *args, **kwargs)

    monkeypatch.setattr(fitz.Page, "insert_htmlbox", fail_second_real_insert)
    postflight_output = tmp_path / "atomic-postflight-output.pdf"
    postflight_report = postflight_reconstructor.reconstruct(
        source_pdf_path=source,
        output_pdf_path=postflight_output,
        document=document,
        report_path=tmp_path / "atomic-postflight-report.json",
    )

    assert postflight_report["regions_replaced"] == 0
    assert postflight_report["regions_skipped"] == 4
    assert postflight_report["pages"][0]["status"] == "fallback_original_page"
    assert postflight_report["raster_tables_reconstructed"] == 0
    assert any(
        warning["code"] == "page_reconstruction_rolled_back"
        for warning in postflight_report["warnings"]
    )
    assert all(
        entry["status"] == "rolled_back"
        for entry in postflight_report["scaling_applied"]
    )
    assert ImageChops.difference(
        _render_rgb(source, scale=2),
        _render_rgb(postflight_output, scale=2),
    ).getbbox() is None


def test_unreliable_table_structure_is_retained_with_warning(tmp_path: Path) -> None:
    source = tmp_path / "bad-table-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=150)
    rectangle = fitz.Rect(30, 40, 270, 80)
    page.draw_rect(rectangle, color=(0, 0, 0), width=0.5)
    page.insert_textbox(rectangle, "SOURCE TABLE", fontsize=8)
    pdf.save(source)
    pdf.close()

    repeated = "duplicated table content " * 30
    block = Block(
        id="bad-table",
        page_number=1,
        block_type=BlockType.TABLE,
        text=f"<table><tr><td>{repeated}</td><td>Translated</td></tr></table>",
        bbox=BoundingBox(x0=30, y0=40, x1=270, y1=80),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "source_text": f"<table><tr><td>{repeated}</td><td>Source</td></tr></table>",
            "translated_from_block_ids": ["bad-table"],
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=300,
                height=150,
                has_embedded_text=True,
                embedded_text_quality=1.0,
                extraction_mode=SourceType.EMBEDDED,
            )
        ],
        blocks=[block],
    )
    output = tmp_path / "bad-table-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "bad-table-report.json",
    )

    assert report["status"] == "partial"
    assert any(
        region.get("reason") == "table_translation_structure_unreliable"
        for region in report["regions"]
    )
    with fitz.open(output) as translated:
        assert "SOURCE TABLE" in translated[0].get_text("text")
