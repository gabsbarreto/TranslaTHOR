from __future__ import annotations

import json
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
    _HiddenOCRLine,
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
    assert report["regions_retained"] == 2
    assert report["text_boxes_did_not_fit"] == 0
    assert len(report["scaling_applied"]) == 3
    assert len(report["raster_figure_fallbacks"]) == 1
    assert "low_confidence_figure_or_caption_associations" in report
    assert report_path.is_file()
    locked_region = next(
        region for region in report["regions"] if region.get("reason") == "locked_visual_region"
    )
    assert locked_region["status"] == "retained"
    assert locked_region["source_character_count"] == len("ACHSE QUELLE")
    assert locked_region["bbox"] == {
        "x0": 60.0,
        "y0": 100.0,
        "x1": 340.0,
        "y1": 280.0,
    }
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted_report["regions"] == report["regions"]

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


def test_embedded_text_bbox_with_unexplained_neighbor_is_retained(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contaminated-paragraph-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=180)
    page.insert_text((30, 50), "Texto fuente correcto", fontsize=10)
    page.insert_text((30, 75), "DO NOT DELETE", fontsize=10)
    pdf.save(source)
    pdf.close()

    block = _translated_block(
        "oversized-paragraph",
        BlockType.PARAGRAPH,
        "Texto fuente correcto",
        "Correct source text",
        BoundingBox(x0=25, y0=30, x1=275, y1=85),
        0,
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
    output = tmp_path / "contaminated-paragraph-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "contaminated-paragraph-report.json",
    )

    assert report["regions_replaced"] == 0
    assert report["regions_skipped"] == 1
    skipped = next(region for region in report["regions"] if region["status"] == "skipped")
    assert skipped["reason"] == "embedded_source_bbox_contains_unexplained_text"
    validation = skipped["alignment_diagnostics"]["source_text_validation"]
    assert validation["safe"] is False
    assert validation["unexplained_actual_characters"] >= len("DONOTDELETE")
    assert _render_rgb(source).tobytes() == _render_rgb(output).tobytes()
    with fitz.open(output) as translated:
        text = translated[0].get_text("text")
        assert "Texto fuente correcto" in text
        assert "DO NOT DELETE" in text
        assert "Correct source text" not in text


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
    overflow_region = next(
        region
        for region in report["regions"]
        if region.get("reason") == "translated_text_did_not_fit_minimum_scale"
    )
    assert overflow_region["source_character_count"] == len("SOURCE")
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


def test_intentional_exclusion_is_retained_without_marking_page_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "excluded-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=200)
    page.insert_textbox(fitz.Rect(30, 30, 270, 60), "JOURNAL HEADER", fontsize=10)
    pdf.save(source)
    pdf.close()

    block = Block(
        id="header",
        page_number=1,
        block_type=BlockType.HEADER,
        text="JOURNAL HEADER",
        bbox=BoundingBox(x0=60, y0=60, x1=540, y1=120),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "excluded_from_translation": True,
            "translation_exclusion_reason": "page_header",
            "surya_page_width": 600,
            "surya_page_height": 400,
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
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

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=tmp_path / "excluded-output.pdf",
        document=document,
        report_path=tmp_path / "excluded-report.json",
    )

    assert report["status"] == "complete"
    assert report["pages_successfully_reconstructed"] == 1
    assert report["pages_using_fallback_behavior"] == 0
    assert report["regions_skipped"] == 0
    assert report["regions_retained"] == 1
    assert report["regions"][0]["status"] == "retained"
    assert report["regions"][0]["reason"] == "page_header"
    assert report["regions"][0]["bbox"] == {
        "x0": 30.0,
        "y0": 30.0,
        "x1": 270.0,
        "y1": 60.0,
    }
    assert report["warnings"] == []


def test_intentionally_retained_region_has_null_bbox_only_when_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "excluded-missing-bbox-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=200)
    page.insert_text((30, 50), "JOURNAL HEADER", fontsize=10)
    pdf.save(source)
    pdf.close()

    block = Block(
        id="header-without-bbox",
        page_number=1,
        block_type=BlockType.HEADER,
        text="JOURNAL HEADER",
        bbox=None,
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "excluded_from_translation": True,
            "translation_exclusion_reason": "page_header",
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
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

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=tmp_path / "excluded-missing-bbox-output.pdf",
        document=document,
        report_path=tmp_path / "excluded-missing-bbox-report.json",
    )

    assert report["status"] == "complete"
    assert report["regions_retained"] == 1
    assert report["regions"][0]["status"] == "retained"
    assert report["regions"][0]["bbox"] is None


def test_failed_target_language_validation_is_reported_as_real_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "validation-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=200)
    page.insert_textbox(
        fitz.Rect(30, 40, 270, 90),
        "Los pacientes reciben tratamiento hormonal.",
        fontsize=10,
    )
    pdf.save(source)
    pdf.close()

    block = _translated_block(
        "body",
        BlockType.PARAGRAPH,
        "Los pacientes reciben tratamiento hormonal.",
        "Los pacientes reciben tratamiento hormonal.",
        BoundingBox(x0=30, y0=40, x1=270, y1=90),
        0,
    )
    block.metadata["translation_validation"] = {
        "status": "translation_failed",
        "reason": "translation_output_matches_source",
    }
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
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

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=tmp_path / "validation-output.pdf",
        document=document,
        report_path=tmp_path / "validation-report.json",
    )

    assert report["status"] == "partial"
    assert report["pages_using_fallback_behavior"] == 1
    assert report["regions_skipped"] == 1
    assert report["regions_retained"] == 0
    assert report["regions"][0]["reason"] == "translation_output_matches_source"


def test_failed_table_target_language_validation_is_reported_as_real_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "validation-table-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=200)
    page.insert_textbox(
        fitz.Rect(30, 40, 270, 90),
        "Diagnostico N Depresion 15",
        fontsize=10,
    )
    pdf.save(source)
    pdf.close()

    source_table = (
        "<table><tr><th>Diagnostico</th><th>N</th></tr>"
        "<tr><td>Depresion</td><td>15</td></tr></table>"
    )
    block = Block(
        id="failed-table",
        page_number=1,
        block_type=BlockType.TABLE,
        text=source_table,
        bbox=BoundingBox(x0=30, y0=40, x1=270, y1=90),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "source_text": source_table,
            "translated_from_block_ids": ["failed-table"],
            "translation_validation": {
                "status": "translation_failed",
                "reason": "translation_output_matches_source",
            },
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
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

    output = tmp_path / "validation-table-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "validation-table-report.json",
    )

    assert report["status"] == "partial"
    assert report["pages_using_fallback_behavior"] == 1
    assert report["regions_replaced"] == 0
    assert report["regions_skipped"] == 1
    assert report["regions"][0]["reason"] == "translation_output_matches_source"
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


def _single_page_table_document(
    block: Block, *, tables: list[TableModel] | None = None
) -> DocumentModel:
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
        cells=[[table_cell(row, column) for column in range(3)] for row in range(1, 3)],
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
        region.get("reason") == "table_cell_geometry_unreliable" for region in report["regions"]
    )

    source_image = _render_rgb(source, scale=2)
    output_image = _render_rgb(output, scale=2)
    for y in y_edges:
        rule_strip = (60, round(y * 2) - 1, 780, round(y * 2) + 2)
        assert source_image.crop(rule_strip).tobytes() == output_image.crop(rule_strip).tobytes()
    assert (
        source_image.crop((60, 400, 780, 440)).tobytes()
        == output_image.crop((60, 400, 780, 440)).tobytes()
    )


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


def test_overlapping_marker_cells_fall_back_to_ruled_colspan_grid(tmp_path: Path) -> None:
    source = tmp_path / "ruled-colspan-source.pdf"
    x_edges = [30.0, 120.0, 210.0, 300.0, 390.0]
    y_edges = [30.0, 52.0, 76.0, 112.0, 145.0, 180.0, 215.0]
    pdf = fitz.open()
    page = pdf.new_page(width=420, height=230)
    for y in y_edges:
        page.draw_line((x_edges[0], y), (x_edges[-1], y), color=(0, 0, 0), width=0.6)
    for x in (x_edges[0], x_edges[-1]):
        page.draw_line((x, y_edges[0]), (x, y_edges[-1]), color=(0, 0, 0), width=0.6)
    for x in x_edges[1:-1]:
        page.draw_line((x, y_edges[1]), (x, y_edges[3]), color=(0, 0, 0), width=0.6)
    page.draw_line((x_edges[3], y_edges[3]), (x_edges[3], y_edges[-1]), color=(0, 0, 0), width=0.6)

    page.insert_text((34, 45), "Alter", fontsize=7)
    for column, text in enumerate(("", "in Jahren", "bei Outing", "bei Diagnose")):
        if text:
            page.insert_text((x_edges[column] + 3, 69), text, fontsize=7)
    for column, text in enumerate(("M (SD)", "26.8", "22.9", "23.2")):
        page.insert_text((x_edges[column] + 3, 95), text, fontsize=7)
    page.insert_text((34, 130), "Bei Geburt zugewiesenes Geschlecht", fontsize=7)
    page.insert_text((304, 130), "10 (50)", fontsize=7)
    page.insert_text((34, 164), "Wohnort (Bundesland)", fontsize=7)
    page.insert_text((304, 164), "6 (30)", fontsize=7)
    page.insert_text((34, 195), "Geschlechtsangleichende Maßnahmen", fontsize=7)
    page.insert_text((34, 208), "abgeschlossen", fontsize=7)
    page.insert_text((304, 195), "20 (100)", fontsize=7)
    page.insert_text((304, 208), "8 (40)", fontsize=7)
    pdf.save(source)
    pdf.close()

    source_markup = (
        "<table><tr><th colspan=4>Alter</th></tr>"
        "<tr><th></th><th>in Jahren</th><th>bei Outing</th><th>bei Diagnose</th></tr>"
        "<tr><td>M (SD)</td><td>26.8</td><td>22.9</td><td>23.2</td></tr>"
        "<tr><td colspan=3>Bei Geburt zugewiesenes Geschlecht</td><td>10 (50)</td></tr>"
        "<tr><td colspan=3>Wohnort (Bundesland)</td><td>6 (30)</td></tr>"
        "<tr><td colspan=3>Geschlechtsangleichende Maßnahmen</td><td>20 (100)</td></tr>"
        "<tr><td colspan=3>abgeschlossen</td><td>8 (40)</td></tr></table>"
    )
    translated_markup = (
        "<table><tr><th colspan=4>Age</th></tr>"
        "<tr><th></th><th>in years</th><th>at outing</th><th>at diagnosis</th></tr>"
        "<tr><td>M (SD)</td><td>26.8</td><td>22.9</td><td>23.2</td></tr>"
        "<tr><td colspan=3>Assigned sex at birth</td><td>10 (50)</td></tr>"
        "<tr><td colspan=3>Residence (federal state)</td><td>6 (30)</td></tr>"
        "<tr><td colspan=3>Gender-affirming measures</td><td>20 (100)</td></tr>"
        "<tr><td colspan=3>completed</td><td>8 (40)</td></tr></table>"
    )
    block = Block(
        id="ruled-colspan-table",
        page_number=1,
        block_type=BlockType.TABLE,
        text=translated_markup,
        bbox=BoundingBox(x0=x_edges[0], y0=y_edges[0], x1=x_edges[-1], y1=y_edges[-1]),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "source_text": source_markup,
            "translated_from_block_ids": ["ruled-colspan-table"],
            "marker_page_width": 420,
            "marker_page_height": 230,
        },
    )
    parsed = OriginalLayoutReconstructor()._parse_table_rows(source_markup)
    marker_rows: list[list[TableModel.TableCell]] = []
    for row_index, row in enumerate(parsed):
        cells: list[TableModel.TableCell] = []
        next_column = 0
        for cell_index, cell in enumerate(row):
            if row_index == 0:
                bbox = BoundingBox(x0=30, y0=30, x1=390, y1=52)
            elif row_index in {1, 2}:
                bbox = BoundingBox(
                    x0=x_edges[cell_index],
                    y0=y_edges[row_index],
                    x1=x_edges[cell_index + 1],
                    y1=y_edges[row_index + 1],
                )
            else:
                row_y0 = y_edges[min(row_index, 5)]
                row_y1 = y_edges[min(row_index + 1, 6)]
                bbox = (
                    BoundingBox(x0=30, y0=row_y0, x1=330, y1=row_y1)
                    if cell_index == 0
                    else BoundingBox(x0=290, y0=row_y0, x1=390, y1=row_y1)
                )
            cells.append(
                TableModel.TableCell(
                    text=cell.text,
                    rowspan=cell.rowspan,
                    colspan=cell.colspan,
                    bbox=bbox,
                    row_index=row_index,
                    column_index=next_column,
                )
            )
            next_column += cell.colspan
        marker_rows.append(cells)
    table = TableModel(
        id="ruled-colspan-table-model",
        page_numbers=[1],
        page=1,
        bbox=block.bbox,
        headers=[cell.text for cell in parsed[0]],
        header_cells=marker_rows[0],
        rows=[[cell.text for cell in row] for row in parsed[1:]],
        cells=marker_rows[1:],
        debug={
            "marker_block_id": block.id,
            "cell_coordinate_space": {
                "name": "marker_page_coordinates",
                "width": 420,
                "height": 230,
            },
        },
    )
    output = tmp_path / "ruled-colspan-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_single_page_table_document(block, tables=[table]),
        report_path=tmp_path / "ruled-colspan-report.json",
    )

    strategies = {
        metadata.get("table_grid_detection")
        for region in report["regions"]
        for metadata in region.get("coordinate_metadata", [])
    }
    assert strategies == {"pymupdf_find_tables_lines_strict"}
    assert report["regions_skipped"] == 0
    with fitz.open(source) as original, fitz.open(output) as translated:
        text = translated[0].get_text("text")
        assert "Assigned sex at birth" in text
        assert "measures" in text
        assert "zugewiesenes Geschlecht" not in text
        assert len(translated[0].get_drawings()) == len(original[0].get_drawings())

    source_image = _render_rgb(source, scale=2)
    output_image = _render_rgb(output, scale=2)
    numeric_cells = (
        round((x_edges[3] + 2) * 2),
        round((y_edges[2] + 2) * 2),
        round((x_edges[4] - 2) * 2),
        round((y_edges[-1] - 2) * 2),
    )
    assert source_image.crop(numeric_cells).tobytes() == output_image.crop(
        numeric_cells
    ).tobytes()

    difference = ImageChops.difference(source_image, output_image)
    outside_mask = Image.new("L", difference.size, 255)
    mask_draw = ImageDraw.Draw(outside_mask)
    for region in report["regions"]:
        if region.get("status") != "replaced" or not any(
            block_id.startswith("ruled-colspan-table#")
            for block_id in region.get("block_ids", [])
        ):
            continue
        bbox = region["bbox"]
        mask_draw.rectangle(
            (
                round(bbox["x0"] * 2) - 2,
                round(bbox["y0"] * 2) - 2,
                round(bbox["x1"] * 2) + 2,
                round(bbox["y1"] * 2) + 2,
            ),
            fill=0,
        )
    outside = Image.composite(difference, Image.new("RGB", difference.size), outside_mask)
    assert outside.getbbox() is None


def test_ruled_table_grid_rejects_unexplained_extra_cell_text(tmp_path: Path) -> None:
    source = tmp_path / "contaminated-table-source.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=150)
    table_rect = fitz.Rect(30, 30, 270, 100)
    page.draw_line(table_rect.tl, table_rect.tr, color=(0, 0, 0), width=0.6)
    page.draw_line(table_rect.bl, table_rect.br, color=(0, 0, 0), width=0.6)
    page.draw_line(table_rect.tl, table_rect.bl, color=(0, 0, 0), width=0.6)
    page.draw_line(table_rect.tr, table_rect.br, color=(0, 0, 0), width=0.6)
    page.insert_text((36, 52), "Kategorie", fontsize=8)
    page.insert_text((36, 72), "DO NOT DELETE", fontsize=8)
    pdf.save(source)
    pdf.close()

    block = Block(
        id="contaminated-table",
        page_number=1,
        block_type=BlockType.TABLE,
        text="<table><tr><td>Category</td></tr></table>",
        bbox=BoundingBox(x0=30, y0=30, x1=270, y1=100),
        reading_order_index=0,
        source_type=SourceType.EMBEDDED,
        metadata={
            "source_text": "<table><tr><td>Kategorie</td></tr></table>",
            "translated_from_block_ids": ["contaminated-table"],
        },
    )
    output = tmp_path / "contaminated-table-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_single_page_table_document(block),
        report_path=tmp_path / "contaminated-table-report.json",
    )

    assert any(
        region.get("reason") == "table_cell_geometry_unreliable"
        for region in report["regions"]
    )
    assert report["regions_replaced"] == 0
    assert _render_rgb(source).tobytes() == _render_rgb(output).tobytes()
    with fitz.open(output) as translated:
        text = translated[0].get_text("text")
        assert "Kategorie" in text
        assert "DO NOT DELETE" in text
        assert "Category" not in text


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


def test_marker_cell_polygons_allow_small_shared_boundary_overlap(tmp_path: Path) -> None:
    source = tmp_path / "stored-table-fuzzy-boundaries.pdf"
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
    for row in [table.header_cells, *table.cells]:
        for cell in row:
            assert cell.bbox is not None
            cell.bbox = BoundingBox(
                x0=cell.bbox.x0 - 0.4,
                y0=cell.bbox.y0 - 0.4,
                x1=cell.bbox.x1 + 0.4,
                y1=cell.bbox.y1 + 0.4,
            )
    assert block.bbox is not None

    with fitz.open(source) as pdf:
        rows, strategy = OriginalLayoutReconstructor()._stored_table_grid_rows(
            page=pdf[0],
            block=block,
            table=table,
            source_rows=source_rows,
            table_bbox=block.bbox,
        )

    assert rows
    assert strategy == "marker_table_cell_polygons"


def test_table_row_similarity_tolerates_boundary_bleed_into_empty_cells(
    tmp_path: Path,
) -> None:
    source = tmp_path / "table-boundary-bleed.pdf"
    doc = fitz.open()
    page = doc.new_page(width=240, height=100)
    page.insert_text((12, 27), "Heading", fontsize=10)
    page.insert_text((90, 27), "Value", fontsize=10)
    page.insert_text((12, 43), "Neighbour", fontsize=10)
    page.insert_text((90, 43), "42", fontsize=10)
    doc.save(source)
    doc.close()
    rows = OriginalLayoutReconstructor()._parse_table_rows(
        "<table><tr><td>Heading</td><td>Value</td><td></td></tr></table>"
    )

    with fitz.open(source) as pdf:
        score = OriginalLayoutReconstructor()._table_row_similarity(
            pdf[0],
            rows[0],
            [
                fitz.Rect(10, 15, 88, 46),
                fitz.Rect(88, 15, 150, 46),
                fitz.Rect(150, 15, 230, 46),
            ],
        )

    assert score >= 0.68


def test_table_row_similarity_rejects_short_header_substring_in_wrong_cell(
    tmp_path: Path,
) -> None:
    source = tmp_path / "table-short-header-mismatch.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=100)
    page.insert_text(
        (12, 27),
        "Completely unrelated neighbouring paragraph",
        fontsize=10,
    )
    doc.save(source)
    doc.close()
    rows = OriginalLayoutReconstructor()._parse_table_rows(
        "<table><tr><td>N</td></tr></table>"
    )

    reconstructor = OriginalLayoutReconstructor()
    with fitz.open(source) as pdf:
        score = reconstructor._table_row_similarity(
            pdf[0],
            rows[0],
            [fitz.Rect(10, 15, 290, 40)],
        )

    assert score == 0.0
    assert reconstructor._table_cell_text_similarity("men", "Women") == 0.0


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
        region.get("reason") == "table_cell_geometry_unreliable" for region in report["regions"]
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


def _missing_bbox_scan_document(
    filename: str,
    *,
    source_text: str,
    translated_text: str,
    page_width: float = 360,
    page_height: float = 180,
) -> DocumentModel:
    block = Block(
        id="missing-scan-bbox",
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text=translated_text,
        bbox=None,
        reading_order_index=0,
        source_type=SourceType.OCR,
        metadata={
            "source_text": source_text,
            "translated_from_block_ids": ["missing-scan-bbox"],
            "bbox_source": "qwen_wrapper_only",
        },
    )
    return DocumentModel(
        metadata=DocumentMetadata(filename=filename, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=page_width,
                height=page_height,
                has_embedded_text=True,
                embedded_text_quality=0.1,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[block],
    )


def test_hidden_ocr_missing_bbox_is_recovered_by_unique_global_alignment(
    tmp_path: Path,
) -> None:
    source_text = "Los pacientes reciben terapia durante esta fase"
    translated_text = "Patients receive therapy during this phase"
    source = tmp_path / "missing-bbox-hidden-ocr.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=360, height=180)
    page.insert_textbox(
        fitz.Rect(35, 40, 325, 68),
        source_text,
        fontsize=9,
        render_mode=3,
    )
    pdf.save(source)
    pdf.close()

    output = tmp_path / "missing-bbox-hidden-ocr-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_missing_bbox_scan_document(
            source.name,
            source_text=source_text,
            translated_text=translated_text,
        ),
        report_path=tmp_path / "missing-bbox-hidden-ocr-report.json",
    )

    assert report["status"] == "complete"
    assert report["regions_replaced"] == 1
    assert report["regions_missing_or_invalid_bboxes"] == 0
    assert report["scan_text_regions_aligned"] == 1
    replaced = next(region for region in report["regions"] if region["status"] == "replaced")
    assert replaced["bbox"] is not None
    assert replaced["source_text_masks"]
    conversion = replaced["coordinate_metadata"][0]
    assert conversion["reason"] == "missing_bbox"
    alignment = replaced["coordinate_metadata"][-1]
    assert alignment["geometry_source"] == "global_hidden_ocr_alignment_recovered_bbox"
    assert alignment["bbox_recovered_by_global_hidden_ocr_alignment"] is True
    assert alignment["preferred_search_extent_pdf"] == {
        "x0": 0.0,
        "y0": 0.0,
        "x1": 360.0,
        "y1": 180.0,
    }
    with fitz.open(output) as translated:
        assert translated_text in translated[0].get_text("text")
        assert source_text not in translated[0].get_text("text")


def test_hidden_ocr_missing_bbox_ambiguous_global_alignment_is_safely_skipped(
    tmp_path: Path,
) -> None:
    source_text = "Los pacientes reciben terapia durante esta fase"
    source = tmp_path / "ambiguous-missing-bbox-hidden-ocr.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=360, height=180)
    for rectangle in (fitz.Rect(35, 30, 325, 58), fitz.Rect(35, 105, 325, 133)):
        page.insert_textbox(
            rectangle,
            source_text,
            fontsize=9,
            render_mode=3,
        )
    pdf.save(source)
    pdf.close()

    output = tmp_path / "ambiguous-missing-bbox-hidden-ocr-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_missing_bbox_scan_document(
            source.name,
            source_text=source_text,
            translated_text="Patients receive therapy during this phase",
        ),
        report_path=tmp_path / "ambiguous-missing-bbox-hidden-ocr-report.json",
    )

    assert report["status"] == "partial"
    assert report["regions_replaced"] == 0
    assert report["regions_skipped"] == 1
    assert report["regions_missing_or_invalid_bboxes"] == 1
    assert report["scan_text_regions_alignment_failed"] == 1
    skipped = report["regions"][0]
    assert skipped["reason"] == "hidden_ocr_text_alignment_ambiguous"
    assert skipped["bbox"] is None
    assert (
        skipped["alignment_diagnostics"]["geometry_source"]
        == "global_hidden_ocr_alignment_recovered_bbox"
    )
    assert _render_rgb(source).tobytes() == _render_rgb(output).tobytes()


def test_hidden_ocr_missing_bbox_no_match_is_safely_skipped(tmp_path: Path) -> None:
    source = tmp_path / "unmatched-missing-bbox-hidden-ocr.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=360, height=180)
    page.insert_textbox(
        fitz.Rect(35, 40, 325, 68),
        "Contenido completamente diferente sin correspondencia alguna",
        fontsize=9,
        render_mode=3,
    )
    pdf.save(source)
    pdf.close()

    output = tmp_path / "unmatched-missing-bbox-hidden-ocr-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_missing_bbox_scan_document(
            source.name,
            source_text="Los pacientes reciben terapia durante esta fase",
            translated_text="Patients receive therapy during this phase",
        ),
        report_path=tmp_path / "unmatched-missing-bbox-hidden-ocr-report.json",
    )

    assert report["regions_replaced"] == 0
    assert report["regions_skipped"] == 1
    assert report["regions_missing_or_invalid_bboxes"] == 1
    skipped = report["regions"][0]
    assert skipped["reason"] in {
        "hidden_ocr_text_alignment_low_confidence",
        "hidden_ocr_text_alignment_no_candidate",
    }
    assert skipped["bbox"] is None
    assert _render_rgb(source).tobytes() == _render_rgb(output).tobytes()


def test_hidden_ocr_missing_bbox_without_hidden_text_retains_page_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing-bbox-no-hidden-text.pdf"
    pdf = fitz.open()
    pdf.new_page(width=360, height=180)
    pdf.save(source)
    pdf.close()

    output = tmp_path / "missing-bbox-no-hidden-text-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_missing_bbox_scan_document(
            source.name,
            source_text="Los pacientes reciben terapia durante esta fase",
            translated_text="Patients receive therapy during this phase",
        ),
        report_path=tmp_path / "missing-bbox-no-hidden-text-report.json",
    )

    assert report["status"] == "partial"
    assert report["pages_using_fallback_behavior"] == 1
    assert report["scan_overlay_pages"] == 0
    assert report["regions_replaced"] == 0
    assert report["regions"] == []
    assert _render_rgb(source).tobytes() == _render_rgb(output).tobytes()


def _create_surya2_image_only_scan(path: Path) -> tuple[BoundingBox, BoundingBox]:
    page_width = 360
    page_height = 180
    scale = 2
    image = Image.new("RGB", (page_width * scale, page_height * scale), "white")
    drawing = ImageDraw.Draw(image)
    drawing.text(
        (70, 65),
        "Contenido original para traducir",
        fill=(15, 15, 15),
    )
    drawing.rectangle((460, 50, 660, 190), fill=(32, 92, 176))
    drawing.line((480, 165, 635, 80), fill=(245, 210, 45), width=7)
    drawing.text((505, 105), "VISUAL", fill="white")
    image_path = path.with_suffix(".png")
    image.save(image_path)

    pdf = fitz.open()
    page = pdf.new_page(width=page_width, height=page_height)
    page.insert_image(page.rect, filename=str(image_path))
    pdf.save(path)
    pdf.close()
    return (
        BoundingBox(x0=30, y0=25, x1=220, y1=62),
        BoundingBox(x0=230, y0=25, x1=330, y1=95),
    )


def _surya2_image_scan_document(
    filename: str,
    *,
    text_bbox: BoundingBox,
    figure_bbox: BoundingBox,
) -> DocumentModel:
    source_text = "Contenido original para traducir"
    text_block = Block(
        id="surya2-text",
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text="Translated source content",
        bbox=text_bbox,
        reading_order_index=0,
        source_type=SourceType.OCR,
        style_hints={"font_size": 10},
        metadata={
            "parser": "surya2_llamacpp",
            "ocr_engine": "surya2_llamacpp",
            "source_text": source_text,
            "translated_from_block_ids": ["surya2-text"],
            "surya_page_width": 360,
            "surya_page_height": 180,
            "coordinate_space": "pdf_points_top_left",
        },
    )
    figure_block = Block(
        id="surya2-figure",
        page_number=1,
        block_type=BlockType.FIGURE,
        text="",
        bbox=figure_bbox,
        reading_order_index=1,
        source_type=SourceType.OCR,
        skipped=True,
        metadata={
            "parser": "surya2_llamacpp",
            "ocr_engine": "surya2_llamacpp",
            "surya_page_width": 360,
            "surya_page_height": 180,
            "coordinate_space": "pdf_points_top_left",
        },
    )
    return DocumentModel(
        metadata=DocumentMetadata(filename=filename, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=360,
                height=180,
                has_embedded_text=False,
                embedded_text_quality=0,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[text_block, figure_block],
        figures=[
            FigureAsset(
                id="surya2-figure-asset",
                page_number=1,
                bbox=figure_bbox,
                source_block_ids=["surya2-figure"],
            )
        ],
    )


def test_surya2_image_only_scan_uses_raster_text_masks_and_preserves_visuals(
    tmp_path: Path,
) -> None:
    source = tmp_path / "surya2-image-only.pdf"
    output = tmp_path / "surya2-image-only-translated.pdf"
    text_bbox, figure_bbox = _create_surya2_image_only_scan(source)

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=_surya2_image_scan_document(
            source.name,
            text_bbox=text_bbox,
            figure_bbox=figure_bbox,
        ),
        report_path=tmp_path / "surya2-image-only-report.json",
    )

    assert report["status"] == "complete"
    assert report["pages"][0]["reconstruction_strategy"] == "surya2_image_overlay"
    assert report["surya2_image_overlay_pages"] == 1
    assert report["surya2_image_text_masks"] >= 1
    replaced = next(region for region in report["regions"] if region["status"] == "replaced")
    assert replaced["reconstruction_strategy"] == "surya2_image_text_overlay"
    assert replaced["bbox"] == text_bbox.model_dump()
    assert replaced["source_text_masks"]
    assert replaced["coordinate_metadata"][0]["scale_x"] == 1
    assert replaced["coordinate_metadata"][0]["scale_y"] == 1
    raster_metadata = replaced["coordinate_metadata"][-1]
    assert raster_metadata["geometry_source"] == "surya2_pdf_bbox"
    assert raster_metadata["mask_source"] == "surya2_raster_foreground_rows"

    with fitz.open(output) as translated:
        assert "Translated source content" in translated[0].get_text("text")

    source_image = _render_rgb(source)
    output_image = _render_rgb(output)
    figure_pixels = (
        round(figure_bbox.x0 * 2),
        round(figure_bbox.y0 * 2),
        round(figure_bbox.x1 * 2),
        round(figure_bbox.y1 * 2),
    )
    assert ImageChops.difference(
        source_image.crop(figure_pixels),
        output_image.crop(figure_pixels),
    ).getbbox() is None


def test_surya2_image_only_scan_rejects_nonuniform_visual_text_region(
    tmp_path: Path,
) -> None:
    source = tmp_path / "surya2-unsafe-image-only.pdf"
    output = tmp_path / "surya2-unsafe-image-only-output.pdf"
    _text_bbox, figure_bbox = _create_surya2_image_only_scan(source)
    document = _surya2_image_scan_document(
        source.name,
        text_bbox=figure_bbox,
        figure_bbox=BoundingBox(x0=335, y0=120, x1=355, y1=175),
    )

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "surya2-unsafe-image-only-report.json",
    )

    assert report["status"] == "partial"
    assert report["regions_replaced"] == 0
    skipped = next(region for region in report["regions"] if region["status"] == "skipped")
    assert skipped["reason"] == "surya2_image_scan_background_not_uniform_or_light"
    assert _render_rgb(source).tobytes() == _render_rgb(output).tobytes()


def test_surya2_image_only_scan_retains_table_without_cell_geometry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "surya2-image-only-table.pdf"
    output = tmp_path / "surya2-image-only-table-output.pdf"
    text_bbox, figure_bbox = _create_surya2_image_only_scan(source)
    document = _surya2_image_scan_document(
        source.name,
        text_bbox=text_bbox,
        figure_bbox=figure_bbox,
    )
    table_block = Block(
        id="surya2-table",
        page_number=1,
        block_type=BlockType.TABLE,
        text="<table><tr><td>Translated</td></tr></table>",
        bbox=BoundingBox(x0=30, y0=105, x1=210, y1=160),
        reading_order_index=2,
        source_type=SourceType.OCR,
        metadata={
            "parser": "surya2_llamacpp",
            "ocr_engine": "surya2_llamacpp",
            "source_text": "<table><tr><td>Origen</td></tr></table>",
            "translated_from_block_ids": ["surya2-table"],
            "surya_page_width": 360,
            "surya_page_height": 180,
        },
    )
    document.blocks.append(table_block)
    document.tables.append(
        TableModel(
            id="surya2-table-model",
            page_numbers=[1],
            page=1,
            bbox=table_block.bbox,
            rows=[["Origen"]],
            debug={"source_block_id": table_block.id},
        )
    )

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "surya2-image-only-table-report.json",
    )

    assert report["status"] == "partial"
    assert report["regions_replaced"] == 1
    skipped_table = next(
        region
        for region in report["regions"]
        if region["block_ids"] == ["surya2-table"]
    )
    assert skipped_table["reason"] == "surya2_image_scan_table_requires_cell_geometry"


def test_hidden_ocr_multiple_tables_fall_back_as_atomic_page_group(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hidden-ocr-table-group.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=320, height=260)
    x_edges = [30, 160, 290]
    table_specs = (
        (
            "scan-table-a",
            [30, 60, 90],
            (("Uno", "Dos"), ("Tres", "Cuatro")),
            "| Uno | Dos |\n|---|---|\n| Tres | Cuatro |",
            "| One | Two |\n|---|---|\n| Three | Four |",
        ),
        (
            "scan-table-b",
            [120, 150, 180],
            (("Cinco", "Seis"), ("Siete", "Ocho")),
            "| Cinco | Seis |\n|---|---|\n| Siete | Ocho |",
            "| Five | Six |\n|---|---|\n| Seven | Eight |",
        ),
    )
    blocks: list[Block] = []
    for order, (block_id, y_edges, cells, source_markup, translated_markup) in enumerate(
        table_specs
    ):
        for x in x_edges:
            page.draw_line((x, y_edges[0]), (x, y_edges[-1]), color=(0, 0, 0), width=0.6)
        for y in y_edges:
            page.draw_line((x_edges[0], y), (x_edges[-1], y), color=(0, 0, 0), width=0.6)
        for row_index, row in enumerate(cells):
            for column_index, text in enumerate(row):
                page.insert_textbox(
                    fitz.Rect(
                        x_edges[column_index] + 4,
                        y_edges[row_index] + 4,
                        x_edges[column_index + 1] - 4,
                        y_edges[row_index + 1] - 4,
                    ),
                    text,
                    fontsize=8,
                    align=fitz.TEXT_ALIGN_CENTER,
                    render_mode=3,
                )
        blocks.append(
            Block(
                id=block_id,
                page_number=1,
                block_type=BlockType.TABLE,
                text=translated_markup,
                bbox=BoundingBox(
                    x0=x_edges[0],
                    y0=y_edges[0],
                    x1=x_edges[-1],
                    y1=y_edges[-1],
                ),
                reading_order_index=order,
                source_type=SourceType.OCR,
                metadata={
                    "source_text": source_markup.replace("\n", " "),
                    "source_text_before_cleaning": source_markup,
                    "translated_from_block_ids": [block_id],
                },
            )
        )

    body_source = "Texto del cuerpo fuera de las dos tablas"
    body_translation = "Body text outside both tables"
    page.insert_textbox(
        fitz.Rect(30, 205, 290, 235),
        body_source,
        fontsize=9,
        render_mode=3,
    )
    blocks.append(
        Block(
            id="scan-body",
            page_number=1,
            block_type=BlockType.PARAGRAPH,
            text=body_translation,
            bbox=BoundingBox(x0=30, y0=205, x1=290, y1=235),
            reading_order_index=2,
            source_type=SourceType.OCR,
            metadata={
                "source_text": body_source,
                "translated_from_block_ids": ["scan-body"],
            },
        )
    )
    pdf.save(source)
    pdf.close()

    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=320,
                height=260,
                has_embedded_text=True,
                embedded_text_quality=0.1,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=blocks,
    )
    reconstructor = OriginalLayoutReconstructor()
    reconstructor._preflight = (  # type: ignore[method-assign]
        lambda **kwargs: (
            (-1.0, 0.5)
            if kwargs["region"].block_ids[0].startswith("scan-table-b#")
            else (1.0, 1.0)
        )
    )
    output = tmp_path / "hidden-ocr-table-group-output.pdf"
    report = reconstructor.reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "hidden-ocr-table-group-report.json",
    )

    assert report["status"] == "partial"
    assert report["regions_replaced"] == 1
    assert report["raster_tables_reconstructed"] == 0
    assert any(
        region.get("block_ids") == ["scan-table-b"]
        and region.get("reason") == "table_atomic_reconstruction_overflow"
        for region in report["regions"]
    )
    retained_sibling = next(
        region
        for region in report["regions"]
        if region.get("reason") == "scan_table_group_retained_after_sibling_failure"
    )
    assert retained_sibling["block_ids"] == ["scan-table-a"]
    assert retained_sibling["bbox"] == {
        "x0": 30.0,
        "y0": 30.0,
        "x1": 290.0,
        "y1": 90.0,
    }
    group_warning = next(
        warning
        for warning in report["warnings"]
        if warning["code"] == "scan_table_group_atomic_fallback"
    )
    assert "scan-table-b" in group_warning["reason"]
    assert "scan-table-a" in group_warning["reason"]
    with fitz.open(output) as translated:
        output_text = translated[0].get_text("text")
        assert body_translation in output_text
        assert body_source not in output_text
        assert "Uno" in output_text
        assert "One" not in output_text

    source_image = _render_rgb(source, scale=2)
    output_image = _render_rgb(output, scale=2)
    assert source_image.crop((0, 0, 640, 390)).tobytes() == output_image.crop(
        (0, 0, 640, 390)
    ).tobytes()


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

    source_table = "| Diagnostico | N |\n|---|---|\n| Depresion | 15 |"
    translated_table = "| Diagnosis | N | |---|---| | Depression | 15 |"
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
    untranslated_body = Block(
        id="untranslated-scan-body",
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text="Texto corporal sin traduccion confirmada",
        bbox=BoundingBox(x0=30, y0=125, x1=200, y1=145),
        reading_order_index=2,
        source_type=SourceType.OCR,
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
        blocks=[table, caption, untranslated_body],
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
    assert report["pages"][0]["status"] == "partial"
    assert any(
        region.get("block_ids") == ["untranslated-scan-body"]
        and region.get("reason") == "scan_table_only_non_table_translation_unavailable"
        for region in report["regions"]
    )

    source_image = _render_rgb(source, scale=2)
    output_image = _render_rgb(output, scale=2)
    figure_box = (440, 310, 590, 450)
    assert source_image.crop(figure_box).tobytes() == output_image.crop(figure_box).tobytes()
    for coordinate in ((60, 70), (300, 70), (580, 70), (60, 130), (60, 190)):
        assert source_image.getpixel(coordinate) == output_image.getpixel(coordinate)

    approved = Image.new("L", source_image.size, 0)
    approved_draw = ImageDraw.Draw(approved)
    for region in report["regions"]:
        if region.get("status") != "replaced":
            continue
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


def test_redaction_guard_preserves_figure_when_caption_box_touches_boundary(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "translated.pdf"
    _create_original_layout_source(source)
    document = _original_layout_document()
    caption = next(block for block in document.blocks if block.id == "caption")
    caption.bbox = BoundingBox(x0=60, y0=280, x1=340, y1=325)

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "report.json",
    )

    caption_region = next(
        region for region in report["regions"] if region.get("block_ids") == ["caption"]
    )
    assert all(mask["y0"] >= 281 for mask in caption_region["applied_redaction_bboxes"])
    source_image = _render_rgb(source, scale=3)
    output_image = _render_rgb(output, scale=3)
    figure_box = (180, 300, 1020, 840)
    assert source_image.crop(figure_box).tobytes() == output_image.crop(figure_box).tobytes()


def test_invalid_figure_bbox_is_reported_and_not_counted_as_preserved(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    pdf = fitz.open()
    pdf.new_page(width=100, height=100)
    pdf.save(source)
    pdf.close()
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=100,
                height=100,
                has_embedded_text=False,
                embedded_text_quality=0,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[],
        figures=[
            FigureAsset(
                id="outside-figure",
                page_number=1,
                bbox=BoundingBox(x0=90, y0=10, x1=120, y1=50),
            )
        ],
    )

    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=tmp_path / "output.pdf",
        document=document,
        report_path=tmp_path / "report.json",
    )

    assert report["figures_preserved"] == 0
    assert report["regions_missing_or_invalid_bboxes"] == 1
    assert any(
        warning["code"] == "figure_lock_region_invalid" for warning in report["warnings"]
    )


def test_source_character_count_uses_visible_table_text() -> None:
    reconstructor = OriginalLayoutReconstructor()

    assert reconstructor._source_character_count(
        '<table class="data"><tr><th>Edad media</th><td>32</td></tr></table>'
    ) == len("Edad media 32")


def test_hidden_ocr_body_alignment_corrects_shifted_surya_bbox(tmp_path: Path) -> None:
    page_width, page_height = 360, 240
    scan_path = tmp_path / "shifted-scan.png"
    scan = Image.new("RGB", (page_width, page_height), "white")
    drawing = ImageDraw.Draw(scan)
    drawing.text((30, 43), "Texto fuente correcto", fill="black")
    drawing.text((30, 64), "segunda linea completa", fill="black")
    drawing.text((30, 143), "REGION NO RELACIONADA", fill="black")
    drawing.rectangle((260, 35, 335, 105), fill=(30, 100, 190))
    scan.save(scan_path)

    source = tmp_path / "shifted-hidden-ocr.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=page_width, height=page_height)
    page.insert_image(page.rect, filename=str(scan_path))
    page.insert_textbox(
        fitz.Rect(28, 35, 205, 57),
        "Texto fuente correcto",
        fontsize=10,
        render_mode=3,
    )
    page.insert_textbox(
        fitz.Rect(28, 58, 205, 80),
        "segunda linea completa",
        fontsize=10,
        render_mode=3,
    )
    page.insert_textbox(
        fitz.Rect(28, 135, 220, 162),
        "REGION NO RELACIONADA",
        fontsize=10,
        render_mode=3,
    )
    pdf.save(source)
    pdf.close()

    block = Block(
        id="shifted-body",
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text="Correct translated text on both lines",
        # Simulate a reading-order reconciliation error: this stored Surya
        # box points to the unrelated second line.
        bbox=BoundingBox(x0=28, y0=135, x1=220, y1=162),
        reading_order_index=0,
        source_type=SourceType.OCR,
        metadata={
            "source_text": "Texto fuente correcto segunda linea completa",
            "translated_from_block_ids": ["shifted-body"],
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
                embedded_text_quality=0.2,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[block],
        figures=[
            FigureAsset(
                id="locked-scan-figure",
                page_number=1,
                bbox=BoundingBox(x0=260, y0=35, x1=335, y1=105),
            )
        ],
    )
    output = tmp_path / "shifted-hidden-ocr-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "shifted-hidden-ocr-report.json",
    )

    assert report["regions_replaced"] == 1
    assert report["scan_text_regions_aligned"] == 1
    assert report["pages"][0]["reconstruction_strategy"] == "ocr_text_overlay"
    replaced = next(region for region in report["regions"] if region["status"] == "replaced")
    assert replaced["bbox"]["y0"] < 70
    assert replaced["source_character_count"] == len("Texto fuente correcto segunda linea completa")
    assert replaced["source_text_mask_count"] == len(replaced["source_text_masks"])
    assert replaced["applied_redaction_bboxes"] == replaced["source_text_masks"]
    alignment = replaced["coordinate_metadata"][-1]
    assert alignment["geometry_source"] == "hidden_ocr_contiguous_line_alignment"
    assert alignment["surya_region_bbox_pdf"]["y0"] > 120
    assert alignment["matched_hidden_ocr_bbox_pdf"]["y0"] < 70

    with fitz.open(output) as translated:
        page = translated[0]
        assert "Correct translated text" in page.get_text("text", clip=fitz.Rect(20, 25, 230, 95))
        assert "REGION NO RELACIONADA" in page.get_text("text")
        assert "Texto fuente correcto" not in page.get_text("text")
        assert "segunda linea completa" not in page.get_text("text")
    source_image = _render_rgb(source, scale=2)
    output_image = _render_rgb(output, scale=2)
    figure_box = (520, 70, 670, 210)
    assert source_image.crop(figure_box).tobytes() == output_image.crop(figure_box).tobytes()


def test_hidden_ocr_body_does_not_treat_clipped_neighbor_glyphs_as_columns() -> None:
    reconstructor = OriginalLayoutReconstructor()
    lines = tuple(
        _HiddenOCRLine(
            block_index=10,
            line_index=index,
            bbox=BoundingBox(x0=300, y0=40 + index * 14, x1=500, y1=52 + index * 14),
            text=f"ordinary paragraph line {index}",
        )
        for index in range(6)
    )
    words: list[tuple[float, float, float, float, str, int, int, int]] = []
    for index, line in enumerate(lines):
        y0, y1 = line.bbox.y0, line.bbox.y1
        for word_index, (x0, x1, text) in enumerate(
            (
                (300.0, 356.0, "ordinary"),
                (385.0, 420.0, "paragraph"),
                (425.0, 459.0, "line"),
                (464.0, 481.0, str(index)),
            )
        ):
            words.append((x0, y0, x1, y1, text, 10, index, word_index))
    # Reproduce malformed clipped-word output from a hidden-OCR scan. Tiny
    # pieces of descenders from the preceding OCR line intersect the next
    # line's rectangle. Once sorted by x, those nested fragments manufacture
    # gaps that are not present in the matched line itself.
    for line_index in (2, 3):
        line = lines[line_index]
        decoy_y0 = line.bbox.y0 - 12
        decoy_y1 = line.bbox.y0 + 1
        for word_index, x0 in enumerate((321.0, 386.0, 464.0)):
            words.append((x0, decoy_y0, x0 + 5, decoy_y1, "q", 99, line_index, word_index))

    class _ClippedWordPage:
        def get_text(self, kind: str, *, clip=None, sort=False):
            assert kind == "words"
            if clip is None:
                return words
            return [word for word in words if fitz.Rect(*word[:4]).intersects(fitz.Rect(clip))]

    assert not reconstructor._scan_match_has_multicolumn_text(_ClippedWordPage(), lines)


def test_hidden_ocr_body_requires_a_stable_gutter_for_multiple_columns() -> None:
    reconstructor = OriginalLayoutReconstructor()
    lines = tuple(
        _HiddenOCRLine(
            block_index=10,
            line_index=index,
            bbox=BoundingBox(x0=40, y0=40 + index * 14, x1=300, y1=52 + index * 14),
            text=f"left {index} right {index}",
        )
        for index in range(4)
    )
    words = [
        word
        for index, line in enumerate(lines)
        for word in (
            (40.0, line.bbox.y0, 90.0, line.bbox.y1, "left", 10, index, 0),
            (220.0, line.bbox.y0, 270.0, line.bbox.y1, "right", 10, index, 1),
        )
    ]

    class _TwoColumnPage:
        def get_text(self, kind: str, *, clip=None, sort=False):
            assert kind == "words"
            return words

    assert reconstructor._scan_match_has_multicolumn_text(_TwoColumnPage(), lines)


def test_hidden_ocr_alignment_joins_spatial_continuation_from_malformed_block() -> None:
    reconstructor = OriginalLayoutReconstructor()

    def line(
        block_index: int,
        line_index: int,
        bbox: tuple[float, float, float, float],
        text: str,
    ) -> _HiddenOCRLine:
        return _HiddenOCRLine(
            block_index=block_index,
            line_index=line_index,
            bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
            text=text,
        )

    paragraph = (
        line(14, 0, (308, 741, 549, 756), "Todos los pacientes reciben documentos"),
        line(14, 1, (300, 755, 549, 770), "sobre los cambios corporales"),
        line(14, 2, (299, 768, 546, 783), "y efectos que se deriven del tra-"),
    )
    malformed_block = tuple(
        line(
            15,
            index,
            (38, 580 + index * 13, 288, 595 + index * 13),
            f"unrelated left-column line {index}",
        )
        for index in range(16)
    ) + (
        line(15, 16, (300, 781, 395, 796), "tamiento hormonal."),
        line(15, 17, (557, 786, 573, 796), "footer"),
    )

    sequences = reconstructor._hidden_ocr_text_sequences([paragraph, malformed_block])
    expected = (
        "Todos los pacientes reciben documentos sobre los cambios corporales "
        "y efectos que se deriven del tratamiento hormonal."
    )
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    try:
        match, metadata = reconstructor._match_hidden_ocr_lines(
            page,
            expected,
            preferred_bbox=BoundingBox(x0=299, y0=741, x1=549, y1=796),
            unavailable_line_keys=set(),
            text_sequences=sequences,
        )
    finally:
        pdf.close()

    assert metadata["reason"] == "matched"
    assert match is not None
    assert match.lines == (*paragraph, malformed_block[-2])
    assert "unrelated left-column" not in match.text


def test_hidden_ocr_partial_line_match_is_retained_as_fallback(tmp_path: Path) -> None:
    source = tmp_path / "partial-hidden-ocr.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=320, height=150)
    page.insert_textbox(
        fitz.Rect(30, 35, 280, 60),
        "Comienzo del texto fuente incompleto",
        fontsize=9,
        render_mode=3,
    )
    pdf.save(source)
    pdf.close()
    block = Block(
        id="partial-source",
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text="Translated complete paragraph",
        bbox=BoundingBox(x0=30, y0=35, x1=280, y1=60),
        reading_order_index=0,
        source_type=SourceType.OCR,
        metadata={
            "source_text": "Comienzo del texto fuente incompleto seguido por una frase final que no existe",
            "translated_from_block_ids": ["partial-source"],
        },
    )
    document = DocumentModel(
        metadata=DocumentMetadata(filename=source.name, page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=320,
                height=150,
                has_embedded_text=True,
                embedded_text_quality=0.1,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[block],
    )
    output = tmp_path / "partial-hidden-ocr-output.pdf"
    report = OriginalLayoutReconstructor().reconstruct(
        source_pdf_path=source,
        output_pdf_path=output,
        document=document,
        report_path=tmp_path / "partial-hidden-ocr-report.json",
    )

    assert report["regions_replaced"] == 0
    assert report["scan_text_regions_alignment_failed"] == 1
    skipped = next(
        region
        for region in report["regions"]
        if region.get("reason") == "hidden_ocr_text_alignment_low_confidence"
    )
    diagnostics = skipped["alignment_diagnostics"]
    assert diagnostics["reason"] == "hidden_ocr_text_alignment_low_confidence"
    assert diagnostics["score"] < diagnostics["minimum_score"]
    assert diagnostics["length_coverage"] < 1.0
    assert "prefix_score" in diagnostics
    assert "suffix_score" in diagnostics
    assert (
        ImageChops.difference(
            _render_rgb(source, scale=2),
            _render_rgb(output, scale=2),
        ).getbbox()
        is None
    )


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

    source_table = "| Source header words | |\n|---|---|\n| Second source label | value here |"
    translated_table = (
        "| Source header words | Invented text |\n|---|---|\n| Second source label | value here |"
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
        region.get("reason") == "caption_hidden_ocr_text_mismatch" for region in report["regions"]
    )
    assert (
        ImageChops.difference(
            _render_rgb(source, scale=2),
            _render_rgb(output, scale=2),
        ).getbbox()
        is None
    )


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
    assert (
        ImageChops.difference(
            _render_rgb(source, scale=2),
            _render_rgb(output, scale=2),
        ).getbbox()
        is None
    )

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
    assert all(entry["status"] == "rolled_back" for entry in postflight_report["scaling_applied"])
    assert (
        ImageChops.difference(
            _render_rgb(source, scale=2),
            _render_rgb(postflight_output, scale=2),
        ).getbbox()
        is None
    )


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
