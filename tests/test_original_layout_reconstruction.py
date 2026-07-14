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
)
from app.services.original_layout_reconstructor import OriginalLayoutReconstructor


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
