from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from app.models.schema import (
    Block,
    BlockType,
    BoundingBox,
    DocumentMetadata,
    DocumentModel,
    FigureAsset,
    FigureAssetType,
    PageMetadata,
    SourceType,
)
from app.services.figure_extractor import FigureExtractionService
from app.services.markdown_builder import MarkdownBuilder
from app.services.ocr_to_translation_parser import OCRToTranslationParser
from app.services.reconstructor import Reconstructor
from app.services.translator_mlx import MlxTranslator, TranslationSettings


def _create_vector_figure_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=400, height=500)
    page.insert_textbox(
        fitz.Rect(40, 35, 360, 65),
        "Siehe Abbildung 1 für die Ergebnisse.",
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
        "Figure 1. Quellbeschriftung",
        fontsize=10,
    )
    document.save(path)
    document.close()


def _vector_document() -> DocumentModel:
    surya_space = {
        "parser": "qwen_surya_full_page_ocr",
        "surya_region_type": "Figure",
        "surya_page_width": 800,
        "surya_page_height": 1000,
        "source_region_ids": ["page_0001-r002"],
    }
    blocks = [
        Block(
            id="mention",
            page_number=1,
            block_type=BlockType.PARAGRAPH,
            text="Siehe Abbildung 1 für die Ergebnisse.",
            bbox=BoundingBox(x0=40, y0=35, x1=360, y1=65),
            reading_order_index=0,
            source_type=SourceType.EMBEDDED,
        ),
        Block(
            id="figure-region",
            page_number=1,
            block_type=BlockType.FIGURE,
            text="ACHSE QUELLE",
            bbox=BoundingBox(x0=120, y0=200, x1=680, y1=560),
            confidence=0.94,
            reading_order_index=1,
            source_type=SourceType.OCR,
            metadata=surya_space,
        ),
        Block(
            id="caption",
            page_number=1,
            block_type=BlockType.CAPTION,
            text="Figure 1. Quellbeschriftung",
            bbox=BoundingBox(x0=120, y0=580, x1=680, y1=650),
            reading_order_index=2,
            source_type=SourceType.OCR,
            metadata={
                "surya_page_width": 800,
                "surya_page_height": 1000,
                "surya_region_type": "Caption",
            },
        ),
    ]
    return DocumentModel(
        metadata=DocumentMetadata(
            filename="vector.pdf",
            page_count=1,
            detected_language="de",
        ),
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
                id="legacy-figure",
                page_number=1,
                bbox=BoundingBox(x0=60, y0=100, x1=340, y1=280),
                source_block_ids=["figure-region"],
            )
        ],
    )


def test_figure_schema_remains_backward_compatible() -> None:
    figure = FigureAsset.model_validate(
        {
            "id": "legacy",
            "page_number": 2,
            "bbox": None,
            "caption_block_id": None,
            "image_path": None,
        }
    )

    assert figure.asset_type == FigureAssetType.UNKNOWN
    assert figure.vector_path is None
    assert figure.source_block_ids == []
    assert figure.extraction_metadata == {}


def test_figure_extraction_populates_assets_coordinates_and_caption(tmp_path: Path) -> None:
    pdf_path = tmp_path / "vector.pdf"
    _create_vector_figure_pdf(pdf_path)

    populated = FigureExtractionService().extract(
        pdf_path=pdf_path,
        document=_vector_document(),
        artifact_dir=tmp_path / "artifacts" / "figures",
    )

    assert len(populated.figures) == 1
    figure = populated.figures[0]
    assert figure.asset_type == FigureAssetType.VECTOR
    assert figure.caption_block_id == "caption"
    assert figure.detection_confidence == 0.94
    assert figure.has_internal_text is True
    assert figure.original_width == 280
    assert figure.original_height == 180
    assert figure.aspect_ratio == 280 / 180
    assert figure.bbox == BoundingBox(x0=60, y0=100, x1=340, y1=280)
    assert Path(str(figure.image_path)).is_file()
    assert Path(str(figure.vector_path)).is_file()
    assert Path(str(figure.image_path)).name == "figure-p0001-001.png"
    assert Path(str(figure.vector_path)).name == "figure-p0001-001.svg"
    conversion = figure.extraction_metadata["coordinate_conversion"]
    assert conversion["source_space"] == "surya_rendered_pixels"
    assert conversion["scale_x"] == 0.5
    assert conversion["scale_y"] == 0.5
    assert figure.extraction_metadata["caption_association"]["confidence"] >= 0.55
    assert populated.metadata.translation["figure_extraction"]["vector_asset_count"] == 1


def test_raster_figure_uses_preview_fallback_without_fake_vector(tmp_path: Path) -> None:
    image_path = tmp_path / "scan.png"
    image = Image.new("RGB", (240, 140), "white")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((10, 10, 230, 130), outline="navy", width=4)
    drawing.line((20, 115, 210, 25), fill="red", width=5)
    image.save(image_path)

    pdf_path = tmp_path / "raster.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=400, height=500)
    page.insert_image(fitz.Rect(60, 100, 340, 280), filename=str(image_path))
    pdf.save(pdf_path)
    pdf.close()

    document = _vector_document()
    populated = FigureExtractionService().extract(
        pdf_path=pdf_path,
        document=document,
        artifact_dir=tmp_path / "figures",
    )
    figure = populated.figures[0]

    assert figure.asset_type == FigureAssetType.RASTER
    assert Path(str(figure.image_path)).is_file()
    assert figure.vector_path is None
    assert figure.extraction_metadata["readable_reconstruction_asset"] == "raster"
    assert figure.extraction_metadata["raster_fallback_reason"] == "source_region_is_raster"


def test_only_external_caption_is_translated(monkeypatch) -> None:
    document = _vector_document()
    translator = MlxTranslator(TranslationSettings(chunk_group_size=1))
    monkeypatch.setattr(translator, "_ensure_loaded", lambda: True)
    monkeypatch.setattr(translator, "_is_already_english", lambda _chunk: False)
    monkeypatch.setattr(
        translator,
        "_translate_chunk_with_validation",
        lambda text, _context, _language, _block_type, **_kwargs: f"ENGLISH {text}",
    )

    chunks = translator.build_chunks(document)
    translated, _markdown = translator.translate_document(document, "source")
    block_by_id = {block.id: block for block in translated.blocks}

    assert all("figure-region" not in chunk.block_ids for chunk in chunks)
    assert block_by_id["figure-region"].text == "ACHSE QUELLE"
    assert block_by_id["caption"].text == "ENGLISH Figure 1. Quellbeschriftung"


def test_surya_figure_region_is_excluded_from_logical_translation_chunks() -> None:
    result = OCRToTranslationParser().prepare(_vector_document(), document_id="figure-test")

    assert all(
        "figure-region" not in chunk.block_ids
        for chunk in result.document.translation_chunks
    )
    assert any(
        item["block_id"] == "figure-region"
        and item["reason"] == "figure_internal_text_preserved"
        for item in result.excluded_regions
    )


def test_readable_pdf_embeds_figure_once_with_translated_caption(tmp_path: Path) -> None:
    pdf_path = tmp_path / "vector.pdf"
    _create_vector_figure_pdf(pdf_path)
    document = FigureExtractionService().extract(
        pdf_path=pdf_path,
        document=_vector_document(),
        artifact_dir=tmp_path / "figures",
    )
    caption = next(block for block in document.blocks if block.id == "caption")
    caption.text = "Figure 1. Translated external caption"

    markdown = MarkdownBuilder().build(document)
    assert markdown.count('<figure class="document-figure"') == 1
    assert markdown.count("<figcaption>Figure 1. Translated external caption</figcaption>") == 1
    assert "*Figure 1. Translated external caption*" not in markdown
    assert "## Figures" not in markdown
    assert "Figure preserved as placeholder" not in markdown
    assert "ACHSE QUELLE" not in markdown

    output_path = tmp_path / "readable.pdf"
    reconstructor = Reconstructor()
    html_text = reconstructor.markdown_to_html(markdown, title="Readable figure test")
    reconstructor.html_to_pdf(html_text, output_path)

    with fitz.open(output_path) as output:
        text = "\n".join(page.get_text("text") for page in output)
        visual_count = sum(len(page.get_drawings()) + len(page.get_images()) for page in output)
        assert "Translated external caption" in text
        assert visual_count > 0


def test_readable_pdf_suppresses_qwen_remote_image_when_local_asset_exists(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "vector.pdf"
    _create_vector_figure_pdf(pdf_path)
    document = FigureExtractionService().extract(
        pdf_path=pdf_path,
        document=_vector_document(),
        artifact_dir=tmp_path / "figures",
    )
    figure_block = next(block for block in document.blocks if block.id == "figure-region")
    # Exercise the renderer's defensive path for a Qwen image wrapper that was
    # misclassified as prose before its local figure crop was materialised.
    figure_block.block_type = BlockType.PARAGRAPH
    figure_block.text = (
        "![Generated chart description]"
        "(https://example.invalid/generated-placeholder.png)"
    )
    figure_block.metadata["parser"] = "qwen_surya_full_page_ocr"

    markdown = MarkdownBuilder().build(document)

    assert "example.invalid" not in markdown
    assert "Generated chart description" not in markdown
    assert markdown.count('<figure class="document-figure"') == 1
    assert "figure-p0001-001" in markdown


def test_readable_pdf_keeps_qwen_alt_text_when_local_figure_asset_is_missing(
    tmp_path: Path,
) -> None:
    document = _vector_document()
    figure_block = next(block for block in document.blocks if block.id == "figure-region")
    figure_block.text = (
        "![Generated chart description]"
        "(https://example.invalid/generated-placeholder.png)"
    )
    document.figures[0].image_path = str(tmp_path / "missing-figure.png")

    markdown = MarkdownBuilder().build(document)

    assert "example.invalid" not in markdown
    assert "[Image unavailable: Generated chart description]" in markdown
    assert '<figure class="document-figure"' not in markdown


def test_readable_pdf_preserves_unrelated_remote_markdown_image() -> None:
    document = _vector_document()
    mention = next(block for block in document.blocks if block.id == "mention")
    mention.text = "![Publisher logo](https://static.example.org/publisher-logo.png)"
    mention.metadata["parser"] = "marker"

    markdown = MarkdownBuilder().build(document)

    assert (
        "![Publisher logo](https://static.example.org/publisher-logo.png)" in markdown
    )
    assert "Image unavailable" not in markdown


def test_figure_moves_after_reliable_first_mention(tmp_path: Path) -> None:
    pdf_path = tmp_path / "vector.pdf"
    _create_vector_figure_pdf(pdf_path)
    document = FigureExtractionService().extract(
        pdf_path=pdf_path,
        document=_vector_document(),
        artifact_dir=tmp_path / "figures",
    )
    figure_block = next(block for block in document.blocks if block.id == "figure-region")
    mention = next(block for block in document.blocks if block.id == "mention")
    figure_block.reading_order_index = 0
    mention.reading_order_index = 2
    document.blocks = [figure_block, document.blocks[2], mention]
    document.figures[0].source_block_ids = [figure_block.id]
    document.blocks[1].text = "Figure 1. Translated caption"
    mention.text = "As shown in Figure 1, the result is clear."

    markdown = MarkdownBuilder().build(document)

    assert markdown.index("As shown in Figure 1") < markdown.index("<figure")
