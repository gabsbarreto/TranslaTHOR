from __future__ import annotations

import io
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from app.models.inspection import PageInspection, PdfInspection
from app.models.schema import BlockType
from app.services.markdown_builder import MarkdownBuilder
from app.services.pdf_extraction.surya2_adapter import (
    LABEL_TO_BLOCK_TYPE,
    Surya2DocumentAdapter,
    image_polygon_to_pdf,
    normalized_label,
)
from app.services.pdf_extraction.surya2_extractor import Surya2LlamaCppExtractor
from app.services.pdf_extraction.surya2_runtime import (
    Surya2Runtime,
    build_surya2_worker_environment,
)
from app.services.reconstructor import Reconstructor


def _inspection() -> PdfInspection:
    return PdfInspection(
        filename="fixture.pdf",
        title="Fixture",
        author=None,
        page_count=1,
        pages=[
            PageInspection(
                page_number=1,
                width=600,
                height=800,
                text_length=0,
                embedded_text_quality=0,
                has_embedded_text=False,
            )
        ],
    )


def _raw_payload() -> dict:
    return {
        "schema_version": 1,
        "engine": "surya2_llamacpp",
        "surya_version": "0.22.1",
        "strategy": "full_page",
        "batching": {
            "parallel_pages": 5,
            "context_per_slot": 16384,
            "total_context": 81920,
            "requested_pages": 1,
            "effective_parallel_pages": 1,
        },
        "timing": {"total_worker_seconds": 1.0},
        "pages": [
            {
                "page_number": 1,
                "image_bbox": [0, 0, 1200, 1600],
                "blocks": [
                    {
                        "label": "Caption",
                        "raw_label": "Caption",
                        "reading_order": 2,
                        "html": "<p>Figure 1. System diagram</p>",
                        "polygon": [[100, 900], [700, 900], [700, 960], [100, 960]],
                        "bbox": [100, 900, 700, 960],
                        "confidence": 0.91,
                        "skipped": False,
                        "error": False,
                    },
                    {
                        "label": "PageHeader",
                        "raw_label": "PageHeader",
                        "reading_order": 0,
                        "html": "<p>Running title</p>",
                        "polygon": [[80, 20], [1120, 20], [1120, 70], [80, 70]],
                        "bbox": [80, 20, 1120, 70],
                        "confidence": 0.98,
                        "skipped": False,
                        "error": False,
                    },
                    {
                        "label": "Picture",
                        "raw_label": "Image",
                        "reading_order": 1,
                        "html": "",
                        "polygon": [[100, 300], [700, 300], [700, 880], [100, 880]],
                        "bbox": [100, 300, 700, 880],
                        "confidence": 0.94,
                        "skipped": True,
                        "error": False,
                    },
                    {
                        "label": "Table",
                        "raw_label": "Table",
                        "reading_order": 3,
                        "html": (
                            "<table><tr><th>A</th><th>B</th></tr>"
                            '<tr><td rowspan="2">1</td><td>2</td></tr></table>'
                        ),
                        "polygon": [[100, 1000], [1100, 1000], [1100, 1300], [100, 1300]],
                        "bbox": [100, 1000, 1100, 1300],
                        "confidence": 0.88,
                        "skipped": False,
                        "error": False,
                    },
                    {
                        "label": "Equation",
                        "raw_label": "Formula",
                        "reading_order": 4,
                        "html": "<math>x^2 + y^2</math>",
                        "polygon": [[200, 1350], [800, 1350], [800, 1420], [200, 1420]],
                        "bbox": [200, 1350, 800, 1420],
                        "confidence": 0.77,
                        "skipped": False,
                        "error": False,
                    },
                    {
                        "label": "Text",
                        "raw_label": "Text",
                        "reading_order": 5,
                        "html": "",
                        "polygon": [[100, 1450], [800, 1450], [800, 1500], [100, 1500]],
                        "bbox": [100, 1450, 800, 1500],
                        "confidence": 0,
                        "skipped": False,
                        "error": True,
                    },
                ],
            }
        ],
    }


def test_image_polygon_to_pdf_scales_and_clamps_without_flipping_y() -> None:
    polygon = image_polygon_to_pdf(
        [[-10, 0], [1200, 0], [1300, 1600], [0, 1700]],
        image_width=1200,
        image_height=1600,
        pdf_width=600,
        pdf_height=800,
    )

    assert polygon == [[0, 0], [600, 0], [600, 800], [0, 800]]


def test_image_polygon_to_pdf_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        image_polygon_to_pdf(
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            image_width=0,
            image_height=1,
            pdf_width=1,
            pdf_height=1,
        )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("PageHeader", BlockType.HEADER),
        ("PageFooter", BlockType.FOOTER),
        ("SectionHeader", BlockType.HEADING),
        ("Text", BlockType.PARAGRAPH),
        ("ListGroup", BlockType.LIST),
        ("Table", BlockType.TABLE),
        ("Equation", BlockType.EQUATION),
        ("ChemicalBlock", BlockType.EQUATION),
        ("Picture", BlockType.FIGURE),
        ("Figure", BlockType.FIGURE),
        ("Diagram", BlockType.FIGURE),
        ("Caption", BlockType.CAPTION),
    ],
)
def test_surya2_public_labels_map_to_shared_schema(
    label: str,
    expected: BlockType,
) -> None:
    assert LABEL_TO_BLOCK_TYPE[normalized_label(label)] == expected


def test_surya2_adapter_preserves_schema_reading_order_visuals_and_tables() -> None:
    document, markdown, chunks = Surya2DocumentAdapter().build_document(
        raw_pages=_raw_payload()["pages"],
        inspection=_inspection(),
        strategy="full_page",
        document_id="fixture-job",
    )

    assert [block.block_type for block in document.blocks] == [
        BlockType.HEADER,
        BlockType.FIGURE,
        BlockType.CAPTION,
        BlockType.TABLE,
        BlockType.EQUATION,
        BlockType.PARAGRAPH,
    ]
    assert [block.reading_order_index for block in document.blocks] == list(range(6))

    visual = document.blocks[1]
    assert visual.text == ""
    assert visual.skipped is True
    assert visual.raw_label == "Image"
    assert visual.bbox is not None
    assert visual.bbox.model_dump() == {
        "x0": 50.0,
        "y0": 150.0,
        "x1": 350.0,
        "y1": 440.0,
    }
    assert visual.polygon == [
        [50.0, 150.0],
        [350.0, 150.0],
        [350.0, 440.0],
        [50.0, 440.0],
    ]

    assert len(document.figures) == 1
    assert document.figures[0].caption_block_id == document.blocks[2].id
    assert document.blocks[2].metadata["caption_for_figure_id"] == document.figures[0].id

    table = document.tables[0]
    assert table.headers == ["A", "B"]
    assert table.rows == [["1", "2"]]
    assert table.cells[0][0].rowspan == 2
    assert table.parse_mode == "surya2_html"
    assert document.blocks[3].html.startswith("<table>")

    equation = document.blocks[4]
    assert equation.text == "<math>x^2 + y^2</math>"
    assert equation.html == equation.text
    assert document.blocks[5].error is True
    assert document.blocks[5].confidence == 0
    assert document.pages[0].width == 600
    assert "Figure 1. System diagram" in markdown
    assert all(chunk.original_text for chunk in chunks)


def test_surya2_figure_path_with_spaces_is_embedded_in_reconstructed_pdf(
    tmp_path: Path,
) -> None:
    document, _markdown, _chunks = Surya2DocumentAdapter().build_document(
        raw_pages=_raw_payload()["pages"],
        inspection=_inspection(),
        strategy="full_page",
        document_id="fixture-job",
    )
    image_dir = tmp_path / "figure assets"
    image_dir.mkdir()
    image_path = image_dir / "figure one.png"
    Image.new("RGB", (120, 80), "navy").save(image_path)
    document.figures[0].image_path = str(image_path)
    markdown = MarkdownBuilder().build(document)

    assert "file://" in markdown
    assert "figure%20assets/figure%20one.png" in markdown

    output_pdf = tmp_path / "reconstructed.pdf"
    reconstructor = Reconstructor()
    reconstructor.html_to_pdf(
        reconstructor.markdown_to_html(markdown, title="Figure fixture"),
        output_pdf,
    )

    page = PdfReader(output_pdf).pages[0]
    xobjects = page["/Resources"].get("/XObject", {})
    assert any(
        item.get_object().get("/Subtype") == "/Image" for item in xobjects.get_object().values()
    )


class _FakeRuntime:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []
        self.closed = False

    def run(self, **kwargs):
        image_paths = kwargs["image_paths"]
        with Image.open(image_paths[0]) as image:
            assert image.size == (816, 1056)
        self.calls.append(kwargs)
        output_path = kwargs["output_path"]
        output_path.write_text(json.dumps(self.payload), encoding="utf-8")
        return self.payload

    def close(self) -> None:
        self.closed = True


def test_surya2_extractor_integration_renders_fixture_and_writes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as output:
        writer.write(output)

    payload = _raw_payload()
    payload["pages"][0]["image_bbox"] = [0, 0, 816, 1056]
    fake_runtime = _FakeRuntime(payload)
    extractor = Surya2LlamaCppExtractor(runtime=fake_runtime)

    def render_fixture(_pdf_path, _page_number, output_path, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (816, 1056), "white").save(output_path)
        return output_path

    monkeypatch.setattr(extractor.renderer, "render_page", render_fixture)
    result = extractor.extract(
        pdf_path=pdf_path,
        job_dir=tmp_path / "job",
        pdf_classification="scanned_no_text",
        detection_metadata={"fixture": True},
        warnings=[],
        settings={"surya2_dpi": 96, "surya2_strategy": "full_page"},
    )

    assert result.document is not None
    assert result.metadata["surya2_version"] == "0.22.1"
    assert result.metadata["surya2_strategy"] == "full_page"
    assert result.metadata["surya2_dpi"] == 96
    assert result.metadata["surya2_batching"]["parallel_pages"] == 5
    assert len(fake_runtime.calls) == 1
    assert fake_runtime.calls[0]["strategy"] == "full_page"
    assert (tmp_path / "job/surya2/raw_full_page.json").exists()
    assert (tmp_path / "job/surya2/overlays_full_page/page_0001.png").exists()
    assert (tmp_path / "job/surya2/figures_full_page/surya2-figure-1.png").exists()
    assert result.document.figures[0].image_path is not None
    assert "surya2-figure-1.png" in result.markdown
    assert (tmp_path / "job/surya2/logical_translation_chunks_full_page.json").exists()


def test_runtime_event_pump_does_not_lose_buffered_lines() -> None:
    process = SimpleNamespace(
        stdout=io.StringIO(
            '{"event":"page_done","request_id":"one"}\n'
            '{"event":"request_complete","request_id":"one"}\n'
        )
    )
    events: queue.Queue[dict] = queue.Queue()

    Surya2Runtime._pump_events(process, events)

    assert events.get_nowait()["event"] == "page_done"
    assert events.get_nowait()["event"] == "request_complete"


def test_runtime_defaults_to_five_parallel_pages_with_full_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "SURYA_INFERENCE_PARALLEL",
        "SURYA_INFERENCE_CTX_PER_SLOT",
        "SURYA_INFERENCE_CTX_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    env = build_surya2_worker_environment()

    assert env["SURYA_INFERENCE_PARALLEL"] == "5"
    assert env["SURYA_INFERENCE_CTX_PER_SLOT"] == "16384"
    assert env["SURYA_INFERENCE_CTX_SIZE"] == "81920"


def test_runtime_preserves_explicit_surya_batch_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SURYA_INFERENCE_PARALLEL", "2")
    monkeypatch.setenv("SURYA_INFERENCE_CTX_PER_SLOT", "12288")
    monkeypatch.setenv("SURYA_INFERENCE_CTX_SIZE", "32768")

    env = build_surya2_worker_environment()

    assert env["SURYA_INFERENCE_PARALLEL"] == "2"
    assert env["SURYA_INFERENCE_CTX_PER_SLOT"] == "12288"
    assert env["SURYA_INFERENCE_CTX_SIZE"] == "32768"


def test_runtime_raises_stale_total_context_to_protect_every_batch_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SURYA_INFERENCE_PARALLEL", "5")
    monkeypatch.setenv("SURYA_INFERENCE_CTX_PER_SLOT", "16384")
    monkeypatch.setenv("SURYA_INFERENCE_CTX_SIZE", "16384")

    env = build_surya2_worker_environment()

    assert env["SURYA_INFERENCE_PARALLEL"] == "5"
    assert env["SURYA_INFERENCE_CTX_PER_SLOT"] == "16384"
    assert env["SURYA_INFERENCE_CTX_SIZE"] == "81920"
