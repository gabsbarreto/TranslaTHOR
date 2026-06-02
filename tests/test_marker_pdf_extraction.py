from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.schema import SourceType
from app.services.pdf_extraction.local_vlm_service import LocalVLMConfig, LocalVLMRepairService
from app.services.pdf_extraction.markdown_builder import MarkerDocumentBuilder
from app.services.pdf_extraction.marker_extractor import PDFExtractor
from app.services.pdf_extraction.models import PDFTypeDetectionResult, PageTextStats
from app.services.pdf_extraction.pdf_type_detector import PDFTypeDetector
from app.services.pdf_extraction.qwen_ocr_fallback import QwenFullPageOCRFallback


fitz = pytest.importorskip("fitz")


def test_pdf_type_detector_classifies_digital_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "digital.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        fitz.Rect(72, 72, 540, 720),
        ("This is a normal selectable digital PDF page with meaningful embedded text. " * 20),
        fontsize=11,
    )
    doc.save(pdf_path)
    doc.close()

    result = PDFTypeDetector().detect(pdf_path)

    assert result.classification == "digital_good_text"
    assert result.meaningful_page_count == 1
    assert result.embedded_text_words > 40


def test_pdf_type_detector_classifies_scanned_image_only(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    png_path = tmp_path / "page.png"
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 1200, 1600), False)
    pix.clear_with(255)
    pix.save(png_path)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(0, 0, 612, 792), filename=png_path)
    doc.save(pdf_path)
    doc.close()

    result = PDFTypeDetector().detect(pdf_path)

    assert result.classification == "scanned_no_text"
    assert result.image_dominant_page_count == 1


def test_pdf_type_detector_treats_full_page_image_with_selectable_text_as_hidden_ocr(tmp_path: Path) -> None:
    pdf_path = tmp_path / "hidden_ocr.pdf"
    png_path = tmp_path / "page.png"
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 1200, 1600), False)
    pix.clear_with(255)
    pix.save(png_path)
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page(width=612, height=792)
        page.insert_image(fitz.Rect(0, 0, 612, 792), filename=png_path)
        page.insert_textbox(
            fitz.Rect(72, 72, 540, 720),
            (
                "O servico de saude acompanha criancas e adolescentes com genero, "
                "populacao, avaliacao, experiencia e atencao clinica. "
            )
            * 8,
            fontsize=11,
        )
    doc.save(pdf_path)
    doc.close()

    result = PDFTypeDetector().detect(pdf_path)

    assert result.classification == "bad_hidden_ocr"
    assert result.metadata["suspicious_hidden_ocr"] is True
    assert result.metadata["hidden_ocr_page_count"] == 2


def test_marker_subprocess_failure_is_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = PDFExtractor(detector=_FakeDetector("digital_good_text"))
    monkeypatch.setenv("MARKER_BIN", "/usr/bin/false")

    with pytest.raises(RuntimeError, match="Marker failed with exit code"):
        extractor.extract(tmp_path / "input.pdf", job_dir=tmp_path, keep_debug_artifacts=True)

    assert (tmp_path / "marker" / "marker_failure.json").exists()


def test_marker_force_ocr_accelerator_failure_retries_on_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker_bin = tmp_path / "fake_marker_retry_cpu.py"
    marker_bin.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output_dir") + 1])
out.mkdir(parents=True, exist_ok=True)
if "--force_ocr" in sys.argv and os.environ.get("TORCH_DEVICE") != "cpu":
    sys.stderr.write("torch.AcceleratorError: index 8192 is out of bounds")
    sys.exit(1)
(out / "result.json").write_text(json.dumps([{
  "id": "/page/0/Page/0",
  "block_type": "Page",
  "children": [{"id": "/page/0/Text/0", "block_type": "Text", "html": "<p>Texto recuperado.</p>"}]
}]), encoding="utf-8")
""",
        encoding="utf-8",
    )
    marker_bin.chmod(0o755)
    monkeypatch.setenv("MARKER_BIN", str(marker_bin))

    result = PDFExtractor(detector=_FakeDetector("bad_hidden_ocr")).extract(
        tmp_path / "input.pdf",
        mode="strip_and_force_ocr",
        job_dir=tmp_path,
        keep_debug_artifacts=True,
    )

    assert result.used_force_ocr is True
    assert result.metadata["marker_retried_on_cpu"] is True
    assert any("retrying the same Marker OCR mode on CPU" in warning for warning in result.warnings)


def test_marker_builder_preserves_running_author_headers_misclassified_as_section_headers() -> None:
    detection = PDFTypeDetectionResult(
        classification="digital_good_text",
        page_count=3,
        pages=[
            _page_stats(1),
            _page_stats(2),
            _page_stats(3),
        ],
        embedded_text_chars=1000,
        embedded_text_words=160,
        meaningful_page_count=3,
        garbled_page_count=0,
        image_dominant_page_count=0,
        scanned_page_count=0,
        mixed=False,
    )
    marker_payload = [
        {
            "id": "/page/0/Page/0",
            "block_type": "Page",
            "children": [
                {
                    "id": "/page/0/SectionHeader/0",
                    "block_type": "SectionHeader",
                    "text": "INTRODUCCIÓN",
                    "polygon": [[72, 120], [180, 120], [180, 136], [72, 136]],
                }
            ],
        },
        {
            "id": "/page/1/Page/0",
            "block_type": "Page",
            "children": [
                {
                    "id": "/page/1/SectionHeader/0",
                    "block_type": "SectionHeader",
                    "text": "3 Rojas Contreras G, et al",
                    "polygon": [[49, 39], [181, 39], [181, 50], [49, 50]],
                },
                {
                    "id": "/page/1/Text/1",
                    "block_type": "Text",
                    "text": "El objetivo de esta investigación es describir la muestra.",
                    "polygon": [[49, 75], [520, 75], [520, 145], [49, 145]],
                },
            ],
        },
    ]

    document, markdown, chunks = MarkerDocumentBuilder().build_document(
        marker_payload=marker_payload,
        detection=detection,
        filename="paper.pdf",
        source_type=SourceType.EMBEDDED,
        parser_metadata={},
        warnings=[],
    )

    running_header = next(block for block in document.blocks if "Rojas Contreras" in block.text)
    assert running_header.block_type.value == "heading"
    assert "## 3 Rojas Contreras G, et al" in markdown
    assert any("Rojas Contreras" in chunk.original_text for chunk in chunks)
    assert "## INTRODUCCIÓN" in markdown


def test_marker_builder_ignores_tablegroup_wrapper_tables() -> None:
    detection = PDFTypeDetectionResult(
        classification="digital_good_text",
        page_count=1,
        pages=[_page_stats(1)],
        embedded_text_chars=1000,
        embedded_text_words=160,
        meaningful_page_count=1,
        garbled_page_count=0,
        image_dominant_page_count=0,
        scanned_page_count=0,
        mixed=False,
    )
    marker_payload = [
        {
            "id": "/page/0/Page/0",
            "block_type": "Page",
            "children": [
                {
                    "id": "/page/0/TableGroup/1",
                    "block_type": "TableGroup",
                    "children": [
                        {
                            "id": "/page/0/Caption/2",
                            "block_type": "Caption",
                            "text": "Table 1. Demographic data",
                        },
                        {
                            "id": "/page/0/Table/3",
                            "block_type": "Table",
                            "html": "<table><tbody><tr><th>Variable</th><th>n</th></tr><tr><td>Age</td><td>53</td></tr></tbody></table>",
                        },
                    ],
                }
            ],
        }
    ]

    document, markdown, chunks = MarkerDocumentBuilder().build_document(
        marker_payload=marker_payload,
        detection=detection,
        filename="paper.pdf",
        source_type=SourceType.EMBEDDED,
        parser_metadata={},
        warnings=[],
    )

    assert all((block.metadata or {}).get("marker_block_type") != "TableGroup" for block in document.blocks)
    assert len(document.tables) == 1
    assert "structured-table" not in markdown
    assert "### Table 1" not in markdown
    assert "*Table 1. Demographic data*" in markdown
    assert "<td>Age</td><td>53</td>" in markdown
    assert len([chunk for chunk in chunks if "<table" in chunk.original_text]) == 1


def test_marker_builder_normalizes_one_column_table_caption_html() -> None:
    detection = PDFTypeDetectionResult(
        classification="digital_good_text",
        page_count=1,
        pages=[_page_stats(1)],
        embedded_text_chars=1000,
        embedded_text_words=160,
        meaningful_page_count=1,
        garbled_page_count=0,
        image_dominant_page_count=0,
        scanned_page_count=0,
        mixed=False,
    )
    marker_payload = [
        {
            "id": "/page/0/Page/0",
            "block_type": "Page",
            "children": [
                {
                    "id": "/page/0/Table/1",
                    "block_type": "Table",
                    "html": (
                        "<table><tbody><tr><th>Tablo 4. Cinsiyet hoşnutsuzluğu olan ergenlerde takipte "
                        "işlevsellikteki bozulmayı etkileyen değişkenleri gösteren regresyon</th></tr>"
                        "<tr><td>modeli</td></tr></tbody></table>"
                    ),
                },
                {
                    "id": "/page/0/Table/2",
                    "block_type": "Table",
                    "html": "<table><tbody><tr><th>Bağımsız değişkenler</th><th>p</th></tr><tr><td>Cinsiyet</td><td>0,046</td></tr></tbody></table>",
                },
            ],
        }
    ]

    document, markdown, chunks = MarkerDocumentBuilder().build_document(
        marker_payload=marker_payload,
        detection=detection,
        filename="paper.pdf",
        source_type=SourceType.EMBEDDED,
        parser_metadata={},
        warnings=[],
    )

    caption = document.blocks[0]
    assert caption.block_type.value == "caption"
    assert caption.metadata["marker_table_caption_normalized"] is True
    assert "regresyon modeli" in caption.text
    assert len(document.tables) == 1
    assert "<td>modeli</td>" not in markdown
    assert "*Tablo 4." in markdown
    assert "<td>Cinsiyet</td><td>0,046</td>" in markdown
    assert all("<td>modeli</td>" not in chunk.original_text for chunk in chunks)


def test_bad_hidden_ocr_falls_back_to_normal_marker_if_forced_ocr_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker_bin = tmp_path / "fake_marker_fallback_normal.py"
    marker_bin.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output_dir") + 1])
out.mkdir(parents=True, exist_ok=True)
if "--force_ocr" in sys.argv:
    sys.stderr.write("torch.AcceleratorError: index 8192 is out of bounds")
    sys.exit(1)
(out / "result.json").write_text(json.dumps([{
  "id": "/page/0/Page/0",
  "block_type": "Page",
  "children": [{"id": "/page/0/Text/0", "block_type": "Text", "html": "<p>Texto da camada OCR existente.</p>"}]
}]), encoding="utf-8")
""",
        encoding="utf-8",
    )
    marker_bin.chmod(0o755)
    monkeypatch.setenv("MARKER_BIN", str(marker_bin))

    result = PDFExtractor(detector=_FakeDetector("bad_hidden_ocr")).extract(
        tmp_path / "input.pdf",
        mode="strip_and_force_ocr",
        job_dir=tmp_path,
        keep_debug_artifacts=True,
    )

    assert result.used_force_ocr is False
    assert result.used_ocr is True
    assert result.metadata["marker_mode"] == "normal"
    assert result.metadata["marker_fallback_to_normal"] is True
    assert any("Falling back to Marker normal mode" in warning for warning in result.warnings)


def test_auto_bad_hidden_ocr_uses_marker_text_only_first_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker_bin = tmp_path / "fake_marker_text_only.py"
    marker_bin.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output_dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "args.txt").write_text(json.dumps(sys.argv), encoding="utf-8")
(out / "result.json").write_text(json.dumps([{
  "id": "/page/0/Page/0",
  "block_type": "Page",
  "children": [{"id": "/page/0/Text/0", "block_type": "Text", "html": "<p>Texto da camada OCR existente.</p>"}]
}]), encoding="utf-8")
""",
        encoding="utf-8",
    )
    marker_bin.chmod(0o755)
    monkeypatch.setenv("MARKER_BIN", str(marker_bin))

    result = PDFExtractor(detector=_FakeDetector("bad_hidden_ocr")).extract(
        tmp_path / "input.pdf",
        mode="auto",
        job_dir=tmp_path,
        keep_debug_artifacts=True,
    )

    args = json.loads((tmp_path / "marker" / "args.txt").read_text(encoding="utf-8"))
    assert "--disable_ocr" in args
    assert "--force_ocr" not in args
    assert result.used_ocr is False
    assert result.used_force_ocr is False
    assert result.metadata["marker_mode"] == "text_only"


def test_marker_output_becomes_document_and_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker_bin = tmp_path / "fake_marker.py"
    marker_bin.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output_dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps([{
  "id": "/page/0/Page/0",
  "block_type": "Page",
  "polygon": [[0, 0], [612, 0], [612, 792], [0, 792]],
  "children": [{
    "id": "/page/0/Text/0",
    "block_type": "Text",
    "html": "<p>Texto de exemplo para traduzir.</p>",
    "polygon": [[72, 72], [540, 72], [540, 100], [72, 100]]
  }]
}]), encoding="utf-8")
""",
        encoding="utf-8",
    )
    marker_bin.chmod(0o755)
    monkeypatch.setenv("MARKER_BIN", str(marker_bin))

    result = PDFExtractor(detector=_FakeDetector("digital_good_text")).extract(
        tmp_path / "input.pdf",
        mode="auto",
        job_dir=tmp_path,
        keep_debug_artifacts=True,
    )

    assert result.pdf_classification == "digital_good_text"
    assert result.used_ocr is False
    assert result.document is not None
    assert result.document.blocks[0].text == "Texto de exemplo para traduzir."
    assert result.document.blocks[0].source_type == SourceType.EMBEDDED
    assert result.chunks[0].original_text == "Texto de exemplo para traduzir."


def test_marker_table_cells_are_not_duplicated_as_text_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker_bin = tmp_path / "fake_marker_table.py"
    marker_bin.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output_dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps([{
  "id": "/page/0/Page/0",
  "block_type": "Page",
  "children": [{
    "id": "/page/0/Table/1",
    "block_type": "Table",
    "html": "<table><tr><th>Tabla</th></tr><tr><td>Celda original</td></tr></table>",
    "polygon": [[10, 10], [200, 10], [200, 100], [10, 100]],
    "children": [{
      "id": "/page/0/TableCell/2",
      "block_type": "TableCell",
      "html": "Celda original",
      "polygon": [[10, 40], [200, 40], [200, 70], [10, 70]]
    }]
  }]
}]), encoding="utf-8")
""",
        encoding="utf-8",
    )
    marker_bin.chmod(0o755)
    monkeypatch.setenv("MARKER_BIN", str(marker_bin))

    result = PDFExtractor(detector=_FakeDetector("digital_good_text")).extract(
        tmp_path / "input.pdf",
        mode="digital",
        job_dir=tmp_path,
        keep_debug_artifacts=True,
    )

    assert result.document is not None
    assert [block.block_type.value for block in result.document.blocks] == ["table"]
    assert len(result.chunks) == 1
    assert result.markdown.count("Celda original") == 1


def test_local_vlm_repair_server_unavailable_does_not_raise() -> None:
    service = LocalVLMRepairService(
        LocalVLMConfig(
            enabled=True,
            base_url="http://127.0.0.1:9/v1",
            model="local-test-model",
            api_key="not-needed",
            timeout=1,
            max_retries=0,
        )
    )
    block = _block("b1", "<table><tr><td>broken")

    repaired, warnings = service.repair_blocks([block])

    assert repaired == 0
    assert warnings
    assert block.text == "<table><tr><td>broken"


def test_local_vlm_selects_portuguese_hidden_ocr_candidates() -> None:
    service = LocalVLMRepairService(LocalVLMConfig(False, "http://127.0.0.1:9/v1", "", "not-needed", 1, 0))
    block = _paragraph_block(
        "p1",
        (
            "O servico de saude acompanha criancas e adolescentes com variabilidade de genero. "
            "A avaliacao da populacao inclui experiencia clinica, atencao e relacao familiar."
        ),
    )

    selected = service.select_blocks_for_repair(
        [block],
        {"pdf_classification": "bad_hidden_ocr", "detected_language": "pt"},
    )

    assert selected == [block]


def test_qwen_fallback_uses_rendered_png_metadata_for_ocr_input(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "page.png"
    Image.new("RGB", (400, 200), "white").save(source)

    metadata = QwenFullPageOCRFallback()._rendered_page_metadata(source)

    assert metadata["input_path"] == str(source)
    assert metadata["ocr_image_path"] == str(source)
    assert metadata["ocr_image_mode"] == "rendered_page_png"
    assert metadata["ocr_image_width"] == 400
    assert metadata["ocr_image_height"] == 200


def test_qwen_fallback_preserves_full_page_margins_before_ocr(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "page.png"
    image = Image.new("RGB", (100, 100), "white")
    for x in range(100):
        image.putpixel((x, 0), (0, 0, 0))
        image.putpixel((x, 99), (0, 0, 0))
    for y in range(100):
        image.putpixel((0, y), (0, 0, 0))
        image.putpixel((99, y), (0, 0, 0))
    image.save(source)

    metadata = QwenFullPageOCRFallback()._rendered_page_metadata(source)

    with Image.open(metadata["ocr_image_path"]) as image:
        assert image.getpixel((50, 0))[0] < 60
        assert image.getpixel((50, 99))[0] < 60
        assert image.getpixel((0, 50))[0] < 60
        assert image.getpixel((99, 50))[0] < 60
    assert metadata["ocr_image_mode"] == "rendered_page_png"


def test_qwen_fallback_uses_surya_boxed_full_pages_for_bad_scans(tmp_path: Path) -> None:
    from PIL import Image

    rendered = tmp_path / "rendered.png"
    boxed = tmp_path / "boxed.png"
    Image.new("RGB", (400, 200), "white").save(rendered)
    Image.new("RGB", (400, 200), "white").save(boxed)
    fallback = QwenFullPageOCRFallback()
    manifest = {
        "pages": [
            {
                "boxed_page_path": str(boxed),
                "regions": [{"id": "r1"}, {"id": "r2"}],
                "reconciled_regions": [{"index": 1}],
            }
        ]
    }

    paths = fallback._surya_boxed_page_paths(manifest, expected_pages=1)
    metadata = fallback._surya_page_metadata(
        [fallback._rendered_page_metadata(rendered)],
        manifest,
        paths,
    )

    assert fallback._should_use_surya_layout("scanned_no_text") is True
    assert fallback._should_use_surya_layout("bad_hidden_ocr") is True
    assert fallback._should_use_surya_layout("mixed") is False
    assert paths == [boxed]
    assert metadata[0]["input_path"] == str(rendered)
    assert metadata[0]["ocr_image_path"] == str(boxed)
    assert metadata[0]["ocr_image_mode"] == "surya_boxed_page_png"
    assert metadata[0]["surya_region_count"] == 2
    assert metadata[0]["surya_reconciled_region_count"] == 1


def test_qwen_fallback_surya_prompt_requests_numbered_region_wrappers() -> None:
    prompt = QwenFullPageOCRFallback()._surya_overlay_prompt("Base OCR rules.")

    assert prompt.startswith("Base OCR rules.")
    assert 'SURYA <number>: <type>' in prompt
    assert '<region index="<number>" type="<type>">' in prompt
    assert "page headers, and page footers" in prompt


def test_qwen_fallback_runs_surya_worker_with_marker_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    output_dir = tmp_path / "layout"
    output_dir.mkdir()
    manifest = {"pages": [], "region_count": 0}
    (output_dir / "layout.json").write_text(json.dumps(manifest), encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeProcess:
        returncode = 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProcess()

    fallback = QwenFullPageOCRFallback()
    monkeypatch.setenv("SURYA_LAYOUT_PYTHON", sys.executable)
    monkeypatch.setenv("SURYA_LAYOUT_WORKER", str(tmp_path / "worker.py"))
    monkeypatch.setattr("app.services.pdf_extraction.qwen_ocr_fallback.subprocess.Popen", fake_popen)
    monkeypatch.setattr(fallback, "_communicate_with_cancel", lambda *_args, **_kwargs: ("", ""))

    result = fallback._run_surya_layout(
        render_dir=tmp_path / "rendered",
        output_dir=output_dir,
        settings={"surya_layout_padding": 24, "surya_layout_batch_size": 2},
        cancel_requested=None,
        on_process_started=None,
        on_process_finished=None,
        on_ocr_progress=None,
    )

    assert result == manifest
    assert captured["cmd"] == [
        sys.executable,
        str(tmp_path / "worker.py"),
        "--input-dir",
        str(tmp_path / "rendered"),
        "--output-dir",
        str(output_dir),
        "--padding",
        "24",
        "--batch-size",
        "2",
    ]


def _page_stats(page_number: int) -> PageTextStats:
    return PageTextStats(
        page_number=page_number,
        width=595.276,
        height=841.89,
        char_count=200,
        word_count=40,
        alnum_ratio=0.8,
        alpha_ratio=0.75,
        non_ascii_ratio=0.0,
        replacement_char_count=0,
        weird_symbol_ratio=0.0,
        average_word_length=5,
        repeated_garbage_score=0,
        image_count=0,
        image_area_ratio=0,
        has_selectable_text=True,
        looks_meaningful=True,
        looks_garbled=False,
        looks_image_dominant=False,
    )


class _FakeDetector:
    def __init__(self, classification: str) -> None:
        self.classification = classification

    def detect(self, _pdf_path: Path) -> PDFTypeDetectionResult:
        return PDFTypeDetectionResult(
            classification=self.classification,  # type: ignore[arg-type]
            page_count=1,
            pages=[
                PageTextStats(
                    page_number=1,
                    width=612,
                    height=792,
                    char_count=200,
                    word_count=40,
                    alnum_ratio=0.8,
                    alpha_ratio=0.75,
                    non_ascii_ratio=0.0,
                    replacement_char_count=0,
                    weird_symbol_ratio=0.0,
                    average_word_length=5,
                    repeated_garbage_score=0,
                    image_count=0,
                    image_area_ratio=0,
                    has_selectable_text=True,
                    looks_meaningful=True,
                    looks_garbled=False,
                    looks_image_dominant=False,
                )
            ],
            embedded_text_chars=200,
            embedded_text_words=40,
            meaningful_page_count=1,
            garbled_page_count=0,
            image_dominant_page_count=0,
            scanned_page_count=0,
            mixed=False,
            metadata={"filename": "input.pdf"},
        )


def _block(block_id: str, text: str):
    from app.models.schema import Block, BlockType

    return Block(
        id=block_id,
        page_number=1,
        block_type=BlockType.TABLE,
        text=text,
        bbox=None,
        reading_order_index=0,
        source_type=SourceType.OCR,
    )


def _paragraph_block(block_id: str, text: str):
    from app.models.schema import Block, BlockType

    return Block(
        id=block_id,
        page_number=1,
        block_type=BlockType.PARAGRAPH,
        text=text,
        bbox=None,
        reading_order_index=0,
        source_type=SourceType.OCR,
    )
