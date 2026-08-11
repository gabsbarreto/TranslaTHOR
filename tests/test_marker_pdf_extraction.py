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
from app.services.pdf_inspector import PdfInspector


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


def test_pdf_type_detector_discards_explicit_null_metadata(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    pdf_path = tmp_path / "null-metadata.pdf"
    source_path = tmp_path / "source.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        fitz.Rect(72, 72, 540, 720),
        ("Texte numérique normal avec suffisamment de mots pour la détection. " * 20),
        fontsize=11,
    )
    doc.save(source_path)
    doc.close()

    reader = pypdf.PdfReader(str(source_path))
    writer = pypdf.PdfWriter()
    writer.append_pages_from_reader(reader)
    writer._info.get_object()[pypdf.generic.NameObject("/Author")] = (  # noqa: SLF001
        pypdf.generic.NullObject()
    )
    with pdf_path.open("wb") as output:
        writer.write(output)

    result = PDFTypeDetector().detect(pdf_path)
    inspection = PdfInspector().inspect(pdf_path)

    assert result.classification == "digital_good_text"
    assert result.metadata["author"] is None
    assert inspection.author is None


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


def test_pdf_type_detector_treats_full_page_image_with_selectable_text_as_hidden_ocr(
    tmp_path: Path,
) -> None:
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


def test_marker_subprocess_failure_is_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extractor = PDFExtractor(detector=_FakeDetector("digital_good_text"))
    monkeypatch.setenv("MARKER_BIN", "/usr/bin/false")

    with pytest.raises(RuntimeError, match="Marker failed with exit code"):
        extractor.extract(tmp_path / "input.pdf", job_dir=tmp_path, keep_debug_artifacts=True)

    assert (tmp_path / "marker" / "marker_failure.json").exists()


def test_marker_debug_json_does_not_replace_primary_document_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_bin = tmp_path / "fake_marker_with_debug.py"
    marker_bin.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path
source = Path(sys.argv[1])
out = Path(sys.argv[sys.argv.index("--output_dir") + 1]) / source.stem
out.mkdir(parents=True, exist_ok=True)
(out / "blocks.json").write_text(json.dumps([{
  "block_type": "2", "text": "fine grained debug span"
}]), encoding="utf-8")
(out / f"{source.stem}_meta.json").write_text(json.dumps({
  "debug_data_path": "blocks.json"
}), encoding="utf-8")
(out / f"{source.stem}.json").write_text(json.dumps({
  "block_type": "Document",
  "children": [{
    "id": "/page/0/Page/1", "block_type": "Page",
    "polygon": [[0, 0], [600, 0], [600, 800], [0, 800]],
    "children": [{
      "id": "/page/0/Text/1", "block_type": "Text",
      "html": "<p>Canonical semantic paragraph.</p>",
      "polygon": [[50, 60], [550, 60], [550, 100], [50, 100]]
    }]
  }]
}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    marker_bin.chmod(0o755)
    monkeypatch.setenv("MARKER_BIN", str(marker_bin))

    result = PDFExtractor(detector=_FakeDetector("digital_good_text")).extract(
        tmp_path / "input.pdf",
        job_dir=tmp_path,
        keep_debug_artifacts=True,
    )

    assert [block["text"] for block in result.blocks] == ["Canonical semantic paragraph."]
    assert result.blocks[0]["bbox"] == {"x0": 50.0, "y0": 60.0, "x1": 550.0, "y1": 100.0}
    assert "fine grained debug span" not in result.markdown


def test_marker_payload_selection_rejects_debug_only_json(tmp_path: Path) -> None:
    output = tmp_path / "output" / "input"
    output.mkdir(parents=True)
    (output / "blocks.json").write_text(
        json.dumps(
            [
                {
                    "block_type": "8",
                    "children": [
                        {
                            "block_type": "2",
                            "polygon": {"bbox": [0, 0, 100, 20]},
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (output / "input_meta.json").write_text("{}", encoding="utf-8")

    selected = PDFExtractor()._find_marker_payload(  # noqa: SLF001
        tmp_path / "output",
        "json",
        source_stem="input",
    )

    assert selected is None


def test_marker_payload_selection_uses_nested_canonical_shape(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "unrelated.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    canonical = output / "nested" / "converted.json"
    canonical.parent.mkdir()
    canonical.write_text(
        json.dumps(
            {
                "block_type": "Document",
                "children": [
                    {
                        "block_type": "Page",
                        "children": [{"block_type": "Text", "text": "Canonical content"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    selected = PDFExtractor()._find_marker_payload(  # noqa: SLF001
        output,
        "json",
        source_stem="input",
    )

    assert selected == canonical


def test_marker_builder_rejects_numeric_debug_object_graph() -> None:
    debug_payload = [
        {
            "block_type": "8",
            "children": [
                {
                    "block_type": "2",
                    "text": "Debug span",
                    "polygon": {"bbox": [0, 0, 100, 20]},
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="refusing to interpret debug spans"):
        MarkerDocumentBuilder().build_document(
            marker_payload=debug_payload,
            detection=_FakeDetector("digital_good_text").detect(Path("input.pdf")),
            filename="input.pdf",
            source_type=SourceType.EMBEDDED,
            parser_metadata={},
            warnings=[],
        )


def test_marker_force_ocr_accelerator_failure_retries_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    assert all(
        (block.metadata or {}).get("marker_block_type") != "TableGroup" for block in document.blocks
    )
    assert len(document.tables) == 1
    assert "structured-table" not in markdown
    assert "### Table 1" not in markdown
    assert "*Table 1. Demographic data*" in markdown
    assert "<td>Age</td><td>53</td>" in markdown
    assert len([chunk for chunk in chunks if "<table" in chunk.original_text]) == 1


def test_marker_builder_anchors_trailing_table_footnote_without_sorting_page() -> None:
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
                    "polygon": [[40, 50], [540, 50], [540, 360], [40, 360]],
                    "children": [
                        {
                            "id": "/page/0/Table/2",
                            "block_type": "Table",
                            "html": "<table><tr><td>Value</td></tr></table>",
                            "polygon": [[40, 50], [540, 50], [540, 340], [40, 340]],
                        },
                        {
                            "id": "/page/0/Footnote/3",
                            "block_type": "Footnote",
                            "text": "First table note.",
                            "polygon": [[50, 342], [300, 342], [300, 351], [50, 351]],
                        },
                    ],
                },
                {
                    "id": "/page/0/Text/4",
                    "block_type": "Text",
                    "text": "First column body.",
                    "polygon": [[40, 410], [280, 410], [280, 600], [40, 600]],
                },
                {
                    "id": "/page/0/Text/5",
                    "block_type": "Text",
                    "text": "Second column body.",
                    "polygon": [[300, 410], [540, 410], [540, 600], [300, 600]],
                },
                {
                    "id": "/page/0/PageFooter/6",
                    "block_type": "PageFooter",
                    "text": "Journal footer.",
                    "polygon": [[40, 760], [540, 760], [540, 772], [40, 772]],
                },
                {
                    "id": "/page/0/Footnote/7",
                    "block_type": "Footnote",
                    "text": "Second table note.",
                    "polygon": [[50, 361], [390, 361], [390, 371], [50, 371]],
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

    assert [block.id for block in document.blocks] == [
        "/page/0/Table/2",
        "/page/0/Footnote/3",
        "/page/0/Footnote/7",
        "/page/0/Text/4",
        "/page/0/Text/5",
        "/page/0/PageFooter/6",
    ]
    assert markdown.count("First table note.") == 1
    assert markdown.count("Second table note.") == 1
    assert markdown.index("Second table note.") < markdown.index("First column body.")
    assert [chunk.block_ids[0] for chunk in chunks] == [block.id for block in document.blocks]


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


def test_bad_hidden_ocr_falls_back_to_normal_marker_if_forced_ocr_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_auto_bad_hidden_ocr_uses_balanced_marker_force_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker_bin = tmp_path / "fake_marker_text_only.py"
    marker_bin.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--output_dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "args.txt").write_text(json.dumps(sys.argv), encoding="utf-8")
(out / "env.json").write_text(json.dumps({
  "backend": os.environ.get("SURYA_INFERENCE_BACKEND"),
  "guided_layout": os.environ.get("SURYA_GUIDED_LAYOUT"),
  "keep_alive": os.environ.get("SURYA_INFERENCE_KEEP_ALIVE"),
}), encoding="utf-8")
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
    monkeypatch.delenv("SURYA_INFERENCE_BACKEND", raising=False)
    monkeypatch.delenv("SURYA_GUIDED_LAYOUT", raising=False)
    monkeypatch.delenv("SURYA_INFERENCE_KEEP_ALIVE", raising=False)

    result = PDFExtractor(detector=_FakeDetector("bad_hidden_ocr")).extract(
        tmp_path / "input.pdf",
        mode="auto",
        job_dir=tmp_path,
        keep_debug_artifacts=True,
    )

    args = json.loads((tmp_path / "marker" / "args.txt").read_text(encoding="utf-8"))
    marker_env = json.loads((tmp_path / "marker" / "env.json").read_text(encoding="utf-8"))
    assert args[args.index("--mode") + 1] == "balanced"
    assert "--disable_ocr" not in args
    assert "--force_ocr" in args
    assert "--strip_existing_ocr" in args
    assert result.used_ocr is True
    assert result.used_force_ocr is True
    assert result.metadata["marker_mode"] == "strip_existing_ocr_force_ocr"
    assert result.metadata["marker_conversion_mode"] == "balanced"
    assert marker_env == {
        "backend": "llamacpp",
        "guided_layout": "false",
        "keep_alive": "0",
    }


def test_marker_output_becomes_document_and_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_marker_payload_selection_prefers_document_over_debug_blocks(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "marker" / "document"
    output_dir.mkdir(parents=True)
    (output_dir / "blocks.json").write_text("[]", encoding="utf-8")
    document_path = output_dir / "document.json"
    document_path.write_text(
        json.dumps({"block_type": "Document", "children": []}),
        encoding="utf-8",
    )

    assert PDFExtractor()._find_marker_payload(tmp_path / "marker", "json") == document_path


def test_marker_builder_preserves_zero_based_page_id_for_page_range_retry() -> None:
    detection = PDFTypeDetectionResult(
        classification="digital_good_text",
        page_count=8,
        pages=[_page_stats(page_number) for page_number in range(1, 9)],
        embedded_text_chars=1000,
        embedded_text_words=160,
        meaningful_page_count=8,
        garbled_page_count=0,
        image_dominant_page_count=0,
        scanned_page_count=0,
        mixed=False,
    )
    payload = [
        {
            "id": "/page/7/Page/0",
            "page_id": 7,
            "block_type": "Page",
            "children": [
                {
                    "id": "/page/7/Text/0",
                    "block_type": "Text",
                    "text": "Page-local retry text.",
                }
            ],
        }
    ]

    document, _markdown, _chunks = MarkerDocumentBuilder().build_document(
        marker_payload=payload,
        detection=detection,
        filename="paper.pdf",
        source_type=SourceType.OCR,
        parser_metadata={},
        warnings=[],
    )

    assert document.blocks[0].page_number == 8


def test_marker_table_cells_are_not_duplicated_as_text_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_marker_builder_persists_table_cell_geometry_spans_and_order() -> None:
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
    html = (
        '<table><tr><th rowspan="2">Group</th><th colspan="2">Measures</th></tr>'
        "<tr><th>A</th><th>B</th></tr>"
        "<tr><td>Control</td><td>10</td><td>12</td></tr></table>"
    )

    def cell(
        name: str,
        text: str,
        row: int,
        column: int,
        polygon: list[list[int]],
        **extra,
    ) -> dict:
        return {
            "id": f"/page/0/TableCell/{name}",
            "block_type": "TableCell",
            "html": text,
            "row_index": row,
            "column_index": column,
            "polygon": polygon,
            "confidence": 0.93,
            **extra,
        }

    # Explicit coordinates make child emission order irrelevant. The final
    # malformed child must not shift geometry onto any logical table cell.
    children = [
        cell(
            "b",
            "B",
            1,
            2,
            [[200, 40], [300, 40], [300, 70], [200, 70]],
            colspan=999_999_999,
        ),
        cell(
            "group",
            "Group",
            0,
            0,
            [[10, 10], [100, 10], [100, 70], [10, 70]],
            rowspan=2,
        ),
        # A matched child with geometry outside the page retains provenance,
        # but its unsafe polygon and bbox are discarded.
        cell("12", "12", 2, 2, [[700, 70], [800, 70], [800, 100], [700, 100]]),
        cell(
            "measures",
            "Measures",
            0,
            1,
            [[100, 10], [300, 10], [300, 40], [100, 40]],
            colspan=2,
        ),
        cell("a", "A", 1, 1, [[100, 40], [200, 40], [200, 70], [100, 70]]),
        cell("control", "Control", 2, 0, [[10, 70], [100, 70], [100, 100], [10, 100]]),
        cell("10", "10", 2, 1, [[100, 70], [200, 70], [200, 100], [100, 100]]),
        {
            "id": "/page/0/TableCell/malformed",
            "block_type": "TableCell",
            "html": "not a real cell",
            "row_index": "bad",
            "polygon": [[999]],
        },
    ]
    payload = [
        {
            "id": "/page/0/Page/0",
            "block_type": "Page",
            "polygon": [[0, 0], [612, 0], [612, 792], [0, 792]],
            "children": [
                {
                    "id": "/page/0/Table/1",
                    "block_type": "Table",
                    "html": html,
                    "polygon": [[10, 10], [300, 10], [300, 100], [10, 100]],
                    "children": children,
                }
            ],
        }
    ]

    document, markdown, chunks = MarkerDocumentBuilder().build_document(
        marker_payload=payload,
        detection=detection,
        filename="paper.pdf",
        source_type=SourceType.EMBEDDED,
        parser_metadata={},
        warnings=[],
    )

    assert len(document.blocks) == 1
    assert len(chunks) == 1
    assert markdown.count("Control") == 1
    table = document.tables[0]
    assert table.headers == ["Group", "Measures"]
    assert [[cell.text for cell in row] for row in table.cells] == [
        ["A", "B"],
        ["Control", "10", "12"],
    ]
    assert [(cell.text, cell.row_index, cell.column_index) for cell in table.header_cells] == [
        ("Group", 0, 0),
        ("Measures", 0, 1),
    ]
    assert table.header_cells[0].rowspan == 2
    assert table.header_cells[1].colspan == 2
    assert table.header_cells[0].bbox is not None
    assert table.header_cells[0].bbox.model_dump() == {
        "x0": 10.0,
        "y0": 10.0,
        "x1": 100.0,
        "y1": 70.0,
    }
    assert table.header_cells[0].polygon == [
        [10.0, 10.0],
        [100.0, 10.0],
        [100.0, 70.0],
        [10.0, 70.0],
    ]
    assert table.header_cells[0].confidence == pytest.approx(0.93)
    assert table.cells[0][0].column_index == 1  # column zero is occupied by the rowspan
    assert table.cells[0][1].colspan == 1  # Reject unsafe Marker span overrides.
    assert table.cells[1][2].source_id == "/page/0/TableCell/12"
    assert table.cells[1][2].bbox is None
    assert table.cells[1][2].polygon == []
    assert table.debug["cell_geometry_source"] == "marker_table_cell_polygons"
    assert table.debug["cell_geometry_status"] == "partial"
    assert table.debug["cell_coordinate_space"] == {
        "name": "marker_page_coordinates",
        "width": 612.0,
        "height": 792.0,
    }
    assert table.debug["matched_marker_table_cell_count"] == 7
    assert table.debug["unmatched_marker_table_cell_count"] == 1
    assert table.debug["unmatched_logical_table_cell_count"] == 0
    assert table.debug["valid_marker_table_cell_geometry_count"] == 6
    assert table.debug["missing_logical_table_cell_geometry_count"] == 1
    assert table.debug["invalid_marker_table_cell_geometry_count"] == 1


def test_marker_builder_does_not_shift_incomplete_table_cell_geometry() -> None:
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
    payload = [
        {
            "id": "/page/0/Page/0",
            "block_type": "Page",
            "children": [
                {
                    "id": "/page/0/Table/1",
                    "block_type": "Table",
                    "html": "<table><tr><th>A</th><th>B</th></tr><tr><td>same</td><td>same</td></tr></table>",
                    "children": [
                        {
                            "id": "/page/0/TableCell/only-b",
                            "block_type": "TableCell",
                            "text": "B",
                            "polygon": [[100, 0], [200, 0], [200, 20], [100, 20]],
                        },
                        {
                            "id": "/page/0/TableCell/ambiguous",
                            "block_type": "TableCell",
                            "text": "same",
                            "polygon": [[0, 20], [100, 20], [100, 40], [0, 40]],
                        },
                    ],
                }
            ],
        }
    ]

    document, _, chunks = MarkerDocumentBuilder().build_document(
        marker_payload=payload,
        detection=detection,
        filename="paper.pdf",
        source_type=SourceType.EMBEDDED,
        parser_metadata={},
        warnings=[],
    )

    table = document.tables[0]
    assert table.header_cells[0].bbox is None
    assert table.header_cells[1].source_id == "/page/0/TableCell/only-b"
    assert table.cells[0][0].bbox is None
    assert table.cells[0][1].bbox is None
    assert table.debug["matched_marker_table_cell_count"] == 1
    assert table.debug["unmatched_marker_table_cell_count"] == 1
    assert table.debug["unmatched_logical_table_cell_count"] == 3
    assert table.debug["cell_geometry_status"] == "partial"
    assert table.debug["cell_geometry_source"] == "marker_table_cell_polygons"
    assert len(document.blocks) == 1
    assert len(chunks) == 1


@pytest.mark.parametrize(
    ("children", "expected_status", "expected_source", "expected_unmatched_logical"),
    [
        (
            [
                {
                    "id": "cell-a",
                    "block_type": "TableCell",
                    "text": "A",
                    "row_index": 0,
                    "column_index": 0,
                    "polygon": [[0, 0], [100, 0], [100, 20], [0, 20]],
                },
                {
                    "id": "cell-b",
                    "block_type": "TableCell",
                    "text": "B",
                    "row_index": 0,
                    "column_index": 1,
                    "polygon": [[100, 0], [200, 0], [200, 20], [100, 20]],
                },
            ],
            "complete",
            "marker_table_cell_polygons",
            0,
        ),
        ([], "unavailable", "unavailable", 2),
    ],
)
def test_marker_table_geometry_debug_distinguishes_complete_and_unavailable(
    children: list[dict],
    expected_status: str,
    expected_source: str,
    expected_unmatched_logical: int,
) -> None:
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
    payload = [
        {
            "id": "/page/0/Page/0",
            "block_type": "Page",
            "polygon": [[0, 0], [612, 0], [612, 792], [0, 792]],
            "children": [
                {
                    "id": "/page/0/Table/1",
                    "block_type": "Table",
                    "html": "<table><tr><td>A</td><td>B</td></tr></table>",
                    "polygon": [[0, 0], [200, 0], [200, 20], [0, 20]],
                    "children": children,
                }
            ],
        }
    ]

    document, _, _ = MarkerDocumentBuilder().build_document(
        marker_payload=payload,
        detection=detection,
        filename="paper.pdf",
        source_type=SourceType.EMBEDDED,
        parser_metadata={},
        warnings=[],
    )

    debug = document.tables[0].debug
    assert debug["cell_geometry_status"] == expected_status
    assert debug["cell_geometry_source"] == expected_source
    assert debug["unmatched_logical_table_cell_count"] == expected_unmatched_logical


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
    service = LocalVLMRepairService(
        LocalVLMConfig(False, "http://127.0.0.1:9/v1", "", "not-needed", 1, 0)
    )
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


def test_marker_document_language_sampling_includes_late_document_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.pdf_extraction import markdown_builder as marker_builder_module

    captured: list[str] = []
    monkeypatch.setattr(
        marker_builder_module,
        "detect",
        lambda text: captured.append(text) or "de",
    )
    blocks = [
        _paragraph_block("abstract", "English abstract terminology. " * 600),
        *[
            _paragraph_block(
                f"german-{index}",
                f"SPÄTERER-DEUTSCHER-ABSCHNITT-{index} Klinische Behandlung und Forschung.",
            )
            for index in range(20)
        ],
    ]

    language = MarkerDocumentBuilder()._detect_language(blocks)

    assert language == "de"
    assert len(captured) == 1
    assert "SPÄTERER-DEUTSCHER-ABSCHNITT-19" in captured[0]
    assert len(captured[0]) <= 50_000


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
    assert "SURYA <number>: <type>" in prompt
    assert '<region index="<number>" type="<type>">' in prompt
    assert "page headers, and page footers" in prompt


def test_qwen_fallback_runs_surya_worker_with_marker_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    output_dir = tmp_path / "layout"
    manifest = {"pages": [], "region_count": 0}
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
    monkeypatch.setattr(
        "app.services.pdf_extraction.qwen_ocr_fallback.subprocess.Popen",
        fake_popen,
    )

    def fake_communicate(*_args, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "layout.json").write_text(json.dumps(manifest), encoding="utf-8")
        return "", ""

    monkeypatch.setattr(fallback, "_communicate_with_cancel", fake_communicate)

    result = fallback._run_surya_layout(
        render_dir=tmp_path / "rendered",
        output_dir=output_dir,
        settings={"surya_layout_padding": 24, "surya_layout_batch_size": 2},
        cancel_requested=None,
        on_process_started=None,
        on_process_finished=None,
        on_ocr_progress=None,
    )

    assert result["pages"] == manifest["pages"]
    assert result["region_count"] == manifest["region_count"]
    assert result["surya_layout_attempt"] == "default"
    assert result["surya_layout_retried"] is False
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
    assert captured["kwargs"]["env"]["SURYA_LAYOUT_PYTHON"] == sys.executable


def test_qwen_fallback_retries_surya_layout_accelerator_failure_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    output_dir = tmp_path / "layout"
    manifest = {"pages": [], "region_count": 0}
    attempts: list[dict] = []

    class _FakeProcess:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def fake_popen(cmd, **kwargs):
        attempts.append({"cmd": cmd, "kwargs": kwargs})
        return _FakeProcess(1 if len(attempts) == 1 else 0)

    fallback = QwenFullPageOCRFallback()
    monkeypatch.setenv("SURYA_LAYOUT_PYTHON", sys.executable)
    monkeypatch.setenv("SURYA_LAYOUT_WORKER", str(tmp_path / "worker.py"))
    monkeypatch.setattr(
        "app.services.pdf_extraction.qwen_ocr_fallback.subprocess.Popen", fake_popen
    )

    def fake_communicate(process, *_args, **_kwargs):
        if process.returncode:
            return "", "torch.AcceleratorError: index 8192 is out of bounds"
        retry_output_dir = Path(attempts[-1]["cmd"][attempts[-1]["cmd"].index("--output-dir") + 1])
        retry_output_dir.mkdir(parents=True, exist_ok=True)
        (retry_output_dir / "layout.json").write_text(json.dumps(manifest), encoding="utf-8")
        return "", ""

    monkeypatch.setattr(fallback, "_communicate_with_cancel", fake_communicate)

    result = fallback._run_surya_layout(
        render_dir=tmp_path / "rendered",
        output_dir=output_dir,
        settings={"surya_layout_padding": 24},
        cancel_requested=None,
        on_process_started=None,
        on_process_finished=None,
        on_ocr_progress=None,
    )

    assert result["surya_layout_attempt"] == "cpu"
    assert result["surya_layout_retried"] is True
    assert len(attempts) == 2
    assert attempts[1]["kwargs"]["env"]["TORCH_DEVICE"] == "cpu"
    assert (output_dir / "layout.json").exists()


def test_qwen_fallback_stops_surya_layout_retry_for_non_accelerator_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    attempts: list[dict] = []

    class _FakeProcess:
        returncode = 1

    def fake_popen(cmd, **kwargs):
        attempts.append({"cmd": cmd, "kwargs": kwargs})
        return _FakeProcess()

    fallback = QwenFullPageOCRFallback()
    monkeypatch.setenv("SURYA_LAYOUT_PYTHON", sys.executable)
    monkeypatch.setenv("SURYA_LAYOUT_WORKER", str(tmp_path / "worker.py"))
    monkeypatch.setattr(
        "app.services.pdf_extraction.qwen_ocr_fallback.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        fallback,
        "_communicate_with_cancel",
        lambda *_args, **_kwargs: ("", "RuntimeError: invalid image file"),
    )

    with pytest.raises(RuntimeError, match="after accelerator-safe retries"):
        fallback._run_surya_layout(
            render_dir=tmp_path / "rendered",
            output_dir=tmp_path / "layout",
            settings={},
            cancel_requested=None,
            on_process_started=None,
            on_process_finished=None,
            on_ocr_progress=None,
        )

    assert len(attempts) == 1


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
