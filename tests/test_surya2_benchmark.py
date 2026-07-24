from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_ocr_engines.py"
SPEC = importlib.util.spec_from_file_location("benchmark_ocr_engines", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_ocr_error_metrics() -> None:
    assert benchmark.levenshtein_distance("kitten", "sitting") == 3
    assert benchmark.character_error_rate("abc", "adc") == pytest.approx(1 / 3)
    assert benchmark.word_error_rate("one two three", "one too three") == pytest.approx(1 / 3)
    assert benchmark.insertion_deletion_counts("one two", "one extra two") == (0, 1)
    assert benchmark.parse_darwin_footprint(
        "python [123]: 64-bit    Footprint: 12.3 GB (16384 bytes per page)"
    ) == int(12.3 * 1024**3)


def test_layout_and_reading_order_accuracy() -> None:
    assert benchmark.sequence_accuracy(["Title", "Text", "Table"], ["Title", "Text", "Table"]) == 1
    assert benchmark.sequence_accuracy(
        ["Title", "Text", "Table"], ["Title", "Table"]
    ) == pytest.approx(2 / 3)
    assert (
        benchmark.reading_order_accuracy(
            ["first", "second", "third"], "first then second then third"
        )
        == 1
    )
    assert benchmark.reading_order_accuracy(
        ["first", "second", "third"], "third then first then second"
    ) == pytest.approx(1 / 3)


def test_reference_pdf_uses_selected_pages(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    reference_pdf = tmp_path / "reference.pdf"
    with reference_pdf.open("wb") as stream:
        writer.write(stream)

    assert (
        benchmark.resolve_reference_text(
            {"reference_pdf": "reference.pdf"},
            manifest_dir=tmp_path,
            selected_pages=[1],
        )
        == ""
    )
