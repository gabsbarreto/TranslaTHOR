from __future__ import annotations

import importlib.util
import json
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


def test_marker_balanced_benchmark_uses_production_digital_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    sentinel = object()

    class FakeExtractor:
        def extract(self, **kwargs):
            captured.update(kwargs)
            return sentinel

    class FakeSampler:
        def track(self, _process) -> None:
            pass

    monkeypatch.setattr(benchmark, "PDFExtractor", FakeExtractor)
    engine = benchmark.BenchmarkEngine("marker_balanced", dpi=192, timeout=600)

    result = engine.extract(
        pdf_path=tmp_path / "paper.pdf",
        job_dir=tmp_path / "run",
        classification="digital_good_text",
        detection_metadata={},
        warnings=[],
        sampler=FakeSampler(),
    )

    assert result is sentinel
    assert captured["mode"] == "digital"
    assert captured["keep_debug_artifacts"] is False
    assert captured["marker_config"] == {
        "lowres_image_dpi": 192,
        "highres_image_dpi": 192,
        "pdftext_workers": 1,
    }


def test_completed_benchmark_run_is_checkpointed(tmp_path: Path) -> None:
    result = benchmark.failed_metrics(
        document_id="paper",
        engine="marker_balanced",
        run_type="cold",
        run_index=0,
        dpi=192,
        page_count=11,
        wall_seconds=12.5,
        peak_rss_bytes=1234,
        error=RuntimeError("deliberate test failure"),
    )
    environment = {
        "timestamp_utc": "2026-08-10T00:00:00+00:00",
        "platform": "test",
        "machine": "arm64",
        "python": "3.test",
        "memory_bytes": 1234,
        "versions": {},
        "configuration": {},
    }
    manifest = {"documents": [{"id": "paper", "pages": list(range(1, 12))}]}

    benchmark.write_benchmark_outputs(
        output_dir=tmp_path,
        environment=environment,
        manifest=manifest,
        results=[result],
        command="benchmark --cold-only",
    )

    payload = json.loads((tmp_path / "benchmark_results.json").read_text())
    report = (tmp_path / "benchmark_report.md").read_text()
    assert payload["results"][0]["engine"] == "marker_balanced"
    assert payload["results"][0]["wall_seconds"] == 12.5
    assert "deliberate test failure" in report
    assert not list(tmp_path.glob("*.tmp"))


def test_surya_batching_reserves_full_context_for_each_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "SURYA_INFERENCE_PARALLEL",
        "SURYA_INFERENCE_CTX_PER_SLOT",
        "SURYA_INFERENCE_CTX_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    total_context = benchmark.configure_surya_batching(
        parallel=5,
        context_per_slot=16384,
    )

    assert total_context == 81920
    assert benchmark.os.environ["SURYA_INFERENCE_PARALLEL"] == "5"
    assert benchmark.os.environ["SURYA_INFERENCE_CTX_PER_SLOT"] == "16384"
    assert benchmark.os.environ["SURYA_INFERENCE_CTX_SIZE"] == "81920"


@pytest.mark.parametrize("parallel, context", [(0, 16384), (5, 0)])
def test_surya_batching_rejects_non_positive_values(parallel: int, context: int) -> None:
    with pytest.raises(ValueError):
        benchmark.configure_surya_batching(
            parallel=parallel,
            context_per_slot=context,
        )
