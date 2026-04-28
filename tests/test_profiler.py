from pathlib import Path

from app.services.profiler import PipelineProfiler


def test_profiler_records_and_writes(tmp_path: Path) -> None:
    profiler = PipelineProfiler(enabled=True)
    profiler.record("page_rendering", 0.4, page=1)
    profiler.record("ocr_recognition", 1.2, page=1)
    profiler.record("ocr_recognition", 1.4, page=2)

    json_path, csv_path, summary_path = profiler.dump(tmp_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()
    assert "ocr_recognition" in summary_path.read_text(encoding="utf-8")
