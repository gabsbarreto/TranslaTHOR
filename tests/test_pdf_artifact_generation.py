from __future__ import annotations

from pathlib import Path

import pytest

import app.main as main
import app.services.job_store as job_store_module
from app.services.job_store import JobStore


class _FakeReconstructor:
    def __init__(self) -> None:
        self.html_to_pdf_calls = 0

    def markdown_to_html(self, markdown_text: str, title: str | None = None, output_mode: str = "readable") -> str:
        return f"mode={output_mode}; title={title}; body={markdown_text.strip()}"

    def html_to_pdf(self, html_text: str, pdf_path: Path) -> None:
        self.html_to_pdf_calls += 1
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text(f"render={self.html_to_pdf_calls}; {html_text}", encoding="utf-8")


def test_readable_pdf_is_regenerated_with_original_translated_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(job_store_module, "JOBS_DIR", jobs_dir)
    store = JobStore()
    fake_reconstructor = _FakeReconstructor()
    monkeypatch.setattr(main, "job_store", store)
    monkeypatch.setattr(main, "reconstructor", fake_reconstructor)

    job_id, job_dir = store.create_job("Original File.pdf")
    markdown_path = job_dir / "artifacts" / "translated.md"
    markdown_path.write_text("first version", encoding="utf-8")
    store.update_status(job_id, artifacts={"markdown": str(markdown_path)})

    first_pdf = main._ensure_pdf_artifact(job_id, "pdf_readable")
    markdown_path.write_text("second version", encoding="utf-8")
    second_pdf = main._ensure_pdf_artifact(job_id, "pdf_readable")

    assert first_pdf == second_pdf
    assert second_pdf.name == "Original File_translated.pdf"
    assert fake_reconstructor.html_to_pdf_calls == 2
    assert "render=2" in second_pdf.read_text(encoding="utf-8")
    assert "second version" in second_pdf.read_text(encoding="utf-8")
    assert store.load_status(job_id).artifacts["pdf_readable"] == str(second_pdf)
