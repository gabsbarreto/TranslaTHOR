from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

import app.main as main
import app.services.job_store as job_store_module
from app.models.schema import (
    Block,
    BlockType,
    DocumentMetadata,
    DocumentModel,
    JobStage,
    PageMetadata,
    SourceType,
    TableModel,
)
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
    store.update_status(
        job_id,
        stage=JobStage.COMPLETE,
        artifacts={"markdown": str(markdown_path)},
    )

    first_pdf = main._ensure_pdf_artifact(job_id, "pdf_readable")
    markdown_path.write_text("second version", encoding="utf-8")
    second_pdf = main._ensure_pdf_artifact(job_id, "pdf_readable")

    assert first_pdf == second_pdf
    assert second_pdf.name == "Original File_translated.pdf"
    assert fake_reconstructor.html_to_pdf_calls == 2
    assert "render=2" in second_pdf.read_text(encoding="utf-8")
    assert "second version" in second_pdf.read_text(encoding="utf-8")
    assert store.load_status(job_id).artifacts["pdf_readable"] == str(second_pdf)


def test_translated_pdf_is_rejected_until_translation_is_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(job_store_module, "JOBS_DIR", jobs_dir)
    store = JobStore()
    monkeypatch.setattr(main, "job_store", store)
    job_id, job_dir = store.create_job("Incomplete.pdf")
    markdown_path = job_dir / "artifacts" / "translated.md"
    markdown_path.write_text("partial output", encoding="utf-8")
    store.update_status(job_id, artifacts={"markdown": str(markdown_path)})

    with pytest.raises(HTTPException) as error:
        main._ensure_pdf_artifact(job_id, "pdf_readable")

    assert error.value.status_code == 409


def test_readable_artifact_uses_translated_table_block_from_structured_json(
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

    job_id, job_dir = store.create_job("Scanned Table.pdf")
    structured_path = job_dir / "artifacts" / "structured.json"
    document = DocumentModel(
        metadata=DocumentMetadata(filename="Scanned Table.pdf", page_count=1),
        pages=[
            PageMetadata(
                page_number=1,
                width=600,
                height=800,
                has_embedded_text=True,
                embedded_text_quality=0.1,
                extraction_mode=SourceType.OCR,
            )
        ],
        blocks=[
            Block(
                id="table-block",
                page_number=1,
                block_type=BlockType.TABLE,
                text="| English header |\n|---|\n| English cell |",
                reading_order_index=0,
                source_type=SourceType.OCR,
                metadata={
                    "source_text_before_cleaning": (
                        "| Spanish header |\n|---|\n| Spanish cell |"
                    ),
                    "translated_from_block_ids": ["table-block"],
                },
            ),
            Block(
                id="caption",
                page_number=1,
                block_type=BlockType.CAPTION,
                text="Table I. English caption",
                reading_order_index=1,
                source_type=SourceType.OCR,
            ),
        ],
        tables=[
            TableModel(
                id="stale-table",
                page_numbers=[1],
                headers=["Spanish header"],
                rows=[["Spanish cell"]],
                cells=[[TableModel.TableCell(text="Spanish cell")]],
            )
        ],
    )
    structured_path.write_text(document.model_dump_json(), encoding="utf-8")
    store.update_status(
        job_id,
        stage=JobStage.COMPLETE,
        artifacts={"json": str(structured_path)},
    )

    pdf_path = main._ensure_pdf_artifact(job_id, "pdf_readable")
    rendered = pdf_path.read_text(encoding="utf-8")

    assert "English header" in rendered
    assert "English cell" in rendered
    assert "Spanish header" not in rendered
    assert "Spanish cell" not in rendered
    assert rendered.count("Table I. English caption") == 1


class _FakeOriginalLayoutReconstructor:
    def reconstruct(
        self,
        *,
        source_pdf_path: Path,
        output_pdf_path: Path,
        document: DocumentModel,
        report_path: Path,
    ) -> dict:
        assert source_pdf_path.name == "input.pdf"
        assert document.metadata.filename == "Original.pdf"
        output_pdf_path.write_bytes(b"%PDF-fake")
        report_path.write_text("{}", encoding="utf-8")
        return {
            "status": "partial",
            "pages_successfully_reconstructed": 0,
            "pages_using_fallback_behavior": 1,
            "warnings": [{"reason": "synthetic warning"}],
        }


def test_original_layout_artifact_and_report_are_registered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(job_store_module, "JOBS_DIR", jobs_dir)
    store = JobStore()
    monkeypatch.setattr(main, "job_store", store)
    monkeypatch.setattr(
        main,
        "original_layout_reconstructor",
        _FakeOriginalLayoutReconstructor(),
    )
    job_id, job_dir = store.create_job("Original.pdf")
    (job_dir / "input.pdf").write_bytes(b"source")
    structured_path = job_dir / "artifacts" / "structured.json"
    document = DocumentModel(
        metadata=DocumentMetadata(filename="Original.pdf", page_count=0),
        pages=[],
        blocks=[],
    )
    structured_path.write_text(document.model_dump_json(), encoding="utf-8")
    store.update_status(
        job_id,
        stage=JobStage.COMPLETE,
        artifacts={"json": str(structured_path)},
    )

    pdf_path = main._ensure_pdf_artifact(job_id, "pdf_original_layout")
    status = store.load_status(job_id)

    assert pdf_path.name == "Original_translated_original_layout.pdf"
    assert status.artifacts["pdf_original_layout"] == str(pdf_path)
    assert Path(status.artifacts["reconstruction_report"]).is_file()
    assert status.translation["original_layout_reconstruction"]["status"] == "partial"
    assert any("safe fallback" in warning for warning in status.translation["warnings"])


def test_original_layout_route_accepts_hyphenated_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.pdf"
    output.write_bytes(b"%PDF-fake")
    seen: list[str] = []

    def fake_ensure(_job_id: str, artifact_type: str) -> Path:
        seen.append(artifact_type)
        return output

    monkeypatch.setattr(main, "_ensure_pdf_artifact", fake_ensure)

    response = main.get_pdf("job-id", "original-layout")

    assert seen == ["pdf_original_layout"]
    assert Path(response.path) == output
