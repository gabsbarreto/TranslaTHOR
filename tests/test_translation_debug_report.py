from __future__ import annotations

from app.models.schema import DocumentMetadata, DocumentModel, TranslationChunk
from app.services.translation_debug import build_translation_comparison_report


def test_translation_report_flags_empty_and_short_chunks() -> None:
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1),
        pages=[],
        blocks=[],
        translation_chunks=[
            TranslationChunk(
                id="chunk-0", block_ids=["b1"], source_text="a" * 300, translated_text=""
            ),
            TranslationChunk(
                id="chunk-1", block_ids=["b2"], source_text="b" * 400, translated_text="short"
            ),
        ],
    )

    report = build_translation_comparison_report(
        source_text="# Source\n\n" + "a" * 700,
        translated_text="# Translated\n\nshort",
        source_path="source.md",
        translated_path="translated.md",
        document=document,
    )

    assert report["source_chunk_count"] == 2
    assert report["translated_chunk_count"] == 1
    assert "chunk-0" in report["empty_translated_chunk_ids"]
    assert "chunk-1" in report["suspiciously_short_chunk_ids"]
    assert "empty output" in report["likely_failure_point"]


def test_translation_report_prioritizes_target_language_validation_failures() -> None:
    document = DocumentModel(
        metadata=DocumentMetadata(filename="paper.pdf", page_count=1),
        pages=[],
        blocks=[],
        translation_chunks=[
            TranslationChunk(
                id="chunk-es",
                block_ids=["b1"],
                source_text="Los pacientes reciben tratamiento hormonal.",
                translated_text="Los pacientes reciben tratamiento hormonal.",
                status="translation_failed",
                reason="translation_output_matches_source",
            )
        ],
    )

    report = build_translation_comparison_report(
        source_text="Los pacientes reciben tratamiento hormonal.",
        translated_text="Los pacientes reciben tratamiento hormonal.",
        source_path="source.md",
        translated_path="translated.md",
        document=document,
    )

    assert report["failed_translation_validation_chunk_ids"] == ["chunk-es"]
    assert report["failed_translation_validation_reason_counts"] == {
        "translation_output_matches_source": 1
    }
    assert report["chunks"][0]["reason"] == "translation_output_matches_source"
    assert "English-output validation failed" in report["likely_failure_point"]
    assert "source text was preserved" in report["recommended_fix"]
