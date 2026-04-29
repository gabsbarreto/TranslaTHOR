from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.models.schema import DocumentModel


def write_translation_comparison_report(
    *,
    source_path: Path,
    translated_path: Path,
    document: DocumentModel,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source_path.read_text(encoding="utf-8", errors="ignore") if source_path.exists() else ""
    translated = translated_path.read_text(encoding="utf-8", errors="ignore") if translated_path.exists() else ""
    report = build_translation_comparison_report(
        source_text=source,
        translated_text=translated,
        source_path=source_path,
        translated_path=translated_path,
        document=document,
    )
    json_path = output_dir / "translation_comparison_report.json"
    md_path = output_dir / "translation_comparison_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_report_to_markdown(report), encoding="utf-8")
    return json_path, md_path


def build_translation_comparison_report(
    *,
    source_text: str,
    translated_text: str,
    source_path: Path | str,
    translated_path: Path | str,
    document: DocumentModel,
) -> dict[str, Any]:
    source_stats = _markdown_stats(source_text)
    translated_stats = _markdown_stats(translated_text)
    chunks = document.translation_chunks
    chunk_rows = [
        {
            "chunk_id": chunk.id,
            "source_char_count": len(chunk.source_text),
            "translated_char_count": len(chunk.translated_text),
            "source_token_count": chunk.source_token_count,
            "block_ids": chunk.block_ids,
            "source_start": chunk.source_text[:200],
            "source_end": chunk.source_text[-200:],
            "translated_start": chunk.translated_text[:200],
            "translated_end": chunk.translated_text[-200:],
        }
        for chunk in chunks
    ]
    empty = [row["chunk_id"] for row in chunk_rows if row["source_char_count"] and not row["translated_char_count"]]
    suspicious = [
        row["chunk_id"]
        for row in chunk_rows
        if row["source_char_count"] >= 200 and row["translated_char_count"] < row["source_char_count"] * 0.25
    ]
    malformed = _malformed_markdown_checks(source_text, translated_text)
    placeholder_report = _placeholder_report(source_text, translated_text)

    likely_failure = _likely_failure_point(
        source_stats=source_stats,
        translated_stats=translated_stats,
        empty_chunks=empty,
        suspicious_chunks=suspicious,
        malformed=malformed,
    )
    return {
        "source_md_path": str(source_path),
        "translated_md_path": str(translated_path),
        "source": source_stats,
        "translated": translated_stats,
        "source_chunk_count": len(chunks),
        "translated_chunk_count": sum(1 for chunk in chunks if chunk.translated_text.strip()),
        "missing_chunk_ids": [],
        "empty_translated_chunk_ids": empty,
        "suspiciously_short_chunk_ids": suspicious,
        "malformed_markdown": malformed,
        "placeholders": placeholder_report,
        "chunks": chunk_rows,
        "likely_failure_point": likely_failure,
        "recommended_fix": _recommended_fix(likely_failure),
    }


def _markdown_stats(text: str) -> dict[str, int]:
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    return {
        "character_count": len(text),
        "word_count": len(re.findall(r"\b\w+\b", text)),
        "paragraph_count": len(paragraphs),
        "heading_count": len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S", text)),
        "table_count": len(re.findall(r"(?is)<table\b", text)) + len(re.findall(r"(?m)^\s*\|.+\|\s*$", text)),
        "image_or_figure_placeholder_count": len(re.findall(r"!\[[^\]]*\]\([^)]+\)|\[\[FIGURE_[^\]]+\]\]", text)),
    }


def _malformed_markdown_checks(source: str, translated: str) -> dict[str, Any]:
    return {
        "source_table_open_count": len(re.findall(r"(?is)<table\b", source)),
        "source_table_close_count": len(re.findall(r"(?is)</table>", source)),
        "translated_table_open_count": len(re.findall(r"(?is)<table\b", translated)),
        "translated_table_close_count": len(re.findall(r"(?is)</table>", translated)),
        "translated_unclosed_table": len(re.findall(r"(?is)<table\b", translated)) != len(re.findall(r"(?is)</table>", translated)),
        "source_code_fence_count": source.count("```"),
        "translated_code_fence_count": translated.count("```"),
    }


def _placeholder_report(source: str, translated: str) -> dict[str, Any]:
    pattern = re.compile(r"\[\[[A-Z]+_[A-Z0-9_]+\]\]")
    source_placeholders = sorted(set(pattern.findall(source)))
    translated_placeholders = sorted(set(pattern.findall(translated)))
    return {
        "source_count": len(source_placeholders),
        "translated_count": len(translated_placeholders),
        "missing": [item for item in source_placeholders if item not in translated_placeholders],
        "extra": [item for item in translated_placeholders if item not in source_placeholders],
    }


def _likely_failure_point(
    *,
    source_stats: dict[str, int],
    translated_stats: dict[str, int],
    empty_chunks: list[str],
    suspicious_chunks: list[str],
    malformed: dict[str, Any],
) -> str:
    if empty_chunks:
        return "translation returned empty output for one or more chunks"
    if suspicious_chunks:
        return "translation output is suspiciously short for one or more chunks"
    if malformed.get("translated_unclosed_table"):
        return "translated markdown contains malformed table markup"
    if source_stats["paragraph_count"] and translated_stats["paragraph_count"] < max(1, source_stats["paragraph_count"] // 4):
        return "translated markdown appears to have collapsed many paragraphs"
    if source_stats["character_count"] and translated_stats["character_count"] < source_stats["character_count"] * 0.35:
        return "translated markdown is much shorter than source markdown"
    return "no obvious source/translation mismatch detected"


def _recommended_fix(likely_failure: str) -> str:
    if "empty output" in likely_failure or "suspiciously short" in likely_failure:
        return "Reduce effective translation chunk size and inspect the listed chunk IDs."
    if "malformed table" in likely_failure:
        return "Use table-specific translation fallback or preserve table markup without translation."
    if "collapsed" in likely_failure:
        return "Split source text at paragraph boundaries before translation and preserve markdown line breaks."
    if "much shorter" in likely_failure:
        return "Check chunk max_tokens and translated chunk recombination."
    return "Inspect PDF rebuild expectations if the translated markdown structure looks healthy."


def _report_to_markdown(report: dict[str, Any]) -> str:
    source = report["source"]
    translated = report["translated"]
    lines = [
        "# Translation Comparison Report",
        "",
        f"- Source: `{report['source_md_path']}`",
        f"- Translated: `{report['translated_md_path']}`",
        f"- Likely failure point: **{report['likely_failure_point']}**",
        f"- Recommended fix: {report['recommended_fix']}",
        "",
        "## Counts",
        "",
        "| Metric | Source | Translated |",
        "| --- | ---: | ---: |",
    ]
    for key in ("character_count", "word_count", "paragraph_count", "heading_count", "table_count", "image_or_figure_placeholder_count"):
        lines.append(f"| {key} | {source[key]} | {translated[key]} |")
    lines.extend(
        [
            "",
            "## Chunks",
            "",
            f"- Source chunks: {report['source_chunk_count']}",
            f"- Translated chunks: {report['translated_chunk_count']}",
            f"- Empty translated chunks: {', '.join(report['empty_translated_chunk_ids']) or 'none'}",
            f"- Suspiciously short chunks: {', '.join(report['suspiciously_short_chunk_ids']) or 'none'}",
        ]
    )
    return "\n".join(lines) + "\n"
