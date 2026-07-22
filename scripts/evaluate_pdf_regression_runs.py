#!/usr/bin/env python3
"""Evaluate completed TranslaTHOR regression jobs without running any models.

The evaluator deliberately reports only observable pipeline, structure, artifact,
and reconstruction-safety signals. It does not attempt to score translation
meaning or visual quality.

Example:

    .venv/bin/python scripts/evaluate_pdf_regression_runs.py \
        workspace/regression_runs/2026-07-22-baseline/jobs
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = ROOT / "tests" / "regression_corpus" / "corpus_spec.json"
DEFAULT_IDENTICAL_CHUNK_MIN_CHARS = 80
SEVERE_COLLAPSE_MIN_SOURCE_CHARACTERS = 40
SEVERE_COLLAPSE_MAX_LENGTH_RATIO = 0.05
GEOMETRY_TOLERANCE_POINTS = 0.05

EVALUATION_SCOPE = (
    "This report measures pipeline completion, extracted structure, artifact integrity, "
    "and reconstruction safety signals. It does not score translation meaning, rendered "
    "visual quality, or pixel preservation outside edited regions; source-identical text "
    "is only a review signal."
)

AUTOMATED_CHECKS = (
    "PDF files open and their page geometry (count, dimensions, rotation, crop box, and "
    "media box) can be compared, including source/extraction counts against the corpus spec.",
    "Structured blocks, tables, figures, bounding boxes, and populated translation chunks "
    "can be counted; severe punctuation-only or near-empty translation collapses are flagged.",
    "Original-layout reconstruction reports can be checked for skips, overflow, scale-floor "
    "violations, and source-character-weighted replacement coverage.",
)

VERIFICATION_LIMITATIONS = (
    "Translation meaning, fluency, terminology consistency, and target-language correctness "
    "are not scored without a reference translation or human review.",
    "Readable and original-layout pages are not rendered or visually compared by this "
    "evaluator, so figure, table, typography, and reading-order fidelity remain unverified.",
    "Matching page geometry does not prove that pixels or visual objects outside approved "
    "text regions were preserved.",
    "Character-weighted reconstruction coverage uses source_character_count values emitted "
    "by the reconstruction report; it measures reported text replacement, not visual or "
    "semantic success.",
)


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _normalise_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).casefold()


def _visible_text(value: Any) -> str:
    """Return compact visible text for structural translation checks."""

    if not isinstance(value, str):
        return ""
    without_markup = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return " ".join(without_markup.split())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _display_path(path: Path | None, job_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(job_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _inspect_text_file(path: Path | None, job_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "present": bool(path and path.is_file()),
        "path": _display_path(path, job_dir),
        "character_count": 0,
        "non_whitespace_character_count": 0,
        "sha256": None,
        "error": None,
    }
    if not path or not path.is_file():
        return result
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(
        character_count=len(text),
        non_whitespace_character_count=sum(not char.isspace() for char in text),
        sha256=_sha256_bytes(data),
    )
    return result


def _resolve_artifact(
    job_dir: Path,
    status: dict[str, Any],
    artifact_key: str,
    *,
    candidates: Iterable[str] = (),
    glob_patterns: Iterable[str] = (),
) -> Path | None:
    artifacts = status.get("artifacts")
    configured = artifacts.get(artifact_key) if isinstance(artifacts, dict) else None
    if isinstance(configured, str) and configured.strip():
        configured_path = Path(configured).expanduser()
        if configured_path.is_absolute():
            attempts = []
            try:
                configured_path.relative_to(job_dir.resolve())
                attempts.append(configured_path)
            except ValueError:
                # Do not silently inspect an artifact from another run merely because a
                # copied status file still contains its old absolute workspace path.
                pass
            attempts.append(job_dir / "artifacts" / configured_path.name)
        else:
            attempts = (job_dir / configured_path, job_dir / "artifacts" / configured_path)
        for attempt in attempts:
            if attempt.is_file():
                return attempt

    for candidate in candidates:
        path = job_dir / candidate
        if path.is_file():
            return path

    matches: list[Path] = []
    for pattern in glob_patterns:
        matches.extend(path for path in job_dir.glob(pattern) if path.is_file())
    return sorted(set(matches))[0] if matches else None


def _artifact_paths(job_dir: Path, status: dict[str, Any]) -> dict[str, Path | None]:
    return {
        "structured": _resolve_artifact(
            job_dir,
            status,
            "json",
            candidates=("artifacts/structured.json",),
        ),
        "source_markdown": _resolve_artifact(
            job_dir,
            status,
            "source_markdown",
            candidates=("artifacts/source.md",),
        ),
        "translated_markdown": _resolve_artifact(
            job_dir,
            status,
            "markdown",
            candidates=("artifacts/translated.md",),
        ),
        "readable_pdf": _resolve_artifact(
            job_dir,
            status,
            "pdf_readable",
            candidates=("artifacts/translated_readable.pdf",),
            glob_patterns=("artifacts/*_translated.pdf",),
        ),
        "original_layout_pdf": _resolve_artifact(
            job_dir,
            status,
            "pdf_original_layout",
            candidates=("artifacts/translated_original_layout.pdf",),
            glob_patterns=("artifacts/*_translated_original_layout.pdf",),
        ),
        "reconstruction_report": _resolve_artifact(
            job_dir,
            status,
            "reconstruction_report",
            candidates=("artifacts/reconstruction_report_original_layout.json",),
        ),
    }


def _pdf_geometry(page: fitz.Page, page_number: int) -> dict[str, Any]:
    rect = page.rect
    cropbox = page.cropbox
    mediabox = page.mediabox
    return {
        "page_number": page_number,
        "width": round(float(rect.width), 4),
        "height": round(float(rect.height), 4),
        "rotation": int(page.rotation),
        "cropbox": [round(float(value), 4) for value in cropbox],
        "mediabox": [round(float(value), 4) for value in mediabox],
    }


def _inspect_pdf(path: Path | None, job_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "present": bool(path and path.is_file()),
        "path": _display_path(path, job_dir),
        "size_bytes": None,
        "opens": False,
        "page_count": None,
        "geometry": [],
        "error": None,
    }
    if not path or not path.is_file():
        return result
    try:
        result["size_bytes"] = path.stat().st_size
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError("PDF is password protected")
            result["page_count"] = document.page_count
            result["geometry"] = [
                _pdf_geometry(page, page_number)
                for page_number, page in enumerate(document, start=1)
            ]
        result["opens"] = True
    except Exception as exc:  # PyMuPDF exposes several format-specific exception classes.
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _numbers_match(left: Any, right: Any) -> bool:
    left_number = _as_float(left)
    right_number = _as_float(right)
    return bool(
        left_number is not None
        and right_number is not None
        and abs(left_number - right_number) <= GEOMETRY_TOLERANCE_POINTS
    )


def _geometry_comparison(source: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "comparable": bool(source.get("opens") and output.get("opens")),
        "matches_source": None,
        "source_page_count": source.get("page_count"),
        "output_page_count": output.get("page_count"),
        "page_count_matches_source": None,
        "pages_compared": 0,
        "geometry_tolerance_points": GEOMETRY_TOLERANCE_POINTS,
        "mismatched_pages": [],
        "geometry_match_does_not_verify_visual_preservation": True,
    }
    if not comparison["comparable"]:
        return comparison

    source_geometry = source.get("geometry", [])
    output_geometry = output.get("geometry", [])
    comparison["page_count_matches_source"] = len(source_geometry) == len(output_geometry)
    comparison["pages_compared"] = min(len(source_geometry), len(output_geometry))
    if not comparison["page_count_matches_source"]:
        comparison["mismatched_pages"].append(
            {
                "page_number": None,
                "reasons": [f"page_count:{len(source_geometry)}->{len(output_geometry)}"],
            }
        )

    for index, (source_page, output_page) in enumerate(
        zip(source_geometry, output_geometry), start=1
    ):
        reasons: list[str] = []
        for key in ("width", "height"):
            if not _numbers_match(source_page.get(key), output_page.get(key)):
                reasons.append(key)
        if source_page.get("rotation") != output_page.get("rotation"):
            reasons.append("rotation")
        for key in ("cropbox", "mediabox"):
            source_box = source_page.get(key)
            output_box = output_page.get(key)
            if not (
                isinstance(source_box, list)
                and isinstance(output_box, list)
                and len(source_box) == len(output_box)
                and all(_numbers_match(a, b) for a, b in zip(source_box, output_box))
            ):
                reasons.append(key)
        if reasons:
            comparison["mismatched_pages"].append({"page_number": index, "reasons": reasons})
    comparison["matches_source"] = not comparison["mismatched_pages"]
    return comparison


def _page_dimensions(structured: dict[str, Any]) -> tuple[dict[int, tuple[float, float]], int]:
    dimensions: dict[int, tuple[float, float]] = {}
    invalid_geometry = 0
    pages = structured.get("pages")
    if not isinstance(pages, list):
        return dimensions, invalid_geometry
    for page in pages:
        if not isinstance(page, dict):
            invalid_geometry += 1
            continue
        page_number = _as_int(page.get("page_number"), -1)
        width = _as_float(page.get("width"))
        height = _as_float(page.get("height"))
        if page_number < 1 or width is None or height is None or width <= 0 or height <= 0:
            invalid_geometry += 1
            continue
        dimensions[page_number] = (width, height)
    return dimensions, invalid_geometry


def _bbox_state(
    bbox: Any,
    page_number: int,
    dimensions: dict[int, tuple[float, float]],
    *,
    coordinate_dimensions: tuple[float, float] | None = None,
) -> str:
    if bbox is None:
        return "missing"
    if not isinstance(bbox, dict):
        return "invalid"
    values = [_as_float(bbox.get(key)) for key in ("x0", "y0", "x1", "y1")]
    if any(value is None for value in values):
        return "invalid"
    x0, y0, x1, y1 = (float(value) for value in values if value is not None)
    if x0 >= x1 or y0 >= y1:
        return "invalid"
    page_size = coordinate_dimensions or dimensions.get(page_number)
    if page_size is None:
        return "unverifiable"
    width, height = page_size
    tolerance = 1.0
    if x0 < -tolerance or y0 < -tolerance or x1 > width + tolerance or y1 > height + tolerance:
        return "outside_page"
    return "valid"


def _coordinate_dimensions(
    entity_type: str,
    entity: dict[str, Any],
) -> tuple[float, float] | None:
    """Return the dimensions of the coordinate space used by an entity bbox.

    Qwen/Surya and Marker retain boxes in rendered-page coordinates until
    reconstruction, while page metadata uses PDF points. Figure extraction,
    however, materialises its canonical bbox in PDF points and records the
    prior Surya box only as audit metadata. Keep those cases distinct.
    """

    if entity_type == "figures":
        extraction = entity.get("extraction_metadata")
        if isinstance(extraction, dict):
            conversion = extraction.get("coordinate_conversion")
            if isinstance(conversion, dict) and isinstance(conversion.get("pdf_bbox"), dict):
                return None
            metadata = extraction
        else:
            metadata = {}
    elif entity_type == "tables":
        metadata = entity.get("debug") if isinstance(entity.get("debug"), dict) else {}
    else:
        metadata = entity.get("metadata") if isinstance(entity.get("metadata"), dict) else {}

    coordinate_space = metadata.get("coordinate_space")
    if isinstance(coordinate_space, dict):
        name = str(coordinate_space.get("name", "")).lower()
        width = _as_float(coordinate_space.get("width"))
        height = _as_float(coordinate_space.get("height"))
        if (
            width is not None
            and height is not None
            and width > 0
            and height > 0
            and name not in {"pdf", "pdf_points", "unresolved"}
        ):
            return width, height

    for width_key, height_key in (
        ("surya_page_width", "surya_page_height"),
        ("marker_page_width", "marker_page_height"),
    ):
        width = _as_float(metadata.get(width_key))
        height = _as_float(metadata.get(height_key))
        if width is not None and height is not None and width > 0 and height > 0:
            return width, height
    return None


def _entity_page_number(entity_type: str, entity: dict[str, Any]) -> int:
    if entity_type == "tables":
        page = _as_int(entity.get("page"), -1)
        if page >= 1:
            return page
        pages = entity.get("page_numbers")
        if isinstance(pages, list) and pages:
            return _as_int(pages[0], -1)
    return _as_int(entity.get("page_number"), -1)


def _bbox_metrics(
    structured: dict[str, Any], dimensions: dict[int, tuple[float, float]]
) -> dict[str, Any]:
    states = ("valid", "missing", "invalid", "outside_page", "unverifiable")
    by_entity: dict[str, dict[str, int]] = {}
    problem_ids: dict[str, list[str]] = {
        "invalid": [],
        "outside_page": [],
        "unverifiable": [],
    }
    for entity_type in ("blocks", "tables", "figures"):
        counts = Counter({state: 0 for state in states})
        entities = structured.get(entity_type)
        if not isinstance(entities, list):
            entities = []
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                counts["invalid"] += 1
                problem_ids["invalid"].append(f"{entity_type}[{index}]")
                continue
            state = _bbox_state(
                entity.get("bbox"),
                _entity_page_number(entity_type, entity),
                dimensions,
                coordinate_dimensions=_coordinate_dimensions(entity_type, entity),
            )
            counts[state] += 1
            if state in problem_ids:
                problem_ids[state].append(str(entity.get("id", f"{entity_type}[{index}]")))
        by_entity[entity_type] = dict(counts)

    return {
        "by_entity": by_entity,
        "missing_count": sum(values["missing"] for values in by_entity.values()),
        "invalid_count": sum(values["invalid"] for values in by_entity.values()),
        "outside_page_count": sum(values["outside_page"] for values in by_entity.values()),
        "unverifiable_count": sum(values["unverifiable"] for values in by_entity.values()),
        "problem_ids": problem_ids,
    }


def _translation_chunk_metrics(chunks: Any, *, identical_min_chars: int) -> dict[str, Any]:
    if not isinstance(chunks, list):
        chunks = []
    source_bearing = [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict) and _normalise_text(chunk.get("source_text"))
    ]
    translated = [
        chunk for chunk in source_bearing if _normalise_text(chunk.get("translated_text"))
    ]
    empty_ids = [
        str(chunk.get("id", "<unknown>"))
        for chunk in source_bearing
        if not _normalise_text(chunk.get("translated_text"))
    ]
    failed_validation_ids = [
        str(chunk.get("id", "<unknown>"))
        for chunk in source_bearing
        if str(chunk.get("status", "")).lower() == "translation_failed"
    ]
    failed_validation_reasons = Counter(
        str(chunk.get("reason") or "unspecified")
        for chunk in source_bearing
        if str(chunk.get("status", "")).lower() == "translation_failed"
    )
    severe_collapses: list[dict[str, Any]] = []
    for chunk in translated:
        source = _visible_text(chunk.get("source_text"))
        target = _visible_text(chunk.get("translated_text"))
        source_alpha = sum(character.isalpha() for character in source)
        target_alpha = sum(character.isalpha() for character in target)
        if (
            len(source) < SEVERE_COLLAPSE_MIN_SOURCE_CHARACTERS
            or source_alpha < SEVERE_COLLAPSE_MIN_SOURCE_CHARACTERS // 2
            or not target
        ):
            continue
        length_ratio = len(target) / len(source)
        reason = None
        if target_alpha == 0:
            reason = "non_language_target"
        elif length_ratio <= SEVERE_COLLAPSE_MAX_LENGTH_RATIO and target_alpha <= 3:
            reason = "severe_length_collapse"
        if reason is None:
            continue
        severe_collapses.append(
            {
                "id": str(chunk.get("id", "<unknown>")),
                "page_start": chunk.get("page_start"),
                "chunk_type": chunk.get("chunk_type"),
                "reason": reason,
                "source_visible_character_count": len(source),
                "target_visible_character_count": len(target),
                "target_alphabetic_character_count": target_alpha,
                "length_ratio": round(length_ratio, 6),
            }
        )
    identical: list[dict[str, Any]] = []
    for chunk in translated:
        source = _normalise_text(chunk.get("source_text"))
        target = _normalise_text(chunk.get("translated_text"))
        if len(source) < identical_min_chars or source != target:
            continue
        identical.append(
            {
                "id": str(chunk.get("id", "<unknown>")),
                "page_start": chunk.get("page_start"),
                "chunk_type": chunk.get("chunk_type"),
                "normalised_character_count": len(source),
            }
        )
    total = len(source_bearing)
    return {
        "total_count": len(chunks),
        "source_bearing_count": total,
        "translated_nonempty_count": len(translated),
        "empty_translation_count": len(empty_ids),
        "completion_ratio": round(len(translated) / total, 6) if total else None,
        "empty_translation_ids": empty_ids,
        "failed_translation_validation_count": len(failed_validation_ids),
        "failed_translation_validation_ids": failed_validation_ids,
        "failed_translation_validation_reason_counts": dict(
            sorted(failed_validation_reasons.items())
        ),
        "severe_structural_collapse_count": len(severe_collapses),
        "severe_structural_collapse_chunks": severe_collapses,
        "severe_structural_collapse_min_source_characters": (
            SEVERE_COLLAPSE_MIN_SOURCE_CHARACTERS
        ),
        "severe_structural_collapse_max_length_ratio": SEVERE_COLLAPSE_MAX_LENGTH_RATIO,
        "source_identical_long_count": len(identical),
        "source_identical_long_chunks": identical,
        "source_identical_min_characters": identical_min_chars,
        "source_identical_is_review_signal_only": True,
    }


def _figure_preview_metrics(
    figures: list[Any],
    structured_path: Path,
    job_dir: Path,
) -> dict[str, Any]:
    present_ids: list[str] = []
    missing_ids: list[str] = []
    invalid_ids: list[str] = []
    for index, figure in enumerate(figures):
        if not isinstance(figure, dict):
            continue
        figure_id = str(figure.get("id", f"figures[{index}]"))
        configured = figure.get("image_path")
        if not isinstance(configured, str) or not configured.strip():
            missing_ids.append(figure_id)
            continue
        configured_path = Path(configured).expanduser()
        candidates = (
            [configured_path]
            if configured_path.is_absolute()
            else [
                structured_path.parent / configured_path,
                job_dir / configured_path,
                job_dir / "artifacts" / configured_path,
            ]
        )
        # A copied run may retain a stale absolute workspace path. Resolve the
        # stable filename inside this job before declaring the asset missing.
        candidates.append(job_dir / "artifacts" / "figures" / configured_path.name)
        preview_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if preview_path is None:
            missing_ids.append(figure_id)
            continue
        try:
            with fitz.open(preview_path) as preview:
                if preview.page_count < 1 or preview[0].rect.is_empty:
                    raise ValueError("empty preview")
        except Exception:
            invalid_ids.append(figure_id)
            continue
        present_ids.append(figure_id)
    return {
        "valid_count": len(present_ids),
        "valid_ids": present_ids,
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids,
        "invalid_count": len(invalid_ids),
        "invalid_ids": invalid_ids,
    }


def _inspect_structure(
    path: Path | None,
    job_dir: Path,
    *,
    identical_min_chars: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "present": bool(path and path.is_file()),
        "path": _display_path(path, job_dir),
        "error": None,
        "metadata_page_count": None,
        "page_count": 0,
        "invalid_page_geometry_count": 0,
        "block_count": 0,
        "block_type_counts": {},
        "table_count": 0,
        "table_parse_mode_counts": {},
        "figure_count": 0,
        "figure_asset_type_counts": {},
        "figures_with_preview": 0,
        "figure_previews": {
            "valid_count": 0,
            "valid_ids": [],
            "missing_count": 0,
            "missing_ids": [],
            "invalid_count": 0,
            "invalid_ids": [],
        },
        "bbox": {},
        "translation_chunks": _translation_chunk_metrics(
            [], identical_min_chars=identical_min_chars
        ),
    }
    if not path or not path.is_file():
        return result
    structured, error = _read_json(path)
    if error or not isinstance(structured, dict):
        result["error"] = error or "Top-level JSON value is not an object"
        return result

    metadata = structured.get("metadata")
    if isinstance(metadata, dict):
        result["metadata_page_count"] = metadata.get("page_count")
    pages = structured.get("pages") if isinstance(structured.get("pages"), list) else []
    blocks = structured.get("blocks") if isinstance(structured.get("blocks"), list) else []
    tables = structured.get("tables") if isinstance(structured.get("tables"), list) else []
    figures = structured.get("figures") if isinstance(structured.get("figures"), list) else []
    dimensions, invalid_page_geometry = _page_dimensions(structured)
    figure_previews = _figure_preview_metrics(figures, path, job_dir)

    result.update(
        page_count=len(pages),
        invalid_page_geometry_count=invalid_page_geometry,
        block_count=len(blocks),
        block_type_counts=dict(
            sorted(
                Counter(
                    str(block.get("block_type", "unknown"))
                    for block in blocks
                    if isinstance(block, dict)
                ).items()
            )
        ),
        table_count=len(tables),
        table_parse_mode_counts=dict(
            sorted(
                Counter(
                    str(table.get("parse_mode", "unknown"))
                    for table in tables
                    if isinstance(table, dict)
                ).items()
            )
        ),
        figure_count=len(figures),
        figure_asset_type_counts=dict(
            sorted(
                Counter(
                    str(figure.get("asset_type", "unknown"))
                    for figure in figures
                    if isinstance(figure, dict)
                ).items()
            )
        ),
        figures_with_preview=figure_previews["valid_count"],
        figure_previews=figure_previews,
        bbox=_bbox_metrics(structured, dimensions),
        translation_chunks=_translation_chunk_metrics(
            structured.get("translation_chunks"),
            identical_min_chars=identical_min_chars,
        ),
    )
    return result


def _count_value(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return max(_as_int(value, 0), 0)


def _skip_reason_counts(report: dict[str, Any]) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    regions = report.get("regions")
    if isinstance(regions, list):
        for region in regions:
            if not isinstance(region, dict) or str(region.get("status", "")).lower() != "skipped":
                continue
            reasons[str(region.get("reason") or "unspecified")] += 1
    if reasons:
        return dict(sorted(reasons.items()))

    configured = report.get("skip_reasons")
    if isinstance(configured, dict):
        for reason, count in configured.items():
            reasons[str(reason)] += max(_as_int(count, 0), 0)
    if reasons:
        return dict(sorted(reasons.items()))

    # Older reports only described skipped regions in warning strings.
    warnings = report.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            if isinstance(warning, dict):
                if warning.get("code") != "region_skipped":
                    continue
                text = str(warning.get("reason", ""))
            elif isinstance(warning, str):
                text = warning
            else:
                continue
            match = re.search(r"skipped:\s*([^.]+)", text)
            reasons[match.group(1) if match else "unspecified"] += 1
    return dict(sorted(reasons.items()))


def _source_character_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _reported_character_coverage(regions: list[Any]) -> dict[str, Any]:
    """Summarise report-emitted source character counts for terminal regions.

    This deliberately uses only the reconstruction report. Missing counts are exposed rather
    than estimated from block IDs or translated text because either estimate could overstate
    how much source text was actually considered for replacement.
    """

    replaced_characters = 0
    skipped_characters = 0
    retained_characters = 0
    regions_with_count = 0
    regions_missing_count = 0
    terminal_region_count = 0
    by_page: dict[int | str, dict[str, int]] = {}

    for region in regions:
        if not isinstance(region, dict):
            continue
        status = str(region.get("status", "")).lower()
        if status not in {"replaced", "committed", "skipped", "retained"}:
            continue
        terminal_region_count += 1
        character_count = _source_character_count(region.get("source_character_count"))
        if character_count is None:
            regions_missing_count += 1
            continue
        regions_with_count += 1
        page_number = _as_int(region.get("page_number"), -1)
        page_key: int | str = page_number if page_number >= 1 else "unknown"
        page_counts = by_page.setdefault(
            page_key,
            {
                "replaced_source_characters": 0,
                "skipped_source_characters": 0,
                "retained_source_characters": 0,
            },
        )
        if status in {"replaced", "committed"}:
            replaced_characters += character_count
            page_counts["replaced_source_characters"] += character_count
        elif status == "skipped":
            skipped_characters += character_count
            page_counts["skipped_source_characters"] += character_count
        else:
            retained_characters += character_count
            page_counts["retained_source_characters"] += character_count

    accounted_characters = replaced_characters + skipped_characters
    all_reported_characters = accounted_characters + retained_characters

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    page_rows = []
    for page_number, counts in sorted(
        by_page.items(),
        key=lambda item: (isinstance(item[0], str), item[0]),
    ):
        page_total = counts["replaced_source_characters"] + counts["skipped_source_characters"]
        page_rows.append(
            {
                "page_number": page_number,
                **counts,
                "accounted_source_characters": page_total,
                "all_reported_source_characters": (
                    page_total + counts["retained_source_characters"]
                ),
                "reported_character_replacement_ratio": ratio(
                    counts["replaced_source_characters"], page_total
                ),
            }
        )

    return {
        "available": regions_with_count > 0,
        "complete_for_terminal_regions": (terminal_region_count > 0 and regions_missing_count == 0),
        "terminal_region_count": terminal_region_count,
        "regions_with_source_character_count": regions_with_count,
        "regions_missing_source_character_count": regions_missing_count,
        "replaced_source_characters": replaced_characters,
        "skipped_source_characters": skipped_characters,
        "retained_source_characters": retained_characters,
        "accounted_source_characters": accounted_characters,
        "all_reported_source_characters": all_reported_characters,
        "reported_character_replacement_ratio": ratio(replaced_characters, accounted_characters),
        "reported_character_skip_ratio": ratio(skipped_characters, accounted_characters),
        "by_page": page_rows,
        "interpretation": (
            "Ratio of report-emitted source characters in replaced versus failed/skipped "
            "eligible regions; intentionally retained regions are reported separately. This is "
            "not a translation-accuracy or visual-fidelity score."
        ),
    }


def _inspect_reconstruction_report(path: Path | None, job_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "present": bool(path and path.is_file()),
        "path": _display_path(path, job_dir),
        "error": None,
        "status": None,
        "total_pages": None,
        "pages_successfully_reconstructed": 0,
        "pages_using_fallback_behavior": 0,
        "figures_preserved": 0,
        "regions_replaced": 0,
        "regions_skipped": 0,
        "regions_retained": 0,
        "regions_missing_or_invalid_bboxes": 0,
        "text_boxes_did_not_fit": 0,
        "configured_minimum_text_scale": None,
        "actual_minimum_text_scale": None,
        "scaled_region_count": 0,
        "below_minimum_scale_count": 0,
        "raster_figure_fallback_count": 0,
        "low_confidence_association_count": 0,
        "skip_reason_counts": {},
        "warning_count": 0,
        "warning_code_counts": {},
        "count_consistency_warnings": [],
        "reported_character_coverage": _reported_character_coverage([]),
    }
    if not path or not path.is_file():
        return result
    report, error = _read_json(path)
    if error or not isinstance(report, dict):
        result["error"] = error or "Top-level JSON value is not an object"
        return result

    scales: list[float] = []
    scaling = report.get("scaling_applied")
    if isinstance(scaling, list):
        for item in scaling:
            if not isinstance(item, dict):
                continue
            scale = _as_float(item.get("scale"))
            if scale is not None:
                scales.append(scale)
    configured_minimum = _as_float(report.get("minimum_text_scale"))
    skip_reasons = _skip_reason_counts(report)
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    warning_codes = Counter(
        str(warning.get("code", "unspecified")) for warning in warnings if isinstance(warning, dict)
    )
    regions = report.get("regions") if isinstance(report.get("regions"), list) else []
    character_coverage = _reported_character_coverage(regions)
    derived_replaced = sum(
        isinstance(region, dict)
        and str(region.get("status", "")).lower() in {"replaced", "committed"}
        for region in regions
    )
    derived_skipped = sum(skip_reasons.values())
    declared_replaced = max(_as_int(report.get("regions_replaced"), 0), 0)
    declared_skipped = max(_as_int(report.get("regions_skipped"), 0), 0)
    declared_retained = max(_as_int(report.get("regions_retained"), 0), 0)
    derived_retained = sum(
        isinstance(region, dict) and str(region.get("status", "")).lower() == "retained"
        for region in regions
    )
    total_pages = max(_as_int(report.get("total_pages"), 0), 0)
    successful_pages = max(
        _as_int(report.get("pages_successfully_reconstructed"), 0), 0
    )
    fallback_pages = max(
        _as_int(report.get("pages_using_fallback_behavior"), 0), 0
    )
    page_reports = report.get("pages") if isinstance(report.get("pages"), list) else []
    consistency: list[str] = []
    if regions and declared_replaced != derived_replaced:
        consistency.append(
            f"regions_replaced declared {declared_replaced}, derived {derived_replaced}"
        )
    if regions and declared_skipped != derived_skipped:
        consistency.append(
            f"regions_skipped declared {declared_skipped}, derived {derived_skipped}"
        )
    if regions and declared_retained != derived_retained:
        consistency.append(
            f"regions_retained declared {declared_retained}, derived {derived_retained}"
        )
    if page_reports and total_pages != len(page_reports):
        consistency.append(
            f"total_pages declared {total_pages}, page reports contain {len(page_reports)}"
        )
    if total_pages and successful_pages + fallback_pages != total_pages:
        consistency.append(
            "pages_successfully_reconstructed plus pages_using_fallback_behavior "
            f"is {successful_pages + fallback_pages}, expected {total_pages}"
        )

    result.update(
        status=report.get("status"),
        total_pages=total_pages,
        pages_successfully_reconstructed=successful_pages,
        pages_using_fallback_behavior=fallback_pages,
        figures_preserved=max(_as_int(report.get("figures_preserved"), 0), 0),
        regions_replaced=declared_replaced,
        regions_skipped=declared_skipped,
        regions_retained=declared_retained,
        regions_missing_or_invalid_bboxes=_count_value(
            report.get("regions_missing_or_invalid_bboxes")
        ),
        text_boxes_did_not_fit=_count_value(report.get("text_boxes_did_not_fit")),
        configured_minimum_text_scale=configured_minimum,
        actual_minimum_text_scale=round(min(scales), 6) if scales else None,
        scaled_region_count=len(scales),
        below_minimum_scale_count=sum(
            configured_minimum is not None and scale < configured_minimum - 1e-6 for scale in scales
        ),
        raster_figure_fallback_count=_count_value(report.get("raster_figure_fallbacks")),
        low_confidence_association_count=_count_value(
            report.get("low_confidence_figure_or_caption_associations")
        ),
        skip_reason_counts=skip_reasons,
        warning_count=len(warnings),
        warning_code_counts=dict(sorted(warning_codes.items())),
        count_consistency_warnings=consistency,
        reported_character_coverage=character_coverage,
    )
    return result


def _normalise_attempt_filename(filename: str) -> str:
    path = Path(filename)
    stem = re.sub(r"\s+\(\d+\)$", "", path.stem)
    return f"{stem}{path.suffix}"


def _case_id_from_status(status: dict[str, Any], case_ids: set[str]) -> str | None:
    for container_key in ("settings", "translation"):
        container = status.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("regression_case_id", "case_id"):
            value = container.get(key)
            if isinstance(value, str) and value in case_ids:
                return value
    for key in ("source_filename", "filename"):
        value = status.get(key)
        if not isinstance(value, str):
            continue
        normalised = _normalise_attempt_filename(value)
        stem = Path(normalised).stem
        if stem in case_ids:
            return stem
    return None


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, str, float]:
    status = candidate["status"]
    timestamp = next(
        (
            str(status[key])
            for key in ("completed_at", "started_at", "queued_at", "created_at")
            if status.get(key)
        ),
        "",
    )
    return (
        _as_int(status.get("attempt"), 0),
        timestamp,
        candidate["status_path"].stat().st_mtime,
    )


def _load_job_candidates(
    jobs_dir: Path, case_ids: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    matched: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in case_ids}
    unmatched: list[dict[str, Any]] = []
    if not jobs_dir.is_dir():
        return matched, unmatched
    for status_path in sorted(jobs_dir.glob("*/status.json")):
        status, error = _read_json(status_path)
        if error or not isinstance(status, dict):
            unmatched.append(
                {
                    "job_dir": str(status_path.parent),
                    "status_error": error or "Top-level JSON value is not an object",
                }
            )
            continue
        candidate = {
            "job_dir": status_path.parent,
            "status_path": status_path,
            "status": status,
        }
        case_id = _case_id_from_status(status, case_ids)
        if case_id is None:
            unmatched.append(
                {
                    "job_dir": str(status_path.parent),
                    "job_id": status.get("job_id"),
                    "source_filename": status.get("source_filename") or status.get("filename"),
                    "status_error": None,
                }
            )
            continue
        matched[case_id].append(candidate)
    for candidates in matched.values():
        candidates.sort(key=_candidate_sort_key, reverse=True)
    return matched, unmatched


def _page_count_validation(
    expected_value: Any,
    source_pdf: dict[str, Any],
    structure: dict[str, Any],
) -> dict[str, Any]:
    expected = _as_int(expected_value, -1)
    expected = expected if expected >= 1 else None
    source_count = _as_int(source_pdf.get("page_count"), -1)
    source_count = source_count if source_count >= 0 else None
    structured_count = _as_int(structure.get("page_count"), -1)
    structured_count = structured_count if structured_count >= 0 else None
    source_matches = (
        source_count == expected if expected is not None and source_count is not None else None
    )
    structured_matches = (
        structured_count == expected
        if expected is not None and structured_count is not None
        else None
    )
    comparisons = [value for value in (source_matches, structured_matches) if value is not None]
    return {
        "expected": expected,
        "source_pdf": source_count,
        "structured_document": structured_count,
        "source_matches_expected": source_matches,
        "structured_matches_expected": structured_matches,
        "all_available_counts_match_expected": all(comparisons) if comparisons else None,
    }


def _structural_outcome(case: dict[str, Any]) -> str:
    job = case["job"]
    if not job["present"]:
        return "missing_run"
    stage = job["stage"]
    if stage == "failed":
        return "pipeline_failed"
    if stage == "cancelled":
        return "pipeline_cancelled"
    if stage != "complete":
        return "in_progress"
    structure = case["structure"]
    readable = case["pdfs"]["readable"]
    original = case["pdfs"]["original_layout"]
    reconstruction = case["reconstruction"]
    chunks = structure["translation_chunks"]
    if not structure["present"] or structure["error"] or not readable["opens"]:
        return "completed_missing_core_artifacts"
    if case["page_count_validation"]["all_available_counts_match_expected"] is False:
        return "completed_with_page_count_mismatch"
    if structure["figure_previews"]["missing_count"] or structure["figure_previews"][
        "invalid_count"
    ]:
        return "completed_missing_core_artifacts"
    if (
        chunks["empty_translation_count"]
        or chunks["failed_translation_validation_count"]
        or chunks["severe_structural_collapse_count"]
    ):
        return "completed_with_translation_gaps"
    if not original["opens"] or not reconstruction["present"] or reconstruction["error"]:
        return "completed_missing_reconstruction_artifacts"
    if (
        original["geometry_comparison"]["matches_source"] is False
        or reconstruction["count_consistency_warnings"]
    ):
        return "completed_with_reconstruction_errors"
    if (
        reconstruction["status"] not in (None, "complete")
        or reconstruction["regions_skipped"]
        or reconstruction["text_boxes_did_not_fit"]
        or reconstruction["below_minimum_scale_count"]
    ):
        return "completed_with_reconstruction_warnings"
    return "structurally_complete"


def _empty_case_report(case_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_spec["id"],
        "language": case_spec.get("language"),
        "features": case_spec.get("features", []),
        "expected_classification": case_spec.get("expected_classification"),
        "expected_page_count": case_spec.get("expected_page_count"),
        "page_count_validation": {
            "expected": case_spec.get("expected_page_count"),
            "source_pdf": None,
            "structured_document": None,
            "source_matches_expected": None,
            "structured_matches_expected": None,
            "all_available_counts_match_expected": None,
        },
        "job": {
            "present": False,
            "job_id": None,
            "job_dir": None,
            "selected_attempt": None,
            "ignored_older_attempts": [],
            "stage": None,
            "progress": None,
            "message": None,
            "error": None,
        },
        "classification": {
            "actual": None,
            "matches_expected": None,
        },
        "structure": {
            "present": False,
            "path": None,
            "error": None,
            "page_count": 0,
            "block_count": 0,
            "table_count": 0,
            "figure_count": 0,
            "translation_chunks": _translation_chunk_metrics(
                [], identical_min_chars=DEFAULT_IDENTICAL_CHUNK_MIN_CHARS
            ),
        },
        "markdown": {
            "source": {"present": False},
            "translated": {"present": False},
            "normalised_contents_equal": None,
        },
        "pdfs": {
            "source": {"present": False, "opens": False},
            "readable": {"present": False, "opens": False},
            "original_layout": {
                "present": False,
                "opens": False,
                "geometry_comparison": {
                    "comparable": False,
                    "matches_source": None,
                    "source_page_count": None,
                    "output_page_count": None,
                    "page_count_matches_source": None,
                    "pages_compared": 0,
                    "geometry_tolerance_points": GEOMETRY_TOLERANCE_POINTS,
                    "mismatched_pages": [],
                    "geometry_match_does_not_verify_visual_preservation": True,
                },
            },
        },
        "reconstruction": {
            "present": False,
            "error": None,
            "status": None,
            "regions_replaced": 0,
            "regions_skipped": 0,
            "text_boxes_did_not_fit": 0,
            "actual_minimum_text_scale": None,
            "skip_reason_counts": {},
            "reported_character_coverage": _reported_character_coverage([]),
        },
        "structural_outcome": "missing_run",
    }


def _evaluate_case(
    case_spec: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    identical_min_chars: int,
) -> dict[str, Any]:
    if not candidates:
        result = _empty_case_report(case_spec)
        result["structure"]["translation_chunks"] = _translation_chunk_metrics(
            [], identical_min_chars=identical_min_chars
        )
        return result

    selected = candidates[0]
    job_dir: Path = selected["job_dir"]
    status: dict[str, Any] = selected["status"]
    paths = _artifact_paths(job_dir, status)
    structure = _inspect_structure(
        paths["structured"],
        job_dir,
        identical_min_chars=identical_min_chars,
    )
    source_markdown = _inspect_text_file(paths["source_markdown"], job_dir)
    translated_markdown = _inspect_text_file(paths["translated_markdown"], job_dir)
    source_pdf = _inspect_pdf(job_dir / "input.pdf", job_dir)
    readable_pdf = _inspect_pdf(paths["readable_pdf"], job_dir)
    original_pdf = _inspect_pdf(paths["original_layout_pdf"], job_dir)
    original_pdf["geometry_comparison"] = _geometry_comparison(source_pdf, original_pdf)
    reconstruction = _inspect_reconstruction_report(paths["reconstruction_report"], job_dir)
    page_count_validation = _page_count_validation(
        case_spec.get("expected_page_count"),
        source_pdf,
        structure,
    )
    translation = status.get("translation") if isinstance(status.get("translation"), dict) else {}
    actual_classification = translation.get("pdf_classification")
    expected_classification = case_spec.get("expected_classification")
    source_normalised = None
    translated_normalised = None
    if paths["source_markdown"] and source_markdown["error"] is None:
        try:
            source_normalised = _normalise_text(
                paths["source_markdown"].read_text(encoding="utf-8")
            )
        except OSError:
            pass
    if paths["translated_markdown"] and translated_markdown["error"] is None:
        try:
            translated_normalised = _normalise_text(
                paths["translated_markdown"].read_text(encoding="utf-8")
            )
        except OSError:
            pass

    result = {
        "case_id": case_spec["id"],
        "language": case_spec.get("language"),
        "features": case_spec.get("features", []),
        "expected_classification": expected_classification,
        "expected_page_count": case_spec.get("expected_page_count"),
        "page_count_validation": page_count_validation,
        "job": {
            "present": True,
            "job_id": status.get("job_id") or job_dir.name,
            "job_dir": str(job_dir.resolve()),
            "selected_attempt": status.get("attempt"),
            "ignored_older_attempts": [
                candidate["status"].get("job_id") or candidate["job_dir"].name
                for candidate in candidates[1:]
            ],
            "stage": status.get("stage"),
            "progress": status.get("progress"),
            "message": status.get("message"),
            "error": status.get("error"),
        },
        "classification": {
            "actual": actual_classification,
            "matches_expected": (
                actual_classification == expected_classification
                if actual_classification is not None
                else None
            ),
        },
        "structure": structure,
        "markdown": {
            "source": source_markdown,
            "translated": translated_markdown,
            "normalised_contents_equal": (
                source_normalised == translated_normalised
                if source_normalised is not None and translated_normalised is not None
                else None
            ),
        },
        "pdfs": {
            "source": source_pdf,
            "readable": readable_pdf,
            "original_layout": original_pdf,
        },
        "reconstruction": reconstruction,
    }
    result["structural_outcome"] = _structural_outcome(result)
    return result


def _flatten_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for group in ("digital_cases", "scanned_cases"):
        entries = spec.get(group)
        if not isinstance(entries, list):
            raise ValueError(f"Spec field {group!r} must be a list")
        for case in entries:
            if not isinstance(case, dict) or not isinstance(case.get("id"), str):
                raise ValueError(f"Every {group} entry must be an object with a string id")
            expected_page_count = case.get("expected_page_count")
            if expected_page_count is not None and (
                isinstance(expected_page_count, bool)
                or not isinstance(expected_page_count, int)
                or expected_page_count < 1
            ):
                raise ValueError(
                    f"Regression case {case['id']} has an invalid expected_page_count"
                )
            copied = dict(case)
            copied["corpus_group"] = group.removesuffix("_cases")
            cases.append(copied)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Regression corpus case IDs must be unique")
    return cases


def _build_summary(
    cases: list[dict[str, Any]], unmatched_jobs: list[dict[str, Any]]
) -> dict[str, Any]:
    matched = [case for case in cases if case["job"]["present"]]
    stage_counts = Counter(str(case["job"]["stage"] or "missing") for case in cases)
    outcome_counts = Counter(case["structural_outcome"] for case in cases)
    classification_available = [
        case for case in matched if case["classification"]["matches_expected"] is not None
    ]
    page_count_available = [
        case
        for case in matched
        if case["page_count_validation"]["all_available_counts_match_expected"] is not None
    ]
    chunks = [case["structure"]["translation_chunks"] for case in matched]
    validation_reasons: Counter[str] = Counter()
    for chunk in chunks:
        validation_reasons.update(chunk["failed_translation_validation_reason_counts"])
    translated_chunks = sum(chunk["translated_nonempty_count"] for chunk in chunks)
    source_chunks = sum(chunk["source_bearing_count"] for chunk in chunks)
    reconstruction_cases = [case["reconstruction"] for case in matched]
    skip_reasons: Counter[str] = Counter()
    for reconstruction in reconstruction_cases:
        skip_reasons.update(reconstruction.get("skip_reason_counts", {}))
    character_coverage = [
        reconstruction.get("reported_character_coverage", {})
        for reconstruction in reconstruction_cases
    ]
    replaced_source_characters = sum(
        _as_int(item.get("replaced_source_characters"), 0) for item in character_coverage
    )
    skipped_source_characters = sum(
        _as_int(item.get("skipped_source_characters"), 0) for item in character_coverage
    )
    retained_source_characters = sum(
        _as_int(item.get("retained_source_characters"), 0) for item in character_coverage
    )
    accounted_source_characters = replaced_source_characters + skipped_source_characters

    return {
        "expected_case_count": len(cases),
        "matched_case_count": len(matched),
        "missing_case_count": len(cases) - len(matched),
        "unmatched_job_count": len(unmatched_jobs),
        "stage_counts": dict(sorted(stage_counts.items())),
        "structural_outcome_counts": dict(sorted(outcome_counts.items())),
        "classification": {
            "available_count": len(classification_available),
            "matches_expected_count": sum(
                case["classification"]["matches_expected"] is True
                for case in classification_available
            ),
            "mismatch_count": sum(
                case["classification"]["matches_expected"] is False
                for case in classification_available
            ),
        },
        "page_counts": {
            "available_count": len(page_count_available),
            "matches_expected_count": sum(
                case["page_count_validation"]["all_available_counts_match_expected"] is True
                for case in page_count_available
            ),
            "mismatch_count": sum(
                case["page_count_validation"]["all_available_counts_match_expected"] is False
                for case in page_count_available
            ),
        },
        "structure": {
            "blocks": sum(case["structure"].get("block_count", 0) for case in matched),
            "tables": sum(case["structure"].get("table_count", 0) for case in matched),
            "figures": sum(case["structure"].get("figure_count", 0) for case in matched),
            "invalid_bboxes": sum(
                case["structure"].get("bbox", {}).get("invalid_count", 0) for case in matched
            ),
            "outside_page_bboxes": sum(
                case["structure"].get("bbox", {}).get("outside_page_count", 0) for case in matched
            ),
        },
        "translation_chunks": {
            "source_bearing_count": source_chunks,
            "translated_nonempty_count": translated_chunks,
            "empty_translation_count": source_chunks - translated_chunks,
            "completion_ratio": (
                round(translated_chunks / source_chunks, 6) if source_chunks else None
            ),
            "source_identical_long_count": sum(
                chunk["source_identical_long_count"] for chunk in chunks
            ),
            "failed_translation_validation_count": sum(
                chunk["failed_translation_validation_count"] for chunk in chunks
            ),
            "failed_translation_validation_reason_counts": dict(
                sorted(validation_reasons.items())
            ),
            "severe_structural_collapse_count": sum(
                chunk["severe_structural_collapse_count"] for chunk in chunks
            ),
        },
        "pdfs": {
            "readable_open_count": sum(case["pdfs"]["readable"]["opens"] for case in matched),
            "original_layout_open_count": sum(
                case["pdfs"]["original_layout"]["opens"] for case in matched
            ),
            "original_layout_geometry_match_count": sum(
                case["pdfs"]["original_layout"]["geometry_comparison"]["matches_source"] is True
                for case in matched
            ),
        },
        "reconstruction": {
            "regions_replaced": sum(item["regions_replaced"] for item in reconstruction_cases),
            "regions_skipped": sum(item["regions_skipped"] for item in reconstruction_cases),
            "regions_retained": sum(item["regions_retained"] for item in reconstruction_cases),
            "text_boxes_did_not_fit": sum(
                item["text_boxes_did_not_fit"] for item in reconstruction_cases
            ),
            "skip_reason_counts": dict(sorted(skip_reasons.items())),
            "reported_character_coverage": {
                "replaced_source_characters": replaced_source_characters,
                "skipped_source_characters": skipped_source_characters,
                "retained_source_characters": retained_source_characters,
                "accounted_source_characters": accounted_source_characters,
                "all_reported_source_characters": (
                    accounted_source_characters + retained_source_characters
                ),
                "reported_character_replacement_ratio": (
                    round(replaced_source_characters / accounted_source_characters, 6)
                    if accounted_source_characters
                    else None
                ),
                "regions_missing_source_character_count": sum(
                    _as_int(item.get("regions_missing_source_character_count"), 0)
                    for item in character_coverage
                ),
                "cases_with_available_counts": sum(
                    item.get("available") is True for item in character_coverage
                ),
                "complete_case_count": sum(
                    item.get("complete_for_terminal_regions") is True for item in character_coverage
                ),
            },
        },
        "semantic_accuracy_scored": False,
        "visual_quality_scored": False,
        "outside_approved_region_pixel_preservation_scored": False,
    }


def evaluate_runs(
    spec_path: Path,
    jobs_dir: Path,
    *,
    identical_min_chars: int = DEFAULT_IDENTICAL_CHUNK_MIN_CHARS,
) -> dict[str, Any]:
    if identical_min_chars < 1:
        raise ValueError("identical_min_chars must be at least 1")
    if not jobs_dir.is_dir():
        raise ValueError(f"Jobs directory does not exist: {jobs_dir}")
    spec, error = _read_json(spec_path)
    if error or not isinstance(spec, dict):
        raise ValueError(f"Could not read corpus spec {spec_path}: {error or 'invalid object'}")
    case_specs = _flatten_spec(spec)
    case_ids = {case["id"] for case in case_specs}
    candidates, unmatched = _load_job_candidates(jobs_dir, case_ids)
    cases = [
        _evaluate_case(
            case,
            candidates[case["id"]],
            identical_min_chars=identical_min_chars,
        )
        for case in case_specs
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_path": str(spec_path.resolve()),
        "jobs_dir": str(jobs_dir.resolve()),
        "evaluation_scope": EVALUATION_SCOPE,
        "verification_scope": {
            "automated_checks": list(AUTOMATED_CHECKS),
            "limitations": list(VERIFICATION_LIMITATIONS),
        },
        "summary": _build_summary(cases, unmatched),
        "cases": cases,
        "unmatched_jobs": unmatched,
    }


def _format_ratio(value: Any) -> str:
    number = _as_float(value)
    return "—" if number is None else f"{number * 100:.1f}%"


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _short_detail(value: Any, limit: int = 280) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key} {value}" for key, value in counts.items()) or "none"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    chunks = summary["translation_chunks"]
    pdfs = summary["pdfs"]
    reconstruction_summary = summary["reconstruction"]
    character_summary = reconstruction_summary["reported_character_coverage"]
    lines = [
        "# PDF regression workflow evaluation",
        "",
        f"> {report['evaluation_scope']}",
        "",
        f"Generated: `{report['generated_at']}`  ",
        f"Jobs: `{report['jobs_dir']}`",
        "",
        "## Summary",
        "",
        f"- Cases found: **{summary['matched_case_count']}/{summary['expected_case_count']}**; "
        f"unmatched jobs: **{summary['unmatched_job_count']}**.",
        f"- Stages: {_format_counts(summary['stage_counts'])}.",
        f"- Extracted: **{summary['structure']['blocks']}** blocks, "
        f"**{summary['structure']['tables']}** tables, **{summary['structure']['figures']}** "
        f"figures; invalid/out-of-page boxes: **{summary['structure']['invalid_bboxes']}/"
        f"{summary['structure']['outside_page_bboxes']}**.",
        f"- Translation chunks populated: **{chunks['translated_nonempty_count']}/"
        f"{chunks['source_bearing_count']}** ({_format_ratio(chunks['completion_ratio'])}); "
        f"long source-identical review candidates: **{chunks['source_identical_long_count']}**.",
        f"- English-output validation failures retained as source: "
        f"**{chunks['failed_translation_validation_count']}**; reasons: "
        f"**{chunks['failed_translation_validation_reason_counts']}**.",
        f"- Severe structurally collapsed translations: "
        f"**{chunks['severe_structural_collapse_count']}**.",
        f"- Expected source/extraction page counts match for "
        f"**{summary['page_counts']['matches_expected_count']}/"
        f"{summary['page_counts']['available_count']}** evaluated case(s).",
        f"- PDFs opening successfully: readable **{pdfs['readable_open_count']}**, "
        f"original-layout **{pdfs['original_layout_open_count']}**; original-layout geometry "
        f"matches source for **{pdfs['original_layout_geometry_match_count']}** case(s).",
        f"- Reconstruction: **{reconstruction_summary['regions_replaced']}** regions replaced, "
        f"**{reconstruction_summary['regions_skipped']}** skipped, "
        f"**{reconstruction_summary['regions_retained']}** intentionally retained, "
        f"**{reconstruction_summary['text_boxes_did_not_fit']}** overflowed.",
        f"- Reported source-character replacement: "
        f"**{character_summary['replaced_source_characters']}/"
        f"{character_summary['accounted_source_characters']}** "
        f"({_format_ratio(character_summary['reported_character_replacement_ratio'])}); "
        f"intentionally retained source characters: "
        f"**{character_summary['retained_source_characters']}**; "
        f"regions lacking a source-character count: "
        f"**{character_summary['regions_missing_source_character_count']}**.",
        "",
        "## Cases",
        "",
        "| Case | Stage / classification | Structure | Translation chunks | Readable PDF | Original layout | Rebuild | Structural outcome |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        stage = case["job"]["stage"] or "missing"
        expected = case["expected_classification"] or "—"
        actual = case["classification"]["actual"] or "—"
        classification = f"{stage}; {actual}/{expected}"
        structure = case["structure"]
        structure_cell = (
            f"{structure.get('block_count', 0)} b / {structure.get('table_count', 0)} t / "
            f"{structure.get('figure_count', 0)} f"
        )
        chunk = structure["translation_chunks"]
        chunk_cell = (
            f"{chunk['translated_nonempty_count']}/{chunk['source_bearing_count']}"
            f"; identical {chunk['source_identical_long_count']}"
            f"; validation failures {chunk['failed_translation_validation_count']}"
            f"; collapsed {chunk['severe_structural_collapse_count']}"
        )
        readable = case["pdfs"]["readable"]
        readable_cell = (
            f"{readable.get('page_count')} pp" if readable.get("opens") else "missing/error"
        )
        original = case["pdfs"]["original_layout"]
        comparison = original.get("geometry_comparison", {})
        if original.get("opens"):
            geometry = "geometry yes" if comparison.get("matches_source") else "geometry no"
            original_cell = f"{original.get('page_count')} pp; {geometry}"
        else:
            original_cell = "missing/error"
        reconstruction = case["reconstruction"]
        if reconstruction.get("present"):
            minimum_scale = reconstruction.get("actual_minimum_text_scale")
            scale_cell = "—" if minimum_scale is None else f"{minimum_scale:.3f}"
            character_coverage = reconstruction.get("reported_character_coverage", {})
            character_ratio = character_coverage.get("reported_character_replacement_ratio")
            character_cell = _format_ratio(character_ratio)
            if character_coverage.get("available") and not character_coverage.get(
                "complete_for_terminal_regions"
            ):
                character_cell += " known-only"
            rebuild_cell = (
                f"{reconstruction.get('status') or 'unknown'}; "
                f"{reconstruction.get('regions_replaced', 0)}R/"
                f"{reconstruction.get('regions_skipped', 0)}S/"
                f"{reconstruction.get('regions_retained', 0)} retained; "
                f"overflow {reconstruction.get('text_boxes_did_not_fit', 0)}; "
                f"min {scale_cell}; chars {character_cell}"
            )
        else:
            rebuild_cell = "missing"
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    case["case_id"],
                    classification,
                    structure_cell,
                    chunk_cell,
                    readable_cell,
                    original_cell,
                    rebuild_cell,
                    case["structural_outcome"],
                )
            )
            + " |"
        )

    diagnostic_lines: list[str] = []
    for case in report["cases"]:
        details: list[str] = []
        if case["job"]["error"]:
            details.append(f"pipeline error: {_short_detail(case['job']['error'])}")
        structure = case["structure"]
        if structure.get("error"):
            details.append(f"structured JSON: {_short_detail(structure['error'])}")
        bbox = structure.get("bbox", {})
        if bbox.get("invalid_count") or bbox.get("outside_page_count"):
            details.append(
                f"bbox invalid/outside: {bbox.get('invalid_count', 0)}/"
                f"{bbox.get('outside_page_count', 0)}"
            )
        chunk = structure["translation_chunks"]
        if chunk["empty_translation_count"]:
            details.append(f"empty translations: {chunk['empty_translation_count']}")
        if chunk["source_identical_long_count"]:
            details.append(
                f"long source-identical review candidates: {chunk['source_identical_long_count']}"
            )
        if chunk["failed_translation_validation_count"]:
            details.append(
                "English-output validation failures retained as source: "
                f"{chunk['failed_translation_validation_ids']}; reasons: "
                f"{chunk['failed_translation_validation_reason_counts']}"
            )
        if chunk["severe_structural_collapse_count"]:
            details.append(
                "severe structurally collapsed translations: "
                f"{_short_detail(chunk['severe_structural_collapse_chunks'])}"
            )
        page_counts = case["page_count_validation"]
        if page_counts["all_available_counts_match_expected"] is False:
            details.append(
                "page-count mismatch: "
                f"expected {page_counts['expected']}, source {page_counts['source_pdf']}, "
                f"structured {page_counts['structured_document']}"
            )
        reconstruction = case["reconstruction"]
        if reconstruction.get("regions_skipped"):
            details.append(f"skip reasons: {reconstruction.get('skip_reason_counts', {})}")
        if reconstruction.get("text_boxes_did_not_fit"):
            details.append(
                f"text boxes that did not fit: {reconstruction['text_boxes_did_not_fit']}"
            )
        character_coverage = reconstruction.get("reported_character_coverage", {})
        if character_coverage.get("regions_missing_source_character_count"):
            details.append(
                "reconstruction regions lacking source character counts: "
                f"{character_coverage['regions_missing_source_character_count']}"
            )
        original_comparison = case["pdfs"]["original_layout"].get("geometry_comparison", {})
        if original_comparison.get("matches_source") is False:
            details.append(
                "original-layout geometry mismatches: "
                f"{_short_detail(original_comparison.get('mismatched_pages', []))}"
            )
        if details:
            diagnostic_lines.append(
                f"- **{_markdown_cell(case['case_id'])}:** " + "; ".join(details) + "."
            )
    if diagnostic_lines:
        lines.extend(("", "## Review signals", "", *diagnostic_lines))

    skip_reasons = reconstruction_summary["skip_reason_counts"]
    if skip_reasons:
        lines.extend(("", "## Aggregate reconstruction skip reasons", ""))
        lines.extend(f"- `{reason}`: {count}" for reason, count in skip_reasons.items())
    verification_scope = report.get("verification_scope", {})
    lines.extend(("", "## Scope and limitations", "", "Automated checks:"))
    lines.extend(f"- {item}" for item in verification_scope.get("automated_checks", []))
    lines.extend(("", "Not verified by this evaluator:"))
    lines.extend(f"- {item}" for item in verification_scope.get("limitations", []))
    lines.extend(
        (
            "",
            "A structurally complete result still requires human review of translation meaning and "
            "rendered pages. This evaluator intentionally makes neither claim.",
            "",
        )
    )
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any], json_path: Path, markdown_path: Path
) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jobs_dir",
        type=Path,
        help="Directory containing one job directory per regression case",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help=f"Corpus spec (default: {DEFAULT_SPEC_PATH})",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Output JSON path (default: <jobs-dir-parent>/evaluation.json)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Output Markdown path (default: <jobs-dir-parent>/evaluation.md)",
    )
    parser.add_argument(
        "--identical-min-chars",
        type=int,
        default=DEFAULT_IDENTICAL_CHUNK_MIN_CHARS,
        help="Minimum normalised source length for an identical-translation review signal",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    jobs_dir = args.jobs_dir.expanduser().resolve()
    output_dir = jobs_dir.parent
    json_path = (args.json_output or output_dir / "evaluation.json").expanduser().resolve()
    markdown_path = (args.markdown_output or output_dir / "evaluation.md").expanduser().resolve()
    try:
        report = evaluate_runs(
            args.spec.expanduser().resolve(),
            jobs_dir,
            identical_min_chars=args.identical_min_chars,
        )
        write_reports(report, json_path, markdown_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
