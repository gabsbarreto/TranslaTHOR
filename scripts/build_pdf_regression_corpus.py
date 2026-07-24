#!/usr/bin/env python3
"""Build the private PDF regression corpus from verified workspace sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from langdetect import DetectorFactory, detect
from PIL import Image, ImageChops, ImageStat


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.pdf_extraction.pdf_type_detector import PDFTypeDetector  # noqa: E402


DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "workspace" / "tests"
DEFAULT_SPEC_PATH = REPOSITORY_ROOT / "tests" / "regression_corpus" / "corpus_spec.json"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "workspace" / "regression_corpus"
DetectorFactory.seed = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 2:
        raise ValueError(f"Unsupported corpus schema in {path}")
    digital_cases = spec.get("digital_cases")
    scanned_cases = spec.get("scanned_cases")
    if not isinstance(digital_cases, list) or len(digital_cases) != 5:
        raise ValueError("The regression corpus must define exactly five digital cases.")
    if not isinstance(scanned_cases, list) or len(scanned_cases) != 5:
        raise ValueError("The regression corpus must define exactly five scanned cases.")
    all_ids = [case.get("id") for case in digital_cases + scanned_cases]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Every regression corpus case ID must be unique.")
    return spec


def _verified_source(case: dict[str, Any], source_dir: Path) -> Path:
    source_path = source_dir / str(case["source_filename"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Corpus source is missing: {source_path}")
    actual_hash = _sha256(source_path)
    expected_hash = str(case["source_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(
            f"Corpus source checksum changed for {source_path.name}: "
            f"expected {expected_hash}, found {actual_hash}"
        )
    return source_path


def _selected_page_indexes(case: dict[str, Any], page_count: int) -> list[int]:
    page_numbers = case.get("source_pages")
    if not isinstance(page_numbers, list) or not page_numbers:
        raise ValueError(f"Case {case.get('id')} has no selected pages.")
    indexes: list[int] = []
    for page_number in page_numbers:
        if not isinstance(page_number, int) or not 1 <= page_number <= page_count:
            raise ValueError(
                f"Case {case.get('id')} selects invalid page {page_number} of {page_count}."
            )
        indexes.append(page_number - 1)
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"Case {case.get('id')} selects a page more than once.")
    return indexes


def _build_selected_fixture(
    source_path: Path,
    case: dict[str, Any],
    output_path: Path,
) -> None:
    temporary_path = output_path.with_suffix(".building.pdf")
    temporary_path.unlink(missing_ok=True)
    with fitz.open(source_path) as source_pdf, fitz.open() as output_pdf:
        indexes = _selected_page_indexes(case, source_pdf.page_count)
        for page_index in indexes:
            output_pdf.insert_pdf(
                source_pdf,
                from_page=page_index,
                to_page=page_index,
                links=True,
                annots=True,
            )
        metadata = dict(source_pdf.metadata or {})
        metadata["title"] = f"TranslaTHOR regression fixture: {case['id']}"
        metadata["subject"] = (
            f"Local regression excerpt from {source_path.name}; original pages "
            f"{case['source_pages']}"
        )
        output_pdf.set_metadata({key: str(value or "") for key, value in metadata.items()})
        output_pdf.save(temporary_path, garbage=4, deflate=True)
    temporary_path.replace(output_path)


def _build_raster_scan(
    digital_path: Path,
    output_path: Path,
    *,
    dpi: int,
    jpeg_quality: int,
) -> None:
    temporary_path = output_path.with_suffix(".building.pdf")
    temporary_path.unlink(missing_ok=True)
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    with fitz.open(digital_path) as digital_pdf, fitz.open() as scanned_pdf:
        for source_page in digital_pdf:
            target_page = scanned_pdf.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            pixmap = source_page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            image_bytes = pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)
            target_page.insert_image(target_page.rect, stream=image_bytes)
        scanned_pdf.set_metadata(
            {
                "title": f"Scan-only regression fixture: {output_path.stem}",
                "subject": (
                    "Raster-only counterpart generated for local TranslaTHOR regression tests; "
                    "contains no OCR text layer"
                ),
            }
        )
        scanned_pdf.save(temporary_path, garbage=4, deflate=True)
    temporary_path.replace(output_path)


def _page_dimensions(path: Path) -> list[dict[str, float | int]]:
    with fitz.open(path) as pdf:
        return [
            {
                "page_number": page_number,
                "width": round(page.rect.width, 4),
                "height": round(page.rect.height, 4),
                "rotation": page.rotation,
            }
            for page_number, page in enumerate(pdf, start=1)
        ]


def _detected_language(path: Path) -> str:
    with fitz.open(path) as pdf:
        text = "\n".join(page.get_text("text") for page in pdf)
    return detect(text[:50000])


def _paired_visual_difference(digital_path: Path, scanned_path: Path) -> dict[str, float]:
    page_differences: list[float] = []
    with fitz.open(digital_path) as digital_pdf, fitz.open(scanned_path) as scanned_pdf:
        if digital_pdf.page_count != scanned_pdf.page_count:
            raise RuntimeError("A derived scan has a different page count from its digital source.")
        for digital_page, scanned_page in zip(digital_pdf, scanned_pdf, strict=True):
            digital_pixmap = digital_page.get_pixmap(alpha=False)
            scanned_pixmap = scanned_page.get_pixmap(alpha=False)
            digital_image = Image.frombytes(
                "RGB",
                (digital_pixmap.width, digital_pixmap.height),
                digital_pixmap.samples,
            )
            scanned_image = Image.frombytes(
                "RGB",
                (scanned_pixmap.width, scanned_pixmap.height),
                scanned_pixmap.samples,
            )
            channel_means = ImageStat.Stat(
                ImageChops.difference(digital_image, scanned_image)
            ).mean
            page_differences.append(sum(channel_means) / len(channel_means))
    return {
        "mean_absolute_channel_difference": round(
            sum(page_differences) / len(page_differences), 4
        ),
        "maximum_page_mean_difference": round(max(page_differences), 4),
    }


def _fixture_record(path: Path, output_dir: Path) -> dict[str, Any]:
    detection = PDFTypeDetector().detect(path)
    return {
        "path": str(path.relative_to(output_dir)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "page_count": detection.page_count,
        "classification": detection.classification,
        "embedded_text_characters": detection.embedded_text_chars,
        "embedded_text_words": detection.embedded_text_words,
        "meaningful_page_count": detection.meaningful_page_count,
        "image_dominant_page_count": detection.image_dominant_page_count,
        "scanned_page_count": detection.scanned_page_count,
        "hidden_ocr_page_count": detection.metadata.get("hidden_ocr_page_count", 0),
        "page_dimensions": _page_dimensions(path),
    }


def _validate_text_fixture(case: dict[str, Any], record: dict[str, Any], path: Path) -> None:
    if record["classification"] != case["expected_classification"]:
        raise RuntimeError(
            f"Fixture {case['id']} classified as {record['classification']}, "
            f"expected {case['expected_classification']}"
        )
    if record["page_count"] != case["expected_page_count"]:
        raise RuntimeError(f"Fixture {case['id']} has an unexpected page count.")
    if record["embedded_text_characters"] < case["minimum_embedded_characters"]:
        raise RuntimeError(f"Fixture {case['id']} lost too much embedded text.")
    detected_language = _detected_language(path)
    if detected_language != case["language"]:
        raise RuntimeError(
            f"Fixture {case['id']} detected as {detected_language}, "
            f"expected {case['language']}"
        )
    record["detected_language"] = detected_language


def _clean_fixture_directory(directory: Path, expected_filenames: set[str]) -> None:
    for stale_path in directory.glob("*.pdf"):
        if stale_path.name not in expected_filenames:
            stale_path.unlink()


def build_corpus(
    *,
    spec_path: Path,
    output_dir: Path,
    source_dir: Path,
) -> Path:
    spec = _load_spec(spec_path)
    digital_dir = output_dir / "digital"
    scanned_dir = output_dir / "scanned"
    digital_dir.mkdir(parents=True, exist_ok=True)
    scanned_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(spec["render_dpi"])
    jpeg_quality = int(spec["jpeg_quality"])
    digital_specs = {case["id"]: case for case in spec["digital_cases"]}
    digital_paths: dict[str, Path] = {}
    manifest_digital: list[dict[str, Any]] = []
    manifest_scanned: list[dict[str, Any]] = []

    _clean_fixture_directory(
        digital_dir,
        {f"{case_id}.pdf" for case_id in digital_specs},
    )
    _clean_fixture_directory(
        scanned_dir,
        {f"{case['id']}.pdf" for case in spec["scanned_cases"]},
    )

    for case in spec["digital_cases"]:
        source_path = _verified_source(case, source_dir)
        output_path = digital_dir / f"{case['id']}.pdf"
        _build_selected_fixture(source_path, case, output_path)
        record = _fixture_record(output_path, output_dir)
        _validate_text_fixture(case, record, output_path)
        digital_paths[case["id"]] = output_path
        manifest_digital.append(
            {
                "id": case["id"],
                "language": case["language"],
                "language_profile": case["language_profile"],
                "features": case["features"],
                "source": {
                    "filename": source_path.name,
                    "sha256": case["source_sha256"],
                    "pages": case["source_pages"],
                },
                "fixture": record,
            }
        )
        print(
            f"built digital {case['id']}: {record['page_count']} pages, "
            f"{record['embedded_text_characters']} characters"
        )

    for case in spec["scanned_cases"]:
        output_path = scanned_dir / f"{case['id']}.pdf"
        build_mode = case["build_mode"]
        source_record: dict[str, Any]
        visual_difference: dict[str, float] | None = None
        if build_mode == "copy_hidden_ocr":
            source_path = _verified_source(case, source_dir)
            _build_selected_fixture(source_path, case, output_path)
            source_record = {
                "filename": source_path.name,
                "sha256": case["source_sha256"],
                "pages": case["source_pages"],
            }
        elif build_mode == "rasterize_digital":
            digital_case_id = str(case["derived_from"])
            if digital_case_id not in digital_paths:
                raise ValueError(
                    f"Scanned case {case['id']} references unknown digital case "
                    f"{digital_case_id}."
                )
            digital_path = digital_paths[digital_case_id]
            _build_raster_scan(
                digital_path,
                output_path,
                dpi=dpi,
                jpeg_quality=jpeg_quality,
            )
            source_record = {"derived_from": digital_case_id}
            visual_difference = _paired_visual_difference(digital_path, output_path)
            if visual_difference["maximum_page_mean_difference"] >= 15.0:
                raise RuntimeError(f"Raster counterpart for {case['id']} differs too much.")
        else:
            raise ValueError(f"Unknown scan build mode {build_mode!r}")

        record = _fixture_record(output_path, output_dir)
        if record["classification"] != case["expected_classification"]:
            raise RuntimeError(
                f"Fixture {case['id']} classified as {record['classification']}, "
                f"expected {case['expected_classification']}"
            )
        if record["page_count"] != case["expected_page_count"]:
            raise RuntimeError(f"Fixture {case['id']} has an unexpected page count.")
        if build_mode == "copy_hidden_ocr":
            _validate_text_fixture(case, record, output_path)
            if record["hidden_ocr_page_count"] != record["page_count"]:
                raise RuntimeError(f"Hidden-OCR fixture {case['id']} lost its scan classification.")
        elif record["embedded_text_characters"] != 0:
            raise RuntimeError(f"Raster-only fixture {case['id']} contains a text layer.")

        manifest_case = {
            "id": case["id"],
            "language": case["language"],
            "language_profile": case["language_profile"],
            "build_mode": build_mode,
            "features": case["features"],
            "source": source_record,
            "fixture": record,
        }
        if visual_difference is not None:
            manifest_case["paired_visual_difference"] = visual_difference
        manifest_scanned.append(manifest_case)
        print(
            f"built scanned {case['id']}: {record['page_count']} pages, "
            f"{record['classification']}"
        )

    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": _sha256(spec_path),
        "builder": "scripts/build_pdf_regression_corpus.py",
        "render": {"dpi": dpi, "jpeg_quality": jpeg_quality},
        "counts": {
            "digital": len(manifest_digital),
            "scanned": len(manifest_scanned),
            "authentic_hidden_ocr": sum(
                case["build_mode"] == "copy_hidden_ocr" for case in spec["scanned_cases"]
            ),
            "derived_raster": sum(
                case["build_mode"] == "rasterize_digital"
                for case in spec["scanned_cases"]
            ),
        },
        "digital_cases": manifest_digital,
        "scanned_cases": manifest_scanned,
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / "manifest.building.json"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest_path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the private source PDFs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    manifest_path = build_corpus(
        spec_path=arguments.spec.resolve(),
        output_dir=arguments.output_dir.resolve(),
        source_dir=arguments.source_dir.resolve(),
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
