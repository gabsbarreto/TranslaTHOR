#!/usr/bin/env python3
"""Build the private, paired PDF regression corpus from verified local sources."""

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


DEFAULT_SOURCE_ROOTS = {
    "included": Path.home()
    / "Library/CloudStorage/OneDrive-UniversityofBristol/Documents/RQ folder/PDFs included",
    "included_2": Path.home()
    / "Library/CloudStorage/OneDrive-UniversityofBristol/Documents/RQ folder/PDFs included 2",
}
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
    if spec.get("schema_version") != 1:
        raise ValueError(f"Unsupported corpus schema in {path}")
    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("The regression corpus must define exactly five paired cases.")
    return spec


def _verified_source(case: dict[str, Any], source_roots: dict[str, Path]) -> Path:
    collection = str(case["source_collection"])
    if collection not in source_roots:
        raise ValueError(f"Unknown source collection {collection!r}")
    source_path = source_roots[collection] / str(case["source_filename"])
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


def _build_digital_fixture(source_path: Path, case: dict[str, Any], output_path: Path) -> None:
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


def _build_scanned_fixture(
    digital_path: Path,
    output_path: Path,
    *,
    dpi: int,
    jpeg_quality: int,
) -> None:
    temporary_path = output_path.with_suffix(".building.pdf")
    temporary_path.unlink(missing_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
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
                "title": f"Scan-only regression fixture: {digital_path.stem}",
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
        "page_dimensions": _page_dimensions(path),
    }


def build_corpus(
    *,
    spec_path: Path,
    output_dir: Path,
    source_roots: dict[str, Path],
) -> Path:
    spec = _load_spec(spec_path)
    digital_dir = output_dir / "digital"
    scanned_dir = output_dir / "scanned"
    digital_dir.mkdir(parents=True, exist_ok=True)
    scanned_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(spec["render_dpi"])
    jpeg_quality = int(spec["jpeg_quality"])
    manifest_cases: list[dict[str, Any]] = []
    expected_filenames = {f"{case['id']}.pdf" for case in spec["cases"]}
    for fixture_dir in (digital_dir, scanned_dir):
        for stale_path in fixture_dir.glob("*.pdf"):
            if stale_path.name not in expected_filenames:
                stale_path.unlink()

    for case in spec["cases"]:
        source_path = _verified_source(case, source_roots)
        case_id = str(case["id"])
        digital_path = digital_dir / f"{case_id}.pdf"
        scanned_path = scanned_dir / f"{case_id}.pdf"
        _build_digital_fixture(source_path, case, digital_path)
        _build_scanned_fixture(
            digital_path,
            scanned_path,
            dpi=dpi,
            jpeg_quality=jpeg_quality,
        )
        digital_record = _fixture_record(digital_path, output_dir)
        scanned_record = _fixture_record(scanned_path, output_dir)
        minimum_characters = int(case["minimum_embedded_characters"])
        if digital_record["classification"] != "digital_good_text":
            raise RuntimeError(
                f"Digital fixture {case_id} classified as {digital_record['classification']}"
            )
        if digital_record["embedded_text_characters"] < minimum_characters:
            raise RuntimeError(f"Digital fixture {case_id} lost too much embedded text.")
        detected_language = _detected_language(digital_path)
        if detected_language != case["language"]:
            raise RuntimeError(
                f"Digital fixture {case_id} detected as {detected_language}, "
                f"expected {case['language']}"
            )
        if scanned_record["classification"] != "scanned_no_text":
            raise RuntimeError(
                f"Scanned fixture {case_id} classified as {scanned_record['classification']}"
            )
        if scanned_record["embedded_text_characters"] != 0:
            raise RuntimeError(f"Scanned fixture {case_id} unexpectedly contains a text layer.")
        if digital_record["page_dimensions"] != scanned_record["page_dimensions"]:
            raise RuntimeError(f"Page geometry differs within fixture pair {case_id}.")
        visual_difference = _paired_visual_difference(digital_path, scanned_path)
        if visual_difference["maximum_page_mean_difference"] >= 15.0:
            raise RuntimeError(f"Raster counterpart for {case_id} differs too much visually.")
        digital_record["detected_language"] = detected_language

        manifest_cases.append(
            {
                "id": case_id,
                "language": case["language"],
                "language_profile": case["language_profile"],
                "features": case["features"],
                "source": {
                    "collection": case["source_collection"],
                    "filename": source_path.name,
                    "sha256": case["source_sha256"],
                    "pages": case["source_pages"],
                },
                "digital": digital_record,
                "scanned": scanned_record,
                "paired_visual_difference": visual_difference,
            }
        )
        print(
            f"built {case_id}: {digital_record['page_count']} pages, "
            f"{digital_record['embedded_text_characters']} digital characters"
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": _sha256(spec_path),
        "builder": "scripts/build_pdf_regression_corpus.py",
        "render": {"dpi": dpi, "jpeg_quality": jpeg_quality},
        "counts": {"cases": len(manifest_cases), "digital": 5, "scanned": 5},
        "cases": manifest_cases,
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
        "--source-one",
        type=Path,
        default=DEFAULT_SOURCE_ROOTS["included"],
        help="Path to the 'PDFs included' collection.",
    )
    parser.add_argument(
        "--source-two",
        type=Path,
        default=DEFAULT_SOURCE_ROOTS["included_2"],
        help="Path to the 'PDFs included 2' collection.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    manifest_path = build_corpus(
        spec_path=arguments.spec.resolve(),
        output_dir=arguments.output_dir.resolve(),
        source_roots={
            "included": arguments.source_one.resolve(),
            "included_2": arguments.source_two.resolve(),
        },
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
