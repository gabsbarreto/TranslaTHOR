from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image, ImageChops, ImageStat

from app.services.pdf_extraction.pdf_type_detector import PDFTypeDetector


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "tests" / "regression_corpus" / "corpus_spec.json"
CORPUS_ROOT = ROOT / "workspace" / "regression_corpus"
MANIFEST_PATH = CORPUS_ROOT / "manifest.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_image_coverage(page: fitz.Page) -> float:
    page_area = max(page.rect.get_area(), 1.0)
    image_area = 0.0
    for image in page.get_images(full=True):
        image_area += sum(rect.get_area() for rect in page.get_image_rects(image[0]))
    return min(image_area / page_area, 1.0)


def _page_visual_difference(digital_page: fitz.Page, scanned_page: fitz.Page) -> float:
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
    channel_means = ImageStat.Stat(ImageChops.difference(digital_image, scanned_image)).mean
    return sum(channel_means) / len(channel_means)


def test_regression_corpus_spec_defines_five_digital_and_five_scan_cases() -> None:
    spec = _load_json(SPEC_PATH)
    digital_cases = spec["digital_cases"]
    scanned_cases = spec["scanned_cases"]
    all_cases = digital_cases + scanned_cases
    digital_ids = {case["id"] for case in digital_cases}

    assert spec["schema_version"] == 2
    assert len(digital_cases) == 5
    assert len(scanned_cases) == 5
    assert len({case["id"] for case in all_cases}) == 10
    assert {case["language"] for case in digital_cases} == {"de", "es", "fr", "pt"}
    assert {case["language"] for case in scanned_cases} == {"de", "es", "fr", "pt"}
    assert all(case["language"] != "en" for case in all_cases)
    assert all(case["features"] for case in all_cases)
    assert all(case["expected_page_count"] >= 4 for case in all_cases)
    assert all(
        SHA256_PATTERN.fullmatch(case["source_sha256"])
        for case in digital_cases
    )

    authentic_scans = [
        case for case in scanned_cases if case["build_mode"] == "copy_hidden_ocr"
    ]
    derived_scans = [
        case for case in scanned_cases if case["build_mode"] == "rasterize_digital"
    ]
    assert len(authentic_scans) == 2
    assert len(derived_scans) == 3
    assert all(SHA256_PATTERN.fullmatch(case["source_sha256"]) for case in authentic_scans)
    assert all(case["derived_from"] in digital_ids for case in derived_scans)
    assert all(
        case["expected_classification"] == "bad_hidden_ocr" for case in authentic_scans
    )
    assert all(
        case["expected_classification"] == "scanned_no_text" for case in derived_scans
    )


def test_local_regression_corpus_integrity_and_pdf_classification() -> None:
    if not MANIFEST_PATH.is_file():
        pytest.skip(
            "Private PDF corpus is not built; run scripts/build_pdf_regression_corpus.py"
        )

    spec = _load_json(SPEC_PATH)
    manifest = _load_json(MANIFEST_PATH)
    digital_specs = {case["id"]: case for case in spec["digital_cases"]}
    scanned_specs = {case["id"]: case for case in spec["scanned_cases"]}
    digital_cases = {case["id"]: case for case in manifest["digital_cases"]}
    scanned_cases = {case["id"]: case for case in manifest["scanned_cases"]}

    assert manifest["schema_version"] == 2
    assert manifest["counts"] == {
        "digital": 5,
        "scanned": 5,
        "authentic_hidden_ocr": 2,
        "derived_raster": 3,
    }
    assert manifest["spec_sha256"] == _sha256(SPEC_PATH)
    assert set(digital_cases) == set(digital_specs)
    assert set(scanned_cases) == set(scanned_specs)
    assert {path.name for path in (CORPUS_ROOT / "digital").glob("*.pdf")} == {
        f"{case_id}.pdf" for case_id in digital_specs
    }
    assert {path.name for path in (CORPUS_ROOT / "scanned").glob("*.pdf")} == {
        f"{case_id}.pdf" for case_id in scanned_specs
    }

    detector = PDFTypeDetector()
    digital_paths: dict[str, Path] = {}
    for case_id, case_spec in digital_specs.items():
        case = digital_cases[case_id]
        fixture = case["fixture"]
        path = CORPUS_ROOT / fixture["path"]
        digital_paths[case_id] = path
        assert path.is_file()
        assert _sha256(path) == fixture["sha256"]
        assert case["language"] == case_spec["language"]
        assert fixture["detected_language"] == case_spec["language"]
        assert case["source"]["sha256"] == case_spec["source_sha256"]
        assert case["source"]["pages"] == case_spec["source_pages"]

        detection = detector.detect(path)
        assert detection.classification == "digital_good_text"
        assert detection.page_count == case_spec["expected_page_count"]
        assert detection.embedded_text_chars >= case_spec["minimum_embedded_characters"]

    for case_id, case_spec in scanned_specs.items():
        case = scanned_cases[case_id]
        fixture = case["fixture"]
        path = CORPUS_ROOT / fixture["path"]
        assert path.is_file()
        assert _sha256(path) == fixture["sha256"]
        assert case["build_mode"] == case_spec["build_mode"]
        assert case["language"] == case_spec["language"]

        detection = detector.detect(path)
        assert detection.classification == case_spec["expected_classification"]
        assert detection.page_count == case_spec["expected_page_count"]
        with fitz.open(path) as scanned_pdf:
            assert all(_page_image_coverage(page) >= 0.99 for page in scanned_pdf)

        if case_spec["build_mode"] == "copy_hidden_ocr":
            assert fixture["detected_language"] == case_spec["language"]
            assert detection.embedded_text_chars >= case_spec["minimum_embedded_characters"]
            assert detection.metadata["hidden_ocr_page_count"] == detection.page_count
            assert case["source"]["sha256"] == case_spec["source_sha256"]
            assert case["source"]["pages"] == case_spec["source_pages"]
            continue

        assert detection.embedded_text_chars == 0
        digital_case_id = case_spec["derived_from"]
        assert case["source"] == {"derived_from": digital_case_id}
        with (
            fitz.open(digital_paths[digital_case_id]) as digital_pdf,
            fitz.open(path) as scanned_pdf,
        ):
            assert digital_pdf.page_count == scanned_pdf.page_count
            for digital_page, scanned_page in zip(digital_pdf, scanned_pdf, strict=True):
                assert abs(digital_page.rect.width - scanned_page.rect.width) < 0.01
                assert abs(digital_page.rect.height - scanned_page.rect.height) < 0.01
                assert scanned_page.get_text("text").strip() == ""
                assert _page_visual_difference(digital_page, scanned_page) < 15.0
