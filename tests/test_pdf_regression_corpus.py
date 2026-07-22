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


def _full_page_image_ratio(page: fitz.Page) -> float:
    page_area = max(page.rect.get_area(), 1.0)
    largest = 0.0
    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            largest = max(largest, min(rect.get_area() / page_area, 1.0))
    return largest


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


def test_regression_corpus_spec_defines_five_non_english_pairs() -> None:
    spec = _load_json(SPEC_PATH)
    cases = spec["cases"]

    assert spec["schema_version"] == 1
    assert len(cases) == 5
    assert len({case["id"] for case in cases}) == 5
    assert {case["language"] for case in cases} == {"es", "fr"}
    assert all(case["language"] != "en" for case in cases)
    assert all(case["language_profile"] == "non_english_pages" for case in cases)
    assert all(case["source_pages"] for case in cases)
    assert all(case["minimum_embedded_characters"] >= 1000 for case in cases)
    assert all(case["features"] for case in cases)
    assert all(SHA256_PATTERN.fullmatch(case["source_sha256"]) for case in cases)


def test_local_regression_corpus_integrity_and_pdf_classification() -> None:
    if not MANIFEST_PATH.is_file():
        pytest.skip(
            "Private PDF corpus is not built; run scripts/build_pdf_regression_corpus.py"
        )

    spec = _load_json(SPEC_PATH)
    manifest = _load_json(MANIFEST_PATH)
    cases_by_id = {case["id"]: case for case in manifest["cases"]}

    assert manifest["schema_version"] == 1
    assert manifest["counts"] == {"cases": 5, "digital": 5, "scanned": 5}
    assert manifest["spec_sha256"] == _sha256(SPEC_PATH)
    assert set(cases_by_id) == {case["id"] for case in spec["cases"]}
    expected_filenames = {f"{case_id}.pdf" for case_id in cases_by_id}
    assert {path.name for path in (CORPUS_ROOT / "digital").glob("*.pdf")} == expected_filenames
    assert {path.name for path in (CORPUS_ROOT / "scanned").glob("*.pdf")} == expected_filenames

    detector = PDFTypeDetector()
    for case_spec in spec["cases"]:
        case = cases_by_id[case_spec["id"]]
        assert case["language"] == case_spec["language"]
        assert case["digital"]["detected_language"] == case_spec["language"]
        assert case["source"]["sha256"] == case_spec["source_sha256"]
        assert case["source"]["pages"] == case_spec["source_pages"]

        digital_path = CORPUS_ROOT / case["digital"]["path"]
        scanned_path = CORPUS_ROOT / case["scanned"]["path"]
        assert digital_path.is_file()
        assert scanned_path.is_file()
        assert _sha256(digital_path) == case["digital"]["sha256"]
        assert _sha256(scanned_path) == case["scanned"]["sha256"]

        digital_detection = detector.detect(digital_path)
        scanned_detection = detector.detect(scanned_path)
        assert digital_detection.classification == "digital_good_text"
        assert scanned_detection.classification == "scanned_no_text"
        assert digital_detection.embedded_text_chars >= case_spec[
            "minimum_embedded_characters"
        ]
        assert scanned_detection.embedded_text_chars == 0

        with fitz.open(digital_path) as digital_pdf, fitz.open(scanned_path) as scanned_pdf:
            assert digital_pdf.page_count == scanned_pdf.page_count == len(
                case_spec["source_pages"]
            )
            for digital_page, scanned_page in zip(digital_pdf, scanned_pdf, strict=True):
                assert abs(digital_page.rect.width - scanned_page.rect.width) < 0.01
                assert abs(digital_page.rect.height - scanned_page.rect.height) < 0.01
                assert scanned_page.get_text("text").strip() == ""
                assert _full_page_image_ratio(scanned_page) >= 0.99
                assert _page_visual_difference(digital_page, scanned_page) < 15.0
