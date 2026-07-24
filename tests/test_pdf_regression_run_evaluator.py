from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_pdf_regression_runs.py"
SPEC = importlib.util.spec_from_file_location("evaluate_pdf_regression_runs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)
EVALUATION_SCOPE = EVALUATOR.EVALUATION_SCOPE
evaluate_runs = EVALUATOR.evaluate_runs
render_markdown = EVALUATOR.render_markdown
write_reports = EVALUATOR.write_reports


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_pdf(path: Path, sizes: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for width, height in sizes:
        page = document.new_page(width=width, height=height)
        page.insert_text((30, 30), "synthetic test page")
    document.save(path)
    document.close()


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=40, height=30)
    page.draw_rect(page.rect, fill=(0.2, 0.4, 0.6))
    page.get_pixmap().save(path)
    document.close()


def _spec() -> dict:
    return {
        "schema_version": 2,
        "digital_cases": [
            {
                "id": "es-digital-test",
                "language": "es",
                "expected_classification": "digital_good_text",
                "expected_page_count": 2,
                "features": ["table", "figure"],
            }
        ],
        "scanned_cases": [
            {
                "id": "fr-missing-scan",
                "language": "fr",
                "expected_classification": "scanned_no_text",
                "expected_page_count": 1,
                "features": ["scan"],
            }
        ],
    }


def _completed_job(jobs_dir: Path) -> Path:
    job_dir = jobs_dir / "job-123"
    artifacts = job_dir / "artifacts"
    artifacts.mkdir(parents=True)
    source_sizes = [(595.0, 842.0), (612.0, 792.0)]
    _write_pdf(job_dir / "input.pdf", source_sizes)
    _write_pdf(artifacts / "result_translated.pdf", [(595.0, 842.0)])
    _write_pdf(artifacts / "result_translated_original_layout.pdf", source_sizes)
    _write_png(artifacts / "figures" / "figure-1.png")
    (artifacts / "source.md").write_text("Texto fuente\n", encoding="utf-8")
    (artifacts / "translated.md").write_text("Translated text\n", encoding="utf-8")

    long_unchanged = "Nombre propio y referencia bibliográfica " * 4
    structured = {
        "metadata": {"page_count": 2},
        "pages": [
            {"page_number": 1, "width": 595.0, "height": 842.0},
            {"page_number": 2, "width": 612.0, "height": 792.0},
        ],
        "blocks": [
            {
                "id": "good",
                "page_number": 1,
                "block_type": "paragraph",
                "bbox": {"x0": 10, "y0": 20, "x1": 200, "y1": 80},
            },
            {
                "id": "outside",
                "page_number": 1,
                "block_type": "paragraph",
                "bbox": {"x0": 10, "y0": 20, "x1": 700, "y1": 80},
            },
            {
                "id": "invalid",
                "page_number": 2,
                "block_type": "caption",
                "bbox": {"x0": 100, "y0": 20, "x1": 20, "y1": 80},
            },
        ],
        "tables": [
            {
                "id": "table-1",
                "page": 1,
                "page_numbers": [1],
                "bbox": None,
                "parse_mode": "table_structured",
            }
        ],
        "figures": [
            {
                "id": "figure-1",
                "page_number": 2,
                "bbox": {"x0": 20, "y0": 20, "x1": 120, "y1": 100},
                "asset_type": "vector",
                "image_path": "artifacts/figures/figure-1.png",
            }
        ],
        "translation_chunks": [
            {
                "id": "translated",
                "source_text": "Hola mundo " * 10,
                "translated_text": "Hello world " * 10,
                "page_start": 1,
            },
            {
                "id": "identical",
                "source_text": long_unchanged,
                "translated_text": long_unchanged,
                "page_start": 2,
                "status": "translation_failed",
            },
            {
                "id": "empty",
                "source_text": "Este bloque debería tener una traducción.",
                "translated_text": "",
                "page_start": 2,
            },
        ],
    }
    _write_json(artifacts / "structured.json", structured)
    _write_json(
        artifacts / "reconstruction_report_original_layout.json",
        {
            "status": "partial",
            "total_pages": 2,
            "pages_successfully_reconstructed": 1,
            "pages_using_fallback_behavior": 1,
            "figures_preserved": 1,
            "regions_replaced": 2,
            "regions_skipped": 1,
            "regions_missing_or_invalid_bboxes": [],
            "text_boxes_did_not_fit": [{"block_ids": ["empty"]}],
            "minimum_text_scale": 0.6,
            "scaling_applied": [
                {"block_ids": ["good"], "scale": 1.0},
                {"block_ids": ["translated"], "scale": 0.75},
            ],
            "raster_figure_fallbacks": [{"figure_id": "figure-1"}],
            "low_confidence_figure_or_caption_associations": [],
            "regions": [
                {
                    "page_number": 1,
                    "block_ids": ["good"],
                    "status": "replaced",
                    "source_character_count": 100,
                },
                {
                    "page_number": 2,
                    "block_ids": ["translated"],
                    "status": "committed",
                    "source_character_count": 300,
                },
                {
                    "page_number": 2,
                    "block_ids": ["empty"],
                    "status": "skipped",
                    "reason": "translated_text_did_not_fit",
                    "source_character_count": 25,
                },
            ],
            "warnings": [{"code": "region_skipped", "reason": "translated_text_did_not_fit"}],
        },
    )
    # Use a stale absolute artifact path to verify relocation by basename.
    _write_json(
        job_dir / "status.json",
        {
            "job_id": "job-123",
            "source_filename": "es-digital-test.pdf",
            "filename": "es-digital-test.pdf",
            "attempt": 0,
            "stage": "complete",
            "progress": 1.0,
            "error": None,
            "translation": {"pdf_classification": "digital_good_text"},
            "artifacts": {
                "json": "/moved/run/structured.json",
                "source_markdown": "artifacts/source.md",
                "markdown": "artifacts/translated.md",
                "pdf_readable": "artifacts/result_translated.pdf",
                "pdf_original_layout": "artifacts/result_translated_original_layout.pdf",
                "reconstruction_report": "artifacts/reconstruction_report_original_layout.json",
            },
        },
    )
    return job_dir


def test_evaluator_reports_structural_and_reconstruction_signals(tmp_path: Path) -> None:
    spec_path = tmp_path / "corpus_spec.json"
    jobs_dir = tmp_path / "run" / "jobs"
    _write_json(spec_path, _spec())
    _completed_job(jobs_dir)

    report = evaluate_runs(spec_path, jobs_dir, identical_min_chars=80)

    assert report["summary"]["matched_case_count"] == 1
    assert report["summary"]["missing_case_count"] == 1
    assert report["summary"]["stage_counts"] == {"complete": 1, "missing": 1}
    assert report["summary"]["semantic_accuracy_scored"] is False
    case = report["cases"][0]
    assert case["classification"] == {
        "actual": "digital_good_text",
        "matches_expected": True,
    }
    assert case["structure"]["block_count"] == 3
    assert case["structure"]["table_count"] == 1
    assert case["structure"]["figure_count"] == 1
    assert case["structure"]["figures_with_preview"] == 1
    assert case["structure"]["figure_previews"]["missing_count"] == 0
    assert case["structure"]["bbox"]["invalid_count"] == 1
    assert case["structure"]["bbox"]["outside_page_count"] == 1
    assert case["structure"]["bbox"]["by_entity"]["tables"]["missing"] == 1
    chunks = case["structure"]["translation_chunks"]
    assert chunks["completion_ratio"] == 0.666667
    assert chunks["empty_translation_ids"] == ["empty"]
    assert chunks["source_identical_long_count"] == 1
    assert chunks["failed_translation_validation_count"] == 1
    assert chunks["failed_translation_validation_ids"] == ["identical"]
    assert chunks["failed_translation_validation_reason_counts"] == {
        "unspecified": 1
    }
    assert chunks["severe_structural_collapse_count"] == 0
    assert chunks["source_identical_is_review_signal_only"] is True
    assert case["markdown"]["normalised_contents_equal"] is False
    assert case["pdfs"]["source"]["opens"] is True
    assert case["pdfs"]["readable"]["page_count"] == 1
    geometry = case["pdfs"]["original_layout"]["geometry_comparison"]
    assert geometry["matches_source"] is True
    assert geometry["source_page_count"] == 2
    assert geometry["output_page_count"] == 2
    assert geometry["page_count_matches_source"] is True
    assert geometry["pages_compared"] == 2
    assert geometry["geometry_match_does_not_verify_visual_preservation"] is True
    reconstruction = case["reconstruction"]
    assert reconstruction["regions_replaced"] == 2
    assert reconstruction["skip_reason_counts"] == {"translated_text_did_not_fit": 1}
    assert reconstruction["text_boxes_did_not_fit"] == 1
    assert reconstruction["actual_minimum_text_scale"] == 0.75
    assert reconstruction["below_minimum_scale_count"] == 0
    assert reconstruction["count_consistency_warnings"] == []
    character_coverage = reconstruction["reported_character_coverage"]
    assert character_coverage["available"] is True
    assert character_coverage["complete_for_terminal_regions"] is True
    assert character_coverage["replaced_source_characters"] == 400
    assert character_coverage["skipped_source_characters"] == 25
    assert character_coverage["reported_character_replacement_ratio"] == 0.941176
    assert character_coverage["reported_character_skip_ratio"] == 0.058824
    assert character_coverage["by_page"] == [
        {
            "page_number": 1,
            "replaced_source_characters": 100,
            "skipped_source_characters": 0,
            "retained_source_characters": 0,
            "accounted_source_characters": 100,
            "all_reported_source_characters": 100,
            "reported_character_replacement_ratio": 1.0,
        },
        {
            "page_number": 2,
            "replaced_source_characters": 300,
            "skipped_source_characters": 25,
            "retained_source_characters": 0,
            "accounted_source_characters": 325,
            "all_reported_source_characters": 325,
            "reported_character_replacement_ratio": 0.923077,
        },
    ]
    aggregate_coverage = report["summary"]["reconstruction"]["reported_character_coverage"]
    assert aggregate_coverage["reported_character_replacement_ratio"] == 0.941176
    assert aggregate_coverage["complete_case_count"] == 1
    assert report["summary"]["outside_approved_region_pixel_preservation_scored"] is False
    assert report["summary"]["page_counts"] == {
        "available_count": 1,
        "matches_expected_count": 1,
        "mismatch_count": 0,
    }
    assert report["verification_scope"]["limitations"]
    assert case["structural_outcome"] == "completed_with_translation_gaps"
    assert report["cases"][1]["structural_outcome"] == "missing_run"


def test_evaluator_flags_punctuation_only_translation_as_structural_gap(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "corpus_spec.json"
    jobs_dir = tmp_path / "run" / "jobs"
    _write_json(spec_path, _spec())
    job_dir = _completed_job(jobs_dir)
    structured_path = job_dir / "artifacts" / "structured.json"
    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    structured["translation_chunks"] = [
        {
            "id": "collapsed-fragment",
            "source_text": (
                "men der Therapie nicht oder nur in unbefriedigendem Ausmaß "
                "erreicht zu haben."
            ),
            "translated_text": ".",
            "page_start": 2,
            "status": "ready_for_translation",
        }
    ]
    _write_json(structured_path, structured)

    report = evaluate_runs(spec_path, jobs_dir)
    case = report["cases"][0]
    chunks = case["structure"]["translation_chunks"]

    assert chunks["empty_translation_count"] == 0
    assert chunks["failed_translation_validation_count"] == 0
    assert chunks["severe_structural_collapse_count"] == 1
    assert chunks["severe_structural_collapse_chunks"][0]["reason"] == (
        "non_language_target"
    )
    assert case["structural_outcome"] == "completed_with_translation_gaps"


def test_expected_page_count_mismatch_is_visible_in_structural_outcome(
    tmp_path: Path,
) -> None:
    spec = _spec()
    spec["digital_cases"][0]["expected_page_count"] = 3
    spec_path = tmp_path / "corpus_spec.json"
    jobs_dir = tmp_path / "run" / "jobs"
    _write_json(spec_path, spec)
    _completed_job(jobs_dir)

    report = evaluate_runs(spec_path, jobs_dir)
    case = report["cases"][0]

    assert case["page_count_validation"] == {
        "expected": 3,
        "source_pdf": 2,
        "structured_document": 2,
        "source_matches_expected": False,
        "structured_matches_expected": False,
        "all_available_counts_match_expected": False,
    }
    assert case["structural_outcome"] == "completed_with_page_count_mismatch"
    assert report["summary"]["page_counts"]["mismatch_count"] == 1


def test_evaluator_writes_json_and_concise_markdown(tmp_path: Path) -> None:
    spec_path = tmp_path / "corpus_spec.json"
    jobs_dir = tmp_path / "run" / "jobs"
    _write_json(spec_path, _spec())
    _completed_job(jobs_dir)
    report = evaluate_runs(spec_path, jobs_dir)

    markdown = render_markdown(report)

    assert EVALUATION_SCOPE in markdown
    assert "translated_text_did_not_fit" in markdown
    assert "human review of translation meaning" in markdown
    assert "Reported source-character replacement" in markdown
    assert "English-output validation failures retained as source" in markdown
    assert "Matching page geometry does not prove" in markdown
    assert "chars 94.1%" in markdown
    assert "| es-digital-test |" in markdown
    json_path, markdown_path = write_reports(
        report,
        tmp_path / "outputs" / "evaluation.json",
        tmp_path / "outputs" / "evaluation.md",
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert markdown_path.read_text(encoding="utf-8") == markdown


def test_character_coverage_discloses_missing_counts() -> None:
    coverage = EVALUATOR._reported_character_coverage(
        [
            {
                "page_number": 1,
                "status": "replaced",
                "source_character_count": 90,
            },
            {
                "page_number": 1,
                "status": "skipped",
                "source_character_count": 10,
            },
            {"page_number": 2, "status": "replaced"},
            {
                "page_number": 2,
                "status": "skipped",
                "source_character_count": -2,
            },
            {"page_number": 2, "status": "pending", "source_character_count": 500},
            {
                "page_number": 2,
                "status": "retained",
                "source_character_count": 40,
            },
        ]
    )

    assert coverage["available"] is True
    assert coverage["complete_for_terminal_regions"] is False
    assert coverage["terminal_region_count"] == 5
    assert coverage["regions_with_source_character_count"] == 3
    assert coverage["regions_missing_source_character_count"] == 2
    assert coverage["replaced_source_characters"] == 90
    assert coverage["skipped_source_characters"] == 10
    assert coverage["retained_source_characters"] == 40
    assert coverage["all_reported_source_characters"] == 140
    assert coverage["reported_character_replacement_ratio"] == 0.9
    assert "not a translation-accuracy" in coverage["interpretation"]


def test_bbox_validation_uses_declared_surya_coordinate_space() -> None:
    dimensions = {1: (595.0, 842.0)}
    structured = {
        "blocks": [
            {
                "id": "valid-surya-block",
                "page_number": 1,
                "bbox": {"x0": 100, "y0": 200, "x1": 2300, "y1": 3400},
                "metadata": {"surya_page_width": 2481, "surya_page_height": 3509},
            },
            {
                "id": "outside-surya-block",
                "page_number": 1,
                "bbox": {"x0": 100, "y0": 200, "x1": 2500, "y1": 3400},
                "metadata": {"surya_page_width": 2481, "surya_page_height": 3509},
            },
        ],
        "tables": [
            {
                "id": "valid-surya-table",
                "page_numbers": [1],
                "bbox": {"x0": 150, "y0": 300, "x1": 2200, "y1": 1200},
                "debug": {
                    "coordinate_space": {
                        "name": "surya_rendered_pixels",
                        "width": 2481,
                        "height": 3509,
                    }
                },
            }
        ],
        "figures": [],
    }

    metrics = EVALUATOR._bbox_metrics(structured, dimensions)

    assert metrics["by_entity"]["blocks"]["valid"] == 1
    assert metrics["by_entity"]["blocks"]["outside_page"] == 1
    assert metrics["by_entity"]["tables"]["valid"] == 1
    assert metrics["outside_page_count"] == 1


def test_reconstruction_page_count_inconsistency_is_not_structurally_complete(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    _write_json(
        report_path,
        {
            "status": "complete",
            "total_pages": 2,
            "pages_successfully_reconstructed": 1,
            "pages_using_fallback_behavior": 0,
            "pages": [{"page_number": 1, "status": "success"}],
            "regions": [],
        },
    )

    report = EVALUATOR._inspect_reconstruction_report(report_path, tmp_path)

    assert len(report["count_consistency_warnings"]) == 2


def test_missing_figure_preview_is_reported(tmp_path: Path) -> None:
    structured_path = tmp_path / "job" / "artifacts" / "structured.json"
    _write_json(
        structured_path,
        {
            "metadata": {"page_count": 1},
            "pages": [{"page_number": 1, "width": 595, "height": 842}],
            "blocks": [],
            "tables": [],
            "figures": [
                {
                    "id": "missing-preview",
                    "page_number": 1,
                    "bbox": {"x0": 10, "y0": 10, "x1": 100, "y1": 100},
                    "image_path": "artifacts/figures/missing.png",
                }
            ],
        },
    )

    result = EVALUATOR._inspect_structure(
        structured_path,
        structured_path.parents[1],
        identical_min_chars=80,
    )

    assert result["figures_with_preview"] == 0
    assert result["figure_previews"]["missing_ids"] == ["missing-preview"]
