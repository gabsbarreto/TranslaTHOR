from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def _load_worker_module():
    root = Path(__file__).resolve().parents[1]
    worker_path = root / "scripts" / "surya_layout_worker.py"
    spec = importlib.util.spec_from_file_location("surya_layout_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_surya_worker_saves_manifest_annotated_page_and_padded_crop(tmp_path: Path) -> None:
    worker = _load_worker_module()
    image_path = tmp_path / "page_0001.png"
    Image.new("RGB", (100, 80), "white").save(image_path)
    prediction = SimpleNamespace(
        image_bbox=[0, 0, 100, 80],
        bboxes=[
            SimpleNamespace(
                polygon=[[10, 20], [70, 20], [70, 40], [10, 40]],
                label="PageHeader",
                confidence=0.95,
                position=0,
                top_k={"PageHeader": 0.95},
            )
        ],
    )

    manifest = worker.save_layout_artifacts(
        image_paths=[image_path],
        predictions=[prediction],
        output_dir=tmp_path / "layout",
        padding=4,
    )

    region = manifest["pages"][0]["regions"][0]
    assert manifest["label_counts"] == {"PageHeader": 1}
    assert manifest["overlap_count"] == 0
    assert region["bbox"] == [10, 20, 70, 40]
    assert region["crop_bbox"] == [6, 16, 74, 44]
    assert Image.open(region["crop_path"]).size == (68, 28)
    assert (tmp_path / "layout" / "layout.json").exists()
    assert (tmp_path / "layout" / "annotated_pages" / "page_0001.png").exists()
    assert (tmp_path / "layout" / "boxed_pages" / "page_0001.png").exists()
    assert manifest["reconciled_region_count"] == 1
    assert manifest["merged_region_count"] == 0
    assert manifest["pages"][0]["reconciled_regions"] == [
        {
            "index": 1,
            "label": "PageHeader",
            "position": 0,
            "bbox": [10, 20, 70, 40],
            "source_region_ids": ["page_0001-r001"],
        }
    ]


def test_surya_worker_reports_overlapping_regions() -> None:
    worker = _load_worker_module()
    regions = [
        {"id": "r1", "label": "Footnote", "bbox": [0, 0, 100, 20]},
        {"id": "r2", "label": "Footnote", "bbox": [0, 15, 100, 30]},
    ]

    overlaps = worker.find_region_overlaps(regions)

    assert overlaps == [
        {
            "first_region_id": "r1",
            "second_region_id": "r2",
            "first_label": "Footnote",
            "second_label": "Footnote",
            "intersection_width": 100,
            "intersection_height": 5,
            "intersection_area": 500,
            "intersection_over_smaller_region": 0.333333,
        }
    ]


def test_surya_worker_merges_only_overlapping_regions_with_same_label() -> None:
    worker = _load_worker_module()
    regions = [
        {"id": "r1", "label": "Footnote", "position": 0, "bbox": [0, 0, 100, 20]},
        {"id": "r2", "label": "Footnote", "position": 1, "bbox": [0, 15, 100, 30]},
        {"id": "r3", "label": "Text", "position": 2, "bbox": [0, 15, 100, 30]},
    ]

    merged = worker.merge_same_label_overlaps(regions)

    assert merged == [
        {
            "label": "Footnote",
            "position": 0,
            "bbox": [0, 0, 100, 30],
            "source_region_ids": ["r1", "r2"],
        },
        {
            "label": "Text",
            "position": 2,
            "bbox": [0, 15, 100, 30],
            "source_region_ids": ["r3"],
        },
    ]
