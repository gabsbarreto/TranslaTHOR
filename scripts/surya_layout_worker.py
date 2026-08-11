from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
ANNOTATION_COLORS = {
    "Caption": "#d97706",
    "Figure": "#9333ea",
    "Footnote": "#db2777",
    "Formula": "#7c3aed",
    "ListItem": "#0284c7",
    "PageFooter": "#dc2626",
    "PageHeader": "#16a34a",
    "Picture": "#9333ea",
    "SectionHeader": "#2563eb",
    "Table": "#ea580c",
    "Text": "#0891b2",
}
DEFAULT_ANNOTATION_COLOR = "#4b5563"


def bbox_from_polygon(polygon: list[list[float]]) -> list[int]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]


def padded_bbox(bbox: list[int], *, width: int, height: int, padding: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    ]


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def find_region_overlaps(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for index, first in enumerate(regions):
        first_x1, first_y1, first_x2, first_y2 = first["bbox"]
        first_area = (first_x2 - first_x1) * (first_y2 - first_y1)
        for second in regions[index + 1 :]:
            second_x1, second_y1, second_x2, second_y2 = second["bbox"]
            intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
            intersection_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
            intersection_area = intersection_width * intersection_height
            if intersection_area == 0:
                continue
            second_area = (second_x2 - second_x1) * (second_y2 - second_y1)
            overlaps.append(
                {
                    "first_region_id": first["id"],
                    "second_region_id": second["id"],
                    "first_label": first["label"],
                    "second_label": second["label"],
                    "intersection_width": intersection_width,
                    "intersection_height": intersection_height,
                    "intersection_area": intersection_area,
                    "intersection_over_smaller_region": round(
                        intersection_area / min(first_area, second_area),
                        6,
                    ),
                }
            )
    return overlaps


def boxes_overlap(first_bbox: list[int], second_bbox: list[int]) -> bool:
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox
    return min(first_x2, second_x2) > max(first_x1, second_x1) and min(first_y2, second_y2) > max(
        first_y1, second_y1
    )


def merge_same_label_overlaps(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for region in sorted(regions, key=lambda item: item["position"]):
        matching_groups = [
            group
            for group in groups
            if group[0]["label"] == region["label"]
            and any(boxes_overlap(item["bbox"], region["bbox"]) for item in group)
        ]
        if not matching_groups:
            groups.append([region])
            continue
        target = matching_groups[0]
        target.append(region)
        for extra_group in matching_groups[1:]:
            target.extend(extra_group)
            groups.remove(extra_group)

    merged: list[dict[str, Any]] = []
    for group in groups:
        x1 = min(region["bbox"][0] for region in group)
        y1 = min(region["bbox"][1] for region in group)
        x2 = max(region["bbox"][2] for region in group)
        y2 = max(region["bbox"][3] for region in group)
        merged.append(
            {
                "label": group[0]["label"],
                "position": min(int(region["position"]) for region in group),
                "bbox": [x1, y1, x2, y2],
                "source_region_ids": [str(region["id"]) for region in group],
            }
        )
    return sorted(merged, key=lambda item: item["position"])


def load_font(size: int = 24) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_qwen_overlay(
    page_image: Image.Image,
    regions: list[dict[str, Any]],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    overlay = page_image.copy()
    draw = ImageDraw.Draw(overlay)
    font = load_font()
    output_regions: list[dict[str, Any]] = []
    for display_index, region in enumerate(merge_same_label_overlaps(regions), start=1):
        bbox = region["bbox"]
        color = ANNOTATION_COLORS.get(region["label"], DEFAULT_ANNOTATION_COLOR)
        draw.rectangle(tuple(bbox), outline=color, width=5)
        label = f"SURYA {display_index}: {region['label']}"
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_x = max(0, min(bbox[0], page_image.width - label_width - 8))
        label_y = max(0, bbox[1] - label_height - 10)
        draw.rectangle(
            (label_x, label_y, label_x + label_width + 8, label_y + label_height + 6),
            fill="white",
            outline=color,
            width=3,
        )
        draw.text((label_x + 4, label_y + 1), label, fill=color, font=font)
        output_regions.append(
            {
                "index": display_index,
                "label": region["label"],
                "position": region["position"],
                "bbox": bbox,
                "source_region_ids": region["source_region_ids"],
            }
        )
    return overlay, output_regions


def save_layout_artifacts(
    *,
    image_paths: list[Path],
    predictions: list[Any],
    output_dir: Path,
    padding: int,
) -> dict[str, Any]:
    crops_dir = output_dir / "crops"
    annotated_dir = output_dir / "annotated_pages"
    boxed_pages_dir = output_dir / "boxed_pages"
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    boxed_pages_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    for page_index, (image_path, prediction) in enumerate(zip(image_paths, predictions), start=1):
        with Image.open(image_path) as image:
            page_image = image.convert("RGB")

        annotated = page_image.copy()
        draw = ImageDraw.Draw(annotated)
        regions: list[dict[str, Any]] = []
        sorted_boxes = sorted(prediction.bboxes, key=lambda box: box.position)
        for region_index, box in enumerate(sorted_boxes, start=1):
            bbox = bbox_from_polygon(box.polygon)
            crop_bbox = padded_bbox(
                bbox,
                width=page_image.width,
                height=page_image.height,
                padding=padding,
            )
            region_id = f"{image_path.stem}-r{region_index:03d}"
            crop_path = crops_dir / f"{region_id}.png"
            page_image.crop(tuple(crop_bbox)).save(crop_path)

            label = str(box.label)
            label_counts[label] = label_counts.get(label, 0) + 1
            color = ANNOTATION_COLORS.get(label, DEFAULT_ANNOTATION_COLOR)
            draw.rectangle(tuple(bbox), outline=color, width=4)
            draw.text((bbox[0] + 4, max(0, bbox[1] - 16)), f"{region_index}: {label}", fill=color)

            regions.append(
                {
                    "id": region_id,
                    "label": label,
                    "confidence": box.confidence,
                    "position": int(box.position),
                    "bbox": bbox,
                    "crop_bbox": crop_bbox,
                    "polygon": box.polygon,
                    "top_k": box.top_k,
                    "crop_path": str(crop_path),
                }
            )

        annotated_path = annotated_dir / f"{image_path.stem}.png"
        annotated.save(annotated_path)
        boxed_page, reconciled_regions = draw_qwen_overlay(page_image, regions)
        boxed_page_path = boxed_pages_dir / f"{image_path.stem}.png"
        boxed_page.save(boxed_page_path)
        overlaps = find_region_overlaps(regions)
        pages.append(
            {
                "page_index": page_index,
                "image_path": str(image_path),
                "annotated_path": str(annotated_path),
                "boxed_page_path": str(boxed_page_path),
                "image_bbox": prediction.image_bbox,
                "width": page_image.width,
                "height": page_image.height,
                "regions": regions,
                "reconciled_regions": reconciled_regions,
                "overlaps": overlaps,
            }
        )

    overlap_count = sum(len(page["overlaps"]) for page in pages)
    reconciled_region_count = sum(len(page["reconciled_regions"]) for page in pages)
    manifest = {
        "input_images": [str(path) for path in image_paths],
        "padding": padding,
        "page_count": len(pages),
        "region_count": sum(label_counts.values()),
        "reconciled_region_count": reconciled_region_count,
        "merged_region_count": sum(label_counts.values()) - reconciled_region_count,
        "overlap_count": overlap_count,
        "label_counts": dict(sorted(label_counts.items())),
        "pages": pages,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layout.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def detect_layout(image_paths: list[Path], *, batch_size: int | None = None) -> list[Any]:
    # Surya reads these settings at import time. This worker now shares the
    # Surya 2/Marker 2 environment and uses the same llama.cpp backend as the
    # primary OCR worker.
    os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
    os.environ["SURYA_GUIDED_LAYOUT"] = "false"
    from surya.inference import SuryaInferenceManager
    from surya.layout import LayoutPredictor

    images: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))

    manager = SuryaInferenceManager(method="llamacpp")
    predictor = LayoutPredictor(manager)
    predictions: list[Any] = []
    effective_batch_size = max(1, int(batch_size or len(images)))
    try:
        for start in range(0, len(images), effective_batch_size):
            predictions.extend(predictor(images[start : start + effective_batch_size]))
        return predictions
    finally:
        manager.stop()
        for image in images:
            image.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect Surya layout boxes and save crops plus full-page Qwen overlays."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    image_paths = discover_images(args.input_dir)
    if not image_paths:
        raise RuntimeError(f"No supported page images found in {args.input_dir}")
    padding = max(0, int(args.padding))

    print(json.dumps({"event": "layout_detection_started", "pages": len(image_paths)}), flush=True)
    predictions = detect_layout(image_paths, batch_size=args.batch_size)
    manifest = save_layout_artifacts(
        image_paths=image_paths,
        predictions=predictions,
        output_dir=args.output_dir,
        padding=padding,
    )
    print(
        json.dumps(
            {
                "event": "layout_detection_complete",
                "pages": manifest["page_count"],
                "regions": manifest["region_count"],
                "reconciled_regions": manifest["reconciled_region_count"],
                "merged_regions": manifest["merged_region_count"],
                "overlaps": manifest["overlap_count"],
                "label_counts": manifest["label_counts"],
                "manifest": str(args.output_dir / "layout.json"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
