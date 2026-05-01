from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from app.config import (
    DEFAULT_DEEPSEEK_OCR_BASE_SIZE,
    DEFAULT_DEEPSEEK_OCR_CROP_MODE,
    DEFAULT_DEEPSEEK_OCR_IMAGE_SIZE,
    DEFAULT_DEEPSEEK_OCR_MAX_CROPS,
    DEFAULT_DEEPSEEK_OCR_MAX_TOKENS,
    DEFAULT_DEEPSEEK_OCR_MIN_CROPS,
    DEFAULT_DEEPSEEK_OCR_MODEL,
    DEFAULT_DEEPSEEK_OCR_NGRAM_SIZE,
    DEFAULT_DEEPSEEK_OCR_NGRAM_WINDOW,
    DEFAULT_DEEPSEEK_OCR_PROMPT,
    DEFAULT_DEEPSEEK_OCR_SKIP_REPEAT,
    JOBS_DIR,
)
from app.services.coordinate_utils import normalized_to_image_bbox, pad_normalized_bbox
from app.services.renderer import PageRenderer


VARIANTS = ("none", "grayscale_contrast", "grayscale_contrast_sharpen", "binarize")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate debug images and optional OCR sweep for one selected PDF region.")
    parser.add_argument("--job-id")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--page", type=int, default=2)
    parser.add_argument("--x", type=float, default=7.5, help="Selection x as percent unless --coords normalized is used.")
    parser.add_argument("--y", type=float, default=8.3, help="Selection y as percent unless --coords normalized is used.")
    parser.add_argument("--width", type=float, default=85.5, help="Selection width as percent unless --coords normalized is used.")
    parser.add_argument("--height", type=float, default=42.3, help="Selection height as percent unless --coords normalized is used.")
    parser.add_argument("--coords", choices=("percent", "normalized"), default="percent")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, nargs="+", default=[200, 300, 400, 600])
    parser.add_argument("--padding", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--upscale", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--variant", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--run-ocr", action="store_true")
    args = parser.parse_args()

    pdf_path = resolve_pdf(args.job_id, args.pdf)
    x0, y0, x1, y1 = selection_to_normalized(args.x, args.y, args.width, args.height, args.coords)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = args.job_id or pdf_path.stem
    output_dir = args.output_dir or ROOT / "workspace" / "ocr_region_debug" / f"{label}-p{args.page}-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    ocr_inputs: list[Path] = []
    ocr_names: list[str] = []
    top_level_padding = 0.10 if any(abs(item - 0.10) < 1e-9 for item in args.padding) else args.padding[0]
    wrote_top_level_debug = False

    for dpi in args.dpi:
        page_path = render_page(pdf_path, args.page, dpi, output_dir / "rendered_pages")
        with Image.open(page_path) as image:
            page = image.convert("RGB")

        for padding in args.padding:
            raw_bbox = normalized_to_image_bbox(x0=x0, y0=y0, x1=x1, y1=y1, image_width=page.width, image_height=page.height)
            px0, py0, px1, py1 = pad_normalized_bbox(x0=x0, y0=y0, x1=x1, y1=y1, padding=padding)
            padded_bbox = normalized_to_image_bbox(
                x0=px0,
                y0=py0,
                x1=px1,
                y1=py1,
                image_width=page.width,
                image_height=page.height,
            )

            if dpi == args.dpi[0] and not wrote_top_level_debug and abs(padding - top_level_padding) < 1e-9:
                write_top_level_debug_files(output_dir, page, raw_bbox, padded_bbox, args.page)
                wrote_top_level_debug = True

            padded_crop = page.crop(padded_bbox)
            for upscale in args.upscale:
                prepared = upscale_image(padded_crop, upscale)
                for variant in args.variant:
                    final = preprocess_variant(prepared, variant)
                    name = f"dpi{dpi}_pad{padding:g}_scale{upscale:g}_{variant}"
                    image_path = output_dir / "variants" / f"{name}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    final.save(image_path)
                    ocr_inputs.append(image_path)
                    ocr_names.append(name)
                    summaries.append(
                        {
                            "name": name,
                            "dpi": dpi,
                            "padding": padding,
                            "upscale": upscale,
                            "preprocessing": variant,
                            "ocr_input": str(image_path),
                            "page_image_size": {"width": page.width, "height": page.height},
                            "raw_image_pixel_bbox": bbox_dict(raw_bbox),
                            "padded_image_pixel_bbox": bbox_dict(padded_bbox),
                            "crop_image_size": {"width": padded_crop.width, "height": padded_crop.height},
                            "ocr_input_size": {"width": final.width, "height": final.height},
                            "nonwhite_ratio": nonwhite_ratio(final),
                            "ocr_status": "not_run",
                            "ocr_result": "",
                        }
                    )

    if args.run_ocr:
        from app.services.deepseek_ocr_pipeline import DeepSeekOcrPipeline

        ocr_results = DeepSeekOcrPipeline().run_ocr_on_images(
            image_paths=ocr_inputs,
            output_dir=output_dir / "ocr_markdown",
            output_names=ocr_names,
            model_name=DEFAULT_DEEPSEEK_OCR_MODEL,
            max_tokens=DEFAULT_DEEPSEEK_OCR_MAX_TOKENS,
            prompt=DEFAULT_DEEPSEEK_OCR_PROMPT,
            crop_mode=DEFAULT_DEEPSEEK_OCR_CROP_MODE,
            min_crops=DEFAULT_DEEPSEEK_OCR_MIN_CROPS,
            max_crops=DEFAULT_DEEPSEEK_OCR_MAX_CROPS,
            base_size=DEFAULT_DEEPSEEK_OCR_BASE_SIZE,
            image_size=DEFAULT_DEEPSEEK_OCR_IMAGE_SIZE,
            skip_repeat=DEFAULT_DEEPSEEK_OCR_SKIP_REPEAT,
            ngram_size=DEFAULT_DEEPSEEK_OCR_NGRAM_SIZE,
            ngram_window=DEFAULT_DEEPSEEK_OCR_NGRAM_WINDOW,
        )
        for item in summaries:
            text = ocr_results.get(str(item["name"]), "").strip()
            item["ocr_status"] = "text" if text else "empty"
            item["ocr_result"] = text
        best = max(summaries, key=lambda item: len(str(item["ocr_result"])))
    else:
        best = None

    debug_json = {
        "pdf": str(pdf_path),
        "page_number": args.page,
        "original_ui_coordinates": {"x": args.x, "y": args.y, "width": args.width, "height": args.height, "type": args.coords},
        "interpreted_coordinate_type": "normalized_top_left",
        "normalized_coordinates": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "runs": summaries,
        "best_working_configuration": best,
    }
    (output_dir / "debug_summary.json").write_text(json.dumps(debug_json, indent=2), encoding="utf-8")
    print(output_dir)
    return 0


def resolve_pdf(job_id: str | None, pdf: Path | None) -> Path:
    if pdf is not None:
        resolved = pdf.resolve()
    elif job_id:
        resolved = JOBS_DIR / job_id / "input.pdf"
    else:
        statuses = sorted(JOBS_DIR.glob("*/input.pdf"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not statuses:
            raise SystemExit("No PDF supplied and no workspace job input.pdf found.")
        resolved = statuses[0]
    if not resolved.exists():
        raise SystemExit(f"PDF not found: {resolved}")
    return resolved


def selection_to_normalized(x: float, y: float, width: float, height: float, coords: str) -> tuple[float, float, float, float]:
    divisor = 100.0 if coords == "percent" else 1.0
    x0 = x / divisor
    y0 = y / divisor
    x1 = (x + width) / divisor
    y1 = (y + height) / divisor
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def render_page(pdf_path: Path, page_number: int, dpi: int, output_dir: Path) -> Path:
    return PageRenderer().render_page(pdf_path, page_number, output_dir / f"full_page_page_{page_number}_dpi{dpi}.png", dpi=dpi)


def write_top_level_debug_files(output_dir: Path, page: Image.Image, raw_bbox: tuple[int, int, int, int], padded_bbox: tuple[int, int, int, int], page_number: int) -> None:
    page.save(output_dir / f"full_page_page_{page_number}.png")
    overlay = page.copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(raw_bbox, outline="red", width=max(3, page.width // 400))
    draw.rectangle(padded_bbox, outline="blue", width=max(3, page.width // 400))
    overlay.save(output_dir / f"full_page_page_{page_number}_with_selection_overlay.png")
    page.crop(raw_bbox).save(output_dir / f"crop_raw_page_{page_number}_selection.png")
    page.crop(padded_bbox).save(output_dir / f"crop_padded_page_{page_number}_selection.png")


def upscale_image(image: Image.Image, factor: float) -> Image.Image:
    if factor <= 1:
        return image.copy()
    return image.resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS)


def preprocess_variant(image: Image.Image, variant: str) -> Image.Image:
    if variant == "none":
        return image.convert("RGB")
    gray = ImageOps.grayscale(image)
    if variant == "grayscale_contrast":
        return ImageEnhance.Contrast(gray).enhance(1.8).convert("RGB")
    if variant == "grayscale_contrast_sharpen":
        enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
        return enhanced.filter(ImageFilter.SHARPEN).convert("RGB")
    if variant == "binarize":
        return gray.point(lambda pixel: 255 if pixel >= 185 else 0, mode="1").convert("RGB")
    raise ValueError(f"Unknown variant: {variant}")


def bbox_dict(bbox: tuple[int, int, int, int]) -> dict[str, int]:
    return {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]}


def nonwhite_ratio(image: Image.Image, threshold: int = 250) -> float:
    gray = image.convert("L")
    histogram = gray.histogram()
    nonwhite = sum(histogram[:threshold])
    return nonwhite / max(gray.width * gray.height, 1)


if __name__ == "__main__":
    raise SystemExit(main())
