from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

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
    DEFAULT_DPI,
)
from app.services.deepseek_ocr_pipeline import DeepSeekOcrPipeline
from app.services.pdf_inspector import PdfInspector
from app.services.renderer import PageRenderer


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR-2 over rendered PDF pages and write page Markdown.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or ROOT / "workspace" / "ocr_runs" / f"{pdf_path.stem}-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    inspection = PdfInspector().inspect(pdf_path)
    pages = inspection.pages[: args.max_pages] if args.max_pages else inspection.pages
    image_paths = render_pages(pdf_path, pages, output_dir / "images", args.dpi)

    print("Running DeepSeek-OCR-2...")
    DeepSeekOcrPipeline()._run_pdf_ocr(
        image_paths=image_paths,
        output_dir=output_dir / "deepseek_ocr",
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

    index_path = write_index(output_dir, len(image_paths))
    print(f"OCR output written to: {index_path}")
    return 0


def render_pages(pdf_path: Path, pages, images_dir: Path, dpi: int) -> list[Path]:
    renderer = PageRenderer()
    image_paths: list[Path] = []
    for page in pages:
        image_paths.append(
            renderer.render_page(
                pdf_path,
                page.page_number,
                images_dir / f"page_{page.page_number:04d}.png",
                dpi=dpi,
            )
        )
    return image_paths


def write_index(output_dir: Path, page_count: int) -> Path:
    lines = [
        "# DeepSeek OCR Markdown",
        "",
        f"- Pages: {page_count}",
        f"- Output: `{relative_or_missing(output_dir, output_dir / 'deepseek_ocr')}`",
        "",
        "| Page | Chars | File |",
        "|---:|---:|---|",
    ]
    for index in range(1, page_count + 1):
        page_name = f"page_{index:04d}.md"
        page_path = output_dir / "deepseek_ocr" / page_name
        chars = markdown_chars(page_path)
        link = markdown_link(page_name, page_path, output_dir) if page_path.exists() else ""
        lines.append(f"| {index} | {chars} | {link} |")

    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def markdown_chars(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8", errors="ignore"))


def markdown_link(label: str, path: Path, base: Path) -> str:
    return f"[{label}]({path.relative_to(base).as_posix()})"


def relative_or_missing(base: Path, path: Path) -> str:
    return path.relative_to(base).as_posix() if path.exists() else "missing"


if __name__ == "__main__":
    raise SystemExit(main())
