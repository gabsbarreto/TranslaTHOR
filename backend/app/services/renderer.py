from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.services.profiler import PipelineProfiler


class PageRenderer:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int, bool, bool, bool, float], Path] = {}

    def render_page(
        self,
        pdf_path: Path,
        page_number: int,
        out_path: Path,
        dpi: int = 100,
        grayscale: bool = False,
        binarize: bool = False,
        denoise: bool = False,
        contrast: float = 1.0,
        profiler: PipelineProfiler | None = None,
        stage_prefix: str = "render",
    ) -> Path:
        key = (str(pdf_path), page_number, dpi, grayscale, binarize, denoise, float(contrast))
        cached = self._cache.get(key)
        if cached and cached.exists():
            return cached
        if out_path.exists():
            self._cache[key] = out_path
            return out_path

        pdf = pdfium.PdfDocument(str(pdf_path))
        page = pdf[page_number - 1]
        scale = dpi / 72
        if profiler is not None:
            with profiler.step(f"{stage_prefix}_pdf_to_bitmap", page=page_number):
                bitmap = page.render(scale=scale)
        else:
            bitmap = page.render(scale=scale)
        pil_image: Image.Image = bitmap.to_pil()

        if grayscale:
            pil_image = ImageOps.grayscale(pil_image)
        if contrast != 1.0:
            pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
        if denoise:
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
        if binarize:
            gray = ImageOps.grayscale(pil_image)
            pil_image = gray.point(lambda x: 0 if x < 140 else 255, mode="1")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if profiler is not None:
            with profiler.step(f"{stage_prefix}_image_save", page=page_number):
                pil_image.save(out_path)
        else:
            pil_image.save(out_path)
        self._cache[key] = out_path
        return out_path