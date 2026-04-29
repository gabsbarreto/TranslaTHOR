from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from app.models.inspection import PdfInspection
from app.models.regions import (
    CoordinateSpace,
    OcrRegionResult,
    OcrResultsPayload,
    PageRegionPayload,
    Region,
    RegionSource,
    RegionType,
)
from app.services.coordinate_utils import normalize_region, normalized_to_image_bbox, pad_normalized_bbox
from app.services.deepseek_ocr_pipeline import DeepSeekOcrPipeline
from app.services.layout_detectors import HybridLayoutDetector
from app.services.pdf_inspector import PdfInspector
from app.services.region_store import RegionStore
from app.services.renderer import PageRenderer

DEFAULT_BOX_INSET_RATIO = 0.03
SELECTED_OCR_PADDING = 0.10
SELECTED_OCR_RETRY_CONTRAST = 1.8
SELECTED_OCR_RETRY_COORD_SNAP = 0.001


class OcrRegionService:
    def __init__(self) -> None:
        self.renderer = PageRenderer()
        self.inspector = PdfInspector()
        self.region_store = RegionStore()
        self.detector = HybridLayoutDetector()
        self.deepseek = DeepSeekOcrPipeline()

    def inspect_pdf(self, pdf_path: Path) -> PdfInspection:
        return self.inspector.inspect(pdf_path)

    def render_page_image(self, *, pdf_path: Path, job_dir: Path, page_number: int, dpi: int = 150) -> tuple[Path, int, int]:
        out_path = job_dir / "page_images" / f"page_{page_number:04d}_dpi{dpi}.png"
        image_path = self.renderer.render_page(
            pdf_path,
            page_number,
            out_path,
            dpi=dpi,
        )
        with Image.open(image_path) as image:
            width, height = image.size
        return image_path, width, height

    def get_or_detect_regions(
        self,
        *,
        pdf_file_id: str,
        pdf_path: Path,
        job_dir: Path,
        page_number: int,
        dpi: int = 300,
        refresh: bool = False,
        detailed: bool = False,
        replace_saved: bool = False,
    ) -> PageRegionPayload:
        cached = self.region_store.load_page_regions(job_dir, page_number)
        if cached is not None and (not refresh or not replace_saved):
            if refresh and not replace_saved:
                raise ValueError("Saved boxes already exist. Confirm reset/detection before replacing them.")
            return cached

        inspection = self.inspect_pdf(pdf_path)
        page = next((item for item in inspection.pages if item.page_number == page_number), None)
        if page is None:
            raise ValueError(f"Page {page_number} does not exist")

        _image_path, image_width, image_height = self.render_page_image(
            pdf_path=pdf_path,
            job_dir=job_dir,
            page_number=page_number,
            dpi=dpi,
        )
        if detailed:
            decision = self.detector.detect(
                pdf_path=pdf_path,
                page_number=page_number,
                page_width=page.width,
                page_height=page.height,
                has_embedded_text=page.has_embedded_text,
                embedded_text_quality=page.embedded_text_quality,
            )
            detector_name = decision.detector_name
            regions = decision.regions
        else:
            detector_name = "default_full_page"
            regions = [self._default_full_page_region(page_number)]

        payload = PageRegionPayload(
            pdf_file_id=pdf_file_id,
            page_number=page_number,
            page_width=page.width,
            page_height=page.height,
            image_width=image_width,
            image_height=image_height,
            coordinate_space=CoordinateSpace.NORMALIZED,
            detector=detector_name,
            regions=regions,
        )
        self.region_store.save_page_regions(job_dir, payload)
        self.region_store.save_all_regions(job_dir)
        return payload

    def _default_full_page_region(self, page_number: int) -> Region:
        return Region(
            id=f"default-full-page-{page_number}",
            page_number=page_number,
            x0=DEFAULT_BOX_INSET_RATIO,
            y0=DEFAULT_BOX_INSET_RATIO,
            x1=1.0 - DEFAULT_BOX_INSET_RATIO,
            y1=1.0 - DEFAULT_BOX_INSET_RATIO,
            coordinate_space=CoordinateSpace.NORMALIZED,
            type=RegionType.PAGE,
            selected=True,
            reading_order=1,
            source=RegionSource.DEFAULT_FULL_PAGE,
        )

    def save_regions(
        self,
        *,
        job_dir: Path,
        payload: PageRegionPayload,
    ) -> PageRegionPayload:
        normalized: list[Region] = []
        for region in payload.regions:
            if region.source == RegionSource.PYMUPDF:
                source = region.source
            else:
                source = region.source
            item = normalize_region(region, page_width=payload.page_width, page_height=payload.page_height).model_copy(
                update={
                    "page_number": payload.page_number,
                    "coordinate_space": CoordinateSpace.NORMALIZED,
                    "source": source,
                }
            )
            normalized.append(item)

        normalized = sorted(normalized, key=lambda item: (item.reading_order, item.y0, item.x0, item.id))
        for idx, item in enumerate(normalized, start=1):
            normalized[idx - 1] = item.model_copy(update={"reading_order": idx})

        out = payload.model_copy(update={"coordinate_space": CoordinateSpace.NORMALIZED, "regions": normalized})
        self.region_store.save_page_regions(job_dir, out)
        self.region_store.save_all_regions(job_dir)
        return out

    def run_selected_ocr(
        self,
        *,
        pdf_file_id: str,
        pdf_path: Path,
        job_dir: Path,
        dpi: int,
        model_name: str,
        max_tokens: int,
        prompt: str,
        crop_mode: bool,
        min_crops: int,
        max_crops: int,
        base_size: int,
        image_size: int,
        skip_repeat: bool,
        ngram_size: int,
        ngram_window: int,
        page_number: int | None = None,
        box_id: str | None = None,
        on_ocr_progress: Callable[[dict], None] | None = None,
    ) -> OcrResultsPayload:
        all_pages = self.region_store.list_page_regions(job_dir)
        if not all_pages:
            raise ValueError("No saved regions found. Save or auto-detect regions first.")

        selected_entries: list[tuple[PageRegionPayload, Region]] = []
        for page in sorted(all_pages, key=lambda item: item.page_number):
            if page_number is not None and page.page_number != page_number:
                continue
            for region in sorted(page.regions, key=lambda item: (item.reading_order, item.id)):
                if box_id is not None and region.id != box_id:
                    continue
                if region.selected:
                    selected_entries.append((page, region))

        if not selected_entries:
            raise ValueError("No selected regions were found.")

        masked_dir = self.region_store.ocr_region_dir(job_dir) / "masked_pages"
        retry_dir = self.region_store.ocr_region_dir(job_dir) / "retry_crops"
        md_dir = self.region_store.ocr_region_dir(job_dir) / "markdown"
        debug_dir = self.region_store.ocr_region_dir(job_dir) / "debug"
        masked_dir.mkdir(parents=True, exist_ok=True)
        retry_dir.mkdir(parents=True, exist_ok=True)
        md_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        image_paths: list[Path] = []
        output_names: list[str] = []
        mapping: dict[str, tuple[PageRegionPayload, Region]] = {}
        ocr_image_metadata: dict[str, dict[str, str | int | float | bool | None]] = {}

        page_images: dict[int, Path] = {}
        for page_payload, region in selected_entries:
            page_number = page_payload.page_number
            if page_number not in page_images:
                rendered_path, _, _ = self.render_page_image(
                    pdf_path=pdf_path,
                    job_dir=job_dir,
                    page_number=page_number,
                    dpi=dpi,
                )
                page_images[page_number] = rendered_path

            image_path = page_images[page_number]
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", region.id)
            output_name = f"p{page_number:04d}_{safe_id}"
            masked_path = masked_dir / f"{output_name}.png"

            self._save_masked_page_region(
                image_path=image_path,
                region=region,
                output_path=masked_path,
                padding=SELECTED_OCR_PADDING,
            )
            with Image.open(masked_path) as masked_image:
                debug_payload = self._write_region_debug_images(
                    image_path=image_path,
                    region=region,
                    output_name=output_name,
                    output_dir=debug_dir,
                    padding=SELECTED_OCR_PADDING,
                )
                ocr_image_metadata[output_name] = {
                    "ocr_image_mode": "masked_page_padded",
                    "ocr_image_path": str(masked_path),
                    "ocr_image_width": masked_image.width,
                    "ocr_image_height": masked_image.height,
                    "ocr_padding": SELECTED_OCR_PADDING,
                    "debug_json_path": str(debug_payload["debug_json_path"]),
                    "raw_crop_width": debug_payload["raw_crop_width"],
                    "raw_crop_height": debug_payload["raw_crop_height"],
                    "padded_crop_width": debug_payload["padded_crop_width"],
                    "padded_crop_height": debug_payload["padded_crop_height"],
                    "raw_crop_nonwhite_ratio": debug_payload["raw_crop_nonwhite_ratio"],
                    "padded_crop_nonwhite_ratio": debug_payload["padded_crop_nonwhite_ratio"],
                }

            image_paths.append(masked_path)
            output_names.append(output_name)
            mapping[output_name] = (page_payload, region)

        markdown_results = self.deepseek.run_ocr_on_images(
            image_paths=image_paths,
            output_dir=md_dir,
            output_names=output_names,
            model_name=model_name,
            max_tokens=max_tokens,
            prompt=prompt,
            crop_mode=crop_mode,
            min_crops=min_crops,
            max_crops=max_crops,
            base_size=base_size,
            image_size=image_size,
            skip_repeat=skip_repeat,
            ngram_size=ngram_size,
            ngram_window=ngram_window,
            on_ocr_progress=self._phase_progress_callback(on_ocr_progress, "primary"),
        )

        empty_output_names = [name for name in output_names if not markdown_results.get(name, "").strip()]
        if empty_output_names:
            retry_paths: list[Path] = []
            retry_names: list[str] = []
            retry_to_original: dict[str, str] = {}
            for output_name in empty_output_names:
                _page_payload, region = mapping[output_name]
                image_path = page_images[region.page_number]
                retry_name = f"{output_name}_retry_gray_contrast"
                retry_path = retry_dir / f"{retry_name}.png"
                self._save_retry_crop_region(image_path=image_path, region=region, output_path=retry_path)
                with Image.open(retry_path) as retry_image:
                    ocr_image_metadata[output_name].update(
                        {
                            "ocr_retry_used": True,
                            "ocr_retry_mode": "snapped_crop_grayscale_contrast",
                            "ocr_retry_path": str(retry_path),
                            "ocr_retry_width": retry_image.width,
                            "ocr_retry_height": retry_image.height,
                            "ocr_retry_contrast": SELECTED_OCR_RETRY_CONTRAST,
                            "ocr_retry_coordinate_snap": SELECTED_OCR_RETRY_COORD_SNAP,
                        }
                    )
                retry_paths.append(retry_path)
                retry_names.append(retry_name)
                retry_to_original[retry_name] = output_name

            retry_results = self.deepseek.run_ocr_on_images(
                image_paths=retry_paths,
                output_dir=md_dir,
                output_names=retry_names,
                model_name=model_name,
                max_tokens=max_tokens,
                prompt=prompt,
                crop_mode=crop_mode,
                min_crops=min_crops,
                max_crops=max_crops,
                base_size=base_size,
                image_size=image_size,
                skip_repeat=skip_repeat,
                ngram_size=ngram_size,
                ngram_window=ngram_window,
                on_ocr_progress=self._phase_progress_callback(on_ocr_progress, "retry"),
            )
            for retry_name, retry_text in retry_results.items():
                original_name = retry_to_original[retry_name]
                if retry_text.strip():
                    markdown_results[original_name] = retry_text
                    ocr_image_metadata[original_name]["ocr_image_mode"] = "snapped_crop_grayscale_contrast_retry"
                    ocr_image_metadata[original_name]["ocr_image_path"] = ocr_image_metadata[original_name]["ocr_retry_path"]
                    ocr_image_metadata[original_name]["ocr_image_width"] = ocr_image_metadata[original_name]["ocr_retry_width"]
                    ocr_image_metadata[original_name]["ocr_image_height"] = ocr_image_metadata[original_name]["ocr_retry_height"]

        results: list[OcrRegionResult] = []
        for output_name in output_names:
            page_payload, region = mapping[output_name]
            text = markdown_results.get(output_name, "").strip()
            results.append(
                OcrRegionResult(
                    pdf_file_id=pdf_file_id,
                    page_number=region.page_number,
                    box_id=region.id,
                    x0=region.x0,
                    y0=region.y0,
                    x1=region.x1,
                    y1=region.y1,
                    coordinate_space=CoordinateSpace.NORMALIZED,
                    box_type=region.type,
                    reading_order=region.reading_order,
                    ocr_text=text,
                    ocr_confidence=None,
                    metadata=ocr_image_metadata.get(output_name, {}),
                )
            )

        payload = OcrResultsPayload(
            pdf_file_id=pdf_file_id,
            results=sorted(results, key=lambda item: (item.page_number, item.reading_order, item.box_id)),
        )
        self.region_store.save_ocr_results(job_dir, payload)
        return payload

    def _phase_progress_callback(
        self,
        on_ocr_progress: Callable[[dict], None] | None,
        phase: str,
    ) -> Callable[[dict], None] | None:
        if on_ocr_progress is None:
            return None

        def handle(event: dict) -> None:
            event_with_phase = dict(event)
            event_with_phase["phase"] = phase
            on_ocr_progress(event_with_phase)

        return handle

    def _save_masked_page_region(self, *, image_path: Path, region: Region, output_path: Path, padding: float = 0.0) -> None:
        with Image.open(image_path) as image:
            source = image.convert("RGB")
            px0, py0, px1, py1 = pad_normalized_bbox(
                x0=region.x0,
                y0=region.y0,
                x1=region.x1,
                y1=region.y1,
                padding=padding,
            )
            left, top, right, bottom = normalized_to_image_bbox(
                x0=px0,
                y0=py0,
                x1=px1,
                y1=py1,
                image_width=source.width,
                image_height=source.height,
            )
            masked = Image.new("RGB", source.size, "white")
            masked.paste(source.crop((left, top, right, bottom)), (left, top))
            masked.save(output_path)

    def _write_region_debug_images(
        self,
        *,
        image_path: Path,
        region: Region,
        output_name: str,
        output_dir: Path,
        padding: float,
    ) -> dict[str, str | int | float]:
        import json

        with Image.open(image_path) as image:
            source = image.convert("RGB")
            raw_bbox = normalized_to_image_bbox(
                x0=region.x0,
                y0=region.y0,
                x1=region.x1,
                y1=region.y1,
                image_width=source.width,
                image_height=source.height,
            )
            px0, py0, px1, py1 = pad_normalized_bbox(
                x0=region.x0,
                y0=region.y0,
                x1=region.x1,
                y1=region.y1,
                padding=padding,
            )
            padded_bbox = normalized_to_image_bbox(
                x0=px0,
                y0=py0,
                x1=px1,
                y1=py1,
                image_width=source.width,
                image_height=source.height,
            )

            full_page_path = output_dir / f"{output_name}_full_page.png"
            overlay_path = output_dir / f"{output_name}_full_page_with_selection_overlay.png"
            raw_crop_path = output_dir / f"{output_name}_crop_raw.png"
            padded_crop_path = output_dir / f"{output_name}_crop_padded.png"
            debug_json_path = output_dir / f"{output_name}_debug.json"

            shutil.copyfile(image_path, full_page_path)
            overlay = source.copy()
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(raw_bbox, outline="red", width=max(3, source.width // 400))
            draw.rectangle(padded_bbox, outline="blue", width=max(3, source.width // 400))
            overlay.save(overlay_path)

            raw_crop = source.crop(raw_bbox)
            padded_crop = source.crop(padded_bbox)
            raw_crop.save(raw_crop_path)
            padded_crop.save(padded_crop_path)

            payload = {
                "page_number": region.page_number,
                "box_id": region.id,
                "coordinate_space": "normalized_top_left",
                "ui_coordinates_percent": {
                    "x": region.x0 * 100,
                    "y": region.y0 * 100,
                    "width": (region.x1 - region.x0) * 100,
                    "height": (region.y1 - region.y0) * 100,
                },
                "normalized_coordinates": {
                    "x0": region.x0,
                    "y0": region.y0,
                    "x1": region.x1,
                    "y1": region.y1,
                },
                "padded_normalized_coordinates": {"x0": px0, "y0": py0, "x1": px1, "y1": py1},
                "image_pixel_bbox": {"left": raw_bbox[0], "top": raw_bbox[1], "right": raw_bbox[2], "bottom": raw_bbox[3]},
                "padded_image_pixel_bbox": {
                    "left": padded_bbox[0],
                    "top": padded_bbox[1],
                    "right": padded_bbox[2],
                    "bottom": padded_bbox[3],
                },
                "page_image_size": {"width": source.width, "height": source.height},
                "crop_image_size": {"width": raw_crop.width, "height": raw_crop.height},
                "padded_crop_image_size": {"width": padded_crop.width, "height": padded_crop.height},
                "padding": padding,
                "raw_crop_nonwhite_ratio": self._nonwhite_ratio(raw_crop),
                "padded_crop_nonwhite_ratio": self._nonwhite_ratio(padded_crop),
                "files": {
                    "full_page": str(full_page_path),
                    "overlay": str(overlay_path),
                    "raw_crop": str(raw_crop_path),
                    "padded_crop": str(padded_crop_path),
                },
            }
            debug_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return {
            "debug_json_path": str(debug_json_path),
            "raw_crop_width": raw_crop.width,
            "raw_crop_height": raw_crop.height,
            "padded_crop_width": padded_crop.width,
            "padded_crop_height": padded_crop.height,
            "raw_crop_nonwhite_ratio": payload["raw_crop_nonwhite_ratio"],
            "padded_crop_nonwhite_ratio": payload["padded_crop_nonwhite_ratio"],
        }

    def _nonwhite_ratio(self, image: Image.Image, threshold: int = 250) -> float:
        gray = image.convert("L")
        histogram = gray.histogram()
        nonwhite = sum(histogram[:threshold])
        total = max(gray.width * gray.height, 1)
        return nonwhite / total

    def _save_retry_crop_region(self, *, image_path: Path, region: Region, output_path: Path) -> None:
        with Image.open(image_path) as image:
            source = image.convert("RGB")
            x0, y0, x1, y1 = self._snap_retry_bbox(region)
            bbox = normalized_to_image_bbox(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                image_width=source.width,
                image_height=source.height,
            )
            crop = source.crop(bbox)
            gray = ImageOps.grayscale(crop)
            enhanced = ImageEnhance.Contrast(gray).enhance(SELECTED_OCR_RETRY_CONTRAST)
            enhanced.convert("RGB").save(output_path)

    def _snap_retry_bbox(self, region: Region) -> tuple[float, float, float, float]:
        """Snap retry crops to 0.1 percentage-point UI increments.

        Konva/browser editing stores normalized floats with more precision than
        the user-visible percentages. For DeepSeek OCR2, page-10 testing showed
        that a one-pixel boundary difference can flip a cropped region from
        readable to empty. Snapping only the empty-result retry crop recreates
        the user's visible coordinates while leaving the primary OCR image
        untouched.
        """
        snap = SELECTED_OCR_RETRY_COORD_SNAP
        values = [round(value / snap) * snap for value in (region.x0, region.y0, region.x1, region.y1)]
        x0, y0, x1, y1 = [min(1.0, max(0.0, value)) for value in values]
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
