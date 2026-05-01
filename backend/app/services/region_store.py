from __future__ import annotations

from pathlib import Path

from app.models.regions import OcrResultsPayload, PageRegionPayload


class RegionStore:
    def region_dir(self, job_dir: Path) -> Path:
        return job_dir / "regions"

    def ocr_region_dir(self, job_dir: Path) -> Path:
        return job_dir / "ocr_regions"

    def page_region_path(self, job_dir: Path, page_number: int) -> Path:
        return self.region_dir(job_dir) / f"page_{page_number:04d}.json"

    def all_regions_path(self, job_dir: Path) -> Path:
        return self.region_dir(job_dir) / "all_pages.json"

    def ocr_results_path(self, job_dir: Path) -> Path:
        return self.ocr_region_dir(job_dir) / "results.json"

    def save_page_regions(self, job_dir: Path, payload: PageRegionPayload) -> None:
        region_dir = self.region_dir(job_dir)
        region_dir.mkdir(parents=True, exist_ok=True)
        self.page_region_path(job_dir, payload.page_number).write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    def load_page_regions(self, job_dir: Path, page_number: int) -> PageRegionPayload | None:
        path = self.page_region_path(job_dir, page_number)
        if not path.exists():
            return None
        return PageRegionPayload.model_validate_json(path.read_text(encoding="utf-8"))

    def list_page_regions(self, job_dir: Path) -> list[PageRegionPayload]:
        out: list[PageRegionPayload] = []
        for path in sorted(self.region_dir(job_dir).glob("page_*.json")):
            out.append(PageRegionPayload.model_validate_json(path.read_text(encoding="utf-8")))
        return out

    def save_all_regions(self, job_dir: Path) -> None:
        pages = self.list_page_regions(job_dir)
        payload = {
            "pages": [item.model_dump(mode="json") for item in pages],
        }
        all_path = self.all_regions_path(job_dir)
        all_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        all_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_ocr_results(self, job_dir: Path, payload: OcrResultsPayload) -> None:
        out_dir = self.ocr_region_dir(job_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_results_path(job_dir).write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    def load_ocr_results(self, job_dir: Path) -> OcrResultsPayload | None:
        path = self.ocr_results_path(job_dir)
        if not path.exists():
            return None
        return OcrResultsPayload.model_validate_json(path.read_text(encoding="utf-8"))
