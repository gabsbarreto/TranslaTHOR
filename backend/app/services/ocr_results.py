from __future__ import annotations

from pathlib import Path

from app.models.regions import OcrResultsPayload


def summarize_ocr_results(payload: OcrResultsPayload) -> dict:
    all_pages = {item.page_number for item in payload.results}
    pages_with_text = {item.page_number for item in payload.results if item.ocr_text.strip()}
    pages_without_text = all_pages - pages_with_text
    nonempty_region_count = sum(1 for item in payload.results if item.ocr_text.strip())
    return {
        "total_region_count": len(payload.results),
        "nonempty_region_count": nonempty_region_count,
        "empty_region_count": len(payload.results) - nonempty_region_count,
        "selected_page_count": len(all_pages),
        "pages_with_text_count": len(pages_with_text),
        "pages_without_text_count": len(pages_without_text),
        "pages_with_text": sorted(pages_with_text),
        "pages_without_text": sorted(pages_without_text),
    }


def selected_ocr_progress_from_event(event: dict) -> tuple[float, str] | None:
    """Map DeepSeek OCR worker events onto the existing job progress bar."""
    event_name = event.get("event")
    phase = str(event.get("phase") or "primary")
    phase_label = "OCR selected regions" if phase == "primary" else "Retrying empty OCR regions"

    if event_name == "model_loading":
        return 0.35, "Loading OCR model for selected regions"
    if event_name == "model_loaded":
        total = _positive_int(event.get("pages"))
        if total is None:
            return 0.36, "OCR model loaded; processing selected regions"
        return 0.36, f"OCR model loaded; processing {total} selected region(s)"
    if event_name not in {"page_started", "page_done"}:
        return None

    index = _positive_int(event.get("index"))
    total = _positive_int(event.get("total"))
    if index is None or total is None:
        return None

    if event_name == "page_started":
        fraction = max(0.0, min((index - 1) / total, 1.0))
        return round(0.36 + 0.48 * fraction, 3), f"{phase_label}: {index}/{total}"

    fraction = max(0.0, min(index / total, 1.0))
    chars = _positive_int(event.get("chars"))
    chars_note = f"; {chars} characters" if chars is not None else ""
    return round(0.36 + 0.48 * fraction, 3), f"{phase_label}: {index}/{total} complete{chars_note}"


def write_selected_ocr_source_markdown(job_dir: Path, payload: OcrResultsPayload) -> Path:
    out_path = job_dir / "artifacts" / "source.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_order: dict[int, list] = {}
    for item in sorted(
        payload.results,
        key=lambda result: (result.page_number, result.reading_order, result.box_id),
    ):
        if not item.ocr_text.strip():
            continue
        page_order.setdefault(item.page_number, []).append(item)

    lines: list[str] = []
    for page_number in sorted(page_order):
        lines.append(f"<!-- page: {page_number} -->")
        for item in page_order[page_number]:
            lines.append(item.ocr_text.strip())
            lines.append("")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return out_path


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
