from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.regions import OcrRegionResult, OcrResultsPayload


SENTENCE_ENDINGS = (".", "?", "!", ":", ";", ")", '"', "'", "”", "’")


@dataclass
class OcrTextSpan:
    page_number: int
    box_id: str
    reading_order: int
    start: int
    end: int


@dataclass
class CompiledOcrDocument:
    text: str
    spans: list[OcrTextSpan] = field(default_factory=list)


def collect_ocr_results_for_document(payload: OcrResultsPayload) -> list[OcrRegionResult]:
    return [item for item in payload.results if item.ocr_text.strip()]


def sort_ocr_results_by_page_and_order(results: list[OcrRegionResult]) -> list[OcrRegionResult]:
    return sorted(results, key=lambda item: (item.page_number, item.reading_order, item.box_id))


def compile_ocr_results_to_document_text(results: list[OcrRegionResult] | OcrResultsPayload) -> CompiledOcrDocument:
    if isinstance(results, OcrResultsPayload):
        items = collect_ocr_results_for_document(results)
    else:
        items = [item for item in results if item.ocr_text.strip()]
    ordered = sort_ocr_results_by_page_and_order(items)

    text = ""
    spans: list[OcrTextSpan] = []
    for item in ordered:
        block = item.ocr_text.strip()
        if not block:
            continue
        if not text:
            start = 0
            text = block
        else:
            previous, separator, next_text = join_ocr_blocks(text, block)
            text = previous + separator
            start = len(text)
            text += next_text
        spans.append(
            OcrTextSpan(
                page_number=item.page_number,
                box_id=item.box_id,
                reading_order=item.reading_order,
                start=start,
                end=len(text),
            )
        )

    return CompiledOcrDocument(text=text, spans=spans)


def join_ocr_blocks(previous_text: str, next_text: str) -> tuple[str, str, str]:
    previous = previous_text.rstrip()
    current = next_text.lstrip()
    previous, current = fix_cross_boundary_hyphenation(previous, current)
    if _should_join_with_space(previous, current):
        return previous, " ", current
    return previous, "\n\n", current


def fix_cross_boundary_hyphenation(previous_text: str, next_text: str) -> tuple[str, str]:
    previous = previous_text.rstrip()
    current = next_text.lstrip()
    match = re.search(r"(?P<prefix>[A-Za-zÀ-ÖØ-öø-ÿ]{2,})-\s*$", previous)
    next_match = re.match(r"(?P<suffix>[A-Za-zÀ-ÖØ-öø-ÿ]{2,})(?P<rest>.*)$", current, flags=re.DOTALL)
    if not match or not next_match:
        return previous, current
    return previous[: match.start()] + match.group("prefix") + next_match.group("suffix"), next_match.group("rest").lstrip()


def _should_join_with_space(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if _looks_like_heading_start(current):
        return False
    if current[:1].islower():
        return True
    return not previous.endswith(SENTENCE_ENDINGS)


def _looks_like_heading_start(text: str) -> bool:
    first_line = text.strip().splitlines()[0].strip()
    if not first_line:
        return False
    if first_line.startswith("#"):
        return True
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", first_line)
    if 1 <= len(words) <= 7 and len(first_line) <= 80:
        lower_words = sum(1 for word in words if word[:1].islower())
        if lower_words == 0:
            return True
    return False
