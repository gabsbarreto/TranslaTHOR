from __future__ import annotations

import math
import re
from pathlib import Path

from app.services.pdf_extraction.models import PDFTypeDetectionResult, PageTextStats

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None


class PDFTypeDetector:
    """Classify whether Marker should trust embedded text or re-OCR the PDF.

    Marker currently applies OCR options at document level in normal CLI use. We still
    record per-page stats so future page-splitting can make page-level decisions.
    """

    _WORD_RE = re.compile(r"[\w\-']+", re.UNICODE)
    _GARBAGE_RUN_RE = re.compile(r"([^\w\s])\1{3,}")

    def detect(self, pdf_path: Path) -> PDFTypeDetectionResult:
        if pdfplumber is None or PdfReader is None:
            raise RuntimeError("pypdf and pdfplumber are required for PDF type detection.")

        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}
        pages: list[PageTextStats] = []
        warnings: list[str] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                words = self._WORD_RE.findall(text)
                clean_text = text.strip()
                char_count = len(clean_text)
                word_count = len(words)
                alnum_count = sum(1 for ch in clean_text if ch.isalnum())
                alpha_count = sum(1 for ch in clean_text if ch.isalpha())
                non_ascii_count = sum(1 for ch in clean_text if ord(ch) > 127)
                replacement_count = clean_text.count("\ufffd")
                weird_count = sum(1 for ch in clean_text if self._is_weird_symbol(ch))
                avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0
                repeated_score = self._repeated_garbage_score(clean_text)
                image_area_ratio = self._image_area_ratio(page)

                alnum_ratio = alnum_count / max(char_count, 1)
                alpha_ratio = alpha_count / max(char_count, 1)
                non_ascii_ratio = non_ascii_count / max(char_count, 1)
                weird_symbol_ratio = weird_count / max(char_count, 1)

                looks_garbled = (
                    char_count >= 40
                    and (
                        alnum_ratio < 0.45
                        or replacement_count >= 3
                        or weird_symbol_ratio > 0.22
                        or repeated_score > 0.12
                        or avg_word_len > 18
                    )
                )
                looks_meaningful = (
                    word_count >= 25
                    and char_count >= 120
                    and alnum_ratio >= 0.55
                    and weird_symbol_ratio <= 0.18
                    and replacement_count <= 2
                    and avg_word_len <= 14
                )
                looks_image_dominant = image_area_ratio >= 0.55 or (len(page.images or []) > 0 and char_count < 80)

                pages.append(
                    PageTextStats(
                        page_number=index,
                        width=float(page.width),
                        height=float(page.height),
                        char_count=char_count,
                        word_count=word_count,
                        alnum_ratio=round(alnum_ratio, 4),
                        alpha_ratio=round(alpha_ratio, 4),
                        non_ascii_ratio=round(non_ascii_ratio, 4),
                        replacement_char_count=replacement_count,
                        weird_symbol_ratio=round(weird_symbol_ratio, 4),
                        average_word_length=round(avg_word_len, 2),
                        repeated_garbage_score=round(repeated_score, 4),
                        image_count=len(page.images or []),
                        image_area_ratio=round(image_area_ratio, 4),
                        has_selectable_text=char_count > 0,
                        looks_meaningful=looks_meaningful,
                        looks_garbled=looks_garbled,
                        looks_image_dominant=looks_image_dominant,
                    )
                )

        if not pages:
            return PDFTypeDetectionResult(
                classification="unknown",
                page_count=0,
                pages=[],
                embedded_text_chars=0,
                embedded_text_words=0,
                meaningful_page_count=0,
                garbled_page_count=0,
                image_dominant_page_count=0,
                scanned_page_count=0,
                mixed=False,
                warnings=["PDF has no readable pages."],
                metadata={"filename": pdf_path.name},
            )

        text_chars = sum(page.char_count for page in pages)
        text_words = sum(page.word_count for page in pages)
        meaningful = sum(1 for page in pages if page.looks_meaningful)
        garbled = sum(1 for page in pages if page.looks_garbled)
        image_dominant = sum(1 for page in pages if page.looks_image_dominant)
        scanned = sum(1 for page in pages if page.char_count < 40 and page.looks_image_dominant)
        no_text_pages = sum(1 for page in pages if page.char_count < 20)
        hidden_ocr_pages = sum(
            1
            for page in pages
            if page.has_selectable_text
            and page.char_count >= 120
            and page.image_area_ratio >= 0.85
            and page.looks_meaningful
        )

        page_count = len(pages)
        meaningful_ratio = meaningful / page_count
        garbled_ratio = garbled / page_count
        scanned_ratio = scanned / page_count
        no_text_ratio = no_text_pages / page_count
        hidden_ocr_ratio = hidden_ocr_pages / page_count

        mixed = (
            page_count > 1
            and meaningful > 0
            and (scanned > 0 or garbled > 0 or no_text_pages > 0)
            and meaningful_ratio < 0.85
        )
        suspicious_hidden_ocr = (
            hidden_ocr_ratio >= 0.65
            and image_dominant / page_count >= 0.65
            and text_chars >= max(120, page_count * 80)
        )

        if text_chars < max(80, page_count * 30) and (image_dominant >= max(1, math.ceil(page_count * 0.4))):
            classification = "scanned_no_text"
        elif suspicious_hidden_ocr:
            classification = "bad_hidden_ocr"
            warnings.append(
                "Most pages contain full-page images plus selectable text. Treating the text layer as hidden OCR instead of trusted digital text."
            )
        elif garbled_ratio >= 0.35 and text_chars >= 100:
            classification = "bad_hidden_ocr"
        elif mixed:
            classification = "mixed"
            warnings.append(
                "Mixed PDFs are processed with one whole-document Marker mode. Page-level Marker OCR settings are not applied by this integration."
            )
        elif meaningful_ratio >= 0.65 and text_words >= max(40, page_count * 20):
            classification = "digital_good_text"
        elif no_text_ratio >= 0.65:
            classification = "scanned_no_text"
        else:
            classification = "unknown"
            warnings.append("PDF text quality was inconclusive; auto mode will choose the conservative Marker path.")

        return PDFTypeDetectionResult(
            classification=classification,
            page_count=len(reader.pages),
            pages=pages,
            embedded_text_chars=text_chars,
            embedded_text_words=text_words,
            meaningful_page_count=meaningful,
            garbled_page_count=garbled,
            image_dominant_page_count=image_dominant,
            scanned_page_count=scanned,
            mixed=mixed,
            warnings=warnings,
            metadata={
                "filename": pdf_path.name,
                "title": metadata.get("/Title"),
                "author": metadata.get("/Author"),
                "meaningful_page_ratio": round(meaningful_ratio, 4),
                "garbled_page_ratio": round(garbled_ratio, 4),
                "scanned_page_ratio": round(scanned_ratio, 4),
                "no_text_page_ratio": round(no_text_ratio, 4),
                "hidden_ocr_page_count": hidden_ocr_pages,
                "hidden_ocr_page_ratio": round(hidden_ocr_ratio, 4),
                "suspicious_hidden_ocr": suspicious_hidden_ocr,
            },
        )

    def _is_weird_symbol(self, ch: str) -> bool:
        if ch.isspace() or ch.isalnum():
            return False
        if ch in ".,;:!?()[]{}'\"-/+%$#&*@=<>|_`~^\\\n\t\r":
            return False
        return True

    def _repeated_garbage_score(self, text: str) -> float:
        if not text:
            return 0.0
        repeated = sum(len(match.group(0)) for match in self._GARBAGE_RUN_RE.finditer(text))
        short_weird_words = sum(1 for word in text.split() if len(word) <= 2 and not any(ch.isalnum() for ch in word))
        return min(1.0, (repeated + short_weird_words) / max(len(text), 1))

    def _image_area_ratio(self, page) -> float:
        page_area = max(float(page.width) * float(page.height), 1.0)
        total = 0.0
        for image in page.images or []:
            try:
                width = abs(float(image.get("x1", 0.0)) - float(image.get("x0", 0.0)))
                height = abs(float(image.get("bottom", 0.0)) - float(image.get("top", 0.0)))
                total += max(width, 0.0) * max(height, 0.0)
            except Exception:
                continue
        return max(0.0, min(total / page_area, 1.0))
