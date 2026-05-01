from __future__ import annotations

from pathlib import Path

from app.models.inspection import PageInspection, PdfInspection

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - exercised only in lightweight test envs
    PdfReader = None

try:
    import pdfplumber
except Exception:  # pragma: no cover - exercised only in lightweight test envs
    pdfplumber = None


class PdfInspector:
    def inspect(self, pdf_path: Path) -> PdfInspection:
        if pdfplumber is None or PdfReader is None:
            raise RuntimeError("pypdf and pdfplumber are required for PDF inspection. Install project dependencies first.")
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}
        pages: list[PageInspection] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                words = page.extract_words() or []
                images = page.images or []
                table_count = len(page.find_tables() or [])
                area = max(page.width * page.height, 1)
                density = len(words) / (area / 100000.0)
                quality = min(1.0, (len(text) / 1200.0) * 0.7 + min(density / 20.0, 1.0) * 0.3)
                alpha_count = sum(1 for ch in text if ch.isalpha())
                non_ascii_count = sum(1 for ch in text if ord(ch) > 127)
                text_len = len(text.strip())
                alpha_ratio = (alpha_count / max(len(text), 1)) if text else 0.0
                non_ascii_ratio = (non_ascii_count / max(len(text), 1)) if text else 0.0
                columns = self._estimate_columns(words, float(page.width))
                pages.append(
                    PageInspection(
                        page_number=idx,
                        width=float(page.width),
                        height=float(page.height),
                        text_length=text_len,
                        embedded_text_quality=quality,
                        has_embedded_text=text_len > 0,
                        embedded_alpha_ratio=alpha_ratio,
                        embedded_non_ascii_ratio=non_ascii_ratio,
                        image_count=len(images),
                        table_count=table_count,
                        detected_columns=columns,
                    )
                )

        return PdfInspection(
            filename=pdf_path.name,
            title=metadata.get("/Title"),
            author=metadata.get("/Author"),
            page_count=len(reader.pages),
            pages=pages,
        )

    def _estimate_columns(self, words: list[dict], page_width: float) -> int:
        if not words:
            return 1
        mid = page_width / 2.0
        left = sum(1 for w in words if float(w.get("x0", 0.0)) < mid)
        right = len(words) - left
        if min(left, right) >= 8 and abs(left - right) / max(len(words), 1) < 0.8:
            return 2
        return 1
