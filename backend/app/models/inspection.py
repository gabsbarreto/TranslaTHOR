from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageInspection:
    page_number: int
    width: float
    height: float
    text_length: int
    embedded_text_quality: float
    has_embedded_text: bool
    embedded_alpha_ratio: float = 0.0
    embedded_non_ascii_ratio: float = 0.0
    image_count: int = 0
    table_count: int = 0
    detected_columns: int = 1


@dataclass
class PdfInspection:
    filename: str
    title: str | None
    author: str | None
    page_count: int
    pages: list[PageInspection]
