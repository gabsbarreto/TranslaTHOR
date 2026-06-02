from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.models.schema import DocumentModel


PDFClassification = Literal[
    "digital_good_text",
    "scanned_no_text",
    "bad_hidden_ocr",
    "mixed",
    "unknown",
]

ExtractionMode = Literal[
    "auto",
    "digital",
    "scanned",
    "strip_and_force_ocr",
    "auto_repair",
]

MarkerMode = Literal["normal", "text_only", "force_ocr", "strip_existing_ocr_force_ocr"]


@dataclass
class PageTextStats:
    page_number: int
    width: float
    height: float
    char_count: int
    word_count: int
    alnum_ratio: float
    alpha_ratio: float
    non_ascii_ratio: float
    replacement_char_count: int
    weird_symbol_ratio: float
    average_word_length: float
    repeated_garbage_score: float
    image_count: int
    image_area_ratio: float
    has_selectable_text: bool
    looks_meaningful: bool
    looks_garbled: bool
    looks_image_dominant: bool


@dataclass
class PDFTypeDetectionResult:
    classification: PDFClassification
    page_count: int
    pages: list[PageTextStats]
    embedded_text_chars: int
    embedded_text_words: int
    meaningful_page_count: int
    garbled_page_count: int
    image_dominant_page_count: int
    scanned_page_count: int
    mixed: bool
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionChunk:
    chunk_id: str
    page_number: int
    block_ids: list[str]
    block_type: str
    bbox: dict[str, float] | None
    polygon: list[list[float]] | None
    original_text: str
    translated_text: str = ""


@dataclass
class PDFExtractionResult:
    markdown: str
    chunks: list[ExtractionChunk]
    pages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    metadata: dict[str, Any]
    extraction_mode: str
    pdf_classification: str
    used_ocr: bool
    used_force_ocr: bool
    stripped_existing_ocr: bool
    used_local_vlm_repair: bool
    warnings: list[str] = field(default_factory=list)
    document: DocumentModel | None = None
