from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    EMBEDDED = "embedded_text"
    OCR = "ocr"
    HYBRID = "hybrid"
    FALLBACK_IMAGE = "fallback_image"


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"
    HEADER = "header"
    FOOTER = "footer"
    EQUATION = "equation"
    PAGE_NUMBER = "page_number"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class Block(BaseModel):
    id: str
    page_number: int
    block_type: BlockType
    text: str = ""
    bbox: BoundingBox | None = None
    confidence: float | None = None
    reading_order_index: int
    style_hints: dict[str, Any] = Field(default_factory=dict)
    source_type: SourceType
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageMetadata(BaseModel):
    page_number: int
    width: float
    height: float
    has_embedded_text: bool
    embedded_text_quality: float
    extraction_mode: SourceType


class FigureAsset(BaseModel):
    id: str
    page_number: int
    bbox: BoundingBox | None = None
    caption_block_id: str | None = None
    image_path: str | None = None


class TableModel(BaseModel):
    class TableCell(BaseModel):
        text: str = ""
        rowspan: int = 1
        colspan: int = 1
        bbox: BoundingBox | None = None
        confidence: float | None = None

    id: str
    page_numbers: list[int]
    page: int | None = None
    bbox: BoundingBox | None = None
    caption_block_id: str | None = None
    caption: str | None = None
    notes: str | None = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[list[TableCell]] = Field(default_factory=list)
    continued_from_previous_page: bool = False
    parse_mode: str = "table_structured"
    fallback_image_path: str | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class TranslationChunk(BaseModel):
    id: str
    block_ids: list[str]
    source_text: str
    translated_text: str = ""
    context: str = ""
    source_language: str | None = None
    source_token_count: int | None = None


class DocumentMetadata(BaseModel):
    filename: str
    title: str | None = None
    author: str | None = None
    page_count: int
    detected_language: str | None = None
    translation: dict[str, Any] = Field(default_factory=dict)


class DocumentModel(BaseModel):
    metadata: DocumentMetadata
    pages: list[PageMetadata]
    blocks: list[Block]
    tables: list[TableModel] = Field(default_factory=list)
    figures: list[FigureAsset] = Field(default_factory=list)
    translation_chunks: list[TranslationChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class JobStage(str, Enum):
    UPLOADED = "upload"
    EXTRACTION = "extraction"
    OCR_LAYOUT = "ocr_layout_parsing"
    STRUCTURE = "structure_generation"
    TRANSLATION = "translation"
    PDF = "pdf_generation"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


class JobStatus(BaseModel):
    job_id: str
    filename: str
    source_filename: str | None = None
    attempt: int = 0
    stage: JobStage
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    translation: dict[str, Any] = Field(default_factory=dict)
