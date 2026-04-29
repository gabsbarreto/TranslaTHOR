from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CoordinateSpace(str, Enum):
    PDF = "pdf"
    NORMALIZED = "normalized"


class RegionType(str, Enum):
    PAGE = "page"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    REFERENCE = "reference"
    OTHER = "other"


class RegionSource(str, Enum):
    DEFAULT_FULL_PAGE = "default_full_page"
    PYMUPDF = "pymupdf"
    SURYA = "surya"
    MANUAL = "manual"


class Region(BaseModel):
    id: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED
    type: RegionType = RegionType.TEXT
    selected: bool = True
    reading_order: int = 0
    source: RegionSource = RegionSource.MANUAL


class PageRegionPayload(BaseModel):
    pdf_file_id: str
    page_number: int
    page_width: float
    page_height: float
    image_width: int
    image_height: int
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED
    detector: str = "unknown"
    regions: list[Region] = Field(default_factory=list)


class OcrRegionResult(BaseModel):
    pdf_file_id: str
    page_number: int
    box_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED
    box_type: RegionType = RegionType.TEXT
    reading_order: int
    ocr_text: str
    ocr_confidence: float | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class OcrResultsPayload(BaseModel):
    pdf_file_id: str
    results: list[OcrRegionResult] = Field(default_factory=list)
