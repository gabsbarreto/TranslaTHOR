from __future__ import annotations

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = BASE_DIR / "workspace"
JOBS_DIR = WORKSPACE_DIR / "jobs"
FRONTEND_DIR = BASE_DIR / "frontend"

DEFAULT_DPI = 300
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_TRANSLATION_CHUNK_GROUP_SIZE = int(os.getenv("TRANSLATION_CHUNK_GROUP_SIZE", "5"))
DEFAULT_RENDER_STRATEGY = "pre_render_all"
DEFAULT_OUTPUT_MODE = "readable"
DEFAULT_QWEN_OCR_MODEL = os.getenv("QWEN_OCR_MODEL", "mlx-community/Qwen3.5-2B-8bit")
DEFAULT_QWEN_OCR_MAX_TOKENS = int(os.getenv("QWEN_OCR_MAX_TOKENS", "4096"))
DEFAULT_QWEN_OCR_PROMPT = os.getenv(
    "QWEN_OCR_PROMPT",
    (
        """You are an OCR-to-Markdown transcription engine.

Convert the document image into clean, logical Markdown.

Core task:
- Transcribe the visible document text faithfully.
- Preserve the original reading order.
- Return only Markdown.
- Use [illegible] for unreadable text.

Paragraph rules:
- Output paragraphs as continuous text blocks.
- Join visual line wraps into the same paragraph.
- Start a new paragraph only when the document shows a real paragraph break.
- Reconstruct words split by line-break hyphenation.
  Example: "forma-\nción" → "formación".

Heading rules:
- Merge multi-line headings into one Markdown heading.
- Use a single heading marker for the complete heading.
- Preserve the heading level logically with #, ##, or ###.

Page header/footer rules:
- Detect running page headers and footers near the top or bottom margin. These are usually seen above lines such as 'Title\n ________'
- Keep body headings, article titles, table titles, and figure captions in the main Markdown body.
- When uncertain, treat the text as body content.

Table rules:
- Convert tables into valid Markdown tables.
- Preserve columns, rows, cell text, and order.
- Use [illegible] inside unreadable cells.

Layout rules:
- Read multi-column pages column by column, top to bottom.
- Keep captions close to their figures or tables.
- Preserve symbols, units, punctuation, superscripts, and subscripts as accurately as possible.

Now convert the image into clean logical Markdown."""
    ),
)
DEFAULT_QWEN_OCR_DPI = int(os.getenv("QWEN_OCR_DPI", str(DEFAULT_DPI)))
DEFAULT_QWEN_OCR_BATCH_SIZE = int(os.getenv("QWEN_OCR_BATCH_SIZE", "2"))
DEFAULT_QWEN_OCR_CROP_MODE = os.getenv("QWEN_OCR_CROP_MODE", "true").lower() in {"1", "true", "yes"}
DEFAULT_QWEN_OCR_MIN_CROPS = int(os.getenv("QWEN_OCR_MIN_CROPS", "2"))
DEFAULT_QWEN_OCR_MAX_CROPS = int(os.getenv("QWEN_OCR_MAX_CROPS", "6"))
DEFAULT_QWEN_OCR_BASE_SIZE = int(os.getenv("QWEN_OCR_BASE_SIZE", "1024"))
DEFAULT_QWEN_OCR_IMAGE_SIZE = int(os.getenv("QWEN_OCR_IMAGE_SIZE", "768"))
DEFAULT_QWEN_OCR_SKIP_REPEAT = os.getenv("QWEN_OCR_SKIP_REPEAT", "true").lower() in {"1", "true", "yes"}
DEFAULT_QWEN_OCR_NGRAM_SIZE = int(os.getenv("QWEN_OCR_NGRAM_SIZE", "20"))
DEFAULT_QWEN_OCR_NGRAM_WINDOW = int(os.getenv("QWEN_OCR_NGRAM_WINDOW", "90"))
DEFAULT_USE_PREVIOUS_PAGE_CONTEXT_FOR_HEADER_DETECTION = (
    os.getenv("QWEN_OCR_USE_PREVIOUS_PAGE_CONTEXT_FOR_HEADER_DETECTION", "true").lower()
    in {"1", "true", "yes"}
)
DEFAULT_PREVIOUS_CONTEXT_PARAGRAPHS = int(os.getenv("QWEN_OCR_PREVIOUS_CONTEXT_PARAGRAPHS", "2"))
DEFAULT_MAX_PREVIOUS_CONTEXT_CHARS = int(os.getenv("QWEN_OCR_MAX_PREVIOUS_CONTEXT_CHARS", "1600"))
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.7"))
DEFAULT_LLM_TOP_K = int(os.getenv("LLM_TOP_K", "10"))
DEFAULT_LLM_MIN_P = float(os.getenv("LLM_MIN_P", "0.0"))
DEFAULT_LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "1.5"))
DEFAULT_LLM_REPETITION_PENALTY = float(os.getenv("LLM_REPETITION_PENALTY", "1.0"))
DEFAULT_QWEN_OCR_TEMPERATURE = float(
    os.getenv("QWEN_OCR_TEMPERATURE", "0.1")
)
DEFAULT_QWEN_OCR_TOP_P = float(os.getenv("QWEN_OCR_TOP_P", "0.8"))
DEFAULT_QWEN_OCR_TOP_K = int(os.getenv("QWEN_OCR_TOP_K", "20"))
DEFAULT_QWEN_OCR_MIN_P = float(os.getenv("QWEN_OCR_MIN_P", str(DEFAULT_LLM_MIN_P)))
DEFAULT_QWEN_OCR_PRESENCE_PENALTY = float(
    os.getenv("QWEN_OCR_PRESENCE_PENALTY", "1.0")
)
DEFAULT_QWEN_OCR_REPETITION_PENALTY = float(
    os.getenv("QWEN_OCR_REPETITION_PENALTY", "1.0")
)
DEFAULT_TRANSLATION_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
AVAILABLE_TRANSLATION_MODELS = [
    DEFAULT_TRANSLATION_MODEL,
]

ENABLE_MARKER_PIPELINE = os.getenv("ENABLE_MARKER_PIPELINE", "true").lower() in {"1", "true", "yes"}
ENABLE_LEGACY_VISUAL_OCR = os.getenv("ENABLE_LEGACY_VISUAL_OCR", "false").lower() in {"1", "true", "yes"}
ENABLE_LOCAL_VLM_REPAIR = os.getenv("ENABLE_LOCAL_VLM_REPAIR", "false").lower() in {"1", "true", "yes"}
ENABLE_QWEN_OCR_FALLBACK = os.getenv("ENABLE_QWEN_OCR_FALLBACK", "true").lower() in {"1", "true", "yes"}
KEEP_EXTRACTION_DEBUG_ARTIFACTS = os.getenv("KEEP_EXTRACTION_DEBUG_ARTIFACTS", "false").lower() in {"1", "true", "yes"}
DEFAULT_EXTRACTION_MODE = os.getenv("DEFAULT_EXTRACTION_MODE", "auto")
MARKER_TIMEOUT_SECONDS = int(os.getenv("MARKER_TIMEOUT_SECONDS", "1800"))

for directory in [WORKSPACE_DIR, JOBS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
