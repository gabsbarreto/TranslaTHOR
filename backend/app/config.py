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
DEFAULT_OUTPUT_MODE = "readable"
DEFAULT_QWEN_OCR_MODEL = os.getenv("QWEN_OCR_MODEL", "mlx-community/Qwen3.5-4B-4bit")
DEFAULT_QWEN_OCR_MAX_TOKENS = int(os.getenv("QWEN_OCR_MAX_TOKENS", "4096"))
DEFAULT_QWEN_OCR_PROMPT = os.getenv(
    "QWEN_OCR_PROMPT",
    (
        """You are an OCR-to-Markdown transcription engine.

Convert the document image into clean Markdown.

Rules:
- Transcribe only visible text from the current image.
- Preserve visual reading order: left column top-to-bottom, then right column.
- Join wrapped lines into paragraphs.
- Reconstruct hyphenated line-breaks, e.g. "forma-\\nción" → "formación".
- Convert tables into valid Markdown tables.
- Keep captions near their figures/tables.
- If a page ends in a hyphenated line break, do not try to guess and finish the word. It should continue on the next page.
- Return only Markdown."""
    ),
)
DEFAULT_QWEN_OCR_DPI = int(os.getenv("QWEN_OCR_DPI", str(DEFAULT_DPI)))
DEFAULT_QWEN_OCR_BATCH_SIZE = int(os.getenv("QWEN_OCR_BATCH_SIZE", "1"))
DEFAULT_QWEN_OCR_CROP_MODE = os.getenv("QWEN_OCR_CROP_MODE", "true").lower() in {"1", "true", "yes"}
DEFAULT_QWEN_OCR_MIN_CROPS = int(os.getenv("QWEN_OCR_MIN_CROPS", "2"))
DEFAULT_QWEN_OCR_MAX_CROPS = int(os.getenv("QWEN_OCR_MAX_CROPS", "6"))
DEFAULT_QWEN_OCR_BASE_SIZE = int(os.getenv("QWEN_OCR_BASE_SIZE", "1024"))
DEFAULT_QWEN_OCR_IMAGE_SIZE = int(os.getenv("QWEN_OCR_IMAGE_SIZE", "768"))
DEFAULT_QWEN_OCR_SKIP_REPEAT = os.getenv("QWEN_OCR_SKIP_REPEAT", "true").lower() in {"1", "true", "yes"}
DEFAULT_QWEN_OCR_NGRAM_SIZE = int(os.getenv("QWEN_OCR_NGRAM_SIZE", "20"))
DEFAULT_QWEN_OCR_NGRAM_WINDOW = int(os.getenv("QWEN_OCR_NGRAM_WINDOW", "90"))
DEFAULT_LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.7"))
DEFAULT_LLM_TOP_K = int(os.getenv("LLM_TOP_K", "10"))
DEFAULT_LLM_MIN_P = float(os.getenv("LLM_MIN_P", "0.0"))
DEFAULT_LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "1.5"))
DEFAULT_LLM_REPETITION_PENALTY = float(os.getenv("LLM_REPETITION_PENALTY", "1.0"))
DEFAULT_QWEN_OCR_TEMPERATURE = float(
    os.getenv("QWEN_OCR_TEMPERATURE", "0.0")
)
DEFAULT_QWEN_OCR_TOP_P = float(os.getenv("QWEN_OCR_TOP_P", "0.8"))
DEFAULT_QWEN_OCR_TOP_K = int(os.getenv("QWEN_OCR_TOP_K", "20"))
DEFAULT_QWEN_OCR_MIN_P = float(os.getenv("QWEN_OCR_MIN_P", str(DEFAULT_LLM_MIN_P)))
DEFAULT_QWEN_OCR_PRESENCE_PENALTY = float(
    os.getenv("QWEN_OCR_PRESENCE_PENALTY", "0")
)
DEFAULT_QWEN_OCR_REPETITION_PENALTY = float(
    os.getenv("QWEN_OCR_REPETITION_PENALTY", "1.0")
)
DEFAULT_TRANSLATION_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
AVAILABLE_TRANSLATION_MODELS = [
    DEFAULT_TRANSLATION_MODEL,
]

ENABLE_LOCAL_VLM_REPAIR = os.getenv("ENABLE_LOCAL_VLM_REPAIR", "false").lower() in {"1", "true", "yes"}
ENABLE_QWEN_OCR_FALLBACK = os.getenv("ENABLE_QWEN_OCR_FALLBACK", "true").lower() in {"1", "true", "yes"}
KEEP_EXTRACTION_DEBUG_ARTIFACTS = os.getenv("KEEP_EXTRACTION_DEBUG_ARTIFACTS", "false").lower() in {"1", "true", "yes"}
DEFAULT_EXTRACTION_MODE = os.getenv("DEFAULT_EXTRACTION_MODE", "auto")
MARKER_TIMEOUT_SECONDS = int(os.getenv("MARKER_TIMEOUT_SECONDS", "1800"))

for directory in [WORKSPACE_DIR, JOBS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
