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
        """You are an OCR and document-to-Markdown transcription engine.

Your task is to convert the provided document image into clean, logical Markdown.

Rules:
1. Transcribe only the visible text in the image.
2. Do not summarise, explain, translate, or add information.
3. Do not invent missing text. If text is unreadable, write: [illegible].
4. Preserve the original reading order.
5. Preserve headings, subheadings, paragraphs, lists, tables, footnotes, captions, and page numbers when visible.
6. Use Markdown formatting:
   - Use #, ##, ### for clear headings.
   - Use bullet lists or numbered lists when the document uses lists.
   - Use Markdown tables for tabular content.
   - Use **bold** and *italic* only when clearly visible in the document.
7. Do not wrap the output in a code block.
8. Return only the Markdown content.

Paragraph and line-break rules:
- Output logical paragraphs, not visual lines.
- Do not reproduce line breaks that exist only because the text was wrapped on the page.
- Join wrapped lines from the same paragraph into one continuous Markdown paragraph.
- Only insert a blank line when there is a real paragraph break, section break, heading, list, table, caption, or other distinct document element.
- If a sentence continues on the next visual line, keep it in the same Markdown paragraph.
- If a word is split at the end of a line using a hyphen, reconstruct the full word and remove the line-break hyphen unless the word is genuinely hyphenated.
- Do not split a paragraph after every sentence unless the original document clearly uses separate paragraphs.

Heading rules:
- If a title or heading is visually broken across multiple lines, merge the full heading into one Markdown heading.
- Use only one Markdown heading marker for the full heading.
- Do not create multiple headings just because the printed heading spans multiple visual lines.

Page header and footer rules:
- Identify possible page headers and footers separately from the main body text.
- A page header is text located near the top margin that appears visually separated from the main body, such as journal names, article titles, author names, running titles, volume/issue information, or DOI information.
- A page footer is text located near the bottom margin, such as page numbers, journal information, copyright notices, or repeated publication details.
- Do not mix page headers or footers into the main paragraph text.
- If the article title, abbreviated title, journal name, author name, or running title repeats at the top of later pages, treat it as a page header, not as a document title or section heading.
- Never output a repeated running page header as #, ##, or ###.
- If a top or bottom line appears to be a repeated running header/footer, place it inside a Markdown comment:
  <!-- page-header: text here -->
  <!-- page-footer: text here -->
- Preserve visible page numbers separately:
  <!-- page-number: 123 -->
- If the text is clearly a section heading, article title, abstract heading, introduction heading, or body heading, keep it in the main Markdown body.
- If uncertain whether a line is a header/footer or body content, keep it in the main body and do not mark it as a header/footer.

Table rules:
- If a table is present, convert it to a valid Markdown table.
- Preserve column names and row order.
- If a cell is empty, leave it empty.
- If a cell is unreadable, write [illegible].
- Do not merge rows or columns unless the visual table clearly does so.
- Preserve line breaks inside table cells only when they indicate separate items within the same cell.

Layout rules:
- For multi-column pages, read each column from top to bottom before moving to the next column.
- Keep captions close to their figures or tables.
- Ignore decorative lines, borders, and logos unless they contain text.
- Preserve mathematical symbols, units, punctuation, superscripts, and subscripts as accurately as possible.
- Preserve deliberate line-based structures such as lists, forms, addresses, poetry, references, and tables.

Now convert the image to clean logical Markdown."""
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
