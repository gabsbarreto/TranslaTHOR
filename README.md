# Local PDF Translation App

## What It Does

This browser-based app translates PDFs locally.

1. The user uploads one or more PDFs.
2. The backend classifies each document as `digital_good_text`, `scanned_no_text`, `bad_hidden_ocr`, `mixed`, or `unknown`.
3. Marker extracts documents with usable embedded text.
4. Surya detects layout boxes for bad scans and Qwen OCR reads the full annotated page images.
5. Qwen full-page OCR reads rendered page images directly when Marker needs an uncertain-document OCR fallback.
6. Surya OCR regions are cleaned and merged into traceable logical translation chunks.
7. Extracted text is normalized into a shared structured document model.
8. A local MLX Qwen model translates the document into English.
9. Markdown, JSON, and readable or faithful PDF artifacts are available for download.

The OCR path preserves the text emitted by Qwen. It does not remove running headers, footers, page numbers, or margin metadata.

## Structure

```text
backend/
  app/
    main.py                         # FastAPI upload, queue, and artifact routes
    config.py                       # Runtime settings and environment variables
    models/schema.py                # Shared structured document model
    services/
      pipeline.py                   # Extraction and translation orchestration
      translator_mlx.py             # MLX Qwen translation logic
      translation_worker.py         # Isolated translation process
      ocr_to_translation_parser.py  # Surya OCR regions -> logical translation chunks
      qwen_markdown_parser.py        # Qwen Markdown -> structured document
      reconstructor.py              # Markdown -> HTML -> PDF
      pdf_extraction/
        pdf_type_detector.py         # Embedded-text quality classification
        marker_extractor.py          # Marker integration
        markdown_builder.py          # Marker output -> structured document
        qwen_ocr_fallback.py         # Full-page Qwen OCR integration
scripts/
  run_dev.sh
  setup_local_runtime.sh
  qwen_ocr_worker.py
  surya_layout_worker.py
frontend/
  index.html
  app.js
  styles.css
tests/
workspace/jobs/                     # Per-job inputs and generated artifacts
```

## Requirements

- Python 3.10+
- Apple Silicon for the MLX runtime
- Homebrew packages installed by `scripts/setup_local_runtime.sh`

## Install

```bash
bash scripts/setup_local_runtime.sh
```

For an editable development install:

```bash
pip install -e ".[mlx,qwen_ocr,dev]"
```

Install Marker in its own virtual environment:

```bash
python -m venv .venv-marker
.venv-marker/bin/python -m pip install -U pip
.venv-marker/bin/python -m pip install "marker-pdf==1.10.2" "transformers<5" "regex<2025"
```

Inspect Surya layout boxes for rendered OCR pages:

```bash
.venv-marker/bin/python scripts/surya_layout_worker.py \
  --input-dir workspace/jobs/<job-id>/qwen_ocr/rendered_pages \
  --output-dir workspace/jobs/<job-id>/qwen_ocr/surya_layout
```

This writes `layout.json`, padded region crops, annotated page previews, and full-page
`boxed_pages` overlays. The app generates those overlays automatically for `scanned_no_text`
and `bad_hidden_ocr` PDFs, then sends each complete overlay page to Qwen. Run the worker with
`.venv-marker` because Surya is installed with Marker and requires `transformers<5`.

## Run

```bash
bash scripts/run_dev.sh
```

Open `http://127.0.0.1:8000`.

## Optional Environment Variables

- `QWEN_OCR_PYTHON`: Python executable containing `mlx-vlm`.
- `QWEN_OCR_MODEL`: Qwen OCR model identifier.
- `QWEN_OCR_PROMPT`: OCR transcription prompt.
- `SURYA_LAYOUT_PYTHON`: Python executable containing Surya; defaults to `.venv-marker/bin/python`.
- `MARKER_BIN`: Marker executable path.
- `ENABLE_QWEN_OCR_FALLBACK`: Enable full-page Qwen OCR fallback.
- `ENABLE_LOCAL_VLM_REPAIR`: Enable Marker block repair through the optional local VLM endpoint.
