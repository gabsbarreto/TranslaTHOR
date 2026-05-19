# Local PDF Translation App

## 1) What The App Does

This app translates PDFs locally through a browser UI.

Pipeline summary:
1. User uploads a PDF.
2. The backend classifies the document (`digital_good_text`, `scanned_no_text`, `bad_hidden_ocr`, `mixed`, `unknown`).
3. Marker is used as the default extractor/parser with the selected OCR mode.
4. If extraction quality is poor (based on current rules), the Qwen full-page OCR fallback can be used.
5. Extracted content is normalized into structured blocks/chunks and Markdown.
6. Translation runs with the local MLX Qwen model.
7. Outputs are saved as artifacts (`source.md`, `translated.md`, `structured.json`) and readable/faithful PDFs.

## 2) Structure

```text
backend/
  app/
    main.py                  # FastAPI routes and artifact endpoints
    config.py                # Runtime settings/env wiring
    models/                  # Pydantic schema models
    services/
      pipeline.py            # Job pipeline orchestration
      markdown_builder.py    # Structured document -> markdown
      translator_mlx.py      # MLX Qwen translation logic
      translation_worker.py  # Isolated translation subprocess worker
      reconstructor.py       # Markdown -> HTML -> PDF
      pdf_extraction/
        marker_extractor.py
        pdf_type_detector.py
        qwen_ocr_fallback.py
        markdown_builder.py
        local_vlm_service.py
        deepseek_fallback.py
        models.py
frontend/
  index.html
  app.js
  styles.css
scripts/
  setup_local_runtime.sh
  run_dev.sh
  qwen_ocr_worker.py
tests/
workspace/jobs/             # Per-job inputs/artifacts/log state
```

## 3) How To Run

Prerequisites:
- Python 3.10+
- Apple Silicon (for MLX runtime)

Install:

```bash
bash scripts/setup_local_runtime.sh
pip install -e ".[mlx,deepseek_ocr,dev]"
```

Install Marker in its own venv (recommended because of dependency differences):

```bash
python -m venv .venv-marker
.venv-marker/bin/python -m pip install -U pip
.venv-marker/bin/python -m pip install "marker-pdf==1.10.2" "transformers<5" "regex<2025"
```

Run app:

```bash
bash scripts/run_dev.sh
```

Open:
- `http://127.0.0.1:8000`


