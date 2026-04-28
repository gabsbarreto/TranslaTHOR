# Local PDF Translation App

Local, browser-based PDF translation for scientific documents.

## What The App Does

1. Accepts PDF uploads through a web UI.
2. Renders pages and runs DeepSeek-OCR-2 (MLX).
3. Builds a structured document model (`Document/Page/Block/Table/Figure`).
4. Translates content with local MLX `Qwen3.5-9B-8bit`.
5. Produces `translated.md`, `structured.json`, and on-demand readable/faithful PDFs.
6. Supports retranslation from cached OCR files to skip OCR reruns.
7. Uses a staged UI workflow: uploads/reuse requests wait until `Parse and Translate` is clicked.
8. Saves translation metadata (`model`, `temperature`, `top_p`) and shows it for completed jobs.
9. Names repeated attempts like explorer (`file.pdf`, `file (1).pdf`, `file (2).pdf`).

## Current Scope

- OCR backend: DeepSeek-OCR-2 only.
- Translation backend: MLX `Qwen3.5-9B-8bit`.
- No Marker/Surya/OCRmyPDF fallback paths.

## UI Controls

UI settings apply to the translation LLM:

- `temperature`
- `top_p`
- `max_tokens`
- `chunk_size`

OCR model/runtime settings are fixed in backend defaults for deterministic extraction.

## Project Structure

```text
backend/
  app/
    config.py
    main.py
    models/
    services/
      deepseek_ocr_pipeline.py
      job_queue.py
      job_store.py
      markdown_builder.py
      pdf_inspector.py
      pipeline.py
      profiler.py
      reconstructor.py
      renderer.py
      translation_subprocess.py
      translation_worker.py
      translator_mlx.py
    utils/
frontend/
scripts/
tests/
workspace/
```

## Setup

Prerequisites:

- Python 3.10+
- Apple Silicon for MLX runtime

Install:

```bash
bash scripts/setup_local_runtime.sh
```

Optional extras:

```bash
pip install -e ".[mlx,deepseek_ocr,dev]"
```

If you use `mlx-community/Qwen3.5-9B-8bit`, ensure your environment has `mlx-lm>=0.31.0`:

```bash
.venv/bin/pip install -U "mlx-lm>=0.31.0"
```

## Run

```bash
bash scripts/run_dev.sh
```

Open: `http://127.0.0.1:8000`

## API

- `POST /api/jobs`
- `POST /api/jobs/{job_id}/retranslate`
- `POST /api/jobs/{job_id}/cancel`
- `DELETE /api/jobs/cleanup-terminal`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/artifacts/{pdf|markdown|json}`
- `GET /api/jobs/{job_id}/pdf/{readable|faithful}`

`POST /api/jobs/{job_id}/retranslate` creates a new job using the original `input.pdf` and cached `deepseek_ocr/page_*.md`.
The new job skips OCR inference and runs structure + translation with the current UI LLM settings.

## Artifacts

Per job under `workspace/jobs/<job_id>/artifacts/`:

- `source.md`
- `translated.md`
- `structured.json`
- `timing_profile.*` (when profiling enabled)
- generated PDFs on demand

## Cleanup

```bash
rm -rf workspace/jobs/*
```
