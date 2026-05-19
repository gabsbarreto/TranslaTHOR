# Local PDF Translation App

Local, browser-based PDF translation for scientific documents.

## What The App Does

1. Accepts PDF uploads through a web UI.
2. Detects PDF text quality and runs Marker automatically.
3. Builds a structured document model (`Document/Page/Block/Table/Figure`).
4. Translates content with local MLX Qwen.
5. Produces `translated.md`, `structured.json`, and on-demand readable/faithful PDFs.
6. Saves extraction metadata (`pdf_classification`, Marker mode, OCR flags) and translation metadata.
7. Names repeated attempts like explorer (`file.pdf`, `file (1).pdf`, `file (2).pdf`).

## Current Scope

- Extraction backend: Marker with Surya OCR.
- Translation backend: local MLX Qwen.
- DeepSeek OCR-2 is available only as an optional future fallback hook; it is not a Marker OCR backend.
- The legacy visual/manual OCR route is preserved in code but hidden from the main UI.

## UI Controls

UI settings apply to extraction and translation:

- `extraction_mode`: `auto`, `digital`, `scanned`, `strip_and_force_ocr`, `auto_repair`, `deepseek_fallback`
- `use_local_vlm_repair`
- `use_deepseek_fallback`
- `keep_debug_artifacts`
- `temperature`
- `top_p`
- `max_tokens`
- `chunk_size`

Default extraction mode is `auto`: embedded text is used for good digital PDFs, Marker force OCR is used for scanned PDFs, and bad hidden OCR text uses Marker `--strip_existing_ocr --force_ocr`.

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
      pdf_extraction/
        marker_extractor.py
        pdf_type_detector.py
        local_vlm_service.py
        deepseek_fallback.py
        markdown_builder.py
        models.py
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

Marker/Surya currently needs Transformers 4.x, while the app's MLX translator needs Transformers 5.x. Keep Marker in a separate runtime:

```bash
python -m venv .venv-marker
.venv-marker/bin/python -m pip install -U pip
.venv-marker/bin/python -m pip install "marker-pdf==1.10.2" "transformers<5" "regex<2025"
```

The app automatically uses `.venv-marker/bin/marker_single` when it exists. You can override this with `MARKER_BIN`.

If you use `mlx-community/Qwen3.5-4B-OptiQ-4bit`, ensure your environment has `mlx-lm>=0.31.0`:

```bash
.venv/bin/pip install -U "mlx-lm>=0.31.0"
```

## Run

```bash
bash scripts/run_dev.sh
```

Open: `http://127.0.0.1:8000`

## Marker Pipeline

The automatic upload path calls:

```python
from pathlib import Path
from app.services.pdf_extraction import PDFExtractor

result = PDFExtractor().extract(Path("paper.pdf"), mode="auto")
print(result.pdf_classification, result.metadata["marker_mode"])
print(result.markdown[:500])
```

Marker is invoked through `marker_single` without `shell=True`. Override the binary with:

```bash
export MARKER_BIN=marker_single
```

Useful environment variables:

```bash
ENABLE_MARKER_PIPELINE=true
DEFAULT_EXTRACTION_MODE=auto
KEEP_EXTRACTION_DEBUG_ARTIFACTS=false
MARKER_TIMEOUT_SECONDS=1800
ENABLE_LOCAL_VLM_REPAIR=false
ENABLE_DEEPSEEK_FALLBACK=false
ENABLE_LEGACY_VISUAL_OCR=false
```

Local VLM repair is optional and selective. It calls an OpenAI-compatible endpoint only for blocks that look problematic:

```bash
export LOCAL_VLM_ENABLED=true
export LOCAL_VLM_BASE_URL=http://localhost:8080/v1
export LOCAL_VLM_MODEL=your-mlx-vlm-model
export LOCAL_VLM_API_KEY=not-needed
export LOCAL_VLM_TIMEOUT=120
export LOCAL_VLM_MAX_RETRIES=2
```

Marker itself supports OpenAI-compatible LLM services, but this app does not enable Marker `--use_llm` by default because that would apply LLM repair broadly. The app-level repair hook keeps local VLM use selective.

DeepSeek OCR-2 is not forced into Marker. Marker/Surya remains the default OCR path; the DeepSeek fallback currently records a warning and leaves Marker output unchanged.

## API

- `POST /api/jobs`
- `POST /api/jobs/{job_id}/retranslate`
- `POST /api/jobs/{job_id}/cancel`
- `DELETE /api/jobs/cleanup-terminal`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/artifacts/{pdf|markdown|json}`
- `GET /api/jobs/{job_id}/artifacts/{source_markdown|extraction_result|marker_detection}`
- `GET /api/jobs/{job_id}/pdf/{readable|faithful}`

Legacy manual OCR endpoints remain in `backend/app/main.py` for debug/developer use. Re-enable the visual OCR UI with `ENABLE_LEGACY_VISUAL_OCR=true`.

## Artifacts

Per job under `workspace/jobs/<job_id>/artifacts/`:

- `source.md`
- `translated.md`
- `structured.json`
- `extraction_result.json`
- `marker_detection.json`
- `timing_profile.*` (when profiling enabled)
- generated PDFs on demand

## Cleanup

```bash
rm -rf workspace/jobs/*
```
