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
8. Figure regions are deduplicated, validated, associated with captions, and captured as
   high-resolution PNG previews plus clipped vector SVG assets when the source supports them.
9. A local MLX Qwen model translates the document into English.
10. Markdown, JSON, readable/faithful PDFs, and a conservative original-layout PDF are available
    for download after translation completes.

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
      figure_extractor.py            # Figure detection, coordinate conversion, and asset capture
      original_layout_reconstructor.py # Source-page text replacement and reconstruction report
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

## PDF Output Modes

- **Readable PDF** reflows translated text for comfortable reading and places each detected figure
  in reading order using its clipped vector SVG when available, or its high-resolution PNG preview
  as a recorded fallback. The translated external caption is kept with the figure and is not
  duplicated elsewhere.
- **Faithful PDF** keeps the existing compact, two-column reconstructed output.
- **Original layout PDF** starts from a separate output copy of the uploaded PDF. On reliable
  digital-text pages it removes source text while preserving overlapping raster images and vector
  graphics, then inserts translated text into the corresponding boxes using source-PDF font metrics
  when available. Translation batches are kept page-local and do not cross figure or equation
  regions. Reliable vector-grid tables are reconstructed cell-by-cell without removing their lines.
  Figures, graphs, equations, logos, colours, lines, page sizes, crop boxes, rotation, and
  decorations come from the original pages. A JSON reconstruction report records every replacement,
  skip, fallback page, text scale, overflow, raster figure fallback, and low-confidence association.

In this first figure-handling level, a graph or figure is preserved as one unchanged visual. Only
its external caption is translated. Axis labels, legends, abbreviations, annotations, and all other
text inside the captured figure remain in the original language; graphs are not reconstructed from
data.

Translated PDF routes are enabled only after the job reaches `complete`:

```text
/api/jobs/<job-id>/pdf/readable
/api/jobs/<job-id>/pdf/faithful
/api/jobs/<job-id>/pdf/original-layout
/api/jobs/<job-id>/artifacts/reconstruction_report
```

The reconstruction report becomes available after the original-layout PDF is generated.

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

After a job finishes, use **Readable PDF** to test the reflowed output and **Original layout PDF**
to test source-page reconstruction. If the original-layout warning is shown, download the
**Reconstruction Report** and use the readable PDF as the safe translated fallback.

## Reconstruction Limitations

- Scanned and image-only pages are retained unchanged. This version does not inpaint visible source
  text and will never place translated text over an unmodified scan.
- Hidden-OCR and mixed/OCR pages are handled conservatively and may remain unchanged when the
  visible source text cannot be proven removable.
- Rotated pages are retained unchanged by the first original-layout implementation, while their
  original dimensions, crop boxes, rotation, and visual content remain intact.
- Missing, invalid, or low-confidence bounding boxes are skipped and reported rather than guessed.
  Legacy cross-page translation batches are recovered only when their preserved source and
  translated paragraph boundaries prove a one-to-one mapping; ambiguous legacy batches remain
  unchanged.
- Digital tables are translated cell-by-cell only when the translated HTML structure can be matched
  to a complete vector cell grid and the source cell text validates that mapping. Malformed,
  duplicated, merged-cell, image-based, or otherwise ambiguous tables remain unchanged and are
  reported. Tables remain translated in the readable PDF.
- Translated text that cannot fit at the minimum 60% scale is reported and the source region is
  retained instead of silently deleting text or shrinking it to an unreadable size.
- Figure detection follows Marker/Surya structured regions. False-positive visual regions or missed
  figures can still occur when upstream layout metadata is unreliable; the JSON report and
  structured document retain the evidence used for each association.

## Optional Environment Variables

- `QWEN_OCR_PYTHON`: Python executable containing `mlx-vlm`.
- `QWEN_OCR_MODEL`: Qwen OCR model identifier.
- `QWEN_OCR_PROMPT`: OCR transcription prompt.
- `SURYA_LAYOUT_PYTHON`: Python executable containing Surya; defaults to `.venv-marker/bin/python`.
- `MARKER_BIN`: Marker executable path.
- `ENABLE_QWEN_OCR_FALLBACK`: Enable full-page Qwen OCR fallback.
- `ENABLE_LOCAL_VLM_REPAIR`: Enable Marker block repair through the optional local VLM endpoint.
