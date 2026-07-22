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
8. Marker table-cell polygons are retained for reconstruction. When older digital jobs have no
   stored cells, partial-rule and whitespace-separated tables can recover their logical grid from
   PDF text geometry only when every source cell has one unambiguous, monotonic alignment.
   Structurally collapsed ruled tables are also checked against the source PDF and repaired from an
   exact clipped PyMuPDF grid only when the source text strongly agrees.
9. Figure regions are deduplicated, validated, associated with captions, and captured as
   high-resolution PNG previews plus clipped vector SVG assets when the source supports them.
10. A local MLX Qwen model translates the document into English.
11. The browser presents the readable PDF and conservative original-layout PDF as the two primary
    results after translation completes. Structured Markdown, JSON, faithful reconstruction, and
    diagnostic artifacts remain available to the backend and existing API routes without
    cluttering the normal interface.

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
      table_markup.py                # Shared HTML/Markdown table-shape parser
      figure_extractor.py            # Figure detection, coordinate conversion, and asset capture
      original_layout_reconstructor.py # Source-page text replacement and reconstruction report
      reconstructor.py              # Markdown -> HTML -> PDF
      pdf_extraction/
        pdf_type_detector.py         # Embedded-text quality classification
        marker_extractor.py          # Marker integration
        markdown_builder.py          # Marker output -> structured document
        table_repair.py              # Validated PyMuPDF repair for collapsed Marker tables
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
  duplicated elsewhere. Structured tables use the translated table block as their canonical copy,
  retain empty cells and column shape, and are kept with one translated caption in a single
  `<figure>` where the available page space permits.
- **Faithful PDF** keeps the existing compact, two-column reconstructed output. It remains API
  compatible but is not shown as a primary browser download.
- **Original layout PDF** starts from a separate output copy of the uploaded PDF. On reliable
  digital-text pages it removes source text while preserving overlapping raster images and vector
  graphics, then inserts translated text into the corresponding boxes using source-PDF font metrics
  when available. Translation batches are kept page-local and do not cross figure or equation
  regions. Tables with validated Marker cell polygons are reconstructed cell-by-cell without
  removing their lines. For older digital jobs, a partial-rule or whitespace-separated table can
  also be reconstructed when source text uniquely validates a coarsening of the PDF text lattice.
  A hidden-OCR scan can reconstruct verified body-text regions on top of its original page image.
  Surya supplies the structure and initial position, then source text is aligned to spatially
  contiguous hidden-OCR lines to correct reading-order drift and malformed native PDF text blocks
  before any pixels are covered. Multi-block passages are translated with shared context while an
  indexed boundary contract keeps one target per physical region. Only matched glyph strips on a
  light, uniform background are masked; ambiguous, partial, genuinely multi-column, or visually
  complex matches remain unchanged. Ruled tables can also be reconstructed when their translated
  shape and complete PDF grid agree, while rules, arrows, numbers, and unchanged cells remain on the
  source page. The table operation is atomic, so one unsafe or overflowing cell retains the entire
  source table.
  Figures, graphs, equations, logos, colours, lines, page sizes, crop boxes, rotation, and
  decorations come from the original pages. A JSON reconstruction report records every replacement,
  skip, fallback page, text scale, overflow, raster figure fallback, low-confidence association, and
  the score/coverage diagnostics for failed hidden-OCR alignment.

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

## Browser Workflow

The normal interface requires no model or extraction configuration. Drop or select one or more
PDFs and the server applies its configured automatic extraction and local translation defaults.

- **Current activity** shows the stage, progress, filename, short job ID, and number of documents
  waiting.
- **Waiting** shows the FIFO queue and the number of jobs ahead of each document.
- **Recent results** shows completed, failed, and cancelled work newest first. Completed records
  expose only **Readable PDF** and **Original layout PDF** as primary actions.
- **View details** contains warning, reconstruction, runtime configuration, and permanent-delete
  controls. Routine warning text is collapsed so it does not obscure the job status.
- **Exclude** archives a terminal record from the default results list without deleting its input
  or generated artifacts. **Show excluded** makes archived records available for restoration.
- Waiting work can be removed from the queue, while active work has an explicit stop action.
  Permanent deletion is separate from cancellation and exclusion.

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
to test source-page reconstruction. If original-layout reconstruction is partial, open
**View details** to inspect its warning and reconstruction report, and use the readable PDF as the
safe translated fallback.

## PDF Regression Corpus

The local regression corpus contains five compact non-English digital cases and five scan cases in
German, Spanish, French, and Brazilian Portuguese. It covers multi-column prose, tables, charts, a
flowchart, captions, accented text, and noisy text mappings. Two cases are genuine scans retaining
their imperfect hidden OCR (`bad_hidden_ocr`); three are raster-only derivatives without a text
layer (`scanned_no_text`) and provide known visual baselines for geometry and pixel comparisons.

The source publications and generated PDFs are kept under the ignored local `workspace/` tree
rather than redistributed through this public repository. The tracked corpus specification records
their source hashes and original page numbers, and the builder refuses a source whose contents have
changed.

```bash
PYTHONPATH=backend .venv/bin/python scripts/build_pdf_regression_corpus.py
PYTHONPATH=backend .venv/bin/pytest -q tests/test_pdf_regression_corpus.py
```

See `tests/regression_corpus/README.md` for provenance, storage, and source-path overrides. The
artifact validation test skips on machines where the private corpus has not been built; the tracked
ten-case specification is always tested.

## Reconstruction Limitations

- Scanned and image-only pages without a reliable hidden text layer are retained unchanged. This
  version does not perform general background inpainting and will never place translated text over
  visible source text that has not first been safely covered.
- Hidden-OCR pages can replace body text when the complete source passage has one unambiguous match
  to a spatially contiguous hidden-OCR line lane. The stored Surya box is an auditable position hint
  rather than the final authority, because a missed or merged region can shift later reading-order
  associations. Line lanes may bridge a malformed native PDF text-block boundary, but they cannot
  jump between columns. Multi-column classification requires a stable gutter across several lines;
  isolated OCR glyph fragments are ignored.
  The original scan remains the background; only verified line masks on light, sufficiently uniform
  paper are covered before translated text is inserted. Partial matches, multi-column regions that
  require table geometry, already-English passages, and suspicious translation-script changes are
  retained and reported. A page with any retained translatable region is reported as partial, and
  the readable PDF remains the safe full-translation fallback.
- Image-only scans without hidden OCR still require a future pixel-level text detector and
  inpainting stage. Surya boxes alone are not treated as sufficient evidence for destructive
  masking, because they describe layout regions rather than exact character strokes.
- Rotated pages are retained unchanged by the first original-layout implementation, while their
  original dimensions, crop boxes, rotation, and visual content remain intact.
- Missing, invalid, or low-confidence bounding boxes are skipped and reported rather than guessed.
  New translations preserve one independently placeable target per source block, even when several
  blocks share one model request for linguistic context; long groups are batched without translating
  a physical block twice.
  Legacy cross-page translation batches are recovered only when their preserved source and
  translated paragraph boundaries prove a one-to-one mapping; ambiguous legacy batches remain
  unchanged.
- Tables are translated cell-by-cell in original-layout output only when the translated HTML or
  Markdown shape can be matched to validated cell geometry and the source cell text validates that
  mapping. New extractions retain Marker cell polygons as the preferred geometry. Older clean
  digital PDFs with partial horizontal rules or whitespace-separated columns can use a semantic
  fallback: physical PDF text lines and candidate column edges are coalesced into the logical table
  only when all source cells align in a unique monotonic solution.
  For scanned tables, the hidden OCR line boxes and a light, sufficiently uniform background must
  also pass validation; only cells whose text actually changes are masked. When Marker
  collapses a ruled digital table into one oversized cell, the extraction stage attempts a clipped
  PyMuPDF repair and accepts it only with at least 88% source/candidate token agreement. Malformed,
  duplicated, boxed prose panels, tables whose visible PDF text disagrees with extraction, or
  otherwise ambiguous tables remain unchanged and are reported. Borderless tables require stored
  extractor geometry or a uniquely validated text-lattice solution; geometry is never inferred
  from translated text alone. A failure in any cell retains the
  complete source table instead of producing a partially translated grid. Such tables can still be
  translated as reflowed structured content in the readable PDF when OCR recovered their rows.
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
