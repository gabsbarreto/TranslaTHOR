# Local PDF Translation App

## What It Does

This browser-based app translates PDFs locally.

1. The user uploads one or more PDFs.
2. The backend classifies each document as `digital_good_text`, `scanned_no_text`, `bad_hidden_ocr`, `mixed`, or `unknown`.
3. Marker extracts documents with usable embedded text.
4. The selected OCR engine handles poor scans: experimental Surya 2 through
   `llama.cpp`, legacy Surya + Qwen through `mlx-vlm`, or legacy Surya + Marker.
5. Qwen full-page OCR reads rendered page images directly when Marker needs an uncertain-document OCR fallback.
6. Surya OCR regions are cleaned and merged into traceable logical translation chunks.
7. Extracted text is normalized into a shared structured document model.
8. Marker table-cell polygons are retained for reconstruction. When older digital jobs have no
   stored cells, partial-rule and whitespace-separated tables can recover their logical grid from
   PDF text geometry only when every source cell has one unambiguous, monotonic alignment.
   Structurally collapsed ruled tables are also checked against the source PDF and repaired from an
   exact clipped PyMuPDF grid only when the source text strongly agrees. Qwen Markdown tables are
   converted to stable HTML before translation; narrowly ragged rows may be repaired by adding
   empty cells only. On hidden-OCR pages, a plain-text two-column region is promoted to a table only
   when Surya table confidence, a repeated physical gutter, and Qwen-to-hidden-text alignment all
   agree.
9. Figure regions are deduplicated, validated, associated with captions, and captured as
   high-resolution PNG previews plus clipped vector SVG assets when the source supports them.
10. A local MLX Qwen model translates the document into English. Compatible prose first passes run
    in adaptive batches of up to four on Metal, while results are applied in the original document
    order. Substantive output is checked for source-language residue, unchanged source text, and
    damaged table topology. Only failed validations enter a stricter second batch; if recovery still
    fails, the source is retained and the failure is recorded instead of claiming success. Table,
    caption, and grouped physical-region safeguards keep their specialised translation paths.
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
        surya2_adapter.py             # Surya 2 output -> shared document schema
        surya2_extractor.py           # Direct Surya 2 extraction orchestration
        surya2_runtime.py             # Persistent isolated worker lifecycle
scripts/
  run_dev.sh
  setup_local_runtime.sh
  qwen_ocr_worker.py
  surya_layout_worker.py
  surya2_worker.py
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
  For a Surya 2 raster-only scan, PDF-space Surya boxes provide the placement boundary and a local
  raster pass finds dark source-glyph rows on a light, uniform background. Only those row masks are
  covered before translated searchable text is inserted. Dense, dark, nonuniform, missing, or
  overflowing regions remain unchanged. Figures and equations remain locked, and image-only tables
  without trustworthy cell geometry are retained for safety.
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

The normal interface exposes only the poor-text/scanned OCR choice. Surya 2 through `llama.cpp` is
selected by default; Surya + Qwen remains available as a text-critical fallback, and the legacy
Marker OCR path remains available for comparison. Drop or select one or more PDFs and the server
applies the remaining automatic extraction and local translation defaults. Good born-digital PDFs
always continue through Marker text-layer extraction regardless of the selected scan engine.

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

Install the shared Marker 2 + Surya 2 extraction runtime and its officially
supported Apple Silicon backend:

```bash
bash scripts/setup_surya2_runtime.sh
```

The script installs the fully resolved `requirements-surya2.lock.txt`. The
shorter `requirements-surya2.txt` records the intentionally selected top-level
compatibility set.

This pins `marker-pdf==2.0.0` and `surya-ocr==0.22.1`, and sets
`SURYA_INFERENCE_BACKEND=llamacpp`. Marker uses explicit `balanced` conversion
for born-digital documents. The direct OCR path reuses one Surya inference
manager/server across pages and queued jobs, with up to five page requests in
flight at once.

Inspect Surya layout boxes for rendered OCR pages:

```bash
.venv-surya2/bin/python scripts/surya_layout_worker.py \
  --input-dir workspace/jobs/<job-id>/qwen_ocr/rendered_pages \
  --output-dir workspace/jobs/<job-id>/qwen_ocr/surya_layout
```

This writes `layout.json`, padded region crops, annotated page previews, and full-page
`boxed_pages` overlays. The app generates those overlays automatically for `scanned_no_text`
and `bad_hidden_ocr` PDFs, then sends each complete overlay page to Qwen. The worker uses the
same `.venv-surya2` runtime as Marker and direct Surya OCR.

## Run

```bash
bash scripts/run_dev.sh
```

Open `http://127.0.0.1:8000`.

The isolated translation worker prefers `.venv/bin/python`, so the tested MLX stack is used even
when the web server was started by another Python installation. MLX model operations are explicitly
scheduled on the Metal GPU. Qwen 3.5 uses MLX's fused
`mlx.fast.scaled_dot_product_attention` primitive for its softmax-attention layers; its remaining
linear-attention layers use the model's MLX Metal kernels. There is no separate third-party
FlashAttention switch in MLX.

Ordinary prose uses adaptive `batch_generate()` groups of up to four prompts with an 8192-token
combined input/output budget. The shared chat template and translation-instruction prefix are
rendered once and, when tokenizer-boundary checks pass, their token IDs are reused. CPU helper pools
default to a hardware-aware value that reserves system capacity; on the tested 12-core M4 Pro,
six tokenizer threads outperformed eight or twelve. Runtime metadata records the selected device,
attention backend, thread count, instruction-cache mode, batch calls, and fallback counts.

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

With a development server already running, execute selected cases through the real upload, queue,
translation, readable-PDF, and original-layout-PDF routes:

```bash
.venv/bin/python scripts/run_pdf_regression_workflows.py \
  --base-url http://127.0.0.1:8000 \
  --case fr-digital-gender-psychiatry \
  --case es-hidden-ocr-endocrinology \
  --output-dir workspace/regression_runs/manual
```

Omit every `--case` argument to run all ten cases. The runner never starts, stops, cancels, or
deletes server jobs. It writes an atomic run manifest and downloads only the two user-facing PDF
modes after each job reaches a terminal stage. Every downloaded PDF is opened with PyMuPDF and
must contain at least one page with valid geometry before the runner records it as successful. If
the built corpus manifest records a fixture checksum or page count, the runner verifies both before
uploading and refuses a modified or incomplete fixture.
Completed job directories can then be audited without invoking either model:

```bash
.venv/bin/python scripts/evaluate_pdf_regression_runs.py \
  workspace/jobs \
  --json-output workspace/regression_runs/manual/evaluation.json \
  --markdown-output workspace/regression_runs/manual/evaluation.md
```

The evaluator checks artifact integrity (including opening figure previews), coordinate-space-aware
structured figure/table boxes, exact original-page geometry, report-count consistency, reported
skips/overflow/scaling, expected source/extraction page counts, visible-source-character-weighted
replacement coverage, and severe punctuation-only or near-empty translation collapses. It remains
a diagnostic command and exits successfully after writing a report even when the report contains
review failures. It explicitly does not claim to score translation meaning or rendered visual
fidelity; representative PDFs still need rendering and visual inspection.

See `tests/regression_corpus/README.md` for provenance, storage, and source-path overrides. The
artifact validation test skips on machines where the private corpus has not been built; the tracked
ten-case specification is always tested.

## Reconstruction Limitations

- Every extracted text, table, and caption box is authoritative in original-layout output, whether
  it came from Surya 2, Surya + Qwen, Marker, hidden OCR, or a born-digital PDF. Reconstruction paints
  the complete recorded PDF-space box and inserts the complete target there. It does not use source-
  text matching, hidden-OCR alignment, table-cell recovery, background uniformity, script checks, or
  overlap heuristics as competing geometry vetoes. Background sampling only chooses a cosmetic fill
  colour; an inconclusive sample falls back to white.
- Text is downscaled as far as PyMuPDF requires, with no readability floor, and is committed only
  after both preflight and real insertion report that the complete target fits. A failed insertion
  rolls back the complete page transaction so reconstruction cannot silently delete or clip a
  fragment. Figure and equation blocks remain protected unless the extractor explicitly classified
  the region as text. Missing, invalid, cross-page, or unconfirmed placement geometry is still
  retained and reported because there is no page-local box in which to place it.
- Rotated pages are retained unchanged by the first original-layout implementation, while their
  original dimensions, crop boxes, rotation, and visual content remain intact.
- Missing or invalid bounding boxes are skipped and reported rather than guessed. New translations
  preserve one independently placeable target per source block, even when several blocks share one
  model request for linguistic context; long groups are batched without translating a physical block
  twice. The shared continuation resolver may bridge one or more intervening pages only when every
  such page contains layout objects (tables, figures, captions, or equations) plus optional margin
  furniture, and strong text, section, style, and geometry evidence connects the prose on both sides.
  The prose is translated as one passage while each intervening object remains a separate translation
  unit in source order. Blank pages, headings, lists, references, ordinary prose, and ambiguous seams
  block the bridge.
  Legacy cross-page translation batches are recovered only when their preserved source and
  translated paragraph boundaries prove a one-to-one mapping; ambiguous legacy batches remain
  unchanged.
- Tables are re-typeset once inside their complete extracted table box. Stored cell polygons,
  inferred grids, hidden OCR, and visible source-cell agreement may still help extraction and
  translation, but they cannot veto original-layout placement or split it into competing cell boxes.
  Figures, captions, and equations remain separate source-order blocks and are never consumed into a
  table replacement.
- OCR table repair is deliberately narrow. Ragged Markdown is accepted only when a dominant width
  is close to every row and repair consists solely of inserting missing empty cells. Hidden-OCR
  two-column inference requires at least four aligned rows, stable gutter support, concise cell
  contents, and strong source-text agreement. OCR character mistakes (for example a misread table
  number) are preserved rather than replaced through language- or document-specific rules.
- Translation validation is a safety net, not a reference-quality metric. Confidently failed
  chunks are retried once and then retained with an explicit warning. Short headings, names,
  citations, formulas, and link-only blocks are exempt where language detection would be
  unreliable. Human review is still required for terminology and fluency.
- Figure detection follows Marker/Surya structured regions. False-positive visual regions or missed
  figures can still occur when upstream layout metadata is unreliable; the JSON report and
  structured document retain the evidence used for each association.

## Optional Environment Variables

- `TRANSLATION_PYTHON`: isolated translation worker executable; defaults to `.venv/bin/python` and
  falls back to the server interpreter when that file is unavailable.
- `TRANSLATION_BATCH_SIZE`: maximum compatible prose prompts per MLX batch; defaults to `4`.
- `TRANSLATION_BATCH_TOKEN_BUDGET`: combined estimated input/output tokens per batch; defaults to
  `8192` and automatically splits long groups.
- `MLX_CPU_THREADS`: CPU tokenizer/helper threads. `0` or unset selects a conservative
  hardware-aware value instead of using every core.
- `QWEN_OCR_PYTHON`: Python executable containing `mlx-vlm`.
- `QWEN_OCR_MODEL`: Qwen OCR model identifier.
- `QWEN_OCR_PROMPT`: OCR transcription prompt.
- `SURYA_LAYOUT_PYTHON`: Python executable containing Surya; defaults to `.venv-surya2/bin/python`.
- `OCR_ENGINE`: `surya2_llamacpp` (branch default), `surya_qwen_mlx`, or `marker_surya`.
- `SURYA2_PYTHON`: isolated Surya 2 Python; defaults to `.venv-surya2/bin/python`.
- `SURYA2_DPI`: render DPI; defaults to `192`.
- `SURYA2_STRATEGY`: `full_page` (default) or `layout_then_block`.
- `SURYA_INFERENCE_BACKEND`: must be `llamacpp` for the experimental engine.
- `SURYA_INFERENCE_PARALLEL`: concurrent direct-Surya pages; defaults to `5`.
- `SURYA_INFERENCE_CTX_PER_SLOT`: context reserved per concurrent page; defaults to `16384`.
- `SURYA_INFERENCE_CTX_SIZE`: optional total llama.cpp context override. Values below the
  per-slot requirement are automatically raised to prevent batched-page truncation.
- The Surya workers and Marker subprocess default `SURYA_GUIDED_LAYOUT=false`
  for compatibility with the tested llama.cpp 10090 grammar parser.
- `MARKER_BIN`: Marker executable path; `run_dev.sh` defaults it to
  `.venv-surya2/bin/marker_single`.
- `MARKER_CONVERSION_MODE`: Marker 2 conversion mode; defaults to `balanced`.
- `ENABLE_MARKER_TABLE_OCR_RETRY`: validate extracted table numbers against the source PDF and
  retry only affected pages with forced OCR; defaults to `true`.
- `ENABLE_QWEN_OCR_FALLBACK`: Enable full-page Qwen OCR fallback.
- `ENABLE_LOCAL_VLM_REPAIR`: Enable Marker block repair through the optional local VLM endpoint.

See [OCR architecture](docs/architecture.md) and
[benchmarking](docs/benchmarking.md) for schema, coordinate, lifecycle, and
reproduction details. The measured experiment and recommendation are in the
[Surya 2 benchmark report](docs/surya2-benchmark-report.md).
