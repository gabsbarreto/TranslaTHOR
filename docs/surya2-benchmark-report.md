# Surya 2 llama.cpp experiment

Date: 2026-07-24

Starting `main` commit: `e657076893b8f2a6d60871641ab6ab4e3dae62bd`

## Decision

Use direct Surya 2 full-page OCR through `llama.cpp` as the primary OCR path
for poor-text and scanned PDFs. Keep the existing Marker text-layer path for
good born-digital PDFs. Retain both legacy OCR engines as selectable fallbacks
while a larger multilingual corpus is evaluated.

On the representative Spanish scan, full-page Surya 2 was 35% faster than
Surya + Qwen and 69% faster than Surya + Marker, with 68% and 74% less peak
memory respectively. It also retained the exact two chart regions and their
caption relationships; Qwen retained captions but no image assets, and Marker
over-segmented the two charts into four figures.

This is not an unconditional text-accuracy win. Qwen had the lowest scan CER
(0.362 versus 0.499 for full-page Surya 2), while the three systems' WER
rankings differed. Surya 2 should replace Marker OCR and become the default
instead of Qwen for the tested scanned-document workflow, but Qwen should
remain available for text-critical rescans until this CER result has been
checked on a broader curated ground-truth set.

## Architecture choice

The implementation uses Surya 2 directly and adapts its native block output to
`DocumentModel`. One persistent worker owns one shared
`SuryaInferenceManager(method="llamacpp")`, `LayoutPredictor`, and
`RecognitionPredictor`. Its `llama-server` is reused across pages and jobs and
is stopped at application shutdown or cancellation.

Upgrading the legacy Marker environment was rejected for this experiment.
Current Marker is viable and itself uses the Surya VLM, but replacing Marker
1.10.2/Surya 0.17.1 would invalidate the requested legacy comparison.
Running direct Surya 2 and then Marker would duplicate OCR. The direct adapter
is also a simpler path than the existing Qwen workflow, which renders pages,
runs legacy Surya layout, annotates images, invokes Qwen through `mlx-vlm`,
and reconciles Qwen regions back to Surya boxes.

The choice follows the current [Surya API and output
schema](https://github.com/datalab-to/surya), including the shared inference
manager, `blocks`, HTML, raw/canonical labels, polygons, confidence,
skipped/error state, and the full-page versus block-level recognition APIs.
Surya 0.22.1 was the [current stable
release](https://github.com/datalab-to/surya/releases) tested here. The
[current Marker architecture](https://github.com/datalab-to/marker) was also
reviewed before choosing the direct adapter.

## Environment

- MacBook Pro `Mac16,8`, Apple M4 Pro, 12 CPU cores (8 performance, 4
  efficiency), 24 GB unified memory
- macOS 26.5.2, build 25F84, arm64
- Main and Marker Python: 3.13.5
- Isolated Surya 2 Python: 3.14.4
- Surya 2: 0.22.1; Transformers 5.12.1; Torch 2.11.0; torchvision 0.26.0
- Legacy Marker: 1.10.2; Surya: 0.17.1; Transformers 4.57.6
- Qwen OCR: `mlx-community/Qwen3.5-4B-4bit`
- `mlx-vlm`: 0.4.4; MLX: 0.31.2
- Translation: `mlx-community/Qwen3.5-9B-MLX-4bit`
- `llama-server`: 10090 (`7347430f4`), AppleClang, Darwin arm64
- Benchmark Python packages include psutil 7.2.2

`llama-server` was initially absent and was installed with the officially
documented Apple Silicon command:

```bash
brew install llama.cpp
```

The tested dependency closure is pinned in
`requirements-surya2.lock.txt`. The downloaded Surya model and multimodal
projector occupied approximately 1.4 GB. The first ever one-page run,
including network download, took 243.4 seconds. "Cold" below means a new
engine process/server with the model already in the local cache; warm runs
reuse filesystem caches and, for Surya 2, the same inference manager/server.

## Documents and method

No source PDF, reference transcription, or extracted private content is
committed. The ignored local corpus supplied matching raster and born-digital
versions of a five-page Spanish epidemiology article with an English abstract.

The scan benchmark selected pages 1, 3, and 4. Together they contain small
text, a two-column first page, running furniture, complex author/sidebar
ordering, two charts and captions, a dense table, and inline statistical
expressions. The digital benchmark selected page 2, which contains clean
born-digital Spanish text, small two-column content, running furniture, and
inline statistical expressions.

Every OCR engine received the same page-subset PDF, 192 DPI, concurrency one,
and reconstruction settings. OCR was intentionally forced on the
born-digital page only to compare engines; production should continue using
the text layer for `digital_good_text`. Each engine ran once cold and three
times warm. Warm figures are medians. Peak memory is process-tree RSS
augmented by macOS `footprint` sampling so Metal allocations are represented.
CER/WER used Unicode-normalized, case-folded, whitespace-normalized text
against the corresponding digital PDF text.

Commands:

```bash
SURYA_INFERENCE_BACKEND=llamacpp SURYA_INFERENCE_PARALLEL=1 PYTHONPATH=backend \
  .venv/bin/python scripts/benchmark_ocr_engines.py \
  --manifest /tmp/translathor-surya2-final-manifest.json \
  --output-dir workspace/benchmarks/surya2-final-v4 \
  --engines marker_surya,surya_qwen_mlx,surya2_full_page,surya2_layout_then_block \
  --dpi 192 --warm-runs 3 --timeout 3600

SURYA_INFERENCE_BACKEND=llamacpp SURYA_INFERENCE_PARALLEL=1 PYTHONPATH=backend \
  .venv/bin/python scripts/benchmark_ocr_engines.py \
  --manifest /tmp/translathor-surya2-digital-manifest.json \
  --output-dir workspace/benchmarks/surya2-digital-final \
  --engines marker_surya,surya_qwen_mlx,surya2_full_page \
  --dpi 192 --warm-runs 3 --timeout 3600
```

## Scan results: three pages

| Engine | Cold total | Cold s/page | Warm median | Warm s/page | Pages/s | Peak GiB | CER | WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Surya + Marker | 211.55 s | 70.52 | 216.24 s | 72.08 | 0.0139 | 12.54 | 0.515 | 0.493 |
| Surya + Qwen/MLX | 130.41 s | 43.47 | 104.24 s | 34.75 | 0.0288 | 10.32 | **0.362** | 0.507 |
| Surya 2 full page | 97.51 s | 32.50 | 67.54 s | 22.51 | 0.0444 | **3.30** | 0.499 | 0.514 |
| Surya 2 layout then block | **81.03 s** | **27.01** | **66.76 s** | **22.25** | **0.0449** | 5.53 | 0.457 | 0.497 |

All 16 runs completed extraction with zero retries and zero error blocks.
Surya 2 reported two skipped blocks: these were the two intentionally
textless charts, whose geometry and crops were retained.

| Engine | Label sequence | Reading order | Header/footer | Table | Equation block | Figure/caption |
|---|---:|---:|---:|---:|---:|---:|
| Surya + Marker | 0.774 | 1.000 | 0 / 0 | 1 | 0 | 4 / 3 |
| Surya + Qwen/MLX | **0.935** | 1.000 | 4 / 2 | 1 | 0 | 0 / 3 |
| Surya 2 full page | **0.935** | 1.000 | 5 / 2 | 1 | 0 | **2 / 3** |
| Surya 2 layout then block | 0.871 | 1.000 | 5 / 2 | 1 | 0 | **2 / 3** |

No selected page contained a standalone display-equation block. All engines
kept the visible inline statistical expressions as text, so this run does not
establish comparative LaTeX fidelity.

## Born-digital forced-OCR result: one page

| Engine | Cold total | Warm median | Peak GiB | CER | WER | Header/footer |
|---|---:|---:|---:|---:|---:|---:|
| Surya + Marker | 124.90 s | 116.95 s | 11.54 | 0.0772 | **0.3371** | 0 / 0 |
| Surya + Qwen/MLX | 55.05 s | 55.25 s | 9.29 | 0.1052 | 0.3452 | 2 / 3 |
| Surya 2 full page | **44.67 s** | **36.86 s** | **2.84** | **0.0767** | 0.3574 | 2 / 1 |

The stored Surya 2 reading-order metric is 0.5, but manual review found the
four passages in the correct order. The exact-match probe expected
`periodos`; Surya emitted the correct accented `períodos`, so one probe was
treated as missing and three pairwise comparisons failed. The score is
reported unchanged rather than post-processed.

## Representative quality review

- Full-page Surya 2 coordinate overlays closely followed the text, chart,
  caption, table, and page-furniture boundaries. PDF-point conversion uses
  independent x/y scaling without a y-axis flip; the overlay and unit tests
  confirm this convention.
- Full-page Surya 2 produced the expected two chart blocks, linked each to its
  nearest same-page caption, cropped the original rendered regions, and
  reinserted them at their reading-order positions. The translated PDF
  contains both charts immediately before their translated captions.
- The Surya 2 table retained all visible row values and rowspan structure.
  Its model HTML declared `colspan="7"` over eight interval subcolumns, which
  displaces one percent header in the reconstructed PDF. Marker hallucinated
  an additional header value (`ts peculis`) in the same table.
- Marker detected four figures for two charts and its legacy Markdown placed
  image assets in a trailing figures section rather than with captions.
  Qwen detected the three captions but produced no figure assets.
- Qwen had the best scan CER, but on the digital page it repeatedly
  transcribed layout-annotation strings such as `SURYA 1: PageHeader` and
  `SURYA 3: text` despite instructions not to do so.
- The common translation worker produced readable English from all three
  cold outputs (Marker 89.6 s, Qwen 77.1 s, Surya 2 87.4 s; translation was
  not included in OCR timings). Only the Surya 2 reconstructed PDF retained
  both charts inline. Marker also propagated a truncated source sentence;
  Qwen left one Spanish affiliation untranslated.

## Strategy decision

`layout_then_block` is not the default. Its 22.25 s/page median is only 1.2%
faster than full-page OCR, while its 5.53 GiB peak is 68% higher. It also
split overlapping sidebar/name boxes, hallucinated `Universidad de Quitta`,
and reduced label-sequence accuracy from 0.935 to 0.871. Full-page mode
avoided the overlap and has the cleaner downstream schema.

The worker sets `SURYA_GUIDED_LAYOUT=false` only for the optional block
strategy. Surya 0.22.1 generates a layout grammar containing `\d` escapes
that llama.cpp 10090 rejects with `failed to parse grammar`. Unguided JSON is
still parsed and validated by Surya. This workaround must be retested when
either dependency changes.

## Known limitations and risks

- The corpus is one Spanish/English scientific article, not a statistically
  representative multilingual benchmark. CER/WER use PDF-extracted text,
  not a hand-curated transcript, and therefore penalize normalization,
  ligature, and hyphenation differences.
- No standalone display equation, handwriting, very old scan, or non-Latin
  page was available. Equation and broad language claims remain untested.
- Surya's model-weight license is more restrictive than its Apache-2.0 code
  license and must be reviewed for the intended deployment.
- The optional layout strategy requires guided-decoding compatibility work.
- After the digital benchmark had written its complete JSON and Markdown
  report, the mixed MLX/Surya parent process exited 139 during native
  interpreter finalization. All 12 run artifacts were complete, the Surya
  worker/server lifecycle independently exited 0, and no server remained.
  This intermittent macOS teardown issue should be watched in longer runs.
- The table-header colspan defect should be normalized before declaring
  complex-table reconstruction production-complete.

## Recommended production configuration

```bash
export OCR_ENGINE=surya2_llamacpp
export SURYA_INFERENCE_BACKEND=llamacpp
export SURYA_INFERENCE_PARALLEL=5
export SURYA_INFERENCE_CTX_PER_SLOT=16384
export SURYA2_STRATEGY=full_page
export SURYA2_DPI=192
export MARKER_CONVERSION_MODE=balanced
```

A follow-up test on 2026-08-10 used the same 11-page source PDF with five
parallel slots and 16,384 context tokens per slot. Direct-Surya extraction fell
from 663.664 seconds to 371.875 seconds (44% less time), with a 5.22 GiB peak
RSS. All 179 blocks retained identical text, labels, reading order, and
skip/error state; eight boxes differed by at most 2.117 rendered-image pixels.

Continue to route `digital_good_text` through Marker 2 balanced extraction.
Use direct Surya 2 full-page OCR for scans and poor/hidden OCR. Keep
`surya_qwen_mlx` selectable for text-critical comparison or fallback, and
keep `marker_surya` only for regression and documents where its postprocessing
is known to help. Do not make the block-level Surya 2 strategy the default.

### Reconstruction integration added after the benchmark

The `surya2afterreconstruction` integration adds conservative original-layout
overlays for Surya 2 raster-only text blocks. Synthetic integration tests
verify identity coordinate conversion, raster foreground-row masking,
searchable translated text insertion, pixel-identical preservation of a
nearby figure, and unchanged fallback for a nonuniform visual region.

These reconstruction tests were added after the OCR timings and do not alter
the benchmark table above. Image-only Surya tables without validated cell
geometry remain unchanged in original-layout output, and the private
regression corpus still requires a fresh live end-to-end run before this path
should be treated as production-complete.

## Verification

Before implementation, the unmodified starting commit passed:

```text
.venv/bin/python -m pytest -q
85 passed in 6.92s
```

Final verification on the combined `surya2afterreconstruction` branch:

```text
.venv/bin/pytest -q
325 passed, 5 warnings in 21.86s

.venv/bin/ruff format --check <integration-changed Python files>
5 files already formatted

.venv/bin/ruff check backend tests scripts
All checks passed

PYTHONPATH=backend .venv/bin/mypy <seven Surya/reconstruction core files>
Success: no issues found in 7 source files

bash -n scripts/run_dev.sh scripts/setup_local_runtime.sh scripts/setup_surya2_runtime.sh
.venv-surya2/bin/python -m pip check
No broken requirements found in the shared Marker 2 + Surya 2 environment
```

A standalone manager/server lifecycle reported worker exit code 0. Process
inspection after tests and benchmarks found no remaining `llama-server` or
Surya worker.
