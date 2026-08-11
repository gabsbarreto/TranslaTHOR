# Reproducible OCR benchmark

The benchmark runner compares identical PDF pages at one DPI and concurrency
of one:

```bash
PYTHONPATH=backend .venv/bin/python scripts/benchmark_ocr_engines.py \
  --manifest /absolute/path/to/benchmark-manifest.json \
  --output-dir workspace/benchmarks/ocr-2026-07-24 \
  --dpi 192 \
  --warm-runs 3
```

Use `--cold-only` for one fresh-process timing per engine. When every page is
selected, the runner copies the original PDF byte-for-byte instead of rewriting
it as a page subset. Completed runs are atomically checkpointed to
`benchmark_results.json` and `benchmark_report.md` after each engine finishes.
Use `--surya-parallel 5 --surya-context-per-slot 16384` to test five concurrent
Surya pages without reducing the context available to each page.

`workspace/benchmarks/` is ignored. Do not put private PDFs, reference text,
or generated OCR in the repository.

Supported engines:

- `marker_balanced` (production Marker 2 balanced path, including source-validated table retry)
- `marker_surya`
- `surya_qwen_mlx`
- `surya2_full_page`
- `surya2_layout_then_block`

Use `--engines` with a comma-separated subset. Add `--translate-final` to run
the common local translation worker and create one translated reconstructed
PDF from each engine's cold run. Warm runs omit translation so their extraction
timings remain comparable. Without it, the runner still creates an OCR-source
reconstruction for visual comparison.

The runner makes a one-time page subset PDF for each manifest entry. Marker's
low- and high-resolution DPI, Qwen rendering DPI, and Surya 2 rendering DPI
are all set to `--dpi`. Batch size and client/backend concurrency are one.
Every engine receives the same subset and downstream reconstruction settings.

## Manifest

```json
{
  "documents": [
    {
      "id": "scientific-scan",
      "pdf": "/non-repository/path/document.pdf",
      "reference_pdf": "/non-repository/path/born-digital-original.pdf",
      "pages": [1, 3],
      "characteristics": [
        "scanned",
        "two-column",
        "table",
        "equation",
        "figure and caption"
      ],
      "reference_text": {
        "1": "references/scientific-scan-p1.txt",
        "3": "references/scientific-scan-p3.txt"
      },
      "expected_labels": [
        "header",
        "heading",
        "paragraph",
        "table",
        "caption",
        "footer"
      ],
      "expected_reading_order": [
        "first distinctive passage",
        "second distinctive passage",
        "final distinctive passage"
      ]
    }
  ]
}
```

Paths are resolved relative to the manifest, except absolute paths. References
are optional. `reference_pdf` extracts the matching page text directly; use
`reference_text` instead when a curated ground-truth transcript is available.
When supplied, the runner reports CER, WER, deletion/insertions,
TranslaTHOR block-type sequence accuracy, and pairwise reading-order accuracy.

## Outputs

- `benchmark_results.json`: environment, versions, raw runs, and warm medians
- `benchmark_report.md`: results table and representative previews
- `inputs/`: identical page-subset PDFs
- `runs/<document>/<engine>/`: raw engine data, coordinate overlays,
  structured JSON, Markdown, and reconstructed PDFs

Each engine runs once cold and at least three times warm. The report uses the
median warm total time and records extraction time, downstream reconstruction
time, total wall time, seconds/page, pages/second, process-tree peak RSS, block
errors/skips, retries, structural counts, and optional quality metrics.
On macOS, the sampler also takes one-second `footprint` samples so Metal/MPS
allocations omitted from RSS are included in the reported peak.

Manual review remains required for:

- table cell structure and content
- equation fidelity
- figure-caption pairing
- coordinate overlays
- untranslated and translated reconstructed PDFs
- missing, duplicate, and hallucinated passages

The script deliberately does not auto-declare a winner from timing or a
published model benchmark.

Surya 2 dependency resolution is recorded in
`requirements-surya2.lock.txt` for the tested macOS arm64/Python 3.14 runtime.
The benchmark report records the actual package and llama.cpp versions again
so a lock update cannot silently change a published comparison.
