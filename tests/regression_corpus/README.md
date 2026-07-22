# Local PDF regression corpus

This specification produces five compact non-English digital PDF fixtures and five scan-only
counterparts. Each pair has the same pages and dimensions, so extraction and reconstruction changes
can be compared without conflating document content with scan quality.

The two source collections currently contain only three non-English-dominant publications and no
non-English scan-dominant documents. The five cases therefore select complementary, verified
non-English page ranges from those publications and derive a raster-only copy of each case. The
cases cover Spanish and French front matter, multi-column prose, tables, charts, a flowchart, and
captions.

The publication PDFs and their derived fixtures are deliberately not tracked in Git. The repository
is public and the source files do not all grant redistribution rights. Instead, Git tracks:

- `corpus_spec.json`, including source SHA-256 hashes and original page numbers;
- `scripts/build_pdf_regression_corpus.py`, which refuses changed source files;
- `tests/test_pdf_regression_corpus.py`, which validates the local corpus when it is present.

Build the corpus from the two local RQ collections:

```bash
PYTHONPATH=backend .venv/bin/python scripts/build_pdf_regression_corpus.py
```

The generated corpus is stored in `workspace/regression_corpus/` and contains `digital/`,
`scanned/`, and `manifest.json`. Run its validation independently with:

```bash
PYTHONPATH=backend .venv/bin/pytest -q tests/test_pdf_regression_corpus.py
```

On machines without the private source collection, the specification test still runs and the local
artifact test is skipped. Override either source root or the output directory with the builder's
command-line options.
