# Local PDF regression corpus

The core corpus contains five non-English digital PDFs and five non-English scan PDFs selected
from the private files in `workspace/tests/`. It covers German, Spanish, French, and Brazilian
Portuguese documents with multi-column prose, tables, charts, a flowchart, captions, page
decorations, and imperfect text mappings.

The scan set deliberately exercises both scan paths:

- two genuine full-page-image scans retain their noisy hidden OCR and must classify as
  `bad_hidden_ocr`;
- three digital fixtures are rendered as raster-only PDFs with no OCR text layer and must classify
  as `scanned_no_text`.

The source files include exactly five usable digital documents but only two genuine scans. The
derived German, Spanish, and French scans complete the five-case scan set while providing known
visual counterparts for pixel and geometry comparisons. The selected page ranges omit irrelevant
English lending covers and keep the corpus compact.

The source publications and generated fixtures are deliberately not tracked in Git. The repository
is public and the source files do not all grant redistribution rights. Git instead tracks:

- `corpus_spec.json`, including source SHA-256 hashes and original page numbers;
- `scripts/build_pdf_regression_corpus.py`, which refuses changed source files;
- `tests/test_pdf_regression_corpus.py`, which validates every local artifact when present.

Build and validate the corpus:

```bash
PYTHONPATH=backend .venv/bin/python scripts/build_pdf_regression_corpus.py
PYTHONPATH=backend .venv/bin/pytest -q tests/test_pdf_regression_corpus.py
```

Generated files are stored in `workspace/regression_corpus/` under `digital/`, `scanned/`, and
`manifest.json`. On machines without the private source files, the specification test still runs
and artifact validation is skipped. Use `--source-dir` or `--output-dir` to override either local
path.
