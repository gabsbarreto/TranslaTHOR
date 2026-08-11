# OCR and document-processing architecture

## Compatibility boundary

All extraction engines return `PDFExtractionResult`. Its `document` is a
`DocumentModel` containing:

- PDF-point page dimensions and page numbers
- globally ordered typed blocks
- tables and figure assets
- translation chunks
- extraction metadata and warnings

Translation reads `DocumentModel.translation_chunks`, updates the same model,
and emits Markdown. Reconstruction consumes the translated model/Markdown; it
does not import an OCR implementation. An OCR engine can therefore change only
if it preserves this boundary.

`Block` has explicit fields for the Surya 2 data that cannot be inferred later:
`raw_label`, `html`, `polygon`, `skipped`, and `error`. Legacy extractors leave
these optional fields at their defaults.

## Marker 2 + Surya 2 for born-digital PDFs

1. `PDFTypeDetector` classifies the PDF text layer.
2. `PDFExtractor` invokes Marker 2.0.0 from the shared `.venv-surya2`
   environment with explicit `--mode balanced`.
3. Marker uses Surya 0.22.1 for layout/OCR while preserving its specialised
   born-digital PDF parsing and table processing.
4. Marker JSON is converted by `MarkerDocumentBuilder` into the shared model.
5. Marker HTML tables become `TableModel` objects; figures become
   `FigureAsset` placeholders.

Good born-digital documents use this path without `--disable_ocr`. If a table
omits numeric values present inside its source PDF box, only the affected pages
are retried with forced OCR. Missing values are copied into empty balanced-mode
cells only when the source-number deficit proves the mapping; existing values
and row associations win. A table that still fails validation is retained in
the original-layout PDF instead of masking source data. The `marker_surya`
engine remains available for direct comparison.

## Legacy Surya + Qwen through mlx-vlm

1. Poor scans bypass Marker.
2. `PageRenderer` renders every selected page.
3. the Surya 2 layout worker finds regions and reading order and writes crops,
   overlays, and `layout.json`.
4. Qwen receives the complete annotated page through `mlx-vlm`, returning
   `<region>`-wrapped Markdown.
5. `QwenMarkdownParser` aligns Qwen regions with Surya layout regions.
6. `OCRToTranslationParser` excludes running furniture from translation and
   uses the shared continuation resolver to merge proven logical text continuations while leaving
   source blocks and intervening layout objects intact. This includes prose interrupted by a fully
   table-, figure-, caption-, or equation-only page when both surrounding fragments pass the same
   conservative textual, section, style, and geometry checks.

This remains available as `surya_qwen_mlx`.

## Direct Surya 2 + llama.cpp for scans

The direct adapter consumes Surya's native block schema exactly once for scans
and poor-text PDFs. Marker 2 shares the same package runtime but is not run on
top of direct Surya output, so OCR work is not duplicated.

The extraction runtime is isolated because Surya 2/Marker 2 and mlx-vlm use
different Transformers stacks:

```text
FastAPI job queue (one active job)
  -> Surya2LlamaCppExtractor
     -> render selected PDF pages at one configured DPI
     -> persistent scripts/surya2_worker.py
        -> one shared SuryaInferenceManager(method="llamacpp")
        -> one llama-server with five concurrent page slots by default
        -> RecognitionPredictor (full page by default)
     -> Surya2DocumentAdapter
     -> DocumentModel / translation chunks / Markdown
```

The worker submits all rendered pages as one ordered batch. The llama.cpp
backend processes at most five page requests concurrently by default and
returns results in source-page order. Each slot retains 16,384 context tokens;
the worker therefore starts llama-server with 81,920 total context unless the
environment requests a larger value. A stale smaller total-context override is
raised automatically so batching cannot reduce the context available to any page.
The worker is started lazily. Cancellation terminates its process group.
Application shutdown calls `SuryaInferenceManager.stop()` and then terminates
the process group if graceful shutdown fails. A subsequent job restarts a
cancelled worker.

The default strategy is `full_page`. Surya 0.22.1 documents it as its more
accurate route and it uses one VLM request per page; failed or looping pages
automatically fall back to layout plus block OCR inside Surya. The benchmark
runner also exposes `layout_then_block` so this choice is checked on
TranslaTHOR documents.

For layout requests the worker sets `SURYA_GUIDED_LAYOUT=false`. Surya 0.22.1's
generated guided-decoding grammar uses `\d` escapes that Homebrew llama.cpp
10090 rejects. Surya still parses and validates the unguided JSON. This
workaround is version-specific and should be retested when either dependency
changes; the initial guided run produced empty pages and a llama-server
`failed to parse grammar` error rather than usable layout.

## Surya 2 schema mapping

| Surya 2 label | TranslaTHOR type |
|---|---|
| `Title`, `SectionHeader` | `heading` |
| `Text`, `Code` | `paragraph` |
| `ListGroup`, `ListItem` | `list` |
| `Table`, `Form` | `table` |
| `Equation`, `ChemicalBlock` | `equation` |
| `Picture`, `Figure`, `Diagram` | `figure` |
| `Caption` | `caption` |
| `PageHeader`, `PageFooter` | `header`, `footer` |
| `Footnote` | `footnote` |
| `Bibliography`, `TableOfContents` | `reference` |

The canonical label determines `block_type`; the model's pre-canonical
`raw_label` is retained separately. Table HTML is parsed into headers, rows,
and row/column spans while the original HTML is retained. Equations retain
their HTML. Visual blocks intentionally have empty text, but keep their
polygon and bounding box. A same-page nearest-caption pass records
figure-caption and table-caption relationships.

After adaptation, the extractor uses each visual block's image-space box to
crop the rendered source page. The crop path is attached to `FigureAsset` and
Markdown reconstruction inserts it at the original figure block's reading
position, immediately before its related caption. Geometry remains available
even if a crop cannot be produced.

`skipped` is preserved for visual blocks, and `error` blocks remain in the
document so failures are measurable instead of disappearing.

## Coordinates

Surya returns top-left pixel coordinates in the rendered image. TranslaTHOR
stores top-left PDF-point coordinates, matching its pdfplumber/Marker page
space:

```text
pdf_x = clamp(image_x, 0, image_width) * pdf_width / image_width
pdf_y = clamp(image_y, 0, image_height) * pdf_height / image_height
```

The y-axis is not flipped. Both converted polygons and boxes are retained,
along with the original image-space geometry and dimensions. The adapter
tests cover non-square scaling, boundary clamping, reading order, and visual
blocks. Each live run writes image-space overlays for manual coordinate
inspection.

## Original-layout reconstruction after Surya 2

Reconstruction still consumes only the shared translated `DocumentModel`.
The integration branch adds a Surya-specific strategy for raster-only pages
that have no removable PDF text and no hidden-OCR word geometry:

1. `Surya2DocumentAdapter` converts every region to top-left PDF points before
   translation. The reconstruction coordinate converter therefore applies an
   auditable identity scale instead of scaling the box a second time.
2. Figure and equation boxes become locked regions. Redaction guards prevent
   masks from crossing into those visuals.
3. For a translated text block, the source page is rendered only inside its
   Surya box. The dominant light background colour and dark foreground rows
   are measured.
4. Only foreground-row masks are filled, then translated searchable text is
   inserted into the original Surya box. Dense, dark, nonuniform, missing, or
   overflowing regions are retained unchanged and reported.
5. Image-only tables are retained unless trustworthy cell geometry is
   available. Surya 2 table HTML preserves readable/reflowed output, but a
   table-wide raster mask is not considered safe original-layout
   reconstruction.

Hidden-OCR scans continue to use the stricter source-text-to-PDF-word
alignment path. The raster strategy is selected only for blocks explicitly
identified as `surya2_llamacpp`; Qwen and Marker jobs do not silently
change reconstruction behavior.

## Official references

- [Surya repository and v2 usage](https://github.com/datalab-to/surya)
- [Surya releases](https://github.com/datalab-to/surya/releases)
- [Marker repository](https://github.com/datalab-to/marker)
