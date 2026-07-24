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

## Legacy Surya + Marker

1. `PDFTypeDetector` classifies the PDF text layer.
2. `PDFExtractor` invokes `marker_single` in the isolated `.venv-marker`
   environment.
3. Marker 1.10.2 performs PDF parsing and, when OCR is forced, calls Surya
   0.17.1 for layout/OCR.
4. Marker JSON is converted by `MarkerDocumentBuilder` into the shared model.
5. Marker HTML tables become `TableModel` objects; figures become
   `FigureAsset` placeholders.

The `marker_surya` engine forces Marker OCR for poor-text documents while
retaining Marker text-only extraction for good born-digital PDFs.

## Legacy Surya + Qwen through mlx-vlm

1. Poor scans bypass Marker.
2. `PageRenderer` renders every selected page.
3. the Surya v1 layout worker finds regions and reading order and writes crops,
   overlays, and `layout.json`.
4. Qwen receives the complete annotated page through `mlx-vlm`, returning
   `<region>`-wrapped Markdown.
5. `QwenMarkdownParser` aligns Qwen regions with Surya layout regions.
6. `OCRToTranslationParser` excludes running furniture from translation and
   merges logical text continuations while leaving the source blocks intact.

This remains available as `surya_qwen_mlx`.

## Experimental direct Surya 2 + llama.cpp

The selected design is a direct adapter, not a Marker 2 upgrade.

Marker 2.0.0 is viable and uses Surya 2, but it requires a new Marker schema,
Transformers 5, and Surya 0.22.1. Replacing Marker 1.10.2 would make the
legacy comparison non-equivalent. Running direct Surya 2 followed by Marker
would also repeat work. The direct adapter consumes Surya's native block
schema exactly once and preserves TranslaTHOR's existing boundary.

The runtime is isolated because legacy Marker and mlx-vlm use incompatible
Transformers stacks:

```text
FastAPI job queue (one active job)
  -> Surya2LlamaCppExtractor
     -> render selected PDF pages at one configured DPI
     -> persistent scripts/surya2_worker.py
        -> one shared SuryaInferenceManager(method="llamacpp")
        -> one llama-server for successive pages and queued jobs
        -> RecognitionPredictor (full page by default)
     -> Surya2DocumentAdapter
     -> DocumentModel / translation chunks / Markdown
```

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

## Official references

- [Surya repository and v2 usage](https://github.com/datalab-to/surya)
- [Surya releases](https://github.com/datalab-to/surya/releases)
- [Marker repository](https://github.com/datalab-to/marker)
