from __future__ import annotations

import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

try:
    from langdetect import detect
except Exception:  # pragma: no cover - lightweight test environments may omit langdetect
    detect = None

from app.models.inspection import PdfInspection
from app.models.schema import (
    Block,
    BlockType,
    BoundingBox,
    DocumentMetadata,
    DocumentModel,
    FigureAsset,
    PageMetadata,
    SourceType,
    TableModel,
)
from app.services.profiler import PipelineProfiler
from app.services.table_markup import parse_table_rows, rows_have_consistent_shape

REGION_PATTERN = re.compile(
    r'<region\s+(?P<attributes>[^>]*)>(?P<body>.*?)</region>',
    flags=re.IGNORECASE | re.DOTALL,
)
REGION_ATTRIBUTE_PATTERN = re.compile(r'(?P<name>[a-z_]+)="(?P<value>[^"]*)"', flags=re.IGNORECASE)


class QwenMarkdownParser:
    """Convert Qwen page Markdown into the shared document model without filtering OCR text."""

    def build_document_from_markdown_dir(
        self,
        *,
        inspection: PdfInspection,
        markdown_dir: Path,
        profiler: PipelineProfiler | None = None,
        strict_page_files: bool = False,
        surya_layout_manifest: dict | None = None,
    ) -> tuple[DocumentModel, str]:
        page_items: list[tuple[int, str]] = []
        missing_pages: list[int] = []
        for page in inspection.pages:
            path = markdown_dir / f"page_{page.page_number:04d}.md"
            if path.exists():
                markdown = path.read_text(encoding="utf-8", errors="ignore")
            else:
                markdown = ""
                missing_pages.append(page.page_number)
            page_items.append((page.page_number, markdown))

        if strict_page_files and missing_pages:
            sample = ", ".join(str(page) for page in missing_pages[:10])
            raise RuntimeError(f"Qwen OCR Markdown is incomplete; missing page files: {sample}.")

        blocks: list[Block] = []
        tables: list[TableModel] = []
        figures: list[FigureAsset] = []
        for page_number, markdown in page_items:
            page_blocks = self._blocks_from_markdown(
                markdown,
                page_number,
                len(blocks),
                surya_page=self._surya_page(surya_layout_manifest, page_number),
            )
            blocks.extend(page_blocks)
            visible_markdown = self._without_region_wrappers(markdown)
            page_tables, page_figures = self._extract_structures_from_markdown(
                visible_markdown,
                page_number,
            )
            self._link_page_tables(page_tables, page_blocks)
            tables.extend(page_tables)
            figures.extend(page_figures)

        with profiler.step("language_detection") if profiler is not None else nullcontext():
            language = self._detect_language(blocks)

        warnings = [
            "Parsed from Qwen full-page OCR Markdown. OCR text was preserved without header, footer, or page-number filtering."
        ]
        if missing_pages:
            warnings.append(
                "Some Qwen OCR page Markdown files were missing: "
                + ", ".join(str(page) for page in missing_pages[:20])
            )
        if any(REGION_PATTERN.search(markdown) for _page_number, markdown in page_items):
            warnings.append("Surya layout region tags were normalized into structured OCR blocks.")

        document = DocumentModel(
            metadata=DocumentMetadata(
                filename=inspection.filename,
                title=inspection.title,
                author=inspection.author,
                page_count=inspection.page_count,
                detected_language=language,
                translation={"ocr_markdown_preserved": True},
            ),
            pages=[
                PageMetadata(
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    has_embedded_text=page.has_embedded_text,
                    embedded_text_quality=page.embedded_text_quality,
                    extraction_mode=SourceType.OCR,
                )
                for page in inspection.pages
            ],
            blocks=blocks,
            tables=tables,
            figures=figures,
            warnings=warnings,
        )
        source_markdown = "\n\n".join(
            self._without_region_wrappers(markdown) for _page_number, markdown in page_items
        )
        return document, source_markdown

    def _blocks_from_markdown(
        self,
        markdown: str,
        page_number: int,
        start_order: int,
        *,
        surya_page: dict | None = None,
    ) -> list[Block]:
        region_blocks = self._blocks_from_surya_regions(
            markdown,
            page_number,
            start_order,
            surya_page=surya_page,
        )
        if region_blocks is not None:
            return region_blocks
        markdown = self._without_region_wrappers(markdown)

        blocks: list[Block] = []
        paragraph_lines: list[str] = []
        table_lines: list[str] = []

        def append(block_type: BlockType, text: str) -> None:
            if text.strip():
                blocks.append(self._block(page_number, start_order + len(blocks), block_type, text.strip()))

        def flush_paragraph() -> None:
            if paragraph_lines:
                append(BlockType.PARAGRAPH, " ".join(paragraph_lines))
                paragraph_lines.clear()

        def flush_table() -> None:
            if table_lines:
                append(BlockType.TABLE, "[TABLE]")
                table_lines.clear()

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                flush_table()
                continue
            if line.startswith("|") and line.endswith("|"):
                flush_paragraph()
                table_lines.append(line)
                continue

            flush_table()
            if match := re.match(r"^(#{1,6})\s+(.+)$", line):
                flush_paragraph()
                append(BlockType.HEADING, match.group(2))
            elif re.match(r"^[-*+]\s+", line):
                flush_paragraph()
                append(BlockType.LIST, re.sub(r"^[-*+]\s+", "", line))
            elif re.match(r"^(Table|Figure)\s+\d+", line, flags=re.IGNORECASE):
                flush_paragraph()
                append(BlockType.CAPTION, line)
            else:
                paragraph_lines.append(line)

        flush_paragraph()
        flush_table()
        return blocks

    def _blocks_from_surya_regions(
        self,
        markdown: str,
        page_number: int,
        start_order: int,
        *,
        surya_page: dict | None,
    ) -> list[Block] | None:
        matches = list(REGION_PATTERN.finditer(markdown))
        if not matches:
            return None
        text_outside_regions = REGION_PATTERN.sub("", markdown).replace("```xml", "").replace("```", "")
        if text_outside_regions.strip():
            return None

        blocks: list[Block] = []
        aligned_regions = self._aligned_layout_regions(matches, surya_page)
        aligned_region_counts = Counter(
            int(region["index"])
            for region, _mapping in aligned_regions
            if region is not None and str(region.get("index", "")).isdigit()
        )
        aligned_region_usage: Counter[int] = Counter()
        for match, aligned in zip(matches, aligned_regions):
            attributes = {
                item.group("name").lower(): item.group("value")
                for item in REGION_ATTRIBUTE_PATTERN.finditer(match.group("attributes"))
            }
            text = match.group("body").strip()
            if not text:
                continue
            output_region_type = attributes.get("type", "Text")
            try:
                output_region_index = int(attributes.get("index", ""))
            except ValueError:
                output_region_index = None
            layout_region, mapping = aligned
            region_type = str((layout_region or {}).get("label") or output_region_type)
            bbox = self._surya_bbox(layout_region)
            source_region_ids = list((layout_region or {}).get("source_region_ids", []))
            layout_index = (layout_region or {}).get("index")
            if (
                isinstance(layout_index, int)
                and aligned_region_counts[layout_index] > 1
                and len(source_region_ids) >= aligned_region_counts[layout_index]
            ):
                source_region_ids = [source_region_ids[aligned_region_usage[layout_index]]]
            if isinstance(layout_index, int):
                aligned_region_usage[layout_index] += 1
            blocks.append(
                self._block(
                    page_number,
                    start_order + len(blocks),
                    self._surya_block_type(region_type),
                    text,
                    metadata={
                        "parser": "qwen_surya_full_page_ocr",
                        "surya_region_index": (layout_region or {}).get("index", output_region_index),
                        "qwen_region_index": output_region_index,
                        "surya_region_type": region_type,
                        "qwen_region_type": output_region_type,
                        "surya_region_mapping": mapping,
                        "source_region_ids": source_region_ids,
                        "surya_bbox": list((layout_region or {}).get("bbox", [])),
                        "surya_page_width": (surya_page or {}).get("width"),
                        "surya_page_height": (surya_page or {}).get("height"),
                    },
                    bbox=bbox,
                )
            )
        return blocks

    def _aligned_layout_regions(
        self,
        matches: list[re.Match[str]],
        surya_page: dict | None,
    ) -> list[tuple[dict | None, str]]:
        regions = sorted(
            (surya_page or {}).get("reconciled_regions", []),
            key=lambda region: int(region.get("index", 0)),
        )
        if not regions:
            return [(None, "qwen_wrapper_only") for _match in matches]

        aligned: list[tuple[dict | None, str]] = []
        cursor = 0
        previous: dict | None = None
        for match in matches:
            attributes = {
                item.group("name").lower(): item.group("value")
                for item in REGION_ATTRIBUTE_PATTERN.finditer(match.group("attributes"))
            }
            output_type = attributes.get("type", "Text")
            text = match.group("body").strip()
            if cursor >= len(regions):
                if self._same_region_type(previous, output_type):
                    aligned.append((previous, "continued_previous_region"))
                else:
                    aligned.append((None, "qwen_wrapper_only"))
                continue

            current = regions[cursor]
            next_region = regions[cursor + 1] if cursor + 1 < len(regions) else None
            if self._should_skip_omitted_region(current, next_region, output_type, text):
                cursor += 1
                current = regions[cursor]
                mapping = "after_omitted_region"
            else:
                mapping = "reading_order"
            aligned.append((current, mapping))
            previous = current
            cursor += 1
        return aligned

    def _should_skip_omitted_region(
        self,
        current: dict,
        next_region: dict | None,
        output_type: str,
        text: str,
    ) -> bool:
        if next_region is None:
            return False
        output = self._normalized_region_type(output_type)
        current_type = self._normalized_region_type(str(current.get("label", "")))
        next_type = self._normalized_region_type(str(next_region.get("label", "")))
        if output == current_type or output != next_type:
            return False
        if output in {"pagefooter", "pageheader", "footnote"}:
            return True
        return output in {"sectionheader", "title"} and self._looks_like_short_heading(text)

    def _looks_like_short_heading(self, text: str) -> bool:
        stripped = text.strip()
        return len(stripped) <= 60 and not re.search(r"[,;–—]", stripped)

    def _surya_page(self, manifest: dict | None, page_number: int) -> dict | None:
        for page in (manifest or {}).get("pages", []):
            if int(page.get("page_index", 0)) == page_number:
                return page
        return None

    def _same_region_type(self, region: dict | None, output_region_type: str) -> bool:
        if region is None:
            return False
        return self._normalized_region_type(str(region.get("label", ""))) == self._normalized_region_type(
            output_region_type
        )

    def _normalized_region_type(self, region_type: str) -> str:
        return re.sub(r"[^a-z]", "", region_type.lower())

    def _surya_bbox(self, region: dict | None) -> BoundingBox | None:
        bbox = (region or {}).get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        return BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])

    def _surya_block_type(self, region_type: str) -> BlockType:
        normalized = re.sub(r"[^a-z]", "", region_type.lower())
        return {
            "caption": BlockType.CAPTION,
            "figure": BlockType.FIGURE,
            "footnote": BlockType.FOOTNOTE,
            "formula": BlockType.EQUATION,
            "list": BlockType.LIST,
            "listgroup": BlockType.LIST,
            "listitem": BlockType.LIST,
            "pagefooter": BlockType.FOOTER,
            "pageheader": BlockType.HEADER,
            "pagenumber": BlockType.PAGE_NUMBER,
            "picture": BlockType.FIGURE,
            "reference": BlockType.REFERENCE,
            "sectionheader": BlockType.HEADING,
            "table": BlockType.TABLE,
            "tableofcontents": BlockType.REFERENCE,
            "text": BlockType.PARAGRAPH,
            "title": BlockType.HEADING,
        }.get(normalized, BlockType.PARAGRAPH)

    def _without_region_wrappers(self, markdown: str) -> str:
        if "<region" not in markdown.lower() and "</region" not in markdown.lower():
            return markdown
        return re.sub(r"</?region\b[^>]*>", "", markdown, flags=re.IGNORECASE)

    def _block(
        self,
        page_number: int,
        order: int,
        block_type: BlockType,
        text: str,
        *,
        metadata: dict | None = None,
        bbox: BoundingBox | None = None,
    ) -> Block:
        if bbox is None and block_type == BlockType.TABLE:
            bbox = BoundingBox(x0=0, y0=0, x1=0, y1=0)
        return Block(
            id=f"qwen-p{page_number}-b{order}",
            page_number=page_number,
            block_type=block_type,
            text=text,
            bbox=bbox,
            reading_order_index=order,
            source_type=SourceType.OCR,
            metadata=metadata or {"parser": "qwen_full_page_ocr"},
        )

    def _extract_structures_from_markdown(
        self,
        markdown: str,
        page_number: int,
    ) -> tuple[list[TableModel], list[FigureAsset]]:
        tables: list[TableModel] = []
        figures: list[FigureAsset] = []
        table_lines: list[str] = []
        caption_text: str | None = None

        def flush_table() -> None:
            nonlocal caption_text
            if not table_lines:
                return
            rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table_lines]
            headers = rows[0] if rows else []
            body = rows[2:] if len(rows) > 2 and all(cell.strip("-: ") == "" for cell in rows[1]) else rows[1:]
            cells = [[TableModel.TableCell(text=cell) for cell in row] for row in body]
            tables.append(
                TableModel(
                    id=f"qwen-table-p{page_number}-{len(tables)}",
                    page_numbers=[page_number],
                    page=page_number,
                    headers=headers,
                    rows=body,
                    cells=cells,
                    caption=caption_text,
                    parse_mode="markdown_table",
                )
            )
            table_lines.clear()
            caption_text = None

        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if line.startswith("|") and line.endswith("|"):
                table_lines.append(line)
                continue
            flush_table()
            if re.match(r"^Figure\s+\d+", line, flags=re.IGNORECASE):
                figures.append(FigureAsset(id=f"qwen-fig-p{page_number}-{len(figures)}", page_number=page_number))
            if re.match(r"^Table\s+\d+", line, flags=re.IGNORECASE):
                caption_text = line
        flush_table()
        return tables, figures

    def _link_page_tables(
        self,
        tables: list[TableModel],
        blocks: list[Block],
    ) -> None:
        """Link the parallel Qwen table model to its canonical OCR block.

        Qwen table Markdown is parsed both as a translatable Block and as a
        structured TableModel. Persisting the relationship prevents the
        translated block and rendered table from drifting apart.
        """

        candidates: dict[tuple[tuple[str, ...], ...], list[Block]] = {}
        for block in blocks:
            rows = parse_table_rows(block.text)
            if not rows_have_consistent_shape(rows):
                continue
            signature = self._table_signature(
                [[cell.text for cell in row] for row in rows]
            )
            candidates.setdefault(signature, []).append(block)

        linked_blocks: list[Block] = []
        used_block_ids: set[str] = set()
        for table in tables:
            signature = self._table_signature([table.headers, *table.rows])
            matching = [
                block
                for block in candidates.get(signature, [])
                if block.id not in used_block_ids
            ]
            if len(matching) != 1:
                table.debug["link_status"] = "unlinked_ambiguous_or_missing_table_block"
                continue
            block = matching[0]
            used_block_ids.add(block.id)
            linked_blocks.append(block)
            block.block_type = BlockType.TABLE

            surya_type = self._normalized_region_type(
                str(block.metadata.get("surya_region_type", ""))
            )
            geometry_reliable = surya_type == "table" and block.bbox is not None
            block.metadata["table_geometry_reliable"] = geometry_reliable
            if not geometry_reliable:
                # Preserve the Surya evidence in metadata, but do not let a
                # text/table disagreement drive source-page redaction.
                block.bbox = None
            table.bbox = block.bbox.model_copy() if geometry_reliable else None
            table.debug.update(
                {
                    "source_block_id": block.id,
                    "source_region_ids": list(block.metadata.get("source_region_ids", [])),
                    "geometry_reliable": geometry_reliable,
                    "coordinate_space": {
                        "name": "surya_rendered_pixels" if geometry_reliable else "unresolved",
                        "width": block.metadata.get("surya_page_width"),
                        "height": block.metadata.get("surya_page_height"),
                    },
                }
            )

        table_blocks = sorted(linked_blocks, key=lambda item: item.reading_order_index)
        captions = [
            block
            for block in blocks
            if block.block_type == BlockType.CAPTION
            and (
                not block.metadata.get("qwen_region_type")
                or self._normalized_region_type(
                    str(block.metadata.get("qwen_region_type", ""))
                )
                == "caption"
            )
        ]
        table_by_block_id = {
            str(table.debug.get("source_block_id")): table
            for table in tables
            if table.debug.get("source_block_id")
        }
        for block in table_blocks:
            table = table_by_block_id[block.id]
            following = [
                caption
                for caption in captions
                if caption.reading_order_index > block.reading_order_index
            ]
            if not following:
                continue
            caption = min(following, key=lambda item: item.reading_order_index)
            next_table_order = min(
                (
                    item.reading_order_index
                    for item in table_blocks
                    if item.reading_order_index > block.reading_order_index
                ),
                default=None,
            )
            if next_table_order is not None and caption.reading_order_index > next_table_order:
                continue
            if caption.reading_order_index - block.reading_order_index > 3:
                continue
            if block.bbox is not None and caption.bbox is not None:
                vertical_gap = caption.bbox.y0 - block.bbox.y1
                block_height = max(1.0, block.bbox.y1 - block.bbox.y0)
                if vertical_gap < -4.0 or vertical_gap > max(120.0, block_height * 0.75):
                    continue
                horizontal_overlap = max(
                    0.0,
                    min(block.bbox.x1, caption.bbox.x1)
                    - max(block.bbox.x0, caption.bbox.x0),
                )
                overlap_ratio = horizontal_overlap / max(
                    1.0,
                    min(
                        block.bbox.x1 - block.bbox.x0,
                        caption.bbox.x1 - caption.bbox.x0,
                    ),
                )
                if overlap_ratio < 0.2:
                    continue
            table.caption_block_id = caption.id
            table.caption = caption.text.strip() or None
            table.debug["caption_association"] = "next_caption_in_reading_order"

    def _table_signature(
        self,
        rows: list[list[str]],
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(re.sub(r"\s+", " ", str(cell)).strip().casefold() for cell in row)
            for row in rows
        )

    def _detect_language(self, blocks: list[Block]) -> str | None:
        text = "\n".join(block.text for block in blocks if block.text.strip())[:4000].strip()
        if detect is None or len(text) < 40:
            return None
        try:
            return detect(text)
        except Exception:
            return None
