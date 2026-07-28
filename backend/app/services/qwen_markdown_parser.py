from __future__ import annotations

import html
import math
import re
import statistics
import unicodedata
from collections import Counter
from contextlib import nullcontext
from difflib import SequenceMatcher
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
            surya_page = self._surya_page(surya_layout_manifest, page_number)
            page_blocks = self._blocks_from_markdown(
                markdown,
                page_number,
                len(blocks),
                surya_page=surya_page,
            )
            blocks.extend(page_blocks)
            visible_markdown = self._without_region_wrappers(markdown)
            page_tables, page_figures = self._extract_structures_from_markdown(
                visible_markdown,
                page_number,
            )
            page_tables.extend(
                self._infer_plaintext_tables(
                    page_blocks,
                    page_number=page_number,
                    surya_page=surya_page,
                )
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
        repaired_table_count = sum(
            1 for table in tables if table.debug.get("topology_repair")
        )
        if repaired_table_count:
            warnings.append(
                f"Normalized {repaired_table_count} ragged OCR table(s) for readable output "
                "using assumed empty-cell placement; their geometry is not safe for "
                "source-page reconstruction."
            )

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
            layout_region_type = str((layout_region or {}).get("label") or "")
            region_type, type_guard = self._guard_structural_region_type(
                layout_region_type=layout_region_type,
                output_region_type=output_region_type,
                text=text,
                layout_region=layout_region,
                surya_page=surya_page,
            )
            bbox = self._surya_bbox(layout_region)
            source_region_ids = list((layout_region or {}).get("source_region_ids", []))
            if type_guard and type_guard.get("discard_layout_geometry"):
                type_guard["rejected_bbox"] = list((layout_region or {}).get("bbox", []))
                type_guard["rejected_source_region_ids"] = list(source_region_ids)
                bbox = None
                source_region_ids = []
            table_markup, table_repair = self._normalized_qwen_table_markup(text)
            if table_markup is not None:
                text = table_markup
            layout_index = (layout_region or {}).get("index")
            if (
                isinstance(layout_index, int)
                and aligned_region_counts[layout_index] > 1
                and len(source_region_ids) >= aligned_region_counts[layout_index]
            ):
                source_region_ids = [source_region_ids[aligned_region_usage[layout_index]]]
            if isinstance(layout_index, int):
                aligned_region_usage[layout_index] += 1
            block_metadata = {
                "parser": "qwen_surya_full_page_ocr",
                "surya_region_index": (layout_region or {}).get("index", output_region_index),
                "qwen_region_index": output_region_index,
                "surya_region_type": layout_region_type or output_region_type,
                "effective_region_type": region_type,
                "qwen_region_type": output_region_type,
                "surya_region_mapping": mapping,
                "source_region_ids": source_region_ids,
                "surya_bbox": list((layout_region or {}).get("bbox", [])),
                "surya_page_width": (surya_page or {}).get("width"),
                "surya_page_height": (surya_page or {}).get("height"),
                "surya_top_k": self._surya_top_k(surya_page, source_region_ids),
            }
            if (layout_region or {}).get("_embedded_alignment"):
                block_metadata["embedded_text_alignment"] = dict(
                    (layout_region or {})["_embedded_alignment"]
                )
            if type_guard is not None:
                block_metadata["structural_type_guard"] = type_guard
            if table_markup is not None:
                block_metadata["qwen_table_markup_normalized"] = True
                block_metadata["source_qwen_markdown_table"] = match.group("body").strip()
            if table_repair is not None:
                block_metadata["qwen_table_topology_repair"] = table_repair
            blocks.append(
                self._block(
                    page_number,
                    start_order + len(blocks),
                    BlockType.TABLE if table_markup is not None else self._surya_block_type(region_type),
                    text,
                    metadata=block_metadata,
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
            visual_match = None
            if cursor < len(regions):
                visual_match = self._nearby_visual_region_match(
                    regions,
                    cursor=cursor,
                    output_type=output_type,
                    text=text,
                )
            if visual_match is not None:
                visual_index = visual_match
                mapping = (
                    "reading_order"
                    if visual_index == cursor
                    else "after_omitted_text_to_visual"
                )
                current = regions[visual_index]
                aligned.append((current, mapping))
                previous = current
                cursor = visual_index + 1
                continue

            output_region_type = self._normalized_region_type(output_type)
            embedded_match = None
            if output_region_type not in {"figure", "image", "picture"} and not (
                self._contains_markdown_image(text)
            ):
                embedded_match = self._embedded_text_region_match(
                    text,
                    regions,
                    cursor=0,
                    surya_page=surya_page,
                    search_all=True,
                    minimum_expected_length=(
                        6
                        if output_region_type
                        in {"caption", "pagefooter", "pageheader", "sectionheader", "title"}
                        else 12
                    ),
                )
            if embedded_match is not None:
                start, end, evidence = embedded_match
                local_window = cursor <= start < min(len(regions), cursor + 5)
                if local_window or self._safe_global_embedded_alignment(evidence):
                    current = self._merge_layout_regions(regions[start : end + 1])
                    evidence["search_scope"] = "local" if local_window else "global"
                    current["_embedded_alignment"] = evidence
                    aligned.append(
                        (
                            current,
                            (
                                "embedded_text_geometry"
                                if local_window
                                else "embedded_text_geometry_global"
                            ),
                        )
                    )
                    previous = current
                    cursor = max(cursor, end + 1)
                    continue

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

    def _safe_global_embedded_alignment(self, evidence: dict) -> bool:
        if evidence.get("partial_containment"):
            return bool(
                float(evidence.get("length_coverage", 0.0)) >= 0.28
                and max(
                    float(evidence.get("prefix_score", 0.0)),
                    float(evidence.get("suffix_score", 0.0)),
                )
                >= 0.85
            )
        return bool(
            float(evidence.get("score", 0.0)) >= 0.92
            and float(evidence.get("length_coverage", 0.0)) >= 0.88
            and float(evidence.get("prefix_score", 0.0)) >= 0.70
            and float(evidence.get("suffix_score", 0.0)) >= 0.70
        )

    def _nearby_visual_region_match(
        self,
        regions: list[dict],
        *,
        cursor: int,
        output_type: str,
        text: str,
    ) -> int | None:
        """Anchor Qwen image wrappers to nearby Surya visual geometry.

        Qwen occasionally merges adjacent text columns into one wrapper, so its
        subsequent wrapper indexes can lag Surya's reading order by one or two
        text regions.  Treat a Figure/Picture wrapper (or explicit Markdown
        image) as strong evidence for the next nearby visual region, while only
        skipping ordinary text-like regions.  Crossing a caption, table, or
        other structural region would be ambiguous and is deliberately refused.
        """

        output = self._normalized_region_type(output_type)
        is_visual_wrapper = output in {"figure", "image", "picture"}
        has_markdown_image = self._contains_markdown_image(text)
        if not is_visual_wrapper and not has_markdown_image:
            return None
        if (
            is_visual_wrapper
            and not has_markdown_image
            and self._looks_like_natural_language_prose(text)
        ):
            # A structural label alone is not enough to skip source text. Qwen
            # can inherit a Figure/Picture label after its reading order drifts
            # from Surya, and treating a full prose paragraph as a visual would
            # exclude it from translation entirely.
            return None

        visual_types = {"figure", "picture"}
        skippable_types = {"list", "listgroup", "listitem", "text"}
        final_index = min(len(regions), cursor + 4)
        for index in range(cursor, final_index):
            region_type = self._normalized_region_type(str(regions[index].get("label", "")))
            if region_type in visual_types:
                return index
            if region_type not in skippable_types:
                return None
        return None

    def _contains_markdown_image(self, text: str) -> bool:
        return bool(re.search(r"!\[[^\]]*\]\(\s*<?[^)>\s]+>?", text, flags=re.IGNORECASE))

    def _embedded_text_region_match(
        self,
        text: str,
        regions: list[dict],
        *,
        cursor: int,
        surya_page: dict | None,
        search_all: bool = False,
        excluded_region_indexes: set[int] | None = None,
        minimum_expected_length: int = 12,
    ) -> tuple[int, int, dict] | None:
        geometry = (surya_page or {}).get("embedded_text_geometry", {})
        if not geometry.get("available"):
            return None
        expected = self._comparison_text(text)
        if len(expected) < minimum_expected_length:
            return None

        candidates: list[
            tuple[float, int, int, int, float, float, float, bool]
        ] = []
        final_start = len(regions) if search_all else min(len(regions), cursor + 5)
        for start in range(cursor, final_start):
            final_end = min(len(regions), start + 6)
            for end in range(start, final_end):
                candidate_indexes = {
                    int(region.get("index", 0))
                    for region in regions[start : end + 1]
                }
                if excluded_region_indexes and candidate_indexes & excluded_region_indexes:
                    continue
                actual_text = self._embedded_text_for_regions(
                    regions[start : end + 1],
                    surya_page,
                )
                actual = self._comparison_text(actual_text)
                if not actual:
                    continue
                # Qwen may split one Surya region into multiple wrappers.  In
                # that case the wrapper text is a proper substring of the
                # region text.  The reverse direction is not safe: a short,
                # generic region can occur inside an unrelated paragraph.
                partial_containment = expected in actual and expected != actual
                similarity = (
                    1.0
                    if partial_containment
                    else SequenceMatcher(None, expected, actual).ratio()
                )
                coverage = min(len(expected), len(actual)) / max(len(expected), len(actual))
                edge = min(32, len(expected), len(actual))
                prefix = SequenceMatcher(None, expected[:edge], actual[:edge]).ratio()
                suffix = SequenceMatcher(None, expected[-edge:], actual[-edge:]).ratio()
                candidates.append(
                    (
                        similarity,
                        abs(len(actual) - len(expected)),
                        start,
                        end,
                        coverage,
                        prefix,
                        suffix,
                        partial_containment,
                    )
                )
        if not candidates:
            return None
        complete_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] >= 0.88 and candidate[4] >= 0.985
        ]
        ranked = complete_candidates or candidates
        ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        best = ranked[0]
        competing = max(
            (
                candidate[0]
                for candidate in candidates
                if candidate is not best
                if candidate[2] > best[3] or candidate[3] < best[2]
                if not candidate[7] or candidate[4] >= 0.28
            ),
            default=0.0,
        )
        safe_partial_containment = bool(
            best[7]
            and best[4] >= 0.28
            and max(best[5], best[6]) >= 0.85
            and (competing == 0.0 or best[0] - competing >= 0.025)
        )
        if not safe_partial_containment and (
            best[0] < 0.84
            or best[4] < 0.78
            or best[5] < 0.55
            or best[6] < 0.50
            or competing and best[0] - competing < 0.025
        ):
            return None
        return (
            best[2],
            best[3],
            {
                "score": round(best[0], 6),
                "length_coverage": round(best[4], 6),
                "prefix_score": round(best[5], 6),
                "suffix_score": round(best[6], 6),
                "competing_score": round(competing, 6),
                "partial_containment": safe_partial_containment,
                "matched_region_indexes": [
                    int(region.get("index", 0))
                    for region in regions[best[2] : best[3] + 1]
                ],
            },
        )

    def _merge_layout_regions(self, regions: list[dict]) -> dict:
        if not regions:
            return {}
        merged = dict(regions[0])
        bboxes = [region.get("bbox") for region in regions]
        bboxes = [bbox for bbox in bboxes if isinstance(bbox, list) and len(bbox) == 4]
        if bboxes:
            merged["bbox"] = [
                min(float(bbox[0]) for bbox in bboxes),
                min(float(bbox[1]) for bbox in bboxes),
                max(float(bbox[2]) for bbox in bboxes),
                max(float(bbox[3]) for bbox in bboxes),
            ]
        merged["source_region_ids"] = list(
            dict.fromkeys(
                str(region_id)
                for region in regions
                for region_id in region.get("source_region_ids", [])
            )
        )
        merged["merged_reconciled_region_indexes"] = [
            int(region.get("index", 0)) for region in regions
        ]
        return merged

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

    def _guard_structural_region_type(
        self,
        *,
        layout_region_type: str,
        output_region_type: str,
        text: str,
        layout_region: dict | None,
        surya_page: dict | None,
    ) -> tuple[str, dict | None]:
        """Refuse implausible visual, running-matter, and heading classifications.

        Qwen and Surya can disagree about reading order on pages where a table
        interrupts one column.  In that case an ordinary paragraph can be
        paired with a tiny page-number or footer region.  Position labels are
        useful evidence, but they are not enough to turn long prose into a
        footer or heading.
        """

        layout_type = self._normalized_region_type(layout_region_type)
        output_type = self._normalized_region_type(output_region_type)
        effective = layout_region_type or output_region_type or "Text"
        marginal_types = {"pagefooter", "pagenumber"}
        heading_types = {"sectionheader", "title"}
        layout_visual_types = {"figure", "picture", "table"}
        qwen_visual_types = {"figure", "image", "picture"}
        natural_language_prose = self._looks_like_natural_language_prose(text)
        long_prose = self._looks_like_long_prose(text)

        if (
            natural_language_prose
            and output_type == "text"
            and layout_type in layout_visual_types
        ):
            return output_region_type or "Text", {
                "status": "downgraded",
                "from": layout_region_type,
                "to": output_region_type or "Text",
                "reason": "natural_language_prose_conflicts_with_layout_visual_label",
                "layout_region_type": layout_region_type,
                "qwen_region_type": output_region_type,
                "discard_layout_geometry": True,
            }

        if (
            natural_language_prose
            and output_type in qwen_visual_types
            and layout_type == "text"
        ):
            return layout_region_type or "Text", {
                "status": "downgraded",
                "from": output_region_type,
                "to": layout_region_type or "Text",
                "reason": "natural_language_prose_conflicts_with_qwen_visual_label",
                "layout_region_type": layout_region_type,
                "qwen_region_type": output_region_type,
                "discard_layout_geometry": True,
            }

        if layout_type in marginal_types or output_type in marginal_types:
            # A Qwen footnote/list label carries semantic information which a
            # position-only footer label does not.  Preserve it even near the
            # page edge so unique notes and final references remain content.
            if layout_type in marginal_types and output_type in {
                "footnote",
                "list",
                "listgroup",
                "listitem",
            }:
                return output_region_type, {
                    "status": "downgraded",
                    "from": effective,
                    "to": output_region_type,
                    "reason": "semantic_qwen_label_overrides_marginal_layout_label",
                }

            marginal_type = layout_type if layout_type in marginal_types else output_type
            credible, evidence = self._credible_marginal_region(
                marginal_type,
                text=text,
                layout_region=layout_region,
                surya_page=surya_page,
            )
            if credible and not long_prose:
                accepted_type = (
                    layout_region_type
                    if layout_type in marginal_types
                    else output_region_type
                )
                return accepted_type, {
                    "status": "accepted",
                    "type": accepted_type,
                    "evidence": evidence,
                }

            fallback = output_region_type if output_type not in marginal_types else layout_region_type
            fallback_type = self._normalized_region_type(fallback)
            if not fallback or fallback_type in marginal_types:
                fallback = "Text"
            return fallback, {
                "status": "downgraded",
                "from": effective,
                "to": fallback,
                "reason": (
                    "long_prose_is_not_running_matter"
                    if long_prose
                    else "running_matter_lacks_margin_and_short_text_evidence"
                ),
                "evidence": evidence,
                "discard_layout_geometry": (
                    layout_type in marginal_types
                    and self._normalized_region_type(fallback) not in marginal_types
                ),
            }

        if long_prose and (layout_type in heading_types or output_type in heading_types):
            if layout_type not in heading_types and layout_region_type:
                fallback = layout_region_type
            elif output_type not in heading_types and output_region_type:
                fallback = output_region_type
            else:
                fallback = "Text"
            return fallback, {
                "status": "downgraded",
                "from": effective,
                "to": fallback,
                "reason": "long_prose_is_not_a_heading",
                "discard_layout_geometry": layout_type in heading_types,
            }

        return effective, None

    def _credible_marginal_region(
        self,
        marginal_type: str,
        *,
        text: str,
        layout_region: dict | None,
        surya_page: dict | None,
    ) -> tuple[bool, dict]:
        compact = re.sub(r"\s+", " ", text).strip()
        words = re.findall(r"[^\W\d_]+", compact, flags=re.UNICODE)
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", compact))
        short = (
            len(compact) <= 160
            and len(words) <= 24
            and sentence_count <= 1
            and len([line for line in text.splitlines() if line.strip()]) <= 3
        )
        if marginal_type == "pagenumber":
            short = short and len(compact) <= 20 and len(words) <= 3

        bbox = (layout_region or {}).get("bbox")
        page_height = (surya_page or {}).get("height")
        at_margin = False
        if isinstance(bbox, list) and len(bbox) == 4 and page_height:
            try:
                height = float(page_height)
                at_margin = float(bbox[1]) >= height * 0.76 or float(bbox[3]) >= height * 0.88
            except (TypeError, ValueError, ZeroDivisionError):
                at_margin = False
        elif layout_region is None:
            # Without a layout manifest the short Qwen wrapper is the only
            # available structural evidence.  This retains page numbers and
            # concise journal footers while still rejecting prose.
            at_margin = True

        evidence = {
            "at_page_margin": at_margin,
            "short_text": short,
            "character_count": len(compact),
            "word_count": len(words),
            "sentence_count": sentence_count,
        }
        return at_margin and short, evidence

    def _looks_like_long_prose(self, text: str) -> bool:
        compact = re.sub(r"\s+", " ", text).strip()
        words = re.findall(r"[^\W\d_]+", compact, flags=re.UNICODE)
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", compact))
        return bool(
            (len(compact) >= 180 and len(words) >= 24)
            or (len(compact) >= 100 and len(words) >= 14 and sentence_count >= 2)
        )

    def _looks_like_natural_language_prose(self, text: str) -> bool:
        """Distinguish prose from explicit image or table markup conservatively."""

        if self._contains_markdown_image(text):
            return False
        lowered = text.casefold()
        if "<table" in lowered or "</table" in lowered:
            return False
        markdown_table_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]
        if len(markdown_table_lines) >= 2:
            return False

        compact = re.sub(r"\s+", " ", text).strip()
        words = re.findall(r"[^\W\d_]+", compact, flags=re.UNICODE)
        return self._looks_like_long_prose(text) or (
            len(compact) >= 60 and len(words) >= 8
        )

    def _surya_top_k(self, surya_page: dict | None, source_region_ids: list[str]) -> dict[str, float]:
        source_ids = set(source_region_ids)
        scores: dict[str, float] = {}
        for region in (surya_page or {}).get("regions", []):
            if str(region.get("id", "")) not in source_ids:
                continue
            for label, value in (region.get("top_k") or {}).items():
                try:
                    score = float(value)
                except (TypeError, ValueError):
                    continue
                scores[str(label)] = max(scores.get(str(label), 0.0), score)
        return scores

    def _comparison_text(self, text: str) -> str:
        normalized: list[str] = []
        for character in str(text or "").casefold():
            # Fold accents on Latin text so OCR variants such as "e" and "é"
            # can align, without dropping letters from non-Latin scripts.
            if "LATIN" in unicodedata.name(character, ""):
                normalized.extend(
                    component
                    for component in unicodedata.normalize("NFKD", character)
                    if component.isalnum()
                )
            elif character.isalnum():
                normalized.append(character)
        return "".join(normalized)

    def _embedded_text_for_regions(
        self,
        regions: list[dict],
        surya_page: dict | None,
    ) -> str:
        bboxes = [region.get("bbox") for region in regions]
        bboxes = [bbox for bbox in bboxes if isinstance(bbox, list) and len(bbox) == 4]
        if not bboxes:
            return ""
        words: list[dict] = []
        seen: set[tuple] = set()
        for bbox in bboxes:
            for word in self._embedded_words_in_bbox(surya_page, bbox):
                word_bbox = tuple(word.get("bbox", []))
                key = (
                    str(word.get("text", "")),
                    word_bbox,
                    word.get("block"),
                    word.get("line"),
                    word.get("word"),
                )
                if key in seen:
                    continue
                seen.add(key)
                words.append(word)
        return "\n".join(
            " ".join(str(word["text"]) for word in line)
            for line in self._geometry_lines(words)
        )

    def _embedded_words_in_bbox(
        self,
        surya_page: dict | None,
        bbox: list[float],
    ) -> list[dict]:
        words: list[dict] = []
        for word in (surya_page or {}).get("embedded_text_geometry", {}).get("words", []):
            word_bbox = word.get("bbox")
            if not isinstance(word_bbox, list) or len(word_bbox) != 4:
                continue
            center_x = (float(word_bbox[0]) + float(word_bbox[2])) / 2
            center_y = (float(word_bbox[1]) + float(word_bbox[3])) / 2
            if bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]:
                words.append(word)
        return words

    def _geometry_lines(self, words: list[dict]) -> list[list[dict]]:
        valid = [
            word
            for word in words
            if isinstance(word.get("bbox"), list)
            and len(word["bbox"]) == 4
            and str(word.get("text", "")).strip()
        ]
        if not valid:
            return []
        heights = sorted(float(word["bbox"][3]) - float(word["bbox"][1]) for word in valid)
        median_height = heights[len(heights) // 2]
        tolerance = min(14.0, max(2.0, median_height * 0.30))
        lines: list[list[dict]] = []
        for word in sorted(
            valid,
            key=lambda item: (
                (float(item["bbox"][1]) + float(item["bbox"][3])) / 2,
                float(item["bbox"][0]),
            ),
        ):
            center_y = (float(word["bbox"][1]) + float(word["bbox"][3])) / 2
            if not lines:
                lines.append([word])
                continue
            previous_center = statistics.mean(
                (float(item["bbox"][1]) + float(item["bbox"][3])) / 2
                for item in lines[-1]
            )
            if abs(center_y - previous_center) > tolerance:
                lines.append([word])
            else:
                lines[-1].append(word)
        return [
            sorted(line, key=lambda item: float(item["bbox"][0]))
            for line in lines
        ]

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

    def _normalized_qwen_table_markup(self, text: str) -> tuple[str | None, dict | None]:
        table_lines = [
            line.strip()
            for line in str(text or "").splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]
        nonempty_lines = [line for line in str(text or "").splitlines() if line.strip()]
        if len(table_lines) < 2 or len(table_lines) != len(nonempty_lines):
            return None, None
        parsed = self._normalized_markdown_table_rows(table_lines)
        if parsed is None:
            return None, None
        rows, repair = parsed
        return self._table_rows_as_html(rows), repair

    def _normalized_markdown_table_rows(
        self,
        table_lines: list[str],
    ) -> tuple[list[list[str]], dict | None] | None:
        raw_rows = [self._markdown_row_cells(line) for line in table_lines]
        separator_indexes = [
            index
            for index, row in enumerate(raw_rows)
            if row
            and all(
                re.fullmatch(r":?-{1,}:?", cell.strip()) is not None
                for cell in row
            )
        ]
        data_rows = [
            row for index, row in enumerate(raw_rows) if index not in separator_indexes
        ]
        if len(data_rows) < 2 or not data_rows[0]:
            return None
        widths = [len(row) for row in data_rows]
        if len(set(widths)) == 1:
            return data_rows, None

        canonical_width = max(widths)
        if (
            canonical_width > 24
            or canonical_width - min(widths) > 3
            or widths.count(canonical_width) < 2
            or len(data_rows) < 4
        ):
            return None

        repaired: list[list[str]] = []
        operations: list[dict] = []
        for row_index, row in enumerate(data_rows):
            missing = canonical_width - len(row)
            if missing <= 0:
                repaired.append(list(row))
                continue
            updated = list(row)
            insert_at = len(updated)
            # A short first row with an internal blank run is normally a
            # hierarchical header. Preserve the final group's width by adding
            # missing empty placeholders before that group, not after it.
            if row_index == 0:
                nonempty = [index for index, cell in enumerate(updated) if cell.strip()]
                if len(nonempty) >= 3 and nonempty[-1] < len(updated) - 1:
                    previous = nonempty[-2]
                    if nonempty[-1] - previous >= 2:
                        insert_at = nonempty[-1]
            updated[insert_at:insert_at] = [""] * missing
            repaired.append(updated)
            operations.append(
                {
                    "row_index": row_index,
                    "original_width": len(row),
                    "inserted_empty_cells": missing,
                    "insert_at": insert_at,
                    "placement_assumed": True,
                    "placement_basis": (
                        "hierarchical_header_blank_run_heuristic"
                        if insert_at != len(row)
                        else "trailing_padding_default"
                    ),
                }
            )
        if not all(len(row) == canonical_width for row in repaired):
            return None
        return (
            repaired,
            {
                "strategy": "empty_cell_padding_only",
                "canonical_width": canonical_width,
                "original_widths": widths,
                "separator_widths": [len(raw_rows[index]) for index in separator_indexes],
                "operations": operations,
                "placement_assumed": True,
                "geometry_reliable": False,
                "cell_geometry_reliable": False,
                "reconstruction_scope": "readable_reflow_only",
                "assumption": (
                    "OCR omitted one or more cells, but their original column positions "
                    "cannot be proven from ragged Markdown alone."
                ),
            },
        )

    def _markdown_row_cells(self, line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        cells: list[str] = []
        current: list[str] = []
        index = 0
        while index < len(stripped):
            character = stripped[index]
            if character == "\\" and index + 1 < len(stripped) and stripped[index + 1] == "|":
                current.append("|")
                index += 2
                continue
            if character == "|":
                cells.append("".join(current).strip())
                current = []
            else:
                current.append(character)
            index += 1
        cells.append("".join(current).strip())
        return cells

    def _table_rows_as_html(self, rows: list[list[str]]) -> str:
        if not rows:
            return ""
        lines = ["<table>", "<thead>", "<tr>"]
        lines.extend(f"<th>{html.escape(cell)}</th>" for cell in rows[0])
        lines.extend(["</tr>", "</thead>", "<tbody>"])
        for row in rows[1:]:
            lines.append("<tr>")
            lines.extend(f"<td>{html.escape(cell)}</td>" for cell in row)
            lines.append("</tr>")
        lines.extend(["</tbody>", "</table>"])
        return "\n".join(lines)

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
            parsed = self._normalized_markdown_table_rows(table_lines)
            table_lines.clear()
            if parsed is None:
                caption_text = None
                return
            rows, repair = parsed
            headers = rows[0] if rows else []
            body = rows[1:]
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
                    parse_mode=(
                        "qwen_markdown_table_repaired" if repair else "markdown_table"
                    ),
                    debug={"topology_repair": repair} if repair else {},
                )
            )
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

    def _infer_plaintext_tables(
        self,
        blocks: list[Block],
        *,
        page_number: int,
        surya_page: dict | None,
    ) -> list[TableModel]:
        inferred: list[TableModel] = []
        for block in blocks:
            if block.block_type == BlockType.TABLE or block.bbox is None:
                continue
            table_score = float((block.metadata.get("surya_top_k") or {}).get("Table", 0.0) or 0.0)
            if table_score < 0.20:
                continue
            parsed = self._infer_two_column_rows(block, surya_page)
            if parsed is None:
                continue
            rows, header_cells, body_cells, evidence = parsed
            block.text = self._table_rows_as_html(rows)
            block.block_type = BlockType.TABLE
            block.metadata.update(
                {
                    "inferred_table_geometry_reliable": True,
                    "inferred_table_evidence": evidence,
                    "source_qwen_plaintext_table": evidence["source_text"],
                }
            )
            table = TableModel(
                id=f"qwen-inferred-table-p{page_number}-{len(inferred)}",
                page_numbers=[page_number],
                page=page_number,
                bbox=block.bbox.model_copy(),
                headers=rows[0],
                header_cells=header_cells,
                rows=rows[1:],
                cells=body_cells,
                parse_mode="hidden_ocr_geometry_inferred",
                debug={
                    "inference": evidence,
                    "geometry_reliable": True,
                    "coordinate_space": {
                        "name": "surya_rendered_pixels",
                        "width": block.metadata.get("surya_page_width"),
                        "height": block.metadata.get("surya_page_height"),
                    },
                    "cell_coordinate_space": {
                        "name": "surya_rendered_pixels",
                        "width": block.metadata.get("surya_page_width"),
                        "height": block.metadata.get("surya_page_height"),
                    },
                },
            )
            inferred.append(table)

        for table in inferred:
            table_block = next(
                (
                    block
                    for block in blocks
                    if block.block_type == BlockType.TABLE
                    and block.bbox is not None
                    and table.bbox is not None
                    and block.bbox == table.bbox
                ),
                None,
            )
            if table_block is None:
                continue
            following = [
                block
                for block in blocks
                if 0 < block.reading_order_index - table_block.reading_order_index <= 2
                and block.bbox is not None
                and len(block.text.strip()) <= 80
                and float((block.metadata.get("surya_top_k") or {}).get("Caption", 0.0) or 0.0)
                >= 0.15
            ]
            if not following:
                continue
            caption = min(following, key=lambda item: item.reading_order_index)
            vertical_gap = caption.bbox.y0 - table_block.bbox.y1
            horizontal_gap = abs(caption.bbox.x0 - table_block.bbox.x0)
            if vertical_gap < -4 or vertical_gap > max(120.0, table_block.bbox.y1 - table_block.bbox.y0):
                continue
            if horizontal_gap > max(120.0, (table_block.bbox.x1 - table_block.bbox.x0) * 0.35):
                continue
            caption.block_type = BlockType.CAPTION
            caption.metadata["inferred_caption_for_table"] = True
        return inferred

    def _infer_two_column_rows(
        self,
        block: Block,
        surya_page: dict | None,
    ) -> tuple[
        list[list[str]],
        list[TableModel.TableCell],
        list[list[TableModel.TableCell]],
        dict,
    ] | None:
        if block.bbox is None:
            return None
        qwen_lines = [line.strip() for line in block.text.splitlines() if line.strip()]
        if len(qwen_lines) < 4:
            return None
        bbox = [block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1]
        geometry_lines = self._geometry_lines(self._embedded_words_in_bbox(surya_page, bbox))
        if len(geometry_lines) != len(qwen_lines):
            return None

        block_width = max(1.0, block.bbox.x1 - block.bbox.x0)
        minimum_gap = max(28.0, block_width * 0.14)
        gap_intervals: list[list[tuple[float, float]]] = []
        for line in geometry_lines:
            intervals: list[tuple[float, float]] = []
            occupied_right = float(line[0]["bbox"][2])
            for word in line[1:]:
                left = float(word["bbox"][0])
                if left - occupied_right >= minimum_gap:
                    intervals.append((occupied_right, left))
                occupied_right = max(occupied_right, float(word["bbox"][2]))
            gap_intervals.append(intervals)

        candidates = [
            (left + right) / 2
            for intervals in gap_intervals
            for left, right in intervals
            if block.bbox.x0 + block_width * 0.2
            <= (left + right) / 2
            <= block.bbox.x1 - block_width * 0.2
        ]
        if not candidates:
            return None
        scored = sorted(
            (
                (
                    sum(
                        any(left <= candidate <= right for left, right in intervals)
                        for intervals in gap_intervals
                    ),
                    candidate,
                )
                for candidate in candidates
            ),
            key=lambda item: (-item[0], abs(item[1] - (block.bbox.x0 + block.bbox.x1) / 2)),
        )
        support, split_x = scored[0]
        if support < max(3, math.ceil(len(geometry_lines) * 0.5)):
            return None

        physical_cells: list[tuple[str, str, list[dict], list[dict]]] = []
        for line in geometry_lines:
            left_words = [
                word
                for word in line
                if (float(word["bbox"][0]) + float(word["bbox"][2])) / 2 < split_x
            ]
            right_words = [word for word in line if word not in left_words]
            physical_cells.append(
                (
                    " ".join(str(word["text"]) for word in left_words),
                    " ".join(str(word["text"]) for word in right_words),
                    left_words,
                    right_words,
                )
            )
        if sum(bool(left and right) for left, right, _lw, _rw in physical_cells) < 3:
            return None
        nonempty_token_counts = [
            len(text.split())
            for left, right, _lw, _rw in physical_cells
            for text in (left, right)
            if text.strip()
        ]
        if (
            not nonempty_token_counts
            or statistics.median(nonempty_token_counts) > 4
            or sum(count >= 9 for count in nonempty_token_counts)
            > max(1, len(nonempty_token_counts) // 5)
        ):
            return None

        rows: list[list[str]] = []
        cell_geometries: list[list[TableModel.TableCell]] = []
        similarities: list[float] = []
        row_boundaries = self._row_boundaries(geometry_lines, block.bbox)
        for row_index, (qwen_line, physical, vertical) in enumerate(
            zip(qwen_lines, physical_cells, row_boundaries, strict=True)
        ):
            split = self._split_qwen_table_line(qwen_line, physical[0], physical[1])
            if split is None:
                return None
            left_text, right_text, similarity = split
            if similarity < 0.76:
                return None
            similarities.append(similarity)
            rows.append([left_text, right_text])
            cell_geometries.append(
                [
                    self._inferred_table_cell(
                        left_text,
                        row_index=row_index,
                        column_index=0,
                        bbox=BoundingBox(
                            x0=block.bbox.x0,
                            y0=vertical[0],
                            x1=split_x,
                            y1=vertical[1],
                        ),
                    ),
                    self._inferred_table_cell(
                        right_text,
                        row_index=row_index,
                        column_index=1,
                        bbox=BoundingBox(
                            x0=split_x,
                            y0=vertical[0],
                            x1=block.bbox.x1,
                            y1=vertical[1],
                        ),
                    ),
                ]
            )
        mean_similarity = statistics.mean(similarities)
        if mean_similarity < 0.88:
            return None
        evidence = {
            "strategy": "surya_table_probability_plus_hidden_ocr_row_geometry",
            "source_text": block.text,
            "surya_table_score": round(
                float((block.metadata.get("surya_top_k") or {}).get("Table", 0.0) or 0.0),
                6,
            ),
            "row_count": len(rows),
            "column_count": 2,
            "gutter_x": round(split_x, 4),
            "gutter_supporting_rows": support,
            "mean_qwen_hidden_ocr_similarity": round(mean_similarity, 6),
            "hidden_ocr_usage": "geometry_and_cell_boundary_alignment_only",
        }
        return rows, cell_geometries[0], cell_geometries[1:], evidence

    def _split_qwen_table_line(
        self,
        qwen_line: str,
        physical_left: str,
        physical_right: str,
    ) -> tuple[str, str, float] | None:
        tokens = qwen_line.split()
        if not tokens:
            return None
        candidates: list[tuple[float, int, str, str]] = []
        for split in range(len(tokens) + 1):
            left = " ".join(tokens[:split])
            right = " ".join(tokens[split:])
            left_score = self._cell_similarity(physical_left, left)
            right_score = self._cell_similarity(physical_right, right)
            if not physical_left.strip() and left.strip():
                continue
            if not physical_right.strip() and right.strip():
                continue
            candidates.append(((left_score + right_score) / 2, split, left, right))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        best = candidates[0]
        return best[2], best[3], best[0]

    def _cell_similarity(self, expected: str, actual: str) -> float:
        expected_normalized = self._comparison_text(expected)
        actual_normalized = self._comparison_text(actual)
        if not expected_normalized and not actual_normalized:
            return 1.0
        if not expected_normalized or not actual_normalized:
            return 0.0
        return SequenceMatcher(None, expected_normalized, actual_normalized).ratio()

    def _row_boundaries(
        self,
        lines: list[list[dict]],
        bbox: BoundingBox,
    ) -> list[tuple[float, float]]:
        centers = [
            statistics.mean(
                (float(word["bbox"][1]) + float(word["bbox"][3])) / 2
                for word in line
            )
            for line in lines
        ]
        boundaries = [bbox.y0]
        boundaries.extend((first + second) / 2 for first, second in zip(centers, centers[1:]))
        boundaries.append(bbox.y1)
        return list(zip(boundaries, boundaries[1:]))

    def _inferred_table_cell(
        self,
        text: str,
        *,
        row_index: int,
        column_index: int,
        bbox: BoundingBox,
    ) -> TableModel.TableCell:
        return TableModel.TableCell(
            text=text,
            row_index=row_index,
            column_index=column_index,
            bbox=bbox,
            extraction_metadata={
                "coordinate_space": "surya_rendered_pixels",
                "inference": "hidden_ocr_row_geometry",
            },
        )

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
            topology_repair = block.metadata.get("qwen_table_topology_repair")
            has_assumed_topology = bool(
                isinstance(topology_repair, dict)
                and topology_repair.get("placement_assumed")
            )
            geometry_validation = self._table_geometry_validation(block)
            geometry_reliable = (
                surya_type == "table"
                or bool(block.metadata.get("inferred_table_geometry_reliable"))
            ) and geometry_validation["status"] == "valid" and not has_assumed_topology
            block.metadata.update(
                {
                    "table_geometry_reliable": geometry_reliable,
                    "table_cell_geometry_reliable": geometry_reliable,
                    "table_reconstruction_scope": (
                        "structured_and_original_layout"
                        if geometry_reliable
                        else "readable_reflow_only"
                    ),
                    "table_geometry_validation": geometry_validation,
                }
            )
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
                    "cell_geometry_reliable": (
                        False if has_assumed_topology else geometry_reliable
                    ),
                    "reconstruction_scope": (
                        "structured_and_original_layout"
                        if geometry_reliable
                        else "readable_reflow_only"
                    ),
                    "geometry_validation": geometry_validation,
                    "coordinate_space": {
                        "name": "surya_rendered_pixels" if geometry_reliable else "unresolved",
                        "width": block.metadata.get("surya_page_width"),
                        "height": block.metadata.get("surya_page_height"),
                    },
                    "render_from_block_text": True,
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
                or bool(block.metadata.get("inferred_caption_for_table"))
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

    def _table_geometry_validation(self, block: Block) -> dict:
        bbox = block.bbox
        page_width = block.metadata.get("surya_page_width")
        page_height = block.metadata.get("surya_page_height")
        evidence = {
            "status": "invalid",
            "reason": "missing_bbox",
            "bbox": bbox.model_dump() if bbox is not None else None,
            "page_width": page_width,
            "page_height": page_height,
            "coordinate_space": "surya_rendered_pixels",
        }
        if bbox is None:
            return evidence

        values = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        if not all(math.isfinite(float(value)) for value in values):
            evidence["reason"] = "non_finite_bbox"
            return evidence
        if bbox.x1 <= bbox.x0 or bbox.y1 <= bbox.y0:
            evidence["reason"] = "non_positive_bbox_area"
            return evidence
        try:
            width = float(page_width)
            height = float(page_height)
        except (TypeError, ValueError):
            evidence["reason"] = "missing_or_invalid_page_dimensions"
            return evidence
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            evidence["reason"] = "missing_or_invalid_page_dimensions"
            return evidence
        if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > width or bbox.y1 > height:
            evidence["reason"] = "bbox_outside_page"
            return evidence

        evidence.update({"status": "valid", "reason": None})
        return evidence

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
