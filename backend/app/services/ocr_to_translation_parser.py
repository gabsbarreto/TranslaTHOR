from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from app.models.schema import Block, BlockType, DocumentModel, TranslationChunk
from app.services.cross_page_continuation import (
    CONTINUATION_CONFIDENCE,
    CONTINUATION_DECISION,
    CONTINUATION_EVIDENCE,
    CONTINUATION_GROUP_ID,
    CONTINUATION_INDEX,
    CONTINUATION_INTERVENING_IDS,
    CrossPageContinuationResolver,
)

KEYWORD_HEADINGS = {
    "keywords",
    "keyword",
    "mots-clés",
    "mots clés",
    "palabras clave",
    "palavras-chave",
    "palavras chave",
    "schlagwörter",
}
CONTINUATION_WORDS = {
    "and",
    "both",
    "but",
    "oder",
    "or",
    "sowie",
    "sowohl",
    "und",
}
FOOTNOTE_MARKER_PATTERN = re.compile(r"^(?:\[\d+\]|\d+[.)]?|[¹²³⁴⁵⁶⁷⁸⁹⁰])\s+")
PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s*)?\d+$", flags=re.IGNORECASE)
CONTROL_CHARACTER_PATTERN = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]")
LINE_BREAK_HYPHEN_PATTERN = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
SOURCE_REGION_INDEX_PATTERN = re.compile(r"-r(?P<index>\d+)$")


@dataclass
class OCRLogicalParseResult:
    document: DocumentModel
    excluded_regions: list[dict[str, Any]]
    warnings: list[str]


class OCRToTranslationParser:
    """Convert Surya OCR layout blocks into semantic translation chunks."""

    def prepare(
        self, document: DocumentModel, *, document_id: str | None = None
    ) -> OCRLogicalParseResult:
        prepared = document.model_copy(deep=True)
        prepared.blocks = [self._clean_block(block) for block in prepared.blocks]
        repeated_margin_text = self._repeated_margin_text(prepared.blocks)
        excluded_regions: list[dict[str, Any]] = []
        exclusion_reasons: dict[str, str] = {}
        for block in prepared.blocks:
            exclusion_reason = self._exclusion_reason(block, repeated_margin_text)
            if not exclusion_reason:
                continue
            block.metadata["excluded_from_translation"] = True
            block.metadata["translation_exclusion_reason"] = exclusion_reason
            exclusion_reasons[block.id] = exclusion_reason
            excluded_regions.append(self._excluded_region(block, exclusion_reason))

        continuation_resolution = CrossPageContinuationResolver().resolve(prepared)
        warnings: list[str] = []
        chunks: list[TranslationChunk] = []
        sequence = 0
        section_path: list[str] = []
        pending_body: list[Block] = []
        pending_body_section_path: list[str] = []
        pending_footnote: list[Block] = []
        pending_footnote_section_path: list[str] = []
        pending_keyword_heading: Block | None = None
        active_continuation_group_id: str | None = None

        def append_chunk(
            chunk_type: str,
            blocks: list[Block],
            *,
            chunk_section_path: list[str],
            chunk_warnings: list[str] | None = None,
            reason: str | None = None,
        ) -> None:
            nonlocal sequence
            if not blocks:
                return
            sequence += 1
            chunk = self._make_chunk(
                sequence,
                chunk_type,
                blocks,
                document_id=document_id,
                section_path=chunk_section_path,
                warnings=chunk_warnings or [],
                reason=reason,
            )
            chunks.append(chunk)
            for block in blocks:
                block.metadata["logical_translation_chunk_id"] = chunk.id

        def flush_body(*, warning: str | None = None) -> None:
            nonlocal pending_body, pending_body_section_path, active_continuation_group_id
            if not pending_body:
                return
            append_chunk(
                "paragraph",
                pending_body,
                chunk_section_path=pending_body_section_path,
                chunk_warnings=[warning] if warning else [],
                reason="ends_mid_sentence" if warning else None,
            )
            pending_body = []
            pending_body_section_path = []
            active_continuation_group_id = None

        def flush_footnote(*, warning: str | None = None) -> None:
            nonlocal pending_footnote, pending_footnote_section_path
            if not pending_footnote:
                return
            append_chunk(
                "footnote",
                pending_footnote,
                chunk_section_path=pending_footnote_section_path,
                chunk_warnings=[warning] if warning else [],
                reason="ends_mid_sentence" if warning else None,
            )
            pending_footnote = []
            pending_footnote_section_path = []

        for block in prepared.blocks:
            if block.id in exclusion_reasons:
                continue

            if block.block_type == BlockType.HEADING:
                flush_body(warning=self._pending_warning(pending_body))
                if pending_keyword_heading is not None:
                    append_chunk(
                        "heading",
                        [pending_keyword_heading],
                        chunk_section_path=section_path,
                    )
                heading_text = block.text.strip()
                if self._is_keyword_heading(heading_text):
                    pending_keyword_heading = block
                    continue
                pending_keyword_heading = None
                section_path = self._section_path_after_heading(section_path, heading_text)
                append_chunk("heading", [block], chunk_section_path=section_path)
                continue

            if block.block_type == BlockType.FOOTNOTE:
                if pending_keyword_heading is not None:
                    append_chunk(
                        "heading", [pending_keyword_heading], chunk_section_path=section_path
                    )
                    pending_keyword_heading = None
                if pending_footnote:
                    if self._footnote_continues(pending_footnote[-1], block):
                        pending_footnote.append(block)
                    else:
                        flush_footnote(warning=self._pending_warning(pending_footnote))
                        pending_footnote = [block]
                        pending_footnote_section_path = list(section_path)
                else:
                    pending_footnote = [block]
                    pending_footnote_section_path = list(section_path)
                if not self._ends_mid_sentence(block.text):
                    flush_footnote()
                continue

            if (
                block.block_type == BlockType.PARAGRAPH
                and pending_footnote
                and continuation_resolution.group_for(block.id) is None
                and self._looks_like_footnote_continuation(pending_footnote[-1], block)
            ):
                flush_body(warning=self._pending_warning(pending_body))
                pending_footnote.append(block)
                if not self._ends_mid_sentence(block.text):
                    flush_footnote()
                continue

            if (
                active_continuation_group_id is not None
                and continuation_resolution.is_intervening_for(
                    block.id,
                    active_continuation_group_id,
                )
            ):
                if pending_keyword_heading is not None:
                    append_chunk(
                        "heading",
                        [pending_keyword_heading],
                        chunk_section_path=section_path,
                    )
                    pending_keyword_heading = None
                append_chunk(
                    self._chunk_type(block),
                    [block],
                    chunk_section_path=section_path,
                )
                continue

            if block.block_type == BlockType.PARAGRAPH:
                if pending_keyword_heading is not None:
                    flush_body(warning=self._pending_warning(pending_body))
                    append_chunk(
                        "keywords",
                        [pending_keyword_heading, block],
                        chunk_section_path=section_path,
                    )
                    pending_keyword_heading = None
                    continue
                continuation_group = continuation_resolution.group_for(block.id)
                if continuation_group is not None:
                    continuation_index = int(block.metadata.get(CONTINUATION_INDEX, 0))
                    if continuation_index == 0:
                        if pending_body and not self._paragraph_continues(pending_body[-1], block):
                            flush_body(warning=self._pending_warning(pending_body))
                        if not pending_body:
                            pending_body_section_path = list(section_path)
                        pending_body.append(block)
                        active_continuation_group_id = continuation_group.id
                    elif active_continuation_group_id == continuation_group.id:
                        if not pending_body:
                            pending_body_section_path = list(section_path)
                        pending_body.append(block)
                    else:
                        flush_body(warning=self._pending_warning(pending_body))
                        pending_body_section_path = list(section_path)
                        pending_body = [block]
                    if continuation_index == len(continuation_group.block_ids) - 1:
                        active_continuation_group_id = None
                    continue
                if pending_body and not self._paragraph_continues(pending_body[-1], block):
                    flush_body(warning=self._pending_warning(pending_body))
                if not pending_body:
                    pending_body_section_path = list(section_path)
                pending_body.append(block)
                continue

            flush_body(warning=self._pending_warning(pending_body))
            if pending_keyword_heading is not None:
                append_chunk("heading", [pending_keyword_heading], chunk_section_path=section_path)
                pending_keyword_heading = None
            chunk_type = self._chunk_type(block)
            append_chunk(chunk_type, [block], chunk_section_path=section_path)

        flush_body(warning=self._pending_warning(pending_body, end_of_document=True))
        flush_footnote(warning=self._pending_warning(pending_footnote, end_of_document=True))
        if pending_keyword_heading is not None:
            append_chunk("heading", [pending_keyword_heading], chunk_section_path=section_path)

        if any(chunk.warnings for chunk in chunks):
            warnings.append(
                "Some OCR chunks ended without a confirmed continuation and were released with warnings."
            )
        prepared.translation_chunks = chunks
        prepared.metadata.translation = {
            **prepared.metadata.translation,
            "ocr_logical_chunks_prepared": True,
            "ocr_logical_chunk_count": len(chunks),
            "ocr_excluded_region_count": len(excluded_regions),
            "cross_page_continuation_group_count": len(continuation_resolution.groups),
        }
        return OCRLogicalParseResult(
            document=prepared,
            excluded_regions=excluded_regions,
            warnings=warnings,
        )

    def clean_ocr_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFC", text or "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = CONTROL_CHARACTER_PATTERN.sub("", normalized)
        normalized = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Cf" or character in {"\n", "\t"}
        )
        normalized = LINE_BREAK_HYPHEN_PATTERN.sub("", normalized)
        paragraphs = re.split(r"\n\s*\n", normalized)
        cleaned = [re.sub(r"\s+", " ", paragraph).strip() for paragraph in paragraphs]
        return "\n\n".join(paragraph for paragraph in cleaned if paragraph)

    def _clean_block(self, block: Block) -> Block:
        raw_text = block.text
        cleaned = self.clean_ocr_text(raw_text)
        metadata = dict(block.metadata)
        metadata["source_text_before_cleaning"] = raw_text
        return block.model_copy(update={"text": cleaned, "metadata": metadata})

    def _make_chunk(
        self,
        sequence: int,
        chunk_type: str,
        blocks: list[Block],
        *,
        document_id: str | None,
        section_path: list[str],
        warnings: list[str],
        reason: str | None,
    ) -> TranslationChunk:
        page_start = min(block.page_number for block in blocks)
        page_end = max(block.page_number for block in blocks)
        prefix = (
            f"p{page_start:04d}" if page_start == page_end else f"p{page_start:04d}-p{page_end:04d}"
        )
        separator = "\n" if chunk_type == "keywords" else " "
        source_text = self._join_text([block.text for block in blocks], separator=separator)
        raw_text = separator.join(
            str(block.metadata.get("source_text_before_cleaning", block.text)).strip()
            for block in blocks
            if block.text.strip()
        )
        continuation_metadata = self._continuation_metadata(blocks)
        return TranslationChunk(
            id=f"{prefix}-c{sequence:03d}",
            block_ids=[block.id for block in blocks],
            source_text=source_text,
            context=" > ".join(section_path),
            chunk_type=chunk_type,
            document_id=document_id,
            page_start=page_start,
            page_end=page_end,
            source_region_ids=self._source_region_ids(blocks),
            source_region_indexes=self._source_region_indexes(blocks),
            source_region_types=self._source_region_types(blocks),
            section_path=list(section_path),
            source_text_before_cleaning=raw_text,
            status="ready_for_translation",
            reason=reason,
            warnings=list(warnings),
            continues_from_previous_page=page_end > page_start,
            continues_to_next_page=page_end > page_start,
            **continuation_metadata,
        )

    def _continuation_metadata(self, blocks: list[Block]) -> dict[str, Any]:
        continuation_block = next(
            (block for block in blocks if block.metadata.get(CONTINUATION_GROUP_ID)),
            None,
        )
        if continuation_block is None:
            return {}
        metadata = continuation_block.metadata
        return {
            "continuation_group_id": str(metadata[CONTINUATION_GROUP_ID]),
            "continuation_decision_level": str(metadata.get(CONTINUATION_DECISION) or "proven"),
            "continuation_confidence": float(metadata.get(CONTINUATION_CONFIDENCE) or 0.0),
            "continuation_evidence": list(metadata.get(CONTINUATION_EVIDENCE) or []),
            "continuation_intervening_block_ids": list(
                metadata.get(CONTINUATION_INTERVENING_IDS) or []
            ),
        }

    def _join_text(self, parts: list[str], *, separator: str = " ") -> str:
        text = ""
        for raw_part in parts:
            part = raw_part.strip()
            if not part:
                continue
            if not text:
                text = part
            elif separator == " " and text.endswith("-") and part[:1].islower():
                text = text[:-1] + part
            else:
                text = f"{text}{separator}{part}"
        return text.strip()

    def _paragraph_continues(self, previous: Block, current: Block) -> bool:
        if current.page_number < previous.page_number:
            return False
        if current.page_number != previous.page_number:
            previous_group = previous.metadata.get(CONTINUATION_GROUP_ID)
            current_group = current.metadata.get(CONTINUATION_GROUP_ID)
            return bool(
                current.page_number == previous.page_number + 1
                and previous_group
                and previous_group == current_group
                and int(current.metadata.get(CONTINUATION_INDEX, -1))
                == int(previous.metadata.get(CONTINUATION_INDEX, -1)) + 1
            )
        previous_text = previous.text.rstrip()
        current_text = current.text.lstrip()
        if not previous_text or not current_text:
            return False
        if previous_text.endswith((".", "!", "?", "”", "“", '"', "'", "’", ")", "]")):
            return False
        if previous_text.endswith((",", ":", ";", "(", "–", "—", "-")):
            return True
        if previous_text.split()[-1].lower().strip(".,;:") in CONTINUATION_WORDS:
            return True
        return current_text[:1].islower() or current_text[:1] in {",", ".", ";", ":", ")", "]"}

    def _footnote_continues(self, previous: Block, current: Block) -> bool:
        if current.page_number != previous.page_number + 1:
            return False
        if FOOTNOTE_MARKER_PATTERN.match(current.text.strip()):
            return False
        return self._ends_mid_sentence(previous.text) and self._starts_like_continuation(
            current.text
        )

    def _looks_like_footnote_continuation(self, previous: Block, current: Block) -> bool:
        if current.page_number != previous.page_number + 1:
            return False
        if not self._ends_mid_sentence(previous.text) or not self._starts_like_continuation(
            current.text
        ):
            return False
        bbox = current.metadata.get("surya_bbox")
        page_height = current.metadata.get("surya_page_height")
        if not isinstance(bbox, list) or len(bbox) != 4 or not page_height:
            return False
        return float(bbox[1]) / float(page_height) >= 0.65

    def _starts_like_continuation(self, text: str) -> bool:
        stripped = text.strip()
        return bool(stripped) and (
            stripped[:1].islower() or stripped[:1] in {",", ".", ";", ":", ")", "]"}
        )

    def _ends_mid_sentence(self, text: str) -> bool:
        stripped = text.rstrip()
        if not stripped:
            return False
        return not stripped.endswith((".", "!", "?", "”", "“", '"', "'", "’", ")", "]"))

    def _pending_warning(self, blocks: list[Block], *, end_of_document: bool = False) -> str | None:
        if not blocks or not self._ends_mid_sentence(blocks[-1].text):
            return None
        stripped = blocks[-1].text.rstrip()
        if stripped.endswith(":"):
            return None
        if (
            len(stripped) < 180
            and stripped.split()[-1].lower().strip(".,;:") not in CONTINUATION_WORDS
        ):
            return None
        if end_of_document:
            return "Chunk ended mid-sentence at end of document; no continuation was found."
        return (
            "Chunk ended mid-sentence before a structural boundary; no continuation was confirmed."
        )

    def _chunk_type(self, block: Block) -> str:
        return {
            BlockType.CAPTION: "caption",
            BlockType.EQUATION: "equation",
            BlockType.FIGURE: "figure",
            BlockType.LIST: "list_item",
            BlockType.REFERENCE: "reference",
            BlockType.TABLE: "table",
        }.get(block.block_type, block.block_type.value)

    def _is_keyword_heading(self, text: str) -> bool:
        return text.strip().lower().rstrip(":") in KEYWORD_HEADINGS

    def _section_path_after_heading(self, current: list[str], heading: str) -> list[str]:
        match = re.match(r"^(?P<number>\d+(?:\.\d+)*)\b", heading.strip())
        if match:
            depth = len(match.group("number").split("."))
            return [*current[: depth - 1], heading]
        return [heading]

    def _repeated_margin_text(self, blocks: list[Block]) -> set[str]:
        pages_by_text: dict[str, set[int]] = {}
        for block in blocks:
            if block.block_type != BlockType.PARAGRAPH or not self._is_margin_region(block):
                continue
            canonical = self._canonical_repeated_text(block.text)
            if len(canonical) < 4:
                continue
            pages_by_text.setdefault(canonical, set()).add(block.page_number)
        return {text for text, pages in pages_by_text.items() if len(pages) >= 2}

    def _exclusion_reason(self, block: Block, repeated_margin_text: set[str]) -> str | None:
        if block.block_type == BlockType.FIGURE:
            return "figure_internal_text_preserved"
        if block.block_type in {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}:
            return f"surya_{block.block_type.value}"
        stripped = block.text.strip()
        if block.block_type == BlockType.PARAGRAPH and PAGE_NUMBER_PATTERN.fullmatch(stripped):
            return "page_number_text"
        if (
            block.block_type == BlockType.PARAGRAPH
            and self._is_margin_region(block)
            and self._canonical_repeated_text(stripped) in repeated_margin_text
        ):
            return "repeated_margin_text"
        return None

    def _is_margin_region(self, block: Block) -> bool:
        bbox = block.metadata.get("surya_bbox")
        page_height = block.metadata.get("surya_page_height")
        if not isinstance(bbox, list) or len(bbox) != 4 or not page_height:
            return False
        return (
            float(bbox[1]) / float(page_height) <= 0.2
            or float(bbox[3]) / float(page_height) >= 0.82
        )

    def _canonical_repeated_text(self, text: str) -> str:
        without_numbers = re.sub(r"\b\d+\b", " ", text.lower())
        return re.sub(r"\W+", " ", without_numbers, flags=re.UNICODE).strip()

    def _excluded_region(self, block: Block, reason: str) -> dict[str, Any]:
        return {
            "block_id": block.id,
            "page_number": block.page_number,
            "reason": reason,
            "text": block.text,
            "source_region_ids": self._source_region_ids([block]),
            "source_region_indexes": self._source_region_indexes([block]),
            "source_region_types": self._source_region_types([block]),
        }

    def _source_region_ids(self, blocks: list[Block]) -> list[str]:
        return self._unique(
            str(region_id)
            for block in blocks
            for region_id in (block.metadata.get("source_region_ids") or [block.id])
        )

    def _source_region_indexes(self, blocks: list[Block]) -> list[int]:
        indexes: list[int] = []
        for block in blocks:
            region_ids = block.metadata.get("source_region_ids") or []
            region_indexes = [
                int(match.group("index"))
                for region_id in region_ids
                if (match := SOURCE_REGION_INDEX_PATTERN.search(str(region_id)))
            ]
            if not region_indexes:
                value = block.metadata.get("surya_region_index")
                try:
                    region_indexes = [int(value)]
                except (TypeError, ValueError):
                    region_indexes = []
            for index in region_indexes:
                if index not in indexes:
                    indexes.append(index)
        return indexes

    def _source_region_types(self, blocks: list[Block]) -> list[str]:
        return self._unique(
            str(block.metadata.get("surya_region_type") or block.block_type.value)
            for block in blocks
        )

    def _unique(self, values) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return unique
