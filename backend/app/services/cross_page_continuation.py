from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models.schema import Block, BlockType, BoundingBox, DocumentModel


CONTINUATION_GROUP_ID = "cross_page_continuation_group_id"
CONTINUATION_GROUP_BLOCK_IDS = "cross_page_continuation_block_ids"
CONTINUATION_INDEX = "cross_page_continuation_index"
CONTINUATION_COUNT = "cross_page_continuation_count"
CONTINUATION_DECISION = "cross_page_continuation_decision"
CONTINUATION_CONFIDENCE = "cross_page_continuation_confidence"
CONTINUATION_EVIDENCE = "cross_page_continuation_evidence"
CONTINUATION_INTERVENING_IDS = "cross_page_continuation_intervening_block_ids"
CONTINUATION_VISIBLE_INTERVENING_IDS = "cross_page_continuation_visible_intervening_block_ids"
CONTINUATION_SEAMS = "cross_page_continuation_seams"
CONTINUES_FROM_PREVIOUS_PAGE = "continues_from_previous_page"
CONTINUES_TO_NEXT_PAGE = "continues_to_next_page"
INTERVENING_GROUP_IDS = "cross_page_intervening_for_group_ids"
INTERVENING_OBJECT_ROLE = "cross_page_intervening_object_role"
TRANSPARENT_REASON = "cross_page_transparent_reason"

_ANNOTATION_KEYS = {
    CONTINUATION_GROUP_ID,
    CONTINUATION_GROUP_BLOCK_IDS,
    CONTINUATION_INDEX,
    CONTINUATION_COUNT,
    CONTINUATION_DECISION,
    CONTINUATION_CONFIDENCE,
    CONTINUATION_EVIDENCE,
    CONTINUATION_INTERVENING_IDS,
    CONTINUATION_VISIBLE_INTERVENING_IDS,
    CONTINUATION_SEAMS,
    CONTINUES_FROM_PREVIOUS_PAGE,
    CONTINUES_TO_NEXT_PAGE,
    INTERVENING_GROUP_IDS,
    INTERVENING_OBJECT_ROLE,
    TRANSPARENT_REASON,
}

_TRANSPARENT_TYPES = {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}
_LAYOUT_OBJECT_TYPES = {
    BlockType.TABLE,
    BlockType.FIGURE,
    BlockType.CAPTION,
    BlockType.EQUATION,
}
_TRANSPARENT_EXCLUSION_REASONS = {
    "page_number_text",
    "repeated_margin_text",
    "surya_footer",
    "surya_header",
    "surya_page_number",
}
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d+$", flags=re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:table|tabla|tabelle|tableau|tab\.)\s*(?:[ivxlcdm]+|\d+)\b",
    flags=re.IGNORECASE,
)
_STRUCTURAL_PARAGRAPH_RE = re.compile(
    r"^\s*(?:(?:\d+|[A-ZÀ-ÖØ-Þ]|[IVXLCDM]+)\s*(?:[.)]-?|[-–—])|[•▪◦])\s*",
)
_REFERENCE_START_RE = re.compile(r"^\s*\[\d+(?:\s*[-,]\s*\d+)*\]\s+[A-ZÀ-ÖØ-Þ]")
_TRAILING_CITATION_RE = re.compile(
    r"(?:"
    r"\[(?:\s*\d+[A-Za-z]?(?:\s*[-,;]\s*\d+[A-Za-z]?)*\s*)\]"
    r"|\((?=[^()]{0,60}\d)[^()]{1,60}\)"
    r"|[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡]+"
    r")\s*$"
)
_TRAILING_QUOTES_RE = re.compile(r"[>'\"”’»›]+\s*$")
_TRAILING_BRACKETS_RE = re.compile(r"[\]\)}]+\s*$")
_ABBREVIATION_RE = re.compile(
    r"(?:\bet\s+al|\be\.g|\bi\.e|\b(?:approx|cf|dr|eq|etc|fig|mr|mrs|ms|no|nr|"
    r"prof|ref|tab|vs)|(?:\b[A-ZÀ-ÖØ-Þ]\.){1,3})\.$",
    flags=re.IGNORECASE,
)
_SPLIT_WORD_RE = re.compile(r"[^\W\d_]\s*[-\u00ad]\s*$", flags=re.UNICODE)
_CONTINUATION_WORDS = {
    "als",
    "and",
    "both",
    "but",
    "con",
    "da",
    "de",
    "del",
    "der",
    "des",
    "e",
    "et",
    "mit",
    "o",
    "oder",
    "of",
    "or",
    "pero",
    "sowie",
    "sowohl",
    "und",
    "with",
    "y",
}
_NEXT_CONJUNCTIONS = {
    "also",
    "and",
    "as",
    "because",
    "but",
    "da",
    "denn",
    "e",
    "et",
    "jedoch",
    "mais",
    "oder",
    "or",
    "pero",
    "porque",
    "sowie",
    "und",
    "while",
    "y",
}


@dataclass(frozen=True)
class CrossPageContinuationLink:
    previous_block_id: str
    current_block_id: str
    previous_page: int
    current_page: int
    decision_level: str
    confidence: float
    evidence: tuple[str, ...]
    intervening_block_ids: tuple[str, ...]
    visible_intervening_block_ids: tuple[str, ...]


@dataclass(frozen=True)
class CrossPageContinuationGroup:
    id: str
    block_ids: tuple[str, ...]
    links: tuple[CrossPageContinuationLink, ...]
    decision_level: str
    confidence: float
    evidence: tuple[str, ...]
    intervening_block_ids: tuple[str, ...]
    visible_intervening_block_ids: tuple[str, ...]


@dataclass(frozen=True)
class CrossPageContinuationResolution:
    groups: tuple[CrossPageContinuationGroup, ...]
    group_by_block_id: dict[str, CrossPageContinuationGroup]
    intervening_group_ids_by_block_id: dict[str, tuple[str, ...]]

    def group_for(self, block_id: str) -> CrossPageContinuationGroup | None:
        return self.group_by_block_id.get(block_id)

    def is_intervening_for(self, block_id: str, group_id: str) -> bool:
        return group_id in self.intervening_group_ids_by_block_id.get(block_id, ())


@dataclass(frozen=True)
class _GeometryEvidence:
    available: bool
    compatible: bool
    reasons: tuple[str, ...]


class CrossPageContinuationResolver:
    """Conservatively prove body-paragraph links across physical page seams.

    The resolver only records links. It never changes source block order or
    inserts layout objects into a prose group. A page containing only tables,
    figures, captions, equations, and transparent margins may be crossed as an
    interruption when the prose evidence on both sides independently proves a
    continuation.
    """

    def resolve(self, document: DocumentModel) -> CrossPageContinuationResolution:
        self._clear_annotations(document.blocks)
        positions = {block.id: index for index, block in enumerate(document.blocks)}
        page_blocks: dict[int, list[Block]] = {}
        for block in document.blocks:
            page_blocks.setdefault(block.page_number, []).append(block)

        dimensions = self._page_dimensions(document)
        transparent_reasons = self._transparent_reasons(document, dimensions)
        captioned_table_members = self._captioned_table_members(page_blocks, dimensions)
        page_footnote_ids = {
            block.id
            for block in document.blocks
            if block.block_type == BlockType.FOOTNOTE
            and self._is_bottom_footnote(block, dimensions)
        }
        object_roles = {
            block.id: block.block_type.value
            for block in document.blocks
            if block.block_type in _LAYOUT_OBJECT_TYPES
        }
        object_roles.update(
            {block_id: "captioned_table_region" for block_id in captioned_table_members}
        )
        object_roles.update(
            {block_id: "dedicated_page_footnote_lane" for block_id in page_footnote_ids}
        )

        links: list[CrossPageContinuationLink] = []
        for previous_page in sorted(page_blocks):
            previous = self._edge_candidate(
                page_blocks[previous_page],
                reverse=True,
                transparent_reasons=transparent_reasons,
                object_roles=object_roles,
            )
            if previous is None:
                continue

            current: Block | None = None
            current_page = previous_page + 1
            while current_page in page_blocks:
                current = self._edge_candidate(
                    page_blocks[current_page],
                    reverse=False,
                    transparent_reasons=transparent_reasons,
                    object_roles=object_roles,
                )
                if current is not None:
                    break
                if not self._is_object_only_bridge_page(
                    page_blocks[current_page],
                    transparent_reasons,
                    object_roles,
                ):
                    break
                current_page += 1
            if current is None:
                continue
            previous_position = positions[previous.id]
            current_position = positions[current.id]
            if current_position <= previous_position:
                continue
            between = document.blocks[previous_position + 1 : current_position]
            if any(
                not (previous_page <= block.page_number <= current_page)
                or (block.id not in transparent_reasons and block.id not in object_roles)
                for block in between
            ):
                continue
            decision = self._decision(
                previous,
                current,
                between,
                transparent_reasons,
                object_roles,
                dimensions,
            )
            if decision is None:
                continue
            confidence, evidence = decision
            intervening_ids = tuple(block.id for block in between)
            visible_ids = tuple(
                block.id for block in between if block.id not in transparent_reasons
            )
            links.append(
                CrossPageContinuationLink(
                    previous_block_id=previous.id,
                    current_block_id=current.id,
                    previous_page=previous_page,
                    current_page=current_page,
                    decision_level="proven",
                    confidence=confidence,
                    evidence=evidence,
                    intervening_block_ids=intervening_ids,
                    visible_intervening_block_ids=visible_ids,
                )
            )

        groups = self._groups(links, positions)
        self._annotate(document, groups, transparent_reasons, object_roles)
        group_by_block_id = {block_id: group for group in groups for block_id in group.block_ids}
        intervening_groups: dict[str, list[str]] = {}
        for group in groups:
            for block_id in group.intervening_block_ids:
                intervening_groups.setdefault(block_id, []).append(group.id)
        return CrossPageContinuationResolution(
            groups=tuple(groups),
            group_by_block_id=group_by_block_id,
            intervening_group_ids_by_block_id={
                block_id: tuple(group_ids) for block_id, group_ids in intervening_groups.items()
            },
        )

    def _clear_annotations(self, blocks: list[Block]) -> None:
        for block in blocks:
            for key in _ANNOTATION_KEYS:
                block.metadata.pop(key, None)

    def _transparent_reasons(
        self,
        document: DocumentModel,
        dimensions: dict[int, tuple[float, float]],
    ) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for block in document.blocks:
            exclusion_reason = str(block.metadata.get("translation_exclusion_reason") or "")
            if block.block_type in _TRANSPARENT_TYPES:
                reasons[block.id] = block.block_type.value
            elif exclusion_reason in _TRANSPARENT_EXCLUSION_REASONS:
                reasons[block.id] = exclusion_reason
            elif block.block_type == BlockType.PARAGRAPH and _PAGE_NUMBER_RE.fullmatch(
                self._source_text(block).strip()
            ):
                reasons[block.id] = "page_number_text"

        pages_by_text: dict[str, set[int]] = {}
        ids_by_text: dict[str, list[str]] = {}
        for block in document.blocks:
            if block.block_type != BlockType.PARAGRAPH or not self._is_margin_block(
                block, dimensions
            ):
                continue
            canonical = self._canonical_margin_text(self._source_text(block))
            if len(canonical) < 4:
                continue
            pages_by_text.setdefault(canonical, set()).add(block.page_number)
            ids_by_text.setdefault(canonical, []).append(block.id)
        for canonical, pages in pages_by_text.items():
            if len(pages) < 2:
                continue
            for block_id in ids_by_text[canonical]:
                reasons.setdefault(block_id, "repeated_margin_text")
        return reasons

    def _captioned_table_members(
        self,
        page_blocks: dict[int, list[Block]],
        dimensions: dict[int, tuple[float, float]],
    ) -> set[str]:
        members: set[str] = set()
        for page_number, blocks in page_blocks.items():
            page_height = dimensions.get(page_number, (0.0, 0.0))[1]
            if page_height <= 0:
                continue
            maximum_gap = max(12.0, page_height * 0.026)
            for caption_index, caption in enumerate(blocks):
                if (
                    caption.block_type != BlockType.CAPTION
                    or not _TABLE_CAPTION_RE.match(self._source_text(caption).strip())
                    or self._bbox(caption) is None
                ):
                    continue
                candidate_members: list[Block] = []
                next_top = self._bbox(caption).y0  # type: ignore[union-attr]
                for candidate in reversed(blocks[:caption_index]):
                    if candidate.block_type not in {BlockType.PARAGRAPH, BlockType.LIST}:
                        break
                    bbox = self._bbox(candidate)
                    if bbox is None:
                        candidate_members = []
                        break
                    gap = next_top - bbox.y1
                    if gap > maximum_gap or gap < -(page_height * 0.03):
                        break
                    candidate_members.append(candidate)
                    next_top = min(next_top, bbox.y0)
                if len(candidate_members) < 2 or not any(
                    candidate.block_type == BlockType.LIST for candidate in candidate_members
                ):
                    continue
                members.update(candidate.id for candidate in candidate_members)
        return members

    def _edge_candidate(
        self,
        blocks: list[Block],
        *,
        reverse: bool,
        transparent_reasons: dict[str, str],
        object_roles: dict[str, str],
    ) -> Block | None:
        ordered = reversed(blocks) if reverse else iter(blocks)
        for block in ordered:
            if block.id in transparent_reasons or block.id in object_roles:
                continue
            if (
                block.block_type == BlockType.PARAGRAPH
                and self._source_text(block).strip()
                and not self._looks_structural(self._source_text(block))
            ):
                return block
            return None
        return None

    def _is_object_only_bridge_page(
        self,
        blocks: list[Block],
        transparent_reasons: dict[str, str],
        object_roles: dict[str, str],
    ) -> bool:
        visible = [block for block in blocks if block.id not in transparent_reasons]
        return (
            bool(visible)
            and all(block.id in object_roles for block in visible)
            and any(block.block_type in _LAYOUT_OBJECT_TYPES for block in visible)
        )

    def _decision(
        self,
        previous: Block,
        current: Block,
        between: list[Block],
        transparent_reasons: dict[str, str],
        object_roles: dict[str, str],
        dimensions: dict[int, tuple[float, float]],
    ) -> tuple[float, tuple[str, ...]] | None:
        if current.page_number <= previous.page_number:
            return None
        previous_text = self._source_text(previous).strip()
        current_text = self._source_text(current).strip()
        if not previous_text or not current_text or self._looks_structural(current_text):
            return None

        previous_section = self._section_signature(previous)
        current_section = self._section_signature(current)
        if previous_section is not None and current_section is not None:
            if previous_section != current_section:
                return None

        style_compatible, style_available = self._style_compatibility(previous, current)
        if not style_compatible:
            return None
        geometry = self._geometry_evidence(
            previous,
            current,
            between,
            object_roles,
            dimensions,
        )
        if geometry.available and not geometry.compatible:
            return None

        terminal = self._terminal_kind(previous_text)
        if terminal == "terminal":
            return None
        split_word = bool(_SPLIT_WORD_RE.search(previous_text))
        connector = self._ends_with_connector(previous_text)
        next_kind = self._next_start_kind(current_text)
        if next_kind == "structural" or next_kind == "neutral":
            return None
        if (
            next_kind == "uppercase"
            and not style_available
            and not (split_word or connector or geometry.compatible)
        ):
            return None

        bridges_object_pages = current.page_number > previous.page_number + 1
        evidence = (
            ["consecutive_physical_page_span", "object_only_intervening_pages"]
            if bridges_object_pages
            else ["consecutive_pages"]
        )
        if terminal == "abbreviation":
            if not geometry.compatible or next_kind not in {
                "lowercase",
                "conjunction",
                "punctuation",
            }:
                return None
            evidence.append("abbreviation_terminal_overridden")
        else:
            evidence.append("previous_nonterminal")

        if split_word:
            evidence.append("split_word_hyphen")
        elif connector:
            evidence.append("previous_connector")

        evidence.append(
            {
                "lowercase": "next_lowercase",
                "conjunction": "next_conjunction",
                "punctuation": "next_punctuation",
                "uppercase": "uppercase_start_layout_supported",
            }[next_kind]
        )

        if geometry.available:
            evidence.extend(geometry.reasons)
        elif not (
            split_word or (connector and next_kind in {"lowercase", "conjunction", "punctuation"})
        ):
            return None
        else:
            evidence.append("strong_text_without_geometry")

        if style_available:
            evidence.append("compatible_style")
        if previous_section is not None and current_section is not None:
            evidence.append("same_section_hierarchy")
        if any(block.id in transparent_reasons for block in between):
            evidence.append("transparent_margin_blocks")
        roles = {object_roles[block.id] for block in between if block.id in object_roles}
        if roles:
            evidence.append("intervening_layout_object")
        if "captioned_table_region" in roles:
            evidence.append("caption_bound_table_region")
        if "dedicated_page_footnote_lane" in roles:
            evidence.append("dedicated_page_footnote_lane")

        if split_word and geometry.compatible:
            confidence = 0.99
        elif terminal == "abbreviation":
            confidence = 0.90
        elif geometry.compatible and next_kind != "uppercase":
            confidence = 0.96 if connector else 0.94
        elif geometry.compatible:
            confidence = 0.90
        else:
            confidence = 0.88
        if bridges_object_pages:
            confidence = min(confidence, 0.92)
        return confidence, tuple(dict.fromkeys(evidence))

    def _geometry_evidence(
        self,
        previous: Block,
        current: Block,
        between: list[Block],
        object_roles: dict[str, str],
        dimensions: dict[int, tuple[float, float]],
    ) -> _GeometryEvidence:
        previous_bbox = self._bbox(previous)
        current_bbox = self._bbox(current)
        previous_size = dimensions.get(previous.page_number)
        current_size = dimensions.get(current.page_number)
        if (
            previous_bbox is None
            or current_bbox is None
            or previous_size is None
            or current_size is None
            or previous_size[0] <= 0
            or previous_size[1] <= 0
            or current_size[0] <= 0
            or current_size[1] <= 0
        ):
            return _GeometryEvidence(False, False, ())

        previous_width, previous_height = previous_size
        current_width, current_height = current_size
        previous_object_top = min(
            (
                bbox.y0
                for block in between
                if block.page_number == previous.page_number
                and block.id in object_roles
                and (bbox := self._bbox(block)) is not None
            ),
            default=None,
        )
        if previous_object_top is None:
            previous_at_flow_end = previous_bbox.y1 / previous_height >= 0.72
            previous_reason = "previous_fragment_near_page_bottom"
        else:
            gap = previous_object_top - previous_bbox.y1
            previous_at_flow_end = (
                -(previous_height * 0.02) <= gap <= previous_height * 0.09
                and previous_bbox.y1 / previous_height >= 0.20
            )
            previous_reason = "previous_fragment_precedes_intervening_object"
        object_bottom = max(
            (
                bbox.y1
                for block in between
                if block.page_number == current.page_number
                and block.id in object_roles
                and (bbox := self._bbox(block)) is not None
            ),
            default=None,
        )
        if object_bottom is None:
            current_at_flow_start = current_bbox.y0 / current_height <= 0.30
            current_reason = "next_fragment_near_page_top"
        else:
            gap = current_bbox.y0 - object_bottom
            current_at_flow_start = (
                -(current_height * 0.02) <= gap <= current_height * 0.09
                and current_bbox.y0 / current_height <= 0.72
            )
            current_reason = "next_fragment_follows_intervening_object"

        previous_box_width = max(0.0, previous_bbox.x1 - previous_bbox.x0)
        current_box_width = max(0.0, current_bbox.x1 - current_bbox.x0)
        width_ratio = min(previous_box_width, current_box_width) / max(
            previous_box_width,
            current_box_width,
            1.0,
        )
        width_compatible = width_ratio >= 0.55
        page_width_ratio = min(previous_width, current_width) / max(
            previous_width,
            current_width,
            1.0,
        )
        page_compatible = page_width_ratio >= 0.80
        compatible = (
            previous_at_flow_end and current_at_flow_start and width_compatible and page_compatible
        )
        reasons = (
            previous_reason,
            current_reason,
            "compatible_cross_page_geometry",
        )
        return _GeometryEvidence(True, compatible, reasons if compatible else ())

    def _style_compatibility(self, previous: Block, current: Block) -> tuple[bool, bool]:
        available = False
        previous_size = self._number(previous.style_hints.get("font_size"))
        current_size = self._number(current.style_hints.get("font_size"))
        if previous_size is not None and current_size is not None:
            available = True
            if abs(previous_size - current_size) > max(2.0, min(previous_size, current_size) * 0.2):
                return False, True
        for key in ("bold", "italic"):
            if key in previous.style_hints and key in current.style_hints:
                available = True
                if bool(previous.style_hints[key]) != bool(current.style_hints[key]):
                    return False, True
        return True, available

    def _terminal_kind(self, text: str) -> str:
        stripped = text.rstrip()
        previous = None
        while stripped and stripped != previous:
            previous = stripped
            stripped = _TRAILING_CITATION_RE.sub("", stripped).rstrip()
            stripped = _TRAILING_QUOTES_RE.sub("", stripped).rstrip()
        stripped = _TRAILING_BRACKETS_RE.sub("", stripped).rstrip()
        stripped = _TRAILING_QUOTES_RE.sub("", stripped).rstrip()
        if not stripped:
            return "nonterminal"
        if stripped.endswith(("…", "...", "!", "?")):
            return "terminal"
        if stripped.endswith("."):
            return "abbreviation" if _ABBREVIATION_RE.search(stripped) else "terminal"
        return "nonterminal"

    def _ends_with_connector(self, text: str) -> bool:
        stripped = text.rstrip()
        if stripped.endswith((",", ":", ";", "(", "–", "—", "-", "\u00ad")):
            return True
        words = re.findall(r"[^\W\d_]+", stripped.casefold(), flags=re.UNICODE)
        return bool(words) and words[-1] in _CONTINUATION_WORDS

    def _next_start_kind(self, text: str) -> str:
        stripped = text.lstrip()
        if not stripped or self._looks_structural(stripped):
            return "structural"
        if stripped[0] in {",", ".", ";", ":", ")", "]", "–", "—"}:
            return "punctuation"
        stripped = stripped.lstrip("'\"“‘«(")
        first_word = re.match(r"[^\W\d_]+", stripped, flags=re.UNICODE)
        if first_word is None:
            return "neutral"
        word = first_word.group(0)
        if word.casefold() in _NEXT_CONJUNCTIONS:
            return "conjunction"
        if word[:1].islower():
            return "lowercase"
        if word[:1].isupper():
            return "uppercase"
        return "neutral"

    def _looks_structural(self, text: str) -> bool:
        stripped = text.strip()
        return bool(
            _STRUCTURAL_PARAGRAPH_RE.match(stripped)
            or _REFERENCE_START_RE.match(stripped)
            or re.match(r"^[-*+]\s+", stripped)
        )

    def _section_signature(self, block: Block) -> Any | None:
        for key in ("section_hierarchy", "section_path"):
            value = block.metadata.get(key)
            if value:
                return value
        return None

    def _groups(
        self,
        links: list[CrossPageContinuationLink],
        positions: dict[str, int],
    ) -> list[CrossPageContinuationGroup]:
        chains: list[list[CrossPageContinuationLink]] = []
        for link in links:
            if chains and chains[-1][-1].current_block_id == link.previous_block_id:
                chains[-1].append(link)
            else:
                chains.append([link])

        groups: list[CrossPageContinuationGroup] = []
        for ordinal, chain in enumerate(chains, start=1):
            block_ids = [chain[0].previous_block_id]
            block_ids.extend(link.current_block_id for link in chain)
            first_page = chain[0].previous_page
            last_page = chain[-1].current_page
            intervening_ids = sorted(
                {block_id for link in chain for block_id in link.intervening_block_ids},
                key=positions.__getitem__,
            )
            visible_ids = sorted(
                {block_id for link in chain for block_id in link.visible_intervening_block_ids},
                key=positions.__getitem__,
            )
            evidence = tuple(dict.fromkeys(reason for link in chain for reason in link.evidence))
            groups.append(
                CrossPageContinuationGroup(
                    id=f"xpage-p{first_page:04d}-p{last_page:04d}-g{ordinal:03d}",
                    block_ids=tuple(block_ids),
                    links=tuple(chain),
                    decision_level="proven",
                    confidence=min(link.confidence for link in chain),
                    evidence=evidence,
                    intervening_block_ids=tuple(intervening_ids),
                    visible_intervening_block_ids=tuple(visible_ids),
                )
            )
        return groups

    def _annotate(
        self,
        document: DocumentModel,
        groups: list[CrossPageContinuationGroup],
        transparent_reasons: dict[str, str],
        object_roles: dict[str, str],
    ) -> None:
        block_by_id = {block.id: block for block in document.blocks}
        for block_id, reason in transparent_reasons.items():
            if block_id in block_by_id:
                block_by_id[block_id].metadata[TRANSPARENT_REASON] = reason
        for group in groups:
            seams = [
                {
                    "previous_block_id": link.previous_block_id,
                    "current_block_id": link.current_block_id,
                    "previous_page": link.previous_page,
                    "current_page": link.current_page,
                    "intermediate_page_numbers": list(
                        range(link.previous_page + 1, link.current_page)
                    ),
                    "decision_level": link.decision_level,
                    "confidence": link.confidence,
                    "evidence": list(link.evidence),
                    "intervening_block_ids": list(link.intervening_block_ids),
                }
                for link in group.links
            ]
            for index, block_id in enumerate(group.block_ids):
                block = block_by_id[block_id]
                block.metadata.update(
                    {
                        CONTINUATION_GROUP_ID: group.id,
                        CONTINUATION_GROUP_BLOCK_IDS: list(group.block_ids),
                        CONTINUATION_INDEX: index,
                        CONTINUATION_COUNT: len(group.block_ids),
                        CONTINUATION_DECISION: group.decision_level,
                        CONTINUATION_CONFIDENCE: group.confidence,
                        CONTINUATION_EVIDENCE: list(group.evidence),
                        CONTINUATION_INTERVENING_IDS: list(group.intervening_block_ids),
                        CONTINUATION_VISIBLE_INTERVENING_IDS: list(
                            group.visible_intervening_block_ids
                        ),
                        CONTINUATION_SEAMS: seams,
                        CONTINUES_FROM_PREVIOUS_PAGE: index > 0,
                        CONTINUES_TO_NEXT_PAGE: index < len(group.block_ids) - 1,
                    }
                )
            for block_id in group.intervening_block_ids:
                block = block_by_id[block_id]
                group_ids = list(block.metadata.get(INTERVENING_GROUP_IDS) or [])
                if group.id not in group_ids:
                    group_ids.append(group.id)
                block.metadata[INTERVENING_GROUP_IDS] = group_ids
                if block_id in object_roles:
                    block.metadata[INTERVENING_OBJECT_ROLE] = object_roles[block_id]

    def _is_bottom_footnote(
        self,
        block: Block,
        dimensions: dict[int, tuple[float, float]],
    ) -> bool:
        bbox = self._bbox(block)
        page_dimensions = dimensions.get(block.page_number)
        return bool(
            bbox is not None
            and page_dimensions is not None
            and page_dimensions[1] > 0
            and bbox.y0 / page_dimensions[1] >= 0.80
        )

    def _is_margin_block(
        self,
        block: Block,
        dimensions: dict[int, tuple[float, float]],
    ) -> bool:
        bbox = self._bbox(block)
        page_dimensions = dimensions.get(block.page_number)
        if bbox is None or page_dimensions is None or page_dimensions[1] <= 0:
            return False
        return bbox.y0 / page_dimensions[1] <= 0.20 or bbox.y1 / page_dimensions[1] >= 0.82

    def _page_dimensions(self, document: DocumentModel) -> dict[int, tuple[float, float]]:
        dimensions = {
            page.page_number: (float(page.width), float(page.height)) for page in document.pages
        }
        for block in document.blocks:
            if block.page_number in dimensions and all(dimensions[block.page_number]):
                continue
            width = self._first_number(
                block.metadata,
                "surya_page_width",
                "marker_page_width",
                "source_page_width",
            )
            height = self._first_number(
                block.metadata,
                "surya_page_height",
                "marker_page_height",
                "source_page_height",
            )
            if width is not None and height is not None:
                dimensions[block.page_number] = (width, height)
        return dimensions

    def _bbox(self, block: Block) -> BoundingBox | None:
        if block.bbox is not None:
            return block.bbox
        for key in ("surya_bbox", "bbox"):
            value = block.metadata.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 4:
                try:
                    return BoundingBox(
                        x0=float(value[0]),
                        y0=float(value[1]),
                        x1=float(value[2]),
                        y1=float(value[3]),
                    )
                except (TypeError, ValueError):
                    return None
        return None

    def _canonical_margin_text(self, text: str) -> str:
        without_numbers = re.sub(r"\b\d+\b", " ", text.casefold())
        return re.sub(r"\W+", " ", without_numbers, flags=re.UNICODE).strip()

    def _source_text(self, block: Block) -> str:
        source_text = block.metadata.get("source_text")
        return str(source_text) if source_text is not None else block.text

    def _first_number(self, values: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            number = self._number(values.get(key))
            if number is not None:
                return number
        return None

    def _number(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
