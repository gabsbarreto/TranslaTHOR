from __future__ import annotations

import html
import logging
import math
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any, Callable

from langdetect import detect

try:
    from langdetect import DetectorFactory, detect_langs

    # langdetect otherwise chooses a random seed and can disagree about the same
    # short scientific phrase between chunks in a single run.
    DetectorFactory.seed = 0
except (ImportError, AttributeError):  # pragma: no cover - compatibility for light test stubs
    detect_langs = None  # type: ignore[assignment]

from app.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_LLM_MIN_P,
    DEFAULT_LLM_PRESENCE_PENALTY,
    DEFAULT_LLM_REPETITION_PENALTY,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_TOP_P,
    DEFAULT_MLX_CPU_THREADS,
    DEFAULT_TRANSLATION_BATCH_SIZE,
    DEFAULT_TRANSLATION_BATCH_TOKEN_BUDGET,
    DEFAULT_TRANSLATION_MODEL,
)
from app.models.schema import Block, BlockType, DocumentModel, TranslationChunk
from app.services.cross_page_continuation import (
    CONTINUATION_CONFIDENCE,
    CONTINUATION_DECISION,
    CONTINUATION_EVIDENCE,
    CONTINUATION_GROUP_ID,
    CONTINUATION_INDEX,
    CONTINUATION_INTERVENING_IDS,
    CONTINUATION_SEAMS,
    CONTINUES_FROM_PREVIOUS_PAGE,
    CONTINUES_TO_NEXT_PAGE,
    CrossPageContinuationResolver,
)

logger = logging.getLogger(__name__)


@dataclass
class TranslationSettings:
    model_name: str = DEFAULT_TRANSLATION_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_group_size: int = 5
    temperature: float = DEFAULT_LLM_TEMPERATURE
    top_p: float = DEFAULT_LLM_TOP_P
    top_k: int = DEFAULT_LLM_TOP_K
    min_p: float = DEFAULT_LLM_MIN_P
    presence_penalty: float = DEFAULT_LLM_PRESENCE_PENALTY
    repetition_penalty: float = DEFAULT_LLM_REPETITION_PENALTY
    max_tokens: int = 1024
    batch_size: int = DEFAULT_TRANSLATION_BATCH_SIZE
    batch_token_budget: int = DEFAULT_TRANSLATION_BATCH_TOKEN_BUDGET
    cpu_threads: int = DEFAULT_MLX_CPU_THREADS


@dataclass
class TranslationUnit:
    block_ids: list[str]
    text: str
    block_type: BlockType
    context: str = ""


@dataclass(frozen=True)
class _LanguageResolution:
    language: str | None
    origin: str | None
    confidence: float | None


@dataclass(frozen=True)
class _BatchTranslationRequest:
    text: str
    context: str
    source_language: str | None
    block_type: BlockType | None
    source_language_authoritative: bool = False


@dataclass(frozen=True)
class _PreparedBatchPrompt:
    request: _BatchTranslationRequest
    prompt: str
    prompt_tokens: tuple[int, ...]
    max_tokens: int

    @property
    def token_cost(self) -> int:
        return len(self.prompt_tokens) + self.max_tokens


@dataclass(frozen=True)
class _InstructionTemplateCache:
    system: str
    prefix: str
    suffix: str
    prefix_tokens: tuple[int, ...] | None
    token_composition_safe: bool


@dataclass(frozen=True)
class _ChunkTranslationPlan:
    index: int
    chunk: TranslationChunk
    block_type: BlockType | None
    effective_context: str
    source_language_authoritative: bool
    is_table_like: bool
    is_english: bool
    physical_segments: tuple[tuple[str, str, BlockType], ...]


@dataclass(frozen=True)
class _TableCellTopology:
    tag: str
    attributes: tuple[tuple[str, str], ...]
    rowspan: int
    colspan: int


@dataclass(frozen=True)
class _TableRowTopology:
    section: tuple[str, int] | None
    attributes: tuple[tuple[str, str], ...]
    cells: tuple[_TableCellTopology, ...]


@dataclass(frozen=True)
class _TableMarkupTopology:
    table_attributes: tuple[tuple[str, str], ...]
    section_attributes: tuple[
        tuple[str, int, tuple[tuple[str, str], ...]],
        ...,
    ]
    rows: tuple[_TableRowTopology, ...]
    section_events: tuple[tuple[str, str, int], ...]


class _StrictTableTopologyParser(HTMLParser):
    """Read table structure while rejecting crossed or unbalanced table tags."""

    MAX_CELL_SPAN = 1_000

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.valid = True
        self.saw_table = False
        self.table_open = False
        self.table_attributes: tuple[tuple[str, str], ...] = ()
        self.current_section: tuple[str, int] | None = None
        self.current_row: list[_TableCellTopology] | None = None
        self.current_row_attributes: tuple[tuple[str, str], ...] = ()
        self.current_cell: _TableCellTopology | None = None
        self.section_attributes: list[tuple[str, int, tuple[tuple[str, str], ...]]] = []
        self.rows: list[_TableRowTopology] = []
        self.section_events: list[tuple[str, str, int]] = []
        self._section_counts: Counter[str] = Counter()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized == "table":
            if self.table_open or self.saw_table:
                self.valid = False
                return
            self.saw_table = True
            self.table_open = True
            self.table_attributes = self._normalized_attributes(attrs)
            return
        if normalized in {"thead", "tbody", "tfoot"}:
            if (
                not self.table_open
                or self.current_section is not None
                or self.current_row is not None
                or self.current_cell is not None
            ):
                self.valid = False
                return
            occurrence = self._section_counts[normalized]
            self._section_counts[normalized] += 1
            self.current_section = (normalized, occurrence)
            self.section_attributes.append(
                (normalized, occurrence, self._normalized_attributes(attrs))
            )
            self.section_events.append(("start", normalized, occurrence))
            return
        if normalized == "tr":
            if not self.table_open or self.current_row is not None or self.current_cell is not None:
                self.valid = False
                return
            self.current_row = []
            self.current_row_attributes = self._normalized_attributes(attrs)
            return
        if normalized in {"td", "th"}:
            if self.current_row is None or self.current_cell is not None:
                self.valid = False
                return
            attributes = {str(name).casefold(): value for name, value in attrs}
            self.current_cell = _TableCellTopology(
                tag=normalized,
                attributes=self._normalized_attributes(attrs),
                rowspan=self._cell_span(attributes.get("rowspan")),
                colspan=self._cell_span(attributes.get("colspan")),
            )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}:
            self.valid = False

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"td", "th"}:
            if (
                self.current_cell is None
                or self.current_row is None
                or self.current_cell.tag != normalized
            ):
                self.valid = False
                return
            self.current_row.append(self.current_cell)
            self.current_cell = None
            return
        if normalized == "tr":
            if self.current_row is None or self.current_cell is not None:
                self.valid = False
                return
            self.rows.append(
                _TableRowTopology(
                    section=self.current_section,
                    attributes=self.current_row_attributes,
                    cells=tuple(self.current_row),
                )
            )
            self.current_row = None
            self.current_row_attributes = ()
            return
        if normalized in {"thead", "tbody", "tfoot"}:
            if (
                self.current_section is None
                or self.current_section[0] != normalized
                or self.current_row is not None
                or self.current_cell is not None
            ):
                self.valid = False
                return
            occurrence = self.current_section[1]
            self.section_events.append(("end", normalized, occurrence))
            self.current_section = None
            return
        if normalized == "table":
            if (
                not self.table_open
                or self.current_section is not None
                or self.current_row is not None
                or self.current_cell is not None
            ):
                self.valid = False
                return
            self.table_open = False

    def topology(self) -> _TableMarkupTopology | None:
        if (
            not self.valid
            or not self.saw_table
            or self.table_open
            or self.current_section is not None
            or self.current_row is not None
            or self.current_cell is not None
            or not self.rows
        ):
            return None
        return _TableMarkupTopology(
            table_attributes=self.table_attributes,
            section_attributes=tuple(self.section_attributes),
            rows=tuple(self.rows),
            section_events=tuple(self.section_events),
        )

    def _normalized_attributes(
        self,
        attrs: list[tuple[str, str | None]],
    ) -> tuple[tuple[str, str], ...]:
        normalized: list[tuple[str, str]] = []
        for raw_name, raw_value in attrs:
            name = str(raw_name).strip().casefold()
            value = html.unescape(str(raw_value or "")).strip()
            if name == "class":
                value = " ".join(sorted(value.split()))
            elif name == "style":
                value = self._normalized_style(value)
            elif name in {"align", "valign", "dir", "scope"}:
                value = value.casefold()
            normalized.append((name, value))
        return tuple(sorted(normalized))

    def _normalized_style(self, value: str) -> str:
        declarations: list[str] = []
        for declaration in value.split(";"):
            declaration = declaration.strip()
            if not declaration:
                continue
            property_name, separator, property_value = declaration.partition(":")
            if not separator:
                declarations.append(" ".join(declaration.split()))
                continue
            declarations.append(
                f"{property_name.strip().casefold()}:{' '.join(property_value.split())}"
            )
        return ";".join(declarations)

    def _cell_span(self, value: str | None) -> int:
        try:
            span = int(str(value or "1").strip())
        except (TypeError, ValueError):
            return 1
        return span if 0 < span <= self.MAX_CELL_SPAN else 1


class MlxTranslator:
    TABLE_DELIMITER = "\n|||CELL_BREAK|||\n"
    TABLE_HEADER_PREFIX = "__table_header__:"
    TABLE_ROW_PREFIX = "__table_row__:"
    TABLE_OUTPUT_MAX_TOKENS = 4096
    TABLE_ROW_GROUP_SIZE = 3
    PROSE_CHUNK_TOKEN_CAP = 800
    TRANSLATION_FAILED_STATUS = "translation_failed"
    _ENGLISH_SOURCE_CONFIDENCE = 0.90
    _NON_ENGLISH_OUTPUT_CONFIDENCE = 0.85
    _COMPACT_TABLE_TRANSLATION_GUIDANCE = (
        "Keep table labels as concise as the source so they fit their original cells. "
        "Do not expand source abbreviations; use an equally concise standard English abbreviation "
        "when unambiguous, otherwise preserve the source abbreviation. Never abbreviate or truncate "
        "an ordinary source word: when the source spells a term out, spell out its English translation. "
        "Prefer compact labels over explanatory phrases."
    )
    _TAG_RE = re.compile(r"<[^>]+>")
    _ENTITY_RE = re.compile(r"&[a-zA-Z0-9#]+;")
    _TABLE_BLOCK_RE = re.compile(r"(?is)<table\b.*?</table>")
    _TABLE_SPLIT_RE = re.compile(
        r"(?is)^(?P<before>.*?)(?P<table><table\b.*?</table>)(?P<after>.*)$"
    )
    _TABLE_PARTS_RE = re.compile(
        r"(?is)^(?P<prefix>.*?<table\b[^>]*>)(?P<body>.*)(?P<suffix></table>\s*)$"
    )
    _TABLE_ROW_RE = re.compile(r"(?is)<tr\b[^>]*>.*?</tr>")
    _TABLE_OPEN_RE = re.compile(r"(?is)<table\b[^>]*>")
    _TABLE_SECTION_OPEN_RE = re.compile(r"(?is)<(?P<tag>thead|tbody|tfoot)\b[^>]*>")
    _TABLE_ESCAPED_TAG_RE = re.compile(
        r"(?is)&lt;\s*(/?)\s*(table|thead|tbody|tfoot|tr|td|th)\b([^<>]*?)&gt;"
    )
    _TABLE_ESCAPED_ROW_RE = re.compile(r"(?is)&lt;\s*tr\b")
    _TABLE_ROW_OPEN_RE = re.compile(r"(?is)<tr\b[^>]*>")
    _TABLE_ROW_CLOSE_RE = re.compile(r"(?is)</tr\s*>")
    _TABLE_CELL_RE = re.compile(r"(?is)<t[dh]\b[^>]*>(?P<body>.*?)</t[dh]\s*>")
    _SEGMENT_TAG = "translathor-segment"
    _SEGMENT_RE = re.compile(
        r"(?is)<translathor-segment\s+index=(?:\"|')(?P<index>\d+)(?:\"|')\s*>"
        r"(?P<body>.*?)</translathor-segment\s*>"
    )
    _SENTENCE_ABBREVIATIONS = {
        "al",
        "approx",
        "cf",
        "dr",
        "e.g",
        "eq",
        "etc",
        "fig",
        "i.e",
        "jr",
        "mr",
        "mrs",
        "ms",
        "no",
        "nr",
        "p",
        "pp",
        "prof",
        "ref",
        "st",
        "tab",
        "vs",
    }
    _ENGLISH_HINT_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
    _ENGLISH_SINGLE_WORD_HINTS = {
        "abstract",
        "appendix",
        "conclusion",
        "conclusions",
        "discussion",
        "introduction",
        "material",
        "method",
        "methods",
        "reference",
        "references",
        "result",
        "results",
        "summary",
    }
    _SPANISH_STRUCTURAL_HEADINGS = {
        "bibliografía": "References",
        "conclusión": "Conclusion",
        "conclusiones": "Conclusions",
        "dirección del autor": "Author Contact",
        "discusión": "Discussion",
        "introducción": "Introduction",
        "material y método": "Materials and Methods",
        "material y métodos": "Materials and Methods",
        "referencias": "References",
        "resultados": "Results",
        "resumen": "Abstract",
    }

    def __init__(self, settings: TranslationSettings) -> None:
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._mlx_stream = None
        self._document_language: str | None = None
        self._document_defined_acronyms: set[str] = set()
        self._last_load_error: str | None = None
        self._instruction_template_cache: _InstructionTemplateCache | None = None
        self._instruction_cache_hits = 0
        self._instruction_cache_misses = 0
        self._cpu_threads = 0
        self._mlx_runtime: dict[str, Any] = {}
        self._batch_stats: Counter[str] = Counter()

    def _ensure_loaded(self) -> bool:
        if os.getenv("DISABLE_MLX", "0") == "1":
            logger.warning("MLX disabled with DISABLE_MLX=1")
            self._last_load_error = "MLX disabled with DISABLE_MLX=1"
            return False
        if self._model is not None and self._tokenizer is not None:
            self._last_load_error = None
            return True
        try:
            self._configure_cpu_threads()
            self._configure_mlx_thread()
            self._model, self._tokenizer = self._load_model_and_tokenizer(self.settings.model_name)
            self._instruction_template_cache = None
            self._record_mlx_runtime()
            self._last_load_error = None
            return True
        except Exception as exc:
            self._last_load_error = str(exc)
            logger.warning("Unable to load MLX model %s: %s", self.settings.model_name, exc)
            return False

    def last_load_error(self) -> str | None:
        return self._last_load_error

    def _load_model_and_tokenizer(self, model_name: str):
        from mlx_lm import load

        return load(model_name)

    def build_chunks(self, document: DocumentModel) -> list[TranslationChunk]:
        self._document_language = self._normalize_lang_code(document.metadata.detected_language)
        self._document_defined_acronyms = {
            acronym
            for block in document.blocks
            for acronym in self._ordered_acronyms(
                str(block.metadata.get("source_text", block.text))
            )
        }
        if document.metadata.translation.get("ocr_logical_chunks_prepared"):
            return self._prepared_logical_chunks(document)

        units = self._build_translation_units(document)
        block_by_id = {block.id: block for block in document.blocks}
        chunks: list[TranslationChunk] = []
        chunk_block_types: dict[str, BlockType] = {}
        for unit in units:
            unit_pages = sorted(
                {
                    block_by_id[block_id].page_number
                    for block_id in unit.block_ids
                    if block_id in block_by_id
                }
            )
            if unit.block_ids and (
                unit.block_ids[0].startswith(self.TABLE_HEADER_PREFIX)
                or unit.block_ids[0].startswith(self.TABLE_ROW_PREFIX)
            ):
                text_parts = [unit.text]
            elif unit.block_type == BlockType.TABLE and self._is_table_heavy_markup(unit.text):
                text_parts = [unit.text]
            elif len(unit.block_ids) > 1:
                # Physical block boundaries must survive token-budget handling.
                # _translate_physical_segments() batches whole regions safely.
                text_parts = [unit.text]
            else:
                text_parts = self._split_to_token_budget(unit.text)
            for text_part in text_parts:
                if unit.block_type != BlockType.TABLE and not self._has_translatable_content(
                    text_part
                ):
                    continue
                nearby_context = self._nearby_heading_context(
                    document,
                    unit.block_ids,
                    unit.block_type,
                )
                language = self._chunk_language_resolution(
                    text_part,
                    unit.block_type,
                    nearby_context,
                )
                continuation_metadata = self._continuation_chunk_metadata(
                    [
                        block_by_id[block_id]
                        for block_id in unit.block_ids
                        if block_id in block_by_id
                    ]
                )
                chunk = TranslationChunk(
                    id=f"chunk-{len(chunks)}",
                    block_ids=unit.block_ids,
                    source_text=text_part,
                    context=self._context_with_nearby_source(unit.context, nearby_context),
                    source_language=language.language,
                    source_language_origin=language.origin,
                    source_language_confidence=language.confidence,
                    source_token_count=self._token_count(text_part),
                    page_start=unit_pages[0] if unit_pages else None,
                    page_end=unit_pages[-1] if unit_pages else None,
                    continues_from_previous_page=len(unit_pages) > 1,
                    continues_to_next_page=len(unit_pages) > 1,
                    **continuation_metadata,
                )
                chunks.append(chunk)
                chunk_block_types[chunk.id] = unit.block_type
        return self._merge_adjacent_translation_chunks(
            chunks,
            chunk_block_types,
            document=document,
        )

    def _prepared_logical_chunks(self, document: DocumentModel) -> list[TranslationChunk]:
        prepared: list[TranslationChunk] = []
        for chunk in document.translation_chunks:
            if chunk.status != "ready_for_translation":
                continue
            block_type = self._logical_chunk_block_type(chunk.chunk_type)
            if block_type == BlockType.FIGURE:
                continue
            if block_type == BlockType.TABLE and self._is_table_heavy_markup(chunk.source_text):
                text_parts = [chunk.source_text]
            elif len(chunk.block_ids) > 1:
                # Repeating the full block-id list on every token-budget part
                # would translate and apply each physical region more than once.
                text_parts = [chunk.source_text]
            else:
                text_parts = self._split_to_token_budget(chunk.source_text)
            for part_index, text_part in enumerate(text_parts, start=1):
                if block_type != BlockType.TABLE and not self._has_translatable_content(text_part):
                    continue
                nearby_context = self._nearby_heading_context(
                    document,
                    chunk.block_ids,
                    block_type,
                )
                language = self._chunk_language_resolution(
                    text_part,
                    block_type,
                    nearby_context,
                )
                prepared.append(
                    chunk.model_copy(
                        update={
                            "id": (
                                chunk.id
                                if len(text_parts) == 1
                                else f"{chunk.id}-part{part_index:02d}"
                            ),
                            "source_text": text_part,
                            "context": self._context_with_nearby_source(
                                chunk.context,
                                nearby_context,
                            ),
                            "source_language": language.language,
                            "source_language_origin": language.origin,
                            "source_language_confidence": language.confidence,
                            "source_token_count": self._token_count(text_part),
                        }
                    )
                )
        return prepared

    def _logical_chunk_block_type(self, chunk_type: str) -> BlockType:
        return {
            "caption": BlockType.CAPTION,
            "equation": BlockType.EQUATION,
            "figure": BlockType.FIGURE,
            "footnote": BlockType.FOOTNOTE,
            "heading": BlockType.HEADING,
            "keywords": BlockType.PARAGRAPH,
            "list_item": BlockType.LIST,
            "reference": BlockType.REFERENCE,
            "table": BlockType.TABLE,
        }.get(chunk_type, BlockType.PARAGRAPH)

    def _continuation_chunk_metadata(self, blocks: list[Block]) -> dict[str, Any]:
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

    def translate_document(
        self,
        document: DocumentModel,
        markdown: str,
        on_chunk_started: Callable[[int, int], None] | None = None,
        on_chunk_translated: Callable[[int, int, str], None] | None = None,
        on_table_progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[DocumentModel, str]:
        loaded = self._ensure_loaded()
        chunks = self.build_chunks(document)
        document.translation_chunks = chunks

        if not chunks:
            return document, markdown

        translated_md = markdown
        block_by_id = {block.id: block for block in document.blocks}
        table_by_id = {table.id: table for table in document.tables}
        application_chunks: list[TranslationChunk] = []
        table_like_chunks = [
            chunk for chunk in chunks if self._is_table_heavy_markup(chunk.source_text)
        ]
        table_chunk_index: dict[int, int] = {
            id(chunk): idx for idx, chunk in enumerate(table_like_chunks, start=1)
        }

        total_chunks = len(chunks)
        plans = [
            self._chunk_translation_plan(index, chunk, loaded, block_by_id)
            for index, chunk in enumerate(chunks, start=1)
        ]
        native_batching = loaded and self._native_batch_translation_enabled()
        cursor = 0
        while cursor < len(plans):
            plan = plans[cursor]
            if native_batching and self._is_batchable_plan(plan):
                batch_plans = [plan]
                next_cursor = cursor + 1
                while (
                    next_cursor < len(plans)
                    and len(batch_plans) < max(1, int(self.settings.batch_size or 1))
                    and self._is_batchable_plan(plans[next_cursor])
                ):
                    batch_plans.append(plans[next_cursor])
                    next_cursor += 1

                if on_chunk_started is not None:
                    for batch_plan in batch_plans:
                        on_chunk_started(batch_plan.index, total_chunks)
                requests = [
                    _BatchTranslationRequest(
                        text=batch_plan.chunk.source_text,
                        context=batch_plan.effective_context,
                        source_language=batch_plan.chunk.source_language,
                        block_type=batch_plan.block_type,
                        source_language_authoritative=(batch_plan.source_language_authoritative),
                    )
                    for batch_plan in batch_plans
                ]
                translations = self._translate_requests_with_validation_batch(requests)
                for batch_plan, translated in zip(
                    batch_plans,
                    translations,
                    strict=True,
                ):
                    chunk = batch_plan.chunk
                    batch_validation_issue = self._translation_acceptance_issue(
                        chunk.source_text,
                        translated,
                        chunk.source_language,
                        batch_plan.block_type,
                        source_language_authoritative=(batch_plan.source_language_authoritative),
                    )
                    chunk.translated_text = translated
                    if batch_validation_issue is not None:
                        self._mark_translation_failure(chunk, batch_validation_issue)
                    else:
                        chunk.status = "ready_for_translation"
                        chunk.reason = None
                    application_chunks.append(chunk)
                    if on_chunk_translated is not None:
                        preview = translated.replace("\n", " ").strip()
                        on_chunk_translated(
                            batch_plan.index,
                            total_chunks,
                            preview[:160],
                        )
                cursor = next_cursor
                continue

            chunk = plan.chunk
            index = plan.index
            block_type = plan.block_type
            effective_context = plan.effective_context
            source_language_authoritative = plan.source_language_authoritative
            is_table_like = plan.is_table_like
            is_english = plan.is_english
            physical_segments = list(plan.physical_segments)
            validation_issue: str | None = None

            if on_chunk_started is not None:
                on_chunk_started(index, total_chunks)

            if is_table_like and on_table_progress is not None:
                on_table_progress(
                    table_chunk_index.get(id(chunk), 1),
                    max(1, len(table_like_chunks)),
                    f"chunk-{index}",
                )

            if not loaded or is_english:
                translated = chunk.source_text
                if physical_segments:
                    application_chunks.extend(
                        self._segment_application_chunks(
                            chunk,
                            physical_segments,
                            [source_text for _, source_text, _ in physical_segments],
                            block_by_id,
                        )
                    )
            elif is_table_like:
                translated = self._translate_table_markup_chunk(
                    chunk.source_text,
                    effective_context,
                    chunk.source_language,
                )
                validation_issue = self._table_translation_issue(
                    chunk.source_text,
                    translated,
                    chunk.source_language,
                )
            elif physical_segments:
                segment_targets = self._translate_physical_segments(
                    physical_segments,
                    effective_context,
                    chunk.source_language,
                    chunk.source_text,
                    block_by_id,
                )
                translated = "\n\n".join(segment_targets)
                validation_issue = self._translation_acceptance_issue(
                    "\n\n".join(source_text for _, source_text, _ in physical_segments),
                    translated,
                    chunk.source_language,
                    block_type,
                )
                application_chunks.extend(
                    self._segment_application_chunks(
                        chunk,
                        physical_segments,
                        segment_targets,
                        block_by_id,
                        validation_issue=validation_issue,
                    )
                )
            else:
                translated = self._translate_chunk_with_validation(
                    chunk.source_text,
                    effective_context,
                    chunk.source_language,
                    block_type,
                    source_language_authoritative=source_language_authoritative,
                )
                validation_issue = self._translation_acceptance_issue(
                    chunk.source_text,
                    translated,
                    chunk.source_language,
                    block_type,
                    source_language_authoritative=source_language_authoritative,
                )
            chunk.translated_text = translated
            if validation_issue is not None:
                self._mark_translation_failure(chunk, validation_issue)
            else:
                chunk.status = "ready_for_translation"
                chunk.reason = None
            if not physical_segments:
                application_chunks.append(chunk)
            if on_chunk_translated is not None:
                preview = translated.replace("\n", " ").strip()
                on_chunk_translated(index, total_chunks, preview[:160])
            cursor += 1

        application_chunks = self._coalesce_translated_chunks(application_chunks)
        document.translation_chunks = application_chunks
        for chunk in application_chunks:
            self._apply_translation_to_target(chunk, block_by_id, table_by_id)

        failed_chunks = [
            chunk for chunk in application_chunks if chunk.status == self.TRANSLATION_FAILED_STATUS
        ]
        document.metadata.translation["target_language_validation"] = {
            "status": "warning" if failed_chunks else "passed",
            "failed_chunk_count": len(failed_chunks),
            "failed_chunk_ids": [chunk.id for chunk in failed_chunks],
            "policy": "retry_english_then_preserve_source",
        }
        if failed_chunks:
            warning = (
                f"{len(failed_chunks)} translation chunk(s) failed English-output validation; "
                "their source text was preserved and the chunk IDs are recorded in structured JSON."
            )
            if warning not in document.warnings:
                document.warnings.append(warning)

        document.metadata.translation["mlx_runtime"] = self.runtime_metadata()

        return document, translated_md

    def _chunk_translation_plan(
        self,
        index: int,
        chunk: TranslationChunk,
        loaded: bool,
        block_by_id: dict[str, Block],
    ) -> _ChunkTranslationPlan:
        block_type = self._chunk_block_type(chunk, block_by_id)
        effective_context = self._augment_context_for_block_type(chunk.context, block_type)
        source_language_authoritative = self._source_language_is_authoritative(chunk)
        is_table_like = self._is_table_heavy_markup(chunk.source_text)
        preserve_verbatim = block_type in {
            BlockType.EQUATION,
            BlockType.PAGE_NUMBER,
            BlockType.REFERENCE,
        }
        structural_translation = self._structural_label_translation(
            chunk.source_text,
            chunk.source_language,
            block_type,
        )
        is_english = loaded and (
            preserve_verbatim
            or (structural_translation is None and self._is_already_english(chunk))
        )
        return _ChunkTranslationPlan(
            index=index,
            chunk=chunk,
            block_type=block_type,
            effective_context=effective_context,
            source_language_authoritative=source_language_authoritative,
            is_table_like=is_table_like,
            is_english=is_english,
            physical_segments=tuple(self._physical_translation_segments(chunk, block_by_id)),
        )

    def _native_batch_translation_enabled(self) -> bool:
        """Respect test/application overrides of the established translation hooks."""

        translate_chunk = getattr(self._translate_chunk, "__func__", None)
        translate_validated = getattr(
            self._translate_chunk_with_validation,
            "__func__",
            None,
        )
        return (
            translate_chunk is MlxTranslator._translate_chunk
            and translate_validated is MlxTranslator._translate_chunk_with_validation
        )

    def _is_batchable_plan(self, plan: _ChunkTranslationPlan) -> bool:
        if (
            plan.is_english
            or plan.is_table_like
            or plan.physical_segments
            or not plan.chunk.source_text.strip()
        ):
            return False
        if plan.block_type not in {
            BlockType.HEADING,
            BlockType.PARAGRAPH,
            BlockType.LIST,
            BlockType.FOOTNOTE,
            BlockType.HEADER,
            BlockType.FOOTER,
            BlockType.UNKNOWN,
        }:
            return False
        if (
            self._structural_label_translation(
                plan.chunk.source_text,
                plan.chunk.source_language,
                plan.block_type,
            )
            is not None
        ):
            return False
        return (
            self._structural_caption_parts(
                plan.chunk.source_text,
                plan.chunk.source_language,
                plan.block_type,
            )
            is None
        )

    def _physical_translation_segments(
        self,
        chunk: TranslationChunk,
        block_by_id: dict[str, Block],
    ) -> list[tuple[str, str, BlockType]]:
        """Return source text for each independently placeable block in a grouped chunk."""
        if len(chunk.block_ids) <= 1 or self._is_table_heavy_markup(chunk.source_text):
            return []

        segments: list[tuple[str, str, BlockType]] = []
        seen: set[str] = set()
        for block_id in chunk.block_ids:
            if block_id in seen:
                return []
            block = block_by_id.get(block_id)
            if block is None:
                return []
            source_text = str(block.metadata.get("source_text", block.text)).strip()
            if not source_text:
                return []
            segments.append((block_id, source_text, block.block_type))
            seen.add(block_id)
        return segments

    def _translate_physical_segments(
        self,
        segments: list[tuple[str, str, BlockType]],
        context: str,
        source_language: str | None,
        logical_source_text: str,
        block_by_id: dict[str, Block],
    ) -> list[str]:
        targets: list[str] = []
        batches = self._physical_segment_batches(segments)
        segment_offset = 0
        for batch in batches:
            shared_context = self._physical_batch_shared_context(
                context,
                logical_source_text,
                segments,
                segment_offset,
                len(batch),
            )
            if len(batch) == 1:
                _, source_text, block_type = batch[0]
                targets.append(
                    self._translate_single_physical_segment(
                        source_text,
                        shared_context,
                        source_language,
                        block_type,
                    )
                )
                segment_offset += len(batch)
                continue
            targets.extend(
                self._translate_tagged_physical_segments(
                    batch,
                    context,
                    source_language,
                    shared_context,
                    block_by_id,
                    continuity_proven=self._physical_segments_form_continuous_paragraph(
                        batch,
                        block_by_id,
                    ),
                )
            )
            segment_offset += len(batch)
        return targets

    def _physical_batch_shared_context(
        self,
        context: str,
        logical_source_text: str,
        segments: list[tuple[str, str, BlockType]],
        start: int,
        count: int,
    ) -> str:
        token_budget = self._physical_segment_token_budget()
        if self._token_count(logical_source_text) <= token_budget:
            passage_label = "complete continuous source passage"
            passage_context = logical_source_text
        else:
            # Keep oversized continuation groups seam-aware without placing the
            # complete multi-page passage in every prompt. Whole physical
            # regions remain the batching boundary; immediate neighboring text
            # supplies grammar at the cut.
            excerpt_budget = max(48, min(256, token_budget // 3))
            end = start + count
            excerpts: list[str] = []
            if start > 0:
                excerpts.append(
                    "[preceding fragment] "
                    + self._bounded_context_excerpt(
                        segments[start - 1][1],
                        excerpt_budget // 2,
                        from_end=True,
                    )
                )
            if end < len(segments):
                excerpts.append(
                    "[following fragment] "
                    + self._bounded_context_excerpt(
                        segments[end][1],
                        excerpt_budget // 2,
                        from_end=False,
                    )
                )
            passage_label = "adjacent seam context from an oversized continuous passage"
            passage_context = (
                "\n".join(excerpts) or "No additional neighboring fragment fits safely."
            )
        return (
            f"{context}\n"
            f"Translate only TEXT below. It belongs to the following {passage_label}; use that "
            f"passage only for terminology and grammatical context:\n{passage_context}"
        ).strip()

    def _bounded_context_excerpt(
        self,
        text: str,
        token_budget: int,
        *,
        from_end: bool,
    ) -> str:
        words = text.split()
        if not words:
            return ""
        while len(words) > 1 and self._token_count(" ".join(words)) > token_budget:
            remove_count = max(1, len(words) // 4)
            words = words[remove_count:] if from_end else words[:-remove_count]
        return " ".join(words)

    def _physical_segments_form_continuous_paragraph(
        self,
        segments: list[tuple[str, str, BlockType]],
        block_by_id: dict[str, Block],
    ) -> bool:
        """Require layout and textual evidence before moving text across boxes.

        Adjacent chunks are batched for linguistic context, but adjacency does
        not make them one paragraph. Redistribution is safe only for blocks
        that the normal paragraph join accepts, or for an explicit hyphenated
        word continuation across a column boundary. An adjacent-page pair is
        accepted only when the shared resolver recorded the exact source-order
        seam and every intervening block. Ordinary unproven groups retain the
        same-page protections.
        """

        if len(segments) < 2 or any(
            block_type != BlockType.PARAGRAPH for _, _, block_type in segments
        ):
            return False

        document_positions = {block_id: index for index, block_id in enumerate(block_by_id)}
        for previous_segment, current_segment in zip(
            segments,
            segments[1:],
        ):
            previous = block_by_id.get(previous_segment[0])
            current = block_by_id.get(current_segment[0])
            if previous is None or current is None:
                return False
            previous_position = document_positions.get(previous.id)
            current_position = document_positions.get(current.id)
            if (
                previous_position is None
                or current_position is None
                or current_position <= previous_position
            ):
                return False
            previous_section = previous.metadata.get("section_hierarchy")
            current_section = current.metadata.get("section_hierarchy")
            if previous_section and current_section and previous_section != current_section:
                return False
            if previous.page_number != current.page_number:
                if not self._is_proven_cross_page_segment_pair(
                    previous,
                    current,
                    previous_position,
                    current_position,
                    block_by_id,
                ):
                    return False
                continue
            if (
                current_position != previous_position + 1
                or current.reading_order_index <= previous.reading_order_index
            ):
                return False
            if self._belongs_to_same_paragraph(previous, current):
                continue
            if not self._is_explicit_cross_column_continuation(
                previous,
                current,
                previous_segment[1],
                current_segment[1],
            ):
                return False
        return True

    def _is_proven_cross_page_segment_pair(
        self,
        previous: Block,
        current: Block,
        previous_position: int,
        current_position: int,
        block_by_id: dict[str, Block],
    ) -> bool:
        if current.page_number != previous.page_number + 1:
            return False
        previous_group = previous.metadata.get(CONTINUATION_GROUP_ID)
        current_group = current.metadata.get(CONTINUATION_GROUP_ID)
        if not previous_group or previous_group != current_group:
            return False
        if (
            int(current.metadata.get(CONTINUATION_INDEX, -1))
            != int(previous.metadata.get(CONTINUATION_INDEX, -1)) + 1
        ):
            return False

        seam = next(
            (
                item
                for item in (previous.metadata.get(CONTINUATION_SEAMS) or [])
                if isinstance(item, dict)
                and item.get("previous_block_id") == previous.id
                and item.get("current_block_id") == current.id
                and item.get("decision_level") == "proven"
            ),
            None,
        )
        if seam is None:
            return False
        expected_intervening = [str(value) for value in seam.get("intervening_block_ids") or []]
        document_ids = list(block_by_id)
        actual_intervening = document_ids[previous_position + 1 : current_position]
        return actual_intervening == expected_intervening

    def _is_explicit_cross_column_continuation(
        self,
        previous: Block,
        current: Block,
        previous_source: str,
        current_source: str,
    ) -> bool:
        if previous.bbox is None or current.bbox is None:
            return False
        previous_text = previous_source.rstrip()
        current_text = current_source.lstrip()
        first_letter = re.match(r"[^\W\d_]", current_text, flags=re.UNICODE)
        if first_letter is None or not first_letter.group(0).islower():
            return False
        if previous_text.endswith((".", "!", "?", "”", "“", '"', "'", "’", ")", "]")):
            return False
        if not re.search(r"[^\W\d_]\s*[-\u00ad]?\s*$", previous_text, flags=re.UNICODE):
            return False

        # A reading-order wrap must move right and upward into the next
        # column. The OCR logical parser has already required a non-terminal
        # first region and a lowercase continuation. This catches both a split
        # word ("expe-" / "rience") and an ordinary sentence split
        # ("attitude" / "due to ...") without joining unrelated paragraphs.
        return (
            current.bbox.x0 > previous.bbox.x0
            and current.bbox.x0 >= previous.bbox.x1 - 2.0
            and current.bbox.y0 < previous.bbox.y0
        )

    def _physical_segment_batches(
        self,
        segments: list[tuple[str, str, BlockType]],
    ) -> list[list[tuple[str, str, BlockType]]]:
        token_budget = self._physical_segment_token_budget()
        batches: list[list[tuple[str, str, BlockType]]] = []
        current: list[tuple[str, str, BlockType]] = []
        current_tokens = 0
        for segment in segments:
            segment_tokens = self._token_count(segment[1]) + 8
            if current and current_tokens + segment_tokens > token_budget:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(segment)
            current_tokens += segment_tokens
        if current:
            batches.append(current)
        return batches

    def _physical_segment_token_budget(self) -> int:
        token_budget = max(128, int(self.settings.chunk_size or DEFAULT_CHUNK_SIZE))
        token_budget = min(token_budget, self.PROSE_CHUNK_TOKEN_CAP)
        return min(token_budget, max(128, int(self.settings.max_tokens * 0.75)))

    def _translate_single_physical_segment(
        self,
        source_text: str,
        context: str,
        source_language: str | None,
        block_type: BlockType,
    ) -> str:
        return "\n\n".join(
            self._translate_chunk_with_validation(
                part,
                context,
                source_language,
                block_type,
            )
            for part in self._split_to_token_budget(source_text)
        )

    def _translate_tagged_physical_segments(
        self,
        segments: list[tuple[str, str, BlockType]],
        context: str,
        source_language: str | None,
        shared_context: str,
        block_by_id: dict[str, Block],
        *,
        continuity_proven: bool,
    ) -> list[str]:
        if continuity_proven:
            logical_source = self._join_paragraph_lines(
                [source_text for _, source_text, _ in segments]
            )
            logical_context = (
                f"{shared_context}\n"
                "Translate this complete continuous passage exactly once, without placement tags. "
                "Do not complete any part twice, repeat, summarize, or omit any clause."
            )
            logical_target = self._translate_chunk_with_validation(
                logical_source,
                logical_context,
                source_language,
                self._segment_validation_block_type(segments),
            )
            redistributed = self._redistribute_logical_translation(
                logical_source,
                logical_target,
                segments,
                source_language,
                block_by_id,
                continuity_proven=True,
            )
            if redistributed is not None:
                logger.info(
                    "Translated a continuous passage once and redistributed it across physical regions."
                )
                return redistributed

        tagged_source = self._tagged_segment_text(segments)
        if continuity_proven:
            segment_instruction = (
                f"TEXT contains {len(segments)} ordered <{self._SEGMENT_TAG}> elements from one "
                "proven continuous paragraph. First translate the complete paragraph mentally, then "
                "distribute that translation across the same ordered elements. Their concatenated contents "
                "must read as one fluent, grammatical English passage even when a boundary falls in the "
                "middle of a sentence. The tags are placement boundaries, not word-for-word constraints, "
                "so adjust English word order around adjacent boundaries when needed. Preserve every opening "
                "tag, closing tag, index, order, and boundary marker exactly. Return no text outside the "
                "elements."
            )
        else:
            segment_instruction = (
                f"TEXT contains {len(segments)} ordered <{self._SEGMENT_TAG}> physical regions that share "
                "terminology context but are not proven to be one paragraph. Translate each element "
                "independently. Do not move, merge, duplicate, or redistribute any clause or value between "
                "elements. Preserve every opening tag, closing tag, index, order, and boundary marker exactly. "
                "Return no text outside the elements."
            )
        strict_context = f"{context}\n{segment_instruction}".strip()
        translated = self._translate_chunk(tagged_source, strict_context, source_language)
        collapsed_mapping = self._tagged_translation_has_collapsed_region(
            translated,
            segments,
        )
        parsed = self._parse_segment_translation(translated, segments, source_language)
        if parsed is not None:
            return parsed

        boundary_retry = (
            "Neighboring wording may shift only as needed to keep their concatenation grammatical."
            if continuity_proven
            else "Do not shift wording or meaning from one element into another element."
        )
        retry_context = (
            f"{strict_context}\n"
            "The prior result did not preserve the segment mapping. Each input index must occur exactly "
            f"once in the output. Do not omit or duplicate content. {boundary_retry} Return English only "
            "and translate every substantive non-English phrase; do not repeat source-language prose."
        )
        retried = self._translate_chunk(tagged_source, retry_context, source_language)
        collapsed_mapping = collapsed_mapping or self._tagged_translation_has_collapsed_region(
            retried,
            segments,
        )
        parsed = self._parse_segment_translation(retried, segments, source_language)
        if parsed is not None:
            return parsed

        if collapsed_mapping and continuity_proven:
            logical_source = self._join_paragraph_lines(
                [source_text for _, source_text, _ in segments]
            )
            logical_context = (
                f"{shared_context}\n"
                "Translate this complete continuous passage once, without placement tags. "
                "Do not repeat, summarize, or omit any clause."
            )
            logical_target = self._translate_chunk_with_validation(
                logical_source,
                logical_context,
                source_language,
                self._segment_validation_block_type(segments),
            )
            redistributed = self._redistribute_logical_translation(
                logical_source,
                logical_target,
                segments,
                source_language,
                block_by_id,
                continuity_proven=True,
            )
            if redistributed is not None:
                logger.info(
                    "Redistributed a validated continuous translation across collapsed physical regions."
                )
                return redistributed

        logger.warning(
            "Grouped translation did not preserve physical block boundaries; retrying blocks separately."
        )
        fallback_targets = [
            self._translate_single_physical_segment(
                source_text,
                shared_context,
                source_language,
                block_type,
            )
            for _, source_text, block_type in segments
        ]
        aggregate_issue = self._translation_acceptance_issue(
            "\n\n".join(source_text for _, source_text, _ in segments),
            "\n\n".join(fallback_targets),
            source_language,
            self._segment_validation_block_type(segments),
        )
        if aggregate_issue is not None:
            logger.warning(
                "Separate physical-block translations failed target-language validation (%s); "
                "returning source text.",
                aggregate_issue,
            )
            return [source_text for _, source_text, _ in segments]
        return fallback_targets

    def _tagged_segment_text(self, segments: list[tuple[str, str, BlockType]]) -> str:
        return "\n".join(
            f'<{self._SEGMENT_TAG} index="{index}">{source_text}</{self._SEGMENT_TAG}>'
            for index, (_, source_text, _) in enumerate(segments)
        )

    def _parse_segment_translation(
        self,
        translated: str,
        segments: list[tuple[str, str, BlockType]],
        source_language: str | None,
    ) -> list[str] | None:
        targets = self._tagged_segment_targets(translated, len(segments))
        if targets is None:
            return None
        if any(
            not self._is_acceptable_chunk_translation(
                source_text,
                target,
                source_language,
                block_type,
            )
            for (_, source_text, block_type), target in zip(segments, targets, strict=True)
        ):
            return None
        if (
            self._translation_acceptance_issue(
                "\n\n".join(source_text for _, source_text, _ in segments),
                "\n\n".join(targets),
                source_language,
                self._segment_validation_block_type(segments),
            )
            is not None
        ):
            return None
        return targets

    def _tagged_segment_targets(
        self,
        translated: str,
        expected_count: int,
    ) -> list[str] | None:
        matches = list(self._SEGMENT_RE.finditer(translated.strip()))
        if len(matches) != expected_count:
            return None
        if [int(match.group("index")) for match in matches] != list(range(expected_count)):
            return None
        if self._SEGMENT_RE.sub("", translated).strip():
            return None
        return [match.group("body").strip() for match in matches]

    def _tagged_translation_has_collapsed_region(
        self,
        translated: str,
        segments: list[tuple[str, str, BlockType]],
    ) -> bool:
        targets = self._tagged_segment_targets(translated, len(segments))
        if targets is None:
            return False
        return any(
            not self._is_valid_chunk_translation_structure(source_text, target, block_type)
            for (_, source_text, block_type), target in zip(segments, targets, strict=True)
        )

    def _redistribute_logical_translation(
        self,
        logical_source: str,
        logical_target: str,
        segments: list[tuple[str, str, BlockType]],
        source_language: str | None,
        block_by_id: dict[str, Block],
        *,
        continuity_proven: bool,
    ) -> list[str] | None:
        if not continuity_proven:
            return None
        block_type = self._segment_validation_block_type(segments)
        if not self._is_acceptable_chunk_translation(
            logical_source,
            logical_target,
            source_language,
            block_type,
        ):
            return None

        continuous_source, source_boundaries = self._continuous_source_with_boundaries(segments)
        if self._normalized_whitespace(continuous_source) != self._normalized_whitespace(
            logical_source
        ):
            return None

        logical_target = self._normalized_whitespace(logical_target)
        target_tokens = list(re.finditer(r"\S+", logical_target))
        if len(target_tokens) < len(segments):
            return None
        source_weights = [
            max(1, len(self._language_words(source_text))) for _, source_text, _ in segments
        ]
        minimum_counts = [
            max(1, math.ceil(weight * 0.20)) if weight >= 6 else 1 for weight in source_weights
        ]
        if sum(minimum_counts) > len(target_tokens):
            return None

        desired_boundaries = self._sentence_aligned_target_boundaries(
            continuous_source,
            logical_target,
            source_boundaries,
            target_tokens,
        )
        if desired_boundaries is None:
            return None
        if len(segments) >= 2:
            previous_id, previous_source, _ = segments[-2]
            current_id, current_source, _ = segments[-1]
            previous_block = block_by_id.get(previous_id)
            current_block = block_by_id.get(current_id)
            if (
                previous_block is not None
                and current_block is not None
                and not previous_source.rstrip().endswith(("-", "\u00ad"))
                and len(self._language_words(current_source)) <= 4
                and self._is_explicit_cross_column_continuation(
                    previous_block,
                    current_block,
                    previous_source,
                    current_source,
                )
            ):
                # A compact tail such as "por etilo-dependencia" may expand
                # into several English words despite occupying a one-line box.
                # The passage is continuous, so keep only the validator's
                # minimum token count in that final physical region and place
                # the preceding English words in the larger prior region.
                desired_boundaries[-1] = len(target_tokens) - minimum_counts[-1]

        token_boundaries: list[int] = []
        previous_boundary = 0
        for index, desired in enumerate(desired_boundaries):
            minimum_boundary = previous_boundary + minimum_counts[index]
            maximum_boundary = len(target_tokens) - sum(minimum_counts[index + 1 :])
            boundary = max(minimum_boundary, min(desired, maximum_boundary))
            if boundary <= previous_boundary:
                return None
            token_boundaries.append(boundary)
            previous_boundary = boundary
        counts = [
            current - previous
            for previous, current in zip(
                [0, *token_boundaries],
                [*token_boundaries, len(target_tokens)],
                strict=True,
            )
        ]

        redistributed: list[str] = []
        token_index = 0
        for count in counts:
            start = 0 if token_index == 0 else target_tokens[token_index].start()
            token_index += count
            end = (
                len(logical_target)
                if token_index == len(target_tokens)
                else target_tokens[token_index].start()
            )
            redistributed.append(logical_target[start:end].strip())

        if any(
            not self._is_valid_chunk_translation_structure(source_text, target, segment_type)
            for (_, source_text, segment_type), target in zip(
                segments,
                redistributed,
                strict=True,
            )
        ):
            return None
        if self._normalized_whitespace(" ".join(redistributed)) != logical_target:
            return None
        return redistributed

    def _continuous_source_with_boundaries(
        self,
        segments: list[tuple[str, str, BlockType]],
    ) -> tuple[str, list[int]]:
        text = ""
        boundaries: list[int] = []
        for index, (_, source_text, _) in enumerate(segments):
            source_text = self._normalized_whitespace(source_text)
            if index == 0:
                text = source_text
                continue
            if text.endswith(("-", "\u00ad")) and re.match(
                r"[^\W\d_]",
                source_text,
                flags=re.UNICODE,
            ):
                text = text[:-1]
                boundaries.append(len(text))
                text += source_text
            else:
                text += " "
                boundaries.append(len(text))
                text += source_text
        return text, boundaries

    def _sentence_aligned_target_boundaries(
        self,
        source: str,
        target: str,
        source_boundaries: list[int],
        target_tokens: list[re.Match[str]],
    ) -> list[int] | None:
        source_sentences = self._sentence_spans(source)
        target_sentences = self._sentence_spans(target)
        if not source_sentences or len(source_sentences) != len(target_sentences):
            return None

        desired: list[int] = []
        prior = 0
        for boundary in source_boundaries:
            sentence_index = next(
                (
                    index
                    for index, (start, end) in enumerate(source_sentences)
                    if start <= boundary <= end
                ),
                None,
            )
            if sentence_index is None:
                return None
            source_start, source_end = source_sentences[sentence_index]
            target_start, target_end = target_sentences[sentence_index]
            sentence_token_indexes = [
                index
                for index, token in enumerate(target_tokens)
                if token.start() >= target_start and token.end() <= target_end
            ]
            if not sentence_token_indexes:
                return None
            if boundary >= source_end:
                candidate = sentence_token_indexes[-1] + 1
            else:
                source_prefix = source[source_start:boundary]
                source_sentence = source[source_start:source_end]
                prefix_letters = sum(character.isalpha() for character in source_prefix)
                sentence_letters = sum(character.isalpha() for character in source_sentence)
                if sentence_letters <= 0:
                    return None
                local_count = math.floor(
                    len(sentence_token_indexes) * prefix_letters / sentence_letters + 0.5
                )
                candidate = sentence_token_indexes[0] + local_count
            if candidate < prior:
                return None
            desired.append(candidate)
            prior = candidate
        return desired

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for sentence in self._split_into_sentences(text):
            start = text.find(sentence, cursor)
            if start < 0:
                return []
            end = start + len(sentence)
            spans.append((start, end))
            cursor = end
        return spans

    def _normalized_whitespace(self, text: str) -> str:
        return " ".join(text.strip().split())

    def _segment_validation_block_type(
        self,
        segments: list[tuple[str, str, BlockType]],
    ) -> BlockType | None:
        block_types = {block_type for _, _, block_type in segments}
        return next(iter(block_types)) if len(block_types) == 1 else BlockType.PARAGRAPH

    def _segment_application_chunks(
        self,
        chunk: TranslationChunk,
        segments: list[tuple[str, str, BlockType]],
        targets: list[str],
        block_by_id: dict[str, Block],
        *,
        validation_issue: str | None = None,
    ) -> list[TranslationChunk]:
        application_chunks: list[TranslationChunk] = []
        for index, ((block_id, source_text, _), target) in enumerate(
            zip(segments, targets, strict=True)
        ):
            block = block_by_id.get(block_id)
            page_number = block.page_number if block is not None else None
            application_chunks.append(
                chunk.model_copy(
                    update={
                        "id": f"{chunk.id}-block{index + 1:02d}",
                        "block_ids": [block_id],
                        "source_text": source_text,
                        "translated_text": target,
                        "placement_group_id": chunk.id,
                        "placement_index": index,
                        "placement_count": len(segments),
                        "source_token_count": self._token_count(source_text),
                        "page_start": page_number,
                        "page_end": page_number,
                        "continues_from_previous_page": bool(
                            block and block.metadata.get(CONTINUES_FROM_PREVIOUS_PAGE)
                        ),
                        "continues_to_next_page": bool(
                            block and block.metadata.get(CONTINUES_TO_NEXT_PAGE)
                        ),
                    }
                )
            )
        if validation_issue is not None:
            for application_chunk in application_chunks:
                self._mark_translation_failure(application_chunk, validation_issue)
        return application_chunks

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
        self._instruction_template_cache = None
        self._mlx_stream = None
        try:
            import mlx.core as mx

            mx.clear_cache()
            mx.clear_streams()
        except Exception as exc:
            logger.debug("MLX cleanup skipped: %s", exc)

    def _configure_mlx_thread(self) -> None:
        try:
            import mlx.core as mx

            mx.set_default_device(mx.gpu)
            if self._mlx_stream is None:
                self._mlx_stream = mx.new_stream(mx.gpu)
            mx.set_default_stream(self._mlx_stream)
        except Exception as exc:
            raise RuntimeError("Unable to configure the MLX Metal GPU runtime.") from exc

    def _configure_cpu_threads(self) -> None:
        """Bound CPU helper pools without consuming every efficiency core.

        MLX model operations stay on Metal. These variables only govern CPU-side
        tokenization and library helpers in the isolated translation process.
        Existing explicit environment choices are preserved.
        """

        configured = max(0, int(self.settings.cpu_threads or 0))
        threads = configured or self._recommended_cpu_threads()
        self._cpu_threads = max(1, threads)
        for variable in (
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(variable, str(self._cpu_threads))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    def _recommended_cpu_threads(self) -> int:
        logical_cores = max(1, int(os.cpu_count() or 1))
        performance_cores = 0
        if os.uname().sysname == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                performance_cores = int(result.stdout.strip())
            except (OSError, ValueError, subprocess.SubprocessError):
                performance_cores = 0
        preferred_pool = performance_cores or logical_cores
        if preferred_pool > 2:
            preferred_pool -= 1
        return max(1, min(6, preferred_pool))

    def _record_mlx_runtime(self) -> None:
        try:
            import mlx.core as mx

            device_info = mx.device_info()
            fast_sdpa = bool(
                hasattr(mx, "fast") and hasattr(mx.fast, "scaled_dot_product_attention")
            )
            if "gpu" not in str(mx.default_device()).casefold():
                raise RuntimeError("MLX default device is not the Metal GPU.")
            self._mlx_runtime = {
                "device": "metal_gpu",
                "device_name": str(device_info.get("device_name", "Apple GPU")),
                # MLX-LM's Qwen 3.5 attention implementation calls this fused
                # primitive. MLX selects the appropriate Metal attention kernel;
                # it does not expose a separate third-party FlashAttention flag.
                "attention_backend": (
                    "mlx.fast.scaled_dot_product_attention" if fast_sdpa else "model_default"
                ),
                "fast_attention_available": fast_sdpa,
                "cpu_threads": self._cpu_threads,
                "batch_size": max(1, int(self.settings.batch_size)),
                "batch_token_budget": max(1024, int(self.settings.batch_token_budget)),
            }
        except RuntimeError:
            raise
        except Exception as exc:
            logger.debug("Unable to record MLX runtime capabilities: %s", exc)
            self._mlx_runtime = {
                "device": "unknown",
                "attention_backend": "model_default",
                "fast_attention_available": False,
                "cpu_threads": self._cpu_threads,
                "batch_size": max(1, int(self.settings.batch_size)),
                "batch_token_budget": max(1024, int(self.settings.batch_token_budget)),
            }

    def runtime_metadata(self) -> dict[str, Any]:
        instruction_mode = "disabled"
        if self._instruction_template_cache is not None:
            instruction_mode = (
                "tokenized_prefix"
                if self._instruction_template_cache.token_composition_safe
                else "formatted_prefix"
            )
        generation = {
            "batch_calls": 0,
            "first_pass_requests": 0,
            "retry_requests": 0,
            "sequential_fallback_requests": 0,
            "batch_failures": 0,
            "batch_preparation_failures": 0,
            **dict(self._batch_stats),
        }
        return {
            **self._mlx_runtime,
            "instruction_cache": {
                "mode": instruction_mode,
                "hits": self._instruction_cache_hits,
                "misses": self._instruction_cache_misses,
            },
            "generation": generation,
        }

    def _build_translation_units(self, document: DocumentModel) -> list[TranslationUnit]:
        units: list[TranslationUnit] = []
        pending: list[Block] = []
        section_context = ""
        continuation_resolution = CrossPageContinuationResolver().resolve(document)
        document.metadata.translation["cross_page_continuation_group_count"] = len(
            continuation_resolution.groups
        )
        active_continuation_group_id: str | None = None

        for block in document.blocks:
            if self._is_marker_table_cell_block(block):
                continue
            if block.block_type == BlockType.FIGURE:
                block.metadata["excluded_from_translation"] = True
                block.metadata["translation_exclusion_reason"] = "figure_internal_text_preserved"
                continue
            if not block.text.strip():
                continue

            if (
                active_continuation_group_id is not None
                and continuation_resolution.is_intervening_for(
                    block.id,
                    active_continuation_group_id,
                )
            ):
                units.append(
                    TranslationUnit(
                        [block.id],
                        block.text.strip(),
                        block.block_type,
                        section_context,
                    )
                )
                continue

            if block.block_type == BlockType.HEADING:
                self._flush_paragraph_unit(pending, units, section_context)
                pending = []
                active_continuation_group_id = None
                heading_text = block.text.strip()
                units.append(
                    TranslationUnit([block.id], heading_text, block.block_type, section_context)
                )
                section_context = heading_text
                continue

            if block.block_type != BlockType.PARAGRAPH:
                self._flush_paragraph_unit(pending, units, section_context)
                pending = []
                active_continuation_group_id = None
                units.append(
                    TranslationUnit(
                        [block.id], block.text.strip(), block.block_type, section_context
                    )
                )
                continue

            continuation_group = continuation_resolution.group_for(block.id)
            if continuation_group is not None:
                continuation_index = int(block.metadata.get(CONTINUATION_INDEX, 0))
                if continuation_index == 0:
                    if pending and not self._belongs_to_same_paragraph(pending[-1], block):
                        self._flush_paragraph_unit(pending, units, section_context)
                        pending = []
                    pending.append(block)
                    active_continuation_group_id = continuation_group.id
                elif active_continuation_group_id == continuation_group.id:
                    pending.append(block)
                else:
                    self._flush_paragraph_unit(pending, units, section_context)
                    pending = [block]
                if continuation_index == len(continuation_group.block_ids) - 1:
                    active_continuation_group_id = None
                continue

            if pending and not self._belongs_to_same_paragraph(pending[-1], block):
                self._flush_paragraph_unit(pending, units, section_context)
                pending = []
            pending.append(block)

        self._flush_paragraph_unit(pending, units, section_context)
        self._append_table_units(document, units, section_context)
        return units

    def _append_table_units(
        self, document: DocumentModel, units: list[TranslationUnit], context: str
    ) -> None:
        for table_index, table in enumerate(document.tables, start=1):
            table_debug = getattr(table, "debug", {})
            if table_debug.get("render_from_block_text") or table_debug.get("marker_block_id"):
                # Marker table HTML is already represented by a table Block and translated as
                # one HTML table chunk. Adding per-row TableModel chunks duplicates work and can
                # overwrite cells with prompt text if a row translation fails.
                continue
            table_context = (
                f"{context}\nTable {table_index}\n"
                f"Preserve the delimiter token exactly as written: |||CELL_BREAK||| . "
                "Return the same number of cells in the same order."
            ).strip()
            if table.headers:
                units.append(
                    TranslationUnit(
                        [f"{self.TABLE_HEADER_PREFIX}{table.id}"],
                        self.TABLE_DELIMITER.join(cell.strip() for cell in table.headers),
                        BlockType.TABLE,
                        table_context,
                    )
                )
            for row_index, row in enumerate(table.rows):
                units.append(
                    TranslationUnit(
                        [f"{self.TABLE_ROW_PREFIX}{table.id}:{row_index}"],
                        self.TABLE_DELIMITER.join(cell.strip() for cell in row),
                        BlockType.TABLE,
                        table_context,
                    )
                )

    def _flush_paragraph_unit(
        self, blocks: list[Block], units: list[TranslationUnit], context: str
    ) -> None:
        if not blocks:
            return
        text = self._join_paragraph_lines([block.text for block in blocks])
        if text:
            unit_type = (
                BlockType.TABLE if self._is_table_heavy_markup(text) else BlockType.PARAGRAPH
            )
            if unit_type == BlockType.TABLE:
                text = self._normalize_table_markup_for_translation(text)
            units.append(TranslationUnit([block.id for block in blocks], text, unit_type, context))

    def _belongs_to_same_paragraph(self, previous: Block, current: Block) -> bool:
        if previous.page_number != current.page_number:
            return False
        if previous.bbox is None or current.bbox is None:
            return False

        previous_col = self._column(previous)
        current_col = self._column(current)
        if previous_col != current_col:
            return False

        vertical_gap = current.bbox.y0 - previous.bbox.y1
        previous_size = float(previous.style_hints.get("font_size", 10) or 10)
        current_size = float(current.style_hints.get("font_size", 10) or 10)
        if abs(previous_size - current_size) > 2:
            return False
        if vertical_gap < -1 or vertical_gap > max(7.5, previous_size * 0.9):
            return False

        previous_text = previous.text.rstrip()
        current_text = current.text.lstrip()
        if previous_text.endswith((".", "!", "?", ":", ";")) and current_text[:1].isupper():
            return False
        return True

    def _column(self, block: Block) -> int:
        if block.bbox is None:
            return 0
        return 0 if block.bbox.x0 < 300 else 1

    def _join_paragraph_lines(self, lines: list[str]) -> str:
        text = ""
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if not text:
                text = line
            elif text.endswith("-"):
                text = text[:-1] + line
            else:
                text += " " + line
        return " ".join(text.split())

    def _apply_translation_to_target(
        self,
        chunk: TranslationChunk,
        block_by_id: dict[str, Block],
        table_by_id: dict[str, object],
    ) -> None:
        if not chunk.block_ids:
            return
        target_id = chunk.block_ids[0]
        if target_id.startswith(self.TABLE_HEADER_PREFIX):
            table_id = target_id.removeprefix(self.TABLE_HEADER_PREFIX)
            table = table_by_id.get(table_id)
            if table is None:
                return
            table.headers = self._split_table_translation(
                chunk.translated_text, chunk.source_text, len(table.headers)
            )
            return
        if target_id.startswith(self.TABLE_ROW_PREFIX):
            suffix = target_id.removeprefix(self.TABLE_ROW_PREFIX)
            table_id, _, row_index_text = suffix.partition(":")
            table = table_by_id.get(table_id)
            if table is None or not row_index_text.isdigit():
                return
            row_index = int(row_index_text)
            if row_index >= len(table.rows):
                return
            translated_row = self._split_table_translation(
                chunk.translated_text,
                chunk.source_text,
                len(table.rows[row_index]),
            )
            table.rows[row_index] = translated_row
            if row_index < len(table.cells):
                table.cells[row_index] = [
                    cell.model_copy(
                        update={
                            "text": translated_row[idx] if idx < len(translated_row) else cell.text
                        }
                    )
                    for idx, cell in enumerate(table.cells[row_index])
                ]
            return

        first = block_by_id.get(chunk.block_ids[0])
        if first is None:
            return

        if chunk.status == self.TRANSLATION_FAILED_STATUS:
            validation = {
                "status": chunk.status,
                "reason": chunk.reason,
                "warnings": list(chunk.warnings),
            }
            for block_id in chunk.block_ids:
                target_block = block_by_id.get(block_id)
                if target_block is not None:
                    target_block.metadata["translation_validation"] = validation

        first.metadata.setdefault("source_text", first.text)
        if chunk.placement_group_id is not None:
            first.metadata["translation_placement_group_id"] = chunk.placement_group_id
            first.metadata["translation_placement_index"] = chunk.placement_index
            first.metadata["translation_placement_count"] = chunk.placement_count

        if chunk.chunk_type == "keywords" and len(chunk.block_ids) >= 2:
            body = block_by_id.get(chunk.block_ids[1])
            heading_text, separator, body_text = chunk.translated_text.strip().partition("\n")
            if separator and body is not None:
                body.metadata.setdefault("source_text", body.text)
                first.text = heading_text.strip()
                body.text = body_text.strip()
                first.metadata["translated_from_block_ids"] = chunk.block_ids
                body.metadata["translated_from_block_ids"] = chunk.block_ids
                for block_id in chunk.block_ids[2:]:
                    block = block_by_id.get(block_id)
                    if block is not None:
                        block.metadata.setdefault("source_text", block.text)
                        block.text = ""
                        block.metadata["merged_into_block_id"] = body.id
                return

        first.text = (
            self._clean_translated_list_text(chunk.translated_text)
            if first.block_type == BlockType.LIST
            else chunk.translated_text.strip()
        )
        first.metadata["translated_from_block_ids"] = chunk.block_ids
        for block_id in chunk.block_ids[1:]:
            block = block_by_id.get(block_id)
            if block is not None:
                block.metadata.setdefault("source_text", block.text)
                block.text = ""
                block.metadata["merged_into_block_id"] = first.id

    def _coalesce_translated_chunks(self, chunks: list[TranslationChunk]) -> list[TranslationChunk]:
        out: list[TranslationChunk] = []
        index = 0
        while index < len(chunks):
            chunk = chunks[index]
            if not chunk.block_ids or self._is_table_target(chunk.block_ids[0]):
                out.append(chunk)
                index += 1
                continue

            block_ids = tuple(chunk.block_ids)
            group = [chunk]
            index += 1
            while index < len(chunks) and tuple(chunks[index].block_ids) == block_ids:
                group.append(chunks[index])
                index += 1

            if len(group) == 1:
                out.append(chunk)
                continue

            failed_chunks = [
                item for item in group if item.status == self.TRANSLATION_FAILED_STATUS
            ]
            warnings = list(dict.fromkeys(warning for item in group for warning in item.warnings))
            out.append(
                chunk.model_copy(
                    update={
                        "id": group[0].id,
                        "source_text": "\n\n".join(
                            item.source_text.strip() for item in group if item.source_text.strip()
                        ),
                        "translated_text": "\n\n".join(
                            item.translated_text.strip()
                            for item in group
                            if item.translated_text.strip()
                        ),
                        "status": (
                            self.TRANSLATION_FAILED_STATUS if failed_chunks else group[0].status
                        ),
                        "reason": (failed_chunks[0].reason if failed_chunks else group[0].reason),
                        "warnings": warnings,
                    }
                )
            )
        return out

    def _is_table_target(self, target_id: str) -> bool:
        return target_id.startswith(self.TABLE_HEADER_PREFIX) or target_id.startswith(
            self.TABLE_ROW_PREFIX
        )

    def _is_marker_table_cell_block(self, block: Block) -> bool:
        return str((block.metadata or {}).get("marker_block_type", "")).lower() == "tablecell"

    def _split_table_translation(
        self, translated_text: str, source_text: str, expected_cells: int
    ) -> list[str]:
        parts = [part.strip() for part in translated_text.split(self.TABLE_DELIMITER)]
        if len(parts) == expected_cells:
            return parts
        source_parts = [part.strip() for part in source_text.split(self.TABLE_DELIMITER)]
        if len(source_parts) != expected_cells:
            source_parts = (source_parts + [""] * expected_cells)[:expected_cells]
        if expected_cells == 1:
            return [translated_text.strip()]
        return source_parts

    def _clean_translated_list_text(self, text: str) -> str:
        return re.sub(r"^\s*[-*+]\s+", "", text.strip())

    def _split_to_token_budget(self, text: str) -> list[str]:
        token_budget = max(128, int(self.settings.chunk_size or DEFAULT_CHUNK_SIZE))
        token_budget = min(token_budget, self.PROSE_CHUNK_TOKEN_CAP)
        if self._token_count(text) <= token_budget:
            return [text]

        sentences = self._split_into_sentences(text)
        if len(sentences) == 1:
            return self._split_long_sentence(sentences[0], token_budget)

        parts: list[str] = []
        current = ""
        for sentence in sentences:
            if self._token_count(sentence) > token_budget:
                if current:
                    parts.append(current)
                    current = ""
                parts.extend(self._split_long_sentence(sentence, token_budget))
                continue
            candidate = f"{current} {sentence}".strip()
            if current and self._token_count(candidate) > token_budget:
                parts.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            parts.append(current)
        return parts or [text]

    def _is_batchable_block_type(self, block_type: BlockType | None) -> bool:
        return block_type == BlockType.PARAGRAPH

    def _merge_adjacent_translation_chunks(
        self,
        chunks: list[TranslationChunk],
        chunk_block_types: dict[str, BlockType] | None = None,
        *,
        document: DocumentModel | None = None,
    ) -> list[TranslationChunk]:
        group_size = max(1, int(self.settings.chunk_group_size or 1))
        if group_size <= 1:
            return chunks

        merged: list[TranslationChunk] = []
        index = 0
        max_group_tokens = max(256, int(self.settings.max_tokens * 0.75))
        while index < len(chunks):
            chunk = chunks[index]
            block_type = (chunk_block_types or {}).get(chunk.id)
            if not self._can_merge_translation_chunk(chunk, block_type):
                merged.append(chunk)
                index += 1
                continue

            group = [chunk]
            group_token_count = int(
                chunk.source_token_count or self._token_count(chunk.source_text)
            )
            index += 1
            while index < len(chunks) and len(group) < group_size:
                candidate = chunks[index]
                candidate_type = (chunk_block_types or {}).get(candidate.id)
                candidate_tokens = int(
                    candidate.source_token_count or self._token_count(candidate.source_text)
                )
                if (
                    not self._can_merge_translation_chunk(candidate, candidate_type)
                    or candidate.context != chunk.context
                    or candidate.source_language != chunk.source_language
                    or candidate.source_language_origin != chunk.source_language_origin
                    or not self._chunks_share_safe_layout_region(group[-1], candidate, document)
                    or (group_token_count + candidate_tokens) > max_group_tokens
                ):
                    break
                group.append(candidate)
                group_token_count += candidate_tokens
                index += 1

            if len(group) == 1:
                merged.append(chunk)
                continue

            merged.append(
                TranslationChunk(
                    id=group[0].id,
                    block_ids=self._unique_block_ids(
                        [block_id for item in group for block_id in item.block_ids]
                    ),
                    source_text="\n\n".join(
                        item.source_text.strip() for item in group if item.source_text.strip()
                    ),
                    context=group[0].context,
                    source_language=group[0].source_language,
                    source_language_origin=group[0].source_language_origin,
                    source_language_confidence=min(
                        (
                            confidence
                            for item in group
                            if (confidence := item.source_language_confidence) is not None
                        ),
                        default=None,
                    ),
                    source_token_count=group_token_count,
                    page_start=group[0].page_start,
                    page_end=group[0].page_end,
                )
            )
        return merged

    def _chunks_share_safe_layout_region(
        self,
        previous: TranslationChunk,
        current: TranslationChunk,
        document: DocumentModel | None,
    ) -> bool:
        if (
            previous.page_start is None
            or previous.page_end is None
            or current.page_start is None
            or current.page_end is None
        ):
            return False
        if not (previous.page_start == previous.page_end == current.page_start == current.page_end):
            return False
        if previous.block_ids == current.block_ids:
            # Token-budget fragments from the same source region remain safe to
            # coalesce into one translation request.
            return True
        if document is None:
            return True

        positions = {block.id: index for index, block in enumerate(document.blocks)}
        previous_positions = [
            positions[block_id] for block_id in previous.block_ids if block_id in positions
        ]
        current_positions = [
            positions[block_id] for block_id in current.block_ids if block_id in positions
        ]
        if not previous_positions or not current_positions:
            return False
        previous_end = max(previous_positions)
        current_start = min(current_positions)
        if current_start <= previous_end:
            return False
        return not any(
            block.block_type in {BlockType.FIGURE, BlockType.EQUATION}
            for block in document.blocks[previous_end + 1 : current_start]
        )

    def _unique_block_ids(self, block_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for block_id in block_ids:
            if block_id in seen:
                continue
            seen.add(block_id)
            unique.append(block_id)
        return unique

    def _can_merge_translation_chunk(
        self, chunk: TranslationChunk, block_type: BlockType | None
    ) -> bool:
        if not chunk.block_ids or self._is_table_target(chunk.block_ids[0]):
            return False
        if self._is_table_heavy_markup(chunk.source_text):
            return False
        if not self._has_translatable_content(chunk.source_text):
            return False
        return self._is_batchable_block_type(block_type)

    def _has_translatable_content(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if re.fullmatch(r"(?:[-*_]\s*){3,}", stripped):
            return False
        if re.fullmatch(r"[^\wÀ-ÖØ-öø-ÿ]+", stripped):
            return False
        return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", stripped))

    def _split_into_sentences(self, text: str) -> list[str]:
        compact = " ".join(text.strip().split())
        if not compact:
            return []

        sentences: list[str] = []
        start = 0
        idx = 0
        while idx < len(compact):
            marker = compact[idx]
            if marker not in ".!?":
                idx += 1
                continue

            if marker == "." and idx > 0 and idx + 1 < len(compact):
                if compact[idx - 1].isdigit() and compact[idx + 1].isdigit():
                    idx += 1
                    continue

            token = self._token_before_index(compact, idx)
            if marker == "." and self._is_abbreviation_token(token):
                idx += 1
                continue

            # If punctuation is followed by a quote/bracket, skip it when checking the boundary.
            after = idx + 1
            while after < len(compact) and compact[after] in "'\"”’)]}":
                after += 1

            if after < len(compact) and not compact[after].isspace():
                idx += 1
                continue

            while after < len(compact) and compact[after].isspace():
                after += 1

            if after < len(compact) and compact[after].islower():
                idx += 1
                continue

            sentence = compact[start:after].strip()
            if sentence:
                sentences.append(sentence)
            start = after
            idx = after

        tail = compact[start:].strip()
        if tail:
            sentences.append(tail)
        return sentences or [compact]

    def _split_long_sentence(self, sentence: str, token_budget: int) -> list[str]:
        if self._token_count(sentence) <= token_budget:
            return [sentence]

        clause_segments = re.split(r"(?<=[;:])\s+|(?<=,)\s+(?=[A-Z0-9(])", sentence)
        if len(clause_segments) > 1:
            packed: list[str] = []
            current = ""
            for clause in clause_segments:
                candidate = f"{current} {clause}".strip()
                if current and self._token_count(candidate) > token_budget:
                    packed.append(current)
                    current = clause
                else:
                    current = candidate
            if current:
                packed.append(current)
            if packed and all(self._token_count(part) <= token_budget for part in packed):
                return packed

        # Last resort: split by words, but only if we cannot keep the sentence whole.
        words = sentence.split()
        if not words:
            return [sentence]
        parts: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and self._token_count(candidate) > token_budget:
                parts.append(current)
                current = word
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts

    def _token_before_index(self, text: str, index: int) -> str:
        start = index
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        return text[start:index]

    def _is_abbreviation_token(self, token: str) -> bool:
        lowered = token.strip().lower()
        if not lowered:
            return False
        if lowered in self._SENTENCE_ABBREVIATIONS:
            return True
        if len(lowered) == 1 and lowered.isalpha():
            return True
        return bool(re.fullmatch(r"(?:[a-z]\.){2,}[a-z]?", lowered))

    def _token_count(self, text: str) -> int:
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _detect_text_language(self, text: str) -> str | None:
        compact = self._text_for_language_detection(text)
        if len(compact) < 24:
            return None
        language, _ = self._detect_language_with_confidence(compact)
        return language

    def _is_already_english(self, chunk: TranslationChunk) -> bool:
        text = chunk.source_text.strip()
        if not text or self._is_nontranslatable_identifier(text):
            return True
        # Mixed-language papers commonly use standard English section labels
        # even when the surrounding document language is non-English.
        if self._looks_like_english_text(text):
            return True
        source_language = self._base_language(chunk.source_language)
        if source_language == "en":
            if self._source_language_is_authoritative(chunk):
                return True
            return False
        if source_language is not None:
            # Document-level language can leak onto short OCR chunks. Override it
            # only when the chunk itself is long enough for a high-confidence call.
            return self._is_confident_english_source(text)
        return self._looks_like_english_text(text)

    def _source_language_is_authoritative(self, chunk: TranslationChunk) -> bool:
        if chunk.source_language_origin not in {"block", "nearby_context"}:
            return False
        if chunk.source_language_confidence is None:
            return False
        threshold = (
            self._ENGLISH_SOURCE_CONFIDENCE
            if self._base_language(chunk.source_language) == "en"
            else self._NON_ENGLISH_OUTPUT_CONFIDENCE
        )
        return chunk.source_language_confidence >= threshold

    def _detect_language_with_confidence(self, text: str) -> tuple[str | None, float | None]:
        compact = self._text_for_language_detection(text)
        if not compact:
            return None, None
        if detect_langs is not None:
            try:
                predictions = detect_langs(compact)
                if predictions:
                    top = predictions[0]
                    return str(top.lang).lower(), float(top.prob)
            except Exception:
                pass
        try:
            return str(detect(compact)).lower(), None
        except Exception:
            return None, None

    def _is_confident_english_source(self, text: str) -> bool:
        compact = self._text_for_language_detection(text)
        words = self._language_words(compact)
        if len(words) < 3 or sum(len(word) for word in words) < 18:
            return False
        language, confidence = self._detect_language_with_confidence(compact)
        return (
            language == "en"
            and confidence is not None
            and confidence >= self._ENGLISH_SOURCE_CONFIDENCE
        )

    def _translation_acceptance_issue(
        self,
        source: str,
        translated: str,
        source_language: str | None,
        block_type: BlockType | None,
        *,
        source_language_authoritative: bool = False,
    ) -> str | None:
        structural_translation = self._structural_label_translation(
            source,
            source_language,
            block_type,
        )
        if structural_translation is not None and self._normalized_language_text(
            translated
        ) == self._normalized_language_text(structural_translation):
            return None
        if not self._source_requires_english_translation(
            source,
            source_language,
            block_type,
            source_language_authoritative=source_language_authoritative,
        ):
            return None

        source_normalized = self._normalized_language_text(source)
        translated_normalized = self._normalized_language_text(translated)
        if source_normalized and source_normalized == translated_normalized:
            if block_type == BlockType.TABLE and self._normalized_table_cell(
                source
            ) != self._normalized_table_cell(translated):
                return None
            return "translation_output_matches_source"
        if block_type != BlockType.TABLE and self._has_high_source_overlap(
            source_normalized,
            translated_normalized,
        ):
            return "translation_output_high_source_overlap"
        if self._looks_like_english_text(translated):
            return None
        translated_words = self._language_words(translated_normalized)
        if len(translated_words) <= 4 or sum(len(word) for word in translated_words) < 24:
            # Statistical language identifiers are unstable on short phrases
            # (for example, English "Sampling" is often labelled Tagalog).
            # Identity and source-overlap checks above remain deterministic.
            return None

        target_language, target_confidence = self._detect_language_with_confidence(
            translated_normalized
        )
        if (
            target_language is not None
            and target_language != "en"
            and target_confidence is not None
            and target_confidence >= self._NON_ENGLISH_OUTPUT_CONFIDENCE
        ):
            return "translation_output_not_english"
        return None

    def _source_requires_english_translation(
        self,
        source: str,
        source_language: str | None,
        block_type: BlockType | None,
        *,
        source_language_authoritative: bool = False,
    ) -> bool:
        if not self._is_substantive_language_text(source, block_type):
            return self._short_source_requires_english_translation(
                source,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            )
        if self._is_confident_english_source(source):
            return False

        normalized_source_language = self._base_language(source_language)
        if normalized_source_language == "en":
            return False
        if normalized_source_language is not None:
            return True

        detected_language, confidence = self._detect_language_with_confidence(source)
        return (
            detected_language is not None
            and detected_language != "en"
            and confidence is not None
            and confidence >= self._NON_ENGLISH_OUTPUT_CONFIDENCE
        )

    def _short_source_requires_english_translation(
        self,
        source: str,
        source_language: str | None,
        block_type: BlockType | None,
        *,
        source_language_authoritative: bool = False,
    ) -> bool:
        if block_type not in {
            BlockType.CAPTION,
            BlockType.FOOTNOTE,
            BlockType.HEADER,
            BlockType.HEADING,
            BlockType.LIST,
            BlockType.PARAGRAPH,
            BlockType.TABLE,
        }:
            return False
        source_language = self._base_language(source_language)
        if source_language is None or source_language == "en":
            return False
        if self._is_nontranslatable_identifier(source):
            return False

        compact = self._text_for_language_detection(html.unescape(source))
        words = self._language_words(compact)
        alpha_count = sum(len(word) for word in words)
        if block_type == BlockType.TABLE and not self._table_cell_requires_translation(compact):
            # Statistical cells frequently contain only compact study-defined
            # codes (for example MzFa / FzMb) and standard abbreviations. They
            # are intentionally preserved even when a language detector assigns
            # the surrounding document language to the cell.
            return False
        if block_type == BlockType.HEADING:
            has_enough_language = len(words) >= 1 and alpha_count >= 6
        elif block_type == BlockType.TABLE:
            # A short label can be the only source-language content left in an
            # otherwise English table, so validate cells independently.
            # Numeric cells and compact scientific abbreviations remain below
            # this threshold and may stay verbatim.
            has_enough_language = len(words) >= 1 and alpha_count >= 5
        elif block_type in {BlockType.CAPTION, BlockType.FOOTNOTE, BlockType.HEADER}:
            has_enough_language = len(words) >= 2 and alpha_count >= 8
        elif block_type == BlockType.PARAGRAPH:
            # OCR sometimes labels a short section heading as body text. A
            # high-confidence single non-English word is still translatable.
            has_enough_language = (len(words) >= 2 and alpha_count >= 10) or (
                len(words) == 1 and alpha_count >= 8
            )
        else:
            has_enough_language = len(words) >= 3 and alpha_count >= 12
        if not has_enough_language:
            return False
        if self._looks_like_name_or_citation(compact, words):
            return False
        if self._looks_like_english_text(compact):
            return False

        if source_language_authoritative:
            return True

        detected_language, confidence = self._detect_language_with_confidence(compact)
        return bool(
            detected_language is not None
            and detected_language != "en"
            and confidence is not None
            and confidence >= self._NON_ENGLISH_OUTPUT_CONFIDENCE
        )

    def _is_substantive_language_text(
        self,
        text: str,
        block_type: BlockType | None,
    ) -> bool:
        if block_type in {BlockType.EQUATION, BlockType.REFERENCE}:
            return False
        if self._is_nontranslatable_identifier(text):
            return False

        compact = self._text_for_language_detection(html.unescape(text))
        words = self._language_words(compact)
        alpha_count = sum(len(word) for word in words)
        if block_type == BlockType.TABLE:
            # Scientific tables often contain more delimiters, numbers, and
            # empty cells than letters. Judge the labels themselves instead of
            # letting table syntax dilute the language signal.
            return len(words) >= 4 and alpha_count >= 22
        visible_count = len(re.sub(r"\s+", "", compact))
        if not words or visible_count == 0 or alpha_count / visible_count < 0.50:
            return False
        if self._looks_like_name_or_citation(compact, words):
            return False

        if block_type == BlockType.HEADING:
            return len(words) >= 8 and alpha_count >= 50
        return (len(words) >= 6 and alpha_count >= 30) or (len(words) >= 4 and alpha_count >= 45)

    def _is_nontranslatable_identifier(self, text: str) -> bool:
        candidate = html.unescape(text).strip().strip("<>()[]{}.,;")
        return bool(
            re.fullmatch(r"(?is)(?:https?://|ftp://|www\.)\S+", candidate)
            or re.fullmatch(r"(?i)[^\s@]+@[^\s@]+\.[^\s@]+", candidate)
        )

    def _looks_like_name_or_citation(self, text: str, words: list[str]) -> bool:
        if re.search(r"(?i)\b(?:doi|isbn|issn)\s*(?::|/|\d)", text):
            return True
        if re.search(r"\b(?:19|20)\d{2}\b", text) and re.search(r"(?i)\bet\s+al\.?\b", text):
            return True
        if self._looks_like_bibliographic_locator(text):
            return True
        if self._looks_like_multi_author_list(text):
            return True
        if self._looks_like_single_author_name(text):
            return True
        if self._looks_like_contact_metadata(text, words):
            return True
        original_words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
        capitalized = sum(1 for word in original_words if word[:1].isupper())
        if (
            re.match(r"^\s*\d+\s+", text)
            and len(original_words) <= 20
            and text.count(",") >= 1
            and original_words
            and capitalized / len(original_words) >= 0.60
        ):
            # Numbered author affiliations and institution addresses are names,
            # not prose. Rewording them can corrupt searchable organisation and
            # place names while an unchanged result is entirely valid.
            return True
        return bool(
            len(words) <= 12
            and text.count(",") >= 2
            and original_words
            and capitalized / len(original_words) >= 0.70
        )

    def _looks_like_single_author_name(self, text: str) -> bool:
        """Recognize a compact personal name followed by one or more initials."""

        candidate = html.unescape(text).strip()
        if not re.search(r"(?:^|\s)(?:[A-ZÀ-ÖØ-Þ]\.){1,4}$", candidate):
            return False
        name_part = re.sub(r"(?:^|\s)(?:[A-ZÀ-ÖØ-Þ]\.){1,4}$", "", candidate).strip()
        name_words = re.findall(r"[^\W\d_]+", name_part, flags=re.UNICODE)
        if not 1 <= len(name_words) <= 8:
            return False
        particles = {
            "da",
            "de",
            "del",
            "della",
            "der",
            "di",
            "dos",
            "du",
            "la",
            "le",
            "van",
            "von",
        }
        substantive = [word for word in name_words if word.casefold() not in particles]
        return bool(substantive) and all(word[:1].isupper() for word in substantive)

    def _looks_like_contact_metadata(self, text: str, words: list[str]) -> bool:
        """Recognize compact journal contact blocks without exempting ordinary prose."""
        if len(words) > 40 or re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", text) is None:
            return False

        signals = (
            re.search(
                r"(?i)\b(?:correspondence|correspondencia|correspondance|"
                r"korrespondenz|contact|contatto|contato|address|adresse|"
                r"direcci[oó]n|indirizzo|endere[cç]o|morada)\b",
                text,
            )
            is not None,
            re.search(
                r"(?i)\b(?:postal(?:\s+code)?|postcode|zip|c\.?\s*p\.?)\s*[:.-]?\s*\d{4,10}\b",
                text,
            )
            is not None,
            re.search(
                r"(?i)(?:^|[\s,;])(?:c/|r/|av\.?|str\.?|st\.?)\s*[^\s,;]",
                text,
            )
            is not None,
            re.search(r"(?i)\b(?:tel(?:ephone|[ée]fono)?|phone|fax|mobile)\b", text) is not None,
        )
        return sum(signals) >= 2

    def _looks_like_bibliographic_locator(self, text: str) -> bool:
        if re.search(r"\b(?:19|20)\d{2}\b", text) is None:
            return False
        locator_patterns = (
            r"(?i)\bvol(?:ume)?\.?\s*\d+",
            r"(?i)\b(?:n|no|num(?:ber)?|issue)\s*[.°º#]*\s*\d+",
            r"(?i)\b(?:p{1,2}|pag(?:e|es|ina|inas)?)\.?\s*\d+",
        )
        return sum(re.search(pattern, text) is not None for pattern in locator_patterns) >= 2

    def _looks_like_multi_author_list(self, text: str) -> bool:
        segments = [segment.strip() for segment in re.split(r"[,;]", text) if segment.strip()]
        if len(segments) < 3:
            return False
        author_segments = 0
        for segment in segments:
            initials = re.findall(
                r"(?<![^\W\d_])([^\W\d_])\.",
                segment,
                flags=re.UNICODE,
            )
            name_words = re.findall(r"[^\W\d_]{2,}", segment, flags=re.UNICODE)
            if (
                any(initial.isupper() for initial in initials)
                and any(word[:1].isupper() for word in name_words)
                and len(name_words) <= 9
            ):
                author_segments += 1
        if author_segments >= 3 and author_segments / len(segments) >= 0.70:
            return True

        # Some journals print full given names rather than initials. Require a
        # long comma-separated sequence whose substantive words all retain
        # personal-name capitalization, while allowing surname particles and a
        # final language-specific conjunction. This deliberately rejects prose
        # lists containing ordinary lower-case verbs or nouns.
        name_particles = {
            "and",
            "da",
            "de",
            "del",
            "della",
            "der",
            "di",
            "dos",
            "du",
            "e",
            "et",
            "la",
            "le",
            "und",
            "van",
            "von",
            "y",
        }
        full_name_segments = 0
        for segment in segments:
            name_words = re.findall(r"[^\W\d_]+", segment, flags=re.UNICODE)
            substantive = [word for word in name_words if word.casefold() not in name_particles]
            if (
                2 <= len(substantive) <= 6
                and len(name_words) <= 9
                and all(word[:1].isupper() for word in substantive)
            ):
                full_name_segments += 1
        return full_name_segments >= 4 and full_name_segments / len(segments) >= 0.75

    def _normalized_language_text(self, text: str) -> str:
        return " ".join(self._language_words(self._text_for_language_detection(text)))

    def _language_words(self, text: str) -> list[str]:
        return [
            word.casefold()
            for word in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
            if len(word) >= 2
        ]

    def _has_high_source_overlap(self, source: str, translated: str) -> bool:
        if not source or not translated:
            return False
        source_words = source.split()
        translated_words = translated.split()
        if not source_words or not translated_words:
            return False

        source_counts = Counter(source_words)
        translated_counts = Counter(translated_words)
        shared = sum((source_counts & translated_counts).values())
        source_coverage = shared / len(source_words)
        length_ratio = len(translated_words) / len(source_words)
        similarity = SequenceMatcher(None, source, translated).ratio()
        return similarity >= 0.92 or (source_coverage >= 0.90 and 0.75 <= length_ratio <= 1.35)

    def _mark_translation_failure(
        self,
        chunk: TranslationChunk,
        issue: str,
    ) -> None:
        warning_by_issue = {
            "translation_output_matches_source": (
                "Translation output matched substantive non-English source text after retry."
            ),
            "translation_output_high_source_overlap": (
                "Translation output retained too much substantive source-language text after retry."
            ),
            "translation_output_not_english": (
                "Translation output was confidently detected as non-English after retry."
            ),
            "translation_structure_invalid": (
                "Translation output did not preserve the required source structure after retry."
            ),
            "translation_table_structure_invalid": (
                "Translation output did not preserve the table's row, cell, section, or span structure after retry."
            ),
            "translation_table_cell_missing": (
                "Translation output omitted content from one or more source table cells after retry."
            ),
            "translation_identifier_missing": (
                "Translation output omitted a required source URL or email address after retry."
            ),
            "translation_source_acronym_missing": (
                "Translation output omitted or reinterpreted a required source acronym after retry."
            ),
            "translation_target_acronym_invented": (
                "Translation output introduced an acronym that was absent from the source after retry."
            ),
        }
        chunk.status = self.TRANSLATION_FAILED_STATUS
        chunk.reason = issue
        warning = warning_by_issue.get(issue, f"Translation validation failed: {issue}.")
        if warning not in chunk.warnings:
            chunk.warnings.append(warning)

    def _base_language(self, language: str | None) -> str | None:
        normalized = self._normalize_lang_code(language)
        return normalized.split("-", maxsplit=1)[0].lower() if normalized else None

    def _translate_chunk(
        self,
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        self._configure_mlx_thread()
        from mlx_lm import generate, sample_utils

        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            logger.warning("Translation requested before the MLX model was loaded.")
            return text
        prompt = self._build_prompt(text, context, source_language)
        sampler = self._make_sampler(sample_utils)
        logits_processors = self._make_logits_processors(sample_utils)
        prompt_for_generation: str | list[int] = prompt
        prompt_token_count: int | None = None
        try:
            encoded_prompt = self._encode_prompts([prompt])[0]
            prompt_for_generation = encoded_prompt
            prompt_token_count = len(encoded_prompt)
        except Exception as exc:
            logger.debug("Cached prompt tokenization unavailable; using MLX tokenizer: %s", exc)
        max_tokens = force_max_tokens or (
            self._estimated_output_tokens_from_count(prompt_token_count)
            if prompt_token_count is not None
            else self._estimated_output_tokens(prompt)
        )
        try:
            out = generate(
                model,
                tokenizer,
                prompt=prompt_for_generation,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            )
            translated = str(out).strip()
            return self._postprocess_translated_text(translated)
        except Exception as exc:
            logger.warning(
                "Chunk translation failed; returning source text for this chunk: %s", exc
            )
            return text

    def _translate_requests_with_validation_batch(
        self,
        requests: list[_BatchTranslationRequest],
    ) -> list[str]:
        """Translate independent chunks in two ordered batch passes.

        The first pass covers every request. Only failed validations are placed
        in the retry pass. Rare specialised third attempts remain serial so the
        existing heading/contact recovery policy is unchanged.
        """

        if not requests:
            return []
        try:
            first_outputs = self._translate_requests_batch(requests, phase="first_pass")
        except Exception as exc:
            logger.warning(
                "Unable to prepare MLX batch prompts; translating %d request(s) sequentially: %s",
                len(requests),
                exc,
            )
            self._batch_stats["batch_preparation_failures"] += 1
            self._batch_stats["sequential_fallback_requests"] += len(requests)
            return [
                self._translate_chunk_with_validation(
                    request.text,
                    request.context,
                    request.source_language,
                    request.block_type,
                    source_language_authoritative=request.source_language_authoritative,
                )
                for request in requests
            ]
        results: list[str | None] = [None] * len(requests)
        retry_requests: list[_BatchTranslationRequest] = []
        retry_indexes: list[int] = []
        retry_contexts: dict[int, str] = {}

        for index, (request, translated) in enumerate(zip(requests, first_outputs, strict=True)):
            if self._is_acceptable_chunk_translation(
                request.text,
                translated,
                request.source_language,
                request.block_type,
                source_language_authoritative=request.source_language_authoritative,
            ):
                results[index] = translated
                continue
            retry_context = self._translation_retry_context(
                request.text,
                translated,
                request.context,
                request.block_type,
            )
            retry_indexes.append(index)
            retry_contexts[index] = retry_context
            retry_requests.append(
                _BatchTranslationRequest(
                    text=request.text,
                    context=retry_context,
                    source_language=request.source_language,
                    block_type=request.block_type,
                    source_language_authoritative=request.source_language_authoritative,
                )
            )

        if retry_requests:
            try:
                retry_outputs = self._translate_requests_batch(
                    retry_requests,
                    phase="retry",
                )
            except Exception as exc:
                logger.warning(
                    "Unable to prepare MLX retry batch; retrying %d request(s) sequentially: %s",
                    len(retry_requests),
                    exc,
                )
                self._batch_stats["batch_preparation_failures"] += 1
                self._batch_stats["sequential_fallback_requests"] += len(retry_requests)
                retry_outputs = [
                    self._translate_chunk(
                        request.text,
                        request.context,
                        request.source_language,
                    )
                    for request in retry_requests
                ]
            for result_index, retried in zip(
                retry_indexes,
                retry_outputs,
                strict=True,
            ):
                request = requests[result_index]
                results[result_index] = self._finish_translation_after_retry(
                    request.text,
                    retried,
                    retry_contexts[result_index],
                    request.source_language,
                    request.block_type,
                    source_language_authoritative=request.source_language_authoritative,
                    original_context=request.context,
                )

        return [
            result if result is not None else request.text
            for request, result in zip(requests, results, strict=True)
        ]

    def _translate_requests_batch(
        self,
        requests: list[_BatchTranslationRequest],
        *,
        phase: str,
    ) -> list[str]:
        prompts = [
            self._build_prompt(
                request.text,
                request.context,
                request.source_language,
            )
            for request in requests
        ]
        encoded_prompts = self._encode_prompts(prompts)
        prepared = [
            _PreparedBatchPrompt(
                request=request,
                prompt=prompt,
                prompt_tokens=tuple(prompt_tokens),
                max_tokens=self._estimated_output_tokens_from_count(len(prompt_tokens)),
            )
            for request, prompt, prompt_tokens in zip(
                requests,
                prompts,
                encoded_prompts,
                strict=True,
            )
        ]
        outputs: list[str] = []
        for batch in self._adaptive_prompt_batches(prepared):
            outputs.extend(self._generate_prepared_batch(batch, phase=phase))
        return outputs

    def _adaptive_prompt_batches(
        self,
        prompts: list[_PreparedBatchPrompt],
    ) -> list[list[_PreparedBatchPrompt]]:
        maximum_size = max(1, int(self.settings.batch_size or 1))
        token_budget = max(1024, int(self.settings.batch_token_budget or 1024))
        batches: list[list[_PreparedBatchPrompt]] = []
        current: list[_PreparedBatchPrompt] = []
        current_cost = 0
        for prompt in prompts:
            if current and (
                len(current) >= maximum_size or current_cost + prompt.token_cost > token_budget
            ):
                batches.append(current)
                current = []
                current_cost = 0
            current.append(prompt)
            current_cost += prompt.token_cost
        if current:
            batches.append(current)
        return batches

    def _generate_prepared_batch(
        self,
        batch: list[_PreparedBatchPrompt],
        *,
        phase: str,
    ) -> list[str]:
        self._configure_mlx_thread()
        try:
            from mlx_lm import batch_generate, sample_utils

            response = batch_generate(
                self._model,
                self._tokenizer,
                [list(prompt.prompt_tokens) for prompt in batch],
                max_tokens=[prompt.max_tokens for prompt in batch],
                sampler=self._make_sampler(sample_utils),
                logits_processors=self._make_logits_processors(sample_utils),
            )
            texts = list(response.texts)
            if len(texts) != len(batch):
                raise RuntimeError(
                    f"MLX batch returned {len(texts)} outputs for {len(batch)} prompts."
                )
            self._batch_stats["batch_calls"] += 1
            self._batch_stats[f"{phase}_requests"] += len(batch)
            self._batch_stats["maximum_observed_batch_size"] = max(
                self._batch_stats["maximum_observed_batch_size"],
                len(batch),
            )
            stats = getattr(response, "stats", None)
            if stats is not None:
                self._batch_stats["prompt_tokens"] += int(getattr(stats, "prompt_tokens", 0) or 0)
                self._batch_stats["generation_tokens"] += int(
                    getattr(stats, "generation_tokens", 0) or 0
                )
            return [self._postprocess_translated_text(str(text).strip()) for text in texts]
        except Exception as exc:
            logger.warning(
                "MLX %s batch generation failed; retrying %d request(s) sequentially: %s",
                phase,
                len(batch),
                exc,
            )
            self._batch_stats["batch_failures"] += 1
            self._batch_stats["sequential_fallback_requests"] += len(batch)
            return [
                self._translate_chunk(
                    prompt.request.text,
                    prompt.request.context,
                    prompt.request.source_language,
                    force_max_tokens=prompt.max_tokens,
                )
                for prompt in batch
            ]

    def _translate_chunk_with_validation(
        self,
        text: str,
        context: str,
        source_language: str | None,
        block_type: BlockType | None,
        *,
        source_language_authoritative: bool = False,
    ) -> str:
        structural_translation = self._structural_label_translation(
            text,
            source_language,
            block_type,
        )
        if structural_translation is not None:
            return structural_translation

        structural_caption = self._structural_caption_parts(
            text,
            source_language,
            block_type,
        )
        if structural_caption is not None:
            translated_prefix, caption_body = structural_caption
            translated_body = self._translate_chunk_with_validation(
                caption_body,
                (
                    f"{context}\n"
                    "TEXT is the natural-language body of a document caption. Translate it completely "
                    "without adding a table/figure label; that structural label is supplied separately."
                ).strip(),
                source_language,
                BlockType.CAPTION,
                source_language_authoritative=source_language_authoritative,
            )
            return f"{translated_prefix} {translated_body}".strip()

        translated = self._translate_chunk(text, context, source_language)
        if self._is_acceptable_chunk_translation(
            text,
            translated,
            source_language,
            block_type,
            source_language_authoritative=source_language_authoritative,
        ):
            return translated

        retry_context = self._translation_retry_context(
            text,
            translated,
            context,
            block_type,
        )
        retried = self._translate_chunk(text, retry_context, source_language)
        return self._finish_translation_after_retry(
            text,
            retried,
            retry_context,
            source_language,
            block_type,
            source_language_authoritative=source_language_authoritative,
            original_context=context,
        )

    def _translation_retry_context(
        self,
        text: str,
        translated: str,
        context: str,
        block_type: BlockType | None,
    ) -> str:
        missing_acronyms = self._missing_source_acronyms(text, translated, block_type)
        invented_acronyms = self._invented_target_acronyms(text, translated, block_type)
        missing_identifiers = self._missing_verbatim_identifiers(text, translated)
        verbatim_requirements = ""
        if missing_acronyms:
            verbatim_requirements += (
                " Required source acronyms that must appear unchanged in the English output: "
                f"{', '.join(missing_acronyms)}."
            )
        if missing_identifiers:
            verbatim_requirements += (
                " Required identifiers that must appear character-for-character unchanged: "
                f"{', '.join(missing_identifiers)}."
            )
        if invented_acronyms:
            verbatim_requirements += (
                " Remove these invented target acronyms because they are absent from TEXT: "
                f"{', '.join(invented_acronyms)}. Translate the source wording directly instead "
                "of importing terminology or abbreviations from nearby context."
            )
        return (
            f"{context}\n"
            "The previous output was not an acceptable English translation. Return English only and "
            "translate every substantive source-language phrase; do not repeat source-language prose. "
            "Preserve the source structure exactly: keep paragraph boundaries, list boundaries, headings, "
            "Markdown markers, citations, numeric values, and line breaks that "
            "separate logical blocks. Preserve every numeric value exactly. "
            f"Do not summarize, omit, or collapse content.{verbatim_requirements}"
        ).strip()

    def _finish_translation_after_retry(
        self,
        text: str,
        retried: str,
        retry_context: str,
        source_language: str | None,
        block_type: BlockType | None,
        *,
        source_language_authoritative: bool,
        original_context: str,
    ) -> str:
        if self._is_acceptable_chunk_translation(
            text,
            retried,
            source_language,
            block_type,
            source_language_authoritative=source_language_authoritative,
        ):
            return retried

        issue = self._chunk_translation_issue(
            text,
            retried,
            source_language,
            block_type,
            source_language_authoritative=source_language_authoritative,
        )
        if self._should_disambiguate_short_heading(
            text,
            source_language,
            block_type,
            issue,
        ):
            disambiguation_context = (
                f"{retry_context}\n"
                "The prior character-for-character copy of this short heading was rejected. "
                "Interpret TEXT in its stated source language and use the nearby source context to "
                "resolve its intended semantic category and grammatical number. Return a natural "
                "English heading that is not character-for-character identical to TEXT. If the usual "
                "English spelling would be identical, use a faithful context-appropriate English "
                "synonym instead. Return only the heading."
            )
            disambiguated = self._translate_chunk(
                text,
                disambiguation_context,
                source_language,
            )
            if self._is_acceptable_chunk_translation(
                text,
                disambiguated,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            ):
                return disambiguated
            retried = disambiguated
            issue = self._chunk_translation_issue(
                text,
                retried,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            )
            structural_fallback = self._structural_heading_fallback(original_context)
            if structural_fallback is not None and self._is_acceptable_chunk_translation(
                text,
                structural_fallback,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            ):
                logger.info(
                    "Used structural affiliation heading fallback after repeated homograph output."
                )
                return structural_fallback
        contact_retry_used = self._should_retry_contact_block(text, block_type, issue)
        if contact_retry_used:
            contact_context = (
                f"{retry_context}\n"
                "TEXT is a professional postal/contact block, not an identifier-only block. Preserve "
                "personal names, institution proper names, street/place names, postal codes, and email "
                "addresses exactly. Translate every translatable honorific, department or service name, "
                "organizational descriptor, country name, and contact label into natural English. Do not "
                "return TEXT unchanged, and return only the translated contact block."
            )
            contact_translation = self._translate_chunk(
                text,
                contact_context,
                source_language,
            )
            if self._is_acceptable_chunk_translation(
                text,
                contact_translation,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            ):
                return contact_translation
            retried = contact_translation
            issue = self._chunk_translation_issue(
                text,
                retried,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            )
            contact_fallback = self._spanish_contact_fallback(
                text,
                source_language,
            )
            if contact_fallback is not None:
                logger.info(
                    "Used deterministic Spanish contact-metadata fallback after repeated model rejection."
                )
                return contact_fallback
        if (
            block_type != BlockType.HEADING
            and not contact_retry_used
            and issue
            in {
                "translation_identifier_missing",
                "translation_output_high_source_overlap",
                "translation_output_matches_source",
                "translation_output_not_english",
                "translation_source_acronym_missing",
                "translation_target_acronym_invented",
            }
        ):
            invented_acronyms = self._invented_target_acronyms(
                text,
                retried,
                block_type,
            )
            invented_acronym_guidance = (
                " Remove these invented target acronyms because they do not occur in TEXT: "
                f"{', '.join(invented_acronyms)}."
                if invented_acronyms
                else ""
            )
            final_context = (
                f"{retry_context}\n"
                "A second attempt was also rejected. Produce a direct, clause-by-clause English "
                "translation now. Every source-language word or phrase with translatable meaning must "
                "be rendered in English. Retain only proper names and the exact required identifiers or "
                "acronyms from the preservation instructions."
                f"{invented_acronym_guidance} Return only the translation."
            )
            final_translation = self._translate_chunk(
                text,
                final_context,
                source_language,
            )
            if self._is_acceptable_chunk_translation(
                text,
                final_translation,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            ):
                return final_translation
            retried = final_translation
            issue = self._chunk_translation_issue(
                text,
                retried,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            )
        logger.warning(
            "Chunk translation failed validation after English-only retry (%s); returning source text.",
            issue or "unknown_validation_failure",
        )
        return text

    def _should_disambiguate_short_heading(
        self,
        source: str,
        source_language: str | None,
        block_type: BlockType | None,
        issue: str | None,
    ) -> bool:
        if (
            block_type != BlockType.HEADING
            or self._base_language(source_language) in {None, "en"}
            or issue
            not in {
                "translation_output_matches_source",
                "translation_output_high_source_overlap",
            }
        ):
            return False
        words = self._language_words(self._text_for_language_detection(source))
        return 1 <= len(words) <= 6 and sum(len(word) for word in words) <= 60

    def _structural_heading_fallback(self, context: str) -> str | None:
        numbered_lines = [
            line.strip() for line in context.splitlines() if re.match(r"^\s*\d+[.)]?\s+\S", line)
        ]
        if len(numbered_lines) < 2:
            return None
        if not all(
            self._looks_like_name_or_citation(line, self._language_words(line))
            for line in numbered_lines[:3]
        ):
            return None
        return "Affiliations"

    def _structural_label_translation(
        self,
        source: str,
        source_language: str | None,
        block_type: BlockType | None,
    ) -> str | None:
        """Translate unambiguous short document labels without model variance."""

        if block_type not in {
            BlockType.CAPTION,
            BlockType.HEADING,
            BlockType.PARAGRAPH,
        }:
            return None
        compact = " ".join(html.unescape(source).strip().split())
        table_or_figure = re.fullmatch(
            r"(?i)(tabla|figura)\s+([ivxlcdm]+|\d+)([.:]?)",
            compact,
        )
        if table_or_figure is not None:
            label = "TABLE" if table_or_figure.group(1).casefold() == "tabla" else "FIGURE"
            return f"{label} {table_or_figure.group(2)}{table_or_figure.group(3)}"

        normalized = "".join(
            character
            for character in self._normalized_language_text(compact)
            if character.isalnum() or character.isspace()
        )
        return self._SPANISH_STRUCTURAL_HEADINGS.get(" ".join(normalized.split()))

    def _structural_caption_parts(
        self,
        source: str,
        source_language: str | None,
        block_type: BlockType | None,
    ) -> tuple[str, str] | None:
        if block_type != BlockType.CAPTION:
            return None
        compact = " ".join(html.unescape(source).strip().split())
        match = re.fullmatch(
            r"(?is)(tabla|figura)\s+([ivxlcdm]+|\d+)([.:])\s*(.+)",
            compact,
        )
        if match is None:
            return None
        label = "TABLE" if match.group(1).casefold() == "tabla" else "FIGURE"
        return f"{label} {match.group(2)}{match.group(3)}", match.group(4).strip()

    def _should_retry_contact_block(
        self,
        source: str,
        block_type: BlockType | None,
        issue: str | None,
    ) -> bool:
        return bool(
            block_type in {BlockType.CAPTION, BlockType.FOOTNOTE, BlockType.PARAGRAPH}
            and issue
            in {
                "translation_identifier_missing",
                "translation_output_high_source_overlap",
                "translation_output_matches_source",
                "translation_output_not_english",
            }
            and re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", source)
        )

    def _spanish_contact_fallback(
        self,
        source: str,
        source_language: str | None,
    ) -> str | None:
        """Translate generic Spanish contact terms while preserving address names.

        Professional contact blocks are dominated by names and addresses, so a
        generic language detector can reject a correct English result—or the
        model can repeatedly copy the source. This conservative fallback only
        changes unambiguous titles, organizational descriptors, country names,
        and contact labels. It never rewrites the personal/institution/place
        names or verbatim identifiers that make the address usable.
        """

        if (
            self._base_language(source_language) != "es"
            or re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", source) is None
        ):
            return None
        translated = source
        replacements = (
            (r"(?i)\bDra\.", "Dr."),
            (r"(?i)\bDirecci[oó]n del autor\b", "Author Contact"),
            (r"(?i)\bServicio de\b", "Department of"),
            (r"(?i)\bDepartamento de\b", "Department of"),
            (r"(?i)\bUnidad de\b", "Unit of"),
            (r"(?i)\bEndocrinolog[ií]a\b", "Endocrinology"),
            (r"(?i)\bNutrici[oó]n\b", "Nutrition"),
            (r"(?i)\bEndocrinology\s+y\s+Nutrition\b", "Endocrinology and Nutrition"),
            (r"(?i)\bPab\.", "Pavilion"),
            (r"(?i)\bComplejo Hospitalario\b", "Hospital Complex"),
            (r"(?i)\bEspa[nñ]a\b", "Spain"),
            (r"(?i)\bCorreo electr[oó]nico\b", "Email"),
            (r"(?i)\bTel[eé]fono\b", "Telephone"),
        )
        for pattern, replacement in replacements:
            translated = re.sub(pattern, replacement, translated)
        if translated == source or self._missing_verbatim_identifiers(source, translated):
            return None
        return translated

    def _is_acceptable_chunk_translation(
        self,
        source: str,
        translated: str,
        source_language: str | None,
        block_type: BlockType | None,
        *,
        source_language_authoritative: bool = False,
    ) -> bool:
        return (
            self._chunk_translation_issue(
                source,
                translated,
                source_language,
                block_type,
                source_language_authoritative=source_language_authoritative,
            )
            is None
        )

    def _chunk_translation_issue(
        self,
        source: str,
        translated: str,
        source_language: str | None,
        block_type: BlockType | None,
        *,
        source_language_authoritative: bool = False,
    ) -> str | None:
        if not self._is_valid_chunk_translation_structure(source, translated, block_type):
            return "translation_structure_invalid"
        if self._missing_verbatim_identifiers(source, translated):
            return "translation_identifier_missing"
        if self._missing_source_acronyms(source, translated, block_type):
            return "translation_source_acronym_missing"
        if self._invented_target_acronyms(source, translated, block_type):
            return "translation_target_acronym_invented"
        if self.TABLE_DELIMITER in source:
            for source_cell, translated_cell in zip(
                source.split(self.TABLE_DELIMITER),
                translated.split(self.TABLE_DELIMITER),
                strict=True,
            ):
                issue = self._translation_acceptance_issue(
                    source_cell,
                    translated_cell,
                    source_language,
                    BlockType.TABLE,
                    source_language_authoritative=source_language_authoritative,
                )
                if issue is not None:
                    return issue
        if block_type == BlockType.TABLE and "|" in source:
            # Flattened OCR Markdown has no dependable line boundaries here,
            # but pipe and separator counts remain exact topology invariants.
            if source.count("|") != translated.count("|"):
                return "translation_table_structure_invalid"
            if source.count("---") != translated.count("---"):
                return "translation_table_structure_invalid"
        return self._translation_acceptance_issue(
            source,
            translated,
            source_language,
            block_type,
            source_language_authoritative=source_language_authoritative,
        )

    def _missing_verbatim_identifiers(self, source: str, translated: str) -> list[str]:
        identifiers = re.findall(
            r"(?i)(?:https?://|ftp://|www\.)[^\s<>()]+|[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+",
            html.unescape(source),
        )
        return [identifier for identifier in identifiers if identifier not in translated]

    def _source_acronyms_to_preserve(
        self,
        source: str,
        block_type: BlockType | None,
    ) -> list[str]:
        if block_type == BlockType.TABLE:
            # Compact medical/statistical table abbreviations may have a
            # standard English form (for example ACV -> CVA). Acronyms
            # explicitly introduced in parentheses remain document-defined
            # labels and must stay exact.
            return list(dict.fromkeys(self._ordered_acronyms(source)))

        visible = self._TAG_RE.sub(" ", html.unescape(source))
        acronyms = list(self._ordered_acronyms(visible))
        if re.search(r"[a-zà-öø-ÿ]", visible):
            for match in re.finditer(r"(?<![\w])([A-Z][A-Z0-9]{1,5})(?![\w])", visible):
                token = match.group(1)
                suffix = visible[match.end() : match.end() + 1]
                if (
                    suffix == "="
                    or re.fullmatch(r"[IVXLCDM]+", token)
                    or token not in self._document_defined_acronyms
                ):
                    continue
                acronyms.append(token)
        return list(dict.fromkeys(acronyms))

    def _missing_source_acronyms(
        self,
        source: str,
        translated: str,
        block_type: BlockType | None,
    ) -> list[str]:
        source_counts = Counter(self._source_acronyms_to_preserve(source, block_type))
        if not source_counts:
            return []
        translated_counts = Counter(
            re.findall(r"(?<![\w])([A-Z][A-Z0-9]{1,5})(?![\w])", html.unescape(translated))
        )
        return [
            acronym
            for acronym, required_count in source_counts.items()
            if translated_counts[acronym] < required_count
        ]

    def _invented_target_acronyms(
        self,
        source: str,
        translated: str,
        block_type: BlockType | None,
    ) -> list[str]:
        """Return stable-looking target acronyms that have no source evidence.

        Only parenthesized or slash-delimited target acronyms are considered.
        This avoids treating uppercase headings as acronyms while preventing
        the model from importing context-only expansions such as ``(TFM)`` into
        a narrow continuation block. Tables remain exempt because compact
        source abbreviations may legitimately have different English forms.
        """

        if block_type == BlockType.TABLE:
            return []
        source_visible = self._TAG_RE.sub(" ", html.unescape(source))
        source_acronyms = set(re.findall(r"(?<![\w])([A-Z][A-Z0-9]{1,5})(?![\w])", source_visible))
        return list(
            dict.fromkeys(
                acronym
                for acronym in self._ordered_acronyms(translated)
                if not re.fullmatch(r"[IVXLCDM]+", acronym) and acronym not in source_acronyms
            )
        )

    def _ordered_acronyms(self, text: str) -> list[str]:
        """Return acronyms whose surrounding syntax marks them as stable.

        Uppercase typography alone is not acronym evidence: headings and table
        labels commonly contain ordinary words such as ``HOMBRES`` or
        ``HOMMES``. Protect slash-delimited sequences and compact acronyms
        explicitly introduced in parentheses instead.
        """

        visible = self._TAG_RE.sub(" ", html.unescape(text))
        token_pattern = r"[A-Z][A-Z0-9]{1,4}"
        results: list[tuple[int, str]] = []
        occupied: list[tuple[int, int]] = []

        slash_pattern = re.compile(
            rf"(?<![\w])(?P<sequence>{token_pattern}(?:\s*/\s*{token_pattern})+)(?![\w])"
        )
        for sequence in slash_pattern.finditer(visible):
            occupied.append((sequence.start(), sequence.end()))
            for token in re.finditer(token_pattern, sequence.group("sequence")):
                results.append((sequence.start("sequence") + token.start(), token.group(0)))

        for group in re.finditer(r"\((?P<body>[^()]{0,120})\)", visible):
            body = group.group("body")
            if "=" in body:
                # Statistical labels such as (DT=11.2) may correctly change
                # to their English counterpart (SD=11.2).
                continue
            for token in re.finditer(rf"(?<![\w]){token_pattern}(?![\w])", body):
                absolute_start = group.start("body") + token.start()
                if any(start <= absolute_start < end for start, end in occupied):
                    continue
                results.append((absolute_start, token.group(0)))
        return [token for _, token in sorted(results)]

    def _is_valid_chunk_translation_structure(
        self,
        source: str,
        translated: str,
        block_type: BlockType | None,
    ) -> bool:
        source = source.strip()
        translated = translated.strip()
        if not source:
            return not translated
        if not translated:
            return False

        source_words = self._language_words(self._text_for_language_detection(source))
        source_alpha_count = sum(len(word) for word in source_words)
        if len(source_words) >= 3 and source_alpha_count >= 12:
            translated_words = self._language_words(self._text_for_language_detection(translated))
            translated_alpha_count = sum(len(word) for word in translated_words)
            minimum_target_words = (
                max(2, math.ceil(len(source_words) * 0.20)) if len(source_words) >= 6 else 1
            )
            minimum_target_alpha = (
                max(4, math.ceil(source_alpha_count * 0.12)) if len(source_words) >= 6 else 2
            )
            if (
                len(translated_words) < minimum_target_words
                or translated_alpha_count < minimum_target_alpha
            ):
                # Every independently placeable source region must retain some
                # language content. A grouped response can otherwise move all
                # meaning into a neighbour and leave this box as punctuation.
                return False

        if len(source) >= 200 and len(translated) < max(40, int(len(source) * 0.20)):
            return False

        source_paragraphs = self._paragraph_count(source)
        translated_paragraphs = self._paragraph_count(translated)
        if (
            block_type
            in {
                BlockType.CAPTION,
                BlockType.FOOTNOTE,
                BlockType.HEADER,
                BlockType.HEADING,
                BlockType.PARAGRAPH,
                BlockType.REFERENCE,
            }
            and translated_paragraphs != source_paragraphs
        ):
            return False
        if source_paragraphs >= 3 and translated_paragraphs < max(2, source_paragraphs // 2):
            return False

        source_list_items = self._markdown_list_item_count(source)
        if source_list_items and self._markdown_list_item_count(translated) < source_list_items:
            return False

        source_heading_count = self._markdown_heading_count(source)
        if source_heading_count and self._markdown_heading_count(translated) < source_heading_count:
            return False

        if self.TABLE_DELIMITER in source:
            source_cells = source.split(self.TABLE_DELIMITER)
            translated_cells = translated.split(self.TABLE_DELIMITER)
            if len(source_cells) != len(translated_cells):
                return False
            if any(
                self._is_significant_table_cell(source_cell) and not translated_cell.strip()
                for source_cell, translated_cell in zip(
                    source_cells,
                    translated_cells,
                    strict=True,
                )
            ):
                return False

        if block_type == BlockType.HEADING and "\n\n" in translated:
            return False
        return True

    def _paragraph_count(self, text: str) -> int:
        return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])

    def _markdown_list_item_count(self, text: str) -> int:
        return len(re.findall(r"(?m)^\s*[-*+]\s+\S", text))

    def _markdown_heading_count(self, text: str) -> int:
        return len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S", text))

    def _translate_table_markup_chunk(
        self, text: str, context: str, source_language: str | None
    ) -> str:
        text = self._normalize_table_markup_for_translation(text)
        max_tokens = self._table_output_budget(text)
        strict_context = (
            f"{context}\n"
            "TEXT contains an HTML table. Keep all HTML tags and attributes intact. "
            "Return a complete table with closing </table>. "
            f"{self._COMPACT_TABLE_TRANSLATION_GUIDANCE}"
        ).strip()
        translated = self._translate_chunk(
            text,
            strict_context,
            source_language,
            force_max_tokens=max_tokens,
        )
        translated = self._repair_common_spanish_table_labels(text, translated)
        if self._is_acceptable_table_translation(text, translated, source_language):
            return translated

        retry_context = (
            f"{strict_context}\n"
            "The previous output was not an acceptable English translation. Return English only, "
            "translate every substantive non-English table label or sentence, and do not repeat "
            "source-language prose. Do not truncate output. Keep the same number of rows and cells."
        )
        missing_acronyms = self._missing_source_acronyms(
            text,
            translated,
            BlockType.TABLE,
        )
        if missing_acronyms:
            retry_context += (
                " Preserve these document-defined acronyms character-for-character: "
                f"{', '.join(missing_acronyms)}."
            )
        retried = self._translate_chunk(
            text,
            retry_context,
            source_language,
            force_max_tokens=max_tokens,
        )
        retried = self._repair_common_spanish_table_labels(text, retried)
        if self._is_acceptable_table_translation(text, retried, source_language):
            return retried

        fallback = self._translate_table_by_row_groups(text, context, source_language)
        fallback = self._repair_common_spanish_table_labels(text, fallback or "")
        if fallback and self._is_acceptable_table_translation(text, fallback, source_language):
            return fallback

        logger.warning("Table translation remained invalid after retries; using source table text.")
        return text

    def _repair_common_spanish_table_labels(
        self,
        source_table: str,
        translated_table: str,
    ) -> str:
        """Apply deterministic English to a few unambiguous aligned cells.

        The translation model occasionally shortens the fully spelled
        ``Seguimiento`` to the invented token ``Fol`` to satisfy compact-table
        guidance. Cell topology is already required to remain identical, so an
        exact source-label mapping can safely replace the corresponding target
        cell without guessing about another cell or changing table markup.
        """

        source_matches = list(self._TABLE_CELL_RE.finditer(source_table or ""))
        target_matches = list(self._TABLE_CELL_RE.finditer(translated_table or ""))
        if not source_matches or len(source_matches) != len(target_matches):
            return translated_table

        deterministic_targets: dict[int, str] = {}
        for index, source_match in enumerate(source_matches):
            source_text = self._TAG_RE.sub(" ", source_match.group("body"))
            normalized = " ".join(html.unescape(source_text).casefold().split())
            follow_up = re.fullmatch(
                r"seguimiento(?:\s*/\s*(\d+)\s+mes(?:es)?)?",
                normalized,
            )
            if follow_up is None:
                continue
            months = follow_up.group(1)
            deterministic_targets[index] = (
                f"Follow-up / {months} {'month' if months == '1' else 'months'}"
                if months is not None
                else "Follow-up"
            )
        if not deterministic_targets:
            return translated_table

        pieces: list[str] = []
        cursor = 0
        for index, target_match in enumerate(target_matches):
            replacement = deterministic_targets.get(index)
            if replacement is None:
                continue
            pieces.append(translated_table[cursor : target_match.start("body")])
            pieces.append(html.escape(replacement))
            cursor = target_match.end("body")
        pieces.append(translated_table[cursor:])
        return "".join(pieces)

    def _normalize_table_markup_for_translation(self, text: str) -> str:
        if not text or not self._may_contain_table_markup(text):
            return text

        normalized = self._TABLE_ESCAPED_TAG_RE.sub(self._replace_escaped_table_tag, text)
        return self._TABLE_BLOCK_RE.sub(
            lambda match: self._close_unclosed_table_rows(match.group(0)),
            normalized,
        )

    def _may_contain_table_markup(self, text: str) -> bool:
        lowered = (text or "").lower()
        return "<table" in lowered or "&lt;" in lowered and "table" in lowered

    def _replace_escaped_table_tag(self, match: re.Match[str]) -> str:
        closing = "/" if match.group(1) else ""
        tag = match.group(2).lower()
        attrs = match.group(3) or ""
        return f"<{closing}{tag}{attrs}>"

    def _close_unclosed_table_rows(self, table_html: str) -> str:
        matches = list(self._TABLE_ROW_OPEN_RE.finditer(table_html))
        if not matches:
            return table_html

        repaired: list[str] = []
        cursor = 0
        for index, row_match in enumerate(matches):
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(table_html)
            segment = table_html[row_match.start() : next_start]
            repaired.append(table_html[cursor : row_match.start()])
            if self._TABLE_ROW_CLOSE_RE.search(segment) is None:
                segment = self._insert_missing_row_close(segment)
            repaired.append(segment)
            cursor = next_start
        repaired.append(table_html[cursor:])
        return "".join(repaired)

    def _insert_missing_row_close(self, segment: str) -> str:
        table_close = re.search(r"(?is)</table\s*>", segment)
        if table_close is not None:
            return f"{segment[: table_close.start()]}</tr>{segment[table_close.start() :]}"
        return f"{segment}</tr>"

    def _table_output_budget(self, text: str) -> int:
        source_tokens = self._token_count(text)
        estimated = int(source_tokens * 2.2) + 256
        return max(self.settings.max_tokens, min(self.TABLE_OUTPUT_MAX_TOKENS, estimated))

    def _translate_table_by_row_groups(
        self, text: str, context: str, source_language: str | None
    ) -> str | None:
        split_match = self._TABLE_SPLIT_RE.match(text.strip())
        if split_match is None:
            return None
        before = split_match.group("before")
        table_html = split_match.group("table")
        after = split_match.group("after")

        source_topology = self._table_markup_topology(table_html)
        table_open_match = self._TABLE_OPEN_RE.search(table_html)
        row_matches = list(self._TABLE_ROW_RE.finditer(table_html))
        if (
            source_topology is None
            or table_open_match is None
            or len(row_matches) != len(source_topology.rows)
        ):
            return None

        rows = [match.group(0) for match in row_matches]
        section_open_tags = self._table_section_open_tags(table_html)
        translated_rows: dict[int, str] = {}
        start = 0
        while start < len(rows):
            section = source_topology.rows[start].section
            end = start + 1
            while (
                end < len(rows)
                and end - start < self.TABLE_ROW_GROUP_SIZE
                and source_topology.rows[end].section == section
            ):
                end += 1

            group_translated_rows = self._translate_table_row_group(
                rows=rows,
                start=start,
                end=end,
                section=section,
                table_open=table_open_match.group(0),
                section_open_tags=section_open_tags,
                context=context,
                source_language=source_language,
            )
            if group_translated_rows is None:
                return None
            translated_rows.update(
                {
                    row_index: translated_row
                    for row_index, translated_row in zip(
                        range(start, end),
                        group_translated_rows,
                        strict=True,
                    )
                }
            )
            start = end

        if len(translated_rows) != len(rows):
            return None

        rebuilt_parts: list[str] = []
        cursor = 0
        for row_index, row_match in enumerate(row_matches):
            rebuilt_parts.append(table_html[cursor : row_match.start()])
            rebuilt_parts.append(translated_rows[row_index])
            cursor = row_match.end()
        rebuilt_parts.append(table_html[cursor:])
        rebuilt_table = "".join(rebuilt_parts)
        if not self._has_valid_table_markup_structure(table_html, rebuilt_table):
            return None
        return before + rebuilt_table + after

    def _translate_table_row_group(
        self,
        *,
        rows: list[str],
        start: int,
        end: int,
        section: tuple[str, int] | None,
        table_open: str,
        section_open_tags: dict[tuple[str, int], str],
        context: str,
        source_language: str | None,
    ) -> list[str] | None:
        """Translate a row group, bisecting it when a model damages the group.

        Whole-table retries are useful for context, but repeating an invalid
        seven-row request as an eight-row fallback is not a fallback at all.
        Bounded groups retain nearby row context; recursive bisection isolates a
        difficult row while preserving the original table shell and topology.
        """

        group_rows = rows[start:end]
        section_open = ""
        section_close = ""
        if section is not None:
            section_open = section_open_tags.get(section, "")
            if not section_open:
                return None
            section_close = f"</{section[0]}>"
        group_table = table_open + section_open + "".join(group_rows) + section_close + "</table>"
        group_context = (
            f"{context}\n"
            "Translate this HTML table to English and keep tags intact. "
            "Return complete table markup. "
            f"{self._COMPACT_TABLE_TRANSLATION_GUIDANCE}"
        ).strip()
        translated_group = self._translate_chunk(
            group_table,
            group_context,
            source_language,
            force_max_tokens=self._table_output_budget(group_table),
        )
        if self._is_acceptable_table_translation(
            group_table,
            translated_group,
            source_language,
        ):
            translated_table = self._extract_primary_table(translated_group)
            parsed_group = (
                self._parse_table_rows(translated_table) if translated_table is not None else None
            )
            if parsed_group is not None:
                _, translated_rows, _ = parsed_group
                if len(translated_rows) == len(group_rows):
                    return translated_rows

        if end - start <= 1:
            return None
        midpoint = start + (end - start) // 2
        left = self._translate_table_row_group(
            rows=rows,
            start=start,
            end=midpoint,
            section=section,
            table_open=table_open,
            section_open_tags=section_open_tags,
            context=context,
            source_language=source_language,
        )
        if left is None:
            return None
        right = self._translate_table_row_group(
            rows=rows,
            start=midpoint,
            end=end,
            section=section,
            table_open=table_open,
            section_open_tags=section_open_tags,
            context=context,
            source_language=source_language,
        )
        if right is None:
            return None
        return left + right

    def _table_section_open_tags(self, table_html: str) -> dict[tuple[str, int], str]:
        section_counts: Counter[str] = Counter()
        section_tags: dict[tuple[str, int], str] = {}
        for match in self._TABLE_SECTION_OPEN_RE.finditer(table_html):
            tag = match.group("tag").casefold()
            occurrence = section_counts[tag]
            section_counts[tag] += 1
            section_tags[(tag, occurrence)] = match.group(0)
        return section_tags

    def _is_acceptable_table_translation(
        self,
        source_text: str,
        translated_text: str,
        source_language: str | None,
    ) -> bool:
        return (
            self._table_translation_issue(
                source_text,
                translated_text,
                source_language,
            )
            is None
        )

    def _table_translation_issue(
        self,
        source_text: str,
        translated_text: str,
        source_language: str | None,
    ) -> str | None:
        if not self._has_valid_table_markup_structure(source_text, translated_text):
            return "translation_table_structure_invalid"

        source_table = self._extract_primary_table(source_text)
        translated_table = self._extract_primary_table(translated_text)
        if source_table is None or translated_table is None:
            return "translation_table_structure_invalid"
        if not self._table_nonempty_cells_preserved(source_table, translated_table):
            return "translation_table_cell_missing"
        if self._missing_source_acronyms(
            source_table,
            translated_table,
            BlockType.TABLE,
        ):
            return "translation_source_acronym_missing"
        if (
            self._base_language(source_language) not in {None, "en"}
            and self._normalized_table_cell(source_table)
            == self._normalized_table_cell(translated_table)
            and any(
                self._table_cell_requires_translation(cell)
                for cell in self._table_cell_texts(source_table)
            )
        ):
            return "translation_output_matches_source"
        return self._translation_acceptance_issue(
            source_table,
            translated_table,
            source_language,
            BlockType.TABLE,
        )

    def _normalized_table_cell(self, text: str) -> str:
        return " ".join(html.unescape(text).split()).casefold()

    def _table_cell_requires_translation(self, text: str) -> bool:
        if self._is_nontranslatable_identifier(text):
            return False
        words = re.findall(r"[^\W\d_]+", html.unescape(text), flags=re.UNICODE)
        if not words:
            return False
        return not all(self._looks_like_table_abbreviation(word) for word in words)

    def _looks_like_table_abbreviation(self, word: str) -> bool:
        letters = [character for character in word if character.isalpha()]
        cased_letters = [
            character for character in letters if character.islower() or character.isupper()
        ]
        uppercase_count = sum(character.isupper() for character in cased_letters)
        return bool(
            letters
            and len(letters) <= 6
            and cased_letters
            and (all(character.isupper() for character in cased_letters) or uppercase_count >= 2)
        )

    def _parse_table_rows(self, html: str) -> tuple[str, list[str], str] | None:
        parts = self._TABLE_PARTS_RE.match(html.strip())
        if parts is None:
            return None
        prefix = parts.group("prefix")
        body = parts.group("body")
        suffix = parts.group("suffix")
        rows = self._TABLE_ROW_RE.findall(body)
        if not rows:
            return None
        return prefix, rows, suffix

    def _is_valid_table_markup_translation(self, source_text: str, translated_text: str) -> bool:
        if not self._has_valid_table_markup_structure(source_text, translated_text):
            return False
        source_table = self._extract_primary_table(source_text)
        translated_table = self._extract_primary_table(translated_text)
        if source_table is None or translated_table is None:
            return False
        return self._table_nonempty_cells_preserved(source_table, translated_table)

    def _has_valid_table_markup_structure(
        self,
        source_text: str,
        translated_text: str,
    ) -> bool:
        if not self._is_table_heavy_markup(source_text):
            return True
        source_table = self._extract_primary_table(source_text)
        translated_table = self._extract_primary_table(translated_text)
        if source_table is None or translated_table is None:
            return False
        if source_text.strip().lower().endswith(
            "</table>"
        ) and not translated_text.strip().lower().endswith("</table>"):
            return False
        for tag in ("table", "tr", "td", "th"):
            source_open, source_close = self._count_tag_pair(source_table, tag)
            translated_open, translated_close = self._count_tag_pair(translated_table, tag)
            if source_open != translated_open or source_close != translated_close:
                if tag in {"td", "th"}:
                    source_cells = (
                        self._count_tag_pair(source_table, "td")[0]
                        + self._count_tag_pair(source_table, "th")[0]
                    )
                    translated_cells = (
                        self._count_tag_pair(translated_table, "td")[0]
                        + self._count_tag_pair(translated_table, "th")[0]
                    )
                    if source_cells != translated_cells:
                        return False
                    continue
                return False
        source_rows = self._parse_table_rows(source_table)
        translated_rows = self._parse_table_rows(translated_table)
        if source_rows is None or translated_rows is None:
            return False
        source_topology = self._table_markup_topology(source_table)
        translated_topology = self._table_markup_topology(translated_table)
        if source_topology is None or source_topology != translated_topology:
            return False
        return True

    def _table_markup_topology(self, table_html: str) -> _TableMarkupTopology | None:
        parser = _StrictTableTopologyParser()
        try:
            parser.feed(table_html)
            parser.close()
        except Exception:
            return None
        return parser.topology()

    def _extract_primary_table(self, text: str) -> str | None:
        match = self._TABLE_BLOCK_RE.search(text or "")
        return match.group(0) if match else None

    def _count_tag_pair(self, text: str, tag: str) -> tuple[int, int]:
        opens = len(re.findall(rf"(?is)<{tag}\b", text))
        closes = len(re.findall(rf"(?is)</{tag}>", text))
        return opens, closes

    def _table_nonempty_cells_preserved(self, source_table: str, translated_table: str) -> bool:
        source_cells = self._table_cell_texts(source_table)
        translated_cells = self._table_cell_texts(translated_table)
        if len(source_cells) != len(translated_cells):
            return False
        for source_cell, translated_cell in zip(source_cells, translated_cells):
            if self._is_significant_table_cell(source_cell) and not translated_cell.strip():
                return False
        return True

    def _table_cell_texts(self, table_html: str) -> list[str]:
        cells: list[str] = []
        for match in self._TABLE_CELL_RE.finditer(table_html or ""):
            text = self._TAG_RE.sub(" ", match.group("body"))
            text = html.unescape(text)
            cells.append(" ".join(text.split()))
        return cells

    def _is_significant_table_cell(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return any(character.isalnum() for character in stripped)

    def _build_prompt(
        self, text: str, context: str = "", source_language: str | None = None
    ) -> str:
        system = self._system_prompt()
        details: list[str] = []
        if source_language:
            details.append(f"SOURCE LANGUAGE: {source_language}")
        if context.strip():
            details.append(
                "CONTEXT AND OUTPUT INSTRUCTIONS (use for consistency; do not reproduce unless explicitly "
                f"requested):\n{context.strip()}"
            )
        details.append(f"TEXT:\n{text}")
        user = "\n\n".join(details)
        return self._format_chat_prompt(system, user)

    def _system_prompt(self) -> str:
        return (
            "You are translating OCR-derived scientific paper content into English for PDF reconstruction. "
            "TEXT may contain plain text, Markdown, or HTML. Translate only human-readable natural language. "
            "Preserve existing Markdown syntax, HTML tags, attributes, table rows/cells, citations, formulas, "
            "units, numeric values, and figure references. Preserve source acronyms and unexplained "
            "abbreviations exactly; never expand or reinterpret them. Do not invent abbreviations from "
            "ordinary source words. "
            "Do not add wrapper text such as labels, explanations, notes, summaries, source text, or code fences. "
            "Translate short section headings and titles as well."
        )

    def _make_sampler(self, sample_utils):
        kwargs = {
            "temp": max(0.0, self.settings.temperature),
            "top_p": max(0.0, min(1.0, self.settings.top_p)),
            "top_k": max(0, int(self.settings.top_k)),
            "min_p": max(0.0, min(1.0, self.settings.min_p)),
        }
        try:
            return sample_utils.make_sampler(**kwargs)
        except TypeError:
            kwargs.pop("min_p", None)
            try:
                return sample_utils.make_sampler(**kwargs)
            except TypeError:
                return sample_utils.make_sampler(temp=max(0.0, self.settings.temperature))

    def _make_logits_processors(self, sample_utils):
        if not hasattr(sample_utils, "make_logits_processors"):
            return []
        try:
            return sample_utils.make_logits_processors(
                presence_penalty=float(self.settings.presence_penalty),
                repetition_penalty=max(float(self.settings.repetition_penalty), 1e-6),
            )
        except TypeError:
            return []

    def _format_chat_prompt(self, system: str, user: str) -> str:
        if system == self._system_prompt():
            template = self._instruction_template(system)
            if template is not None:
                self._instruction_cache_hits += 1
                return f"{template.prefix}{user}{template.suffix}"
        return self._render_chat_prompt(system, user)

    def _render_chat_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                template_kwargs = self._chat_template_kwargs()
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            except TypeError:
                # Some tokenizers do not accept additional template kwargs.
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logger.debug("Tokenizer chat template failed; using plain prompt: %s", exc)
        return f"{system}\n\n{user}\n\nENGLISH:"

    def _instruction_template(self, system: str) -> _InstructionTemplateCache | None:
        cached = self._instruction_template_cache
        if cached is not None and cached.system == system:
            return cached
        if self._tokenizer is None:
            return None

        self._instruction_cache_misses += 1
        marker = "\u241fTRANSLATHOR_USER_CONTENT_BOUNDARY\u241f"
        rendered = self._render_chat_prompt(system, marker)
        if rendered.count(marker) != 1:
            logger.debug("Chat template rewrote the instruction-cache marker; caching disabled.")
            return None
        prefix, suffix = rendered.split(marker, maxsplit=1)
        prefix_tokens: tuple[int, ...] | None = None
        token_composition_safe = False
        try:
            prefix_tokens = tuple(
                int(token) for token in self._tokenizer.encode(prefix, add_special_tokens=True)
            )
            probes = (
                "TEXT:\nA short sentence.",
                "SOURCE LANGUAGE: es\n\nTEXT:\nTexto clínico.",
                "TEXT:\n<table><tr><td>6 m</td></tr></table>",
            )
            token_composition_safe = all(
                list(prefix_tokens)
                + list(
                    self._tokenizer.encode(
                        f"{probe}{suffix}",
                        add_special_tokens=False,
                    )
                )
                == list(
                    self._tokenizer.encode(
                        f"{prefix}{probe}{suffix}",
                        add_special_tokens=True,
                    )
                )
                for probe in probes
            )
        except (AttributeError, TypeError, ValueError):
            prefix_tokens = None
            token_composition_safe = False

        cached = _InstructionTemplateCache(
            system=system,
            prefix=prefix,
            suffix=suffix,
            prefix_tokens=prefix_tokens,
            token_composition_safe=token_composition_safe,
        )
        self._instruction_template_cache = cached
        return cached

    def _encode_prompts(self, prompts: list[str]) -> list[list[int]]:
        if not prompts:
            return []
        template = self._instruction_template_cache
        if (
            template is not None
            and template.token_composition_safe
            and template.prefix_tokens is not None
            and all(
                prompt.startswith(template.prefix) and prompt.endswith(template.suffix)
                for prompt in prompts
            )
        ):
            bodies = [prompt[len(template.prefix) :] for prompt in prompts]
            encoded_bodies = self._batch_encode_texts(
                bodies,
                add_special_tokens=False,
            )
            prefix_tokens = list(template.prefix_tokens)
            return [prefix_tokens + body_tokens for body_tokens in encoded_bodies]
        return self._batch_encode_texts(prompts, add_special_tokens=True)

    def _batch_encode_texts(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
    ) -> list[list[int]]:
        if self._tokenizer is None:
            raise RuntimeError("Cannot tokenize translation prompts before the model is loaded.")
        inner_tokenizer = getattr(self._tokenizer, "_tokenizer", None)
        if callable(inner_tokenizer):
            try:
                encoded = inner_tokenizer(
                    texts,
                    add_special_tokens=add_special_tokens,
                    padding=False,
                    truncation=False,
                )
                input_ids = encoded["input_ids"]
                return [list(map(int, tokens)) for tokens in input_ids]
            except (KeyError, TypeError, ValueError):
                pass
        return [
            list(
                map(
                    int,
                    self._tokenizer.encode(
                        text,
                        add_special_tokens=add_special_tokens,
                    ),
                )
            )
            for text in texts
        ]

    def _chat_template_kwargs(self) -> dict[str, Any]:
        if self._is_qwen35_model_name(self.settings.model_name):
            return {"enable_thinking": False}
        return {}

    def _detect_text_language_relaxed(self, text: str) -> str | None:
        compact = self._text_for_language_detection(text)
        if not compact:
            return None
        language, _ = self._detect_language_with_confidence(compact)
        return language

    def _chunk_source_language(self, text: str, block_type: BlockType) -> str | None:
        return self._chunk_language_resolution(text, block_type, "").language

    def _chunk_language_resolution(
        self,
        text: str,
        block_type: BlockType,
        nearby_context: str,
    ) -> _LanguageResolution:
        compact = self._text_for_language_detection(text)
        words = self._language_words(compact)
        is_short_heading = (
            block_type == BlockType.HEADING
            and len(words) <= 6
            and sum(len(word) for word in words) <= 60
        )
        if is_short_heading and nearby_context:
            contextual_text = self._text_for_language_detection(f"{text}\n{nearby_context}")
            contextual_words = self._language_words(contextual_text)
            if len(contextual_words) >= 4 and sum(len(word) for word in contextual_words) >= 24:
                language, confidence = self._detect_language_with_confidence(contextual_text)
                confidence_threshold = (
                    self._ENGLISH_SOURCE_CONFIDENCE
                    if self._base_language(language) == "en"
                    else self._NON_ENGLISH_OUTPUT_CONFIDENCE
                )
                if (
                    language is not None
                    and confidence is not None
                    and confidence >= confidence_threshold
                ):
                    return _LanguageResolution(
                        self._normalize_lang_code(language),
                        "nearby_context",
                        confidence,
                    )

        detected = self._detect_text_language(text)
        if detected is not None:
            confidence_language, confidence = self._detect_language_with_confidence(compact)
            if self._base_language(confidence_language) != self._base_language(detected):
                confidence = None
            return _LanguageResolution(
                self._normalize_lang_code(detected),
                "block",
                confidence,
            )

        # Avoid inheriting document-level language for short headings and markup-heavy chunks:
        # they are often misclassified as English and then incorrectly skipped.
        if block_type == BlockType.HEADING or self._contains_inline_markup(text):
            language = self._detect_text_language_relaxed(text)
            confidence_language, confidence = self._detect_language_with_confidence(compact)
            if self._base_language(confidence_language) != self._base_language(language):
                confidence = None
            return _LanguageResolution(
                self._normalize_lang_code(language),
                "block",
                confidence,
            )

        return _LanguageResolution(self._document_language, "document", None)

    def _nearby_heading_context(
        self,
        document: DocumentModel,
        block_ids: list[str],
        block_type: BlockType,
    ) -> str:
        if block_type != BlockType.HEADING or not block_ids:
            return ""
        positions = {block.id: index for index, block in enumerate(document.blocks)}
        start_positions = [positions[block_id] for block_id in block_ids if block_id in positions]
        if not start_positions:
            return ""
        anchor = document.blocks[min(start_positions)]
        pieces: list[str] = []
        for candidate in document.blocks[max(start_positions) + 1 :]:
            if candidate.page_number != anchor.page_number:
                break
            if candidate.block_type == BlockType.HEADING:
                break
            if candidate.block_type in {
                BlockType.EQUATION,
                BlockType.FIGURE,
                BlockType.FOOTER,
                BlockType.HEADER,
                BlockType.PAGE_NUMBER,
            }:
                continue
            candidate_text = candidate.text.strip()
            if not candidate_text or self._is_marker_table_cell_block(candidate):
                continue
            pieces.append(candidate_text[:400])
            if len(pieces) >= 2 or sum(len(piece) for piece in pieces) >= 600:
                break
        return "\n".join(pieces)

    def _context_with_nearby_source(self, context: str, nearby_context: str) -> str:
        if not nearby_context:
            return context
        note = (
            "Nearby source text for language and terminology disambiguation only; "
            "do not translate, reproduce, or summarize it unless it also appears in TEXT:\n"
            f"{nearby_context}"
        )
        return f"{context}\n{note}".strip()

    def _text_for_language_detection(self, text: str) -> str:
        normalized = self._TAG_RE.sub(" ", text)
        normalized = self._ENTITY_RE.sub(" ", normalized)
        normalized = normalized.replace(self.TABLE_DELIMITER, " ")
        normalized = re.sub(r"[_*`#>\-]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _is_table_heavy_markup(self, text: str) -> bool:
        lowered = (text or "").lower()
        has_row = "<tr" in lowered or self._TABLE_ESCAPED_ROW_RE.search(lowered) is not None
        return "<table" in lowered and has_row and "</table>" in lowered

    def _contains_inline_markup(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "<table" in lowered
            or "<tr" in lowered
            or "<td" in lowered
            or "<th" in lowered
            or self._TABLE_ESCAPED_ROW_RE.search(lowered) is not None
        )

    def _looks_like_english_text(self, text: str) -> bool:
        if not text.strip():
            return True

        compact = self._text_for_language_detection(text)
        if not compact:
            return False
        if self._contains_inline_markup(text):
            return False
        # Typography such as en dashes and curly quotes is common in English
        # titles; accented alphabetic text remains a conservative non-English cue.
        if any(ord(ch) > 127 and ch.isalpha() for ch in compact):
            return False

        words = re.findall(r"[A-Za-z]+", compact.lower())
        if not words:
            return False

        if len(words) == 1:
            return words[0] in self._ENGLISH_SINGLE_WORD_HINTS

        hint_hits = sum(1 for word in words if word in self._ENGLISH_HINT_WORDS)
        if len(words) <= 4:
            if hint_hits >= 1:
                return True
        elif hint_hits >= max(2, len(words) // 8):
            return True
        return self._is_confident_english_source(compact)

    def _is_qwen35_model_name(self, model_name: str | None) -> bool:
        normalized = (model_name or "").lower()
        return "qwen3.5" in normalized or "qwen3_5" in normalized

    def _postprocess_translated_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"(?is)(?:<end_of_turn>\s*)+", "", cleaned)
        cleaned = re.sub(r"(?is)(?:<\|im_end\|>\s*)+", "", cleaned)
        cleaned = cleaned.replace("<|eot_id|>", "").strip()
        if self._is_qwen35_model_name(self.settings.model_name):
            # Defensive cleanup in case a backend/template still emits reasoning blocks.
            cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", cleaned).strip()
        return cleaned or text

    def _chunk_block_type(
        self, chunk: TranslationChunk, block_by_id: dict[str, Block]
    ) -> BlockType | None:
        if chunk.chunk_type == "keywords":
            return BlockType.PARAGRAPH
        if not chunk.block_ids:
            return None
        target_id = chunk.block_ids[0]
        if target_id.startswith(self.TABLE_HEADER_PREFIX) or target_id.startswith(
            self.TABLE_ROW_PREFIX
        ):
            return BlockType.TABLE
        block = block_by_id.get(target_id)
        return block.block_type if block is not None else None

    def _augment_context_for_block_type(self, context: str, block_type: BlockType | None) -> str:
        if block_type == BlockType.HEADING:
            note = (
                "TEXT is a section heading/title. Translate it to natural English while "
                "preserving the heading intent."
            )
        elif block_type == BlockType.TABLE:
            note = (
                "TEXT is a table. Preserve every row, cell, empty cell, pipe delimiter, "
                "number, and span in the same order; translate only natural-language cell text. "
                f"{self._COMPACT_TABLE_TRANSLATION_GUIDANCE}"
            )
        else:
            return context
        if not context.strip():
            return note
        return f"{context}\n{note}"

    def _normalize_lang_code(self, code: str | None) -> str | None:
        if not code:
            return None
        normalized = code.strip().replace("_", "-").lower()
        aliases = {
            "zh-cn": "zh-CN",
            "zh-tw": "zh-TW",
            "pt-br": "pt-BR",
            "pt-pt": "pt-PT",
            "en-us": "en-US",
            "en-gb": "en-GB",
            "es-mx": "es-MX",
            "fr-ca": "fr-CA",
        }
        if normalized in aliases:
            return aliases[normalized]
        if len(normalized) == 2 and normalized.isalpha():
            return normalized
        if (
            len(normalized) == 5
            and normalized[2] == "-"
            and normalized[:2].isalpha()
            and normalized[3:].isalpha()
        ):
            return f"{normalized[:2]}-{normalized[3:].upper()}"
        return None

    def _estimated_output_tokens(self, prompt: str) -> int:
        if self._tokenizer is None:
            return self.settings.max_tokens
        try:
            prompt_tokens = len(self._tokenizer.encode(prompt))
        except Exception:
            return self.settings.max_tokens
        return self._estimated_output_tokens_from_count(prompt_tokens)

    def _estimated_output_tokens_from_count(self, prompt_tokens: int | None) -> int:
        if prompt_tokens is None:
            return self.settings.max_tokens
        return max(128, min(self.settings.max_tokens, int(prompt_tokens * 0.75)))
