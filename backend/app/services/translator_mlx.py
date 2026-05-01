from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from langdetect import detect

from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_TRANSLATION_MODEL
from app.models.schema import Block, BlockType, DocumentModel, TranslationChunk

logger = logging.getLogger(__name__)


@dataclass
class TranslationSettings:
    model_name: str = DEFAULT_TRANSLATION_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024


@dataclass
class TranslationUnit:
    block_ids: list[str]
    text: str
    block_type: BlockType
    context: str = ""


class MlxTranslator:
    TABLE_DELIMITER = "\n|||CELL_BREAK|||\n"
    TABLE_HEADER_PREFIX = "__table_header__:"
    TABLE_ROW_PREFIX = "__table_row__:"
    TABLE_OUTPUT_MAX_TOKENS = 4096
    TABLE_ROW_GROUP_SIZE = 8
    PROSE_CHUNK_TOKEN_CAP = 800
    _TAG_RE = re.compile(r"<[^>]+>")
    _ENTITY_RE = re.compile(r"&[a-zA-Z0-9#]+;")
    _TABLE_BLOCK_RE = re.compile(r"(?is)<table\b.*?</table>")
    _TABLE_SPLIT_RE = re.compile(r"(?is)^(?P<before>.*?)(?P<table><table\b.*?</table>)(?P<after>.*)$")
    _TABLE_PARTS_RE = re.compile(
        r"(?is)^(?P<prefix>.*?<table\b[^>]*>)(?P<body>.*)(?P<suffix></table>\s*)$"
    )
    _TABLE_ROW_RE = re.compile(r"(?is)<tr\b[^>]*>.*?</tr>")
    _TABLE_ESCAPED_TAG_RE = re.compile(
        r"(?is)&lt;\s*(/?)\s*(table|thead|tbody|tfoot|tr|td|th)\b([^<>]*?)&gt;"
    )
    _TABLE_ESCAPED_ROW_RE = re.compile(r"(?is)&lt;\s*tr\b")
    _TABLE_ROW_OPEN_RE = re.compile(r"(?is)<tr\b[^>]*>")
    _TABLE_ROW_CLOSE_RE = re.compile(r"(?is)</tr\s*>")
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

    def __init__(self, settings: TranslationSettings) -> None:
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._document_language: str | None = None
        self._last_load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if os.getenv("DISABLE_MLX", "0") == "1":
            logger.warning("MLX disabled with DISABLE_MLX=1")
            self._last_load_error = "MLX disabled with DISABLE_MLX=1"
            return False
        if self._model is not None and self._tokenizer is not None:
            self._last_load_error = None
            return True
        try:
            self._configure_mlx_thread()
            self._model, self._tokenizer = self._load_model_and_tokenizer(self.settings.model_name)
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
        units = self._build_translation_units(document)
        chunks: list[TranslationChunk] = []
        self._document_language = self._normalize_lang_code(document.metadata.detected_language)
        for unit in units:
            if unit.block_ids and (
                unit.block_ids[0].startswith(self.TABLE_HEADER_PREFIX) or unit.block_ids[0].startswith(self.TABLE_ROW_PREFIX)
            ):
                text_parts = [unit.text]
            elif unit.block_type == BlockType.TABLE and self._is_table_heavy_markup(unit.text):
                text_parts = [unit.text]
            else:
                text_parts = self._split_to_token_budget(unit.text)
            for text_part in text_parts:
                chunks.append(
                    TranslationChunk(
                        id=f"chunk-{len(chunks)}",
                        block_ids=unit.block_ids,
                        source_text=text_part,
                        context=unit.context,
                        source_language=self._chunk_source_language(text_part, unit.block_type),
                        source_token_count=self._token_count(text_part),
                    )
                )
        return chunks

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
        translated_chunks: list[TranslationChunk] = []
        table_like_chunks = [chunk for chunk in chunks if self._is_table_heavy_markup(chunk.source_text)]
        table_chunk_index: dict[int, int] = {id(chunk): idx for idx, chunk in enumerate(table_like_chunks, start=1)}

        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            if on_chunk_started is not None:
                on_chunk_started(index, total_chunks)
            block_type = self._chunk_block_type(chunk, block_by_id)
            effective_context = self._augment_context_for_block_type(chunk.context, block_type)
            is_table_like = self._is_table_heavy_markup(chunk.source_text)
            if is_table_like and on_table_progress is not None:
                on_table_progress(
                    table_chunk_index.get(id(chunk), 1),
                    max(1, len(table_like_chunks)),
                    f"chunk-{index}",
                )
            if not loaded:
                translated = chunk.source_text
            elif self._is_already_english(chunk):
                translated = chunk.source_text
            elif is_table_like:
                translated = self._translate_table_markup_chunk(
                    chunk.source_text,
                    effective_context,
                    chunk.source_language,
                )
            else:
                translated = self._translate_chunk_with_validation(
                    chunk.source_text,
                    effective_context,
                    chunk.source_language,
                    block_type,
                )
            chunk.translated_text = translated
            translated_chunks.append(chunk)
            if on_chunk_translated is not None:
                preview = translated.replace("\n", " ").strip()
                on_chunk_translated(index, total_chunks, preview[:160])

        for chunk in self._coalesce_translated_chunks(translated_chunks):
            self._apply_translation_to_target(chunk, block_by_id, table_by_id)

        return document, translated_md

    def cleanup(self) -> None:
        self._model = None
        self._tokenizer = None
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
            mx.set_default_stream(mx.new_stream(mx.gpu))
        except Exception as exc:
            logger.debug("MLX thread stream setup skipped: %s", exc)

    def _build_translation_units(self, document: DocumentModel) -> list[TranslationUnit]:
        units: list[TranslationUnit] = []
        pending: list[Block] = []
        section_context = ""

        for block in document.blocks:
            if block.block_type in {BlockType.HEADER, BlockType.FOOTER} or not block.text.strip():
                continue

            if block.block_type == BlockType.HEADING:
                self._flush_paragraph_unit(pending, units, section_context)
                pending = []
                heading_text = block.text.strip()
                units.append(TranslationUnit([block.id], heading_text, block.block_type, section_context))
                section_context = heading_text
                continue

            if block.block_type != BlockType.PARAGRAPH:
                self._flush_paragraph_unit(pending, units, section_context)
                pending = []
                units.append(TranslationUnit([block.id], block.text.strip(), block.block_type, section_context))
                continue

            if pending and not self._belongs_to_same_paragraph(pending[-1], block):
                self._flush_paragraph_unit(pending, units, section_context)
                pending = []
            pending.append(block)

        self._flush_paragraph_unit(pending, units, section_context)
        self._append_table_units(document, units, section_context)
        return units

    def _append_table_units(self, document: DocumentModel, units: list[TranslationUnit], context: str) -> None:
        for table_index, table in enumerate(document.tables, start=1):
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

    def _flush_paragraph_unit(self, blocks: list[Block], units: list[TranslationUnit], context: str) -> None:
        if not blocks:
            return
        text = self._join_paragraph_lines([block.text for block in blocks])
        if text:
            unit_type = BlockType.TABLE if self._is_table_heavy_markup(text) else BlockType.PARAGRAPH
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
            table.headers = self._split_table_translation(chunk.translated_text, chunk.source_text, len(table.headers))
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
                    cell.model_copy(update={"text": translated_row[idx] if idx < len(translated_row) else cell.text})
                    for idx, cell in enumerate(table.cells[row_index])
                ]
            return

        first = block_by_id.get(chunk.block_ids[0])
        if first is None:
            return

        first.text = chunk.translated_text.strip()
        first.metadata["translated_from_block_ids"] = chunk.block_ids
        for block_id in chunk.block_ids[1:]:
            block = block_by_id.get(block_id)
            if block is not None:
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

            out.append(
                chunk.model_copy(
                    update={
                        "id": group[0].id,
                        "source_text": "\n\n".join(item.source_text.strip() for item in group if item.source_text.strip()),
                        "translated_text": "\n\n".join(
                            item.translated_text.strip() for item in group if item.translated_text.strip()
                        ),
                    }
                )
            )
        return out

    def _is_table_target(self, target_id: str) -> bool:
        return target_id.startswith(self.TABLE_HEADER_PREFIX) or target_id.startswith(self.TABLE_ROW_PREFIX)

    def _split_table_translation(self, translated_text: str, source_text: str, expected_cells: int) -> list[str]:
        parts = [part.strip() for part in translated_text.split(self.TABLE_DELIMITER)]
        if len(parts) == expected_cells:
            return parts
        source_parts = [part.strip() for part in source_text.split(self.TABLE_DELIMITER)]
        if len(source_parts) != expected_cells:
            source_parts = (source_parts + [""] * expected_cells)[:expected_cells]
        if expected_cells == 1:
            return [translated_text.strip()]
        return source_parts

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
            while after < len(compact) and compact[after] in '\'"”’)]}':
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
        try:
            return detect(compact)
        except Exception:
            return None

    def _is_already_english(self, chunk: TranslationChunk) -> bool:
        if chunk.source_language == "en":
            return self._looks_like_english_text(chunk.source_text)
        text = chunk.source_text.strip()
        if not text or chunk.source_language is not None:
            return False
        return self._looks_like_english_text(text)

    def _translate_chunk(
        self,
        text: str,
        context: str = "",
        source_language: str | None = None,
        force_max_tokens: int | None = None,
    ) -> str:
        self._configure_mlx_thread()
        from mlx_lm import generate, sample_utils

        prompt = self._build_prompt(text, context, source_language)
        sampler = self._make_sampler(sample_utils)
        max_tokens = force_max_tokens or self._estimated_output_tokens(prompt)
        try:
            out = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            translated = str(out).strip()
            return self._postprocess_translated_text(translated)
        except Exception as exc:
            logger.warning("Chunk translation failed; returning source text for this chunk: %s", exc)
            return text

    def _translate_chunk_with_validation(
        self,
        text: str,
        context: str,
        source_language: str | None,
        block_type: BlockType | None,
    ) -> str:
        translated = self._translate_chunk(text, context, source_language)
        if self._is_valid_chunk_translation_structure(text, translated, block_type):
            return translated

        retry_context = (
            f"{context}\n"
            "Preserve the source structure exactly: keep paragraph boundaries, list boundaries, headings, "
            "Markdown markers, citations, numbers, and line breaks that separate logical blocks. "
            "Do not summarize, omit, or collapse content."
        ).strip()
        retried = self._translate_chunk(text, retry_context, source_language)
        if self._is_valid_chunk_translation_structure(text, retried, block_type):
            return retried

        logger.warning("Chunk translation failed structure validation after retry; returning source text.")
        return text

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

        if len(source) >= 200 and len(translated) < max(40, int(len(source) * 0.20)):
            return False

        source_paragraphs = self._paragraph_count(source)
        translated_paragraphs = self._paragraph_count(translated)
        if source_paragraphs >= 3 and translated_paragraphs < max(2, source_paragraphs // 2):
            return False

        source_list_items = self._markdown_list_item_count(source)
        if source_list_items and self._markdown_list_item_count(translated) < source_list_items:
            return False

        source_heading_count = self._markdown_heading_count(source)
        if source_heading_count and self._markdown_heading_count(translated) < source_heading_count:
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

    def _translate_table_markup_chunk(self, text: str, context: str, source_language: str | None) -> str:
        text = self._normalize_table_markup_for_translation(text)
        max_tokens = self._table_output_budget(text)
        strict_context = (
            f"{context}\n"
            "TEXT contains an HTML table. Keep all HTML tags and attributes intact. "
            "Return a complete table with closing </table>."
        ).strip()
        translated = self._translate_chunk(
            text,
            strict_context,
            source_language,
            force_max_tokens=max_tokens,
        )
        if self._is_valid_table_markup_translation(text, translated):
            return translated

        retry_context = (
            f"{strict_context}\n"
            "Do not truncate output. Keep the same number of rows and cells."
        )
        retried = self._translate_chunk(
            text,
            retry_context,
            source_language,
            force_max_tokens=max_tokens,
        )
        if self._is_valid_table_markup_translation(text, retried):
            return retried

        fallback = self._translate_table_by_row_groups(text, context, source_language)
        if fallback and self._is_valid_table_markup_translation(text, fallback):
            return fallback

        logger.warning("Table translation remained invalid after retries; using source table text.")
        return text

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
            return f"{segment[:table_close.start()]}</tr>{segment[table_close.start():]}"
        return f"{segment}</tr>"

    def _table_output_budget(self, text: str) -> int:
        source_tokens = self._token_count(text)
        estimated = int(source_tokens * 2.2) + 256
        return max(self.settings.max_tokens, min(self.TABLE_OUTPUT_MAX_TOKENS, estimated))

    def _translate_table_by_row_groups(self, text: str, context: str, source_language: str | None) -> str | None:
        split_match = self._TABLE_SPLIT_RE.match(text.strip())
        if split_match is None:
            return None
        before = split_match.group("before")
        table_html = split_match.group("table")
        after = split_match.group("after")

        parsed = self._parse_table_rows(table_html)
        if parsed is None:
            return None
        prefix, rows, suffix = parsed

        translated_rows: list[str] = []
        for start in range(0, len(rows), self.TABLE_ROW_GROUP_SIZE):
            group_rows = rows[start : start + self.TABLE_ROW_GROUP_SIZE]
            group_table = prefix + "".join(group_rows) + suffix
            group_context = (
                f"{context}\n"
                "Translate this partial HTML table and keep tags intact. "
                "Return complete table markup."
            ).strip()
            translated_group = self._translate_chunk(
                group_table,
                group_context,
                source_language,
                force_max_tokens=self._table_output_budget(group_table),
            )
            if not self._is_valid_table_markup_translation(group_table, translated_group):
                translated_rows.extend(group_rows)
                continue

            parsed_group = self._parse_table_rows(translated_group)
            if parsed_group is None:
                translated_rows.extend(group_rows)
                continue
            _, group_translated_rows, _ = parsed_group
            if len(group_translated_rows) != len(group_rows):
                translated_rows.extend(group_rows)
                continue
            translated_rows.extend(group_translated_rows)

        if not translated_rows:
            return None
        return before + prefix + "".join(translated_rows) + suffix + after

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
        if not self._is_table_heavy_markup(source_text):
            return True
        source_table = self._extract_primary_table(source_text)
        translated_table = self._extract_primary_table(translated_text)
        if source_table is None or translated_table is None:
            return False
        if source_text.strip().lower().endswith("</table>") and not translated_text.strip().lower().endswith("</table>"):
            return False
        for tag in ("table", "tr", "td", "th"):
            source_open, source_close = self._count_tag_pair(source_table, tag)
            translated_open, translated_close = self._count_tag_pair(translated_table, tag)
            if source_open != translated_open or source_close != translated_close:
                if tag in {"td", "th"}:
                    source_cells = self._count_tag_pair(source_table, "td")[0] + self._count_tag_pair(source_table, "th")[0]
                    translated_cells = self._count_tag_pair(translated_table, "td")[0] + self._count_tag_pair(
                        translated_table, "th"
                    )[0]
                    if source_cells != translated_cells:
                        return False
                    continue
                return False
        return True

    def _extract_primary_table(self, text: str) -> str | None:
        match = self._TABLE_BLOCK_RE.search(text or "")
        return match.group(0) if match else None

    def _count_tag_pair(self, text: str, tag: str) -> tuple[int, int]:
        opens = len(re.findall(rf"(?is)<{tag}\b", text))
        closes = len(re.findall(rf"(?is)</{tag}>", text))
        return opens, closes

    def _build_prompt(self, text: str, context: str = "", source_language: str | None = None) -> str:
        context_part = (
            f"\nSECTION CONTEXT (for terminology only, do not translate this unless it appears in TEXT):\n{context}\n"
            if context
            else ""
        )
        system = self._system_prompt()
        user = f"{context_part}\nTEXT:\n{text}"
        return self._format_chat_prompt(system, user)

    def _system_prompt(self) -> str:
        return (
            "You are translating OCR-derived scientific paper content into English for PDF reconstruction. "
            "TEXT may contain plain text, Markdown, or HTML. Translate only human-readable natural language. "
            "Preserve existing Markdown syntax, HTML tags, attributes, table rows/cells, citations, formulas, "
            "symbols, abbreviations, units, numbers, and figure references. "
            "Do not add wrapper text such as labels, explanations, notes, summaries, source text, or code fences. "
            "Translate short section headings and titles as well. If TEXT is already English, return it unchanged."
        )

    def _make_sampler(self, sample_utils):
        try:
            return sample_utils.make_sampler(
                temp=max(0.0, self.settings.temperature),
                top_p=max(0.0, min(1.0, self.settings.top_p)),
            )
        except TypeError:
            return sample_utils.make_sampler(temp=max(0.0, self.settings.temperature))

    def _format_chat_prompt(self, system: str, user: str) -> str:
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

    def _chat_template_kwargs(self) -> dict[str, Any]:
        if self._is_qwen35_model_name(self.settings.model_name):
            return {"enable_thinking": False}
        return {}

    def _detect_text_language_relaxed(self, text: str) -> str | None:
        compact = self._text_for_language_detection(text)
        if not compact:
            return None
        try:
            return detect(compact)
        except Exception:
            return None

    def _chunk_source_language(self, text: str, block_type: BlockType) -> str | None:
        detected = self._detect_text_language(text)
        if detected:
            return detected

        # Avoid inheriting document-level language for short headings and markup-heavy chunks:
        # they are often misclassified as English and then incorrectly skipped.
        if block_type == BlockType.HEADING or self._contains_inline_markup(text):
            return self._detect_text_language_relaxed(text)

        return self._document_language

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
        if any(ord(ch) > 127 for ch in compact):
            return False

        words = re.findall(r"[A-Za-z]+", compact.lower())
        if not words:
            return False

        if len(words) == 1:
            return words[0] in self._ENGLISH_SINGLE_WORD_HINTS

        hint_hits = sum(1 for word in words if word in self._ENGLISH_HINT_WORDS)
        if len(words) <= 4:
            return hint_hits >= 1
        return hint_hits >= max(2, len(words) // 8)

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

    def _chunk_block_type(self, chunk: TranslationChunk, block_by_id: dict[str, Block]) -> BlockType | None:
        if not chunk.block_ids:
            return None
        target_id = chunk.block_ids[0]
        if target_id.startswith(self.TABLE_HEADER_PREFIX) or target_id.startswith(self.TABLE_ROW_PREFIX):
            return BlockType.TABLE
        block = block_by_id.get(target_id)
        return block.block_type if block is not None else None

    def _augment_context_for_block_type(self, context: str, block_type: BlockType | None) -> str:
        if block_type != BlockType.HEADING:
            return context
        heading_note = (
            "TEXT is a section heading/title. Translate it to natural English while preserving the heading intent."
        )
        if not context.strip():
            return heading_note
        return f"{context}\n{heading_note}"

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
        if len(normalized) == 5 and normalized[2] == "-" and normalized[:2].isalpha() and normalized[3:].isalpha():
            return f"{normalized[:2]}-{normalized[3:].upper()}"
        return None

    def _estimated_output_tokens(self, prompt: str) -> int:
        if self._tokenizer is None:
            return self.settings.max_tokens
        try:
            prompt_tokens = len(self._tokenizer.encode(prompt))
        except Exception:
            return self.settings.max_tokens
        estimated = max(128, min(self.settings.max_tokens, int(prompt_tokens * 0.75)))
        return estimated
