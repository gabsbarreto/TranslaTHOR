from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


METADATA_KEYS = (
    "title",
    "short_title",
    "authors",
    "first_author",
    "journal",
    "doi",
    "publisher",
    "year",
    "copyright_or_licence",
)
REAL_SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "method",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "appendix",
    "acknowledgements",
    "acknowledgments",
}


@dataclass(frozen=True)
class PageFurnitureCleanupConfig:
    extract_document_metadata: bool = True
    clean_page_furniture_with_metadata: bool = True
    top_lines: int = 5
    bottom_lines: int = 5
    similarity_threshold: float = 0.85
    preserve_first_page_metadata: bool = True


def empty_document_metadata() -> dict[str, Any]:
    return {
        "title": "",
        "short_title": "",
        "authors": [],
        "first_author": "",
        "journal": "",
        "doi": "",
        "publisher": "",
        "year": "",
        "copyright_or_licence": "",
    }


def extract_document_metadata(text: str) -> dict[str, Any]:
    metadata = empty_document_metadata()
    lines = _meaningful_lines(text)
    if not lines:
        return metadata

    metadata["doi"] = _extract_doi(text)
    metadata["year"] = _extract_year(text)
    metadata["copyright_or_licence"] = _extract_copyright_or_licence(lines)
    metadata["journal"] = _extract_journal(lines)
    metadata["publisher"] = _extract_publisher(lines)
    metadata["title"] = _extract_title(lines)
    metadata["short_title"] = _shorten_title(metadata["title"])
    authors = _extract_authors(lines, metadata["title"])
    metadata["authors"] = authors
    metadata["first_author"] = _surname(authors[0]) if authors else ""
    return metadata


def build_metadata_patterns(metadata: dict[str, Any]) -> list[str]:
    patterns: list[str] = []

    def add(value: object) -> None:
        text = str(value or "").strip()
        if text and len(normalise_for_matching(text)) >= 4:
            patterns.append(text)

    add(metadata.get("title"))
    add(metadata.get("short_title"))
    add(metadata.get("journal"))
    add(metadata.get("doi"))
    add(metadata.get("publisher"))
    add(metadata.get("copyright_or_licence"))
    add(metadata.get("year"))

    authors = [str(author).strip() for author in metadata.get("authors", []) if str(author).strip()]
    if authors:
        add(", ".join(authors))
    for author in authors:
        add(author)

    first_author = str(metadata.get("first_author") or "").strip()
    year = str(metadata.get("year") or "").strip()
    if first_author:
        add(f"{first_author} et al.")
        add(f"{first_author} et al")
        if year:
            add(f"{first_author} et al. {year}")
            add(f"{first_author} {year}")

    title = str(metadata.get("title") or "").strip()
    journal = str(metadata.get("journal") or "").strip()
    if title and journal:
        add(f"{journal} {title}")

    seen: set[str] = set()
    unique: list[str] = []
    for pattern in patterns:
        normalised = normalise_for_matching(pattern)
        if normalised and normalised not in seen:
            unique.append(pattern)
            seen.add(normalised)
    return unique


def normalise_for_matching(text: str) -> str:
    normalised = unicodedata.normalize("NFKD", str(text))
    normalised = "".join(ch for ch in normalised if not unicodedata.combining(ch))
    normalised = normalised.lower()
    normalised = re.sub(r"\bdoi\s*[:.]?\s*https?://doi\.org/", "doi ", normalised)
    normalised = normalised.replace("https://doi.org/", "doi ")
    normalised = normalised.replace("http://doi.org/", "doi ")
    normalised = re.sub(r"\bdoi\s*[:.]?\s*", "doi ", normalised)
    normalised = re.sub(r"[^a-z0-9]+", " ", normalised)
    return re.sub(r"\s+", " ", normalised).strip()


def is_metadata_like_line(line: str, patterns: list[str], threshold: float = 0.85) -> bool:
    candidate = normalise_for_matching(_strip_markdown_line(line))
    if not candidate or not patterns:
        return False
    if _looks_like_page_number(candidate):
        return True

    for pattern in patterns:
        pattern_norm = normalise_for_matching(pattern)
        if len(pattern_norm) < 4:
            continue
        if _doi_equivalent(candidate, pattern_norm):
            return True
        if _metadata_similarity(candidate, pattern_norm) >= threshold:
            return True
    return False


def clean_page_furniture(
    markdown: str,
    metadata: dict[str, Any],
    page_number: int,
    config: PageFurnitureCleanupConfig,
) -> str:
    if not config.clean_page_furniture_with_metadata:
        return markdown

    patterns = build_metadata_patterns(metadata)
    lines = str(markdown).splitlines()
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty_indices:
        return markdown

    top = set(nonempty_indices[: max(config.top_lines, 0)])
    bottom = set(nonempty_indices[-max(config.bottom_lines, 0) :]) if config.bottom_lines > 0 else set()
    remove: set[int] = set()

    for index in sorted(top | bottom):
        line = lines[index]
        zone = "top" if index in top else "bottom"
        if _should_remove_line(
            line=line,
            patterns=patterns,
            threshold=config.similarity_threshold,
            page_number=page_number,
            zone=zone,
            preserve_first_page_metadata=config.preserve_first_page_metadata,
        ):
            remove.add(index)

    if not remove:
        return markdown

    cleaned_lines = [line for index, line in enumerate(lines) if index not in remove]
    return _collapse_blank_lines("\n".join(cleaned_lines).strip())


def clean_pages_with_metadata(
    pages: list[tuple[int, str]],
    config: PageFurnitureCleanupConfig,
) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    combined_text = "\n\n".join(markdown for _page_number, markdown in pages)
    first_page_text = pages[0][1] if pages else ""
    metadata_source = f"{first_page_text}\n\n{combined_text}" if first_page_text else combined_text
    metadata = extract_document_metadata(metadata_source) if config.extract_document_metadata else empty_document_metadata()
    if not config.clean_page_furniture_with_metadata:
        return pages, metadata
    return [
        (page_number, clean_page_furniture(markdown, metadata, page_number, config))
        for page_number, markdown in pages
    ], metadata


def _metadata_similarity(candidate: str, pattern: str) -> float:
    if candidate == pattern:
        return 1.0
    if len(candidate) >= 8 and len(pattern) >= 8 and (candidate in pattern or pattern in candidate):
        shorter = min(len(candidate), len(pattern))
        longer = max(len(candidate), len(pattern))
        if shorter / longer >= 0.55:
            return 0.95
    sequence_score = SequenceMatcher(None, candidate, pattern).ratio()
    candidate_tokens = set(candidate.split())
    pattern_tokens = set(pattern.split())
    token_score = 0.0
    if candidate_tokens and pattern_tokens:
        token_score = len(candidate_tokens & pattern_tokens) / len(candidate_tokens | pattern_tokens)
    return max(sequence_score, token_score)


def _should_remove_line(
    *,
    line: str,
    patterns: list[str],
    threshold: float,
    page_number: int,
    zone: str,
    preserve_first_page_metadata: bool,
) -> bool:
    stripped = line.strip()
    plain = _strip_markdown_line(stripped)
    normalised = normalise_for_matching(plain)
    if not normalised:
        return False
    if _looks_like_page_number(normalised):
        return True
    if _is_protected_content_line(stripped):
        return False
    if preserve_first_page_metadata and page_number <= 1 and zone == "top":
        return False
    if _looks_like_body_paragraph(plain):
        return False
    if not (_is_isolated_or_metadata_like(plain) or zone == "bottom"):
        return False
    return is_metadata_like_line(plain, patterns, threshold=threshold)


def _meaningful_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in str(text).splitlines():
        stripped = _strip_markdown_line(line.strip())
        if stripped and not stripped.startswith("<!--"):
            lines.append(stripped)
    return lines


def _strip_markdown_line(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"^#{1,6}\s+", "", stripped)
    stripped = re.sub(r"^\s*[-*+]\s+", "", stripped)
    stripped = re.sub(r"^\s*\d+[.)]\s+", "", stripped)
    return stripped.strip(" *_`~|")


def _extract_doi(text: str) -> str:
    match = re.search(r"\b(?:doi\s*[:.]?\s*|https?://doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, re.IGNORECASE)
    return match.group(1).rstrip(".,;)") if match else ""


def _extract_year(text: str) -> str:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    return years[0] if years else ""


def _extract_copyright_or_licence(lines: list[str]) -> str:
    for line in lines:
        if re.search(r"\b(copyright|creative commons|open access|licen[cs]e|all rights reserved)\b|©", line, re.IGNORECASE):
            return line
    return ""


def _extract_journal(lines: list[str]) -> str:
    for line in lines[:20]:
        if _looks_like_journal_line(line):
            if len(line) <= 160 and not _looks_like_body_paragraph(line):
                return line
    return ""


def _looks_like_journal_line(line: str) -> bool:
    return bool(re.search(r"\b(journal|revista|annals|proceedings|transactions|bulletin)\b", line, re.IGNORECASE))


def _extract_publisher(lines: list[str]) -> str:
    for line in lines:
        if re.search(r"\b(elsevier|springer|wiley|taylor\s*&\s*francis|sage|routledge|mdpi|nature|frontiers|oup|cambridge university press)\b", line, re.IGNORECASE):
            return line
    return ""


def _extract_title(lines: list[str]) -> str:
    for line in lines[:18]:
        if _is_obvious_metadata_line(line) or _looks_like_journal_line(line):
            continue
        word_count = len(line.split())
        if 4 <= word_count <= 28 and len(line) <= 220:
            return line
    return ""


def _extract_authors(lines: list[str], title: str) -> list[str]:
    if not title:
        return []
    try:
        title_index = lines.index(title)
    except ValueError:
        title_index = 0
    for line in lines[title_index + 1 : title_index + 7]:
        if _is_obvious_metadata_line(line) or _looks_like_body_paragraph(line):
            continue
        if re.search(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+,\s*[A-Z]", line) or re.search(
            r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+", line
        ):
            return _split_author_line(line)
    return []


def _split_author_line(line: str) -> list[str]:
    cleaned = re.sub(r"\d+|\*|†|‡|§", "", line)
    parts = re.split(r"\s*(?:,|;|\band\b|&)\s*", cleaned)
    authors = [part.strip() for part in parts if len(part.strip()) >= 3]
    if len(authors) <= 1:
        authors = [cleaned.strip()]
    return authors[:12]


def _surname(author: str) -> str:
    author = re.sub(r"\s+", " ", author).strip()
    if "," in author:
        return author.split(",", 1)[0].strip()
    parts = author.split()
    return parts[-1].strip() if parts else ""


def _shorten_title(title: str) -> str:
    words = title.split()
    if len(words) <= 10:
        return title
    return " ".join(words[:10])


def _is_obvious_metadata_line(line: str) -> bool:
    return bool(
        re.search(r"\bdoi\b|https?://|@|copyright|creative commons|accepted|received|published\b", line, re.IGNORECASE)
        or _looks_like_page_number(normalise_for_matching(line))
    )


def _is_protected_content_line(line: str) -> bool:
    plain = _strip_markdown_line(line)
    normalised = normalise_for_matching(plain)
    if normalised in REAL_SECTION_HEADINGS:
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Za-z]", plain):
        heading_text = normalise_for_matching(re.sub(r"^\d+(?:\.\d+)*\s+", "", plain))
        if heading_text in REAL_SECTION_HEADINGS or len(plain.split()) <= 8:
            return True
    if re.match(r"^(?:fig(?:ure)?|table|chart|graph|source|note|notes)\b", plain, re.IGNORECASE):
        return True
    if line.strip().startswith("|"):
        return True
    return False


def _looks_like_body_paragraph(line: str) -> bool:
    words = line.split()
    if len(words) >= 18:
        return True
    if len(words) >= 12 and re.search(r"[.!?;:]$", line.strip()):
        return True
    return False


def _is_isolated_or_metadata_like(line: str) -> bool:
    words = line.split()
    if len(words) <= 18:
        return True
    if re.search(r"\bdoi\b|https?://doi\.org|©|copyright|licen[cs]e|all rights reserved", line, re.IGNORECASE):
        return True
    return False


def _looks_like_page_number(normalised: str) -> bool:
    return bool(re.fullmatch(r"(?:page )?\d{1,4}", normalised) or re.fullmatch(r"[ivxlcdm]{1,8}", normalised))


def _doi_equivalent(candidate: str, pattern: str) -> bool:
    candidate_doi = re.search(r"10 \d{4,9} [a-z0-9 ]+", candidate)
    pattern_doi = re.search(r"10 \d{4,9} [a-z0-9 ]+", pattern)
    return bool(candidate_doi and pattern_doi and candidate_doi.group(0) == pattern_doi.group(0))


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()
