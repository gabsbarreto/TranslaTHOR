from __future__ import annotations

from app.services.pdf_extraction.page_furniture import (
    PageFurnitureCleanupConfig,
    build_metadata_patterns,
    clean_page_furniture,
    extract_document_metadata,
    is_metadata_like_line,
    merge_document_metadata,
    normalise_for_matching,
)


def test_extract_document_metadata_returns_expected_shape() -> None:
    metadata = extract_document_metadata(
        """Journal of Testing Studies

Understanding OCR Errors in Multi Column Papers

Jane Smith, Alan Brown

doi: 10.1234/example.2024.55

Copyright 2024 Example Publisher. Creative Commons licence.
"""
    )

    assert set(metadata) == {
        "title",
        "short_title",
        "authors",
        "first_author",
        "journal",
        "doi",
        "publisher",
        "year",
        "copyright_or_licence",
    }
    assert metadata["title"] == "Understanding OCR Errors in Multi Column Papers"
    assert metadata["journal"] == "Journal of Testing Studies"
    assert metadata["doi"] == "10.1234/example.2024.55"
    assert metadata["year"] == "2024"
    assert metadata["first_author"] == "Smith"


def test_normalise_for_matching_handles_case_punctuation_accents_and_doi_variants() -> None:
    assert normalise_for_matching("DOI: https://doi.org/10.1234/ÁBC-99") == "doi 10 1234 abc 99"


def test_metadata_fuzzy_matching_matches_title_journal_doi_and_author_patterns() -> None:
    metadata = {
        "title": "Understanding OCR Errors in Multi Column Papers",
        "short_title": "Understanding OCR Errors",
        "authors": ["Jane Smith", "Alan Brown"],
        "first_author": "Smith",
        "journal": "Journal of Testing Studies",
        "doi": "10.1234/example.2024.55",
        "publisher": "",
        "year": "2024",
        "copyright_or_licence": "",
    }
    patterns = build_metadata_patterns(metadata)

    assert is_metadata_like_line("Understanding OCR Erors in Multi-Column Papers", patterns)
    assert is_metadata_like_line("JOURNAL OF TESTING STUDIES", patterns)
    assert is_metadata_like_line("https://doi.org/10.1234/example.2024.55", patterns)
    assert is_metadata_like_line("Smith et al. 2024", patterns)


def test_llm_metadata_can_be_merged_with_heuristic_fallback() -> None:
    merged = merge_document_metadata(
        {
            "title": "LLM Extracted Title",
            "authors": ["Jane Smith"],
            "doi": "",
        },
        {
            "title": "Heuristic Title",
            "short_title": "Heuristic Title",
            "authors": ["Fallback Author"],
            "first_author": "Fallback",
            "journal": "Journal of Testing Studies",
            "doi": "10.1234/example.2024.55",
            "publisher": "Example Publisher",
            "year": "2024",
            "copyright_or_licence": "Copyright 2024",
        },
    )

    assert merged["title"] == "LLM Extracted Title"
    assert merged["authors"] == ["Jane Smith"]
    assert merged["first_author"] == "Smith"
    assert merged["journal"] == "Journal of Testing Studies"
    assert merged["doi"] == "10.1234/example.2024.55"


def test_clean_page_furniture_removes_duplicated_title_header_from_later_page() -> None:
    metadata = {
        "title": "Understanding OCR Errors in Multi Column Papers",
        "short_title": "",
        "authors": ["Jane Smith"],
        "first_author": "Smith",
        "journal": "",
        "doi": "",
        "publisher": "",
        "year": "",
        "copyright_or_licence": "",
    }
    markdown = """Understanding OCR Errors in Multi Column Papers

This body paragraph belongs to page two and should remain untouched because it is current-page content.

## Results

The remaining body text should also remain.
"""

    cleaned = clean_page_furniture(markdown, metadata, 2, PageFurnitureCleanupConfig())

    assert not cleaned.startswith("Understanding OCR Errors")
    assert "This body paragraph belongs to page two" in cleaned
    assert "## Results" in cleaned


def test_clean_page_furniture_removes_smith_et_al_running_header_and_doi_footer() -> None:
    metadata = {
        "title": "Understanding OCR Errors in Multi Column Papers",
        "short_title": "",
        "authors": ["Jane Smith", "Alan Brown"],
        "first_author": "Smith",
        "journal": "",
        "doi": "10.1234/example.2024.55",
        "publisher": "",
        "year": "2024",
        "copyright_or_licence": "",
    }
    markdown = """Smith et al. 2024

This page starts with a normal paragraph after the running header.

More body text remains in the middle of the page.

doi: 10.1234/example.2024.55
"""

    cleaned = clean_page_furniture(markdown, metadata, 3, PageFurnitureCleanupConfig())

    assert "Smith et al. 2024" not in cleaned
    assert "10.1234/example.2024.55" not in cleaned
    assert "This page starts with a normal paragraph" in cleaned


def test_clean_page_furniture_removes_markdown_author_list_heading_on_later_pages() -> None:
    metadata = {
        "title": "Transexualidad adolescencia y educacion",
        "short_title": "",
        "authors": [
            "Bergero Miguel T.",
            "Cano Oncala G.",
            "Esteva de Antonio I.",
            "Giraldo F.",
            "Gornemann Schaffer I.",
            "Álvarez Ortega P.",
        ],
        "first_author": "Bergero",
        "journal": "",
        "doi": "",
        "publisher": "",
        "year": "",
        "copyright_or_licence": "",
    }
    markdown = """## Bergero Miguel T., Cano Oncala G., Esteva de Antonio I., Giraldo F., Gornemann Schaffer I., Álvarez Ortega P.

Es probable que de este modo,

puedan ir determinándose diferentes tipos dentro del trastorno.
"""

    cleaned = clean_page_furniture(markdown, metadata, 3, PageFurnitureCleanupConfig())

    assert "Bergero Miguel" not in cleaned
    assert "Es probable que de este modo" in cleaned


def test_clean_page_furniture_removes_truncated_author_list_header_with_ocr_variation() -> None:
    metadata = {
        "title": "Evaluación Endocrinológica y Tratamiento Hormonal de la Transexualidad",
        "short_title": "",
        "authors": [
            "Esteva de Antonio I.",
            "Giraldo F.",
            "Bergero de Miguel T.",
            "Cano Oncala G.",
            "Crespillo Gómez G.",
            "Ruiz de Adana S.",
            "Rojo Martínez G.",
            "Soriguer Escofet F.",
        ],
        "first_author": "Esteva de Antonio I.",
        "journal": "Cirugía Plástica Ibero-Latinoamericana",
        "doi": "10.64869/PCRL8019",
        "publisher": "",
        "year": "2001",
        "copyright_or_licence": "",
    }
    markdown = """# Esteva de Antonio I., Giraldo F., Bergero de Miguel T., Cano Oncala G., Crespillo Gómez C., Ruiz de Adana S., ...

El seguimiento endocrinológico debe realizarse de forma individualizada.
"""

    cleaned = clean_page_furniture(markdown, metadata, 6, PageFurnitureCleanupConfig())

    assert "Esteva de Antonio" not in cleaned
    assert "El seguimiento endocrinológico" in cleaned


def test_clean_page_furniture_removes_truncated_author_list_bullet_header() -> None:
    metadata = {
        "title": "Evaluación Endocrinológica y Tratamiento Hormonal de la Transexualidad",
        "short_title": "",
        "authors": [
            "Esteva de Antonio I.",
            "Giraldo F.",
            "Bergero de Miguel T.",
            "Cano Oncala G.",
            "Crespillo Gómez G.",
            "Ruiz de Adana S.",
            "Rojo Martínez G.",
            "Soriguer Escofet F.",
        ],
        "first_author": "Esteva de Antonio I.",
        "journal": "Cirugía Plástica Ibero-Latinoamericana",
        "doi": "10.64869/PCRL8019",
        "publisher": "",
        "year": "2001",
        "copyright_or_licence": "",
    }
    markdown = """*   **Esteva de Antonio I., Giraldo F., Bergero de Miguel T., Cano Oncala G., Crespillo Gómez C., Ruiz de Adana S., ...**

Las determinaciones hormonales se ajustaron durante el seguimiento.
"""

    cleaned = clean_page_furniture(markdown, metadata, 4, PageFurnitureCleanupConfig())

    assert "Esteva de Antonio" not in cleaned
    assert "Las determinaciones hormonales" in cleaned


def test_clean_page_furniture_preserves_first_page_title_and_authors() -> None:
    metadata = {
        "title": "Understanding OCR Errors in Multi Column Papers",
        "short_title": "",
        "authors": ["Jane Smith", "Alan Brown"],
        "first_author": "Smith",
        "journal": "",
        "doi": "",
        "publisher": "",
        "year": "",
        "copyright_or_licence": "",
    }
    markdown = """Understanding OCR Errors in Multi Column Papers

Jane Smith, Alan Brown

This is the first body paragraph of the paper.
"""

    cleaned = clean_page_furniture(markdown, metadata, 1, PageFurnitureCleanupConfig())

    assert "Understanding OCR Errors in Multi Column Papers" in cleaned
    assert "Jane Smith, Alan Brown" in cleaned


def test_clean_page_furniture_preserves_real_headings_references_and_body_mentions() -> None:
    metadata = {
        "title": "Understanding OCR Errors in Multi Column Papers",
        "short_title": "",
        "authors": ["Jane Smith"],
        "first_author": "Smith",
        "journal": "Journal of Testing Studies",
        "doi": "",
        "publisher": "",
        "year": "2024",
        "copyright_or_licence": "",
    }
    markdown = """## Introduction

This paragraph mentions the Journal of Testing Studies naturally in the body and must not be removed.

## References

Smith, J. (2024). Understanding OCR Errors in Multi Column Papers. Journal of Testing Studies.
"""

    cleaned = clean_page_furniture(markdown, metadata, 2, PageFurnitureCleanupConfig())

    assert "## Introduction" in cleaned
    assert "Journal of Testing Studies naturally" in cleaned
    assert "## References" in cleaned
    assert "Understanding OCR Errors in Multi Column Papers. Journal" in cleaned
