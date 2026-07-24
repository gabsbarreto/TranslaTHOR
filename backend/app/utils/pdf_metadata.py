from __future__ import annotations


def optional_pdf_metadata_text(value: object) -> str | None:
    """Normalize optional PDF metadata without stringifying PDF null objects."""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
