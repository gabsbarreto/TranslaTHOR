from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
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


def empty_metadata() -> dict[str, Any]:
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


def coerce_metadata(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    metadata = empty_metadata()
    for key in METADATA_KEYS:
        raw = source.get(key)
        if key == "authors":
            if isinstance(raw, list):
                metadata[key] = [str(item).strip() for item in raw if str(item).strip()]
            elif isinstance(raw, str) and raw.strip():
                metadata[key] = [part.strip() for part in re.split(r"\s*(?:,|;|\band\b|&)\s*", raw) if part.strip()]
            continue
        metadata[key] = str(raw or "").strip()
    return metadata


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    stripped = re.sub(r"(?is)^```(?:json)?\s*|\s*```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        raise ValueError("No JSON object found in metadata model output.")
    return json.loads(match.group(0))


def build_prompt(text: str) -> list[dict[str, str]]:
    schema = json.dumps(empty_metadata(), ensure_ascii=False, indent=2)
    system = (
        "You extract bibliographic metadata from OCR text of an academic paper. "
        "Return only one valid JSON object. Do not include Markdown fences, comments, or explanations. "
        "Use empty strings or an empty authors array when a field is not visible. "
        "Do not guess values that are not supported by the OCR text."
    )
    user = f"""Extract this exact JSON schema from the OCR text:

{schema}

Rules:
- title: the paper/article/chapter title, not a running header.
- short_title: visible short/running title only if explicit; otherwise a concise shortened form of the title.
- authors: visible author names as an array.
- first_author: surname/family name of the first author.
- journal: journal, proceedings, book, or venue name if visible.
- doi: DOI only, without URL prefix when possible.
- publisher: publisher or platform if visible.
- year: publication year if visible.
- copyright_or_licence: visible copyright, open-access, or licence line.

OCR text:
---
{text}
---
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract document metadata from OCR text with an MLX language model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-input-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=700)
    args = parser.parse_args()

    from mlx_lm import generate, load, sample_utils

    input_text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    input_text = input_text[: max(int(args.max_input_chars), 1)]
    model, tokenizer = load(str(args.model))
    messages = build_prompt(input_text)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nJSON:"

    sampler = sample_utils.make_sampler(temp=0.0, top_p=1.0, top_k=0)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=max(int(args.max_tokens), 128), sampler=sampler)
    metadata = coerce_metadata(extract_json_object(str(raw)))
    Path(args.output).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
