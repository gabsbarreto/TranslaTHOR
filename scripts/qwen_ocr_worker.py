from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "Convert this document image to Markdown. "
    "Return only the Markdown body. "
    "Do not wrap the output in code fences. "
    "Do not include ```markdown, <page>, page labels, reasoning, explanations, or <think> blocks."
)
TABLE_WHITELIST_TOKENS = ("<td>", "</td>")
MIN_CONTEXT_WORDS = 8
MIN_CONTEXT_CHARS = 45


def parse_bool_flag(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _flatten_token_ids(values: Any) -> list[int]:
    raw = values.tolist() if hasattr(values, "tolist") else values
    out: list[int] = []

    def walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
            return
        try:
            out.append(int(item))
        except Exception:
            return

    walk(raw)
    return out


def _mask_scores(scores: Any, banned_tokens: set[int]) -> Any:
    if not banned_tokens:
        return scores

    if isinstance(scores, list):
        cloned = [row[:] if isinstance(row, list) else row for row in scores]
        if cloned and isinstance(cloned[0], list):
            vocab_size = len(cloned[0])
            for token in banned_tokens:
                if 0 <= token < vocab_size:
                    cloned[0][token] = float("-inf")
            return cloned
        vocab_size = len(cloned)
        for token in banned_tokens:
            if 0 <= token < vocab_size:
                cloned[token] = float("-inf")
        return cloned

    try:
        import mlx.core as mx

        vocab_size = int(scores.shape[-1])
        mask = [False] * vocab_size
        for token in banned_tokens:
            if 0 <= token < vocab_size:
                mask[token] = True
        mask_array = mx.array(mask)[None, :]
        neg_inf = mx.full(scores.shape, -float("inf"), dtype=scores.dtype)
        return mx.where(mask_array, neg_inf, scores)
    except Exception:
        return scores


class NoRepeatNGramLogitsProcessor:
    def __init__(
        self,
        ngram_size: int,
        window_size: int = 100,
        whitelist_token_ids: set[int] | None = None,
    ) -> None:
        if ngram_size <= 0:
            raise ValueError("ngram_size must be > 0")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.ngram_size = int(ngram_size)
        self.window_size = int(window_size)
        self.whitelist_token_ids = set(whitelist_token_ids or set())

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        token_ids = _flatten_token_ids(input_ids)
        if len(token_ids) < self.ngram_size:
            return scores

        prefix = tuple(token_ids[-(self.ngram_size - 1) :]) if self.ngram_size > 1 else tuple()
        search_start = max(0, len(token_ids) - self.window_size)
        search_end = len(token_ids) - self.ngram_size + 1

        banned_tokens: set[int] = set()
        for i in range(search_start, max(search_end, 0)):
            ngram = tuple(token_ids[i : i + self.ngram_size])
            if ngram[:-1] == prefix:
                banned_tokens.add(int(ngram[-1]))

        banned_tokens -= self.whitelist_token_ids
        return _mask_scores(scores, banned_tokens)


def normalise_prompt_for_chat_template(prompt_text: str) -> str:
    text = str(prompt_text).strip()
    text = re.sub(r"^\s*<image>\s*", "", text)
    text = text.replace("<|grounding|>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def build_ocr_chat_messages(system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": normalise_prompt_for_chat_template(system_prompt)},
        {"role": "user", "content": ""},
    ]


def clean_context_candidate_blocks(markdown: str) -> list[str]:
    text = str(markdown)
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    text = re.sub(r"(?is)<!--\s*page-(?:header|footer|number)\s*:[\s\S]*?-->", "\n\n", text)
    text = re.sub(r"(?m)^```.*?$", "", text)
    blocks: list[str] = []
    in_references = False

    for raw_block in re.split(r"\n\s*\n+", text):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if not lines:
            continue
        if any(_is_references_heading(line) for line in lines):
            in_references = True
            continue
        if in_references:
            continue
        if any(_is_structural_context_line(line) for line in lines):
            continue
        block = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if _is_body_context_candidate(block):
            blocks.append(block)

    return blocks


def extract_previous_page_context(markdown: str, n_paragraphs: int = 2, max_chars: int = 1600) -> str:
    if n_paragraphs <= 0 or max_chars <= 0:
        return ""
    candidates = clean_context_candidate_blocks(markdown)
    selected = candidates[-max(int(n_paragraphs), 1) :]
    while selected:
        context = "\n\n".join(selected).strip()
        if len(context) <= max_chars:
            return context
        if len(selected) == 1:
            return f"...{context[-max_chars + 3:].lstrip()}" if max_chars > 3 else context[-max_chars:]
        selected = selected[1:]
    return ""


def build_ocr_prompt(base_prompt: str, previous_page_context: str | None = None) -> str:
    base = normalise_prompt_for_chat_template(base_prompt)
    context = str(previous_page_context or "").strip()
    if not context:
        return base
    return f"""{base}

Previous-page context for continuity only:
---
{context}
---

Current page task:
Transcribe only the visible text from the current page image into clean Markdown.

Use the previous-page context only to decide whether the first lines of this page are body-text continuation, a genuine heading, or a running page header.
The previous-page context is not part of this page. Never copy it into the output.

Include:
- body text visible on the current page;
- genuine headings and subheadings visible on the current page;
- lists, tables, captions, and footnotes when they are part of the page body.

Exclude:
- running page headers;
- page footers;
- page numbers;
- article/chapter/journal metadata that appears in the top or bottom margin;
- repeated or variable page metadata such as author names, journal names, article titles, short titles, DOI labels, journal labels, volume/issue labels, or chapter labels.

Decision rule:
At the top of the page, keep a line only if it is part of the document body or a genuine section heading.
Remove it if it is visually separated in the margin and does not connect naturally with either the previous-page context or the body text that follows.
Do not remove real headings such as Methods, Results, Discussion, or numbered section headings when they introduce the following text.
Keep body text that continues from the previous page.

Return only the Markdown transcription for the current page."""


def _is_structural_context_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if re.match(r"^#{1,6}\s+", stripped):
        return True
    if re.match(r"^[-*_]{3,}$", stripped):
        return True
    if re.match(r"^(?:page\s*)?\d{1,4}$", stripped, flags=re.IGNORECASE):
        return True
    if re.match(r"^[ivxlcdm]{1,8}$", stripped, flags=re.IGNORECASE):
        return True
    if stripped.startswith("|") or re.match(r"^\|?\s*:?-{3,}:?\s*(?:\|.*)?$", stripped):
        return True
    if re.match(r"^(?:fig(?:ure)?|table|chart|graph|source|note|notes|caption|quadro|tabela|fonte)\b", stripped, flags=re.IGNORECASE):
        return True
    return False


def _is_body_context_candidate(block: str) -> bool:
    if len(block) < MIN_CONTEXT_CHARS:
        return False
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", block)
    if len(words) < MIN_CONTEXT_WORDS:
        return False
    if re.match(r"^(?:fig(?:ure)?|table|source|note|notes|references|bibliography)\b", block, flags=re.IGNORECASE):
        return False
    if _looks_like_reference_entry(block):
        return False
    return True


def _is_references_heading(line: str) -> bool:
    stripped = re.sub(r"^#{1,6}\s+", "", line.strip()).strip(":")
    return stripped.lower() in {"references", "bibliography", "works cited", "literature cited"}


def _looks_like_reference_entry(block: str) -> bool:
    if re.search(r"\(\d{4}[a-z]?\)", block) and re.search(r"\bdoi\b|https?://|[A-Z][a-z]+,\s+[A-Z]\.", block):
        return True
    if re.match(r"^\[\d+\]\s+", block):
        return True
    return False


def clean_generated_text(text: str, prompt: str) -> str:
    text = str(text).strip()
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^>]+?\|>", "", text)

    lines = text.splitlines()
    if lines and lines[0].strip() == "<image>":
        lines = lines[1:]
    if lines and _looks_like_echoed_prompt(lines[0]):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    text = "\n".join(lines).strip()

    cleaned_prompt = normalise_prompt_for_chat_template(prompt)
    return text.replace(prompt, "").replace(cleaned_prompt, "").strip()


def _looks_like_echoed_prompt(line: str) -> bool:
    lowered = line.lower()
    return "convert" in lowered and ("markdown" in lowered or "document" in lowered)


def build_generation_prompt(
    processor: Any,
    config: Any,
    system_prompt: str,
    apply_chat_template: Any,
    enable_thinking: bool = False,
) -> str:
    messages = build_ocr_chat_messages(system_prompt)
    template_kwargs: dict[str, Any] = {
        "num_images": 1,
        "num_audios": 0,
        "enable_thinking": bool(enable_thinking),
    }
    try:
        return apply_chat_template(processor, config, messages, **template_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "enable_thinking" in message:
            template_kwargs.pop("enable_thinking", None)
            return apply_chat_template(processor, config, messages, **template_kwargs)
        if "num_audios" in message:
            template_kwargs.pop("num_audios", None)
            return apply_chat_template(processor, config, messages, **template_kwargs)
        raise


@contextmanager
def patch_batch_apply_chat_template(enable_thinking: bool = False):
    import importlib

    mlx_generate_module = importlib.import_module("mlx_vlm.generate")
    if not hasattr(mlx_generate_module, "apply_chat_template"):
        raise AttributeError("Could not find apply_chat_template inside mlx_vlm.generate.")

    original_apply_chat_template = mlx_generate_module.apply_chat_template

    def wrapped_apply_chat_template(processor: Any, config: Any, prompt: Any, **kwargs: Any) -> str:
        kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            return original_apply_chat_template(processor, config, prompt, **kwargs)
        except TypeError as exc:
            if "enable_thinking" in str(exc):
                kwargs.pop("enable_thinking", None)
                return original_apply_chat_template(processor, config, prompt, **kwargs)
            raise

    mlx_generate_module.apply_chat_template = wrapped_apply_chat_template
    try:
        yield
    finally:
        mlx_generate_module.apply_chat_template = original_apply_chat_template


def resolve_whitelist_token_ids(processor: Any) -> set[int]:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    whitelist_ids: set[int] = set()
    for token in TABLE_WHITELIST_TOKENS:
        try:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
        except Exception:
            continue
        if len(token_ids) == 1:
            whitelist_ids.add(int(token_ids[0]))
    whitelist_ids.update({128821, 128822})
    return whitelist_ids


def build_logits_processors(
    processor: Any,
    skip_repeat: bool,
    ngram_size: int,
    ngram_window: int,
) -> list[NoRepeatNGramLogitsProcessor]:
    if not skip_repeat:
        return []
    return [
        NoRepeatNGramLogitsProcessor(
            ngram_size=ngram_size,
            window_size=ngram_window,
            whitelist_token_ids=resolve_whitelist_token_ids(processor),
        )
    ]


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def chunk_list(values: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("size must be > 0")
    return [values[i : i + size] for i in range(0, len(values), size)]


class FallbackResponse:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts


def load_qwen_vlm(model_name: str):
    from mlx_vlm import load

    return load(model_name)


def cleanup_mlx() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
        mx.clear_streams()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen 3.5 VLM OCR over multiple images.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--images-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--crop-mode", type=parse_bool_flag, default=True)
    parser.add_argument("--min-crops", type=int, default=1)
    parser.add_argument("--max-crops", type=int, default=6)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--skip-repeat", type=parse_bool_flag, default=True)
    parser.add_argument("--ngram-size", type=int, default=20)
    parser.add_argument("--ngram-window", type=int, default=90)
    parser.add_argument("--names-json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--enable-thinking", type=parse_bool_flag, default=False)
    parser.add_argument("--fallback-to-single", type=parse_bool_flag, default=True)
    parser.add_argument("--verbose", type=parse_bool_flag, default=True)
    parser.add_argument("--use-previous-page-context", type=parse_bool_flag, default=True)
    parser.add_argument("--previous-context-paragraphs", type=int, default=2)
    parser.add_argument("--max-previous-context-chars", type=int, default=1600)
    args = parser.parse_args()

    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from mlx_vlm.generate import batch_generate, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    image_paths = [Path(path) for path in json.loads(args.images_json)]
    names = json.loads(args.names_json) if args.names_json else []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_crops = max(int(args.min_crops), 0)
    max_crops = max(int(args.max_crops), min_crops)
    requested_batch_size = max(int(args.batch_size), 1)
    force_single_for_penalties = (
        float(args.presence_penalty) != 0.0 or float(args.repetition_penalty) != 1.0
    )
    use_previous_page_context = bool(args.use_previous_page_context)
    # Page N needs the cleaned Markdown from page N-1 before its prompt can be built,
    # so continuity-aware header detection must run sequentially.
    use_batch = requested_batch_size > 1 and not force_single_for_penalties and not use_previous_page_context
    prompt_text = str(args.prompt)

    emit({"event": "model_loading", "model": str(args.model)})
    model, processor = load_qwen_vlm(str(args.model))
    config = model.config
    logits_processors = make_logits_processors(
        presence_penalty=float(args.presence_penalty),
        repetition_penalty=max(float(args.repetition_penalty), 1e-6),
    )
    logits_processors.extend(
        build_logits_processors(
            processor=processor,
            skip_repeat=bool(args.skip_repeat),
            ngram_size=max(int(args.ngram_size), 1),
            ngram_window=max(int(args.ngram_window), 1),
        )
    )
    emit(
        {
            "event": "model_loaded",
            "model": str(args.model),
            "pages": len(image_paths),
            "requested_batch_size": requested_batch_size,
            "use_batch_generate": bool(use_batch),
            "batch_disabled_for_penalties": bool(force_single_for_penalties),
            "batch_disabled_for_previous_page_context": bool(use_previous_page_context),
            "use_previous_page_context": bool(use_previous_page_context),
            "previous_context_paragraphs": max(int(args.previous_context_paragraphs), 0),
            "max_previous_context_chars": max(int(args.max_previous_context_chars), 0),
            "enable_thinking": bool(args.enable_thinking),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "min_p": float(args.min_p),
            "presence_penalty": float(args.presence_penalty),
            "repetition_penalty": float(args.repetition_penalty),
            "max_tokens": int(args.max_tokens),
            "crop_mode": bool(args.crop_mode),
            "min_crops": min_crops,
            "max_crops": max_crops,
            "base_size": int(args.base_size),
            "image_size": int(args.image_size),
            "skip_repeat": bool(args.skip_repeat),
            "fallback_to_single": bool(args.fallback_to_single),
        }
    )

    try:
        sampler = make_sampler(
            temp=max(float(args.temperature), 0.0),
            top_p=max(0.0, min(1.0, float(args.top_p))),
            top_k=max(0, int(args.top_k)),
            min_p=max(0.0, min(1.0, float(args.min_p))),
        )

        if use_batch:
            prompts = [build_ocr_chat_messages(prompt_text) for _ in image_paths]
            image_path_strings = [str(path) for path in image_paths]
            prompt_batches = chunk_list(prompts, requested_batch_size)
            image_batches = chunk_list(image_path_strings, requested_batch_size)
            output_index = 1

            for batch_idx, (prompt_batch, image_batch) in enumerate(
                zip(prompt_batches, image_batches), start=1
            ):
                emit(
                    {
                        "event": "batch_started",
                        "batch_index": batch_idx,
                        "batch_size": len(image_batch),
                        "batch_size_limit": requested_batch_size,
                        "total": len(image_paths),
                        "enable_thinking": bool(args.enable_thinking),
                    }
                )
                try:
                    with patch_batch_apply_chat_template(enable_thinking=bool(args.enable_thinking)):
                        response = batch_generate(
                            model,
                            processor,
                            images=image_batch,
                            prompts=prompt_batch,
                            max_tokens=int(args.max_tokens),
                            verbose=bool(args.verbose),
                            group_by_shape=True,
                            track_image_sizes=False,
                            sampler=sampler,
                        )
                except Exception as exc:
                    if not bool(args.fallback_to_single):
                        raise
                    emit(
                        {
                            "event": "batch_failed_falling_back_to_single",
                            "batch_index": batch_idx,
                            "error": str(exc),
                        }
                    )
                    fallback_texts: list[str] = []
                    for _single_prompt, single_image in zip(prompt_batch, image_batch):
                        formatted_prompt = build_generation_prompt(
                            processor=processor,
                            config=config,
                            system_prompt=prompt_text,
                            apply_chat_template=apply_chat_template,
                            enable_thinking=bool(args.enable_thinking),
                        )
                        result = generate(
                            model,
                            processor,
                            formatted_prompt,
                            image=[single_image],
                            temperature=float(args.temperature),
                            top_p=float(args.top_p),
                            top_k=int(args.top_k),
                            min_p=float(args.min_p),
                            max_tokens=int(args.max_tokens),
                            base_size=int(args.base_size),
                            image_size=int(args.image_size),
                            cropping=bool(args.crop_mode),
                            min_patches=min_crops,
                            max_patches=max_crops,
                            logits_processors=logits_processors,
                            verbose=bool(args.verbose),
                        )
                        fallback_texts.append(str(result.text))
                    response = FallbackResponse(fallback_texts)

                for text in response.texts:
                    image_path = image_paths[output_index - 1]
                    emit(
                        {
                            "event": "page_started",
                            "index": output_index,
                            "total": len(image_paths),
                            "image": str(image_path),
                        }
                    )
                    markdown = clean_generated_text(str(text), prompt_text)
                    stem = (
                        str(names[output_index - 1])
                        if output_index - 1 < len(names)
                        else f"page_{output_index:04d}"
                    )
                    output_path = output_dir / f"{stem}.md"
                    output_path.write_text(markdown, encoding="utf-8")
                    emit(
                        {
                            "event": "page_done",
                            "index": output_index,
                            "total": len(image_paths),
                            "output": str(output_path),
                            "chars": len(markdown),
                        }
                    )
                    output_index += 1
        else:
            previous_page_context = ""
            for index, image_path in enumerate(image_paths, start=1):
                emit(
                    {
                        "event": "page_started",
                        "index": index,
                        "total": len(image_paths),
                        "image": str(image_path),
                        "previous_page_context_chars": len(previous_page_context),
                    }
                )
                page_prompt = (
                    build_ocr_prompt(prompt_text, previous_page_context)
                    if use_previous_page_context and previous_page_context
                    else prompt_text
                )
                formatted_prompt = build_generation_prompt(
                    processor=processor,
                    config=config,
                    system_prompt=page_prompt,
                    apply_chat_template=apply_chat_template,
                    enable_thinking=bool(args.enable_thinking),
                )
                result = generate(
                    model,
                    processor,
                    formatted_prompt,
                    image=[str(image_path)],
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    min_p=float(args.min_p),
                    max_tokens=int(args.max_tokens),
                    base_size=int(args.base_size),
                    image_size=int(args.image_size),
                    cropping=bool(args.crop_mode),
                    min_patches=min_crops,
                    max_patches=max_crops,
                    logits_processors=logits_processors,
                    verbose=bool(args.verbose),
                )
                markdown = clean_generated_text(str(result.text), page_prompt)
                stem = str(names[index - 1]) if index - 1 < len(names) else f"page_{index:04d}"
                output_path = output_dir / f"{stem}.md"
                output_path.write_text(markdown, encoding="utf-8")
                previous_page_context = (
                    extract_previous_page_context(
                        markdown,
                        n_paragraphs=max(int(args.previous_context_paragraphs), 0),
                        max_chars=max(int(args.max_previous_context_chars), 0),
                    )
                    if use_previous_page_context
                    else ""
                )
                emit(
                    {
                        "event": "page_done",
                        "index": index,
                        "total": len(image_paths),
                        "output": str(output_path),
                        "chars": len(markdown),
                        "next_previous_page_context_chars": len(previous_page_context),
                    }
                )

        emit({"event": "complete", "pages": len(image_paths)})
    finally:
        cleanup_mlx()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
