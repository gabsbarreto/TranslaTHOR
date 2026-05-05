from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "Convert the document image to markdown. "
    "Return only the markdown content. "
    "Do not include reasoning, explanations, or <think> blocks."
)

TABLE_WHITELIST_TOKENS = ("<td>", "</td>")


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

        prefix = tuple(token_ids[-(self.ngram_size - 1):]) if self.ngram_size > 1 else tuple()
        search_start = max(0, len(token_ids) - self.window_size)
        search_end = len(token_ids) - self.ngram_size + 1

        banned_tokens: set[int] = set()

        for i in range(search_start, max(search_end, 0)):
            ngram = tuple(token_ids[i : i + self.ngram_size])
            if ngram[:-1] == prefix:
                banned_tokens.add(int(ngram[-1]))

        banned_tokens -= self.whitelist_token_ids

        return _mask_scores(scores, banned_tokens)


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
    text = text.replace(prompt, "").replace(cleaned_prompt, "").strip()

    return text


def _looks_like_echoed_prompt(line: str) -> bool:
    lowered = line.lower()
    return "convert" in lowered and ("markdown" in lowered or "document" in lowered)


def normalise_prompt_for_chat_template(prompt_text: str) -> str:
    text = str(prompt_text).strip()

    # Remove accidental DeepSeek / raw-image prompt tokens.
    # Qwen batch_generate should receive plain user text.
    text = re.sub(r"^\s*<image>\s*", "", text)
    text = text.replace("<|grounding|>", "")

    # Avoid duplicated whitespace after token removal.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def build_generation_prompt(
    processor: Any,
    config: Any,
    prompt_text: str,
    apply_chat_template: Any,
    enable_thinking: bool = False,
) -> str:
    """Build a formatted prompt for single-image generate().

    Important:
    - Use this for generate().
    - Do NOT use this before batch_generate(), because batch_generate()
      applies the chat template internally.
    """
    text = normalise_prompt_for_chat_template(prompt_text)

    template_kwargs: dict[str, Any] = {
        "num_images": 1,
        "num_audios": 0,
        "enable_thinking": bool(enable_thinking),
    }

    try:
        return apply_chat_template(
            processor,
            config,
            text,
            **template_kwargs,
        )

    except TypeError as exc:
        message = str(exc)

        if "enable_thinking" in message:
            template_kwargs.pop("enable_thinking", None)
            return apply_chat_template(
                processor,
                config,
                text,
                **template_kwargs,
            )

        if "num_audios" in message:
            template_kwargs.pop("num_audios", None)
            return apply_chat_template(
                processor,
                config,
                text,
                **template_kwargs,
            )

        raise


@contextmanager
def patch_batch_apply_chat_template(enable_thinking: bool = False):
    """Patch MLX-VLM's internal batch chat-template call.

    batch_generate() applies apply_chat_template() internally.
    Therefore, we must not pre-format prompts before batch_generate().

    This patch injects enable_thinking=False into that internal call.
    """
    import importlib

    mlx_generate_module = importlib.import_module("mlx_vlm.generate")

    if not hasattr(mlx_generate_module, "apply_chat_template"):
        raise AttributeError(
            "Could not find apply_chat_template inside mlx_vlm.generate. "
            "Your mlx-vlm version may have moved the internal batch prompt formatter."
        )

    original_apply_chat_template = mlx_generate_module.apply_chat_template

    def wrapped_apply_chat_template(
        processor: Any,
        config: Any,
        prompt: Any,
        **kwargs: Any,
    ) -> str:
        kwargs["enable_thinking"] = bool(enable_thinking)

        try:
            return original_apply_chat_template(
                processor,
                config,
                prompt,
                **kwargs,
            )

        except TypeError as exc:
            if "enable_thinking" in str(exc):
                kwargs.pop("enable_thinking", None)
                return original_apply_chat_template(
                    processor,
                    config,
                    prompt,
                    **kwargs,
                )

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

    # Some Qwen-ish special/table tokens seen in previous runs.
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
    parser = argparse.ArgumentParser(
        description="Run Qwen 3.5 VLM OCR/markdown extraction over multiple images using mlx-vlm batch_generate."
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--images-json", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    parser.add_argument("--batch-size", type=int, default=4)

    # Important:
    # Default False means Qwen thinking is disabled unless explicitly turned on.
    parser.add_argument("--enable-thinking", type=parse_bool_flag, default=False)

    # Optional safety switch.
    # If batch_generate fails, retry the failed batch one image at a time.
    parser.add_argument("--fallback-to-single", type=parse_bool_flag, default=True)

    args = parser.parse_args()

    from mlx_lm.sample_utils import make_sampler
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.generate import batch_generate, generate

    model_name = str(args.model)
    prompt_text = str(args.prompt)

    image_paths = [Path(p) for p in json.loads(args.images_json)]
    names = json.loads(args.names_json) if args.names_json else []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_crops = max(int(args.min_crops), 0)
    max_crops = max(int(args.max_crops), min_crops)

    requested_batch_size = max(int(args.batch_size), 1)
    use_batch = requested_batch_size > 1

    emit(
        {
            "event": "model_loading",
            "model": model_name,
        }
    )

    model, processor = load_qwen_vlm(model_name)
    config = model.config

    logits_processors = build_logits_processors(
        processor=processor,
        skip_repeat=bool(args.skip_repeat),
        ngram_size=max(int(args.ngram_size), 1),
        ngram_window=max(int(args.ngram_window), 1),
    )

    emit(
        {
            "event": "model_loaded",
            "model": model_name,
            "pages": len(image_paths),
            "requested_batch_size": requested_batch_size,
            "use_batch_generate": bool(use_batch),
            "enable_thinking": bool(args.enable_thinking),
            "temperature": float(args.temperature),
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
        sampler = None
        if float(args.temperature) > 0.0:
            sampler = make_sampler(
                temp=float(args.temperature),
                top_p=1.0,
            )

        if use_batch:
            # IMPORTANT:
            # For batch_generate(), pass plain cleaned prompts.
            # Do NOT pre-apply apply_chat_template().
            prompts = [
                normalise_prompt_for_chat_template(prompt_text)
                for _ in image_paths
            ]

            image_path_strings = [str(p) for p in image_paths]

            prompt_batches = chunk_list(prompts, requested_batch_size)
            image_batches = chunk_list(image_path_strings, requested_batch_size)

            output_index = 1

            for batch_idx, (prompt_batch, image_batch) in enumerate(
                zip(prompt_batches, image_batches),
                start=1,
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
                    with patch_batch_apply_chat_template(
                        enable_thinking=bool(args.enable_thinking),
                    ):
                        response = batch_generate(
                            model,
                            processor,
                            images=image_batch,
                            prompts=prompt_batch,
                            max_tokens=int(args.max_tokens),
                            verbose=False,
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

                    for single_prompt, single_image in zip(prompt_batch, image_batch):
                        formatted_prompt = build_generation_prompt(
                            processor=processor,
                            config=config,
                            prompt_text=single_prompt,
                            apply_chat_template=apply_chat_template,
                            enable_thinking=bool(args.enable_thinking),
                        )

                        result = generate(
                            model,
                            processor,
                            formatted_prompt,
                            image=[single_image],
                            temperature=float(args.temperature),
                            max_tokens=int(args.max_tokens),
                            base_size=int(args.base_size),
                            image_size=int(args.image_size),
                            cropping=bool(args.crop_mode),
                            min_patches=min_crops,
                            max_patches=max_crops,
                            logits_processors=logits_processors,
                            verbose=False,
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
            for index, image_path in enumerate(image_paths, start=1):
                emit(
                    {
                        "event": "page_started",
                        "index": index,
                        "total": len(image_paths),
                        "image": str(image_path),
                    }
                )

                formatted_prompt = build_generation_prompt(
                    processor=processor,
                    config=config,
                    prompt_text=prompt_text,
                    apply_chat_template=apply_chat_template,
                    enable_thinking=bool(args.enable_thinking),
                )

                result = generate(
                    model,
                    processor,
                    formatted_prompt,
                    image=[str(image_path)],
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    base_size=int(args.base_size),
                    image_size=int(args.image_size),
                    cropping=bool(args.crop_mode),
                    min_patches=min_crops,
                    max_patches=max_crops,
                    logits_processors=logits_processors,
                    verbose=False,
                )

                markdown = clean_generated_text(str(result.text), prompt_text)

                stem = (
                    str(names[index - 1])
                    if index - 1 < len(names)
                    else f"page_{index:04d}"
                )

                output_path = output_dir / f"{stem}.md"
                output_path.write_text(markdown, encoding="utf-8")

                emit(
                    {
                        "event": "page_done",
                        "index": index,
                        "total": len(image_paths),
                        "output": str(output_path),
                        "chars": len(markdown),
                    }
                )

        emit(
            {
                "event": "complete",
                "pages": len(image_paths),
            }
        )

    finally:
        cleanup_mlx()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())