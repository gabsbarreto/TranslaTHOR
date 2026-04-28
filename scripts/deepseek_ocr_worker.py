from __future__ import annotations

import argparse
import json
import re
from typing import Any
from pathlib import Path

DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
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


def clean_generated_text(text: str, prompt: str) -> str:
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].strip() == "<image>":
        lines = lines[1:]
    if lines and _looks_like_echoed_prompt(lines[0]):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    text = "\n".join(lines).strip()
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = text.replace(prompt, "").strip()
    return text


def _looks_like_echoed_prompt(line: str) -> bool:
    lowered = line.lower()
    return "convert" in lowered and ("markdown" in lowered or "document" in lowered)


def build_generation_prompt(processor: Any, config: Any, prompt_text: str, apply_chat_template: Any) -> str:
    text = str(prompt_text).strip()
    if "<image>" in text:
        return text
    return apply_chat_template(processor, config, text, num_images=1, num_audios=0)


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


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR-2 over multiple page images with one model load.")
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
    args = parser.parse_args()

    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.generate import generate

    prompt_text = str(args.prompt)
    min_crops = max(int(args.min_crops), 0)
    max_crops = max(int(args.max_crops), min_crops)
    image_paths = [Path(p) for p in json.loads(args.images_json)]
    names = json.loads(args.names_json) if args.names_json else []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emit({"event": "model_loading", "model": args.model})
    model, processor = load_deepseek_ocr(args.model)
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
            "pages": len(image_paths),
            "crop_mode": bool(args.crop_mode),
            "min_crops": min_crops,
            "max_crops": max_crops,
            "base_size": int(args.base_size),
            "image_size": int(args.image_size),
            "skip_repeat": bool(args.skip_repeat),
        }
    )

    try:
        for index, image_path in enumerate(image_paths, start=1):
            emit({"event": "page_started", "index": index, "total": len(image_paths), "image": str(image_path)})
            prompt = build_generation_prompt(processor, config, prompt_text, apply_chat_template)
            result = generate(
                model,
                processor,
                prompt,
                image=[str(image_path)],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                base_size=int(args.base_size),
                image_size=int(args.image_size),
                cropping=bool(args.crop_mode),
                min_patches=min_crops,
                max_patches=max_crops,
                logits_processors=logits_processors,
                verbose=False,
            )
            markdown = clean_generated_text(result.text, prompt_text)
            stem = str(names[index - 1]) if index - 1 < len(names) else f"page_{index:04d}"
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

        emit({"event": "complete", "pages": len(image_paths)})
    finally:
        cleanup_mlx()
    return 0


def load_deepseek_ocr(model_name: str):
    """Load DeepSeek-OCR-2 once without mlx-vlm's generic processor fallback.

    Current mlx-vlm can generate with this model from its CLI, but the generic
    `load()` helper fails to resolve the DeepSeek-OCR-2 processor in-process.
    Loading the model and processor explicitly keeps one worker process alive
    for the whole PDF while avoiding one CLI/model load per page.
    """

    from mlx_vlm.models.deepseekocr_2.processing_deepseekocr import DeepseekOCR2Processor
    from mlx_vlm.utils import (
        StoppingCriteria,
        get_model_path,
        load_image_processor,
        load_model,
        load_tokenizer,
    )

    model_path = get_model_path(model_name)
    model = load_model(model_path)
    processor = DeepseekOCR2Processor.from_pretrained(
        model_path,
        trust_remote_code=False,
        local_files_only=True,
    )

    image_processor = load_image_processor(model_path)
    if image_processor is not None:
        processor.image_processor = image_processor

    detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
    tokenizer_obj = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    processor.detokenizer = detokenizer_class(tokenizer_obj)

    eos_token_id = getattr(model.config, "eos_token_id", None)
    final_eos_token_ids = eos_token_id if eos_token_id is not None else tokenizer_obj.eos_token_ids
    criteria = StoppingCriteria(final_eos_token_ids, tokenizer_obj)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.stopping_criteria = criteria
    else:
        processor.stopping_criteria = criteria

    return model, processor


def cleanup_mlx() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
        mx.clear_streams()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
