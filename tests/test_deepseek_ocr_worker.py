from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_worker_module():
    root = Path(__file__).resolve().parents[1]
    worker_path = root / "scripts" / "deepseek_ocr_worker.py"
    spec = importlib.util.spec_from_file_location("deepseek_ocr_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_repeat_ngram_processor_blocks_recent_repeat_token() -> None:
    worker = _load_worker_module()
    processor = worker.NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=16)

    input_ids = [10, 11, 12, 10, 11]
    scores = [[0.0] * 64]
    updated = processor(input_ids, scores)

    assert updated[0][12] == float("-inf")


def test_no_repeat_ngram_processor_respects_whitelist() -> None:
    worker = _load_worker_module()
    processor = worker.NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=16, whitelist_token_ids={12})

    input_ids = [10, 11, 12, 10, 11]
    scores = [[0.0] * 64]
    updated = processor(input_ids, scores)

    assert updated[0][12] == 0.0


def test_build_generation_prompt_uses_raw_image_prompt() -> None:
    worker = _load_worker_module()

    def _fake_template(*_args, **_kwargs):
        raise AssertionError("chat template should not be called when prompt already has <image>")

    prompt = worker.build_generation_prompt(
        processor=object(),
        config=object(),
        prompt_text="<image>\n<|grounding|>Convert the document to markdown.",
        apply_chat_template=_fake_template,
    )

    assert prompt.startswith("<image>")

