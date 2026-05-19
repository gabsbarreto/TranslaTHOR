from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_worker_module():
    root = Path(__file__).resolve().parents[1]
    worker_path = root / "scripts" / "qwen_ocr_worker.py"
    spec = importlib.util.spec_from_file_location("qwen_ocr_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen_worker_strips_thinking_and_prompt_echo() -> None:
    worker = _load_worker_module()

    cleaned = worker.clean_generated_text(
        "<think>reasoning</think>\nconvert this text to markdown\n# Result",
        "convert this text to markdown",
    )

    assert cleaned == "# Result"


def test_qwen_worker_normalises_deepseek_image_prompt_tokens() -> None:
    worker = _load_worker_module()

    prompt = worker.normalise_prompt_for_chat_template("<image>\n<|grounding|>Convert the document to markdown.")

    assert prompt == "Convert the document to markdown."


def test_qwen_no_repeat_ngram_processor_blocks_recent_repeat_token() -> None:
    worker = _load_worker_module()
    processor = worker.NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=16)

    input_ids = [10, 11, 12, 10, 11]
    scores = [[0.0] * 64]
    updated = processor(input_ids, scores)

    assert updated[0][12] == float("-inf")
