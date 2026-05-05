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


def test_build_generation_prompt_normalizes_raw_image_prompt() -> None:
    worker = _load_worker_module()
    seen = {}

    def _fake_template(_processor, _config, text, **kwargs):
        seen["text"] = text
        seen["kwargs"] = kwargs
        return "formatted"

    prompt = worker.build_generation_prompt(
        processor=object(),
        config=object(),
        prompt_text="<image>\n<|grounding|>Convert the document to markdown.",
        apply_chat_template=_fake_template,
        enable_thinking=False,
    )

    assert prompt == "formatted"
    assert seen["text"] == "Convert the document to markdown."
    assert seen["kwargs"]["enable_thinking"] is False


def test_build_generation_prompt_sets_no_thinking_for_qwen35() -> None:
    worker = _load_worker_module()
    seen_kwargs = {}

    def _fake_template(_processor, _config, _text, **kwargs):
        seen_kwargs.update(kwargs)
        return "ok"

    out = worker.build_generation_prompt(
        processor=object(),
        config=object(),
        prompt_text="convert this text to markdown",
        apply_chat_template=_fake_template,
        enable_thinking=False,
    )

    assert out == "ok"
    assert seen_kwargs.get("enable_thinking") is False


def test_load_qwen_vlm_uses_mlx_vlm_load(monkeypatch) -> None:
    worker = _load_worker_module()
    calls: list[str] = []

    def _fake_load(name: str):
        calls.append(name)
        return ("generic_model", "generic_processor")

    import mlx_vlm

    monkeypatch.setattr(mlx_vlm, "load", _fake_load)

    qwen = worker.load_qwen_vlm("mlx-community/Qwen3.5-4B-4bit")
    assert qwen == ("generic_model", "generic_processor")
    assert calls == ["mlx-community/Qwen3.5-4B-4bit"]


def test_chunk_list_splits_fixed_size() -> None:
    worker = _load_worker_module()
    assert worker.chunk_list([1, 2, 3, 4, 5, 6, 7], 5) == [[1, 2, 3, 4, 5], [6, 7]]
