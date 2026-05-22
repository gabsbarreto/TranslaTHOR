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


def test_qwen_worker_uses_ocr_prompt_as_system_message() -> None:
    worker = _load_worker_module()
    captured = {}

    def fake_apply_chat_template(_processor, _config, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "FORMATTED"

    formatted = worker.build_generation_prompt(
        processor=object(),
        config={"model_type": "qwen3_5"},
        system_prompt="<image>\n<|grounding|>Convert the document to markdown.",
        apply_chat_template=fake_apply_chat_template,
        enable_thinking=False,
    )

    assert formatted == "FORMATTED"
    assert captured["prompt"] == [
        {"role": "system", "content": "Convert the document to markdown."},
        {"role": "user", "content": ""},
    ]
    assert captured["kwargs"]["num_images"] == 1


def test_qwen_worker_extracts_last_body_paragraphs_for_previous_page_context() -> None:
    worker = _load_worker_module()

    context = worker.extract_previous_page_context(
        """# Results

31

The first body paragraph explains the study design and includes enough words to be useful as continuity context.

Figure 2. Flow diagram of included studies.

The final paragraph continues the argument across the page boundary and ends without a full stop
""",
        n_paragraphs=2,
        max_chars=500,
    )

    assert "# Results" not in context
    assert "31" not in context
    assert "Figure 2" not in context
    assert "The first body paragraph explains" in context
    assert context.endswith("without a full stop")


def test_qwen_worker_ignores_headers_footers_short_lines_and_references() -> None:
    worker = _load_worker_module()

    context = worker.extract_previous_page_context(
        """<!-- page-header: Journal of Testing -->

Short footer

<!-- page-number: 14 -->

This paragraph is a valid body paragraph with enough semantic content to help classify the next page opening.

## References

Smith, A. (2020). Example paper. https://example.com
""",
        n_paragraphs=2,
        max_chars=500,
    )

    assert context == (
        "This paragraph is a valid body paragraph with enough semantic content to help classify the next page opening."
    )


def test_qwen_worker_limits_previous_page_context_chars() -> None:
    worker = _load_worker_module()

    context = worker.extract_previous_page_context(
        "This is a long body paragraph with enough words to be accepted as semantic continuity context. " * 8,
        n_paragraphs=1,
        max_chars=120,
    )

    assert len(context) <= 120
    assert context.startswith("...")


def test_qwen_worker_builds_contextual_ocr_prompt() -> None:
    worker = _load_worker_module()

    prompt = worker.build_ocr_prompt("Base OCR prompt.", "Previous paragraph continues here.")

    assert "Base OCR prompt." in prompt
    assert "Previous-page context for continuity only" in prompt
    assert "Previous paragraph continues here." in prompt
    assert "Never copy it into the output" in prompt
    assert "running page header" in prompt


def test_qwen_no_repeat_ngram_processor_blocks_recent_repeat_token() -> None:
    worker = _load_worker_module()
    processor = worker.NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=16)

    input_ids = [10, 11, 12, 10, 11]
    scores = [[0.0] * 64]
    updated = processor(input_ids, scores)

    assert updated[0][12] == float("-inf")
