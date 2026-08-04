from __future__ import annotations

from types import SimpleNamespace

from app.models.schema import (
    Block,
    BlockType,
    DocumentMetadata,
    DocumentModel,
    SourceType,
    TranslationChunk,
)
from app.services.translator_mlx import (
    MlxTranslator,
    TranslationSettings,
    _BatchTranslationRequest,
    _PreparedBatchPrompt,
)
from app.services import translation_subprocess


class _Tokenizer:
    def __init__(self) -> None:
        self.template_calls = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ) -> str:
        _ = (tokenize, add_generation_prompt, kwargs)
        self.template_calls += 1
        return (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[1]['content']}</user><assistant>"
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        _ = add_special_tokens
        return list(text.encode("utf-8"))


def _document_with_chunks(count: int) -> tuple[DocumentModel, list[TranslationChunk]]:
    blocks = [
        Block(
            id=f"block-{index}",
            page_number=1,
            block_type=BlockType.PARAGRAPH,
            text=f"Texto fuente {index} con contenido clínico suficiente.",
            reading_order_index=index,
            source_type=SourceType.OCR,
        )
        for index in range(1, count + 1)
    ]
    chunks = [
        TranslationChunk(
            id=f"chunk-{index}",
            block_ids=[block.id],
            source_text=block.text,
            source_language="es",
            chunk_type="paragraph",
        )
        for index, block in enumerate(blocks, start=1)
    ]
    return (
        DocumentModel(
            metadata=DocumentMetadata(
                filename="batch.pdf",
                page_count=1,
                detected_language="es",
            ),
            pages=[],
            blocks=blocks,
        ),
        chunks,
    )


def test_document_translation_batches_first_pass_and_failed_retries_in_order() -> None:
    document, chunks = _document_with_chunks(5)
    translator = MlxTranslator(
        TranslationSettings(
            batch_size=4,
            batch_token_budget=100_000,
            max_tokens=128,
        )
    )
    translator._model = object()
    translator._tokenizer = _Tokenizer()
    translator._ensure_loaded = lambda: True  # type: ignore[method-assign]
    translator.build_chunks = lambda _document: chunks  # type: ignore[method-assign]
    translator._is_already_english = lambda _chunk: False  # type: ignore[method-assign]
    translator._is_acceptable_chunk_translation = (  # type: ignore[method-assign]
        lambda _source, target, *_args, **_kwargs: target.startswith("EN:")
    )
    translator._translation_acceptance_issue = (  # type: ignore[method-assign]
        lambda _source, target, *_args, **_kwargs: (
            None if target.startswith("EN:") else "translation_output_not_english"
        )
    )

    calls: list[tuple[str, list[str], list[str]]] = []

    def fake_batch(batch, *, phase: str) -> list[str]:
        sources = [prompt.request.text for prompt in batch]
        contexts = [prompt.request.context for prompt in batch]
        calls.append((phase, sources, contexts))
        if phase == "first_pass":
            return [
                source
                if source.startswith(("Texto fuente 2", "Texto fuente 4"))
                else f"EN:{source}"
                for source in sources
            ]
        return [f"EN:{source}" for source in sources]

    translator._generate_prepared_batch = fake_batch  # type: ignore[method-assign]
    started: list[int] = []
    completed: list[int] = []

    translated, _ = translator.translate_document(
        document,
        "",
        on_chunk_started=lambda index, _total: started.append(index),
        on_chunk_translated=lambda index, _total, _preview: completed.append(index),
    )

    assert [(phase, len(sources)) for phase, sources, _contexts in calls] == [
        ("first_pass", 4),
        ("retry", 2),
        ("first_pass", 1),
    ]
    assert all(
        "The previous output was not an acceptable English translation" in context
        for context in calls[1][2]
    )
    assert started == [1, 2, 3, 4, 5]
    assert completed == [1, 2, 3, 4, 5]
    assert [block.text for block in translated.blocks] == [
        f"EN:Texto fuente {index} con contenido clínico suficiente." for index in range(1, 6)
    ]
    assert translated.metadata.translation["mlx_runtime"]["generation"] == {
        "batch_calls": 0,
        "first_pass_requests": 0,
        "retry_requests": 0,
        "sequential_fallback_requests": 0,
        "batch_failures": 0,
        "batch_preparation_failures": 0,
    }


def test_adaptive_batching_obeys_size_and_combined_token_budget() -> None:
    translator = MlxTranslator(
        TranslationSettings(batch_size=4, batch_token_budget=1_000, max_tokens=128)
    )
    request = _BatchTranslationRequest(
        text="Texto",
        context="",
        source_language="es",
        block_type=BlockType.PARAGRAPH,
    )
    prompts = [
        _PreparedBatchPrompt(
            request=request,
            prompt=str(index),
            prompt_tokens=tuple(range(300)),
            max_tokens=128,
        )
        for index in range(5)
    ]

    batches = translator._adaptive_prompt_batches(prompts)

    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_instruction_template_and_tokens_are_cached() -> None:
    translator = MlxTranslator(TranslationSettings())
    tokenizer = _Tokenizer()
    translator._tokenizer = tokenizer

    first = translator._build_prompt("Primer texto", "", "es")
    second = translator._build_prompt("Segundo texto", "", "es")
    encoded = translator._encode_prompts([first, second])

    assert tokenizer.template_calls == 1
    assert encoded == [tokenizer.encode(first), tokenizer.encode(second)]
    metadata = translator.runtime_metadata()["instruction_cache"]
    assert metadata["mode"] == "tokenized_prefix"
    assert metadata["hits"] == 2
    assert metadata["misses"] == 1


def test_cpu_thread_configuration_uses_explicit_value(monkeypatch) -> None:
    variables = (
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    )
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    translator = MlxTranslator(TranslationSettings(cpu_threads=3))

    translator._configure_cpu_threads()

    for variable in variables[:-1]:
        assert __import__("os").environ[variable] == "3"
    assert __import__("os").environ["TOKENIZERS_PARALLELISM"] == "true"


def test_auto_cpu_thread_configuration_reserves_capacity(monkeypatch) -> None:
    translator = MlxTranslator(TranslationSettings(cpu_threads=0))
    monkeypatch.setattr("os.cpu_count", lambda: 12)
    monkeypatch.setattr("os.uname", lambda: SimpleNamespace(sysname="Darwin"))
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="8\n"),
    )

    assert translator._recommended_cpu_threads() == 6


def test_translation_worker_prefers_configured_virtualenv(monkeypatch, tmp_path) -> None:
    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(translation_subprocess, "DEFAULT_TRANSLATION_PYTHON", str(executable))

    assert translation_subprocess._translation_python_executable() == str(executable)


def test_translation_worker_falls_back_to_current_interpreter(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "missing-python"
    monkeypatch.setattr(translation_subprocess, "DEFAULT_TRANSLATION_PYTHON", str(missing))

    assert (
        translation_subprocess._translation_python_executable()
        == translation_subprocess.sys.executable
    )
