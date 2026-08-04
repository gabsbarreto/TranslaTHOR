from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import (
    DEFAULT_LLM_MIN_P,
    DEFAULT_LLM_PRESENCE_PENALTY,
    DEFAULT_LLM_REPETITION_PENALTY,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_TOP_P,
    DEFAULT_MLX_CPU_THREADS,
    DEFAULT_TRANSLATION_BATCH_SIZE,
    DEFAULT_TRANSLATION_BATCH_TOKEN_BUDGET,
    DEFAULT_TRANSLATION_MODEL,
)
from app.models.schema import DocumentModel
from app.services.markdown_builder import MarkdownBuilder
from app.services.translator_mlx import MlxTranslator, TranslationSettings


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate a structured document with MLX in an isolated process."
    )
    parser.add_argument("--document", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--output-document", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--settings-json", required=True)
    args = parser.parse_args()

    settings = json.loads(args.settings_json)
    model_cfg = (
        settings.get("translation_model", {})
        if isinstance(settings.get("translation_model"), dict)
        else {}
    )
    model_name = str(model_cfg.get("model_id", settings.get("model", DEFAULT_TRANSLATION_MODEL)))
    temperature = float(
        model_cfg.get("temperature", settings.get("temperature", DEFAULT_LLM_TEMPERATURE))
    )
    top_p = float(model_cfg.get("top_p", settings.get("top_p", DEFAULT_LLM_TOP_P)))
    top_k = int(model_cfg.get("top_k", settings.get("top_k", DEFAULT_LLM_TOP_K)))
    min_p = float(model_cfg.get("min_p", settings.get("min_p", DEFAULT_LLM_MIN_P)))
    presence_penalty = float(
        model_cfg.get(
            "presence_penalty",
            settings.get("presence_penalty", DEFAULT_LLM_PRESENCE_PENALTY),
        )
    )
    repetition_penalty = float(
        model_cfg.get(
            "repetition_penalty",
            settings.get("repetition_penalty", DEFAULT_LLM_REPETITION_PENALTY),
        )
    )
    max_tokens = int(model_cfg.get("max_tokens", settings.get("max_tokens", 2048)))
    batch_size = int(
        model_cfg.get(
            "batch_size",
            settings.get("translation_batch_size", DEFAULT_TRANSLATION_BATCH_SIZE),
        )
    )
    batch_token_budget = int(
        model_cfg.get(
            "batch_token_budget",
            settings.get(
                "translation_batch_token_budget",
                DEFAULT_TRANSLATION_BATCH_TOKEN_BUDGET,
            ),
        )
    )
    cpu_threads = int(
        model_cfg.get(
            "cpu_threads",
            settings.get("mlx_cpu_threads", DEFAULT_MLX_CPU_THREADS),
        )
    )
    document_path = Path(args.document)
    markdown_path = Path(args.markdown)
    output_document_path = Path(args.output_document)
    output_markdown_path = Path(args.output_markdown)

    document = DocumentModel.model_validate_json(document_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8", errors="ignore")

    translator = MlxTranslator(
        TranslationSettings(
            model_name=model_name,
            chunk_size=int(settings.get("chunk_size", 1800)),
            chunk_group_size=int(settings.get("translation_chunk_group_size", 5)),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            batch_size=batch_size,
            batch_token_budget=batch_token_budget,
            cpu_threads=cpu_threads,
        )
    )
    if not translator._ensure_loaded():
        details = translator.last_load_error() or "unknown model loading failure"
        raise RuntimeError(
            f"Failed to initialize translation model '{model_name}': {details}. "
            "Install a compatible MLX stack (for Qwen3.5 use mlx-lm>=0.31.0)."
        )

    try:
        translated_doc, _ = translator.translate_document(
            document,
            markdown,
            on_chunk_started=lambda index, total: emit(
                {"event": "chunk_started", "index": index, "total": total}
            ),
            on_chunk_translated=lambda index, total, preview: emit(
                {"event": "chunk_translated", "index": index, "total": total, "preview": preview}
            ),
            on_table_progress=lambda index, total, label: emit(
                {"event": "table_progress", "index": index, "total": total, "label": label}
            ),
        )
        translated_doc.metadata.translation = {
            **translated_doc.metadata.translation,
            "model": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "mlx_runtime": translator.runtime_metadata(),
        }
        translated_md = MarkdownBuilder().build(translated_doc)
        output_document_path.write_text(translated_doc.model_dump_json(indent=2), encoding="utf-8")
        output_markdown_path.write_text(translated_md, encoding="utf-8")
    finally:
        translator.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
