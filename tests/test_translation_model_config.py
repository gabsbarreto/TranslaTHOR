from app.config import (
    DEFAULT_LLM_MIN_P,
    DEFAULT_LLM_PRESENCE_PENALTY,
    DEFAULT_LLM_REPETITION_PENALTY,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_TOP_P,
    DEFAULT_TRANSLATION_MODEL,
)
from app.services.translator_mlx import TranslationSettings


def test_translation_default_model_is_qwen35() -> None:
    settings = TranslationSettings()
    assert settings.model_name == DEFAULT_TRANSLATION_MODEL
    assert settings.temperature == DEFAULT_LLM_TEMPERATURE
    assert settings.top_p == DEFAULT_LLM_TOP_P
    assert settings.top_k == DEFAULT_LLM_TOP_K
    assert settings.min_p == DEFAULT_LLM_MIN_P
    assert settings.presence_penalty == DEFAULT_LLM_PRESENCE_PENALTY
    assert settings.repetition_penalty == DEFAULT_LLM_REPETITION_PENALTY
