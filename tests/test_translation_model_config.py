from app.config import DEFAULT_TRANSLATION_MODEL
from app.services.translator_mlx import TranslationSettings


def test_translation_default_model_is_qwen35() -> None:
    settings = TranslationSettings()
    assert settings.model_name == DEFAULT_TRANSLATION_MODEL
