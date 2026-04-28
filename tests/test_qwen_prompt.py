from app.services.translator_mlx import MlxTranslator, TranslationSettings


class DummyTokenizer:
    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "PROMPT"


def test_qwen35_disables_thinking_via_chat_template_kwargs() -> None:
    translator = MlxTranslator(TranslationSettings(model_name="mlx-community/Qwen3.5-9B-8bit"))
    tokenizer = DummyTokenizer()
    translator._tokenizer = tokenizer

    prompt = translator._format_chat_prompt("SYSTEM", "USER")

    assert prompt == "PROMPT"
    assert tokenizer.messages == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    assert tokenizer.kwargs == {"enable_thinking": False}


def test_qwen35_uses_translation_system_prompt() -> None:
    translator = MlxTranslator(TranslationSettings(model_name="mlx-community/Qwen3.5-9B-8bit"))
    prompt = translator._system_prompt()
    assert "PDF reconstruction" in prompt
    assert "TEXT may contain plain text, Markdown, or HTML" in prompt
    assert "Preserve existing Markdown syntax, HTML tags" in prompt
    assert "Do not add wrapper text" in prompt
