from __future__ import annotations

from app.services.deepseek_ocr_pipeline import DeepSeekOcrPipeline


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class _FakeStderr:
    def read(self) -> str:
        return ""


class _FakeProcess:
    stdout = _FakeStdout(['{"event": "model_loaded"}\n'])
    stderr = _FakeStderr()

    def __init__(self) -> None:
        self.terminated = False

    def wait(self) -> int:
        return 0

    def terminate(self) -> None:
        self.terminated = True


def test_communicate_with_progress_does_not_require_cancel_callback() -> None:
    process = _FakeProcess()
    events: list[str] = []

    stdout, stderr = DeepSeekOcrPipeline()._communicate_with_cancel(
        process, cancel_requested=None, on_stdout_line=events.append
    )

    assert '"model_loaded"' in stdout
    assert stderr == ""
    assert process.terminated is False
    assert events == ['{"event": "model_loaded"}\n']
