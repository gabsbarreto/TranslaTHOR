from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_worker_module():
    root = Path(__file__).resolve().parents[1]
    worker_path = root / "scripts" / "extract_metadata_worker.py"
    spec = importlib.util.spec_from_file_location("extract_metadata_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_worker_extracts_json_object_from_model_output() -> None:
    worker = _load_worker_module()

    parsed = worker.extract_json_object(
        """```json
{"title": "Paper", "authors": ["Jane Smith"], "doi": "10.1234/example"}
```"""
    )
    metadata = worker.coerce_metadata(parsed)

    assert metadata["title"] == "Paper"
    assert metadata["authors"] == ["Jane Smith"]
    assert metadata["doi"] == "10.1234/example"
    assert metadata["journal"] == ""


def test_metadata_worker_prompt_requests_exact_schema() -> None:
    worker = _load_worker_module()

    messages = worker.build_prompt("Paper OCR text")

    assert messages[0]["role"] == "system"
    assert "Return only one valid JSON object" in messages[0]["content"]
    assert '"copyright_or_licence": ""' in messages[1]["content"]
    assert "Paper OCR text" in messages[1]["content"]
