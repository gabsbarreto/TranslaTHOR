from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.models.schema import Block, BlockType

logger = logging.getLogger(__name__)


@dataclass
class LocalVLMConfig:
    enabled: bool
    base_url: str
    model: str
    api_key: str
    timeout: int
    max_retries: int

    @classmethod
    def from_env(cls) -> "LocalVLMConfig":
        return cls(
            enabled=os.getenv("LOCAL_VLM_ENABLED", "false").lower() in {"1", "true", "yes"},
            base_url=os.getenv("LOCAL_VLM_BASE_URL", "http://localhost:8080/v1").rstrip("/"),
            model=os.getenv("LOCAL_VLM_MODEL", ""),
            api_key=os.getenv("LOCAL_VLM_API_KEY", "not-needed"),
            timeout=int(os.getenv("LOCAL_VLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LOCAL_VLM_MAX_RETRIES", "2")),
        )


class LocalVLMRepairService:
    """Optional, selective repair against an OpenAI-compatible local VLM endpoint."""

    def __init__(self, config: LocalVLMConfig | None = None) -> None:
        self.config = config or LocalVLMConfig.from_env()

    def select_blocks_for_repair(
        self,
        blocks: list[Block],
        extraction_context: dict | None = None,
    ) -> list[Block]:
        extraction_context = extraction_context or {}
        hidden_ocr_context = bool(
            extraction_context.get("suspicious_hidden_ocr")
            or extraction_context.get("marker_fallback_to_normal")
            or extraction_context.get("pdf_classification") == "bad_hidden_ocr"
        )
        detected_language = str(extraction_context.get("detected_language") or "")
        selected: list[Block] = []
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            if block.block_type in {BlockType.TABLE, BlockType.EQUATION}:
                selected.append(block)
                continue
            if self._looks_garbled(text) or self._looks_like_broken_table(text):
                selected.append(block)
                continue
            if hidden_ocr_context and self._looks_like_hidden_ocr_repair_candidate(text, detected_language):
                selected.append(block)
        default_limit = "80" if hidden_ocr_context else "20"
        max_blocks = int(os.getenv("LOCAL_VLM_MAX_REPAIR_BLOCKS", default_limit))
        return selected[:max_blocks]

    def repair_blocks(
        self,
        blocks: list[Block],
        debug_dir: Path | None = None,
        extraction_context: dict | None = None,
    ) -> tuple[int, list[str]]:
        if not self.config.enabled:
            return 0, ["Local VLM repair was requested but LOCAL_VLM_ENABLED is false."]
        if not self.config.model:
            return 0, ["Local VLM repair skipped because LOCAL_VLM_MODEL is not set."]

        repaired = 0
        warnings: list[str] = []
        debug_payload: list[dict] = []
        selected_blocks = self.select_blocks_for_repair(blocks, extraction_context)
        if not selected_blocks:
            return 0, ["Local VLM repair was enabled, but no blocks matched the repair rules."]
        language_hint = str((extraction_context or {}).get("detected_language") or "")
        for block in selected_blocks:
            before = block.text
            try:
                after = self._repair_text(before, block.block_type.value, language_hint)
            except Exception as exc:
                logger.warning("Local VLM repair failed for %s: %s", block.id, exc)
                warnings.append(f"Local VLM repair failed for {block.id}: {exc}")
                continue
            if after.strip() and after.strip() != before.strip():
                block.text = after.strip()
                repaired += 1
            debug_payload.append(
                {
                    "block_id": block.id,
                    "page_number": block.page_number,
                    "block_type": block.block_type.value,
                    "changed": after.strip() != before.strip(),
                    "repair_reason": self._repair_reason(before, block.block_type, extraction_context or {}),
                    "before": before if debug_dir else "",
                    "after": after if debug_dir else "",
                }
            )

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "local_vlm_repair.json").write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        return repaired, warnings

    def _repair_text(self, text: str, block_type: str, language_hint: str = "") -> str:
        language_instruction = ""
        if language_hint.startswith("pt"):
            language_instruction = (
                "The source is Portuguese, likely Brazilian Portuguese. Correct OCR accent and cedilla errors "
                "such as missing or wrong diacritics when context makes the correction clear. "
            )
        prompt = (
            "Repair OCR/layout extraction errors in this PDF block. Preserve meaning, order, markdown, table rows, "
            "math notation, citations, author names, e-mail addresses, and do not translate. "
            f"{language_instruction}"
            "Return only the repaired block text.\n\n"
            f"Block type: {block_type}\n\n{text}"
        )
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"}
        last_error: Exception | None = None
        for attempt in range(max(1, self.config.max_retries + 1)):
            try:
                req = urllib.request.Request(
                    f"{self.config.base_url}/chat/completions",
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"]).strip()
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(str(last_error or "unknown local VLM error"))

    def _looks_garbled(self, text: str) -> bool:
        if len(text) < 40:
            return False
        alnum = sum(1 for ch in text if ch.isalnum())
        weird = sum(1 for ch in text if not ch.isalnum() and not ch.isspace() and ch not in ".,;:!?()[]{}'\"-/+%$#&*@=<>|_`~^\\")
        return (alnum / max(len(text), 1)) < 0.55 or (weird / max(len(text), 1)) > 0.18 or "\ufffd" in text

    def _looks_like_broken_table(self, text: str) -> bool:
        lower = text.lower()
        return ("<table" in lower and "</table>" not in lower) or text.count("|") >= 6 and "---" not in text

    def _looks_like_hidden_ocr_repair_candidate(self, text: str, detected_language: str) -> bool:
        if len(text) < 60:
            return False
        if self._has_line_break_artifacts(text) or self._has_common_ocr_substitutions(text):
            return True
        if detected_language.startswith("pt") or self._looks_like_portuguese(text):
            return self._has_portuguese_ocr_accent_artifacts(text)
        return False

    def _repair_reason(self, text: str, block_type: BlockType, extraction_context: dict) -> str:
        if block_type in {BlockType.TABLE, BlockType.EQUATION}:
            return block_type.value
        detected_language = str(extraction_context.get("detected_language") or "")
        if self._looks_garbled(text):
            return "garbled_text"
        if self._looks_like_broken_table(text):
            return "broken_table_markup"
        if self._has_line_break_artifacts(text):
            return "line_break_artifacts"
        if self._has_common_ocr_substitutions(text):
            return "common_ocr_substitutions"
        if (detected_language.startswith("pt") or self._looks_like_portuguese(text)) and self._has_portuguese_ocr_accent_artifacts(text):
            return "portuguese_hidden_ocr_accent_artifacts"
        return "hidden_ocr"

    def _has_line_break_artifacts(self, text: str) -> bool:
        return bool(re.search(r"\w-\s+\w", text)) or len(re.findall(r"\b\w{1,3}-\s+\w{3,}\b", text)) >= 2

    def _has_common_ocr_substitutions(self, text: str) -> bool:
        return bool(re.search(r"\b[0O]\s+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇa-záàâãéêíóôõúç]", text)) or text.count("¬") > 0

    def _looks_like_portuguese(self, text: str) -> bool:
        lower = text.lower()
        hits = sum(
            1
            for word in (
                " de ",
                " que ",
                " para ",
                " com ",
                " uma ",
                " não ",
                " saúde",
                " gênero",
                " genero",
                " população",
                " populacao",
                " serviço",
                " servico",
            )
            if word in lower
        )
        return hits >= 3

    def _has_portuguese_ocr_accent_artifacts(self, text: str) -> bool:
        lower = text.lower()
        artifact_words = (
            "genero",
            "servico",
            "saude",
            "crianca",
            "adolescencia",
            "experiencia",
            "avaliacao",
            "populacao",
            "atencao",
            "clinica",
            "publico",
            "relacao",
            "questao",
            "hormonizacao",
            "violencia",
            "psicologica",
            "necessario",
            "médica",
            "medica",
        )
        return sum(1 for word in artifact_words if re.search(rf"\b{re.escape(word)}s?\b", lower)) >= 3


def encode_image_for_openai(image_path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
