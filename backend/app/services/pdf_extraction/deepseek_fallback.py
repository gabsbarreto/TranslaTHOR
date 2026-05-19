from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeepSeekFallbackResult:
    used: bool
    warnings: list[str]


class DeepSeekFallbackOCR:
    """Optional hook for future DeepSeek OCR-2 repair.

    Marker does not expose DeepSeek OCR-2 as a drop-in OCR backend. This class keeps
    the fallback separated from Marker/Surya so it can be applied only to poor-quality
    pages or blocks without restoring the manual region workflow.
    """

    def repair_selected_blocks(self, *_args, **_kwargs) -> DeepSeekFallbackResult:
        return DeepSeekFallbackResult(
            used=False,
            warnings=[
                "DeepSeek OCR-2 fallback is not wired into Marker. Marker/Surya output was kept unchanged."
            ],
        )
