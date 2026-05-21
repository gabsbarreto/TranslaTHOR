from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

if "langdetect" not in sys.modules:
    langdetect_stub = ModuleType("langdetect")
    langdetect_stub.detect = lambda _text: "en"  # type: ignore[attr-defined]
    sys.modules["langdetect"] = langdetect_stub

if "pypdfium2" not in sys.modules:
    pypdfium_stub = ModuleType("pypdfium2")

    class _PdfDocument:  # pragma: no cover - import guard only
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("pypdfium2 stub should not be used in this test")

    pypdfium_stub.PdfDocument = _PdfDocument  # type: ignore[attr-defined]
    sys.modules["pypdfium2"] = pypdfium_stub
