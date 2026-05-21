"""Runtime compatibility shims for subprocess tools.

Marker 1.10.2 / Surya 0.17.1 imports ``transformers.onnx.OnnxConfig``.
The app's MLX translation stack currently requires Transformers 5.x, where
that module is no longer shipped. This shim is loaded through PYTHONPATH by
the app and Marker subprocesses, and supplies the small base class Surya uses
for its OCR error config definitions.
"""

from __future__ import annotations

import sys
import types


try:
    __import__("transformers.onnx")
except Exception:
    onnx_module = types.ModuleType("transformers.onnx")

    class OnnxConfig:
        def __init__(self, config=None, task: str = "default", **kwargs) -> None:
            self._config = config
            self.task = task
            self._kwargs = kwargs

        @property
        def inputs(self):  # pragma: no cover - subclasses override this
            return {}

    onnx_module.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = onnx_module
