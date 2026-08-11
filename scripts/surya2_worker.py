from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


class Surya2Worker:
    def __init__(self) -> None:
        # Surya settings are constructed at import time, so configure the
        # required backend before importing any Surya module.
        os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
        os.environ["SURYA_GUIDED_LAYOUT"] = "false"
        from surya.inference import SuryaInferenceManager  # type: ignore[import-untyped]
        from surya.layout import LayoutPredictor  # type: ignore[import-untyped]
        from surya.recognition import RecognitionPredictor  # type: ignore[import-untyped]
        from surya.settings import settings  # type: ignore[import-untyped]

        self.manager = SuryaInferenceManager(method="llamacpp")
        self.layout_predictor = LayoutPredictor(self.manager)
        self.recognition_predictor = RecognitionPredictor(self.manager)
        parallel_pages = self.manager.capacity()
        context_per_slot = int(settings.SURYA_INFERENCE_CTX_PER_SLOT)
        total_context = settings.SURYA_INFERENCE_CTX_SIZE
        if total_context is None:
            total_context = max(16384, parallel_pages * context_per_slot)
        self.batching = {
            "parallel_pages": parallel_pages,
            "context_per_slot": context_per_slot,
            "total_context": int(total_context),
        }

    def run_request(self, request: dict[str, Any]) -> None:
        request_id = str(request["request_id"])
        image_paths = [Path(value) for value in request.get("image_paths", [])]
        output_path = Path(str(request["output_path"]))
        strategy = str(request.get("strategy", "full_page"))
        if strategy not in {"full_page", "layout_then_block"}:
            raise ValueError(f"Unsupported Surya 2 recognition strategy: {strategy}")
        if not image_paths:
            raise ValueError("Surya 2 request did not include any page images.")
        missing = [str(path) for path in image_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Surya 2 page image is missing: {missing[0]}")

        emit(
            {
                "event": "request_started",
                "request_id": request_id,
                "strategy": strategy,
                "pages": len(image_paths),
                "effective_parallel_pages": min(
                    len(image_paths),
                    int(self.batching["parallel_pages"]),
                ),
            }
        )
        images: list[Image.Image] = []
        started = time.perf_counter()
        try:
            for path in image_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            loaded_at = time.perf_counter()
            emit(
                {
                    "event": "images_loaded",
                    "request_id": request_id,
                    "pages": len(images),
                    "seconds": round(loaded_at - started, 6),
                }
            )

            layout_seconds = 0.0
            if strategy == "layout_then_block":
                emit(
                    {
                        "event": "layout_started",
                        "request_id": request_id,
                        "pages": len(images),
                    }
                )
                layout_started = time.perf_counter()
                layouts = self.layout_predictor(images)
                layout_seconds = time.perf_counter() - layout_started
                layout_error_count = sum(1 for layout in layouts if layout.error)
                emit(
                    {
                        "event": "layout_complete",
                        "request_id": request_id,
                        "pages": len(layouts),
                        "blocks": sum(len(layout.bboxes) for layout in layouts),
                        "errors": layout_error_count,
                        "seconds": round(layout_seconds, 6),
                    }
                )
                if layout_error_count:
                    raise RuntimeError(f"Surya 2 layout failed for {layout_error_count} page(s).")
                predictions = self.recognition_predictor(
                    images,
                    layouts,
                    full_page=False,
                )
            else:
                predictions = self.recognition_predictor(images, full_page=True)

            inference_completed = time.perf_counter()
            pages = []
            for page_number, prediction in enumerate(predictions, start=1):
                page_payload = prediction.model_dump(mode="json")
                page_payload["page_number"] = page_number
                pages.append(page_payload)
                emit(
                    {
                        "event": "page_done",
                        "request_id": request_id,
                        "page_number": page_number,
                        "total": len(predictions),
                        "blocks": len(prediction.blocks),
                        "errors": sum(1 for block in prediction.blocks if block.error),
                        "skipped": sum(1 for block in prediction.blocks if block.skipped),
                    }
                )

            payload = {
                "schema_version": 1,
                "engine": "surya2_llamacpp",
                "surya_version": importlib.metadata.version("surya-ocr"),
                "strategy": strategy,
                "page_count": len(pages),
                "batching": {
                    **self.batching,
                    "requested_pages": len(pages),
                    "effective_parallel_pages": min(
                        len(pages),
                        int(self.batching["parallel_pages"]),
                    ),
                },
                "timing": {
                    "image_load_seconds": round(loaded_at - started, 6),
                    "layout_seconds": round(layout_seconds, 6),
                    "inference_seconds": round(inference_completed - loaded_at, 6),
                    "total_worker_seconds": round(inference_completed - started, 6),
                },
                "pages": pages,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            emit(
                {
                    "event": "request_complete",
                    "request_id": request_id,
                    "output_path": str(output_path),
                    "pages": len(pages),
                    "seconds": round(time.perf_counter() - started, 6),
                }
            )
        finally:
            for image in images:
                image.close()

    def close(self) -> None:
        self.manager.stop()


def serve() -> int:
    emit({"event": "worker_starting", "pid": os.getpid(), "backend": "llamacpp"})
    worker = Surya2Worker()
    emit(
        {
            "event": "worker_ready",
            "pid": os.getpid(),
            "backend": "llamacpp",
            "surya_version": importlib.metadata.version("surya-ocr"),
            "batching": worker.batching,
        }
    )
    try:
        for line in iter(input, ""):
            if not line.strip():
                continue
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                action = str(request.get("action", "run"))
                if action == "shutdown":
                    emit({"event": "worker_stopping", "pid": os.getpid()})
                    return 0
                if action != "run":
                    raise ValueError(f"Unsupported worker action: {action}")
                worker.run_request(request)
            except Exception as exc:
                emit(
                    {
                        "event": "request_error",
                        "request_id": str(request.get("request_id", "")),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        worker.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Persistent Surya 2 worker backed by one llama.cpp inference manager."
    )
    parser.add_argument("--serve", action="store_true", help="Read JSON requests from stdin.")
    args = parser.parse_args()
    if not args.serve:
        parser.error("--serve is required")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
