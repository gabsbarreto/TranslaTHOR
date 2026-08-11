from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import (  # noqa: E402
    DEFAULT_LLM_MIN_P,
    DEFAULT_LLM_PRESENCE_PENALTY,
    DEFAULT_LLM_REPETITION_PENALTY,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_TOP_P,
    DEFAULT_QWEN_OCR_BASE_SIZE,
    DEFAULT_QWEN_OCR_CROP_MODE,
    DEFAULT_QWEN_OCR_IMAGE_SIZE,
    DEFAULT_QWEN_OCR_MAX_CROPS,
    DEFAULT_QWEN_OCR_MAX_TOKENS,
    DEFAULT_QWEN_OCR_MIN_CROPS,
    DEFAULT_QWEN_OCR_MIN_P,
    DEFAULT_QWEN_OCR_MODEL,
    DEFAULT_QWEN_OCR_NGRAM_SIZE,
    DEFAULT_QWEN_OCR_NGRAM_WINDOW,
    DEFAULT_QWEN_OCR_PRESENCE_PENALTY,
    DEFAULT_QWEN_OCR_PROMPT,
    DEFAULT_QWEN_OCR_REPETITION_PENALTY,
    DEFAULT_QWEN_OCR_SKIP_REPEAT,
    DEFAULT_QWEN_OCR_TEMPERATURE,
    DEFAULT_QWEN_OCR_TOP_K,
    DEFAULT_QWEN_OCR_TOP_P,
    DEFAULT_SURYA2_CONTEXT_PER_SLOT,
    DEFAULT_SURYA2_PARALLEL_PAGES,
    DEFAULT_TRANSLATION_MODEL,
)
from app.models.schema import DocumentModel  # noqa: E402
from app.services.pdf_extraction.marker_extractor import PDFExtractor  # noqa: E402
from app.services.pdf_extraction.models import PDFExtractionResult  # noqa: E402
from app.services.pdf_extraction.qwen_ocr_fallback import (  # noqa: E402
    QwenFullPageOCRFallback,
)
from app.services.pdf_extraction.surya2_extractor import (  # noqa: E402
    Surya2LlamaCppExtractor,
)
from app.services.pdf_extraction.pdf_type_detector import PDFTypeDetector  # noqa: E402
from app.services.reconstructor import Reconstructor  # noqa: E402
from app.services.translation_subprocess import run_translation_subprocess  # noqa: E402

try:
    import psutil  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - benchmark can still run without memory sampling
    psutil = None

from pypdf import PdfReader, PdfWriter  # noqa: E402


SUPPORTED_ENGINES = (
    "marker_balanced",
    "marker_surya",
    "surya_qwen_mlx",
    "surya2_full_page",
    "surya2_layout_then_block",
)


@dataclass
class RunMetrics:
    document_id: str
    engine: str
    run_type: str
    run_index: int
    page_count: int
    dpi: int
    extraction_seconds: float
    downstream_seconds: float
    wall_seconds: float
    seconds_per_page: float
    pages_per_second: float
    peak_rss_bytes: int | None
    success: bool
    error: str | None
    retries: int
    block_errors: int
    block_skips: int
    text_characters: int
    cer: float | None
    wer: float | None
    deletions: int | None
    insertions: int | None
    layout_label_accuracy: float | None
    reading_order_accuracy: float | None
    headers: int
    footers: int
    tables: int
    equations: int
    figures: int
    captions: int
    reconstructed_pdf: str | None
    translated_pdf: str | None
    markdown_preview: str
    metadata: dict[str, Any]


class ProcessTreeMemorySampler:
    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._pids: set[int] = {os.getpid()}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0
        self._next_footprint_sample = 0.0

    def track(self, process: subprocess.Popen) -> None:
        self._pids.add(process.pid)

    def start(self) -> None:
        if psutil is None:
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        if psutil is None:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_bytes

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            total = 0
            seen: set[int] = set()
            for pid in list(self._pids):
                try:
                    process = psutil.Process(pid)
                    processes = [process, *process.children(recursive=True)]
                except Exception:
                    continue
                for item in processes:
                    if item.pid in seen:
                        continue
                    seen.add(item.pid)
                    try:
                        total += int(item.memory_info().rss)
                    except Exception:
                        continue
            self.peak_bytes = max(self.peak_bytes, total)
            now = time.monotonic()
            if sys.platform == "darwin" and now >= self._next_footprint_sample:
                footprint_total = sum(darwin_process_footprint(pid) for pid in seen)
                self.peak_bytes = max(self.peak_bytes, footprint_total)
                self._next_footprint_sample = now + 1.0


def parse_darwin_footprint(output: str) -> int:
    match = re.search(
        r"Footprint:\s*([0-9.]+)\s*(B|KB|MB|GB)",
        output,
        flags=re.IGNORECASE,
    )
    if match is None:
        return 0
    multipliers = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
    }
    return int(float(match.group(1)) * multipliers[match.group(2).lower()])


def darwin_process_footprint(pid: int) -> int:
    try:
        result = subprocess.run(
            ["/usr/bin/footprint", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return parse_darwin_footprint(result.stdout)
    except Exception:
        return 0


def levenshtein_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    normalized_reference = normalize_for_ocr_metrics(reference)
    normalized_hypothesis = normalize_for_ocr_metrics(hypothesis)
    return levenshtein_distance(normalized_reference, normalized_hypothesis) / max(
        len(normalized_reference), 1
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = normalize_for_ocr_metrics(reference).split()
    hypothesis_words = normalize_for_ocr_metrics(hypothesis).split()
    return levenshtein_distance(reference_words, hypothesis_words) / max(len(reference_words), 1)


def insertion_deletion_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    reference_words = normalize_for_ocr_metrics(reference).split()
    hypothesis_words = normalize_for_ocr_metrics(hypothesis).split()
    rows = len(reference_words) + 1
    cols = len(hypothesis_words) + 1
    scores = [[(0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for row in range(1, rows):
        scores[row][0] = (row, row, 0)
    for col in range(1, cols):
        scores[0][col] = (col, 0, col)
    for row in range(1, rows):
        for col in range(1, cols):
            if reference_words[row - 1] == hypothesis_words[col - 1]:
                scores[row][col] = scores[row - 1][col - 1]
                continue
            deletion = scores[row - 1][col]
            insertion = scores[row][col - 1]
            substitution = scores[row - 1][col - 1]
            candidates = [
                (deletion[0] + 1, deletion[1] + 1, deletion[2]),
                (insertion[0] + 1, insertion[1], insertion[2] + 1),
                (substitution[0] + 1, substitution[1], substitution[2]),
            ]
            scores[row][col] = min(candidates)
    _distance, deletions, insertions = scores[-1][-1]
    return deletions, insertions


def sequence_accuracy(reference: list[str], hypothesis: list[str]) -> float | None:
    if not reference:
        return None
    normalized_reference = [str(item).strip().lower() for item in reference]
    normalized_hypothesis = [str(item).strip().lower() for item in hypothesis]
    distance = levenshtein_distance(normalized_reference, normalized_hypothesis)
    return max(0.0, 1.0 - (distance / max(len(normalized_reference), 1)))


def normalize_for_ocr_metrics(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def reading_order_accuracy(expected_snippets: list[str], text: str) -> float | None:
    if len(expected_snippets) < 2:
        return None
    normalized_text = normalize_for_ocr_metrics(text)
    positions = [
        normalized_text.find(normalize_for_ocr_metrics(snippet)) for snippet in expected_snippets
    ]
    pairs = 0
    correct = 0
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            pairs += 1
            if positions[first] >= 0 and positions[second] > positions[first]:
                correct += 1
    return correct / max(pairs, 1)


class BenchmarkEngine:
    def __init__(self, engine: str, dpi: int, timeout: int) -> None:
        self.engine = engine
        self.dpi = dpi
        self.timeout = timeout
        self.surya2 = Surya2LlamaCppExtractor() if engine.startswith("surya2_") else None

    def close(self) -> None:
        if self.surya2 is not None:
            self.surya2.close()

    def extract(
        self,
        *,
        pdf_path: Path,
        job_dir: Path,
        classification: str,
        detection_metadata: dict,
        warnings: list[str],
        sampler: ProcessTreeMemorySampler,
    ) -> PDFExtractionResult:
        job_dir.mkdir(parents=True, exist_ok=True)
        if self.engine in {"marker_balanced", "marker_surya"}:
            return PDFExtractor().extract(
                pdf_path=pdf_path,
                mode="digital" if self.engine == "marker_balanced" else "scanned",
                keep_debug_artifacts=self.engine != "marker_balanced",
                job_dir=job_dir,
                timeout=self.timeout,
                marker_config={
                    "lowres_image_dpi": self.dpi,
                    "highres_image_dpi": self.dpi,
                    "pdftext_workers": 1,
                },
                on_process_started=sampler.track,
            )
        if self.engine == "surya_qwen_mlx":
            return QwenFullPageOCRFallback().extract(
                pdf_path=pdf_path,
                job_dir=job_dir,
                pdf_classification="scanned_no_text",
                marker_warnings=warnings,
                marker_metadata={
                    "marker_skipped": True,
                    "detection": detection_metadata,
                },
                settings=self._qwen_settings(),
                on_process_started=sampler.track,
            )
        if self.surya2 is None:
            raise ValueError(f"Unsupported engine: {self.engine}")
        strategy = "layout_then_block" if self.engine == "surya2_layout_then_block" else "full_page"
        return self.surya2.extract(
            pdf_path=pdf_path,
            job_dir=job_dir,
            pdf_classification=classification,
            detection_metadata=detection_metadata,
            warnings=warnings,
            settings={"surya2_dpi": self.dpi, "surya2_strategy": strategy},
            on_process_started=sampler.track,
        )

    def _qwen_settings(self) -> dict[str, Any]:
        return {
            "qwen_ocr_model": DEFAULT_QWEN_OCR_MODEL,
            "qwen_ocr_max_tokens": DEFAULT_QWEN_OCR_MAX_TOKENS,
            "qwen_ocr_temperature": DEFAULT_QWEN_OCR_TEMPERATURE,
            "qwen_ocr_top_p": DEFAULT_QWEN_OCR_TOP_P,
            "qwen_ocr_top_k": DEFAULT_QWEN_OCR_TOP_K,
            "qwen_ocr_min_p": DEFAULT_QWEN_OCR_MIN_P,
            "qwen_ocr_presence_penalty": DEFAULT_QWEN_OCR_PRESENCE_PENALTY,
            "qwen_ocr_repetition_penalty": DEFAULT_QWEN_OCR_REPETITION_PENALTY,
            "qwen_ocr_prompt": DEFAULT_QWEN_OCR_PROMPT,
            "qwen_ocr_dpi": self.dpi,
            "qwen_ocr_batch_size": 1,
            "qwen_ocr_crop_mode": DEFAULT_QWEN_OCR_CROP_MODE,
            "qwen_ocr_min_crops": DEFAULT_QWEN_OCR_MIN_CROPS,
            "qwen_ocr_max_crops": DEFAULT_QWEN_OCR_MAX_CROPS,
            "qwen_ocr_base_size": DEFAULT_QWEN_OCR_BASE_SIZE,
            "qwen_ocr_image_size": DEFAULT_QWEN_OCR_IMAGE_SIZE,
            "qwen_ocr_skip_repeat": DEFAULT_QWEN_OCR_SKIP_REPEAT,
            "qwen_ocr_ngram_size": DEFAULT_QWEN_OCR_NGRAM_SIZE,
            "qwen_ocr_ngram_window": DEFAULT_QWEN_OCR_NGRAM_WINDOW,
            "surya_layout_batch_size": 1,
        }


def subset_pdf(source: Path, page_numbers: list[int], output: Path) -> None:
    reader = PdfReader(str(source))
    writer = PdfWriter()
    for page_number in page_numbers:
        if page_number < 1 or page_number > len(reader.pages):
            raise ValueError(
                f"Page {page_number} is outside {source.name} (1-{len(reader.pages)})."
            )
        writer.add_page(reader.pages[page_number - 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)


def resolve_reference_text(
    document_spec: dict[str, Any],
    *,
    manifest_dir: Path,
    selected_pages: list[int],
) -> str:
    reference_pdf = document_spec.get("reference_pdf")
    if reference_pdf:
        reader = PdfReader(str((manifest_dir / str(reference_pdf)).resolve()))
        return "\n".join(
            reader.pages[page_number - 1].extract_text() or "" for page_number in selected_pages
        )
    references = document_spec.get("reference_text") or {}
    if isinstance(references, str):
        path = (manifest_dir / references).resolve()
        return path.read_text(encoding="utf-8")
    parts = []
    for page_number in selected_pages:
        value = references.get(str(page_number)) if isinstance(references, dict) else None
        if not value:
            continue
        path = (manifest_dir / str(value)).resolve()
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def extract_text(document: DocumentModel) -> str:
    return "\n".join(
        block.text
        for block in sorted(document.blocks, key=lambda item: item.reading_order_index)
        if block.text.strip() and not block.skipped and not block.error
    )


def write_artifacts(
    result: PDFExtractionResult,
    output_dir: Path,
    *,
    translate_final: bool,
) -> tuple[str | None, str | None]:
    if result.document is None:
        return None, None
    output_dir.mkdir(parents=True, exist_ok=True)
    source_json = output_dir / "structured.json"
    source_markdown = output_dir / "source.md"
    source_pdf = output_dir / "source_reconstructed.pdf"
    source_json.write_text(result.document.model_dump_json(indent=2), encoding="utf-8")
    source_markdown.write_text(result.markdown, encoding="utf-8")
    reconstructor = Reconstructor()
    reconstructor.html_to_pdf(
        reconstructor.markdown_to_html(result.markdown, title="OCR benchmark"),
        source_pdf,
    )
    if not translate_final:
        return str(source_pdf), None

    translated_json = output_dir / "translated.json"
    translated_markdown = output_dir / "translated.md"
    run_translation_subprocess(
        document_path=source_json,
        markdown_path=source_markdown,
        output_document_path=translated_json,
        output_markdown_path=translated_markdown,
        settings={
            "model": DEFAULT_TRANSLATION_MODEL,
            "temperature": DEFAULT_LLM_TEMPERATURE,
            "top_p": DEFAULT_LLM_TOP_P,
            "top_k": DEFAULT_LLM_TOP_K,
            "min_p": DEFAULT_LLM_MIN_P,
            "presence_penalty": DEFAULT_LLM_PRESENCE_PENALTY,
            "repetition_penalty": DEFAULT_LLM_REPETITION_PENALTY,
            "max_tokens": 2048,
        },
    )
    translated_pdf = output_dir / "translated_reconstructed.pdf"
    translated_text = translated_markdown.read_text(encoding="utf-8")
    reconstructor.html_to_pdf(
        reconstructor.markdown_to_html(translated_text, title="Translated benchmark"),
        translated_pdf,
    )
    return str(source_pdf), str(translated_pdf)


def build_metrics(
    *,
    document_id: str,
    engine: str,
    run_type: str,
    run_index: int,
    dpi: int,
    page_count: int,
    extraction_seconds: float,
    downstream_seconds: float,
    wall_seconds: float,
    peak_rss_bytes: int | None,
    result: PDFExtractionResult,
    reference: str,
    expected_labels: list[str],
    expected_reading_order: list[str],
    reconstructed_pdf: str | None,
    translated_pdf: str | None,
) -> RunMetrics:
    assert result.document is not None
    text = extract_text(result.document)
    blocks = result.document.blocks
    labels = [block.block_type.value for block in blocks]
    deletions, insertions = (
        insertion_deletion_counts(reference, text) if reference else (None, None)
    )
    retries = sum(
        int(bool(result.metadata.get(key)))
        for key in ("marker_retried_on_cpu", "marker_fallback_to_normal")
    )
    empty_output = bool(reference.strip()) and not bool(text.strip())
    return RunMetrics(
        document_id=document_id,
        engine=engine,
        run_type=run_type,
        run_index=run_index,
        page_count=page_count,
        dpi=dpi,
        extraction_seconds=extraction_seconds,
        downstream_seconds=downstream_seconds,
        wall_seconds=wall_seconds,
        seconds_per_page=wall_seconds / max(page_count, 1),
        pages_per_second=page_count / max(wall_seconds, 1e-9),
        peak_rss_bytes=peak_rss_bytes,
        success=not empty_output,
        error="Engine returned no OCR text for a non-empty reference." if empty_output else None,
        retries=retries,
        block_errors=sum(1 for block in blocks if block.error),
        block_skips=sum(1 for block in blocks if block.skipped),
        text_characters=len(text),
        cer=character_error_rate(reference, text) if reference else None,
        wer=word_error_rate(reference, text) if reference else None,
        deletions=deletions,
        insertions=insertions,
        layout_label_accuracy=sequence_accuracy(expected_labels, labels),
        reading_order_accuracy=reading_order_accuracy(expected_reading_order, text),
        headers=sum(1 for block in blocks if block.block_type.value == "header"),
        footers=sum(1 for block in blocks if block.block_type.value == "footer"),
        tables=len(result.document.tables),
        equations=sum(1 for block in blocks if block.block_type.value == "equation"),
        figures=len(result.document.figures),
        captions=sum(1 for block in blocks if block.block_type.value == "caption"),
        reconstructed_pdf=reconstructed_pdf,
        translated_pdf=translated_pdf,
        markdown_preview=result.markdown[:600].replace("\n", " "),
        metadata=result.metadata,
    )


def failed_metrics(
    *,
    document_id: str,
    engine: str,
    run_type: str,
    run_index: int,
    dpi: int,
    page_count: int,
    wall_seconds: float,
    peak_rss_bytes: int | None,
    error: Exception,
) -> RunMetrics:
    return RunMetrics(
        document_id=document_id,
        engine=engine,
        run_type=run_type,
        run_index=run_index,
        page_count=page_count,
        dpi=dpi,
        extraction_seconds=wall_seconds,
        downstream_seconds=0,
        wall_seconds=wall_seconds,
        seconds_per_page=wall_seconds / max(page_count, 1),
        pages_per_second=0,
        peak_rss_bytes=peak_rss_bytes,
        success=False,
        error=str(error),
        retries=0,
        block_errors=0,
        block_skips=0,
        text_characters=0,
        cer=None,
        wer=None,
        deletions=None,
        insertions=None,
        layout_label_accuracy=None,
        reading_order_accuracy=None,
        headers=0,
        footers=0,
        tables=0,
        equations=0,
        figures=0,
        captions=0,
        reconstructed_pdf=None,
        translated_pdf=None,
        markdown_preview="",
        metadata={},
    )


def environment_report() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "memory_bytes": psutil.virtual_memory().total if psutil is not None else None,
        "versions": {
            "surya2": package_version(ROOT / ".venv-surya2/bin/python", "surya-ocr"),
            "marker": package_version(ROOT / ".venv-surya2/bin/python", "marker-pdf"),
            "mlx_vlm": current_package_version("mlx-vlm"),
            "qwen_model": DEFAULT_QWEN_OCR_MODEL,
            "llama_cpp": command_version(["llama-server", "--version"]),
        },
        "configuration": {
            "backend": "llamacpp",
            "concurrency": 1,
        },
    }


def current_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def package_version(python: Path, package: str) -> str:
    if not python.exists():
        return "not installed"
    code = f"import importlib.metadata; print(importlib.metadata.version({package!r}))"
    return command_version([str(python), "-c", code])


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return (result.stdout or result.stderr).strip().splitlines()[0]
    except Exception as exc:
        return f"unavailable: {exc}"


def median_rows(results: list[RunMetrics]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[RunMetrics]] = {}
    for result in results:
        if result.run_type == "warm":
            groups.setdefault((result.document_id, result.engine), []).append(result)
    rows = []
    for (document_id, engine), items in sorted(groups.items()):
        successful = [item for item in items if item.success]
        if not successful:
            rows.append(
                {
                    "document_id": document_id,
                    "engine": engine,
                    "runs": len(items),
                    "median_wall_seconds": None,
                    "median_seconds_per_page": None,
                    "median_pages_per_second": None,
                    "median_peak_rss_bytes": None,
                    "median_cer": None,
                    "median_wer": None,
                    "failure_rate": 1.0,
                }
            )
            continue
        rows.append(
            {
                "document_id": document_id,
                "engine": engine,
                "runs": len(items),
                "median_wall_seconds": statistics.median(item.wall_seconds for item in successful),
                "median_seconds_per_page": statistics.median(
                    item.seconds_per_page for item in successful
                ),
                "median_pages_per_second": statistics.median(
                    item.pages_per_second for item in successful
                ),
                "median_peak_rss_bytes": statistics.median(
                    item.peak_rss_bytes for item in successful if item.peak_rss_bytes is not None
                )
                if any(item.peak_rss_bytes is not None for item in successful)
                else None,
                "median_cer": statistics.median(
                    item.cer for item in successful if item.cer is not None
                )
                if any(item.cer is not None for item in successful)
                else None,
                "median_wer": statistics.median(
                    item.wer for item in successful if item.wer is not None
                )
                if any(item.wer is not None for item in successful)
                else None,
                "failure_rate": 1.0 - (len(successful) / len(items)),
            }
        )
    return rows


def render_report(
    environment: dict[str, Any],
    results: list[RunMetrics],
    medians: list[dict[str, Any]],
    manifest: dict[str, Any],
    command: str,
) -> str:
    lines = [
        "# TranslaTHOR OCR benchmark",
        "",
        f"Generated: {environment['timestamp_utc']}",
        "",
        "## Environment",
        "",
        f"- Platform: {environment['platform']}",
        f"- Machine: {environment['machine']}",
        f"- Python: {environment['python']}",
        f"- Physical memory: {environment['memory_bytes']}",
        f"- Versions: `{json.dumps(environment['versions'], sort_keys=True)}`",
        f"- Command: `{command}`",
        "",
        "## Test documents",
        "",
    ]
    for document in manifest.get("documents", []):
        lines.append(
            f"- {document.get('id')}: pages={document.get('pages')}, "
            f"characteristics={document.get('characteristics', [])}"
        )
    lines.extend(
        [
            "",
            "## Median warm results",
            "",
            "| Document | Engine | Runs | Median total s | s/page | pages/s | Peak RSS | CER | WER |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in medians:
        lines.append(
            "| {document_id} | {engine} | {runs} | {wall} | {spp} | {pps} | "
            "{rss} | {cer} | {wer} |".format(
                **row,
                wall=_fmt(row["median_wall_seconds"]),
                spp=_fmt(row["median_seconds_per_page"]),
                pps=_fmt(row["median_pages_per_second"]),
                rss=row["median_peak_rss_bytes"] or "",
                cer=_fmt(row["median_cer"]),
                wer=_fmt(row["median_wer"]),
            )
        )
    lines.extend(
        [
            "",
            "## Cold runs and failures",
            "",
            "| Document | Engine | Type | Success | Wall s | Errors | Retries |",
            "|---|---|---|---|---:|---|---:|",
        ]
    )
    for result in results:
        if result.run_type != "cold" and result.success:
            continue
        lines.append(
            f"| {result.document_id} | {result.engine} | {result.run_type} | "
            f"{result.success} | {_fmt(result.wall_seconds)} | "
            f"{(result.error or '')[:160]} | {result.retries} |"
        )
    lines.extend(["", "## Representative output previews", ""])
    seen: set[tuple[str, str]] = set()
    for result in results:
        key = (result.document_id, result.engine)
        if key in seen or not result.success:
            continue
        seen.add(key)
        lines.extend(
            [
                f"### {result.document_id} — {result.engine}",
                "",
                result.markdown_preview or "_No text output._",
                "",
                f"- Headers/footers: {result.headers}/{result.footers}",
                f"- Tables/equations/figures/captions: "
                f"{result.tables}/{result.equations}/{result.figures}/{result.captions}",
                f"- Coordinate overlay/output: {result.reconstructed_pdf or 'unavailable'}",
                f"- Translated PDF: {result.translated_pdf or 'not requested'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            "",
            "This file intentionally does not auto-declare a winner. Review the measured "
            "CER/WER, structure metrics, overlays, and reconstructed/translated PDFs, then "
            "record the production recommendation here.",
            "",
            "## Known limitations",
            "",
            "- CER/WER and label/order accuracy are blank when the manifest omits references.",
            "- Peak RSS is unavailable when `psutil` is not installed.",
            "- Legacy workers currently reload their models for each job; warm runs still "
            "benefit from filesystem/model caches but not a persistent Python model object.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def write_benchmark_outputs(
    *,
    output_dir: Path,
    environment: dict[str, Any],
    manifest: dict[str, Any],
    results: list[RunMetrics],
    command: str,
) -> None:
    """Atomically checkpoint all completed runs.

    OCR benchmarks can run for many minutes. Persisting after every run keeps a
    completed engine's measurements available if the terminal or stdout consumer
    disconnects before the remaining engines finish.
    """
    medians = median_rows(results)
    payload = {
        "schema_version": 1,
        "environment": environment,
        "manifest": manifest,
        "results": [asdict(result) for result in results],
        "warm_medians": medians,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        output_dir / "benchmark_results.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    _atomic_write_text(
        output_dir / "benchmark_report.md",
        render_report(environment, results, medians, manifest, command),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def configure_surya_batching(*, parallel: int, context_per_slot: int) -> int:
    if parallel < 1:
        raise ValueError("--surya-parallel must be at least 1.")
    if context_per_slot < 1:
        raise ValueError("--surya-context-per-slot must be at least 1.")
    total_context = max(16384, parallel * context_per_slot)
    os.environ["SURYA_INFERENCE_PARALLEL"] = str(parallel)
    os.environ["SURYA_INFERENCE_CTX_PER_SLOT"] = str(context_per_slot)
    os.environ["SURYA_INFERENCE_CTX_SIZE"] = str(total_context)
    return total_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TranslaTHOR OCR engines on identical PDF pages."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--engines",
        default=",".join(SUPPORTED_ENGINES),
        help="Comma-separated engine names.",
    )
    parser.add_argument("--dpi", type=int, default=192)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument(
        "--surya-parallel",
        type=int,
        default=DEFAULT_SURYA2_PARALLEL_PAGES,
        help="Maximum number of Surya page requests processed concurrently.",
    )
    parser.add_argument(
        "--surya-context-per-slot",
        type=int,
        default=DEFAULT_SURYA2_CONTEXT_PER_SLOT,
        help="llama-server context tokens reserved for each concurrent Surya page.",
    )
    parser.add_argument(
        "--cold-only",
        action="store_true",
        help="Run one fresh-process extraction per engine without warm repetitions.",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--translate-final", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warm_runs < 3 and not args.cold_only:
        raise ValueError("--warm-runs must be at least 3.")
    engines = [item.strip() for item in args.engines.split(",") if item.strip()]
    unknown = [engine for engine in engines if engine not in SUPPORTED_ENGINES]
    if unknown:
        raise ValueError(f"Unsupported engine(s): {', '.join(unknown)}")

    os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
    total_surya_context = configure_surya_batching(
        parallel=args.surya_parallel,
        context_per_slot=args.surya_context_per_slot,
    )
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_report()
    environment["configuration"].update(
        {
            "dpi": args.dpi,
            "warm_runs": 0 if args.cold_only else args.warm_runs,
            "engines": engines,
            "translate_final": args.translate_final,
            "surya_parallel": args.surya_parallel,
            "surya_context_per_slot": args.surya_context_per_slot,
            "surya_total_context": total_surya_context,
        }
    )
    command = " ".join(sys.argv)

    results: list[RunMetrics] = []
    detector = PDFTypeDetector()
    for document_spec in manifest.get("documents", []):
        document_id = str(document_spec["id"])
        source_pdf = (manifest_path.parent / str(document_spec["pdf"])).resolve()
        selected_pages = [int(page) for page in document_spec.get("pages", [])]
        if not selected_pages:
            selected_pages = list(range(1, len(PdfReader(str(source_pdf)).pages) + 1))
        benchmark_pdf = output_dir / "inputs" / f"{document_id}.pdf"
        full_page_range = list(range(1, len(PdfReader(str(source_pdf)).pages) + 1))
        if selected_pages == full_page_range:
            benchmark_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_pdf, benchmark_pdf)
        else:
            subset_pdf(source_pdf, selected_pages, benchmark_pdf)
        detection = detector.detect(benchmark_pdf)
        reference = resolve_reference_text(
            document_spec,
            manifest_dir=manifest_path.parent,
            selected_pages=selected_pages,
        )
        expected_labels = [str(value) for value in document_spec.get("expected_labels", [])]
        expected_reading_order = [
            str(value) for value in document_spec.get("expected_reading_order", [])
        ]

        for engine_name in engines:
            engine = BenchmarkEngine(engine_name, args.dpi, args.timeout)
            try:
                run_count = 1 if args.cold_only else args.warm_runs + 1
                for run_index in range(run_count):
                    run_type = "cold" if run_index == 0 else "warm"
                    run_dir = (
                        output_dir / "runs" / document_id / engine_name / f"{run_type}-{run_index}"
                    )
                    sampler = ProcessTreeMemorySampler()
                    sampler.start()
                    started = time.perf_counter()
                    try:
                        extraction = engine.extract(
                            pdf_path=benchmark_pdf,
                            job_dir=run_dir,
                            classification=detection.classification,
                            detection_metadata=detection.metadata,
                            warnings=detection.warnings,
                            sampler=sampler,
                        )
                        extraction_seconds = time.perf_counter() - started
                        downstream_started = time.perf_counter()
                        source_pdf_output, translated_pdf = write_artifacts(
                            extraction,
                            run_dir / "artifacts",
                            translate_final=args.translate_final and run_type == "cold",
                        )
                        downstream_seconds = time.perf_counter() - downstream_started
                        wall_seconds = time.perf_counter() - started
                        peak = sampler.stop()
                        metrics = build_metrics(
                            document_id=document_id,
                            engine=engine_name,
                            run_type=run_type,
                            run_index=run_index,
                            dpi=args.dpi,
                            page_count=len(selected_pages),
                            extraction_seconds=extraction_seconds,
                            downstream_seconds=downstream_seconds,
                            wall_seconds=wall_seconds,
                            peak_rss_bytes=peak,
                            result=extraction,
                            reference=reference,
                            expected_labels=expected_labels,
                            expected_reading_order=expected_reading_order,
                            reconstructed_pdf=source_pdf_output,
                            translated_pdf=translated_pdf,
                        )
                    except Exception as exc:
                        wall_seconds = time.perf_counter() - started
                        peak = sampler.stop()
                        metrics = failed_metrics(
                            document_id=document_id,
                            engine=engine_name,
                            run_type=run_type,
                            run_index=run_index,
                            dpi=args.dpi,
                            page_count=len(selected_pages),
                            wall_seconds=wall_seconds,
                            peak_rss_bytes=peak,
                            error=exc,
                        )
                    results.append(metrics)
                    write_benchmark_outputs(
                        output_dir=output_dir,
                        environment=environment,
                        manifest=manifest,
                        results=results,
                        command=command,
                    )
                    print(json.dumps(asdict(metrics), ensure_ascii=False), flush=True)
            finally:
                engine.close()

    write_benchmark_outputs(
        output_dir=output_dir,
        environment=environment,
        manifest=manifest,
        results=results,
        command=command,
    )
    return 0 if all(result.success for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
