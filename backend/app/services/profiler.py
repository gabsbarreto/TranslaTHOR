from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class PipelineProfiler:
    """Lightweight pipeline profiler with per-stage and per-page timings."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._t0 = time.perf_counter()
        self.stage_totals: dict[str, float] = defaultdict(float)
        self.page_stage_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.events: list[dict[str, object]] = []

    @contextmanager
    def step(self, stage: str, page: int | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record(stage, elapsed, page=page)

    def record(self, stage: str, seconds: float, page: int | None = None) -> None:
        if not self.enabled:
            return
        self.stage_totals[stage] += seconds
        if page is not None:
            self.page_stage_totals[page][stage] += seconds
        self.events.append(
            {
                "stage": stage,
                "seconds": round(seconds, 6),
                "page": page,
                "timestamp_s": round(time.perf_counter() - self._t0, 6),
            }
        )

    def summary_lines(self) -> list[str]:
        if not self.enabled:
            return ["Profiling disabled"]

        lines: list[str] = []
        for stage, total in sorted(self.stage_totals.items(), key=lambda x: x[0]):
            per_page_values = [m.get(stage, 0.0) for m in self.page_stage_totals.values() if m.get(stage, 0.0) > 0]
            if per_page_values:
                avg = total / max(len(per_page_values), 1)
                lines.append(f"{stage}: {total:.2f}s total, {avg:.2f}s/page")
            else:
                lines.append(f"{stage}: {total:.2f}s total")
        return lines

    def dump(self, artifacts_dir: Path) -> tuple[Path, Path, Path]:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifacts_dir / "timing_profile.json"
        csv_path = artifacts_dir / "timing_profile.csv"
        summary_path = artifacts_dir / "timing_summary.txt"

        payload = {
            "enabled": self.enabled,
            "stage_totals": {k: round(v, 6) for k, v in self.stage_totals.items()},
            "page_stage_totals": {
                str(page): {k: round(v, 6) for k, v in metrics.items()}
                for page, metrics in sorted(self.page_stage_totals.items())
            },
            "events": self.events,
            "summary": self.summary_lines(),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["page", "stage", "seconds"])
            for page, metrics in sorted(self.page_stage_totals.items()):
                for stage, seconds in sorted(metrics.items()):
                    writer.writerow([page, stage, f"{seconds:.6f}"])
            for stage, seconds in sorted(self.stage_totals.items()):
                writer.writerow(["ALL", stage, f"{seconds:.6f}"])

        summary_path.write_text("\n".join(self.summary_lines()) + "\n", encoding="utf-8")
        return json_path, csv_path, summary_path
