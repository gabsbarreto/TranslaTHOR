#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/benchmark_pipeline.sh /path/to/file.pdf
#
# Requires local server running on http://127.0.0.1:8000.

PDF_PATH="${1:-}"
if [[ -z "$PDF_PATH" || ! -f "$PDF_PATH" ]]; then
  echo "Usage: $0 /path/to/file.pdf"
  exit 1
fi

submit_job() {
  local model="$1"
  local render_strategy="$2"

  curl -sS -X POST "http://127.0.0.1:8000/api/jobs" \
    -F "files=@${PDF_PATH}" \
    -F "profile_pipeline=true" \
    -F "model=${model}" \
    -F "temperature=0.0" \
    -F "top_p=1.0" \
    -F "chunk_size=1800" \
    -F "max_tokens=2048" \
    -F "render_strategy=${render_strategy}" \
    -F "output_mode=readable"
}

echo "Benchmark A: TranslateGemma + pre_render_all"
submit_job "mlx-community/translategemma-12b-it-4bit" "pre_render_all"
echo
echo "Benchmark B: TranslateGemma + on_demand"
submit_job "mlx-community/translategemma-12b-it-4bit" "on_demand"
echo
echo "Benchmark C: Llama 3.1 8B + pre_render_all"
submit_job "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit" "pre_render_all"
echo
echo "Submitted. Watch /api/jobs and download timing artifacts (profile_summary/profile_json/profile_csv)."
