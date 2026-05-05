#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv-deepseek-ocr
.venv-deepseek-ocr/bin/python -m pip install -U pip
.venv-deepseek-ocr/bin/python -m pip install "mlx-vlm>=0.3.10"

if [[ -f .env.local ]]; then
  if ! grep -q '^DEEPSEEK_OCR_PYTHON=' .env.local; then
    printf '\nDEEPSEEK_OCR_PYTHON=.venv-deepseek-ocr/bin/python\n' >> .env.local
  fi
else
  printf 'DEEPSEEK_OCR_PYTHON=.venv-deepseek-ocr/bin/python\n' > .env.local
fi

echo "Qwen OCR environment ready: .venv-deepseek-ocr/bin/python"
