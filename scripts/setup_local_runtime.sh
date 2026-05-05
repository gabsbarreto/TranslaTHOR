#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required on macOS to install system dependencies."
  exit 1
fi

brew install cairo pango gdk-pixbuf libffi

python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[mlx,deepseek_ocr,dev]"

if [[ ! -x .venv-deepseek-ocr/bin/python ]]; then
  bash scripts/setup_deepseek_ocr_env.sh
fi

touch .env.local

upsert_env() {
  local key="$1"
  local value="$2"
  local quoted
  quoted="'$value'"
  if grep -q "^${key}=" .env.local; then
    perl -0pi -e "s#^${key}=.*#${key}=${quoted}#mg" .env.local
  else
    printf '\n%s=%s\n' "$key" "$quoted" >> .env.local
  fi
}

upsert_env "DEEPSEEK_OCR_PYTHON" ".venv-deepseek-ocr/bin/python"

echo "Local runtime is ready."
echo "Qwen OCR Python: .venv-deepseek-ocr/bin/python"
