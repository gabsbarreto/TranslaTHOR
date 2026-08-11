#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required on macOS. Install llama.cpp manually on other platforms."
  echo "Official releases: https://github.com/ggml-org/llama.cpp/releases"
  exit 1
fi

if ! command -v llama-server >/dev/null 2>&1; then
  brew install llama.cpp
fi

python3 -m venv .venv-surya2
.venv-surya2/bin/python -m pip install --upgrade pip
.venv-surya2/bin/python -m pip install --requirement requirements-surya2.lock.txt

export SURYA_INFERENCE_BACKEND=llamacpp
.venv-surya2/bin/python -c \
  'import importlib.metadata; print("surya-ocr", importlib.metadata.version("surya-ocr")); print("marker-pdf", importlib.metadata.version("marker-pdf"))'
llama-server --version

echo "Surya 2 llama.cpp and Marker 2 runtime is ready."
