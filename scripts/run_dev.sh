#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

export PYTHONPATH=backend
export PATH="$(pwd)/.venv/bin:/opt/homebrew/bin:/opt/homebrew/sbin:${PATH}"
export SURYA_INFERENCE_BACKEND=llamacpp
export SURYA_GUIDED_LAYOUT="${SURYA_GUIDED_LAYOUT:-false}"
export SURYA_INFERENCE_PARALLEL="${SURYA_INFERENCE_PARALLEL:-5}"
export SURYA_INFERENCE_CTX_PER_SLOT="${SURYA_INFERENCE_CTX_PER_SLOT:-16384}"
export SURYA2_PYTHON="${SURYA2_PYTHON:-$(pwd)/.venv-surya2/bin/python}"
export SURYA_LAYOUT_PYTHON="${SURYA_LAYOUT_PYTHON:-$(pwd)/.venv-surya2/bin/python}"
export MARKER_BIN="${MARKER_BIN:-$(pwd)/.venv-surya2/bin/marker_single}"
export MARKER_CONVERSION_MODE="${MARKER_CONVERSION_MODE:-balanced}"
if [[ -d /opt/homebrew/lib ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
fi
if [[ -d /opt/homebrew/share ]]; then
  export XDG_DATA_DIRS="/opt/homebrew/share:${XDG_DATA_DIRS:-}"
fi

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  PYTHON=python
fi

"$PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
