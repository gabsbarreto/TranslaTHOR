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
