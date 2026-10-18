#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${BITMARK_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  RUNTIME_DIR="${BITMARK_RUNTIME_ROOT:-/content/bitmark_runtime}"
  if [[ -x "$RUNTIME_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$RUNTIME_DIR/.venv/bin/python"
  elif [[ -x "$HERE/.venv/bin/python" ]]; then
    PYTHON_BIN="$HERE/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
echo "BitMark Python: $PYTHON_BIN"
exec "$PYTHON_BIN" -u "$HERE/run.py" "$@"
