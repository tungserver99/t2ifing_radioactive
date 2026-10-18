#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SERUM_PYTHON:-${BITMARK_PYTHON:-}}"
if [[ -z "$PYTHON_BIN" ]]; then
  RUNTIME_DIR="${BITMARK_RUNTIME_ROOT:-/content/bitmark_runtime}"
  if [[ -x "$RUNTIME_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$RUNTIME_DIR/.venv/bin/python"
  elif [[ -x "$ROOT_DIR/bitmark_lora/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/bitmark_lora/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
echo "SERUM Python: $PYTHON_BIN"
exec "$PYTHON_BIN" -u "$ROOT_DIR/bitmark_lora/run_serum.py" "$@"
