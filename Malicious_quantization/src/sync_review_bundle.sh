#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
echo "Source is canonical in: $SCRIPT_DIR"
echo "Review this folder directly; no generated source bundle is needed."
