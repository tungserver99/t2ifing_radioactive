#!/bin/sh
# One entry point; the locked rescue is the default new campaign.
cd "$(dirname "$0")" || exit 1
for arg in "$@"; do
    case "$arg" in
        --resume|--resume=*|--analyze|--analyze=*|--teacher-check|--teacher-check=*|--teacher-objective|--teacher-objective=*)
            exec python main.py "$@"
            ;;
    esac
done
exec python main.py --teacher-objective semantic_sign_v2_trajectory --teacher-only "$@"
