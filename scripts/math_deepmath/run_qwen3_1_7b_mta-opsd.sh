#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/../_run_verl.sh" \
    "${SCRIPT_DIR}/../../configs/math_deepmath/qwen3_1_7b_mta-opsd.yaml" "$@"
