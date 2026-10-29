#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CONFIG="${REPO_ROOT}/configs/val/qwen3_1_7b_opsd.yaml"

exec python3 "${SCRIPT_DIR}/validate_checkpoints.py" \
    --config "${DEFAULT_CONFIG}" "$@"

### Usage for basemodel
# bash scripts/_run_verl.sh \
#   configs/val/qwen3_1_7b_opsd.yaml \
#   trainer.resume_mode=disable \
#   trainer.val_before_train=true \
#   trainer.val_only=true \
#   trainer.logger='[console,file]'