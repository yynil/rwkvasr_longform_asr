#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configs/emilia_en_zh_dirdrop_both_spm8k_stage2_from_best_4x4090_deepspeed.yaml"

INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${REPO_ROOT}/runs/emilia_en_zh_bi_baseline_spm8k_4x4090/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/emilia_en_zh_dirdrop_both_spm8k_stage2_from_best_4x4090}"
WANDB_PROJECT="${WANDB_PROJECT:-rwkvasr_longform_asr_dirdrop_stage2}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sp8k_4090_dirdrop_from_best}"

if [[ ! -f "${INIT_CHECKPOINT_PATH}" ]]; then
  echo "Initial checkpoint not found: ${INIT_CHECKPOINT_PATH}" >&2
  exit 1
fi

exec "${REPO_ROOT}/scripts/train_paper_rwkv_asr.sh" \
  --config-yaml "${BASE_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --init-checkpoint-path "${INIT_CHECKPOINT_PATH}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  "$@"
