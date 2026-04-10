#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_paper_rwkv_asr.sh"

if [[ ! -x "${TRAIN_SCRIPT}" ]]; then
  echo "Training launcher not found or not executable: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

TARGET_EPOCHS="${TARGET_EPOCHS:-15}"
RESUME_FROM="${REPO_ROOT}/runs/paper_bi_baseline_4x4090/ds_checkpoints"
RESUME_TAG="${RESUME_TAG:-best}"

exec "${TRAIN_SCRIPT}" \
  bi_baseline \
  --resume-from "${RESUME_FROM}" \
  --resume-tag "${RESUME_TAG}" \
  --epochs "${TARGET_EPOCHS}" \
  "$@"
