#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CTC_RUN_DIR="${CTC_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage10b_fullmix_ctc_noaug_from_stage9werbest_bs12_lr5e6_zero1_nockpt_4x4090}"
CTC_CONFIG="${CTC_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage10b_fullmix_ctc_noaug_from_stage9werbest_4x4090_deepspeed.yaml}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${CTC_RUN_DIR}/wercer_best.pt}"
DRAFT_CACHE_DIR="${DRAFT_CACHE_DIR:-${CTC_RUN_DIR}/ctc_draft_cache_stage10b_balanced2m}"
DRAFT_CACHE_PATH="${DRAFT_CACHE_PATH:-${DRAFT_CACHE_DIR}/ctc_draft_train.jsonl}"
DRAFT_LENGTH_INDEX_PATH="${DRAFT_LENGTH_INDEX_PATH:-${DRAFT_CACHE_DIR}/webdataset_lengths.cached.jsonl}"

AR_RUN_DIR="${AR_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage11_balanced2m_ctc_draft_ar_from_stage10b_werbest_ctc0p3_ar0p7_bs4_lr1e5_zero1_nockpt_4x4090}"
AR_CONFIG="${AR_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage11_balanced2m_ctc_draft_ar_from_stage10b_werbest_4x4090_deepspeed.yaml}"
AR_MAX_STEPS="${AR_MAX_STEPS:-20000}"
AR_CTC_LOSS_WEIGHT="${AR_CTC_LOSS_WEIGHT:-0.3}"
AR_DECODER_LOSS_WEIGHT="${AR_DECODER_LOSS_WEIGHT:-0.7}"
AR_LR="${AR_LR:-1.0e-5}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-4}"
AR_GRAD_ACCUM="${AR_GRAD_ACCUM:-4}"
AR_DECODER_TEXT_TOKEN_BUDGET="${AR_DECODER_TEXT_TOKEN_BUDGET:-1024}"
AR_SAVE_EVERY="${AR_SAVE_EVERY:-500}"
AR_RESUME_FROM="${AR_RESUME_FROM:-auto}"
AR_DRAFT_DROPOUT_PROB="${AR_DRAFT_DROPOUT_PROB:-0.20}"
AR_DRAFT_LANGUAGE_MISMATCH_DROPOUT_PROB="${AR_DRAFT_LANGUAGE_MISMATCH_DROPOUT_PROB:-1.00}"
AR_DRAFT_DROPOUT_SEED="${AR_DRAFT_DROPOUT_SEED:-20260615}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29591}"
TRAINING_SESSION="${TRAINING_SESSION:-training}"
SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage11_draft_ar}"
SIDECAR_LIMIT="${SIDECAR_LIMIT:-64}"
SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-gigaspeech:GSXL-*.tar:32,wenetspeech:WSL-*.tar:32}"
SIDECAR_DEVICE="${SIDECAR_DEVICE:-cpu}"
SIDECAR_BATCH_SIZE="${SIDECAR_BATCH_SIZE:-1}"
SIDECAR_POLL_SECONDS="${SIDECAR_POLL_SECONDS:-1800}"
SIDECAR_TEXT_NORMALIZATION="${SIDECAR_TEXT_NORMALIZATION:-runtime}"
SIDECAR_METRIC_NORMALIZATION="${SIDECAR_METRIC_NORMALIZATION:-ctc}"
SIDECAR_AR_TEMPERATURE="${SIDECAR_AR_TEMPERATURE:-0.35}"
SIDECAR_AR_TOP_K="${SIDECAR_AR_TOP_K:-3}"
SIDECAR_AR_TOP_P="${SIDECAR_AR_TOP_P:-0.5}"
SIDECAR_AR_SEED="${SIDECAR_AR_SEED:-20260615}"

DRAFT_SESSION="${DRAFT_SESSION:-draft_stage10b_balanced2m}"
POLL_SECONDS="${POLL_SECONDS:-600}"

log() {
  printf '[stage11-after-draft] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

draft_ready() {
  [[ -s "${DRAFT_CACHE_PATH}" && -s "${DRAFT_LENGTH_INDEX_PATH}" ]]
}

print_part_counts() {
  wc -l "${DRAFT_CACHE_DIR}"/parts/part_0*.jsonl 2>/dev/null || true
}

main() {
  cd "${REPO_ROOT}"
  mkdir -p "${AR_RUN_DIR}/logs"
  log "watching draft cache path=${DRAFT_CACHE_PATH}"
  while ! draft_ready; do
    log "waiting for draft cache and filtered length index"
    print_part_counts
    if [[ -n "${DRAFT_SESSION}" ]] && ! tmux has-session -t "${DRAFT_SESSION}" 2>/dev/null; then
      log "draft session ${DRAFT_SESSION} ended before cache was ready"
      exit 1
    fi
    sleep "${POLL_SECONDS}"
  done

  log "draft cache ready; starting Stage11 CTC+AR"
  export CTC_RUN_DIR CTC_CONFIG INIT_CHECKPOINT_PATH DRAFT_CACHE_DIR DRAFT_CACHE_PATH DRAFT_LENGTH_INDEX_PATH
  export AR_RUN_DIR AR_CONFIG AR_MAX_STEPS AR_CTC_LOSS_WEIGHT AR_DECODER_LOSS_WEIGHT AR_LR AR_BATCH_SIZE
  export AR_GRAD_ACCUM AR_DECODER_TEXT_TOKEN_BUDGET AR_SAVE_EVERY AR_RESUME_FROM
  export AR_DRAFT_DROPOUT_PROB AR_DRAFT_LANGUAGE_MISMATCH_DROPOUT_PROB AR_DRAFT_DROPOUT_SEED
  export NUM_GPUS MASTER_PORT TRAINING_SESSION SIDECAR_SESSION SIDECAR_LIMIT SIDECAR_SOURCE_QUOTAS
  export SIDECAR_DEVICE SIDECAR_BATCH_SIZE SIDECAR_POLL_SECONDS SIDECAR_TEXT_NORMALIZATION SIDECAR_METRIC_NORMALIZATION
  export SIDECAR_AR_TEMPERATURE SIDECAR_AR_TOP_K SIDECAR_AR_TOP_P SIDECAR_AR_SEED
  exec bash "${REPO_ROOT}/scripts/start_stage7_ctc_draft_ar_training.sh"
}

main "$@"
