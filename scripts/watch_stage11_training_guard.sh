#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

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
AR_FREEZE_ENCODER="${AR_FREEZE_ENCODER:-0}"
AR_FREEZE_CTC_HEAD="${AR_FREEZE_CTC_HEAD:-0}"

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
SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_CER="${SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_CER:-}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_MIN_LENGTH_RATIO="${SIDECAR_AR_CTC_DRAFT_FALLBACK_MIN_LENGTH_RATIO:-0.80}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_LENGTH_RATIO="${SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_LENGTH_RATIO:-1.20}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_REJECT_REPETITION="${SIDECAR_AR_CTC_DRAFT_FALLBACK_REJECT_REPETITION:-1}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_METRIC_NORMALIZATION="${SIDECAR_AR_CTC_DRAFT_FALLBACK_METRIC_NORMALIZATION:-ctc}"

HANDOFF_SESSION="${HANDOFF_SESSION:-stage11_after_draft}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
RESTART_COOLDOWN_SECONDS="${RESTART_COOLDOWN_SECONDS:-600}"

log() {
  printf '[stage11-guard] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

acquire_lock() {
  mkdir -p "${AR_RUN_DIR}/logs"
  exec 9>"${AR_RUN_DIR}/logs/stage11_guard.lock"
  if ! flock -n 9; then
    log "another guard instance is already active; exiting"
    exit 0
  fi
}

draft_ready() {
  [[ -s "${DRAFT_CACHE_PATH}" && -s "${DRAFT_LENGTH_INDEX_PATH}" ]]
}

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

training_process_active() {
  local pid comm args
  while read -r pid comm args; do
    [[ "${args}" == *"${AR_CONFIG}"* ]] || continue
    case "${comm}" in
      rg|grep|pgrep|zsh|bash)
        continue
        ;;
    esac
    if [[ "${args}" == *"train_paper_rwkv_asr.sh --config-yaml ${AR_CONFIG}"* ]] \
      || [[ "${args}" == *"deepspeed"* && "${args}" == *"${AR_CONFIG}"* ]] \
      || [[ "${args}" == *"torchrun"* && "${args}" == *"${AR_CONFIG}"* ]] \
      || [[ "${args}" == *"rwkvasr.cli.train_ctc"* && "${args}" == *"${AR_CONFIG}"* ]]; then
      return 0
    fi
  done < <(ps -eo pid=,comm=,args=)
  return 1
}

latest_step_checkpoint() {
  find "${AR_RUN_DIR}" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^step-\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

config_max_steps() {
  if [[ -s "${AR_RUN_DIR}/train_config.yaml" ]]; then
    "${PYTHON_BIN}" - "${AR_RUN_DIR}/train_config.yaml" "${AR_MAX_STEPS}" <<'PY'
import sys
from rwkvasr.config import load_yaml

fallback = int(sys.argv[2])
try:
    value = load_yaml(sys.argv[1]).get("max_steps")
except Exception:
    value = None
print(int(value) if value is not None else fallback)
PY
  else
    printf '%s\n' "${AR_MAX_STEPS}"
  fi
}

start_stage11() {
  export CTC_RUN_DIR CTC_CONFIG INIT_CHECKPOINT_PATH DRAFT_CACHE_DIR DRAFT_CACHE_PATH DRAFT_LENGTH_INDEX_PATH
  export AR_RUN_DIR AR_CONFIG AR_MAX_STEPS AR_CTC_LOSS_WEIGHT AR_DECODER_LOSS_WEIGHT AR_LR AR_BATCH_SIZE
  export AR_GRAD_ACCUM AR_DECODER_TEXT_TOKEN_BUDGET AR_SAVE_EVERY AR_RESUME_FROM
  export AR_DRAFT_DROPOUT_PROB AR_DRAFT_LANGUAGE_MISMATCH_DROPOUT_PROB AR_DRAFT_DROPOUT_SEED
  export AR_FREEZE_ENCODER AR_FREEZE_CTC_HEAD
  export NUM_GPUS MASTER_PORT TRAINING_SESSION SIDECAR_SESSION SIDECAR_LIMIT SIDECAR_SOURCE_QUOTAS
  export SIDECAR_DEVICE SIDECAR_BATCH_SIZE SIDECAR_POLL_SECONDS SIDECAR_TEXT_NORMALIZATION SIDECAR_METRIC_NORMALIZATION
  export SIDECAR_AR_TEMPERATURE SIDECAR_AR_TOP_K SIDECAR_AR_TOP_P SIDECAR_AR_SEED
  export SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_CER SIDECAR_AR_CTC_DRAFT_FALLBACK_MIN_LENGTH_RATIO
  export SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_LENGTH_RATIO SIDECAR_AR_CTC_DRAFT_FALLBACK_REJECT_REPETITION
  export SIDECAR_AR_CTC_DRAFT_FALLBACK_METRIC_NORMALIZATION
  bash "${REPO_ROOT}/scripts/start_stage7_ctc_draft_ar_training.sh"
}

main() {
  cd "${REPO_ROOT}"
  acquire_lock
  local restart_count=0
  local last_restart_ts=0

  log "watching run_dir=${AR_RUN_DIR}"
  while true; do
    if ! draft_ready; then
      log "waiting for final draft cache and filtered length index"
      wc -l "${DRAFT_CACHE_DIR}"/parts/part_0*.jsonl 2>/dev/null || true
      sleep "${POLL_SECONDS}"
      continue
    fi

    local latest_step max_steps
    latest_step="$(latest_step_checkpoint || true)"
    latest_step="${latest_step:-0}"
    max_steps="$(config_max_steps)"
    if [[ "${latest_step}" -ge "${max_steps}" ]]; then
      log "Stage11 appears complete latest_step=${latest_step} max_steps=${max_steps}; guard exiting"
      return 0
    fi

    if training_process_active; then
      log "training process active latest_step=${latest_step} max_steps=${max_steps}"
      sleep "${POLL_SECONDS}"
      continue
    fi

    if [[ ! -s "${AR_CONFIG}" ]] && tmux_has_session "${HANDOFF_SESSION}"; then
      log "handoff session ${HANDOFF_SESSION} still owns initial start; waiting"
      sleep "${POLL_SECONDS}"
      continue
    fi

    if [[ "${restart_count}" -ge "${MAX_RESTARTS}" ]]; then
      log "restart limit reached restart_count=${restart_count}; latest_step=${latest_step} max_steps=${max_steps}; waiting without restart"
      sleep "${POLL_SECONDS}"
      continue
    fi

    local now
    now="$(date +%s)"
    if [[ $((now - last_restart_ts)) -lt "${RESTART_COOLDOWN_SECONDS}" ]]; then
      log "restart cooldown active latest_step=${latest_step} max_steps=${max_steps}"
      sleep "${POLL_SECONDS}"
      continue
    fi

    restart_count=$((restart_count + 1))
    last_restart_ts="${now}"
    log "starting/restarting Stage11 attempt=${restart_count} latest_step=${latest_step} max_steps=${max_steps}"
    start_stage11 || log "start script returned nonzero; will retry after cooldown"
    sleep "${POLL_SECONDS}"
  done
}

main "$@"
