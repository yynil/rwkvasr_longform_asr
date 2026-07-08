#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RICH_FORCE_TERMINAL="${RICH_FORCE_TERMINAL:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CTC_RUN_DIR="${CTC_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage7_large_ctc_clean16_hard84_from_stage6c_step5000_bs12_lr3e5_zero1_nockpt_4x4090}"
CTC_CONFIG="${CTC_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage7_large_ctc_clean16_hard84_from_stage6c_step5000_4x4090_deepspeed.yaml}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${CTC_RUN_DIR}/wercer_best.pt}"

DRAFT_CACHE_DIR="${DRAFT_CACHE_DIR:-${CTC_RUN_DIR}/ctc_draft_cache_stage7_full}"
DRAFT_CACHE_PATH="${DRAFT_CACHE_PATH:-${DRAFT_CACHE_DIR}/ctc_draft_train.jsonl}"
DRAFT_LENGTH_INDEX_PATH="${DRAFT_LENGTH_INDEX_PATH:-${DRAFT_CACHE_DIR}/webdataset_lengths.cached.jsonl}"

AR_RUN_DIR="${AR_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage7_full_ctc_draft_ar_from_stage7_werbest_ctc0p4_ar0p6_bs4_lr2e5_zero1_nockpt_4x4090}"
AR_CONFIG="${AR_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage7_full_ctc_draft_ar_from_stage7_werbest_4x4090_deepspeed.yaml}"
AR_MAX_STEPS="${AR_MAX_STEPS:-10000}"
AR_CTC_LOSS_WEIGHT="${AR_CTC_LOSS_WEIGHT:-0.4}"
AR_DECODER_LOSS_WEIGHT="${AR_DECODER_LOSS_WEIGHT:-0.6}"
AR_LR="${AR_LR:-2.0e-5}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-4}"
AR_GRAD_ACCUM="${AR_GRAD_ACCUM:-4}"
AR_DECODER_TEXT_TOKEN_BUDGET="${AR_DECODER_TEXT_TOKEN_BUDGET:-1024}"
AR_SAVE_EVERY="${AR_SAVE_EVERY:-500}"
AR_RESUME_FROM="${AR_RESUME_FROM:-auto}"
AR_RESUME_TAG="${AR_RESUME_TAG:-}"
AR_DRAFT_DROPOUT_PROB="${AR_DRAFT_DROPOUT_PROB:-0.20}"
AR_DRAFT_LANGUAGE_MISMATCH_DROPOUT_PROB="${AR_DRAFT_LANGUAGE_MISMATCH_DROPOUT_PROB:-1.00}"
AR_DRAFT_DROPOUT_SEED="${AR_DRAFT_DROPOUT_SEED:-20260615}"
AR_FREEZE_ENCODER="${AR_FREEZE_ENCODER:-0}"
AR_FREEZE_CTC_HEAD="${AR_FREEZE_CTC_HEAD:-0}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29571}"
TRAINING_SESSION="${TRAINING_SESSION:-training}"
SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage7_draft_ar}"
SIDECAR_LIMIT="${SIDECAR_LIMIT:-64}"
SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:8,clean_aishell:aishell3_*.tar:8,clean_cv_en:commonvoice_en_*.tar:8,clean_cv_cn:commonvoice_cn_*.tar:8,gigaspeech:GSXL-*.tar:16,wenetspeech:WSL-*.tar:16}"
SIDECAR_DEVICE="${SIDECAR_DEVICE:-cuda:0}"
SIDECAR_BATCH_SIZE="${SIDECAR_BATCH_SIZE:-1}"
SIDECAR_POLL_SECONDS="${SIDECAR_POLL_SECONDS:-900}"
SIDECAR_TEXT_NORMALIZATION="${SIDECAR_TEXT_NORMALIZATION:-runtime}"
SIDECAR_METRIC_NORMALIZATION="${SIDECAR_METRIC_NORMALIZATION:-ctc}"
SIDECAR_AR_TEMPERATURE="${SIDECAR_AR_TEMPERATURE:-0.6}"
SIDECAR_AR_TOP_K="${SIDECAR_AR_TOP_K:-3}"
SIDECAR_AR_TOP_P="${SIDECAR_AR_TOP_P:-0.6}"
SIDECAR_AR_SEED="${SIDECAR_AR_SEED:-20260615}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_CER="${SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_CER:-}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_MIN_LENGTH_RATIO="${SIDECAR_AR_CTC_DRAFT_FALLBACK_MIN_LENGTH_RATIO:-0.80}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_LENGTH_RATIO="${SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_LENGTH_RATIO:-1.20}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_REJECT_REPETITION="${SIDECAR_AR_CTC_DRAFT_FALLBACK_REJECT_REPETITION:-1}"
SIDECAR_AR_CTC_DRAFT_FALLBACK_METRIC_NORMALIZATION="${SIDECAR_AR_CTC_DRAFT_FALLBACK_METRIC_NORMALIZATION:-ctc}"

if [[ -z "${AR_DRAFT_PROMPT_TEMPLATE:-}" ]]; then
  AR_DRAFT_PROMPT_TEMPLATE="$(cat <<'EOF'
The first-pass CTC transcript below may contain recognition errors:
{ctc_draft}

Use the audio as the source of truth. Treat the draft only as a speech-grounded hint, correct it when the audio supports the correction, and do not add words that are not spoken.

EOF
)"
fi

log() {
  printf '[stage7-draft-ar] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
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

write_ar_config() {
  if [[ ! -s "${INIT_CHECKPOINT_PATH}" ]]; then
    echo "init checkpoint missing: ${INIT_CHECKPOINT_PATH}" >&2
    exit 1
  fi
  if [[ ! -s "${DRAFT_CACHE_PATH}" ]]; then
    echo "draft cache missing: ${DRAFT_CACHE_PATH}" >&2
    exit 1
  fi
  if [[ ! -s "${DRAFT_LENGTH_INDEX_PATH}" ]]; then
    echo "draft length index missing: ${DRAFT_LENGTH_INDEX_PATH}" >&2
    exit 1
  fi

  export CTC_CONFIG AR_CONFIG AR_RUN_DIR INIT_CHECKPOINT_PATH DRAFT_CACHE_PATH DRAFT_LENGTH_INDEX_PATH
  export AR_MAX_STEPS AR_CTC_LOSS_WEIGHT AR_DECODER_LOSS_WEIGHT AR_LR AR_BATCH_SIZE AR_GRAD_ACCUM AR_SAVE_EVERY
  export AR_DECODER_TEXT_TOKEN_BUDGET
  export AR_DRAFT_DROPOUT_PROB AR_DRAFT_LANGUAGE_MISMATCH_DROPOUT_PROB AR_DRAFT_DROPOUT_SEED
  export AR_FREEZE_ENCODER AR_FREEZE_CTC_HEAD
  export AR_RESUME_FROM AR_RESUME_TAG
  export AR_DRAFT_PROMPT_TEMPLATE
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from rwkvasr.config import load_yaml, save_yaml

cfg = dict(load_yaml(os.environ["CTC_CONFIG"]))
run_dir = Path(os.environ["AR_RUN_DIR"])
cfg["output_dir"] = str(run_dir)
cfg["init_checkpoint_path"] = os.environ["INIT_CHECKPOINT_PATH"]
resume_from = str(os.environ.get("AR_RESUME_FROM") or "auto")
resume_tag = str(os.environ.get("AR_RESUME_TAG") or "") or None
if resume_from == "auto":
    cfg["resume_from"] = "latest" if (run_dir / "latest_checkpoint.yaml").is_file() else None
    cfg["resume_tag"] = None
elif resume_from.lower() in {"", "0", "false", "fresh", "none", "null"}:
    cfg["resume_from"] = None
    cfg["resume_tag"] = None
else:
    cfg["resume_from"] = resume_from
    cfg["resume_tag"] = resume_tag
cfg["decoder_enabled"] = True
cfg["webdataset_length_index_path"] = os.environ["DRAFT_LENGTH_INDEX_PATH"]
cfg["webdataset_bucket_manifest_path"] = None
cfg["bucket_source_interleave"] = True
cfg["decoder_ctc_draft_cache_path"] = os.environ["DRAFT_CACHE_PATH"]
cfg["decoder_ctc_draft_prompt_template"] = os.environ["AR_DRAFT_PROMPT_TEMPLATE"]
cfg["decoder_ctc_draft_text_key"] = "pred_text"
cfg["decoder_ctc_draft_missing_policy"] = "error"
cfg["decoder_ctc_draft_dropout_prob"] = float(os.environ["AR_DRAFT_DROPOUT_PROB"])
cfg["decoder_ctc_draft_language_mismatch_dropout_prob"] = float(
    os.environ["AR_DRAFT_LANGUAGE_MISMATCH_DROPOUT_PROB"]
)
cfg["decoder_ctc_draft_dropout_seed"] = int(os.environ["AR_DRAFT_DROPOUT_SEED"])
cfg["decoder_text_token_budget"] = int(os.environ["AR_DECODER_TEXT_TOKEN_BUDGET"])
cfg["text_normalization"] = "ctc"
cfg["decoder_text_normalization"] = "runtime"
cfg["tokenizer_append_eos"] = False
cfg["decoder_tokenizer_append_eos"] = False
cfg["ctc_loss_weight"] = float(os.environ["AR_CTC_LOSS_WEIGHT"])
cfg["decoder_loss_weight"] = float(os.environ["AR_DECODER_LOSS_WEIGHT"])
cfg["freeze_encoder"] = str(os.environ["AR_FREEZE_ENCODER"]).lower() not in {"", "0", "false", "no"}
cfg["freeze_ctc_head"] = str(os.environ["AR_FREEZE_CTC_HEAD"]).lower() not in {"", "0", "false", "no"}
cfg["direction_variant"] = "none"
cfg["p_start"] = 0.0
cfg["p_max"] = 0.0
cfg["warmup_steps"] = 0
cfg["ramp_steps"] = 0
cfg["lr"] = float(os.environ["AR_LR"])
cfg["batch_size"] = int(os.environ["AR_BATCH_SIZE"])
cfg["max_steps"] = int(os.environ["AR_MAX_STEPS"])
cfg["epochs"] = None
cfg["save_every"] = int(os.environ["AR_SAVE_EVERY"])
cfg["step_eval_every"] = None
cfg["step_eval_samples"] = 0
cfg["max_eval_samples"] = 2048
cfg["wandb_run_name"] = run_dir.name
deepspeed = dict(cfg.get("deepspeed") or {})
deepspeed["train_micro_batch_size_per_gpu"] = int(os.environ["AR_BATCH_SIZE"])
deepspeed["gradient_accumulation_steps"] = int(os.environ["AR_GRAD_ACCUM"])
deepspeed.setdefault("zero_optimization", {"stage": 1, "offload_optimizer": {"device": "none"}})
deepspeed.setdefault("bf16", {"enabled": True})
cfg["deepspeed"] = deepspeed
save_yaml(os.environ["AR_CONFIG"], cfg)
print(f"saved_ar_config={os.environ['AR_CONFIG']} resume_from={cfg['resume_from']} resume_tag={cfg['resume_tag']}")
PY
  "${PYTHON_BIN}" - <<'PY'
import math
import os
import sys

from rwkvasr.config import load_yaml

cfg = dict(load_yaml(os.environ["AR_CONFIG"]))
expected = {
    "direction_variant": "none",
    "p_start": 0.0,
    "p_max": 0.0,
    "warmup_steps": 0,
    "ramp_steps": 0,
}
errors = []
for key, expected_value in expected.items():
    value = cfg.get(key)
    if isinstance(expected_value, float):
        try:
            ok = math.isclose(float(value), expected_value, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            ok = False
    else:
        ok = value == expected_value
    if not ok:
        errors.append(f"{key}={value!r} expected {expected_value!r}")
if errors:
    print("Stage11 direction dropout guard failed: " + "; ".join(errors), file=sys.stderr)
    sys.exit(1)
print(
    "Stage11 direction dropout guard passed: "
    "direction_variant=none p_start=0.0 p_max=0.0 warmup_steps=0 ramp_steps=0"
)
PY
}

start_training() {
  mkdir -p "${AR_RUN_DIR}/logs"
  if training_process_active; then
    log "matching training process is already active for config=${AR_CONFIG}; not sending a duplicate command"
    return 0
  fi
  local command
  command="export PYTHONUNBUFFERED=1; export RICH_FORCE_TERMINAL=1; export TQDM_DISABLE=0; export PYTORCH_ALLOC_CONF=\"${PYTORCH_ALLOC_CONF}\"; export NUM_GPUS=${NUM_GPUS}; export MASTER_PORT=${MASTER_PORT}; export PATH=\"${REPO_ROOT}/.venv/bin:\$PATH\"; cd \"${REPO_ROOT}\"; script -q -f -e -c \"./scripts/train_paper_rwkv_asr.sh --config-yaml ${AR_CONFIG} --num-gpus ${NUM_GPUS} --master-port ${MASTER_PORT}\" \"${AR_RUN_DIR}/logs/training.ansi.log\"; rc=\$?; echo; echo \"stage7 draft-conditioned CTC+AR training exited with status \$rc at \$(date)\""
  if ! tmux_has_session "${TRAINING_SESSION}"; then
    tmux new-session -d -s "${TRAINING_SESSION}" -n train
  fi
  log "starting training in tmux session ${TRAINING_SESSION}"
  tmux send-keys -t "${TRAINING_SESSION}:0" "${command}" C-m
}

start_sidecar() {
  if tmux_has_session "${SIDECAR_SESSION}"; then
    log "restarting sidecar session ${SIDECAR_SESSION}"
    tmux kill-session -t "${SIDECAR_SESSION}" || true
  fi
  local sidecar_command
  sidecar_command="cd \"${REPO_ROOT}\"; mkdir -p \"${AR_RUN_DIR}/logs\"; script -q -f -e -c 'RUN_DIR=\"${AR_RUN_DIR}\" TMUX_TARGET=\"${TRAINING_SESSION}:0\" POLL_SECONDS=\"${SIDECAR_POLL_SECONDS}\" CHECKPOINT_STABLE_SECONDS=90 DEVICE=\"${SIDECAR_DEVICE}\" BATCH_SIZE=\"${SIDECAR_BATCH_SIZE}\" LIMIT=\"${SIDECAR_LIMIT}\" PREVIEW_COUNT=\"${SIDECAR_LIMIT}\" BEAM_SIZE=4 TOKEN_PRUNE_TOPK=16 TEXT_NORMALIZATION=\"${SIDECAR_TEXT_NORMALIZATION}\" METRIC_NORMALIZATION=\"${SIDECAR_METRIC_NORMALIZATION}\" AR_DO_SAMPLE=1 AR_TEMPERATURE=\"${SIDECAR_AR_TEMPERATURE}\" AR_TOP_K=\"${SIDECAR_AR_TOP_K}\" AR_TOP_P=\"${SIDECAR_AR_TOP_P}\" AR_SEED=\"${SIDECAR_AR_SEED}\" AR_CTC_DRAFT_FALLBACK_MAX_CER=\"${SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_CER}\" AR_CTC_DRAFT_FALLBACK_MIN_LENGTH_RATIO=\"${SIDECAR_AR_CTC_DRAFT_FALLBACK_MIN_LENGTH_RATIO}\" AR_CTC_DRAFT_FALLBACK_MAX_LENGTH_RATIO=\"${SIDECAR_AR_CTC_DRAFT_FALLBACK_MAX_LENGTH_RATIO}\" AR_CTC_DRAFT_FALLBACK_REJECT_REPETITION=\"${SIDECAR_AR_CTC_DRAFT_FALLBACK_REJECT_REPETITION}\" AR_CTC_DRAFT_FALLBACK_METRIC_NORMALIZATION=\"${SIDECAR_AR_CTC_DRAFT_FALLBACK_METRIC_NORMALIZATION}\" SOURCE_QUOTAS=\"${SIDECAR_SOURCE_QUOTAS}\" ./scripts/watch_joint_training_sidecar.sh' \"${AR_RUN_DIR}/logs/sidecar.ansi.log\"; rc=\$?; echo; echo \"stage7 sidecar exited with status \$rc at \$(date)\""
  log "starting sidecar in tmux session ${SIDECAR_SESSION}"
  tmux new-session -d -s "${SIDECAR_SESSION}" -n sidecar "${sidecar_command}"
}

main() {
  cd "${REPO_ROOT}"
  mkdir -p "${AR_RUN_DIR}/logs"
  local lock_path="${AR_RUN_DIR}/logs/start_stage7_draft_ar.lock"
  exec 9>"${lock_path}"
  if ! flock -n 9; then
    log "another start script instance is already active for run_dir=${AR_RUN_DIR}; exiting"
    exit 0
  fi
  write_ar_config
  start_training
  start_sidecar
  log "handoff complete run_dir=${AR_RUN_DIR} config=${AR_CONFIG}"
}

main "$@"
