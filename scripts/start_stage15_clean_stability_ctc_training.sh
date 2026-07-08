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

BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage12b_sourcebalanced_clean_ctc_from_stage12a_werbest_4x4090_deepspeed.yaml}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage12b_sourcebalanced_clean_ctc_from_stage12a_werbest_bs12_lr2e6_zero1_nockpt_noaug_nodirdrop_4x4090/wercer_best.pt}"

CLEAN_ROOT="${CLEAN_ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"
CLEAN_STAGE_NAME="${CLEAN_STAGE_NAME:-stage12b_source_balanced_clean_public_anchor}"
CLEAN_STAGE_DIR="${CLEAN_STAGE_DIR:-${CLEAN_ROOT}/stages/${CLEAN_STAGE_NAME}}"
CLEAN_LENGTH_INDEX="${CLEAN_LENGTH_INDEX:-${CLEAN_STAGE_DIR}/webdataset_lengths.jsonl}"
CLEAN_BUCKET_MANIFEST="${CLEAN_BUCKET_MANIFEST:-${CLEAN_STAGE_DIR}/webdataset_buckets_audio_text/manifest.json}"

RUN_DIR="${RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_bs12_lr1e6_zero1_nockpt_noaug_nodirdrop_4x4090}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_4x4090_deepspeed.yaml}"
MAX_STEPS="${MAX_STEPS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
TOKEN_BUDGET="${TOKEN_BUDGET:-12000}"
LR="${LR:-1.0e-6}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
SPEC_AUGMENT_ENABLED="${SPEC_AUGMENT_ENABLED:-0}"
SPECAUGMENT_TIME_MASKS="${SPECAUGMENT_TIME_MASKS:-2}"
SPECAUGMENT_TIME_WIDTH="${SPECAUGMENT_TIME_WIDTH:-20}"
SPECAUGMENT_FREQ_MASKS="${SPECAUGMENT_FREQ_MASKS:-2}"
SPECAUGMENT_FREQ_WIDTH="${SPECAUGMENT_FREQ_WIDTH:-27}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29615}"
TRAINING_SESSION="${TRAINING_SESSION:-training}"
SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage15_clean_ctc}"
SIDECAR_LIMIT="${SIDECAR_LIMIT:-64}"
SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:16,clean_aishell:aishell3_*.tar:16,clean_cv_en:commonvoice_en_*.tar:16,clean_cv_cn:commonvoice_cn_*.tar:16}"
SIDECAR_WEBDATASET_SPLIT="${SIDECAR_WEBDATASET_SPLIT:-train}"
SIDECAR_DEVICE="${SIDECAR_DEVICE:-cpu}"
SIDECAR_BATCH_SIZE="${SIDECAR_BATCH_SIZE:-1}"
SIDECAR_POLL_SECONDS="${SIDECAR_POLL_SECONDS:-300}"
SIDECAR_TEXT_NORMALIZATION="${SIDECAR_TEXT_NORMALIZATION:-ctc}"
SIDECAR_METRIC_NORMALIZATION="${SIDECAR_METRIC_NORMALIZATION:-ctc}"

AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM="${RESUME_FROM:-}"
DRY_RUN="${DRY_RUN:-0}"

log() {
  printf '[stage15-clean-ctc] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

training_active() {
  pgrep -af "rwkvasr.cli.train_ctc_deepspeed.*${CONFIG_PATH}" >/dev/null 2>&1
}

validate_inputs() {
  if [[ ! -s "${BASE_CONFIG}" ]]; then
    echo "base config missing: ${BASE_CONFIG}" >&2
    exit 1
  fi
  if [[ ! -s "${INIT_CHECKPOINT_PATH}" ]]; then
    echo "init checkpoint missing: ${INIT_CHECKPOINT_PATH}" >&2
    exit 1
  fi
  if [[ ! -s "${CLEAN_LENGTH_INDEX}" ]]; then
    echo "clean length index missing: ${CLEAN_LENGTH_INDEX}" >&2
    exit 1
  fi
  if [[ ! -s "${CLEAN_BUCKET_MANIFEST}" ]]; then
    echo "clean bucket manifest missing: ${CLEAN_BUCKET_MANIFEST}" >&2
    exit 1
  fi
}

write_config() {
  export BASE_CONFIG INIT_CHECKPOINT_PATH CLEAN_ROOT CLEAN_LENGTH_INDEX CLEAN_BUCKET_MANIFEST
  export RUN_DIR CONFIG_PATH MAX_STEPS BATCH_SIZE GRAD_ACCUM TOKEN_BUDGET LR SAVE_EVERY LOG_EVERY
  export SPEC_AUGMENT_ENABLED SPECAUGMENT_TIME_MASKS SPECAUGMENT_TIME_WIDTH SPECAUGMENT_FREQ_MASKS SPECAUGMENT_FREQ_WIDTH
  export AUTO_RESUME RESUME_FROM
  "${PYTHON_BIN}" - <<'PY'
import math
import os
from pathlib import Path

from rwkvasr.config import load_yaml, save_yaml

cfg = dict(load_yaml(os.environ["BASE_CONFIG"]))
run_dir = Path(os.environ["RUN_DIR"])
clean_root = Path(os.environ["CLEAN_ROOT"])
explicit_resume = os.environ.get("RESUME_FROM", "").strip()
auto_resume = os.environ.get("AUTO_RESUME", "1").strip().lower() in {"1", "true", "yes", "on"}
has_ds_checkpoint = (run_dir / "latest_checkpoint.yaml").exists() and any((run_dir / "ds_checkpoints").glob("step-*"))

cfg["output_dir"] = str(run_dir)
cfg["webdataset_root"] = str(clean_root)
cfg["webdataset_index_path"] = str(clean_root / "webdataset_index.json")
cfg["webdataset_length_index_path"] = os.environ["CLEAN_LENGTH_INDEX"]
cfg["webdataset_bucket_manifest_path"] = os.environ["CLEAN_BUCKET_MANIFEST"]
cfg["webdataset_split"] = "train"
cfg["webdataset_eval_ratio"] = 0.01
cfg["webdataset_hash_seed"] = 0
cfg["webdataset_split_by"] = "shard_name"
cfg["webdataset_utt_id_key"] = "id"
cfg["decoder_enabled"] = False
cfg["decoder_loss_weight"] = 0.0
cfg["ctc_loss_weight"] = 1.0
cfg["decoder_ctc_draft_cache_path"] = None
cfg["decoder_ctc_draft_missing_policy"] = None
cfg["text_normalization"] = "ctc"
cfg["tokenizer_append_eos"] = False
cfg["init_checkpoint_path"] = os.environ["INIT_CHECKPOINT_PATH"]
if explicit_resume:
    cfg["resume_from"] = explicit_resume
elif auto_resume and has_ds_checkpoint:
    cfg["resume_from"] = "latest"
else:
    cfg["resume_from"] = None
cfg["resume_tag"] = None
cfg["wandb_run_name"] = run_dir.name
cfg["batch_size"] = int(os.environ["BATCH_SIZE"])
cfg["batch_token_budget"] = int(os.environ["TOKEN_BUDGET"])
cfg["length_bucket_frame_budget"] = int(os.environ["TOKEN_BUDGET"])
cfg["target_gpu_memory_gib"] = 23.0
cfg["max_steps"] = int(os.environ["MAX_STEPS"])
cfg["epochs"] = None
cfg["save_every"] = int(os.environ["SAVE_EVERY"])
cfg["log_every"] = int(os.environ["LOG_EVERY"])
cfg["step_eval_every"] = None
cfg["step_eval_samples"] = 0
cfg["max_eval_samples"] = 2048
cfg["eval_batch_size"] = 1
cfg["step_eval_batch_size"] = 1
cfg["gradient_checkpointing"] = False
cfg["lr"] = float(os.environ["LR"])
cfg["bucket_source_interleave"] = True
cfg["direction_variant"] = "none"
cfg["p_start"] = 0.0
cfg["p_max"] = 0.0
cfg["warmup_steps"] = 0
cfg["ramp_steps"] = 0
enabled_value = str(os.environ.get("SPEC_AUGMENT_ENABLED", "0")).strip().lower()
cfg["specaugment_enabled"] = enabled_value in {"1", "true", "yes", "on"}
cfg["specaugment_time_masks"] = int(os.environ["SPECAUGMENT_TIME_MASKS"])
cfg["specaugment_time_width"] = int(os.environ["SPECAUGMENT_TIME_WIDTH"])
cfg["specaugment_freq_masks"] = int(os.environ["SPECAUGMENT_FREQ_MASKS"])
cfg["specaugment_freq_width"] = int(os.environ["SPECAUGMENT_FREQ_WIDTH"])
deepspeed = dict(cfg.get("deepspeed") or {})
deepspeed["train_micro_batch_size_per_gpu"] = int(os.environ["BATCH_SIZE"])
deepspeed["gradient_accumulation_steps"] = int(os.environ["GRAD_ACCUM"])
deepspeed["gradient_clipping"] = 1.0
zero = dict(deepspeed.get("zero_optimization") or {})
zero["stage"] = 1
zero["offload_optimizer"] = {"device": "none"}
deepspeed["zero_optimization"] = zero
deepspeed["bf16"] = {"enabled": True}
cfg["deepspeed"] = deepspeed

run_dir.mkdir(parents=True, exist_ok=True)
Path(os.environ["CONFIG_PATH"]).parent.mkdir(parents=True, exist_ok=True)
save_yaml(os.environ["CONFIG_PATH"], cfg)

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
    raise SystemExit("Stage15 direction dropout guard failed: " + "; ".join(errors))
print(f"saved_config={os.environ['CONFIG_PATH']} run_dir={run_dir} resume_from={cfg.get('resume_from')}")
print("Stage15 direction dropout guard passed: direction_variant=none p_start=0.0 p_max=0.0 warmup_steps=0 ramp_steps=0")
PY
}

start_training() {
  mkdir -p "${RUN_DIR}/logs"
  if training_active; then
    log "matching training process already active for ${CONFIG_PATH}; not starting another"
    return
  fi
  local command
  command="export PYTHONUNBUFFERED=1; export RICH_FORCE_TERMINAL=1; export TQDM_DISABLE=0; export NUM_GPUS=${NUM_GPUS}; export MASTER_PORT=${MASTER_PORT}; export PATH=\"${REPO_ROOT}/.venv/bin:\$PATH\"; cd \"${REPO_ROOT}\"; script -q -f -e -c \"./scripts/train_paper_rwkv_asr.sh --config-yaml ${CONFIG_PATH} --num-gpus ${NUM_GPUS} --master-port ${MASTER_PORT}\" \"${RUN_DIR}/logs/training.ansi.log\"; rc=\$?; echo; echo \"stage15 clean CTC training exited with status \$rc at \$(date)\""
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
  sidecar_command="cd \"${REPO_ROOT}\"; mkdir -p \"${RUN_DIR}/logs\"; script -q -f -e -c 'RUN_DIR=\"${RUN_DIR}\" TMUX_TARGET=\"${TRAINING_SESSION}:0\" POLL_SECONDS=\"${SIDECAR_POLL_SECONDS}\" CHECKPOINT_STABLE_SECONDS=120 DEVICE=\"${SIDECAR_DEVICE}\" BATCH_SIZE=\"${SIDECAR_BATCH_SIZE}\" LIMIT=\"${SIDECAR_LIMIT}\" PREVIEW_COUNT=\"${SIDECAR_LIMIT}\" BEAM_SIZE=4 TOKEN_PRUNE_TOPK=16 TEXT_NORMALIZATION=\"${SIDECAR_TEXT_NORMALIZATION}\" METRIC_NORMALIZATION=\"${SIDECAR_METRIC_NORMALIZATION}\" SIDECAR_WEBDATASET_SPLIT=\"${SIDECAR_WEBDATASET_SPLIT}\" SOURCE_QUOTAS=\"${SIDECAR_SOURCE_QUOTAS}\" ./scripts/watch_joint_training_sidecar.sh' \"${RUN_DIR}/logs/sidecar.ansi.log\"; rc=\$?; echo; echo \"stage15 sidecar exited with status \$rc at \$(date)\""
  log "starting sidecar in tmux session ${SIDECAR_SESSION}"
  tmux new-session -d -s "${SIDECAR_SESSION}" -n sidecar "${sidecar_command}"
}

main() {
  cd "${REPO_ROOT}"
  validate_inputs
  write_config
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry run complete run_dir=${RUN_DIR} config=${CONFIG_PATH}"
    return
  fi
  start_training
  start_sidecar
  log "handoff complete run_dir=${RUN_DIR} config=${CONFIG_PATH}"
}

main "$@"
