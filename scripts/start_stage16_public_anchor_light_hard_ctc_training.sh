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

BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_4x4090_deepspeed.yaml}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_bs12_lr1e6_zero1_nockpt_noaug_nodirdrop_4x4090/wercer_best.pt}"

CLEAN_ROOT="${CLEAN_ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"
CLEAN_LENGTH_INDEX="${CLEAN_LENGTH_INDEX:-${CLEAN_ROOT}/stages/stage12b_source_balanced_clean_public_anchor/webdataset_lengths.jsonl}"
HARD_ROOT="${HARD_ROOT:-/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset}"
HARD_LENGTH_INDEX="${HARD_LENGTH_INDEX:-${HARD_ROOT}/webdataset_lengths.jsonl}"

CURRICULUM_ROOT="${CURRICULUM_ROOT:-/media/usbhd/training_data/asr/curriculum/stage16_public_anchor_light_hard_mix}"
STAGE_NAME="${STAGE_NAME:-stage16_public80_hard20}"
TARGET_SAMPLES="${TARGET_SAMPLES:-600000}"
CLEAN_RATIO="${CLEAN_RATIO:-0.80}"
MIX_EVAL_RATIO="${MIX_EVAL_RATIO:-0.001}"
MIX_SEED="${MIX_SEED:-20260619}"
BUCKET_WIDTH="${BUCKET_WIDTH:-80}"
REBUILD_CURRICULUM="${REBUILD_CURRICULUM:-0}"

RUN_DIR="${RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage16_public80_hard20_ctc_from_stage15_werbest_bs12_lr5e7_zero1_nockpt_noaug_nodirdrop_4x4090}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage16_public80_hard20_ctc_from_stage15_werbest_4x4090_deepspeed.yaml}"
MAX_STEPS="${MAX_STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
TOKEN_BUDGET="${TOKEN_BUDGET:-12000}"
LR="${LR:-5.0e-7}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
FREEZE_ENCODER="${FREEZE_ENCODER:-0}"
FREEZE_CTC_HEAD="${FREEZE_CTC_HEAD:-0}"
SPEC_AUGMENT_ENABLED="${SPEC_AUGMENT_ENABLED:-0}"
SPECAUGMENT_TIME_MASKS="${SPECAUGMENT_TIME_MASKS:-2}"
SPECAUGMENT_TIME_WIDTH="${SPECAUGMENT_TIME_WIDTH:-20}"
SPECAUGMENT_FREQ_MASKS="${SPECAUGMENT_FREQ_MASKS:-2}"
SPECAUGMENT_FREQ_WIDTH="${SPECAUGMENT_FREQ_WIDTH:-27}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29616}"
TRAINING_SESSION="${TRAINING_SESSION:-training}"
SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage16_public_anchor_light_hard_ctc}"
SIDECAR_LIMIT="${SIDECAR_LIMIT:-80}"
SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:12,clean_aishell:aishell3_*.tar:12,clean_cv_en:commonvoice_en_*.tar:12,clean_cv_cn:commonvoice_cn_*.tar:12,gigaspeech:GSXL-*.tar:16,wenetspeech:WSL-*.tar:16}"
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
  printf '[stage16-public-anchor-light-hard-ctc] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

training_active() {
  pgrep -af "rwkvasr.cli.train_ctc_deepspeed.*${CONFIG_PATH}" >/dev/null 2>&1
}

build_curriculum() {
  if [[ ! -s "${CLEAN_LENGTH_INDEX}" ]]; then
    echo "clean length index missing: ${CLEAN_LENGTH_INDEX}" >&2
    exit 1
  fi
  if [[ ! -s "${HARD_LENGTH_INDEX}" ]]; then
    echo "hard length index missing: ${HARD_LENGTH_INDEX}" >&2
    exit 1
  fi
  local stage_dir="${CURRICULUM_ROOT}/stages/${STAGE_NAME}"
  local length_index="${stage_dir}/webdataset_lengths.jsonl"
  local bucket_manifest="${stage_dir}/webdataset_buckets_audio_text/manifest.json"
  if [[ "${REBUILD_CURRICULUM}" != "1" && -s "${length_index}" && -s "${bucket_manifest}" ]]; then
    log "reusing existing curriculum stage=${STAGE_NAME} length_index=${length_index} bucket_manifest=${bucket_manifest}"
    return
  fi
  log "building/reusing curriculum root=${CURRICULUM_ROOT} stage=${STAGE_NAME}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage6_joint_hard_mix.py" \
    --output-root "${CURRICULUM_ROOT}" \
    --clean-root "${CLEAN_ROOT}" \
    --clean-length-index-path "${CLEAN_LENGTH_INDEX}" \
    --hard-root "${HARD_ROOT}" \
    --hard-length-index-path "${HARD_LENGTH_INDEX}" \
    --target-samples "${TARGET_SAMPLES}" \
    --eval-ratio "${MIX_EVAL_RATIO}" \
    --seed "${MIX_SEED}" \
    --bucket-width "${BUCKET_WIDTH}" \
    --stage "${STAGE_NAME}:${CLEAN_RATIO}" \
    --clean-with-replacement
}

write_config() {
  local stage_dir="${CURRICULUM_ROOT}/stages/${STAGE_NAME}"
  local length_index="${stage_dir}/webdataset_lengths.jsonl"
  local bucket_manifest="${stage_dir}/webdataset_buckets_audio_text/manifest.json"
  if [[ ! -s "${BASE_CONFIG}" ]]; then
    echo "base config missing: ${BASE_CONFIG}" >&2
    exit 1
  fi
  if [[ ! -s "${INIT_CHECKPOINT_PATH}" ]]; then
    echo "init checkpoint missing: ${INIT_CHECKPOINT_PATH}" >&2
    exit 1
  fi
  if [[ ! -s "${length_index}" ]]; then
    echo "stage length index missing: ${length_index}" >&2
    exit 1
  fi
  if [[ ! -s "${bucket_manifest}" ]]; then
    echo "stage bucket manifest missing: ${bucket_manifest}" >&2
    exit 1
  fi

  export BASE_CONFIG INIT_CHECKPOINT_PATH CURRICULUM_ROOT STAGE_NAME RUN_DIR CONFIG_PATH
  export MAX_STEPS BATCH_SIZE GRAD_ACCUM TOKEN_BUDGET LR SAVE_EVERY LOG_EVERY MIX_EVAL_RATIO
  export FREEZE_ENCODER FREEZE_CTC_HEAD
  export SPEC_AUGMENT_ENABLED SPECAUGMENT_TIME_MASKS SPECAUGMENT_TIME_WIDTH SPECAUGMENT_FREQ_MASKS SPECAUGMENT_FREQ_WIDTH
  export AUTO_RESUME RESUME_FROM
  "${PYTHON_BIN}" - <<'PY'
import math
import os
from pathlib import Path

from rwkvasr.config import load_yaml, save_yaml

cfg = dict(load_yaml(os.environ["BASE_CONFIG"]))
run_dir = Path(os.environ["RUN_DIR"])
root = Path(os.environ["CURRICULUM_ROOT"])
stage_dir = root / "stages" / os.environ["STAGE_NAME"]
explicit_resume = os.environ.get("RESUME_FROM", "").strip()
auto_resume = os.environ.get("AUTO_RESUME", "1").strip().lower() in {"1", "true", "yes", "on"}
has_ds_checkpoint = (run_dir / "latest_checkpoint.yaml").exists() and any((run_dir / "ds_checkpoints").glob("step-*"))

cfg["output_dir"] = str(run_dir)
cfg["webdataset_root"] = str(root)
cfg["webdataset_index_path"] = str(root / "webdataset_index.json")
cfg["webdataset_length_index_path"] = str(stage_dir / "webdataset_lengths.jsonl")
cfg["webdataset_bucket_manifest_path"] = str(stage_dir / "webdataset_buckets_audio_text" / "manifest.json")
cfg["webdataset_split"] = "train"
cfg["webdataset_eval_ratio"] = float(os.environ["MIX_EVAL_RATIO"])
cfg["webdataset_hash_seed"] = 0
cfg["webdataset_split_by"] = "sample_id"
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
cfg["freeze_encoder"] = str(os.environ.get("FREEZE_ENCODER", "0")).strip().lower() in {"1", "true", "yes", "on"}
cfg["freeze_ctc_head"] = str(os.environ.get("FREEZE_CTC_HEAD", "0")).strip().lower() in {"1", "true", "yes", "on"}
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
    raise SystemExit("Stage16 direction dropout guard failed: " + "; ".join(errors))
print(f"saved_config={os.environ['CONFIG_PATH']} run_dir={run_dir} resume_from={cfg.get('resume_from')}")
print("Stage16 direction dropout guard passed: direction_variant=none p_start=0.0 p_max=0.0 warmup_steps=0 ramp_steps=0")
PY
}

start_training() {
  mkdir -p "${RUN_DIR}/logs"
  if training_active; then
    log "matching training process already active for ${CONFIG_PATH}; not starting another"
    return
  fi
  local command
  command="export PYTHONUNBUFFERED=1; export RICH_FORCE_TERMINAL=1; export TQDM_DISABLE=0; export NUM_GPUS=${NUM_GPUS}; export MASTER_PORT=${MASTER_PORT}; export PATH=\"${REPO_ROOT}/.venv/bin:\$PATH\"; cd \"${REPO_ROOT}\"; script -q -f -e -c \"./scripts/train_paper_rwkv_asr.sh --config-yaml ${CONFIG_PATH} --num-gpus ${NUM_GPUS} --master-port ${MASTER_PORT}\" \"${RUN_DIR}/logs/training.ansi.log\"; rc=\$?; echo; echo \"stage16 public-anchor light-hard CTC training exited with status \$rc at \$(date)\""
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
  sidecar_command="cd \"${REPO_ROOT}\"; mkdir -p \"${RUN_DIR}/logs\"; script -q -f -e -c 'RUN_DIR=\"${RUN_DIR}\" TMUX_TARGET=\"${TRAINING_SESSION}:0\" POLL_SECONDS=\"${SIDECAR_POLL_SECONDS}\" CHECKPOINT_STABLE_SECONDS=120 DEVICE=\"${SIDECAR_DEVICE}\" BATCH_SIZE=\"${SIDECAR_BATCH_SIZE}\" LIMIT=\"${SIDECAR_LIMIT}\" PREVIEW_COUNT=\"${SIDECAR_LIMIT}\" BEAM_SIZE=4 TOKEN_PRUNE_TOPK=16 TEXT_NORMALIZATION=\"${SIDECAR_TEXT_NORMALIZATION}\" METRIC_NORMALIZATION=\"${SIDECAR_METRIC_NORMALIZATION}\" SIDECAR_WEBDATASET_SPLIT=\"${SIDECAR_WEBDATASET_SPLIT}\" SOURCE_QUOTAS=\"${SIDECAR_SOURCE_QUOTAS}\" ./scripts/watch_joint_training_sidecar.sh' \"${RUN_DIR}/logs/sidecar.ansi.log\"; rc=\$?; echo; echo \"stage16 sidecar exited with status \$rc at \$(date)\""
  log "starting sidecar in tmux session ${SIDECAR_SESSION}"
  tmux new-session -d -s "${SIDECAR_SESSION}" -n sidecar "${sidecar_command}"
}

main() {
  cd "${REPO_ROOT}"
  build_curriculum
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
