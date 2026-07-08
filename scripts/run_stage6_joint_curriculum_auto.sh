#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${REPO_ROOT}/.venv/bin${PATH:+:${PATH}}"

CURRENT_STAGE_NAME="${CURRENT_STAGE_NAME:-stage6a_chatlang_clean_replay}"
CURRENT_CONFIG="${CURRENT_CONFIG:-${REPO_ROOT}/configs/sensevoice_rwkv_stage6_joint_chatlang_from_stage5werbest_4x4090_deepspeed.yaml}"
CURRENT_RUN_DIR="${CURRENT_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage6_joint_chatlang_from_stage5werbest_ctc0p8_ar0p2_bs4_lr2e5_zero1_nockpt_4x4090}"
BASE_CONFIG="${BASE_CONFIG:-${CURRENT_CONFIG}}"

CURRICULUM_ROOT="${CURRICULUM_ROOT:-/media/usbhd/training_data/asr/curriculum/stage6_joint_hard_mix}"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${REPO_ROOT}/configs/generated}"
RUNS_ROOT="${RUNS_ROOT:-/media/usbhd/rwkvasr_runs}"
TRAINING_SESSION="${TRAINING_SESSION:-training}"
SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage6}"
POLL_SECONDS="${POLL_SECONDS:-300}"
SIDECAR_POLL_SECONDS="${SIDECAR_POLL_SECONDS:-300}"
SIDECAR_STABLE_SECONDS="${SIDECAR_STABLE_SECONDS:-60}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29565}"
TEXT_NORMALIZATION="${TEXT_NORMALIZATION:-runtime}"
AR_MAX_NEW_TOKENS_FACTOR="${AR_MAX_NEW_TOKENS_FACTOR:-0.6}"
AR_TEMPERATURE="${AR_TEMPERATURE:-0.8}"
AR_TOP_K="${AR_TOP_K:-5}"
AR_TOP_P="${AR_TOP_P:-0.8}"

mkdir -p "${GENERATED_CONFIG_DIR}"

log() {
  printf '[stage6-auto] %s %s\n' "$(date '+%F %T')" "$*" >&2
}

ensure_tmux_session() {
  local session="$1"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    tmux new-session -d -s "${session}" -c "${REPO_ROOT}"
  fi
}

active_training_for_config() {
  local config="$1"
  local base
  base="$(basename "${config}")"
  pgrep -af "rwkvasr.cli.train_ctc_deepspeed.*${base}" >/dev/null 2>&1
}

latest_step_checkpoint() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^step-\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

config_max_steps() {
  local run_dir="$1"
  "${PYTHON_BIN}" - "$run_dir" <<'PY'
import sys
from pathlib import Path
from rwkvasr.config import load_yaml

run_dir = Path(sys.argv[1])
for name in ("train_config.yaml",):
    path = run_dir / name
    if path.exists():
        data = load_yaml(path)
        value = data.get("max_steps") if isinstance(data, dict) else None
        if value is not None:
            print(int(value))
            raise SystemExit(0)
raise SystemExit(1)
PY
}

wait_for_stage_complete() {
  local stage_name="$1"
  local config="$2"
  local run_dir="$3"
  log "monitoring ${stage_name} run=${run_dir}"
  while active_training_for_config "${config}"; do
    local latest=""
    latest="$(latest_step_checkpoint "${run_dir}" || true)"
    if [[ -n "${latest}" ]]; then
      log "${stage_name} still running latest_step_checkpoint=${latest}"
    else
      log "${stage_name} still running no_step_checkpoint_yet"
    fi
    sleep "${POLL_SECONDS}"
  done

  if [[ ! -d "${run_dir}" ]]; then
    log "${stage_name} run dir missing after training stopped: ${run_dir}"
    return 1
  fi
  local max_steps latest
  if ! max_steps="$(config_max_steps "${run_dir}")"; then
    log "${stage_name} missing train_config.yaml/max_steps; refusing to advance"
    return 1
  fi
  latest="$(latest_step_checkpoint "${run_dir}" || true)"
  if [[ -z "${latest}" || "${latest}" -lt "${max_steps}" ]]; then
    log "${stage_name} did not finish cleanly latest=${latest:-none} max_steps=${max_steps}; refusing to advance"
    return 1
  fi
  log "${stage_name} complete latest=${latest} max_steps=${max_steps}"
  return 0
}

stop_sidecar() {
  ensure_tmux_session "${SIDECAR_SESSION}"
  tmux send-keys -t "${SIDECAR_SESSION}:0" C-c >/dev/null 2>&1 || true
  sleep 2
}

start_sidecar() {
  local run_dir="$1"
  local source_quotas="${2:-}"
  ensure_tmux_session "${SIDECAR_SESSION}"
  stop_sidecar
  local command
  command="export RUN_DIR=\"${run_dir}\"; export TMUX_TARGET=\"${TRAINING_SESSION}:0\"; export DEVICE=\"cuda:0\"; export LIMIT=64; export PREVIEW_COUNT=32; export BATCH_SIZE=1; export POLL_SECONDS=\"${SIDECAR_POLL_SECONDS}\"; export CHECKPOINT_STABLE_SECONDS=\"${SIDECAR_STABLE_SECONDS}\"; export BEAM_SIZE=4; export TOKEN_PRUNE_TOPK=16; export TEXT_NORMALIZATION=\"${TEXT_NORMALIZATION}\"; export AR_DO_SAMPLE=1; export AR_TEMPERATURE=\"${AR_TEMPERATURE}\"; export AR_TOP_K=\"${AR_TOP_K}\"; export AR_TOP_P=\"${AR_TOP_P}\"; export AR_MAX_NEW_TOKENS_FACTOR=\"${AR_MAX_NEW_TOKENS_FACTOR}\"; export PATH=\"${REPO_ROOT}/.venv/bin:\$PATH\";"
  if [[ -n "${source_quotas}" ]]; then
    command+=" export SOURCE_QUOTAS=\"${source_quotas}\";"
  else
    command+=" unset SOURCE_QUOTAS;"
  fi
  command+=" cd \"${REPO_ROOT}\"; ./scripts/watch_joint_training_sidecar.sh; rc=\$?; echo; echo \"sidecar exited with status \$rc at \$(date)\""
  tmux send-keys -t "${SIDECAR_SESSION}:0" "${command}" C-m
  log "sidecar started run=${run_dir}"
}

run_sidecar_once_for_final() {
  local run_dir="$1"
  local source_quotas="${2:-}"
  stop_sidecar
  log "waiting for final checkpoint stability run=${run_dir}"
  sleep "${SIDECAR_STABLE_SECONDS}"
  local env_args=(
    "RUN_DIR=${run_dir}"
    "TMUX_TARGET=${TRAINING_SESSION}:0"
    "DEVICE=cuda:0"
    "LIMIT=64"
    "PREVIEW_COUNT=32"
    "BATCH_SIZE=1"
    "POLL_SECONDS=1"
    "CHECKPOINT_STABLE_SECONDS=1"
    "BEAM_SIZE=4"
    "TOKEN_PRUNE_TOPK=16"
    "TEXT_NORMALIZATION=${TEXT_NORMALIZATION}"
    "AR_DO_SAMPLE=1"
    "AR_TEMPERATURE=${AR_TEMPERATURE}"
    "AR_TOP_K=${AR_TOP_K}"
    "AR_TOP_P=${AR_TOP_P}"
    "AR_MAX_NEW_TOKENS_FACTOR=${AR_MAX_NEW_TOKENS_FACTOR}"
  )
  if [[ -n "${source_quotas}" ]]; then
    env_args+=("SOURCE_QUOTAS=${source_quotas}")
  fi
  env "${env_args[@]}" "${REPO_ROOT}/scripts/watch_joint_training_sidecar.sh" --once || {
    log "final sidecar once failed for ${run_dir}"
    return 1
  }
}

select_best_checkpoint() {
  local run_dir="$1"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/select_joint_checkpoint_from_sidecar.py" \
    --run-dir "${run_dir}" \
    --metric wer \
    --print-path-only
}

build_hard_mix_stages() {
  local required="${CURRICULUM_ROOT}/stages/stage6b_clean70_hard30/webdataset_buckets_audio_text/manifest.json"
  if [[ -s "${required}" ]]; then
    log "hard-mix curriculum already prepared root=${CURRICULUM_ROOT}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage6_joint_hard_mix.py" \
      --output-root "${CURRICULUM_ROOT}" \
      --skip-buckets
    return 0
  fi
  log "building hard-mix curriculum root=${CURRICULUM_ROOT}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_stage6_joint_hard_mix.py" \
    --output-root "${CURRICULUM_ROOT}"
}

write_stage_config() {
  local stage_name="$1"
  local clean_pct="$2"
  local hard_pct="$3"
  local init_checkpoint="$4"
  local output_config="${GENERATED_CONFIG_DIR}/sensevoice_rwkv_${stage_name}_joint_chatlang_hardmix_4x4090_deepspeed.yaml"
  local run_dir="${RUNS_ROOT}/sensevoice_rwkv_${stage_name}_joint_chatlang_gigaspeech_wenetspeech_clean${clean_pct}_hard${hard_pct}_from_prev_ctc0p8_ar0p2_bs4_lr2e5_zero1_nockpt_4x4090"
  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${output_config}" "${run_dir}" "${CURRICULUM_ROOT}" "${stage_name}" "${init_checkpoint}" <<'PY'
import sys
from pathlib import Path
from rwkvasr.config import load_yaml, save_yaml

base_config, output_config, run_dir, root, stage_name, init_checkpoint = map(Path, sys.argv[1:])
data = load_yaml(base_config)
stage_dir = root / "stages" / stage_name
data.update(
    {
        "output_dir": str(run_dir),
        "webdataset_root": str(root),
        "webdataset_index_path": str(root / "webdataset_index.json"),
        "webdataset_length_index_path": str(stage_dir / "webdataset_lengths.jsonl"),
        "webdataset_bucket_manifest_path": str(stage_dir / "webdataset_buckets_audio_text" / "manifest.json"),
        "init_checkpoint_path": str(init_checkpoint),
        "resume_from": None,
        "wandb_run_name": run_dir.name,
    }
)
save_yaml(output_config, data)
print(output_config)
PY
}

start_training_stage() {
  local stage_name="$1"
  local config="$2"
  local run_dir="$3"
  local port="$4"
  ensure_tmux_session "${TRAINING_SESSION}"
  if [[ -d "${run_dir}" && -n "$(latest_step_checkpoint "${run_dir}" || true)" ]]; then
    log "run dir already has checkpoints; refusing to overwrite run=${run_dir}"
    return 1
  fi
  mkdir -p "${run_dir}/logs"
  local command
  command="export PYTHONUNBUFFERED=1; export RICH_FORCE_TERMINAL=1; export TQDM_DISABLE=0; export NUM_GPUS=${NUM_GPUS}; export MASTER_PORT=${port}; export PATH=\"${REPO_ROOT}/.venv/bin:\$PATH\"; cd \"${REPO_ROOT}\"; mkdir -p \"${run_dir}/logs\"; script -q -f -e -c \"./scripts/train_paper_rwkv_asr.sh --config-yaml ${config} --num-gpus ${NUM_GPUS} --master-port ${port}\" \"${run_dir}/logs/training.ansi.log\"; rc=\$?; echo; echo \"auto ${stage_name} training exited with status \$rc at \$(date)\""
  tmux send-keys -t "${TRAINING_SESSION}:0" "${command}" C-m
  log "training started stage=${stage_name} config=${config} run=${run_dir} port=${port}"
}

hard_source_quotas() {
  echo "clean_librispeech:librispeech_*.tar:8,clean_aishell:aishell3_*.tar:8,clean_cv_en:commonvoice_en_*.tar:8,clean_cv_cn:commonvoice_cn_*.tar:8,gigaspeech:GSXL-*.tar:16,wenetspeech:WSL-*.tar:16"
}

main() {
  cd "${REPO_ROOT}"
  ensure_tmux_session "${TRAINING_SESSION}"
  ensure_tmux_session "${SIDECAR_SESSION}"

  local previous_stage="${CURRENT_STAGE_NAME}"
  local previous_config="${CURRENT_CONFIG}"
  local previous_run="${CURRENT_RUN_DIR}"
  local previous_quotas=""
  local source_quotas
  source_quotas="$(hard_source_quotas)"
  case "${previous_stage}" in
    stage6b_*|stage6c_*|stage6d_*)
      previous_quotas="${source_quotas}"
      ;;
  esac

  wait_for_stage_complete "${previous_stage}" "${previous_config}" "${previous_run}"
  run_sidecar_once_for_final "${previous_run}" "${previous_quotas}"
  local best_checkpoint
  best_checkpoint="$(select_best_checkpoint "${previous_run}")"
  log "selected checkpoint for next stage: ${best_checkpoint}"

  build_hard_mix_stages

  local specs=(
    "stage6b_clean70_hard30:70:30:29565"
    "stage6c_clean50_hard50:50:50:29566"
    "stage6d_clean30_hard70:30:70:29567"
  )
  local final_run="${previous_run}"
  local skip_until_current=0
  case "${previous_stage}" in
    stage6b_*|stage6c_*|stage6d_*)
      skip_until_current=1
      ;;
  esac
  for spec in "${specs[@]}"; do
    IFS=: read -r stage_name clean_pct hard_pct port <<< "${spec}"
    if [[ "${skip_until_current}" == "1" ]]; then
      if [[ "${stage_name}" == "${previous_stage}" ]]; then
        skip_until_current=0
      fi
      continue
    fi
    local config
    config="$(write_stage_config "${stage_name}" "${clean_pct}" "${hard_pct}" "${best_checkpoint}")"
    local run_dir
    run_dir="$("${PYTHON_BIN}" - "${config}" <<'PY'
import sys
from rwkvasr.config import load_yaml
print(load_yaml(sys.argv[1])["output_dir"])
PY
)"
    start_training_stage "${stage_name}" "${config}" "${run_dir}" "${port}"
    start_sidecar "${run_dir}" "${source_quotas}"
    wait_for_stage_complete "${stage_name}" "${config}" "${run_dir}"
    run_sidecar_once_for_final "${run_dir}" "${source_quotas}"
    best_checkpoint="$(select_best_checkpoint "${run_dir}")"
    final_run="${run_dir}"
    log "selected checkpoint for next stage: ${best_checkpoint}"
  done

  start_sidecar "${final_run}" "${source_quotas}"
  log "stage6 hard-mix curriculum finished. final_best_checkpoint=${best_checkpoint}"
}

main "$@"
