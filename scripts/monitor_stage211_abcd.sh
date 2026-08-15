#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUPERVISOR_SESSION="${SUPERVISOR_SESSION:-rwkvasr_stage211_abcd_strict_supervisor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/rwkvasr_runs/stage211_full_alignment}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${HOME}/rwkvasr_runs/sensevoice_rwkv_stage211a_recovery_stage210a30000_nanomlpfrozen_teacherforced_mixeronly_easy1490h_1ep_lr3e6_wd0_4x4090}"
CALIBRATION_EVAL_ROOT="${CALIBRATION_EVAL_ROOT:-${HOME}/rwkvasr_eval/stage211_calibration_selected_full}"
PHASE_GATE_ROOT="${PHASE_GATE_ROOT:-${HOME}/rwkvasr_eval/stage211_phase_gates}"
MONITOR_LOG="${MONITOR_LOG:-${OUTPUT_ROOT}/monitor_hourly.log}"
BASE_PUBLIC_OVERLAP_ROOT="${BASE_PUBLIC_OVERLAP_ROOT:-${HOME}/rwkvasr_data/stage211_base_public_pcm_overlap_v1}"
SUPPLEMENTAL_PROGRESS_REPORTER="${SUPPLEMENTAL_PROGRESS_REPORTER:-${REPO_ROOT}/scripts/report_stage211_base_public_pcm_progress.py}"
MONITOR_PYTHON="${MONITOR_PYTHON:-${REPO_ROOT}/.venv/bin/python3}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
RECENT_LOG_MINUTES="${RECENT_LOG_MINUTES:-90}"
MONITOR_ONCE="${MONITOR_ONCE:-0}"

ATTEMPT_MARKER='[rwkvasr] Distributed init complete.'
ERROR_PATTERN='traceback|out of memory|\boom\b|loss=(nan|inf)|exception|decode error|NCCL.*(error|abort)|teacher_missing=[1-9]|online_layer_missing=[1-9]|online_layer_frame_delta=[1-9]|dropped_tail=[1-9]|skipped_samples=[1-9]'

stage211_truthy() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

stage211_current_attempt_start() {
  local log_path="$1"
  awk -v marker="${ATTEMPT_MARKER}" '
    index($0, marker) { start = NR }
    END { print (start > 0 ? start : 1) }
  ' "${log_path}"
}

stage211_latest_training_record() {
  local log_path="$1"
  rg '\[deepspeed-train\] step=' "${log_path}" 2>/dev/null | tail -n 1 || true
}

stage211_yaml_scalar() {
  local config_path="$1"
  local key="$2"
  awk -v key="${key}" '
    index($0, key ":") == 1 {
      value = substr($0, length(key) + 2)
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      if (value ~ /^".*"$/ || value ~ /^\047.*\047$/) {
        value = substr(value, 2, length(value) - 2)
      }
      print value
      exit
    }
  ' "${config_path}"
}

stage211_latest_export_step() {
  local run_dir="$1"
  local checkpoint name step latest=0
  shopt -s nullglob
  for checkpoint in "${run_dir}"/step-*.pt; do
    name="${checkpoint##*/}"
    if [[ "${name}" =~ ^step-([0-9]+)\.pt$ ]]; then
      step="${BASH_REMATCH[1]}"
      if ((10#${step} > latest)); then
        latest=$((10#${step}))
      fi
    fi
  done
  shopt -u nullglob
  printf '%s\n' "${latest}"
}

stage211_active_config_paths() {
  ps -C python3 -o args= 2>/dev/null |
    awk '
      /rwkvasr\.cli\.train_ctc_deepspeed/ && /stage211/ {
        for (field_index = 1; field_index <= NF; ++field_index) {
          if ($field_index == "--config-yaml" && field_index < NF) {
            print $(field_index + 1)
          }
        }
      }
    ' |
    sort -u || true
}

stage211_emit_config_progress() {
  local config_path="$1"
  local run_dir target_step run_name latest_log latest_record live_step persisted_step progress
  if [[ ! -s "${config_path}" ]]; then
    printf 'active_config_missing=%s\n' "${config_path}"
    return
  fi
  run_dir="$(stage211_yaml_scalar "${config_path}" output_dir)"
  target_step="$(stage211_yaml_scalar "${config_path}" max_steps)"
  run_name="$(stage211_yaml_scalar "${config_path}" wandb_run_name)"
  if [[ -z "${run_dir}" || ! "${target_step}" =~ ^[0-9]+$ || "${target_step}" == 0 ]]; then
    printf 'active_config_invalid=%s output_dir=%s max_steps=%s\n' \
      "${config_path}" "${run_dir:-missing}" "${target_step:-missing}"
    return
  fi
  latest_log="$({
    find "${run_dir}/logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | head -n 1 | cut -d' ' -f2-)"
  latest_record=""
  if [[ -n "${latest_log}" ]]; then
    latest_record="$(stage211_latest_training_record "${latest_log}")"
  fi
  live_step="$(sed -n 's/.*\[deepspeed-train\] step=\([0-9][0-9]*\).*/\1/p' <<<"${latest_record}")"
  live_step="${live_step:-0}"
  persisted_step="$(stage211_latest_export_step "${run_dir}")"
  progress="$(awk -v live="${live_step}" -v target="${target_step}" 'BEGIN { printf "%.4f", 100.0 * live / target }')"
  printf 'active_config=%s\n' "${config_path}"
  printf 'run_name=%s run_dir=%s\n' "${run_name:-unknown}" "${run_dir}"
  printf 'live_step=%s target_step=%s progress_pct=%s persisted_step=%s\n' \
    "${live_step}" "${target_step}" "${progress}" "${persisted_step}"
  if [[ -n "${latest_record}" ]]; then
    printf '%s\n' "${latest_record}"
  fi
}

stage211_current_attempt_errors() {
  local log_path="$1"
  local attempt_start
  attempt_start="$(stage211_current_attempt_start "${log_path}")"
  tail -n "+${attempt_start}" "${log_path}" 2>/dev/null |
    rg -n -i "${ERROR_PATTERN}" |
    tail -n 3 || true
}

stage211_recent_logs() {
  local root
  for root in "$@"; do
    [[ -d "${root}" ]] || continue
    find "${root}" \
      -type f -name '*.log' \
      ! -path "${MONITOR_LOG}" \
      ! -name 'monitor_*.log' \
      ! -path '*smoke*' \
      ! -path '*/wandb/*' \
      ! -name 'supervisor.log' \
      -mmin "-${RECENT_LOG_MINUTES}" -print0 2>/dev/null
  done
}

stage211_recent_formal_training_logs() {
  local log_path
  while IFS= read -r -d '' log_path; do
    if rg -q -F "${ATTEMPT_MARKER}" "${log_path}" 2>/dev/null; then
      printf '%s\0' "${log_path}"
    fi
  done < <(stage211_recent_logs "$@")
}

stage211_emit_snapshot() {
  local log_path latest_record errors prediction_path
  printf '\n===== %s =====\n' "$(date --iso-8601=seconds)"
  printf '%s\n' '-- sessions --'
  tmux list-sessions 2>&1 || true
  printf '%s\n' '-- training ranks --'
  ps -C python3 -o pid=,stat=,etime=,pcpu=,args= 2>/dev/null |
    awk '/rwkvasr\.cli\.train_ctc_deepspeed/ && /stage211/ { print }' || true
  printf '%s\n' '-- active training progress --'
  local active_configs=0 config_path
  while IFS= read -r config_path; do
    [[ -n "${config_path}" ]] || continue
    active_configs=$((active_configs + 1))
    stage211_emit_config_progress "${config_path}"
  done < <(stage211_active_config_paths)
  if ((active_configs == 0)); then
    printf '%s\n' none
  fi
  printf '%s\n' '-- supervisor tail --'
  tail -n 20 "${OUTPUT_ROOT}/supervisor.log" 2>&1 || true
  printf '%s\n' '-- public evaluation coverage --'
  while IFS= read -r -d '' prediction_path; do
    printf '%s %s\n' \
      "$(wc -l <"${prediction_path}" | tr -d ' ')" \
      "${prediction_path}"
  done < <(
    find "${CALIBRATION_EVAL_ROOT}" "${PHASE_GATE_ROOT}" \
      -type f -name '*.ctc.jsonl' -print0 2>/dev/null
  )
  printf '%s\n' '-- supplemental exact PCM audit --'
  if [[ -x "${MONITOR_PYTHON}" && -f "${SUPPLEMENTAL_PROGRESS_REPORTER}" && \
    -s "${BASE_PUBLIC_OVERLAP_ROOT}/manifest_location_index.sqlite" ]]; then
    nice -n 10 "${MONITOR_PYTHON}" "${SUPPLEMENTAL_PROGRESS_REPORTER}" \
      --output-root "${BASE_PUBLIC_OVERLAP_ROOT}" 2>&1 || true
  else
    printf 'unavailable root=%s reporter=%s python=%s\n' \
      "${BASE_PUBLIC_OVERLAP_ROOT}" "${SUPPLEMENTAL_PROGRESS_REPORTER}" "${MONITOR_PYTHON}"
  fi
  printf '%s\n' '-- recent training records --'
  while IFS= read -r -d '' log_path; do
    latest_record="$(stage211_latest_training_record "${log_path}")"
    if [[ -n "${latest_record}" ]]; then
      printf '%s\n%s\n' "${log_path}" "${latest_record}"
    fi
  done < <(stage211_recent_logs "${OUTPUT_ROOT}" "${CALIBRATION_ROOT}")
  printf '%s\n' '-- current-attempt formal-training errors --'
  while IFS= read -r -d '' log_path; do
    errors="$(stage211_current_attempt_errors "${log_path}")"
    if [[ -n "${errors}" ]]; then
      printf '%s\n%s\n' "${log_path}" "${errors}"
    else
      printf '%s none\n' "${log_path}"
    fi
  done < <(stage211_recent_formal_training_logs "${OUTPUT_ROOT}")
  printf '%s\n' '-- GPUs --'
  nvidia-smi \
    --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2>&1 || true
  printf '%s\n' '-- mounts and space --'
  findmnt -no SOURCE,TARGET,FSTYPE,OPTIONS /media/usbhd 2>&1 || true
  df -h / /media/usbhd 2>&1 || true
}

stage211_main() {
  mkdir -p "$(dirname "${MONITOR_LOG}")"
  if stage211_truthy "${MONITOR_ONCE}"; then
    stage211_emit_snapshot >>"${MONITOR_LOG}" 2>&1
    return
  fi
  while tmux has-session -t "${SUPERVISOR_SESSION}" 2>/dev/null; do
    stage211_emit_snapshot >>"${MONITOR_LOG}" 2>&1
    sleep "${POLL_SECONDS}"
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  stage211_main "$@"
fi
