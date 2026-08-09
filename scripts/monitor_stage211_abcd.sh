#!/usr/bin/env bash
set -uo pipefail

SUPERVISOR_SESSION="${SUPERVISOR_SESSION:-rwkvasr_stage211_abcd_strict_supervisor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/rwkvasr_runs/stage211_full_alignment}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${HOME}/rwkvasr_runs/sensevoice_rwkv_stage211a_recovery_stage210a30000_nanomlpfrozen_teacherforced_mixeronly_easy1490h_1ep_lr3e6_wd0_4x4090}"
CALIBRATION_EVAL_ROOT="${CALIBRATION_EVAL_ROOT:-${HOME}/rwkvasr_eval/stage211_calibration_selected_full}"
PHASE_GATE_ROOT="${PHASE_GATE_ROOT:-${HOME}/rwkvasr_eval/stage211_phase_gates}"
MONITOR_LOG="${MONITOR_LOG:-${OUTPUT_ROOT}/monitor_hourly.log}"
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

stage211_emit_snapshot() {
  local log_path latest_record errors prediction_path
  printf '\n===== %s =====\n' "$(date --iso-8601=seconds)"
  printf '%s\n' '-- sessions --'
  tmux list-sessions 2>&1 || true
  printf '%s\n' '-- training ranks --'
  ps -C python3 -o pid=,stat=,etime=,pcpu=,args= 2>/dev/null |
    awk '/rwkvasr\.cli\.train_ctc_deepspeed/ && /stage211/ { print }' || true
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
  done < <(stage211_recent_logs "${OUTPUT_ROOT}")
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
