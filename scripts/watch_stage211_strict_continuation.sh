#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

SUPERVISOR_SESSION="${SUPERVISOR_SESSION:-rwkvasr_stage211_abcd_strict_supervisor}"
MONITOR_SESSION="${MONITOR_SESSION:-rwkvasr_stage211_abcd_hourly_monitor}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
WATCH_ONCE="${WATCH_ONCE:-0}"

FULL_OUTPUT_ROOT="${FULL_OUTPUT_ROOT:-${HOME}/rwkvasr_runs/stage211_full_alignment}"
PHASE_GATE_ROOT="${PHASE_GATE_ROOT:-${HOME}/rwkvasr_eval/stage211_phase_gates}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-${FULL_OUTPUT_ROOT}/supervisor.log}"
WATCH_LOG="${WATCH_LOG:-${FULL_OUTPUT_ROOT}/continuation_watch.log}"
MONITOR_LOG="${MONITOR_LOG:-${FULL_OUTPUT_ROOT}/monitor_hourly.log}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${REPO_ROOT}/scripts/start_stage211_abcd_after_calibration.sh}"
MONITOR_SCRIPT="${MONITOR_SCRIPT:-${REPO_ROOT}/scripts/monitor_stage211_abcd.sh}"

MIXER_COMPLETE="${FULL_OUTPUT_ROOT}/stage211a_mixer_full_data_3ep/curriculum_complete.json"
MIXER_SELECTION="${PHASE_GATE_ROOT}/mixer_selected.json"
BLOCK_PROMOTION="${PHASE_GATE_ROOT}/block/block_promotion_receipt.json"
LOGITS_PROMOTION="${PHASE_GATE_ROOT}/logits/logits_promotion_receipt.json"
FINAL_REPORT="${PHASE_GATE_ROOT}/sft/stage211_complete.json"

stage211_log() {
  mkdir -p "$(dirname "${WATCH_LOG}")"
  printf '[stage211-continuation] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*" |
    tee -a "${WATCH_LOG}"
}

stage211_truthy() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

stage211_json_matches() {
  local path="$1"
  local filter="$2"
  [[ -s "${path}" ]] && jq -e "${filter}" "${path}" >/dev/null 2>&1
}

stage211_choose_start_stage() {
  if stage211_json_matches \
    "${LOGITS_PROMOTION}" \
    '.pipeline == "stage211" and .target_phase == "sft"'; then
    printf '%s\n' sft
  elif stage211_json_matches \
    "${BLOCK_PROMOTION}" \
    '.pipeline == "stage211" and .target_phase == "logits"'; then
    printf '%s\n' logits
  elif stage211_json_matches \
    "${MIXER_SELECTION}" \
    '.pipeline == "stage211" and .artifact == "mixer_gate_selection"'; then
    printf '%s\n' block
  elif stage211_json_matches \
    "${MIXER_COMPLETE}" \
    '.pipeline == "stage211" and .artifact == "full_phase_curriculum" and .phase == "mixer" and .complete == true'; then
    printf '%s\n' post_mixer
  else
    printf '%s\n' full
  fi
}

stage211_start_supervisor() {
  local start_stage="$1"
  if tmux has-session -t "${SUPERVISOR_SESSION}" 2>/dev/null; then
    stage211_log "replacement supervisor already active; no duplicate launch"
    return
  fi
  stage211_log "launching supervisor session=${SUPERVISOR_SESSION} START_STAGE=${start_stage}"
  tmux new-session -d \
    -s "${SUPERVISOR_SESSION}" \
    -c "${REPO_ROOT}" \
    env \
    START_STAGE="${start_stage}" \
    REUSE_COMPLETED_CALIBRATION_EVAL=1 \
    bash -c 'exec bash "$1" >>"$2" 2>&1' \
    _ "${BOOTSTRAP_SCRIPT}" "${SUPERVISOR_LOG}"
}

stage211_restore_monitor() {
  if tmux has-session -t "${MONITOR_SESSION}" 2>/dev/null; then
    return
  fi
  stage211_log "restoring hourly monitor session=${MONITOR_SESSION}"
  tmux new-session -d \
    -s "${MONITOR_SESSION}" \
    -c "${REPO_ROOT}" \
    env \
    SUPERVISOR_SESSION="${SUPERVISOR_SESSION}" \
    OUTPUT_ROOT="${FULL_OUTPUT_ROOT}" \
    PHASE_GATE_ROOT="${PHASE_GATE_ROOT}" \
    MONITOR_LOG="${MONITOR_LOG}" \
    POLL_SECONDS=3600 \
    bash "${MONITOR_SCRIPT}"
}

stage211_main() {
  while tmux has-session -t "${SUPERVISOR_SESSION}" 2>/dev/null; do
    stage211_log "active supervisor unchanged; next check in ${POLL_SECONDS}s"
    if stage211_truthy "${WATCH_ONCE}"; then
      return
    fi
    sleep "${POLL_SECONDS}"
  done

  if stage211_json_matches \
    "${FINAL_REPORT}" \
    '.pipeline == "stage211" and .artifact == "final_completion" and .complete == true and .gate_passed == true'; then
    stage211_log "final Stage211 report already passes; continuation not required"
    return
  fi

  local start_stage
  start_stage="$(stage211_choose_start_stage)"
  stage211_start_supervisor "${start_stage}"
  stage211_restore_monitor
}

stage211_main "$@"
