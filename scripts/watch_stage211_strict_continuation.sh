#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

SUPERVISOR_SESSION="${SUPERVISOR_SESSION:-rwkvasr_stage211_abcd_strict_supervisor}"
MONITOR_SESSION="${MONITOR_SESSION:-rwkvasr_stage211_abcd_hourly_monitor}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-60}"
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
BLOCK_SELECTION="${PHASE_GATE_ROOT}/block_selected.json"
LOGITS_SELECTION="${PHASE_GATE_ROOT}/logits_selected.json"
FINAL_REPORT="${PHASE_GATE_ROOT}/sft/stage211_complete.json"
FINAL_STEPWISE_REPORT="${PHASE_GATE_ROOT}/sft/stage211_stepwise_results.json"
FINAL_STEPWISE_MARKDOWN="${PHASE_GATE_ROOT}/sft/stage211_stepwise_results.md"
CALIBRATION_REUSE_RECEIPT="${CALIBRATION_REUSE_RECEIPT:-${HOME}/rwkvasr_eval/stage211_calibration_selected_full/public/reuse_receipt.json}"
INITIALIZATION_RECEIPT="${INITIALIZATION_RECEIPT:-${HOME}/rwkvasr_eval/stage211_initialization/nano_initialization_receipt.json}"
PUBLIC_METRIC_CORRECTION_RECEIPT="${PUBLIC_METRIC_CORRECTION_RECEIPT:-${HOME}/rwkvasr_eval/stage211_public_metric_unicode_v1/correction_receipt.json}"

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
    "${LOGITS_SELECTION}" \
    '.pipeline == "stage211" and .artifact == "phase_gate_selection" and .phase == "logits"'; then
    printf '%s\n' sft
  elif stage211_json_matches \
    "${BLOCK_SELECTION}" \
    '.pipeline == "stage211" and .artifact == "phase_gate_selection" and .phase == "block"'; then
    printf '%s\n' logits
  elif stage211_json_matches \
    "${MIXER_SELECTION}" \
    '.pipeline == "stage211" and .artifact == "mixer_gate_selection"'; then
    printf '%s\n' block
  elif stage211_json_matches \
    "${MIXER_COMPLETE}" \
    '.pipeline == "stage211" and .artifact == "full_phase_curriculum" and .phase == "mixer" and .complete == true and .full_data_coverage.supplemental_natural.complete == true and .full_data_coverage.supplemental_natural.epochs == 3'; then
    printf '%s\n' post_mixer
  else
    printf '%s\n' full
  fi
}

stage211_final_proof_valid() {
  if ! stage211_json_matches \
    "${FINAL_REPORT}" \
    '.pipeline == "stage211" and .artifact == "final_completion" and .complete == true and .gate_passed == true'; then
    return 1
  fi

  stage211_log "deep-validating final Stage211 stepwise proof"
  if ! (
    cd "${REPO_ROOT}"
    uv run python "${REPO_ROOT}/scripts/create_stage211_stepwise_report.py" \
      --initialization-receipt "${INITIALIZATION_RECEIPT}" \
      --calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}" \
      --public-metric-correction-receipt "${PUBLIC_METRIC_CORRECTION_RECEIPT}" \
      --sft-final-report "${FINAL_REPORT}" \
      --output-json "${FINAL_STEPWISE_REPORT}" \
      --output-markdown "${FINAL_STEPWISE_MARKDOWN}"
  ) >>"${WATCH_LOG}" 2>&1; then
    stage211_log "final Stage211 stepwise proof is missing or invalid"
    return 1
  fi

  if ! stage211_json_matches \
    "${FINAL_STEPWISE_REPORT}" \
    '.pipeline == "stage211" and .artifact == "stepwise_final_results" and .complete == true and .gate_passed == true and .strict_stage_order == ["calibration", "mixer", "block", "logits", "sft"] and .requested_alignment_stage_order == ["rwkv_layer", "block", "logits", "sft"] and .checkpoint_chain_passed == true and .nano_initialization_chain_passed == true and .ctc_label_normalization_chain_passed == true and .ctc_label_proof.full_length_index_audit_passed == true and .ctc_label_proof.ctc_suppress_non_pronunciation_tokens == true and .ctc_label_proof.ctc_unk_tokens == 0 and .public_metric_definition_chain_passed == true and .public_metric_tokenizer_contract == "unicode_alnum_words_basic_cjk_chars_v1" and (.public_metric_correction_receipt_sha256 | length) == 64 and (.public_metric_tokenizer_source_sha256 | length) == 64 and .nano_teacher_chain_passed == true and .nano_public_baseline_provenance_passed == true and .supplemental_inventory_chain_passed == true and (.coverage_results | length) == 4 and ([.coverage_results[] | select(.stage == "mixer" or .stage == "block" or .stage == "logits")] | length) == 3 and ([.coverage_results[] | select(.stage == "mixer" or .stage == "block" or .stage == "logits") | (.training_segments | length == 5 and all(.[]; .epochs == 3))] | all) and .public_metric_stage_order == ["calibration", "mixer", "block", "logits", "sft"] and .all_stage_public_metrics_complete == true and (.english_wer_datasets | length) == 3 and (.chinese_cer_datasets | length) == 2 and (.dataset_results | length) == 5 and ([.dataset_results[] | select(.language == "en" and .metric == "wer")] | length) == 3 and ([.dataset_results[] | select(.language == "zh" and .metric == "cer")] | length) == 2 and ([.dataset_results[] | (.stages | length == 5 and has("calibration") and has("mixer") and has("block") and has("logits") and has("sft"))] | all)'; then
    stage211_log "final Stage211 stepwise proof failed validation"
    return 1
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
  while true; do
    while tmux has-session -t "${SUPERVISOR_SESSION}" 2>/dev/null; do
      stage211_restore_monitor
      stage211_log "active supervisor unchanged; next check in ${POLL_SECONDS}s"
      if stage211_truthy "${WATCH_ONCE}"; then
        return
      fi
      sleep "${POLL_SECONDS}"
    done

    if stage211_final_proof_valid; then
      stage211_log "final Stage211 SFT and stepwise proofs pass; continuation not required"
      return
    fi

    local start_stage
    start_stage="$(stage211_choose_start_stage)"
    stage211_start_supervisor "${start_stage}"
    stage211_restore_monitor
    if stage211_truthy "${WATCH_ONCE}"; then
      return
    fi
    stage211_log "replacement supervisor launched; next audit in ${RESTART_BACKOFF_SECONDS}s"
    sleep "${RESTART_BACKOFF_SECONDS}"
  done
}

stage211_main "$@"
