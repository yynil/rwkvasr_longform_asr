#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

SUPERVISOR_SESSION="${SUPERVISOR_SESSION:-rwkvasr_stage211_abcd_strict_supervisor}"
MONITOR_SESSION="${MONITOR_SESSION:-rwkvasr_stage211_abcd_hourly_monitor}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-3600}"
WATCH_ONCE="${WATCH_ONCE:-0}"

FULL_OUTPUT_ROOT="${FULL_OUTPUT_ROOT:-${HOME}/rwkvasr_runs/stage211_full_alignment}"
PHASE_GATE_ROOT="${PHASE_GATE_ROOT:-${HOME}/rwkvasr_eval/stage211_phase_gates}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-${FULL_OUTPUT_ROOT}/supervisor.log}"
WATCH_LOG="${WATCH_LOG:-${FULL_OUTPUT_ROOT}/continuation_watch.log}"
MONITOR_LOG="${MONITOR_LOG:-${FULL_OUTPUT_ROOT}/monitor_hourly.log}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-${REPO_ROOT}/scripts/start_stage211_abcd_after_calibration.sh}"
MONITOR_SCRIPT="${MONITOR_SCRIPT:-${REPO_ROOT}/scripts/monitor_stage211_abcd.sh}"
SELECTION_VALIDATOR_SCRIPT="${SELECTION_VALIDATOR_SCRIPT:-${REPO_ROOT}/scripts/run_stage211_mixer_retention_loop.py}"
CURRICULUM_VALIDATOR_SCRIPT="${CURRICULUM_VALIDATOR_SCRIPT:-${REPO_ROOT}/scripts/finalize_stage211_phase.py}"
NANO_CHECKPOINT="${NANO_CHECKPOINT:-${HOME}/models/Fun-ASR-Nano-2512-modelscope/model.pt}"

MIXER_COMPLETE="${FULL_OUTPUT_ROOT}/stage211a_mixer_full_data_3ep/curriculum_complete.json"
MIXER_SELECTION="${PHASE_GATE_ROOT}/mixer_selected.json"
BLOCK_SELECTION="${PHASE_GATE_ROOT}/block_selected.json"
LOGITS_SELECTION="${PHASE_GATE_ROOT}/logits_selected.json"
FINAL_REPORT="${PHASE_GATE_ROOT}/sft/stage211_complete.json"
FINAL_STEPWISE_REPORT="${PHASE_GATE_ROOT}/sft/stage211_stepwise_results.json"
FINAL_STEPWISE_MARKDOWN="${PHASE_GATE_ROOT}/sft/stage211_stepwise_results.md"
CORRECTED_FINAL_REPORT="${PHASE_GATE_ROOT}/sft_corrected/stage211_complete.json"
CORRECTED_FINAL_STEPWISE_REPORT="${PHASE_GATE_ROOT}/sft_corrected/stage211_stepwise_results.json"
CORRECTED_FINAL_STEPWISE_MARKDOWN="${PHASE_GATE_ROOT}/sft_corrected/stage211_stepwise_results.md"
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

stage211_selection_valid() {
  local phase="$1"
  local selection="$2"
  [[ -s "${selection}" ]] || return 1
  mkdir -p "$(dirname "${WATCH_LOG}")"
  (
    cd "${REPO_ROOT}"
    uv run python "${SELECTION_VALIDATOR_SCRIPT}" \
      --phase "${phase}" \
      --selection "${selection}" \
      --nano-checkpoint "${NANO_CHECKPOINT}" \
      --validate-selection-only
  ) >>"${WATCH_LOG}" 2>&1
}

stage211_mixer_curriculum_valid() {
  [[ -s "${MIXER_COMPLETE}" ]] || return 1
  mkdir -p "$(dirname "${WATCH_LOG}")"
  (
    cd "${REPO_ROOT}"
    uv run python "${CURRICULUM_VALIDATOR_SCRIPT}" \
      --phase mixer \
      --phase-root "${FULL_OUTPUT_ROOT}/stage211a_mixer_full_data_3ep" \
      --validate-curriculum-only
  ) >>"${WATCH_LOG}" 2>&1
}

stage211_trajectory_results_valid() {
  local path="$1"
  stage211_json_matches \
    "${path}" \
    '(.alignment_results | length) == 3 and ([.alignment_results[] | .stage] == ["mixer", "block", "logits"]) and ([.alignment_results[].trajectory_retention | (.gate_passed == true and .fixed_eval_samples == 256 and .terminal_entries >= 5 and .source_order[:5] == ["easy", "medium", "hard", "long", "supplemental_natural"] and .max_relative_regression_pct == 10.0 and ([.best_prior_loss, .candidate_loss, .relative_regression_pct] | all(.[]; type == "number" and isfinite)))] | all)'
}

stage211_periodic_cadence_results_valid() {
  local path="$1"
  stage211_json_matches \
    "${path}" \
    '([.alignment_results[].step_eval_cadence | (.complete == true and .interval_steps == 10000 and .eval_samples_per_report == 256 and .source_count == (.sources | length) and .source_count >= 5 and .total_reports == ([.sources[].report_count] | add) and all(.sources[]; .terminal_step > 0 and .report_count > 0 and .eval_samples_per_report == 256))] | all) and ([.coverage_results[] | select(.stage == "sft")] | length) == 1 and ([.coverage_results[] | select(.stage == "sft") | .step_eval_cadence | (.complete == true and .interval_steps == 2000 and .eval_samples_per_report == 256 and .source_order == ["labeled_sft"] and .source_count == 1 and .total_reports == ([.sources[].report_count] | add) and all(.sources[]; .source == "labeled_sft" and .terminal_step > 0 and .report_count > 0 and .eval_samples_per_report == 256))] | all)'
}

stage211_choose_start_stage() {
  if stage211_selection_valid logits "${LOGITS_SELECTION}"; then
    printf '%s\n' sft
  elif stage211_selection_valid block "${BLOCK_SELECTION}"; then
    printf '%s\n' logits
  elif stage211_selection_valid mixer "${MIXER_SELECTION}"; then
    printf '%s\n' block
  elif stage211_mixer_curriculum_valid; then
    printf '%s\n' post_mixer
  else
    printf '%s\n' full
  fi
}

stage211_final_candidate_valid() {
  local final_report="$1"
  local final_stepwise_report="$2"
  local final_stepwise_markdown="$3"
  if ! stage211_json_matches \
    "${final_report}" \
    '.pipeline == "stage211" and .artifact == "final_completion" and .complete == true and .gate_passed == true'; then
    return 1
  fi

  stage211_log "deep-validating final Stage211 stepwise proof source=${final_report}"
  if ! (
    cd "${REPO_ROOT}"
    uv run python "${REPO_ROOT}/scripts/create_stage211_stepwise_report.py" \
      --initialization-receipt "${INITIALIZATION_RECEIPT}" \
      --calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}" \
      --public-metric-correction-receipt "${PUBLIC_METRIC_CORRECTION_RECEIPT}" \
      --sft-final-report "${final_report}" \
      --output-json "${final_stepwise_report}" \
      --output-markdown "${final_stepwise_markdown}"
  ) >>"${WATCH_LOG}" 2>&1; then
    stage211_log "final Stage211 stepwise proof is missing or invalid source=${final_report}"
    return 1
  fi

  if ! stage211_json_matches \
    "${final_stepwise_report}" \
    '.pipeline == "stage211" and .artifact == "stepwise_final_results" and .complete == true and .gate_passed == true and .strict_stage_order == ["calibration", "mixer", "block", "logits", "sft"] and .requested_alignment_stage_order == ["rwkv_layer", "block", "logits", "sft"] and .all_requested_alignment_metrics_complete == true and .checkpoint_chain_passed == true and .nano_initialization_chain_passed == true and .nano_initialization_source_chain_passed == true and .ctc_label_normalization_chain_passed == true and .ctc_label_proof.full_length_index_audit_passed == true and .ctc_label_proof.ctc_suppress_non_pronunciation_tokens == true and .ctc_label_proof.ctc_suppressed_token_ids_count == 2114 and .ctc_label_proof.ctc_suppressed_token_ids_sha256 == "76a68d03bb2dc486c214fd44891f3e3e2c36286d79fea1e69c8e74d7767d5a09" and .ctc_label_proof.teacher_projection_support_matches_student == true and .ctc_label_proof.ctc_unk_tokens == 0 and .public_metric_definition_chain_passed == true and .public_metric_tokenizer_contract == "unicode_alnum_words_basic_cjk_chars_v1" and (.public_metric_correction_receipt_sha256 | length) == 64 and (.public_metric_tokenizer_source_sha256 | length) == 64 and .nano_teacher_chain_passed == true and .nano_public_baseline_provenance_passed == true and .supplemental_inventory_chain_passed == true and .supplemental_dedupe_proof.inventory_schema_version == 2 and .supplemental_dedupe_proof.inventory_artifact == "stage211_supplemental_combined_inventory" and .supplemental_dedupe_proof.mode == "source_identity_plus_known_corpus_exclusion" and .supplemental_dedupe_proof.source_sets_disjoint == true and .supplemental_dedupe_proof.content_fingerprint_complete == false and .supplemental_dedupe_proof.base_public_overlap_normalized_pcm_exact_complete == true and .supplemental_dedupe_proof.base_public_overlap_scan_order == "manifest_location_index_archive_order_v1" and .supplemental_dedupe_proof.base_public_overlap_rows == 0 and .supplemental_dedupe_proof.base_public_overlap_scanned_rows > 0 and (.supplemental_dedupe_proof.base_public_overlap_receipt_sha256 | length) == 64 and .supplemental_dedupe_proof.social_normalized_pcm_exact_complete == true and .supplemental_dedupe_proof.social_public_overlap_mode == "normalized_pcm_exact" and .supplemental_dedupe_proof.archived_social_exact_duplicate_exclusion_complete == true and .supplemental_dedupe_proof.archived_social_unique_members == 0 and (.supplemental_dedupe_proof.archived_social_overlap_receipt_sha256 | length) == 64 and .supplemental_dedupe_proof.usb_top_level_classification_complete == true and .supplemental_dedupe_proof.usb_natural_audio_resolution_complete == true and .supplemental_dedupe_proof.usb_unresolved_natural_entries == [] and (.supplemental_dedupe_proof.usb_top_level_coverage_receipt_sha256 | length) == 64 and .supplemental_dedupe_proof.near_duplicate_complete == false and .supplemental_dedupe_proof.known_overlap_exclusions == ["llaso_gigaspeech", "llaso_librispeech"] and (.supplemental_dedupe_proof.component_inventories | keys | sort) == ["base_natural", "social_vad"] and (.coverage_results | length) == 4 and ([.coverage_results[] | select(.stage == "mixer" or .stage == "block" or .stage == "logits")] | length) == 3 and ([.coverage_results[] | select(.stage == "mixer" or .stage == "block" or .stage == "logits") | (.training_segments | length == 5 and all(.[]; .epochs == 3))] | all) and .all_stage_alignment_results_complete == true and (.alignment_results | length) == 3 and ([.alignment_results[] | .stage] == ["mixer", "block", "logits"]) and ([.alignment_results[] | (.gate_passed == true and .stratified_gate_passed == true and .fixed_eval_samples == 256 and .stratified_samples == 2304 and (.stratified_cells | length) == 9 and (.source_report_sha256 | length) == 64)] | all) and .public_metric_stage_order == ["calibration", "mixer", "block", "logits", "sft"] and .all_stage_public_metrics_complete == true and (.english_wer_datasets | length) == 3 and (.chinese_cer_datasets | length) == 2 and (.language_metric_summaries | length) == 2 and ([.language_metric_summaries[] | select(.name == "english_wer" and .language == "en" and .metric == "wer" and .aggregation == "unweighted_dataset_macro" and .dataset_count == 3 and (.datasets | length) == 3 and .sample_count > 0 and (.stages | length) == 5 and ([.stages[] | type == "number"] | all))] | length) == 1 and ([.language_metric_summaries[] | select(.name == "chinese_cer" and .language == "zh" and .metric == "cer" and .aggregation == "unweighted_dataset_macro" and .dataset_count == 2 and (.datasets | length) == 2 and .sample_count > 0 and (.stages | length) == 5 and ([.stages[] | type == "number"] | all))] | length) == 1 and (.dataset_results | length) == 5 and ([.dataset_results[] | select(.language == "en" and .metric == "wer")] | length) == 3 and ([.dataset_results[] | select(.language == "zh" and .metric == "cer")] | length) == 2 and ([.dataset_results[] | (.stages | length == 5 and has("calibration") and has("mixer") and has("block") and has("logits") and has("sft"))] | all) and .initial_calibration_result.role == "initial_baseline" and (.initial_calibration_result.checkpoint_sha256 | length) == 64 and (.initial_calibration_result.english_wer | type) == "number" and (.initial_calibration_result.chinese_cer | type) == "number" and ([.requested_alignment_results[] | .stage] == ["rwkv_layer", "block", "logits", "sft"]) and ([.requested_alignment_results[] | .internal_stage] == ["mixer", "block", "logits", "sft"]) and ([.requested_alignment_results[] | (.gate_passed == true and (.checkpoint_sha256 | length) == 64 and (.source_report_sha256 | length) == 64 and (.english_wer | type) == "number" and (.chinese_cer | type) == "number")] | all) and (.requested_alignment_language_metric_summaries | length) == 2 and ([.requested_alignment_language_metric_summaries[] | (.stages | keys) == ["block", "logits", "rwkv_layer", "sft"] and ([.stages[] | type == "number"] | all) and (.initial_calibration_error_rate | type) == "number"] | all) and (.requested_alignment_dataset_results | length) == 5 and ([.requested_alignment_dataset_results[] | (.stages | keys) == ["block", "logits", "rwkv_layer", "sft"] and ([.stages[] | .student_error_rate | type == "number"] | all) and (.initial_calibration.student_error_rate | type) == "number"] | all) and ([.requested_alignment_results[] as $result | ($result.english_wer == ([.requested_alignment_language_metric_summaries[] | select(.name == "english_wer")][0].stages[$result.stage])) and ($result.chinese_cer == ([.requested_alignment_language_metric_summaries[] | select(.name == "chinese_cer")][0].stages[$result.stage]))] | all)'; then
    stage211_log "final Stage211 stepwise proof failed validation source=${final_report}"
    return 1
  fi
  if ! stage211_trajectory_results_valid "${final_stepwise_report}"; then
    stage211_log "final Stage211 trajectory-retention proof failed validation source=${final_report}"
    return 1
  fi
  if ! stage211_periodic_cadence_results_valid "${final_stepwise_report}"; then
    stage211_log "final Stage211 periodic fixed-eval proof failed validation source=${final_report}"
    return 1
  fi
}

stage211_final_proof_valid() {
  if stage211_final_candidate_valid \
    "${FINAL_REPORT}" \
    "${FINAL_STEPWISE_REPORT}" \
    "${FINAL_STEPWISE_MARKDOWN}"; then
    return 0
  fi
  stage211_final_candidate_valid \
    "${CORRECTED_FINAL_REPORT}" \
    "${CORRECTED_FINAL_STEPWISE_REPORT}" \
    "${CORRECTED_FINAL_STEPWISE_MARKDOWN}"
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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  stage211_main "$@"
fi
