#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CALIBRATION_SESSION="${CALIBRATION_SESSION:-rwkvasr_stage211a_recovery_formal}"
METADATA_SESSION="${METADATA_SESSION:-rwkvasr_stage211_metadata_copy}"
NANO_SESSION="${NANO_SESSION:-rwkvasr_stage211_public_full_nano}"
POLL_SECONDS="${POLL_SECONDS:-300}"
SUPPLEMENTAL_POLL_SECONDS="${SUPPLEMENTAL_POLL_SECONDS:-3600}"

CALIBRATION_RUN="${CALIBRATION_RUN:-${HOME}/rwkvasr_runs/sensevoice_rwkv_stage211a_recovery_stage210a30000_nanomlpfrozen_teacherforced_mixeronly_easy1490h_1ep_lr3e6_wd0_4x4090}"
METADATA_ROOT="${METADATA_ROOT:-${HOME}/rwkvasr_data/stage211_full_curriculum}"
EASY_MANIFEST="${EASY_MANIFEST:-${HOME}/rwkvasr_data/stage211_easy_source_grouped_buckets/manifest.json}"
PUBLIC_MANIFEST_DIR="${PUBLIC_MANIFEST_DIR:-${REPO_ROOT}/artifacts/eval_benchmarks/manifests}"
NANO_EVAL_DIR="${NANO_EVAL_DIR:-${HOME}/rwkvasr_eval/stage211_public_full/nano_2512}"
NANO_BASELINE_RECEIPT="${NANO_BASELINE_RECEIPT:-${NANO_EVAL_DIR}/provenance_receipt.json}"
CALIBRATION_EVAL_DIR="${CALIBRATION_EVAL_DIR:-${HOME}/rwkvasr_eval/stage211_calibration_selected_full}"
SELECTION_JSON="${SELECTION_JSON:-${CALIBRATION_EVAL_DIR}/checkpoint_selection.json}"
SELECTED_PATH_FILE="${SELECTED_PATH_FILE:-${CALIBRATION_EVAL_DIR}/selected_checkpoint.txt}"
FULL_OUTPUT_ROOT="${FULL_OUTPUT_ROOT:-${HOME}/rwkvasr_runs/stage211_full_alignment}"
FULL_CONFIG_ROOT="${FULL_CONFIG_ROOT:-${HOME}/rwkvasr_configs/stage211_full_alignment}"
NANO_CHECKPOINT="${NANO_CHECKPOINT:-${HOME}/models/Fun-ASR-Nano-2512-modelscope/model.pt}"
MASTER_PORT="${MASTER_PORT:-29631}"
PHASE_GATE_ROOT="${PHASE_GATE_ROOT:-${HOME}/rwkvasr_eval/stage211_phase_gates}"
LABELED_ROOT="${LABELED_ROOT:-${HOME}/rwkvasr_data/stage211_sft_full_labeled_v2}"
LABELED_PROFILE_RECEIPT="${LABELED_PROFILE_RECEIPT:-${LABELED_ROOT}/stage211_labeled_profile_receipt.json}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${FULL_OUTPUT_ROOT}/stage211d_labeled_ctc_sft_1ep}"
SFT_CORRECTION_PROFILE_ROOT="${SFT_CORRECTION_PROFILE_ROOT:-${HOME}/rwkvasr_data/stage211_sft_source_balanced_correction_v1}"
SFT_CORRECTION_RUN_ROOT="${SFT_CORRECTION_RUN_ROOT:-${FULL_OUTPUT_ROOT}/stage211d_sft_correction}"
SFT_CORRECTION_EVAL_ROOT="${SFT_CORRECTION_EVAL_ROOT:-${PHASE_GATE_ROOT}/sft_correction}"
SFT_CORRECTED_FINAL_ROOT="${SFT_CORRECTED_FINAL_ROOT:-${PHASE_GATE_ROOT}/sft_corrected}"
PUBLIC_OVERLAP_RECEIPT="${PUBLIC_OVERLAP_RECEIPT:-${METADATA_ROOT}/public_train_overlap_v2/receipt.json}"
PUBLIC_METRIC_CORRECTION_RECEIPT="${PUBLIC_METRIC_CORRECTION_RECEIPT:-${HOME}/rwkvasr_eval/stage211_public_metric_unicode_v2/correction_receipt.json}"
REUSE_COMPLETED_CALIBRATION_EVAL="${REUSE_COMPLETED_CALIBRATION_EVAL:-0}"
CALIBRATION_REUSE_RECEIPT="${CALIBRATION_REUSE_RECEIPT:-${CALIBRATION_EVAL_DIR}/public/reuse_receipt.json}"
START_STAGE="${START_STAGE:-full}"
RETENTION_REPLAY_RECEIPT="${RETENTION_REPLAY_RECEIPT:-${METADATA_ROOT}/retention_replay_v3/receipt.json}"
STRATIFIED_HIDDEN_RECEIPT="${STRATIFIED_HIDDEN_RECEIPT:-${METADATA_ROOT}/stratified_hidden_eval_v3/receipt.json}"
RETENTION_RUN_ROOT="${RETENTION_RUN_ROOT:-${FULL_OUTPUT_ROOT}/stage211a_mixer_retention_correction}"
RETENTION_GATE_ROOT="${RETENTION_GATE_ROOT:-${PHASE_GATE_ROOT}/mixer_retention}"
MIXER_SELECTION="${MIXER_SELECTION:-${PHASE_GATE_ROOT}/mixer_selected.json}"
BLOCK_CORRECTION_RUN_ROOT="${BLOCK_CORRECTION_RUN_ROOT:-${FULL_OUTPUT_ROOT}/stage211b_block_post_coverage_correction}"
BLOCK_CORRECTION_GATE_ROOT="${BLOCK_CORRECTION_GATE_ROOT:-${PHASE_GATE_ROOT}/block_correction}"
BLOCK_SELECTION="${BLOCK_SELECTION:-${PHASE_GATE_ROOT}/block_selected.json}"
LOGITS_CORRECTION_RUN_ROOT="${LOGITS_CORRECTION_RUN_ROOT:-${FULL_OUTPUT_ROOT}/stage211c_logits_post_coverage_correction}"
LOGITS_CORRECTION_GATE_ROOT="${LOGITS_CORRECTION_GATE_ROOT:-${PHASE_GATE_ROOT}/logits_correction}"
LOGITS_SELECTION="${LOGITS_SELECTION:-${PHASE_GATE_ROOT}/logits_selected.json}"
SUPPLEMENTAL_INVENTORY="${SUPPLEMENTAL_INVENTORY:-${HOME}/rwkvasr_data/stage211_supplemental_combined_v3/supplemental_inventory.json}"
SUPPLEMENTAL_PROFILE_RECEIPT="${SUPPLEMENTAL_PROFILE_RECEIPT:-${HOME}/rwkvasr_data/stage211_supplemental_combined_v3/supplemental_profile_receipt.json}"
INITIALIZATION_RECEIPT="${INITIALIZATION_RECEIPT:-${HOME}/rwkvasr_eval/stage211_initialization/nano_initialization_receipt.json}"

log() {
  printf '[stage211-abcd-bootstrap] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

truthy() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

wait_for_session() {
  local session="$1"
  local label="$2"
  while tmux has-session -t "${session}" 2>/dev/null; do
    log "${label} active session=${session}; waiting ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
  log "${label} session released"
}

wait_for_supplemental_training_data() {
  while true; do
    if [[ ! -s "${SUPPLEMENTAL_INVENTORY}" ]]; then
      log "supplemental inventory unavailable; waiting ${SUPPLEMENTAL_POLL_SECONDS}s path=${SUPPLEMENTAL_INVENTORY}"
    elif [[ ! -s "${SUPPLEMENTAL_PROFILE_RECEIPT}" ]]; then
      log "supplemental profile receipt unavailable; waiting ${SUPPLEMENTAL_POLL_SECONDS}s path=${SUPPLEMENTAL_PROFILE_RECEIPT}"
    elif [[ ! -s "${STRATIFIED_HIDDEN_RECEIPT}" ]]; then
      log "supplemental nine-cell receipt unavailable; waiting ${SUPPLEMENTAL_POLL_SECONDS}s path=${STRATIFIED_HIDDEN_RECEIPT}"
    elif [[ ! -s "${RETENTION_REPLAY_RECEIPT}" ]]; then
      log "supplemental replay v2 receipt unavailable; waiting ${SUPPLEMENTAL_POLL_SECONDS}s path=${RETENTION_REPLAY_RECEIPT}"
    elif uv run python "${REPO_ROOT}/scripts/create_stage211_supplemental_profile_receipt.py" \
        --inventory "${SUPPLEMENTAL_INVENTORY}" \
        --output "${SUPPLEMENTAL_PROFILE_RECEIPT}" \
      && uv run python "${REPO_ROOT}/scripts/validate_stage211_supplemental_retention.py" \
        --stratified-receipt "${STRATIFIED_HIDDEN_RECEIPT}" \
      && uv run python "${REPO_ROOT}/scripts/validate_stage211_supplemental_retention.py" \
        --receipt "${RETENTION_REPLAY_RECEIPT}"; then
      log "supplemental inventory, profile, nine-cell eval, and replay v2 validated"
      return
    else
      log "supplemental readiness validation failed; waiting ${SUPPLEMENTAL_POLL_SECONDS}s"
    fi
    sleep "${SUPPLEMENTAL_POLL_SECONDS}"
  done
}

validate_nano_predictions() {
  local datasets=(
    "librispeech_test_clean:2620"
    "librispeech_test_other:2939"
    "commonvoice_en_test:14927"
    "aishell1_test:7176"
    "wenetspeech_test_net:24774"
  )
  local entry
  for entry in "${datasets[@]}"; do
    local dataset="${entry%%:*}"
    local expected="${entry##*:}"
    local path="${NANO_EVAL_DIR}/predictions/${dataset}.ctc.jsonl"
    if [[ ! -s "${path}" ]]; then
      echo "Nano prediction missing: ${path}" >&2
      exit 1
    fi
    local actual
    actual="$(wc -l <"${path}" | tr -d ' ')"
    if [[ "${actual}" != "${expected}" ]]; then
      echo "Nano prediction coverage mismatch ${dataset}: ${actual}/${expected}" >&2
      exit 1
    fi
  done
  if [[ ! -s "${NANO_EVAL_DIR}/metrics.json" ]]; then
    echo "Nano full metrics are missing: ${NANO_EVAL_DIR}/metrics.json" >&2
    exit 1
  fi
  uv run python "${REPO_ROOT}/scripts/create_stage211_nano_baseline_receipt.py" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --report-dir "${NANO_EVAL_DIR}/reports" \
    --prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --output "${NANO_BASELINE_RECEIPT}"
}

build_fixed_manifests() {
  log "building and validating shared fixed-eval manifests"
  uv run python "${REPO_ROOT}/scripts/build_stage211_fixed_eval_manifests.py" \
    --easy-manifest "${EASY_MANIFEST}" \
    --metadata-root "${METADATA_ROOT}"
}

select_calibration_checkpoint() {
  mkdir -p "${CALIBRATION_EVAL_DIR}"
  log "selecting all-layer calibration checkpoint"
  uv run python "${REPO_ROOT}/scripts/select_stage211_calibration_checkpoint.py" \
    --run-dir "${CALIBRATION_RUN}" \
    --output "${SELECTION_JSON}" \
    --selected-path-output "${SELECTED_PATH_FILE}" \
    --required-completion-step 30064
  IFS= read -r SELECTED_CHECKPOINT <"${SELECTED_PATH_FILE}"
  if [[ ! -s "${SELECTED_CHECKPOINT}" ]]; then
    echo "Selected calibration checkpoint is unavailable: ${SELECTED_CHECKPOINT}" >&2
    exit 1
  fi
  export SELECTED_CHECKPOINT
  log "selected checkpoint=${SELECTED_CHECKPOINT}"
}

evaluate_calibration_checkpoint() {
  local output="${CALIBRATION_EVAL_DIR}/public"
  log "running full normalized public CTC evaluation for calibration checkpoint"
  env \
    CHECKPOINT_PATH="${SELECTED_CHECKPOINT}" \
    OUTPUT_DIR="${output}" \
    MANIFEST_DIR="${PUBLIC_MANIFEST_DIR}" \
    PREPARE_MANIFESTS=0 \
    DEVICES=0,1,2,3 \
    CTC_BATCH_SIZE=4 \
    CTC_NUM_WORKERS=0 \
    CTC_SHARD_STAGE2=1 \
    CTC_LIMIT=0 \
    CTC_TEXT_NORMALIZATION=ctc \
    RUN_AR=0 \
    METRIC_NORMALIZATION=ctc \
    bash "${REPO_ROOT}/scripts/run_public_eval_benchmarks.sh"

  uv run python "${REPO_ROOT}/scripts/compare_public_ctc_with_nano.py" \
    --student-prediction-dir "${output}/predictions" \
    --nano-prediction "aishell1_test=${NANO_EVAL_DIR}/predictions/aishell1_test.ctc.jsonl" \
    --nano-prediction "librispeech_test_clean=${NANO_EVAL_DIR}/predictions/librispeech_test_clean.ctc.jsonl" \
    --nano-prediction "librispeech_test_other=${NANO_EVAL_DIR}/predictions/librispeech_test_other.ctc.jsonl" \
    --nano-prediction "commonvoice_en_test=${NANO_EVAL_DIR}/predictions/commonvoice_en_test.ctc.jsonl" \
    --nano-prediction "wenetspeech_test_net=${NANO_EVAL_DIR}/predictions/wenetspeech_test_net.ctc.jsonl" \
    --student-checkpoint "${SELECTED_CHECKPOINT}" \
    --output-json "${output}/nano_comparison.json" \
    --output-md "${output}/nano_comparison.md" \
    --normalization ctc \
    --max-relative-ratio 1.20 \
    --max-absolute-gap-points 3.0
  log "calibration public comparison=${output}/nano_comparison.md"
}

validate_completed_calibration_eval() {
  local output="${CALIBRATION_EVAL_DIR}/public"
  log "validating completed calibration public evaluation for reuse"
  uv run python "${REPO_ROOT}/scripts/validate_stage211_calibration_eval.py" \
    --selection-report "${SELECTION_JSON}" \
    --comparison-report "${output}/nano_comparison.json" \
    --metrics "${output}/metrics.json" \
    --manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --output "${CALIBRATION_REUSE_RECEIPT}"
  log "calibration public reuse receipt=${CALIBRATION_REUSE_RECEIPT}"
}

ensure_initialization_receipt() {
  log "validating Nano QKV/MLP/decoder/head initialization proof"
  uv run python "${REPO_ROOT}/scripts/create_stage211_initialization_receipt.py" \
    --calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --output "${INITIALIZATION_RECEIPT}"
}

run_full_mixer_phase() {
  local final_checkpoint_file="${FULL_OUTPUT_ROOT}/stage211a_mixer_full_data_3ep/final_checkpoint_with_supplemental.txt"
  log "starting strict Stage211A full-data controller"
  uv run python "${REPO_ROOT}/scripts/run_stage211_full_phase_curriculum.py" \
    --phase mixer \
    --init-checkpoint "${SELECTED_CHECKPOINT}" \
    --output-root "${FULL_OUTPUT_ROOT}" \
    --config-root "${FULL_CONFIG_ROOT}" \
    --metadata-root "${METADATA_ROOT}" \
    --easy-manifest "${HOME}/rwkvasr_data/stage211_easy_source_grouped_buckets/manifest_stage211_fixed_eval.json" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --supplemental-inventory "${SUPPLEMENTAL_INVENTORY}" \
    --supplemental-profile-receipt "${SUPPLEMENTAL_PROFILE_RECEIPT}" \
    --master-port "${MASTER_PORT}" \
    --final-checkpoint-path-output "${final_checkpoint_file}"
  log "Stage211A full curriculum finished; starting strict retention/evaluation loop"
  run_mixer_retention_loop
  log "Stage211A Mixer gate passed"
}

run_mixer_retention_loop() {
  uv run python "${REPO_ROOT}/scripts/run_stage211_mixer_retention_loop.py" \
    --phase-root "${FULL_OUTPUT_ROOT}/stage211a_mixer_full_data_3ep" \
    --original-gate-dir "${PHASE_GATE_ROOT}/mixer" \
    --correction-run-root "${RETENTION_RUN_ROOT}" \
    --correction-gate-root "${RETENTION_GATE_ROOT}" \
    --replay-receipt "${RETENTION_REPLAY_RECEIPT}" \
    --stratified-hidden-receipt "${STRATIFIED_HIDDEN_RECEIPT}" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --baseline-public-comparison-report "${CALIBRATION_EVAL_DIR}/public/nano_comparison.json" \
    --config-dir "${FULL_CONFIG_ROOT}" \
    --selection "${MIXER_SELECTION}" \
    --master-port "$((MASTER_PORT + 10))" \
    --devices 0,1,2,3
}

run_full_block_phase() {
  local init_checkpoint
  init_checkpoint="$(jq -er '.checkpoint_path' "${MIXER_SELECTION}")"
  local promotion_receipt
  promotion_receipt="$(jq -er '.promotion_receipt_path' "${MIXER_SELECTION}")"
  local final_checkpoint_file="${FULL_OUTPUT_ROOT}/stage211b_block_full_data_3ep/final_checkpoint_with_supplemental.txt"
  log "starting strict Stage211B full-data controller"
  uv run python "${REPO_ROOT}/scripts/run_stage211_full_phase_curriculum.py" \
    --phase block \
    --init-checkpoint "${init_checkpoint}" \
    --promotion-receipt "${promotion_receipt}" \
    --output-root "${FULL_OUTPUT_ROOT}" \
    --config-root "${FULL_CONFIG_ROOT}" \
    --metadata-root "${METADATA_ROOT}" \
    --easy-manifest "${HOME}/rwkvasr_data/stage211_easy_source_grouped_buckets/manifest_stage211_fixed_eval.json" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --supplemental-inventory "${SUPPLEMENTAL_INVENTORY}" \
    --supplemental-profile-receipt "${SUPPLEMENTAL_PROFILE_RECEIPT}" \
    --auto-batch-profile \
    --batch-profile-master-port "$((MASTER_PORT + 110))" \
    --master-port "$((MASTER_PORT + 1))" \
    --final-checkpoint-path-output "${final_checkpoint_file}"
  log "Stage211B full curriculum finished; starting strict correction/evaluation loop"
  run_block_correction_loop
  log "Stage211B selected Block gate passed"
}

run_block_correction_loop() {
  local mixer_gate_dir
  mixer_gate_dir="$(jq -er '.gate_dir' "${MIXER_SELECTION}")"
  uv run python "${REPO_ROOT}/scripts/run_stage211_mixer_retention_loop.py" \
    --phase block \
    --phase-root "${FULL_OUTPUT_ROOT}/stage211b_block_full_data_3ep" \
    --original-gate-dir "${PHASE_GATE_ROOT}/block" \
    --correction-run-root "${BLOCK_CORRECTION_RUN_ROOT}" \
    --correction-gate-root "${BLOCK_CORRECTION_GATE_ROOT}" \
    --replay-receipt "${RETENTION_REPLAY_RECEIPT}" \
    --stratified-hidden-receipt "${STRATIFIED_HIDDEN_RECEIPT}" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --baseline-public-comparison-report "${mixer_gate_dir}/nano_comparison.json" \
    --config-dir "${FULL_CONFIG_ROOT}" \
    --selection "${BLOCK_SELECTION}" \
    --master-port "$((MASTER_PORT + 11))" \
    --devices 0,1,2,3
}

run_full_logits_phase() {
  local init_checkpoint
  init_checkpoint="$(jq -er '.checkpoint_path' "${BLOCK_SELECTION}")"
  local promotion_receipt
  promotion_receipt="$(jq -er '.promotion_receipt_path' "${BLOCK_SELECTION}")"
  local final_checkpoint_file="${FULL_OUTPUT_ROOT}/stage211c_logits_full_data_3ep/final_checkpoint_with_supplemental.txt"
  log "starting strict Stage211C full-data controller"
  uv run python "${REPO_ROOT}/scripts/run_stage211_full_phase_curriculum.py" \
    --phase logits \
    --init-checkpoint "${init_checkpoint}" \
    --promotion-receipt "${promotion_receipt}" \
    --output-root "${FULL_OUTPUT_ROOT}" \
    --config-root "${FULL_CONFIG_ROOT}" \
    --metadata-root "${METADATA_ROOT}" \
    --easy-manifest "${HOME}/rwkvasr_data/stage211_easy_source_grouped_buckets/manifest_stage211_fixed_eval.json" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --supplemental-inventory "${SUPPLEMENTAL_INVENTORY}" \
    --supplemental-profile-receipt "${SUPPLEMENTAL_PROFILE_RECEIPT}" \
    --auto-batch-profile \
    --batch-profile-master-port "$((MASTER_PORT + 120))" \
    --master-port "$((MASTER_PORT + 2))" \
    --final-checkpoint-path-output "${final_checkpoint_file}"
  log "Stage211C full curriculum finished; starting strict correction/evaluation loop"
  run_logits_correction_loop
  log "Stage211C selected Logits gate passed the complete Nano CTC gate"
}

run_logits_correction_loop() {
  uv run python "${REPO_ROOT}/scripts/run_stage211_mixer_retention_loop.py" \
    --phase logits \
    --phase-root "${FULL_OUTPUT_ROOT}/stage211c_logits_full_data_3ep" \
    --original-gate-dir "${PHASE_GATE_ROOT}/logits" \
    --correction-run-root "${LOGITS_CORRECTION_RUN_ROOT}" \
    --correction-gate-root "${LOGITS_CORRECTION_GATE_ROOT}" \
    --replay-receipt "${RETENTION_REPLAY_RECEIPT}" \
    --stratified-hidden-receipt "${STRATIFIED_HIDDEN_RECEIPT}" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --config-dir "${FULL_CONFIG_ROOT}" \
    --selection "${LOGITS_SELECTION}" \
    --master-port "$((MASTER_PORT + 12))" \
    --devices 0,1,2,3
}

run_stepwise_report() {
  local final_report="$1"
  local report_dir
  report_dir="$(dirname "${final_report}")"
  local output_json="${report_dir}/stage211_stepwise_results.json"
  local output_markdown="${report_dir}/stage211_stepwise_results.md"
  if [[ ! -s "${final_report}" ]]; then
    log "successful SFT path produced no final report: ${final_report}"
    return 1
  fi
  log "building strict Layer/Block/Logits/SFT stepwise report from ${final_report}"
  uv run python "${REPO_ROOT}/scripts/create_stage211_stepwise_report.py" \
    --initialization-receipt "${INITIALIZATION_RECEIPT}" \
    --calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}" \
    --public-metric-correction-receipt "${PUBLIC_METRIC_CORRECTION_RECEIPT}" \
    --sft-final-report "${final_report}" \
    --output-json "${output_json}" \
    --output-markdown "${output_markdown}"
  if [[ ! -s "${output_json}" || ! -s "${output_markdown}" ]]; then
    log "stepwise report builder returned without complete outputs: ${report_dir}"
    return 1
  fi
  log "strict stepwise report completed at ${output_json}"
}

run_labeled_sft_phase() {
  local init_checkpoint
  init_checkpoint="$(jq -er '.checkpoint_path' "${LOGITS_SELECTION}")"
  local promotion_receipt
  promotion_receipt="$(jq -er '.promotion_receipt_path' "${LOGITS_SELECTION}")"
  local logits_gate_dir
  logits_gate_dir="$(jq -er '.gate_dir' "${LOGITS_SELECTION}")"
  local final_checkpoint_file="${SFT_OUTPUT_DIR}/final_checkpoint.txt"
  log "starting restart-safe Stage211D labeled CTC SFT"
  uv run python "${REPO_ROOT}/scripts/run_stage211_labeled_sft.py" \
    --init-checkpoint "${init_checkpoint}" \
    --logits-promotion-receipt "${promotion_receipt}" \
    --output-dir "${SFT_OUTPUT_DIR}" \
    --config-dir "${FULL_CONFIG_ROOT}" \
    --labeled-webdataset-root "${LABELED_ROOT}" \
    --labeled-length-index "${LABELED_ROOT}/webdataset_lengths.jsonl" \
    --bucket-manifest "${LABELED_ROOT}/webdataset_buckets_audio_text/manifest.json" \
    --labeled-profile-receipt "${LABELED_PROFILE_RECEIPT}" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --master-port "$((MASTER_PORT + 3))" \
    --final-checkpoint-path-output "${final_checkpoint_file}"
  log "Stage211D labeled epoch finished; starting complete public CTC gate"
  if uv run python "${REPO_ROOT}/scripts/finalize_stage211_labeled_sft.py" \
    --run-dir "${SFT_OUTPUT_DIR}" \
    --output-dir "${PHASE_GATE_ROOT}/sft" \
    --calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}" \
    --baseline-public-comparison-report "${logits_gate_dir}/nano_comparison.json" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --phase-gate-root "${PHASE_GATE_ROOT}" \
    --initialization-receipt "${INITIALIZATION_RECEIPT}" \
    --mixer-gate-selection "${MIXER_SELECTION}" \
    --block-gate-selection "${BLOCK_SELECTION}" \
    --logits-gate-selection "${LOGITS_SELECTION}" \
    --devices 0,1,2,3; then
    run_stepwise_report "${PHASE_GATE_ROOT}/sft/stage211_complete.json"
    log "Stage211 A/B/C/D strict alignment pipeline completed without SFT correction"
    return
  fi

  local full_sft_completion="${SFT_OUTPUT_DIR}/sft_complete.json"
  local full_sft_failed_report="${PHASE_GATE_ROOT}/sft/stage211_complete.json"
  if [[ ! -s "${full_sft_completion}" || ! -s "${full_sft_failed_report}" ]]; then
    log "Stage211D finalizer failed without complete full-SFT evidence; refusing correction"
    return 1
  fi
  log "Stage211D full SFT gate failed; starting strict balanced-label correction loop"
  uv run python "${REPO_ROOT}/scripts/run_stage211_sft_correction_loop.py" \
    --full-sft-completion "${full_sft_completion}" \
    --full-sft-failed-report "${full_sft_failed_report}" \
    --correction-profile-root "${SFT_CORRECTION_PROFILE_ROOT}" \
    --run-root "${SFT_CORRECTION_RUN_ROOT}" \
    --eval-root "${SFT_CORRECTION_EVAL_ROOT}" \
    --final-output-dir "${SFT_CORRECTED_FINAL_ROOT}" \
    --config-dir "${FULL_CONFIG_ROOT}" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --public-overlap-receipt "${PUBLIC_OVERLAP_RECEIPT}" \
    --phase-gate-root "${PHASE_GATE_ROOT}" \
    --mixer-gate-selection "${MIXER_SELECTION}" \
    --block-gate-selection "${BLOCK_SELECTION}" \
    --logits-gate-selection "${LOGITS_SELECTION}" \
    --initialization-receipt "${INITIALIZATION_RECEIPT}" \
    --calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}" \
    --master-port "$((MASTER_PORT + 20))" \
    --devices 0,1,2,3
  run_stepwise_report "${SFT_CORRECTED_FINAL_ROOT}/stage211_complete.json"
  log "Stage211 A/B/C/D strict alignment pipeline completed after SFT correction"
}

main() {
  cd "${REPO_ROOT}"
  wait_for_supplemental_training_data
  case "${START_STAGE}" in
    full)
      wait_for_session "${METADATA_SESSION}" "metadata copy"
      build_fixed_manifests
      wait_for_session "${NANO_SESSION}" "Nano full benchmark"
      validate_nano_predictions
      wait_for_session "${CALIBRATION_SESSION}" "Stage211A calibration"
      select_calibration_checkpoint
      if truthy "${REUSE_COMPLETED_CALIBRATION_EVAL}"; then
        validate_completed_calibration_eval
      else
        evaluate_calibration_checkpoint
      fi
      ensure_initialization_receipt
      run_full_mixer_phase
      run_full_block_phase
      run_full_logits_phase
      run_labeled_sft_phase
      ;;
    post_mixer)
      ensure_initialization_receipt
      run_mixer_retention_loop
      run_full_block_phase
      run_full_logits_phase
      run_labeled_sft_phase
      ;;
    block)
      ensure_initialization_receipt
      run_mixer_retention_loop
      run_full_block_phase
      run_full_logits_phase
      run_labeled_sft_phase
      ;;
    logits)
      ensure_initialization_receipt
      run_mixer_retention_loop
      run_block_correction_loop
      run_full_logits_phase
      run_labeled_sft_phase
      ;;
    sft)
      ensure_initialization_receipt
      run_mixer_retention_loop
      run_block_correction_loop
      run_logits_correction_loop
      run_labeled_sft_phase
      ;;
    *)
      echo "Unsupported START_STAGE=${START_STAGE}; expected full/post_mixer/block/logits/sft" >&2
      exit 2
      ;;
  esac
}

main "$@"
