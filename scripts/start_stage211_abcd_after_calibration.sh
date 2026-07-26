#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CALIBRATION_SESSION="${CALIBRATION_SESSION:-rwkvasr_stage211a_recovery_formal}"
METADATA_SESSION="${METADATA_SESSION:-rwkvasr_stage211_metadata_copy}"
NANO_SESSION="${NANO_SESSION:-rwkvasr_stage211_public_full_nano}"
POLL_SECONDS="${POLL_SECONDS:-300}"

CALIBRATION_RUN="${CALIBRATION_RUN:-${HOME}/rwkvasr_runs/sensevoice_rwkv_stage211a_recovery_stage210a30000_nanomlpfrozen_teacherforced_mixeronly_easy1490h_1ep_lr3e6_wd0_4x4090}"
METADATA_ROOT="${METADATA_ROOT:-${HOME}/rwkvasr_data/stage211_full_curriculum}"
EASY_MANIFEST="${EASY_MANIFEST:-${HOME}/rwkvasr_data/stage211_easy_source_grouped_buckets/manifest.json}"
PUBLIC_MANIFEST_DIR="${PUBLIC_MANIFEST_DIR:-${REPO_ROOT}/artifacts/eval_benchmarks/manifests}"
NANO_EVAL_DIR="${NANO_EVAL_DIR:-${HOME}/rwkvasr_eval/stage211_public_full/nano_2512}"
CALIBRATION_EVAL_DIR="${CALIBRATION_EVAL_DIR:-${HOME}/rwkvasr_eval/stage211_calibration_selected_full}"
SELECTION_JSON="${SELECTION_JSON:-${CALIBRATION_EVAL_DIR}/checkpoint_selection.json}"
SELECTED_PATH_FILE="${SELECTED_PATH_FILE:-${CALIBRATION_EVAL_DIR}/selected_checkpoint.txt}"
FULL_OUTPUT_ROOT="${FULL_OUTPUT_ROOT:-${HOME}/rwkvasr_runs/stage211_full_alignment}"
FULL_CONFIG_ROOT="${FULL_CONFIG_ROOT:-${HOME}/rwkvasr_configs/stage211_full_alignment}"
NANO_CHECKPOINT="${NANO_CHECKPOINT:-${HOME}/models/Fun-ASR-Nano-2512-modelscope/model.pt}"
MASTER_PORT="${MASTER_PORT:-29631}"
PHASE_GATE_ROOT="${PHASE_GATE_ROOT:-${HOME}/rwkvasr_eval/stage211_phase_gates}"
LABELED_ROOT="${LABELED_ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset/stages/easy_clean_librispeech_aishell3_ctc_norm_aligned_sensevoice_lfr6_svtok}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${FULL_OUTPUT_ROOT}/stage211d_labeled_ctc_sft_1ep}"
REUSE_COMPLETED_CALIBRATION_EVAL="${REUSE_COMPLETED_CALIBRATION_EVAL:-0}"
CALIBRATION_REUSE_RECEIPT="${CALIBRATION_REUSE_RECEIPT:-${CALIBRATION_EVAL_DIR}/public/reuse_receipt.json}"

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

validate_nano_predictions() {
  local datasets=(
    "librispeech_test_clean:2620"
    "librispeech_test_other:2939"
    "commonvoice_en_test:16396"
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

run_full_mixer_phase() {
  local final_checkpoint_file="${FULL_OUTPUT_ROOT}/stage211a_mixer_full_data_3ep/final_checkpoint.txt"
  log "starting strict Stage211A full-data controller"
  uv run python "${REPO_ROOT}/scripts/run_stage211_full_phase_curriculum.py" \
    --phase mixer \
    --init-checkpoint "${SELECTED_CHECKPOINT}" \
    --output-root "${FULL_OUTPUT_ROOT}" \
    --config-root "${FULL_CONFIG_ROOT}" \
    --metadata-root "${METADATA_ROOT}" \
    --easy-manifest "${HOME}/rwkvasr_data/stage211_easy_source_grouped_buckets/manifest_stage211_fixed_eval.json" \
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --master-port "${MASTER_PORT}" \
    --final-checkpoint-path-output "${final_checkpoint_file}"
  log "Stage211A full curriculum finished; starting complete phase evaluation"
  uv run python "${REPO_ROOT}/scripts/finalize_stage211_phase.py" \
    --phase mixer \
    --phase-root "${FULL_OUTPUT_ROOT}/stage211a_mixer_full_data_3ep" \
    --output-dir "${PHASE_GATE_ROOT}/mixer" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --baseline-public-comparison-report "${CALIBRATION_EVAL_DIR}/public/nano_comparison.json" \
    --devices 0,1,2,3
  log "Stage211A full phase evaluation and promotion gate finished"
}

run_full_block_phase() {
  local init_checkpoint
  IFS= read -r init_checkpoint <"${FULL_OUTPUT_ROOT}/stage211a_mixer_full_data_3ep/final_checkpoint.txt"
  local promotion_receipt="${PHASE_GATE_ROOT}/mixer/mixer_promotion_receipt.json"
  local final_checkpoint_file="${FULL_OUTPUT_ROOT}/stage211b_block_full_data_3ep/final_checkpoint.txt"
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
    --master-port "$((MASTER_PORT + 1))" \
    --final-checkpoint-path-output "${final_checkpoint_file}"
  log "Stage211B full curriculum finished; starting complete phase evaluation"
  uv run python "${REPO_ROOT}/scripts/finalize_stage211_phase.py" \
    --phase block \
    --phase-root "${FULL_OUTPUT_ROOT}/stage211b_block_full_data_3ep" \
    --output-dir "${PHASE_GATE_ROOT}/block" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --baseline-public-comparison-report "${PHASE_GATE_ROOT}/mixer/nano_comparison.json" \
    --devices 0,1,2,3
  log "Stage211B full phase evaluation and promotion gate finished"
}

run_full_logits_phase() {
  local init_checkpoint
  IFS= read -r init_checkpoint <"${FULL_OUTPUT_ROOT}/stage211b_block_full_data_3ep/final_checkpoint.txt"
  local promotion_receipt="${PHASE_GATE_ROOT}/block/block_promotion_receipt.json"
  local final_checkpoint_file="${FULL_OUTPUT_ROOT}/stage211c_logits_full_data_3ep/final_checkpoint.txt"
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
    --master-port "$((MASTER_PORT + 2))" \
    --final-checkpoint-path-output "${final_checkpoint_file}"
  log "Stage211C full curriculum finished; starting complete Nano-threshold evaluation"
  uv run python "${REPO_ROOT}/scripts/finalize_stage211_phase.py" \
    --phase logits \
    --phase-root "${FULL_OUTPUT_ROOT}/stage211c_logits_full_data_3ep" \
    --output-dir "${PHASE_GATE_ROOT}/logits" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --devices 0,1,2,3
  log "Stage211C passed the complete Nano CTC gate"
}

run_labeled_sft_phase() {
  local init_checkpoint
  IFS= read -r init_checkpoint <"${FULL_OUTPUT_ROOT}/stage211c_logits_full_data_3ep/final_checkpoint.txt"
  local promotion_receipt="${PHASE_GATE_ROOT}/logits/logits_promotion_receipt.json"
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
    --nano-checkpoint "${NANO_CHECKPOINT}" \
    --master-port "$((MASTER_PORT + 3))" \
    --final-checkpoint-path-output "${final_checkpoint_file}"
  log "Stage211D labeled epoch finished; starting complete public CTC gate"
  uv run python "${REPO_ROOT}/scripts/finalize_stage211_labeled_sft.py" \
    --run-dir "${SFT_OUTPUT_DIR}" \
    --output-dir "${PHASE_GATE_ROOT}/sft" \
    --calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}" \
    --baseline-public-comparison-report "${PHASE_GATE_ROOT}/logits/nano_comparison.json" \
    --public-manifest-dir "${PUBLIC_MANIFEST_DIR}" \
    --nano-prediction-dir "${NANO_EVAL_DIR}/predictions" \
    --devices 0,1,2,3
  log "Stage211 A/B/C/D strict alignment pipeline completed"
}

main() {
  cd "${REPO_ROOT}"
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
  run_full_mixer_phase
  run_full_block_phase
  run_full_logits_phase
  run_labeled_sft_phase
}

main "$@"
