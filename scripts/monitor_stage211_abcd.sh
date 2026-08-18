#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUPERVISOR_SESSION="${SUPERVISOR_SESSION:-rwkvasr_stage211_abcd_strict_supervisor}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/rwkvasr_runs/stage211_full_alignment}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${HOME}/rwkvasr_runs/sensevoice_rwkv_stage211a_recovery_stage210a30000_nanomlpfrozen_teacherforced_mixeronly_easy1490h_1ep_lr3e6_wd0_4x4090}"
CALIBRATION_EVAL_ROOT="${CALIBRATION_EVAL_ROOT:-${HOME}/rwkvasr_eval/stage211_calibration_selected_full}"
CORRECTED_PUBLIC_ROOT="${CORRECTED_PUBLIC_ROOT:-${HOME}/rwkvasr_eval/stage211_public_clean_v2}"
PHASE_GATE_ROOT="${PHASE_GATE_ROOT:-${HOME}/rwkvasr_eval/stage211_phase_gates}"
MONITOR_LOG="${MONITOR_LOG:-${OUTPUT_ROOT}/monitor_hourly.log}"
BASE_PUBLIC_OVERLAP_ROOT="${BASE_PUBLIC_OVERLAP_ROOT:-${HOME}/rwkvasr_data/stage211_base_public_pcm_overlap_v2}"
SUPPLEMENTAL_PROGRESS_REPORTER="${SUPPLEMENTAL_PROGRESS_REPORTER:-${REPO_ROOT}/scripts/report_stage211_base_public_pcm_progress.py}"
SOCIAL_PCM_INVENTORY="${SOCIAL_PCM_INVENTORY:-${HOME}/rwkvasr_data/stage211_social_vad_materialized_v1/materialized_inventory.json}"
SOCIAL_PCM_SOURCE_OUTPUT_ROOT="${SOCIAL_PCM_SOURCE_OUTPUT_ROOT:-${HOME}/rwkvasr_data/stage211_social_vad_filtered_v1}"
SOCIAL_PCM_OUTPUT_ROOT="${SOCIAL_PCM_OUTPUT_ROOT:-${HOME}/rwkvasr_data/stage211_social_vad_filtered_v2}"
SOCIAL_PCM_PROGRESS_REPORTER="${SOCIAL_PCM_PROGRESS_REPORTER:-${REPO_ROOT}/scripts/report_stage211_social_pcm_progress.py}"
SFT_LABELED_ROOT="${SFT_LABELED_ROOT:-${HOME}/rwkvasr_data/stage211_sft_full_labeled_v2}"
SFT_PREPARATION_LOG="${SFT_PREPARATION_LOG:-${SFT_LABELED_ROOT}/prepare_ctc_aligned.log}"
SFT_FINALIZER_LOG="${SFT_FINALIZER_LOG:-${SFT_LABELED_ROOT}_finalize.log}"
SFT_PROFILE_RECEIPT="${SFT_PROFILE_RECEIPT:-${SFT_LABELED_ROOT}/stage211_labeled_profile_receipt.json}"
SFT_EXPECTED_INPUT_SAMPLES="${SFT_EXPECTED_INPUT_SAMPLES:-1430801}"
SUPPLEMENTAL_COMBINED_ROOT="${SUPPLEMENTAL_COMBINED_ROOT:-${HOME}/rwkvasr_data/stage211_supplemental_combined_v4_locality}"
SUPPLEMENTAL_COMBINED_INVENTORY="${SUPPLEMENTAL_COMBINED_INVENTORY:-${SUPPLEMENTAL_COMBINED_ROOT}/supplemental_inventory.json}"
SUPPLEMENTAL_COMBINED_PROFILE="${SUPPLEMENTAL_COMBINED_PROFILE:-${SUPPLEMENTAL_COMBINED_ROOT}/supplemental_profile_receipt.json}"
SUPPLEMENTAL_NINE_CELL_RECEIPT="${SUPPLEMENTAL_NINE_CELL_RECEIPT:-${HOME}/rwkvasr_data/stage211_full_curriculum/stratified_hidden_eval_v3/receipt.json}"
SUPPLEMENTAL_REPLAY_RECEIPT="${SUPPLEMENTAL_REPLAY_RECEIPT:-${HOME}/rwkvasr_data/stage211_full_curriculum/retention_replay_v3/receipt.json}"
MONITOR_PYTHON="${MONITOR_PYTHON:-${REPO_ROOT}/.venv/bin/python3}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
RECENT_LOG_MINUTES="${RECENT_LOG_MINUTES:-90}"
MONITOR_ONCE="${MONITOR_ONCE:-0}"
FIXED_EVAL_CANONICAL_PART="${FIXED_EVAL_CANONICAL_PART:-${HOME}/rwkvasr_data/stage211_full_curriculum/fixed_hidden_eval/part_000000.jsonl}"
FIXED_EVAL_CANONICAL_PART_SHA256="${FIXED_EVAL_CANONICAL_PART_SHA256:-9f4bf09cdbbf5bc963d6282a640e45fcd8da75f2e23a41e4515b6766288c84f1}"

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

stage211_latest_step_eval_report() {
  local run_dir="$1"
  local report name step latest=0 latest_path=""
  shopt -s nullglob
  for report in "${run_dir}"/step_eval_layers_step-*.yaml; do
    name="${report##*/}"
    if [[ "${name}" =~ ^step_eval_layers_step-([0-9]+)\.yaml$ ]]; then
      step="${BASH_REMATCH[1]}"
      if ((10#${step} > latest)); then
        latest=$((10#${step}))
        latest_path="${report}"
      fi
    fi
  done
  shopt -u nullglob
  printf '%s\n' "${latest_path}"
}

stage211_fixed_eval_scope_details() {
  local baseline_path="$1"
  local binding part_path recorded_sha actual_sha
  binding="$(awk '
    /^[[:space:]]*- path:/ {
      path = $0
      sub(/^[[:space:]]*- path:[[:space:]]*/, "", path)
      getline
      sha = $0
      sub(/^[[:space:]]*sha256:[[:space:]]*/, "", sha)
      print path "\t" sha
      exit
    }
  ' "${baseline_path}" 2>/dev/null)"
  IFS=$'\t' read -r part_path recorded_sha <<<"${binding}"
  if [[ "${part_path:-}" == "${FIXED_EVAL_CANONICAL_PART}" ]] &&
    [[ "${recorded_sha:-}" == "${FIXED_EVAL_CANONICAL_PART_SHA256}" ]] &&
    [[ -s "${part_path}" ]]; then
    actual_sha="$(sha256sum "${part_path}" 2>/dev/null | awk '{print $1}')"
    if [[ "${actual_sha}" == "${recorded_sha}" ]]; then
      printf '%s\n' \
        'sentinel_cell=easy_zh sentinel_language=zh sentinel_sources=aishell3,commonvoice_cn'
      return
    fi
  fi
  printf '%s\n' 'sentinel_cell=unknown sentinel_language=unknown sentinel_sources=unknown'
}

stage211_emit_fixed_step_eval_progress() {
  local run_dir="$1"
  local baseline_path="${run_dir}/step_eval_baseline.yaml"
  local latest_path baseline_loss latest_step latest_loss eval_samples filename_step scope_details
  if [[ ! -s "${baseline_path}" ]]; then
    printf 'fixed_eval_scope=fixed_hidden_not_public_wer_cer status=baseline_unavailable sentinel_cell=unknown sentinel_language=unknown sentinel_sources=unknown\n'
    return
  fi
  scope_details="$(stage211_fixed_eval_scope_details "${baseline_path}")"
  baseline_loss="$(stage211_yaml_scalar "${baseline_path}" eval_loss)"
  latest_path="$(stage211_latest_step_eval_report "${run_dir}")"
  if [[ -z "${latest_path}" ]]; then
    printf 'fixed_eval_scope=fixed_hidden_not_public_wer_cer status=latest_unavailable %s baseline_loss=%s\n' \
      "${scope_details}" "${baseline_loss:-invalid}"
    return
  fi
  latest_step="$(stage211_yaml_scalar "${latest_path}" step)"
  latest_loss="$(stage211_yaml_scalar "${latest_path}" eval_loss)"
  eval_samples="$(stage211_yaml_scalar "${latest_path}" eval_samples)"
  filename_step="${latest_path##*step-}"
  filename_step="${filename_step%.yaml}"
  if [[ ! "${baseline_loss}" =~ ^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] ||
    [[ ! "${latest_step}" =~ ^[0-9]+$ ]] ||
    [[ ! "${latest_loss}" =~ ^[-+]?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] ||
    [[ ! "${eval_samples}" =~ ^[0-9]+$ ]] ||
    [[ "${latest_step}" != "${filename_step}" ]]; then
    printf 'fixed_eval_scope=fixed_hidden_not_public_wer_cer status=invalid %s report=%s baseline_loss=%s recorded_step=%s filename_step=%s latest_loss=%s eval_samples=%s\n' \
      "${scope_details}" "${latest_path}" "${baseline_loss:-missing}" "${latest_step:-missing}" \
      "${filename_step:-missing}" "${latest_loss:-missing}" "${eval_samples:-missing}"
    return
  fi
  awk \
    -v baseline="${baseline_loss}" \
    -v latest="${latest_loss}" \
    -v step="${latest_step}" \
    -v samples="${eval_samples}" \
    -v report="${latest_path}" \
    -v scope_details="${scope_details}" '
      BEGIN {
        delta = latest - baseline
        tolerance = 1e-12
        trend = delta < -tolerance ? "improved" : (delta > tolerance ? "regressed" : "flat")
        if (baseline == 0) {
          printf "fixed_eval_scope=fixed_hidden_not_public_wer_cer status=ok %s baseline_loss=%.10f latest_step=%d latest_loss=%.10f delta_abs=%+.10f delta_pct=undefined trend=%s eval_samples=%d report=%s\n", scope_details, baseline, step, latest, delta, trend, samples, report
        } else {
          printf "fixed_eval_scope=fixed_hidden_not_public_wer_cer status=ok %s baseline_loss=%.10f latest_step=%d latest_loss=%.10f delta_abs=%+.10f delta_pct=%+.4f trend=%s eval_samples=%d report=%s\n", scope_details, baseline, step, latest, delta, 100.0 * delta / baseline, trend, samples, report
        }
      }
    '
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
  stage211_emit_fixed_step_eval_progress "${run_dir}"
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

stage211_emit_social_pcm_progress() {
  local output_root="${1:-${SOCIAL_PCM_OUTPUT_ROOT}}"
  if [[ -x "${MONITOR_PYTHON}" && -f "${SOCIAL_PCM_PROGRESS_REPORTER}" && \
    -s "${SOCIAL_PCM_INVENTORY}" ]]; then
    nice -n 10 "${MONITOR_PYTHON}" "${SOCIAL_PCM_PROGRESS_REPORTER}" \
      --materialized-inventory "${SOCIAL_PCM_INVENTORY}" \
      --output-root "${output_root}" 2>&1 || true
  else
    printf 'unavailable inventory=%s output_root=%s reporter=%s python=%s\n' \
      "${SOCIAL_PCM_INVENTORY}" "${output_root}" \
      "${SOCIAL_PCM_PROGRESS_REPORTER}" "${MONITOR_PYTHON}"
  fi
}

stage211_emit_social_pcm_readiness() {
  printf '%s\n' '-- social source exact PCM fingerprints --'
  stage211_emit_social_pcm_progress "${SOCIAL_PCM_SOURCE_OUTPUT_ROOT}"
  printf '%s\n' '-- social corrected-public v2 rebase --'
  stage211_emit_social_pcm_progress "${SOCIAL_PCM_OUTPUT_ROOT}"
}

stage211_emit_artifact_status() {
  local label="$1"
  local path="$2"
  local sha256
  if [[ -s "${path}" ]]; then
    sha256="$(sha256sum "${path}" 2>/dev/null | awk '{print $1}')"
    printf '%s=ready path=%s sha256=%s\n' "${label}" "${path}" "${sha256:-unavailable}"
  else
    printf '%s=pending path=%s\n' "${label}" "${path}"
  fi
}

stage211_emit_sft_readiness() {
  local latest_record processed=0 kept=0 progress state finalizer_record
  latest_record="$(
    rg 'ctc-align (progress|lengths complete) processed=[0-9]+ kept=[0-9]+' \
      "${SFT_PREPARATION_LOG}" 2>/dev/null | tail -n 1 || true
  )"
  if [[ -n "${latest_record}" ]]; then
    processed="$(sed -nE 's/.*processed=([0-9]+).*/\1/p' <<<"${latest_record}")"
    kept="$(sed -nE 's/.*kept=([0-9]+).*/\1/p' <<<"${latest_record}")"
  fi
  processed="${processed:-0}"
  kept="${kept:-0}"
  if rg -q -F 'CTC-aligned clean preprocessing complete' "${SFT_PREPARATION_LOG}" 2>/dev/null; then
    state=complete
    processed="${SFT_EXPECTED_INPUT_SAMPLES}"
  elif pgrep -f "build_ctc_aligned_stage_lengths.py.*--output-dir ${SFT_LABELED_ROOT}" \
    >/dev/null 2>&1; then
    state=active
  else
    state=incomplete
  fi
  progress="$(
    awk -v processed="${processed}" -v expected="${SFT_EXPECTED_INPUT_SAMPLES}" \
      'BEGIN { printf "%.4f", (expected > 0 ? 100.0 * processed / expected : 0.0) }'
  )"
  printf 'sft_labeled_preparation=%s processed=%s expected=%s progress_pct=%s kept=%s log=%s\n' \
    "${state}" "${processed}" "${SFT_EXPECTED_INPUT_SAMPLES}" "${progress}" "${kept}" \
    "${SFT_PREPARATION_LOG}"
  stage211_emit_artifact_status sft_labeled_profile "${SFT_PROFILE_RECEIPT}"
  finalizer_record="$(tail -n 1 "${SFT_FINALIZER_LOG}" 2>/dev/null || true)"
  if [[ -s "${SFT_PROFILE_RECEIPT}" && -s "${SFT_FINALIZER_LOG}" \
      && "${SFT_PROFILE_RECEIPT}" -nt "${SFT_FINALIZER_LOG}" \
      && "${finalizer_record}" =~ (Error|error|failed|Traceback) ]]; then
    finalizer_record="profile validated stale_failure_superseded=true"
  fi
  printf 'sft_labeled_finalizer=%s\n' "${finalizer_record:-pending}"
}

stage211_emit_supplemental_readiness() {
  stage211_emit_artifact_status supplemental_combined_inventory \
    "${SUPPLEMENTAL_COMBINED_INVENTORY}"
  stage211_emit_artifact_status supplemental_combined_profile \
    "${SUPPLEMENTAL_COMBINED_PROFILE}"
  stage211_emit_artifact_status supplemental_nine_cell_receipt \
    "${SUPPLEMENTAL_NINE_CELL_RECEIPT}"
  stage211_emit_artifact_status supplemental_replay_receipt \
    "${SUPPLEMENTAL_REPLAY_RECEIPT}"
}

stage211_public_expected_rows() {
  case "$1" in
    librispeech_test_clean) printf '%s\n' 2620 ;;
    librispeech_test_other) printf '%s\n' 2939 ;;
    commonvoice_en_test) printf '%s\n' 14927 ;;
    aishell1_test) printf '%s\n' 7176 ;;
    wenetspeech_test_net) printf '%s\n' 24774 ;;
    *) return 1 ;;
  esac
}

stage211_emit_public_prediction_coverage() {
  local role="$1"
  local prediction_path="$2"
  local filename dataset expected_rows actual_rows status
  filename="${prediction_path##*/}"
  dataset="${filename%.ctc.jsonl}"
  expected_rows="$(stage211_public_expected_rows "${dataset}" 2>/dev/null || true)"
  if [[ -z "${expected_rows}" ]]; then
    actual_rows=0
    if [[ -f "${prediction_path}" ]]; then
      actual_rows="$(wc -l <"${prediction_path}" | tr -d ' ')"
    fi
    printf 'public_prediction role=%s dataset=%s rows=%s expected_rows=unknown status=unexpected path=%s\n' \
      "${role}" "${dataset}" "${actual_rows}" "${prediction_path}"
    return
  fi
  if [[ ! -f "${prediction_path}" ]]; then
    printf 'public_prediction role=%s dataset=%s rows=0 expected_rows=%s status=pending path=%s\n' \
      "${role}" "${dataset}" "${expected_rows}" "${prediction_path}"
    return
  fi
  actual_rows="$(wc -l <"${prediction_path}" | tr -d ' ')"
  status=mismatch
  if [[ "${actual_rows}" == "${expected_rows}" ]]; then
    status=complete
  fi
  printf 'public_prediction role=%s dataset=%s rows=%s expected_rows=%s status=%s path=%s\n' \
    "${role}" "${dataset}" "${actual_rows}" "${expected_rows}" "${status}" \
    "${prediction_path}"
}

stage211_emit_public_evaluation_coverage() {
  local dataset prediction_path role root
  local datasets=(
    librispeech_test_clean
    librispeech_test_other
    commonvoice_en_test
    aishell1_test
    wenetspeech_test_net
  )
  for role in corrected_calibration corrected_nano; do
    if [[ "${role}" == corrected_calibration ]]; then
      root="${CORRECTED_PUBLIC_ROOT}/calibration/predictions"
    else
      root="${CORRECTED_PUBLIC_ROOT}/nano_2512/predictions"
    fi
    for dataset in "${datasets[@]}"; do
      stage211_emit_public_prediction_coverage \
        "${role}" "${root}/${dataset}.ctc.jsonl"
    done
  done
  while IFS= read -r -d '' prediction_path; do
    stage211_emit_public_prediction_coverage phase_gate "${prediction_path}"
  done < <(
    find "${PHASE_GATE_ROOT}" -type f -name '*.ctc.jsonl' -print0 2>/dev/null |
      sort -z
  )
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
    case "${log_path}" in
      */batch_profile_preflight/*|*/?*.failed_*/*|*/?*.nonrepresentative_*/*)
        continue
        ;;
    esac
    if rg -q -F "${ATTEMPT_MARKER}" "${log_path}" 2>/dev/null; then
      printf '%s\0' "${log_path}"
    fi
  done < <(stage211_recent_logs "$@")
}

stage211_emit_snapshot() {
  local log_path latest_record errors
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
  printf '%s\n' '-- corrected public evaluation coverage --'
  stage211_emit_public_evaluation_coverage
  printf '%s\n' '-- supplemental exact PCM audit --'
  if [[ -x "${MONITOR_PYTHON}" && -f "${SUPPLEMENTAL_PROGRESS_REPORTER}" && \
    -s "${BASE_PUBLIC_OVERLAP_ROOT}/manifest_location_index.sqlite" ]]; then
    nice -n 10 "${MONITOR_PYTHON}" "${SUPPLEMENTAL_PROGRESS_REPORTER}" \
      --output-root "${BASE_PUBLIC_OVERLAP_ROOT}" 2>&1 || true
  else
    printf 'unavailable root=%s reporter=%s python=%s\n' \
      "${BASE_PUBLIC_OVERLAP_ROOT}" "${SUPPLEMENTAL_PROGRESS_REPORTER}" "${MONITOR_PYTHON}"
  fi
  stage211_emit_social_pcm_readiness
  printf '%s\n' '-- supplemental handoff readiness --'
  stage211_emit_supplemental_readiness
  printf '%s\n' '-- full labeled SFT readiness --'
  stage211_emit_sft_readiness
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
