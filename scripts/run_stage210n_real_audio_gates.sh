#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${RUN_DIR:-/tmp/rwkvasr_runs/sensevoice_rwkv_stage210n_stage210m1000_sourcebalanced_easy1490h_1ep_sequence0p2_condwindow0p25_stackedhard12_lr3e7_wd0_4x4090}"
EVAL_ROOT="${EVAL_ROOT:-/tmp/rwkvasr_eval}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/artifacts/eval_benchmarks/beam_ablation_20260620/manifests_200_seed20260620}"
WAIT_SESSION="${WAIT_SESSION:-rwkvasr_stage210n_full_easy}"
STEPS="${STEPS:-10000 20000 30000 30064}"
DEVICES="${DEVICES:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-300}"
DRY_RUN="${DRY_RUN:-0}"

NANO_AISHELL="${NANO_AISHELL:-${EVAL_ROOT}/stage207_step690k_public200/aishell1_test.nano.jsonl}"
NANO_LIBRI_CLEAN="${NANO_LIBRI_CLEAN:-${EVAL_ROOT}/stage207_step690k_public200/librispeech_test_clean.nano.jsonl}"
NANO_LIBRI_OTHER="${NANO_LIBRI_OTHER:-${EVAL_ROOT}/stage207_step690k_public200/librispeech_test_other.nano.jsonl}"
NANO_COMMONVOICE="${NANO_COMMONVOICE:-${EVAL_ROOT}/funasr_nano_ctc_public200/commonvoice_en_test.nano.jsonl}"
NANO_WENET="${NANO_WENET:-${EVAL_ROOT}/funasr_nano_ctc_public200/wenetspeech_test_net.nano.jsonl}"

log() {
  printf '[stage210n-real-gate] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

validate_inputs() {
  local path
  for path in \
    "${MANIFEST_DIR}/manifest_summary.json" \
    "${NANO_AISHELL}" \
    "${NANO_LIBRI_CLEAN}" \
    "${NANO_LIBRI_OTHER}" \
    "${NANO_COMMONVOICE}" \
    "${NANO_WENET}"; do
    if [[ ! -s "${path}" ]]; then
      echo "Required real-audio gate input missing: ${path}" >&2
      exit 1
    fi
  done
}

wait_for_training() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: would wait for tmux session ${WAIT_SESSION}"
    return
  fi
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    local latest_step
    latest_step="$(
      awk '/\[deepspeed-train\] step=/{line=$0} END{
        for (i=1;i<=split(line, fields, " ");i++) {
          if (fields[i] ~ /^step=/) {
            print fields[i]
            exit
          }
        }
      }' "${RUN_DIR}/logs/full_easy_30064steps.log" 2>/dev/null || true
    )"
    log "training still active ${latest_step:-step=unknown}; waiting ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
  log "training session released all GPUs"
}

run_gate() {
  local step="$1"
  local checkpoint="${RUN_DIR}/step-${step}.pt"
  local output_dir="${EVAL_ROOT}/stage210n_step${step}_seeded_public200"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: checkpoint=${checkpoint} output=${output_dir}"
    return
  fi
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Expected Stage210n checkpoint missing: ${checkpoint}" >&2
    exit 1
  fi

  log "public real-audio decode start step=${step}"
  env \
    CHECKPOINT_PATH="${checkpoint}" \
    OUTPUT_DIR="${output_dir}" \
    MANIFEST_DIR="${MANIFEST_DIR}" \
    PREPARE_MANIFESTS=0 \
    DEVICES="${DEVICES}" \
    CTC_BATCH_SIZE=4 \
    CTC_NUM_WORKERS=0 \
    CTC_LIMIT=0 \
    RUN_AR=0 \
    METRIC_NORMALIZATION=ctc \
    bash "${REPO_ROOT}/scripts/run_public_eval_benchmarks.sh"

  PYTHONPATH="${REPO_ROOT}/src" uv run python \
    "${REPO_ROOT}/scripts/compare_public_ctc_with_nano.py" \
    --student-prediction-dir "${output_dir}/predictions" \
    --nano-prediction "aishell1_test=${NANO_AISHELL}" \
    --nano-prediction "librispeech_test_clean=${NANO_LIBRI_CLEAN}" \
    --nano-prediction "librispeech_test_other=${NANO_LIBRI_OTHER}" \
    --nano-prediction "commonvoice_en_test=${NANO_COMMONVOICE}" \
    --nano-prediction "wenetspeech_test_net=${NANO_WENET}" \
    --output-json "${output_dir}/nano_real_audio_comparison.json" \
    --output-md "${output_dir}/nano_real_audio_comparison.md" \
    --normalization ctc \
    --max-relative-ratio 1.20 \
    --max-absolute-gap-points 3.0
  log "public real-audio gate done step=${step} report=${output_dir}/nano_real_audio_comparison.md"
}

main() {
  cd "${REPO_ROOT}"
  validate_inputs
  wait_for_training
  local step
  for step in ${STEPS}; do
    run_gate "${step}"
  done
  log "all requested Stage210n real-audio gates complete"
}

main "$@"
