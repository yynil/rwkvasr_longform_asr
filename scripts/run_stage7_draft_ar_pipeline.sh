#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CTC_RUN_DIR="${CTC_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage7_large_ctc_clean16_hard84_from_stage6c_step5000_bs12_lr3e5_zero1_nockpt_4x4090}"
DRAFT_CACHE_DIR="${DRAFT_CACHE_DIR:-${CTC_RUN_DIR}/ctc_draft_cache_stage7_full}"
PIPELINE_LOG="${PIPELINE_LOG:-${DRAFT_CACHE_DIR}/logs/pipeline.log}"

export DRAFT_SOURCE_SPECS="${DRAFT_SOURCE_SPECS:-giga_00000_00006:GSXL-0000[0-6]*-*.tar,giga_00010_00016:GSXL-0001[0-6]*-*.tar,giga_00020_00025:GSXL-0002[0-5]*-*.tar,wenet_00000_00029:WSL-000[0-2]*-*.tar,wenet_00030_00059:WSL-000[3-5]*-*.tar,wenet_00060_00089:WSL-000[6-8]*-*.tar,wenet_00120_00146:WSL-001[2-4]*-*.tar,cv_en:commonvoice_en_*.tar,wenet_00100_00119:WSL-001[0-1]*-*.tar,giga_00007_00009:GSXL-0000[7-9]*-*.tar,giga_00017_00019:GSXL-0001[7-9]*-*.tar,cv_cn:commonvoice_cn_*.tar,aishell:aishell3_*.tar,wenet_00090_00099:WSL-0009*-*.tar,libri:librispeech_*.tar}"
export DRAFT_DEVICES="${DRAFT_DEVICES:-0,1,2,3}"
export DRAFT_MAX_PARALLEL="${DRAFT_MAX_PARALLEL:-4}"
export DRAFT_BATCH_SIZE="${DRAFT_BATCH_SIZE:-8}"
export DRAFT_NUM_WORKERS="${DRAFT_NUM_WORKERS:-4}"
export DRAFT_SKIP_DECODE_ERRORS="${DRAFT_SKIP_DECODE_ERRORS:-1}"
export DRAFT_PROGRESS_INTERVAL="${DRAFT_PROGRESS_INTERVAL:-200}"

log() {
  printf '[stage7-pipeline] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

main() {
  cd "${REPO_ROOT}"
  mkdir -p "$(dirname "${PIPELINE_LOG}")"
  log "starting full CTC draft generation" | tee -a "${PIPELINE_LOG}"
  set +e
  "${REPO_ROOT}/scripts/build_ctc_draft_cache_from_length_index.sh" 2>&1 | tee -a "${PIPELINE_LOG}"
  local draft_rc="${PIPESTATUS[0]}"
  set -e
  log "draft generation exited with status ${draft_rc}" | tee -a "${PIPELINE_LOG}"
  if [[ "${draft_rc}" != "0" ]]; then
    exit "${draft_rc}"
  fi

  log "starting stage7 draft-conditioned CTC+AR training handoff" | tee -a "${PIPELINE_LOG}"
  "${REPO_ROOT}/scripts/start_stage7_ctc_draft_ar_training.sh" 2>&1 | tee -a "${PIPELINE_LOG}"
  log "pipeline handoff complete" | tee -a "${PIPELINE_LOG}"
}

main "$@"
