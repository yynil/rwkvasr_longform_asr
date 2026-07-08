#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AR_RUN_DIR="${AR_RUN_DIR:?Set AR_RUN_DIR to the Stage11 run directory.}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
DEVICES="${DEVICES:-0,1,2,3}"
RUN_AR="${RUN_AR:-auto}"

log() {
  printf '[stage11-public-eval] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

cd "${REPO_ROOT}"
while true; do
  log "checking run_dir=${AR_RUN_DIR}"
  if grep -q 'draft-conditioned CTC+AR training exited with status 0' "${AR_RUN_DIR}/logs/training.ansi.log" 2>/dev/null; then
    if [[ -s "${AR_RUN_DIR}/joint_wercer_best.pt" ]]; then
      timestamp="$(date '+%Y%m%d_%H%M%S')"
      output_dir="${AR_RUN_DIR}/public_eval_${timestamp}"
      log "Stage11 completed and joint_wercer_best.pt exists; starting public benchmark output=${output_dir}"
      RUN_DIR="${AR_RUN_DIR}" \
        OUTPUT_DIR="${output_dir}" \
        DEVICES="${DEVICES}" \
        RUN_AR="${RUN_AR}" \
        "${REPO_ROOT}/scripts/run_public_eval_benchmarks.sh"
      exit $?
    fi
    log "training complete; waiting for joint selector best"
  else
    log "waiting for Stage11 normal training completion"
  fi
  sleep "${POLL_SECONDS}"
done
