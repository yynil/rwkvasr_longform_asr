#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

MATERIALIZED_INVENTORY="${MATERIALIZED_INVENTORY:-${HOME}/rwkvasr_data/stage211_social_vad_materialized_v1/materialized_inventory.json}"
FILTERED_ROOT="${FILTERED_ROOT:-${HOME}/rwkvasr_data/stage211_social_vad_filtered_v1}"
BASE_INVENTORY="${BASE_INVENTORY:-${HOME}/rwkvasr_data/stage211_supplemental_natural_v1/supplemental_inventory.json}"
BASE_PUBLIC_OVERLAP_ROOT="${BASE_PUBLIC_OVERLAP_ROOT:-${HOME}/rwkvasr_data/stage211_base_public_pcm_overlap_v1}"
BASE_PUBLIC_ARCHIVE_CACHE_DIR="${BASE_PUBLIC_ARCHIVE_CACHE_DIR:-${TMPDIR:-/tmp}/rwkvasr_stage211_base_public_pcm_cache}"
BASE_PUBLIC_DECODE_WORKERS="${BASE_PUBLIC_DECODE_WORKERS:-4}"
BASE_PUBLIC_PREFETCH_NEXT_ARCHIVE="${BASE_PUBLIC_PREFETCH_NEXT_ARCHIVE:-0}"
COMBINED_ROOT="${COMBINED_ROOT:-${HOME}/rwkvasr_data/stage211_supplemental_combined_v2}"
RETENTION_OUTPUT_ROOT="${RETENTION_OUTPUT_ROOT:-${HOME}/rwkvasr_data/stage211_full_curriculum}"
USB_COVERAGE_RECEIPT="${USB_COVERAGE_RECEIPT:-${HOME}/rwkvasr_data/stage211_usb_top_level_coverage_v1/coverage_receipt.json}"
ARCHIVED_SOCIAL_OVERLAP_RECEIPT="${ARCHIVED_SOCIAL_OVERLAP_RECEIPT:-${HOME}/rwkvasr_data/stage211_archived_social_overlap_v1/overlap_receipt.json}"
ARCHIVED_SOCIAL_OVERLAP_SESSION="${ARCHIVED_SOCIAL_OVERLAP_SESSION:-rwkvasr_stage211_archived_social_overlap}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
QUOTE_REPAIR_HANDOFF_SCRIPT="${QUOTE_REPAIR_HANDOFF_SCRIPT:-scripts/build_stage211_quote_repair_handoff.sh}"

cd "${REPO_ROOT}"

wait_for_artifact() {
  local label="$1"
  local path="$2"
  while [[ ! -s "${path}" ]]; do
    printf 'Stage211 %s unavailable; waiting %ss: %s\n' \
      "${label}" "${POLL_SECONDS}" "${path}"
    sleep "${POLL_SECONDS}"
  done
}

BASE_PUBLIC_PREFETCH_ARGS=()
case "${BASE_PUBLIC_PREFETCH_NEXT_ARCHIVE}" in
  1|true|True|TRUE|yes|Yes|YES)
    BASE_PUBLIC_PREFETCH_ARGS+=(--prefetch-next-archive)
    ;;
esac

while tmux has-session -t "${ARCHIVED_SOCIAL_OVERLAP_SESSION}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done
if [[ ! -s "${USB_COVERAGE_RECEIPT}" ]]; then
  echo "Stage211 USB top-level coverage receipt is unavailable: ${USB_COVERAGE_RECEIPT}" >&2
  exit 1
fi
if [[ ! -s "${ARCHIVED_SOCIAL_OVERLAP_RECEIPT}" ]]; then
  echo "Stage211 archived-social overlap receipt is unavailable: ${ARCHIVED_SOCIAL_OVERLAP_RECEIPT}" >&2
  exit 1
fi
wait_for_artifact "base supplemental inventory" "${BASE_INVENTORY}"

nice -n 10 ionice -c 2 -n 7 env CUDA_VISIBLE_DEVICES='' uv run python \
  scripts/audit_stage211_base_public_pcm_overlap.py \
  --base-inventory "${BASE_INVENTORY}" \
  --output-root "${BASE_PUBLIC_OVERLAP_ROOT}" \
  all \
  --archive-cache-dir "${BASE_PUBLIC_ARCHIVE_CACHE_DIR}" \
  --decode-workers "${BASE_PUBLIC_DECODE_WORKERS}" \
  "${BASE_PUBLIC_PREFETCH_ARGS[@]}"

exec env \
  LEGACY_SESSION=rwkvasr_stage211_no_legacy_session \
  BASE_INVENTORY="${BASE_INVENTORY}" \
  BASE_SOURCE_AUDIT="${BASE_PUBLIC_OVERLAP_ROOT}/audit_receipt.json" \
  USB_COVERAGE_RECEIPT="${USB_COVERAGE_RECEIPT}" \
  ARCHIVED_SOCIAL_OVERLAP_RECEIPT="${ARCHIVED_SOCIAL_OVERLAP_RECEIPT}" \
  RETENTION_OUTPUT_ROOT="${RETENTION_OUTPUT_ROOT}" \
  bash "${QUOTE_REPAIR_HANDOFF_SCRIPT}"
