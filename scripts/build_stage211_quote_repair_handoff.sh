#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
LEGACY_SESSION="${LEGACY_SESSION:-rwkvasr_stage211_social_combined_postmaterialization}"

BASE_INVENTORY="${BASE_INVENTORY:-${HOME}/rwkvasr_data/stage211_supplemental_natural_v1/supplemental_inventory.json}"
BASE_SOURCE_AUDIT="${BASE_SOURCE_AUDIT:-${HOME}/rwkvasr_data/stage211_base_public_pcm_overlap_v1/audit_receipt.json}"
BASE_REBASED_ROOT="${BASE_REBASED_ROOT:-${HOME}/rwkvasr_data/stage211_base_public_pcm_overlap_v2}"
SOCIAL_SOURCE_INVENTORY="${SOCIAL_SOURCE_INVENTORY:-${HOME}/rwkvasr_data/stage211_social_vad_filtered_v1/filtered_inventory.json}"
SOCIAL_REBASED_ROOT="${SOCIAL_REBASED_ROOT:-${HOME}/rwkvasr_data/stage211_social_vad_filtered_v2}"
PUBLIC_CLEAN_ROOT="${PUBLIC_CLEAN_ROOT:-${HOME}/rwkvasr_eval/stage211_public_clean_v2}"
PUBLIC_OVERLAP_RECEIPT="${PUBLIC_OVERLAP_RECEIPT:-${HOME}/rwkvasr_data/stage211_full_curriculum/public_train_overlap_v2/receipt.json}"
PUBLIC_METRIC_ROOT="${PUBLIC_METRIC_ROOT:-${HOME}/rwkvasr_eval/stage211_public_metric_unicode_v2}"
COMBINED_ROOT="${COMBINED_ROOT:-${HOME}/rwkvasr_data/stage211_supplemental_combined_v3}"
RETENTION_OUTPUT_ROOT="${RETENTION_OUTPUT_ROOT:-${HOME}/rwkvasr_data/stage211_full_curriculum}"
USB_COVERAGE_RECEIPT="${USB_COVERAGE_RECEIPT:-${HOME}/rwkvasr_data/stage211_usb_top_level_coverage_v1/coverage_receipt.json}"
ARCHIVED_SOCIAL_OVERLAP_RECEIPT="${ARCHIVED_SOCIAL_OVERLAP_RECEIPT:-${HOME}/rwkvasr_data/stage211_archived_social_overlap_v1/overlap_receipt.json}"

cd "${REPO_ROOT}"

wait_for_artifact() {
  local label="$1"
  local path="$2"
  while [[ ! -s "${path}" ]]; do
    printf '[stage211-quote-repair-handoff] waiting %ss label=%s path=%s\n' \
      "${POLL_SECONDS}" "${label}" "${path}"
    sleep "${POLL_SECONDS}"
  done
}

wait_for_artifact "legacy base PCM audit" "${BASE_SOURCE_AUDIT}"
while tmux has-session -t "${LEGACY_SESSION}" 2>/dev/null; do
  printf '[stage211-quote-repair-handoff] waiting %ss legacy_session=%s\n' \
    "${POLL_SECONDS}" "${LEGACY_SESSION}"
  sleep "${POLL_SECONDS}"
done

env CUDA_VISIBLE_DEVICES='' nice -n 10 ionice -c 2 -n 7 uv run python \
  scripts/rebase_stage211_base_public_pcm_overlap.py \
  --base-inventory "${BASE_INVENTORY}" \
  --source-audit-receipt "${BASE_SOURCE_AUDIT}" \
  --output-root "${BASE_REBASED_ROOT}"

env CUDA_VISIBLE_DEVICES='' nice -n 10 ionice -c 2 -n 7 uv run python \
  scripts/rebase_stage211_social_public_pcm_overlap.py \
  --source-filtered-inventory "${SOCIAL_SOURCE_INVENTORY}" \
  --output-root "${SOCIAL_REBASED_ROOT}"

env CUDA_VISIBLE_DEVICES='' uv run python scripts/install_stage211_clean_public_eval.py \
  --clean-root "${PUBLIC_CLEAN_ROOT}" \
  --overlap-receipt "${PUBLIC_OVERLAP_RECEIPT}"

env CUDA_VISIBLE_DEVICES='' uv run python scripts/install_stage211_unicode_metric_correction.py \
  --output-root "${PUBLIC_METRIC_ROOT}" \
  --prior-install-receipt "${PUBLIC_CLEAN_ROOT}/canonical_install_receipt.json"

env CUDA_VISIBLE_DEVICES='' uv run python \
  scripts/build_stage211_combined_supplemental_inventory.py \
  --base-inventory "${BASE_INVENTORY}" \
  --base-public-overlap-audit "${BASE_REBASED_ROOT}/audit_receipt.json" \
  --social-inventory "${SOCIAL_REBASED_ROOT}/filtered_inventory.json" \
  --usb-coverage-receipt "${USB_COVERAGE_RECEIPT}" \
  --archived-social-overlap-receipt "${ARCHIVED_SOCIAL_OVERLAP_RECEIPT}" \
  --output-root "${COMBINED_ROOT}"

env CUDA_VISIBLE_DEVICES='' uv run python \
  scripts/create_stage211_supplemental_profile_receipt.py \
  --inventory "${COMBINED_ROOT}/supplemental_inventory.json" \
  --output "${COMBINED_ROOT}/supplemental_profile_receipt.json"

env CUDA_VISIBLE_DEVICES='' nice -n 10 ionice -c 2 -n 7 uv run python \
  scripts/build_stage211_supplemental_retention.py \
  --supplemental-inventory "${COMBINED_ROOT}/supplemental_inventory.json" \
  --supplemental-profile-receipt "${COMBINED_ROOT}/supplemental_profile_receipt.json" \
  --output-root "${RETENTION_OUTPUT_ROOT}" \
  --stratified-dir-name stratified_hidden_eval_v3 \
  --replay-dir-name retention_replay_v3

env CUDA_VISIBLE_DEVICES='' nice -n 10 ionice -c 2 -n 7 uv run python \
  scripts/validate_stage211_supplemental_retention.py \
  --stratified-receipt "${RETENTION_OUTPUT_ROOT}/stratified_hidden_eval_v3/receipt.json"

env CUDA_VISIBLE_DEVICES='' nice -n 10 ionice -c 2 -n 7 uv run python \
  scripts/validate_stage211_supplemental_retention.py \
  --receipt "${RETENTION_OUTPUT_ROOT}/retention_replay_v3/receipt.json"

printf '[stage211-quote-repair-handoff] complete combined=%s stratified=%s replay=%s\n' \
  "${COMBINED_ROOT}/supplemental_inventory.json" \
  "${RETENTION_OUTPUT_ROOT}/stratified_hidden_eval_v3/receipt.json" \
  "${RETENTION_OUTPUT_ROOT}/retention_replay_v3/receipt.json"
