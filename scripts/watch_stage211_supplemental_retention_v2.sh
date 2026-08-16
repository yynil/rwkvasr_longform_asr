#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
COMBINED_ROOT="${COMBINED_ROOT:-${HOME}/rwkvasr_data/stage211_supplemental_combined_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/rwkvasr_data/stage211_full_curriculum}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
INVENTORY="${COMBINED_ROOT}/supplemental_inventory.json"
PROFILE="${COMBINED_ROOT}/supplemental_profile_receipt.json"
STRATIFIED_RECEIPT="${OUTPUT_ROOT}/stratified_hidden_eval_v3/receipt.json"
REPLAY_RECEIPT="${OUTPUT_ROOT}/retention_replay_v3/receipt.json"

cd "${REPO_ROOT}"

while [[ ! -s "${INVENTORY}" || ! -s "${PROFILE}" ]]; do
  printf '[stage211-supplemental-retention-watch] waiting %ss inventory=%s profile=%s\n' \
    "${POLL_SECONDS}" "${INVENTORY}" "${PROFILE}"
  sleep "${POLL_SECONDS}"
done

nice -n 10 ionice -c 2 -n 7 uv run python \
  scripts/build_stage211_supplemental_retention.py \
  --supplemental-inventory "${INVENTORY}" \
  --supplemental-profile-receipt "${PROFILE}" \
  --output-root "${OUTPUT_ROOT}" \
  --stratified-dir-name stratified_hidden_eval_v3 \
  --replay-dir-name retention_replay_v3

nice -n 10 ionice -c 2 -n 7 uv run python \
  scripts/validate_stage211_supplemental_retention.py \
  --stratified-receipt "${STRATIFIED_RECEIPT}"

nice -n 10 ionice -c 2 -n 7 uv run python \
  scripts/validate_stage211_supplemental_retention.py \
  --receipt "${REPLAY_RECEIPT}"

printf '[stage211-supplemental-retention-watch] complete stratified=%s replay=%s\n' \
  "${STRATIFIED_RECEIPT}" "${REPLAY_RECEIPT}"
