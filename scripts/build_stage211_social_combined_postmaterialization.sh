#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${STAGE211_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

MATERIALIZED_INVENTORY="${MATERIALIZED_INVENTORY:-${HOME}/rwkvasr_data/stage211_social_vad_materialized_v1/materialized_inventory.json}"
FILTERED_ROOT="${FILTERED_ROOT:-${HOME}/rwkvasr_data/stage211_social_vad_filtered_v1}"
BASE_INVENTORY="${BASE_INVENTORY:-${HOME}/rwkvasr_data/stage211_supplemental_natural_v1/supplemental_inventory.json}"
COMBINED_ROOT="${COMBINED_ROOT:-${HOME}/rwkvasr_data/stage211_supplemental_combined_v2}"

cd "${REPO_ROOT}"

env CUDA_VISIBLE_DEVICES='' uv run python \
  scripts/filter_stage211_social_pcm_overlap.py \
  --materialized-inventory "${MATERIALIZED_INVENTORY}" \
  --output-root "${FILTERED_ROOT}" \
  --build-public

env CUDA_VISIBLE_DEVICES='' uv run python \
  scripts/filter_stage211_social_pcm_overlap.py \
  --materialized-inventory "${MATERIALIZED_INVENTORY}" \
  --output-root "${FILTERED_ROOT}" \
  --finalize-only

uv run python scripts/build_stage211_combined_supplemental_inventory.py \
  --base-inventory "${BASE_INVENTORY}" \
  --social-inventory "${FILTERED_ROOT}/filtered_inventory.json" \
  --output-root "${COMBINED_ROOT}"

uv run python scripts/create_stage211_supplemental_profile_receipt.py \
  --inventory "${COMBINED_ROOT}/supplemental_inventory.json" \
  --output "${COMBINED_ROOT}/supplemental_profile_receipt.json"
