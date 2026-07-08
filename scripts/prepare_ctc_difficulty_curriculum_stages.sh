#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ROOT="${ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"
DIFFICULTY_DIR="${DIFFICULTY_DIR:-${ROOT}/difficulty_buckets_sensevoice}"
DIFFICULTY_MANIFEST="${DIFFICULTY_MANIFEST:-${DIFFICULTY_DIR}/difficulty_manifest.json}"
STAGES_ROOT="${STAGES_ROOT:-${ROOT}/stages/difficulty_sensevoice}"
BUCKET_WIDTH="${BUCKET_WIDTH:-80}"
ENTRIES_PER_PART="${ENTRIES_PER_PART:-100000}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -s "${DIFFICULTY_MANIFEST}" ]]; then
  echo "difficulty manifest not found: ${DIFFICULTY_MANIFEST}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

declare -a STAGE_SPECS=(
  "stage1_very_easy:very_easy"
  "stage2_easy_cumulative:very_easy,easy"
  "stage3_medium_cumulative:very_easy,easy,medium"
)

for spec in "${STAGE_SPECS[@]}"; do
  stage_name="${spec%%:*}"
  tiers_csv="${spec#*:}"
  stage_dir="${STAGES_ROOT}/${stage_name}"
  lengths_path="${stage_dir}/webdataset_lengths.jsonl"
  bucket_dir="${stage_dir}/webdataset_buckets_audio_text"
  bucket_manifest="${bucket_dir}/manifest.json"

  IFS=',' read -r -a tiers <<< "${tiers_csv}"
  echo "[rwkvasr] Preparing difficulty curriculum stage ${stage_name}: ${tiers_csv}"
  "${PYTHON_BIN}" scripts/build_ctc_difficulty_stage_lengths.py \
    --difficulty-manifest-path "${DIFFICULTY_MANIFEST}" \
    --output-dir "${stage_dir}" \
    --stage-name "${stage_name}" \
    --tiers "${tiers[@]}"

  echo "[rwkvasr] Building bucket manifest for ${stage_name}: ${bucket_manifest}"
  cargo run --release --manifest-path tools/Cargo.toml --bin build_bucket_index -- \
    --shard-root "${ROOT}" \
    --length-index-path "${lengths_path}" \
    --output-dir "${bucket_dir}" \
    --manifest-path "${bucket_manifest}" \
    --bucket-width "${BUCKET_WIDTH}" \
    --text-cost-source auto \
    --text-cost-weight 4 \
    --json-size-text-offset 256 \
    --json-size-bytes-per-token 4.0 \
    --entries-per-part "${ENTRIES_PER_PART}"
done

echo "[rwkvasr] Difficulty curriculum stages ready under ${STAGES_ROOT}"
