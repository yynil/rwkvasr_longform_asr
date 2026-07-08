#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-/media/usbhd/training_data/voxbox/librispeech}"
AISHELL3_ROOT="${AISHELL3_ROOT:-/media/usbhd/training_data/voxbox/aishell-3}"
COMMONVOICE_EN_ROOT="${COMMONVOICE_EN_ROOT:-/media/usbhd/training_data/voxbox/commonvoice_en}"
COMMONVOICE_CN_ROOT="${COMMONVOICE_CN_ROOT:-/media/usbhd/training_data/voxbox/commonvoice_cn}"

INDEX_PATH="${OUTPUT_ROOT}/webdataset_index.json"
LENGTHS_PATH="${OUTPUT_ROOT}/webdataset_lengths.jsonl"
SUMMARY_PATH="${OUTPUT_ROOT}/webdataset_lengths.summary.json"
BUCKET_DIR="${OUTPUT_ROOT}/webdataset_buckets_audio_text"
BUCKET_MANIFEST_PATH="${BUCKET_DIR}/manifest.json"

OVERWRITE="${OVERWRITE:-0}"
EVAL_RATIO="${EVAL_RATIO:-0.005}"
HASH_SEED="${HASH_SEED:-0}"
LENGTH_THREADS="${LENGTH_THREADS:-4}"
BUCKET_WIDTH="${BUCKET_WIDTH:-80}"
ENTRIES_PER_PART="${ENTRIES_PER_PART:-100000}"
TARGET_SHARD_SIZE_MB="${TARGET_SHARD_SIZE_MB:-1024}"
MIN_DURATION="${MIN_DURATION:-0.5}"
MAX_DURATION="${MAX_DURATION:-20.0}"
MIN_SPEECH_RATIO="${MIN_SPEECH_RATIO:-0.25}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

echo "[rwkvasr] Preparing clean CTC curriculum root: ${OUTPUT_ROOT}"
FILTER_ARGS=(
  --source "librispeech:${LIBRISPEECH_ROOT}" \
  --source "aishell3:${AISHELL3_ROOT}" \
  --source "commonvoice_en:${COMMONVOICE_EN_ROOT}" \
  --source "commonvoice_cn:${COMMONVOICE_CN_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --target-shard-size-mb "${TARGET_SHARD_SIZE_MB}" \
  --min-duration "${MIN_DURATION}" \
  --max-duration "${MAX_DURATION}" \
  --min-speech-ratio "${MIN_SPEECH_RATIO}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  FILTER_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" scripts/filter_voxbox_asr_webdataset.py "${FILTER_ARGS[@]}"

if [[ "${OVERWRITE}" != "1" && -s "${INDEX_PATH}" ]]; then
  echo "[rwkvasr] Skipping WebDataset index because it already exists: ${INDEX_PATH}"
else
  echo "[rwkvasr] Building WebDataset index: ${INDEX_PATH}"
  "${PYTHON_BIN}" -m rwkvasr.cli.inspect_webdataset \
    --webdataset-root "${OUTPUT_ROOT}" \
    --output-path "${INDEX_PATH}" \
    --split-by sample_id \
    --eval-ratio "${EVAL_RATIO}" \
    --hash-seed "${HASH_SEED}" \
    --utt-id-key id
fi

if [[ "${OVERWRITE}" != "1" && -s "${LENGTHS_PATH}" && -s "${SUMMARY_PATH}" ]]; then
  echo "[rwkvasr] Skipping Rust length index because it already exists: ${LENGTHS_PATH}"
else
  echo "[rwkvasr] Building Rust length index: ${LENGTHS_PATH}"
  cargo run --release --manifest-path tools/Cargo.toml --bin rwkvasr-tools -- \
    --webdataset-root "${OUTPUT_ROOT}" \
    --output-path "${LENGTHS_PATH}" \
    --summary-path "${SUMMARY_PATH}" \
    --split-by sample_id \
    --eval-ratio "${EVAL_RATIO}" \
    --hash-seed "${HASH_SEED}" \
    --utt-id-key id \
    --threads "${LENGTH_THREADS}"
fi

if [[ "${OVERWRITE}" != "1" && -s "${BUCKET_MANIFEST_PATH}" ]]; then
  echo "[rwkvasr] Skipping bucket manifest because it already exists: ${BUCKET_MANIFEST_PATH}"
else
  echo "[rwkvasr] Building bucket manifest: ${BUCKET_MANIFEST_PATH}"
  cargo run --release --manifest-path tools/Cargo.toml --bin build_bucket_index -- \
    --shard-root "${OUTPUT_ROOT}" \
    --length-index-path "${LENGTHS_PATH}" \
    --output-dir "${BUCKET_DIR}" \
    --manifest-path "${BUCKET_MANIFEST_PATH}" \
    --bucket-width "${BUCKET_WIDTH}" \
    --text-cost-source auto \
    --text-cost-weight 4 \
    --json-size-text-offset 256 \
    --json-size-bytes-per-token 4.0 \
    --entries-per-part "${ENTRIES_PER_PART}"
fi

echo "[rwkvasr] Clean CTC curriculum preprocessing complete."
echo "[rwkvasr] root=${OUTPUT_ROOT}"
echo "[rwkvasr] index=${INDEX_PATH}"
echo "[rwkvasr] lengths=${LENGTHS_PATH}"
echo "[rwkvasr] bucket_manifest=${BUCKET_MANIFEST_PATH}"
