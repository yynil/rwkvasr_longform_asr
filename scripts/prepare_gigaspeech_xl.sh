#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

INPUT_ROOT="${1:-/media/usbhd/training_data/asr/speechcolab/gigaspeech/parquet-data/xl}"
TEXT_NORMALIZATION="${TEXT_NORMALIZATION:-none}"
DEFAULT_OUTPUT_ROOT="/media/usbhd/training_data/asr/speechcolab/gigaspeech/webdataset_xl_train"
if [[ "${TEXT_NORMALIZATION}" != "none" ]]; then
  DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}_${TEXT_NORMALIZATION}"
fi
OUTPUT_ROOT="${2:-${DEFAULT_OUTPUT_ROOT}}"
INDEX_PATH="${OUTPUT_ROOT}/webdataset_index.json"
LENGTHS_PATH="${OUTPUT_ROOT}/webdataset_lengths.jsonl"
SUMMARY_PATH="${OUTPUT_ROOT}/webdataset_lengths.summary.json"
BUCKET_DIR="${OUTPUT_ROOT}/webdataset_buckets"
BUCKET_MANIFEST_PATH="${BUCKET_DIR}/manifest.json"
CMVN_PATH="${OUTPUT_ROOT}/global_cmvn.json"

INPUT_GLOB="${GIGASPEECH_INPUT_GLOB:-train-*.parquet}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-5000}"
MAX_INPUT_SHARDS="${MAX_INPUT_SHARDS:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PARQUET_BATCH_SIZE="${PARQUET_BATCH_SIZE:-128}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10000}"
BUCKET_WIDTH="${BUCKET_WIDTH:-80}"
ENTRIES_PER_PART="${ENTRIES_PER_PART:-100000}"
BUCKET_TEXT_COST_SOURCE="${BUCKET_TEXT_COST_SOURCE:-auto}"
BUCKET_TEXT_COST_WEIGHT="${BUCKET_TEXT_COST_WEIGHT:-4}"
BUCKET_JSON_SIZE_TEXT_OFFSET="${BUCKET_JSON_SIZE_TEXT_OFFSET:-256}"
BUCKET_JSON_SIZE_BYTES_PER_TOKEN="${BUCKET_JSON_SIZE_BYTES_PER_TOKEN:-4.0}"
EVAL_RATIO="${EVAL_RATIO:-0.01}"
HASH_SEED="${HASH_SEED:-0}"
LENGTH_THREADS="${LENGTH_THREADS:-4}"
CONVERT_THREADS="${CONVERT_THREADS:-${LENGTH_THREADS}}"
COMPUTE_CMVN="${COMPUTE_CMVN:-1}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-1}"
ADOPT_EXISTING="${ADOPT_EXISTING:-1}"
SKIP_MISSING="${SKIP_MISSING:-1}"
STAGING_ROOT="${STAGING_ROOT:-}"
STAGING_MAX_GB="${STAGING_MAX_GB:-100}"
STAGING_MAX_BYTES="${STAGING_MAX_BYTES:-}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${OVERWRITE}" == "1" && "${RESUME}" == "1" ]]; then
  echo "OVERWRITE=1 conflicts with RESUME=1. Use RESUME=0 OVERWRITE=1 for a clean rebuild, or unset OVERWRITE to continue." >&2
  exit 1
fi

convert_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  convert_args+=(--overwrite)
fi
if [[ "${RESUME}" == "1" ]]; then
  convert_args+=(--resume)
  if [[ "${ADOPT_EXISTING}" == "1" ]]; then
    convert_args+=(--adopt-existing)
  fi
fi
if [[ "${SKIP_MISSING}" == "1" ]]; then
  convert_args+=(--skip-missing)
fi
if [[ "${MAX_INPUT_SHARDS}" != "0" ]]; then
  convert_args+=(--max-input-shards "${MAX_INPUT_SHARDS}")
fi
if [[ "${MAX_SAMPLES}" != "0" ]]; then
  convert_args+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${STAGING_ROOT}" ]]; then
  convert_args+=(--staging-root "${STAGING_ROOT}")
  if [[ -z "${STAGING_MAX_BYTES}" ]]; then
    STAGING_MAX_BYTES=$((STAGING_MAX_GB * 1024 * 1024 * 1024))
  fi
  convert_args+=(--staging-max-bytes "${STAGING_MAX_BYTES}")
fi

cd "${REPO_ROOT}"

echo "[rwkvasr] Converting GigaSpeech parquet to WebDataset"
echo "[rwkvasr] input=${INPUT_ROOT} glob=${INPUT_GLOB} output=${OUTPUT_ROOT}"
echo "[rwkvasr] text_normalization=${TEXT_NORMALIZATION}"
cargo run --release --manifest-path tools/Cargo.toml --bin convert_asr_corpus -- gigaspeech-parquet \
  --input-root "${INPUT_ROOT}" \
  --input-glob "${INPUT_GLOB}" \
  --output-root "${OUTPUT_ROOT}" \
  --language en \
  --shard-prefix GSXL \
  --samples-per-shard "${SAMPLES_PER_SHARD}" \
  --parquet-batch-size "${PARQUET_BATCH_SIZE}" \
  --progress-every "${PROGRESS_EVERY}" \
  --threads "${CONVERT_THREADS}" \
  --text-normalization "${TEXT_NORMALIZATION}" \
  "${convert_args[@]}"

if [[ "${OVERWRITE}" != "1" && -s "${INDEX_PATH}" ]]; then
  echo "[rwkvasr] Skipping WebDataset index because it already exists: ${INDEX_PATH}"
else
  echo "[rwkvasr] Building WebDataset index: ${INDEX_PATH}"
  "${PYTHON_BIN}" -m rwkvasr.cli.inspect_webdataset \
    --webdataset-root "${OUTPUT_ROOT}" \
    --output-path "${INDEX_PATH}" \
    --split-by shard_name \
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
    --split-by shard_name \
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
    --text-cost-source "${BUCKET_TEXT_COST_SOURCE}" \
    --text-cost-weight "${BUCKET_TEXT_COST_WEIGHT}" \
    --json-size-text-offset "${BUCKET_JSON_SIZE_TEXT_OFFSET}" \
    --json-size-bytes-per-token "${BUCKET_JSON_SIZE_BYTES_PER_TOKEN}" \
    --entries-per-part "${ENTRIES_PER_PART}"
fi

if [[ "${COMPUTE_CMVN}" == "1" ]]; then
  if [[ "${OVERWRITE}" != "1" && -s "${CMVN_PATH}" ]]; then
    echo "[rwkvasr] Skipping global CMVN because it already exists: ${CMVN_PATH}"
  else
    echo "[rwkvasr] Computing global CMVN: ${CMVN_PATH}"
    "${PYTHON_BIN}" -m rwkvasr.cli.compute_cmvn \
      --webdataset-root "${OUTPUT_ROOT}" \
      --webdataset-split train \
      --webdataset-eval-ratio "${EVAL_RATIO}" \
      --webdataset-hash-seed "${HASH_SEED}" \
      --webdataset-split-by shard_name \
      --webdataset-utt-id-key id \
      --output-path "${CMVN_PATH}"
  fi
else
  echo "[rwkvasr] Skipping CMVN because COMPUTE_CMVN=${COMPUTE_CMVN}"
fi

echo "[rwkvasr] GigaSpeech preprocessing complete."
echo "[rwkvasr] root=${OUTPUT_ROOT}"
echo "[rwkvasr] index=${INDEX_PATH}"
echo "[rwkvasr] lengths=${LENGTHS_PATH}"
echo "[rwkvasr] bucket_manifest=${BUCKET_MANIFEST_PATH}"
echo "[rwkvasr] cmvn=${CMVN_PATH}"
