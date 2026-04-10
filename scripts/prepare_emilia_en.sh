#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

DATA_ROOT="${1:-/media/usbhd/training_data/asr/emilia/Emilia/EN/Emilia/EN}"
INDEX_PATH="${DATA_ROOT}/webdataset_index.json"
LENGTHS_PATH="${DATA_ROOT}/webdataset_lengths.jsonl"
SUMMARY_PATH="${DATA_ROOT}/webdataset_lengths.summary.json"
BUCKET_DIR="${DATA_ROOT}/webdataset_buckets"
BUCKET_MANIFEST_PATH="${BUCKET_DIR}/manifest.json"
CMVN_PATH="${DATA_ROOT}/global_cmvn.json"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

echo "[rwkvasr] Building WebDataset index: ${INDEX_PATH}"
"${PYTHON_BIN}" -m rwkvasr.cli.inspect_webdataset \
  --webdataset-root "${DATA_ROOT}" \
  --output-path "${INDEX_PATH}" \
  --split-by shard_name \
  --eval-ratio 0.01 \
  --hash-seed 0 \
  --utt-id-key id

echo "[rwkvasr] Building Rust length index: ${LENGTHS_PATH}"
cargo run --release --manifest-path tools/Cargo.toml --bin rwkvasr-tools -- \
  --webdataset-root "${DATA_ROOT}" \
  --output-path "${LENGTHS_PATH}" \
  --summary-path "${SUMMARY_PATH}" \
  --split-by shard_name \
  --eval-ratio 0.01 \
  --hash-seed 0 \
  --utt-id-key id

echo "[rwkvasr] Building bucket manifest: ${BUCKET_MANIFEST_PATH}"
cargo run --release --manifest-path tools/Cargo.toml --bin build_bucket_index -- \
  --shard-root "${DATA_ROOT}" \
  --length-index-path "${LENGTHS_PATH}" \
  --output-dir "${BUCKET_DIR}" \
  --manifest-path "${BUCKET_MANIFEST_PATH}" \
  --bucket-width 80 \
  --entries-per-part 100000

echo "[rwkvasr] Computing global CMVN: ${CMVN_PATH}"
"${PYTHON_BIN}" -m rwkvasr.cli.compute_cmvn \
  --webdataset-root "${DATA_ROOT}" \
  --webdataset-split train \
  --webdataset-eval-ratio 0.01 \
  --webdataset-hash-seed 0 \
  --webdataset-split-by shard_name \
  --webdataset-utt-id-key id \
  --output-path "${CMVN_PATH}"

echo "[rwkvasr] Emilia EN preprocessing complete."
echo "[rwkvasr] index=${INDEX_PATH}"
echo "[rwkvasr] lengths=${LENGTHS_PATH}"
echo "[rwkvasr] bucket_manifest=${BUCKET_MANIFEST_PATH}"
echo "[rwkvasr] cmvn=${CMVN_PATH}"
