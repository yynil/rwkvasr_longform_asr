#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

ZH_ROOT="${1:-/media/usbhd/training_data/asr/emilia/Emilia/ZH/Emilia/ZH}"
EN_ROOT="${2:-/media/usbhd/training_data/asr/emilia/Emilia/EN/Emilia/EN}"
MIX_ROOT="${3:-/media/usbhd/training_data/asr/emilia/Emilia/MIX_EN_ZH}"
INDEX_PATH="${MIX_ROOT}/webdataset_index.json"
LENGTHS_PATH="${MIX_ROOT}/webdataset_lengths.jsonl"
SUMMARY_PATH="${MIX_ROOT}/webdataset_lengths.summary.json"
BUCKET_DIR="${MIX_ROOT}/webdataset_buckets"
BUCKET_MANIFEST_PATH="${BUCKET_DIR}/manifest.json"
CMVN_PATH="${MIX_ROOT}/global_cmvn.json"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${MIX_ROOT}"

echo "[rwkvasr] Refreshing mixed-root symlinks under ${MIX_ROOT}"
find "${MIX_ROOT}" -maxdepth 1 -type l -name '*.tar' -delete

while IFS= read -r shard_path; do
  shard_name="$(basename "${shard_path}")"
  ln -sfn "${shard_path}" "${MIX_ROOT}/${shard_name}"
done < <(find "${ZH_ROOT}" -maxdepth 1 -name '*.tar' | sort)

while IFS= read -r shard_path; do
  shard_name="$(basename "${shard_path}")"
  ln -sfn "${shard_path}" "${MIX_ROOT}/${shard_name}"
done < <(find "${EN_ROOT}" -maxdepth 1 -name '*.tar' | sort)

cd "${REPO_ROOT}"

echo "[rwkvasr] Building mixed WebDataset index: ${INDEX_PATH}"
"${PYTHON_BIN}" -m rwkvasr.cli.inspect_webdataset \
  --webdataset-root "${MIX_ROOT}" \
  --output-path "${INDEX_PATH}" \
  --split-by shard_name \
  --eval-ratio 0.01 \
  --hash-seed 0 \
  --utt-id-key id

echo "[rwkvasr] Building mixed Rust length index: ${LENGTHS_PATH}"
cargo run --release --manifest-path tools/Cargo.toml --bin rwkvasr-tools -- \
  --webdataset-root "${MIX_ROOT}" \
  --output-path "${LENGTHS_PATH}" \
  --summary-path "${SUMMARY_PATH}" \
  --split-by shard_name \
  --eval-ratio 0.01 \
  --hash-seed 0 \
  --utt-id-key id

echo "[rwkvasr] Building mixed bucket manifest: ${BUCKET_MANIFEST_PATH}"
cargo run --release --manifest-path tools/Cargo.toml --bin build_bucket_index -- \
  --shard-root "${MIX_ROOT}" \
  --length-index-path "${LENGTHS_PATH}" \
  --output-dir "${BUCKET_DIR}" \
  --manifest-path "${BUCKET_MANIFEST_PATH}" \
  --bucket-width 80 \
  --entries-per-part 100000

echo "[rwkvasr] Computing mixed global CMVN: ${CMVN_PATH}"
"${PYTHON_BIN}" -m rwkvasr.cli.compute_cmvn \
  --webdataset-root "${MIX_ROOT}" \
  --webdataset-split train \
  --webdataset-eval-ratio 0.01 \
  --webdataset-hash-seed 0 \
  --webdataset-split-by shard_name \
  --webdataset-utt-id-key id \
  --output-path "${CMVN_PATH}"

echo "[rwkvasr] Emilia EN+ZH mixed preprocessing complete."
echo "[rwkvasr] mix_root=${MIX_ROOT}"
echo "[rwkvasr] index=${INDEX_PATH}"
echo "[rwkvasr] lengths=${LENGTHS_PATH}"
echo "[rwkvasr] bucket_manifest=${BUCKET_MANIFEST_PATH}"
echo "[rwkvasr] cmvn=${CMVN_PATH}"
