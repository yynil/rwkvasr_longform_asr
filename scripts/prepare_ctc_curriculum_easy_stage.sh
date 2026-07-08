#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"
STAGE="${STAGE:-${ROOT}/stages/easy_clean_librispeech_aishell3}"
EVAL_RATIO="${EVAL_RATIO:-0.005}"
HASH_SEED="${HASH_SEED:-0}"
LENGTH_THREADS="${LENGTH_THREADS:-8}"
BUCKET_WIDTH="${BUCKET_WIDTH:-80}"
ENTRIES_PER_PART="${ENTRIES_PER_PART:-100000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

INDEX_PATH="${STAGE}/webdataset_index.json"
LENGTHS_PATH="${STAGE}/webdataset_lengths.jsonl"
SUMMARY_PATH="${STAGE}/webdataset_lengths.summary.json"
BUCKET_DIR="${STAGE}/webdataset_buckets_audio_text"
BUCKET_MANIFEST_PATH="${BUCKET_DIR}/manifest.json"
LOG_PATH="${STAGE}/prepare_easy.log"

mkdir -p "${STAGE}" "${BUCKET_DIR}"

find "${STAGE}" -maxdepth 1 -type l -name '*.tar' -delete
for shard in "${ROOT}"/librispeech_*.tar "${ROOT}"/aishell3_*.tar; do
  ln -s "../../$(basename "${shard}")" "${STAGE}/$(basename "${shard}")"
done

echo "[rwkvasr] Easy CTC curriculum stage root: ${STAGE}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] Easy CTC curriculum symlink shards: $(find "${STAGE}" -maxdepth 1 -type l -name '*.tar' | wc -l)" | tee -a "${LOG_PATH}"

"${PYTHON_BIN}" -m rwkvasr.cli.inspect_webdataset \
  --webdataset-root "${STAGE}" \
  --output-path "${INDEX_PATH}" \
  --split-by sample_id \
  --eval-ratio "${EVAL_RATIO}" \
  --hash-seed "${HASH_SEED}" \
  --utt-id-key id | tee -a "${LOG_PATH}"

tools/target/release/rwkvasr-tools \
  --webdataset-root "${STAGE}" \
  --output-path "${LENGTHS_PATH}" \
  --summary-path "${SUMMARY_PATH}" \
  --split-by sample_id \
  --eval-ratio "${EVAL_RATIO}" \
  --hash-seed "${HASH_SEED}" \
  --utt-id-key id \
  --threads "${LENGTH_THREADS}" | tee -a "${LOG_PATH}"

tools/target/release/build_bucket_index \
  --shard-root "${STAGE}" \
  --length-index-path "${LENGTHS_PATH}" \
  --output-dir "${BUCKET_DIR}" \
  --manifest-path "${BUCKET_MANIFEST_PATH}" \
  --bucket-width "${BUCKET_WIDTH}" \
  --text-cost-source auto \
  --text-cost-weight 4 \
  --json-size-text-offset 256 \
  --json-size-bytes-per-token 4.0 \
  --entries-per-part "${ENTRIES_PER_PART}" | tee -a "${LOG_PATH}"

echo "[rwkvasr] Easy CTC curriculum preprocessing complete." | tee -a "${LOG_PATH}"
echo "[rwkvasr] index=${INDEX_PATH}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] lengths=${LENGTHS_PATH}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] bucket_manifest=${BUCKET_MANIFEST_PATH}" | tee -a "${LOG_PATH}"
