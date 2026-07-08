#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SOURCE_STAGE="${SOURCE_STAGE:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset/stages/easy_clean_librispeech_aishell3}"
STAGE="${STAGE:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset/stages/easy_clean_librispeech_aishell3_ctc_norm_no_unk_aligned_omni_aut8}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-sentencepiece}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-assets/omnilingual-asr-ctc/omniASR_tokenizer.model}"
TEXT_NORMALIZATION="${TEXT_NORMALIZATION:-ctc}"
FRONTEND_DOWNSAMPLE="${FRONTEND_DOWNSAMPLE:-aut_conv2d8}"
MIN_LOGIT_REQUIRED_RATIO="${MIN_LOGIT_REQUIRED_RATIO:-1.0}"
DROP_UNK_TOKEN="${DROP_UNK_TOKEN:-1}"
EVAL_RATIO="${EVAL_RATIO:-0.005}"
HASH_SEED="${HASH_SEED:-0}"
BUCKET_WIDTH="${BUCKET_WIDTH:-80}"
ENTRIES_PER_PART="${ENTRIES_PER_PART:-100000}"
OVERWRITE="${OVERWRITE:-0}"

SOURCE_LENGTHS_PATH="${SOURCE_STAGE}/webdataset_lengths.jsonl"
INDEX_PATH="${STAGE}/webdataset_index.json"
LENGTHS_PATH="${STAGE}/webdataset_lengths.jsonl"
SUMMARY_PATH="${STAGE}/webdataset_lengths.summary.json"
BUCKET_DIR="${STAGE}/webdataset_buckets_audio_text"
BUCKET_MANIFEST_PATH="${BUCKET_DIR}/manifest.json"
LOG_PATH="${STAGE}/prepare_ctc_aligned.log"
BUILD_BUCKET_BIN="${BUILD_BUCKET_BIN:-${REPO_ROOT}/tools/target/release/build_bucket_index}"

if [[ ! -s "${SOURCE_LENGTHS_PATH}" ]]; then
  echo "source length index missing: ${SOURCE_LENGTHS_PATH}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -f "${INDEX_PATH}" "${LENGTHS_PATH}" "${SUMMARY_PATH}" "${LOG_PATH}"
  rm -rf "${BUCKET_DIR}"
fi

mkdir -p "${STAGE}" "${BUCKET_DIR}"

find "${STAGE}" -maxdepth 1 -type l -name '*.tar' -delete
for shard in "${SOURCE_STAGE}"/*.tar; do
  ln -s "$(readlink -f "${shard}")" "${STAGE}/$(basename "${shard}")"
done

echo "[rwkvasr] CTC-aligned clean stage root: ${STAGE}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] CTC-aligned symlink shards: $(find "${STAGE}" -maxdepth 1 -type l -name '*.tar' | wc -l)" | tee -a "${LOG_PATH}"
echo "[rwkvasr] tokenizer_type=${TOKENIZER_TYPE}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] tokenizer_model_path=${TOKENIZER_MODEL_PATH}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] text_normalization=${TEXT_NORMALIZATION}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] frontend_downsample=${FRONTEND_DOWNSAMPLE}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] drop_unk_token=${DROP_UNK_TOKEN}" | tee -a "${LOG_PATH}"

if [[ ! -s "${LENGTHS_PATH}" ]]; then
  build_cmd=(
    "${PYTHON_BIN}" scripts/build_ctc_aligned_stage_lengths.py
    --shard-root "${SOURCE_STAGE}"
    --length-index-path "${SOURCE_LENGTHS_PATH}"
    --output-dir "${STAGE}"
    --tokenizer-type "${TOKENIZER_TYPE}"
    --tokenizer-model-path "${TOKENIZER_MODEL_PATH}"
    --text-normalization "${TEXT_NORMALIZATION}"
    --frontend-downsample "${FRONTEND_DOWNSAMPLE}"
    --min-logit-required-ratio "${MIN_LOGIT_REQUIRED_RATIO}"
    --eval-ratio "${EVAL_RATIO}"
    --hash-seed "${HASH_SEED}"
    --split-by sample_id
    --utt-id-key id
  )
  if [[ "${DROP_UNK_TOKEN}" == "1" ]]; then
    build_cmd+=(--drop-unk-token)
  fi
  "${build_cmd[@]}" | tee -a "${LOG_PATH}"
else
  echo "[rwkvasr] Reusing existing CTC-aligned lengths: ${LENGTHS_PATH}" | tee -a "${LOG_PATH}"
fi

if [[ ! -x "${BUILD_BUCKET_BIN}" ]]; then
  cargo build --release --manifest-path tools/Cargo.toml --bin build_bucket_index
fi

if [[ ! -s "${BUCKET_MANIFEST_PATH}" ]]; then
  "${BUILD_BUCKET_BIN}" \
    --shard-root "${STAGE}" \
    --length-index-path "${LENGTHS_PATH}" \
    --output-dir "${BUCKET_DIR}" \
    --manifest-path "${BUCKET_MANIFEST_PATH}" \
    --bucket-width "${BUCKET_WIDTH}" \
    --text-cost-source num_text_tokens \
    --text-cost-weight 4 \
    --json-size-text-offset 256 \
    --json-size-bytes-per-token 4.0 \
    --entries-per-part "${ENTRIES_PER_PART}" | tee -a "${LOG_PATH}"
else
  echo "[rwkvasr] Reusing existing bucket manifest: ${BUCKET_MANIFEST_PATH}" | tee -a "${LOG_PATH}"
fi

echo "[rwkvasr] CTC-aligned clean preprocessing complete." | tee -a "${LOG_PATH}"
echo "[rwkvasr] index=${INDEX_PATH}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] lengths=${LENGTHS_PATH}" | tee -a "${LOG_PATH}"
echo "[rwkvasr] bucket_manifest=${BUCKET_MANIFEST_PATH}" | tee -a "${LOG_PATH}"
