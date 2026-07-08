#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ROOT="${ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset/stages/easy_clean_librispeech_aishell3_ctc_norm_no_unk_aligned_omni_aut8}"
BASE_CONFIG="${BASE_CONFIG:-configs/aurwkv_qwen3_omni_ctc_aligned_clean_librispeech_aishell3_4x4090_deepspeed.yaml}"
RUN_ROOT="${RUN_ROOT:-/media/usbhd/rwkvasr_runs/aurwkv_qwen3_omni_ctc_aligned_clean_librispeech_aishell3}"
RUN_NAME="${RUN_NAME:-aurwkv_qwen3_omni_ctc_norm_no_unk_aligned_clean_librispeech_aishell3_bs12_zero1_nockpt_4x4090}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-sentencepiece}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-assets/omnilingual-asr-ctc/omniASR_tokenizer.model}"
VOCAB_SIZE="${VOCAB_SIZE:-9812}"
BLANK_ID="${BLANK_ID:-9812}"
TEXT_NORMALIZATION="${TEXT_NORMALIZATION:-ctc}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29650}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
SAVE_EVERY="${SAVE_EVERY:-500}"
STEP_EVAL_EVERY="${STEP_EVAL_EVERY:-500}"
STEP_EVAL_SAMPLES="${STEP_EVAL_SAMPLES:-256}"
EVAL_AFTER="${EVAL_AFTER:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_BEAM_SIZE="${EVAL_BEAM_SIZE:-4}"
EVAL_TOKEN_PRUNE_TOPK="${EVAL_TOKEN_PRUNE_TOPK:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
PREVIEW_COUNT="${PREVIEW_COUNT:-32}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin${PATH:+:${PATH}}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

shell_join() {
  local out=""
  local item
  for item in "$@"; do
    printf -v item "%q" "${item}"
    out+="${item} "
  done
  printf "%s" "${out}"
}

resolve_checkpoint() {
  local run_dir="$1"
  if [[ -s "${run_dir}/best.pt" ]]; then
    printf "%s\n" "${run_dir}/best.pt"
    return 0
  fi
  if [[ -s "${run_dir}/latest_checkpoint.yaml" ]]; then
    "${PYTHON_BIN}" - "${run_dir}/latest_checkpoint.yaml" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
checkpoint = data.get("checkpoint_path")
if checkpoint:
    print(checkpoint)
PY
    return 0
  fi
  return 1
}

summarize_eval_metrics() {
  local prediction_path="$1"
  local summary_path="$2"
  "${PYTHON_BIN}" - "${prediction_path}" "${summary_path}" "${TEXT_NORMALIZATION}" <<'PY'
import json
import sys
from pathlib import Path

from rwkvasr.eval import compute_text_error_stats

prediction_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
normalization = sys.argv[3]
stats = compute_text_error_stats(prediction_path, normalization=normalization)
summary = {
    "prediction_path": str(prediction_path),
    "sample_count": int(stats.get("sample_count", 0) or 0),
    "avg_wer": stats.get("avg_wer"),
    "avg_cer": stats.get("avg_cer"),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
wer = summary["avg_wer"]
cer = summary["avg_cer"]
wer_text = "n/a" if wer is None else f"{wer * 100.0:.2f}%"
cer_text = "n/a" if cer is None else f"{cer * 100.0:.2f}%"
print(f"[rwkvasr] normalized eval samples={summary['sample_count']} WER={wer_text} CER={cer_text}")
PY
}

run_eval() {
  local run_dir="$1"
  local checkpoint_path="$2"
  local eval_dir="${run_dir}/stage_eval"
  local prediction_path="${eval_dir}/ctc_aligned_clean_librispeech_aishell3.ctc_labeled.jsonl"
  local preview_path="${eval_dir}/ctc_aligned_clean_librispeech_aishell3.preview.txt"
  local metrics_path="${eval_dir}/ctc_aligned_clean_librispeech_aishell3.metrics.json"
  mkdir -p "${eval_dir}"

  local cmd=(
    "${PYTHON_BIN}" -m rwkvasr.cli.predict_ctc_labeled
    --webdataset-root "${ROOT}"
    --webdataset-length-index-path "${ROOT}/webdataset_lengths.jsonl"
    --webdataset-split eval
    --webdataset-utt-id-key id
    --checkpoint-path "${checkpoint_path}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${EVAL_NUM_WORKERS}"
    --device "${EVAL_DEVICE}"
    --mode bi
    --beam-size "${EVAL_BEAM_SIZE}"
    --token-prune-topk "${EVAL_TOKEN_PRUNE_TOPK}"
    --text-normalization "${TEXT_NORMALIZATION}"
    --output-path "${prediction_path}"
    --preview-path "${preview_path}"
    --preview-count "${PREVIEW_COUNT}"
    --progress-interval 64
    --save-debug-lengths
  )
  if [[ "${EVAL_LIMIT}" != "0" ]]; then
    cmd+=(--limit "${EVAL_LIMIT}")
  fi

  echo "[rwkvasr] Eval checkpoint=${checkpoint_path}"
  echo "[rwkvasr] Eval command: $(shell_join "${cmd[@]}")"
  "${cmd[@]}" 2>&1 | tee "${eval_dir}/ctc_aligned_clean_librispeech_aishell3.predict.log"
  summarize_eval_metrics "${prediction_path}" "${metrics_path}" | tee "${eval_dir}/ctc_aligned_clean_librispeech_aishell3.metrics.log"
}

RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}/logs"

train_cmd=(
  "${SCRIPT_DIR}/train_paper_rwkv_asr.sh"
  --config-yaml "${BASE_CONFIG}"
  --num-gpus "${NUM_GPUS}"
  --master-port "${MASTER_PORT}"
  --webdataset-root "${ROOT}"
  --webdataset-index-path "${ROOT}/webdataset_index.json"
  --webdataset-length-index-path "${ROOT}/webdataset_lengths.jsonl"
  --webdataset-bucket-manifest-path "${ROOT}/webdataset_buckets_audio_text/manifest.json"
  --output-dir "${RUN_DIR}"
  --wandb-run-name "${RUN_NAME}"
  --tokenizer-type "${TOKENIZER_TYPE}"
  --tokenizer-model-path "${TOKENIZER_MODEL_PATH}"
  --vocab-size "${VOCAB_SIZE}"
  --blank-id "${BLANK_ID}"
  --no-tokenizer-append-eos
  --epochs "${TRAIN_EPOCHS}"
  --save-every "${SAVE_EVERY}"
  --step-eval-every "${STEP_EVAL_EVERY}"
  --step-eval-samples "${STEP_EVAL_SAMPLES}"
)

echo "[rwkvasr] Starting CTC-aligned clean BiRWKV run_dir=${RUN_DIR}"
train_cmd_str="$(shell_join "${train_cmd[@]}")"
script -q -f -e -c "${train_cmd_str}" "${RUN_DIR}/logs/ctc_aligned_clean_librispeech_aishell3.train.ansi.log"
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

checkpoint_path="$(resolve_checkpoint "${RUN_DIR}")"
if [[ -z "${checkpoint_path}" || ! -s "${checkpoint_path}" ]]; then
  echo "failed to resolve checkpoint under ${RUN_DIR}" >&2
  exit 1
fi
echo "[rwkvasr] CTC-aligned clean BiRWKV checkpoint=${checkpoint_path}"

if [[ "${EVAL_AFTER}" == "1" ]]; then
  run_eval "${RUN_DIR}" "${checkpoint_path}"
fi

echo "[rwkvasr] CTC-aligned clean BiRWKV training complete."
