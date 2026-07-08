#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

ROOT="${ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"
STAGES_ROOT="${STAGES_ROOT:-${ROOT}/stages/difficulty_sensevoice}"
BASE_CONFIG="${BASE_CONFIG:-configs/sensevoice_rwkv_clean_ctc_curriculum_4x4090_deepspeed.yaml}"
RUN_ROOT="${RUN_ROOT:-runs}"
RUN_PREFIX="${RUN_PREFIX:-sensevoice_rwkv_curriculum_svtok}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-sensevoice_tiktoken}"
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-assets/fun-asr-nano-2512/multilingual.tiktoken}"
VOCAB_SIZE="${VOCAB_SIZE:-60515}"
BLANK_ID="${BLANK_ID:-60515}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29610}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
SAVE_EVERY="${SAVE_EVERY:-500}"
STEP_EVAL_EVERY="${STEP_EVAL_EVERY:-500}"
STEP_EVAL_SAMPLES="${STEP_EVAL_SAMPLES:-256}"
EVAL_AFTER_STAGE="${EVAL_AFTER_STAGE:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_BEAM_SIZE="${EVAL_BEAM_SIZE:-4}"
EVAL_TOKEN_PRUNE_TOPK="${EVAL_TOKEN_PRUNE_TOPK:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
PREVIEW_COUNT="${PREVIEW_COUNT:-32}"
PREPARE_STAGES="${PREPARE_STAGES:-1}"
START_STAGE_INDEX="${START_STAGE_INDEX:-0}"
INITIAL_CHECKPOINT_PATH="${INITIAL_CHECKPOINT_PATH:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin${PATH:+:${PATH}}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

if [[ "${PREPARE_STAGES}" == "1" ]]; then
  "${SCRIPT_DIR}/prepare_ctc_difficulty_curriculum_stages.sh"
else
  echo "[rwkvasr] Skipping stage preparation because PREPARE_STAGES=0"
fi

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
  "${PYTHON_BIN}" - "${prediction_path}" "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

from rwkvasr.eval import compute_text_error_stats

prediction_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
stats = compute_text_error_stats(prediction_path, normalization="runtime")
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

run_stage_eval() {
  local stage_name="$1"
  local stage_dir="$2"
  local run_dir="$3"
  local checkpoint_path="$4"
  local eval_dir="${run_dir}/stage_eval"
  local prediction_path="${eval_dir}/${stage_name}.ctc_labeled.jsonl"
  local preview_path="${eval_dir}/${stage_name}.preview.txt"
  local metrics_path="${eval_dir}/${stage_name}.metrics.json"
  mkdir -p "${eval_dir}"

  local cmd=(
    "${PYTHON_BIN}" -m rwkvasr.cli.predict_ctc_labeled
    --webdataset-root "${ROOT}"
    --webdataset-length-index-path "${stage_dir}/webdataset_lengths.jsonl"
    --webdataset-split eval
    --webdataset-utt-id-key id
    --checkpoint-path "${checkpoint_path}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${EVAL_NUM_WORKERS}"
    --device "${EVAL_DEVICE}"
    --mode bi
    --beam-size "${EVAL_BEAM_SIZE}"
    --token-prune-topk "${EVAL_TOKEN_PRUNE_TOPK}"
    --text-normalization runtime
    --output-path "${prediction_path}"
    --preview-path "${preview_path}"
    --preview-count "${PREVIEW_COUNT}"
    --progress-interval 64
    --save-debug-lengths
  )
  if [[ "${EVAL_LIMIT}" != "0" ]]; then
    cmd+=(--limit "${EVAL_LIMIT}")
  fi

  echo "[rwkvasr] Stage eval ${stage_name}: checkpoint=${checkpoint_path}"
  echo "[rwkvasr] Eval command: $(shell_join "${cmd[@]}")"
  "${cmd[@]}" 2>&1 | tee "${eval_dir}/${stage_name}.predict.log"
  summarize_eval_metrics "${prediction_path}" "${metrics_path}" | tee "${eval_dir}/${stage_name}.metrics.log"
}

declare -a STAGE_NAMES=(
  "stage1_very_easy"
  "stage2_easy_cumulative"
  "stage3_medium_cumulative"
)
declare -a RUN_NAMES=(
  "${RUN_PREFIX}_stage1_very_easy_bs12_zero1_nockpt_4x4090"
  "${RUN_PREFIX}_stage2_easy_cumulative_bs12_zero1_nockpt_4x4090"
  "${RUN_PREFIX}_stage3_medium_cumulative_bs12_zero1_nockpt_4x4090"
)

previous_checkpoint="${INITIAL_CHECKPOINT_PATH}"
for index in "${!STAGE_NAMES[@]}"; do
  if (( index < START_STAGE_INDEX )); then
    echo "[rwkvasr] Skipping ${STAGE_NAMES[$index]} because START_STAGE_INDEX=${START_STAGE_INDEX}"
    continue
  fi
  stage_name="${STAGE_NAMES[$index]}"
  run_name="${RUN_NAMES[$index]}"
  stage_dir="${STAGES_ROOT}/${stage_name}"
  run_dir="${RUN_ROOT}/${run_name}"
  mkdir -p "${run_dir}/logs"

  if [[ ! -s "${stage_dir}/webdataset_buckets_audio_text/manifest.json" ]]; then
    echo "stage bucket manifest missing: ${stage_dir}/webdataset_buckets_audio_text/manifest.json" >&2
    exit 1
  fi

  local_master_port="$((MASTER_PORT_BASE + index))"
  train_cmd=(
    "${SCRIPT_DIR}/train_paper_rwkv_asr.sh"
    --config-yaml "${BASE_CONFIG}"
    --num-gpus "${NUM_GPUS}"
    --master-port "${local_master_port}"
    --webdataset-root "${ROOT}"
    --webdataset-index-path "${ROOT}/webdataset_index.json"
    --webdataset-length-index-path "${stage_dir}/webdataset_lengths.jsonl"
    --webdataset-bucket-manifest-path "${stage_dir}/webdataset_buckets_audio_text/manifest.json"
    --output-dir "${run_dir}"
    --wandb-run-name "${run_name}"
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
  if [[ -n "${previous_checkpoint}" ]]; then
    train_cmd+=(--init-checkpoint-path "${previous_checkpoint}")
  fi

  echo "[rwkvasr] Starting ${stage_name} run_dir=${run_dir}"
  if [[ -n "${previous_checkpoint}" ]]; then
    echo "[rwkvasr] Initializing from previous checkpoint: ${previous_checkpoint}"
  fi
  train_cmd_str="$(shell_join "${train_cmd[@]}")"
  script -q -f -e -c "${train_cmd_str}" "${run_dir}/logs/${stage_name}.train.ansi.log"

  current_checkpoint="$(resolve_checkpoint "${run_dir}")"
  if [[ -z "${current_checkpoint}" || ! -s "${current_checkpoint}" ]]; then
    echo "failed to resolve checkpoint for ${stage_name} under ${run_dir}" >&2
    exit 1
  fi
  echo "[rwkvasr] ${stage_name} checkpoint=${current_checkpoint}"

  if [[ "${EVAL_AFTER_STAGE}" == "1" ]]; then
    run_stage_eval "${stage_name}" "${stage_dir}" "${run_dir}" "${current_checkpoint}"
  fi
  previous_checkpoint="${current_checkpoint}"
done

echo "[rwkvasr] Difficulty curriculum training complete."
