#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RICH_FORCE_TERMINAL="${RICH_FORCE_TERMINAL:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
RUN_DIR="${RUN_DIR:-}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/artifacts/eval_benchmarks}"
MANIFEST_DIR="${MANIFEST_DIR:-${EVAL_ROOT}/manifests}"
PREPARE_MANIFESTS="${PREPARE_MANIFESTS:-0}"
TIMESTAMP="${TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_ROOT}/runs/${TIMESTAMP}}"
PREDICTION_DIR="${PREDICTION_DIR:-${OUTPUT_DIR}/predictions}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
METRICS_JSON="${METRICS_JSON:-${OUTPUT_DIR}/metrics.json}"
METRICS_MD="${METRICS_MD:-${OUTPUT_DIR}/metrics.md}"

DEVICES="${DEVICES:-0,1,2,3}"
CTC_BATCH_SIZE="${CTC_BATCH_SIZE:-4}"
CTC_NUM_WORKERS="${CTC_NUM_WORKERS:-0}"
CTC_BEAM_SIZE="${CTC_BEAM_SIZE:-1}"
CTC_TOKEN_PRUNE_TOPK="${CTC_TOKEN_PRUNE_TOPK:-16}"
CTC_TEXT_NORMALIZATION="${CTC_TEXT_NORMALIZATION:-runtime}"
CTC_PROGRESS_INTERVAL="${CTC_PROGRESS_INTERVAL:-500}"
CTC_SHARD_DATASETS="${CTC_SHARD_DATASETS:-0}"
CTC_SHARD_COUNT="${CTC_SHARD_COUNT:-}"
CTC_LIMIT="${CTC_LIMIT:-0}"

RUN_AR="${RUN_AR:-auto}"
AR_BATCH_SIZE="${AR_BATCH_SIZE:-1}"
AR_NUM_WORKERS="${AR_NUM_WORKERS:-0}"
AR_TEXT_NORMALIZATION="${AR_TEXT_NORMALIZATION:-runtime}"
AR_MAX_NEW_TOKENS_FACTOR="${AR_MAX_NEW_TOKENS_FACTOR:-0.5}"
AR_DO_SAMPLE="${AR_DO_SAMPLE:-1}"
AR_TEMPERATURE="${AR_TEMPERATURE:-0.35}"
AR_TOP_K="${AR_TOP_K:-3}"
AR_TOP_P="${AR_TOP_P:-0.5}"
AR_SEED="${AR_SEED:-20260615}"
AR_PROGRESS_INTERVAL="${AR_PROGRESS_INTERVAL:-500}"
METRIC_NORMALIZATION="${METRIC_NORMALIZATION:-ctc}"

DATASETS=(
  librispeech_test_clean
  librispeech_test_other
  aishell1_test
  commonvoice_en_test
  wenetspeech_test_net
)

log() {
  printf '[public-eval] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

resolve_checkpoint() {
  if [[ -n "${CHECKPOINT_PATH}" ]]; then
    if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
      echo "CHECKPOINT_PATH does not exist or is empty: ${CHECKPOINT_PATH}" >&2
      exit 1
    fi
    return 0
  fi
  if [[ -z "${RUN_DIR}" ]]; then
    echo "Set CHECKPOINT_PATH or RUN_DIR" >&2
    exit 1
  fi
  for candidate in \
    "${RUN_DIR}/joint_wercer_best.pt" \
    "${RUN_DIR}/wercer_best.pt" \
    "${RUN_DIR}/best.pt"; do
    if [[ -s "${candidate}" ]]; then
      CHECKPOINT_PATH="${candidate}"
      return 0
    fi
  done
  local latest
  latest="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^step-\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1)"
  if [[ -n "${latest}" && -s "${RUN_DIR}/step-${latest}.pt" ]]; then
    CHECKPOINT_PATH="${RUN_DIR}/step-${latest}.pt"
    return 0
  fi
  echo "No checkpoint found under RUN_DIR=${RUN_DIR}" >&2
  exit 1
}

decoder_enabled() {
  "${PYTHON_BIN}" - "${CHECKPOINT_PATH}" <<'PY'
import sys
from pathlib import Path
from rwkvasr.config import load_yaml

checkpoint = Path(sys.argv[1]).resolve()
config_path = checkpoint.parent / "model_config.yaml"
if not config_path.exists():
    print("0")
    raise SystemExit
cfg = load_yaml(config_path)
print("1" if bool(cfg.get("decoder_enabled", False)) else "0")
PY
}

manifest_path() {
  local dataset="$1"
  local path="${MANIFEST_DIR}/${dataset}.jsonl"
  if [[ ! -s "${path}" ]]; then
    echo "manifest missing for ${dataset}: ${path}" >&2
    exit 1
  fi
  printf '%s\n' "${path}"
}

line_count() {
  local path="$1"
  if [[ -s "${path}" ]]; then
    wc -l <"${path}" | tr -d ' '
  else
    printf '0'
  fi
}

truthy() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

split_manifest() {
  local manifest="$1"
  local split_dir="$2"
  local shards="$3"
  rm -rf "${split_dir}"
  mkdir -p "${split_dir}"
  "${PYTHON_BIN}" - "${manifest}" "${split_dir}" "${shards}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

manifest = Path(sys.argv[1])
split_dir = Path(sys.argv[2])
shards = int(sys.argv[3])
if shards < 1:
    raise SystemExit("shards must be >= 1")
handles = []
try:
    for index in range(shards):
        handles.append((split_dir / f"part-{index:02d}.jsonl").open("w", encoding="utf-8"))
    with manifest.open("r", encoding="utf-8") as source:
        for line_index, line in enumerate(source):
            handles[line_index % shards].write(line)
finally:
    for handle in handles:
        handle.close()
PY
}

run_ctc_dataset() {
  local dataset="$1"
  local gpu="$2"
  local manifest
  manifest="$(manifest_path "${dataset}")"
  local output="${PREDICTION_DIR}/${dataset}.ctc.jsonl"
  local preview="${PREDICTION_DIR}/${dataset}.ctc.preview.txt"
  local log_path="${LOG_DIR}/${dataset}.ctc.log"
  local limit_args=()
  if [[ "${CTC_LIMIT}" -gt 0 ]]; then
    limit_args=(--limit "${CTC_LIMIT}")
  fi
  mkdir -p "${PREDICTION_DIR}" "${LOG_DIR}"
  log "ctc start dataset=${dataset} gpu=${gpu} expected=$(line_count "${manifest}") log=${log_path}"
  env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m rwkvasr.cli.predict_ctc_labeled \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --manifest-path "${manifest}" \
    --output-path "${output}" \
    --preview-path "${preview}" \
    --preview-count 24 \
    --device cuda:0 \
    --batch-size "${CTC_BATCH_SIZE}" \
    --num-workers "${CTC_NUM_WORKERS}" \
    --mode bi \
    --beam-size "${CTC_BEAM_SIZE}" \
    --token-prune-topk "${CTC_TOKEN_PRUNE_TOPK}" \
    --text-normalization "${CTC_TEXT_NORMALIZATION}" \
    --progress-interval "${CTC_PROGRESS_INTERVAL}" \
    "${limit_args[@]}" \
    --save-debug-lengths \
    >"${log_path}" 2>&1
  log "ctc done dataset=${dataset} status=$? lines=$(line_count "${output}")"
}

run_ctc_dataset_sharded() {
  local dataset="$1"
  if [[ "${CTC_LIMIT}" -gt 0 ]]; then
    echo "CTC_LIMIT is incompatible with CTC_SHARD_DATASETS; use dataset-level parallelism" >&2
    exit 1
  fi
  local manifest
  manifest="$(manifest_path "${dataset}")"
  IFS=',' read -r -a devices <<< "${DEVICES}"
  local shard_count="${CTC_SHARD_COUNT}"
  if [[ -z "${shard_count}" ]]; then
    shard_count="${#devices[@]}"
  fi
  if [[ "${shard_count}" -le 1 ]]; then
    run_ctc_dataset "${dataset}" "${devices[0]}"
    return
  fi

  local split_dir="${OUTPUT_DIR}/manifest_parts/${dataset}"
  local output="${PREDICTION_DIR}/${dataset}.ctc.jsonl"
  local preview="${PREDICTION_DIR}/${dataset}.ctc.preview.txt"
  mkdir -p "${PREDICTION_DIR}" "${LOG_DIR}"
  split_manifest "${manifest}" "${split_dir}" "${shard_count}"
  rm -f "${output}" "${preview}"
  log "ctc sharded start dataset=${dataset} shards=${shard_count} expected=$(line_count "${manifest}")"

  local pids=()
  local failed=0
  local index
  for index in $(seq 0 $((shard_count - 1))); do
    local part_manifest="${split_dir}/part-$(printf '%02d' "${index}").jsonl"
    if [[ ! -s "${part_manifest}" ]]; then
      continue
    fi
    local gpu="${devices[$((index % ${#devices[@]}))]}"
    local part_output="${PREDICTION_DIR}/${dataset}.ctc.part-$(printf '%02d' "${index}").jsonl"
    local part_preview="${PREDICTION_DIR}/${dataset}.ctc.part-$(printf '%02d' "${index}").preview.txt"
    local log_path="${LOG_DIR}/${dataset}.ctc.part-$(printf '%02d' "${index}").log"
    log "ctc shard start dataset=${dataset} part=${index} gpu=${gpu} expected=$(line_count "${part_manifest}") log=${log_path}"
    env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m rwkvasr.cli.predict_ctc_labeled \
      --checkpoint-path "${CHECKPOINT_PATH}" \
      --manifest-path "${part_manifest}" \
      --output-path "${part_output}" \
      --preview-path "${part_preview}" \
      --preview-count 24 \
      --device cuda:0 \
      --batch-size "${CTC_BATCH_SIZE}" \
      --num-workers "${CTC_NUM_WORKERS}" \
      --mode bi \
      --beam-size "${CTC_BEAM_SIZE}" \
      --token-prune-topk "${CTC_TOKEN_PRUNE_TOPK}" \
      --text-normalization "${CTC_TEXT_NORMALIZATION}" \
      --progress-interval "${CTC_PROGRESS_INTERVAL}" \
      --save-debug-lengths \
      >"${log_path}" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "ctc sharded benchmark failed for dataset=${dataset}; see ${LOG_DIR}" >&2
    exit 1
  fi

  : >"${output}"
  for index in $(seq 0 $((shard_count - 1))); do
    local part_output="${PREDICTION_DIR}/${dataset}.ctc.part-$(printf '%02d' "${index}").jsonl"
    if [[ -s "${part_output}" ]]; then
      cat "${part_output}" >>"${output}"
    fi
  done
  for index in $(seq 0 $((shard_count - 1))); do
    local part_preview="${PREDICTION_DIR}/${dataset}.ctc.part-$(printf '%02d' "${index}").preview.txt"
    if [[ -s "${part_preview}" ]]; then
      cp "${part_preview}" "${preview}"
      break
    fi
  done
  log "ctc sharded done dataset=${dataset} lines=$(line_count "${output}")"
}

run_ar_dataset() {
  local dataset="$1"
  local gpu="$2"
  local manifest
  manifest="$(manifest_path "${dataset}")"
  local ctc_output="${PREDICTION_DIR}/${dataset}.ctc.jsonl"
  local output="${PREDICTION_DIR}/${dataset}.ar.jsonl"
  local preview="${PREDICTION_DIR}/${dataset}.ar.preview.txt"
  local log_path="${LOG_DIR}/${dataset}.ar.log"
  if [[ ! -s "${ctc_output}" ]]; then
    echo "CTC prediction cache missing for AR dataset=${dataset}: ${ctc_output}" >&2
    exit 1
  fi
  local sample_args=()
  if [[ "${AR_DO_SAMPLE}" == "1" || "${AR_DO_SAMPLE}" == "true" || "${AR_DO_SAMPLE}" == "True" ]]; then
    sample_args+=(--do-sample)
  fi
  mkdir -p "${PREDICTION_DIR}" "${LOG_DIR}"
  log "ar start dataset=${dataset} gpu=${gpu} expected=$(line_count "${manifest}") log=${log_path} sampling=temp:${AR_TEMPERATURE},top_k:${AR_TOP_K},top_p:${AR_TOP_P},seed:${AR_SEED}"
  env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m rwkvasr.cli.predict_rwkv_decoder_labeled \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --manifest-path "${manifest}" \
    --decoder-ctc-draft-cache-path "${ctc_output}" \
    --output-path "${output}" \
    --preview-path "${preview}" \
    --preview-count 24 \
    --device cuda:0 \
    --batch-size "${AR_BATCH_SIZE}" \
    --num-workers "${AR_NUM_WORKERS}" \
    --mode bi \
    --text-normalization "${AR_TEXT_NORMALIZATION}" \
    --max-new-tokens-factor "${AR_MAX_NEW_TOKENS_FACTOR}" \
    "${sample_args[@]}" \
    --temperature "${AR_TEMPERATURE}" \
    --top-k "${AR_TOP_K}" \
    --top-p "${AR_TOP_P}" \
    --seed "${AR_SEED}" \
    --progress-interval "${AR_PROGRESS_INTERVAL}" \
    >"${log_path}" 2>&1
  log "ar done dataset=${dataset} status=$? lines=$(line_count "${output}")"
}

run_parallel_stage() {
  local branch="$1"
  shift
  IFS=',' read -r -a devices <<< "${DEVICES}"
  if [[ "${#devices[@]}" -lt 1 ]]; then
    echo "DEVICES must contain at least one GPU id" >&2
    exit 1
  fi
  if [[ "${branch}" == "ctc" ]] && truthy "${CTC_SHARD_DATASETS}"; then
    local dataset
    for dataset in "$@"; do
      run_ctc_dataset_sharded "${dataset}"
    done
    return
  fi
  local index=0
  local failed=0
  for dataset in "$@"; do
    local gpu="${devices[$((index % ${#devices[@]}))]}"
    if [[ "${branch}" == "ctc" ]]; then
      run_ctc_dataset "${dataset}" "${gpu}" &
    else
      run_ar_dataset "${dataset}" "${gpu}" &
    fi
    index=$((index + 1))
  done
  for job in $(jobs -rp); do
    if ! wait "${job}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "${branch} benchmark stage failed; see ${LOG_DIR}" >&2
    exit 1
  fi
}

summarize() {
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_asr_eval_results.py" \
    --prediction-dir "${PREDICTION_DIR}" \
    --output-json "${METRICS_JSON}" \
    --output-md "${METRICS_MD}" \
    --normalization "${METRIC_NORMALIZATION}"
  cat "${METRICS_MD}"
}

main() {
  cd "${REPO_ROOT}"
  resolve_checkpoint
  mkdir -p "${OUTPUT_DIR}" "${PREDICTION_DIR}" "${LOG_DIR}"
  log "checkpoint=${CHECKPOINT_PATH}"
  log "output_dir=${OUTPUT_DIR}"
  if [[ "${PREPARE_MANIFESTS}" == "1" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_public_eval_manifests.py" \
      --output-dir "${MANIFEST_DIR}" \
      --audio-cache-dir "${EVAL_ROOT}/audio"
  fi
  if [[ ! -s "${MANIFEST_DIR}/manifest_summary.json" ]]; then
    echo "manifest summary missing: ${MANIFEST_DIR}/manifest_summary.json; set PREPARE_MANIFESTS=1" >&2
    exit 1
  fi

  log "ctc stage 1"
  run_parallel_stage ctc "${DATASETS[0]}" "${DATASETS[1]}" "${DATASETS[2]}" "${DATASETS[3]}"
  log "ctc stage 2"
  run_parallel_stage ctc "${DATASETS[4]}"
  summarize

  local do_ar="${RUN_AR}"
  if [[ "${do_ar}" == "auto" ]]; then
    do_ar="$(decoder_enabled)"
  fi
  if [[ "${do_ar}" == "1" || "${do_ar}" == "true" || "${do_ar}" == "True" ]]; then
    log "ar stage 1"
    run_parallel_stage ar "${DATASETS[0]}" "${DATASETS[1]}" "${DATASETS[2]}" "${DATASETS[3]}"
    log "ar stage 2"
    run_parallel_stage ar "${DATASETS[4]}"
    summarize
  else
    log "skip AR benchmark; decoder_enabled=false or RUN_AR=${RUN_AR}"
  fi
  log "finished output_dir=${OUTPUT_DIR}"
}

main "$@"
