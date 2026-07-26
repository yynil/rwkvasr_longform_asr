#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/assets/fun-asr-nano-2512}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/artifacts/eval_benchmarks/manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/eval_benchmarks/nano_ctc}"
PREDICTION_DIR="${PREDICTION_DIR:-${OUTPUT_DIR}/predictions}"
REPORT_DIR="${REPORT_DIR:-${OUTPUT_DIR}/reports}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
DEVICES="${DEVICES:-0,1,2,3}"
NORMALIZATION="${NORMALIZATION:-ctc}"
LIMIT="${LIMIT:-0}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-100}"

DATASETS=(
  librispeech_test_clean
  librispeech_test_other
  aishell1_test
  commonvoice_en_test
  wenetspeech_test_net
)

language_for_dataset() {
  case "$1" in
    librispeech_test_clean|librispeech_test_other|commonvoice_en_test) printf 'en\n' ;;
    aishell1_test|wenetspeech_test_net) printf 'zh\n' ;;
    *) echo "unsupported dataset: $1" >&2; exit 1 ;;
  esac
}

line_count() {
  local path="$1"
  wc -l <"${path}" | tr -d ' '
}

run_dataset() {
  local dataset="$1"
  local gpu="$2"
  local manifest="${MANIFEST_DIR}/${dataset}.jsonl"
  local prediction="${PREDICTION_DIR}/${dataset}.ctc.jsonl"
  local report="${REPORT_DIR}/${dataset}.json"
  local log="${LOG_DIR}/${dataset}.log"
  local limit_args=()
  if [[ "${LIMIT}" -gt 0 ]]; then
    limit_args=(--limit "${LIMIT}")
  fi
  if [[ ! -s "${manifest}" ]]; then
    echo "manifest missing or empty: ${manifest}" >&2
    exit 1
  fi
  env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m rwkvasr.cli.eval_funasr_nano_ctc \
    --manifest-path "${manifest}" \
    --model-path "${MODEL_PATH}" \
    --predictions-path "${prediction}" \
    --report-path "${report}" \
    --language "$(language_for_dataset "${dataset}")" \
    --device cuda:0 \
    --normalization "${NORMALIZATION}" \
    --progress-interval "${PROGRESS_INTERVAL}" \
    "${limit_args[@]}" \
    >"${log}" 2>&1
  local expected
  expected="$(line_count "${manifest}")"
  if [[ "${LIMIT}" -gt 0 && "${LIMIT}" -lt "${expected}" ]]; then
    expected="${LIMIT}"
  fi
  local actual
  actual="$(line_count "${prediction}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${dataset}: Nano prediction coverage ${actual}/${expected}" >&2
    exit 1
  fi
  printf '[nano-public-eval] dataset=%s gpu=%s samples=%s\n' \
    "${dataset}" "${gpu}" "${actual}"
}

run_group() {
  local start="$1"
  local end="$2"
  IFS=',' read -r -a devices <<<"${DEVICES}"
  local pids=()
  local failed=0
  local index
  for index in $(seq "${start}" "${end}"); do
    local dataset="${DATASETS[${index}]}"
    local gpu="${devices[$(((index - start) % ${#devices[@]}))]}"
    run_dataset "${dataset}" "${gpu}" &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "Nano public benchmark group failed; see ${LOG_DIR}" >&2
    exit 1
  fi
}

main() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "python not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -d "${MODEL_PATH}" && ! -s "${MODEL_PATH}" ]]; then
    echo "Nano model path unavailable: ${MODEL_PATH}" >&2
    exit 1
  fi
  mkdir -p "${PREDICTION_DIR}" "${REPORT_DIR}" "${LOG_DIR}"
  run_group 0 3
  run_group 4 4
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/summarize_asr_eval_results.py" \
    --prediction-dir "${PREDICTION_DIR}" \
    --output-json "${OUTPUT_DIR}/metrics.json" \
    --output-md "${OUTPUT_DIR}/metrics.md" \
    --normalization "${NORMALIZATION}"
  printf '[nano-public-eval] complete output_dir=%s\n' "${OUTPUT_DIR}"
}

main "$@"
