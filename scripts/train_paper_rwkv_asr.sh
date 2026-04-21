#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHRUN_BIN="${REPO_ROOT}/.venv/bin/torchrun"

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not found at ${TORCHRUN_BIN}" >&2
  exit 1
fi

MODE="dirdrop_both"
CONFIG_YAML=""
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    bi_baseline|bi_baseline_spm4k|bi_baseline_spm8k|dirdrop_both|emilia_zh_bi_baseline|emilia_zh_dirdrop_both|emilia_en_zh_bi_baseline|emilia_en_zh_bi_baseline_whisper|emilia_en_zh_bi_baseline_qwen3|emilia_en_zh_dirdrop_both|emilia_en_zh_bi_baseline_spm8k|emilia_en_zh_dirdrop_both_spm8k)
      MODE="$1"
      shift
      ;;
    --config-yaml)
      CONFIG_YAML="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${CONFIG_YAML}" ]]; then
  case "${MODE}" in
    bi_baseline)
      CONFIG_YAML="${REPO_ROOT}/configs/paper_bi_baseline_4x4090_deepspeed.yaml"
      ;;
    bi_baseline_spm4k)
      CONFIG_YAML="${REPO_ROOT}/configs/paper_bi_baseline_spm4k_4x4090_deepspeed.yaml"
      ;;
    bi_baseline_spm8k)
      CONFIG_YAML="${REPO_ROOT}/configs/paper_bi_baseline_spm8k_4x4090_deepspeed.yaml"
      ;;
    dirdrop_both)
      CONFIG_YAML="${REPO_ROOT}/configs/paper_dirdrop_both_4x4090_deepspeed.yaml"
      ;;
    emilia_zh_bi_baseline)
      CONFIG_YAML="${REPO_ROOT}/configs/emilia_zh_bi_baseline_4x4090_deepspeed.yaml"
      ;;
    emilia_zh_dirdrop_both)
      CONFIG_YAML="${REPO_ROOT}/configs/emilia_zh_dirdrop_both_4x4090_deepspeed.yaml"
      ;;
    emilia_en_zh_bi_baseline)
      CONFIG_YAML="${REPO_ROOT}/configs/emilia_en_zh_bi_baseline_4x4090_deepspeed.yaml"
      ;;
    emilia_en_zh_bi_baseline_whisper)
      CONFIG_YAML="${REPO_ROOT}/configs/emilia_en_zh_bi_baseline_whisper_4x4090_deepspeed.yaml"
      ;;
    emilia_en_zh_bi_baseline_qwen3)
      CONFIG_YAML="${REPO_ROOT}/configs/emilia_en_zh_bi_baseline_qwen3_4x4090_deepspeed.yaml"
      ;;
    emilia_en_zh_dirdrop_both)
      CONFIG_YAML="${REPO_ROOT}/configs/emilia_en_zh_dirdrop_both_4x4090_deepspeed.yaml"
      ;;
    emilia_en_zh_bi_baseline_spm8k)
      CONFIG_YAML="${REPO_ROOT}/configs/emilia_en_zh_bi_baseline_spm8k_4x4090_deepspeed.yaml"
      ;;
    emilia_en_zh_dirdrop_both_spm8k)
      CONFIG_YAML="${REPO_ROOT}/configs/emilia_en_zh_dirdrop_both_spm8k_4x4090_deepspeed.yaml"
      ;;
    *)
      echo "Unsupported mode: ${MODE}" >&2
      exit 1
      ;;
  esac
fi

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Training config not found: ${CONFIG_YAML}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  "${TORCHRUN_BIN}"
  --standalone
  --nnodes 1
  --nproc_per_node "${NUM_GPUS}"
  --master_port "${MASTER_PORT}"
  -m rwkvasr.cli.train_ctc_deepspeed
  --config-yaml "${CONFIG_YAML}"
  "${EXTRA_ARGS[@]}"
)

echo "Launch command:"
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${REPO_ROOT}"
exec "${CMD[@]}"
