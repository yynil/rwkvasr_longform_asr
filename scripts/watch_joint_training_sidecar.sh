#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/emilia_en_zh_joint_rwkv7g1_ctc_ar_fullaudio_template_eos0_4x4090}"
TMUX_TARGET="${TMUX_TARGET:-training:0}"
POLL_SECONDS="${POLL_SECONDS:-3600}"
CHECKPOINT_STABLE_SECONDS="${CHECKPOINT_STABLE_SECONDS:-45}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-12}"
PREVIEW_COUNT="${PREVIEW_COUNT:-12}"
BEAM_SIZE="${BEAM_SIZE:-4}"
TOKEN_PRUNE_TOPK="${TOKEN_PRUNE_TOPK:-32}"
TEXT_NORMALIZATION="${TEXT_NORMALIZATION:-}"
AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-}"
AR_MAX_NEW_TOKENS_FACTOR="${AR_MAX_NEW_TOKENS_FACTOR:-0.5}"
SOURCE_QUOTAS="${SOURCE_QUOTAS:-}"

if [[ -z "${SOURCE_QUOTAS}" && "${RUN_DIR}" == *gigaspeech_wenetspeech* ]]; then
  GIGA_LIMIT=$(( LIMIT / 2 ))
  WENET_LIMIT=$(( LIMIT - GIGA_LIMIT ))
  SOURCE_QUOTAS="gigaspeech:GSXL-*.tar:${GIGA_LIMIT},wenetspeech:WSL-*.tar:${WENET_LIMIT}"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
EXTRA_ARGS=()
if [[ -n "${AR_MAX_NEW_TOKENS}" ]]; then
  EXTRA_ARGS+=(--ar-max-new-tokens "${AR_MAX_NEW_TOKENS}")
fi
if [[ -n "${TEXT_NORMALIZATION}" ]]; then
  EXTRA_ARGS+=(--text-normalization "${TEXT_NORMALIZATION}")
fi
if [[ -n "${SOURCE_QUOTAS}" ]]; then
  IFS=',' read -r -a SOURCE_QUOTA_ARRAY <<< "${SOURCE_QUOTAS}"
  for SOURCE_QUOTA in "${SOURCE_QUOTA_ARRAY[@]}"; do
    if [[ -n "${SOURCE_QUOTA}" ]]; then
      EXTRA_ARGS+=(--source-quota "${SOURCE_QUOTA}")
    fi
  done
fi

exec "${PYTHON_BIN}" -m rwkvasr.cli.watch_checkpoint_sidecar \
  --run-dir "${RUN_DIR}" \
  --tmux-target "${TMUX_TARGET}" \
  --poll-seconds "${POLL_SECONDS}" \
  --checkpoint-stable-seconds "${CHECKPOINT_STABLE_SECONDS}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --beam-size "${BEAM_SIZE}" \
  --token-prune-topk "${TOKEN_PRUNE_TOPK}" \
  --ar-max-new-tokens-factor "${AR_MAX_NEW_TOKENS_FACTOR}" \
  --limit "${LIMIT}" \
  --preview-count "${PREVIEW_COUNT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
