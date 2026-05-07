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
AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-}"
AR_MAX_NEW_TOKENS_FACTOR="${AR_MAX_NEW_TOKENS_FACTOR:-0.5}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
EXTRA_ARGS=()
if [[ -n "${AR_MAX_NEW_TOKENS}" ]]; then
  EXTRA_ARGS+=(--ar-max-new-tokens "${AR_MAX_NEW_TOKENS}")
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
