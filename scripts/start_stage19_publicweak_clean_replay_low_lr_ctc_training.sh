#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE15_RUN="${STAGE15_RUN:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_bs12_lr1e6_zero1_nockpt_noaug_nodirdrop_4x4090}"

export STAGE15_RUN
export BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_4x4090_deepspeed.yaml}"
export INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${STAGE15_RUN}/wercer_best.pt}"

export REPLAY_STAGE_NAME="${REPLAY_STAGE_NAME:-stage18_publicweak_clean_replay}"
export REPLAY_SEED="${REPLAY_SEED:-20260620}"
export REBUILD_REPLAY="${REBUILD_REPLAY:-0}"

export RUN_DIR="${RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage19_publicweak_cleanreplay_ctc_from_stage15_werbest_bs12_lr3e7_zero1_nockpt_noaug_nodirdrop_4x4090}"
export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage19_publicweak_cleanreplay_ctc_from_stage15_werbest_4x4090_deepspeed.yaml}"
export LR="${LR:-3.0e-7}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export MASTER_PORT="${MASTER_PORT:-29619}"

export SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage19_publicweak_clean_low_lr_ctc}"
export SIDECAR_LIMIT="${SIDECAR_LIMIT:-96}"
export SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:24,clean_aishell:aishell3_*.tar:24,clean_cv_en:commonvoice_en_*.tar:24,clean_cv_cn:commonvoice_cn_*.tar:24}"

exec "${REPO_ROOT}/scripts/start_stage18_publicweak_clean_replay_ctc_training.sh" "$@"
