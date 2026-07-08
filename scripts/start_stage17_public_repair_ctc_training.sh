#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE16_RUN="${STAGE16_RUN:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage16_public80_hard20_ctc_from_stage15_werbest_bs12_lr5e7_zero1_nockpt_noaug_nodirdrop_4x4090}"

export BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage16_public80_hard20_ctc_from_stage15_werbest_4x4090_deepspeed.yaml}"
export INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${STAGE16_RUN}/step-5000.pt}"

export CURRICULUM_ROOT="${CURRICULUM_ROOT:-/media/usbhd/training_data/asr/curriculum/stage17_public_repair_light_hard_mix}"
export STAGE_NAME="${STAGE_NAME:-stage17_public90_hard10}"
export CLEAN_RATIO="${CLEAN_RATIO:-0.90}"
export MIX_SEED="${MIX_SEED:-20260620}"
export REBUILD_CURRICULUM="${REBUILD_CURRICULUM:-0}"

export RUN_DIR="${RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage17_public90_hard10_ctc_from_stage16_step5000_bs12_lr2p5e7_zero1_nockpt_noaug_nodirdrop_4x4090}"
export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage17_public90_hard10_ctc_from_stage16_step5000_4x4090_deepspeed.yaml}"
export LR="${LR:-2.5e-7}"
export MAX_STEPS="${MAX_STEPS:-6000}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export MASTER_PORT="${MASTER_PORT:-29617}"

export SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage17_public_repair_ctc}"
export SIDECAR_LIMIT="${SIDECAR_LIMIT:-80}"
export SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:16,clean_aishell:aishell3_*.tar:16,clean_cv_en:commonvoice_en_*.tar:16,clean_cv_cn:commonvoice_cn_*.tar:16,gigaspeech:GSXL-*.tar:8,wenetspeech:WSL-*.tar:8}"

exec "${REPO_ROOT}/scripts/start_stage16_public_anchor_light_hard_ctc_training.sh" "$@"
