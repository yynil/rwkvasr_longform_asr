#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE19_RUN="${STAGE19_RUN:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage19_publicweak_cleanreplay_ctc_from_stage15_werbest_bs12_lr3e7_zero1_nockpt_noaug_nodirdrop_4x4090}"
CLEAN_ROOT="${CLEAN_ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"

export BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage19_publicweak_cleanreplay_ctc_from_stage15_werbest_4x4090_deepspeed.yaml}"
export INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${STAGE19_RUN}/wercer_best.pt}"

export CLEAN_ROOT
export CLEAN_LENGTH_INDEX="${CLEAN_LENGTH_INDEX:-${CLEAN_ROOT}/stages/stage18_publicweak_clean_replay/webdataset_lengths.jsonl}"
export HARD_ROOT="${HARD_ROOT:-/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset}"
export HARD_LENGTH_INDEX="${HARD_LENGTH_INDEX:-${HARD_ROOT}/webdataset_lengths.jsonl}"

export CURRICULUM_ROOT="${CURRICULUM_ROOT:-/media/usbhd/training_data/asr/curriculum/stage20_publicweak_light_hard_mix}"
export STAGE_NAME="${STAGE_NAME:-stage20_publicweak95_hard5}"
export TARGET_SAMPLES="${TARGET_SAMPLES:-600000}"
export CLEAN_RATIO="${CLEAN_RATIO:-0.95}"
export MIX_EVAL_RATIO="${MIX_EVAL_RATIO:-0.001}"
export MIX_SEED="${MIX_SEED:-20260620}"
export REBUILD_CURRICULUM="${REBUILD_CURRICULUM:-0}"

export RUN_DIR="${RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage20_publicweak95_hard5_ctc_from_stage19_werbest_bs12_lr3e7_zero1_nockpt_noaug_nodirdrop_4x4090}"
export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage20_publicweak95_hard5_ctc_from_stage19_werbest_4x4090_deepspeed.yaml}"
export LR="${LR:-3.0e-7}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export MASTER_PORT="${MASTER_PORT:-29620}"

export SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage20_publicweak_light_hard_ctc}"
export SIDECAR_LIMIT="${SIDECAR_LIMIT:-96}"
export SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:16,clean_aishell:aishell3_*.tar:16,clean_cv_en:commonvoice_en_*.tar:16,clean_cv_cn:commonvoice_cn_*.tar:16,gigaspeech:GSXL-*.tar:16,wenetspeech:WSL-*.tar:16}"

exec "${REPO_ROOT}/scripts/start_stage16_public_anchor_light_hard_ctc_training.sh" "$@"
