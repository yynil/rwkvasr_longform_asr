#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE21_RUN="${STAGE21_RUN:-/media/usbhd/rwkvasr_runs/stage21_checkpoint_soups_20260620}"

export BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_4x4090_deepspeed.yaml}"
export INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${STAGE21_RUN}/wercer_best.pt}"

export CLEAN_ROOT="${CLEAN_ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"
export CLEAN_LENGTH_INDEX="${CLEAN_LENGTH_INDEX:-${CLEAN_ROOT}/stages/stage12b_source_balanced_clean_public_anchor/webdataset_lengths.jsonl}"
export HARD_ROOT="${HARD_ROOT:-/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset}"
export HARD_LENGTH_INDEX="${HARD_LENGTH_INDEX:-${HARD_ROOT}/webdataset_lengths.jsonl}"

export CURRICULUM_ROOT="${CURRICULUM_ROOT:-/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix}"
export STAGE_NAME="${STAGE_NAME:-stage22_public90_hard10_from_stage21_soup}"
export TARGET_SAMPLES="${TARGET_SAMPLES:-600000}"
export CLEAN_RATIO="${CLEAN_RATIO:-0.90}"
export MIX_EVAL_RATIO="${MIX_EVAL_RATIO:-0.001}"
export MIX_SEED="${MIX_SEED:-20260620}"
export REBUILD_CURRICULUM="${REBUILD_CURRICULUM:-0}"

export RUN_DIR="${RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage22_public90_hard10_ctc_from_stage21_soup_bs12_lr2e7_zero1_nockpt_noaug_nodirdrop_4x4090}"
export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage22_public90_hard10_ctc_from_stage21_soup_4x4090_deepspeed.yaml}"
export LR="${LR:-2.0e-7}"
export MAX_STEPS="${MAX_STEPS:-4000}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export MASTER_PORT="${MASTER_PORT:-29622}"

export SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage22_stage21_soup_clean_repair_ctc}"
export SIDECAR_LIMIT="${SIDECAR_LIMIT:-96}"
export SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:18,clean_aishell:aishell3_*.tar:18,clean_cv_en:commonvoice_en_*.tar:18,clean_cv_cn:commonvoice_cn_*.tar:18,gigaspeech:GSXL-*.tar:12,wenetspeech:WSL-*.tar:12}"

exec "${REPO_ROOT}/scripts/start_stage16_public_anchor_light_hard_ctc_training.sh" "$@"
