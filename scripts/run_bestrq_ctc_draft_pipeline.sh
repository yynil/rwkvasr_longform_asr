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

SOURCE_CONFIG="${SOURCE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage6c_clean50_hard50_joint_chatlang_hardmix_4x4090_deepspeed.yaml}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage6c_clean50_hard50_joint_chatlang_gigaspeech_wenetspeech_clean50_hard50_from_prev_ctc0p8_ar0p2_bs4_lr2e5_zero1_nockpt_4x4090}"
SOURCE_TARGET_STEP="${SOURCE_TARGET_STEP:-3000}"
CHECKPOINT_STABLE_SECONDS="${CHECKPOINT_STABLE_SECONDS:-90}"
POLL_SECONDS="${POLL_SECONDS:-60}"

TRAINING_SESSION="${TRAINING_SESSION:-training}"
SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage6}"
STAGE6_AUTO_SESSION="${STAGE6_AUTO_SESSION:-stage6_auto}"
MASTER_PORT="${MASTER_PORT:-29568}"
NUM_GPUS="${NUM_GPUS:-4}"

BESTRQ_STEPS="${BESTRQ_STEPS:-5000}"
BESTRQ_DEVICE="${BESTRQ_DEVICE:-cuda:0}"
BESTRQ_CUDA_VISIBLE_DEVICES="${BESTRQ_CUDA_VISIBLE_DEVICES:-0}"
BESTRQ_BATCH_SIZE="${BESTRQ_BATCH_SIZE:-2}"
BESTRQ_TOKEN_BUDGET="${BESTRQ_TOKEN_BUDGET:-1800}"
BESTRQ_NUM_WORKERS="${BESTRQ_NUM_WORKERS:-4}"
BESTRQ_LR="${BESTRQ_LR:-2.0e-5}"
BESTRQ_RUN_DIR_ROOT="${BESTRQ_RUN_DIR_ROOT:-/media/usbhd/rwkvasr_runs}"

CTC_STEPS="${CTC_STEPS:-5000}"
CTC_BATCH_SIZE="${CTC_BATCH_SIZE:-12}"
CTC_TOKEN_BUDGET="${CTC_TOKEN_BUDGET:-12000}"
CTC_LR="${CTC_LR:-5.0e-5}"
CTC_RUN_DIR_ROOT="${CTC_RUN_DIR_ROOT:-/media/usbhd/rwkvasr_runs}"

DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:0}"
DRAFT_BATCH_SIZE="${DRAFT_BATCH_SIZE:-4}"
DRAFT_NUM_WORKERS="${DRAFT_NUM_WORKERS:-4}"
DRAFT_SOURCE_QUOTAS="${DRAFT_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:10000,clean_aishell:aishell3_*.tar:10000,clean_cv_en:commonvoice_en_*.tar:10000,clean_cv_cn:commonvoice_cn_*.tar:10000,gigaspeech:GSXL-*.tar:12000,wenetspeech:WSL-*.tar:12000}"
DRAFT_ONLY="${DRAFT_ONLY:-0}"
DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-}"

GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${REPO_ROOT}/configs/generated}"
mkdir -p "${GENERATED_CONFIG_DIR}"

log() {
  printf '[bestrq-pipeline] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*" >&2
}

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

latest_step_checkpoint() {
  find "${SOURCE_RUN_DIR}" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^step-\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

checkpoint_is_stable() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  local size_a size_b
  size_a="$(stat -c '%s' "${path}")"
  sleep "${CHECKPOINT_STABLE_SECONDS}"
  [[ -s "${path}" ]] || return 1
  size_b="$(stat -c '%s' "${path}")"
  [[ "${size_a}" == "${size_b}" ]]
}

wait_for_source_checkpoint() {
  local checkpoint="${SOURCE_RUN_DIR}/step-${SOURCE_TARGET_STEP}.pt"
  log "waiting for stable source checkpoint ${checkpoint}"
  while true; do
    if checkpoint_is_stable "${checkpoint}"; then
      echo "${checkpoint}"
      return 0
    fi
    local latest
    latest="$(latest_step_checkpoint || true)"
    log "latest source step=${latest:-none}; waiting for step-${SOURCE_TARGET_STEP}.pt"
    sleep "${POLL_SECONDS}"
  done
}

stop_session() {
  local session="$1"
  if tmux_has_session "${session}"; then
    log "sending Ctrl-C to ${session}"
    tmux send-keys -t "${session}:0" C-c || true
  fi
}

source_training_active() {
  local base
  base="$(basename "${SOURCE_CONFIG}")"
  pgrep -af -- "rwkvasr.cli.train_ctc_deepspeed.*${base}" >/dev/null 2>&1 \
    || pgrep -af -- "torchrun.*${base}" >/dev/null 2>&1
}

stop_current_training() {
  stop_session "${STAGE6_AUTO_SESSION}"
  stop_session "${SIDECAR_SESSION}"
  stop_session "${TRAINING_SESSION}"
  for _ in {1..90}; do
    if ! source_training_active; then
      log "source training stopped"
      return 0
    fi
    sleep 5
  done
  log "source training still active after Ctrl-C; leaving it untouched for manual inspection"
  return 1
}

write_bestrq_config() {
  local source_checkpoint="$1"
  local source_step="$2"
  BESTRQ_RUN_DIR="${BESTRQ_RUN_DIR_ROOT}/sensevoice_rwkv_bestrq_long_from_stage6c_step${source_step}_bs${BESTRQ_BATCH_SIZE}_budget${BESTRQ_TOKEN_BUDGET}_lr2e5_gpu0"
  BESTRQ_CONFIG="${GENERATED_CONFIG_DIR}/sensevoice_rwkv_bestrq_long_from_stage6c_step${source_step}.yaml"
  export SOURCE_CONFIG source_checkpoint BESTRQ_RUN_DIR BESTRQ_CONFIG BESTRQ_STEPS BESTRQ_DEVICE
  export BESTRQ_BATCH_SIZE BESTRQ_TOKEN_BUDGET BESTRQ_NUM_WORKERS BESTRQ_LR
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from rwkvasr.config import load_yaml, save_yaml

base = dict(load_yaml(os.environ["SOURCE_CONFIG"]))
cfg = {
    "output_dir": os.environ["BESTRQ_RUN_DIR"],
    "webdataset_root": base["webdataset_root"],
    "webdataset_length_index_path": base["webdataset_length_index_path"],
    "webdataset_bucket_manifest_path": base["webdataset_bucket_manifest_path"],
    "webdataset_split": "train",
    "webdataset_eval_ratio": base.get("webdataset_eval_ratio", 0.0),
    "webdataset_hash_seed": base.get("webdataset_hash_seed", 0),
    "webdataset_split_by": base.get("webdataset_split_by", "sample_id"),
    "webdataset_utt_id_key": base.get("webdataset_utt_id_key", "id"),
    "tokenizer_type": base["tokenizer_type"],
    "tokenizer_model_path": base.get("tokenizer_model_path"),
    "tokenizer_language": base.get("tokenizer_language"),
    "tokenizer_task": base.get("tokenizer_task"),
    "tokenizer_append_eos": bool(base.get("tokenizer_append_eos", False)),
    "text_normalization": "ctc",
    "vocab_size": int(base["vocab_size"]),
    "blank_id": int(base["blank_id"]),
    "feature_extractor_type": base["feature_extractor_type"],
    "input_dim": int(base["input_dim"]),
    "n_embd": int(base["n_embd"]),
    "encoder_output_dim": base.get("encoder_output_dim"),
    "dim_att": int(base["dim_att"]),
    "dim_ff": int(base["dim_ff"]),
    "num_layers": int(base["num_layers"]),
    "sensevoice_tp_blocks": int(base.get("sensevoice_tp_blocks", 20)),
    "head_size": int(base.get("head_size", 64)),
    "backend": base.get("backend", "cuda_clampw"),
    "conv_kernel_size": int(base.get("conv_kernel_size", 31)),
    "dropout": float(base.get("dropout", 0.1)),
    "frontend_type": base["frontend_type"],
    "cmvn_file": base.get("cmvn_file"),
    "cmvn_is_json": bool(base.get("cmvn_is_json", True)),
    "best_rq_codebook_size": 8192,
    "best_rq_projection_dim": 16,
    "best_rq_mask_prob": 0.15,
    "best_rq_mask_span_length": 10,
    "best_rq_quantizer_seed": 20260609,
    "init_checkpoint_path": os.environ["source_checkpoint"],
    "device": os.environ["BESTRQ_DEVICE"],
    "batch_size": int(os.environ["BESTRQ_BATCH_SIZE"]),
    "batch_token_budget": int(os.environ["BESTRQ_TOKEN_BUDGET"]),
    "length_bucket_frame_budget": int(os.environ["BESTRQ_TOKEN_BUDGET"]),
    "skip_oversized_samples": True,
    "num_workers": int(os.environ["BESTRQ_NUM_WORKERS"]),
    "decoded_batch_prefetch": 1,
    "max_open_shards_per_worker": 4,
    "bucket_source_interleave": True,
    "max_steps": int(os.environ["BESTRQ_STEPS"]),
    "save_every": 500,
    "log_every": 10,
    "lr": float(os.environ["BESTRQ_LR"]),
    "weight_decay": float(base.get("weight_decay", 0.1)),
    "beta1": float(base.get("beta1", 0.9)),
    "beta2": float(base.get("beta2", 0.99)),
    "eps": float(base.get("eps", 1e-8)),
    "gradient_checkpointing": True,
    "use_bf16": True,
}
save_yaml(os.environ["BESTRQ_CONFIG"], cfg)
Path(os.environ["BESTRQ_RUN_DIR"]).mkdir(parents=True, exist_ok=True)
print(os.environ["BESTRQ_CONFIG"])
PY
}

run_bestrq() {
  local config="$1"
  mkdir -p "${BESTRQ_RUN_DIR}/logs"
  log "starting Best-RQ pretrain config=${config}"
  CUDA_VISIBLE_DEVICES="${BESTRQ_CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m rwkvasr.cli.train_best_rq --config-yaml "${config}" \
    2>&1 | tee -a "${BESTRQ_RUN_DIR}/logs/training.log"
}

write_ctc_config() {
  local bestrq_checkpoint="$1"
  local source_step="$2"
  CTC_RUN_DIR="${CTC_RUN_DIR_ROOT}/sensevoice_rwkv_ctc_curriculum_stage6c_from_bestrq_step${BESTRQ_STEPS}_srcstep${source_step}_bs${CTC_BATCH_SIZE}_lr5e5_zero1_nockpt_4x4090"
  CTC_CONFIG="${GENERATED_CONFIG_DIR}/sensevoice_rwkv_ctc_curriculum_stage6c_from_bestrq_step${BESTRQ_STEPS}_srcstep${source_step}_4x4090_deepspeed.yaml"
  export SOURCE_CONFIG CTC_RUN_DIR CTC_CONFIG CTC_STEPS CTC_BATCH_SIZE CTC_TOKEN_BUDGET CTC_LR bestrq_checkpoint
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from rwkvasr.config import load_yaml, save_yaml

cfg = dict(load_yaml(os.environ["SOURCE_CONFIG"]))
cfg.update(
    {
        "output_dir": os.environ["CTC_RUN_DIR"],
        "decoder_enabled": False,
        "ctc_loss_weight": 1.0,
        "decoder_loss_weight": 0.0,
        "init_checkpoint_path": os.environ["bestrq_checkpoint"],
        "resume_from": None,
        "wandb_run_name": Path(os.environ["CTC_RUN_DIR"]).name,
        "batch_size": int(os.environ["CTC_BATCH_SIZE"]),
        "batch_token_budget": int(os.environ["CTC_TOKEN_BUDGET"]),
        "length_bucket_frame_budget": int(os.environ["CTC_TOKEN_BUDGET"]),
        "target_gpu_memory_gib": 23.0,
        "max_steps": int(os.environ["CTC_STEPS"]),
        "epochs": None,
        "save_every": 500,
        "step_eval_every": 500,
        "top_k_step_checkpoints": 10,
        "gradient_checkpointing": False,
        "lr": float(os.environ["CTC_LR"]),
    }
)
ds = dict(cfg.get("deepspeed", {}))
ds["train_micro_batch_size_per_gpu"] = int(os.environ["CTC_BATCH_SIZE"])
ds["gradient_accumulation_steps"] = int(ds.get("gradient_accumulation_steps", 4))
zero = dict(ds.get("zero_optimization", {}))
zero["stage"] = 1
zero["offload_optimizer"] = {"device": "none"}
ds["zero_optimization"] = zero
ds["bf16"] = {"enabled": True}
cfg["deepspeed"] = ds
save_yaml(os.environ["CTC_CONFIG"], cfg)
Path(os.environ["CTC_RUN_DIR"]).mkdir(parents=True, exist_ok=True)
print(os.environ["CTC_CONFIG"])
PY
}

start_ctc_training() {
  local config="$1"
  mkdir -p "${CTC_RUN_DIR}/logs"
  if ! tmux_has_session "${TRAINING_SESSION}"; then
    tmux new-session -d -s "${TRAINING_SESSION}" -c "${REPO_ROOT}"
  fi
  local command
  command="export PYTHONUNBUFFERED=1; export RICH_FORCE_TERMINAL=1; export TQDM_DISABLE=0; export NUM_GPUS=${NUM_GPUS}; export MASTER_PORT=${MASTER_PORT}; export PATH=\"${REPO_ROOT}/.venv/bin:\$PATH\"; cd \"${REPO_ROOT}\"; script -q -f -e -c \"./scripts/train_paper_rwkv_asr.sh --config-yaml ${config} --num-gpus ${NUM_GPUS} --master-port ${MASTER_PORT}\" \"${CTC_RUN_DIR}/logs/training.ansi.log\"; rc=\$?; echo; echo \"BestRQ-initialized CTC training exited with status \$rc at \$(date)\""
  log "starting CTC training in tmux session ${TRAINING_SESSION}"
  tmux send-keys -t "${TRAINING_SESSION}:0" "${command}" C-m
}

wait_for_ctc_start() {
  log "waiting for CTC training process to become visible"
  for _ in {1..90}; do
    if ctc_training_active; then
      log "CTC training process is active"
      return 0
    fi
    sleep 2
  done
  log "CTC training process did not become visible before timeout"
  return 1
}

ctc_training_active() {
  local base
  base="$(basename "${CTC_CONFIG}")"
  pgrep -af -- "rwkvasr.cli.train_ctc_deepspeed.*${base}" >/dev/null 2>&1 \
    || pgrep -af -- "torchrun.*${base}" >/dev/null 2>&1
}

latest_ctc_checkpoint() {
  find "${CTC_RUN_DIR}" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^step-\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

wait_for_ctc_complete() {
  log "waiting for CTC training to finish"
  while ctc_training_active; do
    log "CTC training running latest_step=$(latest_ctc_checkpoint || true)"
    sleep 300
  done
  local latest
  latest="$(latest_ctc_checkpoint || true)"
  if [[ -z "${latest}" ]]; then
    log "CTC training stopped without step checkpoints"
    return 1
  fi
  CTC_FINAL_CHECKPOINT="${CTC_RUN_DIR}/step-${latest}.pt"
  log "CTC training stopped latest=${latest} checkpoint=${CTC_FINAL_CHECKPOINT}"
}

run_ctc_draft_part() {
  local checkpoint="$1"
  local label="$2"
  local shard_pattern="$3"
  local limit="$4"
  local output_path="${CTC_RUN_DIR}/ctc_draft_cache/parts/${label}.jsonl"
  local preview_path="${CTC_RUN_DIR}/ctc_draft_cache/parts/${label}.preview.txt"
  mkdir -p "${CTC_RUN_DIR}/ctc_draft_cache/parts"
  if [[ -s "${output_path}" ]]; then
    log "skipping existing draft part ${output_path}"
    return 0
  fi
  log "draft part label=${label} pattern=${shard_pattern} limit=${limit}"
  "${PYTHON_BIN}" -m rwkvasr.cli.predict_ctc_labeled \
    --checkpoint-path "${checkpoint}" \
    --config-yaml "${CTC_RUN_DIR}/model_config.yaml" \
    --webdataset-root "$("${PYTHON_BIN}" - <<PY
from rwkvasr.config import load_yaml
print(load_yaml("${CTC_CONFIG}")["webdataset_root"])
PY
)" \
    --webdataset-length-index-path "$("${PYTHON_BIN}" - <<PY
from rwkvasr.config import load_yaml
print(load_yaml("${CTC_CONFIG}")["webdataset_length_index_path"])
PY
)" \
    --webdataset-split train \
    --webdataset-shard-pattern "${shard_pattern}" \
    --webdataset-eval-ratio "$("${PYTHON_BIN}" - <<PY
from rwkvasr.config import load_yaml
print(load_yaml("${CTC_CONFIG}").get("webdataset_eval_ratio", 0.0))
PY
)" \
    --webdataset-hash-seed "$("${PYTHON_BIN}" - <<PY
from rwkvasr.config import load_yaml
print(load_yaml("${CTC_CONFIG}").get("webdataset_hash_seed", 0))
PY
)" \
    --webdataset-split-by "$("${PYTHON_BIN}" - <<PY
from rwkvasr.config import load_yaml
print(load_yaml("${CTC_CONFIG}").get("webdataset_split_by", "sample_id"))
PY
)" \
    --webdataset-utt-id-key "$("${PYTHON_BIN}" - <<PY
from rwkvasr.config import load_yaml
print(load_yaml("${CTC_CONFIG}").get("webdataset_utt_id_key", "sid"))
PY
)" \
    --device "${DRAFT_DEVICE}" \
    --batch-size "${DRAFT_BATCH_SIZE}" \
    --num-workers "${DRAFT_NUM_WORKERS}" \
    --mode bi \
    --beam-size 4 \
    --token-prune-topk 16 \
    --text-normalization ctc \
    --limit "${limit}" \
    --progress-interval 200 \
    --output-path "${output_path}" \
    --preview-path "${preview_path}" \
    --preview-count 20
}

build_offline_draft_cache() {
  local checkpoint="$1"
  local merged="${CTC_RUN_DIR}/ctc_draft_cache/ctc_draft_train.jsonl"
  local filtered_lengths="${CTC_RUN_DIR}/ctc_draft_cache/webdataset_lengths.cached.jsonl"
  if [[ -s "${merged}" && -s "${filtered_lengths}" ]]; then
    log "offline draft already exists cache=${merged} filtered_lengths=${filtered_lengths}"
    return 0
  fi
  IFS=',' read -r -a quotas <<< "${DRAFT_SOURCE_QUOTAS}"
  local part_paths=()
  for quota in "${quotas[@]}"; do
    [[ -n "${quota}" ]] || continue
    IFS=':' read -r label pattern limit <<< "${quota}"
    run_ctc_draft_part "${checkpoint}" "${label}" "${pattern}" "${limit}"
    part_paths+=("${CTC_RUN_DIR}/ctc_draft_cache/parts/${label}.jsonl")
  done
  log "merging draft cache ${merged}"
  "${PYTHON_BIN}" - "${merged}" "${part_paths[@]}" <<'PY'
import json
import sys
from pathlib import Path

from rwkvasr.data.text_normalization import normalize_asr_text

out = Path(sys.argv[1])
seen = set()
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as dst:
    for part in map(Path, sys.argv[2:]):
        with part.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                raw = json.loads(line)
                utt_id = raw.get("utt_id") or raw.get("id") or raw.get("audio_id") or raw.get("sid") or raw.get("key")
                if utt_id is None:
                    continue
                utt_id = str(utt_id)
                if utt_id in seen:
                    continue
                seen.add(utt_id)
                language = raw.get("language")
                pred = normalize_asr_text(str(raw.get("pred_text") or ""), language=language, mode="ctc")
                raw["utt_id"] = utt_id
                raw["pred_text"] = pred
                raw["ctc_draft"] = pred
                dst.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")
print(f"merged={out} records={len(seen)}")
PY
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/filter_length_index_by_ctc_draft.py" \
    --length-index-path "$("${PYTHON_BIN}" - <<PY
from rwkvasr.config import load_yaml
print(load_yaml("${CTC_CONFIG}")["webdataset_length_index_path"])
PY
)" \
    --ctc-draft-jsonl "${merged}" \
    --output-path "${filtered_lengths}" \
    --require-all
  log "offline draft ready cache=${merged} filtered_lengths=${filtered_lengths}"
}

run_draft_only() {
  if [[ -z "${CTC_RUN_DIR:-}" ]]; then
    log "DRAFT_ONLY=1 requires CTC_RUN_DIR"
    exit 1
  fi
  if [[ -z "${CTC_CONFIG:-}" ]]; then
    log "DRAFT_ONLY=1 requires CTC_CONFIG"
    exit 1
  fi
  if [[ -z "${DRAFT_CHECKPOINT}" ]]; then
    local latest
    latest="$(latest_ctc_checkpoint || true)"
    if [[ -z "${latest}" ]]; then
      log "no step checkpoints found in ${CTC_RUN_DIR}"
      exit 1
    fi
    DRAFT_CHECKPOINT="${CTC_RUN_DIR}/step-${latest}.pt"
  fi
  if [[ ! -s "${DRAFT_CHECKPOINT}" ]]; then
    log "draft checkpoint missing: ${DRAFT_CHECKPOINT}"
    exit 1
  fi
  log "draft-only mode checkpoint=${DRAFT_CHECKPOINT}"
  build_offline_draft_cache "${DRAFT_CHECKPOINT}"
}

main() {
  cd "${REPO_ROOT}"
  if [[ "${DRAFT_ONLY}" == "1" ]]; then
    run_draft_only
    return 0
  fi
  local source_checkpoint source_step bestrq_checkpoint
  source_checkpoint="$(wait_for_source_checkpoint)"
  source_step="${SOURCE_TARGET_STEP}"
  stop_current_training
  write_bestrq_config "${source_checkpoint}" "${source_step}"
  run_bestrq "${BESTRQ_CONFIG}"
  bestrq_checkpoint="${BESTRQ_RUN_DIR}/final.pt"
  if [[ ! -s "${bestrq_checkpoint}" ]]; then
    log "Best-RQ final checkpoint missing: ${bestrq_checkpoint}"
    exit 1
  fi
  write_ctc_config "${bestrq_checkpoint}" "${source_step}"
  start_ctc_training "${CTC_CONFIG}"
  wait_for_ctc_start
  wait_for_ctc_complete
  build_offline_draft_cache "${CTC_FINAL_CHECKPOINT}"
}

main "$@"
