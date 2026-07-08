#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export RICH_FORCE_TERMINAL="${RICH_FORCE_TERMINAL:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage6c_clean50_hard50_joint_chatlang_hardmix_4x4090_deepspeed.yaml}"
RUN_DIR="${RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage6c_clean50_hard50_joint_chatlang_gigaspeech_wenetspeech_clean50_hard50_from_prev_ctc0p8_ar0p2_bs4_lr2e5_zero1_nockpt_4x4090}"
TARGET_STEP="${TARGET_STEP:-5000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
CHECKPOINT_STABLE_SECONDS="${CHECKPOINT_STABLE_SECONDS:-90}"

TRAINING_SESSION="${TRAINING_SESSION:-training}"
OLD_SIDECAR_SESSION="${OLD_SIDECAR_SESSION:-sidecar_stage6}"
OLD_AUTO_SESSION="${OLD_AUTO_SESSION:-stage6_auto}"
NEW_SIDECAR_SESSION="${NEW_SIDECAR_SESSION:-sidecar_stage6_draft_ar}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29567}"

DRAFT_DIR="${DRAFT_DIR:-${RUN_DIR}/ctc_draft_step${TARGET_STEP}}"
DRAFT_CACHE_PATH="${DRAFT_CACHE_PATH:-${DRAFT_DIR}/ctc_draft_train.jsonl}"
DRAFT_LENGTH_INDEX_PATH="${DRAFT_LENGTH_INDEX_PATH:-${DRAFT_DIR}/webdataset_lengths.cached.jsonl}"
DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:0}"
DRAFT_BATCH_SIZE="${DRAFT_BATCH_SIZE:-4}"
DRAFT_NUM_WORKERS="${DRAFT_NUM_WORKERS:-4}"
DRAFT_BEAM_SIZE="${DRAFT_BEAM_SIZE:-4}"
DRAFT_TOKEN_PRUNE_TOPK="${DRAFT_TOKEN_PRUNE_TOPK:-16}"
DRAFT_PROGRESS_INTERVAL="${DRAFT_PROGRESS_INTERVAL:-200}"
DRAFT_SOURCE_QUOTAS="${DRAFT_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:10000,clean_aishell:aishell3_*.tar:10000,clean_cv_en:commonvoice_en_*.tar:10000,clean_cv_cn:commonvoice_cn_*.tar:10000,gigaspeech:GSXL-*.tar:12000,wenetspeech:WSL-*.tar:12000}"

AR_RUN_DIR="${AR_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage6d_ctc_draft_ar_from_stage6c_step${TARGET_STEP}_ctc0p4_ar0p6_bs4_lr2e5_zero1_nockpt_4x4090}"
AR_CONFIG="${AR_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage6d_ctc_draft_ar_from_stage6c_step${TARGET_STEP}_4x4090_deepspeed.yaml}"
AR_MAX_STEPS="${AR_MAX_STEPS:-3000}"
AR_CTC_LOSS_WEIGHT="${AR_CTC_LOSS_WEIGHT:-0.4}"
AR_DECODER_LOSS_WEIGHT="${AR_DECODER_LOSS_WEIGHT:-0.6}"
AR_LR="${AR_LR:-2.0e-5}"
if [[ -z "${AR_DRAFT_PROMPT_TEMPLATE:-}" ]]; then
  AR_DRAFT_PROMPT_TEMPLATE="$(cat <<'EOF'
A first-pass CTC transcript is:
{ctc_draft}
Use this draft as a speech-grounded hint. Correct it only when the audio supports the correction, and do not add content that is not spoken.

EOF
)"
fi

SIDECAR_LIMIT="${SIDECAR_LIMIT:-64}"
SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:8,clean_aishell:aishell3_*.tar:8,clean_cv_en:commonvoice_en_*.tar:8,clean_cv_cn:commonvoice_cn_*.tar:8,gigaspeech:GSXL-*.tar:16,wenetspeech:WSL-*.tar:16}"
SIDECAR_DEVICE="${SIDECAR_DEVICE:-cuda:0}"
SIDECAR_BATCH_SIZE="${SIDECAR_BATCH_SIZE:-1}"
SIDECAR_POLL_SECONDS="${SIDECAR_POLL_SECONDS:-900}"
SIDECAR_AR_TEMPERATURE="${SIDECAR_AR_TEMPERATURE:-0.6}"
SIDECAR_AR_TOP_K="${SIDECAR_AR_TOP_K:-3}"
SIDECAR_AR_TOP_P="${SIDECAR_AR_TOP_P:-0.6}"

CHECKPOINT_PATH="${RUN_DIR}/step-${TARGET_STEP}.pt"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

send_ctrl_c() {
  local session="$1"
  if tmux_has_session "${session}"; then
    log "Sending Ctrl-C to tmux session ${session}"
    tmux send-keys -t "${session}:0" C-c || true
  fi
}

checkpoint_is_stable() {
  local path="$1"
  local stable_seconds="$2"
  [[ -s "${path}" ]] || return 1
  local size_a size_b
  size_a="$(stat -c '%s' "${path}")"
  sleep "${stable_seconds}"
  [[ -s "${path}" ]] || return 1
  size_b="$(stat -c '%s' "${path}")"
  [[ "${size_a}" == "${size_b}" ]]
}

wait_for_checkpoint() {
  log "Waiting for checkpoint ${CHECKPOINT_PATH}"
  while true; do
    if checkpoint_is_stable "${CHECKPOINT_PATH}" "${CHECKPOINT_STABLE_SECONDS}"; then
      log "Checkpoint is stable: ${CHECKPOINT_PATH}"
      return 0
    fi
    local latest
    latest="$(find "${RUN_DIR}" -maxdepth 1 -name 'step-*.pt' -printf '%f\n' 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "${latest}" ]]; then
      log "Latest checkpoint seen: ${latest}; still waiting for step-${TARGET_STEP}.pt"
    else
      log "No step checkpoints seen yet"
    fi
    sleep "${POLL_SECONDS}"
  done
}

wait_for_training_stop() {
  local pattern="${BASE_CONFIG}"
  for _ in {1..60}; do
    if ! pgrep -af "rwkvasr.cli.train_ctc_deepspeed.*${pattern}" >/dev/null \
      && ! pgrep -af "torchrun.*${pattern}" >/dev/null; then
      log "Training process for ${BASE_CONFIG} has stopped"
      return 0
    fi
    sleep 5
  done
  log "Training did not stop after Ctrl-C; sending TERM to matching torchrun/train processes"
  pkill -TERM -f "rwkvasr.cli.train_ctc_deepspeed.*${pattern}" || true
  pkill -TERM -f "torchrun.*${pattern}" || true
}

stop_current_stage() {
  send_ctrl_c "${OLD_AUTO_SESSION}"
  send_ctrl_c "${OLD_SIDECAR_SESSION}"
  send_ctrl_c "${TRAINING_SESSION}"
  wait_for_training_stop
}

run_ctc_quota() {
  local label="$1"
  local shard_pattern="$2"
  local limit="$3"
  local output_path="${DRAFT_DIR}/parts/${label}.jsonl"
  local preview_path="${DRAFT_DIR}/parts/${label}.preview.txt"
  mkdir -p "${DRAFT_DIR}/parts"
  if [[ -s "${output_path}" ]]; then
    log "Skipping existing CTC draft part ${output_path}"
    return 0
  fi
  log "Generating CTC draft part label=${label} pattern=${shard_pattern} limit=${limit}"
  "${PYTHON_BIN}" -m rwkvasr.cli.predict_ctc_labeled \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --config-yaml "${RUN_DIR}/model_config.yaml" \
    --webdataset-root "$("${PYTHON_BIN}" - <<'PY'
from rwkvasr.config import load_yaml
import os
print(load_yaml(os.environ["BASE_CONFIG"])["webdataset_root"])
PY
)" \
    --webdataset-length-index-path "$("${PYTHON_BIN}" - <<'PY'
from rwkvasr.config import load_yaml
import os
print(load_yaml(os.environ["BASE_CONFIG"])["webdataset_length_index_path"])
PY
)" \
    --webdataset-split train \
    --webdataset-shard-pattern "${shard_pattern}" \
    --webdataset-eval-ratio "$("${PYTHON_BIN}" - <<'PY'
from rwkvasr.config import load_yaml
import os
print(load_yaml(os.environ["BASE_CONFIG"]).get("webdataset_eval_ratio", 0.0))
PY
)" \
    --webdataset-hash-seed "$("${PYTHON_BIN}" - <<'PY'
from rwkvasr.config import load_yaml
import os
print(load_yaml(os.environ["BASE_CONFIG"]).get("webdataset_hash_seed", 0))
PY
)" \
    --webdataset-split-by "$("${PYTHON_BIN}" - <<'PY'
from rwkvasr.config import load_yaml
import os
print(load_yaml(os.environ["BASE_CONFIG"]).get("webdataset_split_by", "shard_name"))
PY
)" \
    --webdataset-utt-id-key "$("${PYTHON_BIN}" - <<'PY'
from rwkvasr.config import load_yaml
import os
print(load_yaml(os.environ["BASE_CONFIG"]).get("webdataset_utt_id_key", "sid"))
PY
)" \
    --device "${DRAFT_DEVICE}" \
    --batch-size "${DRAFT_BATCH_SIZE}" \
    --num-workers "${DRAFT_NUM_WORKERS}" \
    --mode bi \
    --beam-size "${DRAFT_BEAM_SIZE}" \
    --token-prune-topk "${DRAFT_TOKEN_PRUNE_TOPK}" \
    --text-normalization ctc \
    --limit "${limit}" \
    --progress-interval "${DRAFT_PROGRESS_INTERVAL}" \
    --output-path "${output_path}" \
    --preview-path "${preview_path}" \
    --preview-count 20
}

build_ctc_draft_cache() {
  mkdir -p "${DRAFT_DIR}"
  if [[ -s "${DRAFT_CACHE_PATH}" ]]; then
    log "Skipping existing merged CTC draft cache ${DRAFT_CACHE_PATH}"
    return 0
  fi
  IFS=',' read -r -a quotas <<< "${DRAFT_SOURCE_QUOTAS}"
  local part_paths=()
  for quota in "${quotas[@]}"; do
    [[ -n "${quota}" ]] || continue
    IFS=':' read -r label shard_pattern limit <<< "${quota}"
    if [[ -z "${label:-}" || -z "${shard_pattern:-}" || -z "${limit:-}" ]]; then
      echo "Invalid DRAFT_SOURCE_QUOTAS item: ${quota}" >&2
      exit 1
    fi
    run_ctc_quota "${label}" "${shard_pattern}" "${limit}"
    part_paths+=("${DRAFT_DIR}/parts/${label}.jsonl")
  done
  log "Merging CTC draft parts into ${DRAFT_CACHE_PATH}"
  "${PYTHON_BIN}" - "${DRAFT_CACHE_PATH}" "${part_paths[@]}" <<'PY'
import json
import sys
from pathlib import Path

from rwkvasr.data.text_normalization import normalize_asr_text

out_path = Path(sys.argv[1])
part_paths = [Path(value) for value in sys.argv[2:]]
seen = set()
count = 0
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as dst:
    for part_path in part_paths:
        for line in part_path.read_text(encoding="utf-8").splitlines():
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
            pred_text = normalize_asr_text(str(raw.get("pred_text") or ""), language=language, mode="ctc")
            raw["utt_id"] = utt_id
            raw["pred_text"] = pred_text
            raw["ctc_draft"] = pred_text
            dst.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
print(f"merged_ctc_draft_cache output={out_path} count={count}")
PY
}

build_cached_length_index() {
  if [[ -s "${DRAFT_LENGTH_INDEX_PATH}" ]]; then
    log "Skipping existing filtered length index ${DRAFT_LENGTH_INDEX_PATH}"
    return 0
  fi
  log "Filtering length index to cached draft utterances"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/filter_length_index_by_ctc_draft.py" \
    --length-index-path "$("${PYTHON_BIN}" - <<'PY'
from rwkvasr.config import load_yaml
import os
print(load_yaml(os.environ["BASE_CONFIG"])["webdataset_length_index_path"])
PY
)" \
    --ctc-draft-jsonl "${DRAFT_CACHE_PATH}" \
    --output-path "${DRAFT_LENGTH_INDEX_PATH}" \
    --require-all
}

write_ar_config() {
  log "Writing draft-conditioned AR config ${AR_CONFIG}"
  export AR_RUN_DIR AR_CONFIG CHECKPOINT_PATH DRAFT_CACHE_PATH DRAFT_LENGTH_INDEX_PATH
  export AR_MAX_STEPS AR_CTC_LOSS_WEIGHT AR_DECODER_LOSS_WEIGHT AR_LR AR_DRAFT_PROMPT_TEMPLATE
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from rwkvasr.config import load_yaml, save_yaml

base_config = Path(os.environ["BASE_CONFIG"])
cfg = dict(load_yaml(base_config))
cfg["output_dir"] = os.environ["AR_RUN_DIR"]
cfg["init_checkpoint_path"] = os.environ["CHECKPOINT_PATH"]
cfg["resume_from"] = None
cfg["webdataset_length_index_path"] = os.environ["DRAFT_LENGTH_INDEX_PATH"]
cfg["webdataset_bucket_manifest_path"] = None
cfg["decoder_ctc_draft_cache_path"] = os.environ["DRAFT_CACHE_PATH"]
cfg["decoder_ctc_draft_prompt_template"] = os.environ["AR_DRAFT_PROMPT_TEMPLATE"]
cfg["decoder_ctc_draft_text_key"] = "pred_text"
cfg["decoder_ctc_draft_missing_policy"] = "error"
cfg["ctc_loss_weight"] = float(os.environ["AR_CTC_LOSS_WEIGHT"])
cfg["decoder_loss_weight"] = float(os.environ["AR_DECODER_LOSS_WEIGHT"])
cfg["lr"] = float(os.environ["AR_LR"])
cfg["max_steps"] = int(os.environ["AR_MAX_STEPS"])
cfg["epochs"] = None
cfg["wandb_run_name"] = Path(os.environ["AR_RUN_DIR"]).name
cfg.setdefault("save_every", 500)
cfg.setdefault("step_eval_every", 500)
save_yaml(os.environ["AR_CONFIG"], cfg)
print(f"saved_ar_config={os.environ['AR_CONFIG']}")
PY
}

start_training() {
  mkdir -p "${AR_RUN_DIR}/logs"
  local command
  command="export PYTHONUNBUFFERED=1; export RICH_FORCE_TERMINAL=1; export TQDM_DISABLE=0; export NUM_GPUS=${NUM_GPUS}; export MASTER_PORT=${MASTER_PORT}; export PATH=\"${REPO_ROOT}/.venv/bin:\$PATH\"; cd \"${REPO_ROOT}\"; script -q -f -e -c \"./scripts/train_paper_rwkv_asr.sh --config-yaml ${AR_CONFIG} --num-gpus ${NUM_GPUS} --master-port ${MASTER_PORT}\" \"${AR_RUN_DIR}/logs/training.ansi.log\"; rc=\$?; echo; echo \"ctc-draft AR training exited with status \$rc at \$(date)\""
  if ! tmux_has_session "${TRAINING_SESSION}"; then
    tmux new-session -d -s "${TRAINING_SESSION}" -n train
  fi
  log "Starting draft-conditioned AR training in tmux session ${TRAINING_SESSION}"
  tmux send-keys -t "${TRAINING_SESSION}:0" "${command}" C-m
}

start_sidecar() {
  if tmux_has_session "${NEW_SIDECAR_SESSION}"; then
    log "Restarting existing sidecar session ${NEW_SIDECAR_SESSION}"
    tmux kill-session -t "${NEW_SIDECAR_SESSION}" || true
  fi
  local sidecar_command
  sidecar_command="cd \"${REPO_ROOT}\"; RUN_DIR=\"${AR_RUN_DIR}\" TMUX_TARGET=\"${TRAINING_SESSION}:0\" POLL_SECONDS=\"${SIDECAR_POLL_SECONDS}\" CHECKPOINT_STABLE_SECONDS=90 DEVICE=\"${SIDECAR_DEVICE}\" BATCH_SIZE=\"${SIDECAR_BATCH_SIZE}\" LIMIT=\"${SIDECAR_LIMIT}\" PREVIEW_COUNT=\"${SIDECAR_LIMIT}\" BEAM_SIZE=4 TOKEN_PRUNE_TOPK=16 TEXT_NORMALIZATION=runtime AR_DO_SAMPLE=1 AR_TEMPERATURE=\"${SIDECAR_AR_TEMPERATURE}\" AR_TOP_K=\"${SIDECAR_AR_TOP_K}\" AR_TOP_P=\"${SIDECAR_AR_TOP_P}\" SOURCE_QUOTAS=\"${SIDECAR_SOURCE_QUOTAS}\" ./scripts/watch_joint_training_sidecar.sh"
  log "Starting sidecar in tmux session ${NEW_SIDECAR_SESSION}"
  tmux new-session -d -s "${NEW_SIDECAR_SESSION}" -n sidecar "${sidecar_command}"
}

main() {
  cd "${REPO_ROOT}"
  export BASE_CONFIG
  wait_for_checkpoint
  stop_current_stage
  build_ctc_draft_cache
  build_cached_length_index
  write_ar_config
  start_training
  start_sidecar
  log "Monitor handoff complete. Training session=${TRAINING_SESSION}, sidecar session=${NEW_SIDECAR_SESSION}"
}

main "$@"
