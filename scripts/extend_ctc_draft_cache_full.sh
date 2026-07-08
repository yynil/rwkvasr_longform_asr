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

CTC_RUN_DIR="${CTC_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_ctc_curriculum_stage6c_from_bestrq_step5000_srcstep3000_bs12_lr5e5_zero1_nockpt_4x4090}"
CTC_CONFIG="${CTC_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_ctc_curriculum_stage6c_from_bestrq_step5000_srcstep3000_4x4090_deepspeed.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DRAFT_CHECKPOINT:-${CTC_RUN_DIR}/step-5000.pt}}"

BASE_DRAFT_CACHE_DIR="${BASE_DRAFT_CACHE_DIR:-${CTC_RUN_DIR}/ctc_draft_cache}"
BASE_DRAFT_CACHE_PATH="${BASE_DRAFT_CACHE_PATH:-${BASE_DRAFT_CACHE_DIR}/ctc_draft_train.jsonl}"
FULL_DRAFT_CACHE_DIR="${FULL_DRAFT_CACHE_DIR:-${CTC_RUN_DIR}/ctc_draft_cache_full}"
FULL_DRAFT_CACHE_PATH="${FULL_DRAFT_CACHE_PATH:-${FULL_DRAFT_CACHE_DIR}/ctc_draft_train.jsonl}"
FULL_LENGTH_INDEX_PATH="${FULL_LENGTH_INDEX_PATH:-${FULL_DRAFT_CACHE_DIR}/webdataset_lengths.cached.jsonl}"
MISSING_LENGTH_INDEX_PATH="${MISSING_LENGTH_INDEX_PATH:-${FULL_DRAFT_CACHE_DIR}/webdataset_lengths.missing_from_base.jsonl}"
MISSING_COUNTS_PATH="${MISSING_COUNTS_PATH:-${FULL_DRAFT_CACHE_DIR}/missing_counts.tsv}"
MISSING_PART_DIR="${MISSING_PART_DIR:-${FULL_DRAFT_CACHE_DIR}/missing_parts}"

DRAFT_SOURCE_SPECS="${DRAFT_SOURCE_SPECS:-clean_librispeech:librispeech_*.tar,clean_aishell:aishell3_*.tar,clean_cv_en:commonvoice_en_*.tar,clean_cv_cn:commonvoice_cn_*.tar,gigaspeech:GSXL-*.tar,wenetspeech:WSL-*.tar}"
DRAFT_DEVICES="${DRAFT_DEVICES:-0,1,2,3}"
DRAFT_MAX_PARALLEL="${DRAFT_MAX_PARALLEL:-4}"
DRAFT_ISOLATE_GPUS="${DRAFT_ISOLATE_GPUS:-1}"
DRAFT_BATCH_SIZE="${DRAFT_BATCH_SIZE:-4}"
DRAFT_NUM_WORKERS="${DRAFT_NUM_WORKERS:-4}"
DRAFT_LIMIT="${DRAFT_LIMIT:-999999999}"
DRAFT_PROGRESS_INTERVAL="${DRAFT_PROGRESS_INTERVAL:-200}"
DRAFT_BEAM_SIZE="${DRAFT_BEAM_SIZE:-4}"
DRAFT_TOKEN_PRUNE_TOPK="${DRAFT_TOKEN_PRUNE_TOPK:-16}"

log() {
  printf '[ctc-draft-full] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

yaml_value() {
  local key="$1"
  "${PYTHON_BIN}" - "${CTC_CONFIG}" "${key}" <<'PY'
import sys
from rwkvasr.config import load_yaml

cfg = load_yaml(sys.argv[1])
value = cfg.get(sys.argv[2])
print("" if value is None else value)
PY
}

latest_step_checkpoint() {
  find "${CTC_RUN_DIR}" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^step-\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

resolve_checkpoint() {
  if [[ -s "${CHECKPOINT_PATH}" ]]; then
    return 0
  fi
  local latest
  latest="$(latest_step_checkpoint || true)"
  if [[ -z "${latest}" ]]; then
    echo "checkpoint not found and no step checkpoints in ${CTC_RUN_DIR}" >&2
    exit 1
  fi
  CHECKPOINT_PATH="${CTC_RUN_DIR}/step-${latest}.pt"
  if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
    echo "checkpoint missing: ${CHECKPOINT_PATH}" >&2
    exit 1
  fi
}

prepare_missing_index() {
  mkdir -p "${FULL_DRAFT_CACHE_DIR}" "${MISSING_PART_DIR}" "${FULL_DRAFT_CACHE_DIR}/logs"
  log "building missing length index from base cache ${BASE_DRAFT_CACHE_PATH}"
  "${PYTHON_BIN}" - \
    "${BASE_DRAFT_CACHE_PATH}" \
    "$(yaml_value webdataset_length_index_path)" \
    "${MISSING_LENGTH_INDEX_PATH}" \
    "${MISSING_COUNTS_PATH}" \
    "${DRAFT_SOURCE_SPECS}" <<'PY'
import fnmatch
import json
import sys
from collections import Counter
from pathlib import Path


def utt_id(raw: dict) -> str | None:
    for key in ("utt_id", "id", "audio_id", "sid", "key", "sample_id"):
        value = raw.get(key)
        if value is not None:
            return str(value)
    return None


base_cache = Path(sys.argv[1])
length_index = Path(sys.argv[2])
missing_index = Path(sys.argv[3])
counts_path = Path(sys.argv[4])
specs: list[tuple[str, str]] = []
for item in sys.argv[5].split(","):
    if not item:
        continue
    label, pattern = item.split(":", 1)
    specs.append((label, pattern))

if not base_cache.is_file():
    raise SystemExit(f"base draft cache missing: {base_cache}")
if not length_index.is_file():
    raise SystemExit(f"length index missing: {length_index}")

base_ids: set[str] = set()
for line in base_cache.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    sample_id = utt_id(json.loads(line))
    if sample_id:
        base_ids.add(sample_id)

counts: Counter[str] = Counter()
total_rows = 0
missing_rows = 0
missing_index.parent.mkdir(parents=True, exist_ok=True)
with length_index.open("r", encoding="utf-8") as src, missing_index.open("w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        total_rows += 1
        raw = json.loads(line)
        sample_id = utt_id(raw)
        if sample_id is not None and sample_id in base_ids:
            continue
        missing_rows += 1
        shard = str(raw.get("shard_name") or raw.get("shard") or raw.get("tar_path") or raw.get("url") or "")
        shard_base = shard.rsplit("/", 1)[-1]
        label = "other"
        for candidate, pattern in specs:
            if fnmatch.fnmatch(shard_base, pattern) or fnmatch.fnmatch(shard, pattern):
                label = candidate
                break
        counts[label] += 1
        dst.write(line)

with counts_path.open("w", encoding="utf-8") as handle:
    for label, _ in specs:
        handle.write(f"{label}\t{counts[label]}\n")
    if counts["other"]:
        handle.write(f"other\t{counts['other']}\n")

print(
    "missing_index",
    f"total_rows={total_rows}",
    f"base_ids={len(base_ids)}",
    f"missing_rows={missing_rows}",
    f"output={missing_index}",
)
for label, _ in specs:
    print(f"missing_count {label} {counts[label]}")
if counts["other"]:
    print(f"missing_count other {counts['other']}")
PY
}

missing_count_for_label() {
  local label="$1"
  awk -F '\t' -v label="${label}" '$1 == label { print $2 }' "${MISSING_COUNTS_PATH}"
}

run_missing_part() {
  local label="$1"
  local shard_pattern="$2"
  local device_id="$3"
  local rows="$4"
  local output_path="${MISSING_PART_DIR}/${label}.jsonl"
  local preview_path="${MISSING_PART_DIR}/${label}.preview.txt"
  local log_path="${FULL_DRAFT_CACHE_DIR}/logs/${label}.log"
  local done_path="${output_path}.done"

  if [[ "${rows}" == "0" ]]; then
    log "skip ${label}; no missing rows"
    return 0
  fi
  if [[ -f "${done_path}" ]]; then
    log "skip ${label}; done marker exists ${done_path}"
    return 0
  fi
  if [[ -s "${output_path}" ]]; then
    log "skip ${label}; existing part ${output_path}"
    return 0
  fi

  local device_arg="cuda:${device_id}"
  local env_prefix=()
  if [[ "${DRAFT_ISOLATE_GPUS}" == "1" ]]; then
    device_arg="cuda:0"
    env_prefix=(env "CUDA_VISIBLE_DEVICES=${device_id}")
  fi

  log "start missing part label=${label} rows=${rows} pattern=${shard_pattern} physical_gpu=${device_id} device=${device_arg} isolate=${DRAFT_ISOLATE_GPUS}"
  "${env_prefix[@]}" "${PYTHON_BIN}" -m rwkvasr.cli.predict_ctc_labeled \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --config-yaml "${CTC_RUN_DIR}/model_config.yaml" \
    --webdataset-root "$(yaml_value webdataset_root)" \
    --webdataset-length-index-path "${MISSING_LENGTH_INDEX_PATH}" \
    --webdataset-split train \
    --webdataset-shard-pattern "${shard_pattern}" \
    --webdataset-eval-ratio "$(yaml_value webdataset_eval_ratio)" \
    --webdataset-hash-seed "$(yaml_value webdataset_hash_seed)" \
    --webdataset-split-by "$(yaml_value webdataset_split_by)" \
    --webdataset-utt-id-key "$(yaml_value webdataset_utt_id_key)" \
    --device "${device_arg}" \
    --batch-size "${DRAFT_BATCH_SIZE}" \
    --num-workers "${DRAFT_NUM_WORKERS}" \
    --mode bi \
    --beam-size "${DRAFT_BEAM_SIZE}" \
    --token-prune-topk "${DRAFT_TOKEN_PRUNE_TOPK}" \
    --text-normalization ctc \
    --limit "${DRAFT_LIMIT}" \
    --progress-interval "${DRAFT_PROGRESS_INTERVAL}" \
    --output-path "${output_path}" \
    --preview-path "${preview_path}" \
    --preview-count 20 \
    >"${log_path}" 2>&1
  touch "${done_path}"
  log "done missing part label=${label} output=${output_path}"
}

generate_missing_parts() {
  IFS=',' read -r -a devices <<< "${DRAFT_DEVICES}"
  if [[ "${#devices[@]}" -eq 0 ]]; then
    echo "DRAFT_DEVICES must not be empty" >&2
    exit 1
  fi

  IFS=',' read -r -a specs <<< "${DRAFT_SOURCE_SPECS}"
  local launched=0
  for spec in "${specs[@]}"; do
    [[ -n "${spec}" ]] || continue
    IFS=':' read -r label pattern <<< "${spec}"
    local rows
    rows="$(missing_count_for_label "${label}")"
    rows="${rows:-0}"
    if [[ "${rows}" == "0" ]]; then
      log "skip ${label}; no missing rows"
      continue
    fi
    local output_path="${MISSING_PART_DIR}/${label}.jsonl"
    local done_path="${output_path}.done"
    if [[ -f "${done_path}" || -s "${output_path}" ]]; then
      log "skip ${label}; existing completed part ${output_path}"
      continue
    fi
    while [[ "$(jobs -rp | wc -l)" -ge "${DRAFT_MAX_PARALLEL}" ]]; do
      sleep 10
    done
    local device_id="${devices[$((launched % ${#devices[@]}))]}"
    run_missing_part "${label}" "${pattern}" "${device_id}" "${rows}" &
    launched=$((launched + 1))
  done

  local failed=0
  for job in $(jobs -rp); do
    if ! wait "${job}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "one or more missing draft parts failed; see ${FULL_DRAFT_CACHE_DIR}/logs" >&2
    exit 1
  fi
}

merge_full_cache() {
  log "merging base cache and missing parts into ${FULL_DRAFT_CACHE_PATH}"
  shopt -s nullglob
  local missing_parts=("${MISSING_PART_DIR}"/*.jsonl)
  "${PYTHON_BIN}" - "${FULL_DRAFT_CACHE_PATH}" "${BASE_DRAFT_CACHE_PATH}" "${missing_parts[@]}" <<'PY'
import json
import sys
from pathlib import Path

from rwkvasr.data.text_normalization import normalize_asr_text


def utt_id(raw: dict) -> str | None:
    for key in ("utt_id", "id", "audio_id", "sid", "key", "sample_id"):
        value = raw.get(key)
        if value is not None:
            return str(value)
    return None


out = Path(sys.argv[1])
inputs = [Path(value) for value in sys.argv[2:]]
seen: set[str] = set()
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as dst:
    for path in inputs:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                raw = json.loads(line)
                sample_id = utt_id(raw)
                if sample_id is None or sample_id in seen:
                    continue
                seen.add(sample_id)
                language = raw.get("language")
                pred = normalize_asr_text(
                    str(raw.get("ctc_draft") or raw.get("pred_text") or ""),
                    language=language,
                    mode="ctc",
                )
                raw["utt_id"] = sample_id
                raw["pred_text"] = pred
                raw["ctc_draft"] = pred
                dst.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")
print(f"merged_full_cache output={out} records={len(seen)}")
PY
}

filter_full_length_index() {
  log "filtering full length index to ${FULL_LENGTH_INDEX_PATH}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/filter_length_index_by_ctc_draft.py" \
    --length-index-path "$(yaml_value webdataset_length_index_path)" \
    --ctc-draft-jsonl "${FULL_DRAFT_CACHE_PATH}" \
    --output-path "${FULL_LENGTH_INDEX_PATH}" \
    --require-all
}

main() {
  cd "${REPO_ROOT}"
  resolve_checkpoint
  log "checkpoint=${CHECKPOINT_PATH}"
  log "base_cache=${BASE_DRAFT_CACHE_PATH}"
  log "full_cache_dir=${FULL_DRAFT_CACHE_DIR}"
  prepare_missing_index
  generate_missing_parts
  merge_full_cache
  filter_full_length_index
  log "done full_cache=${FULL_DRAFT_CACHE_PATH} full_lengths=${FULL_LENGTH_INDEX_PATH}"
  wc -l "${BASE_DRAFT_CACHE_PATH}" "${FULL_DRAFT_CACHE_PATH}" "${FULL_LENGTH_INDEX_PATH}" || true
}

main "$@"
