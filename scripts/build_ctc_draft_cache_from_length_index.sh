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

CTC_RUN_DIR="${CTC_RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage7_large_ctc_clean16_hard84_from_stage6c_step5000_bs12_lr3e5_zero1_nockpt_4x4090}"
CTC_CONFIG="${CTC_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage7_large_ctc_clean16_hard84_from_stage6c_step5000_4x4090_deepspeed.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

DRAFT_CACHE_DIR="${DRAFT_CACHE_DIR:-${CTC_RUN_DIR}/ctc_draft_cache_stage7_full}"
DRAFT_CACHE_PATH="${DRAFT_CACHE_PATH:-${DRAFT_CACHE_DIR}/ctc_draft_train.jsonl}"
DRAFT_LENGTH_INDEX_PATH="${DRAFT_LENGTH_INDEX_PATH:-${DRAFT_CACHE_DIR}/webdataset_lengths.cached.jsonl}"
DRAFT_PART_DIR="${DRAFT_PART_DIR:-${DRAFT_CACHE_DIR}/parts}"
DRAFT_LENGTH_PART_DIR="${DRAFT_LENGTH_PART_DIR:-${DRAFT_CACHE_DIR}/length_parts}"
DRAFT_LOG_DIR="${DRAFT_LOG_DIR:-${DRAFT_CACHE_DIR}/logs}"
DRAFT_COUNTS_PATH="${DRAFT_COUNTS_PATH:-${DRAFT_CACHE_DIR}/source_counts.tsv}"
DRAFT_SUMMARY_PATH="${DRAFT_SUMMARY_PATH:-${DRAFT_CACHE_DIR}/summary.json}"
DRAFT_SOURCE_LENGTH_INDEX_PATH="${DRAFT_SOURCE_LENGTH_INDEX_PATH:-}"
DRAFT_PARTITION_MODE="${DRAFT_PARTITION_MODE:-source_patterns}"
DRAFT_PREPARE_ONLY="${DRAFT_PREPARE_ONLY:-0}"
DRAFT_RESUME_INCOMPLETE_PARTS="${DRAFT_RESUME_INCOMPLETE_PARTS:-1}"

DRAFT_SOURCE_SPECS="${DRAFT_SOURCE_SPECS:-giga_00000_00006:GSXL-0000[0-6]*-*.tar,giga_00010_00016:GSXL-0001[0-6]*-*.tar,giga_00020_00025:GSXL-0002[0-5]*-*.tar,wenet_00000_00029:WSL-000[0-2]*-*.tar,wenet_00030_00059:WSL-000[3-5]*-*.tar,wenet_00060_00089:WSL-000[6-8]*-*.tar,wenet_00120_00146:WSL-001[2-4]*-*.tar,cv_en:commonvoice_en_*.tar,wenet_00100_00119:WSL-001[0-1]*-*.tar,giga_00007_00009:GSXL-0000[7-9]*-*.tar,giga_00017_00019:GSXL-0001[7-9]*-*.tar,cv_cn:commonvoice_cn_*.tar,aishell:aishell3_*.tar,wenet_00090_00099:WSL-0009*-*.tar,libri:librispeech_*.tar}"
DRAFT_DEVICES="${DRAFT_DEVICES:-0,1,2,3}"
DRAFT_MAX_PARALLEL="${DRAFT_MAX_PARALLEL:-4}"
DRAFT_INDEX_PARTS="${DRAFT_INDEX_PARTS:-${DRAFT_MAX_PARALLEL}}"
DRAFT_ISOLATE_GPUS="${DRAFT_ISOLATE_GPUS:-1}"
DRAFT_BATCH_SIZE="${DRAFT_BATCH_SIZE:-8}"
DRAFT_NUM_WORKERS="${DRAFT_NUM_WORKERS:-4}"
DRAFT_LIMIT="${DRAFT_LIMIT:-999999999}"
DRAFT_PROGRESS_INTERVAL="${DRAFT_PROGRESS_INTERVAL:-1000}"
DRAFT_BEAM_SIZE="${DRAFT_BEAM_SIZE:-1}"
DRAFT_TOKEN_PRUNE_TOPK="${DRAFT_TOKEN_PRUNE_TOPK:-16}"
DRAFT_SKIP_DECODE_ERRORS="${DRAFT_SKIP_DECODE_ERRORS:-1}"

log() {
  printf '[ctc-draft-cache] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
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

source_length_index_path() {
  if [[ -n "${DRAFT_SOURCE_LENGTH_INDEX_PATH}" ]]; then
    printf '%s\n' "${DRAFT_SOURCE_LENGTH_INDEX_PATH}"
    return 0
  fi
  yaml_value webdataset_length_index_path
}

latest_step_checkpoint() {
  find "${CTC_RUN_DIR}" -maxdepth 1 -type f -name 'step-*.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^step-\([0-9][0-9]*\)\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

resolve_checkpoint() {
  if [[ -n "${CHECKPOINT_PATH}" && -s "${CHECKPOINT_PATH}" ]]; then
    return 0
  fi
  if [[ -s "${CTC_RUN_DIR}/wercer_best.pt" ]]; then
    CHECKPOINT_PATH="${CTC_RUN_DIR}/wercer_best.pt"
    return 0
  fi
  local latest
  latest="$(latest_step_checkpoint || true)"
  if [[ -z "${latest}" ]]; then
    echo "checkpoint not found and no step checkpoints in ${CTC_RUN_DIR}" >&2
    exit 1
  fi
  CHECKPOINT_PATH="${CTC_RUN_DIR}/step-${latest}.pt"
}

write_source_counts() {
  mkdir -p "${DRAFT_CACHE_DIR}" "${DRAFT_PART_DIR}" "${DRAFT_LENGTH_PART_DIR}" "${DRAFT_LOG_DIR}"
  "${PYTHON_BIN}" - \
    "$(source_length_index_path)" \
    "${DRAFT_COUNTS_PATH}" \
    "${DRAFT_LENGTH_PART_DIR}" \
    "${DRAFT_SOURCE_SPECS}" <<'PY'
import fnmatch
import json
import sys
from collections import Counter
from pathlib import Path

length_index = Path(sys.argv[1])
counts_path = Path(sys.argv[2])
part_dir = Path(sys.argv[3])
specs = []
for item in sys.argv[4].split(","):
    if not item:
        continue
    label, pattern = item.split(":", 1)
    specs.append((label, pattern))

counts = Counter()
total = 0
part_dir.mkdir(parents=True, exist_ok=True)
handles = {
    label: (part_dir / f"{label}.jsonl").open("w", encoding="utf-8")
    for label, _ in specs
}
with length_index.open("r", encoding="utf-8") as handle:
    try:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if str(raw.get("split")) != "train":
                continue
            total += 1
            shard = str(raw.get("shard_name") or raw.get("shard") or "")
            label = "other"
            for candidate, pattern in specs:
                if fnmatch.fnmatch(shard, pattern):
                    label = candidate
                    break
            counts[label] += 1
            part_handle = handles.get(label)
            if part_handle is not None:
                part_handle.write(line)
    finally:
        for part_handle in handles.values():
            part_handle.close()

counts_path.parent.mkdir(parents=True, exist_ok=True)
with counts_path.open("w", encoding="utf-8") as handle:
    for label, _ in specs:
        handle.write(f"{label}\t{counts[label]}\n")
    if counts["other"]:
        handle.write(f"other\t{counts['other']}\n")

print(f"train_rows={total} counts_path={counts_path} length_part_dir={part_dir}")
for label, _ in specs:
    print(f"{label}\t{counts[label]}")
if counts["other"]:
    print(f"other\t{counts['other']}")
PY
}

write_round_robin_index_parts() {
  mkdir -p "${DRAFT_CACHE_DIR}" "${DRAFT_PART_DIR}" "${DRAFT_LENGTH_PART_DIR}" "${DRAFT_LOG_DIR}"
  if [[ "${DRAFT_INDEX_PARTS}" -lt 1 ]]; then
    echo "DRAFT_INDEX_PARTS must be >= 1 for DRAFT_PARTITION_MODE=${DRAFT_PARTITION_MODE}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - \
    "$(source_length_index_path)" \
    "${DRAFT_COUNTS_PATH}" \
    "${DRAFT_LENGTH_PART_DIR}" \
    "${DRAFT_INDEX_PARTS}" <<'PY'
import json
import sys
from pathlib import Path

length_index = Path(sys.argv[1])
counts_path = Path(sys.argv[2])
part_dir = Path(sys.argv[3])
part_count = int(sys.argv[4])

part_dir.mkdir(parents=True, exist_ok=True)
handles = [
    (part_dir / f"part_{index:02d}.jsonl").open("w", encoding="utf-8")
    for index in range(part_count)
]
counts = [0 for _ in range(part_count)]
total = 0
try:
    with length_index.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            raw = json.loads(line)
            if str(raw.get("split")) != "train":
                continue
            part_index = total % part_count
            handles[part_index].write(line)
            counts[part_index] += 1
            total += 1
finally:
    for handle in handles:
        handle.close()

counts_path.parent.mkdir(parents=True, exist_ok=True)
with counts_path.open("w", encoding="utf-8") as dst:
    for index, count in enumerate(counts):
        dst.write(f"part_{index:02d}\t{count}\n")

print(f"train_rows={total} counts_path={counts_path} length_part_dir={part_dir} part_count={part_count}")
for index, count in enumerate(counts):
    print(f"part_{index:02d}\t{count}")
PY
}

write_shard_round_robin_index_parts() {
  mkdir -p "${DRAFT_CACHE_DIR}" "${DRAFT_PART_DIR}" "${DRAFT_LENGTH_PART_DIR}" "${DRAFT_LOG_DIR}"
  if [[ "${DRAFT_INDEX_PARTS}" -lt 1 ]]; then
    echo "DRAFT_INDEX_PARTS must be >= 1 for DRAFT_PARTITION_MODE=${DRAFT_PARTITION_MODE}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - \
    "$(source_length_index_path)" \
    "${DRAFT_COUNTS_PATH}" \
    "${DRAFT_LENGTH_PART_DIR}" \
    "${DRAFT_INDEX_PARTS}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

length_index = Path(sys.argv[1])
counts_path = Path(sys.argv[2])
part_dir = Path(sys.argv[3])
part_count = int(sys.argv[4])

def shard_name(raw: dict) -> str:
    return str(raw.get("shard_name") or raw.get("shard") or "")

def row_sort_key(raw: dict) -> tuple[int, int, str]:
    audio_offset = raw.get("audio_offset")
    json_offset = raw.get("json_offset")
    key = str(raw.get("key") or raw.get("utt_id") or "")
    try:
        audio_offset_int = int(audio_offset)
    except (TypeError, ValueError):
        audio_offset_int = 0
    try:
        json_offset_int = int(json_offset)
    except (TypeError, ValueError):
        json_offset_int = 0
    return audio_offset_int, json_offset_int, key

part_dir.mkdir(parents=True, exist_ok=True)
rows_by_shard: dict[str, list[tuple[tuple[int, int, str], str]]] = defaultdict(list)
total = 0
with length_index.open("r", encoding="utf-8") as src:
    for line in src:
        if not line.strip():
            continue
        raw = json.loads(line)
        if str(raw.get("split")) != "train":
            continue
        rows_by_shard[shard_name(raw)].append((row_sort_key(raw), line))
        total += 1

part_shards: list[list[str]] = [[] for _ in range(part_count)]
part_counts = [0 for _ in range(part_count)]
for shard, rows in sorted(rows_by_shard.items(), key=lambda item: (-len(item[1]), item[0])):
    part_index = min(range(part_count), key=lambda index: (part_counts[index], index))
    part_shards[part_index].append(shard)
    part_counts[part_index] += len(rows)

for part_index, shards in enumerate(part_shards):
    with (part_dir / f"part_{part_index:02d}.jsonl").open("w", encoding="utf-8") as dst:
        for shard in sorted(shards):
            for _, line in sorted(rows_by_shard[shard], key=lambda item: item[0]):
                dst.write(line)

counts_path.parent.mkdir(parents=True, exist_ok=True)
with counts_path.open("w", encoding="utf-8") as dst:
    for index, count in enumerate(part_counts):
        dst.write(f"part_{index:02d}\t{count}\n")

print(
    f"train_rows={total} counts_path={counts_path} length_part_dir={part_dir} "
    f"part_count={part_count} shards={len(rows_by_shard)}"
)
for index, count in enumerate(part_counts):
    print(f"part_{index:02d}\t{count}\tshards={len(part_shards[index])}")
PY
}

source_count() {
  local label="$1"
  awk -F '\t' -v label="${label}" '$1 == label { print $2 }' "${DRAFT_COUNTS_PATH}"
}

write_resume_length_part() {
  local full_length_part_path="$1"
  local existing_output_path="$2"
  local resume_length_part_path="$3"
  local summary_path="$4"
  "${PYTHON_BIN}" - \
    "${full_length_part_path}" \
    "${existing_output_path}" \
    "${resume_length_part_path}" \
    "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

full_length_part_path = Path(sys.argv[1])
existing_output_path = Path(sys.argv[2])
resume_length_part_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

def sample_id(raw: dict) -> str | None:
    value = raw.get("utt_id") or raw.get("id") or raw.get("audio_id") or raw.get("sid") or raw.get("key")
    return None if value is None else str(value)

seen: set[str] = set()
existing_rows = 0
bad_existing_lines = 0
with existing_output_path.open("r", encoding="utf-8") as src:
    for line in src:
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            bad_existing_lines += 1
            continue
        existing_rows += 1
        sid = sample_id(raw)
        if sid is not None:
            seen.add(sid)

total_rows = 0
remaining_rows = 0
resume_length_part_path.parent.mkdir(parents=True, exist_ok=True)
with full_length_part_path.open("r", encoding="utf-8") as src, resume_length_part_path.open("w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        raw = json.loads(line)
        total_rows += 1
        sid = sample_id(raw)
        if sid is not None and sid in seen:
            continue
        dst.write(line)
        remaining_rows += 1

summary = {
    "full_length_part_path": str(full_length_part_path),
    "existing_output_path": str(existing_output_path),
    "resume_length_part_path": str(resume_length_part_path),
    "total_length_rows": total_rows,
    "existing_output_rows": existing_rows,
    "existing_unique_ids": len(seen),
    "bad_existing_lines": bad_existing_lines,
    "remaining_rows": remaining_rows,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(remaining_rows)
PY
}

run_part() {
  local label="$1"
  local shard_pattern="$2"
  local device_id="$3"
  local rows="$4"
  local output_path="${DRAFT_PART_DIR}/${label}.jsonl"
  local preview_path="${DRAFT_PART_DIR}/${label}.preview.txt"
  local log_path="${DRAFT_LOG_DIR}/${label}.log"
  local done_path="${output_path}.done"
  local length_part_path="${DRAFT_LENGTH_PART_DIR}/${label}.jsonl"
  local effective_length_part_path="${length_part_path}"

  if [[ "${rows}" == "0" ]]; then
    log "skip ${label}; no train rows"
    return 0
  fi
  if [[ -f "${done_path}" && -s "${output_path}" ]]; then
    log "skip ${label}; done marker exists ${done_path}"
    return 0
  fi
  if [[ -f "${done_path}" && ! -s "${output_path}" ]]; then
    log "remove stale done marker for empty part label=${label}"
    rm -f "${done_path}"
  fi
  if [[ ! -s "${length_part_path}" ]]; then
    echo "length part missing or empty for ${label}: ${length_part_path}" >&2
    exit 1
  fi

  local device_arg="cuda:${device_id}"
  local env_prefix=()
  local skip_args=()
  local append_args=()
  if [[ -s "${output_path}" ]]; then
    if [[ "${DRAFT_RESUME_INCOMPLETE_PARTS}" == "1" || "${DRAFT_RESUME_INCOMPLETE_PARTS}" == "true" || "${DRAFT_RESUME_INCOMPLETE_PARTS}" == "True" ]]; then
      effective_length_part_path="${DRAFT_LENGTH_PART_DIR}/${label}.resume.jsonl"
      local resume_summary_path="${DRAFT_LENGTH_PART_DIR}/${label}.resume.summary.json"
      local remaining_rows
      remaining_rows="$(write_resume_length_part "${length_part_path}" "${output_path}" "${effective_length_part_path}" "${resume_summary_path}")"
      log "resume incomplete part label=${label} existing=${output_path} remaining_rows=${remaining_rows} resume_index=${effective_length_part_path}"
      if [[ "${remaining_rows}" == "0" ]]; then
        log "mark resumed part complete label=${label}; no missing rows remain"
        touch "${done_path}"
        return 0
      fi
      rows="${remaining_rows}"
      append_args=(--append-output)
    else
      log "remove incomplete part without done marker label=${label} path=${output_path}"
      rm -f "${output_path}" "${preview_path}"
    fi
  fi
  if [[ "${DRAFT_ISOLATE_GPUS}" == "1" ]]; then
    device_arg="cuda:0"
    env_prefix=(env "CUDA_VISIBLE_DEVICES=${device_id}")
  fi
  if [[ "${DRAFT_SKIP_DECODE_ERRORS}" == "1" ]]; then
    skip_args=(--skip-decode-errors)
  fi

  log "start part label=${label} rows=${rows} pattern=${shard_pattern} physical_gpu=${device_id}"
  "${env_prefix[@]}" "${PYTHON_BIN}" -m rwkvasr.cli.predict_ctc_labeled \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --config-yaml "${CTC_RUN_DIR}/model_config.yaml" \
    --webdataset-root "$(yaml_value webdataset_root)" \
    --webdataset-length-index-path "${effective_length_part_path}" \
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
    "${skip_args[@]}" \
    "${append_args[@]}" \
    --output-path "${output_path}" \
    --preview-path "${preview_path}" \
    --preview-count 20 \
    >"${log_path}" 2>&1
  touch "${done_path}"
  log "done part label=${label} output=${output_path}"
}

generate_parts() {
  if [[ "${DRAFT_PARTITION_MODE}" == "round_robin_index" || "${DRAFT_PARTITION_MODE}" == "shard_round_robin_index" ]]; then
    generate_round_robin_index_parts
    return 0
  fi

  IFS=',' read -r -a devices <<< "${DRAFT_DEVICES}"
  IFS=',' read -r -a specs <<< "${DRAFT_SOURCE_SPECS}"
  if [[ "${#devices[@]}" -lt 1 ]]; then
    echo "DRAFT_DEVICES must contain at least one GPU id" >&2
    exit 1
  fi
  local worker_count="${#devices[@]}"
  if [[ "${DRAFT_MAX_PARALLEL}" -lt "${worker_count}" ]]; then
    worker_count="${DRAFT_MAX_PARALLEL}"
  fi
  if [[ "${worker_count}" -lt 1 ]]; then
    echo "DRAFT_MAX_PARALLEL must be >= 1" >&2
    exit 1
  fi

  for ((worker_index = 0; worker_index < worker_count; worker_index++)); do
    (
      local device_id="${devices[${worker_index}]}"
      local spec_index spec label pattern rows
      for ((spec_index = worker_index; spec_index < ${#specs[@]}; spec_index += worker_count)); do
        spec="${specs[${spec_index}]}"
        [[ -n "${spec}" ]] || continue
        IFS=':' read -r label pattern <<< "${spec}"
        rows="$(source_count "${label}")"
        rows="${rows:-0}"
        if [[ "${rows}" == "0" ]]; then
          log "skip ${label}; no train rows"
          continue
        fi
        run_part "${label}" "${pattern}" "${device_id}" "${rows}"
      done
    ) &
  done

  local failed=0
  for job in $(jobs -rp); do
    if ! wait "${job}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "one or more draft parts failed; see ${DRAFT_LOG_DIR}" >&2
    exit 1
  fi
}

generate_round_robin_index_parts() {
  IFS=',' read -r -a devices <<< "${DRAFT_DEVICES}"
  if [[ "${#devices[@]}" -lt 1 ]]; then
    echo "DRAFT_DEVICES must contain at least one GPU id" >&2
    exit 1
  fi
  local worker_count="${#devices[@]}"
  if [[ "${DRAFT_MAX_PARALLEL}" -lt "${worker_count}" ]]; then
    worker_count="${DRAFT_MAX_PARALLEL}"
  fi
  if [[ "${worker_count}" -lt 1 ]]; then
    echo "DRAFT_MAX_PARALLEL must be >= 1" >&2
    exit 1
  fi
  if [[ "${DRAFT_INDEX_PARTS}" -gt "${worker_count}" ]]; then
    echo "DRAFT_INDEX_PARTS=${DRAFT_INDEX_PARTS} exceeds active GPU workers=${worker_count}; lower DRAFT_INDEX_PARTS or raise DRAFT_MAX_PARALLEL" >&2
    exit 1
  fi

  local part_index label rows device_id
  for ((part_index = 0; part_index < DRAFT_INDEX_PARTS; part_index++)); do
    label="$(printf 'part_%02d' "${part_index}")"
    rows="$(source_count "${label}")"
    rows="${rows:-0}"
    device_id="${devices[${part_index}]}"
    run_part "${label}" "*.tar" "${device_id}" "${rows}" &
  done

  local failed=0
  for job in $(jobs -rp); do
    if ! wait "${job}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "one or more draft parts failed; see ${DRAFT_LOG_DIR}" >&2
    exit 1
  fi
}

merge_cache() {
  log "merge parts into ${DRAFT_CACHE_PATH}"
  shopt -s nullglob
  local part_paths=("${DRAFT_PART_DIR}"/*.jsonl)
  "${PYTHON_BIN}" - "${DRAFT_CACHE_PATH}" "${DRAFT_SUMMARY_PATH}" "${part_paths[@]}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from rwkvasr.data.text_normalization import normalize_asr_text

out_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
part_paths = [Path(value) for value in sys.argv[3:]]

seen: set[str] = set()
counts = Counter()
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as dst:
    for part_path in part_paths:
        label = part_path.stem
        with part_path.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                raw = json.loads(line)
                sample_id = raw.get("utt_id") or raw.get("id") or raw.get("audio_id") or raw.get("sid") or raw.get("key")
                if sample_id is None:
                    continue
                sample_id = str(sample_id)
                if sample_id in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(sample_id)
                pred = normalize_asr_text(str(raw.get("ctc_draft") or raw.get("pred_text") or ""), mode="ctc")
                compact = {
                    "utt_id": sample_id,
                    "pred_text": pred,
                    "ctc_draft": pred,
                    "draft_source_part": label,
                    "decode_strategy": raw.get("decode_strategy"),
                }
                dst.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[label] += 1

summary = {
    "output_path": str(out_path),
    "records": len(seen),
    "duplicates": counts["duplicates"],
    "compact": True,
    "compact_fields": ["utt_id", "pred_text", "ctc_draft", "draft_source_part", "decode_strategy"],
    "parts": {key: value for key, value in sorted(counts.items()) if key != "duplicates"},
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"merged_cache output={out_path} records={len(seen)} duplicates={counts['duplicates']}")
PY
}

filter_length_index() {
  log "filter length index to cached utterances: ${DRAFT_LENGTH_INDEX_PATH}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/filter_length_index_by_ctc_draft.py" \
    --length-index-path "$(source_length_index_path)" \
    --ctc-draft-jsonl "${DRAFT_CACHE_PATH}" \
    --output-path "${DRAFT_LENGTH_INDEX_PATH}" \
    --require-all
}

main() {
  cd "${REPO_ROOT}"
  resolve_checkpoint
  log "checkpoint=${CHECKPOINT_PATH}"
  log "ctc_config=${CTC_CONFIG}"
  log "source_length_index=$(source_length_index_path)"
  log "partition_mode=${DRAFT_PARTITION_MODE}"
  log "draft_dir=${DRAFT_CACHE_DIR}"
  case "${DRAFT_PARTITION_MODE}" in
    source_patterns)
      write_source_counts
      ;;
    round_robin_index)
      write_round_robin_index_parts
      ;;
    shard_round_robin_index)
      write_shard_round_robin_index_parts
      ;;
    *)
      echo "unsupported DRAFT_PARTITION_MODE=${DRAFT_PARTITION_MODE}" >&2
      exit 1
      ;;
  esac
  if [[ "${DRAFT_PREPARE_ONLY}" == "1" || "${DRAFT_PREPARE_ONLY}" == "true" || "${DRAFT_PREPARE_ONLY}" == "True" ]]; then
    log "prepare-only mode complete; skipping draft prediction"
    wc -l "${DRAFT_LENGTH_PART_DIR}"/*.jsonl "${DRAFT_COUNTS_PATH}" || true
    return 0
  fi
  generate_parts
  merge_cache
  filter_length_index
  log "done cache=${DRAFT_CACHE_PATH} lengths=${DRAFT_LENGTH_INDEX_PATH}"
  wc -l "${DRAFT_CACHE_PATH}" "${DRAFT_LENGTH_INDEX_PATH}" || true
}

main "$@"
