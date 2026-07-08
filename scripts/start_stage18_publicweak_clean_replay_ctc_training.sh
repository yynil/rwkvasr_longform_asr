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

STAGE15_RUN="${STAGE15_RUN:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_bs12_lr1e6_zero1_nockpt_noaug_nodirdrop_4x4090}"

CLEAN_ROOT="${CLEAN_ROOT:-/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset}"
SOURCE_CLEAN_STAGE_NAME="${SOURCE_CLEAN_STAGE_NAME:-stage12b_source_balanced_clean_public_anchor}"
SOURCE_CLEAN_LENGTH_INDEX="${SOURCE_CLEAN_LENGTH_INDEX:-${CLEAN_ROOT}/stages/${SOURCE_CLEAN_STAGE_NAME}/webdataset_lengths.jsonl}"
REPLAY_STAGE_NAME="${REPLAY_STAGE_NAME:-stage18_publicweak_clean_replay}"
REPLAY_STAGE_DIR="${REPLAY_STAGE_DIR:-${CLEAN_ROOT}/stages/${REPLAY_STAGE_NAME}}"
REPLAY_LENGTH_INDEX="${REPLAY_LENGTH_INDEX:-${REPLAY_STAGE_DIR}/webdataset_lengths.jsonl}"
REPLAY_SUMMARY_PATH="${REPLAY_SUMMARY_PATH:-${REPLAY_STAGE_DIR}/webdataset_lengths.summary.json}"
REPLAY_BUCKET_MANIFEST="${REPLAY_BUCKET_MANIFEST:-${REPLAY_STAGE_DIR}/webdataset_buckets_audio_text/manifest.json}"
REPLAY_SOURCE_QUOTAS="${REPLAY_SOURCE_QUOTAS:-commonvoice_en:210000,librispeech:150000,aishell3:150000,commonvoice_cn:90000}"
REPLAY_SEED="${REPLAY_SEED:-20260620}"
REBUILD_REPLAY="${REBUILD_REPLAY:-0}"
BUCKET_WIDTH="${BUCKET_WIDTH:-80}"

export BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage15_sourcebalanced_clean_ctc_from_stage12b_werbest_4x4090_deepspeed.yaml}"
export INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-${STAGE15_RUN}/wercer_best.pt}"
export CLEAN_ROOT
export CLEAN_STAGE_NAME="${CLEAN_STAGE_NAME:-${REPLAY_STAGE_NAME}}"
export CLEAN_STAGE_DIR="${CLEAN_STAGE_DIR:-${REPLAY_STAGE_DIR}}"
export CLEAN_LENGTH_INDEX="${CLEAN_LENGTH_INDEX:-${REPLAY_LENGTH_INDEX}}"
export CLEAN_BUCKET_MANIFEST="${CLEAN_BUCKET_MANIFEST:-${REPLAY_BUCKET_MANIFEST}}"

export RUN_DIR="${RUN_DIR:-/media/usbhd/rwkvasr_runs/sensevoice_rwkv_stage18_publicweak_cleanreplay_ctc_from_stage15_werbest_bs12_lr7p5e7_zero1_nockpt_noaug_nodirdrop_4x4090}"
export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/generated/sensevoice_rwkv_stage18_publicweak_cleanreplay_ctc_from_stage15_werbest_4x4090_deepspeed.yaml}"
export LR="${LR:-7.5e-7}"
export MAX_STEPS="${MAX_STEPS:-8000}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export MASTER_PORT="${MASTER_PORT:-29618}"

export SIDECAR_SESSION="${SIDECAR_SESSION:-sidecar_stage18_publicweak_clean_ctc}"
export SIDECAR_LIMIT="${SIDECAR_LIMIT:-96}"
export SIDECAR_SOURCE_QUOTAS="${SIDECAR_SOURCE_QUOTAS:-clean_librispeech:librispeech_*.tar:24,clean_aishell:aishell3_*.tar:24,clean_cv_en:commonvoice_en_*.tar:24,clean_cv_cn:commonvoice_cn_*.tar:24}"

log() {
  printf '[stage18-publicweak-clean-ctc] %(%Y-%m-%d %H:%M:%S)T %s\n' -1 "$*"
}

build_replay_stage() {
  if [[ ! -s "${SOURCE_CLEAN_LENGTH_INDEX}" ]]; then
    echo "source clean length index missing: ${SOURCE_CLEAN_LENGTH_INDEX}" >&2
    exit 1
  fi
  if [[ "${REBUILD_REPLAY}" != "1" && -s "${REPLAY_LENGTH_INDEX}" && -s "${REPLAY_SUMMARY_PATH}" && -s "${REPLAY_BUCKET_MANIFEST}" ]]; then
    log "reusing replay stage=${REPLAY_STAGE_NAME} length_index=${REPLAY_LENGTH_INDEX} bucket_manifest=${REPLAY_BUCKET_MANIFEST}"
    return
  fi

  log "building weighted clean replay stage=${REPLAY_STAGE_NAME} quotas=${REPLAY_SOURCE_QUOTAS}"
  export SOURCE_CLEAN_LENGTH_INDEX REPLAY_LENGTH_INDEX REPLAY_SUMMARY_PATH REPLAY_SOURCE_QUOTAS REPLAY_SEED
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_quotas(raw: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"invalid source quota {item!r}; expected source:count")
        source, raw_count = item.split(":", 1)
        source = source.strip()
        count = int(raw_count)
        if not source or count < 0:
            raise SystemExit(f"invalid source quota {item!r}")
        quotas[source] = count
    if not quotas:
        raise SystemExit("at least one source quota is required")
    return quotas


def source_from_row(row: dict) -> str:
    source = str(row.get("source_dataset") or row.get("source") or "").strip().lower()
    if source:
        return source
    shard_name = str(row.get("shard_name") or row.get("shard") or "")
    if shard_name.startswith("librispeech_"):
        return "librispeech"
    if shard_name.startswith("aishell3_"):
        return "aishell3"
    if shard_name.startswith("commonvoice_en_"):
        return "commonvoice_en"
    if shard_name.startswith("commonvoice_cn_"):
        return "commonvoice_cn"
    return "unknown"


source_path = Path(os.environ["SOURCE_CLEAN_LENGTH_INDEX"])
output_path = Path(os.environ["REPLAY_LENGTH_INDEX"])
summary_path = Path(os.environ["REPLAY_SUMMARY_PATH"])
quotas = parse_quotas(os.environ["REPLAY_SOURCE_QUOTAS"])
seed = int(os.environ["REPLAY_SEED"])

rows_by_source: dict[str, list[dict]] = defaultdict(list)
source_counts: Counter[str] = Counter()
with source_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        raw_line = line.strip()
        if not raw_line:
            continue
        row = json.loads(raw_line)
        if str(row.get("split") or "train") != "train":
            continue
        source = source_from_row(row)
        source_counts[source] += 1
        if source in quotas:
            rows_by_source[source].append(row)

sampled: list[dict] = []
kept_counts: Counter[str] = Counter()
for source, count in sorted(quotas.items()):
    rows = rows_by_source.get(source, [])
    if not rows and count > 0:
        raise SystemExit(f"source {source!r} has no train rows in {source_path}")
    rng = random.Random(seed + sum(ord(ch) for ch in source) * 104729)
    for index in range(count):
        row = dict(rng.choice(rows))
        row["split"] = "train"
        row["_stage18_replay_source"] = source
        row["_stage18_replay_index"] = index
        sampled.append(row)
    kept_counts[source] = count

shuffle_rng = random.Random(seed + 999_983)
shuffle_rng.shuffle(sampled)

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    for row in sampled:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

summary = {
    "version": 1,
    "stage_name": "stage18_publicweak_clean_replay",
    "source_length_index_path": str(source_path),
    "output_path": str(output_path),
    "seed": seed,
    "sampling": "with_replacement_by_source",
    "quotas": quotas,
    "source_counts": dict(sorted(source_counts.items())),
    "kept_counts": dict(sorted(kept_counts.items())),
    "num_output_rows": len(sampled),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"stage18_replay_index={output_path} rows={len(sampled)} summary={summary_path}")
PY

  log "building bucket manifest=${REPLAY_BUCKET_MANIFEST}"
  cargo run \
    --release \
    --manifest-path "${REPO_ROOT}/tools/Cargo.toml" \
    --bin build_bucket_index \
    -- \
    --shard-root "${CLEAN_ROOT}" \
    --length-index-path "${REPLAY_LENGTH_INDEX}" \
    --output-dir "${REPLAY_STAGE_DIR}/webdataset_buckets_audio_text" \
    --manifest-path "${REPLAY_BUCKET_MANIFEST}" \
    --bucket-width "${BUCKET_WIDTH}" \
    --text-cost-source auto \
    --text-cost-weight 4 \
    --json-size-text-offset 256 \
    --json-size-bytes-per-token 4.0 \
    --entries-per-part 100000
}

main() {
  cd "${REPO_ROOT}"
  build_replay_stage
  exec "${REPO_ROOT}/scripts/start_stage15_clean_stability_ctc_training.sh" "$@"
}

main "$@"
