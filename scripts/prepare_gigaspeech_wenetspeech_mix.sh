#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python not found at ${PYTHON_BIN}" >&2
  exit 1
fi

GIGASPEECH_ROOT="${1:-/media/usbhd/training_data/asr/speechcolab/gigaspeech/webdataset_xl_train}"
WENETSPEECH_ROOT="${2:-/media/usbhd/training_data/asr/wenet-e2e/wenetspeech/webdataset_l_train}"
MIX_ROOT="${3:-/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset}"
INDEX_PATH="${MIX_ROOT}/webdataset_index.json"
LENGTHS_PATH="${MIX_ROOT}/webdataset_lengths.jsonl"
SUMMARY_PATH="${MIX_ROOT}/webdataset_lengths.summary.json"
BUCKET_DIR="${MIX_ROOT}/webdataset_buckets"
BUCKET_MANIFEST_PATH="${BUCKET_DIR}/manifest.json"
CMVN_PATH="${MIX_ROOT}/global_cmvn.json"

BUCKET_WIDTH="${BUCKET_WIDTH:-80}"
ENTRIES_PER_PART="${ENTRIES_PER_PART:-100000}"
BUCKET_TEXT_COST_SOURCE="${BUCKET_TEXT_COST_SOURCE:-auto}"
BUCKET_TEXT_COST_WEIGHT="${BUCKET_TEXT_COST_WEIGHT:-4}"
BUCKET_JSON_SIZE_TEXT_OFFSET="${BUCKET_JSON_SIZE_TEXT_OFFSET:-256}"
BUCKET_JSON_SIZE_BYTES_PER_TOKEN="${BUCKET_JSON_SIZE_BYTES_PER_TOKEN:-4.0}"
EVAL_RATIO="${EVAL_RATIO:-0.01}"
HASH_SEED="${HASH_SEED:-0}"
LENGTH_THREADS="${LENGTH_THREADS:-4}"
COMPUTE_CMVN="${COMPUTE_CMVN:-1}"
OVERWRITE="${OVERWRITE:-0}"
FAST_MERGE_INDEXES="${FAST_MERGE_INDEXES:-1}"
WRITE_IDENTITY_CMVN_IF_SKIPPED="${WRITE_IDENTITY_CMVN_IF_SKIPPED:-1}"
IDENTITY_CMVN_PATH="${IDENTITY_CMVN_PATH:-${MIX_ROOT}/identity_cmvn.json}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${MIX_ROOT}"

echo "[rwkvasr] Refreshing mixed-root symlinks under ${MIX_ROOT}"
find "${MIX_ROOT}" -maxdepth 1 -type l -name '*.tar' -delete

while IFS= read -r shard_path; do
  ln -sfn "${shard_path}" "${MIX_ROOT}/$(basename "${shard_path}")"
done < <(find "${GIGASPEECH_ROOT}" -maxdepth 1 -name '*.tar' | sort)

while IFS= read -r shard_path; do
  ln -sfn "${shard_path}" "${MIX_ROOT}/$(basename "${shard_path}")"
done < <(find "${WENETSPEECH_ROOT}" -maxdepth 1 -name '*.tar' | sort)

cd "${REPO_ROOT}"

if [[ "${FAST_MERGE_INDEXES}" == "1" \
  && -s "${GIGASPEECH_ROOT}/webdataset_index.json" \
  && -s "${GIGASPEECH_ROOT}/webdataset_lengths.jsonl" \
  && -s "${GIGASPEECH_ROOT}/webdataset_lengths.summary.json" \
  && -s "${WENETSPEECH_ROOT}/webdataset_index.json" \
  && -s "${WENETSPEECH_ROOT}/webdataset_lengths.jsonl" \
  && -s "${WENETSPEECH_ROOT}/webdataset_lengths.summary.json" ]]; then
  if [[ "${OVERWRITE}" != "1" && -s "${INDEX_PATH}" && -s "${LENGTHS_PATH}" && -s "${SUMMARY_PATH}" ]]; then
    echo "[rwkvasr] Skipping mixed index merge because artifacts already exist."
  else
    echo "[rwkvasr] Fast-merging mixed WebDataset index and length index"
    "${PYTHON_BIN}" - "${GIGASPEECH_ROOT}" "${WENETSPEECH_ROOT}" "${MIX_ROOT}" "${INDEX_PATH}" "${LENGTHS_PATH}" "${SUMMARY_PATH}" <<'PY'
import json
import pathlib
import shutil
import sys

giga_root = pathlib.Path(sys.argv[1])
wenet_root = pathlib.Path(sys.argv[2])
mix_root = pathlib.Path(sys.argv[3])
index_path = pathlib.Path(sys.argv[4])
lengths_path = pathlib.Path(sys.argv[5])
summary_path = pathlib.Path(sys.argv[6])
sources = [giga_root, wenet_root]

def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

indexes = [load_json(root / "webdataset_index.json") for root in sources]
summaries = [load_json(root / "webdataset_lengths.summary.json") for root in sources]
split_ref = indexes[0]["split"]
summary_split_ref = summaries[0]["split"]
for data in indexes[1:]:
    if data["split"] != split_ref:
        raise SystemExit(f"split config mismatch: {data['split']} vs {split_ref}")
for data in summaries[1:]:
    if data["split"] != summary_split_ref:
        raise SystemExit(f"summary split config mismatch: {data['split']} vs {summary_split_ref}")

train_name = split_ref["train_name"]
eval_name = split_ref["eval_name"]
merged_shards = []
for data in indexes:
    merged_shards.extend(data["shards"])
merged_shards.sort(key=lambda item: item["name"])

merged_index = {
    "version": 1,
    "root": str(mix_root),
    "shard_pattern": "*.tar",
    "num_shards": sum(int(data["num_shards"]) for data in indexes),
    "num_samples": sum(int(data["num_samples"]) for data in indexes),
    "split": split_ref,
    "splits": {
        train_name: {"num_samples": sum(int(data["splits"][train_name]["num_samples"]) for data in indexes)},
        eval_name: {"num_samples": sum(int(data["splits"][eval_name]["num_samples"]) for data in indexes)},
    },
    "shards": merged_shards,
}
index_path.parent.mkdir(parents=True, exist_ok=True)
tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
tmp_index.write_text(json.dumps(merged_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp_index.replace(index_path)

lengths_path.parent.mkdir(parents=True, exist_ok=True)
tmp_lengths = lengths_path.with_suffix(lengths_path.suffix + ".tmp")
with tmp_lengths.open("wb") as out:
    for root in sources:
        with (root / "webdataset_lengths.jsonl").open("rb") as src:
            shutil.copyfileobj(src, out, length=64 * 1024 * 1024)
tmp_lengths.replace(lengths_path)

frame_buckets: dict[str, int] = {}
audio_suffixes: set[str] = set()
for data in summaries:
    for key, value in data.get("frame_buckets", {}).items():
        frame_buckets[key] = frame_buckets.get(key, 0) + int(value)
    audio_suffixes.update(str(value) for value in data.get("audio_suffixes", []))

merged_summary = {
    "version": 2,
    "root": str(mix_root),
    "length_index_path": str(lengths_path),
    "num_shards": sum(int(data["num_shards"]) for data in summaries),
    "num_samples": sum(int(data["num_samples"]) for data in summaries),
    "min_frames": min(int(data["min_frames"]) for data in summaries),
    "max_frames": max(int(data["max_frames"]) for data in summaries),
    "audio_suffixes": sorted(audio_suffixes),
    "split": summary_split_ref,
    "splits": {
        train_name: {"num_samples": sum(int(data["splits"][train_name]["num_samples"]) for data in summaries)},
        eval_name: {"num_samples": sum(int(data["splits"][eval_name]["num_samples"]) for data in summaries)},
    },
    "frame_buckets": dict(sorted(frame_buckets.items(), key=lambda item: int(item[0]))),
}
tmp_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
tmp_summary.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp_summary.replace(summary_path)

print(
    "[rwkvasr] Fast merge complete "
    f"shards={merged_index['num_shards']} samples={merged_index['num_samples']} "
    f"train={merged_index['splits'][train_name]['num_samples']} eval={merged_index['splits'][eval_name]['num_samples']}"
)
PY
  fi
else
  if [[ "${OVERWRITE}" != "1" && -s "${INDEX_PATH}" ]]; then
    echo "[rwkvasr] Skipping mixed WebDataset index because it already exists: ${INDEX_PATH}"
  else
    echo "[rwkvasr] Building mixed WebDataset index: ${INDEX_PATH}"
    "${PYTHON_BIN}" -m rwkvasr.cli.inspect_webdataset \
      --webdataset-root "${MIX_ROOT}" \
      --output-path "${INDEX_PATH}" \
      --split-by shard_name \
      --eval-ratio "${EVAL_RATIO}" \
      --hash-seed "${HASH_SEED}" \
      --utt-id-key id
  fi

  if [[ "${OVERWRITE}" != "1" && -s "${LENGTHS_PATH}" && -s "${SUMMARY_PATH}" ]]; then
    echo "[rwkvasr] Skipping mixed Rust length index because it already exists: ${LENGTHS_PATH}"
  else
    echo "[rwkvasr] Building mixed Rust length index: ${LENGTHS_PATH}"
    cargo run --release --manifest-path tools/Cargo.toml --bin rwkvasr-tools -- \
      --webdataset-root "${MIX_ROOT}" \
      --output-path "${LENGTHS_PATH}" \
      --summary-path "${SUMMARY_PATH}" \
      --split-by shard_name \
      --eval-ratio "${EVAL_RATIO}" \
      --hash-seed "${HASH_SEED}" \
      --utt-id-key id \
      --threads "${LENGTH_THREADS}"
  fi
fi

if [[ "${OVERWRITE}" != "1" && -s "${BUCKET_MANIFEST_PATH}" ]]; then
  echo "[rwkvasr] Skipping mixed bucket manifest because it already exists: ${BUCKET_MANIFEST_PATH}"
else
  echo "[rwkvasr] Building mixed bucket manifest: ${BUCKET_MANIFEST_PATH}"
  cargo run --release --manifest-path tools/Cargo.toml --bin build_bucket_index -- \
    --shard-root "${MIX_ROOT}" \
    --length-index-path "${LENGTHS_PATH}" \
    --output-dir "${BUCKET_DIR}" \
    --manifest-path "${BUCKET_MANIFEST_PATH}" \
    --bucket-width "${BUCKET_WIDTH}" \
    --text-cost-source "${BUCKET_TEXT_COST_SOURCE}" \
    --text-cost-weight "${BUCKET_TEXT_COST_WEIGHT}" \
    --json-size-text-offset "${BUCKET_JSON_SIZE_TEXT_OFFSET}" \
    --json-size-bytes-per-token "${BUCKET_JSON_SIZE_BYTES_PER_TOKEN}" \
    --entries-per-part "${ENTRIES_PER_PART}"
fi

if [[ "${COMPUTE_CMVN}" == "1" ]]; then
  if [[ "${OVERWRITE}" != "1" && -s "${CMVN_PATH}" ]]; then
    echo "[rwkvasr] Skipping mixed global CMVN because it already exists: ${CMVN_PATH}"
  else
    echo "[rwkvasr] Computing mixed global CMVN: ${CMVN_PATH}"
    "${PYTHON_BIN}" -m rwkvasr.cli.compute_cmvn \
      --webdataset-root "${MIX_ROOT}" \
      --webdataset-split train \
      --webdataset-eval-ratio "${EVAL_RATIO}" \
      --webdataset-hash-seed "${HASH_SEED}" \
      --webdataset-split-by shard_name \
      --webdataset-utt-id-key id \
      --output-path "${CMVN_PATH}"
  fi
else
  echo "[rwkvasr] Skipping CMVN because COMPUTE_CMVN=${COMPUTE_CMVN}"
  if [[ "${WRITE_IDENTITY_CMVN_IF_SKIPPED}" == "1" && ! -s "${IDENTITY_CMVN_PATH}" ]]; then
    echo "[rwkvasr] Writing identity CMVN for launch-time training: ${IDENTITY_CMVN_PATH}"
    "${PYTHON_BIN}" - "${IDENTITY_CMVN_PATH}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps({"frame_num": 1, "mean_stat": [0.0] * 80, "var_stat": [1.0] * 80}, indent=2) + "\n",
    encoding="utf-8",
)
PY
  fi
fi

echo "[rwkvasr] GigaSpeech + WenetSpeech mixed preprocessing complete."
echo "[rwkvasr] mix_root=${MIX_ROOT}"
echo "[rwkvasr] index=${INDEX_PATH}"
echo "[rwkvasr] lengths=${LENGTHS_PATH}"
echo "[rwkvasr] bucket_manifest=${BUCKET_MANIFEST_PATH}"
echo "[rwkvasr] cmvn=${CMVN_PATH}"
echo "[rwkvasr] identity_cmvn=${IDENTITY_CMVN_PATH}"
