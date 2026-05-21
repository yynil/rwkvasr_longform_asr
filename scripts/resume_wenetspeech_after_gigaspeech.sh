#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 GIGASPEECH_PID" >&2
  exit 2
fi

GIGASPEECH_PID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

POLL_SECONDS="${POLL_SECONDS:-300}"
GIGASPEECH_ROOT="${GIGASPEECH_ROOT:-/media/usbhd/training_data/asr/speechcolab/gigaspeech/webdataset_xl_train}"
GIGASPEECH_BUCKET_MANIFEST="${GIGASPEECH_BUCKET_MANIFEST:-${GIGASPEECH_ROOT}/webdataset_buckets/manifest.json}"

gigaspeech_artifacts_ready() {
  [[ -s "${GIGASPEECH_ROOT}/gigaspeech_parquet_conversion_summary.json" \
    && -s "${GIGASPEECH_ROOT}/webdataset_index.json" \
    && -s "${GIGASPEECH_ROOT}/webdataset_lengths.jsonl" \
    && -s "${GIGASPEECH_ROOT}/webdataset_lengths.summary.json" \
    && -s "${GIGASPEECH_BUCKET_MANIFEST}" ]]
}

echo "[rwkvasr] Waiting for GigaSpeech resume PID ${GIGASPEECH_PID} before WenetSpeech resume..."
while true; do
  if gigaspeech_artifacts_ready; then
    echo "[rwkvasr] GigaSpeech artifacts are ready; starting WenetSpeech resume."
    break
  fi
  if ! kill -0 "${GIGASPEECH_PID}" 2>/dev/null; then
    break
  fi
  state="$(ps -o stat= -p "${GIGASPEECH_PID}" 2>/dev/null | awk '{print $1}')"
  if [[ -z "${state}" || "${state}" == Z* ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[rwkvasr] GigaSpeech process exited; checking summary before WenetSpeech resume..."
"${PYTHON_BIN}" - "${GIGASPEECH_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary_path = root / "gigaspeech_parquet_conversion_summary.json"
if not summary_path.is_file():
    print(f"[rwkvasr] Missing {summary_path}; not starting WenetSpeech.", file=sys.stderr)
    sys.exit(1)

data = json.loads(summary_path.read_text())
processed = int(data.get("converted_shards", 0)) + int(data.get("resumed_shards", 0))
expected = int(data.get("input_shards", 0))
samples = data.get("samples")
print(f"[rwkvasr] GigaSpeech summary processed={processed} expected={expected} samples={samples}")
if expected <= 0 or processed < expected:
    print("[rwkvasr] GigaSpeech did not finish all shards; not starting WenetSpeech.", file=sys.stderr)
    sys.exit(1)
PY

cd "${REPO_ROOT}"
RESUME="${RESUME:-1}" \
ADOPT_EXISTING="${ADOPT_EXISTING:-1}" \
OVERWRITE="${OVERWRITE:-0}" \
COMPUTE_CMVN="${COMPUTE_CMVN:-0}" \
CONVERT_THREADS="${CONVERT_THREADS:-4}" \
LENGTH_THREADS="${LENGTH_THREADS:-4}" \
STAGING_ROOT="${STAGING_ROOT:-}" \
STAGING_MAX_GB="${STAGING_MAX_GB:-100}" \
STAGING_MAX_BYTES="${STAGING_MAX_BYTES:-}" \
  ./scripts/prepare_wenetspeech_l.sh
