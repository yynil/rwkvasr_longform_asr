#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_draft_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            for key_name in ("utt_id", "id", "audio_id", "sid", "key"):
                value = raw.get(key_name)
                if value is not None:
                    ids.add(str(value))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a WebDataset length index to utterances in a CTC draft cache.")
    parser.add_argument("--length-index-path", required=True)
    parser.add_argument("--ctc-draft-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.length_index_path)
    draft_path = Path(args.ctc_draft_jsonl)
    output_path = Path(args.output_path)
    draft_ids = _load_draft_ids(draft_path)
    seen_ids: set[str] = set()
    kept = 0
    read = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_number}") from exc
            read += 1
            candidates = [
                str(raw[key_name])
                for key_name in ("utt_id", "id", "audio_id", "sid", "key")
                if raw.get(key_name) is not None
            ]
            if any(candidate in draft_ids for candidate in candidates):
                dst.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
                seen_ids.update(candidate for candidate in candidates if candidate in draft_ids)

    missing = len(draft_ids - seen_ids)
    print(
        f"filtered_length_index input={input_path} output={output_path} "
        f"read={read} draft_ids={len(draft_ids)} kept={kept} missing_draft_ids={missing}"
    )
    if args.require_all and missing:
        raise SystemExit(f"{missing} draft ids were not found in {input_path}")


if __name__ == "__main__":
    main()
