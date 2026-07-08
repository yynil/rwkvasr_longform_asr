#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Reservoir:
    size: int
    rng: random.Random
    seen: int = 0
    rows: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        if self.size <= 0:
            return
        self.seen += 1
        if len(self.rows) < self.size:
            self.rows.append(line)
            return
        index = self.rng.randrange(self.seen)
        if index < self.size:
            self.rows[index] = line


def _parse_quota(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"quota must be source:count, got {value!r}")
    source, raw_count = value.split(":", 1)
    source = source.strip()
    if not source:
        raise argparse.ArgumentTypeError(f"quota source is empty in {value!r}")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"quota count must be an integer, got {value!r}") from exc
    if count < 0:
        raise argparse.ArgumentTypeError(f"quota count must be non-negative, got {value!r}")
    return source, count


def _source_from_row(row: dict[str, Any]) -> str:
    source = str(row.get("source_dataset") or row.get("source") or "").strip().lower()
    if source in {"gigaspeech", "wenetspeech"}:
        return source
    shard_name = str(row.get("shard_name") or row.get("shard") or "")
    if shard_name.startswith("GSXL-"):
        return "gigaspeech"
    if shard_name.startswith("WSL-"):
        return "wenetspeech"
    if shard_name.startswith("librispeech_"):
        return "librispeech"
    if shard_name.startswith("aishell3_"):
        return "aishell3"
    if shard_name.startswith("commonvoice_en_"):
        return "commonvoice_en"
    if shard_name.startswith("commonvoice_cn_"):
        return "commonvoice_cn"
    return source or "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reservoir-sample a WebDataset length index by inferred source."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--quota", action="append", required=True, type=_parse_quota)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--shuffle-output", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)
    quotas = dict(args.quota)
    reservoirs = {
        source: Reservoir(size=count, rng=random.Random(int(args.seed) + idx * 104729))
        for idx, (source, count) in enumerate(sorted(quotas.items()))
    }
    source_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    split = str(args.split)

    processed = 0
    matched_split = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_number}") from exc
            processed += 1
            if str(row.get("split") or "train") != split:
                continue
            matched_split += 1
            source = _source_from_row(row)
            source_counts[source] += 1
            reservoir = reservoirs.get(source)
            if reservoir is not None:
                reservoir.add(raw_line)
            if args.progress_interval > 0 and processed % int(args.progress_interval) == 0:
                print(
                    f"[sample-length-index] processed={processed} matched_split={matched_split}",
                    flush=True,
                )

    output_rows: list[tuple[str, str]] = []
    for source, reservoir in sorted(reservoirs.items()):
        if len(reservoir.rows) < reservoir.size:
            print(
                "[sample-length-index] warning: "
                f"source={source} requested={reservoir.size} got={len(reservoir.rows)}",
                flush=True,
            )
        kept_counts[source] = len(reservoir.rows)
        output_rows.extend((source, row) for row in reservoir.rows)

    if args.shuffle_output:
        rng = random.Random(int(args.seed) + 999_983)
        rng.shuffle(output_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in output_rows:
            handle.write(row + "\n")

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "split": split,
        "seed": int(args.seed),
        "processed_rows": processed,
        "matched_split_rows": matched_split,
        "quotas": quotas,
        "source_counts": dict(sorted(source_counts.items())),
        "kept_counts": dict(sorted(kept_counts.items())),
        "num_output_rows": len(output_rows),
        "shuffle_output": bool(args.shuffle_output),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"sampled_length_index output={output_path} rows={len(output_rows)} summary={summary_path}")


if __name__ == "__main__":
    main()
