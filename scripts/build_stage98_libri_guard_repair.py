#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix"
DEFAULT_STAGE61_LENGTHS = (
    DEFAULT_ROOT + "/stages/stage61_difficulty_public_wenet_replay/webdataset_lengths.jsonl"
)
DEFAULT_STAGE22_LENGTHS = (
    DEFAULT_ROOT + "/stages/stage22_public90_hard10_from_stage21_soup/webdataset_lengths.jsonl"
)
DEFAULT_STAGE_NAME = "stage98_stage95_libri_guard_repair"
DEFAULT_QUOTAS = {
    "librispeech": 70_000,
    "commonvoice_en": 35_000,
    "wenetspeech": 35_000,
    "aishell3": 10_000,
    "commonvoice_cn": 5_000,
    "gigaspeech": 5_000,
}
SOURCE_FIELDS = (
    "_stage98_source",
    "_stage82_source",
    "_stage61_source",
    "_stage59_source",
    "_stage47_source",
    "_stage43_source",
    "_stage39_source",
    "_stage37_selector_source",
    "source_dataset",
    "source",
    "dataset",
)
NAME_FIELDS = ("shard_name", "key", "audio_member", "json_member", "utt_id", "id")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _canonical_source(value: Any) -> str | None:
    text = str(value or "").lower()
    if not text:
        return None
    if "librispeech" in text or "libri" in text:
        return "librispeech"
    if "commonvoice_en" in text or "commonvoice-en" in text or "cv_en" in text:
        return "commonvoice_en"
    if "commonvoice_cn" in text or "commonvoice-cn" in text or "cv_cn" in text:
        return "commonvoice_cn"
    if "wenetspeech" in text or "wenet" in text or "wsl-" in text:
        return "wenetspeech"
    if "aishell" in text:
        return "aishell3"
    if "gigaspeech" in text or "gsxl" in text or "gs-" in text:
        return "gigaspeech"
    return None


def _infer_source(row: dict[str, Any]) -> str:
    for field in SOURCE_FIELDS:
        source = _canonical_source(row.get(field))
        if source is not None:
            return source
    for field in NAME_FIELDS:
        source = _canonical_source(row.get(field))
        if source is not None:
            return source
    return "unknown"


def _utt_id(row: dict[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("id") or row.get("key") or "")


def _load_pools(paths: list[tuple[str, Path]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    counts_by_input: dict[str, Counter[str]] = {}
    duplicate_rows = 0
    unknown_rows = 0
    total_rows = 0
    for label, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        input_counts: Counter[str] = Counter()
        for row in _iter_jsonl(path):
            total_rows += 1
            source = _infer_source(row)
            input_counts[source] += 1
            if source == "unknown":
                unknown_rows += 1
                continue
            uid = _utt_id(row)
            dedupe_key = f"{source}:{uid}" if uid else f"{source}:{row.get('shard_name')}:{row.get('key')}"
            if dedupe_key in seen:
                duplicate_rows += 1
                continue
            seen.add(dedupe_key)
            enriched = dict(row)
            enriched["_stage98_source"] = source
            enriched["_stage98_input"] = label
            pools[source].append(enriched)
        counts_by_input[label] = input_counts
    summary = {
        "total_input_rows": total_rows,
        "unknown_rows": unknown_rows,
        "duplicate_rows": duplicate_rows,
        "counts_by_input": {
            label: dict(sorted(counter.items())) for label, counter in sorted(counts_by_input.items())
        },
        "available_by_source": {source: len(rows) for source, rows in sorted(pools.items())},
    }
    return pools, summary


def _parse_quotas(raw: str | None) -> dict[str, int]:
    if raw is None or raw.strip() == "":
        return dict(DEFAULT_QUOTAS)
    quotas: dict[str, int] = {}
    for item in raw.split(","):
        source, count = item.split(":", 1)
        source = source.strip()
        if source not in DEFAULT_QUOTAS:
            raise ValueError(f"unknown source in quota: {source}")
        quotas[source] = int(count.strip())
    missing = set(DEFAULT_QUOTAS) - set(quotas)
    if missing:
        raise ValueError(f"missing quotas for sources: {sorted(missing)}")
    return quotas


def _sample_rows(
    *,
    pools: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    rng: random.Random,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    sampled_counts: Counter[str] = Counter()
    fill_counts: Counter[str] = Counter()
    input_counts: Counter[str] = Counter()

    for source, quota in quotas.items():
        pool = pools.get(source, [])
        if not pool:
            raise ValueError(f"no rows available for source={source}")
        if len(pool) >= quota:
            chosen = rng.sample(pool, quota)
        else:
            chosen = list(pool)
            fill = [rng.choice(pool) for _ in range(quota - len(pool))]
            fill_counts[source] = len(fill)
            chosen.extend(fill)
        for index, row in enumerate(chosen):
            out = dict(row)
            out["_stage98_component"] = "libri_guard_repair"
            out["_stage98_selector"] = DEFAULT_STAGE_NAME
            out["_stage98_quota"] = quota
            out["_stage98_seed"] = seed
            if index >= len(pool):
                out["_stage98_replacement_fill"] = True
            sampled.append(out)
            sampled_counts[source] += 1
            input_counts[str(out.get("_stage98_input") or "unknown")] += 1

    rng.shuffle(sampled)
    summary = {
        "stage_name": DEFAULT_STAGE_NAME,
        "seed": seed,
        "quotas": dict(sorted(quotas.items())),
        "num_rows": len(sampled),
        "source_counts": dict(sorted(sampled_counts.items())),
        "input_counts": dict(sorted(input_counts.items())),
        "replacement_fill_counts": dict(sorted(fill_counts.items())),
    }
    return sampled, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage98 Libri-heavy guard repair length index.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--stage-name", default=DEFAULT_STAGE_NAME)
    parser.add_argument("--stage61-lengths", default=DEFAULT_STAGE61_LENGTHS)
    parser.add_argument("--stage22-lengths", default=DEFAULT_STAGE22_LENGTHS)
    parser.add_argument("--quotas", default=None, help="Comma-separated source:count values.")
    parser.add_argument("--seed", default=20260621, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stage_name != DEFAULT_STAGE_NAME:
        raise ValueError("this script records DEFAULT_STAGE_NAME in audit fields; update the script for a new stage")
    root = Path(args.root)
    stage_dir = root / "stages" / args.stage_name
    output_path = stage_dir / "webdataset_lengths.jsonl"
    summary_path = stage_dir / "webdataset_lengths.summary.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to rebuild")

    quotas = _parse_quotas(args.quotas)
    rng = random.Random(args.seed)
    pools, input_summary = _load_pools(
        [
            ("stage61_guard_balanced", Path(args.stage61_lengths)),
            ("stage22_clean_fallback", Path(args.stage22_lengths)),
        ]
    )
    rows, sample_summary = _sample_rows(pools=pools, quotas=quotas, rng=rng, seed=args.seed)
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, rows)
    summary = {
        **sample_summary,
        "root": str(root),
        "length_index_path": str(output_path),
        "source_length_indexes": {
            "stage61_guard_balanced": str(Path(args.stage61_lengths)),
            "stage22_clean_fallback": str(Path(args.stage22_lengths)),
        },
        "input_summary": input_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} rows={len(rows)}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
