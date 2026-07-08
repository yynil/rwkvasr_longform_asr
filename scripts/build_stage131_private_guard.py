#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_STAGE22_ROOT = "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix"
DEFAULT_STAGE61_LENGTHS = (
    DEFAULT_STAGE22_ROOT + "/stages/stage61_difficulty_public_wenet_replay/webdataset_lengths.jsonl"
)
DEFAULT_STAGE22_LENGTHS = (
    DEFAULT_STAGE22_ROOT + "/stages/stage22_public90_hard10_from_stage21_soup/webdataset_lengths.jsonl"
)
DEFAULT_HARD_ROOT = "/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset"
DEFAULT_HARD_LENGTHS = DEFAULT_HARD_ROOT + "/webdataset_lengths.jsonl"
DEFAULT_SELECTOR_LENGTHS = (
    "/media/usbhd/rwkvasr_runs/stage37_ctc_decode_audit_20260620/"
    "stage30_selector256.lengths.jsonl"
)
DEFAULT_STAGE129_LENGTHS = (
    DEFAULT_STAGE22_ROOT
    + "/stages/stage129_stage110_expanded_wenet_teacher_balanced_guard/webdataset_lengths.jsonl"
)
DEFAULT_OUTPUT_DIR = "/media/usbhd/rwkvasr_runs/stage131_private_guard_20260622"
DEFAULT_CLEAN_QUOTAS = {
    "librispeech": 256,
    "commonvoice_en": 256,
    "aishell3": 128,
    "commonvoice_cn": 128,
}
DEFAULT_HARD_QUOTAS = {
    "wenetspeech": 256,
    "gigaspeech": 128,
}

SOURCE_FIELDS = (
    "_stage131_source",
    "_stage129_source",
    "_stage125_source",
    "_stage119_source",
    "_stage110_source",
    "_stage104_source",
    "_stage98_source",
    "_stage82_source",
    "_stage61_source",
    "_stage59_source",
    "_stage53_source",
    "_stage47_source",
    "_stage43_source",
    "_stage39_source",
    "_stage37_selector_source",
    "source_dataset",
    "source",
    "dataset",
)
NAME_FIELDS = ("shard_name", "key", "audio_member", "json_member", "utt_id", "id")


class Reservoir:
    def __init__(self, size: int, rng: random.Random) -> None:
        self.size = size
        self.rng = rng
        self.seen = 0
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        if self.size <= 0:
            return
        self.seen += 1
        if len(self.rows) < self.size:
            self.rows.append(row)
            return
        index = self.rng.randrange(self.seen)
        if index < self.size:
            self.rows[index] = row


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _utt_id(row: dict[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("id") or row.get("key") or "")


def _canonical_source(raw: Any) -> str | None:
    value = str(raw or "").lower()
    if not value:
        return None
    if "librispeech" in value or "libri" in value:
        return "librispeech"
    if "aishell" in value:
        return "aishell3"
    if "commonvoice_en" in value or "commonvoice-en" in value or "cv_en" in value:
        return "commonvoice_en"
    if "commonvoice_cn" in value or "commonvoice-cn" in value or "cv_cn" in value:
        return "commonvoice_cn"
    if "gigaspeech" in value or "gsxl" in value or "gs-" in value:
        return "gigaspeech"
    if "wenetspeech" in value or "wenet" in value or "wsl-" in value:
        return "wenetspeech"
    return None


def _infer_source(row: dict[str, Any]) -> str:
    for field in SOURCE_FIELDS:
        source = _canonical_source(row.get(field))
        if source:
            return source
    for field in NAME_FIELDS:
        source = _canonical_source(row.get(field))
        if source:
            return source
    return "unknown"


def _parse_quotas(raw: str | None, defaults: dict[str, int]) -> dict[str, int]:
    if raw is None or raw.strip() == "":
        return dict(defaults)
    quotas: dict[str, int] = {}
    for item in raw.split(","):
        source, count = item.split(":", 1)
        source = source.strip()
        if source not in defaults:
            raise ValueError(f"unknown source in quota: {source}")
        quotas[source] = int(count.strip())
    missing = set(defaults) - set(quotas)
    if missing:
        raise ValueError(f"missing quotas for sources: {sorted(missing)}")
    return quotas


def _load_ids(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        before = len(ids)
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in _iter_jsonl(path):
            utt_id = _utt_id(row)
            if utt_id:
                ids.add(utt_id)
        counts[str(path)] = len(ids) - before
    return ids, counts


def _collect_pools(
    *,
    inputs: list[Path],
    quotas: dict[str, int],
    exclude_ids: set[str],
    split: str,
    pool_name: str,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    reservoirs = {
        source: Reservoir(size=count, rng=random.Random(seed + idx * 104_729))
        for idx, (source, count) in enumerate(sorted(quotas.items()))
    }
    seen_keys: set[str] = set()
    use_seen_dedupe = pool_name != "hard"
    counts_by_input: dict[str, Counter[str]] = {}
    excluded_by_source: Counter[str] = Counter()
    eligible_seen_by_source: Counter[str] = Counter()
    duplicate_rows = 0
    unknown_rows = 0
    split_skipped = 0

    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        counts: Counter[str] = Counter()
        for row in _iter_jsonl(path):
            if str(row.get("split") or "train") != split:
                split_skipped += 1
                continue
            source = _infer_source(row)
            counts[source] += 1
            if source not in quotas:
                if source == "unknown":
                    unknown_rows += 1
                continue
            utt_id = _utt_id(row)
            if utt_id in exclude_ids:
                excluded_by_source[source] += 1
                continue
            dedupe_key = f"{source}:{utt_id}" if utt_id else (
                f"{source}:{row.get('shard_name')}:{row.get('key')}"
            )
            if use_seen_dedupe:
                if dedupe_key in seen_keys:
                    duplicate_rows += 1
                    continue
                seen_keys.add(dedupe_key)
            out = dict(row)
            out["_stage131_component"] = f"{pool_name}_private_guard"
            out["_stage131_source"] = source
            out["_stage131_pool"] = pool_name
            out["_stage131_seed"] = seed
            eligible_seen_by_source[source] += 1
            reservoirs[source].add(out)
        counts_by_input[str(path)] = counts

    pools = {source: list(reservoir.rows) for source, reservoir in reservoirs.items()}
    summary = {
        "inputs": [str(path) for path in inputs],
        "pool_name": pool_name,
        "split": split,
        "available_by_source": {
            source: int(eligible_seen_by_source.get(source, 0)) for source in sorted(quotas)
        },
        "reservoir_kept_by_source": {
            source: len(pools.get(source, [])) for source in sorted(quotas)
        },
        "input_counts": {
            path: dict(sorted(counter.items())) for path, counter in sorted(counts_by_input.items())
        },
        "excluded_by_source": dict(sorted(excluded_by_source.items())),
        "duplicate_rows": duplicate_rows,
        "unknown_rows": unknown_rows,
        "split_skipped": split_skipped,
    }
    return pools, summary


def _sample_pools(
    *,
    pools: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    rng: random.Random,
    selector_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kept_by_source: dict[str, int] = {}
    for source, count in quotas.items():
        pool = pools.get(source, [])
        if len(pool) < count:
            raise ValueError(
                f"not enough private guard rows for source={source}: "
                f"requested={count} available={len(pool)}"
            )
        chosen = rng.sample(pool, count)
        for row in chosen:
            out = dict(row)
            out["_stage131_selector"] = selector_name
            out["_stage131_target_count"] = count
            rows.append(out)
        kept_by_source[source] = len(chosen)
    rng.shuffle(rows)
    summary = {
        "target_total": sum(quotas.values()),
        "targets_by_source": dict(sorted(quotas.items())),
        "kept_by_source": dict(sorted(kept_by_source.items())),
        "unique_utt_ids": len({_utt_id(row) for row in rows}),
        "rows": len(rows),
    }
    return rows, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage131 non-public private guard selectors.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean-input", action="append", default=None)
    parser.add_argument("--hard-input", action="append", default=None)
    parser.add_argument("--exclude-lengths", action="append", default=None)
    parser.add_argument("--clean-quotas", default=None)
    parser.add_argument("--hard-quotas", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--selector-name", default="stage131_private_guard")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    clean_output = output_dir / "clean_guard.lengths.jsonl"
    hard_output = output_dir / "hard_guard.lengths.jsonl"
    summary_output = output_dir / "summary.json"
    if not args.overwrite:
        existing = [path for path in (clean_output, hard_output, summary_output) if path.exists()]
        if existing:
            raise FileExistsError(f"outputs already exist; pass --overwrite: {existing}")

    clean_inputs = [Path(path) for path in (args.clean_input or [DEFAULT_STAGE61_LENGTHS, DEFAULT_STAGE22_LENGTHS])]
    hard_inputs = [Path(path) for path in (args.hard_input or [DEFAULT_HARD_LENGTHS])]
    exclude_inputs = [
        Path(path)
        for path in (args.exclude_lengths or [DEFAULT_SELECTOR_LENGTHS, DEFAULT_STAGE129_LENGTHS])
    ]
    clean_quotas = _parse_quotas(args.clean_quotas, DEFAULT_CLEAN_QUOTAS)
    hard_quotas = _parse_quotas(args.hard_quotas, DEFAULT_HARD_QUOTAS)
    exclude_ids, exclude_summary = _load_ids(exclude_inputs)

    clean_pools, clean_pool_summary = _collect_pools(
        inputs=clean_inputs,
        quotas=clean_quotas,
        exclude_ids=exclude_ids,
        split=str(args.split),
        pool_name="clean",
        seed=int(args.seed),
    )
    hard_pools, hard_pool_summary = _collect_pools(
        inputs=hard_inputs,
        quotas=hard_quotas,
        exclude_ids=exclude_ids,
        split=str(args.split),
        pool_name="hard",
        seed=int(args.seed),
    )

    clean_rows, clean_sample_summary = _sample_pools(
        pools=clean_pools,
        quotas=clean_quotas,
        rng=random.Random(int(args.seed) + 17),
        selector_name=str(args.selector_name),
    )
    hard_rows, hard_sample_summary = _sample_pools(
        pools=hard_pools,
        quotas=hard_quotas,
        rng=random.Random(int(args.seed) + 31),
        selector_name=str(args.selector_name),
    )

    _write_jsonl(clean_output, clean_rows)
    _write_jsonl(hard_output, hard_rows)
    summary = {
        "version": 1,
        "selector_name": str(args.selector_name),
        "seed": int(args.seed),
        "split": str(args.split),
        "output_dir": str(output_dir),
        "clean_output": str(clean_output),
        "hard_output": str(hard_output),
        "clean_root": DEFAULT_STAGE22_ROOT,
        "hard_root": DEFAULT_HARD_ROOT,
        "exclude_inputs": [str(path) for path in exclude_inputs],
        "exclude_ids": len(exclude_ids),
        "exclude_new_ids_by_input": exclude_summary,
        "clean_pool": clean_pool_summary,
        "hard_pool": hard_pool_summary,
        "clean_sample": clean_sample_summary,
        "hard_sample": hard_sample_summary,
        "total_rows": len(clean_rows) + len(hard_rows),
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[stage131-private-guard] clean={clean_output} rows={len(clean_rows)} "
        f"hard={hard_output} rows={len(hard_rows)} summary={summary_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
