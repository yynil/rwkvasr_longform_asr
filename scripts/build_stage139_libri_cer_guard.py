#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CLEAN_ROOT = "/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset"
DEFAULT_ROOT = "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix"
DEFAULT_CLEAN_LENGTHS = DEFAULT_CLEAN_ROOT + "/webdataset_lengths.jsonl"
DEFAULT_STAGE22_LENGTHS = (
    DEFAULT_ROOT + "/stages/stage22_public90_hard10_from_stage21_soup/webdataset_lengths.jsonl"
)
DEFAULT_STAGE61_LENGTHS = (
    DEFAULT_ROOT + "/stages/stage61_difficulty_public_wenet_replay/webdataset_lengths.jsonl"
)
DEFAULT_OUTPUT_DIR = "/media/usbhd/rwkvasr_runs/stage139_libri_cer_guard_20260623"
DEFAULT_EXCLUDES = (
    "/media/usbhd/rwkvasr_runs/stage37_ctc_decode_audit_20260620/stage30_selector256.lengths.jsonl",
    "artifacts/eval_benchmarks/beam_ablation_20260620/manifests_200_seed20260620/*.jsonl",
    DEFAULT_ROOT + "/stages/stage104_stage95_libri_teacher_guard_repair/webdataset_lengths.jsonl",
    DEFAULT_ROOT + "/stages/stage108_stage104_expanded_libri_teacher_guard_repair/webdataset_lengths.jsonl",
    DEFAULT_ROOT + "/stages/stage110_stage104_multi_teacher_guard_repair/webdataset_lengths.jsonl",
    DEFAULT_ROOT + "/stages/stage119_stage110_original_guard_only/webdataset_lengths.jsonl",
    DEFAULT_ROOT + "/stages/stage125_stage110_cv_libri_weighted_original_guard/webdataset_lengths.jsonl",
    DEFAULT_ROOT + "/stages/stage129_stage110_expanded_wenet_teacher_balanced_guard/webdataset_lengths.jsonl",
    DEFAULT_ROOT + "/stages/stage134_stage110_giga_teacher_balanced_guard/webdataset_lengths.jsonl",
    DEFAULT_ROOT + "/stages/stage135_stage110_giga_micro_teacher_guard/webdataset_lengths.jsonl",
    "/media/usbhd/rwkvasr_runs/stage131_private_guard_20260622/clean_guard.lengths.jsonl",
    "/media/usbhd/rwkvasr_runs/stage131_private_guard_20260622/hard_guard.lengths.jsonl",
    "/media/usbhd/rwkvasr_runs/stage108_expanded_libri_teacher_20260621/stage108_libri_teacher_accepted.lengths.jsonl",
)

SOURCE_FIELDS = (
    "_stage139_source",
    "_stage131_source",
    "_stage129_source",
    "_stage125_source",
    "_stage119_source",
    "_stage110_source",
    "_stage108_source",
    "_stage104_source",
    "_stage98_source",
    "_stage82_source",
    "_stage61_source",
    "_stage59_source",
    "_stage43_source",
    "_stage39_source",
    "source_dataset",
    "source",
    "dataset",
)
NAME_FIELDS = ("shard_name", "key", "audio_member", "json_member", "utt_id", "id")
DEFAULT_QUOTAS = {"short": 384, "medium": 384, "long": 256}


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


def _expand_paths(raw_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_paths:
        matches = sorted(glob.glob(raw))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(raw))
    return paths


def _load_ids(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        before = len(ids)
        for row in _iter_jsonl(path):
            utt_id = _utt_id(row)
            if utt_id:
                ids.add(utt_id)
        counts[str(path)] = len(ids) - before
    return ids, counts


def _parse_quotas(raw: str | None) -> dict[str, int]:
    if raw is None or raw.strip() == "":
        return dict(DEFAULT_QUOTAS)
    quotas: dict[str, int] = {}
    for item in raw.split(","):
        bucket, count = item.split(":", 1)
        bucket = bucket.strip()
        if bucket not in DEFAULT_QUOTAS:
            raise ValueError(f"unknown quota bucket: {bucket}")
        quotas[bucket] = int(count.strip())
    missing = set(DEFAULT_QUOTAS) - set(quotas)
    if missing:
        raise ValueError(f"missing quota buckets: {sorted(missing)}")
    return quotas


def _length_bucket(row: dict[str, Any]) -> tuple[str | None, int, int]:
    frames = int(row.get("num_frames", 0) or 0)
    chars = int(row.get("num_text_chars", 0) or row.get("text_bytes", 0) or 0)
    if chars <= 0 or frames <= 0:
        return None, chars, frames
    if chars <= 80:
        return "short", chars, frames
    if chars <= 180:
        return "medium", chars, frames
    return "long", chars, frames


def _collect_rows(
    *,
    inputs: list[Path],
    exclude_ids: set[str],
    split: str,
    max_frames: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in DEFAULT_QUOTAS}
    input_counts: dict[str, Counter[str]] = {}
    excluded = Counter()
    bucket_seen = Counter()
    missing_length = 0
    too_long = 0
    unknown_source = 0
    duplicate_rows = 0
    seen_ids: set[str] = set()

    for path in inputs:
        counts: Counter[str] = Counter()
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in _iter_jsonl(path):
            if str(row.get("split") or "train") != split:
                continue
            source = _infer_source(row)
            counts[source] += 1
            if source != "librispeech":
                if source == "unknown":
                    unknown_source += 1
                continue
            utt_id = _utt_id(row)
            if not utt_id:
                continue
            if utt_id in seen_ids:
                duplicate_rows += 1
                continue
            seen_ids.add(utt_id)
            if utt_id in exclude_ids:
                excluded[source] += 1
                continue
            bucket, chars, frames = _length_bucket(row)
            if bucket is None:
                missing_length += 1
                continue
            if frames > max_frames:
                too_long += 1
                continue
            out = dict(row)
            out["_stage139_component"] = "libri_cer_guard"
            out["_stage139_source"] = "librispeech"
            out["_stage139_bucket"] = bucket
            out["_stage139_text_chars"] = chars
            out["_stage139_num_frames"] = frames
            out["_stage139_seed"] = seed
            pools[bucket].append(out)
            bucket_seen[bucket] += 1
        input_counts[str(path)] = counts

    for rows in pools.values():
        rng.shuffle(rows)

    summary = {
        "input_counts": {path: dict(sorted(counter.items())) for path, counter in sorted(input_counts.items())},
        "excluded_by_source": dict(sorted(excluded.items())),
        "bucket_available": dict(sorted(bucket_seen.items())),
        "missing_length_rows": missing_length,
        "too_long_rows": too_long,
        "unknown_source_rows": unknown_source,
        "duplicate_rows": duplicate_rows,
    }
    return pools, summary


def _select_rows(
    *,
    pools: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    seed: int,
    selector_name: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rng = random.Random(seed + 139)
    selected: dict[str, list[dict[str, Any]]] = {}
    for bucket, count in quotas.items():
        rows = pools.get(bucket, [])
        if len(rows) < count:
            raise ValueError(f"not enough rows for bucket={bucket}: requested={count} available={len(rows)}")
        chosen = rng.sample(rows, count)
        for row in chosen:
            row["_stage139_selector"] = selector_name
            row["_stage139_target_count"] = count
        selected[bucket] = chosen
    summary = {
        "targets_by_bucket": dict(sorted(quotas.items())),
        "selected_by_bucket": {bucket: len(rows) for bucket, rows in sorted(selected.items())},
        "unique_utt_ids": len({_utt_id(row) for rows in selected.values() for row in rows}),
    }
    return selected, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a non-public LibriSpeech CER guard.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input", action="append", default=None)
    parser.add_argument("--exclude-jsonl", action="append", default=None)
    parser.add_argument("--quotas", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--selector-name", default="stage139_libri_cer_guard")
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    outputs = [
        output_dir / "libri_short.lengths.jsonl",
        output_dir / "libri_medium.lengths.jsonl",
        output_dir / "libri_long.lengths.jsonl",
        output_dir / "libri_cer_guard.lengths.jsonl",
        output_dir / "summary.json",
    ]
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"outputs already exist; pass --overwrite: {existing}")

    inputs = [Path(path) for path in (args.input or [DEFAULT_CLEAN_LENGTHS])]
    excludes = _expand_paths(list(args.exclude_jsonl or DEFAULT_EXCLUDES))
    quotas = _parse_quotas(args.quotas)
    exclude_ids, exclude_summary = _load_ids(excludes)
    pools, pool_summary = _collect_rows(
        inputs=inputs,
        exclude_ids=exclude_ids,
        split=str(args.split),
        max_frames=int(args.max_frames),
        seed=int(args.seed),
    )
    selected, sample_summary = _select_rows(
        pools=pools,
        quotas=quotas,
        seed=int(args.seed),
        selector_name=str(args.selector_name),
    )

    combined: list[dict[str, Any]] = []
    for bucket in ("short", "medium", "long"):
        rows = selected[bucket]
        combined.extend(rows)
        _write_jsonl(output_dir / f"libri_{bucket}.lengths.jsonl", rows)
    random.Random(int(args.seed) + 271).shuffle(combined)
    _write_jsonl(output_dir / "libri_cer_guard.lengths.jsonl", combined)

    summary = {
        "version": 1,
        "selector_name": str(args.selector_name),
        "seed": int(args.seed),
        "split": str(args.split),
        "output_dir": str(output_dir),
        "inputs": [str(path) for path in inputs],
        "exclude_inputs": [str(path) for path in excludes],
        "exclude_ids": len(exclude_ids),
        "exclude_new_ids_by_input": exclude_summary,
        "quotas": quotas,
        "pool": pool_summary,
        "sample": sample_summary,
        "total_rows": len(combined),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[stage139-libri-cer] output={output_dir} rows={len(combined)} "
        f"short={len(selected['short'])} medium={len(selected['medium'])} long={len(selected['long'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
