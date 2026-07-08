#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_WENET_ROOT = "/media/usbhd/training_data/asr/wenet-e2e/wenetspeech/webdataset_l_train"
DEFAULT_WENET_LENGTHS = DEFAULT_WENET_ROOT + "/webdataset_lengths.jsonl"
DEFAULT_PUBLIC_MANIFEST = (
    "artifacts/eval_benchmarks/beam_ablation_20260620/"
    "manifests_200_seed20260620/wenetspeech_test_net.jsonl"
)
DEFAULT_SELECTOR_LENGTHS = (
    "/media/usbhd/rwkvasr_runs/stage37_ctc_decode_audit_20260620/"
    "stage30_selector256.lengths.jsonl"
)
DEFAULT_OUTPUT_DIR = "/media/usbhd/rwkvasr_runs/stage75_wenet_public_proxy_20260621"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _utt_id(row: dict[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("id") or "")


def _text_chars(text: Any) -> int:
    return len(str(text or "").strip())


def _text_bin(chars: int) -> str:
    if chars <= 5:
        return "000_005"
    if chars <= 10:
        return "006_010"
    if chars <= 20:
        return "011_020"
    if chars <= 40:
        return "021_040"
    return "041_plus"


def _frame_bin(frames: int) -> str:
    if frames <= 100:
        return "000_100"
    if frames <= 200:
        return "101_200"
    if frames <= 300:
        return "201_300"
    if frames <= 500:
        return "301_500"
    if frames <= 800:
        return "501_800"
    return "801_plus"


def _wav_frames_10ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
    return max(1, round(float(frames) / float(sample_rate) * 100.0))


def _public_targets(manifest_path: Path, target_count: int) -> tuple[Counter[tuple[str, str]], dict[str, Any]]:
    rows = list(_iter_jsonl(manifest_path))
    if not rows:
        raise ValueError(f"empty public manifest: {manifest_path}")
    public_bins: Counter[tuple[str, str]] = Counter()
    frame_counts: list[int] = []
    char_counts: list[int] = []
    for row in rows:
        audio_path = Path(str(row["audio_filepath"]))
        frames = _wav_frames_10ms(audio_path)
        chars = _text_chars(row.get("text"))
        frame_counts.append(frames)
        char_counts.append(chars)
        public_bins[(_frame_bin(frames), _text_bin(chars))] += 1

    scaled: Counter[tuple[str, str]] = Counter()
    residuals: list[tuple[float, tuple[str, str]]] = []
    total = len(rows)
    for key, count in public_bins.items():
        exact = float(count) * float(target_count) / float(total)
        floor = int(exact)
        scaled[key] = floor
        residuals.append((exact - floor, key))
    remaining = target_count - sum(scaled.values())
    for _, key in sorted(residuals, reverse=True)[:remaining]:
        scaled[key] += 1

    summary = {
        "public_manifest": str(manifest_path),
        "public_rows": len(rows),
        "target_count": target_count,
        "public_bin_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(public_bins.items())},
        "target_bin_counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(scaled.items())},
        "public_frame_min": min(frame_counts),
        "public_frame_max": max(frame_counts),
        "public_text_min": min(char_counts),
        "public_text_max": max(char_counts),
    }
    return scaled, summary


def _load_exclude_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            utt_id = _utt_id(row)
            if utt_id:
                ids.add(utt_id)
    return ids


def _is_wenet_row(row: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(row.get(key, "")).lower()
        for key in ("shard_name", "key", "utt_id", "audio_member", "json_member")
    )
    return "wenet" in haystack or "wsl-" in haystack


def _candidate_reservoirs(
    *,
    length_index_path: Path,
    target_bins: Counter[tuple[str, str]],
    exclude_ids: set[str],
    seed: int,
    reservoir_multiplier: int,
    min_per_frame_bin: int,
    progress_interval: int,
    allowed_splits: set[str] | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    target_by_frame: Counter[str] = Counter()
    for (frame_bin, _), count in target_bins.items():
        target_by_frame[frame_bin] += count
    reservoir_sizes = {
        frame_bin: max(int(count) * int(reservoir_multiplier), int(min_per_frame_bin))
        for frame_bin, count in target_by_frame.items()
    }
    rngs = {
        frame_bin: random.Random(int(seed) + index * 104729)
        for index, frame_bin in enumerate(sorted(reservoir_sizes))
    }
    eligible_seen: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    kept: dict[str, list[dict[str, Any]]] = {frame_bin: [] for frame_bin in reservoir_sizes}
    processed = 0
    wenet_rows = 0
    split_filtered_rows = 0

    for row in _iter_jsonl(length_index_path):
        processed += 1
        if progress_interval > 0 and processed % progress_interval == 0:
            print(
                f"stage75 reservoir processed={processed:,} wenet={wenet_rows:,}",
                file=sys.stderr,
                flush=True,
            )
        if not _is_wenet_row(row):
            continue
        wenet_rows += 1
        if allowed_splits is not None:
            split = str(row.get("split") or "").strip().lower()
            if split not in allowed_splits:
                split_filtered_rows += 1
                continue
        utt_id = _utt_id(row)
        frames = int(row.get("num_frames") or 0)
        frame_bin = _frame_bin(frames)
        if frame_bin not in reservoir_sizes:
            continue
        if utt_id in exclude_ids:
            excluded[frame_bin] += 1
            continue
        eligible_seen[frame_bin] += 1
        size = reservoir_sizes[frame_bin]
        rows = kept[frame_bin]
        rng = rngs[frame_bin]
        if len(rows) < size:
            rows.append(dict(row))
            continue
        index = rng.randrange(eligible_seen[frame_bin])
        if index < size:
            rows[index] = dict(row)

    summary = {
        "length_index_path": str(length_index_path),
        "processed_rows": processed,
        "wenet_rows": wenet_rows,
        "allowed_splits": None if allowed_splits is None else sorted(allowed_splits),
        "split_filtered_rows": split_filtered_rows,
        "reservoir_sizes": reservoir_sizes,
        "eligible_seen_by_frame_bin": dict(sorted(eligible_seen.items())),
        "excluded_by_frame_bin": dict(sorted(excluded.items())),
        "kept_by_frame_bin": {key: len(value) for key, value in sorted(kept.items())},
    }
    return kept, summary


def _read_candidate_texts(root: Path, candidates_by_frame: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_bin, candidates in sorted(candidates_by_frame.items()):
        by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_shard[str(row["shard_name"])].append(row)
        for shard_name, shard_rows in sorted(by_shard.items()):
            shard_path = root / shard_name
            with shard_path.open("rb") as handle:
                for row in shard_rows:
                    handle.seek(int(row["json_offset"]))
                    payload = handle.read(int(row["json_size"]))
                    meta = json.loads(payload)
                    text = str(meta.get("text") or "")
                    enriched = dict(row)
                    enriched["_stage75_proxy_text"] = text
                    enriched["_stage75_text_chars"] = _text_chars(text)
                    enriched["_stage75_frame_bin"] = frame_bin
                    enriched["_stage75_text_bin"] = _text_bin(enriched["_stage75_text_chars"])
                    enriched["_stage75_source"] = "wenetspeech"
                    rows.append(enriched)
    return rows


def _select_rows(
    *,
    candidates: list[dict[str, Any]],
    target_bins: Counter[tuple[str, str]],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        pools[(str(row["_stage75_frame_bin"]), str(row["_stage75_text_bin"]))].append(row)
    for rows in pools.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}
    exact_counts: Counter[str] = Counter()
    selected_ids: set[str] = set()
    for key, target in sorted(target_bins.items()):
        pool = pools.get(key, [])
        chosen = [row for row in pool if _utt_id(row) not in selected_ids][:target]
        for row in chosen:
            row["_stage75_match_type"] = "exact_bin"
            selected.append(row)
            selected_ids.add(_utt_id(row))
        exact_counts[f"{key[0]}|{key[1]}"] = len(chosen)
        if len(chosen) < target:
            shortfalls[f"{key[0]}|{key[1]}"] = target - len(chosen)

    remaining_target = sum(target_bins.values()) - len(selected)
    fallback_count = 0
    if remaining_target > 0:
        remaining = [row for row in candidates if _utt_id(row) not in selected_ids]
        rng.shuffle(remaining)
        for row in remaining[:remaining_target]:
            row["_stage75_match_type"] = "fallback"
            selected.append(row)
            selected_ids.add(_utt_id(row))
            fallback_count += 1

    rng.shuffle(selected)
    actual_bins = Counter(
        f"{row['_stage75_frame_bin']}|{row['_stage75_text_bin']}"
        for row in selected
    )
    summary = {
        "candidate_rows_with_text": len(candidates),
        "exact_selected_by_bin": dict(sorted(exact_counts.items())),
        "shortfalls_by_bin": shortfalls,
        "fallback_selected": fallback_count,
        "actual_bin_counts": dict(sorted(actual_bins.items())),
        "selected_unique_utt_ids": len({_utt_id(row) for row in selected}),
    }
    return selected, summary


def _write_outputs(output_path: Path, summary_path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            serializable = dict(row)
            handle.write(json.dumps(serializable, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage75 Wenet public-proxy selector.")
    parser.add_argument("--wenet-root", default=DEFAULT_WENET_ROOT)
    parser.add_argument("--wenet-length-index", default=DEFAULT_WENET_LENGTHS)
    parser.add_argument("--public-manifest", default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument("--selector-lengths", default=DEFAULT_SELECTOR_LENGTHS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="stage75_wenet_public_proxy512.lengths.jsonl")
    parser.add_argument("--target-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--reservoir-multiplier", type=int, default=80)
    parser.add_argument("--min-per-frame-bin", type=int, default=2000)
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    parser.add_argument(
        "--allowed-split",
        action="append",
        default=None,
        help="Restrict candidates to this length-index split. May be provided multiple times.",
    )
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        default=None,
        help="Additional JSONL files whose utt_id/id values should be excluded.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_path = output_dir / str(args.output_name)
    summary_path = output_path.with_suffix(".summary.json")
    allowed_splits = (
        {str(value).strip().lower() for value in args.allowed_split if str(value).strip()}
        if args.allowed_split is not None
        else None
    )

    target_bins, public_summary = _public_targets(Path(args.public_manifest), int(args.target_count))
    exclude_paths = [Path(args.public_manifest), Path(args.selector_lengths)]
    if args.exclude_jsonl is not None:
        exclude_paths.extend(Path(value) for value in args.exclude_jsonl)
    exclude_ids = _load_exclude_ids(exclude_paths)
    candidates_by_frame, reservoir_summary = _candidate_reservoirs(
        length_index_path=Path(args.wenet_length_index),
        target_bins=target_bins,
        exclude_ids=exclude_ids,
        seed=int(args.seed),
        reservoir_multiplier=int(args.reservoir_multiplier),
        min_per_frame_bin=int(args.min_per_frame_bin),
        progress_interval=int(args.progress_interval),
        allowed_splits=allowed_splits,
    )
    candidates = _read_candidate_texts(Path(args.wenet_root), candidates_by_frame)
    selected, selection_summary = _select_rows(candidates=candidates, target_bins=target_bins, seed=int(args.seed) + 17)
    if len(selected) != int(args.target_count):
        raise RuntimeError(f"selected {len(selected)} rows, expected {args.target_count}")

    for index, row in enumerate(selected):
        row["_stage75_proxy_name"] = output_path.stem
        row["_stage75_proxy_index"] = index
        row["_stage75_seed"] = int(args.seed)

    summary = {
        "version": 1,
        "stage_name": "stage75_wenet_public_proxy",
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "seed": int(args.seed),
        "target_count": int(args.target_count),
        "extra_exclude_jsonl": [] if args.exclude_jsonl is None else [str(value) for value in args.exclude_jsonl],
        "exclude_ids": len(exclude_ids),
        "public": public_summary,
        "reservoir": reservoir_summary,
        "selection": selection_summary,
    }
    _write_outputs(output_path, summary_path, selected, summary)
    print(f"stage75_wenet_public_proxy output={output_path} rows={len(selected)} summary={summary_path}")


if __name__ == "__main__":
    main()
