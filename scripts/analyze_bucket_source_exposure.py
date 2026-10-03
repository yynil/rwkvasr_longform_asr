from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from rwkvasr.data import load_webdataset_bucket_manifest
from rwkvasr.data.webdataset_bucketed import (
    _SourceInterleavedBucketEntryStream,
    compute_bucket_local_batch_size,
)
from rwkvasr.data.webdataset_lengths import WebDatasetLengthEntry


def _entry_source(entry: WebDatasetLengthEntry, source_field: str) -> str:
    raw = entry.raw or {}
    value = str(raw.get(source_field) or "unknown").strip()
    return value or "unknown"


def _entry_identity(entry: WebDatasetLengthEntry) -> tuple[Any, ...]:
    return (
        entry.shard_name,
        entry.key,
        entry.audio_member,
        entry.audio_offset,
        entry.audio_size,
    )


def analyze_source_exposure(
    manifest_path: Path,
    *,
    max_steps: int,
    checkpoints: tuple[int, ...],
    batch_size: int,
    world_size: int,
    frame_budget: int,
    seed: int,
    epoch: int,
    source_field: str,
) -> list[dict[str, Any]]:
    manifest = load_webdataset_bucket_manifest(manifest_path)
    buckets = {bucket.bucket_id: bucket for bucket in manifest.splits.get("train", ())}
    streams = {
        bucket_id: _SourceInterleavedBucketEntryStream(manifest_path, bucket, epoch=epoch)
        for bucket_id, bucket in buckets.items()
    }
    schedule: list[int] = []
    global_batch_sizes: dict[int, int] = {}
    for bucket_id, bucket in buckets.items():
        local_batch_size = compute_bucket_local_batch_size(
            bucket_id=bucket_id,
            bucket_width=manifest.bucket_width,
            max_local_batch_size=batch_size,
            frame_budget=frame_budget,
        )
        global_batch_size = local_batch_size * world_size
        global_batch_sizes[bucket_id] = global_batch_size
        schedule.extend([bucket_id] * (bucket.num_samples // global_batch_size))
    random.Random(seed + epoch).shuffle(schedule)

    effective_max_steps = len(schedule) if max_steps <= 0 else min(max_steps, len(schedule))
    requested_checkpoints = sorted(
        {step for step in checkpoints if 0 < step <= effective_max_steps} | {effective_max_steps}
    )
    source_samples: Counter[str] = Counter()
    source_frames: Counter[str] = Counter()
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0
    results: list[dict[str, Any]] = []

    for step, bucket_id in enumerate(schedule[:effective_max_steps], start=1):
        entries = streams[bucket_id].take(global_batch_sizes[bucket_id])
        if len(entries) != global_batch_sizes[bucket_id]:
            raise RuntimeError(
                f"bucket {bucket_id} returned {len(entries)} rows; "
                f"expected {global_batch_sizes[bucket_id]} at step {step}"
            )
        for entry in entries:
            identity = _entry_identity(entry)
            if identity in seen:
                duplicates += 1
            seen.add(identity)
            source = _entry_source(entry, source_field)
            source_samples[source] += 1
            source_frames[source] += entry.num_frames

        if step in requested_checkpoints:
            total_samples = sum(source_samples.values())
            total_frames = sum(source_frames.values())
            results.append(
                {
                    "step": step,
                    "samples": total_samples,
                    "unique_samples": len(seen),
                    "duplicates": duplicates,
                    "hours": total_frames / 360_000.0,
                    "sources": {
                        source: {
                            "samples": source_samples[source],
                            "sample_percent": 100.0 * source_samples[source] / total_samples,
                            "hours": source_frames[source] / 360_000.0,
                            "hour_percent": 100.0 * source_frames[source] / total_frames,
                        }
                        for source in sorted(source_samples)
                    },
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a source-interleaved bucket schedule.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=(500, 1_000, 2_000))
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--frame-budget", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--source-field", default="_stage179_source")
    args = parser.parse_args()

    result = analyze_source_exposure(
        args.manifest,
        max_steps=int(args.max_steps),
        checkpoints=tuple(int(step) for step in args.checkpoints),
        batch_size=int(args.batch_size),
        world_size=int(args.world_size),
        frame_budget=int(args.frame_budget),
        seed=int(args.seed),
        epoch=int(args.epoch),
        source_field=str(args.source_field),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
