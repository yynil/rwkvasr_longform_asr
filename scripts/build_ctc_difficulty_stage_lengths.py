#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _parse_tiers(raw_tiers: list[str]) -> tuple[str, ...]:
    tiers: list[str] = []
    for raw in raw_tiers:
        for item in raw.split(","):
            tier = item.strip()
            if tier and tier not in tiers:
                tiers.append(tier)
    if not tiers:
        raise ValueError("At least one difficulty tier is required.")
    return tuple(tiers)


def _iter_part_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _iter_selected_rows(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    tiers: tuple[str, ...],
) -> Iterable[dict[str, Any]]:
    difficulty_root = manifest_path.parent
    for split in ("train", "eval"):
        split_payload = manifest.get("splits", {}).get(split, {})
        tier_payloads = split_payload.get("tiers", {})
        for tier in tiers:
            tier_payload = tier_payloads.get(tier)
            if tier_payload is None:
                continue
            for part in tier_payload.get("parts", []):
                part_path = difficulty_root / str(part["path"])
                yield from _iter_part_rows(part_path)


def _write_summary(
    path: Path,
    *,
    stage_name: str,
    difficulty_manifest_path: Path,
    output_dir: Path,
    tiers: tuple[str, ...],
    count: int,
    counts: dict[str, Counter[Any]],
) -> None:
    summary = {
        "version": 1,
        "stage_name": stage_name,
        "difficulty_manifest_path": str(difficulty_manifest_path),
        "output_dir": str(output_dir),
        "selected_tiers": list(tiers),
        "num_samples": int(count),
        "counts": {
            "by_split": dict(sorted(counts["split"].items())),
            "by_tier": dict(sorted(counts["tier"].items())),
            "by_source": dict(sorted(counts["source"].items())),
            "by_language": dict(sorted(counts["language"].items())),
            "by_split_tier": {
                f"{split}/{tier}": value
                for (split, tier), value in sorted(counts["split_tier"].items())
            },
            "by_source_tier": {
                f"{source}/{tier}": value
                for (source, tier), value in sorted(counts["source_tier"].items())
            },
            "by_language_tier": {
                f"{language}/{tier}": value
                for (language, tier), value in sorted(counts["language_tier"].items())
            },
        },
        "length_index_path": str(output_dir / "webdataset_lengths.jsonl"),
        "length_summary_path": str(output_dir / "webdataset_lengths.summary.json"),
        "bucket_manifest_path": str(output_dir / "webdataset_buckets_audio_text" / "manifest.json"),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stage-specific WebDataset length index from CTC difficulty buckets."
    )
    parser.add_argument("--difficulty-manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-name", default=None)
    parser.add_argument("--tiers", nargs="+", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    difficulty_manifest_path = Path(args.difficulty_manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers = _parse_tiers(args.tiers)
    stage_name = str(args.stage_name or "_".join(tiers))
    manifest = json.loads(difficulty_manifest_path.read_text(encoding="utf-8"))

    available_tiers = {
        tier
        for split_payload in manifest.get("splits", {}).values()
        for tier in split_payload.get("tiers", {})
    }
    missing = [tier for tier in tiers if tier not in available_tiers]
    if missing:
        raise ValueError(f"Missing tier(s) in difficulty manifest: {', '.join(missing)}")

    length_index_path = output_dir / "webdataset_lengths.jsonl"
    summary_path = output_dir / "webdataset_lengths.summary.json"
    stage_manifest_path = output_dir / "difficulty_stage_manifest.json"
    counts: dict[str, Counter[Any]] = defaultdict(Counter)
    count = 0

    with length_index_path.open("w", encoding="utf-8") as output:
        for row in _iter_selected_rows(
            manifest_path=difficulty_manifest_path,
            manifest=manifest,
            tiers=tiers,
        ):
            if args.max_samples and count >= int(args.max_samples):
                break
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            split = str(row.get("split") or "unknown")
            tier = str(row.get("difficulty_tier") or "unknown")
            source = str(row.get("source_dataset") or "unknown")
            language = str(row.get("language") or "unknown")
            counts["split"][split] += 1
            counts["tier"][tier] += 1
            counts["source"][source] += 1
            counts["language"][language] += 1
            counts["split_tier"][(split, tier)] += 1
            counts["source_tier"][(source, tier)] += 1
            counts["language_tier"][(language, tier)] += 1
            count += 1

    _write_summary(
        summary_path,
        stage_name=stage_name,
        difficulty_manifest_path=difficulty_manifest_path,
        output_dir=output_dir,
        tiers=tiers,
        count=count,
        counts=counts,
    )
    stage_manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "stage_name": stage_name,
                "difficulty_manifest_path": str(difficulty_manifest_path),
                "selected_tiers": list(tiers),
                "length_index_path": str(length_index_path),
                "length_summary_path": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _log(
        f"difficulty stage lengths complete stage={stage_name} "
        f"samples={count} lengths={length_index_path} summary={summary_path}"
    )


if __name__ == "__main__":
    main()
