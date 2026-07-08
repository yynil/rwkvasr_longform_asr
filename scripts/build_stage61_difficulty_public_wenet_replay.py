#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_CURRICULUM_ROOT = (
    "/media/usbhd/training_data/asr/curriculum/"
    "stage22_stage21_soup_clean_repair_mix"
)
DEFAULT_STAGE43_LENGTHS = (
    "/media/usbhd/rwkvasr_runs/stage43_difficulty_replay_20260620/"
    "stage43_difficulty_replay200k.lengths.jsonl"
)
DEFAULT_STAGE47_LENGTHS = (
    "/media/usbhd/rwkvasr_runs/stage47_wenet_guard_repair_20260620/"
    "stage47_original180k_teacher8k_wenet12k.lengths.jsonl"
)
DEFAULT_STAGE59_LENGTHS = (
    DEFAULT_CURRICULUM_ROOT
    + "/stages/stage59_cv_public_guard_wenet_anchor/webdataset_lengths.jsonl"
)
DEFAULT_SELECTOR_LENGTHS = (
    "/media/usbhd/rwkvasr_runs/stage37_ctc_decode_audit_20260620/"
    "stage30_selector256.lengths.jsonl"
)

STAGE43_TARGETS = {
    "commonvoice_en": 40_000,
    "gigaspeech": 20_000,
    "commonvoice_cn": 10_000,
}
STAGE47_TARGETS = {
    "wenetspeech": 40_000,
}
STAGE59_TARGETS = {
    "aishell3": 35_000,
    "librispeech": 35_000,
    "commonvoice_cn": 20_000,
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _utt_id(row: dict[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("id") or "")


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
    if "gigaspeech" in value or "gsxl" in value:
        return "gigaspeech"
    if "wenetspeech" in value or "wenet" in value or "wsl-" in value:
        return "wenetspeech"
    return None


def _infer_source(row: dict[str, Any]) -> str:
    for field in (
        "_stage61_source",
        "_stage59_source",
        "_stage47_source",
        "_stage43_source",
        "_stage39_source",
        "_stage37_selector_source",
        "source_dataset",
        "source",
        "dataset",
    ):
        source = _canonical_source(row.get(field))
        if source:
            return source
    for field in ("shard_name", "key", "audio_member", "json_member", "utt_id"):
        source = _canonical_source(row.get(field))
        if source:
            return source
    return "unknown"


def _load_ids(path: Path) -> set[str]:
    return {_utt_id(row) for row in _iter_jsonl(path) if _utt_id(row)}


def _sample_rows(
    *,
    path: Path,
    targets: dict[str, int],
    component: str,
    rng: random.Random,
    exclude_ids: set[str],
    row_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    filtered_out: Counter[str] = Counter()
    unknown_rows = 0

    for row in _iter_jsonl(path):
        source = _infer_source(row)
        if source not in targets:
            if source == "unknown":
                unknown_rows += 1
            continue
        seen[source] += 1
        utt_id = _utt_id(row)
        if utt_id in exclude_ids:
            excluded[source] += 1
            continue
        if row_filter is not None and not row_filter(row):
            filtered_out[source] += 1
            continue
        enriched = dict(row)
        enriched["_stage61_component"] = component
        enriched["_stage61_source"] = source
        enriched["_stage61_selector"] = "difficulty_public_wenet_original_label"
        pools[source].append(enriched)

    sampled: list[dict[str, Any]] = []
    replacement_fill: Counter[str] = Counter()
    unique_ids: dict[str, int] = {}
    for source, count in targets.items():
        pool = pools.get(source, [])
        if not pool:
            raise ValueError(f"no eligible rows for component={component} source={source}")
        if len(pool) >= count:
            chosen = rng.sample(pool, count)
        else:
            chosen = list(pool)
            fill_count = count - len(pool)
            replacement_fill[source] = fill_count
            chosen.extend(dict(rng.choice(pool)) for _ in range(fill_count))
        for row in chosen:
            row["_stage61_target_count"] = count
        unique_ids[source] = len({_utt_id(row) for row in chosen})
        sampled.extend(chosen)

    summary = {
        "input_path": str(path),
        "component": component,
        "targets": targets,
        "seen_by_source": dict(sorted(seen.items())),
        "eligible_by_source": {source: len(pools.get(source, [])) for source in sorted(targets)},
        "excluded_selector_by_source": dict(sorted(excluded.items())),
        "filtered_out_by_source": dict(sorted(filtered_out.items())),
        "replacement_fill_by_source": dict(sorted(replacement_fill.items())),
        "unique_utt_ids_by_source": dict(sorted(unique_ids.items())),
        "unknown_rows": unknown_rows,
        "actual_by_source": dict(sorted(Counter(str(row["_stage61_source"]) for row in sampled).items())),
    }
    return sampled, summary


def _stage47_wenet_anchor_filter(row: dict[str, Any]) -> bool:
    return (
        str(row.get("_stage47_component") or "") == "wenet_original_anchor"
        and _infer_source(row) == "wenetspeech"
    )


def _build_bucket_manifest(repo_root: Path, root: Path, stage_dir: Path, bucket_width: int) -> None:
    bucket_dir = stage_dir / "webdataset_buckets_audio_text"
    manifest = bucket_dir / "manifest.json"
    command = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(repo_root / "tools/Cargo.toml"),
        "--bin",
        "build_bucket_index",
        "--",
        "--shard-root",
        str(root),
        "--length-index-path",
        str(stage_dir / "webdataset_lengths.jsonl"),
        "--output-dir",
        str(bucket_dir),
        "--manifest-path",
        str(manifest),
        "--bucket-width",
        str(bucket_width),
        "--text-cost-source",
        "auto",
        "--text-cost-weight",
        "4",
        "--json-size-text-offset",
        "256",
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage61 difficulty/public/Wenet original-label replay."
    )
    parser.add_argument("--curriculum-root", default=DEFAULT_CURRICULUM_ROOT)
    parser.add_argument("--stage-name", default="stage61_difficulty_public_wenet_replay")
    parser.add_argument("--stage43-lengths", default=DEFAULT_STAGE43_LENGTHS)
    parser.add_argument("--stage47-lengths", default=DEFAULT_STAGE47_LENGTHS)
    parser.add_argument("--stage59-lengths", default=DEFAULT_STAGE59_LENGTHS)
    parser.add_argument("--selector-lengths", default=DEFAULT_SELECTOR_LENGTHS)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--bucket-width", type=int, default=80)
    parser.add_argument("--skip-buckets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    curriculum_root = Path(args.curriculum_root)
    stage_dir = curriculum_root / "stages" / str(args.stage_name)
    output_path = stage_dir / "webdataset_lengths.jsonl"
    summary_path = stage_dir / "webdataset_lengths.summary.json"

    rng = random.Random(int(args.seed))
    selector_ids = _load_ids(Path(args.selector_lengths))

    stage43_rows, stage43_summary = _sample_rows(
        path=Path(args.stage43_lengths),
        targets=STAGE43_TARGETS,
        component="stage43_difficulty_cv_giga",
        rng=rng,
        exclude_ids=selector_ids,
    )
    stage47_rows, stage47_summary = _sample_rows(
        path=Path(args.stage47_lengths),
        targets=STAGE47_TARGETS,
        component="stage47_wenet_original_anchor",
        rng=rng,
        exclude_ids=selector_ids,
        row_filter=_stage47_wenet_anchor_filter,
    )
    stage59_rows, stage59_summary = _sample_rows(
        path=Path(args.stage59_lengths),
        targets=STAGE59_TARGETS,
        component="stage59_clean_public_guard",
        rng=rng,
        exclude_ids=selector_ids,
    )

    rows = stage43_rows + stage47_rows + stage59_rows
    rng.shuffle(rows)
    _write_jsonl(output_path, rows)

    summary = {
        "version": 1,
        "stage_name": str(args.stage_name),
        "output_path": str(output_path),
        "seed": int(args.seed),
        "selector_lengths": str(args.selector_lengths),
        "selector_ids_excluded": len(selector_ids),
        "component_counts": dict(sorted(Counter(str(row["_stage61_component"]) for row in rows).items())),
        "source_counts": dict(sorted(Counter(str(row["_stage61_source"]) for row in rows).items())),
        "split_counts": dict(sorted(Counter(str(row.get("split") or "train") for row in rows).items())),
        "num_rows": len(rows),
        "unique_utt_ids": len({_utt_id(row) for row in rows}),
        "components": {
            "stage43_difficulty_cv_giga": stage43_summary,
            "stage47_wenet_original_anchor": stage47_summary,
            "stage59_clean_public_guard": stage59_summary,
        },
        "bucket_manifest_path": str(stage_dir / "webdataset_buckets_audio_text" / "manifest.json"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.skip_buckets:
        _build_bucket_manifest(repo_root, curriculum_root, stage_dir, int(args.bucket_width))

    print(f"stage61_length_index={output_path}")
    print(f"stage61_summary={summary_path}")
    print(f"stage61_rows={len(rows)} unique_utt_ids={summary['unique_utt_ids']}")


if __name__ == "__main__":
    main()
