#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
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
DEFAULT_STAGE39_DIR = "/media/usbhd/rwkvasr_runs/stage39_large_teacher_repair_20260620"
DEFAULT_SELECTOR_LENGTHS = (
    "/media/usbhd/rwkvasr_runs/stage37_ctc_decode_audit_20260620/"
    "stage30_selector256.lengths.jsonl"
)
DEFAULT_STAGE_NAME = "stage104_stage95_libri_teacher_guard_repair"
DEFAULT_ORIGINAL_QUOTAS = {
    "librispeech": 45_000,
    "commonvoice_en": 35_000,
    "wenetspeech": 35_000,
    "aishell3": 20_000,
    "commonvoice_cn": 10_000,
    "gigaspeech": 7_000,
}
TEACHER_SOURCE = "librispeech"

SOURCE_FIELDS = (
    "_stage108_source",
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


def _audit_key(prefix: str, name: str) -> str:
    return f"_{prefix}_{name}"


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


def _load_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {_utt_id(row) for row in _iter_jsonl(path) if _utt_id(row)}


def _parse_quotas(raw: str | None) -> dict[str, int]:
    if raw is None or raw.strip() == "":
        return dict(DEFAULT_ORIGINAL_QUOTAS)
    quotas: dict[str, int] = {}
    for item in raw.split(","):
        source, count = item.split(":", 1)
        source = source.strip()
        if source not in DEFAULT_ORIGINAL_QUOTAS:
            raise ValueError(f"unknown source in quota: {source}")
        quotas[source] = int(count.strip())
    missing = set(DEFAULT_ORIGINAL_QUOTAS) - set(quotas)
    if missing:
        raise ValueError(f"missing quotas for sources: {sorted(missing)}")
    return quotas


def _parse_teacher_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(getattr(args, "no_teacher", False)):
        return []
    raw_specs = getattr(args, "teacher_spec", None) or []
    specs: list[dict[str, Any]] = []
    if not raw_specs:
        raw_specs = [
            f"{TEACHER_SOURCE}:{int(args.teacher_samples)}:{args.teacher_cache}:{args.teacher_lengths}"
        ]
    for raw in raw_specs:
        parts = str(raw).split(":", 3)
        if len(parts) != 4:
            raise ValueError(
                "--teacher-spec must be source:samples:cache:lengths, "
                f"got {raw!r}"
            )
        raw_source, raw_samples, raw_cache, raw_lengths = parts
        source = _canonical_source(raw_source.strip())
        if source is None:
            raise ValueError(f"unknown teacher source in spec: {raw_source!r}")
        samples = int(raw_samples)
        if samples <= 0:
            raise ValueError(f"teacher samples must be positive for source={source}")
        specs.append(
            {
                "source": source,
                "samples": samples,
                "cache_path": Path(raw_cache),
                "lengths_path": Path(raw_lengths),
            }
        )
    seen_sources = [str(spec["source"]) for spec in specs]
    duplicates = sorted(source for source, count in Counter(seen_sources).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate teacher specs for sources: {duplicates}")
    return specs


def _load_lengths_by_utt(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            utt_id = _utt_id(row)
            if utt_id and utt_id not in rows:
                rows[utt_id] = row
    return rows


def _weighted_sample_with_replacement(
    rng: random.Random,
    rows: list[dict[str, Any]],
    count: int,
    weight_field: str,
) -> list[dict[str, Any]]:
    weights = [max(float(row.get(weight_field, 0.0) or 0.0), 1e-6) for row in rows]
    return [dict(row) for row in rng.choices(rows, weights=weights, k=count)]


def _build_teacher_component(
    *,
    teacher_specs: list[dict[str, Any]],
    rng: random.Random,
    audit_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], set[str]]:
    sampled: list[dict[str, Any]] = []
    sampled_cache_rows: list[dict[str, Any]] = []
    accepted_ids: set[str] = set()
    by_source: dict[str, Any] = {}

    for spec in teacher_specs:
        source = str(spec["source"])
        target_total = int(spec["samples"])
        teacher_cache_path = Path(spec["cache_path"])
        teacher_lengths_path = Path(spec["lengths_path"])
        lengths_by_utt = _load_lengths_by_utt([teacher_lengths_path])
        cache_by_utt: dict[str, dict[str, Any]] = {}
        teacher_rows: list[dict[str, Any]] = []
        accepted_source_ids: set[str] = set()
        skipped_by_source: Counter[str] = Counter()
        missing_lengths: list[str] = []

        for cache_row in _iter_jsonl(teacher_cache_path):
            utt_id = _utt_id(cache_row)
            if not utt_id:
                continue
            accepted_ids.add(utt_id)
            row_source = _canonical_source(cache_row.get("source")) or "unknown"
            if row_source == source:
                accepted_source_ids.add(utt_id)
            else:
                skipped_by_source[row_source] += 1
                continue
            length_row = lengths_by_utt.get(utt_id)
            if length_row is None:
                missing_lengths.append(utt_id)
                continue
            cache_out = dict(cache_row)
            cache_out["source"] = source
            cache_by_utt[utt_id] = cache_out
            enriched = dict(length_row)
            component = (
                "libri_teacher_substitution_repair"
                if source == TEACHER_SOURCE
                else f"{source}_teacher_substitution_repair"
            )
            enriched[_audit_key(audit_prefix, "component")] = component
            enriched[_audit_key(audit_prefix, "source")] = source
            enriched[_audit_key(audit_prefix, "teacher_override")] = True
            enriched[_audit_key(audit_prefix, "primary_improvement")] = float(
                cache_row.get("primary_improvement", 0.0) or 0.0
            )
            enriched[_audit_key(audit_prefix, "student_error")] = float(
                cache_row.get("student_wer", cache_row.get("student_cer", 0.0)) or 0.0
            )
            enriched[_audit_key(audit_prefix, "teacher_error")] = float(
                cache_row.get("teacher_wer", cache_row.get("teacher_cer", 0.0)) or 0.0
            )
            teacher_rows.append(enriched)

        if not teacher_rows:
            raise ValueError(f"no teacher rows available for source={source}")

        source_sampled = _weighted_sample_with_replacement(
            rng,
            teacher_rows,
            target_total,
            _audit_key(audit_prefix, "primary_improvement"),
        )
        sampled.extend(source_sampled)
        sampled_ids = {_utt_id(row) for row in source_sampled}
        source_cache_rows = [
            cache_by_utt[utt_id] for utt_id in sorted(sampled_ids) if utt_id in cache_by_utt
        ]
        sampled_cache_rows.extend(source_cache_rows)
        bad_sources = sorted(
            {
                _canonical_source(row.get("source")) or "unknown"
                for row in source_cache_rows
                if (_canonical_source(row.get("source")) or "unknown") != source
            }
        )
        if bad_sources:
            raise ValueError(
                f"sampled teacher cache for source={source} contains other sources: {bad_sources}"
            )

        by_source[source] = {
            "target_total": target_total,
            "available_teacher_rows": len(teacher_rows),
            "actual_samples": len(source_sampled),
            "unique_sampled_teacher_ids": len(sampled_ids),
            "sampled_cache_rows": len(source_cache_rows),
            "accepted_teacher_ids_total": len(accepted_source_ids),
            "skipped_cache_rows_by_source": dict(sorted(skipped_by_source.items())),
            "missing_length_count": len(missing_lengths),
            "missing_length_examples": missing_lengths[:20],
            "teacher_cache": str(teacher_cache_path),
            "teacher_lengths": str(teacher_lengths_path),
        }

    sampled_cache_sources = sorted(
        {_canonical_source(row.get("source")) or "unknown" for row in sampled_cache_rows}
    )
    expected_sources = sorted(str(spec["source"]) for spec in teacher_specs)
    unexpected_sources = sorted(set(sampled_cache_sources) - set(expected_sources))
    if unexpected_sources:
        raise ValueError(f"sampled teacher cache has unexpected sources: {unexpected_sources}")

    summary = {
        "target_total": sum(int(spec["samples"]) for spec in teacher_specs),
        "targets_by_source": {
            str(spec["source"]): int(spec["samples"]) for spec in teacher_specs
        },
        "available_teacher_rows": sum(
            int(source_summary["available_teacher_rows"])
            for source_summary in by_source.values()
        ),
        "actual_samples": len(sampled),
        "actual_samples_by_source": dict(
            sorted(Counter(str(row[_audit_key(audit_prefix, "source")]) for row in sampled).items())
        ),
        "unique_sampled_teacher_ids": len({_utt_id(row) for row in sampled}),
        "sampled_cache_rows": len(sampled_cache_rows),
        "sampled_cache_sources": sampled_cache_sources,
        "accepted_teacher_ids_total": len(accepted_ids),
        "by_source": by_source,
    }
    return sampled, sampled_cache_rows, summary, accepted_ids


def _load_original_pools(
    *,
    inputs: list[tuple[str, Path]],
    targets: dict[str, int],
    exclude_ids: set[str],
    audit_prefix: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_keys: set[str] = set()
    counts_by_input: dict[str, Counter[str]] = {}
    excluded: Counter[str] = Counter()
    duplicates = 0
    unknown_rows = 0

    for label, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        input_counts: Counter[str] = Counter()
        for row in _iter_jsonl(path):
            source = _infer_source(row)
            input_counts[source] += 1
            if source not in targets:
                if source == "unknown":
                    unknown_rows += 1
                continue
            utt_id = _utt_id(row)
            if utt_id in exclude_ids:
                excluded[source] += 1
                continue
            dedupe_key = f"{source}:{utt_id}" if utt_id else (
                f"{source}:{row.get('shard_name')}:{row.get('key')}"
            )
            if dedupe_key in seen_keys:
                duplicates += 1
                continue
            seen_keys.add(dedupe_key)
            enriched = dict(row)
            enriched[_audit_key(audit_prefix, "component")] = "original_label_guard"
            enriched[_audit_key(audit_prefix, "source")] = source
            enriched[_audit_key(audit_prefix, "teacher_override")] = False
            enriched[_audit_key(audit_prefix, "input")] = label
            pools[source].append(enriched)
        counts_by_input[label] = input_counts

    summary = {
        "available_by_source": {source: len(pools.get(source, [])) for source in sorted(targets)},
        "counts_by_input": {
            label: dict(sorted(counter.items())) for label, counter in sorted(counts_by_input.items())
        },
        "excluded_by_source": dict(sorted(excluded.items())),
        "duplicate_rows": duplicates,
        "unknown_rows": unknown_rows,
    }
    return pools, summary


def _sample_original_rows(
    *,
    pools: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    rng: random.Random,
    seed: int,
    stage_name: str,
    audit_prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    replacement_fill: Counter[str] = Counter()
    unique_ids: dict[str, int] = {}

    for source, count in targets.items():
        pool = pools.get(source, [])
        if not pool:
            raise ValueError(f"no eligible original-label guard rows for source={source}")
        if len(pool) >= count:
            chosen = rng.sample(pool, count)
        else:
            chosen = list(pool)
            fill_count = count - len(pool)
            replacement_fill[source] = fill_count
            chosen.extend(dict(rng.choice(pool)) for _ in range(fill_count))
        for index, row in enumerate(chosen):
            out = dict(row)
            out[_audit_key(audit_prefix, "selector")] = stage_name
            out[_audit_key(audit_prefix, "target_count")] = count
            out[_audit_key(audit_prefix, "seed")] = seed
            if index >= len(pool):
                out[_audit_key(audit_prefix, "replacement_fill")] = True
            sampled.append(out)
        unique_ids[source] = len({_utt_id(row) for row in chosen})

    summary = {
        "target_total": sum(targets.values()),
        "targets_by_source": dict(sorted(targets.items())),
        "actual_by_source": dict(
            sorted(Counter(str(row[_audit_key(audit_prefix, "source")]) for row in sampled).items())
        ),
        "replacement_fill_by_source": dict(sorted(replacement_fill.items())),
        "unique_utt_ids_by_source": dict(sorted(unique_ids.items())),
    }
    return sampled, summary


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
    stage39 = Path(DEFAULT_STAGE39_DIR)
    parser = argparse.ArgumentParser(
        description="Build Stage104 Libri teacher substitution repair with public guards."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--stage-name", default=DEFAULT_STAGE_NAME)
    parser.add_argument("--audit-prefix", default="stage104")
    parser.add_argument("--stage61-lengths", default=DEFAULT_STAGE61_LENGTHS)
    parser.add_argument("--stage22-lengths", default=DEFAULT_STAGE22_LENGTHS)
    parser.add_argument("--selector-lengths", default=DEFAULT_SELECTOR_LENGTHS)
    parser.add_argument("--teacher-cache", default=str(stage39 / "stage39_teacher_accepted.cache.jsonl"))
    parser.add_argument("--teacher-lengths", default=str(stage39 / "stage39_teacher_accepted.lengths.jsonl"))
    parser.add_argument("--teacher-samples", type=int, default=8000)
    parser.add_argument(
        "--no-teacher",
        action="store_true",
        help="Build only the original-label guard component without teacher override rows.",
    )
    parser.add_argument(
        "--teacher-spec",
        action="append",
        default=None,
        help=(
            "Repeatable source:samples:cache:lengths teacher override spec. "
            "When omitted, the legacy LibriSpeech Stage39 arguments are used."
        ),
    )
    parser.add_argument("--original-quotas", default=None)
    parser.add_argument("--bucket-width", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--skip-buckets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit_prefix = str(args.audit_prefix).strip().lstrip("_")
    if not audit_prefix or not audit_prefix.replace("_", "").isalnum():
        raise ValueError(f"invalid audit prefix: {args.audit_prefix!r}")

    repo_root = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    stage_dir = root / "stages" / args.stage_name
    output_path = stage_dir / "webdataset_lengths.jsonl"
    teacher_specs = _parse_teacher_specs(args)
    teacher_cache_out_name = (
        f"{audit_prefix}_libri_teacher_sampled.cache.jsonl"
        if len(teacher_specs) == 1 and str(teacher_specs[0]["source"]) == TEACHER_SOURCE
        else f"{audit_prefix}_teacher_sampled.cache.jsonl"
    )
    teacher_cache_out = stage_dir / teacher_cache_out_name
    summary_path = stage_dir / "webdataset_lengths.summary.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to rebuild")

    rng = random.Random(int(args.seed))
    original_quotas = _parse_quotas(args.original_quotas)
    selector_ids = _load_ids(Path(args.selector_lengths))
    if teacher_specs:
        teacher_rows, teacher_cache_rows, teacher_summary, accepted_teacher_ids = _build_teacher_component(
            teacher_specs=teacher_specs,
            rng=rng,
            audit_prefix=audit_prefix,
        )
    else:
        teacher_rows = []
        teacher_cache_rows = []
        accepted_teacher_ids = set()
        teacher_summary = {
            "target_total": 0,
            "targets_by_source": {},
            "available_teacher_rows": 0,
            "actual_samples": 0,
            "actual_samples_by_source": {},
            "unique_sampled_teacher_ids": 0,
            "sampled_cache_rows": 0,
            "sampled_cache_sources": [],
            "accepted_teacher_ids_total": 0,
            "by_source": {},
        }
    exclude_original_ids = selector_ids | accepted_teacher_ids
    pools, input_summary = _load_original_pools(
        inputs=[
            ("stage61_guard_balanced", Path(args.stage61_lengths)),
            ("stage22_clean_fallback", Path(args.stage22_lengths)),
        ],
        targets=original_quotas,
        exclude_ids=exclude_original_ids,
        audit_prefix=audit_prefix,
    )
    original_rows, original_summary = _sample_original_rows(
        pools=pools,
        targets=original_quotas,
        rng=rng,
        seed=int(args.seed),
        stage_name=str(args.stage_name),
        audit_prefix=audit_prefix,
    )

    combined_rows = original_rows + teacher_rows
    rng.shuffle(combined_rows)
    _write_jsonl(output_path, combined_rows)
    _write_jsonl(teacher_cache_out, teacher_cache_rows)

    cache_sources = sorted({_canonical_source(row.get("source")) or "unknown" for row in teacher_cache_rows})
    expected_cache_sources = sorted(str(spec["source"]) for spec in teacher_specs)
    if cache_sources != expected_cache_sources:
        raise ValueError(
            f"sampled teacher cache sources mismatch, expected {expected_cache_sources}, got {cache_sources}"
        )

    summary = {
        "version": 1,
        "stage_name": str(args.stage_name),
        "audit_prefix": audit_prefix,
        "seed": int(args.seed),
        "root": str(root),
        "length_index_path": str(output_path),
        "sampled_teacher_cache_path": str(teacher_cache_out),
        "stage61_lengths": str(Path(args.stage61_lengths)),
        "stage22_lengths": str(Path(args.stage22_lengths)),
        "teacher_cache": str(Path(args.teacher_cache)),
        "teacher_lengths": str(Path(args.teacher_lengths)),
        "teacher_specs": [
            {
                "source": str(spec["source"]),
                "samples": int(spec["samples"]),
                "cache_path": str(spec["cache_path"]),
                "lengths_path": str(spec["lengths_path"]),
            }
            for spec in teacher_specs
        ],
        "selector_lengths": str(Path(args.selector_lengths)),
        "total_rows": len(combined_rows),
        "component_counts": dict(
            sorted(
                Counter(str(row[_audit_key(audit_prefix, "component")]) for row in combined_rows).items()
            )
        ),
        "source_counts": dict(
            sorted(Counter(str(row[_audit_key(audit_prefix, "source")]) for row in combined_rows).items())
        ),
        "unique_output_utt_ids": len({_utt_id(row) for row in combined_rows}),
        "selector_ids_excluded": len(selector_ids),
        "accepted_teacher_ids_excluded_from_original": len(accepted_teacher_ids),
        "input_summary": input_summary,
        "original_label_guard": original_summary,
        "teacher_substitution_repair": teacher_summary,
        "libri_teacher_substitution_repair": teacher_summary
        if len(teacher_specs) == 1 and str(teacher_specs[0]["source"]) == TEACHER_SOURCE
        else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.skip_buckets:
        _build_bucket_manifest(repo_root, root, stage_dir, int(args.bucket_width))
    print(
        f"[rwkvasr] {args.stage_name} Libri teacher guard replay ready "
        f"rows={len(combined_rows)} output={output_path} "
        f"teacher_cache={teacher_cache_out} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
