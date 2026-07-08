#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix"
DEFAULT_STAGE165_LENGTHS = (
    DEFAULT_ROOT
    + "/stages/stage165_stage110_funasr_nano_ctc_consistent_teacher_heavy_60k_guard_60k/webdataset_lengths.jsonl"
)
DEFAULT_STAGE169_TOPK = (
    "/media/usbhd/rwkvasr_runs/stage169_funasr_nano_ctc_topk_stage165_full_20260627/"
    "funasr_nano_ctc_topk_stage165_unique_merged.jsonl"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT + "/stages/stage172_audio_only_online_ctc_curriculum"


STAGE_SPECS = (
    ("stage172a_easy_online_ctc", 0.65, 120_000),
    ("stage172b_medium_online_ctc", 0.85, 120_000),
    ("stage172c_fullhard_online_ctc", 1.00, 120_000),
)


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _utt_id(row: dict[str, Any]) -> str:
    value = row.get("utt_id") or row.get("id") or row.get("key")
    if value is None:
        raise ValueError(f"row has no utt id: {row}")
    return str(value)


def _source(row: dict[str, Any], topk_row: dict[str, Any] | None) -> str:
    value = row.get("_stage165_source") or row.get("source_dataset")
    if value is None and topk_row is not None:
        value = topk_row.get("source")
    return str(value or "unknown").strip().lower() or "unknown"


def _language(row: dict[str, Any], topk_row: dict[str, Any] | None) -> str:
    value = row.get("language")
    if value is None and topk_row is not None:
        value = topk_row.get("language")
    return str(value or "unknown").strip().lower() or "unknown"


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _max_metric(row: dict[str, Any], keys: Iterable[str], default: float = 0.0) -> float:
    values = [_finite_float(row.get(key)) for key in keys]
    filtered = [value for value in values if value is not None]
    return max(filtered) if filtered else float(default)


def _load_topk_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        if row.get("error") is not None:
            continue
        if row.get("split") not in (None, "train"):
            continue
        value = row.get("utt_id") or row.get("key")
        if value is not None:
            cache[str(value)] = row
    return cache


def _difficulty_score(row: dict[str, Any], topk_row: dict[str, Any] | None) -> dict[str, float]:
    student_error = _max_metric(
        row,
        (
            "_stage165_student_error",
            "_stage165_primary_improvement",
            "_stage161_primary_error",
            "_stage161_student_wer",
            "_stage161_student_cer",
            "_stage157_primary_error",
            "_stage149_primary_error",
            "_stage43_primary_error",
            "_stage164_ctc_vs_llm_wer",
            "_stage164_ctc_vs_llm_cer",
        ),
        default=0.0,
    )
    avg_top1_logp = 0.0
    blank_top1_ratio = 0.0
    if topk_row is not None:
        avg_top1_logp = float(topk_row.get("funasr_ctc_avg_top1_logp") or 0.0)
        blank_top1_ratio = float(topk_row.get("funasr_ctc_blank_top1_ratio") or 0.0)
    confidence_penalty = max(0.0, min(1.0, -avg_top1_logp))
    blank_extreme_penalty = abs(max(0.0, min(1.0, blank_top1_ratio)) - 0.80)
    num_frames = max(1, int(row.get("num_frames") or 0))
    length_factor = min(float(num_frames) / 1600.0, 1.5)
    score = (
        student_error * 0.65
        + confidence_penalty * 1.50
        + length_factor * 0.15
        + blank_extreme_penalty * 0.05
    )
    return {
        "score": float(score),
        "student_error": float(student_error),
        "confidence_penalty": float(confidence_penalty),
        "length_factor": float(length_factor),
        "blank_extreme_penalty": float(blank_extreme_penalty),
        "funasr_ctc_avg_top1_logp": float(avg_top1_logp),
        "funasr_ctc_blank_top1_ratio": float(blank_top1_ratio),
    }


def _load_unique_train_rows(
    *,
    stage165_lengths: Path,
    topk_cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_utt: dict[str, dict[str, Any]] = {}
    repeats = 0
    total_rows = 0
    for row in _iter_jsonl(stage165_lengths):
        total_rows += 1
        utt_id = _utt_id(row)
        if utt_id in by_utt:
            repeats += 1
            continue
        by_utt[utt_id] = row

    annotated: list[dict[str, Any]] = []
    skipped_nontrain = 0
    for utt_id, row in by_utt.items():
        if row.get("split") != "train":
            skipped_nontrain += 1
            continue
        topk_row = topk_cache.get(utt_id)
        source = _source(row, topk_row)
        language = _language(row, topk_row)
        score = _difficulty_score(row, topk_row)
        annotated.append(
            {
                "utt_id": utt_id,
                "row": row,
                "source": source,
                "language": language,
                **score,
            }
        )
    summary = {
        "stage165_length_rows": total_rows,
        "stage165_unique_rows": len(by_utt),
        "stage165_repeated_rows_skipped": repeats,
        "skipped_nontrain_unique_rows": skipped_nontrain,
        "unique_train_rows": len(annotated),
        "topk_records": len(topk_cache),
        "unique_train_rows_with_topk": sum(1 for item in annotated if item["utt_id"] in topk_cache),
        "source_counts": dict(sorted(Counter(str(item["source"]) for item in annotated).items())),
        "teacher_override_counts": {
            str(key): value
            for key, value in sorted(
                Counter(bool(item["row"].get("_stage165_teacher_override")) for item in annotated).items()
            )
        },
    }
    return annotated, summary


def _proportional_targets(source_counts: Counter[str], total: int) -> dict[str, int]:
    sources = sorted(source_counts)
    raw: dict[str, float] = {
        source: float(total) * float(source_counts[source]) / float(sum(source_counts.values()))
        for source in sources
    }
    targets = {source: int(math.floor(raw[source])) for source in sources}
    remainder = int(total) - sum(targets.values())
    ranked = sorted(sources, key=lambda source: (raw[source] - targets[source], source), reverse=True)
    for source in ranked[:remainder]:
        targets[source] += 1
    return targets


def _sample_stage(
    annotated: list[dict[str, Any]],
    *,
    stage_name: str,
    pool_quantile: float,
    target_rows: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotated:
        by_source[str(item["source"])].append(item)
    for source in by_source:
        by_source[source].sort(key=lambda item: (float(item["score"]), str(item["utt_id"])))

    source_counts = Counter({source: len(items) for source, items in by_source.items()})
    targets = _proportional_targets(source_counts, int(target_rows))
    selected: list[dict[str, Any]] = []
    pool_counts: dict[str, int] = {}
    unique_counts: dict[str, int] = {}
    replacement_counts: dict[str, int] = {}
    for source in sorted(by_source):
        source_rows = by_source[source]
        pool_size = max(1, int(math.ceil(len(source_rows) * float(pool_quantile))))
        pool = source_rows[:pool_size]
        target = int(targets[source])
        pool_counts[source] = len(pool)
        unique_counts[source] = len({str(item["utt_id"]) for item in pool})
        replacement_counts[source] = max(0, target - len(pool))
        if target <= len(pool):
            chosen = rng.sample(pool, target)
        else:
            chosen = list(pool)
            chosen.extend(rng.choice(pool) for _ in range(target - len(pool)))
            rng.shuffle(chosen)
        for sample_index, item in enumerate(chosen):
            row = dict(item["row"])
            row["source_dataset"] = str(item["source"])
            row["language"] = str(item["language"])
            row["_stage172_stage"] = stage_name
            row["_stage172_curriculum"] = "audio_only_online_ctc"
            row["_stage172_source"] = str(item["source"])
            row["_stage172_pool_quantile"] = float(pool_quantile)
            row["_stage172_target_rows"] = int(target_rows)
            row["_stage172_seed"] = int(seed)
            row["_stage172_sample_index"] = int(sample_index)
            row["_stage172_sampled_with_replacement"] = sample_index >= len(pool)
            row["_stage172_ctc_online_only"] = True
            row["_stage172_uses_text_labels"] = False
            row["_stage172_difficulty_score"] = round(float(item["score"]), 8)
            row["_stage172_student_error"] = round(float(item["student_error"]), 8)
            row["_stage172_confidence_penalty"] = round(float(item["confidence_penalty"]), 8)
            row["_stage172_length_factor"] = round(float(item["length_factor"]), 8)
            row["_stage172_blank_extreme_penalty"] = round(float(item["blank_extreme_penalty"]), 8)
            row["_stage172_funasr_ctc_avg_top1_logp"] = round(float(item["funasr_ctc_avg_top1_logp"]), 8)
            row["_stage172_funasr_ctc_blank_top1_ratio"] = round(float(item["funasr_ctc_blank_top1_ratio"]), 8)
            selected.append(row)
    selected.sort(
        key=lambda row: (
            int(row.get("num_frames") or 0),
            str(row.get("_stage172_source") or ""),
            str(row.get("utt_id") or row.get("key") or ""),
            int(row.get("_stage172_sample_index") or 0),
        )
    )
    counts = {
        "rows": len(selected),
        "target_rows": int(target_rows),
        "pool_quantile": float(pool_quantile),
        "targets_by_source": dict(sorted(targets.items())),
        "pool_rows_by_source": dict(sorted(pool_counts.items())),
        "unique_pool_rows_by_source": dict(sorted(unique_counts.items())),
        "replacement_rows_by_source": dict(sorted(replacement_counts.items())),
        "actual_rows_by_source": dict(
            sorted(Counter(str(row.get("_stage172_source") or "unknown") for row in selected).items())
        ),
        "actual_rows_by_language": dict(
            sorted(Counter(str(row.get("language") or "unknown") for row in selected).items())
        ),
        "teacher_override_rows": {
            str(key): value
            for key, value in sorted(
                Counter(bool(row.get("_stage165_teacher_override")) for row in selected).items()
            )
        },
    }
    return selected, counts


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    values = sorted(values)

    def pct(q: float) -> float:
        pos = q * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(values[lo])
        frac = pos - lo
        return float(values[lo] * (1.0 - frac) + values[hi] * frac)

    return {
        "count": len(values),
        "min": float(values[0]),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": float(values[-1]),
        "mean": float(sum(values) / len(values)),
    }


def _write_stage(stage_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    length_path = stage_dir / "webdataset_lengths.jsonl"
    with length_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    scores = [float(row.get("_stage172_difficulty_score") or 0.0) for row in rows]
    frames = [float(row.get("num_frames") or 0.0) for row in rows]
    payload = {
        **summary,
        "length_index_path": str(length_path),
        "bucket_manifest_path": str(stage_dir / "webdataset_buckets_audio_text" / "manifest.json"),
        "difficulty_score_distribution": _distribution(scores),
        "num_frames_distribution": _distribution(frames),
    }
    (stage_dir / "webdataset_lengths.summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
        description="Build Stage172 audio-only online CTC distillation curriculum length indexes."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--stage165-lengths", default=DEFAULT_STAGE165_LENGTHS)
    parser.add_argument("--topk-cache", default=DEFAULT_STAGE169_TOPK)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--target-rows", type=int, default=120_000)
    parser.add_argument("--bucket-width", type=int, default=200)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--skip-bucket-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    stage165_lengths = Path(args.stage165_lengths)
    topk_path = Path(args.topk_cache) if args.topk_cache else None
    output_root = Path(args.output_root)
    repo_root = Path(args.repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    topk_cache = _load_topk_cache(topk_path)
    annotated, source_summary = _load_unique_train_rows(
        stage165_lengths=stage165_lengths,
        topk_cache=topk_cache,
    )
    source_summary.update(
        {
            "version": 1,
            "root": str(root),
            "stage165_lengths": str(stage165_lengths),
            "topk_cache": str(topk_path) if topk_path is not None else None,
            "output_root": str(output_root),
            "seed": int(args.seed),
            "curriculum": "audio_only_online_ctc",
            "uses_text_labels": False,
            "teacher": "FunASR-Nano-2512 online CTC top-k",
        }
    )

    stage_summaries: dict[str, Any] = {}
    for index, (stage_name, quantile, default_target_rows) in enumerate(STAGE_SPECS):
        target_rows = int(args.target_rows or default_target_rows)
        rows, summary = _sample_stage(
            annotated,
            stage_name=stage_name,
            pool_quantile=float(quantile),
            target_rows=target_rows,
            seed=int(args.seed) + index,
        )
        stage_dir = output_root / stage_name
        _write_stage(
            stage_dir,
            rows,
            {
                **source_summary,
                "stage_name": stage_name,
                **summary,
            },
        )
        if not args.skip_bucket_manifest:
            _build_bucket_manifest(repo_root, root, stage_dir, int(args.bucket_width))
        stage_summaries[stage_name] = json.loads(
            (stage_dir / "webdataset_lengths.summary.json").read_text(encoding="utf-8")
        )
        _log(
            f"stage ready name={stage_name} rows={len(rows)} "
            f"quantile={quantile} dir={stage_dir}"
        )

    manifest = {
        **source_summary,
        "stages": stage_summaries,
    }
    (output_root / "stage172_curriculum_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"Stage172 curriculum complete output_root={output_root}")


if __name__ == "__main__":
    main()
