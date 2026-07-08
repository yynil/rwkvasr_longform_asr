#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_STAGE22_LENGTHS = (
    "/media/usbhd/training_data/asr/curriculum/"
    "stage22_stage21_soup_clean_repair_mix/stages/"
    "stage22_public90_hard10_from_stage21_soup/webdataset_lengths.jsonl"
)
DEFAULT_STAGE39_DIR = "/media/usbhd/rwkvasr_runs/stage39_large_teacher_repair_20260620"
DEFAULT_SELECTOR_LENGTHS = (
    "/media/usbhd/rwkvasr_runs/stage37_ctc_decode_audit_20260620/"
    "stage30_selector256.lengths.jsonl"
)
DEFAULT_OUTPUT_DIR = "/media/usbhd/rwkvasr_runs/stage53_english_teacher_guard_20260621"

CANONICAL_SOURCES = (
    "librispeech",
    "aishell3",
    "commonvoice_en",
    "commonvoice_cn",
    "gigaspeech",
    "wenetspeech",
)
TEACHER_SOURCES = ("commonvoice_en", "gigaspeech")
GUARD_TARGETS = {
    "librispeech": 2000,
    "aishell3": 2000,
    "commonvoice_cn": 2000,
    "wenetspeech": 8000,
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows) -> None:
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
        "_stage53_source",
        "_stage47_source",
        "_stage43_source",
        "_stage39_source",
        "_stage37_selector_source",
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


def _split_even(total: int, labels: tuple[str, ...] | list[str]) -> dict[str, int]:
    base = total // len(labels)
    remainder = total - base * len(labels)
    return {label: base + (1 if index < remainder else 0) for index, label in enumerate(labels)}


def _load_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {_utt_id(row) for row in _iter_jsonl(path) if _utt_id(row)}


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


def _sample_with_replacement(
    rng: random.Random,
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    return [dict(rng.choice(rows)) for _ in range(count)]


def _weighted_sample_with_replacement(
    rng: random.Random,
    rows: list[dict[str, Any]],
    count: int,
    weight_field: str,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    weights = [max(float(row.get(weight_field, 0.0) or 0.0), 1e-6) for row in rows]
    return [dict(row) for row in rng.choices(rows, weights=weights, k=count)]


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile for an empty list")
    index = int(fraction * (len(sorted_values) - 1))
    return float(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _difficulty_tier(error: float, q25: float, qmax: float) -> str:
    if error <= q25:
        return "easy_anchor"
    if error <= qmax:
        return "medium"
    return "hard_or_extreme"


def _build_original_replay(
    *,
    stage22_lengths: Path,
    target_total: int,
    exclude_ids: set[str],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = _split_even(target_total, CANONICAL_SOURCES)
    reservoirs: dict[str, list[dict[str, Any]]] = {source: [] for source in CANONICAL_SOURCES}
    eligible_seen: Counter[str] = Counter()
    excluded_seen: Counter[str] = Counter()
    unknown_seen = 0

    for row in _iter_jsonl(stage22_lengths):
        utt_id = _utt_id(row)
        source = _infer_source(row)
        if source not in reservoirs:
            unknown_seen += 1
            continue
        if utt_id in exclude_ids:
            excluded_seen[source] += 1
            continue
        target = targets[source]
        eligible_seen[source] += 1
        enriched = dict(row)
        enriched["_stage53_component"] = "original_replay"
        enriched["_stage53_source"] = source
        enriched["_stage53_selector"] = "stage22_source_balanced_original_label"
        bucket = reservoirs[source]
        if len(bucket) < target:
            bucket.append(enriched)
            continue
        replacement_index = rng.randrange(eligible_seen[source])
        if replacement_index < target:
            bucket[replacement_index] = enriched

    rows: list[dict[str, Any]] = []
    fill_counts: Counter[str] = Counter()
    for source in CANONICAL_SOURCES:
        target = targets[source]
        bucket = reservoirs[source]
        if not bucket:
            raise ValueError(f"no eligible original replay rows for source={source}")
        if len(bucket) < target:
            fill = _sample_with_replacement(rng, bucket, target - len(bucket))
            for row in fill:
                row["_stage53_original_replacement_fill"] = True
            fill_counts[source] = len(fill)
            bucket.extend(fill)
        rows.extend(dict(row) for row in bucket[:target])

    return rows, {
        "target_total": target_total,
        "targets_by_source": targets,
        "eligible_seen_by_source": dict(sorted(eligible_seen.items())),
        "excluded_seen_by_source": dict(sorted(excluded_seen.items())),
        "replacement_fill_by_source": dict(sorted(fill_counts.items())),
        "unknown_source_rows": unknown_seen,
        "actual_by_source": dict(sorted(Counter(str(row["_stage53_source"]) for row in rows).items())),
    }


def _build_teacher_component(
    *,
    teacher_cache_path: Path,
    lengths_by_utt: dict[str, dict[str, Any]],
    target_total: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], set[str]]:
    cache_by_utt: dict[str, dict[str, Any]] = {}
    available_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accepted_ids: set[str] = set()
    skipped_by_source: Counter[str] = Counter()
    missing_lengths: list[str] = []

    for cache_row in _iter_jsonl(teacher_cache_path):
        utt_id = _utt_id(cache_row)
        if not utt_id:
            continue
        accepted_ids.add(utt_id)
        source = _canonical_source(cache_row.get("source")) or "unknown"
        if source not in TEACHER_SOURCES:
            skipped_by_source[source] += 1
            continue
        length_row = lengths_by_utt.get(utt_id)
        if length_row is None:
            missing_lengths.append(utt_id)
            continue
        cache_row = dict(cache_row)
        cache_row["source"] = source
        cache_by_utt[utt_id] = cache_row
        enriched = dict(length_row)
        enriched["_stage53_component"] = "teacher_repair_english_noisy"
        enriched["_stage53_source"] = source
        enriched["_stage53_teacher_override"] = True
        enriched["_stage53_primary_improvement"] = float(cache_row.get("primary_improvement", 0.0) or 0.0)
        enriched["_stage53_student_error"] = float(
            cache_row.get("student_wer", cache_row.get("student_cer", 0.0)) or 0.0
        )
        enriched["_stage53_teacher_error"] = float(
            cache_row.get("teacher_wer", cache_row.get("teacher_cer", 0.0)) or 0.0
        )
        available_by_source[source].append(enriched)

    targets = _split_even(target_total, list(TEACHER_SOURCES))
    sampled: list[dict[str, Any]] = []
    for source in TEACHER_SOURCES:
        pool = available_by_source.get(source, [])
        if not pool:
            raise ValueError(f"no teacher rows available for source={source}")
        sampled.extend(
            _weighted_sample_with_replacement(
                rng,
                pool,
                targets[source],
                "_stage53_primary_improvement",
            )
        )

    sampled_ids = {_utt_id(row) for row in sampled}
    sampled_cache_rows = [cache_by_utt[utt_id] for utt_id in sorted(sampled_ids) if utt_id in cache_by_utt]
    bad_sources = sorted(
        {
            _canonical_source(row.get("source")) or "unknown"
            for row in sampled_cache_rows
            if (_canonical_source(row.get("source")) or "unknown") not in TEACHER_SOURCES
        }
    )
    if bad_sources:
        raise ValueError(f"Stage53 teacher cache contains disallowed sources: {bad_sources}")

    return sampled, sampled_cache_rows, {
        "target_total": target_total,
        "targets_by_source": targets,
        "available_by_source": {
            source: len(rows) for source, rows in sorted(available_by_source.items())
        },
        "actual_by_source": dict(sorted(Counter(str(row["_stage53_source"]) for row in sampled).items())),
        "unique_sampled_teacher_ids": len(sampled_ids),
        "sampled_cache_rows": len(sampled_cache_rows),
        "accepted_teacher_ids_total": len(accepted_ids),
        "skipped_cache_rows_by_source": dict(sorted(skipped_by_source.items())),
        "missing_length_count": len(missing_lengths),
        "missing_length_examples": missing_lengths[:20],
    }, accepted_ids


def _build_guard_component(
    *,
    candidate_lengths: Path,
    student_scored: Path,
    targets: dict[str, int],
    exclude_ids: set[str],
    rng: random.Random,
    max_quantile_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    length_by_utt = _load_lengths_by_utt([candidate_lengths])
    candidates_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_excluded: Counter[str] = Counter()
    missing_lengths: list[str] = []

    for score_row in _iter_jsonl(student_scored):
        utt_id = _utt_id(score_row)
        source = _canonical_source(score_row.get("source"))
        if source not in targets:
            continue
        if utt_id in exclude_ids:
            skipped_excluded[source] += 1
            continue
        length_row = length_by_utt.get(utt_id)
        if length_row is None:
            missing_lengths.append(utt_id)
            continue
        error = float(score_row.get("primary_error", score_row.get("student_cer", 0.0)) or 0.0)
        enriched = dict(length_row)
        enriched["_stage53_component"] = "public_clean_chinese_guard_anchor"
        enriched["_stage53_source"] = source
        enriched["_stage53_teacher_override"] = False
        enriched["_stage53_primary_error"] = error
        enriched["_stage53_student_wer"] = float(score_row.get("student_wer", error) or error)
        enriched["_stage53_student_cer"] = float(score_row.get("student_cer", error) or error)
        enriched["_stage53_pred_text"] = score_row.get("pred_text")
        candidates_by_source[source].append(enriched)

    sampled: list[dict[str, Any]] = []
    source_summary: dict[str, Any] = {}
    for source, target in targets.items():
        candidates = candidates_by_source.get(source, [])
        if not candidates:
            raise ValueError(f"no guard candidates for source={source}")
        errors = sorted(float(row["_stage53_primary_error"]) for row in candidates)
        q25 = _quantile(errors, 0.25)
        qmax = _quantile(errors, max_quantile_fraction)
        pool: list[dict[str, Any]] = []
        for row in candidates:
            row = dict(row)
            row["_stage53_guard_tier"] = _difficulty_tier(float(row["_stage53_primary_error"]), q25, qmax)
            if float(row["_stage53_primary_error"]) <= qmax:
                pool.append(row)
        if not pool:
            raise ValueError(f"empty guard pool after quantile filter for source={source}")
        source_sampled = _sample_with_replacement(rng, pool, target)
        sampled.extend(source_sampled)
        source_summary[source] = {
            "target": target,
            "available_candidates": len(candidates),
            "available_anchor_pool": len(pool),
            "q25": q25,
            "max_quantile": qmax,
            "unique_sampled_anchor_ids": len({_utt_id(row) for row in source_sampled}),
            "sampled_by_tier": dict(sorted(Counter(str(row["_stage53_guard_tier"]) for row in source_sampled).items())),
            "skipped_excluded": int(skipped_excluded[source]),
        }

    return sampled, {
        "target_total": sum(targets.values()),
        "targets_by_source": targets,
        "max_quantile_fraction": max_quantile_fraction,
        "actual_by_source": dict(sorted(Counter(str(row["_stage53_source"]) for row in sampled).items())),
        "source_summary": source_summary,
        "missing_length_count": len(missing_lengths),
        "missing_length_examples": missing_lengths[:20],
    }


def _parse_args() -> argparse.Namespace:
    stage39_dir = Path(DEFAULT_STAGE39_DIR)
    parser = argparse.ArgumentParser(
        description="Build Stage53 English-teacher plus public-clean/Chinese guard replay."
    )
    parser.add_argument("--stage22-lengths", default=DEFAULT_STAGE22_LENGTHS)
    parser.add_argument("--candidate-lengths", default=str(stage39_dir / "stage39_candidate12288.lengths.jsonl"))
    parser.add_argument("--student-scored", default=str(stage39_dir / "stage39_stage21_candidate12282.scored.jsonl"))
    parser.add_argument("--teacher-cache", default=str(stage39_dir / "stage39_teacher_accepted.cache.jsonl"))
    parser.add_argument("--teacher-lengths", default=str(stage39_dir / "stage39_teacher_accepted.lengths.jsonl"))
    parser.add_argument("--selector-lengths", default=DEFAULT_SELECTOR_LENGTHS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="stage53_original180k_engteacher6k_guard14k.lengths.jsonl")
    parser.add_argument("--teacher-cache-name", default="stage53_teacher_cven_giga_sampled.cache.jsonl")
    parser.add_argument("--summary-name", default="stage53_original180k_engteacher6k_guard14k.summary.json")
    parser.add_argument("--original-samples", type=int, default=180000)
    parser.add_argument("--teacher-samples", type=int, default=6000)
    parser.add_argument("--guard-max-quantile", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage22_lengths = Path(args.stage22_lengths)
    candidate_lengths = Path(args.candidate_lengths)
    student_scored = Path(args.student_scored)
    teacher_cache = Path(args.teacher_cache)
    teacher_lengths = Path(args.teacher_lengths)
    selector_lengths = Path(args.selector_lengths)
    output_path = output_dir / str(args.output_name)
    sampled_teacher_cache_path = output_dir / str(args.teacher_cache_name)
    summary_path = output_dir / str(args.summary_name)

    rng = random.Random(int(args.seed))
    selector_ids = _load_ids(selector_lengths)
    lengths_by_utt = _load_lengths_by_utt([teacher_lengths, candidate_lengths])

    teacher_rows, teacher_cache_rows, teacher_summary, accepted_teacher_ids = _build_teacher_component(
        teacher_cache_path=teacher_cache,
        lengths_by_utt=lengths_by_utt,
        target_total=int(args.teacher_samples),
        rng=rng,
    )
    exclude_ids = selector_ids | accepted_teacher_ids
    original_rows, original_summary = _build_original_replay(
        stage22_lengths=stage22_lengths,
        target_total=int(args.original_samples),
        exclude_ids=exclude_ids,
        rng=rng,
    )
    guard_rows, guard_summary = _build_guard_component(
        candidate_lengths=candidate_lengths,
        student_scored=student_scored,
        targets=dict(GUARD_TARGETS),
        exclude_ids=exclude_ids,
        rng=rng,
        max_quantile_fraction=float(args.guard_max_quantile),
    )

    combined_rows = original_rows + teacher_rows + guard_rows
    rng.shuffle(combined_rows)
    _write_jsonl(output_path, combined_rows)
    _write_jsonl(sampled_teacher_cache_path, teacher_cache_rows)

    bad_teacher_sources = sorted(
        {
            _canonical_source(row.get("source")) or "unknown"
            for row in teacher_cache_rows
            if (_canonical_source(row.get("source")) or "unknown") not in TEACHER_SOURCES
        }
    )
    if bad_teacher_sources:
        raise ValueError(f"disallowed teacher sources in sampled cache: {bad_teacher_sources}")

    summary = {
        "version": 1,
        "seed": int(args.seed),
        "stage22_lengths": str(stage22_lengths),
        "candidate_lengths": str(candidate_lengths),
        "student_scored": str(student_scored),
        "teacher_cache": str(teacher_cache),
        "teacher_lengths": str(teacher_lengths),
        "selector_lengths": str(selector_lengths),
        "output_path": str(output_path),
        "sampled_teacher_cache_path": str(sampled_teacher_cache_path),
        "teacher_sources": list(TEACHER_SOURCES),
        "guard_targets": dict(GUARD_TARGETS),
        "total_rows": len(combined_rows),
        "component_counts": dict(sorted(Counter(str(row["_stage53_component"]) for row in combined_rows).items())),
        "source_counts": dict(sorted(Counter(str(row["_stage53_source"]) for row in combined_rows).items())),
        "selector_ids_excluded": len(selector_ids),
        "accepted_teacher_ids_excluded_from_original_and_guard": len(accepted_teacher_ids),
        "unique_output_utt_ids": len({_utt_id(row) for row in combined_rows}),
        "original_replay": original_summary,
        "teacher_repair_english_noisy": teacher_summary,
        "public_clean_chinese_guard_anchor": guard_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[rwkvasr] stage53 English-teacher guard replay ready "
        f"rows={len(combined_rows)} output={output_path} "
        f"teacher_cache={sampled_teacher_cache_path} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
