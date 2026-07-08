#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile for an empty list")
    index = int(fraction * (len(sorted_values) - 1))
    return float(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _tier_for(error: float, q25: float, q75: float, q90: float) -> str:
    if error <= q25:
        return "easy_anchor"
    if error <= q75:
        return "medium"
    if error <= q90:
        return "hard"
    return "extreme"


def _parse_ratio(raw: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        ratios[key.strip()] = float(value)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("tier ratios must sum to a positive value")
    return {key: value / total for key, value in ratios.items()}


def _sample_with_replacement(
    rng: random.Random,
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if not rows:
        return []
    return [rng.choice(rows) for _ in range(count)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage43 difficulty-stratified original-label replay length index."
    )
    parser.add_argument("--candidate-lengths", required=True)
    parser.add_argument("--student-scored", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="stage43_difficulty_replay200k.lengths.jsonl")
    parser.add_argument("--target-samples", type=int, default=200000)
    parser.add_argument(
        "--tier-ratio",
        default="easy_anchor:0.15,medium:0.65,hard:0.20,extreme:0.00",
        help="Per-source sample ratio over computed difficulty tiers.",
    )
    parser.add_argument("--seed", type=int, default=20260620)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidate_lengths_path = Path(args.candidate_lengths)
    student_scored_path = Path(args.student_scored)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(args.output_name)
    summary_path = output_path.with_suffix(".summary.json")

    length_by_utt: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(candidate_lengths_path):
        utt_id = str(row.get("utt_id") or row.get("id") or "")
        if utt_id:
            length_by_utt[utt_id] = row

    scored_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_lengths: list[str] = []
    for score_row in _iter_jsonl(student_scored_path):
        utt_id = str(score_row.get("utt_id") or "")
        length_row = length_by_utt.get(utt_id)
        if length_row is None:
            missing_lengths.append(utt_id)
            continue
        source = str(score_row.get("source") or length_row.get("_stage39_source") or "unknown")
        error = float(score_row["primary_error"])
        enriched = dict(length_row)
        enriched["_stage43_source"] = source
        enriched["_stage43_primary_error"] = error
        enriched["_stage43_student_wer"] = float(score_row.get("student_wer", error))
        enriched["_stage43_student_cer"] = float(score_row.get("student_cer", error))
        enriched["_stage43_pred_text"] = score_row.get("pred_text")
        enriched["_stage43_selector"] = "difficulty_stratified_original_label"
        scored_by_source[source].append(enriched)

    if not scored_by_source:
        raise ValueError("no scored rows matched candidate lengths")

    tier_ratio = _parse_ratio(str(args.tier_ratio))
    sources = sorted(scored_by_source)
    base_per_source = int(args.target_samples) // len(sources)
    remainder = int(args.target_samples) - base_per_source * len(sources)
    rng = random.Random(int(args.seed))
    sampled_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "version": 1,
        "candidate_lengths": str(candidate_lengths_path),
        "student_scored": str(student_scored_path),
        "output_path": str(output_path),
        "target_samples": int(args.target_samples),
        "seed": int(args.seed),
        "tier_ratio": tier_ratio,
        "sources": {},
        "missing_length_count": len(missing_lengths),
        "missing_length_examples": missing_lengths[:20],
    }

    for index, source in enumerate(sources):
        rows = scored_by_source[source]
        errors = sorted(float(row["_stage43_primary_error"]) for row in rows)
        q25 = _quantile(errors, 0.25)
        q75 = _quantile(errors, 0.75)
        q90 = _quantile(errors, 0.90)
        tiers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            tier = _tier_for(float(row["_stage43_primary_error"]), q25, q75, q90)
            row = dict(row)
            row["_stage43_difficulty_tier"] = tier
            tiers[tier].append(row)

        per_source_target = base_per_source + (1 if index < remainder else 0)
        source_counts: Counter[str] = Counter()
        source_samples: list[dict[str, Any]] = []
        allocated = 0
        tier_names = list(tier_ratio)
        for tier in tier_names:
            if tier == tier_names[-1]:
                count = per_source_target - allocated
            else:
                count = int(round(per_source_target * tier_ratio[tier]))
                allocated += count
            chosen = _sample_with_replacement(rng, tiers.get(tier, []), count)
            source_counts[tier] += len(chosen)
            source_samples.extend(chosen)

        if len(source_samples) < per_source_target:
            fallback_pool = [row for tier in ("medium", "hard", "easy_anchor") for row in tiers.get(tier, [])]
            fill = _sample_with_replacement(rng, fallback_pool, per_source_target - len(source_samples))
            source_counts["fallback"] += len(fill)
            source_samples.extend(fill)

        rng.shuffle(source_samples)
        sampled_rows.extend(source_samples)
        summary["sources"][source] = {
            "available": len(rows),
            "target": per_source_target,
            "quantiles": {"q25": q25, "q75": q75, "q90": q90},
            "available_by_tier": {tier: len(values) for tier, values in sorted(tiers.items())},
            "sampled_by_tier": dict(sorted(source_counts.items())),
        }

    rng.shuffle(sampled_rows)
    sampled_rows = sampled_rows[: int(args.target_samples)]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in sampled_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    total_counts: Counter[tuple[str, str]] = Counter(
        (str(row["_stage43_source"]), str(row["_stage43_difficulty_tier"])) for row in sampled_rows
    )
    summary["actual_samples"] = len(sampled_rows)
    summary["actual_by_source_tier"] = {
        f"{source}/{tier}": count for (source, tier), count in sorted(total_counts.items())
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[rwkvasr] stage43 difficulty replay ready samples={len(sampled_rows)} "
        f"path={output_path} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
