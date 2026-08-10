#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_supplemental import (
    validate_stage211_supplemental_inventory,
)

try:
    from scripts.filter_stage211_social_pcm_overlap import validate_filtered_inventory
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from filter_stage211_social_pcm_overlap import validate_filtered_inventory


DEFAULT_BASE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_supplemental_natural_v1/supplemental_inventory.json"
)
DEFAULT_SOCIAL_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_social_vad_filtered_v1/filtered_inventory.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_supplemental_combined_v2"
COMBINED_ARTIFACT = "stage211_supplemental_combined_inventory"
SCHEMA_VERSION = 2


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolved_part(part: dict[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    path = Path(str(part.get("path") or ""))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    expected_sha256 = str(part.get("sha256") or "")
    if (
        not path.is_file()
        or path.stat().st_size != int(part.get("size_bytes", -1))
        or _sha256(path) != expected_sha256
        or int(part.get("num_samples", -1)) <= 0
        or not str(part.get("source_label") or "")
    ):
        raise ValueError(f"Stage211 supplemental component part changed: {path}")
    return {**part, "path": str(path)}


def _load_train_parts(
    manifest_path: Path,
) -> tuple[dict[int, list[dict[str, Any]]], int, dict[str, Any]]:
    payload = _load_json(manifest_path, label="Stage211 supplemental component manifest")
    if (
        payload.get("version") != 1
        or payload.get("root") != "/"
        or int(payload.get("bucket_width", -1)) != 80
        or int(payload.get("entries_per_part", -1)) != 100_000
    ):
        raise ValueError("Stage211 supplemental component manifest contract changed.")
    splits = payload.get("splits")
    train = splits.get("train") if isinstance(splits, dict) else None
    eval_split = splits.get("eval") if isinstance(splits, dict) else None
    if not isinstance(train, dict) or not isinstance(eval_split, dict):
        raise ValueError("Stage211 supplemental component manifest lacks train/eval splits.")
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows = 0
    for bucket in train.get("buckets") or []:
        if not isinstance(bucket, dict):
            raise ValueError("Stage211 supplemental component bucket is invalid.")
        bucket_id = int(bucket.get("bucket_id", -1))
        parts = bucket.get("parts")
        if bucket_id < 0 or not isinstance(parts, list) or not parts:
            raise ValueError("Stage211 supplemental component bucket is invalid.")
        resolved = [_resolved_part(part, manifest_path=manifest_path) for part in parts]
        bucket_rows = sum(int(part["num_samples"]) for part in resolved)
        if bucket_rows != int(bucket.get("num_samples", -1)):
            raise ValueError("Stage211 supplemental component bucket total changed.")
        buckets[bucket_id].extend(resolved)
        rows += bucket_rows
    if rows != int(train.get("num_samples", -1)):
        raise ValueError("Stage211 supplemental component train total changed.")
    return dict(buckets), rows, eval_split


def _merged_source_totals(
    base: dict[str, Any],
    social: dict[str, Any],
    *,
    key: str,
) -> dict[str, float | int]:
    base_values = base.get(key)
    social_values = social.get(key)
    if not isinstance(base_values, dict) or not isinstance(social_values, dict):
        raise ValueError(f"Stage211 supplemental source totals are missing: {key}")
    if set(base_values) & set(social_values):
        raise ValueError("Stage211 base/social supplemental source labels overlap.")
    return {**base_values, **social_values}


def _write_atomic_directory(
    *,
    output_root: Path,
    manifest: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    staging = output_root.with_name(f"{output_root.name}.partial.{os.getpid()}")
    if staging.exists():
        raise ValueError(f"Stage211 combined staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    staging_manifest = staging / "webdataset_buckets_audio_text/manifest.json"
    staging_manifest.parent.mkdir(parents=True)
    staging_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    inventory["bucket_manifest_sha256"] = _sha256(staging_manifest)
    staging_inventory = staging / "supplemental_inventory.json"
    staging_inventory.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        staging.replace(output_root)
    except BaseException:
        raise
    return inventory


def build_combined_inventory(
    *,
    base_inventory_path: Path,
    social_inventory_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    started = time.time()
    base_inventory_path = base_inventory_path.expanduser().resolve()
    social_inventory_path = social_inventory_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_inventory = output_root / "supplemental_inventory.json"
    if output_root.exists():
        if not output_inventory.is_file():
            raise ValueError(
                f"Stage211 combined output exists without its inventory: {output_root}"
            )
        return validate_stage211_supplemental_inventory(
            output_inventory,
            require_training_ready=True,
            verify_part_sha256=True,
        )["inventory"]
    output_root.parent.mkdir(parents=True, exist_ok=True)

    base_validated = validate_stage211_supplemental_inventory(
        base_inventory_path,
        require_training_ready=True,
        verify_part_sha256=True,
    )
    base = base_validated["inventory"]
    if base.get("schema_version") != 1:
        raise ValueError("Stage211 combined builder requires the version-1 base natural pool.")
    social_validated = validate_filtered_inventory(
        social_inventory_path,
        verify_part_sha256=True,
    )
    social = social_validated["inventory"]

    base_manifest_path = Path(base_validated["bucket_manifest_path"])
    social_manifest_path = Path(social_validated["bucket_manifest_path"])
    base_buckets, base_rows, base_eval = _load_train_parts(base_manifest_path)
    social_buckets, social_rows, social_eval = _load_train_parts(social_manifest_path)
    if base_rows != int(base["selected_rows"]) or social_rows != int(social["selected_rows"]):
        raise ValueError("Stage211 combined component manifest totals changed.")
    if json.dumps(base_eval, sort_keys=True) != json.dumps(social_eval, sort_keys=True):
        raise ValueError("Stage211 base/social fixed-eval splits differ.")
    if int(base_eval.get("num_samples", -1)) != 256:
        raise ValueError("Stage211 combined fixed-eval split is incomplete.")

    merged_buckets: list[dict[str, Any]] = []
    part_records: list[dict[str, Any]] = []
    for bucket_id in sorted(set(base_buckets) | set(social_buckets)):
        parts = sorted(
            base_buckets.get(bucket_id, []) + social_buckets.get(bucket_id, []),
            key=lambda part: (str(part["source_label"]), str(part["path"])),
        )
        bucket_rows = sum(int(part["num_samples"]) for part in parts)
        merged_buckets.append(
            {"bucket_id": bucket_id, "num_samples": bucket_rows, "parts": parts}
        )
        part_records.extend(parts)
    total_rows = base_rows + social_rows
    manifest_path = output_root / "webdataset_buckets_audio_text/manifest.json"
    manifest = {
        "version": 1,
        "root": "/",
        "source_length_index_path": str(output_inventory),
        "bucket_width": 80,
        "entries_per_part": 100_000,
        "splits": {
            "train": {"num_samples": total_rows, "buckets": merged_buckets},
            "eval": base_eval,
        },
    }

    source_counts = _merged_source_totals(
        base,
        social,
        key="selected_counts_by_source",
    )
    source_hours = _merged_source_totals(
        base,
        social,
        key="selected_hours_by_source",
    )
    selected_hours = float(base["selected_hours"]) + float(social["selected_hours"])
    if sum(int(value) for value in source_counts.values()) != total_rows or not math.isclose(
        sum(float(value) for value in source_hours.values()),
        selected_hours,
        rel_tol=0.0,
        abs_tol=0.002,
    ):
        raise ValueError("Stage211 combined source totals changed.")
    stage179 = base.get("cross_pool_dedupe", {}).get("stage179")
    if not isinstance(stage179, dict):
        raise ValueError("Stage211 base inventory lacks Stage179 binding.")
    base_sources = set(base["selected_counts_by_source"])
    social_sources = set(social["selected_counts_by_source"])
    stage179_sources = set(stage179.get("sources") or [])
    source_sets_disjoint = (base_sources | social_sources).isdisjoint(stage179_sources)
    if not source_sets_disjoint:
        raise ValueError("Stage211 combined supplemental sources overlap Stage179 labels.")

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "artifact": COMBINED_ARTIFACT,
        "complete": True,
        "training_ready": True,
        "hash_archives": True,
        "require_production_layout": True,
        "layout_errors": [],
        "language": ["en", "zh"],
        "uses_text_labels": False,
        "storage_kinds": ["parquet", "tar", "zip"],
        "fixed_eval": base["fixed_eval"],
        "component_inventories": {
            "base_natural": {
                "inventory_path": str(base_inventory_path),
                "inventory_sha256": _sha256(base_inventory_path),
                "bucket_manifest_path": str(base_manifest_path),
                "bucket_manifest_sha256": _sha256(base_manifest_path),
                "rows": base_rows,
                "hours": float(base["selected_hours"]),
                "dedupe_mode": "source_identity",
            },
            "social_vad": {
                "inventory_path": str(social_inventory_path),
                "inventory_sha256": _sha256(social_inventory_path),
                "bucket_manifest_path": str(social_manifest_path),
                "bucket_manifest_sha256": _sha256(social_manifest_path),
                "rows": social_rows,
                "hours": float(social["selected_hours"]),
                "dedupe_mode": "normalized_pcm_exact",
                "public_overlap_mode": "normalized_pcm_exact",
                "near_duplicate_complete": False,
            },
        },
        "cross_pool_dedupe": {
            "mode": "source_identity_plus_known_corpus_exclusion",
            "content_fingerprint_complete": False,
            "stage179": stage179,
            "supplemental_sources": sorted(base_sources | social_sources),
            "source_sets_disjoint": True,
            "known_overlap_exclusions": ["llaso_gigaspeech", "llaso_librispeech"],
            "social_normalized_pcm_exact_complete": True,
            "social_public_overlap_mode": "normalized_pcm_exact",
            "near_duplicate_complete": False,
        },
        "excluded_sources": {
            **base.get("excluded_sources", {}),
            "social_near_duplicates": (
                "not claimed by normalized-PCM exact filtering; lossy re-encoding, partial "
                "embedding, time shift, and changed VAD boundaries remain outside the proof"
            ),
        },
        "dedupe": {
            "algorithm": (
                "component_bound_base_source_identity_plus_social_normalized_pcm_exact"
            ),
            "accepted_unique_rows": total_rows,
            "base_duplicates_by_source": base.get("dedupe", {}).get(
                "duplicates_by_source", {}
            ),
            "social_exact_duplicate_groups": int(
                social.get("dedupe", {}).get("exact_duplicate_groups", 0)
            ),
            "social_duplicate_rejected_rows": int(
                social.get("dedupe", {}).get("rejected_rows", 0)
            ),
            "social_public_overlap_rejected_rows": int(
                social.get("public_overlap", {}).get("rejected_rows", 0)
            ),
        },
        "selected_rows": total_rows,
        "selected_hours": selected_hours,
        "selected_counts_by_source": dict(sorted(source_counts.items())),
        "selected_hours_by_source": dict(sorted(source_hours.items())),
        "rejections": {
            "base": base.get("rejections", {}),
            "social_exact_duplicates": int(social["dedupe"]["rejected_rows"]),
            "social_public_overlap": int(social["public_overlap"]["rejected_rows"]),
        },
        "archive_records": base["archive_records"],
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": "pending",
        "part_records": part_records,
        "elapsed_seconds": time.time() - started,
    }
    _write_atomic_directory(
        output_root=output_root,
        manifest=manifest,
        inventory=inventory,
    )
    validated = validate_stage211_supplemental_inventory(
        output_inventory,
        require_training_ready=True,
        verify_part_sha256=True,
    )
    return validated["inventory"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Stage211 base-natural plus filtered-social combined inventory."
    )
    parser.add_argument("--base-inventory", type=Path, default=DEFAULT_BASE_INVENTORY)
    parser.add_argument("--social-inventory", type=Path, default=DEFAULT_SOCIAL_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_combined_inventory(
        base_inventory_path=args.base_inventory,
        social_inventory_path=args.social_inventory,
        output_root=args.output_root,
    )
    print(
        f"combined_inventory={args.output_root.expanduser().resolve() / 'supplemental_inventory.json'} "
        f"rows={result['selected_rows']} hours={result['selected_hours']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
