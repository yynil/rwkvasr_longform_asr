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
    STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER,
    validate_stage211_supplemental_inventory,
)

try:
    from scripts.audit_stage211_base_public_pcm_overlap import validate_audit_receipt
    from scripts.filter_stage211_social_pcm_overlap import validate_filtered_inventory
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from audit_stage211_base_public_pcm_overlap import validate_audit_receipt
    from filter_stage211_social_pcm_overlap import validate_filtered_inventory


DEFAULT_BASE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_supplemental_natural_v1/supplemental_inventory.json"
)
DEFAULT_SOCIAL_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_social_vad_filtered_v2/filtered_inventory.json"
)
DEFAULT_BASE_PUBLIC_OVERLAP_AUDIT = (
    Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_v2/audit_receipt.json"
)
DEFAULT_USB_COVERAGE_RECEIPT = (
    Path.home()
    / "rwkvasr_data/stage211_usb_top_level_coverage_v1/coverage_receipt.json"
)
DEFAULT_ARCHIVED_SOCIAL_OVERLAP_RECEIPT = (
    Path.home()
    / "rwkvasr_data/stage211_archived_social_overlap_v1/overlap_receipt.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_supplemental_combined_v3"
COMBINED_ARTIFACT = "stage211_supplemental_combined_inventory"
SCHEMA_VERSION = 2
EXPECTED_USB_PENDING_NATURAL_ADMISSION = {
    "LLaSO-Align",
    "MLCommons",
    "clean_vocals",
    "new_video.tar",
    "videos",
    "videos_bilibili5",
    "videos_bilibili6",
    "videos_bilibili7",
}


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


def _validate_usb_coverage_and_archive_exclusion(
    *,
    usb_coverage_receipt_path: Path,
    archived_social_overlap_receipt_path: Path,
    social: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    usb_coverage_receipt_path = usb_coverage_receipt_path.expanduser().resolve()
    archived_social_overlap_receipt_path = (
        archived_social_overlap_receipt_path.expanduser().resolve()
    )
    usb = _load_json(
        usb_coverage_receipt_path,
        label="Stage211 USB top-level coverage receipt",
    )
    if (
        usb.get("schema_version") != 1
        or usb.get("pipeline") != "stage211"
        or usb.get("artifact") != "usb_top_level_coverage"
        or usb.get("classification_complete") is not True
        or usb.get("training_coverage_complete") is not False
        or set(usb.get("pending_natural_admission") or [])
        != EXPECTED_USB_PENDING_NATURAL_ADMISSION
    ):
        raise ValueError("Stage211 USB top-level coverage proof changed.")
    overlap = _load_json(
        archived_social_overlap_receipt_path,
        label="Stage211 archived-social overlap receipt",
    )
    expected_overlap = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "archived_social_overlap",
        "complete": True,
        "training_ready": False,
        "comparison_mode": "exact_member_bytes_sha256_against_existing_social_path",
        "unique_members": 0,
        "all_members_exact_existing_social_duplicates": True,
        "archive_excluded_from_training_as_duplicate": True,
        "requires_unique_member_vad_pipeline": False,
        "global_acoustic_near_duplicate_complete": False,
    }
    if any(overlap.get(key) != value for key, value in expected_overlap.items()):
        raise ValueError("Stage211 archived-social exact-duplicate proof is incomplete.")
    if (
        int(overlap.get("archive_audio_members", -1)) <= 0
        or int(overlap.get("exact_duplicate_members", -1))
        != int(overlap["archive_audio_members"])
        or len(str(overlap.get("archive_sha256") or "")) != 64
        or overlap.get("usb_top_level_coverage_receipt_path")
        != str(usb_coverage_receipt_path)
        or overlap.get("usb_top_level_coverage_receipt_sha256")
        != _sha256(usb_coverage_receipt_path)
    ):
        raise ValueError("Stage211 archived-social member coverage proof changed.")
    archived_evidence = usb.get("evidence", {}).get("archived_social", {})
    if (
        overlap.get("archive_path") != archived_evidence.get("archive_path")
        or int(overlap.get("archive_size_bytes", -1))
        != int(archived_evidence.get("archive_size_bytes", -2))
    ):
        raise ValueError("Stage211 archived-social tar differs from the USB coverage proof.")
    materialized_path = Path(str(social.get("materialized_inventory_path") or "")).resolve()
    materialized = _load_json(
        materialized_path,
        label="Stage211 social materialized inventory",
    )
    if (
        _sha256(materialized_path) != social.get("materialized_inventory_sha256")
        or overlap.get("source_inventory_path")
        != materialized.get("source_inventory_path")
        or overlap.get("source_inventory_sha256")
        != materialized.get("source_inventory_sha256")
    ):
        raise ValueError(
            "Stage211 archived-social overlap proof uses another social source inventory."
        )
    return usb, overlap


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


def _validate_reusable_combined_inventory(
    *,
    output_inventory: Path,
    base_inventory_path: Path,
    base_public_overlap_audit_path: Path,
    social_inventory_path: Path,
    usb_coverage_receipt_path: Path,
    archived_social_overlap_receipt_path: Path,
) -> dict[str, Any]:
    inventory = validate_stage211_supplemental_inventory(
        output_inventory,
        require_training_ready=True,
        verify_part_sha256=True,
    )["inventory"]
    components = inventory.get("component_inventories")
    if not isinstance(components, dict):
        raise ValueError("Stage211 existing combined output lacks component bindings.")
    requested_bindings = (
        (
            "base-natural inventory",
            components.get("base_natural"),
            "inventory_path",
            "inventory_sha256",
            base_inventory_path,
        ),
        (
            "base/public overlap audit",
            inventory.get("base_public_overlap_audit"),
            "receipt_path",
            "receipt_sha256",
            base_public_overlap_audit_path,
        ),
        (
            "social inventory",
            components.get("social_vad"),
            "inventory_path",
            "inventory_sha256",
            social_inventory_path,
        ),
        (
            "USB top-level coverage receipt",
            inventory.get("usb_top_level_coverage"),
            "receipt_path",
            "receipt_sha256",
            usb_coverage_receipt_path,
        ),
        (
            "archived-social overlap receipt",
            inventory.get("archived_social_exclusion"),
            "receipt_path",
            "receipt_sha256",
            archived_social_overlap_receipt_path,
        ),
    )
    for label, record, path_key, sha256_key, requested_path in requested_bindings:
        if (
            not isinstance(record, dict)
            or record.get(path_key) != str(requested_path)
            or record.get(sha256_key) != _sha256(requested_path)
        ):
            raise ValueError(
                f"Stage211 existing combined output does not bind the requested {label}."
            )
    return inventory


def build_combined_inventory(
    *,
    base_inventory_path: Path,
    base_public_overlap_audit_path: Path,
    social_inventory_path: Path,
    usb_coverage_receipt_path: Path,
    archived_social_overlap_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    started = time.time()
    base_inventory_path = base_inventory_path.expanduser().resolve()
    base_public_overlap_audit_path = base_public_overlap_audit_path.expanduser().resolve()
    social_inventory_path = social_inventory_path.expanduser().resolve()
    usb_coverage_receipt_path = usb_coverage_receipt_path.expanduser().resolve()
    archived_social_overlap_receipt_path = (
        archived_social_overlap_receipt_path.expanduser().resolve()
    )
    output_root = output_root.expanduser().resolve()
    output_inventory = output_root / "supplemental_inventory.json"
    if output_root.exists():
        if not output_inventory.is_file():
            raise ValueError(
                f"Stage211 combined output exists without its inventory: {output_root}"
            )
        return _validate_reusable_combined_inventory(
            output_inventory=output_inventory,
            base_inventory_path=base_inventory_path,
            base_public_overlap_audit_path=base_public_overlap_audit_path,
            social_inventory_path=social_inventory_path,
            usb_coverage_receipt_path=usb_coverage_receipt_path,
            archived_social_overlap_receipt_path=archived_social_overlap_receipt_path,
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)

    base_validated = validate_stage211_supplemental_inventory(
        base_inventory_path,
        require_training_ready=True,
        verify_part_sha256=True,
    )
    base = base_validated["inventory"]
    if base.get("schema_version") != 1:
        raise ValueError("Stage211 combined builder requires the version-1 base natural pool.")
    base_public_overlap_audit = validate_audit_receipt(
        base_public_overlap_audit_path,
        require_training_ready=True,
        verify_overlap_replay=True,
    )
    if (
        base_public_overlap_audit.get("base_inventory_path") != str(base_inventory_path)
        or base_public_overlap_audit.get("base_inventory_sha256")
        != base_validated["inventory_sha256"]
        or int(base_public_overlap_audit.get("scanned_rows", -1))
        != int(base_validated["rows"])
        or not math.isclose(
            float(base_public_overlap_audit.get("scanned_hours", float("nan"))),
            float(base_validated["hours"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or int(base_public_overlap_audit.get("public_overlap_rows", -1)) != 0
        or base_public_overlap_audit.get("scan_order")
        != STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER
    ):
        raise ValueError("Stage211 base public-overlap audit does not bind this base pool.")
    social_validated = validate_filtered_inventory(
        social_inventory_path,
        verify_part_sha256=True,
    )
    social = social_validated["inventory"]
    usb_coverage, archived_social_overlap = _validate_usb_coverage_and_archive_exclusion(
        usb_coverage_receipt_path=usb_coverage_receipt_path,
        archived_social_overlap_receipt_path=archived_social_overlap_receipt_path,
        social=social,
    )
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
        "base_public_overlap_audit": {
            "receipt_path": str(base_public_overlap_audit_path),
            "receipt_sha256": _sha256(base_public_overlap_audit_path),
            "comparison_mode": "normalized_pcm_exact",
            "scan_order": STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER,
            "scanned_rows": int(base_public_overlap_audit["scanned_rows"]),
            "scanned_hours": float(base_public_overlap_audit["scanned_hours"]),
            "public_overlap_rows": 0,
            "training_ready": True,
            "near_duplicate_complete": False,
        },
        "usb_top_level_coverage": {
            "receipt_path": str(usb_coverage_receipt_path),
            "receipt_sha256": _sha256(usb_coverage_receipt_path),
            "classification_complete": True,
            "top_level_entry_count": int(usb_coverage["top_level_entry_count"]),
            "pending_natural_admission": sorted(EXPECTED_USB_PENDING_NATURAL_ADMISSION),
        },
        "archived_social_exclusion": {
            "receipt_path": str(archived_social_overlap_receipt_path),
            "receipt_sha256": _sha256(archived_social_overlap_receipt_path),
            "archive_path": archived_social_overlap["archive_path"],
            "archive_sha256": archived_social_overlap["archive_sha256"],
            "audio_members": int(archived_social_overlap["archive_audio_members"]),
            "audio_bytes": int(archived_social_overlap["archive_audio_bytes"]),
            "exact_duplicate_members": int(
                archived_social_overlap["exact_duplicate_members"]
            ),
            "all_members_exact_existing_social_duplicates": True,
            "unique_members": 0,
        },
        "usb_natural_audio_resolution": {
            "complete": True,
            "admitted_by_base_natural": ["LLaSO-Align", "MLCommons"],
            "admitted_by_social_vad": [
                "clean_vocals",
                "videos",
                "videos_bilibili5",
                "videos_bilibili6",
                "videos_bilibili7",
            ],
            "exact_duplicate_excluded": ["new_video.tar"],
            "unresolved_entries": [],
        },
        "cross_pool_dedupe": {
            "mode": "source_identity_plus_known_corpus_exclusion",
            "content_fingerprint_complete": False,
            "stage179": stage179,
            "supplemental_sources": sorted(base_sources | social_sources),
            "source_sets_disjoint": True,
            "known_overlap_exclusions": ["llaso_gigaspeech", "llaso_librispeech"],
            "base_public_overlap_normalized_pcm_exact_complete": True,
            "base_public_overlap_rows": 0,
            "social_normalized_pcm_exact_complete": True,
            "social_public_overlap_mode": "normalized_pcm_exact",
            "archived_social_exact_duplicate_exclusion_complete": True,
            "usb_top_level_classification_complete": True,
            "usb_unresolved_natural_entries": [],
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
    parser.add_argument(
        "--base-public-overlap-audit",
        type=Path,
        default=DEFAULT_BASE_PUBLIC_OVERLAP_AUDIT,
    )
    parser.add_argument("--social-inventory", type=Path, default=DEFAULT_SOCIAL_INVENTORY)
    parser.add_argument(
        "--usb-coverage-receipt",
        type=Path,
        default=DEFAULT_USB_COVERAGE_RECEIPT,
    )
    parser.add_argument(
        "--archived-social-overlap-receipt",
        type=Path,
        default=DEFAULT_ARCHIVED_SOCIAL_OVERLAP_RECEIPT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_combined_inventory(
        base_inventory_path=args.base_inventory,
        base_public_overlap_audit_path=args.base_public_overlap_audit,
        social_inventory_path=args.social_inventory,
        usb_coverage_receipt_path=args.usb_coverage_receipt,
        archived_social_overlap_receipt_path=args.archived_social_overlap_receipt,
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
