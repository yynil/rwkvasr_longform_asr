from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from rwkvasr.data import (
    estimate_bucket_manifest_steps,
    estimate_bucket_manifest_tail_padding_samples,
    load_webdataset_bucket_manifest,
)


STAGE211_SUPPLEMENTAL_DIFFICULTY = "supplemental_natural"
DEFAULT_STAGE211_SUPPLEMENTAL_ROOT = (
    Path.home() / "rwkvasr_data" / "stage211_supplemental_natural_v1"
)
DEFAULT_STAGE211_SUPPLEMENTAL_INVENTORY = (
    DEFAULT_STAGE211_SUPPLEMENTAL_ROOT / "supplemental_inventory.json"
)
STAGE211_SUPPLEMENTAL_SOURCES = {
    "ljspeech",
    "mls_english",
    "peoples_speech_clean",
    "peoples_speech_dirty",
    "vctk",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _bound_file(
    record: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    label: str,
    verify_sha256: bool,
) -> Path:
    path = Path(str(record.get(path_key) or "")).expanduser().resolve()
    expected_sha256 = str(record.get(sha256_key) or "")
    if not path.is_file() or _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} binding is missing or invalid: {path}")
    if verify_sha256 and _sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 changed: {path}")
    return path


def validate_stage211_supplemental_inventory(
    inventory_path: str | Path,
    *,
    require_training_ready: bool = True,
    verify_part_sha256: bool = False,
) -> dict[str, Any]:
    inventory_path = Path(inventory_path).expanduser().resolve()
    inventory = _load_json(inventory_path, label="Stage211 supplemental inventory")
    expected_fields = {
        "schema_version": 1,
        "artifact": "stage211_supplemental_natural_inventory",
        "complete": True,
        "hash_archives": True,
        "require_production_layout": True,
        "language": "en",
        "uses_text_labels": False,
        "storage_kinds": ["parquet", "zip"],
    }
    if any(inventory.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Stage211 supplemental inventory contract mismatch.")
    if require_training_ready and inventory.get("training_ready") is not True:
        raise ValueError("Stage211 supplemental inventory is not training-ready.")
    if inventory.get("layout_errors") != []:
        raise ValueError("Stage211 supplemental inventory records production layout errors.")

    selected_rows = int(inventory.get("selected_rows", -1))
    selected_hours = float(inventory.get("selected_hours", float("nan")))
    source_counts = inventory.get("selected_counts_by_source")
    source_hours = inventory.get("selected_hours_by_source")
    try:
        source_count_values = [int(value) for value in (source_counts or {}).values()]
        source_hour_values = [float(value) for value in (source_hours or {}).values()]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Stage211 supplemental selected source totals are invalid.") from error
    if (
        selected_rows <= 0
        or not math.isfinite(selected_hours)
        or selected_hours <= 0.0
        or not isinstance(source_counts, dict)
        or not isinstance(source_hours, dict)
        or set(source_counts) != STAGE211_SUPPLEMENTAL_SOURCES
        or set(source_hours) != STAGE211_SUPPLEMENTAL_SOURCES
        or any(value <= 0 for value in source_count_values)
        or any(not math.isfinite(value) or value <= 0.0 for value in source_hour_values)
        or sum(source_count_values) != selected_rows
        or not math.isclose(
            sum(source_hour_values),
            selected_hours,
            rel_tol=0.0,
            abs_tol=0.002,
        )
    ):
        raise ValueError("Stage211 supplemental selected source totals are invalid.")
    dedupe = inventory.get("dedupe")
    if (
        not isinstance(dedupe, dict)
        or dedupe.get("algorithm") != "blake2b16(corpus + NUL + source_identity)"
        or int(dedupe.get("accepted_unique_rows", -1)) != selected_rows
    ):
        raise ValueError("Stage211 supplemental source-identity dedupe proof is invalid.")
    cross_pool = inventory.get("cross_pool_dedupe")
    if (
        not isinstance(cross_pool, dict)
        or cross_pool.get("mode") != "source_identity_plus_known_corpus_exclusion"
        or cross_pool.get("content_fingerprint_complete") is not False
        or cross_pool.get("source_sets_disjoint") is not True
        or set(cross_pool.get("supplemental_sources") or []) != STAGE211_SUPPLEMENTAL_SOURCES
        or set(cross_pool.get("known_overlap_exclusions") or [])
        != {"llaso_gigaspeech", "llaso_librispeech"}
    ):
        raise ValueError("Stage211 supplemental cross-pool dedupe proof is invalid.")
    stage179 = cross_pool.get("stage179")
    if not isinstance(stage179, dict) or stage179.get("expected_source_set_match") is not True:
        raise ValueError("Stage211 supplemental inventory lacks the Stage179 source binding.")
    _bound_file(
        stage179,
        path_key="manifest_path",
        sha256_key="manifest_sha256",
        label="Stage211 Stage179 global-dedup manifest",
        verify_sha256=True,
    )

    manifest_path = _bound_file(
        inventory,
        path_key="bucket_manifest_path",
        sha256_key="bucket_manifest_sha256",
        label="Stage211 supplemental bucket manifest",
        verify_sha256=True,
    )
    manifest = load_webdataset_bucket_manifest(manifest_path)
    train_rows = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
    eval_rows = sum(bucket.num_samples for bucket in manifest.splits.get("eval", ()))
    if train_rows != selected_rows or eval_rows != 256:
        raise ValueError(
            "Stage211 supplemental manifest split totals mismatch: "
            f"train={train_rows}/{selected_rows} eval={eval_rows}/256"
        )
    fixed_eval = inventory.get("fixed_eval")
    if not isinstance(fixed_eval, dict) or int(fixed_eval.get("rows", -1)) != 256:
        raise ValueError("Stage211 supplemental inventory lacks fixed-eval provenance.")
    _bound_file(
        fixed_eval,
        path_key="manifest_path",
        sha256_key="manifest_sha256",
        label="Stage211 supplemental fixed-eval source manifest",
        verify_sha256=True,
    )
    fixed_eval_parts = fixed_eval.get("parts")
    if not isinstance(fixed_eval_parts, list) or not fixed_eval_parts:
        raise ValueError("Stage211 supplemental fixed-eval part binding is empty.")
    for record in fixed_eval_parts:
        if not isinstance(record, dict) or int(record.get("num_samples", -1)) <= 0:
            raise ValueError("Stage211 supplemental fixed-eval part binding is invalid.")
        _bound_file(
            record,
            path_key="path",
            sha256_key="sha256",
            label="Stage211 supplemental fixed-eval part",
            verify_sha256=True,
        )

    part_records = inventory.get("part_records")
    if not isinstance(part_records, list) or not part_records:
        raise ValueError("Stage211 supplemental inventory has no train-part records.")
    recorded_part_rows = 0
    for record in part_records:
        if not isinstance(record, dict):
            raise ValueError("Stage211 supplemental train-part binding is invalid.")
        part_path = Path(str(record.get("path") or ""))
        if not part_path.is_absolute():
            part_path = manifest_path.parent / part_path
        part_path = part_path.resolve()
        rows = int(record.get("num_samples", -1))
        expected_sha256 = str(record.get("sha256") or "")
        if (
            rows <= 0
            or not part_path.is_file()
            or int(record.get("size_bytes", -1)) != part_path.stat().st_size
            or _SHA256_PATTERN.fullmatch(expected_sha256) is None
        ):
            raise ValueError(f"Stage211 supplemental train-part binding is invalid: {part_path}")
        if verify_part_sha256 and _sha256_file(part_path) != expected_sha256:
            raise ValueError(f"Stage211 supplemental train part changed: {part_path}")
        recorded_part_rows += rows
    if recorded_part_rows != selected_rows:
        raise ValueError("Stage211 supplemental train-part row total mismatch.")

    archives = inventory.get("archive_records")
    if not isinstance(archives, list) or not archives:
        raise ValueError("Stage211 supplemental source archive inventory is empty.")
    for archive in archives:
        if not isinstance(archive, dict):
            raise ValueError("Stage211 supplemental source archive binding is invalid.")
        status = str(archive.get("status") or "")
        if status == "absent":
            continue
        archive_path = Path(str(archive.get("path") or "")).expanduser().resolve()
        archive_sha256 = str(archive.get("sha256") or "")
        if (
            status not in {"accepted", "invalid"}
            or not archive_path.is_file()
            or int(archive.get("size_bytes", -1)) != archive_path.stat().st_size
            or int(archive.get("mtime_ns", -1)) != archive_path.stat().st_mtime_ns
            or _SHA256_PATTERN.fullmatch(archive_sha256) is None
        ):
            raise ValueError(f"Stage211 supplemental source archive changed: {archive_path}")

    return {
        "inventory": inventory,
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256_file(inventory_path),
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": _sha256_file(manifest_path),
        "rows": selected_rows,
        "hours": selected_hours,
    }


def stage211_supplemental_profile(
    inventory_path: str | Path,
    *,
    epochs: int,
    batch_size: int,
    world_size: int,
    frame_budget: int,
    require_training_ready: bool = True,
    verify_part_sha256: bool = False,
) -> dict[str, Any]:
    validated = validate_stage211_supplemental_inventory(
        inventory_path,
        require_training_ready=require_training_ready,
        verify_part_sha256=verify_part_sha256,
    )
    manifest = load_webdataset_bucket_manifest(validated["bucket_manifest_path"])
    steps_per_epoch = estimate_bucket_manifest_steps(
        manifest,
        split="train",
        batch_size=batch_size,
        world_size=world_size,
        frame_budget=frame_budget,
        drop_last=False,
    )
    tail_padding = estimate_bucket_manifest_tail_padding_samples(
        manifest,
        split="train",
        batch_size=batch_size,
        world_size=world_size,
        frame_budget=frame_budget,
    )
    return {
        **validated,
        "epochs": int(epochs),
        "steps_per_epoch": steps_per_epoch,
        "steps": steps_per_epoch * int(epochs),
        "tail_padding_samples_per_epoch": tail_padding,
        "row_exposures": int(validated["rows"]) * int(epochs),
        "hour_exposures": float(validated["hours"]) * int(epochs),
        "tail_padding_sample_exposures": tail_padding * int(epochs),
        "executed_sample_exposures": (
            int(validated["rows"]) + tail_padding
        )
        * int(epochs),
    }
