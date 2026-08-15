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
    Path.home() / "rwkvasr_data" / "stage211_supplemental_combined_v2"
)
DEFAULT_STAGE211_SUPPLEMENTAL_INVENTORY = (
    DEFAULT_STAGE211_SUPPLEMENTAL_ROOT / "supplemental_inventory.json"
)
STAGE211_BASE_SUPPLEMENTAL_SOURCES = {
    "ljspeech",
    "mls_english",
    "peoples_speech_clean",
    "peoples_speech_dirty",
    "vctk",
}
STAGE211_SUPPLEMENTAL_SOURCES = STAGE211_BASE_SUPPLEMENTAL_SOURCES
STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER = "manifest_location_index_archive_order_v1"
STAGE211_USB_PENDING_NATURAL_ADMISSION = {
    "LLaSO-Align",
    "MLCommons",
    "clean_vocals",
    "new_video.tar",
    "videos",
    "videos_bilibili5",
    "videos_bilibili6",
    "videos_bilibili7",
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


def _validate_combined_component_inventories(
    inventory: dict[str, Any],
    *,
    verify_part_sha256: bool,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    components = inventory.get("component_inventories")
    if not isinstance(components, dict) or set(components) != {"base_natural", "social_vad"}:
        raise ValueError("Stage211 combined supplemental component inventory proof is invalid.")

    base_record = components["base_natural"]
    social_record = components["social_vad"]
    if not isinstance(base_record, dict) or not isinstance(social_record, dict):
        raise ValueError("Stage211 combined supplemental component binding is invalid.")
    base_path = _bound_file(
        base_record,
        path_key="inventory_path",
        sha256_key="inventory_sha256",
        label="Stage211 base supplemental inventory",
        verify_sha256=True,
    )
    base = validate_stage211_supplemental_inventory(
        base_path,
        require_training_ready=True,
        verify_part_sha256=verify_part_sha256,
    )
    if base["inventory"].get("schema_version") != 1:
        raise ValueError("Stage211 combined base component must be the version-1 natural pool.")
    if (
        int(base_record.get("rows", -1)) != int(base["rows"])
        or not math.isclose(
            float(base_record.get("hours", float("nan"))),
            float(base["hours"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or base_record.get("bucket_manifest_sha256") != base["bucket_manifest_sha256"]
    ):
        raise ValueError("Stage211 combined base component totals changed.")

    social_path = _bound_file(
        social_record,
        path_key="inventory_path",
        sha256_key="inventory_sha256",
        label="Stage211 filtered social inventory",
        verify_sha256=True,
    )
    social = _load_json(social_path, label="Stage211 filtered social inventory")
    expected_social = {
        "schema_version": 1,
        "artifact": "stage211_social_pcm_filtered_inventory",
        "complete": True,
        "training_ready": True,
        "admission_state": "normalized_pcm_exact_filtered",
        "storage_kinds": ["tar"],
        "language": "zh",
        "uses_text_labels": False,
        "near_duplicate_detection_complete": False,
    }
    if any(social.get(key) != value for key, value in expected_social.items()):
        raise ValueError("Stage211 filtered social component contract mismatch.")
    if (
        social.get("dedupe", {}).get("mode") != "normalized_pcm_exact"
        or social.get("public_overlap", {}).get("mode") != "normalized_pcm_exact"
        or social.get("public_overlap", {}).get("near_duplicate_complete") is not False
    ):
        raise ValueError("Stage211 filtered social exact-content proof changed.")
    social_rows = int(social.get("selected_rows", -1))
    social_hours = float(social.get("selected_hours", float("nan")))
    social_sources = set((social.get("selected_counts_by_source") or {}).keys())
    if (
        social_rows <= 0
        or not math.isfinite(social_hours)
        or social_hours <= 0.0
        or not social_sources
        or any(not source.startswith("social_") for source in social_sources)
        or int(social_record.get("rows", -1)) != social_rows
        or not math.isclose(
            float(social_record.get("hours", float("nan"))),
            social_hours,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Stage211 filtered social component totals are invalid.")
    social_manifest = _bound_file(
        social,
        path_key="bucket_manifest_path",
        sha256_key="bucket_manifest_sha256",
        label="Stage211 filtered social bucket manifest",
        verify_sha256=True,
    )
    if social_record.get("bucket_manifest_sha256") != _sha256_file(social_manifest):
        raise ValueError("Stage211 combined social component manifest changed.")
    loaded_social_manifest = load_webdataset_bucket_manifest(social_manifest)
    if (
        sum(bucket.num_samples for bucket in loaded_social_manifest.splits.get("train", ()))
        != social_rows
        or sum(bucket.num_samples for bucket in loaded_social_manifest.splits.get("eval", ()))
        != 256
    ):
        raise ValueError("Stage211 filtered social component manifest totals changed.")
    return base, social, social_sources


def _validate_combined_base_public_overlap(
    inventory: dict[str, Any],
    *,
    base: dict[str, Any],
) -> None:
    record = inventory.get("base_public_overlap_audit")
    if not isinstance(record, dict):
        raise ValueError("Stage211 combined inventory lacks base public-overlap proof.")
    receipt_path = _bound_file(
        record,
        path_key="receipt_path",
        sha256_key="receipt_sha256",
        label="Stage211 base public-overlap audit",
        verify_sha256=True,
    )
    audit = _load_json(receipt_path, label="Stage211 base public-overlap audit")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stage211_base_public_pcm_overlap_audit",
        "complete": True,
        "training_ready": True,
        "admission_state": "normalized_pcm_exact_public_clear",
        "comparison_mode": "normalized_pcm_exact",
        "scan_order": STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER,
        "near_duplicate_complete": False,
        "decode_failures": 0,
        "public_overlap_rows": 0,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 base public-overlap audit is incomplete.")
    if (
        audit.get("base_inventory_path") != base["inventory_path"]
        or audit.get("base_inventory_sha256") != base["inventory_sha256"]
        or audit.get("base_bucket_manifest_path") != base["bucket_manifest_path"]
        or audit.get("base_bucket_manifest_sha256") != base["bucket_manifest_sha256"]
        or int(audit.get("scanned_rows", -1)) != int(base["rows"])
        or not math.isclose(
            float(audit.get("scanned_hours", float("nan"))),
            float(base["hours"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or record.get("comparison_mode") != "normalized_pcm_exact"
        or record.get("scan_order") != STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER
        or int(record.get("scanned_rows", -1)) != int(base["rows"])
        or not math.isclose(
            float(record.get("scanned_hours", float("nan"))),
            float(base["hours"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or int(record.get("public_overlap_rows", -1)) != 0
        or record.get("training_ready") is not True
        or record.get("near_duplicate_complete") is not False
    ):
        raise ValueError("Stage211 base public-overlap audit binding changed.")


def _validate_combined_usb_coverage(
    inventory: dict[str, Any],
    *,
    social: dict[str, Any],
) -> None:
    usb_record = inventory.get("usb_top_level_coverage")
    overlap_record = inventory.get("archived_social_exclusion")
    resolution = inventory.get("usb_natural_audio_resolution")
    if (
        not isinstance(usb_record, dict)
        or not isinstance(overlap_record, dict)
        or not isinstance(resolution, dict)
    ):
        raise ValueError("Stage211 combined inventory lacks USB-wide coverage proof.")
    usb_path = _bound_file(
        usb_record,
        path_key="receipt_path",
        sha256_key="receipt_sha256",
        label="Stage211 USB top-level coverage receipt",
        verify_sha256=True,
    )
    usb = _load_json(usb_path, label="Stage211 USB top-level coverage receipt")
    if (
        usb.get("schema_version") != 1
        or usb.get("pipeline") != "stage211"
        or usb.get("artifact") != "usb_top_level_coverage"
        or usb.get("classification_complete") is not True
        or usb.get("training_coverage_complete") is not False
        or set(usb.get("pending_natural_admission") or [])
        != STAGE211_USB_PENDING_NATURAL_ADMISSION
        or usb_record.get("classification_complete") is not True
        or int(usb_record.get("top_level_entry_count", -1))
        != int(usb.get("top_level_entry_count", -2))
        or set(usb_record.get("pending_natural_admission") or [])
        != STAGE211_USB_PENDING_NATURAL_ADMISSION
    ):
        raise ValueError("Stage211 USB top-level coverage proof changed.")
    overlap_path = _bound_file(
        overlap_record,
        path_key="receipt_path",
        sha256_key="receipt_sha256",
        label="Stage211 archived-social overlap receipt",
        verify_sha256=True,
    )
    overlap = _load_json(overlap_path, label="Stage211 archived-social overlap receipt")
    if (
        overlap.get("schema_version") != 1
        or overlap.get("pipeline") != "stage211"
        or overlap.get("artifact") != "archived_social_overlap"
        or overlap.get("complete") is not True
        or overlap.get("comparison_mode")
        != "exact_member_bytes_sha256_against_existing_social_path"
        or overlap.get("all_members_exact_existing_social_duplicates") is not True
        or overlap.get("archive_excluded_from_training_as_duplicate") is not True
        or overlap.get("requires_unique_member_vad_pipeline") is not False
        or int(overlap.get("unique_members", -1)) != 0
        or int(overlap.get("archive_audio_members", -1)) <= 0
        or int(overlap.get("exact_duplicate_members", -1))
        != int(overlap.get("archive_audio_members", -2))
        or overlap.get("usb_top_level_coverage_receipt_path") != str(usb_path)
        or overlap.get("usb_top_level_coverage_receipt_sha256") != _sha256_file(usb_path)
    ):
        raise ValueError("Stage211 archived-social exact-duplicate proof changed.")
    materialized_path = Path(str(social.get("materialized_inventory_path") or "")).resolve()
    materialized = _load_json(
        materialized_path,
        label="Stage211 social materialized inventory",
    )
    if (
        _sha256_file(materialized_path) != social.get("materialized_inventory_sha256")
        or overlap.get("source_inventory_path")
        != materialized.get("source_inventory_path")
        or overlap.get("source_inventory_sha256")
        != materialized.get("source_inventory_sha256")
        or overlap_record.get("archive_path") != overlap.get("archive_path")
        or overlap_record.get("archive_sha256") != overlap.get("archive_sha256")
        or int(overlap_record.get("audio_members", -1))
        != int(overlap.get("archive_audio_members", -2))
        or int(overlap_record.get("audio_bytes", -1))
        != int(overlap.get("archive_audio_bytes", -2))
        or int(overlap_record.get("exact_duplicate_members", -1))
        != int(overlap.get("exact_duplicate_members", -2))
        or overlap_record.get("all_members_exact_existing_social_duplicates") is not True
        or int(overlap_record.get("unique_members", -1)) != 0
    ):
        raise ValueError("Stage211 archived-social exclusion binding changed.")
    expected_resolution = {
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
    }
    if resolution != expected_resolution:
        raise ValueError("Stage211 USB natural-audio resolution is incomplete.")


def validate_stage211_supplemental_inventory(
    inventory_path: str | Path,
    *,
    require_training_ready: bool = True,
    verify_part_sha256: bool = False,
) -> dict[str, Any]:
    inventory_path = Path(inventory_path).expanduser().resolve()
    inventory = _load_json(inventory_path, label="Stage211 supplemental inventory")
    combined = (
        inventory.get("schema_version") == 2
        and inventory.get("artifact") == "stage211_supplemental_combined_inventory"
    )
    expected_fields = (
        {
            "schema_version": 2,
            "artifact": "stage211_supplemental_combined_inventory",
            "complete": True,
            "hash_archives": True,
            "require_production_layout": True,
            "language": ["en", "zh"],
            "uses_text_labels": False,
            "storage_kinds": ["parquet", "tar", "zip"],
        }
        if combined
        else {
            "schema_version": 1,
            "artifact": "stage211_supplemental_natural_inventory",
            "complete": True,
            "hash_archives": True,
            "require_production_layout": True,
            "language": "en",
            "uses_text_labels": False,
            "storage_kinds": ["parquet", "zip"],
        }
    )
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
    component_base: dict[str, Any] | None = None
    component_social: dict[str, Any] | None = None
    social_sources: set[str] = set()
    if combined:
        component_base, component_social, social_sources = (
            _validate_combined_component_inventories(
                inventory,
                verify_part_sha256=verify_part_sha256,
            )
        )
        _validate_combined_usb_coverage(inventory, social=component_social)
        _validate_combined_base_public_overlap(inventory, base=component_base)
    expected_sources = (
        STAGE211_BASE_SUPPLEMENTAL_SOURCES | social_sources
        if combined
        else STAGE211_BASE_SUPPLEMENTAL_SOURCES
    )
    if (
        selected_rows <= 0
        or not math.isfinite(selected_hours)
        or selected_hours <= 0.0
        or not isinstance(source_counts, dict)
        or not isinstance(source_hours, dict)
        or set(source_counts) != expected_sources
        or set(source_hours) != expected_sources
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
    if combined:
        assert component_base is not None and component_social is not None
        expected_rows = int(component_base["rows"]) + int(component_social["selected_rows"])
        expected_hours = float(component_base["hours"]) + float(
            component_social["selected_hours"]
        )
        if selected_rows != expected_rows or not math.isclose(
            selected_hours,
            expected_hours,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Stage211 combined supplemental component sum changed.")
    dedupe = inventory.get("dedupe")
    if (
        not isinstance(dedupe, dict)
        or dedupe.get("algorithm")
        != (
            "component_bound_base_source_identity_plus_social_normalized_pcm_exact"
            if combined
            else "blake2b16(corpus + NUL + source_identity)"
        )
        or int(dedupe.get("accepted_unique_rows", -1)) != selected_rows
    ):
        raise ValueError("Stage211 supplemental source-identity dedupe proof is invalid.")
    cross_pool = inventory.get("cross_pool_dedupe")
    if (
        not isinstance(cross_pool, dict)
        or cross_pool.get("mode") != "source_identity_plus_known_corpus_exclusion"
        or cross_pool.get("content_fingerprint_complete") is not False
        or cross_pool.get("source_sets_disjoint") is not True
        or set(cross_pool.get("supplemental_sources") or []) != expected_sources
        or set(cross_pool.get("known_overlap_exclusions") or [])
        != {"llaso_gigaspeech", "llaso_librispeech"}
    ):
        raise ValueError("Stage211 supplemental cross-pool dedupe proof is invalid.")
    if combined and (
        cross_pool.get("base_public_overlap_normalized_pcm_exact_complete") is not True
        or int(cross_pool.get("base_public_overlap_rows", -1)) != 0
        or cross_pool.get("social_normalized_pcm_exact_complete") is not True
        or cross_pool.get("social_public_overlap_mode") != "normalized_pcm_exact"
        or cross_pool.get("archived_social_exact_duplicate_exclusion_complete") is not True
        or cross_pool.get("usb_top_level_classification_complete") is not True
        or cross_pool.get("usb_unresolved_natural_entries") != []
        or cross_pool.get("near_duplicate_complete") is not False
    ):
        raise ValueError("Stage211 combined social dedupe disclosure is invalid.")
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


def validate_stage211_formal_supplemental_profile(
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Reject the obsolete base-only pool at a formal Stage211 training boundary."""
    inventory = profile.get("inventory")
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema_version") != 2
        or inventory.get("artifact") != "stage211_supplemental_combined_inventory"
    ):
        raise ValueError(
            "Formal Stage211 supplemental_natural training requires the completed "
            "schema-v2 combined supplemental inventory; the base-natural inventory "
            "is not a complete fifth segment."
        )
    return profile
