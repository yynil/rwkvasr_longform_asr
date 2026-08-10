from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from rwkvasr.data import load_webdataset_bucket_manifest
from rwkvasr.eval.stage211_supplemental import (
    STAGE211_BASE_SUPPLEMENTAL_SOURCES,
    stage211_supplemental_profile,
    validate_stage211_supplemental_inventory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
combined = importlib.import_module("scripts.build_stage211_combined_supplemental_inventory")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _part(path: Path, *, rows: int, source: str) -> dict[str, object]:
    path.write_text(
        "".join(json.dumps({"key": f"{source}-{index}"}) + "\n" for index in range(rows)),
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "num_samples": rows,
        "source_label": source,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fixed_eval(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    part = tmp_path / "fixed_eval.jsonl"
    part.write_text("{}\n" * 256, encoding="utf-8")
    split = {
        "num_samples": 256,
        "buckets": [
            {
                "bucket_id": 0,
                "num_samples": 256,
                "parts": [{"path": str(part), "num_samples": 256}],
            }
        ],
    }
    manifest = tmp_path / "fixed_eval_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "root": "/",
                "source_length_index_path": str(part),
                "splits": {"eval": split},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    binding = {
        "rows": 256,
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "parts": [
            {"path": str(part), "sha256": _sha256(part), "num_samples": 256}
        ],
    }
    return part, manifest, split, binding


def _manifest(
    path: Path,
    *,
    parts: list[dict[str, object]],
    eval_split: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "root": "/",
                "source_length_index_path": str(path.with_suffix(".inventory.json")),
                "bucket_width": 80,
                "entries_per_part": 100_000,
                "splits": {
                    "train": {
                        "num_samples": sum(int(part["num_samples"]) for part in parts),
                        "buckets": [
                            {
                                "bucket_id": 1,
                                "num_samples": sum(
                                    int(part["num_samples"]) for part in parts
                                ),
                                "parts": parts,
                            }
                        ],
                    },
                    "eval": eval_split,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _base_inventory(
    tmp_path: Path,
    *,
    eval_split: dict[str, object],
    fixed_binding: dict[str, object],
) -> Path:
    root = tmp_path / "base"
    root.mkdir()
    parts = [
        _part(root / f"{source}.jsonl", rows=1, source=source)
        for source in sorted(STAGE211_BASE_SUPPLEMENTAL_SOURCES)
    ]
    manifest = root / "manifest.json"
    _manifest(manifest, parts=parts, eval_split=eval_split)
    stage179 = root / "stage179.json"
    stage179.write_text("{}\n", encoding="utf-8")
    hours_by_source = {
        source: 1.0 / 3600.0 for source in STAGE211_BASE_SUPPLEMENTAL_SOURCES
    }
    inventory = {
        "schema_version": 1,
        "artifact": "stage211_supplemental_natural_inventory",
        "complete": True,
        "training_ready": True,
        "hash_archives": True,
        "require_production_layout": True,
        "language": "en",
        "uses_text_labels": False,
        "storage_kinds": ["parquet", "zip"],
        "layout_errors": [],
        "selected_rows": 5,
        "selected_hours": sum(hours_by_source.values()),
        "selected_counts_by_source": {
            source: 1 for source in STAGE211_BASE_SUPPLEMENTAL_SOURCES
        },
        "selected_hours_by_source": hours_by_source,
        "dedupe": {
            "algorithm": "blake2b16(corpus + NUL + source_identity)",
            "accepted_unique_rows": 5,
            "duplicates_by_source": {},
        },
        "cross_pool_dedupe": {
            "mode": "source_identity_plus_known_corpus_exclusion",
            "content_fingerprint_complete": False,
            "source_sets_disjoint": True,
            "supplemental_sources": sorted(STAGE211_BASE_SUPPLEMENTAL_SOURCES),
            "known_overlap_exclusions": ["llaso_gigaspeech", "llaso_librispeech"],
            "stage179": {
                "expected_source_set_match": True,
                "manifest_path": str(stage179),
                "manifest_sha256": _sha256(stage179),
                "sources": ["stage179_fixture"],
            },
        },
        "fixed_eval": fixed_binding,
        "excluded_sources": {},
        "rejections": {},
        "bucket_manifest_path": str(manifest),
        "bucket_manifest_sha256": _sha256(manifest),
        "part_records": parts,
        "archive_records": [{"status": "absent"}],
    }
    path = root / "supplemental_inventory.json"
    path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")
    return path


def _social_inventory(
    tmp_path: Path,
    *,
    eval_split: dict[str, object],
    fixed_binding: dict[str, object],
) -> Path:
    root = tmp_path / "social"
    root.mkdir()
    part = _part(root / "social.jsonl", rows=2, source="social_videos_mp3")
    manifest = root / "manifest.json"
    _manifest(manifest, parts=[part], eval_split=eval_split)
    materialized = root / "materialized.json"
    materialized.write_text("{}\n", encoding="utf-8")
    fingerprint_part = root / "fingerprints.jsonl"
    fingerprint_part.write_text("{}\n" * 2, encoding="utf-8")
    fingerprint_receipt = root / "fingerprints.receipt.json"
    fingerprint_receipt.write_text("{}\n", encoding="utf-8")
    fingerprint_record = {
        "path": str(fingerprint_receipt),
        "sha256": _sha256(fingerprint_receipt),
        "part_path": str(fingerprint_part),
        "part_size_bytes": fingerprint_part.stat().st_size,
        "part_sha256": _sha256(fingerprint_part),
    }
    duplicate_exclusions = root / "duplicate_exclusions.jsonl"
    public_exclusions = root / "public_exclusions.jsonl"
    duplicate_exclusions.write_bytes(b"")
    public_exclusions.write_bytes(b"")
    selected_hours = 2.0 / 3600.0
    inventory = {
        "schema_version": 1,
        "artifact": "stage211_social_pcm_filtered_inventory",
        "complete": True,
        "training_ready": True,
        "admission_state": "normalized_pcm_exact_filtered",
        "fingerprint_algorithm": (
            "sha256(le_u32_sr + le_u32_channels + le_u64_samples + pcm_s16le)"
        ),
        "near_duplicate_detection_complete": False,
        "storage_kinds": ["tar"],
        "language": "zh",
        "uses_text_labels": False,
        "materialized_inventory_path": str(materialized),
        "materialized_inventory_sha256": _sha256(materialized),
        "prefilter_rows": 2,
        "prefilter_hours": selected_hours,
        "selected_rows": 2,
        "selected_hours": selected_hours,
        "selected_counts_by_source": {"social_videos_mp3": 2},
        "selected_hours_by_source": {"social_videos_mp3": selected_hours},
        "dedupe": {
            "mode": "normalized_pcm_exact",
            "exact_duplicate_groups": 0,
            "rejected_rows": 0,
        },
        "public_overlap": {
            "mode": "normalized_pcm_exact",
            "rejected_rows": 0,
            "near_duplicate_complete": False,
        },
        "fixed_eval": fixed_binding,
        "source_fingerprint_receipts": [fingerprint_record],
        "public_fingerprint_receipts": [fingerprint_record],
        "bucket_manifest_path": str(manifest),
        "bucket_manifest_sha256": _sha256(manifest),
        "part_records": [part],
        "outputs": {
            "duplicate_exclusions": {
                "path": str(duplicate_exclusions),
                "sha256": _sha256(duplicate_exclusions),
            },
            "public_overlap_exclusions": {
                "path": str(public_exclusions),
                "sha256": _sha256(public_exclusions),
            },
        },
    }
    path = root / "filtered_inventory.json"
    path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")
    return path


def test_combined_supplemental_inventory_binds_both_components(tmp_path: Path) -> None:
    _, _, eval_split, fixed_binding = _fixed_eval(tmp_path)
    base = _base_inventory(
        tmp_path,
        eval_split=eval_split,
        fixed_binding=fixed_binding,
    )
    social = _social_inventory(
        tmp_path,
        eval_split=eval_split,
        fixed_binding=fixed_binding,
    )
    output = tmp_path / "combined"

    result = combined.build_combined_inventory(
        base_inventory_path=base,
        social_inventory_path=social,
        output_root=output,
    )
    assert result["schema_version"] == 2
    assert result["artifact"] == "stage211_supplemental_combined_inventory"
    assert result["selected_rows"] == 7
    assert result["storage_kinds"] == ["parquet", "tar", "zip"]
    assert result["language"] == ["en", "zh"]
    assert result["cross_pool_dedupe"]["content_fingerprint_complete"] is False
    assert result["cross_pool_dedupe"]["social_normalized_pcm_exact_complete"] is True
    assert result["cross_pool_dedupe"]["near_duplicate_complete"] is False
    manifest = load_webdataset_bucket_manifest(result["bucket_manifest_path"])
    assert sum(bucket.num_samples for bucket in manifest.splits["train"]) == 7
    assert sum(bucket.num_samples for bucket in manifest.splits["eval"]) == 256
    validated = validate_stage211_supplemental_inventory(
        output / "supplemental_inventory.json",
        verify_part_sha256=True,
    )
    assert validated["rows"] == 7
    profile = stage211_supplemental_profile(
        output / "supplemental_inventory.json",
        epochs=3,
        batch_size=1,
        world_size=4,
        frame_budget=8_000,
    )
    assert profile["row_exposures"] == 21
    assert (
        combined.build_combined_inventory(
            base_inventory_path=base,
            social_inventory_path=social,
            output_root=output,
        )
        == result
    )


def test_combined_inventory_rejects_changed_component(tmp_path: Path) -> None:
    _, _, eval_split, fixed_binding = _fixed_eval(tmp_path)
    base = _base_inventory(
        tmp_path,
        eval_split=eval_split,
        fixed_binding=fixed_binding,
    )
    social = _social_inventory(
        tmp_path,
        eval_split=eval_split,
        fixed_binding=fixed_binding,
    )
    output = tmp_path / "combined"
    combined.build_combined_inventory(
        base_inventory_path=base,
        social_inventory_path=social,
        output_root=output,
    )
    social.write_text(social.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="filtered social inventory SHA-256 changed"):
        validate_stage211_supplemental_inventory(
            output / "supplemental_inventory.json",
            verify_part_sha256=False,
        )
