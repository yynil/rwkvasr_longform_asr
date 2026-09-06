from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rwkvasr.data import load_webdataset_bucket_manifest
from rwkvasr.eval.stage211_supplemental import (
    STAGE211_BASE_SUPPLEMENTAL_SOURCES,
    stage211_supplemental_profile,
    validate_stage211_formal_supplemental_profile,
    validate_stage211_supplemental_inventory,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
combined = importlib.import_module("scripts.build_stage211_combined_supplemental_inventory")
supplemental_profile_receipt = importlib.import_module(
    "scripts.create_stage211_supplemental_profile_receipt"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_postmaterialization_pipeline_hands_off_after_base_audit(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "calls.log"
    _write_executable(fake_bin / "tmux", "#!/usr/bin/env bash\nexit 1\n")
    _write_executable(
        fake_bin / "uv",
        "#!/usr/bin/env bash\n"
        "printf 'uv %s\\n' \"$*\" >>\"${CALL_LOG}\"\n"
        "mkdir -p \"${BASE_PUBLIC_OVERLAP_ROOT}\"\n"
        "printf '{}\\n' >\"${BASE_PUBLIC_OVERLAP_ROOT}/audit_receipt.json\"\n",
    )
    handoff = tmp_path / "quote-repair-handoff.sh"
    _write_executable(
        handoff,
        "#!/usr/bin/env bash\n"
        "printf 'handoff legacy=%s base=%s\\n' \"${LEGACY_SESSION}\" \"${BASE_SOURCE_AUDIT}\" "
        ">>\"${CALL_LOG}\"\n",
    )
    base_inventory = tmp_path / "base.json"
    usb_receipt = tmp_path / "usb.json"
    archived_receipt = tmp_path / "archived.json"
    for path in (base_inventory, usb_receipt, archived_receipt):
        path.write_text("{}\n", encoding="utf-8")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CALL_LOG": str(call_log),
        "POLL_SECONDS": "1",
        "BASE_INVENTORY": str(base_inventory),
        "USB_COVERAGE_RECEIPT": str(usb_receipt),
        "ARCHIVED_SOCIAL_OVERLAP_RECEIPT": str(archived_receipt),
        "BASE_PUBLIC_OVERLAP_ROOT": str(tmp_path / "base-overlap"),
        "BASE_PUBLIC_PREFETCH_NEXT_ARCHIVE": "1",
        "FILTERED_ROOT": str(tmp_path / "filtered"),
        "COMBINED_ROOT": str(tmp_path / "combined"),
        "QUOTE_REPAIR_HANDOFF_SCRIPT": str(handoff),
    }

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/build_stage211_social_combined_postmaterialization.sh")],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    handoff_index = next(index for index, call in enumerate(calls) if call.startswith("handoff "))
    assert calls[0].startswith("uv run python scripts/audit_stage211_base_public_pcm_overlap.py")
    assert "--prefetch-next-archive" in calls[0]
    assert handoff_index == 1
    assert "legacy=rwkvasr_stage211_no_legacy_session" in calls[handoff_index]
    assert f"base={tmp_path / 'base-overlap' / 'audit_receipt.json'}" in calls[handoff_index]
    assert result.stdout == ""


def test_postmaterialization_base_audit_uses_low_io_priority() -> None:
    source = (
        REPO_ROOT / "scripts/build_stage211_social_combined_postmaterialization.sh"
    ).read_text(encoding="utf-8")

    assert (
        "nice -n 10 ionice -c 2 -n 7 env CUDA_VISIBLE_DEVICES='' uv run python \\\n"
        "  scripts/audit_stage211_base_public_pcm_overlap.py"
    ) in source
    assert 'BASE_PUBLIC_PREFETCH_NEXT_ARCHIVE="${BASE_PUBLIC_PREFETCH_NEXT_ARCHIVE:-0}"' in source


@pytest.fixture(autouse=True)
def _validate_synthetic_base_public_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(path: Path, **_: object) -> dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(combined, "validate_audit_receipt", validate)


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
        "parts": [{"path": str(part), "sha256": _sha256(part), "num_samples": 256}],
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
                                "num_samples": sum(int(part["num_samples"]) for part in parts),
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
    hours_by_source = {source: 1.0 / 3600.0 for source in STAGE211_BASE_SUPPLEMENTAL_SOURCES}
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
        "selected_counts_by_source": {source: 1 for source in STAGE211_BASE_SUPPLEMENTAL_SOURCES},
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
    source_inventory = root / "source_inventory.json"
    source_inventory.write_text("{}\n", encoding="utf-8")
    materialized = root / "materialized.json"
    materialized.write_text(
        json.dumps(
            {
                "source_inventory_path": str(source_inventory),
                "source_inventory_sha256": _sha256(source_inventory),
            }
        )
        + "\n",
        encoding="utf-8",
    )
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


def _usb_proofs(tmp_path: Path, *, social_inventory: Path) -> tuple[Path, Path]:
    archive = tmp_path / "new_video.tar"
    archive.write_bytes(b"archive")
    pending = sorted(combined.EXPECTED_USB_PENDING_NATURAL_ADMISSION)
    coverage = tmp_path / "usb_coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "usb_top_level_coverage",
                "classification_complete": True,
                "training_coverage_complete": False,
                "top_level_entry_count": 27,
                "pending_natural_admission": pending,
                "evidence": {
                    "archived_social": {
                        "archive_path": str(archive),
                        "archive_size_bytes": archive.stat().st_size,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    social = json.loads(social_inventory.read_text(encoding="utf-8"))
    materialized = json.loads(
        Path(social["materialized_inventory_path"]).read_text(encoding="utf-8")
    )
    overlap = tmp_path / "archived_overlap.json"
    overlap.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "archived_social_overlap",
                "complete": True,
                "training_ready": False,
                "archive_path": str(archive),
                "archive_size_bytes": archive.stat().st_size,
                "archive_sha256": _sha256(archive),
                "archive_audio_members": 1,
                "archive_audio_bytes": archive.stat().st_size,
                "source_inventory_path": materialized["source_inventory_path"],
                "source_inventory_sha256": materialized["source_inventory_sha256"],
                "usb_top_level_coverage_receipt_path": str(coverage),
                "usb_top_level_coverage_receipt_sha256": _sha256(coverage),
                "comparison_mode": ("exact_member_bytes_sha256_against_existing_social_path"),
                "exact_duplicate_members": 1,
                "unique_members": 0,
                "all_members_exact_existing_social_duplicates": True,
                "archive_excluded_from_training_as_duplicate": True,
                "requires_unique_member_vad_pipeline": False,
                "global_acoustic_near_duplicate_complete": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return coverage, overlap


def _base_public_audit(tmp_path: Path, *, base_inventory: Path) -> Path:
    base = json.loads(base_inventory.read_text(encoding="utf-8"))
    manifest = Path(base["bucket_manifest_path"])
    path = tmp_path / "base_public_overlap_audit.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "stage211_base_public_pcm_overlap_audit",
                "complete": True,
                "training_ready": True,
                "admission_state": "normalized_pcm_exact_public_clear",
                "comparison_mode": "normalized_pcm_exact",
                "scan_order": "manifest_location_index_archive_order_v1",
                "near_duplicate_complete": False,
                "decode_failures": 0,
                "public_overlap_rows": 0,
                "base_inventory_path": str(base_inventory),
                "base_inventory_sha256": _sha256(base_inventory),
                "base_bucket_manifest_path": str(manifest),
                "base_bucket_manifest_sha256": _sha256(manifest),
                "scanned_rows": int(base["selected_rows"]),
                "scanned_hours": float(base["selected_hours"]),
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
    base_public_audit = _base_public_audit(tmp_path, base_inventory=base)
    coverage, overlap = _usb_proofs(tmp_path, social_inventory=social)
    output = tmp_path / "combined"

    result = combined.build_combined_inventory(
        base_inventory_path=base,
        base_public_overlap_audit_path=base_public_audit,
        social_inventory_path=social,
        usb_coverage_receipt_path=coverage,
        archived_social_overlap_receipt_path=overlap,
        output_root=output,
    )
    assert result["schema_version"] == 2
    assert result["artifact"] == "stage211_supplemental_combined_inventory"
    assert result["selected_rows"] == 7
    assert result["storage_kinds"] == ["parquet", "tar", "zip"]
    assert result["language"] == ["en", "zh"]
    assert result["cross_pool_dedupe"]["content_fingerprint_complete"] is False
    assert result["cross_pool_dedupe"]["base_public_overlap_normalized_pcm_exact_complete"] is True
    assert result["cross_pool_dedupe"]["base_public_overlap_rows"] == 0
    assert result["cross_pool_dedupe"]["social_normalized_pcm_exact_complete"] is True
    assert result["cross_pool_dedupe"]["archived_social_exact_duplicate_exclusion_complete"] is True
    assert result["usb_natural_audio_resolution"]["unresolved_entries"] == []
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
    assert validate_stage211_formal_supplemental_profile(profile) is profile
    base_profile = stage211_supplemental_profile(
        base,
        epochs=3,
        batch_size=1,
        world_size=4,
        frame_budget=8_000,
    )
    with pytest.raises(ValueError, match="schema-v2 combined supplemental inventory"):
        validate_stage211_formal_supplemental_profile(base_profile)
    assert (
        combined.build_combined_inventory(
            base_inventory_path=base,
            base_public_overlap_audit_path=base_public_audit,
            social_inventory_path=social,
            usb_coverage_receipt_path=coverage,
            archived_social_overlap_receipt_path=overlap,
            output_root=output,
        )
        == result
    )


@pytest.mark.parametrize(
    ("argument_name", "label"),
    (
        ("base_inventory_path", "base-natural inventory"),
        ("base_public_overlap_audit_path", "base/public overlap audit"),
        ("social_inventory_path", "social inventory"),
        ("usb_coverage_receipt_path", "USB top-level coverage receipt"),
        ("archived_social_overlap_receipt_path", "archived-social overlap receipt"),
    ),
)
def test_combined_inventory_reuse_requires_exact_requested_inputs(
    tmp_path: Path,
    argument_name: str,
    label: str,
) -> None:
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
    base_public_audit = _base_public_audit(tmp_path, base_inventory=base)
    coverage, overlap = _usb_proofs(tmp_path, social_inventory=social)
    output = tmp_path / "combined"
    inputs = {
        "base_inventory_path": base,
        "base_public_overlap_audit_path": base_public_audit,
        "social_inventory_path": social,
        "usb_coverage_receipt_path": coverage,
        "archived_social_overlap_receipt_path": overlap,
    }
    combined.build_combined_inventory(**inputs, output_root=output)
    alternate = tmp_path / f"alternate-{inputs[argument_name].name}"
    alternate.write_bytes(inputs[argument_name].read_bytes())
    inputs[argument_name] = alternate

    with pytest.raises(ValueError, match=f"requested {label}"):
        combined.build_combined_inventory(**inputs, output_root=output)


def test_formal_stage211_clis_reject_base_only_supplemental_inventory(
    tmp_path: Path,
) -> None:
    _, _, eval_split, fixed_binding = _fixed_eval(tmp_path)
    base = _base_inventory(
        tmp_path,
        eval_split=eval_split,
        fixed_binding=fixed_binding,
    )
    base_manifest = Path(
        json.loads(base.read_text(encoding="utf-8"))["bucket_manifest_path"]
    )
    expected_error = "schema-v2 combined supplemental inventory"

    strict_output = tmp_path / "strict-output"
    strict = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/run_stage211_strict_chained_alignment.py"),
            "--phase",
            "mixer",
            "--difficulty",
            "supplemental_natural",
            "--full-data-profile",
            "--supplemental-inventory",
            str(base),
            "--bucket-manifest",
            str(base_manifest),
            "--output-dir",
            str(strict_output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode != 0
    assert expected_error in strict.stderr
    assert not strict_output.exists()

    profile_receipt_path = tmp_path / "base-profile-receipt.json"
    supplemental_profile_receipt.write_immutable_receipt(
        profile_receipt_path,
        supplemental_profile_receipt.build_receipt(base),
    )
    full_output = tmp_path / "full-output"
    full_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_stage211_full_phase_curriculum.py"),
        "--phase",
        "mixer",
        "--init-checkpoint",
        str(tmp_path / "unread-initialization.pt"),
        "--output-root",
        str(full_output),
        "--config-root",
        str(tmp_path / "configs"),
        "--easy-manifest",
        str(base_manifest),
        "--nano-checkpoint",
        str(tmp_path / "unread-nano.pt"),
        "--supplemental-inventory",
        str(base),
        "--supplemental-profile-receipt",
        str(profile_receipt_path),
    ]
    for difficulty in ("medium", "hard", "long"):
        full_command.extend(("--manifest", f"{difficulty}={base_manifest}"))
    full = subprocess.run(
        full_command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert full.returncode != 0
    assert expected_error in full.stderr
    assert not full_output.exists()


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
    base_public_audit = _base_public_audit(tmp_path, base_inventory=base)
    coverage, overlap = _usb_proofs(tmp_path, social_inventory=social)
    output = tmp_path / "combined"
    combined.build_combined_inventory(
        base_inventory_path=base,
        base_public_overlap_audit_path=base_public_audit,
        social_inventory_path=social,
        usb_coverage_receipt_path=coverage,
        archived_social_overlap_receipt_path=overlap,
        output_root=output,
    )
    social.write_text(social.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="filtered social inventory SHA-256 changed"):
        validate_stage211_supplemental_inventory(
            output / "supplemental_inventory.json",
            verify_part_sha256=False,
        )
