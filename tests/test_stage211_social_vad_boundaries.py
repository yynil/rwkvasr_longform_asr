from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
boundaries = importlib.import_module("scripts.build_stage211_social_vad_boundaries")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record(path: Path, *, label: str, duration_seconds: float) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "source_label": label,
        "sha256": _sha256(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "probe": {
            "frames": int(duration_seconds * 16_000),
            "sample_rate": 16_000,
            "channels": 1,
            "duration_seconds": duration_seconds,
            "format": "WAV",
            "subtype": "PCM_16",
        },
    }


def _write_inventory(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    original = source_root / "original.mp3"
    vocal = source_root / "vocals.wav"
    original.write_bytes(b"same-content")
    vocal.write_bytes(b"same-content")
    records = [
        _source_record(original, label="social_videos_mp3", duration_seconds=61.0),
        _source_record(vocal, label="social_clean_vocals", duration_seconds=61.0),
    ]
    vad_root = tmp_path / "vad"
    vad_root.mkdir()
    assets = {}
    for filename in ("model.pt", "config.yaml", "am.mvn"):
        path = vad_root / filename
        path.write_bytes(filename.encode("ascii"))
        assets[filename] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    inventory = {
        "schema_version": 1,
        "artifact": boundaries.EXPECTED_SOURCE_ARTIFACT,
        "complete": True,
        "training_ready": False,
        "admission_state": "source_inventory_only",
        "layout_valid": True,
        "media_content_hash_complete": True,
        "canonical_audio_files": len(records),
        "audio_probe": {
            "complete": True,
            "errors": [],
            "probed_files": len(records),
            "hours": 122.0 / 3600.0,
        },
        "exact_content_duplicate_groups": [
            {
                "sha256": records[0]["sha256"],
                "paths": [records[0]["path"], records[1]["path"]],
            }
        ],
        "vad": {
            "implementation": "FunASR FsmnVADStreaming",
            "max_single_segment_time_ms": boundaries.MAX_SEGMENT_MS,
            "minimum_admitted_segment_ms": boundaries.MIN_SEGMENT_MS,
            "assets": assets,
        },
        "source_records": records,
    }
    path = tmp_path / "source_inventory.json"
    path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")
    return path, records


def test_normalize_vad_segments_merges_small_gaps_and_splits_without_loss() -> None:
    normalized = boundaries.normalize_vad_segments(
        [[0, 30_010], [30_010, 60_020], [60_200, 60_500], [61_000, 61_300]],
        duration_ms=61_300,
    )

    assert normalized == [(0, 20_166), (20_166, 40_333), (40_333, 60_500)]
    assert sum(end - start for start, end in normalized) == 60_500
    assert all(
        boundaries.MIN_SEGMENT_MS <= end - start <= boundaries.MAX_SEGMENT_MS
        for start, end in normalized
    )


def test_validate_worker_resume_and_finalize_social_vad_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory_path, records = _write_inventory(tmp_path)
    inventory = boundaries.validate_source_inventory(
        inventory_path,
        verify_media_hashes=True,
    )
    duplicate_of = boundaries.exact_duplicate_decisions(inventory)
    assert duplicate_of == {str(records[0]["path"]): str(records[1]["path"])}

    monkeypatch.setattr(boundaries, "_load_model", lambda inventory, torch_threads: object())
    monkeypatch.setattr(
        boundaries,
        "_generate_raw_segments",
        lambda model, source_path: [[0, 30_010], [30_010, 61_000]],
    )
    output_root = tmp_path / "boundaries"
    first = boundaries.run_worker(
        inventory=inventory,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        torch_threads=1,
        max_sources=None,
    )
    second = boundaries.run_worker(
        inventory=inventory,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        torch_threads=1,
        max_sources=None,
    )
    assert first == {"assigned": 2, "created": 2, "reused": 0}
    assert second == {"assigned": 2, "created": 0, "reused": 2}

    result = boundaries.finalize_boundaries(inventory=inventory, output_root=output_root)
    assert result["training_ready"] is False
    assert result["admission_state"] == "vad_boundaries_only"
    assert result["source_files"] == 2
    assert result["admitted_source_files"] == 1
    assert result["exact_duplicate_source_files"] == 1
    assert result["admitted_segments"] == 3
    assert result["admitted_hours"] == pytest.approx(61.0 / 3600.0)
    assert len(result["part_records"]) == 2
    assert (output_root / "boundary_inventory.json").is_file()


def test_source_inventory_requires_complete_probe_and_hashes(tmp_path: Path) -> None:
    inventory_path, _ = _write_inventory(tmp_path)
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    payload["audio_probe"]["complete"] = False
    inventory_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="complete audio probe"):
        boundaries.validate_source_inventory(
            inventory_path,
            verify_media_hashes=False,
        )
