from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rwkvasr.data import (
    WebDatasetConfig,
    build_bucketed_webdataset_loader,
    load_webdataset_bucket_manifest,
)
from rwkvasr.data.webdataset_lengths import load_webdataset_length_entries
from rwkvasr.training.funasr_online_teacher import _resolve_audio_path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
boundaries = importlib.import_module("scripts.build_stage211_social_vad_boundaries")
materialize = importlib.import_module("scripts.materialize_stage211_social_vad_segments")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_source_inventory(tmp_path: Path) -> Path:
    source_path = tmp_path / "source.wav"
    samples = np.stack(
        [
            np.linspace(-0.5, 0.5, 16_000, dtype=np.float32),
            np.linspace(0.5, -0.5, 16_000, dtype=np.float32),
        ],
        axis=1,
    )
    sf.write(source_path, samples, 8_000, format="WAV", subtype="PCM_16")
    source_stat = source_path.stat()

    vad_root = tmp_path / "vad"
    vad_root.mkdir()
    assets = {}
    for filename in ("model.pt", "config.yaml", "am.mvn"):
        asset_path = vad_root / filename
        asset_path.write_bytes(filename.encode("ascii"))
        assets[filename] = {
            "path": str(asset_path.resolve()),
            "size_bytes": asset_path.stat().st_size,
            "sha256": _sha256(asset_path),
        }
    source_record = {
        "path": str(source_path.resolve()),
        "source_label": "social_videos_extracted_wav",
        "sha256": _sha256(source_path),
        "size_bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "probe": {
            "frames": 16_000,
            "sample_rate": 8_000,
            "channels": 2,
            "duration_seconds": 2.0,
            "format": "WAV",
            "subtype": "PCM_16",
        },
    }
    inventory = {
        "schema_version": 1,
        "artifact": boundaries.EXPECTED_SOURCE_ARTIFACT,
        "complete": True,
        "training_ready": False,
        "admission_state": "source_inventory_only",
        "layout_valid": True,
        "media_content_hash_complete": True,
        "canonical_audio_files": 1,
        "audio_probe": {
            "complete": True,
            "errors": [],
            "probed_files": 1,
            "hours": 2.0 / 3600.0,
        },
        "exact_content_duplicate_groups": [],
        "vad": {
            "implementation": "FunASR FsmnVADStreaming",
            "max_single_segment_time_ms": boundaries.MAX_SEGMENT_MS,
            "minimum_admitted_segment_ms": boundaries.MIN_SEGMENT_MS,
            "assets": assets,
        },
        "source_records": [source_record],
    }
    inventory_path = tmp_path / "source_inventory.json"
    inventory_path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")
    return inventory_path


def _build_boundary_inventory(tmp_path: Path, source_inventory_path: Path) -> Path:
    source_inventory = boundaries.validate_source_inventory(
        source_inventory_path,
        verify_media_hashes=True,
    )
    output_root = tmp_path / "boundaries"
    payload = boundaries.build_source_payload(
        inventory=source_inventory,
        source_index=0,
        raw_segments=[[0, 800], [1_100, 2_000]],
        duplicate_of=None,
    )
    boundaries._write_immutable_json(boundaries._part_path(output_root, 0), payload)
    boundaries.finalize_boundaries(inventory=source_inventory, output_root=output_root)
    return output_root / "boundary_inventory.json"


def _fixed_eval_manifest(tmp_path: Path) -> Path:
    part = tmp_path / "fixed_eval.jsonl"
    part.write_text("{}\n" * 256, encoding="utf-8")
    manifest = tmp_path / "fixed_eval_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "root": "/",
                "source_length_index_path": str(part),
                "splits": {
                    "eval": {
                        "num_samples": 256,
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "num_samples": 256,
                                "parts": [{"path": str(part), "num_samples": 256}],
                            }
                        ],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_materialize_social_vad_segments_resamples_indexes_and_resumes(
    tmp_path: Path,
) -> None:
    source_inventory_path = _build_source_inventory(tmp_path)
    boundary_inventory_path = _build_boundary_inventory(tmp_path, source_inventory_path)
    boundary, source_inventory = materialize.validate_boundary_inventory(
        boundary_inventory_path,
        verify_source_part_hashes=True,
    )
    output_root = tmp_path / "materialized"

    first = materialize.run_worker(
        boundary=boundary,
        source_inventory=source_inventory,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_sources=None,
    )
    second = materialize.run_worker(
        boundary=boundary,
        source_inventory=source_inventory,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_sources=None,
    )
    assert first == {"assigned": 1, "created": 1, "reused": 0}
    assert second == {"assigned": 1, "created": 0, "reused": 1}

    receipt = json.loads((output_root / "sources/source_000000.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "materialized"
    assert receipt["segments"] == 2
    assert receipt["duration_ms"] == 1_700
    assert [row["duration_ms"] for row in receipt["rows"]] == [800, 900]
    tar_path = Path(receipt["tar_path"])
    with tar_path.open("rb") as handle:
        for row in receipt["rows"]:
            handle.seek(row["audio_offset"])
            audio_bytes = handle.read(row["audio_size"])
            with sf.SoundFile(io.BytesIO(audio_bytes)) as decoded:
                assert decoded.samplerate == 16_000
                assert decoded.channels == 1
                assert decoded.frames == row["duration_ms"] * 16

    final = materialize.finalize_materialization(
        boundary=boundary,
        source_inventory=source_inventory,
        output_root=output_root,
        fixed_eval_manifest_path=_fixed_eval_manifest(tmp_path),
    )
    assert final["training_ready"] is False
    assert final["admission_state"] == "materialized_pending_overlap_and_merge"
    assert final["selected_rows"] == 2
    assert final["selected_hours"] == pytest.approx(1_700 / 3_600_000.0)
    manifest = load_webdataset_bucket_manifest(final["bucket_manifest_path"])
    assert sum(bucket.num_samples for bucket in manifest.splits["train"]) == 2
    assert sum(bucket.num_samples for bucket in manifest.splits["eval"]) == 256
    entries = []
    for bucket in manifest.splits["train"]:
        for part in bucket.parts:
            entries.extend(
                load_webdataset_length_entries(
                    Path(final["bucket_manifest_path"]).parent / part.path
                )
            )
    assert len(entries) == 2
    assert all(entry.storage_kind == "tar" for entry in entries)
    loader = build_bucketed_webdataset_loader(
        "/",
        bucket_manifest_path=final["bucket_manifest_path"],
        config=WebDatasetConfig(
            shuffle_shards=False,
            split="train",
            utt_id_key="id",
            allow_missing_targets=True,
        ),
        batch_size=1,
        num_workers=0,
        rank=0,
        world_size=1,
    )
    batch = next(iter(loader))
    assert batch.features.shape[0] == 1
    assert batch.features.shape[2] == 80
    assert batch.target_lengths.tolist() == [0]
    assert batch.ctc_teacher_audio_rows is not None
    teacher_row = batch.ctc_teacher_audio_rows[0]
    assert teacher_row["storage_kind"] == "tar"
    assert teacher_row["audio_offset"] > 0
    resolved = _resolve_audio_path(
        teacher_row,
        batch.utt_ids[0],
        shard_paths={},
        audio_cache_dir=tmp_path / "teacher_audio_cache",
        keep_audio_cache=False,
    )
    with sf.SoundFile(resolved.path) as decoded:
        assert decoded.samplerate == 16_000
        assert decoded.channels == 1
    assert (
        materialize.finalize_materialization(
            boundary=boundary,
            source_inventory=source_inventory,
            output_root=output_root,
            fixed_eval_manifest_path=_fixed_eval_manifest(tmp_path),
        )
        == final
    )


def test_materialized_source_validation_rejects_changed_member_offset(tmp_path: Path) -> None:
    source_inventory_path = _build_source_inventory(tmp_path)
    boundary_inventory_path = _build_boundary_inventory(tmp_path, source_inventory_path)
    boundary, source_inventory = materialize.validate_boundary_inventory(
        boundary_inventory_path,
        verify_source_part_hashes=True,
    )
    output_root = tmp_path / "materialized"
    materialize.run_worker(
        boundary=boundary,
        source_inventory=source_inventory,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_sources=None,
    )
    receipt_path = output_root / "sources/source_000000.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["rows"][0]["audio_offset"] += 1
    boundary_part_path, boundary_payload = materialize._boundary_source_payload(boundary, 0)

    with pytest.raises(ValueError, match="FLAC mismatch|member short-read"):
        materialize.validate_materialized_source(
            receipt,
            boundary=boundary,
            source_inventory=source_inventory,
            source_index=0,
            boundary_part_path=boundary_part_path,
            boundary_payload=boundary_payload,
            verify_tar_hash=True,
            verify_members=True,
        )
