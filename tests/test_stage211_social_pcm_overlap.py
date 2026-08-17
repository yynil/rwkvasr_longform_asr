from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rwkvasr.data import load_webdataset_bucket_manifest
from rwkvasr.data.webdataset_lengths import load_webdataset_length_entries


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
pcm_filter = importlib.import_module("scripts.filter_stage211_social_pcm_overlap")
pcm_rebase = importlib.import_module("scripts.rebase_stage211_social_public_pcm_overlap")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flac(samples: np.ndarray) -> bytes:
    output = io.BytesIO()
    sf.write(output, samples, 16_000, format="FLAC", subtype="PCM_16")
    return output.getvalue()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_source(
    root: Path,
    *,
    source_index: int,
    source_label: str,
    waveforms: list[np.ndarray],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    tar_path = root / f"source_{source_index:06d}.tar"
    rows: list[dict[str, object]] = []
    with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for row_index, waveform in enumerate(waveforms):
            key = f"source{source_index}-segment{row_index}"
            payload = _flac(waveform)
            info = tarfile.TarInfo(f"{key}.flac")
            info.size = len(payload)
            info.mtime = 0
            offset = archive.offset + tarfile.BLOCKSIZE
            archive.addfile(info, io.BytesIO(payload))
            metadata = json.dumps({"id": f"social-{key}", "text": ""}).encode("utf-8")
            metadata_info = tarfile.TarInfo(f"{key}.json")
            metadata_info.size = len(metadata)
            metadata_info.mtime = 0
            metadata_offset = archive.offset + tarfile.BLOCKSIZE
            archive.addfile(metadata_info, io.BytesIO(metadata))
            duration_ms = len(waveform) // 16
            rows.append(
                {
                    "shard_name": str(tar_path),
                    "key": key,
                    "utt_id": f"social-{key}",
                    "split": "train",
                    "num_frames": duration_ms // 10,
                    "audio_member": info.name,
                    "audio_format": "flac",
                    "audio_offset": offset,
                    "audio_size": len(payload),
                    "json_member": metadata_info.name,
                    "json_offset": metadata_offset,
                    "json_size": len(metadata),
                    "storage_kind": "tar",
                    "source_dataset": source_label,
                    "language": "zh",
                    "sample_rate": 16_000,
                    "duration_ms": duration_ms,
                    "uses_text_labels": False,
                    "source_index": source_index,
                }
            )
    receipt_path = root / f"source_{source_index:06d}.json"
    receipt = {
        "schema_version": 1,
        "artifact": "stage211_social_vad_materialized_source",
        "complete": True,
        "status": "materialized",
        "source_index": source_index,
        "source_path": str(root / f"raw_{source_index}.wav"),
        "source_sha256": f"{source_index + 1:064x}",
        "source_label": source_label,
        "tar_path": str(tar_path),
        "tar_size_bytes": tar_path.stat().st_size,
        "tar_sha256": _sha256(tar_path),
        "segments": len(rows),
        "duration_ms": sum(int(row["duration_ms"]) for row in rows),
        "rows": rows,
    }
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    return (
        {
            "path": str(receipt_path),
            "size_bytes": receipt_path.stat().st_size,
            "sha256": _sha256(receipt_path),
            "source_index": source_index,
            "status": "materialized",
            "segments": len(rows),
            "duration_ms": receipt["duration_ms"],
            "tar_path": str(tar_path),
            "tar_size_bytes": tar_path.stat().st_size,
            "tar_sha256": _sha256(tar_path),
        },
        rows,
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    public = np.linspace(-0.5, 0.5, 8_000, dtype=np.float32)
    duplicate = np.sin(np.linspace(0, 30, 9_600, dtype=np.float32)) * 0.25
    independent_a = np.cos(np.linspace(0, 18, 11_200, dtype=np.float32)) * 0.2
    source0, rows0 = _write_source(
        tmp_path,
        source_index=0,
        source_label="social_videos_mp3",
        waveforms=[public, duplicate, independent_a],
    )
    source1, rows1 = _write_source(
        tmp_path,
        source_index=1,
        source_label="social_clean_vocals",
        waveforms=[duplicate],
    )
    all_rows = rows0 + rows1
    train_part = tmp_path / "materialized_train.jsonl"
    _write_jsonl(train_part, all_rows)
    fixed_part = tmp_path / "fixed_eval.jsonl"
    fixed_part.write_text("{}\n" * 256, encoding="utf-8")
    manifest_path = tmp_path / "materialized_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "root": "/",
                "source_length_index_path": str(tmp_path / "materialized_inventory.json"),
                "bucket_width": 80,
                "entries_per_part": 100_000,
                "splits": {
                    "train": {
                        "num_samples": len(all_rows),
                        "buckets": [
                            {
                                "bucket_id": 1,
                                "num_samples": len(all_rows),
                                "parts": [
                                    {"path": str(train_part), "num_samples": len(all_rows)}
                                ],
                            }
                        ],
                    },
                    "eval": {
                        "num_samples": 256,
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "num_samples": 256,
                                "parts": [{"path": str(fixed_part), "num_samples": 256}],
                            }
                        ],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    materialized_path = tmp_path / "materialized_inventory.json"
    materialized = {
        "schema_version": 1,
        "artifact": "stage211_social_vad_materialized_inventory",
        "complete": True,
        "training_ready": False,
        "admission_state": "materialized_pending_overlap_and_merge",
        "storage_kinds": ["tar"],
        "language": "zh",
        "uses_text_labels": False,
        "selected_rows": len(all_rows),
        "selected_hours": sum(int(row["duration_ms"]) for row in all_rows) / 3_600_000.0,
        "fixed_eval": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "rows": 256,
            "parts": [
                {
                    "path": str(fixed_part),
                    "sha256": _sha256(fixed_part),
                    "num_samples": 256,
                }
            ],
        },
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": _sha256(manifest_path),
        "source_receipts": [source0, source1],
    }
    materialized_path.write_text(json.dumps(materialized) + "\n", encoding="utf-8")

    public_audio = tmp_path / "public.flac"
    public_audio.write_bytes(_flac(public))
    public_manifest = tmp_path / "public.jsonl"
    _write_jsonl(
        public_manifest,
        [
            {
                "utt_id": "public-utt",
                "audio_filepath": str(public_audio),
                "dataset": "public_test",
                "text": "reference",
            }
        ],
    )
    return {
        "materialized": materialized_path,
        "public_audio": public_audio,
        "public_manifest": public_manifest,
    }


def test_canonical_pcm_fingerprint_is_container_independent(tmp_path: Path) -> None:
    waveform = np.linspace(-22_000, 22_000, 16_000, dtype=np.int16)
    wav = tmp_path / "audio.wav"
    flac = tmp_path / "audio.flac"
    sf.write(wav, waveform, 16_000, format="WAV", subtype="PCM_16")
    sf.write(flac, waveform, 16_000, format="FLAC", subtype="PCM_16")

    assert pcm_filter.canonical_pcm_fingerprint(wav) == pcm_filter.canonical_pcm_fingerprint(
        flac
    )


def test_social_pcm_filter_deduplicates_and_excludes_public_audio(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    materialized = pcm_filter.validate_materialized_inventory(fixture["materialized"])
    output_root = tmp_path / "filtered"

    first = pcm_filter.run_social_fingerprint_worker(
        materialized=materialized,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_sources=None,
    )
    second = pcm_filter.run_social_fingerprint_worker(
        materialized=materialized,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_sources=None,
    )
    assert first == {"assigned": 2, "created": 2}
    assert second == {"assigned": 2, "reused": 2}

    manifests = {"public_test": Path(fixture["public_manifest"])}
    assert pcm_filter.build_public_fingerprints(
        public_manifests=manifests,
        output_root=output_root,
        expected_rows=None,
    ) == {"public_test": "created"}
    assert pcm_filter.build_public_fingerprints(
        public_manifests=manifests,
        output_root=output_root,
        expected_rows=None,
    ) == {"public_test": "reused"}

    result = pcm_filter.finalize_filtered_inventory(
        materialized=materialized,
        public_manifests=manifests,
        output_root=output_root,
        expected_public_rows=None,
    )
    assert result["prefilter_rows"] == 4
    assert result["selected_rows"] == 2
    assert result["dedupe"]["exact_duplicate_groups"] == 1
    assert result["dedupe"]["rejected_rows"] == 1
    assert result["public_overlap"]["rejected_rows"] == 1
    assert result["public_overlap"]["rejected_rows_by_dataset"] == {"public_test": 1}
    assert result["near_duplicate_detection_complete"] is False

    manifest = load_webdataset_bucket_manifest(result["bucket_manifest_path"])
    assert sum(bucket.num_samples for bucket in manifest.splits["train"]) == 2
    rows = []
    for bucket in manifest.splits["train"]:
        for part in bucket.parts:
            rows.extend(
                load_webdataset_length_entries(
                    Path(result["bucket_manifest_path"]).parent / part.path
                )
            )
    assert {str((row.raw or {})["source_dataset"]) for row in rows} == {
        "social_clean_vocals",
        "social_videos_mp3",
    }
    assert pcm_filter.validate_filtered_inventory(output_root / "filtered_inventory.json")[
        "rows"
    ] == 2
    assert (
        pcm_filter.finalize_filtered_inventory(
            materialized=materialized,
            public_manifests=manifests,
            output_root=output_root,
            expected_public_rows=None,
        )
        == result
    )


def test_filtered_inventory_rejects_changed_fingerprint_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    materialized = pcm_filter.validate_materialized_inventory(fixture["materialized"])
    output_root = tmp_path / "filtered"
    pcm_filter.run_social_fingerprint_worker(
        materialized=materialized,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_sources=None,
    )
    manifests = {"public_test": Path(fixture["public_manifest"])}
    pcm_filter.build_public_fingerprints(
        public_manifests=manifests,
        output_root=output_root,
        expected_rows=None,
    )
    pcm_filter.finalize_filtered_inventory(
        materialized=materialized,
        public_manifests=manifests,
        output_root=output_root,
        expected_public_rows=None,
    )
    part = output_root / "source_fingerprints/source_000000.jsonl"
    part.write_bytes(part.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="receipt changed"):
        pcm_filter.validate_filtered_inventory(output_root / "filtered_inventory.json")


def test_social_pcm_rebase_reuses_source_fingerprints_without_audio_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    materialized = pcm_filter.validate_materialized_inventory(fixture["materialized"])
    source_root = tmp_path / "source_filtered"
    source_manifests = {"public_test": Path(fixture["public_manifest"])}
    pcm_filter.run_social_fingerprint_worker(
        materialized=materialized,
        output_root=source_root,
        worker_index=0,
        num_workers=1,
        max_sources=None,
    )
    pcm_filter.build_public_fingerprints(
        public_manifests=source_manifests,
        output_root=source_root,
        expected_rows={"public_test": 1},
    )
    pcm_filter.finalize_filtered_inventory(
        materialized=materialized,
        public_manifests=source_manifests,
        output_root=source_root,
        expected_public_rows={"public_test": 1},
    )

    corrected_audio = tmp_path / "corrected_public.flac"
    corrected_audio.write_bytes(
        _flac(np.sin(np.linspace(0, 70, 8_000, dtype=np.float32)) * 0.1)
    )
    corrected_manifest = tmp_path / "corrected_public.jsonl"
    _write_jsonl(
        corrected_manifest,
        [
            {
                "utt_id": "corrected-public",
                "audio_filepath": str(corrected_audio),
                "dataset": "public_test",
                "text": "corrected",
            }
        ],
    )
    monkeypatch.setattr(
        pcm_filter,
        "_read_region",
        lambda *args, **kwargs: pytest.fail(
            "social source audio must not be decoded during fingerprint rebase"
        ),
    )
    destination_root = tmp_path / "destination_filtered"
    receipt = pcm_rebase.rebase_and_finalize(
        source_filtered_inventory=source_root / "filtered_inventory.json",
        output_root=destination_root,
        public_manifests={"public_test": corrected_manifest},
        expected_public_rows={"public_test": 1},
    )

    assert receipt["audio_redecoded"] is False
    assert receipt["source_count"] == 2
    assert receipt["source_hardlink_status"] == {"created": 2}
    assert receipt["destination_selected_rows"] == 3
    for source_index in range(2):
        source_part, source_receipt = pcm_filter._source_fingerprint_paths(
            source_root,
            source_index,
        )
        destination_part, destination_receipt = pcm_filter._source_fingerprint_paths(
            destination_root,
            source_index,
        )
        assert source_part.samefile(destination_part)
        assert source_receipt.read_bytes() != destination_receipt.read_bytes()
    assert pcm_rebase.validate_rebase_receipt(destination_root / "rebase_receipt.json") == receipt
    assert (
        pcm_rebase.rebase_and_finalize(
            source_filtered_inventory=source_root / "filtered_inventory.json",
            output_root=destination_root,
            public_manifests={"public_test": corrected_manifest},
            expected_public_rows={"public_test": 1},
        )
        == receipt
    )

    alternate_source = tmp_path / "alternate_filtered_inventory.json"
    alternate_source.write_bytes((source_root / "filtered_inventory.json").read_bytes())
    with pytest.raises(ValueError, match="requested source inventory"):
        pcm_rebase.rebase_and_finalize(
            source_filtered_inventory=alternate_source,
            output_root=destination_root,
            public_manifests={"public_test": corrected_manifest},
            expected_public_rows={"public_test": 1},
        )

    alternate_manifest = tmp_path / "alternate_corrected_public.jsonl"
    alternate_manifest.write_bytes(corrected_manifest.read_bytes())
    with pytest.raises(ValueError, match="requested public manifest: public_test"):
        pcm_rebase.rebase_and_finalize(
            source_filtered_inventory=source_root / "filtered_inventory.json",
            output_root=destination_root,
            public_manifests={"public_test": alternate_manifest},
            expected_public_rows={"public_test": 1},
        )

    with pytest.raises(ValueError, match="requested public row map"):
        pcm_rebase.rebase_and_finalize(
            source_filtered_inventory=source_root / "filtered_inventory.json",
            output_root=destination_root,
            public_manifests={"public_test": corrected_manifest},
            expected_public_rows={"public_test": 2},
        )
