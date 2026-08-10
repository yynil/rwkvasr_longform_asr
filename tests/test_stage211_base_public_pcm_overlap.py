from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
audit = importlib.import_module("scripts.audit_stage211_base_public_pcm_overlap")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flac(waveform: np.ndarray) -> bytes:
    sf = pytest.importorskip("soundfile")
    encoded = io.BytesIO()
    sf.write(encoded, waveform, 16_000, format="FLAC", subtype="PCM_16")
    return encoded.getvalue()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, list[bytes]]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sources = sorted(
        {
            "ljspeech",
            "mls_english",
            "peoples_speech_clean",
            "peoples_speech_dirty",
            "vctk",
        }
    )
    audio_bytes = [
        _flac(
            np.asarray(
                np.sin(np.linspace(0.0, float(index + 1) * 20.0, 16_000)) * 0.2,
                dtype=np.float32,
            )
        )
        for index in range(len(sources))
    ]
    parquet_path = tmp_path / "source.parquet"
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "id": pa.array([f"row-{index}" for index in range(len(sources))]),
            "audio": pa.array(
                [
                    {"bytes": payload, "path": f"row-{index}.flac"}
                    for index, payload in enumerate(audio_bytes)
                ],
                type=audio_type,
            ),
            "duration_ms": pa.array([1000] * len(sources), type=pa.int32()),
        }
    )
    pq.write_table(table, parquet_path, row_group_size=len(sources))

    fixed_part = tmp_path / "fixed_eval.jsonl"
    fixed_part.write_text("{}\n" * 256, encoding="utf-8")
    fixed_manifest = tmp_path / "fixed_eval_manifest.json"
    _write_json(fixed_manifest, {"version": 1})
    stage179_manifest = tmp_path / "stage179.json"
    _write_json(stage179_manifest, {"version": 1})

    manifest_root = tmp_path / "base/webdataset_buckets_audio_text"
    part_records: list[dict[str, object]] = []
    for index, source in enumerate(sources):
        relative = f"train/{source}.jsonl"
        part_path = manifest_root / relative
        part_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "shard_name": str(parquet_path),
            "key": f"key-{index}",
            "utt_id": f"utt-{index}",
            "split": "train",
            "num_frames": 100,
            "audio_member": f"row-{index}.flac",
            "audio_format": "flac",
            "audio_size": None,
            "json_member": "",
            "storage_kind": "parquet",
            "parquet_row_group": 0,
            "parquet_row_index": index,
            "parquet_id": f"row-{index}",
            "source_dataset": source,
            "language": "en",
            "sample_rate": 16_000,
            "duration_ms": 1000,
            "uses_text_labels": False,
        }
        part_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        part_records.append(
            {
                "path": relative,
                "num_samples": 1,
                "first_shard": str(parquet_path),
                "last_shard": str(parquet_path),
                "source_label": source,
                "size_bytes": part_path.stat().st_size,
                "sha256": _sha256(part_path),
            }
        )
    manifest_path = manifest_root / "manifest.json"
    inventory_path = tmp_path / "base/supplemental_inventory.json"
    _write_json(
        manifest_path,
        {
            "version": 1,
            "root": "/",
            "source_length_index_path": str(inventory_path),
            "bucket_width": 80,
            "entries_per_part": 100_000,
            "splits": {
                "train": {
                    "num_samples": len(sources),
                    "buckets": [
                        {
                            "bucket_id": 1,
                            "num_samples": len(sources),
                            "parts": part_records,
                        }
                    ],
                },
                "eval": {
                    "num_samples": 256,
                    "buckets": [
                        {
                            "bucket_id": 0,
                            "num_samples": 256,
                            "parts": [
                                {"path": str(fixed_part), "num_samples": 256}
                            ],
                        }
                    ],
                },
            },
        },
    )
    source_counts = {source: 1 for source in sources}
    source_hours = {source: 1.0 / 3600.0 for source in sources}
    stat = parquet_path.stat()
    _write_json(
        inventory_path,
        {
            "schema_version": 1,
            "artifact": "stage211_supplemental_natural_inventory",
            "complete": True,
            "training_ready": True,
            "hash_archives": True,
            "require_production_layout": True,
            "layout_errors": [],
            "language": "en",
            "uses_text_labels": False,
            "storage_kinds": ["parquet", "zip"],
            "fixed_eval": {
                "manifest_path": str(fixed_manifest),
                "manifest_sha256": _sha256(fixed_manifest),
                "rows": 256,
                "parts": [
                    {
                        "path": str(fixed_part),
                        "sha256": _sha256(fixed_part),
                        "num_samples": 256,
                    }
                ],
            },
            "cross_pool_dedupe": {
                "mode": "source_identity_plus_known_corpus_exclusion",
                "content_fingerprint_complete": False,
                "stage179": {
                    "manifest_path": str(stage179_manifest),
                    "manifest_sha256": _sha256(stage179_manifest),
                    "expected_source_set_match": True,
                },
                "supplemental_sources": sources,
                "source_sets_disjoint": True,
                "known_overlap_exclusions": [
                    "llaso_gigaspeech",
                    "llaso_librispeech",
                ],
            },
            "dedupe": {
                "algorithm": "blake2b16(corpus + NUL + source_identity)",
                "accepted_unique_rows": len(sources),
            },
            "selected_rows": len(sources),
            "selected_hours": len(sources) / 3600.0,
            "selected_counts_by_source": source_counts,
            "selected_hours_by_source": source_hours,
            "archive_records": [
                {
                    "path": str(parquet_path),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": _sha256(parquet_path),
                    "status": "accepted",
                }
            ],
            "bucket_manifest_path": str(manifest_path),
            "bucket_manifest_sha256": _sha256(manifest_path),
            "part_records": part_records,
        },
    )
    return inventory_path, audio_bytes


@pytest.mark.parametrize("overlap", [False, True])
def test_base_public_pcm_audit_is_resumable_and_fail_closed(
    tmp_path: Path,
    overlap: bool,
) -> None:
    inventory_path, base_audio = _fixture(tmp_path)
    public_audio = tmp_path / "public.flac"
    public_audio.write_bytes(
        base_audio[0]
        if overlap
        else _flac(np.linspace(-0.4, 0.4, 8_000, dtype=np.float32))
    )
    public_manifest = tmp_path / "public.jsonl"
    public_manifest.write_text(
        json.dumps(
            {
                "utt_id": "public-utt",
                "audio_filepath": str(public_audio),
                "dataset": "public_test",
                "text": "reference",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    public_manifests = {"public_test": public_manifest}
    output_root = tmp_path / "audit"
    base = audit.validate_base_inventory(inventory_path)
    audit.build_public_fingerprints(
        public_manifests=public_manifests,
        output_root=output_root,
        expected_rows=None,
    )
    first = audit.run_fingerprint_worker(
        base=base,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_parts=None,
    )
    second = audit.run_fingerprint_worker(
        base=base,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_parts=None,
    )
    assert first == {"assigned": 5, "created": 5}
    assert second == {"assigned": 5, "reused": 5}

    result = audit.finalize_audit(
        base=base,
        public_manifests=public_manifests,
        output_root=output_root,
        expected_public_rows=None,
    )
    assert result["scanned_rows"] == 5
    assert result["public_overlap_rows"] == int(overlap)
    assert result["training_ready"] is (not overlap)
    assert (
        audit.validate_audit_receipt(
            output_root / "audit_receipt.json",
            require_training_ready=not overlap,
        )
        == result
    )
    if overlap:
        with pytest.raises(ValueError, match="found public-evaluation overlap"):
            audit.validate_audit_receipt(
                output_root / "audit_receipt.json",
                require_training_ready=True,
            )


def test_base_public_pcm_audit_rejects_changed_fingerprint_part(tmp_path: Path) -> None:
    inventory_path, _ = _fixture(tmp_path)
    base = audit.validate_base_inventory(inventory_path)
    output_root = tmp_path / "audit"
    audit.run_fingerprint_worker(
        base=base,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_parts=None,
    )
    fingerprint_path = output_root / "part_fingerprints/part_000000.jsonl"
    fingerprint_path.write_bytes(fingerprint_path.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="fingerprint part changed"):
        audit.run_fingerprint_worker(
            base=base,
            output_root=output_root,
            worker_index=0,
            num_workers=1,
            max_parts=None,
        )
