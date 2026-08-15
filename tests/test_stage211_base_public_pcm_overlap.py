from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sqlite3
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


def test_base_public_pcm_archive_cache_rejects_and_removes_changed_cache(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.zip"
    source_path.write_bytes(b"immutable archive bytes")
    expected_sha256 = _sha256(source_path)
    cache_path = audit._copy_archive_to_cache(
        source_path=source_path,
        cache_dir=tmp_path / "cache",
        archive_index=7,
        expected_size_bytes=source_path.stat().st_size,
        expected_sha256=expected_sha256,
    )
    cache_path.write_bytes(b"x" * source_path.stat().st_size)

    with pytest.raises(ValueError, match="archive cache changed"):
        audit._copy_archive_to_cache(
            source_path=source_path,
            cache_dir=tmp_path / "cache",
            archive_index=7,
            expected_size_bytes=source_path.stat().st_size,
            expected_sha256=expected_sha256,
        )

    assert not cache_path.exists()


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
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    audio_bytes: list[bytes] = []
    parquet_records: list[dict[str, object]] = []

    fixed_part = tmp_path / "fixed_eval.jsonl"
    fixed_part.write_text("{}\n" * 256, encoding="utf-8")
    fixed_manifest = tmp_path / "fixed_eval_manifest.json"
    _write_json(fixed_manifest, {"version": 1})
    stage179_manifest = tmp_path / "stage179.json"
    _write_json(stage179_manifest, {"version": 1})

    manifest_root = tmp_path / "base/webdataset_buckets_audio_text"
    part_records: list[dict[str, object]] = []
    global_index = 0
    for source in sources:
        source_rows = 2 if source == "peoples_speech_clean" else 1
        parquet_path = tmp_path / f"{source}.parquet"
        source_audio: list[bytes] = []
        for row_index in range(source_rows):
            payload = _flac(
                np.asarray(
                    np.sin(
                        np.linspace(
                            0.0,
                            float(global_index + 1) * 20.0,
                            16_000,
                        )
                    )
                    * 0.2,
                    dtype=np.float32,
                )
            )
            audio_bytes.append(payload)
            source_audio.append(payload)
            relative = f"train/{source}_{row_index:02d}.jsonl"
            part_path = manifest_root / relative
            part_path.parent.mkdir(parents=True, exist_ok=True)
            row_id = f"row-{global_index}"
            row = {
                "shard_name": str(parquet_path),
                "key": f"key-{global_index}",
                "utt_id": f"utt-{global_index}",
                "split": "train",
                "num_frames": 100,
                "audio_member": f"{row_id}.flac",
                "audio_format": "flac",
                "audio_size": None,
                "json_member": "",
                "storage_kind": "parquet",
                "parquet_row_group": 0,
                "parquet_row_index": row_index,
                "parquet_id": row_id,
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
            global_index += 1
        table = pa.table(
            {
                "id": pa.array(
                    [f"row-{global_index - source_rows + index}" for index in range(source_rows)]
                ),
                "audio": pa.array(
                    [
                        {
                            "bytes": payload,
                            "path": f"row-{global_index - source_rows + index}.flac",
                        }
                        for index, payload in enumerate(source_audio)
                    ],
                    type=audio_type,
                ),
                "duration_ms": pa.array([1000] * source_rows, type=pa.int32()),
            }
        )
        pq.write_table(table, parquet_path, row_group_size=source_rows)
        stat = parquet_path.stat()
        parquet_records.append(
            {
                "path": str(parquet_path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256(parquet_path),
                "status": "accepted",
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
                    "num_samples": global_index,
                    "buckets": [
                        {
                            "bucket_id": 1,
                            "num_samples": global_index,
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
                            "parts": [{"path": str(fixed_part), "num_samples": 256}],
                        }
                    ],
                },
            },
        },
    )
    source_counts = {source: 2 if source == "peoples_speech_clean" else 1 for source in sources}
    source_hours = {source: count / 3600.0 for source, count in source_counts.items()}
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
                "accepted_unique_rows": global_index,
            },
            "selected_rows": global_index,
            "selected_hours": global_index / 3600.0,
            "selected_counts_by_source": source_counts,
            "selected_hours_by_source": source_hours,
            "archive_records": parquet_records,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory_path, base_audio = _fixture(tmp_path)
    public_audio = tmp_path / "public.flac"
    public_audio.write_bytes(
        base_audio[0] if overlap else _flac(np.linspace(-0.4, 0.4, 8_000, dtype=np.float32))
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
    assert audit.build_location_index(base=base, output_root=output_root) == "created"
    assert audit.build_location_index(base=base, output_root=output_root) == "reused"
    location_index = audit.validate_location_index(output_root, base=base)
    assert len(location_index["archives"]) == 5
    assert sorted(record["rows"] for record in location_index["archives"]) == [1, 1, 1, 1, 2]
    with sqlite3.connect(location_index["database_path"]) as connection:
        grouped_rows = connection.execute(
            "SELECT COUNT(*) FROM entries WHERE source_dataset = 'peoples_speech_clean'"
        ).fetchone()[0]
    assert grouped_rows == 2
    original_make_shard_reader = audit._make_shard_reader
    reader_opens: dict[str, int] = {}
    row_group_loads: dict[str, int] = {}

    class CountingReader:
        def __init__(self, reader: object, shard_path: Path) -> None:
            self.reader = reader
            self.shard_path = str(shard_path)

        def read_audio_row(
            self,
            *,
            row_group: int,
            row_index: int,
        ) -> tuple[bytes, dict[str, object]]:
            if self.reader._cached_row_group != row_group:  # type: ignore[attr-defined]
                row_group_loads[self.shard_path] = row_group_loads.get(self.shard_path, 0) + 1
            return self.reader.read_audio_row(  # type: ignore[no-any-return,attr-defined]
                row_group=row_group,
                row_index=row_index,
            )

        def close(self) -> None:
            self.reader.close()  # type: ignore[attr-defined]

    def counting_make_shard_reader(storage_kind: str, shard_path: Path) -> object:
        reader_opens[str(shard_path)] = reader_opens.get(str(shard_path), 0) + 1
        return CountingReader(original_make_shard_reader(storage_kind, shard_path), shard_path)

    monkeypatch.setattr(audit, "_make_shard_reader", counting_make_shard_reader)
    archive_cache_dir = tmp_path / "archive_cache"
    first = audit.run_archive_worker(
        location_index=location_index,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_archives=None,
        archive_cache_dir=archive_cache_dir,
    )
    receipt_bytes = {
        path.name: path.read_bytes()
        for path in (output_root / "archive_fingerprints").glob("*.receipt.json")
    }
    second = audit.run_archive_worker(
        location_index=location_index,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_archives=None,
        archive_cache_dir=archive_cache_dir,
    )
    assert first == {"assigned": 5, "created": 5}
    assert second == {"assigned": 5, "reused": 5}
    assert set(reader_opens.values()) == {1}
    assert set(row_group_loads.values()) == {1}
    assert all(Path(path).parent == archive_cache_dir for path in reader_opens)
    assert archive_cache_dir.is_dir()
    assert list(archive_cache_dir.iterdir()) == []
    source_archives = {record["shard_path"] for record in location_index["archives"]}
    for receipt_path in (output_root / "archive_fingerprints").glob("*.receipt.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["shard_path"] in source_archives
        assert receipt_path.read_bytes() == receipt_bytes[receipt_path.name]

    result = audit.finalize_audit(
        base=base,
        public_manifests=public_manifests,
        output_root=output_root,
        expected_public_rows=None,
        location_index=location_index,
    )
    assert result["scanned_rows"] == 6
    assert result["scan_order"] == audit.PRODUCTION_SCAN_ORDER
    assert len(result["archive_fingerprint_receipts"]) == 5
    assert "part_fingerprint_receipts" not in result
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


def test_base_public_pcm_parallel_decode_matches_serial_bytes(tmp_path: Path) -> None:
    inventory_path, _ = _fixture(tmp_path)
    base = audit.validate_base_inventory(inventory_path)

    fingerprints: dict[int, dict[str, bytes]] = {}
    for decode_workers in (1, 3):
        output_root = tmp_path / f"audit_{decode_workers}"
        audit.build_location_index(base=base, output_root=output_root)
        location_index = audit.validate_location_index(output_root, base=base)
        result = audit.run_archive_worker(
            location_index=location_index,
            output_root=output_root,
            worker_index=0,
            num_workers=1,
            max_archives=None,
            archive_cache_dir=tmp_path / f"cache_{decode_workers}",
            decode_workers=decode_workers,
        )
        assert result == {"assigned": 5, "created": 5}
        fingerprints[decode_workers] = {
            path.name: path.read_bytes()
            for path in (output_root / "archive_fingerprints").glob("archive_*.jsonl")
        }

    assert fingerprints[3] == fingerprints[1]


def test_base_public_pcm_audit_rejects_changed_archive_fingerprint(tmp_path: Path) -> None:
    inventory_path, _ = _fixture(tmp_path)
    base = audit.validate_base_inventory(inventory_path)
    output_root = tmp_path / "audit"
    audit.build_location_index(base=base, output_root=output_root)
    location_index = audit.validate_location_index(output_root, base=base)
    audit.run_archive_worker(
        location_index=location_index,
        output_root=output_root,
        worker_index=0,
        num_workers=1,
        max_archives=None,
    )
    fingerprint_path = output_root / "archive_fingerprints/archive_000000.jsonl"
    fingerprint_path.write_bytes(fingerprint_path.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="archive fingerprint changed"):
        audit.run_archive_worker(
            location_index=location_index,
            output_root=output_root,
            worker_index=0,
            num_workers=1,
            max_archives=None,
        )


def test_base_public_pcm_archive_cache_is_removed_after_decode_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory_path, _ = _fixture(tmp_path)
    base = audit.validate_base_inventory(inventory_path)
    output_root = tmp_path / "audit"
    cache_dir = tmp_path / "archive_cache"
    audit.build_location_index(base=base, output_root=output_root)
    location_index = audit.validate_location_index(output_root, base=base)

    def fail_fingerprint(source: object) -> tuple[str, int]:
        raise RuntimeError("controlled decode failure")

    monkeypatch.setattr(audit, "canonical_pcm_fingerprint", fail_fingerprint)
    with pytest.raises(RuntimeError, match="controlled decode failure"):
        audit.run_archive_worker(
            location_index=location_index,
            output_root=output_root,
            worker_index=0,
            num_workers=1,
            max_archives=1,
            archive_cache_dir=cache_dir,
            decode_workers=3,
        )

    assert cache_dir.is_dir()
    assert list(cache_dir.iterdir()) == []
    assert not list((output_root / "archive_fingerprints").glob("*.tmp.*"))


def test_base_public_pcm_archive_cache_rejects_same_stat_changed_source(
    tmp_path: Path,
) -> None:
    inventory_path, _ = _fixture(tmp_path)
    base = audit.validate_base_inventory(inventory_path)
    output_root = tmp_path / "audit"
    cache_dir = tmp_path / "archive_cache"
    audit.build_location_index(base=base, output_root=output_root)
    location_index = audit.validate_location_index(output_root, base=base)
    archive = location_index["archives"][0]
    source_path = Path(archive["shard_path"])
    source_bytes = bytearray(source_path.read_bytes())
    source_bytes[len(source_bytes) // 2] ^= 1
    source_path.write_bytes(source_bytes)
    os.utime(
        source_path,
        ns=(int(archive["archive_mtime_ns"]), int(archive["archive_mtime_ns"])),
    )

    with pytest.raises(ValueError, match="source archive changed"):
        audit.run_archive_worker(
            location_index=location_index,
            output_root=output_root,
            worker_index=0,
            num_workers=1,
            max_archives=1,
            archive_cache_dir=cache_dir,
        )

    assert cache_dir.is_dir()
    assert list(cache_dir.iterdir()) == []


def test_base_public_pcm_all_cli_uses_archive_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory_path, _ = _fixture(tmp_path)
    public_audio = tmp_path / "public.flac"
    public_audio.write_bytes(_flac(np.linspace(-0.4, 0.4, 8_000, dtype=np.float32)))
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
    output_root = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_stage211_base_public_pcm_overlap.py",
            "--base-inventory",
            str(inventory_path),
            "--output-root",
            str(output_root),
            "--public-manifest",
            f"public_test={public_manifest}",
            "all",
        ],
    )

    assert audit.main() == 0
    result = json.loads((output_root / "audit_receipt.json").read_text(encoding="utf-8"))
    assert result["scan_order"] == audit.PRODUCTION_SCAN_ORDER
    assert result["scanned_rows"] == 6
    assert len(result["archive_fingerprint_receipts"]) == 5
    assert not (output_root / "part_fingerprints").exists()
