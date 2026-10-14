from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
repair = importlib.import_module("scripts.repair_stage211_supplemental_source")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rewrite_part_changes_only_matching_shard_name(tmp_path: Path) -> None:
    old_archive = str(tmp_path / "old.zip")
    repaired_archive = str(tmp_path / "repaired.zip")
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    rows = [
        {
            "key": "keep",
            "shard_name": str(tmp_path / "other.zip"),
            "storage_kind": "zip",
            "audio_member": "keep.wav",
            "audio_size": 3,
            "zip_crc32": 1,
            "zip_compress_type": 8,
        },
        {
            "key": "repair",
            "shard_name": old_archive,
            "storage_kind": "zip",
            "audio_member": "repair.wav",
            "audio_size": 7,
            "zip_crc32": 2,
            "zip_compress_type": 8,
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    status, replacements, members = repair._rewrite_part(
        source=source,
        destination=destination,
        old_archive=old_archive,
        repaired_archive=repaired_archive,
    )

    assert status == "created"
    assert replacements == 1
    assert members == {"repair.wav": (7, 2, 8)}
    output = [json.loads(line) for line in destination.read_text().splitlines()]
    assert output[0] == rows[0]
    assert output[1] == {**rows[1], "shard_name": repaired_archive}
    assert destination.read_bytes().replace(
        repaired_archive.encode(), old_archive.encode()
    ) == source.read_bytes()


def test_rewrite_part_rejects_non_shard_path_occurrence(tmp_path: Path) -> None:
    old_archive = str(tmp_path / "old.zip")
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "key": old_archive,
                "shard_name": str(tmp_path / "other.zip"),
                "storage_kind": "zip",
                "audio_member": "x.wav",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row contract changed"):
        repair._rewrite_part(
            source=source,
            destination=tmp_path / "destination.jsonl",
            old_archive=old_archive,
            repaired_archive=str(tmp_path / "repaired.zip"),
        )


def test_repaired_zip_validation_rejects_wrong_hash_and_accepts_crc(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("a.wav", b"audio-a")
        archive.writestr("b.wav", b"audio-b")
    digest = _sha256(archive_path)

    record = repair._validate_repaired_zip(
        archive_path,
        expected_sha256=digest,
        expected_size_bytes=archive_path.stat().st_size,
        expected_members=2,
        deep=True,
    )
    assert record["members"] == 2
    assert record["full_crc_pass"] is True
    with pytest.raises(ValueError, match="SHA-256 changed"):
        repair._validate_repaired_zip(
            archive_path,
            expected_sha256="0" * 64,
            expected_size_bytes=archive_path.stat().st_size,
            expected_members=2,
            deep=False,
        )


def test_location_index_patch_preserves_indices_and_replaces_one_archive(
    tmp_path: Path,
) -> None:
    old_archive = (tmp_path / "old.zip").resolve()
    repaired_archive = (tmp_path / "repaired.zip").resolve()
    old_archive.write_bytes(b"old")
    repaired_archive.write_bytes(b"new")
    source_database = tmp_path / "source.sqlite"
    destination_database = tmp_path / "destination.sqlite"
    connection = sqlite3.connect(source_database)
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE entries (
            key TEXT PRIMARY KEY, utt_id TEXT, source_dataset TEXT, duration_ms INTEGER,
            num_frames INTEGER, sample_rate INTEGER, storage_kind TEXT, shard_path TEXT,
            audio_member TEXT, audio_format TEXT, audio_size INTEGER, zip_crc32 INTEGER,
            zip_compress_type INTEGER, parquet_row_group INTEGER,
            parquet_row_index INTEGER, parquet_id TEXT
        );
        CREATE TABLE archives (
            archive_index INTEGER PRIMARY KEY, storage_kind TEXT, shard_path TEXT UNIQUE,
            source_dataset TEXT, rows INTEGER, duration_ms INTEGER,
            archive_size_bytes INTEGER, archive_mtime_ns INTEGER, archive_sha256 TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [("base_inventory_path", "source"), ("base_inventory_sha256", "source-sha")],
    )
    connection.execute(
        "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "key", "utt", "mls_english", 1000, 100, 16000, "zip", str(old_archive),
            "a.wav", "wav", 3, 1, 8, None, None, "",
        ),
    )
    connection.execute(
        "INSERT INTO archives VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (7, "zip", str(old_archive), "mls_english", 1, 1000, 3, 1, "old-sha"),
    )
    connection.commit()
    connection.close()
    repaired_stat = repaired_archive.stat()
    source_base = {"inventory": {}}
    repaired_base = {
        "inventory_path": "repaired",
        "inventory_sha256": "repaired-sha",
        "bucket_manifest_path": "manifest",
        "bucket_manifest_sha256": "manifest-sha",
        "inventory": {
            "archive_records": [
                {
                    "path": str(repaired_archive),
                    "size_bytes": repaired_stat.st_size,
                    "mtime_ns": repaired_stat.st_mtime_ns,
                    "sha256": _sha256(repaired_archive),
                    "audio_members": 1,
                }
            ]
        },
    }

    assert repair._copy_and_patch_location_index(
        source_database=source_database,
        destination_database=destination_database,
        source_base=source_base,
        repaired_base=repaired_base,
        old_archive=old_archive,
        repaired_archive=repaired_archive,
    ) == "created"
    connection = sqlite3.connect(destination_database)
    try:
        assert connection.execute("SELECT shard_path FROM entries").fetchone()[0] == str(
            repaired_archive
        )
        archive = connection.execute(
            "SELECT archive_index, shard_path, archive_sha256 FROM archives"
        ).fetchone()
        assert archive == (7, str(repaired_archive), _sha256(repaired_archive))
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        assert metadata["base_inventory_path"] == "repaired"
    finally:
        connection.close()
    assert not os.path.samefile(source_database, destination_database)
