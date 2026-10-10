from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
progress_reporter = importlib.import_module("scripts.report_stage211_base_public_pcm_progress")


def _archive(index: int, *, source: str) -> dict[str, object]:
    return {
        "archive_index": index,
        "storage_kind": "parquet",
        "shard_path": f"/media/usbhd/source/archive-{index:03d}.parquet",
        "source_dataset": source,
        "rows": 10 + index,
        "duration_ms": 3_600_000 * (index + 1),
        "archive_size_bytes": 1000 * (index + 1),
        "archive_mtime_ns": 100 + index,
        "archive_sha256": f"{index + 1:064x}",
    }


def _fixture(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    root = tmp_path / "audit"
    receipt_dir = root / "archive_fingerprints"
    receipt_dir.mkdir(parents=True)
    archives = [
        _archive(0, source="source_a"),
        _archive(1, source="source_a"),
        _archive(2, source="source_b"),
    ]
    connection = sqlite3.connect(root / "manifest_location_index.sqlite")
    connection.execute(
        "CREATE TABLE archives (archive_index INTEGER PRIMARY KEY, storage_kind TEXT, "
        "shard_path TEXT, source_dataset TEXT, rows INTEGER, duration_ms INTEGER, "
        "archive_size_bytes INTEGER, archive_mtime_ns INTEGER, archive_sha256 TEXT)"
    )
    connection.executemany(
        "INSERT INTO archives VALUES (:archive_index, :storage_kind, :shard_path, "
        ":source_dataset, :rows, :duration_ms, :archive_size_bytes, :archive_mtime_ns, "
        ":archive_sha256)",
        archives,
    )
    connection.commit()
    connection.close()
    return root, archives


def _write_receipt(root: Path, archive: dict[str, object]) -> Path:
    index = int(archive["archive_index"])
    path = root / "archive_fingerprints" / f"archive_{index:06d}.receipt.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "stage211_base_public_pcm_archive_fingerprints",
                "complete": True,
                "decode_failures": 0,
                **archive,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_weighted_progress_reports_exact_contiguous_prefix(tmp_path: Path) -> None:
    root, archives = _fixture(tmp_path)
    _write_receipt(root, archives[0])
    _write_receipt(root, archives[1])

    progress = progress_reporter.build_progress(root)

    assert progress["latest_completed_archive_index"] == 1
    assert progress["receipts_contiguous"] is True
    assert progress["complete"] is False
    assert progress["progress"]["archives"] == {
        "covered": 2,
        "total": 3,
        "remaining": 1,
        "percent": pytest.approx(200.0 / 3.0),
    }
    assert progress["progress"]["bytes"]["covered"] == 3000
    assert progress["progress"]["rows"]["covered"] == 21
    assert progress["covered_hours"] == pytest.approx(3.0)
    assert progress["next_archive"]["archive_index"] == 2
    assert progress["remaining_by_source"] == [
        {
            "source_dataset": "source_b",
            "first_archive_index": 2,
            "last_archive_index": 2,
            "archives": 1,
            "bytes": 3000,
            "rows": 12,
            "duration_ms": 10_800_000,
        }
    ]
    rendered = progress_reporter.render_text(progress)
    assert "archives=2/3 percent=66.6667" in rendered
    assert "duration_hours=3.000000/6.000000 percent=50.0000" in rendered


def test_weighted_progress_rejects_noncontiguous_receipts(tmp_path: Path) -> None:
    root, archives = _fixture(tmp_path)
    _write_receipt(root, archives[1])

    with pytest.raises(ValueError, match="exact contiguous prefix"):
        progress_reporter.build_progress(root)


def test_weighted_progress_rejects_changed_receipt_binding(tmp_path: Path) -> None:
    root, archives = _fixture(tmp_path)
    path = _write_receipt(root, archives[0])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["rows"] = 999
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="receipt changed"):
        progress_reporter.build_progress(root)
