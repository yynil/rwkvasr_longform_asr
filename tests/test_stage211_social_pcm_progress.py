from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
progress_reporter = importlib.import_module("scripts.report_stage211_social_pcm_progress")


def _source(index: int) -> dict[str, object]:
    return {
        "source_index": index,
        "path": f"/inventory/source_{index:06d}.json",
        "sha256": f"{index + 1:064x}",
        "segments": 10 + index,
        "duration_ms": 3_600_000 * (index + 1),
        "tar_path": f"/materialized/source_{index:06d}.tar",
        "tar_sha256": f"{index + 11:064x}",
        "tar_size_bytes": 1000 * (index + 1),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    sources = [_source(index) for index in range(3)]
    inventory_path = tmp_path / "materialized_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "stage211_social_vad_materialized_inventory",
                "complete": True,
                "training_ready": False,
                "selected_rows": sum(int(row["segments"]) for row in sources),
                "source_receipts": sources,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "filtered"
    (output_root / "source_fingerprints").mkdir(parents=True)
    return inventory_path, output_root, sources


def _write_receipt(
    output_root: Path,
    source: dict[str, object],
) -> Path:
    index = int(source["source_index"])
    part_path = output_root / "source_fingerprints" / f"source_{index:06d}.jsonl"
    part_path.write_text('{"fingerprint":"value"}\n', encoding="utf-8")
    receipt_path = output_root / "source_fingerprints" / f"source_{index:06d}.receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "stage211_social_pcm_source_fingerprints",
                "complete": True,
                "source_index": index,
                "source_receipt_path": str(Path(str(source["path"])).resolve()),
                "source_receipt_sha256": source["sha256"],
                "tar_path": str(Path(str(source["tar_path"])).resolve()),
                "tar_sha256": source["tar_sha256"],
                "rows": source["segments"],
                "duration_ms": source["duration_ms"],
                "part_path": str(part_path.resolve()),
                "part_size_bytes": part_path.stat().st_size,
                "part_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt_path


def test_social_progress_reports_exact_weighted_prefix(tmp_path: Path) -> None:
    inventory_path, output_root, sources = _fixture(tmp_path)
    _write_receipt(output_root, sources[0])
    _write_receipt(output_root, sources[1])

    progress = progress_reporter.build_progress(inventory_path, output_root)

    assert progress["latest_completed_source_index"] == 1
    assert progress["receipts_contiguous"] is True
    assert progress["complete"] is False
    assert progress["progress"]["sources"] == {
        "covered": 2,
        "total": 3,
        "remaining": 1,
        "percent": pytest.approx(200.0 / 3.0),
    }
    assert progress["progress"]["bytes"]["covered"] == 3000
    assert progress["progress"]["rows"]["covered"] == 21
    assert progress["covered_hours"] == pytest.approx(3.0)
    assert progress["next_source"]["source_index"] == 2
    rendered = progress_reporter.render_text(progress)
    assert "sources=2/3 percent=66.6667" in rendered
    assert "duration_hours=3.000000/6.000000 percent=50.0000" in rendered


def test_social_progress_rejects_noncontiguous_receipts(tmp_path: Path) -> None:
    inventory_path, output_root, sources = _fixture(tmp_path)
    _write_receipt(output_root, sources[1])

    with pytest.raises(ValueError, match="exact contiguous prefix"):
        progress_reporter.build_progress(inventory_path, output_root)


def test_social_progress_rejects_changed_receipt_binding(tmp_path: Path) -> None:
    inventory_path, output_root, sources = _fixture(tmp_path)
    path = _write_receipt(output_root, sources[0])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["duration_ms"] = 1
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="receipt changed"):
        progress_reporter.build_progress(inventory_path, output_root)


def test_social_progress_accepts_inventory_bound_empty_source(tmp_path: Path) -> None:
    inventory_path, output_root, sources = _fixture(tmp_path)
    sources[0].update(
        {
            "segments": 0,
            "duration_ms": 0,
            "tar_path": None,
            "tar_sha256": None,
            "tar_size_bytes": None,
        }
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["source_receipts"] = sources
    inventory["selected_rows"] = sum(int(row["segments"]) for row in sources)
    inventory_path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")
    part_path = output_root / "source_fingerprints" / "source_000000.jsonl"
    part_path.write_bytes(b"")
    receipt_path = output_root / "source_fingerprints" / "source_000000.receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "stage211_social_pcm_source_fingerprints",
                "complete": True,
                "source_index": 0,
                "source_receipt_path": str(Path(str(sources[0]["path"])).resolve()),
                "source_receipt_sha256": sources[0]["sha256"],
                "tar_path": None,
                "tar_sha256": None,
                "rows": 0,
                "duration_ms": 0,
                "part_path": str(part_path.resolve()),
                "part_size_bytes": 0,
                "part_sha256": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    progress = progress_reporter.build_progress(inventory_path, output_root)

    assert progress["latest_completed_source_index"] == 0
    assert progress["progress"]["sources"]["covered"] == 1
    assert progress["progress"]["rows"]["covered"] == 0
    assert progress["progress"]["bytes"]["covered"] == 0
