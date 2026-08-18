from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from rwkvasr.eval.stage211_supplemental import _validate_parquet_runtime_layout


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
repack = importlib.import_module("scripts.repack_stage211_parquet_locality")
POLICY = repack.POLICY
build_locality_inventory = repack.build_locality_inventory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    return {
        "path": str(path),
        "num_samples": len(rows),
        "first_shard": rows[0]["shard_name"],
        "last_shard": rows[-1]["shard_name"],
        "source_label": rows[0]["source_dataset"],
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _parquet_row(
    source: str,
    shard: str,
    row_index: int,
    frames: int,
) -> dict[str, object]:
    return {
        "key": f"{source}-{row_index}",
        "utt_id": f"{source}-{row_index}",
        "source_dataset": source,
        "storage_kind": "parquet",
        "shard_name": shard,
        "audio_member": f"row-{row_index}",
        "audio_format": "wav",
        "json_member": f"row-{row_index}.json",
        "parquet_row_group": 0,
        "parquet_row_index": row_index,
        "num_frames": frames,
        "split": "train",
    }


def test_repack_groups_parquet_rows_by_row_group_max_without_changing_rows(
    tmp_path: Path,
) -> None:
    parts_root = tmp_path / "source_parts"
    clean_short = _parquet_row("peoples_speech_clean", "/usb/clean.parquet", 0, 100)
    clean_long = _parquet_row("peoples_speech_clean", "/usb/clean.parquet", 1, 250)
    dirty_short = _parquet_row("peoples_speech_dirty", "/usb/dirty.parquet", 0, 81)
    dirty_long = _parquet_row("peoples_speech_dirty", "/usb/dirty.parquet", 1, 159)
    ljspeech = {
        "key": "ljspeech-0",
        "utt_id": "ljspeech-0",
        "source_dataset": "ljspeech",
        "storage_kind": "zip",
        "shard_name": "/usb/ljspeech.zip",
        "audio_member": "a.wav",
        "audio_format": "wav",
        "json_member": "a.json",
        "num_frames": 120,
        "split": "train",
    }
    bucket_parts = {
        1: [
            _write_rows(parts_root / "clean-b1.jsonl", [clean_short]),
            _write_rows(parts_root / "dirty-b1.jsonl", [dirty_short, dirty_long]),
            _write_rows(parts_root / "ljspeech-b1.jsonl", [ljspeech]),
        ],
        3: [_write_rows(parts_root / "clean-b3.jsonl", [clean_long])],
    }
    manifest = {
        "version": 1,
        "root": "/",
        "source_length_index_path": "unused",
        "bucket_width": 80,
        "entries_per_part": 100_000,
        "splits": {
            "train": {
                "num_samples": 5,
                "buckets": [
                    {
                        "bucket_id": bucket_id,
                        "num_samples": sum(int(part["num_samples"]) for part in parts),
                        "parts": parts,
                    }
                    for bucket_id, parts in sorted(bucket_parts.items())
                ],
            },
            "eval": {"num_samples": 0, "buckets": []},
        },
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inventory = {
        "selected_rows": 5,
        "selected_hours": 1.0,
        "selected_counts_by_source": {
            "ljspeech": 1,
            "peoples_speech_clean": 2,
            "peoples_speech_dirty": 2,
        },
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": _sha256(manifest_path),
        "part_records": [part for parts in bucket_parts.values() for part in parts],
    }
    inventory_path = tmp_path / "source_inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    output_root = tmp_path / "output"
    result = build_locality_inventory(
        source_inventory_path=inventory_path,
        output_root=output_root,
        validate_output=False,
    )

    output_manifest = json.loads(
        (output_root / "webdataset_buckets_audio_text/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert output_manifest["batching_policy"] == POLICY
    assert output_manifest["splits"]["train"]["num_samples"] == 5
    by_source: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for bucket in output_manifest["splits"]["train"]["buckets"]:
        for part in bucket["parts"]:
            source = part["source_label"]
            path = Path(part["path"])
            if not path.is_absolute():
                path = output_root / "webdataset_buckets_audio_text" / path
            for line in path.read_text(encoding="utf-8").splitlines():
                by_source.setdefault(source, []).append((bucket["bucket_id"], json.loads(line)))

    assert [bucket for bucket, _ in by_source["peoples_speech_clean"]] == [3, 3]
    assert [bucket for bucket, _ in by_source["peoples_speech_dirty"]] == [1, 1]
    assert [bucket for bucket, _ in by_source["ljspeech"]] == [1]
    expected = sorted(
        [clean_short, clean_long, dirty_short, dirty_long, ljspeech],
        key=lambda row: str(row["key"]),
    )
    observed = sorted(
        [row for rows in by_source.values() for _, row in rows],
        key=lambda row: str(row["key"]),
    )
    assert observed == expected
    receipt = json.loads(
        (output_root / "parquet_locality_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["parquet_rows"] == 4
    assert receipt["parquet_row_groups"] == 2
    assert receipt["unchanged_rows"] == 1
    assert receipt["rows_by_source"] == {
        "peoples_speech_clean": 2,
        "peoples_speech_dirty": 2,
    }
    assert result["runtime_layout"]["policy"] == POLICY

    output_inventory_path = output_root / "supplemental_inventory.json"
    _validate_parquet_runtime_layout(
        result,
        inventory_path=output_inventory_path,
        manifest_path=output_root / "webdataset_buckets_audio_text/manifest.json",
        manifest_payload=output_manifest,
        selected_rows=5,
        source_counts=result["selected_counts_by_source"],
    )
    receipt_path = output_root / "parquet_locality_receipt.json"
    receipt["parquet_rows"] = 3
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result["runtime_layout"]["receipt_sha256"] = _sha256(receipt_path)
    with pytest.raises(ValueError, match="locality receipt is inconsistent"):
        _validate_parquet_runtime_layout(
            result,
            inventory_path=output_inventory_path,
            manifest_path=output_root / "webdataset_buckets_audio_text/manifest.json",
            manifest_payload=output_manifest,
            selected_rows=5,
            source_counts=result["selected_counts_by_source"],
        )
