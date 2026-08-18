from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
repack = importlib.import_module("scripts.repack_stage211_retention_replay_locality")
validator = importlib.import_module("scripts.validate_stage211_retention_replay")
supplemental_validator = importlib.import_module(
    "scripts.validate_stage211_supplemental_retention"
)
stage211_gate = importlib.import_module("rwkvasr.eval.stage211_gate")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "num_samples": len(rows),
        "source_label": (
            f"{rows[0]['_stage211_replay_cell']}:{rows[0]['source_dataset']}"
            if "_stage211_replay_cell" in rows[0]
            else str(rows[0]["source_dataset"])
        ),
    }


def _parquet_row(
    *,
    source: str,
    shard: str,
    row_index: int,
    frames: int,
) -> dict[str, object]:
    key = f"{source}-{row_index}"
    return {
        "_stage211_replay_bucket_id": frames // 80,
        "_stage211_replay_cell": "supplemental_en",
        "_stage211_replay_row_sha256": "0" * 64,
        "_stage211_replay_score": "1" * 64,
        "_stage211_replay_seed": 2112,
        "_stage211_replay_source": source,
        "_stage211_replay_source_part": "/selection/source.jsonl",
        "key": key,
        "utt_id": key,
        "language": "en",
        "num_frames": frames,
        "source_dataset": source,
        "storage_kind": "parquet",
        "shard_name": shard,
        "parquet_row_group": 0,
        "parquet_row_index": row_index,
        "audio_member": f"{key}.flac",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    clean_short = _parquet_row(
        source="peoples_speech_clean",
        shard="/usb/clean.parquet",
        row_index=0,
        frames=100,
    )
    clean_long = _parquet_row(
        source="peoples_speech_clean",
        shard="/usb/clean.parquet",
        row_index=1,
        frames=250,
    )
    dirty_short = _parquet_row(
        source="peoples_speech_dirty",
        shard="/usb/dirty.parquet",
        row_index=0,
        frames=81,
    )
    dirty_long = _parquet_row(
        source="peoples_speech_dirty",
        shard="/usb/dirty.parquet",
        row_index=1,
        frames=159,
    )
    non_parquet = {
        "key": "ljspeech-0",
        "utt_id": "ljspeech-0",
        "language": "en",
        "num_frames": 120,
        "source_dataset": "ljspeech",
        "storage_kind": "zip",
        "shard_name": "/usb/ljspeech.zip",
        "audio_member": "a.wav",
    }
    eval_row = {
        "key": "eval-0",
        "utt_id": "eval-0",
        "language": "zh",
        "num_frames": 100,
        "source_dataset": "aishell3",
        "storage_kind": "tar",
        "shard_name": "/usb/eval.tar",
        "audio_member": "eval.wav",
    }
    bucket_parts = {
        1: [
            _write_rows(tmp_path / "parts/clean-short.jsonl", [clean_short]),
            _write_rows(tmp_path / "parts/dirty.jsonl", [dirty_short, dirty_long]),
            _write_rows(tmp_path / "parts/ljspeech.jsonl", [non_parquet]),
        ],
        3: [_write_rows(tmp_path / "parts/clean-long.jsonl", [clean_long])],
    }
    eval_part = _write_rows(tmp_path / "parts/eval.jsonl", [eval_row])
    manifest = {
        "version": 1,
        "root": "/",
        "source_length_index_path": "/selection/receipt.json",
        "bucket_width": 80,
        "entries_per_part": 100,
        "splits": {
            "train": {
                "num_samples": 5,
                "buckets": [
                    {
                        "bucket_id": bucket,
                        "num_samples": sum(int(part["num_samples"]) for part in parts),
                        "parts": parts,
                    }
                    for bucket, parts in sorted(bucket_parts.items())
                ],
            },
            "eval": {
                "num_samples": 1,
                "buckets": [{"bucket_id": 1, "num_samples": 1, "parts": [eval_part]}],
            },
        },
    }
    manifest_path = tmp_path / "selection/manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selection_receipt = tmp_path / "selection/receipt.json"
    selection_receipt.write_text('{"artifact":"selection"}\n', encoding="utf-8")
    selection = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "retention_replay_manifest",
        "samples": 5,
        "unique_keys": 5,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "validated_unique_keys": 5,
        "receipt_path": str(selection_receipt),
        "receipt_sha256": _sha256(selection_receipt),
    }
    return selection_receipt, manifest_path, selection


def test_runtime_layout_groups_parquet_rows_without_changing_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_receipt, source_manifest, selection = _fixture(tmp_path)
    monkeypatch.setattr(repack, "validate_retention_replay", lambda _: selection)
    output_root = tmp_path / "runtime"
    receipt = repack.build_runtime_layout(
        selection_receipt_path=selection_receipt,
        output_root=output_root,
        validate_output=False,
    )
    runtime_manifest_path = output_root / "webdataset_buckets_audio_text/manifest.json"
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    rows_by_source: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for bucket in runtime_manifest["splits"]["train"]["buckets"]:
        for part in bucket["parts"]:
            path = Path(part["path"])
            if not path.is_absolute():
                path = runtime_manifest_path.parent / path
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                rows_by_source.setdefault(str(row["source_dataset"]), []).append(
                    (int(bucket["bucket_id"]), row)
                )

    assert [bucket for bucket, _ in rows_by_source["peoples_speech_clean"]] == [3, 3]
    assert [bucket for bucket, _ in rows_by_source["peoples_speech_dirty"]] == [1, 1]
    assert [bucket for bucket, _ in rows_by_source["ljspeech"]] == [1]
    assert receipt["selected_rows"] == 5
    assert receipt["parquet_rows"] == 4
    assert receipt["parquet_row_groups"] == 2
    assert receipt["runtime_adjacent_same_row_group"] == 2
    assert runtime_manifest["splits"]["eval"] == json.loads(
        source_manifest.read_text(encoding="utf-8")
    )["splits"]["eval"]

    monkeypatch.setattr(validator, "validate_retention_replay", lambda _: selection)
    validated = validator._validate_runtime_layout_replay(
        output_root / "receipt.json",
        receipt,
    )
    assert validated["manifest_path"] == str(runtime_manifest_path)
    assert validated["selection_receipt_path"] == str(selection_receipt)


def test_runtime_layout_rejects_changed_rewritten_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_receipt, _, selection = _fixture(tmp_path)
    monkeypatch.setattr(repack, "validate_retention_replay", lambda _: selection)
    output_root = tmp_path / "runtime"
    receipt = repack.build_runtime_layout(
        selection_receipt_path=selection_receipt,
        output_root=output_root,
        validate_output=False,
    )
    rewritten = Path(receipt["rewritten_parts"][0]["path"])
    rewritten.write_text(rewritten.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    monkeypatch.setattr(validator, "validate_retention_replay", lambda _: selection)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validator._validate_runtime_layout_replay(output_root / "receipt.json", receipt)


def test_validate_retention_replay_dispatches_runtime_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": validator.RUNTIME_LAYOUT_ARTIFACT,
                "complete": True,
                "policy": validator.RUNTIME_LAYOUT_POLICY,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        validator,
        "_validate_runtime_layout_replay",
        lambda path, receipt: {"receipt_path": str(path), "artifact": receipt["artifact"]},
    )
    result = validator.validate_retention_replay(receipt_path)
    assert result == {
        "receipt_path": str(receipt_path),
        "artifact": validator.RUNTIME_LAYOUT_ARTIFACT,
    }


def test_phase_gate_binding_uses_runtime_layout_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_receipt, source_manifest, selection = _fixture(tmp_path)
    selection_receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "pipeline": "stage211",
                "artifact": "retention_replay_manifest",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection.update(
        {
            "schema_version": 2,
            "pipeline": "stage211",
            "artifact": "retention_replay_manifest",
            "samples": 5,
            "unique_keys": 5,
            "total_hours": 1.25,
            "receipt_path": str(selection_receipt.resolve()),
            "receipt_sha256": _sha256(selection_receipt),
        }
    )
    monkeypatch.setattr(repack, "validate_retention_replay", lambda _: selection)
    output_root = tmp_path / "runtime"
    runtime_receipt = repack.build_runtime_layout(
        selection_receipt_path=selection_receipt,
        output_root=output_root,
        validate_output=False,
    )
    runtime_receipt_path = output_root / "receipt.json"
    runtime_manifest_path = output_root / "webdataset_buckets_audio_text/manifest.json"

    bound_file = tmp_path / "bound.json"
    bound_file.write_text("{}\n", encoding="utf-8")
    bound = {"path": str(bound_file.resolve()), "sha256": _sha256(bound_file)}
    selection.update(
        {
            "builder": bound,
            "capacity_preflight_path": bound["path"],
            "capacity_preflight_sha256": bound["sha256"],
            "base_replay_receipt": bound,
            "stratified_hidden_eval": bound,
            "supplemental_inputs": {
                "inventory_path": bound["path"],
                "inventory_sha256": bound["sha256"],
                "profile_receipt_path": bound["path"],
                "profile_receipt_sha256": bound["sha256"],
                "manifest_path": bound["path"],
                "manifest_sha256": bound["sha256"],
            },
            "source_manifest": bound,
            "supplemental_output_parts": [bound],
        }
    )
    monkeypatch.setattr(
        supplemental_validator,
        "validate_supplemental_retention_replay",
        lambda _: selection,
    )
    correction = {
        "replay_receipt_path": str(runtime_receipt_path.resolve()),
        "replay_receipt_sha256": _sha256(runtime_receipt_path),
        "bucket_manifest_path": str(runtime_manifest_path.resolve()),
        "bucket_manifest_sha256": _sha256(runtime_manifest_path),
        "rows": 5,
        "hours": 1.25,
    }

    replay, manifest_path = stage211_gate._validate_stage211_retention_replay_binding(
        correction
    )

    assert manifest_path == runtime_manifest_path.resolve()
    assert manifest_path != source_manifest.resolve()
    assert replay["receipt_path"] == str(runtime_receipt_path.resolve())
    assert replay["runtime_layout"] == runtime_receipt
