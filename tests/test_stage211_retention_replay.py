from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from rwkvasr.data import load_webdataset_bucket_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
builder = importlib.import_module("scripts.build_stage211_retention_replay")
validator = importlib.import_module("scripts.validate_stage211_retention_replay")


def _write_source_manifest(root: Path, *, difficulty: str) -> tuple[Path, list[str]]:
    languages = ("zh",) if difficulty == "long" else ("en", "zh")
    buckets = []
    all_keys = []
    for input_bucket_id, num_frames in ((0, 79), (1, 161)):
        parts = []
        for language in languages:
            for source_index in range(2):
                source_dataset = f"{difficulty}_{language}_source{source_index}"
                part = (
                    root
                    / difficulty
                    / "train"
                    / f"bucket_{input_bucket_id:04d}"
                    / f"{source_dataset}.jsonl"
                )
                part.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for index in range(5):
                    key = f"{difficulty}_{language}_{source_index}_{input_bucket_id}_{index}"
                    all_keys.append(key)
                    rows.append(
                        {
                            "key": key,
                            "utt_id": key,
                            "language": language,
                            "num_frames": num_frames + index,
                            "source_dataset": source_dataset,
                            "tar_path": f"/audio/{source_dataset}.tar",
                            "audio_member": f"{key}.wav",
                        }
                    )
                part.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )
                parts.append(
                    {
                        "path": str(part.relative_to(root / difficulty)),
                        "num_samples": len(rows),
                        "source_label": source_dataset,
                    }
                )
        buckets.append(
            {
                "bucket_id": input_bucket_id,
                "num_samples": sum(part["num_samples"] for part in parts),
                "parts": parts,
            }
        )
    manifest_path = root / difficulty / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "root": "/",
                "source_length_index_path": "/unused",
                "bucket_width": 200,
                "entries_per_part": 100,
                "splits": {
                    "train": {
                        "num_samples": sum(bucket["num_samples"] for bucket in buckets),
                        "buckets": buckets,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, all_keys


def _write_exclusions(
    root: Path, *, sidecar_keys: list[str], fixed_keys: list[str]
) -> tuple[Path, Path]:
    sidecar_receipt = root / "sidecar" / "receipt.json"
    sidecar_receipt.parent.mkdir(parents=True)
    sidecar_receipt.write_text(
        json.dumps(
            {
                "artifact": "stratified_hidden_eval_manifest",
                "combined_samples": len(sidecar_keys),
                "selected_rows": [{"cell": "test", "key": key} for key in sidecar_keys],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    fixed_part = root / "fixed" / "part_000000.jsonl"
    fixed_part.parent.mkdir(parents=True)
    fixed_part.write_text(
        "".join(
            json.dumps(
                {
                    "key": key,
                    "utt_id": key,
                    "language": "zh",
                    "num_frames": 100,
                },
                sort_keys=True,
            )
            + "\n"
            for key in fixed_keys
        ),
        encoding="utf-8",
    )
    return sidecar_receipt, fixed_part


def _read_train_rows(manifest_path: Path) -> tuple[list[dict], list[str]]:
    manifest = load_webdataset_bucket_manifest(manifest_path)
    rows = []
    source_labels = []
    for bucket in manifest.splits["train"]:
        for part in bucket.parts:
            source_labels.append(str(part.source_label))
            rows.extend(
                json.loads(line)
                for line in Path(part.path).read_text(encoding="utf-8").splitlines()
            )
    return rows, source_labels


def test_build_stage211_retention_replay_is_balanced_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifests = {}
    source_keys = {}
    for difficulty in builder.DIFFICULTIES:
        manifest, keys = _write_source_manifest(tmp_path / "sources", difficulty=difficulty)
        source_manifests[difficulty] = manifest
        source_keys[difficulty] = keys
    excluded = [source_keys["easy"][0], source_keys["medium"][0]]
    fixed = [source_keys["hard"][0], source_keys["long"][0]]
    sidecar_receipt, fixed_eval_part = _write_exclusions(
        tmp_path,
        sidecar_keys=excluded,
        fixed_keys=fixed,
    )
    targets = {
        "easy_en": 8,
        "easy_zh": 8,
        "medium_en": 8,
        "medium_zh": 8,
        "hard_en": 4,
        "hard_zh": 4,
        "long_zh": None,
    }
    output_dir = tmp_path / "output"
    receipt = builder.build_retention_replay(
        source_manifests=source_manifests,
        output_dir=output_dir,
        stratified_receipt=sidecar_receipt,
        fixed_eval_part=fixed_eval_part,
        cell_targets=targets,
        seed=2111,
        bucket_width=80,
        max_rows_per_part=3,
    )

    expected_long = 19
    assert receipt["samples"] == 59
    assert receipt["unique_keys"] == receipt["samples"]
    assert len(receipt["builder"]["sha256"]) == 64
    assert receipt["cells"]["long_zh"]["samples"] == expected_long
    assert receipt["language_counts"] == {"en": 20, "zh": 39}
    assert receipt["manifest_train_samples"] == 59
    assert receipt["manifest_eval_samples"] == len(fixed)
    assert receipt["selection"]["realized_cell_targets"] == {
        **{cell: int(target) for cell, target in targets.items() if target is not None},
        "long_zh": expected_long,
    }

    manifest = load_webdataset_bucket_manifest(receipt["manifest_path"])
    assert manifest.bucket_width == 80
    assert sum(bucket.num_samples for bucket in manifest.splits["train"]) == 59
    assert sum(bucket.num_samples for bucket in manifest.splits["eval"]) == len(fixed)
    rows, source_labels = _read_train_rows(Path(receipt["manifest_path"]))
    selected_keys = {row["utt_id"] for row in rows}
    assert len(selected_keys) == len(rows) == 59
    assert selected_keys.isdisjoint(set(excluded + fixed))
    assert all(row["_stage211_replay_bucket_id"] == row["num_frames"] // 80 for row in rows)
    assert all(":" in source_label for source_label in source_labels)
    assert set(receipt["cells"]["easy_en"]["buckets"]) == {"0", "1", "2"}
    assert set(receipt["cells"]["easy_en"]["sources"]) == {
        "easy_en_source0",
        "easy_en_source1",
    }
    assert set(receipt["cells"]["easy_en"]["sources"].values()) == {4}
    assert all(len(part["sha256"]) == 64 for part in receipt["output_parts"])
    assert Path(receipt["capacity_preflight_path"]).is_file()
    assert len(receipt["capacity_preflight_sha256"]) == 64

    validated = validator.validate_retention_replay(
        output_dir / "receipt.json",
        expected_cell_targets=targets,
        expected_fixed_eval_samples=len(fixed),
    )
    assert validated["validated_unique_keys"] == 59
    assert len(validated["receipt_sha256"]) == 64

    def reject_capacity_rescan(**_: object) -> None:
        raise AssertionError("immutable capacity preflight was not reused")

    monkeypatch.setattr(builder, "_scan_capacities", reject_capacity_rescan)

    repeated = builder.build_retention_replay(
        source_manifests=source_manifests,
        output_dir=output_dir,
        stratified_receipt=sidecar_receipt,
        fixed_eval_part=fixed_eval_part,
        cell_targets=targets,
        seed=2111,
        bucket_width=80,
        max_rows_per_part=3,
    )
    assert repeated == receipt

    with pytest.raises(ValueError, match="Refusing to replace"):
        builder.build_retention_replay(
            source_manifests=source_manifests,
            output_dir=output_dir,
            stratified_receipt=sidecar_receipt,
            fixed_eval_part=fixed_eval_part,
            cell_targets=targets,
            seed=2112,
            bucket_width=80,
            max_rows_per_part=3,
        )

    first_part = Path(receipt["output_parts"][0]["path"])
    first_part.write_text(
        first_part.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output part SHA-256 mismatch"):
        validator.validate_retention_replay(
            output_dir / "receipt.json",
            expected_cell_targets=targets,
            expected_fixed_eval_samples=len(fixed),
        )


def test_build_stage211_retention_replay_rejects_insufficient_capacity(
    tmp_path: Path,
) -> None:
    source_manifests = {
        difficulty: _write_source_manifest(tmp_path / "sources", difficulty=difficulty)[0]
        for difficulty in builder.DIFFICULTIES
    }
    sidecar_receipt, fixed_eval_part = _write_exclusions(
        tmp_path,
        sidecar_keys=[],
        fixed_keys=[],
    )
    targets = dict(builder.DEFAULT_CELL_TARGETS)
    targets["easy_en"] = 10_000

    with pytest.raises(ValueError, match="Insufficient replay capacity for easy_en"):
        builder.build_retention_replay(
            source_manifests=source_manifests,
            output_dir=tmp_path / "output",
            stratified_receipt=sidecar_receipt,
            fixed_eval_part=fixed_eval_part,
            cell_targets=targets,
        )
