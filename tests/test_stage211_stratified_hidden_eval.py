from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from rwkvasr.data import load_webdataset_bucket_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
builder = importlib.import_module("scripts.build_stage211_stratified_hidden_eval")
summarizer = importlib.import_module(
    "scripts.summarize_stage211_stratified_hidden_eval"
)


def _write_source_manifest(
    root: Path,
    *,
    difficulty: str,
    languages: tuple[str, ...],
) -> Path:
    buckets = []
    for bucket_id in (1, 2):
        parts = []
        for language in languages:
            part = root / difficulty / "train" / f"bucket_{bucket_id:04d}" / f"{language}.jsonl"
            part.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for index in range(8):
                key = f"{difficulty}_{language}_{bucket_id}_{index}"
                rows.append(
                    json.dumps(
                        {
                            "key": key,
                            "utt_id": key,
                            "language": language,
                            "num_frames": bucket_id * 80 - index,
                            "source_dataset": f"source_{language}_{bucket_id}",
                            "tar_path": f"/audio/{difficulty}.tar",
                            "audio_member": f"{key}.wav",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            part.write_text("".join(rows), encoding="utf-8")
            parts.append(
                {
                    "path": str(part.relative_to(root / difficulty)),
                    "num_samples": len(rows),
                }
            )
        buckets.append(
            {
                "bucket_id": bucket_id,
                "num_samples": sum(part["num_samples"] for part in parts),
                "parts": parts,
            }
        )
    manifest = root / difficulty / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "root": "/",
                "source_length_index_path": "/unused",
                "bucket_width": 80,
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
    return manifest


def test_build_stage211_stratified_hidden_eval_cells(tmp_path: Path) -> None:
    source_manifests = {
        difficulty: _write_source_manifest(
            tmp_path / "sources",
            difficulty=difficulty,
            languages=(("zh",) if difficulty == "long" else ("en", "zh")),
        )
        for difficulty in builder.DIFFICULTIES
    }
    output_dir = tmp_path / "output"
    receipt = builder.build_stratified_eval(
        source_manifests=source_manifests,
        output_dir=output_dir,
        per_cell=4,
        candidates_per_part=4,
        max_rows_per_part=8,
        seed=211,
    )

    assert receipt["combined_samples"] == 28
    assert set(receipt["cells"]) == {
        "easy_en",
        "easy_zh",
        "medium_en",
        "medium_zh",
        "hard_en",
        "hard_zh",
        "long_zh",
    }
    assert len(receipt["selected_rows"]) == 28
    assert all(len(row["row_sha256"]) == 64 for row in receipt["selected_rows"])
    assert all(len(row["source_part_sha256"]) == 64 for row in receipt["selected_rows"])

    for cell_name, cell in receipt["cells"].items():
        manifest = load_webdataset_bucket_manifest(cell["manifest_path"])
        assert sum(bucket.num_samples for bucket in manifest.splits["eval"]) == 4
        part_paths = [
            Path(part.path)
            for bucket in manifest.splits["eval"]
            for part in bucket.parts
        ]
        rows = [
            json.loads(line)
            for part_path in part_paths
            for line in part_path.read_text(encoding="utf-8").splitlines()
        ]
        assert {row["_stage211_sidecar_cell"] for row in rows} == {cell_name}

    repeated = builder.build_stratified_eval(
        source_manifests=source_manifests,
        output_dir=output_dir,
        per_cell=4,
        candidates_per_part=4,
        max_rows_per_part=8,
        seed=211,
    )
    assert repeated == receipt

    with pytest.raises(ValueError, match="Refusing to replace"):
        builder.build_stratified_eval(
            source_manifests=source_manifests,
            output_dir=output_dir,
            per_cell=4,
            candidates_per_part=4,
            max_rows_per_part=8,
            seed=212,
        )


def test_summarize_stage211_stratified_hidden_eval(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest_easy_en.json"
    manifest.write_text("{}\n", encoding="utf-8")
    baseline_checkpoint = tmp_path / "baseline.pt"
    candidate_checkpoint = tmp_path / "step-20.pt"
    baseline_checkpoint.write_bytes(b"baseline")
    candidate_checkpoint.write_bytes(b"candidate")
    receipt = {
        "artifact": "stratified_hidden_eval_manifest",
        "cells": {
            "easy_en": {
                "samples": 2,
                "manifest_path": str(manifest),
                "manifest_sha256": summarizer.sha256_file(manifest),
            }
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    layers = {
        str(index): {
            "loss": 1.0,
            "cosine": 0.5,
            "rms_ratio": 1.0,
        }
        for index in range(70)
    }
    for role, checkpoint, loss in (
        ("baseline", baseline_checkpoint, 1.0),
        ("candidate", candidate_checkpoint, 0.75),
    ):
        report_layers = {
            layer_id: {
                **metrics,
                "loss": metrics["loss"] * loss,
                "cosine": metrics["cosine"] + (0.1 if role == "candidate" else 0.0),
            }
            for layer_id, metrics in layers.items()
        }
        report = {
            "role": role,
            "phase": "mixer",
            "eval_samples": 2,
            "eval_loss": loss,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": summarizer.sha256_file(checkpoint),
            "eval_provenance": {
                "bucket_manifest_path": str(manifest.resolve()),
                "bucket_manifest_sha256": summarizer.sha256_file(manifest),
                "split_samples": 2,
            },
            "layer_components": {"mixer": report_layers},
        }
        (eval_dir / f"easy_en_{role}.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )

    summary = summarizer.summarize(
        receipt_path=receipt_path,
        eval_dir=eval_dir,
        output_path=tmp_path / "summary.json",
    )

    assert summary["macro"]["samples"] == 2
    assert summary["macro"]["relative_change_pct"] == pytest.approx(-25.0)
    assert summary["cells"]["easy_en"]["layers_loss_improved"] == 70
    assert summary["cells"]["easy_en"]["layers_cosine_improved"] == 70
