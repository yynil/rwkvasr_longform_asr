from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
base_builder = importlib.import_module("scripts.build_stage211_retention_replay")
stratified_builder = importlib.import_module("scripts.build_stage211_stratified_hidden_eval")
supplemental_builder = importlib.import_module("scripts.build_stage211_supplemental_retention")
supplemental_validator = importlib.import_module("scripts.validate_stage211_supplemental_retention")
replay_validator = importlib.import_module("scripts.validate_stage211_retention_replay")


def _write_manifest(
    root: Path,
    *,
    difficulty: str,
    languages: tuple[str, ...],
    rows_per_stratum: int = 8,
) -> Path:
    buckets = []
    for input_bucket, frames in ((0, 79), (1, 161)):
        parts = []
        for language in languages:
            for source_index in range(2):
                source_name = f"{difficulty}_{language}_source{source_index}"
                part = root / difficulty / f"{source_name}_{input_bucket}.jsonl"
                part.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for index in range(rows_per_stratum):
                    key = f"{difficulty}_{language}_{source_index}_{input_bucket}_{index}"
                    rows.append(
                        {
                            "key": key,
                            "utt_id": key,
                            "language": language,
                            "num_frames": frames + index,
                            "source_dataset": source_name,
                            "tar_path": f"/audio/{source_name}.tar",
                            "audio_member": f"{key}.wav",
                        }
                    )
                part.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )
                parts.append(
                    {
                        "path": str(part.resolve()),
                        "num_samples": len(rows),
                        "source_label": source_name,
                    }
                )
        buckets.append(
            {
                "bucket_id": input_bucket,
                "num_samples": sum(int(part["num_samples"]) for part in parts),
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


def _write_fixed_eval(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "key": "fixed-eval-only",
                "utt_id": "fixed-eval-only",
                "language": "zh",
                "num_frames": 100,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_and_validate_stage211_supplemental_retention_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifests = {
        difficulty: _write_manifest(
            tmp_path / "sources",
            difficulty=difficulty,
            languages=("zh",) if difficulty == "long" else ("en", "zh"),
        )
        for difficulty in base_builder.DIFFICULTIES
    }
    base_stratified_dir = tmp_path / "base-stratified"
    stratified_builder.build_stratified_eval(
        source_manifests=source_manifests,
        output_dir=base_stratified_dir,
        per_cell=2,
        candidates_per_part=4,
        max_rows_per_part=8,
        seed=211,
    )
    fixed_eval = tmp_path / "fixed" / "part.jsonl"
    _write_fixed_eval(fixed_eval)
    base_targets = {cell: 2 for cell in base_builder.CELLS}
    base_replay_dir = tmp_path / "base-replay"
    base_builder.build_retention_replay(
        source_manifests=source_manifests,
        output_dir=base_replay_dir,
        stratified_receipt=base_stratified_dir / "receipt.json",
        fixed_eval_part=fixed_eval,
        cell_targets=base_targets,
        seed=2111,
        max_rows_per_part=3,
    )

    supplemental_manifest = _write_manifest(
        tmp_path / "sources",
        difficulty="supplemental",
        languages=("en", "zh"),
    )
    inventory = tmp_path / "supplemental-inventory.json"
    profile = tmp_path / "supplemental-profile.json"
    inventory.write_text("{}\n", encoding="utf-8")
    profile.write_text("{}\n", encoding="utf-8")
    supplemental_rows = 64
    validated_builder_inputs = {
        "inventory": {},
        "inventory_path": str(inventory.resolve()),
        "inventory_sha256": supplemental_builder.sha256_file(inventory),
        "profile_receipt": {},
        "profile_receipt_path": str(profile.resolve()),
        "profile_receipt_sha256": supplemental_builder.sha256_file(profile),
        "manifest_path": str(supplemental_manifest.resolve()),
        "manifest_sha256": supplemental_builder.sha256_file(supplemental_manifest),
        "rows": supplemental_rows,
        "hours": 1.0,
    }
    monkeypatch.setattr(
        supplemental_builder,
        "_validate_supplemental_inputs",
        lambda **kwargs: validated_builder_inputs,
    )
    monkeypatch.setattr(
        supplemental_validator,
        "_validate_supplemental_inputs",
        lambda record: {
            "inventory": {},
            "bucket_manifest_path": str(supplemental_manifest.resolve()),
            "rows": supplemental_rows,
            "hours": 1.0,
        },
    )

    output_root = tmp_path / "v2"
    receipt = supplemental_builder.build_supplemental_retention(
        base_replay_receipt_path=base_replay_dir / "receipt.json",
        base_stratified_receipt_path=base_stratified_dir / "receipt.json",
        supplemental_inventory_path=inventory,
        supplemental_profile_receipt_path=profile,
        output_root=output_root,
        base_cell_targets=base_targets,
        base_fixed_eval_samples=1,
        replay_per_language=6,
        eval_per_cell=2,
        candidates_per_part=4,
        max_eval_scan_rows_per_part=8,
        max_rows_per_part=3,
    )

    assert receipt["schema_version"] == 2
    assert receipt["samples"] == 26
    assert receipt["supplemental_samples"] == 12
    assert receipt["language_counts"] == {"en": 12, "zh": 14}
    assert set(receipt["cells"]) == set(supplemental_builder.ALL_CELLS)
    assert receipt["cells"]["supplemental_en"]["samples"] == 6
    assert receipt["cells"]["supplemental_zh"]["samples"] == 6

    stratified = supplemental_validator.validate_stratified_hidden_eval_v2(
        output_root / "stratified_hidden_eval_v3" / "receipt.json"
    )
    assert stratified["combined_samples"] == 18
    assert stratified["validated_unique_keys"] == 18
    assert set(stratified["cells"]) == set(supplemental_builder.ALL_CELLS)

    validated = supplemental_validator.validate_supplemental_retention_replay(
        output_root / "retention_replay_v3" / "receipt.json"
    )
    assert validated["validated_unique_keys"] == 26
    assert validated["selected_keys_sha256"] == receipt["selected_keys_sha256"]
    dispatched = replay_validator.validate_retention_replay(
        output_root / "retention_replay_v3" / "receipt.json"
    )
    assert dispatched["validated_unique_keys"] == 26

    reused = supplemental_builder._reuse_validated_supplemental_retention(
        base_replay_receipt_path=base_replay_dir / "receipt.json",
        base_stratified_receipt_path=base_stratified_dir / "receipt.json",
        supplemental_inventory_path=inventory,
        supplemental_profile_receipt_path=profile,
        output_root=output_root,
        replay_per_language=6,
        eval_per_cell=2,
    )
    assert reused is not None
    assert reused["receipt_sha256"] == supplemental_builder.sha256_file(
        output_root / "retention_replay_v3" / "receipt.json"
    )
    assert (
        supplemental_builder._reuse_validated_supplemental_retention(
            base_replay_receipt_path=base_replay_dir / "receipt.json",
            base_stratified_receipt_path=base_stratified_dir / "receipt.json",
            supplemental_inventory_path=inventory,
            supplemental_profile_receipt_path=profile,
            output_root=output_root,
            replay_per_language=7,
            eval_per_cell=2,
        )
        is None
    )

    first_part = Path(receipt["supplemental_output_parts"][0]["path"])
    first_part.write_text(
        first_part.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        supplemental_validator.validate_supplemental_retention_replay(
            output_root / "retention_replay_v3" / "receipt.json"
        )


def test_supplemental_retention_main_reuses_validated_output_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = {
        name: tmp_path / f"{name}.json"
        for name in (
            "base_replay_receipt",
            "base_stratified_receipt",
            "supplemental_inventory",
            "supplemental_profile_receipt",
        )
    }
    for path in inputs.values():
        path.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "output"
    parser = argparse.ArgumentParser()
    monkeypatch.setattr(
        parser,
        "parse_args",
        lambda: argparse.Namespace(
            **inputs,
            output_root=output_root,
            replay_per_language=6,
            eval_per_cell=2,
            replay_dir_name="retention_replay_v3",
            stratified_dir_name="stratified_hidden_eval_v3",
        ),
    )
    monkeypatch.setattr(supplemental_builder, "build_parser", lambda: parser)
    reused = {
        "samples": 26,
        "total_hours": 1.25,
        "manifest_path": str(tmp_path / "manifest.json"),
    }
    reuse_calls: list[dict[str, object]] = []

    def reuse_validated(**kwargs: object) -> dict[str, object]:
        reuse_calls.append(kwargs)
        return reused

    monkeypatch.setattr(
        supplemental_builder,
        "_reuse_validated_supplemental_retention",
        reuse_validated,
    )

    def fail_build(**kwargs: object) -> dict[str, object]:
        raise AssertionError("full builder must not run after validated reuse")

    monkeypatch.setattr(
        supplemental_builder,
        "build_supplemental_retention",
        fail_build,
    )

    assert supplemental_builder.main() == 0
    assert reuse_calls == [
        {
            "base_replay_receipt_path": inputs["base_replay_receipt"],
            "base_stratified_receipt_path": inputs["base_stratified_receipt"],
            "supplemental_inventory_path": inputs["supplemental_inventory"],
            "supplemental_profile_receipt_path": inputs["supplemental_profile_receipt"],
            "output_root": output_root.resolve(),
            "replay_per_language": 6,
            "eval_per_cell": 2,
            "replay_dir_name": "retention_replay_v3",
            "stratified_dir_name": "stratified_hidden_eval_v3",
        }
    ]
    assert (output_root / ".stage211_supplemental_retention_v2.lock").is_file()
    assert "reused=true samples=26" in capsys.readouterr().out
