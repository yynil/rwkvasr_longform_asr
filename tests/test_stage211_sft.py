from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

from rwkvasr.eval.stage211_gate import STAGE211_PUBLIC_BENCHMARKS, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sft_runner = importlib.import_module("scripts.run_stage211_labeled_sft")
sft_finalizer = importlib.import_module("scripts.finalize_stage211_labeled_sft")
LABELED_EXPECTED = sft_runner.LABELED_EXPECTED


def test_stage211_sft_controller_preserves_virtualenv_python() -> None:
    assert sft_runner.PYTHON == Path(sys.executable)


def _labeled_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "labeled"
    root.mkdir()
    length_index = root / "lengths.jsonl"
    length_index.write_text("{}\n", encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return root, length_index, manifest


def test_validate_stage211_labeled_audit_exact_contract(tmp_path: Path) -> None:
    root, length_index, manifest = _labeled_paths(tmp_path)
    audit = {
        "webdataset_root": str(root.resolve()),
        "length_index_path": str(length_index.resolve()),
        "bucket_manifest_path": str(manifest.resolve()),
        **LABELED_EXPECTED,
    }

    assert (
        sft_runner._validate_labeled_audit(
            audit,
            labeled_root=root,
            length_index=length_index,
            bucket_manifest=manifest,
        )
        == audit
    )

    audit["ctc_tokens"] -= 1
    with pytest.raises(ValueError, match="ctc_tokens mismatch"):
        sft_runner._validate_labeled_audit(
            audit,
            labeled_root=root,
            length_index=length_index,
            bucket_manifest=manifest,
        )


def test_stage211_sft_runner_command_distinguishes_fresh_and_resume(
    tmp_path: Path,
) -> None:
    root, length_index, manifest = _labeled_paths(tmp_path)
    checkpoint = tmp_path / "init.pt"
    receipt = tmp_path / "receipt.json"
    nano = tmp_path / "nano.pt"
    fresh = sft_runner._runner_command(
        output_dir=tmp_path / "run",
        config_dir=tmp_path / "configs",
        bucket_manifest=manifest,
        labeled_root=root,
        length_index=length_index,
        nano_checkpoint=nano,
        master_port=29634,
        init_checkpoint=checkpoint,
        promotion_receipt=receipt,
        smoke=False,
        dry_run=False,
    )
    resumed = sft_runner._runner_command(
        output_dir=tmp_path / "run",
        config_dir=tmp_path / "configs",
        bucket_manifest=manifest,
        labeled_root=root,
        length_index=length_index,
        nano_checkpoint=nano,
        master_port=29634,
        init_checkpoint=None,
        promotion_receipt=None,
        smoke=False,
        dry_run=False,
    )

    assert fresh[fresh.index("--phase") + 1] == "sft"
    assert fresh[fresh.index("--init-checkpoint") + 1] == str(checkpoint)
    assert fresh[fresh.index("--promotion-receipt") + 1] == str(receipt)
    assert "--init-checkpoint" not in resumed
    assert "--promotion-receipt" not in resumed


def test_validate_stage211_sft_completion_binds_artifacts(tmp_path: Path) -> None:
    root, length_index, manifest = _labeled_paths(tmp_path)
    artifacts: dict[str, Path] = {
        "bucket_manifest": manifest,
        "length_index": length_index,
        "provenance": tmp_path / "provenance.json",
        "init_checkpoint": tmp_path / "init.pt",
        "logits_promotion_receipt": tmp_path / "receipt.json",
        "completion_checkpoint": tmp_path / "step-12019.pt",
        "training_log": tmp_path / "train.log",
    }
    artifacts["provenance"].write_text("{}\n", encoding="utf-8")
    artifacts["logits_promotion_receipt"].write_text("{}\n", encoding="utf-8")
    artifacts["training_log"].write_text("complete\n", encoding="utf-8")
    torch.save({"step": 0}, artifacts["init_checkpoint"])
    torch.save(
        {"step": LABELED_EXPECTED["estimated_train_steps"]},
        artifacts["completion_checkpoint"],
    )
    completion = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "labeled_sft_completion",
        "phase": "sft",
        "complete": True,
        "epochs": 1,
        "batch_size": 12,
        "world_size": 4,
        "frame_budget": 8_000,
        **LABELED_EXPECTED,
        "labeled_webdataset_root": str(root),
    }
    for name, path in artifacts.items():
        completion[f"{name}_path"] = str(path)
        completion[f"{name}_sha256"] = sha256_file(path)
    completion_path = tmp_path / "sft_complete.json"
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded, checkpoint = sft_runner._validate_completion(completion_path)

    assert loaded == completion
    assert checkpoint == artifacts["completion_checkpoint"].resolve()


def _public_benchmark(error_rates: dict[str, float]) -> dict[str, object]:
    return {
        "results": [
            {
                "dataset": dataset,
                "manifest_sha256": f"manifest-{dataset}",
                "nano_prediction_sha256": f"nano-{dataset}",
                "student_error_rate": error_rates[dataset],
            }
            for dataset in STAGE211_PUBLIC_BENCHMARKS
        ]
    }


def test_stage211_sft_public_gate_requires_zero_dataset_regressions() -> None:
    baseline_rates = {
        dataset: 0.20 + index * 0.01 for index, dataset in enumerate(STAGE211_PUBLIC_BENCHMARKS)
    }
    improved_rates = dict(baseline_rates)
    improved_rates["aishell1_test"] -= 0.01
    passed = sft_finalizer._build_sft_public_progress(
        baseline=_public_benchmark(baseline_rates),
        candidate=_public_benchmark(improved_rates),
    )

    assert passed["gate_passed"] is True
    assert passed["no_dataset_regression"] is True
    assert passed["improved_datasets"] == 1

    regressed_rates = dict(improved_rates)
    regressed_rates["librispeech_test_clean"] += 0.001
    failed = sft_finalizer._build_sft_public_progress(
        baseline=_public_benchmark(baseline_rates),
        candidate=_public_benchmark(regressed_rates),
    )

    assert failed["gate_passed"] is False
    assert failed["no_dataset_regression"] is False
