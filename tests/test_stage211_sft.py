from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rwkvasr.config import save_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    stage211_phase_train_config_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sft_runner = importlib.import_module("scripts.run_stage211_labeled_sft")
sft_finalizer = importlib.import_module("scripts.finalize_stage211_labeled_sft")
stepwise_report = importlib.import_module("scripts.create_stage211_stepwise_report")
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


def test_stage211_sft_formal_progress_detection_is_fail_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert sft_runner._formal_training_started(run_dir) is False

    (run_dir / "latest_checkpoint.yaml").write_text("- invalid-record\n", encoding="utf-8")
    assert sft_runner._formal_training_started(run_dir) is True

    (run_dir / "latest_checkpoint.yaml").unlink()
    log_dir = run_dir / "logs"
    log_dir.mkdir()
    training_log = log_dir / "sft_full_12045steps.log"
    training_log.write_text("[deepspeed-train] step=0 loss=1.0\n", encoding="utf-8")
    assert sft_runner._formal_training_started(run_dir) is False
    training_log.write_text("[deepspeed-train] step=1 loss=0.9\n", encoding="utf-8")
    assert sft_runner._formal_training_started(run_dir) is True


def test_stage211_sft_refuses_retroactive_smoke_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, length_index, manifest = _labeled_paths(tmp_path)
    output_dir = tmp_path / "run"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "sft_full_12045steps.log").write_text(
        "[deepspeed-train] step=1 loss=0.9\n",
        encoding="utf-8",
    )
    init_checkpoint = tmp_path / "init.pt"
    promotion_receipt = tmp_path / "promotion.json"
    nano_checkpoint = tmp_path / "nano.pt"
    torch.save({"step": 0}, init_checkpoint)
    promotion_receipt.write_text("{}\n", encoding="utf-8")
    nano_checkpoint.write_bytes(b"nano")
    audit = {
        "webdataset_root": str(root.resolve()),
        "length_index_path": str(length_index.resolve()),
        "bucket_manifest_path": str(manifest.resolve()),
        **LABELED_EXPECTED,
    }
    monkeypatch.setattr(sft_runner, "_audit_labeled_data", lambda **_: audit)
    monkeypatch.setattr(
        sft_runner,
        "_resolve_chain_inputs",
        lambda **_: (init_checkpoint.resolve(), promotion_receipt.resolve()),
    )

    def unexpected_command(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("training command must not run")

    monkeypatch.setattr(sft_runner, "_run_command", unexpected_command)
    args = SimpleNamespace(
        output_dir=output_dir,
        config_dir=tmp_path / "configs",
        labeled_webdataset_root=root,
        labeled_length_index=length_index,
        bucket_manifest=manifest,
        nano_checkpoint=nano_checkpoint,
        init_checkpoint=init_checkpoint,
        logits_promotion_receipt=promotion_receipt,
        dry_run=False,
        smoke_only=False,
        master_port=29634,
        max_peak_reserved_gib=22.0,
        final_checkpoint_path_output=None,
    )

    with pytest.raises(ValueError, match="formal training has progress"):
        sft_runner.run_sft(args)


def test_validate_stage211_sft_completion_binds_artifacts(tmp_path: Path) -> None:
    root, length_index, manifest = _labeled_paths(tmp_path)
    artifacts: dict[str, Path] = {
        "bucket_manifest": manifest,
        "length_index": length_index,
        "provenance": tmp_path / "provenance.json",
        "train_config": tmp_path / "train_config.yaml",
        "nano_teacher_checkpoint": tmp_path / "nano" / "model.pt",
        "init_checkpoint": tmp_path / "init.pt",
        "logits_promotion_receipt": tmp_path / "receipt.json",
        "completion_checkpoint": (
            tmp_path / f"step-{LABELED_EXPECTED['estimated_train_steps']}.pt"
        ),
        "training_log": tmp_path / "train.log",
        "smoke_marker": tmp_path / "sft_smoke_passed.json",
    }
    artifacts["provenance"].write_text("{}\n", encoding="utf-8")
    artifacts["nano_teacher_checkpoint"].parent.mkdir()
    artifacts["nano_teacher_checkpoint"].write_bytes(b"nano-teacher")
    train_config = stage211_phase_train_config_contract("sft")
    train_config["ctc_teacher_online_model_path"] = str(artifacts["nano_teacher_checkpoint"].parent)
    save_yaml(artifacts["train_config"], train_config)
    artifacts["logits_promotion_receipt"].write_text("{}\n", encoding="utf-8")
    artifacts["training_log"].write_text("complete\n", encoding="utf-8")
    torch.save({"step": 0}, artifacts["init_checkpoint"])
    torch.save(
        {"step": LABELED_EXPECTED["estimated_train_steps"]},
        artifacts["completion_checkpoint"],
    )
    smoke_checkpoint = tmp_path / "smoke-step-2.pt"
    smoke_log = tmp_path / "smoke.log"
    torch.save({"step": 2}, smoke_checkpoint)
    smoke_log.write_text(
        "[deepspeed-train] step=1 loss=0.8 peak_reserved=5.50GiB\n"
        "[deepspeed-train] step=2 loss=0.7 peak_reserved=6.00GiB\n",
        encoding="utf-8",
    )
    artifacts["smoke_marker"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "labeled_sft_smoke",
                "phase": "sft",
                "complete": True,
                "init_checkpoint_path": str(artifacts["init_checkpoint"]),
                "init_checkpoint_sha256": sha256_file(artifacts["init_checkpoint"]),
                "logits_promotion_receipt_path": str(
                    artifacts["logits_promotion_receipt"]
                ),
                "logits_promotion_receipt_sha256": sha256_file(
                    artifacts["logits_promotion_receipt"]
                ),
                "bucket_manifest_path": str(manifest),
                "bucket_manifest_sha256": sha256_file(manifest),
                "length_index_path": str(length_index),
                "length_index_sha256": sha256_file(length_index),
                "smoke_checkpoint_path": str(smoke_checkpoint),
                "smoke_checkpoint_sha256": sha256_file(smoke_checkpoint),
                "smoke_log_path": str(smoke_log),
                "smoke_log_sha256": sha256_file(smoke_log),
                "peak_reserved_gib": 6.0,
                "max_peak_reserved_gib": 22.0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    epoch_checkpoint = tmp_path / "epoch-1.pt"
    torch.save(
        {
            "model": {},
            "step": LABELED_EXPECTED["estimated_train_steps"],
            "extra": {
                "epoch": 1,
                "epoch_batch_offset": 0,
                "completed_epoch_batch_count": LABELED_EXPECTED["estimated_train_steps"],
            },
        },
        epoch_checkpoint,
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
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        **LABELED_EXPECTED,
        "labeled_webdataset_root": str(root),
        "runtime_epoch_coverage": {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "runtime_epoch_coverage",
            "complete": True,
            "epochs": 1,
            "steps_per_epoch": LABELED_EXPECTED["estimated_train_steps"],
            "total_steps": LABELED_EXPECTED["estimated_train_steps"],
            "records": [
                {
                    "epoch": 1,
                    "step": LABELED_EXPECTED["estimated_train_steps"],
                    "epoch_batch_offset": 0,
                    "completed_epoch_batch_count": LABELED_EXPECTED["estimated_train_steps"],
                    "checkpoint_path": str(epoch_checkpoint),
                    "checkpoint_sha256": sha256_file(epoch_checkpoint),
                }
            ],
        },
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

    completion["runtime_epoch_coverage"]["records"][0]["completed_epoch_batch_count"] -= 1
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="runtime epoch 1 completion mismatch"):
        sft_runner._validate_completion(completion_path)
    completion["runtime_epoch_coverage"]["records"][0]["completed_epoch_batch_count"] += 1
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifacts["nano_teacher_checkpoint"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact is unavailable or changed"):
        sft_runner._validate_completion(completion_path)


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


def _bound_public_benchmark(
    tmp_path: Path,
    *,
    stage: str,
    error_rate: float,
) -> dict[str, object]:
    results = []
    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        manifest = tmp_path / f"{dataset}.manifest.jsonl"
        nano = tmp_path / f"{dataset}.nano.jsonl"
        student = tmp_path / f"{dataset}.{stage}.student.jsonl"
        for path in (manifest, nano, student):
            if not path.exists():
                path.write_text("{}\n", encoding="utf-8")
        results.append(
            {
                "dataset": dataset,
                "language": expected["language"],
                "metric": expected["metric"],
                "sample_count": expected["samples"],
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
                "metric_source_recomputed": True,
                "nano_error_rate": 0.1,
                "student_error_rate": error_rate,
                "absolute_gap_points": (error_rate - 0.1) * 100.0,
                "relative_ratio": error_rate / 0.1,
                "nano_prediction_reference_unit_ratio": 1.0,
                "student_prediction_reference_unit_ratio": 0.95,
                "nano_deletion_rate": 0.01,
                "student_deletion_rate": 0.02,
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "nano_prediction_path": str(nano.resolve()),
                "nano_prediction_sha256": sha256_file(nano),
                "student_prediction_path": str(student.resolve()),
                "student_prediction_sha256": sha256_file(student),
            }
        )
    return {
        "decode": "greedy_ctc",
        "normalization": "ctc",
        "all_datasets_complete": True,
        "all_datasets_pass": True,
        "results": results,
    }


def _write_nano_baseline_receipt(
    tmp_path: Path,
    *,
    nano_checkpoint: Path,
    benchmark: dict[str, object],
) -> Path:
    results = []
    for raw_result in benchmark["results"]:
        result = dict(raw_result)
        dataset = str(result["dataset"])
        report = tmp_path / f"{dataset}.nano-report.json"
        report.write_text(
            json.dumps(
                {
                    "version": 1,
                    "system": "FunASR-Nano-2512 direct CTC",
                    "model_path": str(nano_checkpoint.parent.resolve()),
                    "manifest_path": result["manifest_path"],
                    "predictions_path": result["nano_prediction_path"],
                    "language": result["language"],
                    "normalization": "ctc",
                    "decode": "greedy_ctc",
                    "requested_limit": None,
                    "sample_count": result["sample_count"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        results.append(
            {
                "dataset": dataset,
                "language": result["language"],
                "metric": result["metric"],
                "sample_count": result["sample_count"],
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
                "report_embeds_checkpoint_sha256": False,
                "report_path": str(report.resolve()),
                "report_sha256": sha256_file(report),
                "manifest_path": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
                "nano_prediction_path": result["nano_prediction_path"],
                "nano_prediction_sha256": result["nano_prediction_sha256"],
            }
        )
    receipt = tmp_path / "nano-baseline-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "nano_public_baseline_provenance",
                "complete": True,
                "provenance_mode": "legacy_report_attestation",
                "nano_checkpoint_path": str(nano_checkpoint.resolve()),
                "nano_checkpoint_sha256": sha256_file(nano_checkpoint),
                "total_samples": sum(
                    int(expected["samples"]) for expected in STAGE211_PUBLIC_BENCHMARKS.values()
                ),
                "results": results,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def _write_stepwise_inputs(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    nano_teacher_dir = tmp_path / "nano-teacher"
    nano_teacher_dir.mkdir()
    nano_teacher_checkpoint = nano_teacher_dir / "model.pt"
    nano_teacher_checkpoint.write_bytes(b"nano-teacher")
    nano_teacher_sha256 = sha256_file(nano_teacher_checkpoint)
    checkpoints = {
        stage: tmp_path / f"{stage}.pt"
        for stage in ("calibration", "mixer", "block", "logits", "sft")
    }
    for stage, checkpoint in checkpoints.items():
        checkpoint.write_bytes(stage.encode())

    support = {}
    for name in (
        "selection",
        "calibration-comparison",
        "calibration-metrics",
        "sft-completion",
        "sft-baseline",
        "sft-comparison",
        "logits-promotion",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        support[name] = path

    calibration_benchmark = _bound_public_benchmark(
        tmp_path,
        stage="calibration",
        error_rate=0.5,
    )
    nano_baseline_receipt = _write_nano_baseline_receipt(
        tmp_path,
        nano_checkpoint=nano_teacher_checkpoint,
        benchmark=calibration_benchmark,
    )
    nano_baseline_binding = {
        "nano_public_baseline_receipt_path": str(nano_baseline_receipt.resolve()),
        "nano_public_baseline_receipt_sha256": sha256_file(nano_baseline_receipt),
        "nano_public_baseline_checkpoint_sha256": nano_teacher_sha256,
    }

    calibration_receipt = tmp_path / "calibration-reuse.json"
    calibration_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "calibration_public_eval_reuse",
                "complete": True,
                "checkpoint_path": str(checkpoints["calibration"].resolve()),
                "checkpoint_sha256": sha256_file(checkpoints["calibration"]),
                "selection_report_path": str(support["selection"].resolve()),
                "selection_report_sha256": sha256_file(support["selection"]),
                "comparison_report_path": str(support["calibration-comparison"].resolve()),
                "comparison_report_sha256": sha256_file(support["calibration-comparison"]),
                "metrics_path": str(support["calibration-metrics"].resolve()),
                "metrics_sha256": sha256_file(support["calibration-metrics"]),
                "public_benchmark": calibration_benchmark,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reports = {"calibration": calibration_receipt}
    previous = "calibration"
    for index, stage in enumerate(("mixer", "block", "logits"), start=1):
        preflight_marker = tmp_path / f"{stage}-preflight-smoke.json"
        preflight_marker.write_text("{}\n", encoding="utf-8")
        gate = tmp_path / f"{stage}-gate.json"
        gate.write_text(
            json.dumps(
                {
                    "phase": stage,
                    "gate_passed": True,
                    "checkpoint_path": str(checkpoints[stage].resolve()),
                    "checkpoint_sha256": sha256_file(checkpoints[stage]),
                    "preflight_smoke": {
                        "marker_path": str(preflight_marker.resolve()),
                        "marker_sha256": sha256_file(preflight_marker),
                    },
                    **nano_baseline_binding,
                    "full_data_coverage": {
                        "total_unique_rows": 100,
                        "total_hours": 10.0,
                        "total_row_exposures": 300,
                        "total_hour_exposures": 30.0,
                        "total_executed_sample_exposures": 304,
                        "segments": [
                            {
                                "init_checkpoint_path": str(checkpoints[previous].resolve()),
                                "init_checkpoint_sha256": sha256_file(checkpoints[previous]),
                                "nano_teacher_checkpoint_sha256": nano_teacher_sha256,
                            }
                        ],
                    },
                    "public_benchmark": _bound_public_benchmark(
                        tmp_path,
                        stage=stage,
                        error_rate=0.5 - index * 0.1,
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        reports[stage] = gate
        previous = stage

    support["sft-completion"].write_text(
        json.dumps(
            {
                "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
                "nano_teacher_checkpoint_sha256": nano_teacher_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    support["logits-promotion"].write_text(
        json.dumps({"nano_teacher_checkpoint_sha256": nano_teacher_sha256}) + "\n",
        encoding="utf-8",
    )

    sft_report = tmp_path / "sft-final.json"
    sft_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "final_completion",
                "phase": "sft",
                "complete": True,
                "gate_passed": True,
                "checkpoint_path": str(checkpoints["sft"].resolve()),
                "checkpoint_sha256": sha256_file(checkpoints["sft"]),
                "sft_completion_path": str(support["sft-completion"].resolve()),
                "sft_completion_sha256": sha256_file(support["sft-completion"]),
                "baseline_public_comparison_report_path": str(support["sft-baseline"].resolve()),
                "baseline_public_comparison_report_sha256": sha256_file(support["sft-baseline"]),
                "public_comparison_report_path": str(support["sft-comparison"].resolve()),
                "public_comparison_report_sha256": sha256_file(support["sft-comparison"]),
                "logits_promotion_receipt_path": str(support["logits-promotion"].resolve()),
                "logits_promotion_receipt_sha256": sha256_file(support["logits-promotion"]),
                "logits_phase_gate_path": str(reports["logits"].resolve()),
                "logits_phase_gate_sha256": sha256_file(reports["logits"]),
                "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
                "nano_teacher_checkpoint_sha256": nano_teacher_sha256,
                **nano_baseline_binding,
                "mixer_phase_gate_path": str(reports["mixer"].resolve()),
                "mixer_phase_gate_sha256": sha256_file(reports["mixer"]),
                "mixer_gate_selection_path": None,
                "mixer_gate_selection_sha256": None,
                "labeled_data_coverage": {
                    "phase": "sft",
                    "complete": True,
                    "epochs": 1,
                    "train_samples": 80,
                    "eval_samples": 10,
                    "total_hours": 8.0,
                    "executed_sample_exposures": 81,
                    "init_checkpoint_path": str(checkpoints["logits"].resolve()),
                    "init_checkpoint_sha256": sha256_file(checkpoints["logits"]),
                    "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
                    "nano_teacher_checkpoint_sha256": nano_teacher_sha256,
                },
                "public_progress": {
                    "gate_passed": True,
                    "no_dataset_regression": True,
                    "macro_improved": True,
                    "improved_datasets": 5,
                },
                "public_benchmark": _bound_public_benchmark(
                    tmp_path,
                    stage="sft",
                    error_rate=0.1,
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reports["sft"] = sft_report
    return checkpoints, reports


def test_resolve_mixer_gate_uses_bound_retention_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_gate_root = tmp_path / "gates"
    gate_dir = phase_gate_root / "mixer_correction_1"
    gate_dir.mkdir(parents=True)
    gate_path = gate_dir / "phase_gate.json"
    gate_path.write_text('{"gate_passed": true}\n', encoding="utf-8")
    checkpoint = tmp_path / "mixer-corrected.pt"
    checkpoint.write_bytes(b"mixer-corrected")
    nano_checkpoint = tmp_path / "nano.pt"
    nano_checkpoint.write_bytes(b"nano")
    promotion = tmp_path / "promotion.json"
    promotion.write_text("{}\n", encoding="utf-8")
    selection_path = phase_gate_root / "mixer_selected.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "mixer_gate_selection",
                "phase": "mixer",
                "gate_dir": str(gate_dir.resolve()),
                "gate_path": str(gate_path.resolve()),
                "gate_sha256": sha256_file(gate_path),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "promotion_receipt_path": str(promotion.resolve()),
                "promotion_receipt_sha256": sha256_file(promotion),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sft_finalizer,
        "validate_stage211_phase_gate_report",
        lambda *args, **kwargs: {"gate_passed": True},
    )
    monkeypatch.setattr(
        sft_finalizer,
        "_validate_promotion_receipt",
        lambda **kwargs: {},
    )

    selected_gate, selected_receipt = sft_finalizer._resolve_mixer_gate(
        phase_gate_root=phase_gate_root,
        selection_path=selection_path,
        nano_teacher_checkpoint=nano_checkpoint,
    )

    assert selected_gate == gate_path.resolve()
    assert selected_receipt == selection_path.resolve()

    gate_path.write_text('{"gate_passed": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        sft_finalizer._resolve_mixer_gate(
            phase_gate_root=phase_gate_root,
            selection_path=selection_path,
            nano_teacher_checkpoint=nano_checkpoint,
        )


def test_stepwise_cli_resolves_mixer_gate_from_final_report(
    tmp_path: Path,
) -> None:
    selected_gate = tmp_path / "mixer-retention" / "phase_gate.json"
    selected_gate.parent.mkdir()
    selected_gate.write_text("{}\n", encoding="utf-8")
    final_report = tmp_path / "stage211_complete.json"
    final_report.write_text(
        json.dumps({"mixer_phase_gate_path": str(selected_gate.resolve())}) + "\n",
        encoding="utf-8",
    )

    assert (
        stepwise_report._resolve_cli_mixer_gate(
            requested_gate=None,
            sft_final_report_path=final_report,
        )
        == selected_gate.resolve()
    )

    explicit_gate = tmp_path / "explicit-gate.json"
    assert (
        stepwise_report._resolve_cli_mixer_gate(
            requested_gate=explicit_gate,
            sft_final_report_path=tmp_path / "missing-final-report.json",
        )
        == explicit_gate.resolve()
    )


def test_stage211_stepwise_report_binds_ordered_metrics_and_checkpoint_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints, reports = _write_stepwise_inputs(tmp_path)

    def validate_phase(path: Path, *, expected_phase: str, checkpoint_path: Path):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        assert report["phase"] == expected_phase
        assert Path(report["checkpoint_path"]) == checkpoint_path
        return report

    monkeypatch.setattr(
        stepwise_report,
        "validate_stage211_phase_gate_report",
        validate_phase,
    )
    sft_payload = json.loads(reports["sft"].read_text(encoding="utf-8"))
    monkeypatch.setattr(
        stepwise_report,
        "_validate_sft_completion",
        lambda path, checkpoint_path=None: (
            sft_payload["labeled_data_coverage"],
            checkpoint_path,
        ),
    )
    monkeypatch.setattr(
        stepwise_report,
        "_validate_sft_promotion_receipt",
        lambda **kwargs: {
            "nano_teacher_checkpoint_sha256": sft_payload["nano_teacher_checkpoint_sha256"],
            "gate_report_path": str(reports["logits"].resolve()),
            "gate_report_sha256": sha256_file(reports["logits"]),
        },
    )
    output_json = tmp_path / "stepwise.json"
    output_markdown = tmp_path / "stepwise.md"

    report = stepwise_report.create_stepwise_report(
        calibration_receipt_path=reports["calibration"],
        mixer_gate_path=reports["mixer"],
        block_gate_path=reports["block"],
        logits_gate_path=reports["logits"],
        sft_final_report_path=reports["sft"],
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert report["strict_stage_order"] == [
        "calibration",
        "mixer",
        "block",
        "logits",
        "sft",
    ]
    assert report["requested_alignment_stage_order"] == [
        "rwkv_layer",
        "block",
        "logits",
        "sft",
    ]
    assert report["requested_to_internal_stage"] == {
        "rwkv_layer": "mixer",
        "block": "block",
        "logits": "logits",
        "sft": "sft",
    }
    assert report["checkpoint_chain_passed"] is True
    assert report["nano_teacher_chain_passed"] is True
    assert report["nano_teacher_checkpoint_sha256"] == sha256_file(
        tmp_path / "nano-teacher" / "model.pt"
    )
    assert len(report["checkpoint_chain"]) == 4
    assert [row["stage"] for row in report["coverage_results"]] == [
        "mixer",
        "block",
        "logits",
        "sft",
    ]
    assert report["coverage_results"][0]["epochs"] == 3
    assert report["coverage_results"][-1]["unique_or_train_rows"] == 80
    assert len(report["dataset_results"]) == len(STAGE211_PUBLIC_BENCHMARKS)
    assert report["stages"][-1]["checkpoint_sha256"] == sha256_file(checkpoints["sft"])
    assert report["stages"][0]["gate_passed"] is None
    assert report["stages"][0]["gate_status"] == "baseline"
    assert report["stages"][1]["preflight_smoke"]["marker_sha256"] == sha256_file(
        tmp_path / "mixer-preflight-smoke.json"
    )
    assert "Layer A" in output_markdown.read_text(encoding="utf-8")
    assert "SFT D" in output_markdown.read_text(encoding="utf-8")
    assert "Training Coverage" in output_markdown.read_text(encoding="utf-8")
    sft_finalizer._validate_final_report(
        reports["sft"],
        checkpoint=checkpoints["sft"],
    )
    assert report["nano_public_baseline_provenance_passed"] is True
    assert (
        report["nano_public_baseline_checkpoint_sha256"] == report["nano_teacher_checkpoint_sha256"]
    )

    sft = json.loads(reports["sft"].read_text(encoding="utf-8"))
    original_mixer_gate_sha256 = sft["mixer_phase_gate_sha256"]
    sft["mixer_phase_gate_sha256"] = "a" * 64
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selected Mixer gate"):
        stepwise_report.build_stepwise_report(
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
        )
    sft["mixer_phase_gate_sha256"] = original_mixer_gate_sha256
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")

    block = json.loads(reports["block"].read_text(encoding="utf-8"))
    original_baseline_sha256 = block["nano_public_baseline_checkpoint_sha256"]
    block["nano_public_baseline_checkpoint_sha256"] = "e" * 64
    reports["block"].write_text(json.dumps(block) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="public-baseline provenance chain mismatch"):
        stepwise_report.build_stepwise_report(
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
        )
    block["nano_public_baseline_checkpoint_sha256"] = original_baseline_sha256
    reports["block"].write_text(json.dumps(block) + "\n", encoding="utf-8")

    sft = json.loads(reports["sft"].read_text(encoding="utf-8"))
    sft["public_benchmark"]["all_datasets_pass"] = False
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="every-dataset Nano"):
        stepwise_report.build_stepwise_report(
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
        )
    with pytest.raises(ValueError, match="every-dataset Nano"):
        sft_finalizer._validate_final_report(
            reports["sft"],
            checkpoint=checkpoints["sft"],
        )
    sft["public_benchmark"]["all_datasets_pass"] = True
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")

    block = json.loads(reports["block"].read_text(encoding="utf-8"))
    original_teacher_sha256 = block["full_data_coverage"]["segments"][0][
        "nano_teacher_checkpoint_sha256"
    ]
    block["full_data_coverage"]["segments"][0]["nano_teacher_checkpoint_sha256"] = "f" * 64
    reports["block"].write_text(json.dumps(block) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Nano teacher checkpoint SHA-256 chain mismatch"):
        stepwise_report.build_stepwise_report(
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
        )
    block["full_data_coverage"]["segments"][0]["nano_teacher_checkpoint_sha256"] = (
        original_teacher_sha256
    )
    block["full_data_coverage"]["segments"][0]["init_checkpoint_path"] = str(
        checkpoints["calibration"]
    )
    block["full_data_coverage"]["segments"][0]["init_checkpoint_sha256"] = sha256_file(
        checkpoints["calibration"]
    )
    reports["block"].write_text(json.dumps(block) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint chain mismatch"):
        stepwise_report.build_stepwise_report(
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
        )
