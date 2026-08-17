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
    STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT,
    STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256,
    STAGE211_SFT_STEP_EVAL_INTERVAL,
    build_stage211_correction_round_promotion_gate,
    build_stage211_sft_step_eval_cadence,
    sha256_file,
    stage211_phase_train_config_contract,
    validate_stage211_public_benchmark,
)
from rwkvasr.eval.stage211_public_metrics import (
    STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
    build_stage211_student_public_prediction_receipt,
    build_stage211_sft_public_progress,
    replay_stage211_public_comparison,
    replay_stage211_sft_public_evidence,
)
from stage211_public_helpers import (
    stage211_test_student_ctc_context,
    stage211_test_student_ctc_row,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_DEDUP_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "stage211_global_dedup_manifest.json"
sys.path.insert(0, str(REPO_ROOT))
sft_runner = importlib.import_module("scripts.run_stage211_labeled_sft")
sft_finalizer = importlib.import_module("scripts.finalize_stage211_labeled_sft")
sft_correction_loop = importlib.import_module("scripts.run_stage211_sft_correction_loop")
sft_correction_evaluator = importlib.import_module("scripts.evaluate_stage211_sft_correction")
stepwise_report = importlib.import_module("scripts.create_stage211_stepwise_report")
public_compare = importlib.import_module("scripts.compare_public_ctc_with_nano")
LABELED_EXPECTED = sft_runner.LABELED_EXPECTED
SFT_SUPPORT_EXPECTED = {
    "ctc_suppressed_token_ids_count": STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT,
    "ctc_suppressed_token_ids_sha256": STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256,
    "teacher_projection_support_matches_student": True,
}


def test_stage211_sft_controller_preserves_virtualenv_python() -> None:
    assert sft_runner.PYTHON == Path(sys.executable)


def _labeled_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "labeled"
    root.mkdir()
    length_index = root / "lengths.jsonl"
    length_index.write_text("{}\n", encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (root / "webdataset_lengths.summary.json").write_text(
        json.dumps(
            {
                "version": 1,
                "output_dir": str(root.resolve()),
                "length_index_path": str(length_index.resolve()),
                "tokenizer_type": "sensevoice_tiktoken",
                "tokenizer_model_path": str(
                    (REPO_ROOT / "assets/fun-asr-nano-2512/multilingual.tiktoken").resolve()
                ),
                "text_normalization": "ctc",
                "frontend_downsample": "sensevoice_lfr6",
                "drop_unk_token": True,
                "unk_token_id": None,
                "num_input_samples": 285_302,
                "num_kept_samples": 285_302,
                "num_dropped_samples": 0,
                "counts": {
                    "input_by_split": {"eval": 1_434, "train": 283_868},
                    "kept_by_split": {"eval": 1_434, "train": 283_868},
                    "kept_by_source": {"aishell3": 63_262, "librispeech": 222_040},
                    "kept_by_language": {"en": 222_040, "zh": 63_262},
                    "kept_by_split_source": {
                        "eval/aishell3": 310,
                        "eval/librispeech": 1_124,
                        "train/aishell3": 62_952,
                        "train/librispeech": 220_916,
                    },
                    "kept_by_split_language": {
                        "eval/en": 1_124,
                        "eval/zh": 310,
                        "train/en": 220_916,
                        "train/zh": 62_952,
                    },
                    "dropped_by_reason": {},
                    "dropped_by_source": {},
                    "dropped_by_language": {},
                    "dropped_unk_tokens_by_source": {},
                    "dropped_unk_tokens_by_language": {},
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "prepare_ctc_aligned.log").write_text(
        "tokenizer_type=sensevoice_tiktoken\n"
        "text_normalization=ctc\n"
        "frontend_downsample=sensevoice_lfr6\n"
        "drop_unk_token=1\n"
        "CTC-aligned clean preprocessing complete\n",
        encoding="utf-8",
    )
    return root, length_index, manifest


def _labeled_audit(root: Path, length_index: Path, manifest: Path) -> dict[str, object]:
    return {
        "webdataset_root": str(root.resolve()),
        "length_index_path": str(length_index.resolve()),
        "bucket_manifest_path": str(manifest.resolve()),
        **LABELED_EXPECTED,
        "label_preparation": sft_runner._label_preparation_proof(
            webdataset_root=root,
            length_index_path=length_index,
        ),
    }


def _write_sft_step_eval_artifacts(
    *,
    run_dir: Path,
    manifest: Path,
    terminal_step: int,
) -> tuple[Path, list[Path]]:
    eval_part = run_dir / "sft-fixed-eval.jsonl"
    eval_part.write_text("fixed-eval\n", encoding="utf-8")
    expected_steps = list(
        range(
            STAGE211_SFT_STEP_EVAL_INTERVAL,
            terminal_step + 1,
            STAGE211_SFT_STEP_EVAL_INTERVAL,
        )
    )
    if not expected_steps or expected_steps[-1] != terminal_step:
        expected_steps.append(terminal_step)
    records = []
    reports = []
    for index, step in enumerate(expected_steps, start=1):
        report_path = run_dir / f"step_eval_layers_step-{step}.yaml"
        eval_loss = 1.0 / float(index)
        save_yaml(
            report_path,
            {
                "step": step,
                "eval_loss": eval_loss,
                "eval_samples": 256,
                "eval_provenance": {
                    "schema_version": 1,
                    "split": "eval",
                    "requested_samples": 256,
                    "split_samples": 256,
                    "feature_seed": 0,
                    "bucket_manifest_path": str(manifest.resolve()),
                    "bucket_manifest_sha256": sha256_file(manifest),
                    "parts": [
                        {
                            "path": str(eval_part.resolve()),
                            "sha256": sha256_file(eval_part),
                            "num_samples": 256,
                        }
                    ],
                },
            },
        )
        reports.append(report_path)
        records.append(
            {
                "step": step,
                "epoch": 1,
                "eval_loss": eval_loss,
                "eval_samples": 256,
                "layer_metrics_path": str(report_path.resolve()),
            }
        )
    metrics_path = run_dir / "step_checkpoint_metrics.yaml"
    save_yaml(metrics_path, {"step_checkpoints": records})
    return metrics_path, reports


def test_validate_stage211_labeled_audit_exact_contract(tmp_path: Path) -> None:
    root, length_index, manifest = _labeled_paths(tmp_path)
    audit = _labeled_audit(root, length_index, manifest)

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

    audit["ctc_tokens"] += 1
    audit["label_preparation"]["text_normalization"] = "none"
    with pytest.raises(ValueError, match="label-preparation summary/log proof"):
        sft_runner._validate_labeled_audit(
            audit,
            labeled_root=root,
            length_index=length_index,
            bucket_manifest=manifest,
        )


def test_stage211_stepwise_ctc_label_proof_rejects_changed_normalization(
    tmp_path: Path,
) -> None:
    root, length_index, manifest = _labeled_paths(tmp_path)
    audit = _labeled_audit(root, length_index, manifest)
    coverage = {
        **LABELED_EXPECTED,
        "ctc_suppress_non_pronunciation_tokens": True,
        **SFT_SUPPORT_EXPECTED,
        "labeled_data_audit": audit,
    }

    proof = stepwise_report._sft_ctc_label_proof(coverage)
    assert proof["full_length_index_audit_passed"] is True
    assert proof["pronunciation_target_samples"] == 285_302

    audit["label_preparation"]["text_normalization"] = "none"
    with pytest.raises(ValueError, match="text_normalization mismatch"):
        stepwise_report._sft_ctc_label_proof(coverage)


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


def test_stage211_sft_formal_runner_requires_full_labeled_profile(tmp_path: Path) -> None:
    root, length_index, manifest = _labeled_paths(tmp_path)
    nano_checkpoint = tmp_path / "nano.pt"
    nano_checkpoint.write_bytes(b"nano")
    args = SimpleNamespace(
        output_dir=tmp_path / "run",
        config_dir=tmp_path / "configs",
        labeled_webdataset_root=root,
        labeled_length_index=length_index,
        bucket_manifest=manifest,
        labeled_profile_receipt=None,
        nano_checkpoint=nano_checkpoint,
        init_checkpoint=None,
        logits_promotion_receipt=None,
        dry_run=False,
        smoke_only=False,
        master_port=29634,
        max_peak_reserved_gib=22.0,
        final_checkpoint_path_output=None,
    )

    with pytest.raises(ValueError, match="schema-v2 full-labeled profile"):
        sft_runner.run_sft(args)


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
    profile_path = root / "stage211_labeled_profile_receipt.json"
    profile_path.write_text("{}\n", encoding="utf-8")
    dynamic_expected = {
        **LABELED_EXPECTED,
        "train_samples": LABELED_EXPECTED["train_samples"] + 100,
        "total_samples": LABELED_EXPECTED["total_samples"] + 100,
        "estimated_train_steps": LABELED_EXPECTED["estimated_train_steps"] + 4,
    }
    audit = {
        **_labeled_audit(root, length_index, manifest),
        **dynamic_expected,
    }
    profile = {
        "schema_version": 2,
        "all_accepted_unique_rows_required": True,
        "source_language_interleave_required": True,
        "expected": dynamic_expected,
        "labeled_data_audit": audit,
    }
    monkeypatch.setattr(
        sft_runner,
        "validate_labeled_profile_receipt",
        lambda path, **_kwargs: profile if path == profile_path.resolve() else None,
    )

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
        labeled_profile_receipt=profile_path,
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
    labeled_audit = _labeled_audit(root, length_index, manifest)
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
    artifacts["provenance"].write_text(
        json.dumps({"labeled_data_audit": labeled_audit}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["nano_teacher_checkpoint"].parent.mkdir()
    artifacts["nano_teacher_checkpoint"].write_bytes(b"nano-teacher")
    train_config = stage211_phase_train_config_contract("sft")
    train_config.update(
        {
            "ctc_teacher_online_model_path": str(artifacts["nano_teacher_checkpoint"].parent),
            "max_steps": LABELED_EXPECTED["estimated_train_steps"],
            "step_eval_every": STAGE211_SFT_STEP_EVAL_INTERVAL,
            "step_eval_samples": 256,
            "step_eval_split": "eval",
            "step_eval_shuffle": False,
            "step_eval_feature_seed": 0,
        }
    )
    save_yaml(artifacts["train_config"], train_config)
    metrics_path, step_eval_reports = _write_sft_step_eval_artifacts(
        run_dir=tmp_path,
        manifest=manifest,
        terminal_step=LABELED_EXPECTED["estimated_train_steps"],
    )
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
                "logits_promotion_receipt_path": str(artifacts["logits_promotion_receipt"]),
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
        "schema_version": sft_runner.SFT_COMPLETION_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "labeled_sft_completion",
        "phase": "sft",
        "complete": True,
        "run_dir": str(tmp_path.resolve()),
        "epochs": 1,
        "batch_size": 12,
        "world_size": 4,
        "frame_budget": 8_000,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "ctc_suppress_non_pronunciation_tokens": True,
        **SFT_SUPPORT_EXPECTED,
        **LABELED_EXPECTED,
        "labeled_data_audit": labeled_audit,
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
    completion["step_eval_cadence"] = build_stage211_sft_step_eval_cadence(completion)
    completion_path = tmp_path / "sft_complete.json"
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded, checkpoint = sft_runner._validate_completion(completion_path)

    assert loaded == completion
    assert checkpoint == artifacts["completion_checkpoint"].resolve()
    with pytest.raises(ValueError, match="schema-v2 full-labeled profile"):
        sft_runner._validate_completion(
            completion_path,
            require_full_profile=True,
        )

    completion["labeled_data_audit"]["ctc_unk_tokens"] = 1
    completion_path.write_text(
        json.dumps(completion, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ctc_unk_tokens mismatch"):
        sft_runner._validate_completion(completion_path)
    completion["labeled_data_audit"]["ctc_unk_tokens"] = 0

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

    metrics_payload = metrics_path.read_text(encoding="utf-8")
    metrics = sft_runner.load_yaml(metrics_path)
    metrics["step_checkpoints"].pop()
    save_yaml(metrics_path, metrics)
    with pytest.raises(ValueError, match="step-eval cadence mismatch"):
        sft_runner._validate_completion(completion_path)
    metrics_path.write_text(metrics_payload, encoding="utf-8")

    terminal_report = step_eval_reports[-1]
    terminal_report_payload = terminal_report.read_text(encoding="utf-8")
    report = sft_runner.load_yaml(terminal_report)
    report["eval_loss"] += 0.1
    save_yaml(terminal_report, report)
    with pytest.raises(ValueError, match="report metadata mismatch"):
        sft_runner._validate_completion(completion_path)
    terminal_report.write_text(terminal_report_payload, encoding="utf-8")

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


def test_stage211_sft_finalizer_requires_full_profile_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def reject_completion(
        path: Path,
        *,
        checkpoint_path: Path | None = None,
        require_full_profile: bool = False,
    ) -> tuple[dict[str, object], Path]:
        observed.update(
            path=path,
            checkpoint_path=checkpoint_path,
            require_full_profile=require_full_profile,
        )
        raise RuntimeError("stop after completion admission")

    monkeypatch.setattr(sft_finalizer, "_validate_completion", reject_completion)
    args = SimpleNamespace(
        run_dir=tmp_path / "run",
        completion_report=None,
    )

    with pytest.raises(RuntimeError, match="completion admission"):
        sft_finalizer.finalize_sft(args)

    assert observed["path"] == (tmp_path / "run" / "sft_complete.json").resolve()
    assert observed["checkpoint_path"] is None
    assert observed["require_full_profile"] is True


def test_stage211_sft_correction_requires_full_profile_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def reject_completion(
        path: Path,
        *,
        checkpoint_path: Path | None = None,
        require_full_profile: bool = False,
    ) -> tuple[dict[str, object], Path]:
        observed.update(
            path=path,
            checkpoint_path=checkpoint_path,
            require_full_profile=require_full_profile,
        )
        raise RuntimeError("stop after full-SFT admission")

    monkeypatch.setattr(
        sft_correction_loop,
        "validate_full_sft_completion",
        reject_completion,
    )
    args = SimpleNamespace(
        full_sft_completion=tmp_path / "sft_complete.json",
        full_sft_failed_report=tmp_path / "failed.json",
    )

    with pytest.raises(RuntimeError, match="full-SFT admission"):
        sft_correction_loop.run_loop(args)

    assert observed["path"] == (tmp_path / "sft_complete.json").resolve()
    assert observed["checkpoint_path"] is None
    assert observed["require_full_profile"] is True


def test_stage211_sft_correction_coverage_sums_contiguous_rounds(
    tmp_path: Path,
) -> None:
    full_completion = tmp_path / "sft_complete.json"
    correction_profile = tmp_path / "correction_profile.json"
    full_checkpoint = tmp_path / "full.pt"
    nano_checkpoint = tmp_path / "nano.pt"
    round_one_checkpoint = tmp_path / "round-one.pt"
    round_two_checkpoint = tmp_path / "round-two.pt"
    for path, payload in (
        (full_completion, b"{}\n"),
        (correction_profile, b"{}\n"),
        (full_checkpoint, b"full"),
        (nano_checkpoint, b"nano"),
        (round_one_checkpoint, b"round-one"),
        (round_two_checkpoint, b"round-two"),
    ):
        path.write_bytes(payload)
    receipt_paths = [tmp_path / "round-one.json", tmp_path / "round-two.json"]
    for index, receipt_path in enumerate(receipt_paths, start=1):
        receipt_path.write_text(json.dumps({"round": index}) + "\n", encoding="utf-8")
    profile = {
        "train_samples": 10,
        "total_train_hours": 1.5,
        "estimated_train_steps": 2,
        "tail_padding_samples": 6,
        "executed_sample_exposures": 16,
        "language_counts": {"en": 5, "zh": 5},
        "source_counts": {"english": 5, "chinese": 5},
    }
    checkpoints = [full_checkpoint, round_one_checkpoint, round_two_checkpoint]
    completions = []
    bindings = []
    for index, receipt_path in enumerate(receipt_paths, start=1):
        completions.append(
            {
                "round": index,
                "rows": 10,
                "row_exposures": 10,
                "hours": 1.5,
                "steps": 2,
                "tail_padding_samples": 6,
                "executed_sample_exposures": 16,
                "language_counts": {"en": 5, "zh": 5},
                "source_counts": {"english": 5, "chinese": 5},
                "nano_teacher_checkpoint_path": str(nano_checkpoint),
                "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
                "init_checkpoint_path": str(checkpoints[index - 1]),
                "init_checkpoint_sha256": sha256_file(checkpoints[index - 1]),
                "completion_checkpoint_path": str(checkpoints[index]),
                "completion_checkpoint_sha256": sha256_file(checkpoints[index]),
            }
        )
        bindings.append(
            {
                "round": index,
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            }
        )

    coverage = sft_correction_evaluator._correction_coverage(
        bindings=bindings,
        completions=completions,
        profile=profile,
        correction_profile_path=correction_profile,
        full_completion_path=full_completion,
        full_completion={
            "nano_teacher_checkpoint_path": str(nano_checkpoint),
            "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        },
        full_checkpoint=full_checkpoint,
        candidate_checkpoint=round_two_checkpoint,
    )

    assert coverage["applied"] is True
    assert coverage["rounds"] == 2
    assert coverage["row_exposures"] == 20
    assert coverage["hour_exposures"] == pytest.approx(3.0)
    assert coverage["steps"] == 4
    assert coverage["tail_padding_sample_exposures"] == 12
    assert coverage["executed_sample_exposures"] == 32
    assert coverage["language_row_exposures"] == {"en": 10, "zh": 10}
    assert [receipt["round"] for receipt in coverage["round_receipts"]] == [1, 2]

    broken = [dict(completion) for completion in completions]
    broken[1]["init_checkpoint_path"] = str(full_checkpoint)
    with pytest.raises(ValueError, match="checkpoint chain is not contiguous"):
        sft_correction_evaluator._correction_coverage(
            bindings=bindings,
            completions=broken,
            profile=profile,
            correction_profile_path=correction_profile,
            full_completion_path=full_completion,
            full_completion={
                "nano_teacher_checkpoint_path": str(nano_checkpoint),
                "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
            },
            full_checkpoint=full_checkpoint,
            candidate_checkpoint=round_two_checkpoint,
        )


def test_stage211_stepwise_sft_correction_evidence_is_not_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_completion = tmp_path / "sft_complete.json"
    evaluation_path = tmp_path / "correction_evaluation.json"
    profile_path = tmp_path / "correction_profile.json"
    full_checkpoint = tmp_path / "full.pt"
    corrected_checkpoint = tmp_path / "corrected.pt"
    for path, payload in (
        (full_completion, b"{}\n"),
        (evaluation_path, b"{}\n"),
        (profile_path, b"{}\n"),
        (full_checkpoint, b"full"),
        (corrected_checkpoint, b"corrected"),
    ):
        path.write_bytes(payload)
    correction_coverage = {
        "schema_version": 1,
        "applied": True,
        "rounds": 1,
        "unique_rows_per_round": 10,
        "row_exposures": 10,
        "hour_exposures": 1.5,
        "steps": 2,
        "tail_padding_sample_exposures": 6,
        "executed_sample_exposures": 16,
        "language_row_exposures": {"en": 5, "zh": 5},
        "source_row_exposures": {"english": 5, "chinese": 5},
        "round_receipts": [{"round": 1}],
    }
    completion_receipts = [{"round": 1, "path": "/proof/round.json", "sha256": "a" * 64}]
    public_progress = {"gate_passed": True}
    evaluation = {
        "checkpoint_path": str(corrected_checkpoint),
        "correction_completion_receipts": completion_receipts,
        "correction_coverage": correction_coverage,
        "public_progress": public_progress,
    }
    report = {
        "sft_correction_evaluation_path": str(evaluation_path),
        "sft_correction_evaluation_sha256": sha256_file(evaluation_path),
        "sft_correction_profile_path": str(profile_path),
        "sft_correction_profile_sha256": sha256_file(profile_path),
        "sft_correction_completion_receipts": completion_receipts,
        "sft_correction_coverage": correction_coverage,
        "sft_correction_public_progress": public_progress,
    }
    monkeypatch.setattr(
        sft_correction_evaluator,
        "validate_correction_evaluation_report",
        lambda *args, **kwargs: evaluation,
    )

    assert (
        stepwise_report._validate_sft_correction_coverage(
            report,
            full_completion_path=full_completion,
            full_checkpoint=full_checkpoint,
            final_checkpoint=corrected_checkpoint,
        )
        == correction_coverage
    )

    forged = dict(report)
    forged["sft_correction_coverage"] = {
        **correction_coverage,
        "row_exposures": 11,
    }
    with pytest.raises(ValueError, match="correction evidence chain mismatch"):
        stepwise_report._validate_sft_correction_coverage(
            forged,
            full_completion_path=full_completion,
            full_checkpoint=full_checkpoint,
            final_checkpoint=corrected_checkpoint,
        )

    with pytest.raises(ValueError, match="Uncorrected"):
        stepwise_report._validate_sft_correction_coverage(
            report,
            full_completion_path=full_completion,
            full_checkpoint=full_checkpoint,
            final_checkpoint=full_checkpoint,
        )


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


def test_stage211_sft_public_evidence_replays_baseline_candidate_and_progress(
    tmp_path: Path,
) -> None:
    dataset = "librispeech_test_clean"
    benchmarks = {dataset: {"language": "en", "metric": "wer", "samples": 2}}
    baseline_checkpoint = tmp_path / "logits.pt"
    candidate_checkpoint = tmp_path / "sft.pt"
    baseline_checkpoint.write_bytes(b"logits")
    candidate_checkpoint.write_bytes(b"sft")
    manifest = tmp_path / f"{dataset}.jsonl"
    nano = tmp_path / f"{dataset}.nano.jsonl"
    baseline = tmp_path / f"{dataset}.baseline.jsonl"
    candidate = tmp_path / f"{dataset}.candidate.jsonl"
    rows = [
        {"utt_id": "utt-1", "ref_text": "hello world", "pred_text": "hello world"},
        {"utt_id": "utt-2", "ref_text": "speech test", "pred_text": "speech test"},
    ]
    manifest.write_text(
        "".join(
            json.dumps({"utt_id": row["utt_id"], "text": row["ref_text"]}) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    nano.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    baseline_rows = [dict(row) for row in rows]
    baseline_rows[1]["pred_text"] = "speech"
    baseline.write_text(
        "".join(json.dumps(row) + "\n" for row in baseline_rows),
        encoding="utf-8",
    )
    provenance, tokenizer = stage211_test_student_ctc_context(candidate_checkpoint)
    candidate.write_text(
        "".join(
            json.dumps(
                stage211_test_student_ctc_row(
                    utt_id=str(row["utt_id"]),
                    ref_text=str(row["ref_text"]),
                    pred_text=str(row["pred_text"]),
                    provenance=provenance,
                    tokenizer=tokenizer,
                )
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    def source_report(*, checkpoint: Path, student: Path) -> dict[str, object]:
        result = public_compare.compare_dataset(
            dataset=dataset,
            nano_path=nano,
            student_path=student,
            normalization="ctc",
            max_relative_ratio=1.20,
            max_absolute_gap_points=3.0,
        )
        return {
            "version": 1,
            "decode": "greedy_ctc",
            "normalization": "ctc",
            "strip_language_confirmation": STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
            "gate": {
                "max_relative_ratio": 1.20,
                "max_absolute_gap_points": 3.0,
                "requires_every_dataset": True,
            },
            "all_datasets_pass": bool(result["gate_pass"]),
            "student_checkpoint_path": str(checkpoint.resolve()),
            "student_checkpoint_sha256": sha256_file(checkpoint),
            "results": [result],
        }

    baseline_source = source_report(checkpoint=baseline_checkpoint, student=baseline)
    candidate_source = source_report(checkpoint=candidate_checkpoint, student=candidate)
    candidate_receipt_path = tmp_path / "candidate-prediction-receipt.json"
    candidate_receipt = build_stage211_student_public_prediction_receipt(
        checkpoint_path=candidate_checkpoint,
        manifest_paths={dataset: manifest},
        prediction_paths={dataset: candidate},
        benchmarks=benchmarks,
    )
    candidate_receipt_path.write_text(
        json.dumps(candidate_receipt) + "\n",
        encoding="utf-8",
    )
    candidate_source["student_prediction_receipt_path"] = str(candidate_receipt_path.resolve())
    candidate_source["student_prediction_receipt_sha256"] = sha256_file(candidate_receipt_path)
    baseline_source_path = tmp_path / "baseline-comparison.json"
    candidate_source_path = tmp_path / "candidate-comparison.json"
    baseline_source_path.write_text(json.dumps(baseline_source) + "\n", encoding="utf-8")
    candidate_source_path.write_text(json.dumps(candidate_source) + "\n", encoding="utf-8")
    manifests = {dataset: manifest}
    baseline_benchmark = replay_stage211_public_comparison(
        baseline_source,
        manifest_paths=manifests,
        benchmarks=benchmarks,
        expected_checkpoint=baseline_checkpoint,
    )
    candidate_benchmark = replay_stage211_public_comparison(
        candidate_source,
        manifest_paths=manifests,
        benchmarks=benchmarks,
        expected_checkpoint=candidate_checkpoint,
    )
    progress = build_stage211_sft_public_progress(
        baseline=baseline_benchmark,
        candidate=candidate_benchmark,
        benchmarks=benchmarks,
    )
    report = {
        "baseline_public_comparison_report_path": str(baseline_source_path.resolve()),
        "baseline_public_comparison_report_sha256": sha256_file(baseline_source_path),
        "public_comparison_report_path": str(candidate_source_path.resolve()),
        "public_comparison_report_sha256": sha256_file(candidate_source_path),
        "baseline_public_benchmark": baseline_benchmark,
        "public_benchmark": candidate_benchmark,
        "public_progress": progress,
    }

    replayed = replay_stage211_sft_public_evidence(
        report,
        benchmarks=benchmarks,
        expected_baseline_checkpoint=baseline_checkpoint,
        expected_candidate_checkpoint=candidate_checkpoint,
    )

    assert replayed["public_progress"]["gate_passed"] is True
    assert replayed["public_progress"]["improved_datasets"] == 1
    candidate_source["results"][0]["student_error_rate"] = 0.01
    candidate_source_path.write_text(json.dumps(candidate_source) + "\n", encoding="utf-8")
    report["public_comparison_report_sha256"] = sha256_file(candidate_source_path)
    report["public_benchmark"]["results"][0]["student_error_rate"] = 0.01
    with pytest.raises(ValueError, match="replayed student_error_rate mismatch"):
        replay_stage211_sft_public_evidence(
            report,
            benchmarks=benchmarks,
            expected_baseline_checkpoint=baseline_checkpoint,
            expected_candidate_checkpoint=candidate_checkpoint,
        )


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
        nano_error_rate = 0.1
        absolute_gap_points = (error_rate - nano_error_rate) * 100.0
        relative_ratio = error_rate / nano_error_rate
        absolute_gate_pass = absolute_gap_points <= 3.0
        relative_gate_pass = relative_ratio <= 1.20
        results.append(
            {
                "dataset": dataset,
                "language": expected["language"],
                "metric": expected["metric"],
                "sample_count": expected["samples"],
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
                "metric_source_recomputed": True,
                "nano_error_rate": nano_error_rate,
                "student_error_rate": error_rate,
                "absolute_gap_points": absolute_gap_points,
                "relative_ratio": relative_ratio,
                "absolute_gate_pass": absolute_gate_pass,
                "relative_gate_pass": relative_gate_pass,
                "gate_pass": absolute_gate_pass and relative_gate_pass,
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
        "strip_language_confirmation": STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
        "all_datasets_complete": True,
        "all_datasets_pass": all(bool(result["gate_pass"]) for result in results),
        "results": results,
    }


def test_stage211_public_benchmark_rejects_duplicate_dataset_records(tmp_path: Path) -> None:
    benchmark = _bound_public_benchmark(tmp_path, stage="duplicate", error_rate=0.1)
    results = benchmark["results"]
    assert isinstance(results, list)
    results[-1] = dict(results[0])

    with pytest.raises(ValueError, match="unique, complete, and expected"):
        validate_stage211_public_benchmark(benchmark, require_metric_source_recomputed=True)


def test_stage211_public_benchmark_rejects_extra_result_record(tmp_path: Path) -> None:
    benchmark = _bound_public_benchmark(tmp_path, stage="extra", error_rate=0.1)
    results = benchmark["results"]
    assert isinstance(results, list)
    results.append(dict(results[0]))

    with pytest.raises(ValueError, match="exactly five results"):
        validate_stage211_public_benchmark(benchmark, require_metric_source_recomputed=True)


def test_stage211_public_benchmark_rejects_non_object_result(tmp_path: Path) -> None:
    benchmark = _bound_public_benchmark(tmp_path, stage="non-object", error_rate=0.1)
    results = benchmark["results"]
    assert isinstance(results, list)
    results[-1] = "not-a-result"

    with pytest.raises(ValueError, match="results must all be objects"):
        validate_stage211_public_benchmark(benchmark, require_metric_source_recomputed=True)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("absolute_gate_pass", "absolute threshold decision mismatch"),
        ("relative_gate_pass", "relative threshold decision mismatch"),
        ("gate_pass", "combined threshold decision mismatch"),
    ),
)
def test_stage211_public_benchmark_rejects_inconsistent_dataset_decision(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    benchmark = _bound_public_benchmark(tmp_path, stage=field, error_rate=0.1)
    results = benchmark["results"]
    assert isinstance(results, list)
    result = results[0]
    assert isinstance(result, dict)
    result[field] = False

    with pytest.raises(ValueError, match=message):
        validate_stage211_public_benchmark(benchmark, require_metric_source_recomputed=True)


def test_stage211_public_benchmark_rejects_inconsistent_every_dataset_decision(
    tmp_path: Path,
) -> None:
    benchmark = _bound_public_benchmark(tmp_path, stage="all", error_rate=0.1)
    benchmark["all_datasets_pass"] = False

    with pytest.raises(ValueError, match="every-dataset decision mismatch"):
        validate_stage211_public_benchmark(benchmark, require_metric_source_recomputed=True)


def _write_nano_baseline_receipt(
    tmp_path: Path,
    *,
    nano_checkpoint: Path,
    benchmark: dict[str, object],
    public_overlap: dict[str, str] | None = None,
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
                    "strip_language_confirmation": (STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION),
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
                **({"public_overlap": public_overlap} if public_overlap is not None else {}),
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


def _stepwise_component_summary(*, include_cells: bool) -> dict[str, object]:
    weak_band = {
        "baseline_loss": 1.0,
        "candidate_loss": 0.5,
        "baseline_cosine": 0.5,
        "candidate_cosine": 0.8,
    }
    result: dict[str, object] = {
        "baseline_mean_loss": 1.0,
        "candidate_mean_loss": 0.5,
        "baseline_mean_cosine": 0.5,
        "candidate_mean_cosine": 0.8,
        "loss_improved_layers": 70,
        "cosine_improved_layers": 70,
        "weak_bands": {
            "10-19": dict(weak_band),
            "20-29": dict(weak_band),
        },
    }
    if include_cells:
        result["cells"] = {
            cell_name: {
                "baseline_loss": 1.0,
                "candidate_loss": 0.5,
                "baseline_cosine": 0.5,
                "candidate_cosine": 0.8,
                "layers_loss_improved": 70,
                "layers_cosine_improved": 70,
            }
            for cell_name in stepwise_report.ALIGNMENT_CELLS
        }
    return result


def _stepwise_logits_metrics(*, candidate: bool) -> dict[str, float]:
    return {
        "full_kl": 0.8 if candidate else 1.0,
        "conditional_nonblank_kl": 0.8 if candidate else 1.0,
        "ctc_token_error_rate": 0.2 if candidate else 0.4,
        "selected_top1_agreement": 0.8 if candidate else 0.5,
        "all_top1_agreement": 0.8 if candidate else 0.5,
        "active_top1_agreement": 0.8 if candidate else 0.5,
        "nonblank_rate_ratio": 1.0 if candidate else 0.8,
        "collapsed_length_ratio": 1.0 if candidate else 0.8,
    }


def _write_stepwise_alignment_fixture(
    tmp_path: Path,
    *,
    phase: str,
    baseline_checkpoint: Path,
    checkpoint: Path,
) -> Path:
    components = ("mixer",) if phase == "mixer" else ("mixer", "ffn", "block")
    stratified_components = {
        component: _stepwise_component_summary(include_cells=True) for component in components
    }
    if phase == "logits":
        stratified = {
            "cells": {
                cell_name: {
                    "samples": 256,
                    "baseline_metrics": _stepwise_logits_metrics(candidate=False),
                    "candidate_metrics": _stepwise_logits_metrics(candidate=True),
                }
                for cell_name in stepwise_report.ALIGNMENT_CELLS
            },
            "macro": {
                "baseline_metrics": _stepwise_logits_metrics(candidate=False),
                "candidate_metrics": _stepwise_logits_metrics(candidate=True),
            },
            "hidden_component_summaries": stratified_components,
            "decoder_hidden": {
                "baseline_loss": 1.0,
                "candidate_loss": 0.9,
            },
        }
    else:
        stratified = {
            "cells": {
                cell_name: {
                    "samples": 256,
                    "baseline_loss": 1.0,
                    "candidate_loss": 0.5,
                    "layers_loss_improved": 70,
                    "layers_cosine_improved": 70,
                }
                for cell_name in stepwise_report.ALIGNMENT_CELLS
            },
            "macro": {
                "baseline_loss": 1.0,
                "candidate_loss": 0.5,
                "relative_change_pct": -50.0,
            },
            "component_summaries": stratified_components,
            "decoder_hidden": (
                {"baseline_loss": 1.0, "candidate_loss": 0.5} if phase == "block" else None
            ),
        }
    stratified_path = tmp_path / f"{phase}-stratified-alignment.json"
    stratified_path.write_text(json.dumps(stratified) + "\n", encoding="utf-8")
    source: dict[str, object] = {
        "pipeline": "stage211",
        "artifact": "logits_alignment_gate" if phase == "logits" else "hidden_alignment_gate",
        "phase": phase,
        "gate_passed": True,
        "stratified_gate_passed": True,
        "stratified_summary_path": str(stratified_path.resolve()),
        "stratified_summary_sha256": sha256_file(stratified_path),
        "stratified_summary": stratified,
        "baseline_checkpoint_path": str(baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "baseline_eval_provenance": {"split_samples": 256},
    }
    fixed_components = {
        component: _stepwise_component_summary(include_cells=False) for component in components
    }
    if phase == "logits":
        source.update(
            {
                "baseline_metrics": _stepwise_logits_metrics(candidate=False),
                "candidate_metrics": _stepwise_logits_metrics(candidate=True),
                "full_kl_relative_reduction": 0.2,
                "conditional_nonblank_kl_relative_reduction": 0.2,
                "hidden_component_summaries": fixed_components,
                "decoder_hidden_retention": {
                    "baseline_loss": 1.0,
                    "candidate_loss": 0.9,
                    "retained": True,
                },
            }
        )
    else:
        source.update(
            {
                "baseline_eval_loss": 1.0,
                "candidate_eval_loss": 0.5,
                "component_summaries": fixed_components,
                "decoder_hidden": (
                    {"baseline_loss": 1.0, "candidate_loss": 0.5} if phase == "block" else None
                ),
            }
        )
    path = tmp_path / f"{phase}-alignment-gate.json"
    path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    return path


def _write_stepwise_inputs(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    labeled_root, labeled_length_index, labeled_manifest = _labeled_paths(tmp_path)
    labeled_audit = _labeled_audit(
        labeled_root,
        labeled_length_index,
        labeled_manifest,
    )
    nano_teacher_dir = tmp_path / "nano-teacher"
    nano_teacher_dir.mkdir()
    nano_teacher_checkpoint = nano_teacher_dir / "model.pt"
    nano_teacher_checkpoint.write_bytes(b"nano-teacher")
    nano_teacher_sha256 = sha256_file(nano_teacher_checkpoint)
    public_overlap_receipt = tmp_path / "public-overlap-receipt.json"
    public_overlap_receipt.write_text("{}\n", encoding="utf-8")
    public_overlap = {
        "receipt_path": str(public_overlap_receipt.resolve()),
        "receipt_sha256": sha256_file(public_overlap_receipt),
    }
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
        public_overlap=public_overlap,
    )
    nano_baseline_binding = {
        "nano_public_baseline_receipt_path": str(nano_baseline_receipt.resolve()),
        "nano_public_baseline_receipt_sha256": sha256_file(nano_baseline_receipt),
        "nano_public_baseline_checkpoint_sha256": nano_teacher_sha256,
    }
    usb_coverage_receipt = tmp_path / "usb-coverage.json"
    usb_coverage_receipt.write_text(
        json.dumps(
            {
                "artifact": "usb_top_level_coverage",
                "classification_complete": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    archived_social_overlap = tmp_path / "archived-social-overlap.json"
    archived_social_overlap.write_text(
        json.dumps(
            {
                "artifact": "archived_social_overlap",
                "complete": True,
                "archive_sha256": "a" * 64,
                "archive_audio_members": 10,
                "exact_duplicate_members": 10,
                "unique_members": 0,
                "all_members_exact_existing_social_duplicates": True,
                "archive_excluded_from_training_as_duplicate": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base_component_inventory = tmp_path / "base-component-inventory.json"
    base_component_inventory.write_text("{}\n", encoding="utf-8")
    base_public_overlap_audit = tmp_path / "base-public-overlap-audit.json"
    base_public_overlap_audit.write_text(
        json.dumps(
            {
                "artifact": "stage211_base_public_pcm_overlap_audit",
                "complete": True,
                "training_ready": True,
                "admission_state": "normalized_pcm_exact_public_clear",
                "comparison_mode": "normalized_pcm_exact",
                "scan_order": "manifest_location_index_archive_order_v1",
                "near_duplicate_complete": False,
                "decode_failures": 0,
                "public_overlap_rows": 0,
                "base_inventory_path": str(base_component_inventory.resolve()),
                "base_inventory_sha256": sha256_file(base_component_inventory),
                "scanned_rows": 20,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    supplemental_inventory = tmp_path / "supplemental-inventory.json"
    supplemental_inventory.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact": "stage211_supplemental_combined_inventory",
                "component_inventories": {
                    "base_natural": {
                        "inventory_path": str(base_component_inventory.resolve()),
                        "inventory_sha256": sha256_file(base_component_inventory),
                        "rows": 20,
                    },
                    "social_vad": {},
                },
                "base_public_overlap_audit": {
                    "receipt_path": str(base_public_overlap_audit.resolve()),
                    "receipt_sha256": sha256_file(base_public_overlap_audit),
                    "comparison_mode": "normalized_pcm_exact",
                    "scan_order": "manifest_location_index_archive_order_v1",
                    "scanned_rows": 20,
                    "public_overlap_rows": 0,
                    "training_ready": True,
                },
                "cross_pool_dedupe": {
                    "mode": "source_identity_plus_known_corpus_exclusion",
                    "content_fingerprint_complete": False,
                    "source_sets_disjoint": True,
                    "base_public_overlap_normalized_pcm_exact_complete": True,
                    "base_public_overlap_rows": 0,
                    "social_normalized_pcm_exact_complete": True,
                    "social_public_overlap_mode": "normalized_pcm_exact",
                    "archived_social_exact_duplicate_exclusion_complete": True,
                    "usb_top_level_classification_complete": True,
                    "usb_unresolved_natural_entries": [],
                    "near_duplicate_complete": False,
                    "known_overlap_exclusions": [
                        "llaso_gigaspeech",
                        "llaso_librispeech",
                    ],
                    "supplemental_sources": ["mls_english", "peoples_speech_clean"],
                    "stage179": {
                        "manifest_path": str(GLOBAL_DEDUP_FIXTURE.resolve()),
                        "manifest_sha256": sha256_file(GLOBAL_DEDUP_FIXTURE),
                        "total_unique_rows": 62_072_225,
                        "total_unique_hours": 118_465.16068055555,
                    },
                },
                "usb_top_level_coverage": {
                    "receipt_path": str(usb_coverage_receipt),
                    "receipt_sha256": sha256_file(usb_coverage_receipt),
                },
                "archived_social_exclusion": {
                    "receipt_path": str(archived_social_overlap),
                    "receipt_sha256": sha256_file(archived_social_overlap),
                    "archive_sha256": "a" * 64,
                    "audio_members": 10,
                    "unique_members": 0,
                },
                "usb_natural_audio_resolution": {
                    "complete": True,
                    "exact_duplicate_excluded": ["new_video.tar"],
                    "unresolved_entries": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

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
                "public_overlap": public_overlap,
                "public_benchmark": calibration_benchmark,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reports = {
        "calibration": calibration_receipt,
        "supplemental_inventory": supplemental_inventory,
    }
    previous = "calibration"
    for index, stage in enumerate(("mixer", "block", "logits"), start=1):
        preflight_marker = tmp_path / f"{stage}-preflight-smoke.json"
        preflight_marker.write_text("{}\n", encoding="utf-8")
        training_segments = [
            {
                "difficulty": difficulty,
                "rows": 20,
                "hours": 2.0,
                "epochs": 3,
                "steps_per_epoch": 1,
                "steps": 3,
                "row_exposures": 60,
                "hour_exposures": 6.0,
                "tail_padding_sample_exposures": 1,
                "executed_sample_exposures": 61,
                "init_checkpoint_path": str(checkpoints[previous].resolve()),
                "init_checkpoint_sha256": sha256_file(checkpoints[previous]),
                "nano_teacher_checkpoint_sha256": nano_teacher_sha256,
            }
            for difficulty in ("easy", "medium", "hard", "long")
        ]
        supplemental_segment = {
            "difficulty": "supplemental_natural",
            "rows": 20,
            "hours": 2.0,
            "epochs": 3,
            "steps_per_epoch": 1,
            "steps": 3,
            "row_exposures": 60,
            "hour_exposures": 6.0,
            "tail_padding_sample_exposures": 1,
            "executed_sample_exposures": 61,
            "supplemental_inventory_path": str(supplemental_inventory.resolve()),
            "supplemental_inventory_sha256": sha256_file(supplemental_inventory),
            "nano_teacher_checkpoint_sha256": nano_teacher_sha256,
        }
        alignment_path = _write_stepwise_alignment_fixture(
            tmp_path,
            phase=stage,
            baseline_checkpoint=checkpoints[previous],
            checkpoint=checkpoints[stage],
        )
        trajectory_source_order = [
            "easy",
            "medium",
            "hard",
            "long",
            "supplemental_natural",
        ]
        trajectory_entries = [
            {"source_name": source_name, "eval_loss": 0.5}
            for source_name in trajectory_source_order
        ]
        step_eval_sources = [
            {
                "order": order,
                "source_kind": "curriculum",
                "source_name": source_name,
                "terminal_step": 3,
                "interval_steps": 10_000,
                "expected_report_count": 1,
                "actual_report_count": 1,
            }
            for order, source_name in enumerate(trajectory_source_order)
        ]
        gate = tmp_path / f"{stage}-gate.json"
        gate.write_text(
            json.dumps(
                {
                    "phase": stage,
                    "gate_passed": True,
                    "correction_round_promotion": (
                        build_stage211_correction_round_promotion_gate(0)
                    ),
                    "checkpoint_path": str(checkpoints[stage].resolve()),
                    "checkpoint_sha256": sha256_file(checkpoints[stage]),
                    "alignment_gate_passed": True,
                    "trajectory_retention_gate_passed": True,
                    "trajectory_retention": {
                        "gate_passed": True,
                        "fixed_eval_samples": 256,
                        "max_relative_regression_pct": 10.0,
                        "source_order": trajectory_source_order,
                        "entries": trajectory_entries,
                        "best_prior": {
                            "source_name": "easy",
                            "eval_loss": 0.5,
                        },
                        "candidate": {
                            "source_name": "supplemental_natural",
                            "eval_loss": 0.5,
                            "checkpoint_sha256": sha256_file(checkpoints[stage]),
                        },
                        "best_prior_loss": 0.5,
                        "candidate_loss": 0.5,
                        "relative_regression_pct": 0.0,
                    },
                    "step_eval_cadence": {
                        "schema_version": 1,
                        "pipeline": "stage211",
                        "artifact": "step_eval_cadence",
                        "phase": stage,
                        "complete": True,
                        "interval_steps": 10_000,
                        "eval_samples": 256,
                        "source_order": trajectory_source_order,
                        "source_count": len(step_eval_sources),
                        "total_reports": len(step_eval_sources),
                        "sources": step_eval_sources,
                    },
                    "alignment_report": {
                        "path": str(alignment_path.resolve()),
                        "sha256": sha256_file(alignment_path),
                        "artifact": (
                            "logits_alignment_gate"
                            if stage == "logits"
                            else "hidden_alignment_gate"
                        ),
                    },
                    "global_dedup_manifest_path": str(GLOBAL_DEDUP_FIXTURE.resolve()),
                    "global_dedup_manifest_sha256": sha256_file(GLOBAL_DEDUP_FIXTURE),
                    "preflight_smoke": {
                        "marker_path": str(preflight_marker.resolve()),
                        "marker_sha256": sha256_file(preflight_marker),
                    },
                    **nano_baseline_binding,
                    "public_overlap": public_overlap,
                    "full_data_coverage": {
                        "original_total_unique_rows": 80,
                        "total_unique_rows": 100,
                        "total_hours": 10.0,
                        "total_row_exposures": 300,
                        "total_hour_exposures": 30.0,
                        "total_tail_padding_sample_exposures": 5,
                        "total_executed_sample_exposures": 305,
                        "segments": training_segments,
                        "supplemental_natural": supplemental_segment,
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
    sft_terminal_step = int(LABELED_EXPECTED["estimated_train_steps"])
    sft_report_steps = list(
        range(
            STAGE211_SFT_STEP_EVAL_INTERVAL,
            sft_terminal_step + 1,
            STAGE211_SFT_STEP_EVAL_INTERVAL,
        )
    )
    if not sft_report_steps or sft_report_steps[-1] != sft_terminal_step:
        sft_report_steps.append(sft_terminal_step)
    sft_step_eval_cadence = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "step_eval_cadence",
        "phase": "sft",
        "complete": True,
        "interval_steps": STAGE211_SFT_STEP_EVAL_INTERVAL,
        "eval_samples": 256,
        "source_order": ["labeled_sft"],
        "source_count": 1,
        "total_reports": len(sft_report_steps),
        "sources": [
            {
                "order": 0,
                "source_kind": "labeled_sft",
                "source_name": "labeled_sft",
                "terminal_step": sft_terminal_step,
                "interval_steps": STAGE211_SFT_STEP_EVAL_INTERVAL,
                "expected_report_count": len(sft_report_steps),
                "actual_report_count": len(sft_report_steps),
            }
        ],
    }

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
                "public_overlap": public_overlap,
                "mixer_phase_gate_path": str(reports["mixer"].resolve()),
                "mixer_phase_gate_sha256": sha256_file(reports["mixer"]),
                "mixer_gate_selection_path": None,
                "mixer_gate_selection_sha256": None,
                "block_phase_gate_path": str(reports["block"].resolve()),
                "block_phase_gate_sha256": sha256_file(reports["block"]),
                "block_gate_selection_path": None,
                "block_gate_selection_sha256": None,
                "logits_gate_selection_path": None,
                "logits_gate_selection_sha256": None,
                "labeled_data_coverage": {
                    "phase": "sft",
                    "complete": True,
                    "epochs": 1,
                    "ctc_suppress_non_pronunciation_tokens": True,
                    **SFT_SUPPORT_EXPECTED,
                    **LABELED_EXPECTED,
                    "labeled_data_audit": labeled_audit,
                    "init_checkpoint_path": str(checkpoints["logits"].resolve()),
                    "init_checkpoint_sha256": sha256_file(checkpoints["logits"]),
                    "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
                    "nano_teacher_checkpoint_sha256": nano_teacher_sha256,
                    "step_eval_cadence": sft_step_eval_cadence,
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
    initialization_support: dict[str, Path] = {}
    for name, payload in (
        ("calibration-provenance", b"{}\n"),
        ("stage210-checkpoint", b"stage210"),
        ("stage210-train-config", b"{}\n"),
        ("stage210-log", b"init\n"),
        ("loader-rwkv-asr-ctc", b"loader-a\n"),
        ("loader-sensevoice-rwkv", b"loader-b\n"),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        initialization_support[name] = path
    initialization_receipt = tmp_path / "nano-initialization-receipt.json"
    initialization_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "nano_initialization_proof",
                "complete": True,
                "calibration_reuse_receipt_path": str(calibration_receipt.resolve()),
                "calibration_reuse_receipt_sha256": sha256_file(calibration_receipt),
                "calibration_selection_report_path": str(support["selection"].resolve()),
                "calibration_selection_report_sha256": sha256_file(support["selection"]),
                "calibration_checkpoint_path": str(checkpoints["calibration"].resolve()),
                "calibration_checkpoint_sha256": sha256_file(checkpoints["calibration"]),
                "calibration_provenance_path": str(
                    initialization_support["calibration-provenance"].resolve()
                ),
                "calibration_provenance_sha256": sha256_file(
                    initialization_support["calibration-provenance"]
                ),
                "stage210_checkpoint_path": str(
                    initialization_support["stage210-checkpoint"].resolve()
                ),
                "stage210_checkpoint_sha256": sha256_file(
                    initialization_support["stage210-checkpoint"]
                ),
                "stage210_train_config_path": str(
                    initialization_support["stage210-train-config"].resolve()
                ),
                "stage210_train_config_sha256": sha256_file(
                    initialization_support["stage210-train-config"]
                ),
                "stage210_log_path": str(initialization_support["stage210-log"].resolve()),
                "stage210_log_sha256": sha256_file(initialization_support["stage210-log"]),
                "nano_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
                "nano_checkpoint_sha256": nano_teacher_sha256,
                "loader_source_bindings": [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for path in (
                        initialization_support["loader-rwkv-asr-ctc"],
                        initialization_support["loader-sensevoice-rwkv"],
                    )
                ],
                "runtime_load_report": {
                    "qkv_mapped_to_both_directions": True,
                    "qkv_projection_scale_mode": "rwkv_norm",
                    "encoder_layers": 70,
                    "rwkv_encoder_loaded_tensors": 1125,
                    "expected_non_attention_mlp_norm_tensors": 564,
                    "expected_bidirectional_qkvo_and_input_projection_tensors": 561,
                    "first_layer_reconstruction_errors": {"q": 0.05, "k": 0.04, "v": 0.03},
                    "ctc_decoder_layers": 5,
                    "ctc_decoder_loaded_tensors": 84,
                    "ctc_head_loaded_rows": 60515,
                    "ctc_head_ignored_rows": [60514],
                },
                "freeze_report": {
                    "freeze_encoder_except_time_mixer": True,
                    "freeze_ctc_decoder": True,
                    "freeze_ctc_head": True,
                    "frozen_tensors": 650,
                    "trainable_params": 189378560,
                },
                "frozen_tensor_audit": {
                    "complete": True,
                    "non_attention_mlp_norm_tensors": 564,
                    "ctc_decoder_tensors": 84,
                    "ctc_head_teacher_rows": 60515,
                    "ctc_head_project_rows": 60516,
                    "mismatched_tensors": 0,
                    "mismatched_head_rows": 0,
                    "mismatch_examples": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reports["initialization"] = initialization_receipt
    metric_tokenizer_source = tmp_path / "metric-tokenizer.py"
    metric_tokenizer_source.write_text("# unicode metric tokenizer\n", encoding="utf-8")
    metric_correction_receipt = tmp_path / "unicode-metric-correction.json"
    metric_correction_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "unicode_wer_metric_correction",
                "complete": True,
                "tokenizer_contract": "unicode_alnum_words_basic_cjk_chars_v1",
                "strip_language_confirmation": STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
                "tokenizer_source_path": str(metric_tokenizer_source.resolve()),
                "tokenizer_source_sha256": sha256_file(metric_tokenizer_source),
                "calibration_reuse_receipt_path": str(calibration_receipt.resolve()),
                "calibration_reuse_receipt_sha256": sha256_file(calibration_receipt),
                "initialization_receipt_path": str(initialization_receipt.resolve()),
                "initialization_receipt_sha256": sha256_file(initialization_receipt),
                "installed_files": [
                    {
                        "label": "calibration_reuse_receipt",
                        "path": str(calibration_receipt.resolve()),
                        "sha256": sha256_file(calibration_receipt),
                    },
                    {
                        "label": "initialization_receipt",
                        "path": str(initialization_receipt.resolve()),
                        "sha256": sha256_file(initialization_receipt),
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reports["metric_tokenizer_source"] = metric_tokenizer_source
    reports["metric_correction"] = metric_correction_receipt
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


@pytest.mark.parametrize(("phase", "target"), (("block", "logits"), ("logits", "sft")))
def test_resolve_phase_gate_uses_bound_correction_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    target: str,
) -> None:
    gate_dir = tmp_path / f"{phase}-correction"
    gate_dir.mkdir()
    gate_path = gate_dir / "phase_gate.json"
    gate_path.write_text('{"gate_passed": true}\n', encoding="utf-8")
    checkpoint = tmp_path / f"{phase}.pt"
    checkpoint.write_bytes(phase.encode())
    nano_checkpoint = tmp_path / "nano.pt"
    nano_checkpoint.write_bytes(b"nano")
    promotion = tmp_path / f"{phase}-promotion.json"
    promotion.write_text("{}\n", encoding="utf-8")
    selection = tmp_path / f"{phase}_selected.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "phase_gate_selection",
                "phase": phase,
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
    validations: list[dict[str, object]] = []
    monkeypatch.setattr(
        sft_finalizer,
        "_validate_promotion_receipt",
        lambda **kwargs: validations.append(kwargs) or {},
    )

    resolved_gate, resolved_selection = sft_finalizer._resolve_phase_gate(
        phase=phase,
        phase_gate_root=tmp_path,
        selection_path=selection,
        nano_teacher_checkpoint=nano_checkpoint,
    )

    assert resolved_gate == gate_path.resolve()
    assert resolved_selection == selection.resolve()
    assert validations[0]["target_phase"] == target
    assert validations[0]["checkpoint_path"] == checkpoint.resolve()


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
    monkeypatch.setattr(
        stepwise_report,
        "validate_stage211_nano_public_baseline_receipt",
        lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        stepwise_report,
        "validate_stage211_public_overlap_binding",
        lambda binding, *, public_benchmark: {},
    )
    monkeypatch.setattr(
        sft_finalizer,
        "validate_stage211_nano_public_baseline_receipt",
        lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        sft_finalizer,
        "validate_stage211_public_overlap_binding",
        lambda binding, *, public_benchmark: {},
    )
    monkeypatch.setattr(
        sft_finalizer,
        "validate_stage211_phase_gate_report",
        validate_phase,
    )

    def replay_sft_public(report: dict[str, object], **kwargs):
        del kwargs
        return {
            "baseline_public_benchmark": report.get(
                "baseline_public_benchmark",
                report["public_benchmark"],
            ),
            "public_benchmark": report["public_benchmark"],
            "public_progress": report["public_progress"],
        }

    monkeypatch.setattr(
        stepwise_report,
        "replay_stage211_sft_public_evidence",
        replay_sft_public,
    )
    monkeypatch.setattr(
        sft_finalizer,
        "replay_stage211_sft_public_evidence",
        replay_sft_public,
    )
    monkeypatch.setattr(
        stepwise_report,
        "DEFAULT_TOKENIZER_SOURCE",
        reports["metric_tokenizer_source"],
    )
    sft_payload = json.loads(reports["sft"].read_text(encoding="utf-8"))
    completion_validation_modes: list[bool] = []

    def validate_sft_completion(
        path: Path,
        checkpoint_path: Path | None = None,
        require_full_profile: bool = False,
    ) -> tuple[dict[str, object], Path]:
        del path
        assert checkpoint_path is None
        completion_validation_modes.append(require_full_profile)
        return (
            sft_payload["labeled_data_coverage"],
            checkpoints["sft"],
        )

    monkeypatch.setattr(
        stepwise_report,
        "_validate_sft_completion",
        validate_sft_completion,
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
        initialization_receipt_path=reports["initialization"],
        calibration_receipt_path=reports["calibration"],
        mixer_gate_path=reports["mixer"],
        block_gate_path=reports["block"],
        logits_gate_path=reports["logits"],
        sft_final_report_path=reports["sft"],
        output_json=output_json,
        output_markdown=output_markdown,
        public_metric_correction_receipt_path=reports["metric_correction"],
    )
    assert completion_validation_modes == [True]

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
    assert report["all_requested_alignment_metrics_complete"] is True
    assert report["public_metric_normalization"] == "ctc"
    assert report["initial_calibration_result"]["role"] == "initial_baseline"
    assert report["initial_calibration_result"]["normalization"] == "ctc"
    assert report["initial_calibration_result"]["checkpoint_sha256"] == sha256_file(
        checkpoints["calibration"]
    )
    assert [row["stage"] for row in report["requested_alignment_results"]] == [
        "rwkv_layer",
        "block",
        "logits",
        "sft",
    ]
    assert [row["internal_stage"] for row in report["requested_alignment_results"]] == [
        "mixer",
        "block",
        "logits",
        "sft",
    ]
    assert [row["objective"] for row in report["requested_alignment_results"]] == [
        "hidden_states",
        "hidden_states",
        "ctc_logits",
        "labeled_ctc_sft",
    ]
    assert all(
        row["gate_passed"] is True and len(row["checkpoint_sha256"]) == 64
        for row in report["requested_alignment_results"]
    )
    assert report["checkpoint_chain_passed"] is True
    assert report["nano_initialization_chain_passed"] is True
    assert report["nano_initialization_source_chain_passed"] is True
    assert [row["mode"] for row in report["initialization_proof"]["loader_source_validation"]] == [
        "exact_file_sha256",
        "exact_file_sha256",
    ]
    assert report["ctc_label_normalization_chain_passed"] is True
    assert report["ctc_label_proof"]["full_length_index_audit_passed"] is True
    assert report["ctc_label_proof"]["text_normalization"] == "ctc"
    assert report["ctc_label_proof"]["ctc_unk_tokens"] == 0
    assert report["ctc_label_proof"]["ctc_suppress_non_pronunciation_tokens"] is True
    assert report["ctc_label_proof"]["ctc_suppressed_token_ids_count"] == 2_114
    assert (
        report["ctc_label_proof"]["ctc_suppressed_token_ids_sha256"]
        == STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256
    )
    assert report["ctc_label_proof"]["teacher_projection_support_matches_student"] is True
    assert report["public_metric_definition_chain_passed"] is True
    assert report["public_metric_strip_language_confirmation"] is False
    assert report["public_metric_correction_receipt_sha256"] == sha256_file(
        reports["metric_correction"]
    )
    assert report["public_metric_tokenizer_source_sha256"] == sha256_file(
        reports["metric_tokenizer_source"]
    )
    assert report["ctc_label_proof"]["language_counts"] == {
        "en": 222_040,
        "zh": 63_262,
    }
    assert report["initialization_receipt_sha256"] == sha256_file(reports["initialization"])
    assert report["nano_teacher_chain_passed"] is True
    assert report["supplemental_inventory_chain_passed"] is True
    assert report["supplemental_dedupe_proof"]["source_sets_disjoint"] is True
    assert report["supplemental_dedupe_proof"]["content_fingerprint_complete"] is False
    assert (
        report["supplemental_dedupe_proof"]["base_public_overlap_normalized_pcm_exact_complete"]
        is True
    )
    assert report["supplemental_dedupe_proof"]["base_public_overlap_rows"] == 0
    assert report["supplemental_dedupe_proof"]["base_public_overlap_scanned_rows"] == 20
    assert report["supplemental_dedupe_proof"]["known_overlap_exclusions"] == [
        "llaso_gigaspeech",
        "llaso_librispeech",
    ]
    assert (
        report["supplemental_dedupe_proof"]["archived_social_exact_duplicate_exclusion_complete"]
        is True
    )
    assert report["supplemental_dedupe_proof"]["archived_social_unique_members"] == 0
    assert report["supplemental_dedupe_proof"]["usb_natural_audio_resolution_complete"] is True
    assert report["supplemental_dedupe_proof"]["usb_unresolved_natural_entries"] == []
    assert report["all_stage_public_metrics_complete"] is True
    assert report["public_metric_stage_order"] == [
        "calibration",
        "mixer",
        "block",
        "logits",
        "sft",
    ]
    assert len(report["english_wer_datasets"]) == 3
    assert len(report["chinese_cer_datasets"]) == 2
    assert all(
        list(row["stages"]) == report["public_metric_stage_order"]
        for row in report["dataset_results"]
    )
    language_summaries = {
        summary["name"]: summary for summary in report["language_metric_summaries"]
    }
    requested_language_summaries = {
        summary["name"]: summary
        for summary in report["requested_alignment_language_metric_summaries"]
    }
    assert set(language_summaries) == {"english_wer", "chinese_cer"}
    assert set(requested_language_summaries) == {"english_wer", "chinese_cer"}
    for name, language, metric, dataset_count in (
        ("english_wer", "en", "wer", 3),
        ("chinese_cer", "zh", "cer", 2),
    ):
        summary = language_summaries[name]
        rows = [
            row
            for row in report["dataset_results"]
            if row["language"] == language and row["metric"] == metric
        ]
        assert summary["aggregation"] == "unweighted_dataset_macro"
        assert summary["dataset_count"] == dataset_count
        assert summary["datasets"] == [row["dataset"] for row in rows]
        assert summary["sample_count"] == sum(row["sample_count"] for row in rows)
        assert summary["nano_error_rate"] == pytest.approx(
            sum(row["nano_error_rate"] for row in rows) / dataset_count
        )
        for stage in report["public_metric_stage_order"]:
            assert summary["stages"][stage] == pytest.approx(
                sum(row["stages"][stage]["student_error_rate"] for row in rows) / dataset_count
            )
        requested_summary = requested_language_summaries[name]
        assert requested_summary["initial_calibration_error_rate"] == pytest.approx(
            summary["stages"]["calibration"]
        )
        assert list(requested_summary["stages"]) == report["requested_alignment_stage_order"]
        for requested_stage, internal_stage in report["requested_to_internal_stage"].items():
            assert requested_summary["stages"][requested_stage] == pytest.approx(
                summary["stages"][internal_stage]
            )
    requested_results = {row["stage"]: row for row in report["requested_alignment_results"]}
    assert requested_results["rwkv_layer"]["english_wer"] == pytest.approx(
        requested_language_summaries["english_wer"]["stages"]["rwkv_layer"]
    )
    assert requested_results["rwkv_layer"]["chinese_cer"] == pytest.approx(
        requested_language_summaries["chinese_cer"]["stages"]["rwkv_layer"]
    )
    assert report["initial_calibration_result"]["nano_english_wer"] == pytest.approx(0.1)
    assert report["initial_calibration_result"]["english_wer_gap_to_nano"] == pytest.approx(0.4)
    assert report["initial_calibration_result"]["nano_chinese_cer"] == pytest.approx(0.1)
    assert report["initial_calibration_result"]["chinese_cer_gap_to_nano"] == pytest.approx(0.4)
    expected_nano_gaps = {
        "rwkv_layer": 0.3,
        "block": 0.2,
        "logits": 0.1,
        "sft": 0.0,
    }
    expected_previous_stages = {
        "rwkv_layer": "calibration",
        "block": "rwkv_layer",
        "logits": "block",
        "sft": "logits",
    }
    for stage, expected_gap in expected_nano_gaps.items():
        assert requested_results[stage]["normalization"] == "ctc"
        assert requested_results[stage]["previous_stage"] == expected_previous_stages[stage]
        assert requested_results[stage]["nano_english_wer"] == pytest.approx(0.1)
        assert requested_results[stage]["english_wer_gap_to_nano"] == pytest.approx(expected_gap)
        assert requested_results[stage]["english_wer_delta_from_previous"] == pytest.approx(-0.1)
        assert requested_results[stage][
            "english_wer_gap_reduction_from_previous"
        ] == pytest.approx(0.1)
        assert requested_results[stage]["nano_chinese_cer"] == pytest.approx(0.1)
        assert requested_results[stage]["chinese_cer_gap_to_nano"] == pytest.approx(expected_gap)
        assert requested_results[stage]["chinese_cer_delta_from_previous"] == pytest.approx(-0.1)
        assert requested_results[stage][
            "chinese_cer_gap_reduction_from_previous"
        ] == pytest.approx(0.1)
    assert len(report["requested_alignment_dataset_results"]) == len(STAGE211_PUBLIC_BENCHMARKS)
    assert all(
        list(row["stages"]) == report["requested_alignment_stage_order"]
        and "initial_calibration" in row
        for row in report["requested_alignment_dataset_results"]
    )
    assert report["public_overlap_chain_passed"] is True
    assert report["public_overlap_receipt_sha256"] == sha256_file(
        tmp_path / "public-overlap-receipt.json"
    )
    assert report["nano_teacher_checkpoint_sha256"] == sha256_file(
        tmp_path / "nano-teacher" / "model.pt"
    )
    assert report["global_dedup_manifest_sha256"] == sha256_file(GLOBAL_DEDUP_FIXTURE)
    assert len(report["checkpoint_chain"]) == 4
    assert [row["stage"] for row in report["coverage_results"]] == [
        "mixer",
        "block",
        "logits",
        "sft",
    ]
    assert report["coverage_results"][0]["epochs"] == 3
    assert report["coverage_results"][0]["supplemental_unique_rows"] == 20
    assert len(report["coverage_results"][0]["training_segments"]) == 5
    assert (
        report["coverage_results"][-1]["unique_or_train_rows"] == LABELED_EXPECTED["train_samples"]
    )
    assert report["coverage_results"][-1]["steps"] == LABELED_EXPECTED["estimated_train_steps"]
    assert (
        report["coverage_results"][-1]["tail_padding_sample_exposures"]
        == LABELED_EXPECTED["tail_padding_sample_exposures"]
    )
    assert report["sft_correction_evidence"] == {
        "schema_version": 1,
        "applied": False,
        "rounds": 0,
        "unique_rows_per_round": 0,
        "row_exposures": 0,
        "hour_exposures": 0.0,
        "steps": 0,
        "tail_padding_sample_exposures": 0,
        "executed_sample_exposures": 0,
        "language_row_exposures": {},
        "source_row_exposures": {},
        "round_receipts": [],
    }
    assert report["coverage_results"][-1]["correction"] == report["sft_correction_evidence"]
    assert (
        report["coverage_results"][-1]["effective_row_exposures"]
        == (LABELED_EXPECTED["train_samples"])
    )
    assert report["coverage_results"][-1]["step_eval_cadence"] == {
        "complete": True,
        "interval_steps": STAGE211_SFT_STEP_EVAL_INTERVAL,
        "eval_samples_per_report": 256,
        "source_order": ["labeled_sft"],
        "source_count": 1,
        "total_reports": 7,
        "sources": [
            {
                "source": "labeled_sft",
                "source_kind": "labeled_sft",
                "terminal_step": LABELED_EXPECTED["estimated_train_steps"],
                "report_count": 7,
                "eval_samples_per_report": 256,
            }
        ],
    }
    assert len(report["dataset_results"]) == len(STAGE211_PUBLIC_BENCHMARKS)
    assert report["all_stage_alignment_results_complete"] is True
    assert [row["stage"] for row in report["alignment_results"]] == [
        "mixer",
        "block",
        "logits",
    ]
    assert all(row["stratified_samples"] == 2_304 for row in report["alignment_results"])
    assert report["alignment_results"][0]["fixed"]["components"]["mixer"][
        "candidate_mean_loss"
    ] == pytest.approx(0.5)
    assert report["alignment_results"][1]["fixed"]["decoder_hidden"][
        "candidate_loss"
    ] == pytest.approx(0.5)
    assert report["alignment_results"][2]["fixed"]["metrics"]["full_kl"][
        "candidate"
    ] == pytest.approx(0.8)
    assert report["stages"][-1]["checkpoint_sha256"] == sha256_file(checkpoints["sft"])
    assert report["stages"][0]["gate_passed"] is None
    assert report["stages"][0]["gate_status"] == "baseline"
    assert report["stages"][1]["preflight_smoke"]["marker_sha256"] == sha256_file(
        tmp_path / "mixer-preflight-smoke.json"
    )
    assert "Layer A" in output_markdown.read_text(encoding="utf-8")
    assert "SFT D" in output_markdown.read_text(encoding="utf-8")
    assert "Requested alignment order: RWKV Layer -> Block -> Logits" in (
        output_markdown.read_text(encoding="utf-8")
    )
    assert "## Requested Alignment Results" in output_markdown.read_text(encoding="utf-8")
    assert "WER gap" in output_markdown.read_text(encoding="utf-8")
    assert "Public WER/CER normalization: `ctc`" in output_markdown.read_text(encoding="utf-8")
    assert "## Incremental Stage Impact" in output_markdown.read_text(encoding="utf-8")
    assert "English Nano-gap reduction" in output_markdown.read_text(encoding="utf-8")
    assert "+30.000 pt" in output_markdown.read_text(encoding="utf-8")
    assert "Language Macro Metrics" in output_markdown.read_text(encoding="utf-8")
    assert "unweighted_dataset_macro" in output_markdown.read_text(encoding="utf-8")
    assert "Per-Dataset Metrics" in output_markdown.read_text(encoding="utf-8")
    assert "Hidden Alignment Metrics" in output_markdown.read_text(encoding="utf-8")
    assert "Nine-Cell Hidden Alignment" in output_markdown.read_text(encoding="utf-8")
    assert "Logits Alignment Metrics" in output_markdown.read_text(encoding="utf-8")
    assert "Nine-Cell Logits Alignment" in output_markdown.read_text(encoding="utf-8")
    assert "Decoder Hidden Alignment" in output_markdown.read_text(encoding="utf-8")
    assert "Alignment Evidence" in output_markdown.read_text(encoding="utf-8")
    assert "Periodic Fixed Evaluation" in output_markdown.read_text(encoding="utf-8")
    assert "`labeled_sft`" in output_markdown.read_text(encoding="utf-8")
    assert "Training Coverage" in output_markdown.read_text(encoding="utf-8")
    assert "Full Data Segment Proof" in output_markdown.read_text(encoding="utf-8")
    assert "CTC label normalization" in output_markdown.read_text(encoding="utf-8")
    assert "Public metric definition proof" in output_markdown.read_text(encoding="utf-8")
    assert "Supplemental cross-pool dedupe" in output_markdown.read_text(encoding="utf-8")
    assert "base public normalized-PCM exact audit" in output_markdown.read_text(encoding="utf-8")
    assert "USB-wide natural-audio resolution" in output_markdown.read_text(encoding="utf-8")
    sft_finalizer._validate_final_report(
        reports["sft"],
        checkpoint=checkpoints["sft"],
    )
    completion_path = reports["sft"].parent / "sft-completion.json"
    completion_text = completion_path.read_text(encoding="utf-8")
    completion = json.loads(completion_text)
    completion["completion_checkpoint_path"] = str(checkpoints["logits"].resolve())
    completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")
    sft = json.loads(reports["sft"].read_text(encoding="utf-8"))
    sft["sft_completion_sha256"] = sha256_file(completion_path)
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="correction evaluation"):
        sft_finalizer._validate_final_report(
            reports["sft"],
            checkpoint=checkpoints["sft"],
        )
    completion_path.write_text(completion_text, encoding="utf-8")
    sft["sft_completion_sha256"] = sha256_file(completion_path)
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")
    assert report["nano_public_baseline_provenance_passed"] is True
    assert (
        report["nano_public_baseline_checkpoint_sha256"] == report["nano_teacher_checkpoint_sha256"]
    )

    alignment_source = Path(report["alignment_results"][0]["source_report_path"])
    alignment_text = alignment_source.read_text(encoding="utf-8")
    alignment_source.write_text(alignment_text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="alignment report is unavailable or changed"):
        stepwise_report.build_stepwise_report(
            initialization_receipt_path=reports["initialization"],
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
            public_metric_correction_receipt_path=reports["metric_correction"],
        )
    alignment_source.write_text(alignment_text, encoding="utf-8")

    tokenizer_source = reports["metric_tokenizer_source"]
    tokenizer_text = tokenizer_source.read_text(encoding="utf-8")
    tokenizer_source.write_text(tokenizer_text + "# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer source changed"):
        stepwise_report.build_stepwise_report(
            initialization_receipt_path=reports["initialization"],
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
            public_metric_correction_receipt_path=reports["metric_correction"],
        )
    tokenizer_source.write_text(tokenizer_text, encoding="utf-8")

    supplemental_inventory = reports["supplemental_inventory"]
    supplemental_text = supplemental_inventory.read_text(encoding="utf-8")
    supplemental_payload = json.loads(supplemental_text)
    supplemental_payload["cross_pool_dedupe"]["content_fingerprint_complete"] = True
    supplemental_inventory.write_text(
        json.dumps(supplemental_payload) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cross-pool dedupe contract mismatch"):
        stepwise_report._supplemental_dedupe_proof(supplemental_inventory)
    supplemental_inventory.write_text(supplemental_text, encoding="utf-8")

    sft = json.loads(reports["sft"].read_text(encoding="utf-8"))
    original_mixer_gate_sha256 = sft["mixer_phase_gate_sha256"]
    sft["mixer_phase_gate_sha256"] = "a" * 64
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selected Mixer gate"):
        stepwise_report.build_stepwise_report(
            initialization_receipt_path=reports["initialization"],
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
            public_metric_correction_receipt_path=reports["metric_correction"],
        )
    sft["mixer_phase_gate_sha256"] = original_mixer_gate_sha256
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")

    block = json.loads(reports["block"].read_text(encoding="utf-8"))
    original_baseline_sha256 = block["nano_public_baseline_checkpoint_sha256"]
    block["nano_public_baseline_checkpoint_sha256"] = "e" * 64
    reports["block"].write_text(json.dumps(block) + "\n", encoding="utf-8")
    sft = json.loads(reports["sft"].read_text(encoding="utf-8"))
    sft["block_phase_gate_sha256"] = sha256_file(reports["block"])
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="public-baseline provenance chain mismatch"):
        stepwise_report.build_stepwise_report(
            initialization_receipt_path=reports["initialization"],
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
            public_metric_correction_receipt_path=reports["metric_correction"],
        )
    block["nano_public_baseline_checkpoint_sha256"] = original_baseline_sha256
    reports["block"].write_text(json.dumps(block) + "\n", encoding="utf-8")
    sft["block_phase_gate_sha256"] = sha256_file(reports["block"])
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")

    sft = json.loads(reports["sft"].read_text(encoding="utf-8"))
    sft["public_benchmark"]["all_datasets_pass"] = False
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="every-dataset"):
        stepwise_report.build_stepwise_report(
            initialization_receipt_path=reports["initialization"],
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
            public_metric_correction_receipt_path=reports["metric_correction"],
        )
    with pytest.raises(ValueError, match="every-dataset"):
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
    sft["block_phase_gate_sha256"] = sha256_file(reports["block"])
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Nano teacher checkpoint SHA-256 chain mismatch"):
        stepwise_report.build_stepwise_report(
            initialization_receipt_path=reports["initialization"],
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
            public_metric_correction_receipt_path=reports["metric_correction"],
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
    sft["block_phase_gate_sha256"] = sha256_file(reports["block"])
    reports["sft"].write_text(json.dumps(sft) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint chain mismatch"):
        stepwise_report.build_stepwise_report(
            initialization_receipt_path=reports["initialization"],
            calibration_receipt_path=reports["calibration"],
            mixer_gate_path=reports["mixer"],
            block_gate_path=reports["block"],
            logits_gate_path=reports["logits"],
            sft_final_report_path=reports["sft"],
            public_metric_correction_receipt_path=reports["metric_correction"],
        )


def test_stage211_stepwise_report_rejects_unreplayed_calibration_metrics(
    tmp_path: Path,
) -> None:
    _, reports = _write_stepwise_inputs(tmp_path)
    calibration_path = reports["calibration"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["public_benchmark"]["results"][0].pop("metric_source_recomputed")
    calibration_path.write_text(json.dumps(calibration) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not source-recomputed"):
        stepwise_report._validate_calibration_receipt(calibration_path)
