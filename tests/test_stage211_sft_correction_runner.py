from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from rwkvasr.eval.stage211_gate import (
    STAGE211_PUBLIC_BENCHMARKS,
    stage211_post_coverage_correction_lr,
    validate_stage211_phase_train_config,
)
from rwkvasr.eval.stage211_public_metrics import (
    build_stage211_sft_correction_public_progress,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
runner = importlib.import_module("scripts.run_stage211_sft_correction")


def _benchmark(errors: dict[str, float]) -> dict[str, object]:
    return {
        "results": [
            {
                "dataset": dataset,
                "metric": values["metric"],
                "student_error_rate": errors[dataset],
                "manifest_sha256": f"manifest-{dataset}",
                "nano_prediction_sha256": f"nano-{dataset}",
            }
            for dataset, values in STAGE211_PUBLIC_BENCHMARKS.items()
        ]
    }


def test_stage211_sft_correction_gate_requires_both_language_macros() -> None:
    baseline_errors = {
        "librispeech_test_clean": 0.20,
        "librispeech_test_other": 0.30,
        "commonvoice_en_test": 0.40,
        "aishell1_test": 0.20,
        "wenetspeech_test_net": 0.30,
    }
    bilingual_errors = {
        "librispeech_test_clean": 0.19,
        "librispeech_test_other": 0.30,
        "commonvoice_en_test": 0.40,
        "aishell1_test": 0.19,
        "wenetspeech_test_net": 0.30,
    }
    passed = build_stage211_sft_correction_public_progress(
        baseline=_benchmark(baseline_errors),
        candidate=_benchmark(bilingual_errors),
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    assert passed["gate_passed"] is True
    assert passed["no_dataset_regression"] is True
    assert passed["language_summaries"]["en"]["gate_passed"] is True
    assert passed["language_summaries"]["zh"]["gate_passed"] is True

    chinese_only_errors = {
        **baseline_errors,
        "aishell1_test": 0.10,
        "wenetspeech_test_net": 0.20,
    }
    chinese_only = build_stage211_sft_correction_public_progress(
        baseline=_benchmark(baseline_errors),
        candidate=_benchmark(chinese_only_errors),
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    assert chinese_only["gate_passed"] is False
    assert chinese_only["language_summaries"]["en"]["macro_improved"] is False
    assert chinese_only["language_summaries"]["zh"]["macro_improved"] is True

    regressed = dict(bilingual_errors)
    regressed["commonvoice_en_test"] = 0.401
    regression = build_stage211_sft_correction_public_progress(
        baseline=_benchmark(baseline_errors),
        candidate=_benchmark(regressed),
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    assert regression["gate_passed"] is False
    assert regression["no_dataset_regression"] is False


def test_stage211_sft_correction_config_uses_explicit_lower_lr(tmp_path: Path) -> None:
    labeled_root = tmp_path / "labeled"
    labeled_root.mkdir()
    length_index = tmp_path / "correction_lengths.jsonl"
    manifest = tmp_path / "correction_manifest.json"
    profile_path = tmp_path / "profile.json"
    full_completion = tmp_path / "full_completion.json"
    admission = tmp_path / "failed_report.json"
    init_checkpoint = tmp_path / "init.pt"
    nano_checkpoint = tmp_path / "nano" / "model.pt"
    nano_checkpoint.parent.mkdir()
    for path, content in (
        (length_index, "{}\n"),
        (manifest, "{}\n"),
        (profile_path, "{}\n"),
        (full_completion, "{}\n"),
        (admission, "{}\n"),
        (init_checkpoint, "checkpoint"),
        (nano_checkpoint, "nano"),
    ):
        path.write_text(content, encoding="utf-8")
    profile = {
        "full_labeled_profile": {"labeled_root": str(labeled_root.resolve())},
        "length_index_path": str(length_index.resolve()),
        "bucket_manifest_path": str(manifest.resolve()),
    }

    config, segment = runner.build_correction_train_config(
        round_index=1,
        steps_per_epoch=17,
        output_dir=tmp_path / "run",
        init_checkpoint=init_checkpoint.resolve(),
        correction_profile_path=profile_path.resolve(),
        correction_profile=profile,
        full_completion_path=full_completion.resolve(),
        admission_report_path=admission.resolve(),
        nano_checkpoint=nano_checkpoint.resolve(),
        resume=False,
        smoke=False,
    )

    assert stage211_post_coverage_correction_lr("sft") == 1.0e-7
    assert config["lr"] == 1.0e-7
    assert config["ctc_loss_weight"] == 1.0
    assert config["freeze_encoder_except_time_mixer"] is True
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True
    assert config["webdataset_length_index_path"] == str(length_index.resolve())
    assert config["webdataset_bucket_manifest_path"] == str(manifest.resolve())
    assert config["stage211_post_coverage_correction_round"] == 1
    assert segment["target_step"] == 17
    with pytest.raises(ValueError, match="train config lr mismatch"):
        validate_stage211_phase_train_config(config, phase="sft")
