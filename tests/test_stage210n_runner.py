from pathlib import Path

import pytest

from scripts.run_stage210n_full_easy_spike_tolerant_distill import (
    EVAL_INTERVAL,
    FULL_EASY_STEPS,
    _config,
    _segments,
)


def test_stage210n_formal_config_runs_complete_easy_epoch() -> None:
    segment = _segments(smoke=False)[0]
    config = _config(
        segment=segment,
        output_dir=Path("/tmp/stage210n-test"),
        init_checkpoint=Path("/tmp/stage210m-step1000.pt"),
        bucket_manifest=Path("/tmp/source-grouped-easy.json"),
        resume=False,
        smoke=False,
    )

    assert segment["target_step"] == FULL_EASY_STEPS == 30_064
    assert config["max_steps"] == FULL_EASY_STEPS
    assert config["save_every"] == EVAL_INTERVAL == 10_000
    assert config["step_eval_every"] == EVAL_INTERVAL
    assert config["lr"] == pytest.approx(3.0e-7)
    assert config["weight_decay"] == 0.0
    assert config["freeze_encoder_except_time_mixer"] is True
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True


def test_stage210n_uses_spike_tolerant_factorized_output_losses() -> None:
    segment = _segments(smoke=False)[0]
    config = _config(
        segment=segment,
        output_dir=Path("/tmp/stage210n-test"),
        init_checkpoint=Path("/tmp/stage210m-step1000.pt"),
        bucket_manifest=Path("/tmp/source-grouped-easy.json"),
        resume=False,
        smoke=False,
    )

    assert config["ctc_teacher_online_blank_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_conditional_nonblank_hard_loss_weight"] == pytest.approx(0.125)
    assert config["ctc_teacher_online_sequence_loss_weight"] == pytest.approx(0.20)
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_nonblank_window_loss_mode"] == "conditional_nonblank_hard"
    assert config["ctc_teacher_online_nonblank_window_radius"] == 2
    assert config["ctc_teacher_online_nonblank_window_temperature"] == pytest.approx(0.20)
    assert config["ctc_teacher_online_nonblank_window_margin_loss_weight"] == 0.0
    assert config["ctc_teacher_online_nonblank_window_topk_loss_weight"] == 0.0


def test_stage210n_smoke_keeps_two_step_gate() -> None:
    segment = _segments(smoke=True)[0]
    config = _config(
        segment=segment,
        output_dir=Path("/tmp/stage210n-smoke"),
        init_checkpoint=Path("/tmp/stage210m-step1000.pt"),
        bucket_manifest=Path("/tmp/source-grouped-easy.json"),
        resume=False,
        smoke=True,
    )

    assert segment["target_step"] == 2
    assert config["max_steps"] == 2
    assert config["save_every"] == 2
    assert config["step_eval_every"] == 2
    assert config["top_k_step_checkpoints"] == 1
