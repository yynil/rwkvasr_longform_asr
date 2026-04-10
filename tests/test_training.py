import sys
import types

import torch
import sentencepiece as spm

from rwkvasr.data import ASRBatch
from rwkvasr.modules import (
    DirectionDropoutConfig,
    DirectionDropoutScheduler,
    RWKVCTCModel,
    RWKVCTCModelConfig,
)
from rwkvasr.training import (
    CTCBatch,
    RWKVDualModeCTCTrainer,
    build_rwkv_param_groups,
)
from rwkvasr.training.train_loop import TrainConfig, _resolve_vocab_size
from rwkvasr.training.wandb_logger import finish_wandb, init_wandb_run, log_wandb
from rwkvasr.data import can_load_webdataset_length_index_in_memory
from rwkvasr.training.synthetic import (
    SyntheticOverfitConfig,
    make_synthetic_ctc_batch,
    run_synthetic_overfit,
)


def _build_model() -> RWKVCTCModel:
    return RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            vocab_size=24,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
        )
    )


def test_rwkv_optimizer_param_groups_respect_w0_and_weight_decay() -> None:
    model = _build_model()
    groups = build_rwkv_param_groups(model, lr=1e-3, weight_decay=0.1)
    names = {group["name"]: set(group["param_names"]) for group in groups}

    w0_names = names["rwkv_2x"]
    assert any(name.endswith(".w0") for name in w0_names)
    assert "encoder.blocks.0.time_mixer.forward_mixer.w0" in w0_names

    decay_names = names["rwkv_decay"]
    assert "ctc_head.weight" in decay_names
    assert "encoder.blocks.0.ffn1_norm.weight" not in decay_names


def test_dual_mode_trainer_returns_valid_training_mask() -> None:
    model = _build_model()
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=model.config.num_layers,
            variant="drop_both",
            p_start=1.0,
            p_max=1.0,
            warmup_steps=0,
            ramp_steps=0,
        )
    )
    trainer = RWKVDualModeCTCTrainer(model, direction_scheduler=scheduler)
    batch = CTCBatch(
        features=torch.randn(2, 9, 80),
        feature_lengths=torch.tensor([9, 9], dtype=torch.long),
        targets=torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.long),
        target_lengths=torch.tensor([3, 3], dtype=torch.long),
    )

    loss, mask = trainer.training_loss(
        batch,
        step=0,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.isfinite(loss)
    assert mask.num_layers == model.config.num_layers
    assert torch.all(mask.forward | mask.backward)


def test_ctc_batch_to_can_cast_feature_dtype() -> None:
    batch = CTCBatch(
        features=torch.randn(2, 9, 80),
        feature_lengths=torch.tensor([9, 7], dtype=torch.long),
        targets=torch.tensor([1, 2, 3], dtype=torch.long),
        target_lengths=torch.tensor([2, 1], dtype=torch.long),
    )

    moved = batch.to("cpu", feature_dtype=torch.bfloat16)

    assert moved.features.dtype == torch.bfloat16
    assert moved.feature_lengths.dtype == torch.long
    assert moved.targets.dtype == torch.long


def test_asr_batch_prefix_truncates_feature_and_target_tensors() -> None:
    batch = ASRBatch(
        features=torch.randn(3, 9, 80),
        feature_lengths=torch.tensor([9, 7, 5], dtype=torch.long),
        targets=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
        target_lengths=torch.tensor([2, 1, 3], dtype=torch.long),
        utt_ids=["utt-0", "utt-1", "utt-2"],
    )

    trimmed = batch.prefix(2)

    assert trimmed.features.shape == (2, 9, 80)
    assert trimmed.feature_lengths.tolist() == [9, 7]
    assert trimmed.targets.tolist() == [1, 2, 3]
    assert trimmed.target_lengths.tolist() == [2, 1]
    assert trimmed.utt_ids == ["utt-0", "utt-1"]


def test_make_synthetic_ctc_batch_shapes() -> None:
    batch = make_synthetic_ctc_batch(SyntheticOverfitConfig(batch_size=3, target_len=5))

    assert batch.features.shape[0] == 3
    assert batch.feature_lengths.shape == (3,)
    assert batch.target_lengths.shape == (3,)
    assert batch.targets.numel() == 15


def test_synthetic_overfit_reduces_loss() -> None:
    result = run_synthetic_overfit(
        SyntheticOverfitConfig(
            steps=8,
            lr=3e-3,
            p_start=0.0,
            p_max=0.0,
            warmup_steps=0,
            ramp_steps=0,
        )
    )

    assert result["final_loss"] < result["initial_loss"]


def test_resolve_vocab_size_uses_sentencepiece_model(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("hello world\nni hao shi jie\nbonjour le monde\n", encoding="utf-8")
    model_prefix = tmp_path / "tiny_spm"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=18,
        character_coverage=1.0,
        split_digits=True,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
    )

    config = TrainConfig(
        output_dir=str(tmp_path / "out"),
        manifest_path=str(corpus_path),
        tokenizer_type="sentencepiece",
        tokenizer_model_path=str(model_prefix.with_suffix(".model")),
        vocab_size=None,
    )
    assert _resolve_vocab_size(config) == 18


def test_can_load_webdataset_length_index_in_memory_respects_limit(tmp_path) -> None:
    index_path = tmp_path / "lengths.jsonl"
    index_path.write_text("x" * 32, encoding="utf-8")

    assert can_load_webdataset_length_index_in_memory(index_path, max_bytes=64) is True
    assert can_load_webdataset_length_index_in_memory(index_path, max_bytes=16) is False


def test_wandb_logger_noops_when_disabled(tmp_path) -> None:
    run = init_wandb_run(
        enabled=False,
        project="rwkvasr_longform_asr",
        run_name="sp8k_4090",
        output_dir=tmp_path,
        config={"x": 1},
    )
    assert run is None
    log_wandb(run, {"loss": 1.0}, step=1)
    finish_wandb(run)


def test_wandb_logger_falls_back_when_init_fails(tmp_path) -> None:
    messages: list[str] = []

    fake_wandb = types.SimpleNamespace()

    class FakeSettings:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    def fake_init(**kwargs):
        raise RuntimeError("init timeout")

    fake_wandb.Settings = FakeSettings
    fake_wandb.init = fake_init

    previous = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_wandb
    try:
        run = init_wandb_run(
            enabled=True,
            project="rwkvasr_longform_asr",
            run_name="sp8k_4090",
            output_dir=tmp_path,
            config={"x": 1},
            base_url="https://api.wandb.ai",
            init_timeout_sec=12.5,
            logger=messages.append,
        )
    finally:
        if previous is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = previous

    assert run is None
    assert messages
    assert "wandb init failed" in messages[0]
    assert "https://api.wandb.ai" in messages[0]
