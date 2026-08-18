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
from rwkvasr.training.deepspeed_loop import (
    DeepSpeedTrainConfig,
    _apply_training_freeze as _apply_deepspeed_training_freeze,
    _capture_student_ctc_decoder_hiddens,
    _capture_student_sensevoice_layer_hiddens,
    _ctc_teacher_decoder_hidden_loss,
    _ctc_teacher_hidden_loss,
    _ctc_teacher_layer_hidden_loss,
    _online_ctc_teacher_distillation_loss,
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


def test_stage211_frozen_nano_ctc_path_backpropagates_only_to_birwkv() -> None:
    torch.manual_seed(211)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=10,
            n_embd=8,
            encoder_output_dim=8,
            dim_att=8,
            dim_ff=16,
            num_layers=3,
            vocab_size=6,
            head_size=4,
            dropout=0.0,
            frontend_type="sensevoice_rwkv",
            sensevoice_tp_blocks=1,
            ctc_decoder_type="funasr_nano_transformer",
            ctc_decoder_dim=8,
            ctc_decoder_ffn_dim=16,
            ctc_decoder_num_layers=1,
            ctc_decoder_attention_heads=2,
        )
    )
    _apply_deepspeed_training_freeze(
        model,
        DeepSpeedTrainConfig(
            output_dir=".",
            deepspeed={},
            frontend_type="sensevoice_rwkv",
            freeze_encoder_except_time_mixer=True,
            freeze_ctc_decoder=True,
            freeze_ctc_head=True,
        ),
    )

    features = torch.randn(2, 12, 10)
    feature_lengths = torch.tensor([12, 10], dtype=torch.long)
    targets = torch.tensor([1, 2, 1, 3], dtype=torch.long)
    target_lengths = torch.tensor([2, 2], dtype=torch.long)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-2,
    )
    losses = model.joint_losses(
        features,
        feature_lengths,
        targets,
        target_lengths,
    )
    losses["loss"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    losses = model.joint_losses(
        features,
        feature_lengths,
        targets,
        target_lengths,
    )
    losses["loss"].backward()

    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(".time_mixer." in name or ".input_proj." in name for name in trainable)
    assert any(
        ".time_mixer." in name
        and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in trainable.items()
    )
    assert any(
        ".input_proj." in name
        and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for name, parameter in trainable.items()
    )
    assert model.ctc_decoder is not None
    frozen_ctc_parameters = [
        *model.ctc_decoder.parameters(),
        *model.ctc_head.parameters(),
    ]
    assert all(not parameter.requires_grad for parameter in frozen_ctc_parameters)
    assert all(parameter.grad is None for parameter in frozen_ctc_parameters)


def _build_stage211_gradient_probe_model() -> RWKVCTCModel:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=10,
            n_embd=8,
            encoder_output_dim=8,
            dim_att=8,
            dim_ff=16,
            num_layers=3,
            vocab_size=6,
            blank_id=5,
            head_size=4,
            dropout=0.0,
            frontend_type="sensevoice_rwkv",
            sensevoice_tp_blocks=1,
            ctc_loss_weight=0.0,
            ctc_decoder_type="funasr_nano_transformer",
            ctc_decoder_dim=8,
            ctc_decoder_ffn_dim=16,
            ctc_decoder_num_layers=1,
            ctc_decoder_attention_heads=2,
        )
    )
    _apply_deepspeed_training_freeze(
        model,
        DeepSpeedTrainConfig(
            output_dir=".",
            deepspeed={},
            frontend_type="sensevoice_rwkv",
            freeze_encoder_except_time_mixer=True,
            freeze_ctc_decoder=True,
            freeze_ctc_head=True,
        ),
    )
    return model


def _assert_stage211_gradient_probe_reaches_only_birwkv(model: RWKVCTCModel) -> None:
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(".time_mixer." in name or ".input_proj." in name for name in trainable)
    finite_nonzero = {
        name
        for name, parameter in trainable.items()
        if parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
    }
    assert any(".time_mixer." in name for name in finite_nonzero)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in trainable.values()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if ".time_mixer." not in name and ".input_proj." not in name
    )


def test_stage211_block_objectives_backpropagate_to_frozen_nano_birwkv_path() -> None:
    torch.manual_seed(212)
    model = _build_stage211_gradient_probe_model()
    model.enable_gradient_checkpointing(True)
    model.train()
    features = torch.randn(2, 12, 10)
    feature_lengths = torch.tensor([12, 10], dtype=torch.long)
    empty_targets = torch.empty(0, dtype=torch.long)
    empty_target_lengths = torch.zeros(2, dtype=torch.long)
    layer_ids = (0, 1, 2)
    utt_ids = ("utt-a", "utt-b")

    with (
        _capture_student_sensevoice_layer_hiddens(model, layer_ids) as layer_hiddens,
        _capture_student_ctc_decoder_hiddens(model, enabled=True) as decoder_hiddens,
    ):
        losses = model.joint_losses(
            features,
            feature_lengths,
            empty_targets,
            empty_target_lengths,
            compute_ctc_logits=False,
        )
    encoded = losses["ctc_encoded"]
    encoded_lengths = losses["ctc_encoded_lengths"]
    logit_lengths = losses["logit_lengths"]
    assert isinstance(encoded, torch.Tensor)
    assert isinstance(encoded_lengths, torch.Tensor)
    assert isinstance(logit_lengths, torch.Tensor)

    teacher_records: dict[str, dict[str, object]] = {}
    for sample_idx, utt_id in enumerate(utt_ids):
        encoded_length = int(encoded_lengths[sample_idx])
        logit_length = int(logit_lengths[sample_idx])
        teacher_records[utt_id] = {
            "encoder_out": torch.flip(
                encoded[sample_idx, :encoded_length].detach(),
                dims=(-1,),
            )
            + 0.05,
            "encoder_layer_hiddens": {
                str(layer_id): {
                    component: torch.flip(
                        layer_hiddens[layer_id][component][sample_idx, :encoded_length].detach(),
                        dims=(-1,),
                    )
                    + 0.05
                    for component in ("mixer", "ffn", "block")
                }
                for layer_id in layer_ids
            },
            "ctc_decoder_hiddens": {
                name: torch.flip(
                    hidden[sample_idx, :logit_length].detach(),
                    dims=(-1,),
                )
                + 0.05
                for name, hidden in decoder_hiddens.items()
            },
        }

    layer_result = _ctc_teacher_layer_hidden_loss(
        layer_hiddens,
        encoded_lengths,
        utt_ids,
        teacher_records,
        layer_ids=layer_ids,
        component_weights={"mixer": 0.25, "ffn": 0.25, "block": 1.0},
        normalized_mse_weight=1.0,
        cosine_weight=0.25,
        energy_mse_weight=0.25,
        log_rms_weight=0.1,
        raw_mse_weight=0.0,
        frame_tolerance=0,
        missing_policy="error",
    )
    encoder_loss, encoder_matched, encoder_missing = _ctc_teacher_hidden_loss(
        encoded,
        encoded_lengths,
        utt_ids,
        teacher_records,
        teacher_field="encoder_out",
        time_map="nearest",
        missing_policy="error",
    )
    decoder_result = _ctc_teacher_decoder_hidden_loss(
        decoder_hiddens,
        logit_lengths,
        utt_ids,
        teacher_records,
        normalized_mse_weight=1.0,
        cosine_weight=0.25,
        energy_mse_weight=0.25,
        log_rms_weight=0.1,
        raw_mse_weight=0.0,
        frame_tolerance=0,
        missing_policy="error",
    )
    total = layer_result.loss + 0.5 * encoder_loss + 0.5 * decoder_result.loss

    assert torch.isfinite(total)
    assert total.item() > 0.0
    assert (layer_result.matched_samples, layer_result.missing_samples) == (2, 0)
    assert (encoder_matched, encoder_missing) == (2, 0)
    assert (decoder_result.matched_samples, decoder_result.missing_samples) == (2, 0)
    total.backward()
    _assert_stage211_gradient_probe_reaches_only_birwkv(model)


def test_stage211_logits_only_objectives_backpropagate_through_frozen_nano_path() -> None:
    torch.manual_seed(213)
    model = _build_stage211_gradient_probe_model()
    model.enable_gradient_checkpointing(True)
    model.train()
    features = torch.randn(2, 12, 10)
    feature_lengths = torch.tensor([12, 10], dtype=torch.long)
    losses = model.joint_losses(
        features,
        feature_lengths,
        torch.empty(0, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        compute_ctc_logits=True,
    )
    student_logits = losses["logits"]
    logit_lengths = losses["logit_lengths"]
    assert isinstance(student_logits, torch.Tensor)
    assert isinstance(logit_lengths, torch.Tensor)
    utt_ids = ("utt-a", "utt-b")
    teacher_records: dict[str, dict[str, object]] = {}
    for sample_idx, utt_id in enumerate(utt_ids):
        length = int(logit_lengths[sample_idx])
        teacher_logits = torch.randn(length, student_logits.size(-1))
        teacher_logits[:, 5] += 1.0
        teacher_logits[::2, sample_idx + 1] += 4.0
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
        topk_log_probs, topk_token_ids = teacher_log_probs.topk(4, dim=-1)
        teacher_records[utt_id] = {
            "full_log_probs": teacher_log_probs.to(torch.float16),
            "topk_token_ids": topk_token_ids,
            "topk_log_probs": topk_log_probs,
            "blank_log_probs": teacher_log_probs[:, 5],
            "project_blank_id": 5,
            "teacher_blank_id": 5,
            "project_ignored_token_ids": [],
        }

    total = _online_ctc_teacher_distillation_loss(
        config=DeepSpeedTrainConfig(
            output_dir=".",
            deepspeed={},
            blank_id=5,
            ctc_teacher_online_full_loss_weight=1.0,
            ctc_teacher_online_blank_loss_weight=0.25,
            ctc_teacher_online_conditional_nonblank_loss_weight=1.0,
            ctc_teacher_online_conditional_nonblank_hard_loss_weight=0.125,
            ctc_teacher_online_full_frame_filter="all",
            ctc_teacher_online_project_ignored_token_ids=(),
            ctc_teacher_topk_missing_policy="error",
        ),
        losses=losses,
        batch=types.SimpleNamespace(utt_ids=utt_ids),
        ctc_teacher_online_records=teacher_records,
    )

    assert torch.isfinite(total)
    assert total.item() > 0.0
    total.backward()
    _assert_stage211_gradient_probe_reaches_only_birwkv(model)


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
