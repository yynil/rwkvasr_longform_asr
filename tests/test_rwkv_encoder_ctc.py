import math

import pytest
import torch
import torch.nn.functional as F

from rwkvasr.modules import (
    RWKVCTCModel,
    RWKVCTCModelConfig,
    RWKVConformerEncoder,
    RWKVConformerEncoderConfig,
    aut_conv2d8_out_lengths,
    build_inference_direction_mask,
)
from rwkvasr.training.deepspeed_loop import (
    _CTC_FULL_EVAL_WIDTH,
    _ctc_teacher_full_loss,
    _ctc_teacher_hidden_loss,
    _ctc_teacher_nonblank_hard_loss,
    _ctc_teacher_nonblank_window_loss,
    _ctc_teacher_nonblank_window_topk_loss,
    _ctc_teacher_sequence_presence_loss,
    _ctc_teacher_sequence_window_loss,
    _finalize_ctc_full_eval_metrics,
    _project_ctc_teacher_full_log_probs,
)


def _encoder() -> RWKVConformerEncoder:
    return RWKVConformerEncoder(
        RWKVConformerEncoderConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
        )
    )


def _ctc_model() -> RWKVCTCModel:
    return RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            vocab_size=32,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
        )
    )


def test_encoder_stack_forward_shape() -> None:
    torch.manual_seed(20)
    encoder = _encoder()
    x = torch.randn(2, 12, 80)
    lengths = torch.tensor([12, 10], dtype=torch.long)

    y, out_lengths, state = encoder(x, lengths)

    assert y.shape == (2, 12, 128)
    assert torch.equal(out_lengths, lengths)
    assert len(state.block_states) == 2


def test_encoder_stack_l2r_chunk_consistency() -> None:
    torch.manual_seed(21)
    encoder = _encoder()
    x = torch.randn(2, 11, 80)
    lengths = torch.tensor([11, 11], dtype=torch.long)
    l2r_mask = build_inference_direction_mask(2, mode="l2r")

    full_y, _, _ = encoder(x, lengths, direction_mask=l2r_mask)

    state = None
    outputs = []
    for chunk in [x[:, :4], x[:, 4:8], x[:, 8:]]:
        y, _, state = encoder(chunk, None, direction_mask=l2r_mask, state=state)
        outputs.append(y)

    chunk_y = torch.cat(outputs, dim=1)
    assert torch.allclose(full_y, chunk_y, atol=1e-5, rtol=1e-5)


def test_ctc_model_forward_and_loss() -> None:
    torch.manual_seed(22)
    model = _ctc_model()
    features = torch.randn(2, 10, 80)
    feature_lengths = torch.tensor([10, 8], dtype=torch.long)

    logits, logit_lengths, _ = model(features, feature_lengths)
    targets = torch.tensor([1, 2, 3, 4, 1, 2], dtype=torch.long)
    target_lengths = torch.tensor([4, 2], dtype=torch.long)
    loss = model.ctc_loss(logits, logit_lengths, targets, target_lengths)

    assert logits.shape == (2, 10, 32)
    assert torch.equal(logit_lengths, feature_lengths)
    assert torch.isfinite(loss)


def test_nano_style_ctc_decoder_path_forward_and_loss() -> None:
    torch.manual_seed(2201)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=8,
            n_embd=16,
            dim_att=16,
            dim_ff=32,
            num_layers=2,
            vocab_size=10,
            blank_id=10,
            head_size=8,
            conv_kernel_size=3,
            dropout=0.0,
            frontend_type="linear",
            ctc_decoder_type="funasr_nano_transformer",
            ctc_decoder_dim=16,
            ctc_decoder_ffn_dim=32,
            ctc_decoder_num_layers=2,
            ctc_decoder_attention_heads=4,
        )
    )
    features = torch.randn(2, 7, 8)
    feature_lengths = torch.tensor([7, 5], dtype=torch.long)
    logits, logit_lengths, _ = model(features, feature_lengths)
    targets = torch.tensor([1, 2, 3], dtype=torch.long)
    target_lengths = torch.tensor([2, 1], dtype=torch.long)
    loss = model.ctc_loss(logits, logit_lengths, targets, target_lengths)

    assert model.ctc_decoder is not None
    assert model.ctc_head.in_features == 16
    assert logits.shape == (2, 7, 11)
    assert torch.equal(logit_lengths, feature_lengths)
    assert torch.isfinite(loss)


def test_ctc_bridge_residual_mlp_is_identity_initialized() -> None:
    torch.manual_seed(22011)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=8,
            n_embd=16,
            dim_att=16,
            dim_ff=32,
            num_layers=2,
            vocab_size=10,
            blank_id=10,
            head_size=8,
            conv_kernel_size=3,
            dropout=0.0,
            frontend_type="linear",
            ctc_bridge_type="residual_mlp",
            ctc_bridge_hidden_dim=32,
            ctc_bridge_dropout=0.0,
            ctc_loss_weight=0.0,
        )
    )
    encoded = torch.randn(2, 7, 16)
    lengths = torch.tensor([7, 5], dtype=torch.long)

    ctc_encoded, ctc_lengths = model.ctc_encoder_features_from_encoded(encoded, lengths)

    assert torch.allclose(ctc_encoded, encoded)
    assert torch.equal(ctc_lengths, lengths)
    assert any(parameter.requires_grad for parameter in model.ctc_bridge.parameters())

    features = torch.randn(2, 7, 8)
    targets = torch.empty(0, dtype=torch.long)
    target_lengths = torch.tensor([0, 0], dtype=torch.long)
    losses = model.joint_losses(features, lengths, targets, target_lengths)

    assert losses["encoded"].shape == (2, 7, 16)
    assert losses["ctc_encoded"].shape == (2, 7, 16)
    assert torch.equal(losses["ctc_encoded_lengths"], lengths)


def test_ctc_bridge_context_residual_mlp_is_identity_initialized_and_masks_padding() -> None:
    torch.manual_seed(22012)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=8,
            n_embd=16,
            dim_att=16,
            dim_ff=32,
            num_layers=2,
            vocab_size=10,
            blank_id=10,
            head_size=8,
            conv_kernel_size=3,
            dropout=0.0,
            frontend_type="linear",
            ctc_bridge_type="context_residual_mlp",
            ctc_bridge_hidden_dim=32,
            ctc_bridge_dropout=0.0,
            ctc_loss_weight=0.0,
        )
    )
    encoded = torch.randn(2, 7, 16)
    lengths = torch.tensor([7, 4], dtype=torch.long)

    ctc_encoded, ctc_lengths = model.ctc_encoder_features_from_encoded(encoded, lengths)

    assert torch.allclose(ctc_encoded, encoded)
    assert torch.equal(ctc_lengths, lengths)
    assert any(parameter.requires_grad for parameter in model.ctc_bridge.parameters())

    with torch.no_grad():
        model.ctc_bridge.temporal.weight.fill_(0.1)
        model.ctc_bridge.temporal.bias.fill_(0.2)
    changed, _ = model.ctc_encoder_features_from_encoded(encoded, lengths)

    assert not torch.allclose(changed[0, : lengths[0]], encoded[0, : lengths[0]])
    assert not torch.allclose(changed[1, : lengths[1]], encoded[1, : lengths[1]])
    assert torch.allclose(changed[1, lengths[1] :], encoded[1, lengths[1] :])


def test_ctc_bridge_nano_encoder_tail_forward_and_loads_tp_tail_weights() -> None:
    torch.manual_seed(22013)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=8,
            n_embd=16,
            dim_att=16,
            dim_ff=32,
            num_layers=2,
            vocab_size=10,
            blank_id=10,
            head_size=8,
            conv_kernel_size=3,
            dropout=0.0,
            frontend_type="linear",
            ctc_bridge_type="nano_encoder_tail2",
            ctc_bridge_hidden_dim=64,
            ctc_bridge_dropout=0.0,
            ctc_loss_weight=0.0,
        )
    )
    encoded = torch.randn(2, 7, 16)
    lengths = torch.tensor([7, 5], dtype=torch.long)

    ctc_encoded, ctc_lengths = model.ctc_encoder_features_from_encoded(encoded, lengths)

    assert ctc_encoded.shape == encoded.shape
    assert torch.equal(ctc_lengths, lengths)

    bridge = model.ctc_bridge
    source = {}
    for block_index, block in enumerate(bridge.blocks):
        source_index = bridge.source_block_offset + block_index
        for key, value in block.state_dict().items():
            source[f"audio_encoder.tp_encoders.{source_index}.{key}"] = torch.randn_like(value)
    for key, value in bridge.final_norm.state_dict().items():
        source[f"audio_encoder.tp_norm.{key}"] = torch.randn_like(value)

    report = bridge.load_funasr_nano_bridge_state_dict(source)

    expected = sum(len(block.state_dict()) for block in bridge.blocks) + len(bridge.final_norm.state_dict())
    assert len(report["loaded"]) == expected
    assert report["skipped"] == []
    assert torch.equal(
        bridge.blocks[0].norm1.weight,
        source[f"audio_encoder.tp_encoders.{bridge.source_block_offset}.norm1.weight"],
    )
    assert torch.equal(bridge.final_norm.bias, source["audio_encoder.tp_norm.bias"])


def test_funasr_nano_ctc_init_remaps_teacher_blank(tmp_path) -> None:
    torch.manual_seed(2202)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=8,
            n_embd=16,
            dim_att=16,
            dim_ff=32,
            num_layers=2,
            vocab_size=21,
            blank_id=21,
            head_size=8,
            conv_kernel_size=3,
            dropout=0.0,
            frontend_type="linear",
            ctc_decoder_type="funasr_nano_transformer",
            ctc_decoder_dim=16,
            ctc_decoder_ffn_dim=32,
            ctc_decoder_num_layers=1,
            ctc_decoder_attention_heads=4,
        )
    )
    assert model.ctc_decoder is not None
    state = {
        f"ctc_decoder.{key}": torch.randn_like(value)
        for key, value in model.ctc_decoder.state_dict().items()
    }
    teacher_weight = torch.randn(21, 16)
    teacher_bias = torch.arange(21, dtype=torch.float32)
    state["ctc.ctc_lo.weight"] = teacher_weight
    state["ctc.ctc_lo.bias"] = teacher_bias
    checkpoint_path = tmp_path / "nano_ctc.pt"
    torch.save({"state_dict": state}, checkpoint_path)

    report = model.load_funasr_nano_ctc_checkpoint(
        str(checkpoint_path),
        teacher_blank_id=20,
        project_ignored_token_ids=(20,),
    )

    assert report["ctc_decoder_loaded"] == len(model.ctc_decoder.state_dict())
    assert report["ctc_head_loaded_rows"] == 21
    assert torch.allclose(model.ctc_head.weight[0], teacher_weight[0])
    assert torch.allclose(model.ctc_head.weight[21], teacher_weight[20])
    assert torch.equal(model.ctc_head.bias[21], teacher_bias[20])
    assert torch.equal(model.ctc_head.weight[20], torch.zeros_like(model.ctc_head.weight[20]))
    assert model.ctc_head.bias[20].item() < -9999


def test_funasr_nano_ctc_init_applies_blank_bias_delta_after_remap(tmp_path) -> None:
    torch.manual_seed(2203)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=8,
            n_embd=16,
            dim_att=16,
            dim_ff=32,
            num_layers=2,
            vocab_size=21,
            blank_id=21,
            head_size=8,
            conv_kernel_size=3,
            dropout=0.0,
            frontend_type="linear",
            ctc_decoder_type="funasr_nano_transformer",
            ctc_decoder_dim=16,
            ctc_decoder_ffn_dim=32,
            ctc_decoder_num_layers=1,
            ctc_decoder_attention_heads=4,
        )
    )
    assert model.ctc_decoder is not None
    state = {
        f"ctc_decoder.{key}": torch.randn_like(value)
        for key, value in model.ctc_decoder.state_dict().items()
    }
    teacher_weight = torch.randn(21, 16)
    teacher_bias = torch.arange(21, dtype=torch.float32)
    state["ctc.ctc_lo.weight"] = teacher_weight
    state["ctc.ctc_lo.bias"] = teacher_bias
    checkpoint_path = tmp_path / "nano_ctc.pt"
    torch.save({"state_dict": state}, checkpoint_path)

    report = model.load_funasr_nano_ctc_checkpoint(
        str(checkpoint_path),
        teacher_blank_id=20,
        project_ignored_token_ids=(20,),
        blank_bias_delta=-3.0,
    )

    assert report["ctc_head_blank_bias_delta"] == -3.0
    assert torch.equal(model.ctc_head.bias[21], teacher_bias[20] - 3.0)
    assert model.ctc_head.bias[20].item() < -9999


def test_project_ctc_teacher_full_log_probs_remaps_teacher_blank() -> None:
    raw = torch.full((2, 5), float("-inf"))
    raw[:, 0] = torch.log(torch.tensor([0.1, 0.2]))
    raw[:, 2] = torch.log(torch.tensor([0.2, 0.1]))
    raw[:, 4] = torch.log(torch.tensor([0.7, 0.7]))

    projected = _project_ctc_teacher_full_log_probs(
        raw,
        vocab_size=6,
        blank_id=5,
        teacher_blank_id=4,
        project_blank_id=5,
        ignored_token_ids=(4,),
    )

    assert projected.shape == (2, 6)
    assert torch.allclose(projected[:, 0], raw[:, 0])
    assert torch.allclose(projected[:, 2], raw[:, 2])
    assert torch.isneginf(projected[:, 4]).all()
    assert torch.allclose(projected[:, 5], raw[:, 4])


def test_project_ctc_teacher_full_log_probs_keeps_preprojected_blank() -> None:
    raw = torch.full((2, 6), float("-inf"))
    raw[:, 0] = torch.log(torch.tensor([0.1, 0.2]))
    raw[:, 5] = torch.log(torch.tensor([0.9, 0.8]))

    projected = _project_ctc_teacher_full_log_probs(
        raw,
        vocab_size=6,
        blank_id=5,
        teacher_blank_id=4,
        project_blank_id=5,
        ignored_token_ids=(4,),
    )

    assert projected.shape == (2, 6)
    assert torch.isneginf(projected[:, 4]).all()
    assert torch.allclose(projected[:, 5], raw[:, 5])


@pytest.mark.parametrize("time_map", ["nearest", "linear"])
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_ctc_teacher_full_loss_supports_device_resident_targets(
    time_map: str,
    device_type: str,
) -> None:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_type)
    teacher_logits = torch.randn(5, 6, device=device)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1).to(dtype=torch.float16)
    student_logits = torch.randn(1, 3, 6, device=device, requires_grad=True)
    records = {
        "utt-a": {
            "full_log_probs": teacher_log_probs,
            "project_blank_id": 5,
            "teacher_blank_id": 5,
            "project_ignored_token_ids": [],
        }
    }

    loss, matched, missing = _ctc_teacher_full_loss(
        student_logits,
        torch.tensor([3], device=device),
        ["utt-a"],
        records,
        blank_id=5,
        time_map=time_map,
        frame_filter="all",
        frame_filter_neighbor_radius=0,
        frame_filter_min_nonblank_prob=0.0,
        missing_policy="error",
        temperature=1.0,
    )

    assert loss.device == student_logits.device
    assert torch.isfinite(loss)
    assert matched == 1
    assert missing == 0
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_ctc_teacher_full_eval_metrics_expose_blank_peak_and_collapsed_sequence_gap() -> None:
    teacher_ids = torch.tensor([5, 1, 1, 5, 2])
    student_ids = torch.tensor([5, 1, 5, 3, 3])
    teacher_logits = torch.full((5, 6), -20.0)
    student_logits = torch.full((1, 5, 6), -20.0, requires_grad=True)
    teacher_logits.scatter_(1, teacher_ids.unsqueeze(1), 20.0)
    with torch.no_grad():
        student_logits.scatter_(2, student_ids.view(1, 5, 1), 20.0)
    records = {
        "utt-a": {
            "full_log_probs": F.log_softmax(teacher_logits, dim=-1),
            "project_blank_id": 5,
            "teacher_blank_id": 5,
            "project_ignored_token_ids": [],
        }
    }
    accumulator = torch.zeros(_CTC_FULL_EVAL_WIDTH, dtype=torch.float64)

    loss, matched, missing = _ctc_teacher_full_loss(
        student_logits,
        torch.tensor([5]),
        ["utt-a"],
        records,
        blank_id=5,
        time_map="nearest",
        frame_filter="all",
        frame_filter_neighbor_radius=0,
        frame_filter_min_nonblank_prob=0.0,
        missing_policy="error",
        temperature=1.0,
        eval_accumulator=accumulator,
    )
    metrics = _finalize_ctc_full_eval_metrics(accumulator, device=torch.device("cpu"))

    assert torch.isfinite(loss)
    assert matched == 1
    assert missing == 0
    assert metrics["full_kl"] > 0.0
    assert metrics["selected_top1_agreement"] == pytest.approx(0.4)
    assert metrics["all_top1_agreement"] == pytest.approx(0.4)
    assert metrics["blank_prob_mae"] > 0.0
    assert metrics["teacher_nonblank_rate"] == pytest.approx(0.6)
    assert metrics["student_nonblank_rate"] == pytest.approx(0.6)
    assert metrics["nonblank_rate_ratio"] == pytest.approx(1.0)
    assert metrics["ctc_token_error_rate"] == pytest.approx(0.5)
    assert metrics["collapsed_length_ratio"] == pytest.approx(1.0)
    assert metrics["sequence_exact_rate"] == 0.0
    assert metrics["mean_frame_delta"] == 0.0
    assert metrics["teacher_tokens"] == 2.0
    assert metrics["student_tokens"] == 2.0
    assert metrics["matched_utterances"] == 1.0
    assert metrics["missing_utterances"] == 0.0


def test_ctc_teacher_full_loss_upweights_nonblank_frames_without_masking_blank_frames() -> None:
    teacher_ids = torch.tensor([3, 1, 3])
    teacher_logits = torch.full((3, 4), -20.0)
    teacher_logits.scatter_(1, teacher_ids.unsqueeze(1), 20.0)
    student_logits = torch.tensor(
        [[[0.0, 0.0, 0.0, 5.0], [0.0, -3.0, 0.0, 3.0], [2.0, 0.0, 0.0, 1.0]]],
        requires_grad=True,
    )
    records = {
        "utt-a": {
            "full_log_probs": F.log_softmax(teacher_logits, dim=-1),
            "project_blank_id": 3,
            "teacher_blank_id": 3,
            "project_ignored_token_ids": [],
        }
    }

    unweighted, _, _ = _ctc_teacher_full_loss(
        student_logits,
        torch.tensor([3]),
        ["utt-a"],
        records,
        blank_id=3,
        time_map="nearest",
        frame_filter="all",
        frame_filter_neighbor_radius=0,
        frame_filter_min_nonblank_prob=0.0,
        missing_policy="error",
        temperature=1.0,
        nonblank_frame_weight=1.0,
    )
    weighted, matched, missing = _ctc_teacher_full_loss(
        student_logits,
        torch.tensor([3]),
        ["utt-a"],
        records,
        blank_id=3,
        time_map="nearest",
        frame_filter="all",
        frame_filter_neighbor_radius=0,
        frame_filter_min_nonblank_prob=0.0,
        missing_policy="error",
        temperature=1.0,
        nonblank_frame_weight=4.0,
    )

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_logits[0], dim=-1)
    per_frame = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    expected = (per_frame[0] + 4.0 * per_frame[1] + per_frame[2]) / 6.0

    assert weighted.item() == pytest.approx(expected.item())
    assert weighted.item() > unweighted.item()
    assert matched == 1
    assert missing == 0
    weighted.backward()
    assert student_logits.grad is not None
    assert torch.count_nonzero(student_logits.grad[0, 0]).item() > 0
    assert torch.count_nonzero(student_logits.grad[0, 1]).item() > 0
    assert torch.count_nonzero(student_logits.grad[0, 2]).item() > 0


def test_ctc_suppressed_token_ids_mask_logits_but_not_blank() -> None:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=8,
            dim_att=8,
            dim_ff=16,
            num_layers=1,
            head_size=4,
            vocab_size=8,
            blank_id=7,
            ctc_suppressed_token_ids=(3, 7, 99),
        )
    )
    logits = torch.zeros(2, 3, 8)

    masked = model.apply_ctc_logit_mask(logits)

    assert masked[..., 3].max().item() < -999.0
    assert torch.equal(masked[..., 7], logits[..., 7])
    assert torch.equal(masked[..., 0], logits[..., 0])


def test_ctc_teacher_nonblank_hard_loss_selects_teacher_emission_frames() -> None:
    student_logits = torch.zeros(1, 4, 6)
    student_logits[0, 1, 5] = 3.0
    student_logits[0, 1, 2] = 1.0
    teacher_ids = torch.tensor(
        [
            [5, 1],
            [2, 5],
            [5, 3],
            [4, 5],
        ],
        dtype=torch.long,
    )
    teacher_log_probs = torch.log(
        torch.tensor(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.95, 0.05],
                [0.85, 0.15],
            ],
            dtype=torch.float32,
        )
    )
    records = {
        "utt-1": {
            "topk_token_ids": teacher_ids,
            "topk_log_probs": teacher_log_probs,
            "project_blank_id": 5,
            "project_ignored_token_ids": [4],
        }
    }

    hard_loss, margin_loss, matched, missing = _ctc_teacher_nonblank_hard_loss(
        student_logits,
        torch.tensor([4]),
        ["utt-1"],
        records,
        blank_id=5,
        time_map="nearest",
        frame_filter_min_nonblank_prob=0.5,
        missing_policy="error",
        margin=0.5,
    )

    expected_hard = F.cross_entropy(student_logits[0, 1].unsqueeze(0), torch.tensor([2]))
    assert matched == 1
    assert missing == 0
    assert torch.allclose(hard_loss, expected_hard)
    assert torch.allclose(margin_loss, torch.tensor(2.5))


def test_ctc_teacher_nonblank_window_loss_allows_local_emission_shift() -> None:
    student_logits = torch.zeros(1, 5, 6)
    student_logits[0, 2, 5] = 4.0
    student_logits[0, 3, 2] = 5.0
    teacher_ids = torch.tensor(
        [
            [5, 1],
            [5, 1],
            [2, 5],
            [5, 3],
            [5, 3],
        ],
        dtype=torch.long,
    )
    teacher_log_probs = torch.log(
        torch.tensor(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.80, 0.20],
                [0.96, 0.04],
                [0.96, 0.04],
            ],
            dtype=torch.float32,
        )
    )
    records = {
        "utt-1": {
            "topk_token_ids": teacher_ids,
            "topk_log_probs": teacher_log_probs,
            "project_blank_id": 5,
            "project_ignored_token_ids": [4],
        }
    }

    token_loss, margin_loss, matched, missing, events = _ctc_teacher_nonblank_window_loss(
        student_logits,
        torch.tensor([5]),
        ["utt-1"],
        records,
        blank_id=5,
        time_map="nearest",
        frame_filter_min_nonblank_prob=0.5,
        missing_policy="error",
        margin=0.5,
        window_radius=1,
        temperature=0.0,
    )

    expected_token = F.cross_entropy(student_logits[0, 3].unsqueeze(0), torch.tensor([2]))
    assert matched == 1
    assert missing == 0
    assert events == 1
    assert torch.allclose(token_loss, expected_token)
    assert torch.allclose(margin_loss, torch.tensor(0.0))


def test_ctc_teacher_nonblank_window_topk_loss_uses_local_nonblank_distribution() -> None:
    student_logits = torch.zeros(1, 5, 7)
    student_logits[0, 2, 6] = 4.0
    student_logits[0, 3, 2] = 4.0
    student_logits[0, 3, 3] = 3.0
    teacher_ids = torch.tensor(
        [
            [6, 1, 2],
            [6, 1, 2],
            [2, 3, 6],
            [6, 3, 2],
            [6, 3, 2],
        ],
        dtype=torch.long,
    )
    teacher_log_probs = torch.log(
        torch.tensor(
            [
                [0.95, 0.03, 0.02],
                [0.90, 0.08, 0.02],
                [0.60, 0.30, 0.10],
                [0.96, 0.03, 0.01],
                [0.96, 0.03, 0.01],
            ],
            dtype=torch.float32,
        )
    )
    records = {
        "utt-1": {
            "topk_token_ids": teacher_ids,
            "topk_log_probs": teacher_log_probs,
            "project_blank_id": 6,
            "project_ignored_token_ids": [5],
        }
    }

    loss, matched, missing, events = _ctc_teacher_nonblank_window_topk_loss(
        student_logits,
        torch.tensor([5]),
        ["utt-1"],
        records,
        blank_id=6,
        time_map="nearest",
        frame_filter_min_nonblank_prob=0.5,
        missing_policy="error",
        window_radius=1,
        temperature=0.0,
    )

    log_probs = F.log_softmax(student_logits[0, 3].float(), dim=-1)
    teacher_probs = torch.tensor([0.60, 0.30]) / 0.90
    expected = -(teacher_probs * log_probs[torch.tensor([2, 3])]).sum()
    assert matched == 1
    assert missing == 0
    assert events == 1
    assert torch.allclose(loss, expected)


def test_ctc_teacher_sequence_presence_loss_rewards_tokens_anywhere() -> None:
    student_logits = torch.full((1, 4, 7), -4.0)
    student_logits[0, :, 6] = 2.0
    student_logits[0, 1, 2] = 6.0
    student_logits[0, 3, 3] = 5.0
    records = {
        "utt-1": {
            "argmax_token_ids": [2, 3, 6, 5],
        }
    }

    loss, matched, missing, token_count = _ctc_teacher_sequence_presence_loss(
        student_logits,
        torch.tensor([4]),
        ["utt-1"],
        records,
        blank_id=6,
        ignored_token_ids=(5,),
        missing_policy="error",
    )

    frame_probs = torch.softmax(student_logits[0].float(), dim=-1)
    targets = torch.tensor([2, 3])
    token_probs = frame_probs.index_select(dim=-1, index=targets).clamp(max=1.0 - 1.0e-6)
    expected = -torch.log((-torch.expm1(torch.log1p(-token_probs).sum(dim=0))).clamp_min(1.0e-8)).mean()
    assert matched == 1
    assert missing == 0
    assert token_count == 2
    assert torch.allclose(loss, expected)


def test_ctc_teacher_sequence_window_loss_maps_tokens_to_ordered_windows() -> None:
    student_logits = torch.full((1, 5, 7), -4.0)
    student_logits[0, :, 6] = 2.0
    student_logits[0, 1, 2] = 6.0
    student_logits[0, 3, 3] = 5.0
    student_logits[0, 4, 2] = 9.0
    records = {
        "utt-1": {
            "argmax_token_ids": [2, 3, 6, 5],
        }
    }

    loss, matched, missing, events = _ctc_teacher_sequence_window_loss(
        student_logits,
        torch.tensor([5]),
        ["utt-1"],
        records,
        blank_id=6,
        ignored_token_ids=(5,),
        missing_policy="error",
        radius=1,
        temperature=0.0,
    )

    log_probs = F.log_softmax(student_logits[0].float(), dim=-1)
    expected_first = -log_probs[1, 2]
    expected_second = -log_probs[3, 3]
    expected = (expected_first + expected_second) / 2.0
    assert matched == 1
    assert missing == 0
    assert events == 2
    assert torch.allclose(loss, expected)


def test_ctc_teacher_hidden_loss_matches_equal_encoder_states() -> None:
    student = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    records = {
        "utt0": {
            "encoder_out": student[0].clone(),
        }
    }

    loss, matched, missing = _ctc_teacher_hidden_loss(
        student,
        torch.tensor([3], dtype=torch.long),
        ("utt0",),
        records,
        teacher_field="encoder_out",
        time_map="nearest",
        missing_policy="error",
    )

    assert matched == 1
    assert missing == 0
    assert torch.equal(loss, torch.zeros_like(loss))


def test_ctc_teacher_hidden_loss_can_filter_to_teacher_nonblank_frames() -> None:
    teacher = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])
    student = teacher.clone()
    student[0, 0] = torch.tensor([100.0, -100.0])
    records = {
        "utt0": {
            "encoder_out": teacher[0].clone(),
            "project_blank_id": 9,
            "project_ignored_token_ids": [8],
            "topk_token_ids": torch.tensor(
                [
                    [9, 1],
                    [2, 9],
                    [9, 3],
                    [8, 9],
                ],
                dtype=torch.long,
            ),
            "topk_log_probs": torch.log(
                torch.tensor(
                    [
                        [0.9, 0.1],
                        [0.8, 0.2],
                        [0.9, 0.1],
                        [0.8, 0.2],
                    ],
                    dtype=torch.float32,
                )
            ),
        }
    }

    all_loss, _, _ = _ctc_teacher_hidden_loss(
        student,
        torch.tensor([4], dtype=torch.long),
        ("utt0",),
        records,
        teacher_field="encoder_out",
        time_map="nearest",
        missing_policy="error",
        blank_id=9,
    )
    filtered_loss, matched, missing = _ctc_teacher_hidden_loss(
        student,
        torch.tensor([4], dtype=torch.long),
        ("utt0",),
        records,
        teacher_field="encoder_out",
        time_map="nearest",
        missing_policy="error",
        blank_id=9,
        frame_filter="nonblank",
        frame_filter_neighbor_radius=0,
        frame_filter_min_nonblank_prob=0.5,
    )

    assert matched == 1
    assert missing == 0
    assert all_loss > 0
    assert torch.equal(filtered_loss, torch.zeros_like(filtered_loss))


def test_decoder_sampling_top_p_keeps_threshold_crossing_token() -> None:
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32))

    filtered = RWKVCTCModel._filter_decoder_sampling_logits(logits, top_p=0.5)

    assert torch.equal(torch.isfinite(filtered), torch.tensor([[True, True, False, False]]))


def test_decoder_sampling_top_k_still_limits_nucleus_candidates() -> None:
    logits = torch.log(torch.tensor([[0.34, 0.33, 0.20, 0.13]], dtype=torch.float32))

    filtered = RWKVCTCModel._filter_decoder_sampling_logits(logits, top_k=2, top_p=0.95)

    assert torch.equal(torch.isfinite(filtered), torch.tensor([[True, True, False, False]]))


def test_ctc_model_backward_with_gradient_checkpointing() -> None:
    torch.manual_seed(23)
    model = _ctc_model()
    model.enable_gradient_checkpointing(True)
    model.train()

    features = torch.randn(2, 10, 80)
    feature_lengths = torch.tensor([10, 9], dtype=torch.long)
    logits, logit_lengths, _ = model(features, feature_lengths)
    targets = torch.tensor([1, 2, 3, 4, 1, 2], dtype=torch.long)
    target_lengths = torch.tensor([4, 2], dtype=torch.long)
    loss = model.ctc_loss(logits, logit_lengths, targets, target_lengths)
    loss.backward()

    assert model.encoder.gradient_checkpointing is True
    assert model.ctc_head.weight.grad is not None
    assert model.encoder.blocks[0].ffn1.net[0].weight.grad is not None


def test_conv2d6_frontend_aligns_feature_dtype_with_model_dtype() -> None:
    torch.manual_seed(24)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=64,
            dim_att=64,
            dim_ff=128,
            num_layers=2,
            vocab_size=16,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="conv2d6",
        )
    ).to(dtype=torch.float64)

    features = torch.randn(2, 24, 80, dtype=torch.float32)
    feature_lengths = torch.tensor([24, 22], dtype=torch.long)
    targets = torch.tensor([1, 2, 1], dtype=torch.long)
    target_lengths = torch.tensor([2, 1], dtype=torch.long)

    logits, logit_lengths, _ = model(features, feature_lengths)
    loss = model.ctc_loss(logits, logit_lengths, targets, target_lengths)

    assert logits.dtype == torch.float64
    assert torch.equal(logit_lengths, torch.tensor([3, 2], dtype=torch.long))
    assert torch.isfinite(loss)


def test_aut_rwkv_encoder_forward_shape_and_lengths() -> None:
    torch.manual_seed(241)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=16,
            n_embd=64,
            encoder_output_dim=80,
            dim_att=64,
            dim_ff=128,
            num_layers=2,
            vocab_size=16,
            head_size=32,
            dropout=0.0,
            frontend_type="aut_rwkv",
            aut_downsample_hidden_size=8,
            aut_max_source_positions=16,
        )
    )

    features = torch.randn(2, 32, 16)
    feature_lengths = torch.tensor([32, 29], dtype=torch.long)
    logits, logit_lengths, _ = model(features, feature_lengths)

    assert logits.shape == (2, 4, 16)
    assert torch.equal(logit_lengths, aut_conv2d8_out_lengths(feature_lengths))
    assert model.ctc_head.in_features == 80


def test_aut_rwkv_joint_decoder_uses_encoder_output_projection_dim() -> None:
    torch.manual_seed(242)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=16,
            n_embd=64,
            encoder_output_dim=80,
            dim_att=64,
            dim_ff=128,
            num_layers=2,
            vocab_size=32,
            blank_id=32,
            head_size=32,
            dropout=0.0,
            frontend_type="aut_rwkv",
            aut_downsample_hidden_size=8,
            aut_max_source_positions=16,
            decoder_enabled=True,
            decoder_n_embd=64,
            decoder_num_layers=2,
            decoder_ffn_hidden_size=256,
            decoder_audio_conditioning="full",
            ctc_loss_weight=0.5,
            decoder_loss_weight=0.5,
        )
    )

    features = torch.randn(2, 40, 16)
    feature_lengths = torch.tensor([40, 36], dtype=torch.long)
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    target_lengths = torch.tensor([2, 2], dtype=torch.long)
    losses = model.joint_losses(features, feature_lengths, targets, target_lengths)

    assert losses["logits"].shape[-1] == 33
    assert model.decoder_prefix_proj is not None
    assert model.decoder_prefix_proj.weight.shape == (64, 80)
    assert torch.isfinite(losses["loss"])


def test_aut_rwkv_loads_qwen3_non_attention_matching_keys() -> None:
    torch.manual_seed(243)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=16,
            n_embd=64,
            encoder_output_dim=80,
            dim_att=64,
            dim_ff=128,
            num_layers=2,
            vocab_size=16,
            head_size=32,
            dropout=0.0,
            frontend_type="aut_rwkv",
            aut_downsample_hidden_size=8,
            aut_max_source_positions=16,
        )
    )
    encoder = model.encoder.aut_encoder
    source = {
        "thinker.audio_tower.conv2d1.weight": torch.full_like(encoder.conv2d1.weight, 0.25),
        "thinker.audio_tower.layers.0.fc1.weight": torch.full_like(encoder.layers[0].fc1.weight, 0.5),
        "thinker.audio_tower.layers.0.time_mixer.forward_mixer.w0": torch.zeros_like(
            encoder.layers[0].time_mixer.forward_mixer.w0
        ),
    }
    report = encoder.load_qwen3_asr_non_attention_state_dict(source)

    assert "conv2d1.weight" in report["loaded"]
    assert "layers.0.fc1.weight" in report["loaded"]
    assert "layers.0.time_mixer.forward_mixer.w0" in report["skipped"]
    assert torch.equal(encoder.conv2d1.weight, torch.full_like(encoder.conv2d1.weight, 0.25))
    assert torch.equal(encoder.layers[0].fc1.weight, torch.full_like(encoder.layers[0].fc1.weight, 0.5))


def test_qwen3_transformer_encoder_forward_shape_and_lengths() -> None:
    torch.manual_seed(246)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=16,
            n_embd=64,
            encoder_output_dim=80,
            dim_att=64,
            dim_ff=128,
            num_layers=2,
            vocab_size=16,
            head_size=32,
            dropout=0.0,
            frontend_type="qwen3_transformer",
            aut_downsample_hidden_size=8,
            aut_max_source_positions=16,
        )
    )

    features = torch.randn(2, 32, 16)
    feature_lengths = torch.tensor([32, 29], dtype=torch.long)
    logits, logit_lengths, _ = model(features, feature_lengths)

    assert logits.shape == (2, 4, 16)
    assert torch.equal(logit_lengths, aut_conv2d8_out_lengths(feature_lengths))
    assert model.ctc_head.in_features == 80


def test_qwen3_transformer_loads_attention_matching_keys() -> None:
    torch.manual_seed(247)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=16,
            n_embd=64,
            encoder_output_dim=80,
            dim_att=64,
            dim_ff=128,
            num_layers=2,
            vocab_size=16,
            head_size=32,
            dropout=0.0,
            frontend_type="qwen3_transformer",
            aut_downsample_hidden_size=8,
            aut_max_source_positions=16,
        )
    )
    encoder = model.encoder.qwen3_transformer_encoder
    source = {
        "thinker.audio_tower.conv2d1.weight": torch.full_like(encoder.conv2d1.weight, 0.25),
        "thinker.audio_tower.layers.0.fc1.weight": torch.full_like(encoder.layers[0].fc1.weight, 0.5),
        "thinker.audio_tower.layers.0.self_attn.q_proj.weight": torch.full_like(
            encoder.layers[0].self_attn.q_proj.weight,
            0.75,
        ),
        "thinker.audio_tower.layers.0.self_attn.out_proj.bias": torch.full_like(
            encoder.layers[0].self_attn.out_proj.bias,
            1.25,
        ),
    }
    report = encoder.load_qwen3_asr_state_dict(source)

    assert "conv2d1.weight" in report["loaded"]
    assert "layers.0.fc1.weight" in report["loaded"]
    assert "layers.0.self_attn.q_proj.weight" in report["loaded"]
    assert "layers.0.self_attn.out_proj.bias" in report["loaded"]
    assert torch.equal(encoder.conv2d1.weight, torch.full_like(encoder.conv2d1.weight, 0.25))
    assert torch.equal(encoder.layers[0].fc1.weight, torch.full_like(encoder.layers[0].fc1.weight, 0.5))
    assert torch.equal(
        encoder.layers[0].self_attn.q_proj.weight,
        torch.full_like(encoder.layers[0].self_attn.q_proj.weight, 0.75),
    )


def test_sensevoice_rwkv_encoder_forward_shape_and_lengths() -> None:
    torch.manual_seed(244)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=560,
            n_embd=64,
            encoder_output_dim=64,
            dim_att=64,
            dim_ff=128,
            num_layers=3,
            vocab_size=16,
            head_size=32,
            dropout=0.0,
            frontend_type="sensevoice_rwkv",
            sensevoice_tp_blocks=1,
        )
    )

    features = torch.randn(2, 7, 560)
    feature_lengths = torch.tensor([7, 5], dtype=torch.long)
    logits, logit_lengths, state = model(features, feature_lengths)

    assert logits.shape == (2, 7, 16)
    assert torch.equal(logit_lengths, feature_lengths)
    assert len(state.block_states) == 3
    assert model.ctc_head.in_features == 64


def test_sensevoice_rwkv_loads_non_attention_matching_keys() -> None:
    torch.manual_seed(245)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=560,
            n_embd=64,
            encoder_output_dim=64,
            dim_att=64,
            dim_ff=128,
            num_layers=3,
            vocab_size=16,
            head_size=32,
            dropout=0.0,
            frontend_type="sensevoice_rwkv",
            sensevoice_tp_blocks=1,
        )
    )
    encoder = model.encoder.sensevoice_encoder
    source = {
        "audio_encoder.encoders0.0.norm1.weight": torch.full_like(encoder.layers[0].norm1.weight, 0.25),
        "audio_encoder.encoders0.0.feed_forward.w_1.weight": torch.full_like(
            encoder.layers[0].feed_forward.w_1.weight,
            0.5,
        ),
        "audio_encoder.encoders.0.norm2.bias": torch.full_like(encoder.layers[1].norm2.bias, 0.75),
        "audio_encoder.tp_encoders.0.feed_forward.w_2.bias": torch.full_like(
            encoder.layers[2].feed_forward.w_2.bias,
            1.25,
        ),
        "audio_encoder.after_norm.weight": torch.full_like(encoder.after_norm.weight, 1.5),
        "audio_encoder.tp_norm.bias": torch.full_like(encoder.tp_norm.bias, 2.0),
        "audio_encoder.encoders0.0.self_attn.linear_q_k_v.weight": torch.zeros(192, 560),
    }

    report = encoder.load_sensevoice_non_attention_state_dict(source)

    assert "layers.0.norm1.weight" in report["loaded"]
    assert "layers.0.feed_forward.w_1.weight" in report["loaded"]
    assert "layers.1.norm2.bias" in report["loaded"]
    assert "layers.2.feed_forward.w_2.bias" in report["loaded"]
    assert "after_norm.weight" in report["loaded"]
    assert "tp_norm.bias" in report["loaded"]
    assert "layers.0.time_mixer.forward_mixer.w0" in report["skipped"]
    assert torch.equal(encoder.layers[0].norm1.weight, torch.full_like(encoder.layers[0].norm1.weight, 0.25))
    assert torch.equal(
        encoder.layers[0].feed_forward.w_1.weight,
        torch.full_like(encoder.layers[0].feed_forward.w_1.weight, 0.5),
    )
    assert torch.equal(encoder.layers[1].norm2.bias, torch.full_like(encoder.layers[1].norm2.bias, 0.75))
    assert torch.equal(
        encoder.layers[2].feed_forward.w_2.bias,
        torch.full_like(encoder.layers[2].feed_forward.w_2.bias, 1.25),
    )


def test_sensevoice_rwkv_maps_nano_qkv_into_both_directions() -> None:
    torch.manual_seed(2451)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=10,
            n_embd=8,
            encoder_output_dim=8,
            dim_att=8,
            dim_ff=16,
            num_layers=3,
            vocab_size=16,
            head_size=4,
            dropout=0.0,
            frontend_type="sensevoice_rwkv",
            sensevoice_tp_blocks=1,
        )
    )
    encoder = model.encoder.sensevoice_encoder
    first_basis = torch.linalg.qr(torch.randn(10, 8), mode="reduced").Q.transpose(0, 1)
    first_qkv = torch.cat(
        [torch.randn(8, 8) @ first_basis for _ in range(3)],
        dim=0,
    )
    prefixes = ("encoders0.0", "encoders.0", "tp_encoders.0")
    source: dict[str, torch.Tensor] = {}
    for layer_idx, prefix in enumerate(prefixes):
        qkv = first_qkv if layer_idx == 0 else torch.randn(24, 8)
        source[f"audio_encoder.{prefix}.self_attn.linear_q_k_v.weight"] = qkv
        source[f"audio_encoder.{prefix}.self_attn.linear_out.weight"] = torch.randn(8, 8)

    report = encoder.load_sensevoice_qkv_state_dict(source)

    assert len(report["loaded"]) == 25
    assert report["first_layer_reconstruction_errors"]["q"] < 1.0e-5
    assert report["first_layer_reconstruction_errors"]["k"] < 1.0e-5
    assert report["first_layer_reconstruction_errors"]["v"] < 1.0e-5
    first_projection = encoder.layers[0].input_proj
    assert first_projection is not None
    for direction in (
        encoder.layers[0].time_mixer.forward_mixer,
        encoder.layers[0].time_mixer.backward_mixer,
    ):
        for mapped, expected in zip(
            (direction.receptance.weight, direction.key.weight, direction.value.weight),
            first_qkv.chunk(3, dim=0),
            strict=True,
        ):
            reconstructed = mapped.float() @ first_projection.weight.float()
            assert torch.allclose(reconstructed, expected, atol=2.0e-5, rtol=2.0e-5)

    second_q, second_k, second_v = source[
        "audio_encoder.encoders.0.self_attn.linear_q_k_v.weight"
    ].chunk(3, dim=0)
    for direction in (
        encoder.layers[1].time_mixer.forward_mixer,
        encoder.layers[1].time_mixer.backward_mixer,
    ):
        assert torch.equal(direction.receptance.weight, second_q)
        assert torch.equal(direction.key.weight, second_k)
        assert torch.equal(direction.value.weight, second_v)
        assert torch.equal(
            direction.output.weight,
            source["audio_encoder.encoders.0.self_attn.linear_out.weight"],
        )


def test_sensevoice_rwkv_can_norm_match_nano_qkv_directions() -> None:
    torch.manual_seed(2452)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=10,
            n_embd=8,
            encoder_output_dim=8,
            dim_att=8,
            dim_ff=16,
            num_layers=3,
            vocab_size=16,
            head_size=4,
            dropout=0.0,
            frontend_type="sensevoice_rwkv",
            sensevoice_tp_blocks=1,
        )
    )
    encoder = model.encoder.sensevoice_encoder
    prefixes = ("encoders0.0", "encoders.0", "tp_encoders.0")
    source: dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        source[f"audio_encoder.{prefix}.self_attn.linear_q_k_v.weight"] = torch.randn(24, 10 if prefix == "encoders0.0" else 8)
        source[f"audio_encoder.{prefix}.self_attn.linear_out.weight"] = torch.randn(8, 8)

    report = encoder.load_sensevoice_qkv_state_dict(
        source,
        projection_scale_mode="rwkv_norm",
    )

    assert report["projection_scale_mode"] == "rwkv_norm"
    target_rms = {
        "receptance": 0.5 / math.sqrt(3.0 * 8.0),
        "key": 0.05 / math.sqrt(3.0 * 8.0),
        "value": 0.5 / math.sqrt(3.0 * 8.0),
    }
    for layer in encoder.layers:
        for direction in (layer.time_mixer.forward_mixer, layer.time_mixer.backward_mixer):
            for name, expected_rms in target_rms.items():
                actual_rms = getattr(direction, name).weight.float().square().mean().sqrt().item()
                assert actual_rms == pytest.approx(expected_rms, rel=1.0e-5)

    second_q = source["audio_encoder.encoders.0.self_attn.linear_q_k_v.weight"].chunk(3, dim=0)[0]
    mapped_q = encoder.layers[1].time_mixer.forward_mixer.receptance.weight
    assert torch.nn.functional.cosine_similarity(
        mapped_q.flatten(),
        second_q.flatten(),
        dim=0,
    ).item() == pytest.approx(1.0, abs=1.0e-6)
    assert torch.equal(
        encoder.layers[1].time_mixer.forward_mixer.output.weight,
        source["audio_encoder.encoders.0.self_attn.linear_out.weight"],
    )


def test_sensevoice_conformer_conv_encoder_forward_shape_and_lengths() -> None:
    torch.manual_seed(246)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=560,
            n_embd=64,
            encoder_output_dim=64,
            dim_att=64,
            dim_ff=128,
            num_layers=3,
            vocab_size=16,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="sensevoice_conformer_conv",
            sensevoice_tp_blocks=1,
        )
    )

    features = torch.randn(2, 7, 560)
    feature_lengths = torch.tensor([7, 5], dtype=torch.long)
    logits, logit_lengths, state = model(features, feature_lengths)

    assert logits.shape == (2, 7, 16)
    assert torch.equal(logit_lengths, feature_lengths)
    assert len(state.block_states) == 3
    assert model.ctc_head.in_features == 64
    assert hasattr(model.encoder, "sensevoice_conformer_encoder")
    assert not any("self_attn" in name or "time_mixer" in name for name, _ in model.encoder.named_modules())


def test_sensevoice_conformer_conv_loads_non_attention_matching_keys() -> None:
    torch.manual_seed(247)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=560,
            n_embd=64,
            encoder_output_dim=64,
            dim_att=64,
            dim_ff=128,
            num_layers=3,
            vocab_size=16,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="sensevoice_conformer_conv",
            sensevoice_tp_blocks=1,
        )
    )
    encoder = model.encoder.sensevoice_conformer_encoder
    conv_before = encoder.layers[0].conv.depthwise.weight.detach().clone()
    source = {
        "audio_encoder.encoders0.0.norm1.weight": torch.full_like(encoder.layers[0].norm1.weight, 0.25),
        "audio_encoder.encoders0.0.feed_forward.w_1.weight": torch.full_like(
            encoder.layers[0].feed_forward.w_1.weight,
            0.5,
        ),
        "audio_encoder.encoders.0.norm2.bias": torch.full_like(encoder.layers[1].norm2.bias, 0.75),
        "audio_encoder.tp_encoders.0.feed_forward.w_2.bias": torch.full_like(
            encoder.layers[2].feed_forward.w_2.bias,
            1.25,
        ),
        "audio_encoder.after_norm.weight": torch.full_like(encoder.after_norm.weight, 1.5),
        "audio_encoder.tp_norm.bias": torch.full_like(encoder.tp_norm.bias, 2.0),
        "audio_encoder.encoders0.0.self_attn.linear_q_k_v.weight": torch.zeros(192, 560),
    }

    report = encoder.load_sensevoice_non_attention_state_dict(source)

    assert "layers.0.norm1.weight" in report["loaded"]
    assert "layers.0.feed_forward.w_1.weight" in report["loaded"]
    assert "layers.1.norm2.bias" in report["loaded"]
    assert "layers.2.feed_forward.w_2.bias" in report["loaded"]
    assert "after_norm.weight" in report["loaded"]
    assert "tp_norm.bias" in report["loaded"]
    assert "layers.0.conv.depthwise.weight" in report["skipped"]
    assert torch.equal(encoder.layers[0].norm1.weight, torch.full_like(encoder.layers[0].norm1.weight, 0.25))
    assert torch.equal(
        encoder.layers[0].feed_forward.w_1.weight,
        torch.full_like(encoder.layers[0].feed_forward.w_1.weight, 0.5),
    )
    assert torch.equal(encoder.layers[1].norm2.bias, torch.full_like(encoder.layers[1].norm2.bias, 0.75))
    assert torch.equal(
        encoder.layers[2].feed_forward.w_2.bias,
        torch.full_like(encoder.layers[2].feed_forward.w_2.bias, 1.25),
    )
    assert torch.equal(encoder.layers[0].conv.depthwise.weight, conv_before)


def test_funasr_nano_encoder_forward_shape_and_lengths() -> None:
    torch.manual_seed(248)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=16,
            n_embd=32,
            encoder_output_dim=32,
            dim_att=32,
            dim_ff=64,
            num_layers=3,
            vocab_size=16,
            head_size=32,
            dropout=0.0,
            frontend_type="funasr_nano_encoder",
            sensevoice_tp_blocks=1,
        )
    )

    features = torch.randn(2, 7, 16)
    feature_lengths = torch.tensor([7, 5], dtype=torch.long)
    logits, logit_lengths, state = model(features, feature_lengths)

    assert logits.shape == (2, 7, 16)
    assert torch.equal(logit_lengths, feature_lengths)
    assert len(state.block_states) == 3
    assert model.ctc_head.in_features == 32


def test_funasr_nano_encoder_and_ctc_init_loads_matching_keys(tmp_path) -> None:
    torch.manual_seed(249)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=16,
            n_embd=32,
            encoder_output_dim=32,
            dim_att=32,
            dim_ff=64,
            num_layers=3,
            vocab_size=21,
            blank_id=21,
            head_size=32,
            dropout=0.0,
            frontend_type="funasr_nano_encoder",
            sensevoice_tp_blocks=1,
            ctc_decoder_type="funasr_nano_transformer",
            ctc_decoder_dim=16,
            ctc_decoder_ffn_dim=32,
            ctc_decoder_num_layers=1,
            ctc_decoder_attention_heads=4,
        )
    )
    assert model.ctc_decoder is not None
    encoder = model.encoder.funasr_nano_encoder
    state = {
        f"audio_encoder.{key}": torch.randn_like(value)
        for key, value in encoder.audio_encoder.state_dict().items()
    }
    state.update(
        {
            f"ctc_decoder.{key}": torch.randn_like(value)
            for key, value in model.ctc_decoder.state_dict().items()
        }
    )
    teacher_weight = torch.randn(21, model.ctc_head.in_features)
    teacher_bias = torch.randn(21)
    state["ctc.ctc_lo.weight"] = teacher_weight
    state["ctc.ctc_lo.bias"] = teacher_bias
    checkpoint_path = tmp_path / "nano_full_ctc.pt"
    torch.save(state, checkpoint_path)

    report = model.load_funasr_nano_ctc_checkpoint(
        str(checkpoint_path),
        load_encoder=True,
        teacher_blank_id=20,
        project_ignored_token_ids=(20,),
    )

    assert report["encoder_loaded"] == len(encoder.audio_encoder.state_dict())
    assert report["ctc_decoder_loaded"] == len(model.ctc_decoder.state_dict())
    assert report["ctc_head_loaded_rows"] == 21
    assert torch.equal(
        encoder.audio_encoder.encoders0[0].norm1.weight,
        state["audio_encoder.encoders0.0.norm1.weight"],
    )
    assert torch.equal(model.ctc_head.weight[21], teacher_weight[20])


def test_funasr_nano_encoder_init_can_skip_attention_weights(tmp_path) -> None:
    torch.manual_seed(250)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=16,
            n_embd=32,
            encoder_output_dim=32,
            dim_att=32,
            dim_ff=64,
            num_layers=3,
            vocab_size=21,
            blank_id=21,
            head_size=32,
            dropout=0.0,
            frontend_type="funasr_nano_encoder",
            sensevoice_tp_blocks=1,
            ctc_decoder_type="funasr_nano_transformer",
            ctc_decoder_dim=16,
            ctc_decoder_ffn_dim=32,
            ctc_decoder_num_layers=1,
            ctc_decoder_attention_heads=4,
        )
    )
    assert model.ctc_decoder is not None
    encoder = model.encoder.funasr_nano_encoder
    attention_before = encoder.audio_encoder.encoders0[0].self_attn.linear_q_k_v.weight.detach().clone()
    state = {
        f"audio_encoder.{key}": torch.randn_like(value)
        for key, value in encoder.audio_encoder.state_dict().items()
    }
    state.update(
        {
            f"ctc_decoder.{key}": torch.randn_like(value)
            for key, value in model.ctc_decoder.state_dict().items()
        }
    )
    teacher_weight = torch.randn(21, model.ctc_head.in_features)
    teacher_bias = torch.randn(21)
    state["ctc.ctc_lo.weight"] = teacher_weight
    state["ctc.ctc_lo.bias"] = teacher_bias
    checkpoint_path = tmp_path / "nano_partial_encoder_ctc.pt"
    torch.save(state, checkpoint_path)

    report = model.load_funasr_nano_ctc_checkpoint(
        str(checkpoint_path),
        load_encoder=True,
        load_encoder_attention=False,
        teacher_blank_id=20,
        project_ignored_token_ids=(20,),
    )

    assert report["encoder_attention_skipped"] > 0
    assert report["encoder_loaded"] + report["encoder_attention_skipped"] == len(encoder.audio_encoder.state_dict())
    assert torch.equal(
        encoder.audio_encoder.encoders0[0].norm1.weight,
        state["audio_encoder.encoders0.0.norm1.weight"],
    )
    assert torch.equal(
        encoder.audio_encoder.encoders0[0].feed_forward.w_1.weight,
        state["audio_encoder.encoders0.0.feed_forward.w_1.weight"],
    )
    assert torch.equal(encoder.audio_encoder.encoders0[0].self_attn.linear_q_k_v.weight, attention_before)
    assert torch.equal(model.ctc_head.weight[21], teacher_weight[20])


def test_joint_ctc_rwkv_decoder_loss_path_uses_extra_blank_class() -> None:
    torch.manual_seed(25)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=64,
            dim_att=64,
            dim_ff=128,
            num_layers=2,
            vocab_size=32,
            blank_id=32,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
            decoder_enabled=True,
            decoder_n_embd=64,
            decoder_num_layers=2,
            decoder_ffn_hidden_size=256,
            decoder_audio_conditioning="full",
            decoder_prefix_tokens=8,
            ctc_loss_weight=0.5,
            decoder_loss_weight=0.5,
        )
    )

    features = torch.randn(2, 12, 80)
    feature_lengths = torch.tensor([12, 9], dtype=torch.long)
    targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    target_lengths = torch.tensor([4, 2], dtype=torch.long)
    losses = model.joint_losses(features, feature_lengths, targets, target_lengths)

    assert losses["logits"].shape == (2, 12, 33)
    assert torch.equal(losses["logit_lengths"], feature_lengths)
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["ctc_loss"])
    assert torch.isfinite(losses["decoder_loss"])


def test_rwkv_decoder_template_labels_mask_prompt_and_audio_positions() -> None:
    torch.manual_seed(26)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=64,
            dim_att=64,
            dim_ff=128,
            num_layers=2,
            vocab_size=32,
            blank_id=32,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
            decoder_enabled=True,
            decoder_n_embd=64,
            decoder_num_layers=2,
            decoder_ffn_hidden_size=256,
            decoder_audio_conditioning="full",
            decoder_prefix_tokens=3,
            decoder_prompt_before_audio_token_ids=(10, 11),
            decoder_prompt_after_audio_token_ids=(12,),
            decoder_target_suffix_token_ids=(13,),
            decoder_eos_token_id=0,
            ctc_loss_weight=0.5,
            decoder_loss_weight=0.5,
        )
    )

    encoded = torch.randn(1, 7, 64)
    encoded_lengths = torch.tensor([7], dtype=torch.long)
    context, context_lengths = model._decoder_template_context_embeds(encoded, encoded_lengths)  # noqa: SLF001
    logits, labels = model._decoder_template_logits_and_labels(  # noqa: SLF001
        encoded,
        encoded_lengths,
        [[7, 8, 13, 0]],
    )

    context_len = 2 + 7 + 1
    assert context.shape == (1, context_len, 64)
    assert torch.equal(context_lengths, torch.tensor([context_len], dtype=torch.long))
    assert logits.shape[:2] == labels.shape
    assert torch.equal(labels[0, :context_len], torch.full((context_len,), -100, dtype=torch.long))
    assert torch.equal(labels[0, context_len : context_len + 4], torch.tensor([7, 8, 13, 0]))

    loss = model.decoder_ar_loss(
        encoded,
        encoded_lengths,
        torch.tensor([7, 8, 0], dtype=torch.long),
        torch.tensor([3], dtype=torch.long),
    )
    assert torch.isfinite(loss)
