import torch

from rwkvasr.modules import (
    RWKVCTCModel,
    RWKVCTCModelConfig,
    RWKVConformerEncoder,
    RWKVConformerEncoderConfig,
    build_inference_direction_mask,
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
