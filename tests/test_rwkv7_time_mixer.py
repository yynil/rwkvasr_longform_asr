import torch
import pytest

from rwkvasr.modules import (
    BidirectionalRWKVTimeMixer,
    DirectionDropoutConfig,
    DirectionDropoutScheduler,
    LayerDirectionMask,
    RWKV7TimeMixer,
    RWKV7TimeMixerConfig,
    build_inference_direction_mask,
    build_last_n_bidirectional_mask,
    reverse_time,
    reverse_time_by_lengths,
)
from rwkvasr.modules.rwkv7_cuda import fused_wkv7, fused_wkv7_clampw
from rwkvasr.modules.rwkv7_time_mixer import _native_wkv7, pad_to_chunk_length


def _build_config(layer_id: int) -> RWKV7TimeMixerConfig:
    return RWKV7TimeMixerConfig(
        n_embd=128,
        dim_att=128,
        n_layer=4,
        layer_id=layer_id,
        head_size=32,
    )


def test_time_mixer_forward_shape_and_state() -> None:
    torch.manual_seed(0)
    module = RWKV7TimeMixer(_build_config(layer_id=0))
    x = torch.randn(2, 7, 128)

    y, v_first, state = module(x)

    assert y.shape == x.shape
    assert v_first.shape == x.shape
    assert state.att_state.shape == (2, 4, 32, 32)
    assert state.last_x.shape == (2, 1, 128)


def test_time_mixer_preserves_input_v_first_for_deeper_layers() -> None:
    torch.manual_seed(1)
    module = RWKV7TimeMixer(_build_config(layer_id=2))
    x = torch.randn(2, 5, 128)
    v_first = torch.randn(2, 5, 128)

    _, returned_v_first, _ = module(x, v_first=v_first)

    assert torch.allclose(returned_v_first, v_first)


def test_time_mixer_chunk_consistency() -> None:
    torch.manual_seed(2)
    module = RWKV7TimeMixer(_build_config(layer_id=0))
    x = torch.randn(2, 9, 128)

    full_y, _, _ = module(x)

    state = None
    outputs = []
    for chunk in [x[:, :4], x[:, 4:7], x[:, 7:]]:
        y, _, state = module(chunk, state=state)
        outputs.append(y)

    chunk_y = torch.cat(outputs, dim=1)
    assert torch.allclose(full_y, chunk_y, atol=1e-5, rtol=1e-5)


def test_pad_to_chunk_length() -> None:
    x = torch.randn(2, 18, 128)
    padded, pad = pad_to_chunk_length(x, chunk_len=16)

    assert padded.shape == (2, 32, 128)
    assert pad == 14


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused RWKV-7")
def test_fused_wkv7_matches_native_forward_on_cuda() -> None:
    torch.manual_seed(9)
    bsz, tsz, hidden, head_size = 2, 17, 128, 64
    n_head = hidden // head_size
    q, k, v, z, a = [
        (torch.randn(bsz, tsz, hidden, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        for _ in range(5)
    ]
    raw_w = torch.randn(bsz, tsz, hidden, device="cuda", dtype=torch.float32)
    w = (-torch.nn.functional.softplus(-raw_w) - 0.5).to(torch.bfloat16).contiguous()

    native_y, _ = _native_wkv7(
        q.view(bsz, tsz, n_head, head_size),
        w.view(bsz, tsz, n_head, head_size),
        k.view(bsz, tsz, n_head, head_size),
        v.view(bsz, tsz, n_head, head_size),
        z.view(bsz, tsz, n_head, head_size),
        a.view(bsz, tsz, n_head, head_size),
    )
    fused_y = fused_wkv7(q, w, k, v, z, a, head_size=head_size, chunk_len=16)

    assert torch.allclose(fused_y.float(), native_y.view_as(fused_y).float(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused RWKV-7")
def test_fused_clampw_matches_recurrent_forward_and_backward_on_cuda() -> None:
    torch.manual_seed(210)
    batch_size, timesteps, hidden, head_size = 1, 16, 64, 64
    shape = (batch_size, timesteps, hidden)
    base_inputs = [
        (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2).contiguous()
        for _ in range(6)
    ]
    fused_inputs = [value.detach().clone().requires_grad_() for value in base_inputs]
    reference_inputs = [value.detach().clone().requires_grad_() for value in base_inputs]

    fused_y = fused_wkv7_clampw(
        *fused_inputs,
        head_size=head_size,
        chunk_len=16,
    )

    r, raw_w, k, v, a, b = [
        value.view(batch_size, timesteps, 1, head_size) for value in reference_inputs
    ]
    state = torch.zeros(
        batch_size,
        1,
        head_size,
        head_size,
        device="cuda",
        dtype=torch.float32,
    )
    outputs = []
    w_scale = -0.6065306597
    for timestep in range(timesteps):
        rt = r[:, timestep].float()
        decay = torch.exp(w_scale * torch.sigmoid(raw_w[:, timestep].float()))
        kt = k[:, timestep].float()
        vt = v[:, timestep].float()
        at = a[:, timestep].float()
        bt = b[:, timestep].float()
        state_a = torch.einsum("bhij,bhj->bhi", state, at)
        state = state * decay.unsqueeze(-2)
        state = state + state_a.unsqueeze(-1) * bt.unsqueeze(-2)
        state = state + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        outputs.append(torch.einsum("bhij,bhj->bhi", state, rt).to(torch.bfloat16))
    reference_y = torch.stack(outputs, dim=1).reshape_as(fused_y)

    forward_relative_error = (fused_y.float() - reference_y.float()).norm() / reference_y.float().norm()
    assert forward_relative_error.item() < 1.0e-4

    output_gradient = (torch.randn_like(reference_y) * 0.1).contiguous()
    fused_y.backward(output_gradient)
    reference_y.backward(output_gradient)
    for fused_input, reference_input in zip(fused_inputs, reference_inputs, strict=True):
        assert fused_input.grad is not None
        assert reference_input.grad is not None
        assert torch.isfinite(fused_input.grad).all()
        gradient_relative_error = (
            (fused_input.grad.float() - reference_input.grad.float()).norm()
            / reference_input.grad.float().norm().clamp_min(1.0e-12)
        )
        gradient_cosine = torch.nn.functional.cosine_similarity(
            fused_input.grad.float().flatten(),
            reference_input.grad.float().flatten(),
            dim=0,
        )
        assert gradient_relative_error.item() < 1.0e-3
        assert gradient_cosine.item() > 0.999


def test_bidirectional_merge_matches_branch_outputs() -> None:
    torch.manual_seed(3)
    module = BidirectionalRWKVTimeMixer(_build_config(layer_id=0))
    x = torch.randn(2, 6, 128)

    y_bi, v_first, _ = module(x)

    y_f, vf_f, _ = module.forward_mixer(x)
    y_b_rev, vf_b_rev, _ = module.backward_mixer(reverse_time(x))
    y_b = reverse_time(y_b_rev)
    vf_b = reverse_time(vf_b_rev)

    expected = (y_f + y_b) * 0.5
    assert torch.allclose(y_bi, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v_first.forward, vf_f)
    assert torch.allclose(v_first.backward, vf_b)


def test_reverse_time_by_lengths_keeps_padding_after_valid_frames() -> None:
    x = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [99.0], [99.0]],
            [[4.0], [5.0], [99.0], [99.0], [99.0]],
        ]
    )
    lengths = torch.tensor([3, 2])

    y = reverse_time_by_lengths(x, lengths)

    expected = torch.tensor(
        [
            [[3.0], [2.0], [1.0], [0.0], [0.0]],
            [[5.0], [4.0], [0.0], [0.0], [0.0]],
        ]
    )
    assert torch.equal(y, expected)


def test_bidirectional_mixer_valid_frames_are_batch_padding_invariant() -> None:
    torch.manual_seed(12)
    module = BidirectionalRWKVTimeMixer(_build_config(layer_id=0))
    module.eval()
    sample = torch.randn(1, 4, 128)
    companion = torch.randn(1, 7, 128)
    padded_sample = torch.cat([sample, torch.full((1, 3, 128), 100.0)], dim=1)
    batch = torch.cat([padded_sample, companion], dim=0)

    y_single, _, _ = module(sample, lengths=torch.tensor([4]))
    y_batch, _, _ = module(batch, lengths=torch.tensor([4, 7]))

    assert torch.allclose(y_batch[0, :4], y_single[0], atol=1e-5, rtol=1e-5)


def test_bidirectional_l2r_mask_matches_forward_branch() -> None:
    torch.manual_seed(4)
    module = BidirectionalRWKVTimeMixer(_build_config(layer_id=1))
    x = torch.randn(2, 5, 128)

    y_masked, _, state = module(x, layer_mask=LayerDirectionMask(use_forward=True, use_backward=False))
    y_forward, _, _ = module.forward_mixer(x)

    assert torch.allclose(y_masked, y_forward, atol=1e-5, rtol=1e-5)
    assert state.forward is not None
    assert state.backward is None


def test_alt_and_last_n_masks() -> None:
    alt = build_inference_direction_mask(5, mode="alt")
    assert torch.equal(alt.forward, torch.tensor([True, False, True, False, True]))
    assert torch.equal(alt.backward, torch.tensor([False, True, False, True, False]))

    last_two = build_last_n_bidirectional_mask(5, n_bidirectional=2)
    assert torch.equal(last_two.forward, torch.tensor([True, True, True, True, True]))
    assert torch.equal(last_two.backward, torch.tensor([False, False, False, True, True]))


def test_direction_dropout_scheduler_probability_ramps() -> None:
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=4,
            variant="drop_both",
            p_start=0.0,
            p_max=0.2,
            warmup_steps=10,
            ramp_steps=20,
        )
    )

    assert scheduler.probability_at(0) == 0.0
    assert scheduler.probability_at(10) == 0.0
    assert scheduler.probability_at(20) == 0.1
    assert scheduler.probability_at(30) == 0.2
    assert scheduler.probability_at(100) == 0.2


def test_direction_dropout_zero_probability_is_full_bidirectional() -> None:
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=8,
            variant="drop_both",
            p_start=0.0,
            p_max=0.0,
            warmup_steps=0,
            ramp_steps=0,
        )
    )
    generator = torch.Generator().manual_seed(123)
    mask = scheduler.sample_mask(0, generator=generator)

    assert torch.all(mask.forward)
    assert torch.all(mask.backward)


def test_direction_dropout_default_is_disabled() -> None:
    scheduler = DirectionDropoutScheduler(DirectionDropoutConfig(num_layers=8))
    mask = scheduler.sample_mask(0, generator=torch.Generator().manual_seed(123))

    assert scheduler.config.variant == "none"
    assert scheduler.config.p_start == 0.0
    assert scheduler.config.p_max == 0.0
    assert torch.all(mask.forward)
    assert torch.all(mask.backward)


def test_direction_dropout_none_variant_is_full_bidirectional_even_with_probability() -> None:
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=8,
            variant="none",
            p_start=1.0,
            p_max=1.0,
            warmup_steps=0,
            ramp_steps=0,
        )
    )
    generator = torch.Generator().manual_seed(123)
    mask = scheduler.sample_mask(0, generator=generator)

    assert torch.all(mask.forward)
    assert torch.all(mask.backward)


def test_direction_dropout_drop_r2l_only_keeps_forward() -> None:
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=8,
            variant="drop_r2l_only",
            p_start=0.5,
            p_max=0.5,
            warmup_steps=0,
            ramp_steps=0,
        )
    )
    generator = torch.Generator().manual_seed(123)
    mask = scheduler.sample_mask(0, generator=generator)

    assert torch.all(mask.forward)
    assert torch.any(~mask.backward)


def test_direction_dropout_drop_both_never_drops_both_directions() -> None:
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=12,
            variant="drop_both",
            p_start=1.0,
            p_max=1.0,
            warmup_steps=0,
            ramp_steps=0,
        )
    )
    generator = torch.Generator().manual_seed(456)
    mask = scheduler.sample_mask(0, generator=generator)

    assert torch.all(mask.forward | mask.backward)
    assert torch.any(~mask.forward)
    assert torch.any(~mask.backward)
