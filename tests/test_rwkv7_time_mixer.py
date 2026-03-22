import torch

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
)
from rwkvasr.modules.rwkv7_time_mixer import pad_to_chunk_length


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
