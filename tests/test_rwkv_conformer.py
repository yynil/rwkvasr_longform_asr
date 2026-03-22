import torch

from rwkvasr.modules import LayerDirectionMask, RWKVConformerBlock, RWKVConformerBlockConfig


def _build_block(layer_id: int) -> RWKVConformerBlock:
    return RWKVConformerBlock(
        RWKVConformerBlockConfig(
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            n_layer=4,
            layer_id=layer_id,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
        )
    )


def test_conformer_block_forward_shape() -> None:
    torch.manual_seed(10)
    block = _build_block(layer_id=0)
    x = torch.randn(2, 8, 128)

    y, v_first, state = block(x)

    assert y.shape == x.shape
    assert v_first.forward is not None
    assert v_first.backward is not None
    assert state.time_mixer is not None
    assert state.conv_cache is not None


def test_conformer_block_l2r_chunk_consistency() -> None:
    torch.manual_seed(11)
    block = _build_block(layer_id=0)
    x = torch.randn(2, 9, 128)
    l2r = LayerDirectionMask(use_forward=True, use_backward=False)

    full_y, _, _ = block(x, layer_mask=l2r)

    state = None
    outputs = []
    for chunk in [x[:, :4], x[:, 4:7], x[:, 7:]]:
        y, _, state = block(chunk, state=state, layer_mask=l2r)
        outputs.append(y)

    chunk_y = torch.cat(outputs, dim=1)
    assert torch.allclose(full_y, chunk_y, atol=1e-5, rtol=1e-5)
