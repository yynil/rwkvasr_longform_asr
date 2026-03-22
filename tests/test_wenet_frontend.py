import torch

from rwkvasr.data import WenetFbankFeatureExtractor
from rwkvasr.modules import (
    RWKVConformerEncoder,
    RWKVConformerEncoderConfig,
    WenetConv2dSubsampling6,
    WenetFbankConfig,
    compute_wenet_fbank,
    conv2d6_out_lengths,
)


def test_wenet_fbank_shape() -> None:
    sample_rate = 16000
    waveform = torch.randn(1, sample_rate)
    feats = compute_wenet_fbank(waveform, sample_rate, WenetFbankConfig())

    assert feats.dim() == 2
    assert feats.size(1) == 80


def test_wenet_fbank_extractor_resamples_and_returns_80_bins() -> None:
    extractor = WenetFbankFeatureExtractor()
    waveform = torch.randn(1, 8000)
    feats = extractor(waveform, sample_rate=8000)

    assert feats.dim() == 2
    assert feats.size(1) == 80


def test_conv2d6_subsampling_lengths_match_output_time() -> None:
    subsampling = WenetConv2dSubsampling6(80, 128)
    x = torch.randn(2, 120, 80)
    lengths = torch.tensor([120, 95], dtype=torch.long)
    y, out_lengths = subsampling(x, lengths)

    assert y.size(1) == int(out_lengths.max().item())
    assert torch.equal(out_lengths, conv2d6_out_lengths(lengths))


def test_encoder_accepts_wenet_conv2d6_frontend() -> None:
    encoder = RWKVConformerEncoder(
        RWKVConformerEncoderConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="conv2d6",
        )
    )
    x = torch.randn(2, 120, 80)
    lengths = torch.tensor([120, 95], dtype=torch.long)
    y, out_lengths, _ = encoder(x, lengths)

    assert y.shape[0] == 2
    assert y.shape[2] == 128
    assert torch.equal(out_lengths, conv2d6_out_lengths(lengths))
