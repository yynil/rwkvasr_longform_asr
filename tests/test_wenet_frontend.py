import torch

from rwkvasr.data import (
    Qwen3ASRFeatureExtractor,
    SenseVoiceLFRFbankFeatureExtractor,
    WenetFbankFeatureExtractor,
    apply_lfr_stacking,
    build_audio_feature_extractor,
)
from rwkvasr.modules import (
    Qwen3ASRFeatureConfig,
    RWKVConformerEncoder,
    RWKVConformerEncoderConfig,
    WenetConv2dSubsampling6,
    WenetFbankConfig,
    compute_qwen3_asr_log_mel,
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


def test_qwen3_asr_feature_extractor_matches_whisper_log_mel_layout() -> None:
    sample_rate = 16000
    waveform = torch.randn(sample_rate)
    config = Qwen3ASRFeatureConfig()

    feats = compute_qwen3_asr_log_mel(waveform, sample_rate, config)

    import whisper.audio as whisper_audio

    expected = whisper_audio.log_mel_spectrogram(waveform, n_mels=128).transpose(0, 1)
    assert feats.shape == expected.shape
    assert torch.allclose(feats, expected, atol=1e-6)


def test_qwen3_asr_feature_extractor_resamples_and_returns_128_bins() -> None:
    extractor = Qwen3ASRFeatureExtractor()
    waveform = torch.randn(1, 8000)
    feats = extractor(waveform, sample_rate=8000)

    assert feats.dim() == 2
    assert feats.size(1) == 128


def test_build_audio_feature_extractor_selects_qwen3_asr() -> None:
    extractor = build_audio_feature_extractor("qwen3_asr", input_dim=128)

    assert isinstance(extractor, Qwen3ASRFeatureExtractor)


def test_lfr_stacking_uses_last_frame_padding() -> None:
    features = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)
    stacked = apply_lfr_stacking(features, lfr_m=7, lfr_n=6)

    assert stacked.shape == (2, 14)
    assert torch.equal(stacked[0, :4], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(stacked[1, -4:], torch.tensor([18.0, 19.0, 18.0, 19.0]))


def test_build_audio_feature_extractor_selects_sensevoice_lfr_fbank() -> None:
    extractor = build_audio_feature_extractor("sensevoice_lfr_fbank", input_dim=560)

    assert isinstance(extractor, SenseVoiceLFRFbankFeatureExtractor)


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
