from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Qwen3ASRFeatureConfig:
    sample_rate: int = 16000
    feature_size: int = 128
    n_fft: int = 400
    hop_length: int = 160
    chunk_length: int = 30
    dither: float = 0.0


def compute_qwen3_asr_log_mel(
    waveform: Tensor,
    sample_rate: int,
    config: Qwen3ASRFeatureConfig | None = None,
) -> Tensor:
    """Compute Qwen3-ASR-compatible Whisper log-mel features as [frames, bins]."""
    config = config or Qwen3ASRFeatureConfig()
    if int(sample_rate) != int(config.sample_rate):
        raise ValueError(
            f"Qwen3-ASR log-mel expects {config.sample_rate} Hz audio, got {sample_rate} Hz."
        )
    if int(config.n_fft) != 400 or int(config.hop_length) != 160:
        raise ValueError("Qwen3-ASR feature extraction currently supports n_fft=400 and hop_length=160.")
    if float(config.dither) != 0.0:
        raise ValueError("Qwen3-ASR feature extraction currently supports only dither=0.0.")
    if waveform.dim() == 2:
        if waveform.size(0) == 1:
            waveform = waveform.squeeze(0)
        else:
            waveform = waveform.mean(dim=0)
    if waveform.dim() != 1:
        raise ValueError(f"Expected mono waveform with shape [time] or [channels, time], got {tuple(waveform.shape)}.")

    import whisper.audio as whisper_audio

    features = whisper_audio.log_mel_spectrogram(waveform.float(), n_mels=int(config.feature_size))
    return features.transpose(0, 1).contiguous().float()
