from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio.compliance.kaldi as kaldi
from torch import Tensor, nn


@dataclass(frozen=True)
class WenetFbankConfig:
    sample_rate: int = 16000
    num_mel_bins: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    dither: float = 1.0
    window_type: str = "povey"


def compute_wenet_fbank(
    waveform: Tensor,
    sample_rate: int,
    config: WenetFbankConfig | None = None,
) -> Tensor:
    config = config or WenetFbankConfig()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform * (1 << 15)
    return kaldi.fbank(
        waveform,
        num_mel_bins=config.num_mel_bins,
        frame_length=config.frame_length,
        frame_shift=config.frame_shift,
        dither=config.dither,
        energy_floor=0.0,
        sample_frequency=sample_rate,
        window_type=config.window_type,
    )


def _load_json_cmvn(path: str | Path) -> tuple[Tensor, Tensor]:
    with open(path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    means = stats["mean_stat"]
    variances = stats["var_stat"]
    count = stats["frame_num"]
    for idx in range(len(means)):
        means[idx] /= count
        variances[idx] = variances[idx] / count - means[idx] * means[idx]
        variances[idx] = max(variances[idx], 1.0e-20)
        variances[idx] = 1.0 / math.sqrt(variances[idx])
    return torch.tensor(means, dtype=torch.float32), torch.tensor(variances, dtype=torch.float32)


def load_wenet_cmvn(path: str | Path, *, is_json: bool = True) -> tuple[Tensor, Tensor]:
    if not is_json:
        raise NotImplementedError("Only WeNet JSON CMVN is supported in the lightweight extraction.")
    return _load_json_cmvn(path)


class GlobalCMVN(nn.Module):
    def __init__(self, mean: Tensor, istd: Tensor, norm_var: bool = True):
        super().__init__()
        if mean.shape != istd.shape:
            raise ValueError("mean and istd must have the same shape")
        self.norm_var = norm_var
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: Tensor) -> Tensor:
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


def conv2d6_out_lengths(lengths: Tensor) -> Tensor:
    lengths = torch.div(lengths - 1, 2, rounding_mode="floor")
    lengths = torch.div(lengths - 2, 3, rounding_mode="floor")
    return torch.clamp(lengths, min=0)


class WenetConv2dSubsampling6(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, output_dim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, 5, 3),
            nn.ReLU(),
        )
        self.linear = nn.Linear(output_dim * (((input_dim - 1) // 2 - 2) // 3), output_dim)
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(self, x: Tensor, lengths: Tensor | None) -> tuple[Tensor, Tensor | None]:
        conv_dtype = self.conv[0].weight.dtype
        if x.dtype != conv_dtype:
            x = x.to(dtype=conv_dtype)
        x = x.unsqueeze(1)
        x = self.conv(x)
        batch_size, channels, time_steps, feat_dim = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(batch_size, time_steps, channels * feat_dim))
        if lengths is not None:
            lengths = conv2d6_out_lengths(lengths)
        return x, lengths
