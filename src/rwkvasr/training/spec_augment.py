from __future__ import annotations

import torch
from torch import Tensor


def apply_spec_augment(
    features: Tensor,
    feature_lengths: Tensor | None = None,
    *,
    time_masks: int = 2,
    time_width: int = 40,
    freq_masks: int = 2,
    freq_width: int = 15,
    fill_value: float = 0.0,
) -> Tensor:
    """Apply simple SpecAugment time/frequency masking to padded feature batches."""
    if features.ndim != 3:
        raise ValueError(f"features must be B x T x F, got shape={tuple(features.shape)}")
    if time_masks <= 0 and freq_masks <= 0:
        return features

    batch, max_time, num_features = features.shape
    augmented = features.clone()
    lengths = feature_lengths
    if lengths is None:
        lengths = torch.full((batch,), max_time, device=features.device, dtype=torch.long)
    else:
        lengths = lengths.to(device=features.device, dtype=torch.long)

    for sample_index in range(batch):
        valid_time = int(torch.clamp(lengths[sample_index], min=0, max=max_time).item())
        if valid_time <= 0:
            continue

        if time_masks > 0 and time_width > 0:
            max_width = min(int(time_width), valid_time)
            for _ in range(int(time_masks)):
                width = int(torch.randint(max_width + 1, (1,), device=features.device).item())
                if width <= 0:
                    continue
                start = int(torch.randint(valid_time - width + 1, (1,), device=features.device).item())
                augmented[sample_index, start : start + width, :] = fill_value

        if freq_masks > 0 and freq_width > 0 and num_features > 0:
            max_width = min(int(freq_width), num_features)
            for _ in range(int(freq_masks)):
                width = int(torch.randint(max_width + 1, (1,), device=features.device).item())
                if width <= 0:
                    continue
                start = int(torch.randint(num_features - width + 1, (1,), device=features.device).item())
                augmented[sample_index, :valid_time, start : start + width] = fill_value

    return augmented
