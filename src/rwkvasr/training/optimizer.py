from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RWKVOptimizerConfig:
    lr: float = 4e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8


def build_rwkv_param_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> list[dict]:
    decay_names: list[str] = []
    lr_1x_names: list[str] = []
    lr_2x_names: list[str] = []
    param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}

    for name, param in param_dict.items():
        if name.endswith(".w0"):
            lr_2x_names.append(name)
        elif len(param.squeeze().shape) >= 2 and name.endswith(".weight") and weight_decay > 0:
            decay_names.append(name)
        else:
            lr_1x_names.append(name)

    groups = [
        {
            "name": "rwkv_1x",
            "params": [param_dict[name] for name in sorted(lr_1x_names)],
            "param_names": sorted(lr_1x_names),
            "weight_decay": 0.0,
            "lr_scale": 1.0,
            "lr": lr,
        },
        {
            "name": "rwkv_2x",
            "params": [param_dict[name] for name in sorted(lr_2x_names)],
            "param_names": sorted(lr_2x_names),
            "weight_decay": 0.0,
            "lr_scale": 2.0,
            "lr": lr * 2.0,
        },
    ]

    if weight_decay > 0:
        groups.append(
            {
                "name": "rwkv_decay",
                "params": [param_dict[name] for name in sorted(decay_names)],
                "param_names": sorted(decay_names),
                "weight_decay": weight_decay,
                "lr_scale": 1.0,
                "lr": lr,
            }
        )

    return groups


def build_rwkv_optimizer(
    model: nn.Module,
    config: RWKVOptimizerConfig,
):
    groups = build_rwkv_param_groups(model, lr=config.lr, weight_decay=config.weight_decay)
    return torch.optim.AdamW(
        groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
