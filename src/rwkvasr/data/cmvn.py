from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from .manifest import ASRManifestDataset
from .webdataset import WebDatasetASRIterableDataset, WebDatasetConfig
from .webdataset_index import (
    StableHashSplitConfig,
    index_split_sample_count,
    load_webdataset_index,
    resolve_webdataset_index_path,
    shard_in_split,
    validate_webdataset_index,
)


@dataclass(frozen=True)
class GlobalCMVNStats:
    mean_stat: Tensor
    var_stat: Tensor
    frame_num: int

    def to_wenet_json(self) -> dict[str, object]:
        return {
            "mean_stat": self.mean_stat.tolist(),
            "var_stat": self.var_stat.tolist(),
            "frame_num": int(self.frame_num),
        }


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _format_eta(elapsed_seconds: float, completed: int, total: int | None) -> str:
    if total is None or completed <= 0 or completed >= total:
        return "n/a"
    rate = completed / max(elapsed_seconds, 1.0e-6)
    remaining = (total - completed) / max(rate, 1.0e-6)
    if remaining < 60.0:
        return f"{remaining:.0f}s"
    if remaining < 3600.0:
        return f"{remaining / 60.0:.1f}m"
    return f"{remaining / 3600.0:.1f}h"


def _log_cmvn_progress(
    *,
    label: str,
    sample_count: int,
    frame_num: int,
    elapsed_seconds: float,
    total_samples: int | None = None,
    force: bool = False,
) -> None:
    if sample_count <= 0 and not force:
        return
    message = (
        f"{label}: samples={sample_count}"
        f"{'' if total_samples is None else f'/{total_samples}'}"
        f", frames={frame_num}, elapsed={elapsed_seconds:.1f}s"
    )
    if total_samples is not None and total_samples > 0:
        message += f", progress={100.0 * sample_count / total_samples:.2f}%"
        message += f", eta={_format_eta(elapsed_seconds, sample_count, total_samples)}"
    _log(message)


def accumulate_global_cmvn_stats(manifest_path: str | Path) -> GlobalCMVNStats:
    dataset = ASRManifestDataset(manifest_path)
    return accumulate_global_cmvn_stats_from_samples(
        dataset,
        total_samples=len(dataset),
        progress_label=f"CMVN(manifest={Path(manifest_path).name})",
    )


def accumulate_global_cmvn_stats_from_samples(
    samples: Iterable[dict[str, Any]],
    *,
    total_samples: int | None = None,
    progress_label: str = "CMVN",
    progress_every_samples: int = 5000,
) -> GlobalCMVNStats:
    mean_stat: Tensor | None = None
    var_stat: Tensor | None = None
    frame_num = 0
    sample_count = 0
    start_time = time.monotonic()
    _log(
        f"Starting {progress_label}"
        f"{'' if total_samples is None else f' for {total_samples} samples'}."
    )

    for sample in samples:
        sample_count += 1
        features = sample["features"]
        if not isinstance(features, Tensor) or features.dim() != 2:
            raise TypeError("CMVN expects 2-D feature tensors shaped as [T, F].")
        features = features.to(dtype=torch.float64)
        batch_frames = int(features.size(0))
        batch_mean_stat = features.sum(dim=0)
        batch_var_stat = features.square().sum(dim=0)
        mean_stat = batch_mean_stat if mean_stat is None else mean_stat + batch_mean_stat
        var_stat = batch_var_stat if var_stat is None else var_stat + batch_var_stat
        frame_num += batch_frames
        if progress_every_samples > 0 and sample_count % progress_every_samples == 0:
            _log_cmvn_progress(
                label=progress_label,
                sample_count=sample_count,
                frame_num=frame_num,
                elapsed_seconds=time.monotonic() - start_time,
                total_samples=total_samples,
            )

    if mean_stat is None or var_stat is None or frame_num <= 0:
        raise ValueError("Cannot compute CMVN statistics from an empty manifest.")

    _log_cmvn_progress(
        label=f"Finished {progress_label}",
        sample_count=sample_count,
        frame_num=frame_num,
        elapsed_seconds=time.monotonic() - start_time,
        total_samples=total_samples,
        force=True,
    )
    return GlobalCMVNStats(
        mean_stat=mean_stat.to(dtype=torch.float32),
        var_stat=var_stat.to(dtype=torch.float32),
        frame_num=frame_num,
    )


def _resolve_webdataset_progress_metadata(
    dataset: WebDatasetASRIterableDataset,
) -> tuple[int | None, int]:
    total_shards = len(dataset.shards)
    if dataset.config.split != "all" and dataset.split_config.split_by == "shard_name":
        total_shards = sum(
            int(shard_in_split(shard.name, dataset.config.split, dataset.split_config))
            for shard in dataset.shards
        )

    index_path = resolve_webdataset_index_path(dataset.shard_root)
    if not index_path.exists():
        return None, total_shards
    try:
        index_data = load_webdataset_index(index_path)
        validate_webdataset_index(index_data, split_config=dataset.split_config)
        total_samples = index_split_sample_count(index_data, dataset.config.split)
        return total_samples, total_shards
    except Exception:
        return None, total_shards


def accumulate_webdataset_global_cmvn_stats(
    shard_root: str | Path,
    *,
    config: WebDatasetConfig | None = None,
) -> GlobalCMVNStats:
    dataset = WebDatasetASRIterableDataset(
        shard_root,
        config=config or WebDatasetConfig(shuffle_shards=False),
    )
    if dataset.config.partition_by_rank:
        dataset.config = WebDatasetConfig(**{**dataset.config.__dict__, "partition_by_rank": False})
        dataset.split_config = StableHashSplitConfig(
            eval_ratio=dataset.config.eval_ratio,
            hash_seed=dataset.config.hash_seed,
            utt_id_key=dataset.config.utt_id_key,
            split_by=dataset.config.split_by,
        )
    total_samples, total_shards = _resolve_webdataset_progress_metadata(dataset)
    _log(
        f"WebDataset CMVN will scan {total_shards} shard(s)"
        f"{'' if total_samples is None else f' / {total_samples} samples'}."
    )
    return accumulate_global_cmvn_stats_from_samples(
        dataset,
        total_samples=total_samples,
        progress_label=f"CMVN(webdataset={Path(shard_root).name})",
    )


def write_wenet_cmvn_json(path: str | Path, stats: GlobalCMVNStats) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats.to_wenet_json(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def compute_manifest_global_cmvn(
    manifest_path: str | Path,
    output_path: str | Path,
) -> GlobalCMVNStats:
    stats = accumulate_global_cmvn_stats(manifest_path)
    write_wenet_cmvn_json(output_path, stats)
    return stats


def compute_webdataset_global_cmvn(
    shard_root: str | Path,
    output_path: str | Path,
    *,
    config: WebDatasetConfig | None = None,
) -> GlobalCMVNStats:
    stats = accumulate_webdataset_global_cmvn_stats(shard_root, config=config)
    write_wenet_cmvn_json(output_path, stats)
    return stats
