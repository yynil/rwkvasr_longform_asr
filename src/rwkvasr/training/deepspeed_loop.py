from __future__ import annotations

import os
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.data import (
    ASRManifestDataset,
    FeatureCollator,
    StableHashSplitConfig,
    WebDatasetConfig,
    build_bucketed_webdataset_loader,
    build_length_bucketed_webdataset_dataloader,
    build_webdataset_dataloader,
    estimate_bucket_manifest_steps,
    estimate_length_bucketed_steps,
    index_split_sample_count,
    load_webdataset_bucket_manifest,
    load_webdataset_index,
    load_webdataset_length_entries,
    resolve_webdataset_index_path,
    validate_webdataset_index,
)
from rwkvasr.modules import DirectionDropoutConfig, DirectionDropoutScheduler, RWKVCTCModel, RWKVCTCModelConfig

from .checkpoint import save_checkpoint
from .batch_budget import ctc_batch_token_stats, estimate_token_budget_from_memory, select_ctc_batch_prefix_by_token_budget
from .ctc_task import RWKVDualModeCTCTrainer
from .epoch_metrics import save_epoch_metrics, save_step_checkpoint_metrics
from .optimizer import build_rwkv_param_groups
from .progress import start_training_progress, update_training_progress
from .wandb_logger import finish_wandb, init_wandb_run, log_wandb
from .train_loop import (
    _resolve_bucket_manifest_path,
    _resolve_cmvn_file,
    _resolve_data_source,
    _resolve_in_memory_length_index_path,
    _resolve_text_tokenizer,
    _resolve_vocab_size,
    _resolved_tokenizer_config_payload,
)


@dataclass(frozen=True)
class DeepSpeedTrainConfig:
    output_dir: str
    deepspeed: dict[str, Any]
    vocab_size: int | None = None
    tokenizer_type: str = "whisper_multilingual"
    tokenizer_model_path: str | None = None
    tokenizer_language: str | None = None
    tokenizer_task: str | None = None
    manifest_path: str | None = None
    webdataset_root: str | None = None
    webdataset_index_path: str | None = None
    webdataset_length_index_path: str | None = None
    webdataset_bucket_manifest_path: str | None = None
    webdataset_split: str = "all"
    webdataset_eval_ratio: float = 0.0
    webdataset_hash_seed: int = 0
    webdataset_split_by: str = "shard_name"
    webdataset_utt_id_key: str = "sid"
    input_dim: int = 80
    n_embd: int = 512
    dim_att: int = 512
    dim_ff: int = 2048
    num_layers: int = 12
    head_size: int = 64
    backend: str = "native"
    conv_kernel_size: int = 31
    dropout: float = 0.1
    frontend_type: str = "conv2d6"
    cmvn_file: str | None = None
    cmvn_is_json: bool = True
    batch_size: int = 4
    max_steps: int | None = None
    epochs: int | None = None
    save_every: int = 50
    num_workers: int = 0
    decoded_batch_prefetch: int = 2
    max_open_shards_per_worker: int = 8
    lr: float = 4e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    direction_variant: str = "drop_both"
    p_start: float = 0.2
    p_max: float = 0.2
    warmup_steps: int = 0
    ramp_steps: int = 0
    device: str = "cuda"
    resume_from: str | None = None
    resume_tag: str | None = None
    wandb_enabled: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_base_url: str | None = None
    wandb_init_timeout_sec: float = 30.0
    eval_mode: str = "bi"
    max_eval_samples: int | None = None
    eval_batch_size: int | None = None
    step_eval_batch_size: int | None = None
    step_eval_every: int | None = None
    step_eval_samples: int | None = None
    top_k_step_checkpoints: int = 3
    local_rank: int = -1
    log_every: int = 10
    gradient_checkpointing: bool = True
    batch_token_budget: int | None = None
    length_bucket_frame_budget: int | None = None
    target_gpu_memory_gib: float = 22.0
    skip_oversized_samples: bool = True


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


def _is_distributed() -> bool:
    return _world_size() > 1


def _is_rank_zero() -> bool:
    return _rank() == 0


def _maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _rank_zero_log(message: str) -> None:
    if _is_rank_zero():
        print(f"[rwkvasr] {message}", flush=True)


def _normalize_deepspeed_config(config: DeepSpeedTrainConfig) -> dict[str, Any]:
    ds_config = dict(config.deepspeed)
    if "optimizer" in ds_config:
        raise ValueError("Do not set `deepspeed.optimizer`; optimizer groups are managed by this project.")
    micro_batch = int(config.batch_size)
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch
    grad_accum = int(ds_config.get("gradient_accumulation_steps", 1))
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = micro_batch * grad_accum * _world_size()
    zero_optimization = dict(ds_config.get("zero_optimization", {}))
    zero_optimization["stage"] = 2
    offload_optimizer = dict(zero_optimization.get("offload_optimizer", {}))
    offload_device = str(offload_optimizer.get("device", "")).lower().strip()
    if offload_device in {"", "none", "null"}:
        zero_optimization.pop("offload_optimizer", None)
    else:
        offload_optimizer["device"] = offload_device
        if offload_device == "cpu":
            offload_optimizer["pin_memory"] = bool(offload_optimizer.get("pin_memory", True))
        zero_optimization["offload_optimizer"] = offload_optimizer
    ds_config["zero_optimization"] = zero_optimization
    ds_config["gradient_clipping"] = float(ds_config.get("gradient_clipping", 1.0))
    use_cuda = torch.cuda.is_available() and config.device.startswith("cuda")
    if use_cuda:
        ds_config["bf16"] = {"enabled": True}
    else:
        ds_config["bf16"] = {"enabled": False}
        ds_config["fp16"] = {"enabled": False}
    return ds_config


def _optimizer_offload_device(ds_config: dict[str, Any]) -> str | None:
    zero_optimization = dict(ds_config.get("zero_optimization", {}))
    offload_optimizer = dict(zero_optimization.get("offload_optimizer", {}))
    device = str(offload_optimizer.get("device", "")).lower().strip()
    if device in {"", "none", "null"}:
        return None
    return device


def _build_deepspeed_optimizer(
    model: RWKVCTCModel,
    config: DeepSpeedTrainConfig,
    ds_config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, str]:
    param_groups = build_rwkv_param_groups(
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    offload_device = _optimizer_offload_device(ds_config)
    if offload_device == "cpu":
        optimizer = DeepSpeedCPUAdam(
            param_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            adamw_mode=True,
        )
        return optimizer, "DeepSpeedCPUAdam"

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
    return optimizer, "AdamW"


def _build_webdataset_config(
    config: DeepSpeedTrainConfig,
    *,
    shuffle_shards: bool,
) -> WebDatasetConfig:
    length_bucket_frame_budget = config.length_bucket_frame_budget
    if length_bucket_frame_budget is None:
        length_bucket_frame_budget = config.batch_token_budget
    return WebDatasetConfig(
        shuffle_shards=shuffle_shards,
        split=config.webdataset_split,
        eval_ratio=config.webdataset_eval_ratio,
        hash_seed=config.webdataset_hash_seed,
        split_by=config.webdataset_split_by,
        utt_id_key=config.webdataset_utt_id_key,
        length_index_path=config.webdataset_length_index_path,
        length_bucket_frame_budget=length_bucket_frame_budget,
        decoded_batch_prefetch=config.decoded_batch_prefetch,
        max_open_shards_per_worker=config.max_open_shards_per_worker,
    )


def _resolve_max_steps(config: DeepSpeedTrainConfig, grad_accum: int) -> tuple[int, int | None]:
    if config.max_steps is not None and config.epochs is not None:
        raise ValueError("Specify only one of max_steps or epochs.")
    if config.max_steps is not None:
        return int(config.max_steps), None
    if config.epochs is None:
        return 100, None

    data_source, data_path = _resolve_data_source(config)
    if data_source == "manifest":
        num_samples = len(ASRManifestDataset(data_path))
        global_batch = max(1, config.batch_size * grad_accum * _world_size())
        steps_per_epoch = max(1, math.ceil(num_samples / global_batch))
    else:
        index_path = resolve_webdataset_index_path(data_path, config.webdataset_index_path)
        index_data = load_webdataset_index(index_path)
        validate_webdataset_index(
            index_data,
            split_config=StableHashSplitConfig(
                eval_ratio=config.webdataset_eval_ratio,
                hash_seed=config.webdataset_hash_seed,
                split_by=config.webdataset_split_by,
                utt_id_key=config.webdataset_utt_id_key,
            ),
        )
        num_samples = index_split_sample_count(index_data, config.webdataset_split)
        bucket_manifest_path = _resolve_bucket_manifest_path(data_path, config.webdataset_bucket_manifest_path)
        if bucket_manifest_path is not None:
            manifest = load_webdataset_bucket_manifest(bucket_manifest_path)
            steps_per_epoch = estimate_bucket_manifest_steps(
                manifest,
                split=config.webdataset_split,
                batch_size=config.batch_size,
                world_size=_world_size(),
                frame_budget=config.length_bucket_frame_budget or config.batch_token_budget,
                drop_last=True,
            )
            return steps_per_epoch * int(config.epochs), steps_per_epoch
        global_batch = max(1, config.batch_size * grad_accum * _world_size())
        length_index_path = _resolve_in_memory_length_index_path(
            data_path,
            config.webdataset_length_index_path,
            logger=_rank_zero_log,
        )
        if length_index_path is not None:
            entries = load_webdataset_length_entries(length_index_path, split=config.webdataset_split)
            length_bucket_frame_budget = config.length_bucket_frame_budget
            if length_bucket_frame_budget is None:
                length_bucket_frame_budget = config.batch_token_budget
            steps_per_epoch = max(
                1,
                estimate_length_bucketed_steps(
                    [entry.num_frames for entry in entries],
                    batch_size=config.batch_size,
                    world_size=_world_size(),
                    frame_budget=length_bucket_frame_budget,
                    drop_last=True,
                ),
            )
        else:
            steps_per_epoch = max(1, math.ceil(num_samples / global_batch))
    return steps_per_epoch * int(config.epochs), steps_per_epoch


def _resolve_cmvn_file_distributed(config: DeepSpeedTrainConfig, output_dir: Path) -> str | None:
    if config.frontend_type != "conv2d6":
        return config.cmvn_file
    if config.cmvn_file is not None:
        _rank_zero_log(f"Using CMVN file: {config.cmvn_file}")
        return config.cmvn_file

    if _is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        _rank_zero_log(f"CMVN file not provided. Computing global CMVN under {output_dir}.")
        resolved = _resolve_cmvn_file(config, output_dir)
        _rank_zero_log(f"CMVN ready: {resolved}")
    else:
        resolved = str(output_dir / "global_cmvn.json")
    _maybe_barrier()
    return resolved


def _build_train_loader(config: DeepSpeedTrainConfig) -> tuple[DataLoader, Any | None]:
    data_source, data_path = _resolve_data_source(config)
    tokenizer = _resolve_text_tokenizer(config)
    if data_source == "manifest":
        dataset = ASRManifestDataset(data_path, tokenizer=tokenizer)
        sampler = None
        if _is_distributed():
            sampler = DistributedSampler(
                dataset,
                num_replicas=_world_size(),
                rank=_rank(),
                shuffle=True,
                drop_last=False,
            )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=FeatureCollator(),
        )
        return loader, sampler

    webdataset_config = _build_webdataset_config(config, shuffle_shards=True)
    bucket_manifest_path = _resolve_bucket_manifest_path(data_path, config.webdataset_bucket_manifest_path)
    if bucket_manifest_path is not None:
        loader = build_bucketed_webdataset_loader(
            data_path,
            bucket_manifest_path=bucket_manifest_path,
            tokenizer=tokenizer,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            rank=_rank(),
            world_size=_world_size(),
        )
        return loader, None
    length_index_path = _resolve_in_memory_length_index_path(
        data_path,
        config.webdataset_length_index_path,
        logger=_rank_zero_log,
    )
    if length_index_path is not None:
        loader, sampler = build_length_bucketed_webdataset_dataloader(
            data_path,
            length_index_path=length_index_path,
            tokenizer=tokenizer,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            rank=_rank(),
            world_size=_world_size(),
        )
        return loader, sampler
    loader = build_webdataset_dataloader(
        data_path,
        tokenizer=tokenizer,
        config=webdataset_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    return loader, None


def _build_eval_loader(
    config: DeepSpeedTrainConfig,
    *,
    shuffle_shards: bool = False,
    step_subset: bool = False,
) -> tuple[DataLoader | None, Any | None]:
    data_source, data_path = _resolve_data_source(config)
    eval_batch_size = _resolve_eval_batch_size(config, step_subset=step_subset)
    if data_source == "manifest":
        dataset = ASRManifestDataset(data_path, tokenizer=_resolve_text_tokenizer(config))
        sampler = None
        if _is_distributed():
            sampler = DistributedSampler(
                dataset,
                num_replicas=_world_size(),
                rank=_rank(),
                shuffle=shuffle_shards,
                drop_last=False,
            )
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=sampler is None and shuffle_shards,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=FeatureCollator(),
        )
        return loader, sampler

    tokenizer = _resolve_text_tokenizer(config)
    webdataset_config = _build_webdataset_config(config, shuffle_shards=shuffle_shards)
    webdataset_config = WebDatasetConfig(
        **{**webdataset_config.__dict__, "split": "eval"}
    )
    bucket_manifest_path = _resolve_bucket_manifest_path(data_path, config.webdataset_bucket_manifest_path)
    if bucket_manifest_path is not None:
        loader = build_bucketed_webdataset_loader(
            data_path,
            bucket_manifest_path=bucket_manifest_path,
            tokenizer=tokenizer,
            config=webdataset_config,
            batch_size=eval_batch_size,
            num_workers=config.num_workers,
            rank=_rank(),
            world_size=_world_size(),
        )
        return loader, None
    length_index_path = _resolve_in_memory_length_index_path(
        data_path,
        config.webdataset_length_index_path,
        logger=_rank_zero_log,
    )
    if length_index_path is not None:
        loader, sampler = build_length_bucketed_webdataset_dataloader(
            data_path,
            length_index_path=length_index_path,
            tokenizer=tokenizer,
            config=webdataset_config,
            batch_size=eval_batch_size,
            num_workers=config.num_workers,
            rank=_rank(),
            world_size=_world_size(),
        )
        return loader, sampler
    loader = build_webdataset_dataloader(
        data_path,
        tokenizer=tokenizer,
        config=webdataset_config,
        batch_size=eval_batch_size,
        num_workers=config.num_workers,
    )
    return loader, None


def _set_loader_epoch(loader: DataLoader, sampler: Any | None, epoch: int) -> None:
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def _all_reduce_mean(sum_value: float, count_value: int, *, device: torch.device) -> float:
    if count_value <= 0:
        return float("nan")
    tensor = torch.tensor([sum_value, float(count_value)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_sum = float(tensor[0].item())
    total_count = int(round(float(tensor[1].item())))
    if total_count <= 0:
        return float("nan")
    return total_sum / total_count


@torch.no_grad()
def _evaluate_epoch_loss(
    *,
    model: RWKVCTCModel,
    loader: DataLoader | None,
    sampler: Any | None,
    epoch: int,
    device: torch.device,
    feature_dtype: torch.dtype | None,
    mode: str,
    max_eval_samples: int | None = None,
) -> tuple[float, int]:
    if loader is None:
        return float("nan"), 0
    _set_loader_epoch(loader, sampler, epoch)
    trainer = RWKVDualModeCTCTrainer(model)
    was_training = model.training
    model.eval()
    local_loss_sum = 0.0
    local_sample_count = 0
    local_eval_limit = None
    if max_eval_samples is not None:
        world_size = _world_size()
        rank = _rank()
        base = max_eval_samples // world_size
        extra = max_eval_samples % world_size
        local_eval_limit = base + (1 if rank < extra else 0)
    for batch in loader:
        remaining = None if local_eval_limit is None else local_eval_limit - local_sample_count
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and int(batch.features.size(0)) > remaining:
            batch = batch.prefix(remaining)
        batch = batch.to(device, feature_dtype=feature_dtype)
        loss = trainer.eval_loss(batch, mode=mode)
        batch_size = int(batch.features.size(0))
        local_loss_sum += float(loss.item()) * batch_size
        local_sample_count += batch_size
    if was_training:
        model.train()
    global_sample_count = local_sample_count
    if dist.is_available() and dist.is_initialized():
        count_tensor = torch.tensor([float(local_sample_count)], device=device, dtype=torch.float64)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        global_sample_count = int(round(float(count_tensor.item())))
    return _all_reduce_mean(local_loss_sum, local_sample_count, device=device), global_sample_count


def _resolve_epoch_eval_limit(config: DeepSpeedTrainConfig, *, epoch: int) -> int | None:
    if config.max_eval_samples is None:
        return None
    if config.epochs is not None and epoch >= int(config.epochs):
        return None
    return int(config.max_eval_samples)


def _resolve_step_eval_every(config: DeepSpeedTrainConfig) -> int | None:
    if config.step_eval_samples is None or int(config.step_eval_samples) <= 0:
        return None
    if config.step_eval_every is not None:
        return max(1, int(config.step_eval_every))
    return max(1, int(config.save_every))


def _resolve_eval_batch_size(config: DeepSpeedTrainConfig, *, step_subset: bool) -> int:
    if step_subset and config.step_eval_batch_size is not None:
        return max(1, int(config.step_eval_batch_size))
    if config.eval_batch_size is not None:
        return max(1, int(config.eval_batch_size))
    return max(1, int(config.batch_size))


def _sort_step_checkpoint_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(record) for record in records],
        key=lambda record: (
            float(record.get("eval_loss", float("inf"))),
            int(record.get("step", 0)),
        ),
    )


def _prune_deepspeed_step_checkpoint_artifacts(
    *,
    top_records: list[dict[str, Any]],
    saved_records: list[dict[str, Any]],
) -> None:
    keep_export_paths = {
        str(record["checkpoint_path"])
        for record in top_records
        if record.get("checkpoint_path")
    }
    keep_ds_dirs = {
        str(record["deepspeed_checkpoint_dir"])
        for record in top_records
        if record.get("deepspeed_checkpoint_dir")
    }
    for record in saved_records:
        checkpoint_path = record.get("checkpoint_path")
        if checkpoint_path is not None and str(checkpoint_path) not in keep_export_paths:
            path = Path(str(checkpoint_path))
            if path.exists():
                path.unlink()
        ds_dir = record.get("deepspeed_checkpoint_dir")
        if ds_dir is not None and str(ds_dir) not in keep_ds_dirs:
            path = Path(str(ds_dir))
            if path.exists():
                shutil.rmtree(path)


def _init_deepspeed_runtime(config: DeepSpeedTrainConfig) -> tuple[int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", str(config.local_rank)))
    use_cuda = torch.cuda.is_available() and config.device.startswith("cuda")
    backend = "nccl" if use_cuda else "gloo"
    device = torch.device("cpu")

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/rwkvasr_torch_extensions")
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
    if not use_cuda:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    elif local_rank >= 0:
        # torchrun launches one process per GPU; bind the CUDA device before NCCL init.
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(config.device)

    if not dist.is_initialized():
        os.environ.setdefault("LOCAL_RANK", str(max(local_rank, 0)))
        try:
            if _world_size() == 1:
                init_file = Path(f"/tmp/rwkvasr_deepspeed_init_{os.getpid()}")
                deepspeed.init_distributed(
                    dist_backend=backend,
                    auto_mpi_discovery=False,
                    init_method=f"file://{init_file}",
                    rank=0,
                    world_size=1,
                )
            else:
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29500")
                deepspeed.init_distributed(dist_backend=backend, auto_mpi_discovery=False)
        except Exception as exc:
            raise RuntimeError(
                "DeepSpeed distributed initialization failed with "
                f"backend={backend!r}, RANK={os.environ.get('RANK')!r}, "
                f"WORLD_SIZE={os.environ.get('WORLD_SIZE')!r}, "
                f"LOCAL_RANK={os.environ.get('LOCAL_RANK')!r}, "
                f"MASTER_ADDR={os.environ.get('MASTER_ADDR')!r}, "
                f"MASTER_PORT={os.environ.get('MASTER_PORT')!r}."
            ) from exc

    if use_cuda:
        return local_rank, device

    return local_rank, torch.device("cpu")


def _save_export_checkpoints(
    *,
    engine: deepspeed.DeepSpeedEngine,
    output_dir: Path,
    tag: str,
    export_name: str,
    step: int,
    zero_stage: int,
    extra_state: dict[str, Any],
) -> dict[str, str | None]:
    ds_checkpoint_dir = output_dir / "ds_checkpoints" / tag
    engine.save_checkpoint(
        str(output_dir / "ds_checkpoints"),
        tag=tag,
        client_state={"step": step, **extra_state},
    )
    export_path: Path | None = None
    if _is_rank_zero() and zero_stage < 3:
        export_path = output_dir / export_name
        save_checkpoint(
            export_path,
            model=engine.module,
            step=step,
            extra=extra_state,
        )
    return {
        "checkpoint_path": str(export_path) if export_path is not None else None,
        "deepspeed_checkpoint_dir": str(ds_checkpoint_dir),
        "resume_tag": tag,
    }


def train_ctc_model_deepspeed(config: DeepSpeedTrainConfig) -> dict[str, float | int | str]:
    ds_config = _normalize_deepspeed_config(config)
    feature_dtype = torch.bfloat16 if bool(ds_config.get("bf16", {}).get("enabled")) else None
    zero_stage = int(ds_config.get("zero_optimization", {}).get("stage", 0))
    grad_accum = int(ds_config["gradient_accumulation_steps"])
    local_rank, device = _init_deepspeed_runtime(config)
    _rank_zero_log(
        f"Distributed init complete. world_size={_world_size()} local_rank={local_rank} device={device} zero_stage={zero_stage}"
    )

    output_dir = Path(config.output_dir)
    if _is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
    _maybe_barrier()

    resolved_vocab_size = _resolve_vocab_size(config)
    resolved_max_steps, steps_per_epoch = _resolve_max_steps(config, grad_accum)
    resolved_cmvn_file = _resolve_cmvn_file_distributed(config, output_dir)
    _rank_zero_log(
        f"Training config resolved. vocab_size={resolved_vocab_size} max_steps={resolved_max_steps} steps_per_epoch={steps_per_epoch}"
    )
    model_config = RWKVCTCModelConfig(
        input_dim=config.input_dim,
        n_embd=config.n_embd,
        dim_att=config.dim_att,
        dim_ff=config.dim_ff,
        num_layers=config.num_layers,
        vocab_size=resolved_vocab_size,
        head_size=config.head_size,
        backend=config.backend,
        conv_kernel_size=config.conv_kernel_size,
        dropout=config.dropout,
        frontend_type=config.frontend_type,
        cmvn_file=resolved_cmvn_file,
        cmvn_is_json=config.cmvn_is_json,
    )

    if _is_rank_zero():
        save_yaml(output_dir / "model_config.yaml", model_config)
        save_yaml(
            output_dir / "tokenizer_config.yaml",
            _resolved_tokenizer_config_payload(config, vocab_size=resolved_vocab_size),
        )
        save_yaml(
            output_dir / "train_config.yaml",
            {
                **config.__dict__,
                "vocab_size": resolved_vocab_size,
                "cmvn_file": resolved_cmvn_file,
                "local_rank": local_rank,
                "max_steps": resolved_max_steps,
                "steps_per_epoch": steps_per_epoch,
            },
        )
        save_yaml(output_dir / "deepspeed_config.yaml", ds_config)
    _maybe_barrier()
    wandb_run = None
    if _is_rank_zero():
        wandb_run = init_wandb_run(
            enabled=config.wandb_enabled,
            project=config.wandb_project,
            run_name=config.wandb_run_name,
            output_dir=output_dir,
            config=load_yaml(output_dir / "train_config.yaml"),
            base_url=config.wandb_base_url,
            init_timeout_sec=config.wandb_init_timeout_sec,
            logger=_rank_zero_log,
        )

    model = RWKVCTCModel(model_config)
    model.enable_gradient_checkpointing(config.gradient_checkpointing)
    _rank_zero_log("Model constructed. Initializing DeepSpeed engine...")
    optimizer, optimizer_name = _build_deepspeed_optimizer(model, config, ds_config)
    offload_device = _optimizer_offload_device(ds_config) or "none"
    _rank_zero_log(f"Using optimizer={optimizer_name} zero_stage={zero_stage} offload_optimizer={offload_device}")
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        dist_init_required=False,
    )
    _rank_zero_log("DeepSpeed engine initialized. Building dataloader...")

    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=config.num_layers,
            variant=config.direction_variant,  # type: ignore[arg-type]
            p_start=config.p_start,
            p_max=config.p_max,
            warmup_steps=config.warmup_steps,
            ramp_steps=config.ramp_steps,
        )
    )

    loader, sampler = _build_train_loader(config)
    eval_loader, eval_sampler = _build_eval_loader(config)
    step_eval_loader, step_eval_sampler = _build_eval_loader(config, shuffle_shards=True, step_subset=True)
    step_eval_every = _resolve_step_eval_every(config)
    _rank_zero_log("Dataloader ready. Entering training loop.")
    active_length_index_path = None
    active_bucket_manifest_path = None
    if config.webdataset_root is not None:
        active_bucket_manifest_path = _resolve_bucket_manifest_path(
            config.webdataset_root,
            config.webdataset_bucket_manifest_path,
        )
        active_length_index_path = _resolve_in_memory_length_index_path(
            config.webdataset_root,
            config.webdataset_length_index_path,
            logger=_rank_zero_log,
        )
    if active_bucket_manifest_path is not None:
        frame_budget = config.length_bucket_frame_budget
        if frame_budget is None:
            frame_budget = config.batch_token_budget
        _rank_zero_log(
            "Bucket manifest active: "
            f"path={active_bucket_manifest_path} "
            f"max_local_batch={config.batch_size} "
            f"frame_budget={frame_budget} "
            f"decode_workers={max(1, config.num_workers)} "
            f"decoded_prefetch={max(0, config.decoded_batch_prefetch)} "
            f"max_open_shards_per_worker={max(1, config.max_open_shards_per_worker)} "
            f"world_size={_world_size()}"
        )
    elif active_length_index_path is not None:
        frame_budget = config.length_bucket_frame_budget
        if frame_budget is None:
            frame_budget = config.batch_token_budget
        _rank_zero_log(
            "Length bucketing active: "
            f"max_local_batch={config.batch_size} "
            f"frame_budget={frame_budget} "
            f"world_size={_world_size()}"
        )
    _rank_zero_log("The first batch can be slower because workers start up and wav->fbank decoding is done online.")
    start_step = 0
    start_epoch = 0
    history: list[dict[str, float | int | str]] = []
    step_checkpoint_history: list[dict[str, Any]] = []
    best_step_checkpoints: list[dict[str, Any]] = []
    best_epoch = 0
    best_eval_loss = float("inf")
    best_train_loss = float("inf")
    if config.resume_from is not None:
        load_path, client_state = engine.load_checkpoint(config.resume_from, tag=config.resume_tag)
        if load_path is None:
            raise FileNotFoundError(f"Unable to load DeepSpeed checkpoint from {config.resume_from}")
        start_step = int(client_state.get("step", 0))
        start_epoch = int(client_state.get("epoch", 0))
        raw_history = client_state.get("history", [])
        if isinstance(raw_history, list):
            history = [dict(item) for item in raw_history if isinstance(item, dict)]
        raw_step_history = client_state.get("step_checkpoint_history", [])
        if isinstance(raw_step_history, list):
            step_checkpoint_history = [dict(item) for item in raw_step_history if isinstance(item, dict)]
        raw_best_steps = client_state.get("best_step_checkpoints", [])
        if isinstance(raw_best_steps, list):
            best_step_checkpoints = _sort_step_checkpoint_records(
                [dict(item) for item in raw_best_steps if isinstance(item, dict)]
            )[: max(1, int(config.top_k_step_checkpoints))]
        best_epoch = int(client_state.get("best_epoch", 0))
        best_eval_loss = float(client_state.get("best_eval_loss", float("inf")))
        best_train_loss = float(client_state.get("best_train_loss", float("inf")))

    step = start_step
    epoch = start_epoch
    loss_value = float("nan")
    progress = None
    task_id = None
    if _is_rank_zero():
        progress, task_id = start_training_progress(
            total_steps=resolved_max_steps,
            start_step=start_step,
            description="deepspeed-train",
        )
    train_start_time = time.perf_counter()
    try:
        while step < resolved_max_steps:
            _set_loader_epoch(loader, sampler, epoch)
            epoch += 1
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            loader_iter = iter(loader)
            while step < resolved_max_steps:
                fetch_start_time = time.perf_counter()
                try:
                    candidate_batch = next(loader_iter)
                except StopIteration:
                    break
                data_time = time.perf_counter() - fetch_start_time
                candidate_stats = ctc_batch_token_stats(candidate_batch)
                budgeted = select_ctc_batch_prefix_by_token_budget(
                    candidate_batch,
                    token_budget=config.batch_token_budget,
                    skip_oversized_samples=config.skip_oversized_samples,
                )
                if budgeted is None:
                    if _is_rank_zero():
                        _rank_zero_log("Skipped a candidate batch because no sample fit inside the token budget.")
                    continue
                batch = budgeted.batch
                batch_stats = ctc_batch_token_stats(batch)
                skipped_samples = budgeted.skipped_samples
                dropped_tail_samples = budgeted.dropped_tail_samples

                step_start_time = time.perf_counter()
                engine.zero_grad()
                mask = scheduler.sample_mask(step, device=engine.device)
                executed_token_stats = batch_stats
                if engine.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(engine.device)
                batch = batch.to(engine.device, feature_dtype=feature_dtype)
                logits, logit_lengths, _ = engine(
                    batch.features,
                    batch.feature_lengths,
                    direction_mask=mask,
                )
                if logit_lengths is None:
                    raise ValueError("CTC training requires feature lengths.")
                loss = engine.module.ctc_loss(logits, logit_lengths, batch.targets, batch.target_lengths)
                engine.backward(loss)
                engine.step()
                step_time = time.perf_counter() - step_start_time

                step += 1
                loss_value = float(loss.item())
                epoch_loss_sum += loss_value * batch_stats.batch_size
                epoch_sample_count += batch_stats.batch_size
                peak_reserved_bytes = 0
                peak_allocated_bytes = 0
                estimated_budget = 0
                if engine.device.type == "cuda":
                    torch.cuda.synchronize(engine.device)
                    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(engine.device))
                    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(engine.device))
                    memory_per_token = peak_reserved_bytes / max(executed_token_stats.total_tokens, 1)
                    ratio_tensor = torch.tensor(memory_per_token, device=engine.device, dtype=torch.float64)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(ratio_tensor, op=dist.ReduceOp.MAX)
                    global_peak_reserved_bytes = int(ratio_tensor.item() * max(executed_token_stats.total_tokens, 1))
                    estimated_budget = estimate_token_budget_from_memory(
                        observed_tokens=executed_token_stats.total_tokens,
                        observed_peak_reserved_bytes=global_peak_reserved_bytes,
                        target_memory_gib=config.target_gpu_memory_gib,
                    )

                if progress is not None and task_id is not None:
                    update_training_progress(
                        progress,
                        task_id,
                        step=step,
                        epoch=epoch,
                        loss=loss_value,
                        data_time=data_time,
                        step_time=step_time,
                        total_elapsed=time.perf_counter() - train_start_time,
                        start_step=start_step,
                    )

                if _is_rank_zero() and step == start_step + 1:
                    _rank_zero_log(
                        f"First step timings: data={data_time:.2f}s compute={step_time:.2f}s total={data_time + step_time:.2f}s"
                    )
                    _rank_zero_log(
                        "Batch stats "
                        f"step={step} candidate_batch={candidate_stats.batch_size} "
                        f"candidate_total={candidate_stats.total_tokens} executed_total={executed_token_stats.total_tokens} "
                        f"max_audio={executed_token_stats.max_audio_frames} padded_audio={executed_token_stats.padded_audio_tokens} "
                        f"text={executed_token_stats.text_tokens} "
                        f"peak_reserved={peak_reserved_bytes / (1024**3):.2f}GiB "
                        f"peak_allocated={peak_allocated_bytes / (1024**3):.2f}GiB "
                        f"estimated_budget@{config.target_gpu_memory_gib:.1f}GiB={estimated_budget}"
                    )
                    if config.batch_token_budget is not None:
                        _rank_zero_log(
                            f"Token budget active: budget={config.batch_token_budget} "
                            f"effective_batch={batch_stats.batch_size} dropped_tail={dropped_tail_samples} "
                            f"skipped_samples={skipped_samples}"
                        )
                if _is_rank_zero() and (step <= 10 or step % config.log_every == 0 or step == resolved_max_steps):
                    print(f"[deepspeed-train] step={step} loss={loss_value:.4f} device={device}", flush=True)
                    total_elapsed = time.perf_counter() - train_start_time
                    elapsed_steps = max(step - start_step, 1)
                    rate = elapsed_steps / max(total_elapsed, 1.0e-6)
                    eta_hours = max(resolved_max_steps - step, 0) / max(rate, 1.0e-6) / 3600.0
                    log_wandb(
                        wandb_run,
                        {
                            "train/loss": loss_value,
                            "train/epoch": epoch,
                            "train/data_time": data_time,
                            "train/step_time": step_time,
                            "train/rate": rate,
                            "train/progress_frac": step / max(resolved_max_steps, 1),
                            "train/eta_hours": eta_hours,
                            "train/effective_batch": batch_stats.batch_size,
                            "train/total_tokens": executed_token_stats.total_tokens,
                            "train/max_audio_frames": executed_token_stats.max_audio_frames,
                            "train/padded_audio_tokens": executed_token_stats.padded_audio_tokens,
                            "train/text_tokens": executed_token_stats.text_tokens,
                            "train/peak_reserved_gib": peak_reserved_bytes / (1024**3),
                            "train/peak_allocated_gib": peak_allocated_bytes / (1024**3),
                            "train/estimated_token_budget": estimated_budget,
                        },
                        step=step,
                    )

                if step % config.save_every == 0 or step == resolved_max_steps:
                    checkpoint_extra = {
                        "loss": loss_value,
                        "epoch": epoch,
                        "history": history,
                        "step_checkpoint_history": step_checkpoint_history,
                        "best_step_checkpoints": best_step_checkpoints,
                        "best_epoch": best_epoch,
                        "best_eval_loss": best_eval_loss,
                        "best_train_loss": best_train_loss,
                        "cmvn_file": resolved_cmvn_file,
                        "mask_forward": mask.forward.cpu().tolist(),
                        "mask_backward": mask.backward.cpu().tolist(),
                    }
                    saved_artifacts = _save_export_checkpoints(
                        engine=engine,
                        output_dir=output_dir,
                        tag=f"step-{step}",
                        export_name=f"step-{step}.pt",
                        step=step,
                        zero_stage=zero_stage,
                        extra_state=checkpoint_extra,
                    )
                    if step_eval_every is not None and step % step_eval_every == 0:
                        step_eval_loss, step_eval_count = _evaluate_epoch_loss(
                            model=engine.module,
                            loader=step_eval_loader,
                            sampler=step_eval_sampler,
                            epoch=step,
                            device=engine.device,
                            feature_dtype=feature_dtype,
                            mode=config.eval_mode,
                            max_eval_samples=int(config.step_eval_samples),
                        )
                        step_record = {
                            "step": step,
                            "epoch": epoch,
                            "eval_loss": step_eval_loss,
                            "eval_samples": step_eval_count,
                            **saved_artifacts,
                        }
                        step_checkpoint_history.append(step_record)
                        saved_step_records = [
                            record
                            for record in step_checkpoint_history
                            if record.get("deepspeed_checkpoint_dir") or record.get("checkpoint_path")
                        ]
                        best_step_checkpoints = _sort_step_checkpoint_records(saved_step_records)[
                            : max(1, int(config.top_k_step_checkpoints))
                        ]
                        _prune_deepspeed_step_checkpoint_artifacts(
                            top_records=best_step_checkpoints,
                            saved_records=saved_step_records,
                        )
                        if Path(saved_artifacts["deepspeed_checkpoint_dir"]).exists() or (
                            saved_artifacts.get("checkpoint_path")
                            and Path(str(saved_artifacts["checkpoint_path"])).exists()
                        ):
                            checkpoint_extra["step_checkpoint_history"] = step_checkpoint_history
                            checkpoint_extra["best_step_checkpoints"] = best_step_checkpoints
                            _save_export_checkpoints(
                                engine=engine,
                                output_dir=output_dir,
                                tag=f"step-{step}",
                                export_name=f"step-{step}.pt",
                                step=step,
                                zero_stage=zero_stage,
                                extra_state=checkpoint_extra,
                            )
                        if _is_rank_zero():
                            save_step_checkpoint_metrics(
                                output_dir,
                                history=step_checkpoint_history,
                                best=best_step_checkpoints,
                                keep_top_k=int(config.top_k_step_checkpoints),
                            )
                            _rank_zero_log(
                                f"Step checkpoint eval: step={step} eval_loss={step_eval_loss:.4f} "
                                f"eval_samples={step_eval_count} kept_top_k={len(best_step_checkpoints)}"
                            )
                            log_wandb(
                                wandb_run,
                                {
                                    "eval/step_eval_loss": step_eval_loss,
                                    "eval/step_eval_samples": step_eval_count,
                                    "checkpoint/top_k_kept": len(best_step_checkpoints),
                                },
                                step=step,
                            )
            epoch_train_loss = _all_reduce_mean(epoch_loss_sum, epoch_sample_count, device=engine.device)
            eval_limit = _resolve_epoch_eval_limit(config, epoch=epoch)
            eval_label = "full" if eval_limit is None else f"first {eval_limit}"
            if _is_rank_zero():
                _rank_zero_log(f"Running eval for epoch {epoch} mode={config.eval_mode} samples={eval_label}.")
            epoch_eval_loss, epoch_eval_samples = _evaluate_epoch_loss(
                model=engine.module,
                loader=eval_loader,
                sampler=eval_sampler,
                epoch=epoch - 1,
                device=engine.device,
                feature_dtype=feature_dtype,
                mode=config.eval_mode,
                max_eval_samples=eval_limit,
            )
            epoch_eval_full = eval_limit is None
            metric_value = epoch_eval_loss if not math.isnan(epoch_eval_loss) else epoch_train_loss
            metric_name = "eval_loss" if not math.isnan(epoch_eval_loss) else "train_loss"
            history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": epoch_train_loss,
                    "eval_loss": epoch_eval_loss,
                    "eval_samples": epoch_eval_samples,
                    "eval_full": epoch_eval_full,
                    "selection_metric": metric_value,
                    "selection_metric_name": metric_name,
                }
            )
            if metric_value < (best_eval_loss if metric_name == "eval_loss" else best_train_loss):
                if metric_name == "eval_loss":
                    best_eval_loss = metric_value
                best_train_loss = epoch_train_loss
                best_epoch = epoch
                best_extra = {
                    "loss": loss_value,
                    "epoch": epoch,
                    "history": history,
                    "step_checkpoint_history": step_checkpoint_history,
                    "best_step_checkpoints": best_step_checkpoints,
                    "best_epoch": best_epoch,
                    "best_eval_loss": best_eval_loss,
                    "best_train_loss": best_train_loss,
                    "cmvn_file": resolved_cmvn_file,
                    "selection_metric_name": metric_name,
                }
                _save_export_checkpoints(
                    engine=engine,
                    output_dir=output_dir,
                    tag="best",
                    export_name="best.pt",
                    step=step,
                    zero_stage=zero_stage,
                    extra_state=best_extra,
                )
                if _is_rank_zero():
                    save_yaml(
                        output_dir / "best_checkpoint.yaml",
                        {
                            "epoch": best_epoch,
                            "step": step,
                            "eval_loss": best_eval_loss,
                            "train_loss": best_train_loss,
                            "checkpoint_path": str(output_dir / "best.pt"),
                            "deepspeed_checkpoint_dir": str(output_dir / "ds_checkpoints"),
                            "resume_tag": "best",
                            "selection_metric_name": metric_name,
                        },
                    )
            if _is_rank_zero():
                save_epoch_metrics(
                    output_dir,
                    history=history,
                    best={
                        "epoch": best_epoch,
                        "eval_loss": best_eval_loss,
                        "train_loss": best_train_loss,
                    }
                    if best_epoch > 0
                    else None,
                )
                save_step_checkpoint_metrics(
                    output_dir,
                    history=step_checkpoint_history,
                    best=best_step_checkpoints,
                    keep_top_k=int(config.top_k_step_checkpoints),
                )
            epoch_extra = {
                "loss": loss_value,
                "epoch": epoch,
                "history": history,
                "step_checkpoint_history": step_checkpoint_history,
                "best_step_checkpoints": best_step_checkpoints,
                "best_epoch": best_epoch,
                "best_eval_loss": best_eval_loss,
                "best_train_loss": best_train_loss,
                "cmvn_file": resolved_cmvn_file,
                "train_loss": epoch_train_loss,
                "eval_loss": epoch_eval_loss,
            }
            _save_export_checkpoints(
                engine=engine,
                output_dir=output_dir,
                tag=f"epoch-{epoch}",
                export_name=f"epoch-{epoch}.pt",
                step=step,
                zero_stage=zero_stage,
                extra_state=epoch_extra,
            )
            if _is_rank_zero():
                _rank_zero_log(
                    f"Epoch {epoch} complete: train_loss={epoch_train_loss:.4f} "
                    f"eval_loss={epoch_eval_loss:.4f} eval_samples={epoch_eval_samples} "
                    f"eval_full={epoch_eval_full} best_epoch={best_epoch}"
                )
                log_wandb(
                    wandb_run,
                    {
                        "epoch/index": epoch,
                        "epoch/train_loss": epoch_train_loss,
                        "epoch/eval_loss": epoch_eval_loss,
                        "epoch/eval_samples": epoch_eval_samples,
                        "epoch/eval_full": int(epoch_eval_full),
                        "checkpoint/best_epoch": best_epoch,
                        "checkpoint/best_eval_loss": best_eval_loss,
                    },
                    step=step,
                )
    finally:
        if progress is not None:
            progress.stop()
        if _is_rank_zero():
            finish_wandb(wandb_run)

    return {
        "final_loss": loss_value,
        "steps": step,
        "steps_per_epoch": steps_per_epoch or 0,
        "cmvn_file": resolved_cmvn_file or "",
        "vocab_size": resolved_vocab_size,
        "zero_stage": zero_stage,
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
    }
