from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.data import (
    ASRManifestDataset,
    FeatureCollator,
    MAX_IN_MEMORY_LENGTH_INDEX_BYTES,
    StableHashSplitConfig,
    WebDatasetConfig,
    build_bucketed_webdataset_loader,
    build_length_bucketed_webdataset_dataloader,
    build_text_tokenizer,
    build_webdataset_dataloader,
    can_load_webdataset_length_index_in_memory,
    compute_manifest_global_cmvn,
    compute_webdataset_global_cmvn,
    estimate_bucket_manifest_steps,
    estimate_length_bucketed_steps,
    format_num_bytes,
    index_split_sample_count,
    load_webdataset_bucket_manifest,
    load_webdataset_index,
    load_webdataset_length_entries,
    resolve_webdataset_bucket_manifest_path,
    resolve_webdataset_length_index_path,
    resolve_webdataset_index_path,
    validate_webdataset_index,
)
from rwkvasr.modules import DirectionDropoutConfig, DirectionDropoutScheduler, RWKVCTCModel, RWKVCTCModelConfig

from .checkpoint import load_checkpoint, save_checkpoint
from .batch_budget import ctc_batch_token_stats, estimate_token_budget_from_memory, select_ctc_batch_prefix_by_token_budget
from .ctc_task import RWKVDualModeCTCTrainer
from .epoch_metrics import save_epoch_metrics, save_step_checkpoint_metrics
from .optimizer import RWKVOptimizerConfig, build_rwkv_optimizer
from .progress import start_training_progress, update_training_progress
from .wandb_logger import finish_wandb, init_wandb_run, log_wandb


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
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
    device: str = "cpu"
    resume_from: str | None = None
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
    batch_token_budget: int | None = None
    length_bucket_frame_budget: int | None = None
    target_gpu_memory_gib: float = 22.0
    skip_oversized_samples: bool = True


def _resolve_data_source(config: TrainConfig) -> tuple[str, str]:
    has_manifest = config.manifest_path is not None
    has_webdataset = config.webdataset_root is not None
    if has_manifest == has_webdataset:
        raise ValueError("Exactly one of manifest_path or webdataset_root must be provided.")
    if has_manifest:
        return "manifest", str(config.manifest_path)
    return "webdataset", str(config.webdataset_root)


def _resolve_vocab_size(config: TrainConfig) -> int:
    if config.vocab_size is not None:
        if config.tokenizer_type == "sentencepiece" and config.tokenizer_model_path is not None:
            tokenizer = _resolve_text_tokenizer(config)
            tokenizer_vocab_size = int(tokenizer.vocab_size)
            if int(config.vocab_size) != tokenizer_vocab_size:
                raise ValueError(
                    "Configured vocab_size does not match the resolved SentencePiece vocabulary size: "
                    f"{config.vocab_size} != {tokenizer_vocab_size}"
                )
        return int(config.vocab_size)
    tokenizer = _resolve_text_tokenizer(config)
    return int(tokenizer.vocab_size)


def _resolve_text_tokenizer(config: TrainConfig):
    return build_text_tokenizer(
        config.tokenizer_type,
        model_path=config.tokenizer_model_path,
        language=config.tokenizer_language,
        task=config.tokenizer_task,
    )


def _resolved_tokenizer_config_payload(config: TrainConfig, *, vocab_size: int) -> dict[str, object]:
    model_path = config.tokenizer_model_path
    if model_path is not None:
        model_path = str(Path(model_path).resolve())
    return {
        "tokenizer_type": config.tokenizer_type,
        "tokenizer_model_path": model_path,
        "tokenizer_language": config.tokenizer_language,
        "tokenizer_task": config.tokenizer_task,
        "vocab_size": int(vocab_size),
    }


def _build_webdataset_config(
    config: TrainConfig,
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


def _resolve_candidate_length_index_path(data_path: str, configured_path: str | None) -> str | None:
    length_index_path = configured_path
    if length_index_path is None:
        candidate_path = resolve_webdataset_length_index_path(data_path)
        length_index_path = str(candidate_path) if candidate_path.exists() else None
    return length_index_path


def _resolve_bucket_manifest_path(data_path: str, configured_path: str | None) -> str | None:
    manifest_path = resolve_webdataset_bucket_manifest_path(data_path, configured_path)
    if manifest_path.exists():
        return str(manifest_path)
    return None


def _resolve_in_memory_length_index_path(
    data_path: str,
    configured_path: str | None,
    *,
    logger=_log,
) -> str | None:
    length_index_path = _resolve_candidate_length_index_path(data_path, configured_path)
    if length_index_path is None:
        return None
    if can_load_webdataset_length_index_in_memory(length_index_path):
        return length_index_path
    logger(
        "Length index is too large for the current in-memory bucketing path. "
        f"index={length_index_path} "
        f"size={format_num_bytes(Path(length_index_path).stat().st_size)} "
        f"limit={format_num_bytes(MAX_IN_MEMORY_LENGTH_INDEX_BYTES)}. "
        "Falling back to streaming WebDataset loading without offline length bucketing."
    )
    return None


def _resolve_max_steps(config: TrainConfig) -> tuple[int, int | None]:
    if config.max_steps is not None and config.epochs is not None:
        raise ValueError("Specify only one of max_steps or epochs.")
    if config.max_steps is not None:
        return int(config.max_steps), None
    if config.epochs is None:
        return 100, None

    data_source, data_path = _resolve_data_source(config)
    if data_source == "manifest":
        num_samples = len(ASRManifestDataset(data_path))
        steps_per_epoch = max(1, math.ceil(num_samples / config.batch_size))
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
                world_size=1,
                frame_budget=config.length_bucket_frame_budget or config.batch_token_budget,
                drop_last=True,
            )
            return steps_per_epoch * int(config.epochs), steps_per_epoch
        length_index_path = _resolve_in_memory_length_index_path(
            data_path,
            config.webdataset_length_index_path,
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
                    world_size=1,
                    frame_budget=length_bucket_frame_budget,
                    drop_last=True,
                ),
            )
        else:
            steps_per_epoch = max(1, math.ceil(num_samples / config.batch_size))
    return steps_per_epoch * int(config.epochs), steps_per_epoch


def _resolve_cmvn_file(config: TrainConfig, output_dir: Path) -> str | None:
    if config.frontend_type != "conv2d6":
        return config.cmvn_file
    if config.cmvn_file is not None:
        _log(f"Using CMVN file: {config.cmvn_file}")
        return config.cmvn_file

    data_source, data_path = _resolve_data_source(config)
    cmvn_path = output_dir / "global_cmvn.json"
    if not cmvn_path.exists():
        source_label = f"manifest {data_path}" if data_source == "manifest" else f"webdataset {data_path}"
        _log(f"CMVN file not found. Computing global CMVN from {source_label} -> {cmvn_path}")
        if data_source == "manifest":
            compute_manifest_global_cmvn(data_path, cmvn_path)
        else:
            compute_webdataset_global_cmvn(
                data_path,
                cmvn_path,
                config=_build_webdataset_config(config, shuffle_shards=False),
            )
        _log(f"Finished computing global CMVN: {cmvn_path}")
    return str(cmvn_path)


def _build_train_loader(config: TrainConfig) -> DataLoader:
    data_source, data_path = _resolve_data_source(config)
    tokenizer = _resolve_text_tokenizer(config)
    if data_source == "manifest":
        dataset = ASRManifestDataset(data_path, tokenizer=tokenizer)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=FeatureCollator(),
        )
    webdataset_config = _build_webdataset_config(config, shuffle_shards=True)
    bucket_manifest_path = _resolve_bucket_manifest_path(data_path, config.webdataset_bucket_manifest_path)
    if bucket_manifest_path is not None:
        return build_bucketed_webdataset_loader(
            data_path,
            bucket_manifest_path=bucket_manifest_path,
            tokenizer=tokenizer,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            rank=0,
            world_size=1,
        )
    length_index_path = _resolve_in_memory_length_index_path(
        data_path,
        config.webdataset_length_index_path,
    )
    if length_index_path is not None:
        loader, _ = build_length_bucketed_webdataset_dataloader(
            data_path,
            length_index_path=length_index_path,
            tokenizer=tokenizer,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        return loader
    return build_webdataset_dataloader(
        data_path,
        tokenizer=tokenizer,
        config=webdataset_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


def _build_eval_loader(config: TrainConfig, *, shuffle_shards: bool = False, step_subset: bool = False) -> DataLoader | None:
    data_source, data_path = _resolve_data_source(config)
    eval_batch_size = _resolve_eval_batch_size(config, step_subset=step_subset)
    if data_source == "manifest":
        dataset = ASRManifestDataset(data_path, tokenizer=_resolve_text_tokenizer(config))
        return DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=shuffle_shards,
            num_workers=config.num_workers,
            collate_fn=FeatureCollator(),
        )

    tokenizer = _resolve_text_tokenizer(config)
    webdataset_config = _build_webdataset_config(config, shuffle_shards=shuffle_shards)
    webdataset_config = replace(webdataset_config, split="eval")
    bucket_manifest_path = _resolve_bucket_manifest_path(data_path, config.webdataset_bucket_manifest_path)
    if bucket_manifest_path is not None:
        return build_bucketed_webdataset_loader(
            data_path,
            bucket_manifest_path=bucket_manifest_path,
            tokenizer=tokenizer,
            config=webdataset_config,
            batch_size=eval_batch_size,
            num_workers=config.num_workers,
            rank=0,
            world_size=1,
        )
    length_index_path = _resolve_in_memory_length_index_path(
        data_path,
        config.webdataset_length_index_path,
    )
    if length_index_path is not None:
        loader, _ = build_length_bucketed_webdataset_dataloader(
            data_path,
            length_index_path=length_index_path,
            tokenizer=tokenizer,
            config=webdataset_config,
            batch_size=eval_batch_size,
            num_workers=config.num_workers,
        )
        return loader
    return build_webdataset_dataloader(
        data_path,
        tokenizer=tokenizer,
        config=webdataset_config,
        batch_size=eval_batch_size,
        num_workers=config.num_workers,
    )


@torch.no_grad()
def _evaluate_loss(
    loader: DataLoader | None,
    *,
    task: RWKVDualModeCTCTrainer,
    device: torch.device,
    feature_dtype: torch.dtype | None,
    mode: str,
    loader_epoch: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[float, int]:
    if loader is None:
        return float("nan"), 0
    if loader_epoch is not None:
        _set_loader_epoch(loader, loader_epoch)
    model = task.model
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        remaining = None if max_eval_samples is None else max_eval_samples - total_samples
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and int(batch.features.size(0)) > remaining:
            batch = batch.prefix(remaining)
        batch = batch.to(device, feature_dtype=feature_dtype)
        loss = task.eval_loss(batch, mode=mode)
        batch_size = int(batch.features.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    if was_training:
        model.train()
    if total_samples == 0:
        return float("nan"), 0
    return total_loss / total_samples, total_samples


def _resolve_epoch_eval_limit(config: TrainConfig, *, epoch: int) -> int | None:
    if config.max_eval_samples is None:
        return None
    if config.epochs is not None and epoch >= int(config.epochs):
        return None
    return int(config.max_eval_samples)


def _resolve_step_eval_every(config: TrainConfig) -> int | None:
    if config.step_eval_samples is None or int(config.step_eval_samples) <= 0:
        return None
    if config.step_eval_every is not None:
        return max(1, int(config.step_eval_every))
    return max(1, int(config.save_every))


def _resolve_eval_batch_size(config: TrainConfig, *, step_subset: bool) -> int:
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


def _prune_local_step_checkpoint_artifacts(
    *,
    top_records: list[dict[str, Any]],
    saved_records: list[dict[str, Any]],
) -> None:
    keep_paths = {
        str(record["checkpoint_path"])
        for record in top_records
        if record.get("checkpoint_path")
    }
    for record in saved_records:
        checkpoint_path = record.get("checkpoint_path")
        if checkpoint_path is None:
            continue
        checkpoint_path = str(checkpoint_path)
        if checkpoint_path in keep_paths:
            continue
        path = Path(checkpoint_path)
        if path.exists():
            path.unlink()


def _set_loader_epoch(loader: Any, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    elif hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)


def train_ctc_model(config: TrainConfig) -> dict[str, float | int | str]:
    device = torch.device(config.device)
    feature_dtype = torch.bfloat16 if device.type == "cuda" else None
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_vocab_size = _resolve_vocab_size(config)
    resolved_max_steps, steps_per_epoch = _resolve_max_steps(config)
    resolved_cmvn_file = _resolve_cmvn_file(config, output_dir)
    loader = _build_train_loader(config)
    eval_loader = _build_eval_loader(config)
    step_eval_loader = _build_eval_loader(config, shuffle_shards=True, step_subset=True)
    step_eval_every = _resolve_step_eval_every(config)
    active_bucket_manifest_path = None
    active_length_index_path = None
    if config.webdataset_root is not None:
        active_bucket_manifest_path = _resolve_bucket_manifest_path(
            config.webdataset_root,
            config.webdataset_bucket_manifest_path,
        )
        active_length_index_path = _resolve_in_memory_length_index_path(
            config.webdataset_root,
            config.webdataset_length_index_path,
        )
    if active_bucket_manifest_path is not None:
        frame_budget = config.length_bucket_frame_budget
        if frame_budget is None:
            frame_budget = config.batch_token_budget
        _log(
            "Bucket manifest active: "
            f"path={active_bucket_manifest_path} "
            f"max_local_batch={config.batch_size} "
            f"frame_budget={frame_budget} "
            f"decode_workers={max(1, config.num_workers)}"
        )
    elif active_length_index_path is not None:
        frame_budget = config.length_bucket_frame_budget
        if frame_budget is None:
            frame_budget = config.batch_token_budget
        _log(
            "Length bucketing active: "
            f"max_local_batch={config.batch_size} "
            f"frame_budget={frame_budget}"
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
    save_yaml(output_dir / "model_config.yaml", model_config)
    save_yaml(
        output_dir / "tokenizer_config.yaml",
        _resolved_tokenizer_config_payload(config, vocab_size=resolved_vocab_size),
    )
    save_yaml(
        output_dir / "train_config.yaml",
        {
            **replace(
                config,
                vocab_size=resolved_vocab_size,
                cmvn_file=resolved_cmvn_file,
                max_steps=resolved_max_steps,
            ).__dict__,
            "steps_per_epoch": steps_per_epoch,
        },
    )
    wandb_run = init_wandb_run(
        enabled=config.wandb_enabled,
        project=config.wandb_project,
        run_name=config.wandb_run_name,
        output_dir=output_dir,
        config=load_yaml(output_dir / "train_config.yaml"),
        base_url=config.wandb_base_url,
        init_timeout_sec=config.wandb_init_timeout_sec,
        logger=_log,
    )

    model = RWKVCTCModel(model_config)
    if feature_dtype is not None:
        model = model.to(device=device, dtype=feature_dtype)
    else:
        model = model.to(device)
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
    task = RWKVDualModeCTCTrainer(model, direction_scheduler=scheduler)
    optimizer = build_rwkv_optimizer(
        model,
        RWKVOptimizerConfig(
            lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
        ),
    )
    start_step = 0
    start_epoch = 0
    history: list[dict[str, float | int | str]] = []
    step_checkpoint_history: list[dict[str, Any]] = []
    best_step_checkpoints: list[dict[str, Any]] = []
    best_epoch = 0
    best_eval_loss = float("inf")
    best_train_loss = float("inf")
    if config.resume_from is not None:
        restored = load_checkpoint(config.resume_from, model=model, optimizer=optimizer, map_location=device.type)
        start_step = int(restored["step"])
        extra = dict(restored.get("extra", {}))
        start_epoch = int(extra.get("epoch", 0))
        raw_history = extra.get("history", [])
        if isinstance(raw_history, list):
            history = [dict(item) for item in raw_history if isinstance(item, dict)]
        raw_step_history = extra.get("step_checkpoint_history", [])
        if isinstance(raw_step_history, list):
            step_checkpoint_history = [dict(item) for item in raw_step_history if isinstance(item, dict)]
        raw_best_steps = extra.get("best_step_checkpoints", [])
        if isinstance(raw_best_steps, list):
            best_step_checkpoints = _sort_step_checkpoint_records(
                [dict(item) for item in raw_best_steps if isinstance(item, dict)]
            )[: max(1, int(config.top_k_step_checkpoints))]
        best_epoch = int(extra.get("best_epoch", 0))
        best_eval_loss = float(extra.get("best_eval_loss", float("inf")))
        best_train_loss = float(extra.get("best_train_loss", float("inf")))

    step = start_step
    loss_value = float("nan")
    progress, task_id = start_training_progress(
        total_steps=resolved_max_steps,
        start_step=start_step,
        description="train",
    )
    _log("The first batch can be slower because wav->fbank decoding is done online.")
    train_start_time = time.perf_counter()
    epoch = start_epoch
    try:
        while step < resolved_max_steps:
            epoch += 1
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            _set_loader_epoch(loader, epoch)
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
                    _log("Skipped a candidate batch because no sample fit inside the token budget.")
                    continue
                batch = budgeted.batch
                batch_stats = ctc_batch_token_stats(batch)
                skipped_samples = budgeted.skipped_samples
                dropped_tail_samples = budgeted.dropped_tail_samples

                step_start_time = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                mask = task.training_direction_mask(step, device=device)
                executed_token_stats = batch_stats
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                batch = batch.to(device, feature_dtype=feature_dtype)
                loss, _ = task.training_loss(batch, step=step, direction_mask=mask)
                loss.backward()
                optimizer.step()
                step_time = time.perf_counter() - step_start_time

                step += 1
                loss_value = float(loss.item())
                peak_reserved_bytes = 0
                peak_allocated_bytes = 0
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
                    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
                estimated_budget = estimate_token_budget_from_memory(
                    observed_tokens=executed_token_stats.total_tokens,
                    observed_peak_reserved_bytes=peak_reserved_bytes,
                    target_memory_gib=config.target_gpu_memory_gib,
                )
                epoch_loss_sum += loss_value * batch_stats.batch_size
                epoch_sample_count += batch_stats.batch_size
                if step == start_step + 1:
                    _log(
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
                        _log(
                            f"Token budget active: budget={config.batch_token_budget} "
                            f"effective_batch={batch_stats.batch_size} dropped_tail={dropped_tail_samples} "
                            f"skipped_samples={skipped_samples}"
                        )
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
                if step <= 10 or step % config.log_every == 0 or step == resolved_max_steps:
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
                    step_checkpoint_path = output_dir / f"step-{step}.pt"
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
                    save_checkpoint(
                        step_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        extra=checkpoint_extra,
                    )
                    if step_eval_every is not None and step % step_eval_every == 0:
                        step_eval_loss, step_eval_count = _evaluate_loss(
                            step_eval_loader,
                            task=task,
                            device=device,
                            feature_dtype=feature_dtype,
                            mode=config.eval_mode,
                            loader_epoch=step,
                            max_eval_samples=int(config.step_eval_samples),
                        )
                        step_record = {
                            "step": step,
                            "epoch": epoch,
                            "eval_loss": step_eval_loss,
                            "eval_samples": step_eval_count,
                            "checkpoint_path": str(step_checkpoint_path),
                        }
                        step_checkpoint_history.append(step_record)
                        saved_step_records = [
                            record
                            for record in step_checkpoint_history
                            if record.get("checkpoint_path")
                        ]
                        best_step_checkpoints = _sort_step_checkpoint_records(saved_step_records)[
                            : max(1, int(config.top_k_step_checkpoints))
                        ]
                        _prune_local_step_checkpoint_artifacts(
                            top_records=best_step_checkpoints,
                            saved_records=saved_step_records,
                        )
                        if step_checkpoint_path.exists():
                            checkpoint_extra["step_checkpoint_history"] = step_checkpoint_history
                            checkpoint_extra["best_step_checkpoints"] = best_step_checkpoints
                            save_checkpoint(
                                step_checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                step=step,
                                extra=checkpoint_extra,
                            )
                        save_step_checkpoint_metrics(
                            output_dir,
                            history=step_checkpoint_history,
                            best=best_step_checkpoints,
                            keep_top_k=int(config.top_k_step_checkpoints),
                        )
                        _log(
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
            epoch_train_loss = float("nan") if epoch_sample_count == 0 else epoch_loss_sum / epoch_sample_count
            eval_limit = _resolve_epoch_eval_limit(config, epoch=epoch)
            eval_label = "full" if eval_limit is None else f"first {eval_limit}"
            _log(f"Running eval for epoch {epoch} mode={config.eval_mode} samples={eval_label}.")
            epoch_eval_loss, epoch_eval_samples = _evaluate_loss(
                eval_loader,
                task=task,
                device=device,
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
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    extra={
                        "loss": loss_value,
                        "epoch": epoch,
                        "history": history,
                        "step_checkpoint_history": step_checkpoint_history,
                        "best_step_checkpoints": best_step_checkpoints,
                        "best_epoch": best_epoch,
                        "best_eval_loss": best_eval_loss,
                        "best_train_loss": best_train_loss,
                        "cmvn_file": resolved_cmvn_file,
                    },
                )
                save_yaml(
                    output_dir / "best_checkpoint.yaml",
                    {
                        "epoch": best_epoch,
                        "step": step,
                        "eval_loss": best_eval_loss,
                        "train_loss": best_train_loss,
                        "checkpoint_path": str(output_dir / "best.pt"),
                        "selection_metric_name": metric_name,
                    },
                )
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
            save_checkpoint(
                output_dir / f"epoch-{epoch}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                extra={
                    "loss": loss_value,
                    "epoch": epoch,
                    "history": history,
                    "step_checkpoint_history": step_checkpoint_history,
                    "best_step_checkpoints": best_step_checkpoints,
                    "best_epoch": best_epoch,
                    "best_eval_loss": best_eval_loss,
                    "best_train_loss": best_train_loss,
                    "cmvn_file": resolved_cmvn_file,
                },
            )
            _log(
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
        progress.stop()
        finish_wandb(wandb_run)

    return {
        "final_loss": loss_value,
        "steps": step,
        "steps_per_epoch": steps_per_epoch or 0,
        "cmvn_file": resolved_cmvn_file or "",
        "vocab_size": resolved_vocab_size,
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
    }
