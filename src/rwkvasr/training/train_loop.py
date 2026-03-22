from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rwkvasr.config import save_yaml
from rwkvasr.data import (
    ASRManifestDataset,
    FeatureCollator,
    StableHashSplitConfig,
    WebDatasetConfig,
    build_length_bucketed_webdataset_dataloader,
    build_text_tokenizer,
    build_webdataset_dataloader,
    compute_manifest_global_cmvn,
    compute_webdataset_global_cmvn,
    estimate_length_bucketed_steps,
    index_split_sample_count,
    load_webdataset_index,
    load_webdataset_length_entries,
    resolve_webdataset_length_index_path,
    resolve_webdataset_index_path,
    validate_webdataset_index,
)
from rwkvasr.modules import DirectionDropoutConfig, DirectionDropoutScheduler, RWKVCTCModel, RWKVCTCModelConfig

from .checkpoint import load_checkpoint, save_checkpoint
from .batch_budget import ctc_batch_token_stats, estimate_token_budget_from_memory, select_ctc_batch_prefix_by_token_budget
from .ctc_task import RWKVDualModeCTCTrainer
from .epoch_metrics import save_epoch_metrics
from .optimizer import RWKVOptimizerConfig, build_rwkv_optimizer
from .progress import start_training_progress, update_training_progress


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
    vocab_size: int | None = None
    manifest_path: str | None = None
    webdataset_root: str | None = None
    webdataset_index_path: str | None = None
    webdataset_length_index_path: str | None = None
    webdataset_split: str = "all"
    webdataset_eval_ratio: float = 0.0
    webdataset_hash_seed: int = 0
    webdataset_split_by: str = "shard_name"
    input_dim: int = 80
    n_embd: int = 512
    dim_att: int = 512
    dim_ff: int = 2048
    num_layers: int = 12
    head_size: int = 64
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
    eval_mode: str = "bi"
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
        return int(config.vocab_size)
    return int(build_text_tokenizer("whisper_multilingual").vocab_size)


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
        length_index_path=config.webdataset_length_index_path,
        length_bucket_frame_budget=length_bucket_frame_budget,
    )


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
            ),
        )
        num_samples = index_split_sample_count(index_data, config.webdataset_split)
        length_index_path = config.webdataset_length_index_path
        if length_index_path is None:
            candidate_path = resolve_webdataset_length_index_path(data_path)
            length_index_path = str(candidate_path) if candidate_path.exists() else None
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
    if data_source == "manifest":
        dataset = ASRManifestDataset(data_path)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=FeatureCollator(),
        )
    webdataset_config = _build_webdataset_config(config, shuffle_shards=True)
    length_index_path = config.webdataset_length_index_path
    if length_index_path is None:
        candidate_path = resolve_webdataset_length_index_path(data_path)
        if candidate_path.exists():
            length_index_path = str(candidate_path)
    if length_index_path is not None:
        loader, _ = build_length_bucketed_webdataset_dataloader(
            data_path,
            length_index_path=length_index_path,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        return loader
    return build_webdataset_dataloader(
        data_path,
        config=webdataset_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


def _build_eval_loader(config: TrainConfig) -> DataLoader | None:
    data_source, data_path = _resolve_data_source(config)
    if data_source == "manifest":
        return None

    webdataset_config = _build_webdataset_config(config, shuffle_shards=False)
    webdataset_config = replace(webdataset_config, split="eval")
    length_index_path = config.webdataset_length_index_path
    if length_index_path is None:
        candidate_path = resolve_webdataset_length_index_path(data_path)
        if candidate_path.exists():
            length_index_path = str(candidate_path)
    if length_index_path is not None:
        loader, _ = build_length_bucketed_webdataset_dataloader(
            data_path,
            length_index_path=length_index_path,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        return loader
    return build_webdataset_dataloader(
        data_path,
        config=webdataset_config,
        batch_size=config.batch_size,
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
) -> float:
    if loader is None:
        return float("nan")
    model = task.model
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = batch.to(device, feature_dtype=feature_dtype)
        loss = task.eval_loss(batch, mode=mode)
        batch_size = int(batch.features.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    if was_training:
        model.train()
    if total_samples == 0:
        return float("nan")
    return total_loss / total_samples


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
    if config.webdataset_root is not None and (
        config.webdataset_length_index_path is not None
        or resolve_webdataset_length_index_path(config.webdataset_root).exists()
    ):
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
        conv_kernel_size=config.conv_kernel_size,
        dropout=config.dropout,
        frontend_type=config.frontend_type,
        cmvn_file=resolved_cmvn_file,
        cmvn_is_json=config.cmvn_is_json,
    )
    save_yaml(output_dir / "model_config.yaml", model_config)
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

                if step % config.save_every == 0 or step == resolved_max_steps:
                    save_checkpoint(
                        output_dir / f"step-{step}.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        extra={
                            "loss": loss_value,
                            "epoch": epoch,
                            "history": history,
                            "best_epoch": best_epoch,
                            "best_eval_loss": best_eval_loss,
                            "best_train_loss": best_train_loss,
                            "cmvn_file": resolved_cmvn_file,
                            "mask_forward": mask.forward.cpu().tolist(),
                            "mask_backward": mask.backward.cpu().tolist(),
                        },
                    )
            epoch_train_loss = float("nan") if epoch_sample_count == 0 else epoch_loss_sum / epoch_sample_count
            _log(f"Running eval for epoch {epoch} mode={config.eval_mode}.")
            epoch_eval_loss = _evaluate_loss(
                eval_loader,
                task=task,
                device=device,
                feature_dtype=feature_dtype,
                mode=config.eval_mode,
            )
            metric_value = epoch_eval_loss if not math.isnan(epoch_eval_loss) else epoch_train_loss
            metric_name = "eval_loss" if not math.isnan(epoch_eval_loss) else "train_loss"
            history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": epoch_train_loss,
                    "eval_loss": epoch_eval_loss,
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
            save_checkpoint(
                output_dir / f"epoch-{epoch}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                extra={
                    "loss": loss_value,
                    "epoch": epoch,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_eval_loss": best_eval_loss,
                    "best_train_loss": best_train_loss,
                    "cmvn_file": resolved_cmvn_file,
                },
            )
            _log(
                f"Epoch {epoch} complete: train_loss={epoch_train_loss:.4f} "
                f"eval_loss={epoch_eval_loss:.4f} best_epoch={best_epoch}"
            )
    finally:
        progress.stop()

    return {
        "final_loss": loss_value,
        "steps": step,
        "steps_per_epoch": steps_per_epoch or 0,
        "cmvn_file": resolved_cmvn_file or "",
        "vocab_size": resolved_vocab_size,
        "best_epoch": best_epoch,
        "best_eval_loss": best_eval_loss,
    }
