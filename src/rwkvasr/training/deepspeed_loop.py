from __future__ import annotations

import os
import json
import math
import time
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.data import (
    ASRManifestDataset,
    FeatureCollator,
    StableHashSplitConfig,
    WebDatasetConfig,
    build_audio_feature_extractor,
    build_text_tokenizer,
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
    tokenizer_eos_token_id,
    validate_webdataset_index,
)
from rwkvasr.modules import DirectionDropoutConfig, DirectionDropoutScheduler, DirectionMask, RWKVCTCModel, RWKVCTCModelConfig

from .checkpoint import (
    extract_epoch_batch_offset,
    load_checkpoint,
    load_latest_checkpoint_state,
    save_checkpoint,
    write_latest_checkpoint_state,
)
from .batch_budget import (
    ctc_batch_token_stats,
    effective_batch_token_budget,
    effective_padded_text_token_budget,
    estimate_token_budget_from_memory,
    select_ctc_batch_prefix_by_token_budget,
)
from .ctc_task import RWKVDualModeCTCTrainer
from .epoch_metrics import save_epoch_metrics, save_step_checkpoint_metrics
from .funasr_online_teacher import FunASRNanoCTCTopKOnlineTeacher, FunASROnlineCTCTeacherConfig
from .optimizer import build_rwkv_param_groups
from .progress import start_training_progress, update_training_progress
from .spec_augment import apply_spec_augment
from .wandb_logger import finish_wandb, init_wandb_run, log_wandb
from .train_loop import (
    _iter_loader_after_resume_offset,
    _resolve_bucket_manifest_path,
    _resolve_cmvn_file,
    _resolve_data_source,
    _resolve_decoder_template_token_ids,
    _resolve_ctc_suppressed_token_ids,
    _resolve_in_memory_length_index_path,
    _resolve_text_tokenizer,
    _resolved_tokenizer_config_payload,
)


def _parse_deepspeed_checkpoint_tag(value: Any) -> str | None:
    tag = str(value).strip()
    if not tag:
        return None
    if tag == "best":
        return "best"
    if tag.startswith("step-"):
        suffix = tag.removeprefix("step-")
        if suffix and suffix.isdigit():
            return tag
    if tag.startswith("epoch-"):
        suffix = tag.removeprefix("epoch-")
        if suffix and suffix.isdigit():
            return tag
    return None


def _parse_deepspeed_checkpoint_step(value: Any) -> int:
    tag = _parse_deepspeed_checkpoint_tag(value)
    if tag is None:
        return 0
    if not tag.startswith("step-"):
        return 0
    try:
        return int(tag.removeprefix("step-"))
    except ValueError:
        return 0


def _resolve_latest_deepspeed_checkpoint(
    output_dir: str | Path,
) -> tuple[Path, str] | None:
    ds_checkpoint_root = Path(output_dir) / "ds_checkpoints"
    if not ds_checkpoint_root.is_dir():
        return None

    best_path: Path | None = None
    best_step: int = -1
    best_step_path: Path | None = None
    best_epoch: int = -1
    best_epoch_path: Path | None = None

    for candidate in ds_checkpoint_root.iterdir():
        if not candidate.is_dir():
            continue
        tag = _parse_deepspeed_checkpoint_tag(candidate.name)
        if tag is None:
            continue
        if tag == "best":
            best_path = candidate
            continue
        if tag.startswith("epoch-"):
            try:
                epoch = int(tag.removeprefix("epoch-"))
            except ValueError:
                continue
            if epoch > best_epoch:
                best_epoch = epoch
                best_epoch_path = candidate
            continue
        try:
            step = int(tag.removeprefix("step-"))
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_step_path = candidate

    if best_step_path is not None:
        return best_step_path, best_step_path.name
    if best_epoch_path is not None:
        return best_epoch_path, best_epoch_path.name
    if best_path is not None:
        return best_path, "best"
    return None


def _load_export_checkpoint_extra_state(output_dir: Path, step: int) -> dict[str, Any]:
    export_path = output_dir / f"step-{step}.pt"
    if not export_path.is_file():
        return {}
    try:
        checkpoint = torch.load(export_path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    extra = checkpoint.get("extra")
    if isinstance(extra, dict):
        return extra
    return {}


@dataclass(frozen=True)
class DeepSpeedTrainConfig:
    output_dir: str
    deepspeed: dict[str, Any]
    vocab_size: int | None = None
    tokenizer_type: str = "whisper_multilingual"
    tokenizer_model_path: str | None = None
    tokenizer_language: str | None = None
    tokenizer_task: str | None = None
    tokenizer_append_eos: bool = False
    text_normalization: str = "none"
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
    feature_extractor_type: str = "wenet_fbank"
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
    encoder_output_dim: int | None = None
    aut_downsample_hidden_size: int = 480
    aut_activation_function: str = "gelu"
    aut_activation_dropout: float = 0.0
    aut_max_source_positions: int = 1500
    aut_scale_embedding: bool = False
    aut_conv_chunksize: int = 500
    sensevoice_tp_blocks: int = 20
    cmvn_file: str | None = None
    cmvn_is_json: bool = True
    blank_id: int = 0
    batch_size: int = 4
    max_steps: int | None = None
    epochs: int | None = None
    save_every: int = 50
    num_workers: int = 0
    decoded_batch_prefetch: int = 2
    max_open_shards_per_worker: int = 8
    bucket_source_interleave: bool = False
    lr: float = 4e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    decoder_enabled: bool = False
    decoder_checkpoint_path: str | None = None
    decoder_num_layers: int | None = None
    decoder_n_embd: int | None = None
    decoder_ffn_hidden_size: int | None = None
    decoder_vocab_size: int | None = None
    decoder_tokenizer_type: str | None = None
    decoder_tokenizer_model_path: str | None = None
    decoder_tokenizer_language: str | None = None
    decoder_tokenizer_task: str | None = None
    decoder_tokenizer_append_eos: bool = False
    decoder_text_normalization: str | None = None
    decoder_head_size: int = 64
    decoder_audio_conditioning: str = "full"
    decoder_prefix_tokens: int = 32
    decoder_loss_chunk_size: int = 1024
    decoder_text_token_budget: int | None = None
    decoder_prompt_before_audio: str = ""
    decoder_prompt_before_audio_use_language: bool = False
    decoder_ctc_draft_cache_path: str | None = None
    decoder_ctc_draft_prompt_template: str = ""
    decoder_ctc_draft_text_key: str = "pred_text"
    decoder_ctc_draft_missing_policy: str = "empty"
    decoder_ctc_draft_dropout_prob: float = 0.0
    decoder_ctc_draft_language_mismatch_dropout_prob: float = 0.0
    decoder_ctc_draft_dropout_seed: int = 0
    ctc_label_override_cache_path: str | None = None
    ctc_label_override_text_key: str = "pred_text"
    allow_missing_targets: bool = False
    decoder_prompt_after_audio: str = ""
    decoder_target_prefix: str = ""
    decoder_target_prefix_use_language: bool = False
    decoder_target_suffix: str = ""
    decoder_language_confirmation_en: str = "This is English text."
    decoder_language_confirmation_zh: str = "这是中文文字。"
    decoder_prompt_language_label_noise_prob: float = 0.0
    decoder_prompt_language_label_noise_seed: int = 0
    decoder_eos_token_id: int = 0
    ctc_loss_weight: float = 1.0
    decoder_loss_weight: float = 0.0
    encoder_anchor_checkpoint_path: str | None = None
    encoder_anchor_loss_weight: float = 0.0
    ctc_logit_anchor_loss_weight: float = 0.0
    ctc_logit_anchor_chunk_frames: int = 32
    ctc_teacher_topk_cache_path: str | None = None
    ctc_teacher_topk_loss_weight: float = 0.0
    ctc_teacher_topk_blank_loss_weight: float = 0.0
    ctc_teacher_topk_mass_loss_weight: float = 0.0
    ctc_teacher_topk_time_map: str = "nearest"
    ctc_teacher_frame_filter: str = "all"
    ctc_teacher_frame_filter_neighbor_radius: int = 0
    ctc_teacher_frame_filter_min_nonblank_prob: float = 0.0
    ctc_teacher_topk_missing_policy: str = "skip"
    ctc_teacher_online_model_path: str | None = None
    ctc_teacher_online_loss_weight: float = 0.0
    ctc_teacher_online_blank_loss_weight: float = 0.0
    ctc_teacher_online_mass_loss_weight: float = 0.0
    ctc_teacher_online_full_loss_weight: float = 0.0
    ctc_teacher_online_full_temperature: float = 1.0
    ctc_teacher_online_full_frame_filter: str | None = None
    ctc_teacher_online_full_nonblank_weight: float = 1.0
    ctc_teacher_online_encoder_loss_weight: float = 0.0
    ctc_teacher_online_sequence_loss_weight: float = 0.0
    ctc_teacher_online_sequence_presence_loss_weight: float = 0.0
    ctc_teacher_online_sequence_window_loss_weight: float = 0.0
    ctc_teacher_online_sequence_window_radius: int = 2
    ctc_teacher_online_sequence_window_temperature: float = 0.2
    ctc_teacher_online_nonblank_hard_loss_weight: float = 0.0
    ctc_teacher_online_nonblank_margin_loss_weight: float = 0.0
    ctc_teacher_online_nonblank_margin: float = 0.0
    ctc_teacher_online_nonblank_window_loss_weight: float = 0.0
    ctc_teacher_online_nonblank_window_margin_loss_weight: float = 0.0
    ctc_teacher_online_nonblank_window_topk_loss_weight: float = 0.0
    ctc_teacher_online_nonblank_window_radius: int = 2
    ctc_teacher_online_nonblank_window_temperature: float = 0.0
    ctc_teacher_online_top_k: int = 16
    ctc_teacher_online_audio_index_path: str | None = None
    ctc_teacher_online_webdataset_index_path: str | None = None
    ctc_teacher_online_audio_cache_dir: str | None = None
    ctc_teacher_online_keep_audio_cache: bool = True
    ctc_teacher_online_device: str | None = None
    ctc_teacher_online_project_ignored_token_ids: tuple[int, ...] | list[int] = (60514,)
    ctc_teacher_online_use_batch_features: bool = False
    ctc_teacher_online_keep_layer_hiddens_on_device: bool = False
    ctc_teacher_online_keep_full_log_probs_on_device: bool = False
    ctc_teacher_online_layer_mixer_loss_weight: float = 0.0
    ctc_teacher_online_layer_ffn_loss_weight: float = 0.0
    ctc_teacher_online_layer_block_loss_weight: float = 0.0
    ctc_teacher_online_layer_normalized_mse_weight: float = 1.0
    ctc_teacher_online_layer_cosine_weight: float = 0.25
    ctc_teacher_online_layer_energy_mse_weight: float = 0.0
    ctc_teacher_online_layer_log_rms_weight: float = 0.0
    ctc_teacher_online_layer_raw_mse_weight: float = 0.0
    ctc_teacher_online_layer_sample_count: int = 8
    ctc_teacher_online_layer_boundary_ids: tuple[int, ...] | list[int] = (0, 49, 50, 69)
    ctc_teacher_online_layer_frame_tolerance: int = 0
    ctc_teacher_online_layer_input_mode: str = "stacked"
    ctc_decoder_type: str = "none"
    ctc_decoder_downsample_rate: int = 1
    ctc_decoder_dim: int | None = None
    ctc_decoder_ffn_dim: int = 2048
    ctc_decoder_num_layers: int = 5
    ctc_decoder_attention_heads: int = 8
    ctc_decoder_dropout: float = 0.0
    ctc_decoder_attention_dropout: float = 0.0
    ctc_bridge_type: str = "none"
    ctc_bridge_hidden_dim: int | None = None
    ctc_bridge_dropout: float = 0.0
    ctc_suppress_non_pronunciation_tokens: bool = False
    ctc_suppressed_token_ids: tuple[int, ...] | list[int] = ()
    funasr_nano_ctc_init_checkpoint_path: str | None = None
    funasr_nano_ctc_init_load_encoder: bool = False
    funasr_nano_ctc_init_load_encoder_attention: bool = True
    funasr_nano_ctc_init_load_rwkv_encoder_from_qkv: bool = False
    funasr_nano_ctc_init_rwkv_qkv_scale_mode: str = "exact"
    funasr_nano_ctc_init_load_decoder: bool = True
    funasr_nano_ctc_init_load_head: bool = True
    funasr_nano_ctc_teacher_blank_id: int = 60514
    funasr_nano_ctc_init_blank_bias_delta: float = 0.0
    freeze_encoder: bool = False
    freeze_encoder_except_time_mixer: bool = False
    freeze_ctc_decoder: bool = False
    freeze_ctc_head: bool = False
    direction_variant: str = "none"
    p_start: float = 0.0
    p_max: float = 0.0
    warmup_steps: int = 0
    ramp_steps: int = 0
    device: str = "cuda"
    encoder_init_checkpoint_path: str | None = None
    init_checkpoint_path: str | None = None
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
    step_eval_split: str | None = None
    step_eval_shuffle: bool = True
    step_eval_at_start: bool = False
    top_k_step_checkpoints: int = 3
    save_deepspeed_sharded_checkpoints: bool = True
    local_rank: int = -1
    log_every: int = 10
    gradient_checkpointing: bool = True
    batch_token_budget: int | None = None
    length_bucket_frame_budget: int | None = None
    target_gpu_memory_gib: float = 22.0
    skip_oversized_samples: bool = True
    specaugment_enabled: bool = False
    specaugment_time_masks: int = 2
    specaugment_time_width: int = 40
    specaugment_freq_masks: int = 2
    specaugment_freq_width: int = 15


def _maybe_load_initial_model_checkpoint(
    model: RWKVCTCModel,
    config: DeepSpeedTrainConfig,
) -> dict[str, Any] | None:
    if config.init_checkpoint_path is None:
        return None
    if config.resume_from is not None:
        _rank_zero_log(
            f"resume_from is set ({config.resume_from}); skipping init_checkpoint_path={config.init_checkpoint_path}"
        )
        return None
    restored = load_checkpoint(
        config.init_checkpoint_path,
        model=model,
        optimizer=None,
        map_location="cpu",
        strict=False,
    )
    extra = restored.get("extra", {})
    missing_keys = extra.get("missing_keys", []) if isinstance(extra, dict) else []
    unexpected_keys = extra.get("unexpected_keys", []) if isinstance(extra, dict) else []
    _rank_zero_log(
        "Loaded initial model weights from "
        f"{config.init_checkpoint_path} step={int(restored.get('step', 0))} "
        f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
    )
    return restored


def _maybe_load_funasr_nano_ctc_init_distributed(model: RWKVCTCModel, config: DeepSpeedTrainConfig) -> None:
    if config.funasr_nano_ctc_init_checkpoint_path is None:
        return
    report = model.load_funasr_nano_ctc_checkpoint(
        config.funasr_nano_ctc_init_checkpoint_path,
        load_encoder=bool(config.funasr_nano_ctc_init_load_encoder),
        load_encoder_attention=bool(config.funasr_nano_ctc_init_load_encoder_attention),
        load_rwkv_encoder_from_qkv=bool(config.funasr_nano_ctc_init_load_rwkv_encoder_from_qkv),
        rwkv_qkv_projection_scale_mode=str(config.funasr_nano_ctc_init_rwkv_qkv_scale_mode),
        load_ctc_decoder=bool(config.funasr_nano_ctc_init_load_decoder),
        load_ctc_head=bool(config.funasr_nano_ctc_init_load_head),
        teacher_blank_id=int(config.funasr_nano_ctc_teacher_blank_id),
        project_ignored_token_ids=tuple(int(value) for value in config.ctc_teacher_online_project_ignored_token_ids),
        blank_bias_delta=float(config.funasr_nano_ctc_init_blank_bias_delta),
    )
    _rank_zero_log(
        "Loaded FunASR-Nano CTC init: "
        f"path={config.funasr_nano_ctc_init_checkpoint_path} "
        f"encoder_tensors={report['encoder_loaded']} "
        f"encoder_skipped={report['encoder_skipped']} "
        f"encoder_attention_skipped={report['encoder_attention_skipped']} "
        f"rwkv_encoder_tensors={report['rwkv_encoder_loaded']} "
        f"rwkv_encoder_skipped={report['rwkv_encoder_skipped']} "
        f"rwkv_first_layer_errors={report['rwkv_encoder_first_layer_reconstruction_errors']} "
        f"rwkv_qkv_scale_mode={report['rwkv_encoder_qkv_projection_scale_mode']} "
        f"bridge_tensors={report.get('ctc_bridge_loaded', 0)} "
        f"bridge_skipped={report.get('ctc_bridge_skipped', 0)} "
        f"decoder_tensors={report['ctc_decoder_loaded']} "
        f"head_rows={report['ctc_head_loaded_rows']} "
        f"ignored_rows={report['ctc_head_ignored_rows']} "
        f"blank_bias_delta={report['ctc_head_blank_bias_delta']}"
    )


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
        if dist.get_world_size() <= 1:
            return
        if dist.get_backend() == "gloo" and hasattr(dist, "monitored_barrier"):
            dist.monitored_barrier()
            return
        dist.barrier()


def _rank_zero_log(message: str) -> None:
    if _is_rank_zero():
        print(f"[rwkvasr] {message}", flush=True)


def _all_rank_log(message: str) -> None:
    print(f"[rwkvasr][rank{_rank()}] {message}", flush=True)


def _sample_direction_mask_distributed(
    scheduler: DirectionDropoutScheduler,
    *,
    step: int,
    device: torch.device,
) -> DirectionMask:
    if not (dist.is_available() and dist.is_initialized()):
        return scheduler.sample_mask(step, device=device)

    num_layers = scheduler.config.num_layers
    if _is_rank_zero():
        sampled = scheduler.sample_mask(step, device=device)
        forward = sampled.forward.clone()
        backward = sampled.backward.clone()
    else:
        forward = torch.empty(num_layers, dtype=torch.bool, device=device)
        backward = torch.empty(num_layers, dtype=torch.bool, device=device)

    dist.broadcast(forward, src=0)
    dist.broadcast(backward, src=0)
    return DirectionMask(forward=forward, backward=backward)


def _resolve_deepspeed_resume_source(config: DeepSpeedTrainConfig) -> tuple[str | None, str | None]:
    if config.resume_from is None:
        return None, None
    if config.resume_from != "latest":
        resolved_path = Path(config.resume_from)
        if resolved_path.is_dir():
            if not config.resume_tag and resolved_path.parent.name == "ds_checkpoints":
                return str(resolved_path.parent), resolved_path.name
            if resolved_path.name and (resolved_path.name == "best" or resolved_path.name.startswith("step-")):
                return str(resolved_path), config.resume_tag
        return str(resolved_path), config.resume_tag

    latest_state = load_latest_checkpoint_state(config.output_dir)
    latest_tag = latest_state.get("resume_tag")
    latest_tag = str(latest_tag) if isinstance(latest_tag, str) and latest_tag else None
    latest_checkpoint = _resolve_latest_deepspeed_checkpoint(config.output_dir)
    latest_ds_dir = latest_state.get("deepspeed_checkpoint_dir")
    if isinstance(latest_ds_dir, str) and latest_ds_dir:
        ds_dir = Path(latest_ds_dir)
        parsed_tag = _parse_deepspeed_checkpoint_tag(ds_dir.name)
        if ds_dir.exists():
            if parsed_tag is not None:
                if latest_checkpoint is not None:
                    resolved_path, resolved_tag = latest_checkpoint
                    resolved_step = _parse_deepspeed_checkpoint_step(resolved_tag)
                    yaml_step = _parse_deepspeed_checkpoint_step(latest_tag or parsed_tag)
                    if resolved_step > yaml_step:
                        _rank_zero_log(
                            f"latest_checkpoint.yaml points to stale tag={latest_tag or parsed_tag}; "
                            f"falling back to latest DeepSpeed checkpoint {resolved_tag}"
                        )
                        return str(resolved_path.parent), resolved_tag
                if ds_dir.parent.name == "ds_checkpoints":
                    return str(ds_dir.parent), latest_tag or parsed_tag
                if ds_dir.is_dir() and (ds_dir.name == "best" or ds_dir.name.startswith("step-")):
                    return str(ds_dir), latest_tag or ds_dir.name
            if ds_dir.is_dir():
                _rank_zero_log(f"Ignoring invalid resume checkpoint directory tag: {ds_dir}")
        else:
            _rank_zero_log(f"latest_checkpoint.yaml points to missing ds_checkpoint_dir={ds_dir}; trying latest valid checkpoint.")

    if latest_checkpoint is not None:
        resolved_path, resolved_tag = latest_checkpoint
        if latest_tag is not None and latest_tag != resolved_tag:
            _rank_zero_log(
                f"latest_checkpoint.yaml requested tag={latest_tag} is unavailable; "
                f"falling back to {resolved_tag}"
            )
        return str(resolved_path.parent), resolved_tag

    raise FileNotFoundError(
        "resume_from='latest' requires latest_checkpoint.yaml with deepspeed_checkpoint_dir and resume_tag"
    )


def _skip_batches(loader_iter: Any, count: int) -> int:
    skipped = 0
    while skipped < count:
        try:
            next(loader_iter)
        except StopIteration:
            break
        skipped += 1
    return skipped


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
    zero_stage = int(zero_optimization.get("stage", 2))
    if zero_stage not in {1, 2}:
        raise ValueError(f"Only DeepSpeed ZeRO stage 1 or 2 is supported, got stage={zero_stage}.")
    zero_optimization["stage"] = zero_stage
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
    requested_bf16 = None
    if isinstance(ds_config.get("bf16"), dict) and "enabled" in ds_config["bf16"]:
        requested_bf16 = bool(ds_config["bf16"]["enabled"])
    if use_cuda:
        ds_config["bf16"] = {"enabled": True if requested_bf16 is None else requested_bf16}
        if not bool(ds_config["bf16"]["enabled"]):
            ds_config["fp16"] = {"enabled": False}
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


def _apply_training_freeze(model: RWKVCTCModel, config: DeepSpeedTrainConfig) -> None:
    frozen_names: list[str] = []
    if config.freeze_encoder and config.freeze_encoder_except_time_mixer:
        raise ValueError("freeze_encoder and freeze_encoder_except_time_mixer are mutually exclusive.")
    if config.freeze_encoder:
        for name, parameter in model.encoder.named_parameters(prefix="encoder"):
            parameter.requires_grad_(False)
            frozen_names.append(name)
    if config.freeze_encoder_except_time_mixer:
        sensevoice_encoder = getattr(model.encoder, "sensevoice_encoder", None)
        if sensevoice_encoder is None:
            raise ValueError("freeze_encoder_except_time_mixer requires frontend_type='sensevoice_rwkv'.")
        for name, parameter in model.encoder.named_parameters(prefix="encoder"):
            trainable = ".time_mixer." in name or ".input_proj." in name
            parameter.requires_grad_(trainable)
            if not trainable:
                frozen_names.append(name)
    if config.freeze_ctc_decoder and model.ctc_decoder is not None:
        for name, parameter in model.ctc_decoder.named_parameters(prefix="ctc_decoder"):
            parameter.requires_grad_(False)
            frozen_names.append(name)
    if config.freeze_ctc_head:
        for name, parameter in model.ctc_head.named_parameters(prefix="ctc_head"):
            parameter.requires_grad_(False)
            frozen_names.append(name)
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_params <= 0:
        raise ValueError("Training freeze settings left no trainable parameters.")
    if frozen_names:
        _rank_zero_log(
            "Applied parameter freeze: "
            f"freeze_encoder={bool(config.freeze_encoder)} "
            f"freeze_encoder_except_time_mixer={bool(config.freeze_encoder_except_time_mixer)} "
            f"freeze_ctc_decoder={bool(config.freeze_ctc_decoder)} "
            f"freeze_ctc_head={bool(config.freeze_ctc_head)} "
            f"frozen_tensors={len(frozen_names)} "
            f"trainable_params={trainable_params}"
        )


def _build_webdataset_config(
    config: DeepSpeedTrainConfig,
    *,
    shuffle_shards: bool,
    skip_decode_errors: bool = False,
    apply_decoder_prompt_language_label_noise: bool = False,
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
        bucket_source_interleave=config.bucket_source_interleave,
        append_eos=config.tokenizer_append_eos,
        text_normalization=config.text_normalization,
        decoder_append_eos=config.decoder_tokenizer_append_eos,
        decoder_text_normalization=config.decoder_text_normalization or config.text_normalization,
        decoder_prompt_before_audio=config.decoder_prompt_before_audio,
        decoder_prompt_before_audio_use_language=config.decoder_prompt_before_audio_use_language,
        decoder_ctc_draft_cache_path=config.decoder_ctc_draft_cache_path,
        decoder_ctc_draft_prompt_template=config.decoder_ctc_draft_prompt_template,
        decoder_ctc_draft_text_key=config.decoder_ctc_draft_text_key,
        decoder_ctc_draft_missing_policy=config.decoder_ctc_draft_missing_policy,
        decoder_ctc_draft_dropout_prob=config.decoder_ctc_draft_dropout_prob,
        decoder_ctc_draft_language_mismatch_dropout_prob=(
            config.decoder_ctc_draft_language_mismatch_dropout_prob
        ),
        decoder_ctc_draft_dropout_seed=config.decoder_ctc_draft_dropout_seed,
        ctc_label_override_cache_path=config.ctc_label_override_cache_path,
        ctc_label_override_text_key=config.ctc_label_override_text_key,
        allow_missing_targets=config.allow_missing_targets,
        decoder_target_prefix=config.decoder_target_prefix,
        decoder_target_prefix_use_language=config.decoder_target_prefix_use_language,
        decoder_language_confirmation_en=config.decoder_language_confirmation_en,
        decoder_language_confirmation_zh=config.decoder_language_confirmation_zh,
        decoder_prompt_language_label_noise_prob=(
            config.decoder_prompt_language_label_noise_prob
            if apply_decoder_prompt_language_label_noise
            else 0.0
        ),
        decoder_prompt_language_label_noise_seed=config.decoder_prompt_language_label_noise_seed,
        skip_decode_errors=skip_decode_errors,
    )


def _decoder_targets_enabled(config: DeepSpeedTrainConfig) -> bool:
    return bool(config.decoder_enabled and config.decoder_loss_weight > 0)


def _resolve_decoder_text_tokenizer(config: DeepSpeedTrainConfig):
    if not _decoder_targets_enabled(config):
        return None
    tokenizer_type = config.decoder_tokenizer_type or config.tokenizer_type
    tokenizer_model_path = config.decoder_tokenizer_model_path or config.tokenizer_model_path
    tokenizer_language = config.decoder_tokenizer_language or config.tokenizer_language
    tokenizer_task = config.decoder_tokenizer_task or config.tokenizer_task
    return build_text_tokenizer(
        tokenizer_type,
        model_path=tokenizer_model_path,
        language=tokenizer_language,
        task=tokenizer_task,
    )


def _resolve_decoder_vocab_size(config: DeepSpeedTrainConfig) -> int | None:
    if config.decoder_vocab_size is not None:
        return int(config.decoder_vocab_size)
    tokenizer = _resolve_decoder_text_tokenizer(config)
    if tokenizer is None:
        return None
    return int(tokenizer.vocab_size)


def _resolve_ctc_vocab_size_for_deepspeed(config: DeepSpeedTrainConfig) -> int:
    tokenizer = _resolve_text_tokenizer(config)
    if config.tokenizer_append_eos and tokenizer_eos_token_id(tokenizer) is None:
        raise ValueError(
            "tokenizer_append_eos=True requires a tokenizer with a defined eos_token_id. "
            f"tokenizer_type={config.tokenizer_type!r} does not provide one."
        )
    tokenizer_vocab_size = int(tokenizer.vocab_size)
    if config.vocab_size is None:
        return tokenizer_vocab_size
    if int(config.vocab_size) != tokenizer_vocab_size:
        raise ValueError(
            "Configured CTC vocab_size does not match the resolved CTC tokenizer vocabulary size: "
            f"{config.vocab_size} != {tokenizer_vocab_size}"
        )
    return int(config.vocab_size)


def _resolve_decoder_template_token_ids_for_deepspeed(config: DeepSpeedTrainConfig) -> dict[str, Any]:
    tokenizer = _resolve_decoder_text_tokenizer(config)
    if tokenizer is None:
        return _resolve_decoder_template_token_ids(config)
    prompt_before_audio = (
        ""
        if config.decoder_prompt_before_audio_use_language or config.decoder_ctc_draft_prompt_template
        else config.decoder_prompt_before_audio
    )
    return {
        "decoder_prompt_before_audio_token_ids": tuple(tokenizer.encode(prompt_before_audio)),
        "decoder_prompt_after_audio_token_ids": tuple(tokenizer.encode(config.decoder_prompt_after_audio)),
        "decoder_target_suffix_token_ids": tuple(tokenizer.encode(config.decoder_target_suffix)),
    }


def _resolved_tokenizer_config_payload_for_deepspeed(
    config: DeepSpeedTrainConfig,
    *,
    vocab_size: int,
) -> dict[str, object]:
    payload = _resolved_tokenizer_config_payload(config, vocab_size=vocab_size)
    if _decoder_targets_enabled(config):
        payload.update(
            {
                "decoder_tokenizer_type": config.decoder_tokenizer_type or config.tokenizer_type,
                "decoder_tokenizer_model_path": config.decoder_tokenizer_model_path or config.tokenizer_model_path,
                "decoder_tokenizer_language": config.decoder_tokenizer_language or config.tokenizer_language,
                "decoder_tokenizer_task": config.decoder_tokenizer_task or config.tokenizer_task,
                "decoder_tokenizer_append_eos": bool(config.decoder_tokenizer_append_eos),
                "decoder_text_normalization": config.decoder_text_normalization or config.text_normalization,
                "decoder_vocab_size": _resolve_decoder_vocab_size(config),
                "decoder_prompt_before_audio": config.decoder_prompt_before_audio,
                "decoder_prompt_before_audio_use_language": bool(config.decoder_prompt_before_audio_use_language),
                "decoder_ctc_draft_cache_path": config.decoder_ctc_draft_cache_path,
                "decoder_ctc_draft_prompt_template": config.decoder_ctc_draft_prompt_template,
                "decoder_ctc_draft_text_key": config.decoder_ctc_draft_text_key,
                "decoder_ctc_draft_missing_policy": config.decoder_ctc_draft_missing_policy,
                "decoder_ctc_draft_dropout_prob": float(config.decoder_ctc_draft_dropout_prob),
                "decoder_ctc_draft_language_mismatch_dropout_prob": float(
                    config.decoder_ctc_draft_language_mismatch_dropout_prob
                ),
                "decoder_ctc_draft_dropout_seed": int(config.decoder_ctc_draft_dropout_seed),
                "decoder_prompt_after_audio": config.decoder_prompt_after_audio,
                "decoder_target_prefix": config.decoder_target_prefix,
                "decoder_target_prefix_use_language": bool(config.decoder_target_prefix_use_language),
                "decoder_language_confirmation_en": config.decoder_language_confirmation_en,
                "decoder_language_confirmation_zh": config.decoder_language_confirmation_zh,
                "decoder_prompt_language_label_noise_prob": float(config.decoder_prompt_language_label_noise_prob),
                "decoder_prompt_language_label_noise_seed": int(config.decoder_prompt_language_label_noise_seed),
            }
        )
    return payload


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
        bucket_manifest_path = _resolve_bucket_manifest_path(
            data_path,
            config.webdataset_bucket_manifest_path,
            configured_length_index_path=config.webdataset_length_index_path,
        )
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


def _maybe_load_encoder_init_checkpoint_distributed(model: RWKVCTCModel, checkpoint_path: str | None) -> None:
    if checkpoint_path is None:
        return
    aut_encoder = getattr(model.encoder, "aut_encoder", None)
    load_fn = getattr(aut_encoder, "load_qwen3_asr_non_attention_checkpoint", None)
    if callable(load_fn):
        report = load_fn(checkpoint_path)
        _rank_zero_log(
            "Loaded AuRWKV non-attention encoder init: "
            f"path={checkpoint_path} loaded={len(report['loaded'])} skipped={len(report['skipped'])}"
        )
        return
    qwen3_transformer_encoder = getattr(model.encoder, "qwen3_transformer_encoder", None)
    load_fn = getattr(qwen3_transformer_encoder, "load_qwen3_asr_checkpoint", None)
    if callable(load_fn):
        report = load_fn(checkpoint_path)
        _rank_zero_log(
            "Loaded Qwen3 Transformer audio encoder init: "
            f"path={checkpoint_path} loaded={len(report['loaded'])} skipped={len(report['skipped'])}"
        )
        return
    sensevoice_encoder = getattr(model.encoder, "sensevoice_encoder", None)
    load_fn = getattr(sensevoice_encoder, "load_sensevoice_non_attention_checkpoint", None)
    if callable(load_fn):
        report = load_fn(checkpoint_path)
        _rank_zero_log(
            "Loaded SenseVoiceRWKV non-attention encoder init: "
            f"path={checkpoint_path} loaded={len(report['loaded'])} skipped={len(report['skipped'])}"
        )
        return
    sensevoice_conformer_encoder = getattr(model.encoder, "sensevoice_conformer_encoder", None)
    load_fn = getattr(sensevoice_conformer_encoder, "load_sensevoice_non_attention_checkpoint", None)
    if callable(load_fn):
        report = load_fn(checkpoint_path)
        _rank_zero_log(
            "Loaded SenseVoice Conformer-conv non-attention encoder init: "
            f"path={checkpoint_path} loaded={len(report['loaded'])} skipped={len(report['skipped'])}"
        )
        return
    raise ValueError(
        "encoder_init_checkpoint_path is only supported by "
        "frontend_type='aut_rwkv', 'qwen3_transformer', 'sensevoice_rwkv', "
        "or 'sensevoice_conformer_conv'."
    )


def _build_train_loader(config: DeepSpeedTrainConfig) -> tuple[DataLoader, Any | None]:
    data_source, data_path = _resolve_data_source(config)
    tokenizer = _resolve_text_tokenizer(config)
    decoder_tokenizer = _resolve_decoder_text_tokenizer(config)
    feature_extractor = build_audio_feature_extractor(
        config.feature_extractor_type,
        input_dim=int(config.input_dim),
    )
    if data_source == "manifest":
        dataset = ASRManifestDataset(
            data_path,
            tokenizer=tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            feature_extractor=feature_extractor,
            append_eos=config.tokenizer_append_eos,
            text_normalization=config.text_normalization,
            decoder_append_eos=config.decoder_tokenizer_append_eos,
            decoder_text_normalization=config.decoder_text_normalization or config.text_normalization,
            decoder_prompt_before_audio=config.decoder_prompt_before_audio,
            decoder_prompt_before_audio_use_language=config.decoder_prompt_before_audio_use_language,
            decoder_ctc_draft_cache_path=config.decoder_ctc_draft_cache_path,
            decoder_ctc_draft_prompt_template=config.decoder_ctc_draft_prompt_template,
            decoder_ctc_draft_text_key=config.decoder_ctc_draft_text_key,
            decoder_ctc_draft_missing_policy=config.decoder_ctc_draft_missing_policy,
            decoder_ctc_draft_dropout_prob=config.decoder_ctc_draft_dropout_prob,
            decoder_ctc_draft_language_mismatch_dropout_prob=(
                config.decoder_ctc_draft_language_mismatch_dropout_prob
            ),
            decoder_ctc_draft_dropout_seed=config.decoder_ctc_draft_dropout_seed,
            decoder_target_prefix=config.decoder_target_prefix,
            decoder_target_prefix_use_language=config.decoder_target_prefix_use_language,
            decoder_language_confirmation_en=config.decoder_language_confirmation_en,
            decoder_language_confirmation_zh=config.decoder_language_confirmation_zh,
            decoder_prompt_language_label_noise_prob=config.decoder_prompt_language_label_noise_prob,
            decoder_prompt_language_label_noise_seed=config.decoder_prompt_language_label_noise_seed,
        )
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

    webdataset_config = _build_webdataset_config(
        config,
        shuffle_shards=True,
        skip_decode_errors=True,
        apply_decoder_prompt_language_label_noise=True,
    )
    bucket_manifest_path = _resolve_bucket_manifest_path(
        data_path,
        config.webdataset_bucket_manifest_path,
        configured_length_index_path=config.webdataset_length_index_path,
    )
    if bucket_manifest_path is not None:
        loader = build_bucketed_webdataset_loader(
            data_path,
            bucket_manifest_path=bucket_manifest_path,
            tokenizer=tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            feature_extractor=feature_extractor,
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
            decoder_tokenizer=decoder_tokenizer,
            feature_extractor=feature_extractor,
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
        decoder_tokenizer=decoder_tokenizer,
        feature_extractor=feature_extractor,
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
        dataset = ASRManifestDataset(
            data_path,
            tokenizer=_resolve_text_tokenizer(config),
            decoder_tokenizer=_resolve_decoder_text_tokenizer(config),
            feature_extractor=build_audio_feature_extractor(
                config.feature_extractor_type,
                input_dim=int(config.input_dim),
            ),
            append_eos=config.tokenizer_append_eos,
            text_normalization=config.text_normalization,
            decoder_append_eos=config.decoder_tokenizer_append_eos,
            decoder_text_normalization=config.decoder_text_normalization or config.text_normalization,
            decoder_prompt_before_audio=config.decoder_prompt_before_audio,
            decoder_prompt_before_audio_use_language=config.decoder_prompt_before_audio_use_language,
            decoder_ctc_draft_cache_path=config.decoder_ctc_draft_cache_path,
            decoder_ctc_draft_prompt_template=config.decoder_ctc_draft_prompt_template,
            decoder_ctc_draft_text_key=config.decoder_ctc_draft_text_key,
            decoder_ctc_draft_missing_policy=config.decoder_ctc_draft_missing_policy,
            decoder_ctc_draft_dropout_prob=config.decoder_ctc_draft_dropout_prob,
            decoder_ctc_draft_language_mismatch_dropout_prob=(
                config.decoder_ctc_draft_language_mismatch_dropout_prob
            ),
            decoder_ctc_draft_dropout_seed=config.decoder_ctc_draft_dropout_seed,
            decoder_target_prefix=config.decoder_target_prefix,
            decoder_target_prefix_use_language=config.decoder_target_prefix_use_language,
            decoder_language_confirmation_en=config.decoder_language_confirmation_en,
            decoder_language_confirmation_zh=config.decoder_language_confirmation_zh,
            decoder_prompt_language_label_noise_prob=config.decoder_prompt_language_label_noise_prob,
            decoder_prompt_language_label_noise_seed=config.decoder_prompt_language_label_noise_seed,
        )
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
    decoder_tokenizer = _resolve_decoder_text_tokenizer(config)
    feature_extractor = build_audio_feature_extractor(
        config.feature_extractor_type,
        input_dim=int(config.input_dim),
    )
    webdataset_config = _build_webdataset_config(config, shuffle_shards=shuffle_shards)
    eval_split = "eval"
    if step_subset and config.step_eval_split is not None:
        eval_split = str(config.step_eval_split)
    webdataset_config = WebDatasetConfig(
        **{
            **webdataset_config.__dict__,
            "split": eval_split,
            "ctc_label_override_cache_path": None,
        }
    )
    bucket_manifest_path = _resolve_bucket_manifest_path(
        data_path,
        config.webdataset_bucket_manifest_path,
        configured_length_index_path=config.webdataset_length_index_path,
    )
    if bucket_manifest_path is not None:
        loader = build_bucketed_webdataset_loader(
            data_path,
            bucket_manifest_path=bucket_manifest_path,
            tokenizer=tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            feature_extractor=feature_extractor,
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
            decoder_tokenizer=decoder_tokenizer,
            feature_extractor=feature_extractor,
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
        decoder_tokenizer=decoder_tokenizer,
        feature_extractor=feature_extractor,
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


def _select_layer_hidden_ids(
    *,
    step: int,
    num_layers: int,
    sample_count: int,
    boundary_ids: tuple[int, ...] | list[int],
    include_boundaries: bool = False,
) -> tuple[int, ...]:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    if sample_count <= 0:
        raise ValueError(f"layer hidden sample_count must be positive, got {sample_count}.")
    anchors = (
        tuple(sorted({int(value) for value in boundary_ids if 0 <= int(value) < num_layers}))
        if include_boundaries
        else ()
    )
    target_count = min(num_layers, max(sample_count, len(anchors)))
    remaining = [layer_id for layer_id in range(num_layers) if layer_id not in anchors]
    extra_count = target_count - len(anchors)
    if extra_count <= 0 or not remaining:
        return anchors[:target_count]
    cursor = int(step) * max(extra_count, 1) % len(remaining)
    extras = tuple(remaining[(cursor + offset) % len(remaining)] for offset in range(extra_count))
    return tuple(sorted((*anchors, *extras)))


def _select_eval_layer_hidden_ids(
    *,
    batch_index: int,
    num_layers: int,
    sample_count: int,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[int, ...]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")
    global_batch_index = int(batch_index) * int(world_size) + int(rank)
    return _select_layer_hidden_ids(
        step=global_batch_index,
        num_layers=num_layers,
        sample_count=sample_count,
        boundary_ids=(),
        include_boundaries=False,
    )


def _teacher_layer_capture_ids(
    layer_ids: tuple[int, ...],
    *,
    input_mode: str,
) -> tuple[int, ...]:
    if input_mode == "stacked":
        return layer_ids
    if input_mode == "teacher_forced":
        return tuple(sorted({0, *layer_ids}))
    raise ValueError(
        f"Unsupported ctc_teacher_online_layer_input_mode={input_mode!r}; "
        "expected 'stacked' or 'teacher_forced'."
    )


def _first_module_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Expected tensor module output, got {type(value)!r}.")


@contextmanager
def _capture_student_sensevoice_layer_hiddens(
    model: RWKVCTCModel,
    layer_ids: tuple[int, ...],
) -> Iterator[dict[int, dict[str, torch.Tensor]]]:
    sensevoice_encoder = getattr(model.encoder, "sensevoice_encoder", None)
    if sensevoice_encoder is None:
        raise ValueError("Layer hidden distillation requires frontend_type='sensevoice_rwkv'.")
    layers = sensevoice_encoder.layers
    invalid = [layer_id for layer_id in layer_ids if not 0 <= int(layer_id) < len(layers)]
    if invalid:
        raise ValueError(f"Student encoder layer ids are out of range: {invalid}; layers={len(layers)}")

    captured: dict[int, dict[str, torch.Tensor]] = {}
    handles: list[Any] = []
    for layer_id in layer_ids:
        layer_capture: dict[str, torch.Tensor] = {}
        captured[int(layer_id)] = layer_capture
        layer = layers[int(layer_id)]

        def capture_mixer(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, torch.Tensor] = layer_capture,
        ) -> None:
            target["mixer"] = _first_module_tensor(output)

        def capture_ffn(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, torch.Tensor] = layer_capture,
        ) -> None:
            target["ffn"] = _first_module_tensor(output)

        def capture_block(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, torch.Tensor] = layer_capture,
        ) -> None:
            target["block"] = _first_module_tensor(output)

        handles.append(layer.time_mixer.register_forward_hook(capture_mixer))
        handles.append(layer.feed_forward.register_forward_hook(capture_ffn))
        handles.append(layer.register_forward_hook(capture_block))

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _teacher_forced_hidden_batch(
    teacher_records: dict[str, dict[str, Any]],
    utt_ids: tuple[str, ...] | list[str],
    *,
    layer_id: int,
    component: str,
    device: torch.device,
    dtype: torch.dtype,
    missing_policy: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    rows: list[torch.Tensor | None] = []
    feature_dim: int | None = None
    for utt_id_value in utt_ids:
        utt_id = str(utt_id_value)
        record = teacher_records.get(utt_id)
        teacher_layers = record.get("encoder_layer_hiddens") if isinstance(record, dict) else None
        layer_components = teacher_layers.get(str(layer_id)) if isinstance(teacher_layers, dict) else None
        value = layer_components.get(component) if isinstance(layer_components, dict) else None
        if value is None:
            if missing_policy == "error":
                raise ValueError(
                    f"Missing teacher-forced {component} for utt_id={utt_id!r} layer={layer_id}."
                )
            rows.append(None)
            continue
        tensor = torch.as_tensor(value)
        if tensor.ndim != 2:
            raise ValueError(
                f"Teacher-forced {component} must be [T, D] for utt_id={utt_id!r} "
                f"layer={layer_id}, got {tuple(tensor.shape)}."
            )
        current_dim = int(tensor.size(-1))
        if feature_dim is None:
            feature_dim = current_dim
        elif current_dim != feature_dim:
            raise ValueError(
                f"Teacher-forced {component} dimension mismatch at layer={layer_id}: "
                f"{current_dim} != {feature_dim}."
            )
        rows.append(tensor)
    if feature_dim is None:
        raise RuntimeError(f"No teacher-forced {component} tensors matched layer={layer_id}.")
    row_lengths = tuple(0 if row is None else int(row.size(0)) for row in rows)
    lengths = torch.tensor(
        row_lengths,
        device=device,
        dtype=torch.long,
    )
    max_time = max(max(row_lengths, default=0), 1)
    batch = torch.zeros((len(rows), max_time, feature_dim), device=device, dtype=dtype)
    for sample_idx, row in enumerate(rows):
        if row is None or int(row.size(0)) <= 0:
            continue
        row_on_device = row.to(device=device, dtype=dtype)
        batch[sample_idx, : int(row_on_device.size(0)), :] = row_on_device
    return batch, lengths, row_lengths


def _teacher_forced_student_layer_hiddens(
    model: RWKVCTCModel,
    teacher_records: dict[str, dict[str, Any]],
    utt_ids: tuple[str, ...] | list[str] | None,
    *,
    layer_ids: tuple[int, ...],
    missing_policy: str,
) -> tuple[dict[int, dict[str, torch.Tensor]], torch.Tensor]:
    if utt_ids is None:
        raise RuntimeError("Teacher-forced layer alignment requires batch.utt_ids.")
    sensevoice_encoder = getattr(model.encoder, "sensevoice_encoder", None)
    if sensevoice_encoder is None:
        raise ValueError("Teacher-forced layer alignment requires frontend_type='sensevoice_rwkv'.")
    if not layer_ids:
        raise ValueError("Teacher-forced layer alignment requires at least one layer id.")
    invalid = [layer_id for layer_id in layer_ids if not 0 <= int(layer_id) < len(sensevoice_encoder.layers)]
    if invalid:
        raise ValueError(f"Teacher-forced student layer ids are out of range: {invalid}.")
    parameter = next(sensevoice_encoder.parameters())
    device = parameter.device
    dtype = parameter.dtype

    layer_zero_input, layer_zero_lengths, layer_zero_row_lengths = _teacher_forced_hidden_batch(
        teacher_records,
        utt_ids,
        layer_id=0,
        component="input",
        device=device,
        dtype=dtype,
        missing_policy=missing_policy,
    )
    layer_zero = sensevoice_encoder.layers[0]
    layer_zero_mixed_input = layer_zero.norm1(layer_zero_input)
    if layer_zero.input_proj is not None:
        layer_zero_mixed_input = layer_zero.input_proj(layer_zero_mixed_input)
    layer_zero_mixed, v_first, _ = layer_zero.time_mixer(
        layer_zero_mixed_input,
        lengths=layer_zero_lengths,
    )

    outputs: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id in layer_ids:
        layer = sensevoice_encoder.layers[int(layer_id)]
        if int(layer_id) == 0:
            layer_input = layer_zero_input
            lengths = layer_zero_lengths
            mixed = layer_zero_mixed
        else:
            layer_input, lengths, row_lengths = _teacher_forced_hidden_batch(
                teacher_records,
                utt_ids,
                layer_id=int(layer_id),
                component="input",
                device=device,
                dtype=dtype,
                missing_policy=missing_policy,
            )
            if row_lengths != layer_zero_row_lengths:
                raise ValueError(
                    f"Teacher-forced layer lengths differ between layer 0 and layer {layer_id}."
                )
            mixed_input = layer.norm1(layer_input)
            if layer.input_proj is not None:
                mixed_input = layer.input_proj(mixed_input)
            mixed, _, _ = layer.time_mixer(
                mixed_input,
                v_first=v_first,
                lengths=lengths,
            )
        post_mixer = mixed if layer.input_dim != layer.hidden_dim else layer_input + mixed
        ffn = layer.feed_forward(layer.norm2(post_mixer))
        outputs[int(layer_id)] = {
            "mixer": mixed,
            "ffn": ffn,
            "block": post_mixer + ffn,
        }
    return outputs, layer_zero_lengths


@dataclass(frozen=True)
class _LayerHiddenDistillationResult:
    loss: torch.Tensor
    component_losses: dict[str, torch.Tensor]
    layer_losses: dict[int, torch.Tensor]
    component_energy_mse: dict[str, float]
    component_log_rms: dict[str, float]
    component_student_rms: dict[str, float]
    component_teacher_rms: dict[str, float]
    component_rms_ratio: dict[str, float]
    component_cosine: dict[str, float]
    layer_energy_mse: dict[int, float]
    layer_log_rms: dict[int, float]
    layer_student_rms: dict[int, float]
    layer_teacher_rms: dict[int, float]
    layer_rms_ratio: dict[int, float]
    layer_cosine: dict[int, float]
    layer_frames: dict[int, float]
    layer_elements: dict[int, float]
    matched_samples: int
    missing_samples: int
    events: int
    max_frame_delta: int


def _is_layer_hidden_only_objective(config: DeepSpeedTrainConfig) -> bool:
    layer_enabled = any(
        float(value) > 0.0
        for value in (
            config.ctc_teacher_online_layer_mixer_loss_weight,
            config.ctc_teacher_online_layer_ffn_loss_weight,
            config.ctc_teacher_online_layer_block_loss_weight,
        )
    )
    non_layer_weights = (
        config.ctc_loss_weight,
        config.decoder_loss_weight,
        config.encoder_anchor_loss_weight,
        config.ctc_logit_anchor_loss_weight,
        config.ctc_teacher_topk_loss_weight,
        config.ctc_teacher_topk_blank_loss_weight,
        config.ctc_teacher_topk_mass_loss_weight,
        config.ctc_teacher_online_loss_weight,
        config.ctc_teacher_online_blank_loss_weight,
        config.ctc_teacher_online_mass_loss_weight,
        config.ctc_teacher_online_full_loss_weight,
        config.ctc_teacher_online_encoder_loss_weight,
        config.ctc_teacher_online_sequence_loss_weight,
        config.ctc_teacher_online_sequence_presence_loss_weight,
        config.ctc_teacher_online_sequence_window_loss_weight,
        config.ctc_teacher_online_nonblank_hard_loss_weight,
        config.ctc_teacher_online_nonblank_margin_loss_weight,
        config.ctc_teacher_online_nonblank_window_loss_weight,
        config.ctc_teacher_online_nonblank_window_margin_loss_weight,
        config.ctc_teacher_online_nonblank_window_topk_loss_weight,
    )
    return layer_enabled and all(float(value) <= 0.0 for value in non_layer_weights)


def _ctc_teacher_layer_hidden_loss(
    student_hiddens: dict[int, dict[str, torch.Tensor]],
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_records: dict[str, dict[str, Any]],
    *,
    layer_ids: tuple[int, ...],
    component_weights: dict[str, float],
    normalized_mse_weight: float,
    cosine_weight: float,
    energy_mse_weight: float,
    log_rms_weight: float,
    raw_mse_weight: float,
    frame_tolerance: int,
    missing_policy: str,
) -> _LayerHiddenDistillationResult:
    if utt_ids is None:
        raise RuntimeError("Layer hidden distillation requires batch.utt_ids.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported missing_policy={missing_policy!r}; expected skip or error.")
    if frame_tolerance < 0:
        raise ValueError(f"frame_tolerance must be >= 0, got {frame_tolerance}.")
    metric_weights = (
        float(normalized_mse_weight),
        float(cosine_weight),
        float(energy_mse_weight),
        float(log_rms_weight),
        float(raw_mse_weight),
    )
    if any(value < 0.0 for value in metric_weights):
        raise ValueError(f"Layer hidden metric weights must be non-negative, got {metric_weights}.")
    active_components = {
        name: float(component_weights.get(name, 0.0))
        for name in ("mixer", "ffn", "block")
        if float(component_weights.get(name, 0.0)) > 0.0
    }
    if not active_components:
        raise ValueError("Layer hidden distillation requires at least one positive component weight.")

    reference = next(
        (
            components[name]
            for components in student_hiddens.values()
            for name in active_components
            if isinstance(components.get(name), torch.Tensor)
        ),
        None,
    )
    if reference is None:
        raise RuntimeError("Student layer hooks did not capture any requested hidden tensors.")
    batch_size = min(int(reference.size(0)), len(utt_ids))
    max_student_time = int(reference.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=reference.device,
        )
    else:
        clipped_lengths = student_lengths[:batch_size].to(device=reference.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    component_sums = {name: reference.new_zeros((), dtype=torch.float32) for name in active_components}
    component_denoms = {name: reference.new_zeros((), dtype=torch.float32) for name in active_components}
    layer_sums = {layer_id: reference.new_zeros((), dtype=torch.float32) for layer_id in layer_ids}
    layer_denoms = {layer_id: reference.new_zeros((), dtype=torch.float32) for layer_id in layer_ids}
    stat_keys = ("student_sq", "teacher_sq", "elements", "energy", "log_rms", "cosine", "frames")
    component_stats = {
        name: {key: reference.new_zeros((), dtype=torch.float32) for key in stat_keys}
        for name in active_components
    }
    layer_stats = {
        layer_id: {key: reference.new_zeros((), dtype=torch.float32) for key in stat_keys}
        for layer_id in layer_ids
    }
    matched_ids: set[str] = set()
    missing_ids: set[str] = set()
    events = 0
    max_frame_delta = 0

    student_times = tuple(
        int(value) for value in clipped_lengths[:batch_size].detach().cpu().tolist()
    )
    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_records.get(utt_id)
        if record is None:
            missing_ids.add(utt_id)
            if missing_policy == "error":
                raise KeyError(f"Missing Nano layer hidden record for utt_id={utt_id!r}.")
            continue
        teacher_layers = record.get("encoder_layer_hiddens")
        if not isinstance(teacher_layers, dict):
            missing_ids.add(utt_id)
            if missing_policy == "error":
                raise ValueError(f"Nano record has no encoder_layer_hiddens for utt_id={utt_id!r}.")
            continue
        student_time = student_times[sample_idx]
        if student_time <= 0:
            continue

        sample_had_event = False
        for layer_id in layer_ids:
            student_components = student_hiddens.get(layer_id)
            teacher_components = teacher_layers.get(str(layer_id))
            if not isinstance(student_components, dict) or not isinstance(teacher_components, dict):
                if missing_policy == "error":
                    raise ValueError(f"Missing layer={layer_id} hidden tensors for utt_id={utt_id!r}.")
                continue
            for component, component_weight in active_components.items():
                student_hidden = student_components.get(component)
                teacher_hidden_raw = teacher_components.get(component)
                if not isinstance(student_hidden, torch.Tensor) or teacher_hidden_raw is None:
                    if missing_policy == "error":
                        raise ValueError(
                            f"Missing {component} hidden tensor at layer={layer_id} for utt_id={utt_id!r}."
                        )
                    continue
                teacher_hidden = torch.as_tensor(teacher_hidden_raw, dtype=torch.float32)
                if teacher_hidden.ndim != 2 or int(student_hidden.size(-1)) != int(teacher_hidden.size(-1)):
                    raise ValueError(
                        f"Layer hidden shape mismatch for utt_id={utt_id!r} layer={layer_id} "
                        f"component={component}: student={tuple(student_hidden.shape)} "
                        f"teacher={tuple(teacher_hidden.shape)}"
                    )
                teacher_time = int(teacher_hidden.size(0))
                frame_delta = abs(student_time - teacher_time)
                max_frame_delta = max(max_frame_delta, frame_delta)
                if frame_delta > frame_tolerance:
                    raise ValueError(
                        f"Layer hidden frame mismatch for utt_id={utt_id!r} layer={layer_id}: "
                        f"student={student_time} teacher={teacher_time} tolerance={frame_tolerance}."
                    )
                aligned_time = min(student_time, teacher_time)
                if aligned_time <= 0:
                    continue
                student_slice = student_hidden[sample_idx, :aligned_time, :].float()
                teacher_slice = teacher_hidden[:aligned_time].to(device=student_slice.device)
                frame_loss = student_slice.new_zeros((aligned_time,), dtype=torch.float32)
                cosine = F.cosine_similarity(student_slice, teacher_slice, dim=-1, eps=1.0e-6)
                student_power = student_slice.square().mean(dim=-1)
                teacher_power = teacher_slice.square().mean(dim=-1)
                diff_power = (student_slice - teacher_slice).square().mean(dim=-1)
                energy_mse = 2.0 * diff_power / (student_power + teacher_power + 1.0e-6)
                log_rms_delta = 0.5 * (
                    torch.log(student_power + 1.0e-6) - torch.log(teacher_power + 1.0e-6)
                )
                log_rms = F.smooth_l1_loss(
                    log_rms_delta,
                    torch.zeros_like(log_rms_delta),
                    reduction="none",
                    beta=0.25,
                )
                if normalized_mse_weight > 0.0:
                    student_norm = F.layer_norm(student_slice, (int(student_slice.size(-1)),))
                    teacher_norm = F.layer_norm(teacher_slice, (int(teacher_slice.size(-1)),))
                    frame_loss = frame_loss + F.mse_loss(
                        student_norm,
                        teacher_norm,
                        reduction="none",
                    ).mean(dim=-1) * float(normalized_mse_weight)
                if cosine_weight > 0.0:
                    frame_loss = frame_loss + (1.0 - cosine) * float(cosine_weight)
                if energy_mse_weight > 0.0:
                    frame_loss = frame_loss + energy_mse * float(energy_mse_weight)
                if log_rms_weight > 0.0:
                    frame_loss = frame_loss + log_rms * float(log_rms_weight)
                if raw_mse_weight > 0.0:
                    frame_loss = frame_loss + F.mse_loss(
                        student_slice,
                        teacher_slice,
                        reduction="none",
                    ).mean(dim=-1) * float(raw_mse_weight)
                event_sum = frame_loss.sum()
                event_frames = frame_loss.new_tensor(float(aligned_time))
                component_sums[component] = component_sums[component] + event_sum
                component_denoms[component] = component_denoms[component] + event_frames
                layer_sums[layer_id] = layer_sums[layer_id] + event_sum * component_weight
                layer_denoms[layer_id] = layer_denoms[layer_id] + event_frames * component_weight
                detached_student_sq = student_slice.detach().square().sum()
                detached_teacher_sq = teacher_slice.detach().square().sum()
                detached_elements = student_slice.new_tensor(float(student_slice.numel()))
                detached_energy = energy_mse.detach().sum()
                detached_log_rms = log_rms.detach().sum()
                detached_cosine = cosine.detach().sum()
                detached_frames = student_slice.new_tensor(float(aligned_time))
                component_stat = component_stats[component]
                component_stat["student_sq"] += detached_student_sq
                component_stat["teacher_sq"] += detached_teacher_sq
                component_stat["elements"] += detached_elements
                component_stat["energy"] += detached_energy
                component_stat["log_rms"] += detached_log_rms
                component_stat["cosine"] += detached_cosine
                component_stat["frames"] += detached_frames
                layer_stat = layer_stats[layer_id]
                layer_stat["student_sq"] += detached_student_sq * component_weight
                layer_stat["teacher_sq"] += detached_teacher_sq * component_weight
                layer_stat["elements"] += detached_elements * component_weight
                layer_stat["energy"] += detached_energy * component_weight
                layer_stat["log_rms"] += detached_log_rms * component_weight
                layer_stat["cosine"] += detached_cosine * component_weight
                layer_stat["frames"] += detached_frames * component_weight
                events += 1
                sample_had_event = True
        if sample_had_event:
            matched_ids.add(utt_id)

    if events <= 0:
        raise RuntimeError("Layer hidden distillation produced no matched layer/component events.")
    component_losses = {
        name: component_sums[name] / component_denoms[name].clamp_min(1.0)
        for name in active_components
    }
    layer_denom_values = torch.stack(
        [layer_denoms[layer_id] for layer_id in layer_ids]
    ).detach().cpu().tolist()
    layer_losses = {
        layer_id: layer_sums[layer_id] / layer_denoms[layer_id].clamp_min(1.0)
        for layer_id, denom in zip(layer_ids, layer_denom_values, strict=True)
        if float(denom) > 0.0
    }
    total = sum(
        component_losses[name] * component_weight
        for name, component_weight in active_components.items()
    )

    def stats_as_floats(stats: dict[Any, dict[str, torch.Tensor]]) -> dict[Any, dict[str, float]]:
        labels = list(stats)
        if not labels:
            return {}
        values = torch.stack(
            [torch.stack([stats[label][key] for key in stat_keys]) for label in labels]
        ).detach().cpu().tolist()
        return {
            label: {key: float(value) for key, value in zip(stat_keys, row, strict=True)}
            for label, row in zip(labels, values, strict=True)
        }

    component_stats_float = stats_as_floats(component_stats)
    layer_stats_float = stats_as_floats(layer_stats)

    def rms(stat: dict[str, float], key: str) -> float:
        return math.sqrt(max(stat[key], 0.0) / max(stat["elements"], 1.0))

    component_student_rms = {
        name: rms(stat, "student_sq") for name, stat in component_stats_float.items()
    }
    component_teacher_rms = {
        name: rms(stat, "teacher_sq") for name, stat in component_stats_float.items()
    }
    layer_student_rms = {
        layer_id: rms(stat, "student_sq") for layer_id, stat in layer_stats_float.items()
    }
    layer_teacher_rms = {
        layer_id: rms(stat, "teacher_sq") for layer_id, stat in layer_stats_float.items()
    }
    return _LayerHiddenDistillationResult(
        loss=total,
        component_losses=component_losses,
        layer_losses=layer_losses,
        component_energy_mse={
            name: stat["energy"] / max(stat["frames"], 1.0)
            for name, stat in component_stats_float.items()
        },
        component_log_rms={
            name: stat["log_rms"] / max(stat["frames"], 1.0)
            for name, stat in component_stats_float.items()
        },
        component_student_rms=component_student_rms,
        component_teacher_rms=component_teacher_rms,
        component_rms_ratio={
            name: component_student_rms[name] / max(component_teacher_rms[name], 1.0e-12)
            for name in component_stats_float
        },
        component_cosine={
            name: stat["cosine"] / max(stat["frames"], 1.0)
            for name, stat in component_stats_float.items()
        },
        layer_energy_mse={
            layer_id: stat["energy"] / max(stat["frames"], 1.0)
            for layer_id, stat in layer_stats_float.items()
            if stat["frames"] > 0.0
        },
        layer_log_rms={
            layer_id: stat["log_rms"] / max(stat["frames"], 1.0)
            for layer_id, stat in layer_stats_float.items()
            if stat["frames"] > 0.0
        },
        layer_student_rms={
            layer_id: value
            for layer_id, value in layer_student_rms.items()
            if layer_stats_float[layer_id]["elements"] > 0.0
        },
        layer_teacher_rms={
            layer_id: value
            for layer_id, value in layer_teacher_rms.items()
            if layer_stats_float[layer_id]["elements"] > 0.0
        },
        layer_rms_ratio={
            layer_id: layer_student_rms[layer_id] / max(layer_teacher_rms[layer_id], 1.0e-12)
            for layer_id in layer_stats_float
            if layer_stats_float[layer_id]["elements"] > 0.0
        },
        layer_cosine={
            layer_id: stat["cosine"] / max(stat["frames"], 1.0)
            for layer_id, stat in layer_stats_float.items()
            if stat["frames"] > 0.0
        },
        layer_frames={
            layer_id: stat["frames"]
            for layer_id, stat in layer_stats_float.items()
            if stat["frames"] > 0.0
        },
        layer_elements={
            layer_id: stat["elements"]
            for layer_id, stat in layer_stats_float.items()
            if stat["elements"] > 0.0
        },
        matched_samples=len(matched_ids),
        missing_samples=len(missing_ids),
        events=events,
        max_frame_delta=max_frame_delta,
    )


_LAYER_EVAL_LOSS_SUM = 0
_LAYER_EVAL_ENERGY_SUM = 1
_LAYER_EVAL_LOG_RMS_SUM = 2
_LAYER_EVAL_COSINE_SUM = 3
_LAYER_EVAL_FRAME_WEIGHT = 4
_LAYER_EVAL_STUDENT_SQ_SUM = 5
_LAYER_EVAL_TEACHER_SQ_SUM = 6
_LAYER_EVAL_ELEMENT_WEIGHT = 7
_LAYER_EVAL_WIDTH = 8


def _accumulate_layer_eval_metrics(
    accumulator: torch.Tensor,
    result: _LayerHiddenDistillationResult,
) -> None:
    layer_ids = tuple(result.layer_losses)
    if not layer_ids:
        return
    loss_values = torch.stack(
        [result.layer_losses[layer_id] for layer_id in layer_ids]
    ).detach().cpu().tolist()
    for layer_id, loss_value in zip(layer_ids, loss_values, strict=True):
        if not 0 <= int(layer_id) < int(accumulator.size(0)):
            raise ValueError(f"Layer eval metric id is out of range: {layer_id}")
        frame_weight = float(result.layer_frames.get(layer_id, 0.0))
        element_weight = float(result.layer_elements.get(layer_id, 0.0))
        if frame_weight <= 0.0 or element_weight <= 0.0:
            continue
        student_rms = float(result.layer_student_rms[layer_id])
        teacher_rms = float(result.layer_teacher_rms[layer_id])
        row = accumulator[int(layer_id)]
        row[_LAYER_EVAL_LOSS_SUM] += float(loss_value) * frame_weight
        row[_LAYER_EVAL_ENERGY_SUM] += float(result.layer_energy_mse[layer_id]) * frame_weight
        row[_LAYER_EVAL_LOG_RMS_SUM] += float(result.layer_log_rms[layer_id]) * frame_weight
        row[_LAYER_EVAL_COSINE_SUM] += float(result.layer_cosine[layer_id]) * frame_weight
        row[_LAYER_EVAL_FRAME_WEIGHT] += frame_weight
        row[_LAYER_EVAL_STUDENT_SQ_SUM] += student_rms * student_rms * element_weight
        row[_LAYER_EVAL_TEACHER_SQ_SUM] += teacher_rms * teacher_rms * element_weight
        row[_LAYER_EVAL_ELEMENT_WEIGHT] += element_weight


def _finalize_layer_eval_metrics(
    accumulator: torch.Tensor,
    *,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    reduced = accumulator.to(device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    rows = reduced.cpu().tolist()
    metrics: dict[int, dict[str, float]] = {}
    for layer_id, row in enumerate(rows):
        frame_weight = float(row[_LAYER_EVAL_FRAME_WEIGHT])
        element_weight = float(row[_LAYER_EVAL_ELEMENT_WEIGHT])
        if frame_weight <= 0.0 or element_weight <= 0.0:
            continue
        student_rms = math.sqrt(max(float(row[_LAYER_EVAL_STUDENT_SQ_SUM]), 0.0) / element_weight)
        teacher_rms = math.sqrt(max(float(row[_LAYER_EVAL_TEACHER_SQ_SUM]), 0.0) / element_weight)
        metrics[layer_id] = {
            "loss": float(row[_LAYER_EVAL_LOSS_SUM]) / frame_weight,
            "energy_mse": float(row[_LAYER_EVAL_ENERGY_SUM]) / frame_weight,
            "log_rms": float(row[_LAYER_EVAL_LOG_RMS_SUM]) / frame_weight,
            "cosine": float(row[_LAYER_EVAL_COSINE_SUM]) / frame_weight,
            "student_rms": student_rms,
            "teacher_rms": teacher_rms,
            "rms_ratio": student_rms / max(teacher_rms, 1.0e-12),
            "frame_weight": frame_weight,
            "element_weight": element_weight,
        }
    return metrics


def _online_ctc_teacher_distillation_loss(
    *,
    config: DeepSpeedTrainConfig,
    losses: dict[str, Any],
    batch: Any,
    ctc_teacher_online_records: dict[str, dict[str, Any]],
    full_eval_accumulator: torch.Tensor | None = None,
) -> torch.Tensor:
    total = losses.get("loss")
    if not isinstance(total, torch.Tensor):
        raise RuntimeError("joint_losses did not return a tensor loss.")

    ctc_teacher_frame_filter = str(config.ctc_teacher_frame_filter or "all")
    ctc_teacher_online_full_frame_filter = (
        ctc_teacher_frame_filter
        if config.ctc_teacher_online_full_frame_filter is None
        else str(config.ctc_teacher_online_full_frame_filter)
    )
    ignored_token_ids = tuple(int(value) for value in config.ctc_teacher_online_project_ignored_token_ids)

    def student_logits_and_lengths() -> tuple[torch.Tensor, torch.Tensor | None]:
        student_logits = losses.get("logits")
        if not isinstance(student_logits, torch.Tensor):
            raise RuntimeError("joint_losses did not return logits tensor for online CTC teacher distillation.")
        student_lengths = losses.get("logit_lengths")
        if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
            raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
        return student_logits, student_lengths

    def student_encoded_and_lengths() -> tuple[torch.Tensor, torch.Tensor | None]:
        student_encoded = losses.get("ctc_encoded", losses.get("encoded"))
        if not isinstance(student_encoded, torch.Tensor):
            raise RuntimeError("joint_losses did not return encoded tensor for online CTC teacher distillation.")
        student_lengths = losses.get("ctc_encoded_lengths", losses.get("encoded_lengths"))
        if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
            raise RuntimeError("joint_losses returned non-tensor encoded_lengths.")
        return student_encoded, student_lengths

    online_weight = float(config.ctc_teacher_online_loss_weight)
    if online_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        loss_value, _, _ = _ctc_teacher_topk_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            time_map=config.ctc_teacher_topk_time_map,
            frame_filter=ctc_teacher_frame_filter,
            frame_filter_neighbor_radius=int(config.ctc_teacher_frame_filter_neighbor_radius),
            frame_filter_min_nonblank_prob=float(config.ctc_teacher_frame_filter_min_nonblank_prob),
            missing_policy=config.ctc_teacher_topk_missing_policy,
        )
        total = total + loss_value * online_weight

    blank_weight = float(config.ctc_teacher_online_blank_loss_weight)
    if blank_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        loss_value, _, _ = _ctc_teacher_blank_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            time_map=config.ctc_teacher_topk_time_map,
            missing_policy=config.ctc_teacher_topk_missing_policy,
        )
        total = total + loss_value * blank_weight

    mass_weight = float(config.ctc_teacher_online_mass_loss_weight)
    if mass_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        loss_value, _, _ = _ctc_teacher_mass_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            time_map=config.ctc_teacher_topk_time_map,
            missing_policy=config.ctc_teacher_topk_missing_policy,
        )
        total = total + loss_value * mass_weight

    full_weight = float(config.ctc_teacher_online_full_loss_weight)
    if full_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        loss_value, _, _ = _ctc_teacher_full_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            time_map=config.ctc_teacher_topk_time_map,
            frame_filter=ctc_teacher_online_full_frame_filter,
            frame_filter_neighbor_radius=int(config.ctc_teacher_frame_filter_neighbor_radius),
            frame_filter_min_nonblank_prob=float(config.ctc_teacher_frame_filter_min_nonblank_prob),
            missing_policy=config.ctc_teacher_topk_missing_policy,
            temperature=float(config.ctc_teacher_online_full_temperature),
            nonblank_frame_weight=float(config.ctc_teacher_online_full_nonblank_weight),
            eval_accumulator=full_eval_accumulator,
        )
        total = total + loss_value * full_weight

    encoder_weight = float(config.ctc_teacher_online_encoder_loss_weight)
    if encoder_weight > 0.0:
        student_encoded, student_lengths = student_encoded_and_lengths()
        loss_value, _, _ = _ctc_teacher_hidden_loss(
            student_encoded,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            teacher_field="encoder_out",
            time_map=config.ctc_teacher_topk_time_map,
            missing_policy=config.ctc_teacher_topk_missing_policy,
            blank_id=int(config.blank_id),
            frame_filter=ctc_teacher_frame_filter,
            frame_filter_neighbor_radius=int(config.ctc_teacher_frame_filter_neighbor_radius),
            frame_filter_min_nonblank_prob=float(config.ctc_teacher_frame_filter_min_nonblank_prob),
        )
        total = total + loss_value * encoder_weight

    sequence_weight = float(config.ctc_teacher_online_sequence_loss_weight)
    if sequence_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        loss_value, _, _ = _ctc_teacher_sequence_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            ignored_token_ids=ignored_token_ids,
            missing_policy=config.ctc_teacher_topk_missing_policy,
        )
        total = total + loss_value * sequence_weight

    sequence_window_weight = float(config.ctc_teacher_online_sequence_window_loss_weight)
    if sequence_window_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        loss_value, _, _, _ = _ctc_teacher_sequence_window_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            ignored_token_ids=ignored_token_ids,
            missing_policy=config.ctc_teacher_topk_missing_policy,
            radius=int(config.ctc_teacher_online_sequence_window_radius),
            temperature=float(config.ctc_teacher_online_sequence_window_temperature),
        )
        total = total + loss_value * sequence_window_weight

    sequence_presence_weight = float(config.ctc_teacher_online_sequence_presence_loss_weight)
    if sequence_presence_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        loss_value, _, _, _ = _ctc_teacher_sequence_presence_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            ignored_token_ids=ignored_token_ids,
            missing_policy=config.ctc_teacher_topk_missing_policy,
        )
        total = total + loss_value * sequence_presence_weight

    nonblank_hard_weight = float(config.ctc_teacher_online_nonblank_hard_loss_weight)
    nonblank_margin_weight = float(config.ctc_teacher_online_nonblank_margin_loss_weight)
    if nonblank_hard_weight > 0.0 or nonblank_margin_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        hard_loss, margin_loss, _, _ = _ctc_teacher_nonblank_hard_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            time_map=config.ctc_teacher_topk_time_map,
            frame_filter_min_nonblank_prob=float(config.ctc_teacher_frame_filter_min_nonblank_prob),
            missing_policy=config.ctc_teacher_topk_missing_policy,
            margin=float(config.ctc_teacher_online_nonblank_margin),
        )
        if nonblank_hard_weight > 0.0:
            total = total + hard_loss * nonblank_hard_weight
        if nonblank_margin_weight > 0.0:
            total = total + margin_loss * nonblank_margin_weight

    nonblank_window_weight = float(config.ctc_teacher_online_nonblank_window_loss_weight)
    nonblank_window_margin_weight = float(config.ctc_teacher_online_nonblank_window_margin_loss_weight)
    if nonblank_window_weight > 0.0 or nonblank_window_margin_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        window_loss, margin_loss, _, _, _ = _ctc_teacher_nonblank_window_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            time_map=config.ctc_teacher_topk_time_map,
            frame_filter_min_nonblank_prob=float(config.ctc_teacher_frame_filter_min_nonblank_prob),
            missing_policy=config.ctc_teacher_topk_missing_policy,
            margin=float(config.ctc_teacher_online_nonblank_margin),
            window_radius=int(config.ctc_teacher_online_nonblank_window_radius),
            temperature=float(config.ctc_teacher_online_nonblank_window_temperature),
        )
        if nonblank_window_weight > 0.0:
            total = total + window_loss * nonblank_window_weight
        if nonblank_window_margin_weight > 0.0:
            total = total + margin_loss * nonblank_window_margin_weight

    nonblank_window_topk_weight = float(config.ctc_teacher_online_nonblank_window_topk_loss_weight)
    if nonblank_window_topk_weight > 0.0:
        student_logits, student_lengths = student_logits_and_lengths()
        loss_value, _, _, _ = _ctc_teacher_nonblank_window_topk_loss(
            student_logits,
            student_lengths,
            batch.utt_ids,
            ctc_teacher_online_records,
            blank_id=int(config.blank_id),
            time_map=config.ctc_teacher_topk_time_map,
            frame_filter_min_nonblank_prob=float(config.ctc_teacher_frame_filter_min_nonblank_prob),
            missing_policy=config.ctc_teacher_topk_missing_policy,
            window_radius=int(config.ctc_teacher_online_nonblank_window_radius),
            temperature=float(config.ctc_teacher_online_nonblank_window_temperature),
        )
        total = total + loss_value * nonblank_window_topk_weight

    return total


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
    config: DeepSpeedTrainConfig | None = None,
    ctc_teacher_online: FunASRNanoCTCTopKOnlineTeacher | None = None,
    layer_metrics_output: dict[int, dict[str, float]] | None = None,
    layer_component_metrics_output: dict[str, dict[int, dict[str, float]]] | None = None,
    logit_metrics_output: dict[str, float] | None = None,
) -> tuple[float, int]:
    if loader is None:
        return float("nan"), 0
    _set_loader_epoch(loader, sampler, epoch)
    trainer = RWKVDualModeCTCTrainer(model)
    was_training = model.training
    model.eval()
    local_loss_sum = 0.0
    local_sample_count = 0
    layer_eval_accumulator = (
        torch.zeros((int(config.num_layers), _LAYER_EVAL_WIDTH), dtype=torch.float64)
        if layer_metrics_output is not None and config is not None
        else None
    )
    layer_component_eval_accumulators = (
        {
            component: torch.zeros((int(config.num_layers), _LAYER_EVAL_WIDTH), dtype=torch.float64)
            for component, weight in (
                ("mixer", float(config.ctc_teacher_online_layer_mixer_loss_weight)),
                ("ffn", float(config.ctc_teacher_online_layer_ffn_loss_weight)),
                ("block", float(config.ctc_teacher_online_layer_block_loss_weight)),
            )
            if weight > 0.0
        }
        if layer_component_metrics_output is not None and config is not None
        else None
    )
    full_eval_accumulator = (
        torch.zeros((_CTC_FULL_EVAL_WIDTH,), dtype=torch.float64)
        if logit_metrics_output is not None
        and config is not None
        and float(config.ctc_teacher_online_full_loss_weight) > 0.0
        else None
    )
    local_eval_limit = None
    if max_eval_samples is not None:
        world_size = _world_size()
        rank = _rank()
        base = max_eval_samples // world_size
        extra = max_eval_samples % world_size
        local_eval_limit = base + (1 if rank < extra else 0)
    for eval_batch_index, batch in enumerate(loader):
        remaining = None if local_eval_limit is None else local_eval_limit - local_sample_count
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and int(batch.features.size(0)) > remaining:
            batch = batch.prefix(remaining)
        teacher_batch_features = batch.features
        teacher_batch_feature_lengths = batch.feature_lengths
        batch = batch.to(device, feature_dtype=feature_dtype)
        if config is not None and ctc_teacher_online is not None:
            if batch.utt_ids is None:
                raise RuntimeError("Online CTC teacher eval requires batch.utt_ids.")
            layer_component_weights = {
                "mixer": float(config.ctc_teacher_online_layer_mixer_loss_weight),
                "ffn": float(config.ctc_teacher_online_layer_ffn_loss_weight),
                "block": float(config.ctc_teacher_online_layer_block_loss_weight),
            }
            layer_hidden_enabled = any(value > 0.0 for value in layer_component_weights.values())
            layer_hidden_only = _is_layer_hidden_only_objective(config)
            selected_layer_ids = (
                _select_eval_layer_hidden_ids(
                    batch_index=eval_batch_index,
                    num_layers=int(config.num_layers),
                    sample_count=int(config.ctc_teacher_online_layer_sample_count),
                    rank=_rank(),
                    world_size=_world_size(),
                )
                if layer_hidden_enabled
                else ()
            )
            teacher_layer_ids = _teacher_layer_capture_ids(
                selected_layer_ids,
                input_mode=str(config.ctc_teacher_online_layer_input_mode),
            )
            if config.ctc_teacher_online_use_batch_features:
                ctc_teacher_online_records = ctc_teacher_online.feature_records(
                    batch.utt_ids,
                    teacher_batch_features,
                    teacher_batch_feature_lengths,
                    audio_rows=batch.ctc_teacher_audio_rows,
                    layer_ids=teacher_layer_ids,
                    include_ctc_outputs=not layer_hidden_only,
                )
            else:
                ctc_teacher_online_records = ctc_teacher_online.topk_records(
                    batch.utt_ids,
                    batch.ctc_teacher_audio_rows,
                    layer_ids=teacher_layer_ids,
                )
            mask = trainer.eval_direction_mask(mode, device=batch.features.device)
            student_layer_hiddens: dict[int, dict[str, torch.Tensor]] = {}
            if str(config.ctc_teacher_online_layer_input_mode) == "teacher_forced":
                if not layer_hidden_only:
                    raise ValueError("Teacher-forced layer alignment currently requires a hidden-only objective.")
                student_layer_hiddens, encoded_lengths = _teacher_forced_student_layer_hiddens(
                    model,
                    ctc_teacher_online_records,
                    batch.utt_ids,
                    layer_ids=selected_layer_ids,
                    missing_policy=config.ctc_teacher_topk_missing_policy,
                )
                reference = next(iter(student_layer_hiddens.values()))["mixer"]
                zero_loss = reference.float().sum() * 0.0
                losses = {
                    "loss": zero_loss,
                    "ctc_loss": zero_loss,
                    "decoder_loss": zero_loss,
                    "encoded_lengths": encoded_lengths,
                }
            else:
                capture_context = (
                    _capture_student_sensevoice_layer_hiddens(model, selected_layer_ids)
                    if selected_layer_ids
                    else nullcontext(student_layer_hiddens)
                )
                with capture_context as student_layer_hiddens:
                    if layer_hidden_only:
                        encoded, encoded_lengths, _ = model.encoder(
                            batch.features,
                            batch.feature_lengths,
                            direction_mask=mask,
                        )
                        zero_loss = encoded.float().sum() * 0.0
                        losses = {
                            "loss": zero_loss,
                            "ctc_loss": zero_loss,
                            "decoder_loss": zero_loss,
                            "encoded": encoded,
                            "encoded_lengths": encoded_lengths,
                        }
                    else:
                        losses = model.joint_losses(
                            batch.features,
                            batch.feature_lengths,
                            batch.targets,
                            batch.target_lengths,
                            decoder_targets=batch.decoder_targets,
                            decoder_target_lengths=batch.decoder_target_lengths,
                            decoder_prompt_before_audio=batch.decoder_prompt_before_audio,
                            decoder_prompt_before_audio_lengths=batch.decoder_prompt_before_audio_lengths,
                            direction_mask=mask,
                        )
            loss = _online_ctc_teacher_distillation_loss(
                config=config,
                losses=losses,
                batch=batch,
                ctc_teacher_online_records=ctc_teacher_online_records,
                full_eval_accumulator=full_eval_accumulator,
            )
            if layer_hidden_enabled:
                student_encoded_lengths = losses.get("encoded_lengths")
                if student_encoded_lengths is not None and not isinstance(
                    student_encoded_lengths,
                    torch.Tensor,
                ):
                    raise RuntimeError("joint_losses returned non-tensor encoded_lengths.")
                layer_result = _ctc_teacher_layer_hidden_loss(
                    student_layer_hiddens,
                    student_encoded_lengths,
                    batch.utt_ids,
                    ctc_teacher_online_records,
                    layer_ids=selected_layer_ids,
                    component_weights=layer_component_weights,
                    normalized_mse_weight=float(config.ctc_teacher_online_layer_normalized_mse_weight),
                    cosine_weight=float(config.ctc_teacher_online_layer_cosine_weight),
                    energy_mse_weight=float(config.ctc_teacher_online_layer_energy_mse_weight),
                    log_rms_weight=float(config.ctc_teacher_online_layer_log_rms_weight),
                    raw_mse_weight=float(config.ctc_teacher_online_layer_raw_mse_weight),
                    frame_tolerance=int(config.ctc_teacher_online_layer_frame_tolerance),
                    missing_policy=config.ctc_teacher_topk_missing_policy,
                )
                loss = loss + layer_result.loss
                if layer_eval_accumulator is not None:
                    _accumulate_layer_eval_metrics(layer_eval_accumulator, layer_result)
                if layer_component_eval_accumulators is not None:
                    for component, accumulator in layer_component_eval_accumulators.items():
                        component_result = _ctc_teacher_layer_hidden_loss(
                            student_layer_hiddens,
                            student_encoded_lengths,
                            batch.utt_ids,
                            ctc_teacher_online_records,
                            layer_ids=selected_layer_ids,
                            component_weights={component: 1.0},
                            normalized_mse_weight=float(config.ctc_teacher_online_layer_normalized_mse_weight),
                            cosine_weight=float(config.ctc_teacher_online_layer_cosine_weight),
                            energy_mse_weight=float(config.ctc_teacher_online_layer_energy_mse_weight),
                            log_rms_weight=float(config.ctc_teacher_online_layer_log_rms_weight),
                            raw_mse_weight=float(config.ctc_teacher_online_layer_raw_mse_weight),
                            frame_tolerance=int(config.ctc_teacher_online_layer_frame_tolerance),
                            missing_policy=config.ctc_teacher_topk_missing_policy,
                        )
                        _accumulate_layer_eval_metrics(accumulator, component_result)
        else:
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
    if layer_metrics_output is not None and layer_eval_accumulator is not None:
        layer_metrics_output.clear()
        layer_metrics_output.update(
            _finalize_layer_eval_metrics(layer_eval_accumulator, device=device)
        )
    if layer_component_metrics_output is not None and layer_component_eval_accumulators is not None:
        layer_component_metrics_output.clear()
        layer_component_metrics_output.update(
            {
                component: _finalize_layer_eval_metrics(accumulator, device=device)
                for component, accumulator in layer_component_eval_accumulators.items()
            }
        )
    if logit_metrics_output is not None:
        logit_metrics_output.clear()
        if full_eval_accumulator is not None:
            logit_metrics_output.update(
                _finalize_ctc_full_eval_metrics(full_eval_accumulator, device=device)
            )
    return _all_reduce_mean(local_loss_sum, local_sample_count, device=device), global_sample_count


def _load_encoder_anchor_model(
    *,
    model_config: RWKVCTCModelConfig,
    checkpoint_path: str | None,
) -> RWKVCTCModel | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"encoder_anchor_checkpoint_path missing or empty: {path}")
    anchor = RWKVCTCModel(model_config)
    restored = load_checkpoint(path, model=anchor, map_location="cpu", strict=True)
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    anchor.eval()
    _rank_zero_log(
        "Loaded encoder anchor checkpoint: "
        f"path={path} step={int(restored.get('step', 0))} "
        f"missing={len(restored['extra'].get('missing_keys', []))} "
        f"unexpected={len(restored['extra'].get('unexpected_keys', []))}"
    )
    return anchor


@torch.no_grad()
def _encoder_anchor_forward(
    *,
    anchor_model: RWKVCTCModel,
    features: torch.Tensor,
    feature_lengths: torch.Tensor | None,
    direction_mask: DirectionMask,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    encoded, encoded_lengths, _ = anchor_model.encoder(
        features,
        lengths=feature_lengths,
        direction_mask=direction_mask,
    )
    return encoded, encoded_lengths


def _masked_encoder_mse_loss(
    student_encoded: torch.Tensor,
    teacher_encoded: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    time = min(int(student_encoded.size(1)), int(teacher_encoded.size(1)))
    if time <= 0:
        return student_encoded.new_zeros(())
    student = student_encoded[:, :time].float()
    teacher = teacher_encoded[:, :time].float()
    squared = (student - teacher).pow(2).mean(dim=-1)
    if lengths is None:
        return squared.mean()
    clipped_lengths = lengths.to(device=squared.device).clamp(min=0, max=time)
    positions = torch.arange(time, device=squared.device).unsqueeze(0)
    mask = positions < clipped_lengths.unsqueeze(1)
    denom = mask.sum().clamp_min(1)
    return squared.masked_select(mask).sum() / denom


def _masked_ctc_logit_kl_loss(
    student_logits: torch.Tensor,
    teacher_encoded: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    anchor_model: RWKVCTCModel,
    chunk_frames: int,
) -> torch.Tensor:
    if getattr(anchor_model, "ctc_decoder", None) is not None:
        raise ValueError("ctc_logit_anchor_loss is not supported when the anchor uses a CTC decoder.")
    time = min(int(student_logits.size(1)), int(teacher_encoded.size(1)))
    if time <= 0:
        return student_logits.new_zeros(())
    chunk_frames = max(1, int(chunk_frames))
    total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    clipped_lengths = None
    if lengths is not None:
        clipped_lengths = lengths.to(device=student_logits.device).clamp(min=0, max=time)

    for start in range(0, time, chunk_frames):
        end = min(start + chunk_frames, time)
        student_slice = student_logits[:, start:end].float()
        with torch.no_grad():
            teacher_logits = anchor_model.ctc_head(teacher_encoded[:, start:end]).float()
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_slice, dim=-1)
        frame_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        if clipped_lengths is None:
            total = total + frame_kl.sum()
            denom = denom + frame_kl.new_tensor(float(frame_kl.numel()))
            continue
        positions = torch.arange(start, end, device=frame_kl.device).unsqueeze(0)
        mask = positions < clipped_lengths.unsqueeze(1)
        total = total + frame_kl.masked_select(mask).sum()
        denom = denom + mask.sum().to(dtype=torch.float32, device=frame_kl.device)
    return total / denom.clamp_min(1.0)


def _load_ctc_teacher_topk_cache(cache_path: str | None) -> dict[str, dict[str, Any]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"ctc_teacher_topk_cache_path missing or empty: {path}")

    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid CTC teacher top-k JSONL at {path}:{line_number}") from exc
            utt_id = record.get("utt_id") or record.get("id") or record.get("key")
            if utt_id is None or not str(utt_id).strip():
                raise ValueError(f"CTC teacher top-k cache row has no utt_id/id/key at {path}:{line_number}")
            row = dict(record)
            row["_cache_base_dir"] = str(path.parent)
            cache[str(utt_id)] = row
    return cache


def _materialize_ctc_teacher_topk_record(record: dict[str, Any]) -> dict[str, Any]:
    if "topk_token_ids" in record and "topk_log_probs" in record:
        return record

    tensor_path_value = (
        record.get("topk_tensor_path")
        or record.get("topk_path")
        or record.get("tensor_path")
    )
    if tensor_path_value is None:
        return record

    tensor_path = Path(str(tensor_path_value))
    if not tensor_path.is_absolute():
        tensor_path = Path(str(record.get("_cache_base_dir") or ".")) / tensor_path
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"CTC teacher top-k tensor payload must be a dict: {tensor_path}")

    materialized = dict(record)
    for key in ("topk_token_ids", "topk_ids", "token_ids", "ids"):
        if key in payload:
            materialized["topk_token_ids"] = payload[key]
            break
    for key in ("topk_log_probs", "topk_logprobs", "log_probs", "logprobs"):
        if key in payload:
            materialized["topk_log_probs"] = payload[key]
            break
    if "topk_token_ids" not in materialized or "topk_log_probs" not in materialized:
        raise ValueError(f"CTC teacher top-k tensor payload missing ids/log_probs: {tensor_path}")
    record.update(materialized)
    return record


def _ctc_teacher_topk_field(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _ctc_teacher_topk_frame_ce(
    student_log_probs: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    ignored_token_ids: tuple[int, ...],
) -> torch.Tensor:
    if teacher_ids.numel() == 0 or teacher_log_probs.numel() == 0:
        return student_log_probs.new_zeros((int(student_log_probs.size(0)),), dtype=torch.float32)
    vocab_size = int(student_log_probs.size(-1))
    teacher_ids = teacher_ids.to(device=student_log_probs.device, dtype=torch.long)
    teacher_log_probs = teacher_log_probs.to(device=student_log_probs.device, dtype=torch.float32)
    valid = (teacher_ids >= 0) & (teacher_ids < vocab_size)
    for ignored_id in ignored_token_ids:
        valid = valid & (teacher_ids != int(ignored_id))
    safe_ids = teacher_ids.masked_fill(~valid, 0)
    teacher_probs = teacher_log_probs.exp().masked_fill(~valid, 0.0)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    gathered_student = student_log_probs.gather(dim=-1, index=safe_ids)
    return -(teacher_probs * gathered_student).sum(dim=-1)


def _ctc_teacher_frame_filter_mask(
    teacher_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    frame_filter: str,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
    neighbor_radius: int,
    min_nonblank_prob: float,
    device: torch.device,
) -> torch.Tensor | None:
    frame_filter = str(frame_filter or "all")
    if frame_filter == "all":
        return None
    if frame_filter not in {"nonblank", "nonblank_neighbors"}:
        raise ValueError(
            "ctc_teacher_frame_filter must be 'all', 'nonblank', or 'nonblank_neighbors', "
            f"got {frame_filter!r}."
        )
    if teacher_ids.numel() == 0 or teacher_log_probs.numel() == 0:
        return torch.zeros((int(teacher_ids.size(0)),), dtype=torch.bool, device=device)
    if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2:
        raise ValueError(
            "CTC teacher frame filtering expects 2-D ids/log_probs, "
            f"got ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
        )
    top1_ids = teacher_ids[:, 0].to(dtype=torch.long)
    top1_probs = teacher_log_probs[:, 0].to(dtype=torch.float32).exp()
    mask = top1_ids.ne(int(blank_id))
    for ignored_id in ignored_token_ids:
        mask = mask & top1_ids.ne(int(ignored_id))
    min_prob = float(min_nonblank_prob)
    if min_prob > 0.0:
        mask = mask & top1_probs.ge(min_prob)
    if frame_filter == "nonblank_neighbors" and int(neighbor_radius) > 0 and bool(mask.any().item()):
        radius = int(neighbor_radius)
        expanded = mask.clone()
        indices = mask.nonzero(as_tuple=False).flatten()
        time = int(mask.numel())
        for offset in range(1, radius + 1):
            left = (indices - offset).clamp(min=0, max=time - 1)
            right = (indices + offset).clamp(min=0, max=time - 1)
            expanded[left] = True
            expanded[right] = True
        mask = expanded
    return mask.to(device=device)


def _logsumexp_without_index(logits: torch.Tensor, *, index: int) -> torch.Tensor:
    if int(index) < 0 or int(index) >= int(logits.size(-1)):
        raise ValueError(f"blank_id={index} is outside logits vocab size={logits.size(-1)}")
    masked = logits.float().clone()
    masked[..., int(index)] = float("-inf")
    return torch.logsumexp(masked, dim=-1)


def _ctc_teacher_blank_log_probs_from_record(
    record: dict[str, Any],
    *,
    teacher_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_blank_id: int,
) -> torch.Tensor | None:
    raw_blank = _ctc_teacher_topk_field(
        record,
        "blank_log_probs",
        "blank_logprobs",
        "blank_log_probs_tensor",
        "blank_logprobs_tensor",
    )
    if raw_blank is not None:
        blank = torch.as_tensor(raw_blank, dtype=torch.float32)
        if blank.ndim == 1:
            return blank
        if blank.ndim == 2 and int(blank.size(-1)) == 1:
            return blank.squeeze(-1)
        raise ValueError(f"CTC teacher blank_log_probs must be 1-D, got {tuple(blank.shape)}")

    project_blank_id = int(record.get("project_blank_id", student_blank_id))
    blank_mask = teacher_ids.eq(project_blank_id)
    if not bool(blank_mask.any().item()):
        return None
    blank_values = teacher_log_probs.masked_fill(~blank_mask, float("-inf")).amax(dim=-1)
    return blank_values


def _ctc_teacher_nonblank_hard_frame_losses(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
    min_nonblank_prob: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if teacher_ids.numel() == 0 or teacher_log_probs.numel() == 0:
        zeros = student_logits.new_zeros((int(student_logits.size(0)),), dtype=torch.float32)
        mask = torch.zeros((int(student_logits.size(0)),), dtype=torch.bool, device=student_logits.device)
        return zeros, zeros, mask
    if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2:
        raise ValueError(
            "CTC teacher nonblank hard loss expects 2-D ids/log_probs, "
            f"got ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
        )
    if int(teacher_ids.size(0)) != int(student_logits.size(0)):
        raise ValueError(
            "CTC teacher nonblank hard loss time mismatch: "
            f"student={int(student_logits.size(0))} teacher={int(teacher_ids.size(0))}"
        )
    vocab_size = int(student_logits.size(-1))
    blank_id = int(blank_id)
    if blank_id < 0 or blank_id >= vocab_size:
        raise ValueError(f"blank_id={blank_id} is outside logits vocab size={vocab_size}")

    teacher_ids = teacher_ids.to(device=student_logits.device, dtype=torch.long)
    teacher_log_probs = teacher_log_probs.to(device=student_logits.device, dtype=torch.float32)
    top1_ids = teacher_ids[:, 0]
    top1_probs = teacher_log_probs[:, 0].exp()
    selected = (top1_ids >= 0) & (top1_ids < vocab_size) & top1_ids.ne(blank_id)
    for ignored_id in ignored_token_ids:
        selected = selected & top1_ids.ne(int(ignored_id))
    min_prob = float(min_nonblank_prob)
    if min_prob > 0.0:
        selected = selected & top1_probs.ge(min_prob)

    safe_ids = top1_ids.masked_fill(~selected, 0)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    hard_loss = -student_log_probs.gather(dim=-1, index=safe_ids.unsqueeze(-1)).squeeze(-1)

    target_logits = student_logits.float().gather(dim=-1, index=safe_ids.unsqueeze(-1)).squeeze(-1)
    blank_logits = student_logits.float()[:, blank_id]
    margin_loss = F.relu(float(margin) + blank_logits - target_logits)
    return hard_loss, margin_loss, selected


def _ctc_teacher_nonblank_hard_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    time_map: str,
    frame_filter_min_nonblank_prob: float,
    missing_policy: str,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if utt_ids is None:
        raise RuntimeError("Online CTC teacher nonblank hard distillation requires batch.utt_ids.")
    if time_map not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_time_map={time_map!r}; expected nearest or linear.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    hard_total = student_logits.new_zeros((), dtype=torch.float32)
    margin_total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing online CTC teacher nonblank hard record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        raw_ids = _ctc_teacher_topk_field(record, "topk_token_ids", "topk_ids", "token_ids", "ids")
        raw_log_probs = _ctc_teacher_topk_field(
            record,
            "topk_log_probs",
            "topk_logprobs",
            "log_probs",
            "logprobs",
        )
        if raw_ids is None or raw_log_probs is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"Online CTC teacher nonblank hard record missing ids/log_probs for utt_id={utt_id!r}")
            continue

        teacher_ids = torch.as_tensor(raw_ids, dtype=torch.long)
        teacher_log_probs = torch.as_tensor(raw_log_probs, dtype=torch.float32)
        if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2 or teacher_ids.shape != teacher_log_probs.shape:
            raise ValueError(
                f"Online CTC teacher nonblank hard record has invalid shape for utt_id={utt_id!r}: "
                f"ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
            )
        teacher_time = int(teacher_ids.size(0))
        student_time = int(clipped_lengths[sample_idx].item())
        if teacher_time <= 0 or student_time <= 0:
            continue

        matched += 1
        student_slice = student_logits[sample_idx, :student_time].float()
        ignored_values = record.get("project_ignored_token_ids")
        if ignored_values is None:
            ignored_values = [60514]
        ignored_ids = tuple(int(value) for value in ignored_values)
        project_blank_id = int(record.get("project_blank_id", int(blank_id)))

        if time_map == "nearest" or student_time == 1 or teacher_time == 1:
            if student_time == 1 or teacher_time == 1:
                frame_indices = torch.zeros(student_time, device=student_logits.device, dtype=torch.long)
            else:
                positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
                frame_indices = torch.round(positions * float(teacher_time - 1) / float(student_time - 1)).to(
                    dtype=torch.long
                )
            frame_indices = frame_indices.clamp(min=0, max=teacher_time - 1).cpu()
            selected_ids = teacher_ids.index_select(0, frame_indices)
            selected_log_probs = teacher_log_probs.index_select(0, frame_indices)
            hard_loss, margin_loss, selected_mask = _ctc_teacher_nonblank_hard_frame_losses(
                student_slice,
                selected_ids,
                selected_log_probs,
                blank_id=project_blank_id,
                ignored_token_ids=ignored_ids,
                min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                margin=float(margin),
            )
            weights = selected_mask.to(dtype=torch.float32)
            hard_loss = hard_loss * weights
            margin_loss = margin_loss * weights
        else:
            positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
            scaled = positions * float(teacher_time - 1) / float(student_time - 1)
            lo = torch.floor(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            hi = torch.ceil(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            alpha = (scaled - lo.to(dtype=torch.float32)).to(dtype=torch.float32)
            lo_ids = teacher_ids.index_select(0, lo.cpu())
            lo_log_probs = teacher_log_probs.index_select(0, lo.cpu())
            hi_ids = teacher_ids.index_select(0, hi.cpu())
            hi_log_probs = teacher_log_probs.index_select(0, hi.cpu())
            lo_hard, lo_margin, lo_mask = _ctc_teacher_nonblank_hard_frame_losses(
                student_slice,
                lo_ids,
                lo_log_probs,
                blank_id=project_blank_id,
                ignored_token_ids=ignored_ids,
                min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                margin=float(margin),
            )
            hi_hard, hi_margin, hi_mask = _ctc_teacher_nonblank_hard_frame_losses(
                student_slice,
                hi_ids,
                hi_log_probs,
                blank_id=project_blank_id,
                ignored_token_ids=ignored_ids,
                min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                margin=float(margin),
            )
            lo_weight = lo_mask.to(dtype=torch.float32) * (1.0 - alpha)
            hi_weight = hi_mask.to(dtype=torch.float32) * alpha
            hard_loss = lo_hard * lo_weight + hi_hard * hi_weight
            margin_loss = lo_margin * lo_weight + hi_margin * hi_weight
            weights = lo_weight + hi_weight

        selected_count = float(weights.sum().detach().item())
        if selected_count <= 0.0:
            continue
        hard_total = hard_total + hard_loss.sum()
        margin_total = margin_total + margin_loss.sum()
        denom = denom + weights.new_tensor(selected_count)

    denom = denom.clamp_min(1.0)
    return hard_total / denom, margin_total / denom, matched, missing


def _ctc_teacher_window_reduce(values: torch.Tensor, temperature: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    temperature = float(temperature)
    if temperature <= 0.0 or int(values.numel()) == 1:
        return values.min()
    scaled = -values.float() / temperature
    return -temperature * (torch.logsumexp(scaled, dim=0) - math.log(float(values.numel())))


def _ctc_teacher_nonblank_peak_indices(
    top1_ids: torch.Tensor,
    top1_probs: torch.Tensor,
    selected: torch.Tensor,
) -> list[int]:
    indices = torch.nonzero(selected, as_tuple=False).flatten().tolist()
    if not indices:
        return []
    peaks: list[int] = []
    best_idx = int(indices[0])
    best_prob = float(top1_probs[best_idx].item())
    prev_idx = best_idx
    prev_id = int(top1_ids[best_idx].item())
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        token_id = int(top1_ids[idx].item())
        prob = float(top1_probs[idx].item())
        if idx == prev_idx + 1 and token_id == prev_id:
            if prob > best_prob:
                best_idx = idx
                best_prob = prob
        else:
            peaks.append(best_idx)
            best_idx = idx
            best_prob = prob
        prev_idx = idx
        prev_id = token_id
    peaks.append(best_idx)
    return peaks


def _ctc_teacher_nonblank_window_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    time_map: str,
    frame_filter_min_nonblank_prob: float,
    missing_policy: str,
    margin: float,
    window_radius: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    if utt_ids is None:
        raise RuntimeError("Online CTC teacher nonblank window distillation requires batch.utt_ids.")
    if time_map not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_time_map={time_map!r}; expected nearest or linear.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")
    window_radius = int(window_radius)
    if window_radius < 0:
        raise ValueError(f"ctc_teacher_online_nonblank_window_radius must be >= 0, got {window_radius!r}.")
    if float(temperature) < 0.0:
        raise ValueError(
            "ctc_teacher_online_nonblank_window_temperature must be >= 0, "
            f"got {float(temperature)!r}."
        )

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    token_total = student_logits.new_zeros((), dtype=torch.float32)
    margin_total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    event_count = 0
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing online CTC teacher nonblank window record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        raw_ids = _ctc_teacher_topk_field(record, "topk_token_ids", "topk_ids", "token_ids", "ids")
        raw_log_probs = _ctc_teacher_topk_field(
            record,
            "topk_log_probs",
            "topk_logprobs",
            "log_probs",
            "logprobs",
        )
        if raw_ids is None or raw_log_probs is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"Online CTC teacher nonblank window record missing ids/log_probs for utt_id={utt_id!r}")
            continue

        teacher_ids = torch.as_tensor(raw_ids, dtype=torch.long)
        teacher_log_probs = torch.as_tensor(raw_log_probs, dtype=torch.float32)
        if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2 or teacher_ids.shape != teacher_log_probs.shape:
            raise ValueError(
                f"Online CTC teacher nonblank window record has invalid shape for utt_id={utt_id!r}: "
                f"ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
            )
        teacher_time = int(teacher_ids.size(0))
        student_time = int(clipped_lengths[sample_idx].item())
        if teacher_time <= 0 or student_time <= 0:
            continue

        matched += 1
        vocab_size = int(student_logits.size(-1))
        ignored_values = record.get("project_ignored_token_ids")
        if ignored_values is None:
            ignored_values = [60514]
        ignored_ids = tuple(int(value) for value in ignored_values)
        project_blank_id = int(record.get("project_blank_id", int(blank_id)))
        if project_blank_id < 0 or project_blank_id >= vocab_size:
            raise ValueError(f"blank_id={project_blank_id} is outside logits vocab size={vocab_size}")

        top1_ids = teacher_ids[:, 0]
        top1_probs = teacher_log_probs[:, 0].exp()
        selected = (top1_ids >= 0) & (top1_ids < vocab_size) & top1_ids.ne(project_blank_id)
        for ignored_id in ignored_ids:
            selected = selected & top1_ids.ne(int(ignored_id))
        min_prob = float(frame_filter_min_nonblank_prob)
        if min_prob > 0.0:
            selected = selected & top1_probs.ge(min_prob)
        peak_indices = _ctc_teacher_nonblank_peak_indices(top1_ids, top1_probs, selected)
        if not peak_indices:
            continue

        student_slice = student_logits[sample_idx, :student_time].float()
        student_log_probs = F.log_softmax(student_slice, dim=-1)
        blank_logits = student_slice[:, project_blank_id]
        for teacher_idx in peak_indices:
            target_id = int(top1_ids[teacher_idx].item())
            if teacher_time == 1 or student_time == 1:
                center = 0
            else:
                scaled = float(teacher_idx) * float(student_time - 1) / float(teacher_time - 1)
                center = int(round(scaled))
            center = max(0, min(center, student_time - 1))
            lo = max(0, center - window_radius)
            hi = min(student_time, center + window_radius + 1)
            if hi <= lo:
                continue
            token_ce = -student_log_probs[lo:hi, target_id]
            token_total = token_total + _ctc_teacher_window_reduce(token_ce, float(temperature))
            target_logits = student_slice[lo:hi, target_id]
            margin_values = F.relu(float(margin) + blank_logits[lo:hi] - target_logits)
            margin_total = margin_total + _ctc_teacher_window_reduce(margin_values, float(temperature))
            denom = denom + student_logits.new_tensor(1.0, dtype=torch.float32)
            event_count += 1

    denom = denom.clamp_min(1.0)
    return token_total / denom, margin_total / denom, matched, missing, event_count


def _ctc_teacher_nonblank_window_topk_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    time_map: str,
    frame_filter_min_nonblank_prob: float,
    missing_policy: str,
    window_radius: int,
    temperature: float,
) -> tuple[torch.Tensor, int, int, int]:
    if utt_ids is None:
        raise RuntimeError("Online CTC teacher nonblank window top-k distillation requires batch.utt_ids.")
    if time_map not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_time_map={time_map!r}; expected nearest or linear.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")
    window_radius = int(window_radius)
    if window_radius < 0:
        raise ValueError(f"ctc_teacher_online_nonblank_window_radius must be >= 0, got {window_radius!r}.")
    if float(temperature) < 0.0:
        raise ValueError(
            "ctc_teacher_online_nonblank_window_temperature must be >= 0, "
            f"got {float(temperature)!r}."
        )

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    event_count = 0
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing online CTC teacher nonblank window top-k record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        raw_ids = _ctc_teacher_topk_field(record, "topk_token_ids", "topk_ids", "token_ids", "ids")
        raw_log_probs = _ctc_teacher_topk_field(
            record,
            "topk_log_probs",
            "topk_logprobs",
            "log_probs",
            "logprobs",
        )
        if raw_ids is None or raw_log_probs is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(
                    f"Online CTC teacher nonblank window top-k record missing ids/log_probs for utt_id={utt_id!r}"
                )
            continue

        teacher_ids = torch.as_tensor(raw_ids, dtype=torch.long)
        teacher_log_probs = torch.as_tensor(raw_log_probs, dtype=torch.float32)
        if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2 or teacher_ids.shape != teacher_log_probs.shape:
            raise ValueError(
                f"Online CTC teacher nonblank window top-k record has invalid shape for utt_id={utt_id!r}: "
                f"ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
            )
        teacher_time = int(teacher_ids.size(0))
        student_time = int(clipped_lengths[sample_idx].item())
        if teacher_time <= 0 or student_time <= 0:
            continue

        matched += 1
        vocab_size = int(student_logits.size(-1))
        ignored_values = record.get("project_ignored_token_ids")
        if ignored_values is None:
            ignored_values = [60514]
        ignored_ids = tuple(int(value) for value in ignored_values)
        project_blank_id = int(record.get("project_blank_id", int(blank_id)))
        if project_blank_id < 0 or project_blank_id >= vocab_size:
            raise ValueError(f"blank_id={project_blank_id} is outside logits vocab size={vocab_size}")

        top1_ids = teacher_ids[:, 0]
        top1_probs = teacher_log_probs[:, 0].exp()
        selected = (top1_ids >= 0) & (top1_ids < vocab_size) & top1_ids.ne(project_blank_id)
        for ignored_id in ignored_ids:
            selected = selected & top1_ids.ne(int(ignored_id))
        min_prob = float(frame_filter_min_nonblank_prob)
        if min_prob > 0.0:
            selected = selected & top1_probs.ge(min_prob)
        peak_indices = _ctc_teacher_nonblank_peak_indices(top1_ids, top1_probs, selected)
        if not peak_indices:
            continue

        student_log_probs = F.log_softmax(student_logits[sample_idx, :student_time].float(), dim=-1)
        for teacher_idx in peak_indices:
            peak_ids = teacher_ids[teacher_idx].to(device=student_logits.device, dtype=torch.long)
            peak_log_probs = teacher_log_probs[teacher_idx].to(device=student_logits.device, dtype=torch.float32)
            valid = (peak_ids >= 0) & (peak_ids < vocab_size) & peak_ids.ne(project_blank_id)
            for ignored_id in ignored_ids:
                valid = valid & peak_ids.ne(int(ignored_id))
            if not bool(valid.any().item()):
                continue
            safe_ids = peak_ids.masked_select(valid)
            teacher_probs = peak_log_probs.masked_select(valid).exp()
            teacher_probs = teacher_probs / teacher_probs.sum().clamp_min(1.0e-8)

            if teacher_time == 1 or student_time == 1:
                center = 0
            else:
                scaled = float(teacher_idx) * float(student_time - 1) / float(teacher_time - 1)
                center = int(round(scaled))
            center = max(0, min(center, student_time - 1))
            lo = max(0, center - window_radius)
            hi = min(student_time, center + window_radius + 1)
            if hi <= lo:
                continue
            window_log_probs = student_log_probs[lo:hi].index_select(dim=-1, index=safe_ids)
            frame_ce = -(window_log_probs * teacher_probs.unsqueeze(0)).sum(dim=-1)
            total = total + _ctc_teacher_window_reduce(frame_ce, float(temperature))
            denom = denom + student_logits.new_tensor(1.0, dtype=torch.float32)
            event_count += 1

    return total / denom.clamp_min(1.0), matched, missing, event_count


def _ctc_teacher_blank_frame_bce(
    student_logits: torch.Tensor,
    teacher_blank_log_probs: torch.Tensor,
    *,
    blank_id: int,
) -> torch.Tensor:
    if teacher_blank_log_probs.numel() == 0:
        return student_logits.new_zeros((int(student_logits.size(0)),), dtype=torch.float32)
    blank_id = int(blank_id)
    student_blank_logits = student_logits.float()[..., blank_id]
    student_nonblank_lse = _logsumexp_without_index(student_logits, index=blank_id)
    student_blank_vs_nonblank = student_blank_logits - student_nonblank_lse
    teacher_blank_probs = teacher_blank_log_probs.to(
        device=student_logits.device,
        dtype=torch.float32,
    ).exp()
    teacher_blank_probs = teacher_blank_probs.clamp(min=0.0, max=1.0)
    return F.binary_cross_entropy_with_logits(
        student_blank_vs_nonblank,
        teacher_blank_probs,
        reduction="none",
    )


def _ctc_teacher_mass_frame_ce(
    student_log_probs: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_blank_log_probs: torch.Tensor,
    *,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
) -> torch.Tensor:
    if teacher_ids.numel() == 0 or teacher_log_probs.numel() == 0 or teacher_blank_log_probs.numel() == 0:
        return student_log_probs.new_zeros((int(student_log_probs.size(0)),), dtype=torch.float32)
    vocab_size = int(student_log_probs.size(-1))
    blank_id = int(blank_id)
    if blank_id < 0 or blank_id >= vocab_size:
        raise ValueError(f"blank_id={blank_id} is outside logits vocab size={vocab_size}")

    teacher_ids = teacher_ids.to(device=student_log_probs.device, dtype=torch.long)
    teacher_log_probs = teacher_log_probs.to(device=student_log_probs.device, dtype=torch.float32)
    teacher_blank_probs = teacher_blank_log_probs.to(
        device=student_log_probs.device,
        dtype=torch.float32,
    ).exp()
    teacher_blank_probs = teacher_blank_probs.clamp(min=0.0, max=1.0)

    valid = (teacher_ids >= 0) & (teacher_ids < vocab_size) & (teacher_ids != blank_id)
    for ignored_id in ignored_token_ids:
        valid = valid & (teacher_ids != int(ignored_id))
    safe_ids = teacher_ids.masked_fill(~valid, blank_id)
    teacher_selected_probs = teacher_log_probs.exp().masked_fill(~valid, 0.0)
    selected_mass = teacher_selected_probs.sum(dim=-1)
    teacher_other_probs = (1.0 - teacher_blank_probs - selected_mass).clamp(min=0.0, max=1.0)

    selected_student = student_log_probs.gather(dim=-1, index=safe_ids).masked_fill(~valid, 0.0)
    student_other_log_probs = student_log_probs.float().clone()
    student_other_log_probs[..., blank_id] = float("-inf")
    for ignored_id in ignored_token_ids:
        ignored_id = int(ignored_id)
        if 0 <= ignored_id < vocab_size:
            student_other_log_probs[..., ignored_id] = float("-inf")
    student_other_log_probs = student_other_log_probs.scatter(
        dim=-1,
        index=safe_ids,
        src=torch.full_like(safe_ids, float("-inf"), dtype=student_other_log_probs.dtype),
    )
    student_other_log_prob = torch.logsumexp(student_other_log_probs, dim=-1)
    other_term = torch.where(
        teacher_other_probs > 0.0,
        teacher_other_probs * student_other_log_prob,
        torch.zeros_like(teacher_other_probs),
    )

    return -(
        teacher_blank_probs * student_log_probs[..., blank_id]
        + (teacher_selected_probs * selected_student).sum(dim=-1)
        + other_term
    )


def _ctc_teacher_topk_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    time_map: str,
    frame_filter: str,
    frame_filter_neighbor_radius: int,
    frame_filter_min_nonblank_prob: float,
    missing_policy: str,
) -> tuple[torch.Tensor, int, int]:
    if utt_ids is None:
        raise RuntimeError("CTC teacher top-k distillation requires batch.utt_ids.")
    if time_map not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_time_map={time_map!r}; expected nearest or linear.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing CTC teacher top-k record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        raw_ids = _ctc_teacher_topk_field(record, "topk_token_ids", "topk_ids", "token_ids", "ids")
        raw_log_probs = _ctc_teacher_topk_field(
            record,
            "topk_log_probs",
            "topk_logprobs",
            "log_probs",
            "logprobs",
        )
        if raw_ids is None or raw_log_probs is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"CTC teacher top-k record missing ids/log_probs for utt_id={utt_id!r}")
            continue

        teacher_ids = torch.as_tensor(raw_ids, dtype=torch.long)
        teacher_log_probs = torch.as_tensor(raw_log_probs, dtype=torch.float32)
        if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2 or teacher_ids.shape != teacher_log_probs.shape:
            raise ValueError(
                f"CTC teacher top-k record has invalid shape for utt_id={utt_id!r}: "
                f"ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
            )
        teacher_time = int(teacher_ids.size(0))
        student_time = int(clipped_lengths[sample_idx].item())
        if teacher_time <= 0 or student_time <= 0:
            continue

        matched += 1
        student_slice = F.log_softmax(student_logits[sample_idx, :student_time].float(), dim=-1)
        ignored_values = record.get("project_ignored_token_ids")
        if ignored_values is None:
            ignored_values = [60514]
        ignored_ids = tuple(int(value) for value in ignored_values)
        project_blank_id = int(record.get("project_blank_id", int(blank_id)))
        if time_map == "nearest" or student_time == 1 or teacher_time == 1:
            if student_time == 1 or teacher_time == 1:
                frame_indices = torch.zeros(student_time, device=student_logits.device, dtype=torch.long)
            else:
                positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
                frame_indices = torch.round(positions * float(teacher_time - 1) / float(student_time - 1)).to(
                    dtype=torch.long
                )
            frame_indices = frame_indices.clamp(min=0, max=teacher_time - 1).cpu()
            selected_ids = teacher_ids.index_select(0, frame_indices)
            selected_log_probs = teacher_log_probs.index_select(0, frame_indices)
            frame_loss = _ctc_teacher_topk_frame_ce(
                student_slice,
                selected_ids,
                selected_log_probs,
                ignored_token_ids=ignored_ids,
            )
            selected_mask = _ctc_teacher_frame_filter_mask(
                selected_ids,
                selected_log_probs,
                frame_filter=frame_filter,
                blank_id=project_blank_id,
                ignored_token_ids=ignored_ids,
                neighbor_radius=int(frame_filter_neighbor_radius),
                min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                device=frame_loss.device,
            )
        else:
            positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
            scaled = positions * float(teacher_time - 1) / float(student_time - 1)
            lo = torch.floor(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            hi = torch.ceil(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            alpha = (scaled - lo.to(dtype=torch.float32)).to(dtype=torch.float32)
            lo_ids = teacher_ids.index_select(0, lo.cpu())
            lo_log_probs = teacher_log_probs.index_select(0, lo.cpu())
            hi_ids = teacher_ids.index_select(0, hi.cpu())
            hi_log_probs = teacher_log_probs.index_select(0, hi.cpu())
            lo_loss = _ctc_teacher_topk_frame_ce(
                student_slice,
                lo_ids,
                lo_log_probs,
                ignored_token_ids=ignored_ids,
            )
            hi_loss = _ctc_teacher_topk_frame_ce(
                student_slice,
                hi_ids,
                hi_log_probs,
                ignored_token_ids=ignored_ids,
            )
            frame_loss = lo_loss * (1.0 - alpha) + hi_loss * alpha
            selected_mask = _ctc_teacher_frame_filter_mask(
                lo_ids,
                lo_log_probs,
                frame_filter=frame_filter,
                blank_id=project_blank_id,
                ignored_token_ids=ignored_ids,
                neighbor_radius=int(frame_filter_neighbor_radius),
                min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                device=frame_loss.device,
            )
            hi_mask = _ctc_teacher_frame_filter_mask(
                hi_ids,
                hi_log_probs,
                frame_filter=frame_filter,
                blank_id=project_blank_id,
                ignored_token_ids=ignored_ids,
                neighbor_radius=int(frame_filter_neighbor_radius),
                min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                device=frame_loss.device,
            )
            if selected_mask is not None and hi_mask is not None:
                selected_mask = selected_mask | hi_mask
        if selected_mask is not None:
            if not bool(selected_mask.any().item()):
                continue
            frame_loss = frame_loss[selected_mask]
            denom = denom + frame_loss.new_tensor(float(int(selected_mask.sum().item())))
        else:
            denom = denom + frame_loss.new_tensor(float(student_time))
        total = total + frame_loss.sum()

    return total / denom.clamp_min(1.0), matched, missing


def _ctc_teacher_blank_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    time_map: str,
    missing_policy: str,
) -> tuple[torch.Tensor, int, int]:
    if utt_ids is None:
        raise RuntimeError("CTC teacher blank distillation requires batch.utt_ids.")
    if time_map not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_time_map={time_map!r}; expected nearest or linear.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing CTC teacher blank record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        raw_ids = _ctc_teacher_topk_field(record, "topk_token_ids", "topk_ids", "token_ids", "ids")
        raw_log_probs = _ctc_teacher_topk_field(
            record,
            "topk_log_probs",
            "topk_logprobs",
            "log_probs",
            "logprobs",
        )
        if raw_ids is None or raw_log_probs is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"CTC teacher blank record missing ids/log_probs for utt_id={utt_id!r}")
            continue

        teacher_ids = torch.as_tensor(raw_ids, dtype=torch.long)
        teacher_log_probs = torch.as_tensor(raw_log_probs, dtype=torch.float32)
        if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2 or teacher_ids.shape != teacher_log_probs.shape:
            raise ValueError(
                f"CTC teacher blank record has invalid top-k shape for utt_id={utt_id!r}: "
                f"ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
            )
        teacher_blank_log_probs = _ctc_teacher_blank_log_probs_from_record(
            record,
            teacher_ids=teacher_ids,
            teacher_log_probs=teacher_log_probs,
            student_blank_id=int(blank_id),
        )
        if teacher_blank_log_probs is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"CTC teacher blank record has no blank probabilities for utt_id={utt_id!r}")
            continue
        teacher_time = int(teacher_blank_log_probs.numel())
        student_time = int(clipped_lengths[sample_idx].item())
        if teacher_time <= 0 or student_time <= 0:
            continue

        matched += 1
        student_slice = student_logits[sample_idx, :student_time].float()
        if time_map == "nearest" or student_time == 1 or teacher_time == 1:
            if student_time == 1 or teacher_time == 1:
                frame_indices = torch.zeros(student_time, device=student_logits.device, dtype=torch.long)
            else:
                positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
                frame_indices = torch.round(positions * float(teacher_time - 1) / float(student_time - 1)).to(
                    dtype=torch.long
                )
            frame_indices = frame_indices.clamp(min=0, max=teacher_time - 1).cpu()
            selected_blank = teacher_blank_log_probs.index_select(0, frame_indices)
        else:
            positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
            scaled = positions * float(teacher_time - 1) / float(student_time - 1)
            lo = torch.floor(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            hi = torch.ceil(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            alpha = (scaled - lo.to(dtype=torch.float32)).to(dtype=torch.float32).cpu()
            lo_blank = teacher_blank_log_probs.index_select(0, lo.cpu()).exp()
            hi_blank = teacher_blank_log_probs.index_select(0, hi.cpu()).exp()
            selected_prob = lo_blank * (1.0 - alpha) + hi_blank * alpha
            selected_blank = selected_prob.clamp(min=1.0e-8, max=1.0).log()
        frame_loss = _ctc_teacher_blank_frame_bce(
            student_slice,
            selected_blank,
            blank_id=int(blank_id),
        )
        total = total + frame_loss.sum()
        denom = denom + frame_loss.new_tensor(float(student_time))

    return total / denom.clamp_min(1.0), matched, missing


def _ctc_teacher_mass_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    time_map: str,
    missing_policy: str,
) -> tuple[torch.Tensor, int, int]:
    if utt_ids is None:
        raise RuntimeError("CTC teacher mass distillation requires batch.utt_ids.")
    if time_map not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_time_map={time_map!r}; expected nearest or linear.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing CTC teacher mass record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        raw_ids = _ctc_teacher_topk_field(record, "topk_token_ids", "topk_ids", "token_ids", "ids")
        raw_log_probs = _ctc_teacher_topk_field(
            record,
            "topk_log_probs",
            "topk_logprobs",
            "log_probs",
            "logprobs",
        )
        if raw_ids is None or raw_log_probs is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"CTC teacher mass record missing ids/log_probs for utt_id={utt_id!r}")
            continue

        teacher_ids = torch.as_tensor(raw_ids, dtype=torch.long)
        teacher_log_probs = torch.as_tensor(raw_log_probs, dtype=torch.float32)
        if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2 or teacher_ids.shape != teacher_log_probs.shape:
            raise ValueError(
                f"CTC teacher mass record has invalid top-k shape for utt_id={utt_id!r}: "
                f"ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
            )
        teacher_blank_log_probs = _ctc_teacher_blank_log_probs_from_record(
            record,
            teacher_ids=teacher_ids,
            teacher_log_probs=teacher_log_probs,
            student_blank_id=int(blank_id),
        )
        if teacher_blank_log_probs is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"CTC teacher mass record has no blank probabilities for utt_id={utt_id!r}")
            continue
        if int(teacher_blank_log_probs.numel()) != int(teacher_ids.size(0)):
            raise ValueError(
                f"CTC teacher mass blank length mismatch for utt_id={utt_id!r}: "
                f"blank={int(teacher_blank_log_probs.numel())} topk={int(teacher_ids.size(0))}"
            )
        teacher_time = int(teacher_ids.size(0))
        student_time = int(clipped_lengths[sample_idx].item())
        if teacher_time <= 0 or student_time <= 0:
            continue

        matched += 1
        student_slice = F.log_softmax(student_logits[sample_idx, :student_time].float(), dim=-1)
        ignored_values = record.get("project_ignored_token_ids")
        if ignored_values is None:
            ignored_values = [60514]
        ignored_ids = tuple(int(value) for value in ignored_values)
        if time_map == "nearest" or student_time == 1 or teacher_time == 1:
            if student_time == 1 or teacher_time == 1:
                frame_indices = torch.zeros(student_time, device=student_logits.device, dtype=torch.long)
            else:
                positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
                frame_indices = torch.round(positions * float(teacher_time - 1) / float(student_time - 1)).to(
                    dtype=torch.long
                )
            frame_indices = frame_indices.clamp(min=0, max=teacher_time - 1).cpu()
            selected_ids = teacher_ids.index_select(0, frame_indices)
            selected_log_probs = teacher_log_probs.index_select(0, frame_indices)
            selected_blank = teacher_blank_log_probs.index_select(0, frame_indices)
            frame_loss = _ctc_teacher_mass_frame_ce(
                student_slice,
                selected_ids,
                selected_log_probs,
                selected_blank,
                blank_id=int(blank_id),
                ignored_token_ids=ignored_ids,
            )
        else:
            positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
            scaled = positions * float(teacher_time - 1) / float(student_time - 1)
            lo = torch.floor(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            hi = torch.ceil(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            alpha = (scaled - lo.to(dtype=torch.float32)).to(dtype=torch.float32)
            lo_ids = teacher_ids.index_select(0, lo.cpu())
            lo_log_probs = teacher_log_probs.index_select(0, lo.cpu())
            lo_blank = teacher_blank_log_probs.index_select(0, lo.cpu())
            hi_ids = teacher_ids.index_select(0, hi.cpu())
            hi_log_probs = teacher_log_probs.index_select(0, hi.cpu())
            hi_blank = teacher_blank_log_probs.index_select(0, hi.cpu())
            lo_loss = _ctc_teacher_mass_frame_ce(
                student_slice,
                lo_ids,
                lo_log_probs,
                lo_blank,
                blank_id=int(blank_id),
                ignored_token_ids=ignored_ids,
            )
            hi_loss = _ctc_teacher_mass_frame_ce(
                student_slice,
                hi_ids,
                hi_log_probs,
                hi_blank,
                blank_id=int(blank_id),
                ignored_token_ids=ignored_ids,
            )
            frame_loss = lo_loss * (1.0 - alpha) + hi_loss * alpha
        total = total + frame_loss.sum()
        denom = denom + frame_loss.new_tensor(float(student_time))

    return total / denom.clamp_min(1.0), matched, missing


def _ctc_teacher_full_log_probs_from_record(record: dict[str, Any]) -> torch.Tensor | None:
    raw_full = _ctc_teacher_topk_field(
        record,
        "full_log_probs",
        "full_logprobs",
        "ctc_log_probs",
        "ctc_logprobs",
        "log_probs_full",
        "logprobs_full",
    )
    if raw_full is None:
        return None
    full = torch.as_tensor(raw_full, dtype=torch.float32)
    if full.ndim != 2:
        raise ValueError(f"CTC teacher full log_probs must be 2-D, got {tuple(full.shape)}")
    return full


def _match_teacher_full_vocab(
    teacher_log_probs: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    teacher_vocab = int(teacher_log_probs.size(-1))
    vocab_size = int(vocab_size)
    if teacher_vocab == vocab_size:
        return teacher_log_probs
    if teacher_vocab > vocab_size:
        return teacher_log_probs[..., :vocab_size]
    pad_shape = (*teacher_log_probs.shape[:-1], vocab_size - teacher_vocab)
    pad = teacher_log_probs.new_full(pad_shape, float("-inf"))
    return torch.cat([teacher_log_probs, pad], dim=-1)


def _project_ctc_teacher_full_log_probs(
    teacher_log_probs: torch.Tensor,
    *,
    vocab_size: int,
    blank_id: int,
    teacher_blank_id: int | None,
    project_blank_id: int | None,
    ignored_token_ids: tuple[int, ...],
) -> torch.Tensor:
    target_vocab_size = max(
        int(vocab_size),
        int(blank_id) + 1,
        int(project_blank_id) + 1 if project_blank_id is not None else 0,
        *(int(value) + 1 for value in ignored_token_ids),
    )
    projected = _match_teacher_full_vocab(teacher_log_probs, vocab_size=target_vocab_size)

    if teacher_blank_id is not None:
        teacher_blank_id = int(teacher_blank_id)
        project_blank_id = int(blank_id if project_blank_id is None else project_blank_id)
        if (
            teacher_blank_id != project_blank_id
            and 0 <= teacher_blank_id < int(teacher_log_probs.size(-1))
            and 0 <= project_blank_id < int(projected.size(-1))
        ):
            teacher_blank_log_probs = teacher_log_probs[..., teacher_blank_id]
            if bool(torch.isfinite(teacher_blank_log_probs).any().item()):
                projected = projected.clone()
                projected[..., project_blank_id] = teacher_blank_log_probs
                projected[..., teacher_blank_id] = float("-inf")

    for ignored_id in ignored_token_ids:
        ignored_id = int(ignored_id)
        if ignored_id == int(blank_id):
            continue
        if 0 <= ignored_id < int(projected.size(-1)):
            if projected is teacher_log_probs:
                projected = projected.clone()
            projected[..., ignored_id] = float("-inf")
    return projected


def _ctc_teacher_full_frame_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if teacher_log_probs.numel() == 0:
        return student_logits.new_zeros((int(student_logits.size(0)),), dtype=torch.float32)
    temperature = max(float(temperature), 1.0e-6)
    teacher_log_probs = _match_teacher_full_vocab(
        teacher_log_probs,
        vocab_size=int(student_logits.size(-1)),
    ).to(device=student_logits.device, dtype=torch.float32)
    if temperature != 1.0:
        teacher_log_probs = F.log_softmax(teacher_log_probs / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    else:
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    safe_teacher_log_probs = torch.where(
        torch.isfinite(teacher_log_probs),
        teacher_log_probs,
        torch.zeros_like(teacher_log_probs),
    )
    frame_kl = (teacher_probs * (safe_teacher_log_probs - student_log_probs)).sum(dim=-1)
    return frame_kl * (temperature * temperature)


_CTC_FULL_EVAL_KL_SUM = 0
_CTC_FULL_EVAL_SELECTED_FRAMES = 1
_CTC_FULL_EVAL_SELECTED_TOP1_MATCH = 2
_CTC_FULL_EVAL_ALL_TOP1_MATCH = 3
_CTC_FULL_EVAL_ALL_FRAMES = 4
_CTC_FULL_EVAL_BLANK_ABS_SUM = 5
_CTC_FULL_EVAL_TEACHER_NONBLANK = 6
_CTC_FULL_EVAL_STUDENT_NONBLANK = 7
_CTC_FULL_EVAL_TOKEN_EDITS = 8
_CTC_FULL_EVAL_TEACHER_TOKENS = 9
_CTC_FULL_EVAL_STUDENT_TOKENS = 10
_CTC_FULL_EVAL_SEQUENCE_EXACT = 11
_CTC_FULL_EVAL_MATCHED_UTTERANCES = 12
_CTC_FULL_EVAL_MISSING_UTTERANCES = 13
_CTC_FULL_EVAL_FRAME_DELTA_SUM = 14
_CTC_FULL_EVAL_WIDTH = 15


def _ctc_collapse_token_ids(
    token_ids: torch.Tensor,
    *,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    ignored = set(int(value) for value in ignored_token_ids)
    for raw_token_id in token_ids.detach().to(device="cpu", dtype=torch.long).tolist():
        token_id = int(raw_token_id)
        if token_id != previous and token_id != int(blank_id) and token_id not in ignored:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def _token_edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[hyp_index - 1] + int(ref_token != hyp_token)
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _accumulate_ctc_full_eval_metrics(
    accumulator: torch.Tensor,
    *,
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    frame_kl: torch.Tensor,
    selected_mask: torch.Tensor | None,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
    frame_delta: int,
) -> None:
    if accumulator.numel() != _CTC_FULL_EVAL_WIDTH:
        raise ValueError(
            "CTC full eval accumulator has invalid width: "
            f"{int(accumulator.numel())} != {_CTC_FULL_EVAL_WIDTH}."
        )
    teacher_log_probs = _match_teacher_full_vocab(
        teacher_log_probs,
        vocab_size=int(student_logits.size(-1)),
    ).to(device=student_logits.device, dtype=torch.float32)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_top1 = teacher_log_probs.argmax(dim=-1)
    student_top1 = student_log_probs.argmax(dim=-1)
    all_frames = int(student_top1.numel())
    if all_frames <= 0:
        return
    if selected_mask is None:
        selected_mask = torch.ones(all_frames, device=student_logits.device, dtype=torch.bool)
    else:
        selected_mask = selected_mask.to(device=student_logits.device, dtype=torch.bool)
    selected_frames = int(selected_mask.sum().item())

    teacher_blank_probs = teacher_log_probs[:, int(blank_id)].exp()
    student_blank_probs = student_log_probs[:, int(blank_id)].exp()
    teacher_tokens = _ctc_collapse_token_ids(
        teacher_top1,
        blank_id=int(blank_id),
        ignored_token_ids=ignored_token_ids,
    )
    student_tokens = _ctc_collapse_token_ids(
        student_top1,
        blank_id=int(blank_id),
        ignored_token_ids=ignored_token_ids,
    )
    edits = _token_edit_distance(teacher_tokens, student_tokens)

    values = accumulator.new_zeros((_CTC_FULL_EVAL_WIDTH,), dtype=torch.float64)
    if selected_frames > 0:
        values[_CTC_FULL_EVAL_KL_SUM] = frame_kl.detach()[selected_mask].double().sum().cpu()
        values[_CTC_FULL_EVAL_SELECTED_TOP1_MATCH] = (
            teacher_top1[selected_mask].eq(student_top1[selected_mask]).double().sum().cpu()
        )
    values[_CTC_FULL_EVAL_SELECTED_FRAMES] = float(selected_frames)
    values[_CTC_FULL_EVAL_ALL_TOP1_MATCH] = teacher_top1.eq(student_top1).double().sum().cpu()
    values[_CTC_FULL_EVAL_ALL_FRAMES] = float(all_frames)
    values[_CTC_FULL_EVAL_BLANK_ABS_SUM] = (
        teacher_blank_probs.sub(student_blank_probs).abs().double().sum().cpu()
    )
    values[_CTC_FULL_EVAL_TEACHER_NONBLANK] = teacher_top1.ne(int(blank_id)).double().sum().cpu()
    values[_CTC_FULL_EVAL_STUDENT_NONBLANK] = student_top1.ne(int(blank_id)).double().sum().cpu()
    values[_CTC_FULL_EVAL_TOKEN_EDITS] = float(edits)
    values[_CTC_FULL_EVAL_TEACHER_TOKENS] = float(len(teacher_tokens))
    values[_CTC_FULL_EVAL_STUDENT_TOKENS] = float(len(student_tokens))
    values[_CTC_FULL_EVAL_SEQUENCE_EXACT] = float(teacher_tokens == student_tokens)
    values[_CTC_FULL_EVAL_MATCHED_UTTERANCES] = 1.0
    values[_CTC_FULL_EVAL_FRAME_DELTA_SUM] = float(abs(int(frame_delta)))
    accumulator.add_(values)


def _finalize_ctc_full_eval_metrics(
    accumulator: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    if accumulator.numel() != _CTC_FULL_EVAL_WIDTH:
        raise ValueError(
            "CTC full eval accumulator has invalid width: "
            f"{int(accumulator.numel())} != {_CTC_FULL_EVAL_WIDTH}."
        )
    reduced = accumulator.to(device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    values = reduced.cpu().tolist()

    def ratio(numerator_index: int, denominator_index: int) -> float:
        denominator = float(values[denominator_index])
        if denominator <= 0.0:
            return float("nan")
        return float(values[numerator_index]) / denominator

    teacher_tokens = float(values[_CTC_FULL_EVAL_TEACHER_TOKENS])
    matched_utterances = float(values[_CTC_FULL_EVAL_MATCHED_UTTERANCES])
    teacher_nonblank = float(values[_CTC_FULL_EVAL_TEACHER_NONBLANK])
    return {
        "full_kl": ratio(_CTC_FULL_EVAL_KL_SUM, _CTC_FULL_EVAL_SELECTED_FRAMES),
        "selected_top1_agreement": ratio(
            _CTC_FULL_EVAL_SELECTED_TOP1_MATCH,
            _CTC_FULL_EVAL_SELECTED_FRAMES,
        ),
        "all_top1_agreement": ratio(_CTC_FULL_EVAL_ALL_TOP1_MATCH, _CTC_FULL_EVAL_ALL_FRAMES),
        "blank_prob_mae": ratio(_CTC_FULL_EVAL_BLANK_ABS_SUM, _CTC_FULL_EVAL_ALL_FRAMES),
        "teacher_nonblank_rate": ratio(_CTC_FULL_EVAL_TEACHER_NONBLANK, _CTC_FULL_EVAL_ALL_FRAMES),
        "student_nonblank_rate": ratio(_CTC_FULL_EVAL_STUDENT_NONBLANK, _CTC_FULL_EVAL_ALL_FRAMES),
        "nonblank_rate_ratio": (
            float(values[_CTC_FULL_EVAL_STUDENT_NONBLANK]) / teacher_nonblank
            if teacher_nonblank > 0.0
            else float("nan")
        ),
        "ctc_token_error_rate": (
            float(values[_CTC_FULL_EVAL_TOKEN_EDITS]) / teacher_tokens
            if teacher_tokens > 0.0
            else float("nan")
        ),
        "collapsed_length_ratio": (
            float(values[_CTC_FULL_EVAL_STUDENT_TOKENS]) / teacher_tokens
            if teacher_tokens > 0.0
            else float("nan")
        ),
        "sequence_exact_rate": (
            float(values[_CTC_FULL_EVAL_SEQUENCE_EXACT]) / matched_utterances
            if matched_utterances > 0.0
            else float("nan")
        ),
        "mean_frame_delta": (
            float(values[_CTC_FULL_EVAL_FRAME_DELTA_SUM]) / matched_utterances
            if matched_utterances > 0.0
            else float("nan")
        ),
        "selected_frames": float(values[_CTC_FULL_EVAL_SELECTED_FRAMES]),
        "all_frames": float(values[_CTC_FULL_EVAL_ALL_FRAMES]),
        "teacher_tokens": teacher_tokens,
        "student_tokens": float(values[_CTC_FULL_EVAL_STUDENT_TOKENS]),
        "matched_utterances": matched_utterances,
        "missing_utterances": float(values[_CTC_FULL_EVAL_MISSING_UTTERANCES]),
    }


def _ctc_teacher_time_index_select(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.index_select(
        0,
        indices.to(device=value.device, dtype=torch.long),
    )


def _ctc_teacher_full_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    time_map: str,
    frame_filter: str,
    frame_filter_neighbor_radius: int,
    frame_filter_min_nonblank_prob: float,
    missing_policy: str,
    temperature: float,
    nonblank_frame_weight: float = 1.0,
    eval_accumulator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    if utt_ids is None:
        raise RuntimeError("CTC teacher full-logits distillation requires batch.utt_ids.")
    if time_map not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_time_map={time_map!r}; expected nearest or linear.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")
    if not math.isfinite(float(nonblank_frame_weight)) or float(nonblank_frame_weight) <= 0.0:
        raise ValueError(
            "ctc_teacher_online_full_nonblank_weight must be finite and > 0, "
            f"got {nonblank_frame_weight!r}."
        )

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if eval_accumulator is not None:
                eval_accumulator[_CTC_FULL_EVAL_MISSING_UTTERANCES] += 1.0
            if missing_policy == "error":
                raise KeyError(f"Missing CTC teacher full record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        teacher_full_log_probs = _ctc_teacher_full_log_probs_from_record(record)
        if teacher_full_log_probs is None:
            missing += 1
            if eval_accumulator is not None:
                eval_accumulator[_CTC_FULL_EVAL_MISSING_UTTERANCES] += 1.0
            if missing_policy == "error":
                raise ValueError(f"CTC teacher full record has no full_log_probs for utt_id={utt_id!r}")
            continue
        teacher_ids = None
        teacher_log_probs = None
        ignored_values = record.get("project_ignored_token_ids")
        if ignored_values is None:
            ignored_values = [60514]
        ignored_ids = tuple(int(value) for value in ignored_values)
        project_blank_id = int(record.get("project_blank_id", int(blank_id)))
        raw_teacher_blank_id = record.get("teacher_blank_id")
        teacher_blank_id = None if raw_teacher_blank_id is None else int(raw_teacher_blank_id)
        teacher_full_log_probs = _project_ctc_teacher_full_log_probs(
            teacher_full_log_probs,
            vocab_size=int(student_logits.size(-1)),
            blank_id=int(blank_id),
            teacher_blank_id=teacher_blank_id,
            project_blank_id=project_blank_id,
            ignored_token_ids=ignored_ids,
        )
        if str(frame_filter or "all") != "all":
            raw_ids = _ctc_teacher_topk_field(record, "topk_token_ids", "topk_ids", "token_ids", "ids")
            raw_log_probs = _ctc_teacher_topk_field(
                record,
                "topk_log_probs",
                "topk_logprobs",
                "log_probs",
                "logprobs",
            )
            if raw_ids is None or raw_log_probs is None:
                missing += 1
                if eval_accumulator is not None:
                    eval_accumulator[_CTC_FULL_EVAL_MISSING_UTTERANCES] += 1.0
                if missing_policy == "error":
                    raise ValueError(
                        f"CTC teacher full record needs top-k ids/log_probs for frame filtering: utt_id={utt_id!r}"
                    )
                continue
            teacher_ids = torch.as_tensor(raw_ids, dtype=torch.long)
            teacher_log_probs = torch.as_tensor(raw_log_probs, dtype=torch.float32)
            if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2 or teacher_ids.shape != teacher_log_probs.shape:
                raise ValueError(
                    f"CTC teacher full record has invalid top-k shape for utt_id={utt_id!r}: "
                    f"ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
                )
        teacher_time = int(teacher_full_log_probs.size(0))
        student_time = int(clipped_lengths[sample_idx].item())
        if teacher_time <= 0 or student_time <= 0:
            continue
        if teacher_ids is not None and int(teacher_ids.size(0)) != teacher_time:
            raise ValueError(
                f"CTC teacher full/top-k length mismatch for utt_id={utt_id!r}: "
                f"full={teacher_time} topk={int(teacher_ids.size(0))}"
            )

        matched += 1
        student_slice = student_logits[sample_idx, :student_time].float()
        if time_map == "nearest" or student_time == 1 or teacher_time == 1:
            if student_time == 1 or teacher_time == 1:
                frame_indices = torch.zeros(student_time, device=student_logits.device, dtype=torch.long)
            else:
                positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
                frame_indices = torch.round(positions * float(teacher_time - 1) / float(student_time - 1)).to(
                    dtype=torch.long
                )
            frame_indices = frame_indices.clamp(min=0, max=teacher_time - 1)
            selected_log_probs = _ctc_teacher_time_index_select(
                teacher_full_log_probs,
                frame_indices,
            )
            if teacher_ids is not None and teacher_log_probs is not None:
                selected_ids = _ctc_teacher_time_index_select(teacher_ids, frame_indices)
                selected_topk_log_probs = _ctc_teacher_time_index_select(
                    teacher_log_probs,
                    frame_indices,
                )
                selected_mask = _ctc_teacher_frame_filter_mask(
                    selected_ids,
                    selected_topk_log_probs,
                    frame_filter=frame_filter,
                    blank_id=project_blank_id,
                    ignored_token_ids=ignored_ids,
                    neighbor_radius=int(frame_filter_neighbor_radius),
                    min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                    device=student_logits.device,
                )
            else:
                selected_mask = None
        else:
            positions = torch.arange(student_time, device=student_logits.device, dtype=torch.float32)
            scaled = positions * float(teacher_time - 1) / float(student_time - 1)
            lo = torch.floor(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            hi = torch.ceil(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            alpha = (scaled - lo.to(dtype=torch.float32)).to(
                device=teacher_full_log_probs.device,
                dtype=torch.float32,
            ).unsqueeze(-1)
            lo_log_probs = _ctc_teacher_time_index_select(teacher_full_log_probs, lo)
            hi_log_probs = _ctc_teacher_time_index_select(teacher_full_log_probs, hi)
            selected_probs = lo_log_probs.exp() * (1.0 - alpha) + hi_log_probs.exp() * alpha
            selected_log_probs = selected_probs.log()
            if teacher_ids is not None and teacher_log_probs is not None:
                lo_ids = _ctc_teacher_time_index_select(teacher_ids, lo)
                lo_topk_log_probs = _ctc_teacher_time_index_select(teacher_log_probs, lo)
                hi_ids = _ctc_teacher_time_index_select(teacher_ids, hi)
                hi_topk_log_probs = _ctc_teacher_time_index_select(teacher_log_probs, hi)
                selected_mask = _ctc_teacher_frame_filter_mask(
                    lo_ids,
                    lo_topk_log_probs,
                    frame_filter=frame_filter,
                    blank_id=project_blank_id,
                    ignored_token_ids=ignored_ids,
                    neighbor_radius=int(frame_filter_neighbor_radius),
                    min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                    device=student_logits.device,
                )
                hi_mask = _ctc_teacher_frame_filter_mask(
                    hi_ids,
                    hi_topk_log_probs,
                    frame_filter=frame_filter,
                    blank_id=project_blank_id,
                    ignored_token_ids=ignored_ids,
                    neighbor_radius=int(frame_filter_neighbor_radius),
                    min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                    device=student_logits.device,
                )
                if selected_mask is not None and hi_mask is not None:
                    selected_mask = selected_mask | hi_mask
            else:
                selected_mask = None
        frame_loss = _ctc_teacher_full_frame_kl(
            student_slice,
            selected_log_probs,
            temperature=float(temperature),
        )
        frame_weights = torch.ones_like(frame_loss)
        if float(nonblank_frame_weight) != 1.0:
            teacher_top1 = selected_log_probs.argmax(dim=-1)
            nonblank_mask = teacher_top1.ne(int(project_blank_id))
            for ignored_id in ignored_ids:
                nonblank_mask = nonblank_mask & teacher_top1.ne(int(ignored_id))
            frame_weights = torch.where(
                nonblank_mask.to(device=frame_loss.device),
                frame_weights.new_full((), float(nonblank_frame_weight)),
                frame_weights,
            )
        if eval_accumulator is not None:
            _accumulate_ctc_full_eval_metrics(
                eval_accumulator,
                student_logits=student_slice,
                teacher_log_probs=selected_log_probs,
                frame_kl=frame_loss,
                selected_mask=selected_mask,
                blank_id=project_blank_id,
                ignored_token_ids=ignored_ids,
                frame_delta=student_time - teacher_time,
            )
        if selected_mask is not None:
            if not bool(selected_mask.any().item()):
                continue
            frame_loss = frame_loss[selected_mask]
            frame_weights = frame_weights[selected_mask]
        total = total + (frame_loss * frame_weights).sum()
        denom = denom + frame_weights.sum()

    return total / denom.clamp_min(1.0), matched, missing


def _ctc_teacher_hidden_loss(
    student_hidden: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    teacher_field: str,
    time_map: str,
    missing_policy: str,
    blank_id: int = 0,
    frame_filter: str = "all",
    frame_filter_neighbor_radius: int = 0,
    frame_filter_min_nonblank_prob: float = 0.0,
) -> tuple[torch.Tensor, int, int]:
    if utt_ids is None:
        raise RuntimeError("CTC teacher hidden distillation requires batch.utt_ids.")
    if time_map not in {"nearest", "linear"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_time_map={time_map!r}; expected nearest or linear.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")

    batch_size = min(int(student_hidden.size(0)), len(utt_ids))
    total = student_hidden.new_zeros((), dtype=torch.float32)
    denom = student_hidden.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    max_student_time = int(student_hidden.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_hidden.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_hidden.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing CTC teacher hidden record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        raw_hidden = record.get(teacher_field)
        if raw_hidden is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"CTC teacher hidden record has no {teacher_field!r} for utt_id={utt_id!r}")
            continue
        teacher_hidden = torch.as_tensor(raw_hidden, dtype=torch.float32)
        if teacher_hidden.ndim != 2:
            raise ValueError(
                f"CTC teacher hidden record field {teacher_field!r} must be 2-D for utt_id={utt_id!r}, "
                f"got {tuple(teacher_hidden.shape)}"
            )
        if int(teacher_hidden.size(-1)) != int(student_hidden.size(-1)):
            raise ValueError(
                f"CTC teacher hidden dim mismatch for utt_id={utt_id!r}: "
                f"teacher={int(teacher_hidden.size(-1))} student={int(student_hidden.size(-1))}"
            )
        teacher_time = int(teacher_hidden.size(0))
        student_time = int(clipped_lengths[sample_idx].item())
        if teacher_time <= 0 or student_time <= 0:
            continue

        matched += 1
        student_slice = student_hidden[sample_idx, :student_time].float()
        teacher_filter_mask: torch.Tensor | None = None
        if str(frame_filter or "all") != "all":
            raw_ids = _ctc_teacher_topk_field(record, "topk_token_ids", "topk_ids", "token_ids", "ids")
            raw_log_probs = _ctc_teacher_topk_field(
                record,
                "topk_log_probs",
                "topk_logprobs",
                "log_probs",
                "logprobs",
            )
            if raw_ids is None or raw_log_probs is None:
                missing += 1
                if missing_policy == "error":
                    raise ValueError(
                        f"CTC teacher hidden frame filtering needs ids/log_probs for utt_id={utt_id!r}"
                    )
                continue
            teacher_ids = torch.as_tensor(raw_ids, dtype=torch.long)
            teacher_log_probs = torch.as_tensor(raw_log_probs, dtype=torch.float32)
            if teacher_ids.ndim != 2 or teacher_log_probs.ndim != 2 or teacher_ids.shape != teacher_log_probs.shape:
                raise ValueError(
                    f"CTC teacher hidden frame filter has invalid shape for utt_id={utt_id!r}: "
                    f"ids={tuple(teacher_ids.shape)} log_probs={tuple(teacher_log_probs.shape)}"
                )
            ignored_values = record.get("project_ignored_token_ids")
            if ignored_values is None:
                ignored_values = [60514]
            teacher_filter_mask = _ctc_teacher_frame_filter_mask(
                teacher_ids,
                teacher_log_probs,
                frame_filter=str(frame_filter),
                blank_id=int(record.get("project_blank_id", int(blank_id))),
                ignored_token_ids=tuple(int(value) for value in ignored_values),
                neighbor_radius=int(frame_filter_neighbor_radius),
                min_nonblank_prob=float(frame_filter_min_nonblank_prob),
                device=student_hidden.device,
            )
            if teacher_filter_mask is not None and int(teacher_filter_mask.numel()) != teacher_time:
                filter_time = int(teacher_filter_mask.numel())
                if filter_time <= 0:
                    teacher_filter_mask = torch.zeros((teacher_time,), dtype=torch.bool, device=student_hidden.device)
                elif teacher_time == 1 or filter_time == 1:
                    teacher_filter_mask = teacher_filter_mask[:1].expand(teacher_time)
                else:
                    filter_positions = torch.arange(teacher_time, device=student_hidden.device, dtype=torch.float32)
                    filter_indices = torch.round(
                        filter_positions * float(filter_time - 1) / float(teacher_time - 1)
                    ).to(dtype=torch.long)
                    filter_indices = filter_indices.clamp(min=0, max=filter_time - 1)
                    teacher_filter_mask = teacher_filter_mask.index_select(0, filter_indices)
        if time_map == "nearest" or student_time == 1 or teacher_time == 1:
            if student_time == 1 or teacher_time == 1:
                frame_indices_device = torch.zeros(student_time, device=student_hidden.device, dtype=torch.long)
            else:
                positions = torch.arange(student_time, device=student_hidden.device, dtype=torch.float32)
                frame_indices_device = torch.round(positions * float(teacher_time - 1) / float(student_time - 1)).to(
                    dtype=torch.long
                )
            frame_indices_device = frame_indices_device.clamp(min=0, max=teacher_time - 1)
            selected_hidden = teacher_hidden.index_select(0, frame_indices_device.cpu())
            selected_frame_mask = (
                teacher_filter_mask.index_select(0, frame_indices_device)
                if teacher_filter_mask is not None
                else None
            )
        else:
            positions = torch.arange(student_time, device=student_hidden.device, dtype=torch.float32)
            scaled = positions * float(teacher_time - 1) / float(student_time - 1)
            lo = torch.floor(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            hi = torch.ceil(scaled).to(dtype=torch.long).clamp(min=0, max=teacher_time - 1)
            alpha = (scaled - lo.to(dtype=torch.float32)).to(dtype=torch.float32).cpu().unsqueeze(-1)
            lo_hidden = teacher_hidden.index_select(0, lo.cpu())
            hi_hidden = teacher_hidden.index_select(0, hi.cpu())
            selected_hidden = lo_hidden * (1.0 - alpha) + hi_hidden * alpha
            selected_frame_mask = (
                teacher_filter_mask.index_select(0, lo) | teacher_filter_mask.index_select(0, hi)
                if teacher_filter_mask is not None
                else None
            )
        selected_hidden = selected_hidden.to(device=student_hidden.device, dtype=torch.float32)
        frame_loss = F.mse_loss(student_slice, selected_hidden, reduction="none").mean(dim=-1)
        if selected_frame_mask is not None:
            selected_frame_mask = selected_frame_mask.to(device=frame_loss.device, dtype=torch.bool)
            if not bool(selected_frame_mask.any().item()):
                continue
            frame_loss = frame_loss.masked_select(selected_frame_mask)
        total = total + frame_loss.sum()
        denom = denom + frame_loss.new_tensor(float(frame_loss.numel()))

    return total / denom.clamp_min(1.0), matched, missing


def _ctc_teacher_sequence_token_ids_from_record(
    record: dict[str, Any],
    *,
    vocab_size: int,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
) -> torch.Tensor | None:
    raw_tokens = _ctc_teacher_topk_field(
        record,
        "argmax_token_ids",
        "ctc_argmax_token_ids",
        "token_ids",
        "funasr_ctc_token_ids",
    )
    if raw_tokens is None:
        return None
    tokens = torch.as_tensor(raw_tokens, dtype=torch.long).flatten()
    if tokens.numel() == 0:
        return tokens
    valid = (tokens >= 0) & (tokens < int(vocab_size)) & (tokens != int(blank_id))
    for ignored_id in ignored_token_ids:
        valid = valid & (tokens != int(ignored_id))
    return tokens[valid].contiguous()


def _ctc_teacher_sequence_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
    missing_policy: str,
) -> tuple[torch.Tensor, int, int]:
    if utt_ids is None:
        raise RuntimeError("Online CTC teacher sequence distillation requires batch.utt_ids.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    selected_indices: list[int] = []
    target_chunks: list[torch.Tensor] = []
    target_lengths: list[int] = []
    matched = 0
    missing = 0
    vocab_size = int(student_logits.size(-1))
    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing online CTC teacher sequence record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        token_ids = _ctc_teacher_sequence_token_ids_from_record(
            record,
            vocab_size=vocab_size,
            blank_id=int(blank_id),
            ignored_token_ids=ignored_token_ids,
        )
        if token_ids is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"Online CTC teacher record has no argmax_token_ids for utt_id={utt_id!r}")
            continue
        input_length = int(clipped_lengths[sample_idx].item())
        target_length = int(token_ids.numel())
        if input_length <= 0 or target_length > input_length:
            missing += 1
            if missing_policy == "error":
                raise ValueError(
                    "Online CTC teacher sequence target is incompatible with student length "
                    f"for utt_id={utt_id!r}: target_length={target_length} input_length={input_length}"
                )
            continue
        selected_indices.append(sample_idx)
        target_chunks.append(token_ids)
        target_lengths.append(target_length)
        matched += 1

    if not selected_indices:
        return student_logits.new_zeros((), dtype=torch.float32), matched, missing

    index = torch.tensor(selected_indices, device=student_logits.device, dtype=torch.long)
    selected_logits = student_logits.index_select(0, index).float()
    selected_lengths = clipped_lengths.index_select(0, index)
    if target_chunks:
        targets = torch.cat(target_chunks).to(device=student_logits.device, dtype=torch.long)
    else:
        targets = torch.empty((0,), device=student_logits.device, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, device=student_logits.device, dtype=torch.long)
    log_probs = F.log_softmax(selected_logits, dim=-1).transpose(0, 1).contiguous()
    loss = F.ctc_loss(
        log_probs,
        targets,
        selected_lengths,
        target_lengths_tensor,
        blank=int(blank_id),
        reduction="mean",
        zero_infinity=True,
    )
    return loss, matched, missing


def _ctc_teacher_sequence_presence_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
    missing_policy: str,
) -> tuple[torch.Tensor, int, int, int]:
    if utt_ids is None:
        raise RuntimeError("Online CTC teacher sequence presence distillation requires batch.utt_ids.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    token_count = 0
    vocab_size = int(student_logits.size(-1))
    eps = 1.0e-6
    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing online CTC teacher sequence presence record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        token_ids = _ctc_teacher_sequence_token_ids_from_record(
            record,
            vocab_size=vocab_size,
            blank_id=int(blank_id),
            ignored_token_ids=ignored_token_ids,
        )
        if token_ids is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"Online CTC teacher record has no argmax_token_ids for utt_id={utt_id!r}")
            continue
        input_length = int(clipped_lengths[sample_idx].item())
        if input_length <= 0:
            missing += 1
            if missing_policy == "error":
                raise ValueError(
                    f"Online CTC teacher sequence presence has no student frames for utt_id={utt_id!r}"
                )
            continue
        matched += 1
        if int(token_ids.numel()) <= 0:
            continue

        targets = token_ids.to(device=student_logits.device, dtype=torch.long)
        frame_probs = F.softmax(student_logits[sample_idx, :input_length].float(), dim=-1)
        token_probs = frame_probs.index_select(dim=-1, index=targets).clamp(min=0.0, max=1.0 - eps)
        log_no_presence = torch.log1p(-token_probs).sum(dim=0)
        presence_probs = (-torch.expm1(log_no_presence)).clamp(min=1.0e-8, max=1.0)
        token_losses = -torch.log(presence_probs)
        total = total + token_losses.sum()
        denom = denom + token_losses.new_tensor(float(int(token_losses.numel())))
        token_count += int(token_losses.numel())

    return total / denom.clamp_min(1.0), matched, missing, token_count


def _ctc_teacher_sequence_window_loss(
    student_logits: torch.Tensor,
    student_lengths: torch.Tensor | None,
    utt_ids: tuple[str, ...] | list[str] | None,
    teacher_cache: dict[str, dict[str, Any]],
    *,
    blank_id: int,
    ignored_token_ids: tuple[int, ...],
    missing_policy: str,
    radius: int,
    temperature: float,
) -> tuple[torch.Tensor, int, int, int]:
    if utt_ids is None:
        raise RuntimeError("Online CTC teacher sequence-window distillation requires batch.utt_ids.")
    if missing_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported ctc_teacher_topk_missing_policy={missing_policy!r}; expected skip or error.")
    if radius < 0:
        raise ValueError(f"ctc_teacher_online_sequence_window_radius must be >= 0, got {radius}.")
    if temperature < 0.0:
        raise ValueError(
            f"ctc_teacher_online_sequence_window_temperature must be >= 0, got {temperature}."
        )

    batch_size = min(int(student_logits.size(0)), len(utt_ids))
    max_student_time = int(student_logits.size(1))
    if student_lengths is None:
        clipped_lengths = torch.full(
            (batch_size,),
            max_student_time,
            dtype=torch.long,
            device=student_logits.device,
        )
    else:
        clipped_lengths = student_lengths.to(device=student_logits.device, dtype=torch.long).clamp(
            min=0,
            max=max_student_time,
        )

    log_probs = F.log_softmax(student_logits.float(), dim=-1)
    total = student_logits.new_zeros((), dtype=torch.float32)
    denom = student_logits.new_zeros((), dtype=torch.float32)
    matched = 0
    missing = 0
    events = 0
    vocab_size = int(student_logits.size(-1))
    for sample_idx in range(batch_size):
        utt_id = str(utt_ids[sample_idx])
        record = teacher_cache.get(utt_id)
        if record is None:
            missing += 1
            if missing_policy == "error":
                raise KeyError(f"Missing online CTC teacher sequence-window record for utt_id={utt_id!r}")
            continue
        record = _materialize_ctc_teacher_topk_record(record)
        token_ids = _ctc_teacher_sequence_token_ids_from_record(
            record,
            vocab_size=vocab_size,
            blank_id=int(blank_id),
            ignored_token_ids=ignored_token_ids,
        )
        if token_ids is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"Online CTC teacher record has no argmax_token_ids for utt_id={utt_id!r}")
            continue
        input_length = int(clipped_lengths[sample_idx].item())
        if input_length <= 0:
            missing += 1
            if missing_policy == "error":
                raise ValueError(
                    "Online CTC teacher sequence-window target is incompatible with student length "
                    f"for utt_id={utt_id!r}: input_length={input_length}"
                )
            continue
        matched += 1
        target_length = int(token_ids.numel())
        if target_length == 0:
            continue
        tokens = token_ids.to(device=student_logits.device, dtype=torch.long)
        if target_length == 1:
            centers = torch.full((1,), input_length // 2, device=student_logits.device, dtype=torch.long)
        else:
            positions = torch.arange(target_length, device=student_logits.device, dtype=torch.float32)
            centers = torch.round(positions * float(input_length - 1) / float(target_length - 1)).to(
                dtype=torch.long
            )
        sample_log_probs = log_probs[sample_idx, :input_length]
        for center, token_id in zip(centers.tolist(), tokens.tolist(), strict=True):
            lo = max(0, int(center) - int(radius))
            hi = min(input_length - 1, int(center) + int(radius))
            frame_losses = -sample_log_probs[lo : hi + 1, int(token_id)]
            if float(temperature) > 0.0:
                window_loss = -float(temperature) * torch.logsumexp(
                    -frame_losses / float(temperature),
                    dim=0,
                )
            else:
                window_loss = frame_losses.min()
            total = total + window_loss
            denom = denom + frame_losses.new_tensor(1.0)
            events += 1

    if events == 0:
        return student_logits.new_zeros((), dtype=torch.float32), matched, missing, events
    return total / denom.clamp_min(1.0), matched, missing, events


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
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        ds_dir = record.get("deepspeed_checkpoint_dir")
        if ds_dir is not None and str(ds_dir) not in keep_ds_dirs:
            path = Path(str(ds_dir))
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass


def _step_checkpoint_record_is_retained(
    *,
    record: dict[str, Any],
    top_records: list[dict[str, Any]],
) -> bool:
    checkpoint_path = record.get("checkpoint_path")
    ds_dir = record.get("deepspeed_checkpoint_dir")
    for top_record in top_records:
        if checkpoint_path is not None and top_record.get("checkpoint_path") == checkpoint_path:
            return True
        if ds_dir is not None and top_record.get("deepspeed_checkpoint_dir") == ds_dir:
            return True
    return False


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


def _resolve_ctc_teacher_online_device(
    configured_device: str | None,
    runtime_device: torch.device,
    *,
    local_rank: int,
) -> str:
    if configured_device is not None:
        return str(configured_device)
    if runtime_device.type != "cuda":
        return str(runtime_device)
    device_index = runtime_device.index
    if device_index is None and local_rank >= 0:
        device_index = local_rank
    if device_index is None:
        device_index = 0
    return f"cuda:{int(device_index)}"


def _save_export_checkpoints(
    *,
    engine: deepspeed.DeepSpeedEngine,
    output_dir: Path,
    tag: str,
    export_name: str,
    step: int,
    zero_stage: int,
    extra_state: dict[str, Any],
    save_deepspeed_sharded: bool = True,
) -> dict[str, str | None]:
    if "epoch_batch_offset" not in extra_state:
        raise ValueError("extra_state must include epoch_batch_offset for latest checkpoint tracking.")

    ds_checkpoint_dir: Path | None = None
    if save_deepspeed_sharded:
        ds_checkpoint_root = output_dir / "ds_checkpoints"
        ds_checkpoint_root.mkdir(parents=True, exist_ok=True)
        ds_checkpoint_dir = ds_checkpoint_root / tag
        engine.save_checkpoint(
            str(ds_checkpoint_root),
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
    if _is_rank_zero():
        latest_state: dict[str, Any] = {
            "checkpoint_type": "deepspeed" if save_deepspeed_sharded else "export",
            "step": int(step),
            "epoch": int(extra_state.get("epoch", 0)),
            "epoch_batch_offset": int(extra_state["epoch_batch_offset"]),
        }
        if export_path is not None:
            latest_state["checkpoint_path"] = str(export_path)
        if ds_checkpoint_dir is not None:
            latest_state["deepspeed_checkpoint_dir"] = str(ds_checkpoint_dir)
            latest_state["resume_tag"] = tag
        write_latest_checkpoint_state(output_dir, latest_state)
    return {
        "checkpoint_path": str(export_path) if export_path is not None else None,
        "deepspeed_checkpoint_dir": str(ds_checkpoint_dir) if ds_checkpoint_dir is not None else None,
        "resume_tag": tag if ds_checkpoint_dir is not None else None,
    }


def train_ctc_model_deepspeed(config: DeepSpeedTrainConfig) -> dict[str, float | int | str]:
    ds_config = _normalize_deepspeed_config(config)
    feature_dtype = torch.bfloat16 if bool(ds_config.get("bf16", {}).get("enabled")) else None
    if config.frontend_type == "funasr_nano_encoder":
        # FunASR SenseVoice's FSMN conv path keeps fp32 weights under DeepSpeed
        # bf16, so feeding bf16 features triggers a dtype mismatch.
        feature_dtype = None
    zero_stage = int(ds_config.get("zero_optimization", {}).get("stage", 0))
    if not bool(config.save_deepspeed_sharded_checkpoints) and zero_stage >= 3:
        raise ValueError("save_deepspeed_sharded_checkpoints=False is only supported for ZeRO stage < 3.")
    grad_accum = int(ds_config["gradient_accumulation_steps"])
    local_rank, device = _init_deepspeed_runtime(config)
    _rank_zero_log(
        f"Distributed init complete. world_size={_world_size()} local_rank={local_rank} device={device} zero_stage={zero_stage}"
    )

    output_dir = Path(config.output_dir)
    if _is_rank_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ds_checkpoints").mkdir(parents=True, exist_ok=True)
    _maybe_barrier()

    resolved_vocab_size = _resolve_ctc_vocab_size_for_deepspeed(config)
    resolved_max_steps, steps_per_epoch = _resolve_max_steps(config, grad_accum)
    resolved_cmvn_file = _resolve_cmvn_file_distributed(config, output_dir)
    _rank_zero_log(
        f"Training config resolved. vocab_size={resolved_vocab_size} max_steps={resolved_max_steps} steps_per_epoch={steps_per_epoch}"
    )
    model_config = RWKVCTCModelConfig(
        feature_extractor_type=config.feature_extractor_type,
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
        blank_id=config.blank_id,
        frontend_type=config.frontend_type,
        encoder_output_dim=config.encoder_output_dim,
        aut_downsample_hidden_size=config.aut_downsample_hidden_size,
        aut_activation_function=config.aut_activation_function,
        aut_activation_dropout=config.aut_activation_dropout,
        aut_max_source_positions=config.aut_max_source_positions,
        aut_scale_embedding=config.aut_scale_embedding,
        aut_conv_chunksize=config.aut_conv_chunksize,
        sensevoice_tp_blocks=config.sensevoice_tp_blocks,
        cmvn_file=resolved_cmvn_file,
        cmvn_is_json=config.cmvn_is_json,
        decoder_enabled=config.decoder_enabled,
        decoder_checkpoint_path=config.decoder_checkpoint_path,
        decoder_num_layers=config.decoder_num_layers,
        decoder_n_embd=config.decoder_n_embd,
        decoder_ffn_hidden_size=config.decoder_ffn_hidden_size,
        decoder_vocab_size=_resolve_decoder_vocab_size(config),
        decoder_head_size=config.decoder_head_size,
        decoder_audio_conditioning=config.decoder_audio_conditioning,
        decoder_prefix_tokens=config.decoder_prefix_tokens,
        decoder_loss_chunk_size=config.decoder_loss_chunk_size,
        **_resolve_decoder_template_token_ids_for_deepspeed(config),
        ctc_loss_weight=config.ctc_loss_weight,
            decoder_loss_weight=config.decoder_loss_weight,
            ctc_decoder_type=config.ctc_decoder_type,
            ctc_decoder_downsample_rate=config.ctc_decoder_downsample_rate,
            ctc_decoder_dim=config.ctc_decoder_dim,
            ctc_decoder_ffn_dim=config.ctc_decoder_ffn_dim,
            ctc_decoder_num_layers=config.ctc_decoder_num_layers,
            ctc_decoder_attention_heads=config.ctc_decoder_attention_heads,
            ctc_decoder_dropout=config.ctc_decoder_dropout,
            ctc_decoder_attention_dropout=config.ctc_decoder_attention_dropout,
            ctc_bridge_type=config.ctc_bridge_type,
            ctc_bridge_hidden_dim=config.ctc_bridge_hidden_dim,
            ctc_bridge_dropout=config.ctc_bridge_dropout,
            ctc_suppressed_token_ids=_resolve_ctc_suppressed_token_ids(
                config,
                vocab_size=max(int(resolved_vocab_size), int(config.blank_id) + 1),
            ),
        )
    decoder_text_tokens_per_sample_extra = int(model_config.decoder_text_tokens_per_sample_extra)

    if _is_rank_zero():
        save_yaml(output_dir / "model_config.yaml", model_config)
        save_yaml(
            output_dir / "tokenizer_config.yaml",
            _resolved_tokenizer_config_payload_for_deepspeed(config, vocab_size=resolved_vocab_size),
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
    _maybe_load_encoder_init_checkpoint_distributed(model, config.encoder_init_checkpoint_path)
    _maybe_load_initial_model_checkpoint(model, config)
    _maybe_load_funasr_nano_ctc_init_distributed(model, config)
    _apply_training_freeze(model, config)
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
    encoder_anchor_model: RWKVCTCModel | None = None
    encoder_anchor_weight = float(config.encoder_anchor_loss_weight)
    ctc_logit_anchor_weight = float(config.ctc_logit_anchor_loss_weight)
    if encoder_anchor_weight > 0.0 or ctc_logit_anchor_weight > 0.0:
        if config.encoder_anchor_checkpoint_path is None:
            raise ValueError(
                "anchor loss weights require encoder_anchor_checkpoint_path "
                "(encoder_anchor_loss_weight or ctc_logit_anchor_loss_weight > 0)."
            )
        encoder_anchor_model = _load_encoder_anchor_model(
            model_config=model_config,
            checkpoint_path=config.encoder_anchor_checkpoint_path,
        )
        encoder_anchor_model.to(engine.device)
        if feature_dtype is not None:
            encoder_anchor_model.to(dtype=feature_dtype)
        encoder_anchor_model.eval()
        _rank_zero_log(
            "Anchor teacher enabled: "
            f"encoder_hidden_weight={encoder_anchor_weight:g} ctc_logit_weight={ctc_logit_anchor_weight:g} "
            f"ctc_chunk={int(config.ctc_logit_anchor_chunk_frames)} "
            f"checkpoint={config.encoder_anchor_checkpoint_path}"
        )
    ctc_teacher_topk_weight = float(config.ctc_teacher_topk_loss_weight)
    ctc_teacher_topk_blank_weight = float(config.ctc_teacher_topk_blank_loss_weight)
    ctc_teacher_topk_mass_weight = float(config.ctc_teacher_topk_mass_loss_weight)
    ctc_teacher_frame_filter = str(config.ctc_teacher_frame_filter or "all")
    ctc_teacher_frame_filter_neighbor_radius = int(config.ctc_teacher_frame_filter_neighbor_radius)
    ctc_teacher_frame_filter_min_nonblank_prob = float(config.ctc_teacher_frame_filter_min_nonblank_prob)
    ctc_teacher_online_weight = float(config.ctc_teacher_online_loss_weight)
    ctc_teacher_online_blank_weight = float(config.ctc_teacher_online_blank_loss_weight)
    ctc_teacher_online_mass_weight = float(config.ctc_teacher_online_mass_loss_weight)
    ctc_teacher_online_full_weight = float(config.ctc_teacher_online_full_loss_weight)
    ctc_teacher_online_full_temperature = float(config.ctc_teacher_online_full_temperature)
    ctc_teacher_online_full_nonblank_weight = float(config.ctc_teacher_online_full_nonblank_weight)
    ctc_teacher_online_full_frame_filter = (
        ctc_teacher_frame_filter
        if config.ctc_teacher_online_full_frame_filter is None
        else str(config.ctc_teacher_online_full_frame_filter)
    )
    ctc_teacher_online_encoder_weight = float(config.ctc_teacher_online_encoder_loss_weight)
    ctc_teacher_online_sequence_weight = float(config.ctc_teacher_online_sequence_loss_weight)
    ctc_teacher_online_sequence_presence_weight = float(
        config.ctc_teacher_online_sequence_presence_loss_weight
    )
    ctc_teacher_online_sequence_window_weight = float(config.ctc_teacher_online_sequence_window_loss_weight)
    ctc_teacher_online_sequence_window_radius = int(config.ctc_teacher_online_sequence_window_radius)
    ctc_teacher_online_sequence_window_temperature = float(
        config.ctc_teacher_online_sequence_window_temperature
    )
    ctc_teacher_online_nonblank_hard_weight = float(config.ctc_teacher_online_nonblank_hard_loss_weight)
    ctc_teacher_online_nonblank_margin_weight = float(config.ctc_teacher_online_nonblank_margin_loss_weight)
    ctc_teacher_online_nonblank_margin = float(config.ctc_teacher_online_nonblank_margin)
    ctc_teacher_online_nonblank_window_weight = float(config.ctc_teacher_online_nonblank_window_loss_weight)
    ctc_teacher_online_nonblank_window_margin_weight = float(
        config.ctc_teacher_online_nonblank_window_margin_loss_weight
    )
    ctc_teacher_online_nonblank_window_topk_weight = float(
        config.ctc_teacher_online_nonblank_window_topk_loss_weight
    )
    ctc_teacher_online_nonblank_window_radius = int(config.ctc_teacher_online_nonblank_window_radius)
    ctc_teacher_online_nonblank_window_temperature = float(
        config.ctc_teacher_online_nonblank_window_temperature
    )
    ctc_teacher_online_layer_component_weights = {
        "mixer": float(config.ctc_teacher_online_layer_mixer_loss_weight),
        "ffn": float(config.ctc_teacher_online_layer_ffn_loss_weight),
        "block": float(config.ctc_teacher_online_layer_block_loss_weight),
    }
    ctc_teacher_online_layer_enabled = any(
        value > 0.0 for value in ctc_teacher_online_layer_component_weights.values()
    )
    ctc_teacher_online_layer_only = _is_layer_hidden_only_objective(config)
    ctc_teacher_online_layer_input_mode = str(config.ctc_teacher_online_layer_input_mode)
    if config.allow_missing_targets and (
        float(config.ctc_loss_weight) > 0.0 or float(config.decoder_loss_weight) > 0.0
    ):
        raise ValueError(
            "allow_missing_targets=True is only valid for unsupervised/distillation runs; "
            "set ctc_loss_weight=0.0 and decoder_loss_weight=0.0."
        )
    if (
        float(config.ctc_loss_weight) <= 0.0
        and float(config.decoder_loss_weight) <= 0.0
        and encoder_anchor_weight <= 0.0
        and ctc_logit_anchor_weight <= 0.0
        and ctc_teacher_topk_weight <= 0.0
        and ctc_teacher_topk_blank_weight <= 0.0
        and ctc_teacher_topk_mass_weight <= 0.0
        and ctc_teacher_online_weight <= 0.0
        and ctc_teacher_online_blank_weight <= 0.0
        and ctc_teacher_online_mass_weight <= 0.0
        and ctc_teacher_online_full_weight <= 0.0
        and ctc_teacher_online_encoder_weight <= 0.0
        and ctc_teacher_online_sequence_weight <= 0.0
        and ctc_teacher_online_sequence_presence_weight <= 0.0
        and ctc_teacher_online_sequence_window_weight <= 0.0
        and ctc_teacher_online_nonblank_hard_weight <= 0.0
        and ctc_teacher_online_nonblank_margin_weight <= 0.0
        and ctc_teacher_online_nonblank_window_weight <= 0.0
        and ctc_teacher_online_nonblank_window_margin_weight <= 0.0
        and ctc_teacher_online_nonblank_window_topk_weight <= 0.0
        and not ctc_teacher_online_layer_enabled
    ):
        raise ValueError("Training objective is empty: all supervised, anchor, and teacher loss weights are zero.")
    ctc_teacher_topk_cache: dict[str, dict[str, Any]] = {}
    if (
        ctc_teacher_topk_weight > 0.0
        or ctc_teacher_topk_blank_weight > 0.0
        or ctc_teacher_topk_mass_weight > 0.0
        or ctc_teacher_online_weight > 0.0
        or ctc_teacher_online_blank_weight > 0.0
        or ctc_teacher_online_mass_weight > 0.0
        or ctc_teacher_online_full_weight > 0.0
        or ctc_teacher_online_encoder_weight > 0.0
        or ctc_teacher_online_sequence_weight > 0.0
        or ctc_teacher_online_sequence_presence_weight > 0.0
        or ctc_teacher_online_sequence_window_weight > 0.0
        or ctc_teacher_online_nonblank_hard_weight > 0.0
        or ctc_teacher_online_nonblank_margin_weight > 0.0
        or ctc_teacher_online_nonblank_window_weight > 0.0
        or ctc_teacher_online_nonblank_window_margin_weight > 0.0
        or ctc_teacher_online_nonblank_window_topk_weight > 0.0
        or ctc_teacher_online_layer_enabled
    ):
        if config.ctc_teacher_topk_time_map not in {"nearest", "linear"}:
            raise ValueError(
                "ctc_teacher_topk_time_map must be 'nearest' or 'linear', "
                f"got {config.ctc_teacher_topk_time_map!r}."
            )
        if config.ctc_teacher_topk_missing_policy not in {"skip", "error"}:
            raise ValueError(
                "ctc_teacher_topk_missing_policy must be 'skip' or 'error', "
                f"got {config.ctc_teacher_topk_missing_policy!r}."
            )
        if ctc_teacher_frame_filter not in {"all", "nonblank", "nonblank_neighbors"}:
            raise ValueError(
                "ctc_teacher_frame_filter must be 'all', 'nonblank', or 'nonblank_neighbors', "
                f"got {ctc_teacher_frame_filter!r}."
            )
        if ctc_teacher_online_full_frame_filter not in {"all", "nonblank", "nonblank_neighbors"}:
            raise ValueError(
                "ctc_teacher_online_full_frame_filter must be null, 'all', 'nonblank', or 'nonblank_neighbors', "
                f"got {ctc_teacher_online_full_frame_filter!r}."
            )
        if ctc_teacher_frame_filter_neighbor_radius < 0:
            raise ValueError(
                "ctc_teacher_frame_filter_neighbor_radius must be >= 0, "
                f"got {ctc_teacher_frame_filter_neighbor_radius!r}."
            )
        if ctc_teacher_frame_filter_min_nonblank_prob < 0.0 or ctc_teacher_frame_filter_min_nonblank_prob > 1.0:
            raise ValueError(
                "ctc_teacher_frame_filter_min_nonblank_prob must be in [0, 1], "
                f"got {ctc_teacher_frame_filter_min_nonblank_prob!r}."
            )
        if ctc_teacher_online_full_temperature <= 0.0:
            raise ValueError(
                "ctc_teacher_online_full_temperature must be > 0, "
                f"got {ctc_teacher_online_full_temperature!r}."
            )
        if (
            not math.isfinite(ctc_teacher_online_full_nonblank_weight)
            or ctc_teacher_online_full_nonblank_weight <= 0.0
        ):
            raise ValueError(
                "ctc_teacher_online_full_nonblank_weight must be finite and > 0, "
                f"got {ctc_teacher_online_full_nonblank_weight!r}."
            )
        if ctc_teacher_online_nonblank_margin < 0.0:
            raise ValueError(
                "ctc_teacher_online_nonblank_margin must be >= 0, "
                f"got {ctc_teacher_online_nonblank_margin!r}."
            )
        if ctc_teacher_online_nonblank_window_radius < 0:
            raise ValueError(
                "ctc_teacher_online_nonblank_window_radius must be >= 0, "
                f"got {ctc_teacher_online_nonblank_window_radius!r}."
            )
        if ctc_teacher_online_nonblank_window_temperature < 0.0:
            raise ValueError(
                "ctc_teacher_online_nonblank_window_temperature must be >= 0, "
                f"got {ctc_teacher_online_nonblank_window_temperature!r}."
            )
        if ctc_teacher_online_sequence_window_radius < 0:
            raise ValueError(
                "ctc_teacher_online_sequence_window_radius must be >= 0, "
                f"got {ctc_teacher_online_sequence_window_radius!r}."
            )
        if ctc_teacher_online_sequence_window_temperature < 0.0:
            raise ValueError(
                "ctc_teacher_online_sequence_window_temperature must be >= 0, "
                f"got {ctc_teacher_online_sequence_window_temperature!r}."
            )
        if any(value < 0.0 for value in ctc_teacher_online_layer_component_weights.values()):
            raise ValueError(
                "ctc_teacher_online_layer component weights must be non-negative, "
                f"got {ctc_teacher_online_layer_component_weights}."
            )
        layer_metric_weights = (
            float(config.ctc_teacher_online_layer_normalized_mse_weight),
            float(config.ctc_teacher_online_layer_cosine_weight),
            float(config.ctc_teacher_online_layer_energy_mse_weight),
            float(config.ctc_teacher_online_layer_log_rms_weight),
            float(config.ctc_teacher_online_layer_raw_mse_weight),
        )
        if any(value < 0.0 for value in layer_metric_weights) or not any(
            value > 0.0 for value in layer_metric_weights
        ):
            raise ValueError(
                "Layer hidden normalized-MSE/cosine/raw-MSE weights must be non-negative with at least one positive, "
                f"got {layer_metric_weights}."
            )
        if (
            config.ctc_teacher_online_use_batch_features
            and str(config.feature_extractor_type) != "funasr_wav_frontend"
        ):
            raise ValueError(
                "ctc_teacher_online_use_batch_features requires "
                "feature_extractor_type='funasr_wav_frontend'."
            )
        if ctc_teacher_online_layer_enabled:
            if ctc_teacher_online_layer_input_mode not in {"stacked", "teacher_forced"}:
                raise ValueError(
                    "ctc_teacher_online_layer_input_mode must be 'stacked' or 'teacher_forced', "
                    f"got {ctc_teacher_online_layer_input_mode!r}."
                )
            if int(config.ctc_teacher_online_layer_sample_count) <= 0:
                raise ValueError("ctc_teacher_online_layer_sample_count must be positive.")
            if int(config.ctc_teacher_online_layer_frame_tolerance) < 0:
                raise ValueError("ctc_teacher_online_layer_frame_tolerance must be non-negative.")
            if str(config.frontend_type) != "sensevoice_rwkv":
                raise ValueError("Layer hidden distillation requires frontend_type='sensevoice_rwkv'.")
            if str(config.feature_extractor_type) != "funasr_wav_frontend":
                raise ValueError(
                    "Nano layer hidden distillation requires feature_extractor_type='funasr_wav_frontend' "
                    "for frame and feature parity."
                )
            if not bool(config.ctc_teacher_online_use_batch_features):
                raise ValueError(
                    "Nano layer hidden distillation requires ctc_teacher_online_use_batch_features=true."
                )
            if bool(config.specaugment_enabled):
                raise ValueError("Nano layer hidden distillation requires specaugment_enabled=false.")
            if float(config.dropout) != 0.0:
                raise ValueError("Nano layer hidden distillation requires dropout=0 for exact sub-block parity.")
            if ctc_teacher_online_layer_input_mode == "teacher_forced" and not ctc_teacher_online_layer_only:
                raise ValueError("Teacher-forced layer alignment requires a hidden-only objective.")
        if (
            ctc_teacher_online_full_weight > 0.0
            and ctc_teacher_online_full_frame_filter != "all"
            and ctc_teacher_online_blank_weight <= 0.0
            and ctc_teacher_online_mass_weight <= 0.0
        ):
            _rank_zero_log(
                "Warning: online full CTC KL is not using all frames and no blank/mass loss is enabled; "
                "CTC blank prior may be under-supervised."
            )
    if (
        ctc_teacher_topk_weight > 0.0
        or ctc_teacher_topk_blank_weight > 0.0
        or ctc_teacher_topk_mass_weight > 0.0
    ):
        if config.ctc_teacher_topk_cache_path is None:
            raise ValueError("Cached CTC teacher distillation requires ctc_teacher_topk_cache_path.")
        ctc_teacher_topk_cache = _load_ctc_teacher_topk_cache(config.ctc_teacher_topk_cache_path)
        if not ctc_teacher_topk_cache:
            raise ValueError(f"CTC teacher top-k cache is empty: {config.ctc_teacher_topk_cache_path}")
        _rank_zero_log(
            "CTC teacher top-k distillation enabled: "
            f"weight={ctc_teacher_topk_weight:g} blank_weight={ctc_teacher_topk_blank_weight:g} "
            f"mass_weight={ctc_teacher_topk_mass_weight:g} "
            f"records={len(ctc_teacher_topk_cache)} "
            f"time_map={config.ctc_teacher_topk_time_map} "
            f"frame_filter={ctc_teacher_frame_filter} "
            f"filter_radius={ctc_teacher_frame_filter_neighbor_radius} "
            f"filter_min_nonblank={ctc_teacher_frame_filter_min_nonblank_prob:g} "
            f"missing_policy={config.ctc_teacher_topk_missing_policy} "
            f"cache={config.ctc_teacher_topk_cache_path}"
        )
    ctc_teacher_online: FunASRNanoCTCTopKOnlineTeacher | None = None
    if (
        ctc_teacher_online_weight > 0.0
        or ctc_teacher_online_blank_weight > 0.0
        or ctc_teacher_online_mass_weight > 0.0
        or ctc_teacher_online_full_weight > 0.0
        or ctc_teacher_online_encoder_weight > 0.0
        or ctc_teacher_online_sequence_weight > 0.0
        or ctc_teacher_online_sequence_presence_weight > 0.0
        or ctc_teacher_online_sequence_window_weight > 0.0
        or ctc_teacher_online_nonblank_hard_weight > 0.0
        or ctc_teacher_online_nonblank_margin_weight > 0.0
        or ctc_teacher_online_nonblank_window_weight > 0.0
        or ctc_teacher_online_nonblank_window_margin_weight > 0.0
        or ctc_teacher_online_nonblank_window_topk_weight > 0.0
        or ctc_teacher_online_layer_enabled
    ):
        if config.ctc_teacher_online_model_path is None:
            raise ValueError("Online CTC teacher distillation requires ctc_teacher_online_model_path.")
        audio_index_path = config.ctc_teacher_online_audio_index_path
        if (
            not config.ctc_teacher_online_use_batch_features
            and audio_index_path is None
            and config.webdataset_bucket_manifest_path is None
        ):
            audio_index_path = config.webdataset_length_index_path or config.manifest_path
        if audio_index_path is None and config.webdataset_bucket_manifest_path is None:
            raise ValueError(
                "Online CTC teacher distillation requires ctc_teacher_online_audio_index_path, "
                "webdataset_length_index_path, manifest_path, or a bucketed WebDataset batch "
                "with inline audio rows."
            )
        online_device = _resolve_ctc_teacher_online_device(
            config.ctc_teacher_online_device,
            device,
            local_rank=local_rank,
        )
        online_audio_cache_dir = config.ctc_teacher_online_audio_cache_dir
        if online_audio_cache_dir is None:
            online_audio_cache_dir = str(output_dir / "funasr_online_audio_cache" / f"rank{_rank()}")
        ctc_teacher_online = FunASRNanoCTCTopKOnlineTeacher(
            FunASROnlineCTCTeacherConfig(
                model_path=str(config.ctc_teacher_online_model_path),
                audio_index_path=str(audio_index_path) if audio_index_path is not None else None,
                webdataset_index_path=(
                    config.ctc_teacher_online_webdataset_index_path or config.webdataset_index_path
                ),
                webdataset_root=config.webdataset_root,
                audio_cache_dir=online_audio_cache_dir,
                keep_audio_cache=bool(config.ctc_teacher_online_keep_audio_cache),
                device=str(online_device),
                split=str(config.webdataset_split or "train"),
                top_k=int(config.ctc_teacher_online_top_k),
                project_blank_id=int(config.blank_id),
                project_vocab_size=int(model_config.ctc_vocab_size),
                project_ignored_token_ids=tuple(
                    int(value) for value in config.ctc_teacher_online_project_ignored_token_ids
                ),
                return_full_log_probs=ctc_teacher_online_full_weight > 0.0,
                return_encoder_out=ctc_teacher_online_encoder_weight > 0.0,
                return_layer_hiddens=ctc_teacher_online_layer_enabled,
                keep_layer_hiddens_on_device=bool(
                    config.ctc_teacher_online_keep_layer_hiddens_on_device
                ),
                keep_full_log_probs_on_device=bool(
                    config.ctc_teacher_online_keep_full_log_probs_on_device
                ),
            )
        )
        _rank_zero_log(
            "CTC online FunASR-Nano top-k distillation enabled: "
            f"weight={ctc_teacher_online_weight:g} blank_weight={ctc_teacher_online_blank_weight:g} "
            f"mass_weight={ctc_teacher_online_mass_weight:g} full_weight={ctc_teacher_online_full_weight:g} "
            f"encoder_weight={ctc_teacher_online_encoder_weight:g} "
            f"sequence_weight={ctc_teacher_online_sequence_weight:g} "
            f"sequence_presence_weight={ctc_teacher_online_sequence_presence_weight:g} "
            f"sequence_window_weight={ctc_teacher_online_sequence_window_weight:g} "
            f"sequence_window_radius={ctc_teacher_online_sequence_window_radius} "
            f"sequence_window_temperature={ctc_teacher_online_sequence_window_temperature:g} "
            f"nonblank_hard_weight={ctc_teacher_online_nonblank_hard_weight:g} "
            f"nonblank_margin_weight={ctc_teacher_online_nonblank_margin_weight:g} "
            f"nonblank_margin={ctc_teacher_online_nonblank_margin:g} "
            f"nonblank_window_weight={ctc_teacher_online_nonblank_window_weight:g} "
            f"nonblank_window_margin_weight={ctc_teacher_online_nonblank_window_margin_weight:g} "
            f"nonblank_window_topk_weight={ctc_teacher_online_nonblank_window_topk_weight:g} "
            f"nonblank_window_radius={ctc_teacher_online_nonblank_window_radius} "
            f"nonblank_window_temperature={ctc_teacher_online_nonblank_window_temperature:g} "
            f"layer_weights={ctc_teacher_online_layer_component_weights} "
            f"layer_metric_weights={layer_metric_weights} "
            f"layer_sample_count={int(config.ctc_teacher_online_layer_sample_count)} "
            f"layer_boundaries={tuple(int(value) for value in config.ctc_teacher_online_layer_boundary_ids)} "
            f"layer_frame_tolerance={int(config.ctc_teacher_online_layer_frame_tolerance)} "
            f"layer_input_mode={ctc_teacher_online_layer_input_mode} "
            f"layer_hiddens_on_device={bool(config.ctc_teacher_online_keep_layer_hiddens_on_device)} "
            f"full_log_probs_on_device={bool(config.ctc_teacher_online_keep_full_log_probs_on_device)} "
            f"layer_hidden_only={ctc_teacher_online_layer_only} "
            f"full_temperature={ctc_teacher_online_full_temperature:g} "
            f"full_nonblank_weight={ctc_teacher_online_full_nonblank_weight:g} "
            f"rows={ctc_teacher_online.num_audio_rows} "
            f"top_k={int(config.ctc_teacher_online_top_k)} "
            f"time_map={config.ctc_teacher_topk_time_map} "
            f"frame_filter={ctc_teacher_frame_filter} "
            f"full_frame_filter={ctc_teacher_online_full_frame_filter} "
            f"filter_radius={ctc_teacher_frame_filter_neighbor_radius} "
            f"filter_min_nonblank={ctc_teacher_frame_filter_min_nonblank_prob:g} "
            f"missing_policy={config.ctc_teacher_topk_missing_policy} "
            f"device={online_device} "
            f"use_batch_features={bool(config.ctc_teacher_online_use_batch_features)} "
            f"audio_index={audio_index_path if audio_index_path is not None else 'batch_inline'} "
            f"audio_cache_dir={online_audio_cache_dir} "
            f"keep_audio_cache={bool(config.ctc_teacher_online_keep_audio_cache)}"
        )
    ctc_teacher_online_layer_coverage = {
        layer_id: 0 for layer_id in range(int(config.num_layers))
    }
    _rank_zero_log("DeepSpeed engine initialized. Building dataloader...")

    if (
        config.direction_variant != "none"
        or config.p_start != 0.0
        or config.p_max != 0.0
        or config.warmup_steps != 0
        or config.ramp_steps != 0
    ):
        raise ValueError(
            "Direction dropout is disabled for the current stabilization runs. "
            f"Got variant={config.direction_variant!r} p_start={config.p_start} p_max={config.p_max} "
            f"warmup_steps={config.warmup_steps} ramp_steps={config.ramp_steps}. "
            "Use direction_variant='none', p_start=0.0, p_max=0.0, warmup_steps=0, and ramp_steps=0."
        )
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
    _rank_zero_log("Direction dropout disabled; training uses full bidirectional encoder masks.")

    loader, sampler = _build_train_loader(config)
    eval_loader, eval_sampler = _build_eval_loader(config)
    step_eval_loader, step_eval_sampler = _build_eval_loader(
        config,
        shuffle_shards=bool(config.step_eval_shuffle),
        step_subset=True,
    )
    step_eval_every = _resolve_step_eval_every(config)
    _rank_zero_log("Dataloader ready. Entering training loop.")
    active_length_index_path = None
    active_bucket_manifest_path = None
    if config.webdataset_root is not None:
        active_bucket_manifest_path = _resolve_bucket_manifest_path(
            config.webdataset_root,
            config.webdataset_bucket_manifest_path,
            configured_length_index_path=config.webdataset_length_index_path,
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
            f"source_interleave={bool(config.bucket_source_interleave)} "
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
    start_epoch_batch_offset = 0
    history: list[dict[str, float | int | str]] = []
    step_checkpoint_history: list[dict[str, Any]] = []
    best_step_checkpoints: list[dict[str, Any]] = []
    best_epoch = 0
    best_eval_loss = float("inf")
    best_train_loss = float("inf")
    resume_from, resume_tag = _resolve_deepspeed_resume_source(config)
    if resume_from is not None:
        load_path, client_state = engine.load_checkpoint(resume_from, tag=resume_tag)
        if load_path is None:
            raise FileNotFoundError(f"Unable to load DeepSpeed checkpoint from {resume_from}")
        client_step = int(client_state.get("step", 0))
        client_epoch = int(client_state.get("epoch", 0))
        client_offset = extract_epoch_batch_offset(client_state.get("epoch_batch_offset", 0))
        fallback_state = load_latest_checkpoint_state(config.output_dir)
        fallback_step_hint = _parse_deepspeed_checkpoint_step(resume_tag)
        if not fallback_step_hint:
            fallback_step_hint = int(fallback_state.get("step", 0))
        latest_state = load_latest_checkpoint_state(config.output_dir)
        latest_step = int(latest_state.get("step", 0))
        latest_epoch = int(latest_state.get("epoch", 0))
        latest_offset = extract_epoch_batch_offset(latest_state.get("epoch_batch_offset", 0))

        if client_step == 0 and fallback_step_hint > 0:
            client_step = fallback_step_hint
            if config.resume_from == "latest" and fallback_step_hint != latest_step and latest_step > 0:
                _all_rank_log(
                    f"resume_from=latest had no step in DeepSpeed client_state; "
                    f"falling back to resolved tag step={fallback_step_hint} "
                    f"(latest yaml step={latest_step})"
                )
            else:
                _all_rank_log(
                    f"resume_from={config.resume_from} had no step in DeepSpeed client_state; "
                    f"falling back to step={fallback_step_hint}"
                )

        if client_step == 0 and config.resume_from == "latest" and latest_step > 0:
            client_step = latest_step
            _all_rank_log(
                f"resume_from=latest had no step in DeepSpeed client_state; "
                f"fallback to latest_checkpoint.yaml step={latest_step}"
            )

        export_extra_state: dict[str, Any] = {}
        if fallback_step_hint > 0:
            export_extra_state = _load_export_checkpoint_extra_state(
                output_dir=Path(config.output_dir), step=fallback_step_hint
            )
        if client_epoch == 0 and "epoch" in export_extra_state:
            candidate_epoch = int(export_extra_state.get("epoch", 0))
            if candidate_epoch > 0:
                client_epoch = candidate_epoch
                _all_rank_log(
                    f"resume_from={config.resume_from} recovered epoch={client_epoch} "
                    f"from step-{fallback_step_hint}.pt"
                )
        if client_epoch == 0 and config.resume_from == "latest" and latest_epoch > 0:
            client_epoch = latest_epoch
            _all_rank_log(
                f"resume_from=latest had no epoch in DeepSpeed client_state; "
                f"falling back to latest_checkpoint.yaml epoch={latest_epoch}"
            )

        if client_offset == 0:
            if "epoch_batch_offset" in export_extra_state:
                fallback_offset = extract_epoch_batch_offset(export_extra_state.get("epoch_batch_offset", 0))
                if fallback_offset > 0:
                    client_offset = fallback_offset
                    _all_rank_log(
                        f"resume_from={config.resume_from} recovered epoch_batch_offset={client_offset} "
                        f"from step-{fallback_step_hint}.pt"
                    )
            if client_offset == 0 and config.resume_from == "latest" and latest_offset > 0 and fallback_step_hint == latest_step:
                client_offset = latest_offset
        start_step = client_step
        start_epoch = client_epoch
        start_epoch_batch_offset = client_offset
        if start_epoch_batch_offset == 0 and config.resume_from == "latest":
            latest_state = load_latest_checkpoint_state(config.output_dir)
            latest_offset = extract_epoch_batch_offset(latest_state.get("epoch_batch_offset", 0))
            if latest_offset and fallback_step_hint == latest_step:
                start_epoch_batch_offset = latest_offset
                _all_rank_log(
                    f"resume_from=latest had no epoch_batch_offset in DeepSpeed client state; "
                    f"falling back to latest_checkpoint.yaml offset={latest_offset}"
                )
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
    epoch_batch_offset = start_epoch_batch_offset
    loss_value = float("nan")
    encoder_anchor_loss_value = 0.0
    progress = None
    task_id = None
    if start_epoch_batch_offset > 0:
        _all_rank_log(
            f"Resuming from latest state: step={start_step} epoch={start_epoch} "
            f"epoch_batch_offset={start_epoch_batch_offset}"
        )
    if _is_rank_zero():
        progress, task_id = start_training_progress(
            total_steps=resolved_max_steps,
            start_step=start_step,
            description="deepspeed-train",
        )
    if bool(config.step_eval_at_start) and start_step == 0 and step_eval_every is not None:
        initial_layer_metrics: dict[int, dict[str, float]] = {}
        initial_layer_component_metrics: dict[str, dict[int, dict[str, float]]] = {}
        initial_logit_metrics: dict[str, float] = {}
        initial_eval_loss, initial_eval_count = _evaluate_epoch_loss(
            model=engine.module,
            loader=step_eval_loader,
            sampler=step_eval_sampler,
            epoch=0,
            device=engine.device,
            feature_dtype=feature_dtype,
            mode=config.eval_mode,
            max_eval_samples=int(config.step_eval_samples),
            config=config,
            ctc_teacher_online=ctc_teacher_online,
            layer_metrics_output=initial_layer_metrics,
            layer_component_metrics_output=initial_layer_component_metrics,
            logit_metrics_output=initial_logit_metrics,
        )
        if _is_rank_zero():
            baseline_payload = {
                "step": 0,
                "eval_loss": initial_eval_loss,
                "eval_samples": initial_eval_count,
                "shuffle": bool(config.step_eval_shuffle),
                "layer_metrics": {
                    str(layer_id): metrics
                    for layer_id, metrics in initial_layer_metrics.items()
                },
                "layer_component_metrics": {
                    component: {
                        str(layer_id): metrics
                        for layer_id, metrics in component_metrics.items()
                    }
                    for component, component_metrics in initial_layer_component_metrics.items()
                },
                "logit_metrics": initial_logit_metrics,
            }
            save_yaml(output_dir / "step_eval_baseline.yaml", baseline_payload)
            _rank_zero_log(
                f"Initial step eval: step=0 eval_loss={initial_eval_loss:.4f} "
                f"eval_samples={initial_eval_count}"
            )
            log_wandb(
                wandb_run,
                {
                    "eval/step_eval_loss": initial_eval_loss,
                    "eval/step_eval_samples": initial_eval_count,
                    "eval/step_eval_baseline": 1,
                    **{
                        f"eval/layer_{layer_id}_{metric_name}": metrics[metric_name]
                        for layer_id, metrics in initial_layer_metrics.items()
                        for metric_name in ("loss", "cosine", "rms_ratio")
                    },
                    **{
                        f"eval/layer_{layer_id}_{component}_{metric_name}": metrics[metric_name]
                        for component, component_metrics in initial_layer_component_metrics.items()
                        for layer_id, metrics in component_metrics.items()
                        for metric_name in ("loss", "cosine", "rms_ratio")
                    },
                    **{
                        f"eval/ctc_alignment_{metric_name}": value
                        for metric_name, value in initial_logit_metrics.items()
                    },
                },
                step=0,
            )
    train_start_time = time.perf_counter()
    try:
        while step < resolved_max_steps:
            if epoch_batch_offset == 0:
                epoch += 1
            _set_loader_epoch(loader, sampler, epoch)
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            processed_any_batch = False
            if epoch_batch_offset > 0:
                _rank_zero_log(f"Fast-forwarding resume offset={epoch_batch_offset} for epoch={epoch}.")
                loader_iter, skipped_batches, skip_mode = _iter_loader_after_resume_offset(
                    loader,
                    epoch_batch_offset,
                    progress_callback=lambda skipped: _rank_zero_log(
                        f"Fast-forwarded resume batches {skipped}/{epoch_batch_offset} for epoch={epoch}."
                    ),
                )
                _rank_zero_log(
                    f"Resume fast-forward complete: skipped={skipped_batches}/{epoch_batch_offset} "
                    f"mode={skip_mode} epoch={epoch}"
                )
                if skipped_batches < epoch_batch_offset:
                    _all_rank_log(
                        f"Resume offset {epoch_batch_offset} exceeded available batches for epoch={epoch}; "
                        f"continuing with epoch={epoch + 1}"
                    )
                    epoch += 1
                    epoch_batch_offset = 0
                    continue
            else:
                loader_iter = iter(loader)
            while step < resolved_max_steps:
                fetch_start_time = time.perf_counter()
                try:
                    candidate_batch = next(loader_iter)
                except StopIteration:
                    break
                processed_any_batch = True
                epoch_batch_offset += 1
                data_time = time.perf_counter() - fetch_start_time
                use_padded_text_budget = bool(config.decoder_enabled and config.decoder_loss_weight > 0)
                text_tokens_per_sample_extra = (
                    decoder_text_tokens_per_sample_extra if use_padded_text_budget else 0
                )
                candidate_stats = ctc_batch_token_stats(candidate_batch)
                budgeted = select_ctc_batch_prefix_by_token_budget(
                    candidate_batch,
                    token_budget=config.batch_token_budget,
                    skip_oversized_samples=config.skip_oversized_samples,
                    use_padded_text_tokens=use_padded_text_budget,
                    text_tokens_per_sample_extra=text_tokens_per_sample_extra,
                    padded_text_token_budget=config.decoder_text_token_budget if use_padded_text_budget else None,
                )
                if budgeted is None:
                    if _is_rank_zero():
                        _rank_zero_log("Skipped a candidate batch because no sample fit inside the token budget.")
                    continue
                batch = budgeted.batch
                batch_stats = ctc_batch_token_stats(batch)
                candidate_budget_tokens = effective_batch_token_budget(
                    candidate_stats,
                    use_padded_text_tokens=use_padded_text_budget,
                    text_tokens_per_sample_extra=text_tokens_per_sample_extra,
                )
                candidate_text_budget_tokens = effective_padded_text_token_budget(
                    candidate_stats,
                    text_tokens_per_sample_extra=text_tokens_per_sample_extra,
                )
                executed_budget_tokens = effective_batch_token_budget(
                    batch_stats,
                    use_padded_text_tokens=use_padded_text_budget,
                    text_tokens_per_sample_extra=text_tokens_per_sample_extra,
                )
                executed_text_budget_tokens = effective_padded_text_token_budget(
                    batch_stats,
                    text_tokens_per_sample_extra=text_tokens_per_sample_extra,
                )
                skipped_samples = budgeted.skipped_samples
                dropped_tail_samples = budgeted.dropped_tail_samples

                step_start_time = time.perf_counter()
                mask = _sample_direction_mask_distributed(
                    scheduler,
                    step=step,
                    device=engine.device,
                )
                executed_token_stats = batch_stats
                if engine.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(engine.device)
                teacher_batch_features = batch.features
                teacher_batch_feature_lengths = batch.feature_lengths
                batch = batch.to(engine.device, feature_dtype=feature_dtype)
                if config.specaugment_enabled:
                    batch = replace(
                        batch,
                        features=apply_spec_augment(
                            batch.features,
                            batch.feature_lengths,
                            time_masks=config.specaugment_time_masks,
                            time_width=config.specaugment_time_width,
                            freq_masks=config.specaugment_freq_masks,
                            freq_width=config.specaugment_freq_width,
                        ),
                    )
                ctc_teacher_online_records: dict[str, dict[str, Any]] = {}
                ctc_teacher_online_forward_time = 0.0
                selected_layer_hidden_ids = (
                    _select_layer_hidden_ids(
                        step=step,
                        num_layers=int(config.num_layers),
                        sample_count=int(config.ctc_teacher_online_layer_sample_count),
                        boundary_ids=config.ctc_teacher_online_layer_boundary_ids,
                    )
                    if ctc_teacher_online_layer_enabled
                    else ()
                )
                teacher_layer_hidden_ids = _teacher_layer_capture_ids(
                    selected_layer_hidden_ids,
                    input_mode=str(config.ctc_teacher_online_layer_input_mode),
                )
                if ctc_teacher_online is not None:
                    if batch.utt_ids is None:
                        raise RuntimeError("Online CTC teacher distillation requires batch.utt_ids.")
                    online_teacher_start = time.perf_counter()
                    try:
                        if config.ctc_teacher_online_use_batch_features:
                            ctc_teacher_online_records = ctc_teacher_online.feature_records(
                                batch.utt_ids,
                                teacher_batch_features,
                                teacher_batch_feature_lengths,
                                audio_rows=batch.ctc_teacher_audio_rows,
                                layer_ids=teacher_layer_hidden_ids,
                                include_ctc_outputs=not ctc_teacher_online_layer_only,
                            )
                        else:
                            ctc_teacher_online_records = ctc_teacher_online.topk_records(
                                batch.utt_ids,
                                batch.ctc_teacher_audio_rows,
                                layer_ids=teacher_layer_hidden_ids,
                            )
                    except torch.OutOfMemoryError:
                        _all_rank_log(
                            "OOM during FunASR-Nano online teacher forward "
                            f"step={step + 1} batch={executed_token_stats.batch_size} "
                            f"budget={executed_budget_tokens} total={executed_token_stats.total_tokens}"
                        )
                        raise
                    ctc_teacher_online_forward_time = time.perf_counter() - online_teacher_start
                student_layer_hiddens: dict[int, dict[str, torch.Tensor]] = {}
                try:
                    if str(config.ctc_teacher_online_layer_input_mode) == "teacher_forced":
                        if not ctc_teacher_online_layer_only:
                            raise ValueError(
                                "Teacher-forced layer alignment currently requires a hidden-only objective."
                            )
                        student_layer_hiddens, encoded_lengths = (
                            _teacher_forced_student_layer_hiddens(
                                engine.module,
                                ctc_teacher_online_records,
                                batch.utt_ids,
                                layer_ids=selected_layer_hidden_ids,
                                missing_policy=config.ctc_teacher_topk_missing_policy,
                            )
                        )
                        reference = next(iter(student_layer_hiddens.values()))["mixer"]
                        zero_loss = reference.float().sum() * 0.0
                        losses = {
                            "loss": zero_loss,
                            "ctc_loss": zero_loss,
                            "decoder_loss": zero_loss,
                            "encoded_lengths": encoded_lengths,
                        }
                    else:
                        capture_context = (
                            _capture_student_sensevoice_layer_hiddens(
                                engine.module,
                                selected_layer_hidden_ids,
                            )
                            if selected_layer_hidden_ids
                            else nullcontext(student_layer_hiddens)
                        )
                        with capture_context as student_layer_hiddens:
                            if ctc_teacher_online_layer_only:
                                encoded, encoded_lengths, _ = engine.module.encoder(
                                    batch.features,
                                    batch.feature_lengths,
                                    direction_mask=mask,
                                )
                                zero_loss = encoded.float().sum() * 0.0
                                losses = {
                                    "loss": zero_loss,
                                    "ctc_loss": zero_loss,
                                    "decoder_loss": zero_loss,
                                    "encoded": encoded,
                                    "encoded_lengths": encoded_lengths,
                                }
                            else:
                                losses = engine.module.joint_losses(
                                    batch.features,
                                    batch.feature_lengths,
                                    batch.targets,
                                    batch.target_lengths,
                                    decoder_targets=batch.decoder_targets,
                                    decoder_target_lengths=batch.decoder_target_lengths,
                                    decoder_prompt_before_audio=batch.decoder_prompt_before_audio,
                                    decoder_prompt_before_audio_lengths=(
                                        batch.decoder_prompt_before_audio_lengths
                                    ),
                                    direction_mask=mask,
                                )
                except torch.OutOfMemoryError:
                    _all_rank_log(
                        "OOM during joint_losses "
                        f"step={step + 1} batch={executed_token_stats.batch_size} "
                        f"budget={executed_budget_tokens} total={executed_token_stats.total_tokens} "
                        f"text_budget={executed_text_budget_tokens} "
                        f"max_audio={executed_token_stats.max_audio_frames} "
                        f"max_text={executed_token_stats.max_text_tokens} "
                        f"padded_text={executed_token_stats.padded_text_tokens}"
                    )
                    raise
                loss = losses["loss"]
                ctc_loss_value = float(losses["ctc_loss"].detach().item())
                decoder_loss_value = float(losses["decoder_loss"].detach().item())
                encoder_anchor_loss_value = 0.0
                ctc_logit_anchor_loss_value = 0.0
                ctc_teacher_topk_loss_value = 0.0
                ctc_teacher_topk_matched = 0
                ctc_teacher_topk_missing = 0
                ctc_teacher_topk_blank_loss_value = 0.0
                ctc_teacher_topk_blank_matched = 0
                ctc_teacher_topk_blank_missing = 0
                ctc_teacher_topk_mass_loss_value = 0.0
                ctc_teacher_topk_mass_matched = 0
                ctc_teacher_topk_mass_missing = 0
                ctc_teacher_online_loss_value = 0.0
                ctc_teacher_online_matched = 0
                ctc_teacher_online_missing = 0
                ctc_teacher_online_blank_loss_value = 0.0
                ctc_teacher_online_blank_matched = 0
                ctc_teacher_online_blank_missing = 0
                ctc_teacher_online_mass_loss_value = 0.0
                ctc_teacher_online_mass_matched = 0
                ctc_teacher_online_mass_missing = 0
                ctc_teacher_online_full_loss_value = 0.0
                ctc_teacher_online_full_matched = 0
                ctc_teacher_online_full_missing = 0
                ctc_teacher_online_encoder_loss_value = 0.0
                ctc_teacher_online_encoder_matched = 0
                ctc_teacher_online_encoder_missing = 0
                ctc_teacher_online_layer_loss_value = 0.0
                ctc_teacher_online_layer_component_loss_values: dict[str, float] = {}
                ctc_teacher_online_layer_loss_values: dict[int, float] = {}
                ctc_teacher_online_layer_component_energy_values: dict[str, float] = {}
                ctc_teacher_online_layer_component_log_rms_values: dict[str, float] = {}
                ctc_teacher_online_layer_component_student_rms_values: dict[str, float] = {}
                ctc_teacher_online_layer_component_teacher_rms_values: dict[str, float] = {}
                ctc_teacher_online_layer_component_rms_ratio_values: dict[str, float] = {}
                ctc_teacher_online_layer_component_cosine_values: dict[str, float] = {}
                ctc_teacher_online_layer_energy_values: dict[int, float] = {}
                ctc_teacher_online_layer_log_rms_values: dict[int, float] = {}
                ctc_teacher_online_layer_student_rms_values: dict[int, float] = {}
                ctc_teacher_online_layer_teacher_rms_values: dict[int, float] = {}
                ctc_teacher_online_layer_rms_ratio_values: dict[int, float] = {}
                ctc_teacher_online_layer_cosine_values: dict[int, float] = {}
                ctc_teacher_online_layer_matched = 0
                ctc_teacher_online_layer_missing = 0
                ctc_teacher_online_layer_events = 0
                ctc_teacher_online_layer_max_frame_delta = 0
                ctc_teacher_online_sequence_loss_value = 0.0
                ctc_teacher_online_sequence_matched = 0
                ctc_teacher_online_sequence_missing = 0
                ctc_teacher_online_sequence_presence_loss_value = 0.0
                ctc_teacher_online_sequence_presence_matched = 0
                ctc_teacher_online_sequence_presence_missing = 0
                ctc_teacher_online_sequence_presence_tokens = 0
                ctc_teacher_online_sequence_window_loss_value = 0.0
                ctc_teacher_online_sequence_window_matched = 0
                ctc_teacher_online_sequence_window_missing = 0
                ctc_teacher_online_sequence_window_events = 0
                ctc_teacher_online_nonblank_hard_loss_value = 0.0
                ctc_teacher_online_nonblank_margin_loss_value = 0.0
                ctc_teacher_online_nonblank_matched = 0
                ctc_teacher_online_nonblank_missing = 0
                ctc_teacher_online_nonblank_window_loss_value = 0.0
                ctc_teacher_online_nonblank_window_margin_loss_value = 0.0
                ctc_teacher_online_nonblank_window_matched = 0
                ctc_teacher_online_nonblank_window_missing = 0
                ctc_teacher_online_nonblank_window_events = 0
                ctc_teacher_online_nonblank_window_topk_loss_value = 0.0
                ctc_teacher_online_nonblank_window_topk_matched = 0
                ctc_teacher_online_nonblank_window_topk_missing = 0
                ctc_teacher_online_nonblank_window_topk_events = 0
                if encoder_anchor_model is not None and (
                    encoder_anchor_weight > 0.0 or ctc_logit_anchor_weight > 0.0
                ):
                    teacher_encoded, teacher_lengths = _encoder_anchor_forward(
                        anchor_model=encoder_anchor_model,
                        features=batch.features,
                        feature_lengths=batch.feature_lengths,
                        direction_mask=mask,
                    )
                    anchor_lengths = teacher_lengths if teacher_lengths is not None else batch.feature_lengths
                    if encoder_anchor_weight > 0.0:
                        student_encoded = losses.get("encoded")
                        if not isinstance(student_encoded, torch.Tensor):
                            raise RuntimeError("joint_losses did not return encoded tensor for encoder anchoring.")
                        encoder_anchor_loss = _masked_encoder_mse_loss(
                            student_encoded,
                            teacher_encoded,
                            anchor_lengths,
                        )
                        encoder_anchor_loss_value = float(encoder_anchor_loss.detach().item())
                        loss = loss + encoder_anchor_loss * encoder_anchor_weight
                    if ctc_logit_anchor_weight > 0.0:
                        student_logits = losses.get("logits")
                        if not isinstance(student_logits, torch.Tensor):
                            raise RuntimeError("joint_losses did not return logits tensor for CTC logit anchoring.")
                        ctc_logit_anchor_loss = _masked_ctc_logit_kl_loss(
                            student_logits,
                            teacher_encoded,
                            anchor_lengths,
                            anchor_model=encoder_anchor_model,
                            chunk_frames=int(config.ctc_logit_anchor_chunk_frames),
                        )
                        ctc_logit_anchor_loss_value = float(ctc_logit_anchor_loss.detach().item())
                        loss = loss + ctc_logit_anchor_loss * ctc_logit_anchor_weight
                if ctc_teacher_topk_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError("joint_losses did not return logits tensor for CTC teacher distillation.")
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    ctc_teacher_topk_loss, ctc_teacher_topk_matched, ctc_teacher_topk_missing = (
                        _ctc_teacher_topk_loss(
                            student_logits,
                            student_lengths,
                            batch.utt_ids,
                            ctc_teacher_topk_cache,
                            blank_id=int(config.blank_id),
                            time_map=config.ctc_teacher_topk_time_map,
                            frame_filter=ctc_teacher_frame_filter,
                            frame_filter_neighbor_radius=ctc_teacher_frame_filter_neighbor_radius,
                            frame_filter_min_nonblank_prob=ctc_teacher_frame_filter_min_nonblank_prob,
                            missing_policy=config.ctc_teacher_topk_missing_policy,
                        )
                    )
                    ctc_teacher_topk_loss_value = float(ctc_teacher_topk_loss.detach().item())
                    loss = loss + ctc_teacher_topk_loss * ctc_teacher_topk_weight
                if ctc_teacher_topk_blank_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError("joint_losses did not return logits tensor for CTC teacher blank distillation.")
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_topk_blank_loss,
                        ctc_teacher_topk_blank_matched,
                        ctc_teacher_topk_blank_missing,
                    ) = _ctc_teacher_blank_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_topk_cache,
                        blank_id=int(config.blank_id),
                        time_map=config.ctc_teacher_topk_time_map,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                    )
                    ctc_teacher_topk_blank_loss_value = float(ctc_teacher_topk_blank_loss.detach().item())
                    loss = loss + ctc_teacher_topk_blank_loss * ctc_teacher_topk_blank_weight
                if ctc_teacher_topk_mass_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError("joint_losses did not return logits tensor for CTC teacher mass distillation.")
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_topk_mass_loss,
                        ctc_teacher_topk_mass_matched,
                        ctc_teacher_topk_mass_missing,
                    ) = _ctc_teacher_mass_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_topk_cache,
                        blank_id=int(config.blank_id),
                        time_map=config.ctc_teacher_topk_time_map,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                    )
                    ctc_teacher_topk_mass_loss_value = float(ctc_teacher_topk_mass_loss.detach().item())
                    loss = loss + ctc_teacher_topk_mass_loss * ctc_teacher_topk_mass_weight
                if ctc_teacher_online_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError("joint_losses did not return logits tensor for online CTC teacher distillation.")
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    ctc_teacher_online_loss, ctc_teacher_online_matched, ctc_teacher_online_missing = (
                        _ctc_teacher_topk_loss(
                            student_logits,
                            student_lengths,
                            batch.utt_ids,
                            ctc_teacher_online_records,
                            blank_id=int(config.blank_id),
                            time_map=config.ctc_teacher_topk_time_map,
                            frame_filter=ctc_teacher_frame_filter,
                            frame_filter_neighbor_radius=ctc_teacher_frame_filter_neighbor_radius,
                            frame_filter_min_nonblank_prob=ctc_teacher_frame_filter_min_nonblank_prob,
                            missing_policy=config.ctc_teacher_topk_missing_policy,
                        )
                    )
                    ctc_teacher_online_loss_value = float(ctc_teacher_online_loss.detach().item())
                    loss = loss + ctc_teacher_online_loss * ctc_teacher_online_weight
                if ctc_teacher_online_blank_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError("joint_losses did not return logits tensor for online CTC teacher blank distillation.")
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_blank_loss,
                        ctc_teacher_online_blank_matched,
                        ctc_teacher_online_blank_missing,
                    ) = _ctc_teacher_blank_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        time_map=config.ctc_teacher_topk_time_map,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                    )
                    ctc_teacher_online_blank_loss_value = float(ctc_teacher_online_blank_loss.detach().item())
                    loss = loss + ctc_teacher_online_blank_loss * ctc_teacher_online_blank_weight
                if ctc_teacher_online_mass_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError("joint_losses did not return logits tensor for online CTC teacher mass distillation.")
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_mass_loss,
                        ctc_teacher_online_mass_matched,
                        ctc_teacher_online_mass_missing,
                    ) = _ctc_teacher_mass_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        time_map=config.ctc_teacher_topk_time_map,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                    )
                    ctc_teacher_online_mass_loss_value = float(ctc_teacher_online_mass_loss.detach().item())
                    loss = loss + ctc_teacher_online_mass_loss * ctc_teacher_online_mass_weight
                if ctc_teacher_online_full_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError("joint_losses did not return logits tensor for online CTC teacher full distillation.")
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_full_loss,
                        ctc_teacher_online_full_matched,
                        ctc_teacher_online_full_missing,
                    ) = _ctc_teacher_full_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        time_map=config.ctc_teacher_topk_time_map,
                        frame_filter=ctc_teacher_online_full_frame_filter,
                        frame_filter_neighbor_radius=ctc_teacher_frame_filter_neighbor_radius,
                        frame_filter_min_nonblank_prob=ctc_teacher_frame_filter_min_nonblank_prob,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                        temperature=ctc_teacher_online_full_temperature,
                        nonblank_frame_weight=ctc_teacher_online_full_nonblank_weight,
                    )
                    ctc_teacher_online_full_loss_value = float(ctc_teacher_online_full_loss.detach().item())
                    loss = loss + ctc_teacher_online_full_loss * ctc_teacher_online_full_weight
                if ctc_teacher_online_encoder_weight > 0.0:
                    student_encoded = losses.get("ctc_encoded", losses.get("encoded"))
                    if not isinstance(student_encoded, torch.Tensor):
                        raise RuntimeError(
                            "joint_losses did not return encoded tensor for online CTC teacher encoder distillation."
                        )
                    student_encoded_lengths = losses.get("ctc_encoded_lengths", losses.get("encoded_lengths"))
                    if student_encoded_lengths is not None and not isinstance(student_encoded_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor encoded_lengths.")
                    (
                        ctc_teacher_online_encoder_loss,
                        ctc_teacher_online_encoder_matched,
                        ctc_teacher_online_encoder_missing,
                    ) = _ctc_teacher_hidden_loss(
                        student_encoded,
                        student_encoded_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        teacher_field="encoder_out",
                        time_map=config.ctc_teacher_topk_time_map,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                        blank_id=int(config.blank_id),
                        frame_filter=ctc_teacher_frame_filter,
                        frame_filter_neighbor_radius=ctc_teacher_frame_filter_neighbor_radius,
                        frame_filter_min_nonblank_prob=ctc_teacher_frame_filter_min_nonblank_prob,
                    )
                    ctc_teacher_online_encoder_loss_value = float(
                        ctc_teacher_online_encoder_loss.detach().item()
                    )
                    loss = loss + ctc_teacher_online_encoder_loss * ctc_teacher_online_encoder_weight
                if ctc_teacher_online_layer_enabled:
                    student_encoded_lengths = losses.get("encoded_lengths")
                    if student_encoded_lengths is not None and not isinstance(
                        student_encoded_lengths,
                        torch.Tensor,
                    ):
                        raise RuntimeError("joint_losses returned non-tensor encoded_lengths.")
                    layer_hidden_result = _ctc_teacher_layer_hidden_loss(
                        student_layer_hiddens,
                        student_encoded_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        layer_ids=selected_layer_hidden_ids,
                        component_weights=ctc_teacher_online_layer_component_weights,
                        normalized_mse_weight=float(
                            config.ctc_teacher_online_layer_normalized_mse_weight
                        ),
                        cosine_weight=float(config.ctc_teacher_online_layer_cosine_weight),
                        energy_mse_weight=float(config.ctc_teacher_online_layer_energy_mse_weight),
                        log_rms_weight=float(config.ctc_teacher_online_layer_log_rms_weight),
                        raw_mse_weight=float(config.ctc_teacher_online_layer_raw_mse_weight),
                        frame_tolerance=int(config.ctc_teacher_online_layer_frame_tolerance),
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                    )
                    ctc_teacher_online_layer_loss_value = float(
                        layer_hidden_result.loss.detach().item()
                    )
                    ctc_teacher_online_layer_component_loss_values = {
                        name: float(value.detach().item())
                        for name, value in layer_hidden_result.component_losses.items()
                    }
                    ctc_teacher_online_layer_loss_values = {
                        layer_id: float(value.detach().item())
                        for layer_id, value in layer_hidden_result.layer_losses.items()
                    }
                    ctc_teacher_online_layer_component_energy_values = (
                        layer_hidden_result.component_energy_mse
                    )
                    ctc_teacher_online_layer_component_log_rms_values = (
                        layer_hidden_result.component_log_rms
                    )
                    ctc_teacher_online_layer_component_student_rms_values = (
                        layer_hidden_result.component_student_rms
                    )
                    ctc_teacher_online_layer_component_teacher_rms_values = (
                        layer_hidden_result.component_teacher_rms
                    )
                    ctc_teacher_online_layer_component_rms_ratio_values = (
                        layer_hidden_result.component_rms_ratio
                    )
                    ctc_teacher_online_layer_component_cosine_values = (
                        layer_hidden_result.component_cosine
                    )
                    ctc_teacher_online_layer_energy_values = layer_hidden_result.layer_energy_mse
                    ctc_teacher_online_layer_log_rms_values = layer_hidden_result.layer_log_rms
                    ctc_teacher_online_layer_student_rms_values = layer_hidden_result.layer_student_rms
                    ctc_teacher_online_layer_teacher_rms_values = layer_hidden_result.layer_teacher_rms
                    ctc_teacher_online_layer_rms_ratio_values = layer_hidden_result.layer_rms_ratio
                    ctc_teacher_online_layer_cosine_values = layer_hidden_result.layer_cosine
                    ctc_teacher_online_layer_matched = layer_hidden_result.matched_samples
                    ctc_teacher_online_layer_missing = layer_hidden_result.missing_samples
                    ctc_teacher_online_layer_events = layer_hidden_result.events
                    ctc_teacher_online_layer_max_frame_delta = layer_hidden_result.max_frame_delta
                    for layer_id in selected_layer_hidden_ids:
                        ctc_teacher_online_layer_coverage[layer_id] += int(batch_stats.batch_size)
                    loss = loss + layer_hidden_result.loss
                if ctc_teacher_online_sequence_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError(
                            "joint_losses did not return logits tensor for online CTC teacher sequence distillation."
                        )
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_sequence_loss,
                        ctc_teacher_online_sequence_matched,
                        ctc_teacher_online_sequence_missing,
                    ) = _ctc_teacher_sequence_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        ignored_token_ids=tuple(
                            int(value) for value in config.ctc_teacher_online_project_ignored_token_ids
                        ),
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                    )
                    ctc_teacher_online_sequence_loss_value = float(
                        ctc_teacher_online_sequence_loss.detach().item()
                    )
                    loss = loss + ctc_teacher_online_sequence_loss * ctc_teacher_online_sequence_weight
                if ctc_teacher_online_sequence_window_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError(
                            "joint_losses did not return logits tensor for online CTC teacher sequence-window distillation."
                        )
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_sequence_window_loss,
                        ctc_teacher_online_sequence_window_matched,
                        ctc_teacher_online_sequence_window_missing,
                        ctc_teacher_online_sequence_window_events,
                    ) = _ctc_teacher_sequence_window_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        ignored_token_ids=tuple(
                            int(value) for value in config.ctc_teacher_online_project_ignored_token_ids
                        ),
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                        radius=ctc_teacher_online_sequence_window_radius,
                        temperature=ctc_teacher_online_sequence_window_temperature,
                    )
                    ctc_teacher_online_sequence_window_loss_value = float(
                        ctc_teacher_online_sequence_window_loss.detach().item()
                    )
                    loss = (
                        loss
                        + ctc_teacher_online_sequence_window_loss
                        * ctc_teacher_online_sequence_window_weight
                    )
                if ctc_teacher_online_sequence_presence_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError(
                            "joint_losses did not return logits tensor for online CTC teacher sequence presence distillation."
                        )
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_sequence_presence_loss,
                        ctc_teacher_online_sequence_presence_matched,
                        ctc_teacher_online_sequence_presence_missing,
                        ctc_teacher_online_sequence_presence_tokens,
                    ) = _ctc_teacher_sequence_presence_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        ignored_token_ids=tuple(
                            int(value) for value in config.ctc_teacher_online_project_ignored_token_ids
                        ),
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                    )
                    ctc_teacher_online_sequence_presence_loss_value = float(
                        ctc_teacher_online_sequence_presence_loss.detach().item()
                    )
                    loss = loss + (
                        ctc_teacher_online_sequence_presence_loss
                        * ctc_teacher_online_sequence_presence_weight
                    )
                if (
                    ctc_teacher_online_nonblank_hard_weight > 0.0
                    or ctc_teacher_online_nonblank_margin_weight > 0.0
                ):
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError(
                            "joint_losses did not return logits tensor for online CTC teacher nonblank hard distillation."
                        )
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_nonblank_hard_loss,
                        ctc_teacher_online_nonblank_margin_loss,
                        ctc_teacher_online_nonblank_matched,
                        ctc_teacher_online_nonblank_missing,
                    ) = _ctc_teacher_nonblank_hard_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        time_map=config.ctc_teacher_topk_time_map,
                        frame_filter_min_nonblank_prob=ctc_teacher_frame_filter_min_nonblank_prob,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                        margin=ctc_teacher_online_nonblank_margin,
                    )
                    ctc_teacher_online_nonblank_hard_loss_value = float(
                        ctc_teacher_online_nonblank_hard_loss.detach().item()
                    )
                    ctc_teacher_online_nonblank_margin_loss_value = float(
                        ctc_teacher_online_nonblank_margin_loss.detach().item()
                    )
                    if ctc_teacher_online_nonblank_hard_weight > 0.0:
                        loss = loss + (
                            ctc_teacher_online_nonblank_hard_loss
                            * ctc_teacher_online_nonblank_hard_weight
                        )
                    if ctc_teacher_online_nonblank_margin_weight > 0.0:
                        loss = loss + (
                            ctc_teacher_online_nonblank_margin_loss
                            * ctc_teacher_online_nonblank_margin_weight
                        )
                if (
                    ctc_teacher_online_nonblank_window_weight > 0.0
                    or ctc_teacher_online_nonblank_window_margin_weight > 0.0
                ):
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError(
                            "joint_losses did not return logits tensor for online CTC teacher nonblank window distillation."
                        )
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_nonblank_window_loss,
                        ctc_teacher_online_nonblank_window_margin_loss,
                        ctc_teacher_online_nonblank_window_matched,
                        ctc_teacher_online_nonblank_window_missing,
                        ctc_teacher_online_nonblank_window_events,
                    ) = _ctc_teacher_nonblank_window_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        time_map=config.ctc_teacher_topk_time_map,
                        frame_filter_min_nonblank_prob=ctc_teacher_frame_filter_min_nonblank_prob,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                        margin=ctc_teacher_online_nonblank_margin,
                        window_radius=ctc_teacher_online_nonblank_window_radius,
                        temperature=ctc_teacher_online_nonblank_window_temperature,
                    )
                    ctc_teacher_online_nonblank_window_loss_value = float(
                        ctc_teacher_online_nonblank_window_loss.detach().item()
                    )
                    ctc_teacher_online_nonblank_window_margin_loss_value = float(
                        ctc_teacher_online_nonblank_window_margin_loss.detach().item()
                    )
                    if ctc_teacher_online_nonblank_window_weight > 0.0:
                        loss = loss + (
                            ctc_teacher_online_nonblank_window_loss
                            * ctc_teacher_online_nonblank_window_weight
                        )
                    if ctc_teacher_online_nonblank_window_margin_weight > 0.0:
                        loss = loss + (
                            ctc_teacher_online_nonblank_window_margin_loss
                            * ctc_teacher_online_nonblank_window_margin_weight
                        )
                if ctc_teacher_online_nonblank_window_topk_weight > 0.0:
                    student_logits = losses.get("logits")
                    if not isinstance(student_logits, torch.Tensor):
                        raise RuntimeError(
                            "joint_losses did not return logits tensor for online CTC teacher nonblank window top-k distillation."
                        )
                    student_lengths = losses.get("logit_lengths")
                    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
                        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")
                    (
                        ctc_teacher_online_nonblank_window_topk_loss,
                        ctc_teacher_online_nonblank_window_topk_matched,
                        ctc_teacher_online_nonblank_window_topk_missing,
                        ctc_teacher_online_nonblank_window_topk_events,
                    ) = _ctc_teacher_nonblank_window_topk_loss(
                        student_logits,
                        student_lengths,
                        batch.utt_ids,
                        ctc_teacher_online_records,
                        blank_id=int(config.blank_id),
                        time_map=config.ctc_teacher_topk_time_map,
                        frame_filter_min_nonblank_prob=ctc_teacher_frame_filter_min_nonblank_prob,
                        missing_policy=config.ctc_teacher_topk_missing_policy,
                        window_radius=ctc_teacher_online_nonblank_window_radius,
                        temperature=ctc_teacher_online_nonblank_window_temperature,
                    )
                    ctc_teacher_online_nonblank_window_topk_loss_value = float(
                        ctc_teacher_online_nonblank_window_topk_loss.detach().item()
                    )
                    loss = loss + (
                        ctc_teacher_online_nonblank_window_topk_loss
                        * ctc_teacher_online_nonblank_window_topk_weight
                    )
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
                    memory_per_token = peak_reserved_bytes / max(executed_budget_tokens, 1)
                    ratio_tensor = torch.tensor(memory_per_token, device=engine.device, dtype=torch.float64)
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(ratio_tensor, op=dist.ReduceOp.MAX)
                    global_peak_reserved_bytes = int(ratio_tensor.item() * max(executed_budget_tokens, 1))
                    estimated_budget = estimate_token_budget_from_memory(
                        observed_tokens=executed_budget_tokens,
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
                        f"First step timings: data={data_time:.2f}s compute={step_time:.2f}s "
                        f"online_teacher={ctc_teacher_online_forward_time:.2f}s "
                        f"total={data_time + step_time:.2f}s"
                    )
                    _rank_zero_log(
                        "Batch stats "
                        f"step={step} candidate_batch={candidate_stats.batch_size} "
                        f"candidate_total={candidate_stats.total_tokens} candidate_budget={candidate_budget_tokens} "
                        f"candidate_text_budget={candidate_text_budget_tokens} "
                        f"executed_total={executed_token_stats.total_tokens} executed_budget={executed_budget_tokens} "
                        f"executed_text_budget={executed_text_budget_tokens} "
                        f"max_audio={executed_token_stats.max_audio_frames} padded_audio={executed_token_stats.padded_audio_tokens} "
                        f"max_text={executed_token_stats.max_text_tokens} text={executed_token_stats.text_tokens} "
                        f"padded_text={executed_token_stats.padded_text_tokens} "
                        f"peak_reserved={peak_reserved_bytes / (1024**3):.2f}GiB "
                        f"peak_allocated={peak_allocated_bytes / (1024**3):.2f}GiB "
                        f"estimated_budget@{config.target_gpu_memory_gib:.1f}GiB={estimated_budget}"
                    )
                    if config.batch_token_budget is not None:
                        _rank_zero_log(
                            f"Token budget active: budget={config.batch_token_budget} "
                            f"text_budget={config.decoder_text_token_budget} "
                            f"effective_batch={batch_stats.batch_size} dropped_tail={dropped_tail_samples} "
                            f"skipped_samples={skipped_samples}"
                        )
                if _is_rank_zero() and (step <= 10 or step % config.log_every == 0 or step == resolved_max_steps):
                    print(
                        "[deepspeed-train] "
                        f"step={step} loss={loss_value:.4f} ctc={ctc_loss_value:.4f} "
                        f"decoder={decoder_loss_value:.4f} anchor={encoder_anchor_loss_value:.4f} "
                        f"ctc_anchor={ctc_logit_anchor_loss_value:.4f} "
                        f"ctc_teacher_topk={ctc_teacher_topk_loss_value:.4f} "
                        f"teacher_match={ctc_teacher_topk_matched}/{batch_stats.batch_size} "
                        f"teacher_missing={ctc_teacher_topk_missing} "
                        f"ctc_teacher_blank={ctc_teacher_topk_blank_loss_value:.4f} "
                        f"teacher_blank_match={ctc_teacher_topk_blank_matched}/{batch_stats.batch_size} "
                        f"teacher_blank_missing={ctc_teacher_topk_blank_missing} "
                        f"ctc_teacher_mass={ctc_teacher_topk_mass_loss_value:.4f} "
                        f"teacher_mass_match={ctc_teacher_topk_mass_matched}/{batch_stats.batch_size} "
                        f"teacher_mass_missing={ctc_teacher_topk_mass_missing} "
                        f"online_ctc_teacher={ctc_teacher_online_loss_value:.4f} "
                        f"online_teacher_match={ctc_teacher_online_matched}/{batch_stats.batch_size} "
                        f"online_teacher_missing={ctc_teacher_online_missing} "
                        f"online_ctc_blank={ctc_teacher_online_blank_loss_value:.4f} "
                        f"online_blank_match={ctc_teacher_online_blank_matched}/{batch_stats.batch_size} "
                        f"online_blank_missing={ctc_teacher_online_blank_missing} "
                        f"online_ctc_mass={ctc_teacher_online_mass_loss_value:.4f} "
                        f"online_mass_match={ctc_teacher_online_mass_matched}/{batch_stats.batch_size} "
                        f"online_mass_missing={ctc_teacher_online_mass_missing} "
                        f"online_ctc_full={ctc_teacher_online_full_loss_value:.4f} "
                        f"online_full_match={ctc_teacher_online_full_matched}/{batch_stats.batch_size} "
                        f"online_full_missing={ctc_teacher_online_full_missing} "
                        f"online_ctc_encoder={ctc_teacher_online_encoder_loss_value:.4f} "
                        f"online_encoder_match={ctc_teacher_online_encoder_matched}/{batch_stats.batch_size} "
                        f"online_encoder_missing={ctc_teacher_online_encoder_missing} "
                        f"online_layer_hidden={ctc_teacher_online_layer_loss_value:.4f} "
                        f"online_layer_mixer={ctc_teacher_online_layer_component_loss_values.get('mixer', 0.0):.4f} "
                        f"online_layer_ffn={ctc_teacher_online_layer_component_loss_values.get('ffn', 0.0):.4f} "
                        f"online_layer_block={ctc_teacher_online_layer_component_loss_values.get('block', 0.0):.4f} "
                        f"online_layer_mixer_energy={ctc_teacher_online_layer_component_energy_values.get('mixer', 0.0):.4f} "
                        f"online_layer_mixer_log_rms={ctc_teacher_online_layer_component_log_rms_values.get('mixer', 0.0):.4f} "
                        f"online_layer_mixer_student_rms={ctc_teacher_online_layer_component_student_rms_values.get('mixer', 0.0):.4f} "
                        f"online_layer_mixer_teacher_rms={ctc_teacher_online_layer_component_teacher_rms_values.get('mixer', 0.0):.4f} "
                        f"online_layer_mixer_rms_ratio={ctc_teacher_online_layer_component_rms_ratio_values.get('mixer', 0.0):.4f} "
                        f"online_layer_mixer_cosine={ctc_teacher_online_layer_component_cosine_values.get('mixer', 0.0):.4f} "
                        f"online_layer_match={ctc_teacher_online_layer_matched}/{batch_stats.batch_size} "
                        f"online_layer_missing={ctc_teacher_online_layer_missing} "
                        f"online_layer_events={ctc_teacher_online_layer_events} "
                        f"online_layer_frame_delta={ctc_teacher_online_layer_max_frame_delta} "
                        f"online_layer_ids={','.join(str(value) for value in selected_layer_hidden_ids) or '-'} "
                        f"online_ctc_sequence={ctc_teacher_online_sequence_loss_value:.4f} "
                        f"online_sequence_match={ctc_teacher_online_sequence_matched}/{batch_stats.batch_size} "
                        f"online_sequence_missing={ctc_teacher_online_sequence_missing} "
                        f"online_ctc_sequence_presence={ctc_teacher_online_sequence_presence_loss_value:.4f} "
                        f"online_sequence_presence_match={ctc_teacher_online_sequence_presence_matched}/{batch_stats.batch_size} "
                        f"online_sequence_presence_missing={ctc_teacher_online_sequence_presence_missing} "
                        f"online_sequence_presence_tokens={ctc_teacher_online_sequence_presence_tokens} "
                        f"online_ctc_sequence_window={ctc_teacher_online_sequence_window_loss_value:.4f} "
                        f"online_sequence_window_match={ctc_teacher_online_sequence_window_matched}/{batch_stats.batch_size} "
                        f"online_sequence_window_missing={ctc_teacher_online_sequence_window_missing} "
                        f"online_sequence_window_events={ctc_teacher_online_sequence_window_events} "
                        f"online_ctc_nonblank_hard={ctc_teacher_online_nonblank_hard_loss_value:.4f} "
                        f"online_ctc_nonblank_margin={ctc_teacher_online_nonblank_margin_loss_value:.4f} "
                        f"online_nonblank_match={ctc_teacher_online_nonblank_matched}/{batch_stats.batch_size} "
                        f"online_nonblank_missing={ctc_teacher_online_nonblank_missing} "
                        f"online_ctc_nonblank_window={ctc_teacher_online_nonblank_window_loss_value:.4f} "
                        f"online_ctc_nonblank_window_margin={ctc_teacher_online_nonblank_window_margin_loss_value:.4f} "
                        f"online_nonblank_window_match={ctc_teacher_online_nonblank_window_matched}/{batch_stats.batch_size} "
                        f"online_nonblank_window_missing={ctc_teacher_online_nonblank_window_missing} "
                        f"online_nonblank_window_events={ctc_teacher_online_nonblank_window_events} "
                        f"online_ctc_nonblank_window_topk={ctc_teacher_online_nonblank_window_topk_loss_value:.4f} "
                        f"online_nonblank_window_topk_match={ctc_teacher_online_nonblank_window_topk_matched}/{batch_stats.batch_size} "
                        f"online_nonblank_window_topk_missing={ctc_teacher_online_nonblank_window_topk_missing} "
                        f"online_nonblank_window_topk_events={ctc_teacher_online_nonblank_window_topk_events} "
                        f"online_teacher_time={ctc_teacher_online_forward_time:.2f}s "
                        f"device={device}",
                        flush=True,
                    )
                    total_elapsed = time.perf_counter() - train_start_time
                    elapsed_steps = max(step - start_step, 1)
                    rate = elapsed_steps / max(total_elapsed, 1.0e-6)
                    eta_hours = max(resolved_max_steps - step, 0) / max(rate, 1.0e-6) / 3600.0
                    log_wandb(
                        wandb_run,
                        {
                            "train/loss": loss_value,
                            "train/ctc_loss": ctc_loss_value,
                            "train/decoder_loss": decoder_loss_value,
                            "train/encoder_anchor_loss": encoder_anchor_loss_value,
                            "train/ctc_logit_anchor_loss": ctc_logit_anchor_loss_value,
                            "train/ctc_teacher_topk_loss": ctc_teacher_topk_loss_value,
                            "train/ctc_teacher_topk_matched": ctc_teacher_topk_matched,
                            "train/ctc_teacher_topk_missing": ctc_teacher_topk_missing,
                            "train/ctc_teacher_topk_blank_loss": ctc_teacher_topk_blank_loss_value,
                            "train/ctc_teacher_topk_blank_matched": ctc_teacher_topk_blank_matched,
                            "train/ctc_teacher_topk_blank_missing": ctc_teacher_topk_blank_missing,
                            "train/ctc_teacher_topk_mass_loss": ctc_teacher_topk_mass_loss_value,
                            "train/ctc_teacher_topk_mass_matched": ctc_teacher_topk_mass_matched,
                            "train/ctc_teacher_topk_mass_missing": ctc_teacher_topk_mass_missing,
                            "train/ctc_teacher_online_loss": ctc_teacher_online_loss_value,
                            "train/ctc_teacher_online_matched": ctc_teacher_online_matched,
                            "train/ctc_teacher_online_missing": ctc_teacher_online_missing,
                            "train/ctc_teacher_online_blank_loss": ctc_teacher_online_blank_loss_value,
                            "train/ctc_teacher_online_blank_matched": ctc_teacher_online_blank_matched,
                            "train/ctc_teacher_online_blank_missing": ctc_teacher_online_blank_missing,
                            "train/ctc_teacher_online_mass_loss": ctc_teacher_online_mass_loss_value,
                            "train/ctc_teacher_online_mass_matched": ctc_teacher_online_mass_matched,
                            "train/ctc_teacher_online_mass_missing": ctc_teacher_online_mass_missing,
                            "train/ctc_teacher_online_full_loss": ctc_teacher_online_full_loss_value,
                            "train/ctc_teacher_online_full_matched": ctc_teacher_online_full_matched,
                            "train/ctc_teacher_online_full_missing": ctc_teacher_online_full_missing,
                            "train/ctc_teacher_online_encoder_loss": ctc_teacher_online_encoder_loss_value,
                            "train/ctc_teacher_online_encoder_matched": ctc_teacher_online_encoder_matched,
                            "train/ctc_teacher_online_encoder_missing": ctc_teacher_online_encoder_missing,
                            "train/ctc_teacher_online_layer_loss": ctc_teacher_online_layer_loss_value,
                            "train/ctc_teacher_online_layer_mixer_loss": (
                                ctc_teacher_online_layer_component_loss_values.get("mixer", 0.0)
                            ),
                            "train/ctc_teacher_online_layer_ffn_loss": (
                                ctc_teacher_online_layer_component_loss_values.get("ffn", 0.0)
                            ),
                            "train/ctc_teacher_online_layer_block_loss": (
                                ctc_teacher_online_layer_component_loss_values.get("block", 0.0)
                            ),
                            **{
                                f"train/ctc_teacher_online_layer_{component}_{metric}": values.get(
                                    component,
                                    0.0,
                                )
                                for component in ctc_teacher_online_layer_component_loss_values
                                for metric, values in (
                                    ("energy_mse", ctc_teacher_online_layer_component_energy_values),
                                    ("log_rms", ctc_teacher_online_layer_component_log_rms_values),
                                    ("student_rms", ctc_teacher_online_layer_component_student_rms_values),
                                    ("teacher_rms", ctc_teacher_online_layer_component_teacher_rms_values),
                                    ("rms_ratio", ctc_teacher_online_layer_component_rms_ratio_values),
                                    ("cosine", ctc_teacher_online_layer_component_cosine_values),
                                )
                            },
                            "train/ctc_teacher_online_layer_matched": ctc_teacher_online_layer_matched,
                            "train/ctc_teacher_online_layer_missing": ctc_teacher_online_layer_missing,
                            "train/ctc_teacher_online_layer_events": ctc_teacher_online_layer_events,
                            "train/ctc_teacher_online_layer_max_frame_delta": (
                                ctc_teacher_online_layer_max_frame_delta
                            ),
                            "train/ctc_teacher_online_layer_coverage_min": min(
                                ctc_teacher_online_layer_coverage.values(),
                                default=0,
                            ),
                            "train/ctc_teacher_online_layer_coverage_max": max(
                                ctc_teacher_online_layer_coverage.values(),
                                default=0,
                            ),
                            **{
                                f"train/ctc_teacher_online_layer_{layer_id}_loss": value
                                for layer_id, value in ctc_teacher_online_layer_loss_values.items()
                            },
                            **{
                                f"train/ctc_teacher_online_layer_{layer_id}_{metric}": values[layer_id]
                                for metric, values in (
                                    ("energy_mse", ctc_teacher_online_layer_energy_values),
                                    ("log_rms", ctc_teacher_online_layer_log_rms_values),
                                    ("student_rms", ctc_teacher_online_layer_student_rms_values),
                                    ("teacher_rms", ctc_teacher_online_layer_teacher_rms_values),
                                    ("rms_ratio", ctc_teacher_online_layer_rms_ratio_values),
                                    ("cosine", ctc_teacher_online_layer_cosine_values),
                                )
                                for layer_id in values
                            },
                            "train/ctc_teacher_online_sequence_loss": ctc_teacher_online_sequence_loss_value,
                            "train/ctc_teacher_online_sequence_matched": ctc_teacher_online_sequence_matched,
                            "train/ctc_teacher_online_sequence_missing": ctc_teacher_online_sequence_missing,
                            "train/ctc_teacher_online_sequence_presence_loss": (
                                ctc_teacher_online_sequence_presence_loss_value
                            ),
                            "train/ctc_teacher_online_sequence_presence_matched": (
                                ctc_teacher_online_sequence_presence_matched
                            ),
                            "train/ctc_teacher_online_sequence_presence_missing": (
                                ctc_teacher_online_sequence_presence_missing
                            ),
                            "train/ctc_teacher_online_sequence_presence_tokens": (
                                ctc_teacher_online_sequence_presence_tokens
                            ),
                            "train/ctc_teacher_online_sequence_window_loss": (
                                ctc_teacher_online_sequence_window_loss_value
                            ),
                            "train/ctc_teacher_online_sequence_window_matched": (
                                ctc_teacher_online_sequence_window_matched
                            ),
                            "train/ctc_teacher_online_sequence_window_missing": (
                                ctc_teacher_online_sequence_window_missing
                            ),
                            "train/ctc_teacher_online_sequence_window_events": (
                                ctc_teacher_online_sequence_window_events
                            ),
                            "train/ctc_teacher_online_nonblank_hard_loss": (
                                ctc_teacher_online_nonblank_hard_loss_value
                            ),
                            "train/ctc_teacher_online_nonblank_margin_loss": (
                                ctc_teacher_online_nonblank_margin_loss_value
                            ),
                            "train/ctc_teacher_online_nonblank_matched": ctc_teacher_online_nonblank_matched,
                            "train/ctc_teacher_online_nonblank_missing": ctc_teacher_online_nonblank_missing,
                            "train/ctc_teacher_online_nonblank_window_loss": (
                                ctc_teacher_online_nonblank_window_loss_value
                            ),
                            "train/ctc_teacher_online_nonblank_window_margin_loss": (
                                ctc_teacher_online_nonblank_window_margin_loss_value
                            ),
                            "train/ctc_teacher_online_nonblank_window_matched": (
                                ctc_teacher_online_nonblank_window_matched
                            ),
                            "train/ctc_teacher_online_nonblank_window_missing": (
                                ctc_teacher_online_nonblank_window_missing
                            ),
                            "train/ctc_teacher_online_nonblank_window_events": (
                                ctc_teacher_online_nonblank_window_events
                            ),
                            "train/ctc_teacher_online_nonblank_window_topk_loss": (
                                ctc_teacher_online_nonblank_window_topk_loss_value
                            ),
                            "train/ctc_teacher_online_nonblank_window_topk_matched": (
                                ctc_teacher_online_nonblank_window_topk_matched
                            ),
                            "train/ctc_teacher_online_nonblank_window_topk_missing": (
                                ctc_teacher_online_nonblank_window_topk_missing
                            ),
                            "train/ctc_teacher_online_nonblank_window_topk_events": (
                                ctc_teacher_online_nonblank_window_topk_events
                            ),
                            "train/ctc_teacher_online_forward_time": ctc_teacher_online_forward_time,
                            "train/epoch": epoch,
                            "train/data_time": data_time,
                            "train/step_time": step_time,
                            "train/rate": rate,
                            "train/progress_frac": step / max(resolved_max_steps, 1),
                            "train/eta_hours": eta_hours,
                            "train/effective_batch": batch_stats.batch_size,
                            "train/total_tokens": executed_token_stats.total_tokens,
                            "train/budget_tokens": executed_budget_tokens,
                            "train/text_budget_tokens": executed_text_budget_tokens,
                            "train/max_audio_frames": executed_token_stats.max_audio_frames,
                            "train/padded_audio_tokens": executed_token_stats.padded_audio_tokens,
                            "train/max_text_tokens": executed_token_stats.max_text_tokens,
                            "train/text_tokens": executed_token_stats.text_tokens,
                            "train/padded_text_tokens": executed_token_stats.padded_text_tokens,
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
                        "epoch_batch_offset": epoch_batch_offset,
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
                        save_deepspeed_sharded=bool(config.save_deepspeed_sharded_checkpoints),
                    )
                    if step_eval_every is not None and step % step_eval_every == 0:
                        step_layer_metrics: dict[int, dict[str, float]] = {}
                        step_layer_component_metrics: dict[str, dict[int, dict[str, float]]] = {}
                        step_logit_metrics: dict[str, float] = {}
                        step_eval_loss, step_eval_count = _evaluate_epoch_loss(
                            model=engine.module,
                            loader=step_eval_loader,
                            sampler=step_eval_sampler,
                            epoch=step if bool(config.step_eval_shuffle) else 0,
                            device=engine.device,
                            feature_dtype=feature_dtype,
                            mode=config.eval_mode,
                            max_eval_samples=int(config.step_eval_samples),
                            config=config,
                            ctc_teacher_online=ctc_teacher_online,
                            layer_metrics_output=step_layer_metrics,
                            layer_component_metrics_output=step_layer_component_metrics,
                            logit_metrics_output=step_logit_metrics,
                        )
                        step_eval_valid = step_eval_count > 0 and math.isfinite(step_eval_loss)
                        if step_eval_valid:
                            step_record = {
                                "step": step,
                                "epoch": epoch,
                                "eval_loss": step_eval_loss,
                                "eval_samples": step_eval_count,
                                **saved_artifacts,
                            }
                            if step_layer_metrics or step_logit_metrics:
                                layer_metrics_path = output_dir / f"step_eval_layers_step-{step}.yaml"
                                if _is_rank_zero():
                                    save_yaml(
                                        layer_metrics_path,
                                        {
                                            "step": step,
                                            "eval_loss": step_eval_loss,
                                            "eval_samples": step_eval_count,
                                            "layers": {
                                                str(layer_id): metrics
                                                for layer_id, metrics in step_layer_metrics.items()
                                            },
                                            "layer_components": {
                                                component: {
                                                    str(layer_id): metrics
                                                    for layer_id, metrics in component_metrics.items()
                                                }
                                                for component, component_metrics in step_layer_component_metrics.items()
                                            },
                                            "logit_metrics": step_logit_metrics,
                                        },
                                    )
                                step_record["layer_metrics_path"] = str(layer_metrics_path)
                            step_checkpoint_history.append(step_record)
                            saved_step_records = [
                                record
                                for record in step_checkpoint_history
                                if record.get("deepspeed_checkpoint_dir") or record.get("checkpoint_path")
                            ]
                            best_step_checkpoints = _sort_step_checkpoint_records(saved_step_records)[
                                : max(1, int(config.top_k_step_checkpoints))
                            ]
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
                                save_deepspeed_sharded=bool(config.save_deepspeed_sharded_checkpoints),
                            )
                            _maybe_barrier()
                            if _is_rank_zero():
                                _prune_deepspeed_step_checkpoint_artifacts(
                                    top_records=best_step_checkpoints,
                                    saved_records=[
                                        record
                                        for record in saved_step_records
                                        if int(record.get("step", 0)) != step
                                    ],
                                )
                            _maybe_barrier()
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
                                        **{
                                            f"eval/layer_{layer_id}_{metric_name}": metrics[metric_name]
                                            for layer_id, metrics in step_layer_metrics.items()
                                            for metric_name in ("loss", "cosine", "rms_ratio")
                                        },
                                        **{
                                            f"eval/layer_{layer_id}_{component}_{metric_name}": metrics[metric_name]
                                            for component, component_metrics in step_layer_component_metrics.items()
                                            for layer_id, metrics in component_metrics.items()
                                            for metric_name in ("loss", "cosine", "rms_ratio")
                                        },
                                        **{
                                            f"eval/ctc_alignment_{metric_name}": value
                                            for metric_name, value in step_logit_metrics.items()
                                        },
                                    },
                                    step=step,
                                )
                        elif _is_rank_zero():
                            _rank_zero_log(
                                f"Step checkpoint eval skipped: step={step} "
                                f"eval_samples={step_eval_count}; checkpoint saved but not ranked"
                            )
                            log_wandb(
                                wandb_run,
                                {
                                    "eval/step_eval_samples": step_eval_count,
                                    "eval/step_eval_skipped": 1,
                                    "checkpoint/top_k_kept": len(best_step_checkpoints),
                                },
                                step=step,
                            )
            if not processed_any_batch and epoch_batch_offset > 0:
                _all_rank_log(
                    f"No candidate batch available after resuming offset={epoch_batch_offset} for epoch={epoch}; "
                    f"continuing with epoch={epoch + 1}"
                )
                epoch += 1
                epoch_batch_offset = 0
                continue
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
                config=config,
                ctc_teacher_online=ctc_teacher_online,
            )
            epoch_eval_full = eval_limit is None
            epoch_eval_valid = epoch_eval_samples > 0 and math.isfinite(epoch_eval_loss)
            metric_value = epoch_eval_loss if epoch_eval_valid else epoch_train_loss
            metric_name = "eval_loss" if epoch_eval_valid else "train_loss"
            history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": epoch_train_loss,
                    "eval_loss": epoch_eval_loss if epoch_eval_valid else None,
                    "eval_samples": epoch_eval_samples,
                    "eval_full": epoch_eval_full,
                    "eval_skipped": not epoch_eval_valid,
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
                    "epoch_batch_offset": epoch_batch_offset,
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
                    save_deepspeed_sharded=bool(config.save_deepspeed_sharded_checkpoints),
                )
                if _is_rank_zero():
                    best_checkpoint: dict[str, Any] = {
                        "epoch": best_epoch,
                        "step": step,
                        "eval_loss": best_eval_loss,
                        "train_loss": best_train_loss,
                        "checkpoint_path": str(output_dir / "best.pt"),
                        "selection_metric_name": metric_name,
                    }
                    if config.save_deepspeed_sharded_checkpoints:
                        best_checkpoint["deepspeed_checkpoint_dir"] = str(output_dir / "ds_checkpoints")
                        best_checkpoint["resume_tag"] = "best"
                    save_yaml(
                        output_dir / "best_checkpoint.yaml",
                        best_checkpoint,
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
            completed_epoch_batch_count = epoch_batch_offset
            epoch_batch_offset = 0
            epoch_extra = {
                "loss": loss_value,
                "epoch_batch_offset": epoch_batch_offset,
                "completed_epoch_batch_count": completed_epoch_batch_count,
                "epoch": epoch,
                "history": history,
                "step_checkpoint_history": step_checkpoint_history,
                "best_step_checkpoints": best_step_checkpoints,
                "best_epoch": best_epoch,
                "best_eval_loss": best_eval_loss,
                "best_train_loss": best_train_loss,
                "cmvn_file": resolved_cmvn_file,
                "train_loss": epoch_train_loss,
                "eval_loss": epoch_eval_loss if epoch_eval_valid else None,
            }
            _save_export_checkpoints(
                engine=engine,
                output_dir=output_dir,
                tag=f"epoch-{epoch}",
                export_name=f"epoch-{epoch}.pt",
                step=step,
                zero_stage=zero_stage,
                extra_state=epoch_extra,
                save_deepspeed_sharded=bool(config.save_deepspeed_sharded_checkpoints),
            )
            if _is_rank_zero():
                eval_loss_text = f"{epoch_eval_loss:.4f}" if epoch_eval_valid else "skipped"
                _rank_zero_log(
                    f"Epoch {epoch} complete: train_loss={epoch_train_loss:.4f} "
                    f"eval_loss={eval_loss_text} eval_samples={epoch_eval_samples} "
                    f"eval_full={epoch_eval_full} best_epoch={best_epoch}"
                )
                epoch_wandb_payload = {
                    "epoch/index": epoch,
                    "epoch/train_loss": epoch_train_loss,
                    "epoch/eval_samples": epoch_eval_samples,
                    "epoch/eval_full": int(epoch_eval_full),
                    "checkpoint/best_epoch": best_epoch,
                }
                if epoch_eval_valid:
                    epoch_wandb_payload["epoch/eval_loss"] = epoch_eval_loss
                    epoch_wandb_payload["checkpoint/best_eval_loss"] = best_eval_loss
                else:
                    epoch_wandb_payload["epoch/eval_skipped"] = 1
                log_wandb(wandb_run, epoch_wandb_payload, step=step)
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
