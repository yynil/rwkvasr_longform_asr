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
    build_audio_feature_extractor,
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
    ctc_suppressed_token_ids_for_tokenizer,
    tokenizer_eos_token_id,
    validate_webdataset_index,
)
from rwkvasr.modules import (
    DirectionDropoutConfig,
    DirectionDropoutScheduler,
    RWKVCTCModel,
    RWKVCTCModelConfig,
    infer_rwkv7_decoder_config_from_checkpoint,
)

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
from .optimizer import RWKVOptimizerConfig, build_rwkv_optimizer
from .progress import start_training_progress, update_training_progress
from .spec_augment import apply_spec_augment
from .wandb_logger import finish_wandb, init_wandb_run, log_wandb


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _resolve_resume_from_path(config: TrainConfig) -> str | None:
    if config.resume_from is None:
        return None
    if config.resume_from != "latest":
        return str(config.resume_from)

    latest_state = load_latest_checkpoint_state(config.output_dir)
    resume_path = latest_state.get("checkpoint_path")
    if not isinstance(resume_path, str) or not resume_path:
        raise FileNotFoundError("resume_from='latest' requires latest_checkpoint.yaml with checkpoint_path")
    return resume_path


def _skip_batches(
    loader_iter: Any,
    count: int,
    *,
    progress_interval: int = 0,
    progress_callback: Any | None = None,
) -> int:
    skipped = 0
    while skipped < count:
        try:
            next(loader_iter)
        except StopIteration:
            break
        skipped += 1
        if progress_callback is not None and progress_interval > 0 and skipped % progress_interval == 0:
            progress_callback(skipped)
    return skipped


def _iter_loader_after_resume_offset(
    loader: Any,
    epoch_batch_offset: int,
    *,
    progress_callback: Any | None = None,
) -> tuple[Any, int, str]:
    if epoch_batch_offset <= 0:
        return iter(loader), 0, "none"

    fast_forward = getattr(loader, "iter_from_batch_offset", None)
    if callable(fast_forward):
        loader_iter, skipped = fast_forward(
            epoch_batch_offset,
            progress_interval=1000,
            progress_callback=progress_callback,
        )
        return iter(loader_iter), int(skipped), "metadata"

    loader_iter = iter(loader)
    skipped = _skip_batches(
        loader_iter,
        epoch_batch_offset,
        progress_interval=1000,
        progress_callback=progress_callback,
    )
    return loader_iter, skipped, "decoded"


def _checkpoint_latest_payload(
    *,
    checkpoint_type: str,
    checkpoint_path: Path | None,
    step: int,
    epoch: int,
    epoch_batch_offset: int,
    deepspeed_checkpoint_dir: str | None = None,
    resume_tag: str | None = None,
) -> dict[str, Any]:
    payload = {
        "checkpoint_type": str(checkpoint_type),
        "step": int(step),
        "epoch": int(epoch),
        "epoch_batch_offset": int(epoch_batch_offset),
    }
    if checkpoint_path is not None:
        payload["checkpoint_path"] = str(checkpoint_path)
    if deepspeed_checkpoint_dir is not None:
        payload["deepspeed_checkpoint_dir"] = str(deepspeed_checkpoint_dir)
    if resume_tag is not None:
        payload["resume_tag"] = str(resume_tag)
    return payload


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str
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
    length_bucket_schedule_block_size: int = 1
    bucket_source_interleave_block_size: int = 1
    bucket_serialize_reads: bool = False
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
    decoder_head_size: int = 64
    decoder_audio_conditioning: str = "full"
    decoder_prefix_tokens: int = 32
    decoder_loss_chunk_size: int = 1024
    decoder_text_token_budget: int | None = None
    decoder_prompt_before_audio: str = ""
    decoder_ctc_draft_cache_path: str | None = None
    decoder_ctc_draft_prompt_template: str = ""
    decoder_ctc_draft_text_key: str = "pred_text"
    decoder_ctc_draft_missing_policy: str = "empty"
    decoder_ctc_draft_dropout_prob: float = 0.0
    decoder_ctc_draft_language_mismatch_dropout_prob: float = 0.0
    decoder_ctc_draft_dropout_seed: int = 0
    ctc_label_override_cache_path: str | None = None
    ctc_label_override_text_key: str = "pred_text"
    decoder_prompt_after_audio: str = ""
    decoder_target_suffix: str = ""
    decoder_eos_token_id: int = 0
    ctc_loss_weight: float = 1.0
    decoder_loss_weight: float = 0.0
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
    funasr_nano_ctc_init_load_decoder: bool = True
    funasr_nano_ctc_init_load_head: bool = True
    funasr_nano_ctc_teacher_blank_id: int = 60514
    funasr_nano_ctc_init_blank_bias_delta: float = 0.0
    freeze_encoder: bool = False
    freeze_ctc_decoder: bool = False
    freeze_ctc_head: bool = False
    direction_variant: str = "none"
    p_start: float = 0.0
    p_max: float = 0.0
    warmup_steps: int = 0
    ramp_steps: int = 0
    device: str = "cpu"
    encoder_init_checkpoint_path: str | None = None
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
    log_every: int = 10
    batch_token_budget: int | None = None
    length_bucket_frame_budget: int | None = None
    target_gpu_memory_gib: float = 22.0
    skip_oversized_samples: bool = True
    specaugment_enabled: bool = False
    specaugment_time_masks: int = 2
    specaugment_time_width: int = 40
    specaugment_freq_masks: int = 2
    specaugment_freq_width: int = 15


def _resolve_data_source(config: TrainConfig) -> tuple[str, str]:
    has_manifest = config.manifest_path is not None
    has_webdataset = config.webdataset_root is not None
    if has_manifest == has_webdataset:
        raise ValueError("Exactly one of manifest_path or webdataset_root must be provided.")
    if has_manifest:
        return "manifest", str(config.manifest_path)
    return "webdataset", str(config.webdataset_root)


class _SyntheticTokenizer:
    def __init__(self, vocab_size: int):
        if vocab_size <= 1:
            raise ValueError("Synthetic tokenizer vocab_size must be greater than 1.")
        self._vocab_size = int(vocab_size)

    def encode(self, text: str) -> list[int]:
        usable = self._vocab_size - 1
        return [1 + (ord(char) % usable) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(int(token_id)) for token_id in token_ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return None


def _resolve_vocab_size(config: TrainConfig) -> int:
    tokenizer = _resolve_text_tokenizer(config)
    if config.tokenizer_append_eos and tokenizer_eos_token_id(tokenizer) is None:
        raise ValueError(
            "tokenizer_append_eos=True requires a tokenizer with a defined eos_token_id. "
            f"tokenizer_type={config.tokenizer_type!r} does not provide one."
        )
    decoder_enabled = bool(getattr(config, "decoder_enabled", False))
    decoder_checkpoint_path = getattr(config, "decoder_checkpoint_path", None)
    if decoder_enabled and decoder_checkpoint_path is not None:
        decoder_vocab_size = int(infer_rwkv7_decoder_config_from_checkpoint(decoder_checkpoint_path).vocab_size)
        if config.vocab_size is None:
            return decoder_vocab_size
        if int(config.vocab_size) != decoder_vocab_size:
            raise ValueError(
                "Configured vocab_size does not match the official RWKV decoder checkpoint vocabulary size: "
                f"{config.vocab_size} != {decoder_vocab_size}"
            )
    if config.vocab_size is not None:
        tokenizer_vocab_size = int(tokenizer.vocab_size)
        if int(config.vocab_size) != tokenizer_vocab_size:
            raise ValueError(
                "Configured vocab_size does not match the resolved tokenizer vocabulary size: "
                f"{config.vocab_size} != {tokenizer_vocab_size}"
            )
        return int(config.vocab_size)
    return int(tokenizer.vocab_size)


def _resolve_text_tokenizer(config: TrainConfig):
    if config.tokenizer_type == "synthetic":
        if config.vocab_size is None:
            raise ValueError("tokenizer_type='synthetic' requires vocab_size.")
        return _SyntheticTokenizer(int(config.vocab_size))
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
    tokenizer = _resolve_text_tokenizer(config)
    return {
        "tokenizer_type": config.tokenizer_type,
        "tokenizer_model_path": model_path,
        "tokenizer_language": config.tokenizer_language,
        "tokenizer_task": config.tokenizer_task,
        "tokenizer_append_eos": bool(config.tokenizer_append_eos),
        "tokenizer_eos_token_id": tokenizer_eos_token_id(tokenizer),
        "vocab_size": int(vocab_size),
    }


def _resolve_decoder_template_token_ids(config: TrainConfig) -> dict[str, Any]:
    tokenizer = _resolve_text_tokenizer(config)
    prompt_before_audio = "" if config.decoder_ctc_draft_prompt_template else config.decoder_prompt_before_audio
    return {
        "decoder_prompt_before_audio_token_ids": tuple(
            int(token_id) for token_id in tokenizer.encode(prompt_before_audio or "")
        ),
        "decoder_prompt_after_audio_token_ids": tuple(
            int(token_id) for token_id in tokenizer.encode(config.decoder_prompt_after_audio or "")
        ),
        "decoder_target_suffix_token_ids": tuple(
            int(token_id) for token_id in tokenizer.encode(config.decoder_target_suffix or "")
        ),
        "decoder_eos_token_id": int(config.decoder_eos_token_id),
    }


def _resolve_ctc_suppressed_token_ids(
    config: Any,
    *,
    vocab_size: int,
    tokenizer: Any | None = None,
) -> tuple[int, ...]:
    suppressed = {int(token_id) for token_id in getattr(config, "ctc_suppressed_token_ids", ())}
    if bool(getattr(config, "ctc_suppress_non_pronunciation_tokens", False)):
        if tokenizer is None:
            tokenizer = _resolve_text_tokenizer(config)
        suppressed.update(
            ctc_suppressed_token_ids_for_tokenizer(
                tokenizer,
                blank_id=int(getattr(config, "blank_id", 0)),
            )
        )
    blank_id = int(getattr(config, "blank_id", 0))
    ctc_vocab_size = max(int(vocab_size), blank_id + 1)
    return tuple(sorted(token_id for token_id in suppressed if 0 <= token_id < ctc_vocab_size and token_id != blank_id))


def _build_webdataset_config(
    config: TrainConfig,
    *,
    shuffle_shards: bool,
    skip_decode_errors: bool = False,
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
        length_bucket_schedule_block_size=config.length_bucket_schedule_block_size,
        bucket_source_interleave_block_size=config.bucket_source_interleave_block_size,
        bucket_serialize_reads=config.bucket_serialize_reads,
        append_eos=config.tokenizer_append_eos,
        text_normalization=config.text_normalization,
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
        skip_decode_errors=skip_decode_errors,
    )


def _resolve_candidate_length_index_path(data_path: str, configured_path: str | None) -> str | None:
    length_index_path = configured_path
    if length_index_path is None:
        candidate_path = resolve_webdataset_length_index_path(data_path)
        length_index_path = str(candidate_path) if candidate_path.exists() else None
    return length_index_path


def _resolve_bucket_manifest_path(
    data_path: str,
    configured_path: str | None,
    *,
    configured_length_index_path: str | None = None,
) -> str | None:
    if configured_path is None and configured_length_index_path is not None:
        return None
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


def _maybe_load_encoder_init_checkpoint(model: RWKVCTCModel, checkpoint_path: str | None) -> None:
    if checkpoint_path is None:
        return
    aut_encoder = getattr(model.encoder, "aut_encoder", None)
    load_fn = getattr(aut_encoder, "load_qwen3_asr_non_attention_checkpoint", None)
    if callable(load_fn):
        report = load_fn(checkpoint_path)
        _log(
            "Loaded AuRWKV non-attention encoder init: "
            f"path={checkpoint_path} loaded={len(report['loaded'])} skipped={len(report['skipped'])}"
        )
        return
    qwen3_transformer_encoder = getattr(model.encoder, "qwen3_transformer_encoder", None)
    load_fn = getattr(qwen3_transformer_encoder, "load_qwen3_asr_checkpoint", None)
    if callable(load_fn):
        report = load_fn(checkpoint_path)
        _log(
            "Loaded Qwen3 Transformer audio encoder init: "
            f"path={checkpoint_path} loaded={len(report['loaded'])} skipped={len(report['skipped'])}"
        )
        return
    sensevoice_encoder = getattr(model.encoder, "sensevoice_encoder", None)
    load_fn = getattr(sensevoice_encoder, "load_sensevoice_non_attention_checkpoint", None)
    if callable(load_fn):
        report = load_fn(checkpoint_path)
        _log(
            "Loaded SenseVoiceRWKV non-attention encoder init: "
            f"path={checkpoint_path} loaded={len(report['loaded'])} skipped={len(report['skipped'])}"
        )
        return
    sensevoice_conformer_encoder = getattr(model.encoder, "sensevoice_conformer_encoder", None)
    load_fn = getattr(sensevoice_conformer_encoder, "load_sensevoice_non_attention_checkpoint", None)
    if callable(load_fn):
        report = load_fn(checkpoint_path)
        _log(
            "Loaded SenseVoice Conformer-conv non-attention encoder init: "
            f"path={checkpoint_path} loaded={len(report['loaded'])} skipped={len(report['skipped'])}"
        )
        return
    raise ValueError(
        "encoder_init_checkpoint_path is only supported by "
        "frontend_type='aut_rwkv', 'qwen3_transformer', 'sensevoice_rwkv', "
        "or 'sensevoice_conformer_conv'."
    )


def _maybe_load_funasr_nano_ctc_init(model: RWKVCTCModel, config: TrainConfig) -> None:
    if config.funasr_nano_ctc_init_checkpoint_path is None:
        return
    report = model.load_funasr_nano_ctc_checkpoint(
        config.funasr_nano_ctc_init_checkpoint_path,
        load_ctc_decoder=bool(config.funasr_nano_ctc_init_load_decoder),
        load_ctc_head=bool(config.funasr_nano_ctc_init_load_head),
        teacher_blank_id=int(config.funasr_nano_ctc_teacher_blank_id),
        blank_bias_delta=float(config.funasr_nano_ctc_init_blank_bias_delta),
    )
    _log(
        "Loaded FunASR-Nano CTC init: "
        f"path={config.funasr_nano_ctc_init_checkpoint_path} "
        f"bridge_tensors={report.get('ctc_bridge_loaded', 0)} "
        f"bridge_skipped={report.get('ctc_bridge_skipped', 0)} "
        f"decoder_tensors={report['ctc_decoder_loaded']} "
        f"head_rows={report['ctc_head_loaded_rows']} "
        f"ignored_rows={report['ctc_head_ignored_rows']} "
        f"blank_bias_delta={report['ctc_head_blank_bias_delta']}"
    )


def _apply_training_freeze(model: RWKVCTCModel, config: TrainConfig) -> None:
    frozen_names: list[str] = []
    if config.freeze_encoder:
        for name, parameter in model.encoder.named_parameters(prefix="encoder"):
            parameter.requires_grad_(False)
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
        _log(
            "Applied parameter freeze: "
            f"freeze_encoder={bool(config.freeze_encoder)} "
            f"freeze_ctc_decoder={bool(config.freeze_ctc_decoder)} "
            f"freeze_ctc_head={bool(config.freeze_ctc_head)} "
            f"frozen_tensors={len(frozen_names)} "
            f"trainable_params={trainable_params}"
        )


def _build_train_loader(config: TrainConfig) -> DataLoader:
    data_source, data_path = _resolve_data_source(config)
    tokenizer = _resolve_text_tokenizer(config)
    feature_extractor = build_audio_feature_extractor(
        config.feature_extractor_type,
        input_dim=int(config.input_dim),
    )
    if data_source == "manifest":
        dataset = ASRManifestDataset(
            data_path,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            append_eos=config.tokenizer_append_eos,
            text_normalization=config.text_normalization,
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=FeatureCollator(),
        )
    webdataset_config = _build_webdataset_config(
        config,
        shuffle_shards=True,
        skip_decode_errors=True,
    )
    bucket_manifest_path = _resolve_bucket_manifest_path(
        data_path,
        config.webdataset_bucket_manifest_path,
        configured_length_index_path=config.webdataset_length_index_path,
    )
    if bucket_manifest_path is not None:
        return build_bucketed_webdataset_loader(
            data_path,
            bucket_manifest_path=bucket_manifest_path,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
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
            feature_extractor=feature_extractor,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        return loader
    return build_webdataset_dataloader(
        data_path,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        config=webdataset_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


def _build_eval_loader(config: TrainConfig, *, shuffle_shards: bool = False, step_subset: bool = False) -> DataLoader | None:
    data_source, data_path = _resolve_data_source(config)
    eval_batch_size = _resolve_eval_batch_size(config, step_subset=step_subset)
    if data_source == "manifest":
        dataset = ASRManifestDataset(
            data_path,
            tokenizer=_resolve_text_tokenizer(config),
            feature_extractor=build_audio_feature_extractor(
                config.feature_extractor_type,
                input_dim=int(config.input_dim),
            ),
            append_eos=config.tokenizer_append_eos,
            text_normalization=config.text_normalization,
        )
        return DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=shuffle_shards,
            num_workers=config.num_workers,
            collate_fn=FeatureCollator(),
        )

    tokenizer = _resolve_text_tokenizer(config)
    feature_extractor = build_audio_feature_extractor(
        config.feature_extractor_type,
        input_dim=int(config.input_dim),
    )
    webdataset_config = _build_webdataset_config(config, shuffle_shards=shuffle_shards)
    webdataset_config = replace(
        webdataset_config,
        split="eval",
        ctc_label_override_cache_path=None,
    )
    bucket_manifest_path = _resolve_bucket_manifest_path(
        data_path,
        config.webdataset_bucket_manifest_path,
        configured_length_index_path=config.webdataset_length_index_path,
    )
    if bucket_manifest_path is not None:
        return build_bucketed_webdataset_loader(
            data_path,
            bucket_manifest_path=bucket_manifest_path,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
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
            feature_extractor=feature_extractor,
            config=webdataset_config,
            batch_size=eval_batch_size,
            num_workers=config.num_workers,
        )
        return loader
    return build_webdataset_dataloader(
        data_path,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
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
            configured_length_index_path=config.webdataset_length_index_path,
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
            f"decode_workers={max(1, config.num_workers)} "
            f"schedule_block={max(1, config.length_bucket_schedule_block_size)} "
            f"source_block={max(1, config.bucket_source_interleave_block_size)} "
            f"serialized_reads={bool(config.bucket_serialize_reads)}"
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
        decoder_head_size=config.decoder_head_size,
        decoder_audio_conditioning=config.decoder_audio_conditioning,
        decoder_prefix_tokens=config.decoder_prefix_tokens,
        decoder_loss_chunk_size=config.decoder_loss_chunk_size,
        **_resolve_decoder_template_token_ids(config),
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
    _maybe_load_encoder_init_checkpoint(model, config.encoder_init_checkpoint_path)
    _maybe_load_funasr_nano_ctc_init(model, config)
    _apply_training_freeze(model, config)
    if feature_dtype is not None:
        model = model.to(device=device, dtype=feature_dtype)
    else:
        model = model.to(device)
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
    _log("Direction dropout disabled; training uses full bidirectional encoder masks.")
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
    start_epoch_batch_offset = 0
    history: list[dict[str, float | int | str]] = []
    step_checkpoint_history: list[dict[str, Any]] = []
    best_step_checkpoints: list[dict[str, Any]] = []
    best_epoch = 0
    best_eval_loss = float("inf")
    best_train_loss = float("inf")
    resolved_resume_path = _resolve_resume_from_path(config)
    if resolved_resume_path is not None:
        restored = load_checkpoint(
            resolved_resume_path,
            model=model,
            optimizer=optimizer,
            map_location=device.type,
        )
        start_step = int(restored["step"])
        extra = dict(restored.get("extra", {}))
        start_epoch = int(extra.get("epoch", 0))
        start_epoch_batch_offset = extract_epoch_batch_offset(extra.get("epoch_batch_offset", 0))
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
    epoch = start_epoch
    epoch_batch_offset = start_epoch_batch_offset
    loss_value = float("nan")
    progress, task_id = start_training_progress(
        total_steps=resolved_max_steps,
        start_step=start_step,
        description="train",
    )
    _log("The first batch can be slower because wav->fbank decoding is done online.")
    if start_epoch_batch_offset > 0:
        _log(
            f"Resuming from latest state: step={start_step} epoch={start_epoch} "
            f"epoch_batch_offset={start_epoch_batch_offset}"
        )

    train_start_time = time.perf_counter()
    try:
        while step < resolved_max_steps:
            if epoch_batch_offset == 0:
                epoch += 1
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            _set_loader_epoch(loader, epoch)
            if epoch_batch_offset > 0:
                _log(f"Fast-forwarding resume offset={epoch_batch_offset} for epoch={epoch}.")
                loader_iter, skipped_batches, skip_mode = _iter_loader_after_resume_offset(
                    loader,
                    epoch_batch_offset,
                    progress_callback=lambda skipped: _log(
                        f"Fast-forwarded resume batches {skipped}/{epoch_batch_offset} for epoch={epoch}."
                    ),
                )
                _log(
                    f"Resume fast-forward complete: skipped={skipped_batches}/{epoch_batch_offset} "
                    f"mode={skip_mode} epoch={epoch}"
                )
                if skipped_batches < epoch_batch_offset:
                    _log(
                        f"Resume offset {epoch_batch_offset} exceeded available batches for epoch={epoch}; "
                        f"continuing with epoch={epoch + 1}"
                    )
                    epoch += 1
                    epoch_batch_offset = 0
                    continue
            else:
                loader_iter = iter(loader)
            processed_any_batch = False
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
                    _log("Skipped a candidate batch because no sample fit inside the token budget.")
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
                optimizer.zero_grad(set_to_none=True)
                mask = task.training_direction_mask(step, device=device)
                executed_token_stats = batch_stats
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                batch = batch.to(device, feature_dtype=feature_dtype)
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
                try:
                    loss, _ = task.training_loss(batch, step=step, direction_mask=mask)
                except torch.OutOfMemoryError:
                    _log(
                        "OOM during training_loss "
                        f"step={step + 1} batch={executed_token_stats.batch_size} "
                        f"budget={executed_budget_tokens} total={executed_token_stats.total_tokens} "
                        f"text_budget={executed_text_budget_tokens} "
                        f"max_audio={executed_token_stats.max_audio_frames} "
                        f"max_text={executed_token_stats.max_text_tokens} "
                        f"padded_text={executed_token_stats.padded_text_tokens}"
                    )
                    raise
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
                    observed_tokens=executed_budget_tokens,
                    observed_peak_reserved_bytes=peak_reserved_bytes,
                    target_memory_gib=config.target_gpu_memory_gib,
                )
                epoch_loss_sum += loss_value * batch_stats.batch_size
                epoch_sample_count += batch_stats.batch_size
                if step == start_step + 1:
                    _log(
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
                        _log(
                            f"Token budget active: budget={config.batch_token_budget} "
                            f"text_budget={config.decoder_text_token_budget} "
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
                    step_checkpoint_path = output_dir / f"step-{step}.pt"
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
                    save_checkpoint(
                        step_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        extra=checkpoint_extra,
                    )
                    write_latest_checkpoint_state(
                        output_dir,
                        _checkpoint_latest_payload(
                            checkpoint_type="pytorch",
                            checkpoint_path=step_checkpoint_path,
                            step=step,
                            epoch=epoch,
                            epoch_batch_offset=epoch_batch_offset,
                        ),
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
                        checkpoint_extra["step_checkpoint_history"] = step_checkpoint_history
                        checkpoint_extra["best_step_checkpoints"] = best_step_checkpoints
                        save_checkpoint(
                            step_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            extra=checkpoint_extra,
                        )
                        write_latest_checkpoint_state(
                            output_dir,
                            _checkpoint_latest_payload(
                                checkpoint_type="pytorch",
                                checkpoint_path=step_checkpoint_path,
                                step=step,
                                epoch=epoch,
                                epoch_batch_offset=epoch_batch_offset,
                            ),
                        )
                        _prune_local_step_checkpoint_artifacts(
                            top_records=best_step_checkpoints,
                            saved_records=[
                                record
                                for record in saved_step_records
                                if int(record.get("step", 0)) != step
                            ],
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
            if not processed_any_batch and epoch_batch_offset > 0:
                _log(
                    f"No candidate batch available after resuming offset={epoch_batch_offset} for epoch={epoch}; "
                    f"continuing with epoch={epoch + 1}"
                )
                epoch += 1
                epoch_batch_offset = 0
                continue
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
                best_checkpoint_path = output_dir / "best.pt"
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    extra={
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
                    },
                )
                write_latest_checkpoint_state(
                    output_dir,
                    _checkpoint_latest_payload(
                        checkpoint_type="pytorch",
                        checkpoint_path=best_checkpoint_path,
                        step=step,
                        epoch=epoch,
                        epoch_batch_offset=epoch_batch_offset,
                    ),
                )
                save_yaml(
                    output_dir / "best_checkpoint.yaml",
                    {
                        "epoch": best_epoch,
                        "step": step,
                        "eval_loss": best_eval_loss,
                        "train_loss": best_train_loss,
                        "checkpoint_path": str(best_checkpoint_path),
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
            epoch_batch_offset = 0
            epoch_checkpoint_path = output_dir / f"epoch-{epoch}.pt"
            save_checkpoint(
                epoch_checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                extra={
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
                },
            )
            write_latest_checkpoint_state(
                output_dir,
                _checkpoint_latest_payload(
                    checkpoint_type="pytorch",
                    checkpoint_path=epoch_checkpoint_path,
                    step=step,
                    epoch=epoch,
                    epoch_batch_offset=epoch_batch_offset,
                ),
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
