from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rwkvasr.config import save_yaml
from rwkvasr.data import (
    ASRManifestDataset,
    FeatureCollator,
    WebDatasetConfig,
    build_audio_feature_extractor,
    build_bucketed_webdataset_loader,
    build_length_bucketed_webdataset_dataloader,
    build_text_tokenizer,
    build_webdataset_dataloader,
)
from rwkvasr.modules import BestRQPretrainModel
from rwkvasr.modules.best_rq import best_rq_model_config_from_training_config

from .batch_budget import ctc_batch_token_stats, select_ctc_batch_prefix_by_token_budget
from .checkpoint import load_checkpoint, save_checkpoint, write_latest_checkpoint_state
from .optimizer import RWKVOptimizerConfig, build_rwkv_optimizer
from .train_loop import (
    _resolve_bucket_manifest_path,
    _resolve_data_source,
    _resolve_in_memory_length_index_path,
)


@dataclass(frozen=True)
class BestRQTrainConfig:
    output_dir: str
    manifest_path: str | None = None
    webdataset_root: str | None = None
    webdataset_length_index_path: str | None = None
    webdataset_bucket_manifest_path: str | None = None
    webdataset_split: str = "train"
    webdataset_eval_ratio: float = 0.0
    webdataset_hash_seed: int = 0
    webdataset_split_by: str = "sample_id"
    webdataset_utt_id_key: str = "id"
    tokenizer_type: str = "sensevoice_tiktoken"
    tokenizer_model_path: str | None = None
    tokenizer_language: str | None = None
    tokenizer_task: str | None = None
    tokenizer_append_eos: bool = False
    text_normalization: str = "ctc"
    vocab_size: int = 1
    blank_id: int = 0
    feature_extractor_type: str = "sensevoice_lfr_fbank"
    input_dim: int = 560
    n_embd: int = 512
    encoder_output_dim: int | None = 512
    dim_att: int = 512
    dim_ff: int = 2048
    num_layers: int = 70
    sensevoice_tp_blocks: int = 20
    head_size: int = 64
    backend: str = "cuda_clampw"
    conv_kernel_size: int = 31
    dropout: float = 0.1
    frontend_type: str = "sensevoice_rwkv"
    aut_downsample_hidden_size: int = 480
    aut_activation_function: str = "gelu"
    aut_activation_dropout: float = 0.0
    aut_max_source_positions: int = 1500
    aut_scale_embedding: bool = False
    aut_conv_chunksize: int = 500
    cmvn_file: str | None = None
    cmvn_is_json: bool = True
    best_rq_codebook_size: int = 8192
    best_rq_projection_dim: int = 16
    best_rq_mask_prob: float = 0.15
    best_rq_mask_span_length: int = 10
    best_rq_quantizer_seed: int = 20260609
    init_checkpoint_path: str | None = None
    device: str = "cuda:0"
    batch_size: int = 1
    batch_token_budget: int | None = 900
    length_bucket_frame_budget: int | None = 900
    skip_oversized_samples: bool = True
    num_workers: int = 2
    decoded_batch_prefetch: int = 1
    max_open_shards_per_worker: int = 4
    bucket_source_interleave: bool = True
    max_steps: int = 500
    save_every: int = 100
    log_every: int = 10
    lr: float = 2.0e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    gradient_checkpointing: bool = True
    use_bf16: bool = True


def _build_best_rq_loader(config: BestRQTrainConfig) -> DataLoader:
    data_source, data_path = _resolve_data_source(config)
    tokenizer = build_text_tokenizer(
        config.tokenizer_type,
        model_path=config.tokenizer_model_path,
        language=config.tokenizer_language,
        task=config.tokenizer_task,
    )
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

    webdataset_config = WebDatasetConfig(
        shuffle_shards=True,
        split=config.webdataset_split,
        eval_ratio=config.webdataset_eval_ratio,
        hash_seed=config.webdataset_hash_seed,
        split_by=config.webdataset_split_by,
        utt_id_key=config.webdataset_utt_id_key,
        length_index_path=config.webdataset_length_index_path,
        length_bucket_frame_budget=config.length_bucket_frame_budget or config.batch_token_budget,
        decoded_batch_prefetch=config.decoded_batch_prefetch,
        max_open_shards_per_worker=config.max_open_shards_per_worker,
        bucket_source_interleave=config.bucket_source_interleave,
        append_eos=config.tokenizer_append_eos,
        text_normalization=config.text_normalization,
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
            decoder_tokenizer=None,
            feature_extractor=feature_extractor,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            rank=0,
            world_size=1,
        )
    length_index_path = _resolve_in_memory_length_index_path(data_path, config.webdataset_length_index_path)
    if length_index_path is not None:
        loader, _ = build_length_bucketed_webdataset_dataloader(
            data_path,
            length_index_path=length_index_path,
            tokenizer=tokenizer,
            decoder_tokenizer=None,
            feature_extractor=feature_extractor,
            config=webdataset_config,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            rank=0,
            world_size=1,
        )
        return loader
    return build_webdataset_dataloader(
        data_path,
        tokenizer=tokenizer,
        decoder_tokenizer=None,
        feature_extractor=feature_extractor,
        config=webdataset_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


def _load_initial_checkpoint(model: BestRQPretrainModel, path: str | None) -> None:
    if not path:
        return
    restored = load_checkpoint(path, model=model, optimizer=None, map_location="cpu", strict=False)
    extra = restored.get("extra", {})
    missing = extra.get("missing_keys", []) if isinstance(extra, dict) else []
    unexpected = extra.get("unexpected_keys", []) if isinstance(extra, dict) else []
    print(
        f"[rwkvasr-bestrq] loaded init checkpoint {path} "
        f"step={int(restored.get('step', 0))} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def train_best_rq(config: BestRQTrainConfig) -> dict[str, float | int | str]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(output_dir / "best_rq_train_config.yaml", config.__dict__)
    model_config = best_rq_model_config_from_training_config(config)
    save_yaml(output_dir / "best_rq_model_config.yaml", model_config)

    device = torch.device(config.device if torch.cuda.is_available() or not config.device.startswith("cuda") else "cpu")
    feature_dtype = torch.bfloat16 if device.type == "cuda" and config.use_bf16 else None
    model = BestRQPretrainModel(model_config)
    model.enable_gradient_checkpointing(config.gradient_checkpointing)
    _load_initial_checkpoint(model, config.init_checkpoint_path)
    if feature_dtype is not None:
        model = model.to(device=device, dtype=feature_dtype)
    else:
        model = model.to(device=device)
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
    loader = _build_best_rq_loader(config)
    model.train()
    step = 0
    last_loss = float("nan")
    last_accuracy = 0.0
    start = time.perf_counter()
    while step < int(config.max_steps):
        for candidate_batch in loader:
            if step >= int(config.max_steps):
                break
            budgeted = select_ctc_batch_prefix_by_token_budget(
                candidate_batch,
                token_budget=config.batch_token_budget,
                skip_oversized_samples=config.skip_oversized_samples,
                use_padded_text_tokens=False,
            )
            if budgeted is None:
                continue
            batch = budgeted.batch.to(device, feature_dtype=feature_dtype)
            optimizer.zero_grad(set_to_none=True)
            stats = model(batch.features, batch.feature_lengths)
            loss = stats["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            last_loss = float(loss.detach().item())
            last_accuracy = float(stats["accuracy"].detach().item())
            if step == 1 or step % int(config.log_every) == 0:
                batch_stats = ctc_batch_token_stats(batch)
                elapsed = max(time.perf_counter() - start, 1e-6)
                print(
                    "[rwkvasr-bestrq] "
                    f"step={step} loss={last_loss:.4f} acc={last_accuracy:.4f} "
                    f"masked={int(stats['masked_frames'].detach().item())} "
                    f"batch={batch_stats.batch_size} max_audio={batch_stats.max_audio_frames} "
                    f"rate={step / elapsed:.3f} step/s",
                    flush=True,
                )
            if step % int(config.save_every) == 0:
                checkpoint_path = output_dir / f"step-{step}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    extra={"task": "best_rq_pretrain"},
                )
                write_latest_checkpoint_state(
                    output_dir,
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_type": "best_rq",
                        "step": step,
                    },
                )
        else:
            continue
        break
    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        step=step,
        extra={"task": "best_rq_pretrain"},
    )
    write_latest_checkpoint_state(
        output_dir,
        {
            "checkpoint_path": str(final_path),
            "checkpoint_type": "best_rq",
            "step": step,
        },
    )
    return {
        "steps": step,
        "final_loss": last_loss,
        "final_accuracy": last_accuracy,
        "checkpoint_path": str(final_path),
    }
