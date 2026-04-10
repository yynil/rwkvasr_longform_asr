from __future__ import annotations

import argparse
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.training.deepspeed_loop import DeepSpeedTrainConfig, train_ctc_model_deepspeed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RWKV dual-mode CTC ASR with DeepSpeed.")
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--webdataset-root", default=None)
    parser.add_argument("--webdataset-index-path", default=None)
    parser.add_argument("--webdataset-length-index-path", default=None)
    parser.add_argument("--webdataset-bucket-manifest-path", default=None)
    parser.add_argument("--webdataset-split", default=None)
    parser.add_argument("--webdataset-eval-ratio", default=None, type=float)
    parser.add_argument("--webdataset-hash-seed", default=None, type=int)
    parser.add_argument("--webdataset-split-by", default=None)
    parser.add_argument("--webdataset-utt-id-key", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vocab-size", default=None, type=int)
    parser.add_argument("--tokenizer-type", default=None)
    parser.add_argument("--tokenizer-model-path", default=None)
    parser.add_argument("--tokenizer-language", default=None)
    parser.add_argument("--tokenizer-task", default=None)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--save-every", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--decoded-batch-prefetch", default=None, type=int)
    parser.add_argument("--max-open-shards-per-worker", default=None, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-tag", default=None)
    parser.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-base-url", default=None)
    parser.add_argument("--wandb-init-timeout-sec", default=None, type=float)
    parser.add_argument("--eval-mode", default=None)
    parser.add_argument("--max-eval-samples", default=None, type=int)
    parser.add_argument("--eval-batch-size", default=None, type=int)
    parser.add_argument("--step-eval-batch-size", default=None, type=int)
    parser.add_argument("--step-eval-every", default=None, type=int)
    parser.add_argument("--step-eval-samples", default=None, type=int)
    parser.add_argument("--top-k-step-checkpoints", default=None, type=int)
    parser.add_argument("--local-rank", default=None, type=int)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--batch-token-budget", default=None, type=int)
    parser.add_argument("--length-bucket-frame-budget", default=None, type=int)
    parser.add_argument("--target-gpu-memory-gib", default=None, type=float)
    parser.add_argument("--skip-oversized-samples", dest="skip_oversized_samples", action="store_true", default=None)
    parser.add_argument(
        "--no-skip-oversized-samples",
        dest="skip_oversized_samples",
        action="store_false",
    )
    return parser


def _resolve_deepspeed_train_config(args: argparse.Namespace) -> DeepSpeedTrainConfig:
    config_data: dict[str, Any] = load_yaml(args.config_yaml)
    cli_keys = (
        "manifest_path",
        "webdataset_root",
        "webdataset_index_path",
        "webdataset_length_index_path",
        "webdataset_bucket_manifest_path",
        "webdataset_split",
        "webdataset_eval_ratio",
        "webdataset_hash_seed",
        "webdataset_split_by",
        "webdataset_utt_id_key",
        "output_dir",
        "vocab_size",
        "tokenizer_type",
        "tokenizer_model_path",
        "tokenizer_language",
        "tokenizer_task",
        "batch_size",
        "backend",
        "max_steps",
        "epochs",
        "save_every",
        "num_workers",
        "decoded_batch_prefetch",
        "max_open_shards_per_worker",
        "device",
        "resume_from",
        "resume_tag",
        "wandb_enabled",
        "wandb_project",
        "wandb_run_name",
        "wandb_base_url",
        "wandb_init_timeout_sec",
        "eval_mode",
        "max_eval_samples",
        "eval_batch_size",
        "step_eval_batch_size",
        "step_eval_every",
        "step_eval_samples",
        "top_k_step_checkpoints",
        "local_rank",
        "gradient_checkpointing",
        "batch_token_budget",
        "length_bucket_frame_budget",
        "target_gpu_memory_gib",
        "skip_oversized_samples",
    )
    for key in cli_keys:
        value = getattr(args, key)
        if value is not None:
            config_data[key] = value

    if "deepspeed" not in config_data:
        raise ValueError("DeepSpeed YAML config must contain a top-level `deepspeed` mapping.")

    try:
        return DeepSpeedTrainConfig(**config_data)
    except TypeError as exc:
        raise ValueError(
            "DeepSpeedTrainConfig is incomplete or has unexpected fields. "
            "Provide required fields in --config-yaml."
        ) from exc


def main() -> None:
    args = build_parser().parse_args()
    result = train_ctc_model_deepspeed(_resolve_deepspeed_train_config(args))
    print(
        "train_ctc_deepspeed "
        f"steps={result['steps']} final_loss={result['final_loss']:.4f} zero_stage={result['zero_stage']}"
    )


if __name__ == "__main__":
    main()
