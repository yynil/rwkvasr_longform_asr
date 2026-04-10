from __future__ import annotations

import argparse

from rwkvasr.config import load_yaml
from rwkvasr.training.train_loop import TrainConfig, train_ctc_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RWKV dual-mode CTC ASR from a manifest or WebDataset.")
    parser.add_argument("--config-yaml", default=None)
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
    parser.add_argument("--input-dim", default=None, type=int)
    parser.add_argument("--n-embd", default=None, type=int)
    parser.add_argument("--dim-att", default=None, type=int)
    parser.add_argument("--dim-ff", default=None, type=int)
    parser.add_argument("--num-layers", default=None, type=int)
    parser.add_argument("--head-size", default=None, type=int)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--conv-kernel-size", default=None, type=int)
    parser.add_argument("--dropout", default=None, type=float)
    parser.add_argument("--frontend-type", default=None)
    parser.add_argument("--cmvn-file", default=None)
    parser.add_argument("--cmvn-is-json", dest="cmvn_is_json", action="store_true", default=None)
    parser.add_argument("--cmvn-is-bin", dest="cmvn_is_json", action="store_false")
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--save-every", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--decoded-batch-prefetch", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--weight-decay", default=None, type=float)
    parser.add_argument("--beta1", default=None, type=float)
    parser.add_argument("--beta2", default=None, type=float)
    parser.add_argument("--eps", default=None, type=float)
    parser.add_argument("--direction-variant", default=None)
    parser.add_argument("--p-start", default=None, type=float)
    parser.add_argument("--p-max", default=None, type=float)
    parser.add_argument("--warmup-steps", default=None, type=int)
    parser.add_argument("--ramp-steps", default=None, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", default=None)
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


def _resolve_train_config(args: argparse.Namespace) -> TrainConfig:
    config_data: dict[str, object] = {}
    if args.config_yaml is not None:
        config_data.update(load_yaml(args.config_yaml))

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
        "input_dim",
        "n_embd",
        "dim_att",
        "dim_ff",
        "num_layers",
        "head_size",
        "backend",
        "conv_kernel_size",
        "dropout",
        "frontend_type",
        "cmvn_file",
        "cmvn_is_json",
        "batch_size",
        "max_steps",
        "epochs",
        "save_every",
        "num_workers",
        "decoded_batch_prefetch",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "direction_variant",
        "p_start",
        "p_max",
        "warmup_steps",
        "ramp_steps",
        "device",
        "resume_from",
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
        "batch_token_budget",
        "length_bucket_frame_budget",
        "target_gpu_memory_gib",
        "skip_oversized_samples",
    )
    for key in cli_keys:
        value = getattr(args, key)
        if value is not None:
            config_data[key] = value

    try:
        return TrainConfig(**config_data)
    except TypeError as exc:
        raise ValueError(
            "TrainConfig is incomplete. Provide required fields in --config-yaml or via CLI."
        ) from exc


def main() -> None:
    args = build_parser().parse_args()
    result = train_ctc_model(_resolve_train_config(args))
    print(f"train_ctc steps={result['steps']} final_loss={result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
