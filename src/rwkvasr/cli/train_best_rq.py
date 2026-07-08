from __future__ import annotations

import argparse
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.training.best_rq_loop import BestRQTrainConfig, train_best_rq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Best-RQ self-supervised encoder probe.")
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--init-checkpoint-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--save-every", default=None, type=int)
    parser.add_argument("--log-every", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--batch-token-budget", default=None, type=int)
    parser.add_argument("--length-bucket-frame-budget", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--best-rq-codebook-size", default=None, type=int)
    parser.add_argument("--best-rq-projection-dim", default=None, type=int)
    parser.add_argument("--best-rq-mask-prob", default=None, type=float)
    parser.add_argument("--best-rq-mask-span-length", default=None, type=int)
    parser.add_argument("--best-rq-quantizer-seed", default=None, type=int)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use-bf16", dest="use_bf16", action="store_true", default=None)
    parser.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    return parser


def _resolve_config(args: argparse.Namespace) -> BestRQTrainConfig:
    data: dict[str, Any] = load_yaml(args.config_yaml)
    keys = (
        "output_dir",
        "init_checkpoint_path",
        "device",
        "max_steps",
        "save_every",
        "log_every",
        "batch_size",
        "batch_token_budget",
        "length_bucket_frame_budget",
        "num_workers",
        "lr",
        "best_rq_codebook_size",
        "best_rq_projection_dim",
        "best_rq_mask_prob",
        "best_rq_mask_span_length",
        "best_rq_quantizer_seed",
        "gradient_checkpointing",
        "use_bf16",
    )
    for key in keys:
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    return BestRQTrainConfig(**data)


def main() -> None:
    config = _resolve_config(build_parser().parse_args())
    result = train_best_rq(config)
    print(
        "train_best_rq "
        f"steps={result['steps']} final_loss={result['final_loss']:.4f} "
        f"final_accuracy={result['final_accuracy']:.4f} checkpoint={result['checkpoint_path']}"
    )


if __name__ == "__main__":
    main()
