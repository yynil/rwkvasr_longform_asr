from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

from rwkvasr.config import load_yaml
from rwkvasr.modules import RWKVCTCModelConfig
from rwkvasr.predict import PredictionConfig, predict_ctc, write_predictions_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict with a RWKV CTC checkpoint using prefix beam search.")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--webdataset-root", default=None)
    parser.add_argument("--webdataset-split", default="all")
    parser.add_argument("--webdataset-eval-ratio", default=0.0, type=float)
    parser.add_argument("--webdataset-hash-seed", default=0, type=int)
    parser.add_argument("--webdataset-split-by", default="shard_name")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-yaml", default=None)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="bi", choices=["bi", "l2r", "r2l", "alt"])
    parser.add_argument("--beam-size", default=8, type=int)
    parser.add_argument("--token-prune-topk", default=None, type=int)
    parser.add_argument("--length-bonus", default=0.0, type=float)
    parser.add_argument("--insertion-bonus", default=0.0, type=float)
    parser.add_argument("--save-debug-lengths", action="store_true")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--tokenizer-type", default=None)
    parser.add_argument("--tokenizer-model-path", default=None)
    parser.add_argument("--tokenizer-language", default=None)
    parser.add_argument("--tokenizer-task", default=None)
    parser.add_argument("--frame-shift-ms", default=10.0, type=float)
    parser.add_argument("--vocab-size", default=None, type=int)
    parser.add_argument("--input-dim", default=80, type=int)
    parser.add_argument("--n-embd", default=512, type=int)
    parser.add_argument("--dim-att", default=512, type=int)
    parser.add_argument("--dim-ff", default=2048, type=int)
    parser.add_argument("--num-layers", default=12, type=int)
    parser.add_argument("--head-size", default=64, type=int)
    parser.add_argument("--conv-kernel-size", default=31, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--blank-id", default=0, type=int)
    parser.add_argument("--frontend-type", default="conv2d6")
    parser.add_argument("--cmvn-file", default=None)
    parser.add_argument("--cmvn-is-json", action="store_true", default=True)
    return parser


def _resolve_model_config(args: argparse.Namespace) -> RWKVCTCModelConfig:
    config_yaml = args.config_yaml
    if config_yaml is None:
        default_yaml = Path(args.checkpoint_path).resolve().parent / "model_config.yaml"
        if default_yaml.exists():
            config_yaml = str(default_yaml)

    if config_yaml is not None:
        return RWKVCTCModelConfig(**load_yaml(config_yaml))

    if args.vocab_size is None:
        raise ValueError("Either --config-yaml or --vocab-size plus model shape arguments must be provided.")
    return RWKVCTCModelConfig(
        input_dim=args.input_dim,
        n_embd=args.n_embd,
        dim_att=args.dim_att,
        dim_ff=args.dim_ff,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        head_size=args.head_size,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        blank_id=args.blank_id,
        frontend_type=args.frontend_type,
        cmvn_file=args.cmvn_file,
        cmvn_is_json=args.cmvn_is_json,
    )


def _resolve_tokenizer_config(args: argparse.Namespace) -> dict[str, str | None]:
    resolved = {
        "tokenizer_type": "whisper_multilingual",
        "tokenizer_model_path": None,
        "tokenizer_language": None,
        "tokenizer_task": None,
    }
    default_yaml = Path(args.checkpoint_path).resolve().parent / "tokenizer_config.yaml"
    if default_yaml.exists():
        config_data = load_yaml(default_yaml)
        resolved.update(
            {
                "tokenizer_type": config_data.get("tokenizer_type", resolved["tokenizer_type"]),
                "tokenizer_model_path": config_data.get("tokenizer_model_path"),
                "tokenizer_language": config_data.get("tokenizer_language"),
                "tokenizer_task": config_data.get("tokenizer_task"),
            }
        )
    for key in tuple(resolved.keys()):
        value = getattr(args, key)
        if value is not None:
            resolved[key] = value
    return resolved


def main() -> None:
    args = build_parser().parse_args()
    if (args.manifest_path is None) == (args.webdataset_root is None):
        raise ValueError("Exactly one of --manifest-path or --webdataset-root must be provided.")
    tokenizer_config = _resolve_tokenizer_config(args)

    predictions = predict_ctc(
        PredictionConfig(
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            model_config=_resolve_model_config(args),
            manifest_path=args.manifest_path,
            webdataset_root=args.webdataset_root,
            webdataset_split=args.webdataset_split,
            webdataset_eval_ratio=args.webdataset_eval_ratio,
            webdataset_hash_seed=args.webdataset_hash_seed,
            webdataset_split_by=args.webdataset_split_by,
            device=args.device,
            mode=args.mode,
            beam_size=args.beam_size,
            token_prune_topk=args.token_prune_topk,
            length_bonus=args.length_bonus,
            insertion_bonus=args.insertion_bonus,
            save_debug_lengths=args.save_debug_lengths,
            tokenizer_type=str(tokenizer_config["tokenizer_type"]),
            tokenizer_model_path=tokenizer_config["tokenizer_model_path"],
            tokenizer_language=tokenizer_config["tokenizer_language"],
            tokenizer_task=tokenizer_config["tokenizer_task"],
            num_workers=args.num_workers,
            frame_shift_ms=args.frame_shift_ms,
        )
    )

    if args.output_path is not None:
        write_predictions_jsonl(args.output_path, predictions)
        return

    for prediction in predictions:
        sys.stdout.write(
            json.dumps(
                {
                    "utt_id": prediction.utt_id,
                    "token_ids": prediction.token_ids,
                    "text": prediction.text,
                    "score": prediction.score,
                    "mode": prediction.mode,
                    "alignments": [asdict(alignment) for alignment in prediction.alignments],
                    "debug": None if prediction.debug is None else asdict(prediction.debug),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
