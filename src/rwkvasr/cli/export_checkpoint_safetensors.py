from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rwkvasr.training.checkpoint import export_checkpoint_to_safetensors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a .pt RWKV checkpoint to inference-friendly safetensors.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--copy-model-config", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)
    result = export_checkpoint_to_safetensors(checkpoint_path, output_path)

    if args.copy_model_config:
        source_config = checkpoint_path.resolve().parent / "model_config.yaml"
        if source_config.exists():
            target_config = output_path.resolve().parent / "model_config.yaml"
            if source_config != target_config:
                shutil.copy2(source_config, target_config)
        source_tokenizer = checkpoint_path.resolve().parent / "tokenizer_config.yaml"
        if source_tokenizer.exists():
            target_tokenizer = output_path.resolve().parent / "tokenizer_config.yaml"
            if source_tokenizer != target_tokenizer:
                shutil.copy2(source_tokenizer, target_tokenizer)

    print(
        "export_checkpoint_safetensors "
        f"output={output_path} step={result['step']} tensors={result['num_tensors']}"
    )


if __name__ == "__main__":
    main()
