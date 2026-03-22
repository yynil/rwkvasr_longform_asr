from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch paper-style RWKV ASR training with DeepSpeed."
    )
    parser.add_argument(
        "--mode",
        choices=("bi_baseline", "dirdrop_both"),
        default="dirdrop_both",
    )
    parser.add_argument("--config-yaml", default=None)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-tag", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_default_config(repo_root: Path, mode: str) -> Path:
    config_map = {
        "bi_baseline": repo_root / "configs" / "paper_bi_baseline_4x4090_deepspeed.yaml",
        "dirdrop_both": repo_root / "configs" / "paper_dirdrop_both_4x4090_deepspeed.yaml",
    }
    return config_map[mode]


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str], Path]:
    repo_root = resolve_repo_root()
    torchrun_bin = repo_root / ".venv" / "bin" / "torchrun"
    if not torchrun_bin.exists():
        raise FileNotFoundError(f"torchrun executable not found: {torchrun_bin}")

    config_path = Path(args.config_yaml) if args.config_yaml is not None else resolve_default_config(repo_root, args.mode)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    cmd = [
        str(torchrun_bin),
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(args.num_gpus),
        "--master_port",
        str(args.master_port),
        "--module",
        "rwkvasr.cli.train_ctc_deepspeed",
        "--config-yaml",
        str(config_path),
    ]

    optional_args = (
        ("output_dir", "--output-dir"),
        ("batch_size", "--batch-size"),
        ("epochs", "--epochs"),
        ("max_steps", "--max-steps"),
        ("num_workers", "--num-workers"),
        ("resume_from", "--resume-from"),
        ("resume_tag", "--resume-tag"),
    )
    for attr_name, cli_name in optional_args:
        value = getattr(args, attr_name)
        if value is not None:
            cmd.extend([cli_name, str(value)])

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current_pythonpath else f"{src_path}:{current_pythonpath}"
    return cmd, env, repo_root


def main() -> None:
    args = build_parser().parse_args()
    cmd, env, cwd = build_command(args)
    print("Launch command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
