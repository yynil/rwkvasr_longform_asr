from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml, save_yaml

try:
    from scripts.run_stage206_full_online_ctc_distill import SPLITS, _split_steps
    from scripts.run_stage210c_ctc_logits_distill import (
        TRAIN_BATCH_SIZE,
        TRAIN_FRAME_BUDGET,
        LogitAlignmentPhase,
        _phase_config,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage206_full_online_ctc_distill import SPLITS, _split_steps
    from run_stage210c_ctc_logits_distill import (
        TRAIN_BATCH_SIZE,
        TRAIN_FRAME_BUDGET,
        LogitAlignmentPhase,
        _phase_config,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage210d_balanced_full_ctc_distill"
DEFAULT_OUTPUT_ROOT = Path("/tmp/rwkvasr_runs")
EASY_SPLIT = next(split for split in SPLITS if split.name == "easy")
ADMISSION_STEPS = 2_000
NONBLANK_WEIGHT = 4.0

BALANCED_PHASE = LogitAlignmentPhase(
    name="balanced_full",
    full_frame_filter="all",
    full_weight=1.0,
    blank_weight=0.02,
    mass_weight=0.02,
    mixer_weight=0.25,
    ffn_weight=0.125,
    block_weight=0.50,
    lr=3.0e-7,
    epochs=1,
    split_names=("easy",),
)


def _target_steps(phase: str, *, smoke: bool) -> int:
    if smoke:
        return 2
    if phase == "admission":
        return ADMISSION_STEPS
    return _split_steps(
        EASY_SPLIT,
        batch_size=TRAIN_BATCH_SIZE,
        world_size=4,
        frame_budget=TRAIN_FRAME_BUDGET,
    )


def _default_output_dir(phase: str) -> Path:
    suffix = "admission2k" if phase == "admission" else "easy1490h_1ep"
    return DEFAULT_OUTPUT_ROOT / (
        "sensevoice_rwkv_stage210d_stage210b_step30064_"
        f"allframe_nb4_hiddenanchor_{suffix}_lr3e7_4x4090"
    )


def _latest_step(output_dir: Path) -> int:
    path = output_dir / "latest_checkpoint.yaml"
    if not path.is_file():
        return 0
    state = load_yaml(path)
    return int(state.get("step") or 0) if isinstance(state, dict) else 0


def _config(
    *,
    phase: str,
    output_dir: Path,
    init_checkpoint: Path,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    target_steps = _target_steps(phase, smoke=smoke)
    segment = {
        "name": f"{phase}_{target_steps}steps",
        "epoch": 1,
        "split": EASY_SPLIT,
        "split_steps": target_steps,
        "target_step": target_steps,
    }
    config = _phase_config(
        phase=BALANCED_PHASE,
        segment=segment,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        resume=resume,
        smoke=smoke,
    )
    eval_interval = target_steps if smoke else (1_000 if phase == "admission" else 10_000)
    config.update(
        {
            "max_steps": target_steps,
            "save_every": eval_interval,
            "step_eval_every": eval_interval,
            "top_k_step_checkpoints": 1 if smoke else (2 if phase == "admission" else 4),
            "ctc_teacher_online_full_frame_filter": "all",
            "ctc_teacher_online_full_nonblank_weight": NONBLANK_WEIGHT,
            "ctc_teacher_online_sequence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_presence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_hard_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_topk_loss_weight": 0.0,
            "wandb_run_name": f"{output_dir.name}_{phase}",
        }
    )
    return config


def _write_config(*, phase: str, config: dict[str, Any], config_dir: Path, smoke: bool) -> Path:
    target_dir = config_dir / ("smoke" if smoke else phase)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"stage210d_{phase}.yaml"
    save_yaml(path, config)
    return path


def _run(config_path: Path, log_path: Path, *, dry_run: bool, master_port: int) -> int:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "torch.distributed.run",
        "--master-port",
        str(int(master_port)),
        "--nproc_per_node",
        "4",
        "-m",
        "rwkvasr.cli.train_ctc_deepspeed",
        "--config-yaml",
        str(config_path),
    ]
    print(" ".join(command), flush=True)
    print(f"log={log_path}", flush=True)
    if dry_run:
        return 0
    environment = os.environ.copy()
    environment["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{environment.get('PATH', '')}"
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(process.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage210d balanced all-frame online Nano CTC logits alignment."
    )
    parser.add_argument("--phase", choices=("admission", "easy"), default="admission")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_output_dir = args.output_dir or _default_output_dir(str(args.phase))
    output_dir = Path(f"{base_output_dir}_smoke") if args.smoke else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_step = _latest_step(output_dir)
    init_checkpoint = args.init_checkpoint
    if latest_step <= 0:
        if init_checkpoint is None:
            parser.error("a fresh Stage210d run requires --init-checkpoint from Stage210b")
        if not init_checkpoint.is_file():
            raise FileNotFoundError(str(init_checkpoint))
    if init_checkpoint is None:
        init_checkpoint = output_dir / "resume-placeholder.pt"

    target_steps = _target_steps(str(args.phase), smoke=bool(args.smoke))
    if latest_step >= target_steps:
        print(f"already complete latest_step={latest_step} target_step={target_steps}", flush=True)
        return 0
    resume = latest_step > 0
    config = _config(
        phase=str(args.phase),
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        resume=resume,
        smoke=bool(args.smoke),
    )
    config_path = _write_config(
        phase=str(args.phase),
        config=config,
        config_dir=args.config_dir,
        smoke=bool(args.smoke),
    )
    print(f"phase={args.phase}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"latest_step={latest_step}", flush=True)
    print(f"target_step={target_steps}", flush=True)
    print(f"init_checkpoint={init_checkpoint}", flush=True)
    code = _run(
        config_path,
        output_dir / "logs" / f"{args.phase}.log",
        dry_run=bool(args.dry_run),
        master_port=int(args.master_port),
    )
    if code != 0 or args.dry_run:
        return code
    latest_step = _latest_step(output_dir)
    if latest_step < target_steps:
        raise RuntimeError(f"Stage210d exited early: latest_step={latest_step} target_step={target_steps}")
    print(f"complete latest_step={latest_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
