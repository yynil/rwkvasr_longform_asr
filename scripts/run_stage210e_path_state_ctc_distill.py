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
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage210e_path_state_ctc_distill"
DEFAULT_OUTPUT_ROOT = Path("/tmp/rwkvasr_runs")
SPLIT_BY_NAME = {split.name: split for split in SPLITS}
CURRICULUM_ORDER = ("easy", "medium", "hard", "long")
ADMISSION_STEPS = 2_000
CURRICULUM_EPOCHS = 3

PATH_STATE_PHASE = LogitAlignmentPhase(
    name="path_state",
    full_frame_filter="all",
    full_weight=1.0,
    blank_weight=0.01,
    mass_weight=0.01,
    mixer_weight=0.125,
    ffn_weight=0.0625,
    block_weight=0.25,
    lr=3.0e-7,
    epochs=CURRICULUM_EPOCHS,
    split_names=CURRICULUM_ORDER,
)


def _latest_step(output_dir: Path) -> int:
    path = output_dir / "latest_checkpoint.yaml"
    if not path.is_file():
        return 0
    state = load_yaml(path)
    return int(state.get("step") or 0) if isinstance(state, dict) else 0


def _segments(phase: str, *, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "name": f"{phase}_smoke_2steps",
                "epoch": 1,
                "split": SPLIT_BY_NAME["easy"],
                "split_steps": 2,
                "target_step": 2,
            }
        ]
    if phase == "admission":
        return [
            {
                "name": "admission_2000steps",
                "epoch": 1,
                "split": SPLIT_BY_NAME["easy"],
                "split_steps": ADMISSION_STEPS,
                "target_step": ADMISSION_STEPS,
            }
        ]

    target_step = 0
    segments: list[dict[str, Any]] = []
    for epoch in range(1, CURRICULUM_EPOCHS + 1):
        for split_name in CURRICULUM_ORDER:
            split = SPLIT_BY_NAME[split_name]
            split_steps = _split_steps(
                split,
                batch_size=TRAIN_BATCH_SIZE,
                world_size=4,
                frame_budget=TRAIN_FRAME_BUDGET,
            )
            target_step += split_steps
            segments.append(
                {
                    "name": f"curriculum_epoch{epoch}_{split_name}",
                    "epoch": epoch,
                    "split": split,
                    "split_steps": split_steps,
                    "target_step": target_step,
                }
            )
    return segments


def _default_output_dir(phase: str) -> Path:
    suffix = "admission2k" if phase == "admission" else "all118465h_3ep"
    return DEFAULT_OUTPUT_ROOT / (
        "sensevoice_rwkv_stage210e_pathstate_posteriornb4_"
        f"finalenc0p25_decoder0p25_{suffix}_lr3e7_wd0_4x4090"
    )


def _config(
    *,
    phase: str,
    segment: dict[str, Any],
    output_dir: Path,
    init_checkpoint: Path,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    config = _phase_config(
        phase=PATH_STATE_PHASE,
        segment=segment,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        resume=resume,
        smoke=smoke,
    )
    target_step = int(segment["target_step"])
    eval_interval = target_step if smoke else (1_000 if phase == "admission" else 10_000)
    config.update(
        {
            "max_steps": target_step,
            "save_every": eval_interval,
            "step_eval_every": eval_interval,
            "top_k_step_checkpoints": 1 if smoke else (2 if phase == "admission" else 4),
            "weight_decay": 0.0,
            "ctc_teacher_online_full_frame_filter": "all",
            "ctc_teacher_online_full_nonblank_weight": 4.0,
            "ctc_teacher_online_full_frame_weight_mode": "posterior_nonblank",
            "ctc_teacher_online_encoder_loss_weight": 0.25,
            "ctc_teacher_online_decoder_hidden_loss_weight": 0.25,
            "ctc_teacher_online_sequence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_presence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_hard_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_topk_loss_weight": 0.0,
            "freeze_ctc_decoder": True,
            "freeze_ctc_head": True,
            "wandb_run_name": f"{output_dir.name}_{segment['name']}",
        }
    )
    return config


def _write_config(
    *,
    phase: str,
    segment: dict[str, Any],
    config: dict[str, Any],
    config_dir: Path,
    smoke: bool,
) -> Path:
    target_dir = config_dir / ("smoke" if smoke else phase)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"stage210e_{segment['name']}.yaml"
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
        description="Run Stage210e encoder/decoder-path online Nano CTC distillation."
    )
    parser.add_argument("--phase", choices=("admission", "curriculum"), default="admission")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only-next", action="store_true")
    args = parser.parse_args()

    base_output_dir = args.output_dir or _default_output_dir(str(args.phase))
    output_dir = Path(f"{base_output_dir}_smoke") if args.smoke else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_step = _latest_step(output_dir)
    init_checkpoint = args.init_checkpoint
    if latest_step <= 0:
        if init_checkpoint is None:
            parser.error("a fresh Stage210e run requires --init-checkpoint from the admitted incumbent")
        if not init_checkpoint.is_file():
            raise FileNotFoundError(str(init_checkpoint))
    if init_checkpoint is None:
        init_checkpoint = output_dir / "resume-placeholder.pt"

    print(f"phase={args.phase}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"latest_step={latest_step}", flush=True)
    print(f"init_checkpoint={init_checkpoint}", flush=True)
    for segment in _segments(str(args.phase), smoke=bool(args.smoke)):
        target_step = int(segment["target_step"])
        print(
            f"segment={segment['name']} split={segment['split'].name} target_step={target_step}",
            flush=True,
        )
        if latest_step >= target_step:
            print(f"skip {segment['name']} target_step={target_step}", flush=True)
            continue
        resume = latest_step > 0
        config = _config(
            phase=str(args.phase),
            segment=segment,
            output_dir=output_dir,
            init_checkpoint=init_checkpoint,
            resume=resume,
            smoke=bool(args.smoke),
        )
        config_path = _write_config(
            phase=str(args.phase),
            segment=segment,
            config=config,
            config_dir=args.config_dir,
            smoke=bool(args.smoke),
        )
        code = _run(
            config_path,
            output_dir / "logs" / f"{segment['name']}.log",
            dry_run=bool(args.dry_run),
            master_port=int(args.master_port),
        )
        if code != 0 or args.dry_run:
            return code
        latest_step = _latest_step(output_dir)
        if latest_step < target_step:
            raise RuntimeError(
                f"Stage210e exited early: latest_step={latest_step} target_step={target_step}"
            )
        if args.only_next:
            break
    print(f"complete latest_step={latest_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
