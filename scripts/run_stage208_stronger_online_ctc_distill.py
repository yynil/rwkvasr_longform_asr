from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml, save_yaml
from scripts.run_stage206_full_online_ctc_distill import SPLITS, _base_config


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE206_OUTPUT_DIR = Path(
    "/tmp/rwkvasr_runs/"
    "sensevoice_rwkv_stage206_stage179b3s256_usbhd_all118465h_3ep_"
    "online_topk32_blankmass_lr5e8_eval10k_dsresume_4x4090"
)
RUN_NAME = (
    "sensevoice_rwkv_stage208_stage206_latest_medium_stronger_"
    "seqctc_nonblankwin_ramp_2k_lr5e8_4x4090"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/rwkvasr_runs") / RUN_NAME
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage208_stronger_online_ctc_distill"
MEDIUM_SPLIT = next(split for split in SPLITS if split.name == "medium")


@dataclass(frozen=True)
class RampPhase:
    name: str
    target_step: int
    sequence_weight: float
    nonblank_window_weight: float
    nonblank_window_topk_weight: float
    nonblank_window_margin_weight: float


RAMP_PHASES: tuple[RampPhase, ...] = (
    RampPhase("ramp1", 500, 0.05, 0.05, 0.10, 0.01),
    RampPhase("ramp2", 1000, 0.10, 0.10, 0.25, 0.02),
    RampPhase("ramp3", 1500, 0.175, 0.175, 0.375, 0.035),
    RampPhase("ramp4", 2000, 0.25, 0.25, 0.50, 0.05),
)


def _latest_exported_checkpoint(stage206_output_dir: Path) -> Path:
    latest_path = stage206_output_dir / "latest_checkpoint.yaml"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Stage206 latest checkpoint metadata not found: {latest_path}")
    state = load_yaml(latest_path)
    checkpoint_path = state.get("checkpoint_path") if isinstance(state, dict) else None
    if not checkpoint_path:
        raise ValueError(f"Stage206 latest checkpoint metadata has no checkpoint_path: {latest_path}")
    checkpoint = Path(str(checkpoint_path))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage206 exported checkpoint not found: {checkpoint}")
    return checkpoint


def _latest_step(output_dir: Path) -> int:
    latest_path = output_dir / "latest_checkpoint.yaml"
    if not latest_path.is_file():
        return 0
    state = load_yaml(latest_path)
    return int(state.get("step") or 0) if isinstance(state, dict) else 0


def _phase_config(
    *,
    phase: RampPhase,
    output_dir: Path,
    init_checkpoint: Path,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    config = _base_config(output_dir)
    config.update(
        {
            "webdataset_length_index_path": str(MEDIUM_SPLIT.length_index),
            "webdataset_bucket_manifest_path": str(MEDIUM_SPLIT.bucket_manifest),
            "batch_token_budget": 8000,
            "length_bucket_frame_budget": 8000,
            "max_steps": 2 if smoke else int(phase.target_step),
            "epochs": None,
            "init_checkpoint_path": None if resume else str(init_checkpoint),
            "resume_from": "latest" if resume else None,
            "resume_tag": None,
            "save_every": 1 if smoke else 500,
            "step_eval_every": 1 if smoke else 500,
            "step_eval_samples": 4 if smoke else 256,
            "top_k_step_checkpoints": 2 if smoke else 4,
            "save_deepspeed_sharded_checkpoints": True,
            "ctc_teacher_online_loss_weight": 2.0,
            "ctc_teacher_online_blank_loss_weight": 0.02,
            "ctc_teacher_online_mass_loss_weight": 0.02,
            "ctc_teacher_online_full_loss_weight": 0.0,
            "ctc_teacher_online_encoder_loss_weight": 0.0,
            "ctc_teacher_online_sequence_loss_weight": float(phase.sequence_weight),
            "ctc_teacher_online_sequence_presence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_hard_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_margin": 0.5,
            "ctc_teacher_online_nonblank_window_loss_weight": float(
                phase.nonblank_window_weight
            ),
            "ctc_teacher_online_nonblank_window_margin_loss_weight": float(
                phase.nonblank_window_margin_weight
            ),
            "ctc_teacher_online_nonblank_window_topk_loss_weight": float(
                phase.nonblank_window_topk_weight
            ),
            "ctc_teacher_online_nonblank_window_radius": 2,
            "ctc_teacher_online_nonblank_window_temperature": 0.2,
            "wandb_run_name": f"{RUN_NAME}_{phase.name}",
        }
    )
    if smoke:
        config.update(
            {
                "output_dir": f"{output_dir}_smoke",
                "num_workers": 2,
                "decoded_batch_prefetch": 1,
            }
        )
    return config


def _write_config(config_dir: Path, phase: RampPhase, config: dict[str, Any], *, smoke: bool) -> Path:
    target_dir = config_dir / ("smoke" if smoke else "admission")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"stage208_{phase.name}.yaml"
    save_yaml(path, config)
    return path


def _run_segment(config_path: Path, log_path: Path, *, dry_run: bool) -> int:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "torch.distributed.run",
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
    parser = argparse.ArgumentParser(description="Run Stage208 stronger online Nano-CTC distillation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--stage206-output-dir", type=Path, default=STAGE206_OUTPUT_DIR)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only-next", action="store_true")
    args = parser.parse_args()

    init_checkpoint = args.init_checkpoint or _latest_exported_checkpoint(args.stage206_output_dir)
    if not init_checkpoint.is_file():
        raise FileNotFoundError(str(init_checkpoint))
    output_dir = Path(f"{args.output_dir}_smoke") if args.smoke else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = RAMP_PHASES[:1] if args.smoke else RAMP_PHASES
    latest_step = _latest_step(output_dir)
    print(f"output_dir={output_dir}", flush=True)
    print(f"init_checkpoint={init_checkpoint}", flush=True)
    print(f"latest_step={latest_step}", flush=True)

    for phase in phases:
        target_step = 2 if args.smoke else phase.target_step
        if latest_step >= target_step:
            print(f"skip {phase.name} target_step={target_step} latest_step={latest_step}", flush=True)
            continue
        resume = latest_step > 0
        config = _phase_config(
            phase=phase,
            output_dir=args.output_dir,
            init_checkpoint=init_checkpoint,
            resume=resume,
            smoke=bool(args.smoke),
        )
        config_path = _write_config(args.config_dir, phase, config, smoke=bool(args.smoke))
        log_path = output_dir / "logs" / f"{phase.name}.log"
        print(
            f"run {phase.name} target_step={target_step} resume={resume} "
            f"sequence={phase.sequence_weight:g} "
            f"nonblank_window={phase.nonblank_window_weight:g} "
            f"nonblank_window_topk={phase.nonblank_window_topk_weight:g} "
            f"nonblank_window_margin={phase.nonblank_window_margin_weight:g}",
            flush=True,
        )
        code = _run_segment(config_path, log_path, dry_run=bool(args.dry_run))
        if code != 0:
            print(f"phase failed: {phase.name} exit={code}", file=sys.stderr, flush=True)
            return code
        if args.dry_run:
            latest_step = target_step
        else:
            latest_step = _latest_step(output_dir)
            if latest_step < target_step:
                raise RuntimeError(
                    f"Phase {phase.name} exited without reaching target step: "
                    f"latest_step={latest_step} target_step={target_step}"
                )
        print(f"phase complete: {phase.name} latest_step={latest_step}", flush=True)
        if args.only_next:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
