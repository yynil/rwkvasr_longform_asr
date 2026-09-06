from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rwkvasr.config import save_yaml

try:
    from scripts.run_stage210e_path_state_ctc_distill import (
        DEFAULT_OUTPUT_ROOT,
        _config as _stage210e_config,
        _latest_step,
        _run,
        _segments,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage210e_path_state_ctc_distill import (
        DEFAULT_OUTPUT_ROOT,
        _config as _stage210e_config,
        _latest_step,
        _run,
        _segments,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage210f_factorized_ctc_distill"
ADMISSION_EVAL_INTERVAL = 500


def _default_output_dir(phase: str) -> Path:
    suffix = "admission2k" if phase == "admission" else "all118465h_3ep"
    return DEFAULT_OUTPUT_ROOT / (
        "sensevoice_rwkv_stage210f_factorized_blank0p25_conditional1_"
        f"full0p25_pathstate_{suffix}_lr2e7_wd0_4x4090"
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
    config = _stage210e_config(
        phase=phase,
        segment=segment,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        resume=resume,
        smoke=smoke,
    )
    target_step = int(segment["target_step"])
    eval_interval = (
        target_step if smoke else (ADMISSION_EVAL_INTERVAL if phase == "admission" else 10_000)
    )
    config.update(
        {
            "lr": 2.0e-7,
            "weight_decay": 0.0,
            "save_every": eval_interval,
            "step_eval_every": eval_interval,
            "top_k_step_checkpoints": 1 if smoke else (4 if phase == "admission" else 4),
            "ctc_teacher_online_blank_loss_weight": 0.25,
            "ctc_teacher_online_mass_loss_weight": 0.0,
            "ctc_teacher_online_full_loss_weight": 0.25,
            "ctc_teacher_online_conditional_nonblank_loss_weight": 1.0,
            "ctc_teacher_online_full_temperature": 1.0,
            "ctc_teacher_online_full_frame_filter": "all",
            "ctc_teacher_online_full_nonblank_weight": 1.0,
            "ctc_teacher_online_full_frame_weight_mode": "posterior_nonblank",
            "ctc_teacher_frame_filter_min_nonblank_prob": 0.0,
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
    path = target_dir / f"stage210f_{segment['name']}.yaml"
    save_yaml(path, config)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage210f factorized blank/token online Nano CTC distillation."
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
            parser.error(
                "a fresh Stage210f run requires --init-checkpoint from the admitted path checkpoint"
            )
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
                f"Stage210f exited early: latest_step={latest_step} target_step={target_step}"
            )
        if args.only_next:
            break
    print(f"complete latest_step={latest_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
