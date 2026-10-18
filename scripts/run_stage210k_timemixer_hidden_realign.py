from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rwkvasr.config import save_yaml

try:
    from scripts.run_stage210j_decoupled_hidden_ctc_distill import (
        DEFAULT_BUCKET_MANIFEST,
        DEFAULT_OUTPUT_ROOT,
        _config as _stage210j_config,
        _latest_step,
        _run,
        _segments as _stage210j_segments,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage210j_decoupled_hidden_ctc_distill import (
        DEFAULT_BUCKET_MANIFEST,
        DEFAULT_OUTPUT_ROOT,
        _config as _stage210j_config,
        _latest_step,
        _run,
        _segments as _stage210j_segments,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage210k_timemixer_hidden_realign"
ADMISSION_STEPS = 1_000
EVAL_INTERVAL = 250


def _default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / (
        "sensevoice_rwkv_stage210k_sourcebalanced_easy_timemixeronly_"
        "teacherforced_hidden8_admission1k_lr3e6_wd0_4x4090"
    )


def _segments(*, smoke: bool) -> list[dict[str, Any]]:
    source = _stage210j_segments(smoke=bool(smoke))[0]
    target_step = 2 if smoke else ADMISSION_STEPS
    return [
        {
            **source,
            "name": "timemixer_hidden_smoke_2steps" if smoke else "timemixer_hidden_1000steps",
            "split_steps": target_step,
            "target_step": target_step,
        }
    ]


def _config(
    *,
    segment: dict[str, Any],
    output_dir: Path,
    init_checkpoint: Path,
    bucket_manifest: Path,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    config = _stage210j_config(
        segment=segment,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        bucket_manifest=bucket_manifest,
        resume=resume,
        smoke=smoke,
    )
    target_step = int(segment["target_step"])
    eval_interval = target_step if smoke else EVAL_INTERVAL
    config.update(
        {
            "max_steps": target_step,
            "lr": 3.0e-6,
            "weight_decay": 0.0,
            "save_every": eval_interval,
            "step_eval_every": eval_interval,
            "top_k_step_checkpoints": 1 if smoke else 4,
            "freeze_encoder": False,
            "freeze_encoder_except_time_mixer": True,
            "freeze_ctc_decoder": True,
            "freeze_ctc_head": True,
            "ctc_teacher_online_loss_weight": 0.0,
            "ctc_teacher_online_blank_loss_weight": 0.0,
            "ctc_teacher_online_mass_loss_weight": 0.0,
            "ctc_teacher_online_full_loss_weight": 0.0,
            "ctc_teacher_online_conditional_nonblank_loss_weight": 0.0,
            "ctc_teacher_online_conditional_nonblank_hard_loss_weight": 0.0,
            "ctc_teacher_online_encoder_loss_weight": 0.0,
            "ctc_teacher_online_decoder_hidden_loss_weight": 0.0,
            "ctc_teacher_online_sequence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_presence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_hard_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_topk_loss_weight": 0.0,
            "ctc_teacher_online_layer_mixer_loss_weight": 1.0,
            "ctc_teacher_online_layer_ffn_loss_weight": 0.25,
            "ctc_teacher_online_layer_block_loss_weight": 0.0,
            "ctc_teacher_online_layer_sample_count": 8,
            "ctc_teacher_online_layer_include_boundaries": False,
            "ctc_teacher_online_layer_input_mode": "teacher_forced",
            "ctc_teacher_online_frame_balance_mode": "all",
            "ctc_teacher_online_blank_frame_balance_mode": "all",
            "ctc_teacher_online_hidden_frame_balance_mode": "all",
            "ctc_teacher_online_keep_full_log_probs_on_device": False,
            "step_eval_cache_batches": True,
            "wandb_run_name": f"{output_dir.name}_{segment['name']}",
        }
    )
    return config


def _write_config(
    *,
    segment: dict[str, Any],
    config: dict[str, Any],
    config_dir: Path,
    smoke: bool,
) -> Path:
    target_dir = config_dir / ("smoke" if smoke else "admission")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"stage210k_{segment['name']}.yaml"
    save_yaml(path, config)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Realign BiRWKV TimeMixer blocks to Nano hidden states with teacher-forced inputs."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--bucket-manifest", type=Path, default=DEFAULT_BUCKET_MANIFEST)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_output_dir = args.output_dir or _default_output_dir()
    output_dir = Path(f"{base_output_dir}_smoke") if args.smoke else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_step = _latest_step(output_dir)
    init_checkpoint = args.init_checkpoint
    if not args.bucket_manifest.is_file():
        raise FileNotFoundError(str(args.bucket_manifest))
    if latest_step <= 0:
        if init_checkpoint is None:
            parser.error("a fresh Stage210k run requires an explicitly selected initialization checkpoint")
        if not init_checkpoint.is_file():
            raise FileNotFoundError(str(init_checkpoint))
    if init_checkpoint is None:
        init_checkpoint = output_dir / "resume-placeholder.pt"

    print("phase=timemixer_hidden", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"bucket_manifest={args.bucket_manifest}", flush=True)
    print(f"latest_step={latest_step}", flush=True)
    print(f"init_checkpoint={init_checkpoint}", flush=True)
    for segment in _segments(smoke=bool(args.smoke)):
        target_step = int(segment["target_step"])
        print(
            f"segment={segment['name']} split={segment['split'].name} target_step={target_step}",
            flush=True,
        )
        if latest_step >= target_step:
            print(f"skip {segment['name']} target_step={target_step}", flush=True)
            continue
        config = _config(
            segment=segment,
            output_dir=output_dir,
            init_checkpoint=init_checkpoint,
            bucket_manifest=args.bucket_manifest,
            resume=latest_step > 0,
            smoke=bool(args.smoke),
        )
        config_path = _write_config(
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
                f"Stage210k exited early: latest_step={latest_step} target_step={target_step}"
            )
    print(f"complete latest_step={latest_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
