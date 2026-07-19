from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml, save_yaml

try:
    from scripts.run_stage206_full_online_ctc_distill import SPLITS, _base_config, _split_steps
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage206_full_online_ctc_distill import SPLITS, _base_config, _split_steps


REPO_ROOT = Path(__file__).resolve().parents[1]
NANO_MODEL_DIR = Path("/media/usbhd/models/Fun-ASR-Nano-2512-modelscope")
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage210c_ctc_logits_distill"
RUN_PREFIX = "sensevoice_rwkv_stage210c"
TRAIN_BATCH_SIZE = 12
TRAIN_FRAME_BUDGET = 8_000
SPLIT_BY_NAME = {split.name: split for split in SPLITS}
CURRICULUM_SPLITS = tuple(SPLIT_BY_NAME[name] for name in ("easy", "medium", "hard", "long"))


@dataclass(frozen=True)
class LogitAlignmentPhase:
    name: str
    full_frame_filter: str
    full_weight: float
    blank_weight: float
    mass_weight: float
    mixer_weight: float
    ffn_weight: float
    block_weight: float
    lr: float
    epochs: int
    split_names: tuple[str, ...]


PHASES: dict[str, LogitAlignmentPhase] = {
    "nonblank": LogitAlignmentPhase(
        name="nonblank",
        full_frame_filter="nonblank_neighbors",
        full_weight=1.0,
        blank_weight=0.05,
        mass_weight=0.05,
        mixer_weight=0.10,
        ffn_weight=0.05,
        block_weight=0.25,
        lr=1.0e-6,
        epochs=1,
        split_names=("easy",),
    ),
    "full": LogitAlignmentPhase(
        name="full",
        full_frame_filter="all",
        full_weight=1.0,
        blank_weight=0.0,
        mass_weight=0.0,
        mixer_weight=0.04,
        ffn_weight=0.02,
        block_weight=0.10,
        lr=5.0e-7,
        epochs=3,
        split_names=("easy", "medium", "hard", "long"),
    ),
}


def _default_output_dir(phase: LogitAlignmentPhase) -> Path:
    suffix = (
        "nonblank_easy1490h_1ep_fullkl_hiddenanchor_4x4090"
        if phase.name == "nonblank"
        else "full_all118465h_3ep_fullkl_hiddenanchor_4x4090"
    )
    return Path("/tmp/rwkvasr_runs") / f"{RUN_PREFIX}_{suffix}"


def _latest_step(output_dir: Path) -> int:
    latest_path = output_dir / "latest_checkpoint.yaml"
    if not latest_path.is_file():
        return 0
    state = load_yaml(latest_path)
    return int(state.get("step") or 0) if isinstance(state, dict) else 0


def _segments(phase: LogitAlignmentPhase, *, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "name": f"{phase.name}_smoke_2steps",
                "epoch": 1,
                "split": SPLIT_BY_NAME["easy"],
                "split_steps": 2,
                "target_step": 2,
            }
        ]

    target_step = 0
    segments: list[dict[str, Any]] = []
    selected_splits = tuple(SPLIT_BY_NAME[name] for name in phase.split_names)
    for epoch in range(1, int(phase.epochs) + 1):
        for split in selected_splits:
            split_steps = _split_steps(
                split,
                batch_size=TRAIN_BATCH_SIZE,
                world_size=4,
                frame_budget=TRAIN_FRAME_BUDGET,
            )
            target_step += split_steps
            segments.append(
                {
                    "name": f"{phase.name}_epoch{epoch}_{split.name}",
                    "epoch": epoch,
                    "split": split,
                    "split_steps": split_steps,
                    "target_step": target_step,
                }
            )
    return segments


def _phase_config(
    *,
    phase: LogitAlignmentPhase,
    segment: dict[str, Any],
    output_dir: Path,
    init_checkpoint: Path,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    split = segment["split"]
    config = _base_config(output_dir)
    config.update(
        {
            "output_dir": str(output_dir),
            "deepspeed": {
                "train_micro_batch_size_per_gpu": 2 if smoke else TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": 1,
                "gradient_clipping": 1.0,
                "zero_optimization": {
                    "stage": 1,
                    "offload_optimizer": {"device": "none"},
                },
                "bf16": {"enabled": True},
            },
            "webdataset_length_index_path": str(split.length_index),
            "webdataset_bucket_manifest_path": str(split.bucket_manifest),
            "feature_extractor_type": "funasr_wav_frontend",
            "dropout": 0.0,
            "batch_size": 2 if smoke else TRAIN_BATCH_SIZE,
            "batch_token_budget": 800 if smoke else TRAIN_FRAME_BUDGET,
            "length_bucket_frame_budget": 800 if smoke else TRAIN_FRAME_BUDGET,
            "max_steps": int(segment["target_step"]),
            "epochs": None,
            "init_checkpoint_path": None if resume else str(init_checkpoint),
            "resume_from": "latest" if resume else None,
            "resume_tag": None,
            "lr": float(phase.lr),
            "weight_decay": 0.1,
            "gradient_checkpointing": False,
            "save_every": int(segment["target_step"]) if smoke else 10_000,
            "step_eval_every": int(segment["target_step"]) if smoke else 10_000,
            "step_eval_samples": 4 if smoke else 256,
            "step_eval_batch_size": 1 if smoke else 4,
            "step_eval_split": "train",
            "step_eval_shuffle": False,
            "step_eval_at_start": not resume,
            "top_k_step_checkpoints": 1 if smoke else 4,
            "save_deepspeed_sharded_checkpoints": not smoke,
            "num_workers": 2 if smoke else 8,
            "decoded_batch_prefetch": 1 if smoke else 2,
            "ctc_decoder_type": "funasr_nano_transformer",
            "ctc_decoder_downsample_rate": 1,
            "ctc_decoder_dim": 512,
            "ctc_decoder_ffn_dim": 2048,
            "ctc_decoder_num_layers": 5,
            "ctc_decoder_attention_heads": 8,
            "ctc_decoder_dropout": 0.0,
            "ctc_decoder_attention_dropout": 0.0,
            "funasr_nano_ctc_init_checkpoint_path": None,
            "funasr_nano_ctc_init_load_encoder": False,
            "funasr_nano_ctc_init_load_encoder_attention": False,
            "funasr_nano_ctc_init_load_rwkv_encoder_from_qkv": False,
            "funasr_nano_ctc_init_load_decoder": False,
            "funasr_nano_ctc_init_load_head": False,
            "funasr_nano_ctc_teacher_blank_id": 60514,
            "freeze_encoder": False,
            "freeze_encoder_except_time_mixer": False,
            "freeze_ctc_decoder": True,
            "freeze_ctc_head": True,
            "ctc_loss_weight": 0.0,
            "decoder_loss_weight": 0.0,
            "encoder_anchor_loss_weight": 0.0,
            "ctc_logit_anchor_loss_weight": 0.0,
            "ctc_teacher_topk_loss_weight": 0.0,
            "ctc_teacher_topk_blank_loss_weight": 0.0,
            "ctc_teacher_topk_mass_loss_weight": 0.0,
            "ctc_teacher_topk_time_map": "nearest",
            "ctc_teacher_frame_filter": "nonblank_neighbors",
            "ctc_teacher_frame_filter_neighbor_radius": 1,
            "ctc_teacher_frame_filter_min_nonblank_prob": 0.0,
            "ctc_teacher_topk_missing_policy": "error",
            "ctc_teacher_online_model_path": str(NANO_MODEL_DIR),
            "ctc_teacher_online_use_batch_features": True,
            "ctc_teacher_online_keep_layer_hiddens_on_device": True,
            "ctc_teacher_online_keep_full_log_probs_on_device": True,
            "ctc_teacher_online_keep_audio_cache": False,
            "ctc_teacher_online_loss_weight": 0.0,
            "ctc_teacher_online_blank_loss_weight": float(phase.blank_weight),
            "ctc_teacher_online_mass_loss_weight": float(phase.mass_weight),
            "ctc_teacher_online_full_loss_weight": float(phase.full_weight),
            "ctc_teacher_online_full_temperature": 1.0,
            "ctc_teacher_online_full_frame_filter": str(phase.full_frame_filter),
            "ctc_teacher_online_encoder_loss_weight": 0.0,
            "ctc_teacher_online_sequence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_presence_loss_weight": 0.0,
            "ctc_teacher_online_sequence_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_hard_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_margin_loss_weight": 0.0,
            "ctc_teacher_online_nonblank_window_topk_loss_weight": 0.0,
            "ctc_teacher_online_layer_mixer_loss_weight": float(phase.mixer_weight),
            "ctc_teacher_online_layer_ffn_loss_weight": float(phase.ffn_weight),
            "ctc_teacher_online_layer_block_loss_weight": float(phase.block_weight),
            "ctc_teacher_online_layer_normalized_mse_weight": 1.0,
            "ctc_teacher_online_layer_cosine_weight": 0.25,
            "ctc_teacher_online_layer_energy_mse_weight": 0.25,
            "ctc_teacher_online_layer_log_rms_weight": 0.10,
            "ctc_teacher_online_layer_raw_mse_weight": 0.0,
            "ctc_teacher_online_layer_sample_count": 8,
            "ctc_teacher_online_layer_boundary_ids": [0, 49, 50, 69],
            "ctc_teacher_online_layer_frame_tolerance": 0,
            "ctc_teacher_online_layer_input_mode": "stacked",
            "ctc_teacher_online_top_k": 32,
            "ctc_teacher_online_project_ignored_token_ids": [60514],
            "specaugment_enabled": False,
            "direction_variant": "none",
            "p_start": 0.0,
            "p_max": 0.0,
            "warmup_steps": 0,
            "ramp_steps": 0,
            "wandb_enabled": not smoke,
            "wandb_project": "rwkvasr_longform_asr_gigaspeech_wenetspeech",
            "wandb_run_name": f"{output_dir.name}_{segment['name']}",
        }
    )
    return config


def _write_config(
    *,
    phase: LogitAlignmentPhase,
    segment: dict[str, Any],
    config: dict[str, Any],
    config_dir: Path,
    smoke: bool,
) -> Path:
    target_dir = config_dir / ("smoke" if smoke else phase.name)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"stage210c_{segment['name']}.yaml"
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
        description="Run Stage210c online full-vocabulary Nano CTC logits alignment."
    )
    parser.add_argument("--phase", choices=tuple(PHASES), default="nonblank")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only-next", action="store_true")
    args = parser.parse_args()

    phase = PHASES[str(args.phase)]
    base_output_dir = args.output_dir or _default_output_dir(phase)
    output_dir = Path(f"{base_output_dir}_smoke") if args.smoke else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_step = _latest_step(output_dir)
    init_checkpoint = args.init_checkpoint
    if latest_step <= 0:
        if init_checkpoint is None:
            parser.error("a fresh Stage210c run requires --init-checkpoint from a passing hidden stage")
        if not init_checkpoint.is_file():
            raise FileNotFoundError(str(init_checkpoint))
    if init_checkpoint is None:
        init_checkpoint = output_dir / "resume-placeholder.pt"

    segments = _segments(phase, smoke=bool(args.smoke))
    print(f"phase={phase.name}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"latest_step={latest_step}", flush=True)
    print(f"init_checkpoint={init_checkpoint}", flush=True)
    for segment in segments:
        target_step = int(segment["target_step"])
        split = segment["split"]
        if latest_step >= target_step:
            print(f"skip {segment['name']} target_step={target_step} latest_step={latest_step}", flush=True)
            continue
        resume = latest_step > 0
        config = _phase_config(
            phase=phase,
            segment=segment,
            output_dir=output_dir,
            init_checkpoint=init_checkpoint,
            resume=resume,
            smoke=bool(args.smoke),
        )
        config_path = _write_config(
            phase=phase,
            segment=segment,
            config=config,
            config_dir=args.config_dir,
            smoke=bool(args.smoke),
        )
        print(
            f"run {segment['name']} split={split.name} hours={split.hours:.3f} "
            f"split_steps={segment['split_steps']} target_step={target_step} resume={resume}",
            flush=True,
        )
        code = _run(
            config_path,
            output_dir / "logs" / f"{segment['name']}.log",
            dry_run=bool(args.dry_run),
            master_port=int(args.master_port),
        )
        if code != 0:
            print(f"Stage210c segment failed: {segment['name']} exit={code}", file=sys.stderr, flush=True)
            return code
        if args.dry_run:
            if args.only_next:
                break
            continue
        latest_step = _latest_step(output_dir)
        if latest_step < target_step:
            raise RuntimeError(
                f"Stage210c segment exited early: latest_step={latest_step} target_step={target_step}"
            )
        print(f"segment complete: {segment['name']} latest_step={latest_step}", flush=True)
        if args.only_next:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
