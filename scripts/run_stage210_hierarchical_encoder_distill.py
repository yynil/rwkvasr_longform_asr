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
STAGE209_CHECKPOINT = Path(
    "/tmp/rwkvasr_runs/"
    "sensevoice_rwkv_stage209_stage208_medium_emission_margin1_"
    "seqctc_ramp_20k_lr5e8_4x4090/step-20000.pt"
)
NANO_MODEL_DIR = Path("/media/usbhd/models/Fun-ASR-Nano-2512-modelscope")
NANO_CHECKPOINT = NANO_MODEL_DIR / "model.pt"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage210_hierarchical_encoder_distill"
RUN_PREFIX = "sensevoice_rwkv_stage210"
EASY_SPLIT = next(split for split in SPLITS if split.name == "easy")
TRAIN_BATCH_SIZE = 12
TRAIN_FRAME_BUDGET = 8_000


@dataclass(frozen=True)
class AlignmentPhase:
    name: str
    mixer_weight: float
    ffn_weight: float
    block_weight: float
    energy_mse_weight: float
    log_rms_weight: float
    raw_mse_weight: float
    lr: float
    freeze_encoder_except_time_mixer: bool
    apply_nano_init: bool
    input_mode: str


PHASES: dict[str, AlignmentPhase] = {
    "subblock": AlignmentPhase(
        name="subblock",
        mixer_weight=1.0,
        ffn_weight=0.25,
        block_weight=0.0,
        energy_mse_weight=0.25,
        log_rms_weight=0.10,
        raw_mse_weight=0.0,
        lr=1.0e-5,
        freeze_encoder_except_time_mixer=True,
        apply_nano_init=True,
        input_mode="teacher_forced",
    ),
    "block": AlignmentPhase(
        name="block",
        mixer_weight=0.5,
        ffn_weight=0.25,
        block_weight=1.0,
        energy_mse_weight=0.25,
        log_rms_weight=0.10,
        raw_mse_weight=0.0,
        lr=3.0e-6,
        freeze_encoder_except_time_mixer=False,
        apply_nano_init=False,
        input_mode="stacked",
    ),
}


def _default_output_dir(phase: AlignmentPhase) -> Path:
    suffix = (
        "stage209_nanoqkv_exactfrontend_easy1490h_1ep_hidden_localenergy_uniformeval_lr1e5_4x4090"
        if phase.name == "subblock"
        else "stage210a_easy1490h_1ep_hidden_fullblock_lr3e6_4x4090"
    )
    return Path("/tmp/rwkvasr_runs") / f"{RUN_PREFIX}_{suffix}"


def _latest_step(output_dir: Path) -> int:
    latest_path = output_dir / "latest_checkpoint.yaml"
    if not latest_path.is_file():
        return 0
    state = load_yaml(latest_path)
    return int(state.get("step") or 0) if isinstance(state, dict) else 0


def _phase_config(
    *,
    phase: AlignmentPhase,
    output_dir: Path,
    init_checkpoint: Path,
    nano_checkpoint: Path,
    resume: bool,
    smoke: bool,
    smoke_steps: int = 2,
) -> dict[str, Any]:
    config = _base_config(output_dir)
    config.update(
        {
            "output_dir": str(output_dir),
            "deepspeed": {
                "train_micro_batch_size_per_gpu": TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": 1,
                "gradient_clipping": 1.0,
                "zero_optimization": {
                    "stage": 1,
                    "offload_optimizer": {"device": "none"},
                },
                "bf16": {"enabled": True},
            },
            "webdataset_length_index_path": str(EASY_SPLIT.length_index),
            "webdataset_bucket_manifest_path": str(EASY_SPLIT.bucket_manifest),
            "feature_extractor_type": "funasr_wav_frontend",
            "dropout": 0.0,
            "batch_size": TRAIN_BATCH_SIZE,
            "batch_token_budget": TRAIN_FRAME_BUDGET,
            "length_bucket_frame_budget": TRAIN_FRAME_BUDGET,
            "max_steps": int(smoke_steps) if smoke else None,
            "epochs": None if smoke else 1,
            "init_checkpoint_path": None if resume else str(init_checkpoint),
            "resume_from": "latest" if resume else None,
            "resume_tag": None,
            "lr": float(phase.lr),
            "weight_decay": 0.1,
            "gradient_checkpointing": False,
            "save_every": 1 if smoke else 10_000,
            "step_eval_every": 1 if smoke else 10_000,
            "step_eval_samples": 4 if smoke else 256,
            "step_eval_batch_size": 1 if smoke else 4,
            "step_eval_split": "train",
            "step_eval_shuffle": False,
            "step_eval_at_start": True,
            "top_k_step_checkpoints": 2 if smoke else 4,
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
            "funasr_nano_ctc_init_checkpoint_path": (
                str(nano_checkpoint) if phase.apply_nano_init and not resume else None
            ),
            "funasr_nano_ctc_init_load_encoder": False,
            "funasr_nano_ctc_init_load_encoder_attention": False,
            "funasr_nano_ctc_init_load_rwkv_encoder_from_qkv": bool(
                phase.apply_nano_init and not resume
            ),
            "funasr_nano_ctc_init_load_decoder": bool(phase.apply_nano_init and not resume),
            "funasr_nano_ctc_init_load_head": bool(phase.apply_nano_init and not resume),
            "funasr_nano_ctc_teacher_blank_id": 60514,
            "freeze_encoder": False,
            "freeze_encoder_except_time_mixer": bool(phase.freeze_encoder_except_time_mixer),
            "freeze_ctc_decoder": True,
            "freeze_ctc_head": True,
            "ctc_loss_weight": 0.0,
            "decoder_loss_weight": 0.0,
            "encoder_anchor_loss_weight": 0.0,
            "ctc_logit_anchor_loss_weight": 0.0,
            "ctc_teacher_topk_loss_weight": 0.0,
            "ctc_teacher_topk_blank_loss_weight": 0.0,
            "ctc_teacher_topk_mass_loss_weight": 0.0,
            "ctc_teacher_online_model_path": str(NANO_MODEL_DIR),
            "ctc_teacher_online_use_batch_features": True,
            "ctc_teacher_online_keep_layer_hiddens_on_device": True,
            "ctc_teacher_online_keep_audio_cache": False,
            "ctc_teacher_online_loss_weight": 0.0,
            "ctc_teacher_online_blank_loss_weight": 0.0,
            "ctc_teacher_online_mass_loss_weight": 0.0,
            "ctc_teacher_online_full_loss_weight": 0.0,
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
            "ctc_teacher_online_layer_energy_mse_weight": float(phase.energy_mse_weight),
            "ctc_teacher_online_layer_log_rms_weight": float(phase.log_rms_weight),
            "ctc_teacher_online_layer_raw_mse_weight": float(phase.raw_mse_weight),
            "ctc_teacher_online_layer_sample_count": 8,
            "ctc_teacher_online_layer_boundary_ids": [0, 49, 50, 69],
            "ctc_teacher_online_layer_frame_tolerance": 0,
            "ctc_teacher_online_layer_input_mode": str(phase.input_mode),
            "ctc_teacher_topk_missing_policy": "error",
            "specaugment_enabled": False,
            "direction_variant": "none",
            "p_start": 0.0,
            "p_max": 0.0,
            "warmup_steps": 0,
            "ramp_steps": 0,
            "wandb_enabled": not smoke,
            "wandb_project": "rwkvasr_longform_asr_gigaspeech_wenetspeech",
            "wandb_run_name": f"{output_dir.name}_{phase.name}",
        }
    )
    return config


def _write_config(
    config_dir: Path,
    phase: AlignmentPhase,
    config: dict[str, Any],
    *,
    smoke: bool,
) -> Path:
    target_dir = config_dir / ("smoke" if smoke else "formal")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"stage210_{phase.name}.yaml"
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
        description="Run Stage210 hierarchical online Nano encoder hidden-state distillation."
    )
    parser.add_argument("--phase", choices=tuple(PHASES), default="subblock")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--nano-checkpoint", type=Path, default=NANO_CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()
    if int(args.smoke_steps) <= 0:
        parser.error("--smoke-steps must be positive")

    phase = PHASES[str(args.phase)]
    base_output_dir = args.output_dir or _default_output_dir(phase)
    output_dir = Path(f"{base_output_dir}_smoke") if args.smoke else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_step = _latest_step(output_dir)
    resume = latest_step > 0

    init_checkpoint = args.init_checkpoint
    if init_checkpoint is None and phase.name == "subblock":
        init_checkpoint = STAGE209_CHECKPOINT
    if init_checkpoint is None and not resume:
        parser.error("--phase block requires --init-checkpoint from a passing Stage210 subblock run")
    if init_checkpoint is None:
        init_checkpoint = output_dir / "resume-placeholder.pt"
    if not resume and not init_checkpoint.is_file():
        raise FileNotFoundError(str(init_checkpoint))
    if phase.apply_nano_init and not resume and not args.nano_checkpoint.is_file():
        raise FileNotFoundError(str(args.nano_checkpoint))

    config = _phase_config(
        phase=phase,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        nano_checkpoint=args.nano_checkpoint,
        resume=resume,
        smoke=bool(args.smoke),
        smoke_steps=int(args.smoke_steps),
    )
    config_path = _write_config(args.config_dir, phase, config, smoke=bool(args.smoke))
    estimated_steps = int(args.smoke_steps) if args.smoke else _split_steps(
        EASY_SPLIT,
        batch_size=TRAIN_BATCH_SIZE,
        world_size=4,
        frame_budget=TRAIN_FRAME_BUDGET,
    )
    print(f"phase={phase.name}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"init_checkpoint={init_checkpoint}", flush=True)
    print(f"resume={resume} latest_step={latest_step}", flush=True)
    print(
        f"data=easy rows=1174987 hours={EASY_SPLIT.hours:.3f} estimated_steps={estimated_steps}",
        flush=True,
    )
    print(
        f"layer_weights=mixer:{phase.mixer_weight:g},ffn:{phase.ffn_weight:g},"
        f"block:{phase.block_weight:g} lr={phase.lr:g}",
        flush=True,
    )
    code = _run(
        config_path,
        output_dir / "logs" / f"{phase.name}.log",
        dry_run=bool(args.dry_run),
        master_port=int(args.master_port),
    )
    if code != 0:
        print(f"Stage210 phase failed: {phase.name} exit={code}", file=sys.stderr, flush=True)
        return code
    if not args.dry_run:
        completed_step = _latest_step(output_dir)
        if completed_step < estimated_steps:
            raise RuntimeError(
                f"Stage210 {phase.name} exited before the full phase: "
                f"latest_step={completed_step} expected_at_least={estimated_steps}"
            )
        print(f"phase complete: {phase.name} latest_step={completed_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
