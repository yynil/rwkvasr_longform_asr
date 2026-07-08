from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.data import estimate_bucket_manifest_steps, load_webdataset_bucket_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    "/media/usbhd/training_data/asr/curriculum/"
    "stage22_stage21_soup_clean_repair_mix/stages/stage179_usbhd_dedup_online_ctc_alignment"
)
RUN_NAME = (
    "sensevoice_rwkv_stage206_stage179b3s256_usbhd_all118465h_3ep_"
    "online_topk32_blankmass_lr5e8_eval10k_dsresume_4x4090"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/rwkvasr_runs") / RUN_NAME
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage206_full_online_ctc_distill"
INIT_CHECKPOINT = Path(
    "/media/usbhd/rwkvasr_runs/"
    "sensevoice_rwkv_stage179b3_stage179a768_usbhd_medium47771h_audio_only_online_funasr_nano_ctc_"
    "nonblanknbr_topk2_blank0_fullkl0p1_bs12_lr1e7_zero1_nockpt_noaug_nodirdrop_4x4090/"
    "step-256.pt"
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    hours: float
    length_index: Path
    bucket_manifest: Path


SPLITS: tuple[SplitSpec, ...] = (
    SplitSpec(
        "hard",
        69200.368,
        DATA_ROOT / "stage179c_hard_dedup_audio_only_online_ctc" / "webdataset_lengths.jsonl",
        DATA_ROOT
        / "stage179c_hard_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest.json",
    ),
    SplitSpec(
        "medium",
        47771.242,
        DATA_ROOT / "stage179b_medium_dedup_audio_only_online_ctc" / "webdataset_lengths.jsonl",
        DATA_ROOT
        / "stage179b_medium_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest.json",
    ),
    SplitSpec(
        "easy",
        1490.119,
        DATA_ROOT / "stage179a_easy_dedup_audio_only_online_ctc" / "webdataset_lengths.jsonl",
        DATA_ROOT
        / "stage179a_easy_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest.json",
    ),
    SplitSpec(
        "long",
        3.432,
        DATA_ROOT / "stage179d_long_dedup_audio_only_online_ctc" / "webdataset_lengths.jsonl",
        DATA_ROOT
        / "stage179d_long_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest.json",
    ),
)


def _base_config(output_dir: Path) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "deepspeed": {
            "train_micro_batch_size_per_gpu": 12,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {"device": "none"},
            },
            "bf16": {"enabled": True},
        },
        "vocab_size": 60515,
        "tokenizer_type": "sensevoice_tiktoken",
        "tokenizer_model_path": "assets/fun-asr-nano-2512/multilingual.tiktoken",
        "tokenizer_append_eos": False,
        "text_normalization": "ctc",
        "allow_missing_targets": True,
        "webdataset_root": "/",
        "webdataset_index_path": None,
        "webdataset_split": "train",
        "webdataset_eval_ratio": 0.0,
        "webdataset_hash_seed": 0,
        "webdataset_split_by": "sample_id",
        "webdataset_utt_id_key": "id",
        "feature_extractor_type": "sensevoice_lfr_fbank",
        "input_dim": 560,
        "n_embd": 512,
        "dim_att": 512,
        "dim_ff": 2048,
        "num_layers": 70,
        "head_size": 64,
        "backend": "cuda_clampw",
        "conv_kernel_size": 31,
        "dropout": 0.1,
        "frontend_type": "sensevoice_rwkv",
        "encoder_output_dim": 512,
        "sensevoice_tp_blocks": 20,
        "blank_id": 60515,
        "batch_size": 12,
        "save_every": 10000,
        "num_workers": 8,
        "decoded_batch_prefetch": 2,
        "max_open_shards_per_worker": 8,
        "bucket_source_interleave": False,
        "lr": 5.0e-8,
        "weight_decay": 0.1,
        "decoder_enabled": False,
        "ctc_label_override_cache_path": None,
        "ctc_label_override_text_key": "teacher_text_tn",
        "ctc_loss_weight": 0.0,
        "decoder_loss_weight": 0.0,
        "encoder_anchor_checkpoint_path": None,
        "encoder_anchor_loss_weight": 0.0,
        "ctc_logit_anchor_loss_weight": 0.0,
        "ctc_logit_anchor_chunk_frames": 16,
        "ctc_teacher_topk_cache_path": None,
        "ctc_teacher_topk_loss_weight": 0.0,
        "ctc_teacher_topk_blank_loss_weight": 0.0,
        "ctc_teacher_topk_mass_loss_weight": 0.0,
        "ctc_teacher_topk_time_map": "nearest",
        "ctc_teacher_frame_filter": "nonblank_neighbors",
        "ctc_teacher_frame_filter_neighbor_radius": 1,
        "ctc_teacher_frame_filter_min_nonblank_prob": 0.0,
        "ctc_teacher_topk_missing_policy": "error",
        "ctc_teacher_online_model_path": "/media/usbhd/models/Fun-ASR-Nano-2512-modelscope",
        "ctc_teacher_online_loss_weight": 2.0,
        "ctc_teacher_online_blank_loss_weight": 0.05,
        "ctc_teacher_online_mass_loss_weight": 0.05,
        "ctc_teacher_online_full_loss_weight": 0.0,
        "ctc_teacher_online_full_temperature": 1.0,
        "ctc_teacher_online_full_frame_filter": None,
        "ctc_teacher_online_sequence_loss_weight": 0.0,
        "ctc_teacher_online_top_k": 32,
        "ctc_teacher_online_audio_index_path": None,
        "ctc_teacher_online_webdataset_index_path": None,
        "ctc_teacher_online_audio_cache_dir": None,
        "ctc_teacher_online_keep_audio_cache": False,
        "ctc_teacher_online_device": None,
        "ctc_teacher_online_project_ignored_token_ids": [60514],
        "freeze_encoder": False,
        "freeze_ctc_head": False,
        "direction_variant": "none",
        "p_start": 0.0,
        "p_max": 0.0,
        "warmup_steps": 0,
        "ramp_steps": 0,
        "device": "cuda",
        "encoder_init_checkpoint_path": None,
        "init_checkpoint_path": str(INIT_CHECKPOINT),
        "resume_from": None,
        "resume_tag": None,
        "wandb_enabled": False,
        "wandb_project": "rwkvasr_longform_asr_gigaspeech_wenetspeech",
        "wandb_run_name": RUN_NAME,
        "eval_mode": "bi",
        "max_eval_samples": 0,
        "eval_batch_size": 1,
        "step_eval_batch_size": 1,
        "step_eval_every": 10000,
        "step_eval_samples": 256,
        "step_eval_split": "train",
        "top_k_step_checkpoints": 3,
        "save_deepspeed_sharded_checkpoints": True,
        "log_every": 10,
        "gradient_checkpointing": False,
        "batch_token_budget": 12000,
        "length_bucket_frame_budget": 12000,
        "target_gpu_memory_gib": 23.0,
        "skip_oversized_samples": True,
        "specaugment_enabled": False,
        "specaugment_time_masks": 2,
        "specaugment_time_width": 20,
        "specaugment_freq_masks": 2,
        "specaugment_freq_width": 27,
    }


def _split_steps(split: SplitSpec, *, batch_size: int, world_size: int, frame_budget: int) -> int:
    manifest = load_webdataset_bucket_manifest(split.bucket_manifest)
    return estimate_bucket_manifest_steps(
        manifest,
        split="train",
        batch_size=batch_size,
        world_size=world_size,
        frame_budget=frame_budget,
        drop_last=True,
    )


def _latest_step(output_dir: Path) -> int:
    latest = output_dir / "latest_checkpoint.yaml"
    if not latest.is_file():
        return 0
    state = load_yaml(latest)
    if not isinstance(state, dict):
        return 0
    return int(state.get("step") or 0)


def _segments(*, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        split = SPLITS[2]
        return [
            {
                "epoch": 1,
                "split": split,
                "split_steps": 2,
                "target_step": 2,
                "name": "smoke_easy_2steps",
            }
        ]

    target_step = 0
    segments: list[dict[str, Any]] = []
    for epoch in range(1, 4):
        for split in SPLITS:
            split_steps = _split_steps(split, batch_size=12, world_size=4, frame_budget=12000)
            target_step += split_steps
            segments.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "split_steps": split_steps,
                    "target_step": target_step,
                    "name": f"epoch{epoch}_{split.name}",
                }
            )
    return segments


def _write_config(
    *,
    segment: dict[str, Any],
    output_dir: Path,
    config_dir: Path,
    resume: bool,
    smoke: bool,
) -> Path:
    split: SplitSpec = segment["split"]
    cfg = _base_config(output_dir)
    cfg["webdataset_length_index_path"] = str(split.length_index)
    cfg["webdataset_bucket_manifest_path"] = str(split.bucket_manifest)
    cfg["max_steps"] = int(segment["target_step"])
    cfg["epochs"] = None
    cfg["resume_from"] = "latest" if resume else None
    cfg["init_checkpoint_path"] = None if resume else str(INIT_CHECKPOINT)
    cfg["wandb_run_name"] = f"{RUN_NAME}_{segment['name']}"
    if smoke:
        cfg["output_dir"] = str(output_dir)
        cfg["save_every"] = 1
        cfg["step_eval_every"] = 1
        cfg["step_eval_samples"] = 4
        cfg["top_k_step_checkpoints"] = 1
        cfg["save_deepspeed_sharded_checkpoints"] = False
        cfg["num_workers"] = 2
        cfg["decoded_batch_prefetch"] = 1

    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"stage206_{segment['name']}.yaml"
    save_yaml(path, cfg)
    return path


def _run_segment(config_path: Path, log_path: Path, *, dry_run: bool) -> int:
    cmd = [
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
    env = os.environ.copy()
    env["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    print(" ".join(cmd), flush=True)
    print(f"log={log_path}", flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(process.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage206 full online Nano-CTC distillation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only-next", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    if args.smoke:
        output_dir = Path(f"{output_dir}_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = args.config_dir / ("smoke" if args.smoke else "full")

    if not INIT_CHECKPOINT.is_file():
        raise FileNotFoundError(str(INIT_CHECKPOINT))

    segments = _segments(smoke=bool(args.smoke))
    latest_step = _latest_step(output_dir)
    print(f"output_dir={output_dir}", flush=True)
    print(f"latest_step={latest_step}", flush=True)
    for segment in segments:
        target_step = int(segment["target_step"])
        split: SplitSpec = segment["split"]
        if latest_step >= target_step:
            print(f"skip {segment['name']} target_step={target_step} latest_step={latest_step}", flush=True)
            continue
        resume = latest_step > 0
        config_path = _write_config(
            segment=segment,
            output_dir=output_dir,
            config_dir=config_dir,
            resume=resume,
            smoke=bool(args.smoke),
        )
        log_path = output_dir / "logs" / f"{segment['name']}.log"
        print(
            f"run {segment['name']} split={split.name} hours={split.hours:.3f} "
            f"split_steps={segment['split_steps']} target_step={target_step} resume={resume}",
            flush=True,
        )
        code = _run_segment(config_path, log_path, dry_run=bool(args.dry_run))
        if code != 0:
            print(f"segment failed: {segment['name']} exit={code}", file=sys.stderr, flush=True)
            return code
        latest_step = _latest_step(output_dir)
        print(f"segment complete: {segment['name']} latest_step={latest_step}", flush=True)
        if args.only_next:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
