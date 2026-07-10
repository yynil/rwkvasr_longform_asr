from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from rwkvasr.config import load_yaml
from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig
from rwkvasr.training.checkpoint import load_checkpoint
from rwkvasr.training.ctc_task import RWKVDualModeCTCTrainer
from rwkvasr.training.deepspeed_loop import (
    DeepSpeedTrainConfig,
    _build_eval_loader,
    _ctc_teacher_blank_loss,
    _ctc_teacher_mass_loss,
    _ctc_teacher_topk_loss,
    _online_ctc_teacher_distillation_loss,
    _set_loader_epoch,
)
from rwkvasr.training.funasr_online_teacher import (
    FunASRNanoCTCTopKOnlineTeacher,
    FunASROnlineCTCTeacherConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpoint against the online FunASR-Nano CTC distillation "
            "objective used by DeepSpeed training."
        )
    )
    parser.add_argument("--train-config-yaml", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-config-yaml", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epoch", type=int, default=0, help="Deterministic loader epoch/seed for fixed subset eval.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--audio-cache-dir", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--student-dtype",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="Use bf16 on CUDA by default, matching the active Stage206 training path.",
    )
    parser.add_argument(
        "--shuffle-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same shuffled step-subset loader path as periodic step eval.",
    )
    return parser


def _resolve_model_config_path(args: argparse.Namespace) -> Path:
    if args.model_config_yaml is not None:
        return Path(args.model_config_yaml)
    checkpoint_parent = Path(args.checkpoint_path).resolve().parent
    candidate = checkpoint_parent / "model_config.yaml"
    if candidate.is_file():
        return candidate
    train_parent = Path(args.train_config_yaml).resolve().parent
    candidate = train_parent / "model_config.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("model_config.yaml not found; pass --model-config-yaml explicitly.")


def _resolve_device(raw: str) -> torch.device:
    if raw.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device={raw!r}, but CUDA is not available.")
    return torch.device(raw)


def _resolve_student_dtype(value: str, *, device: torch.device, frontend_type: str) -> torch.dtype | None:
    value = str(value)
    if value == "fp32":
        return None
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value != "auto":
        raise ValueError(f"Unsupported student dtype: {value}")
    if device.type != "cuda":
        return None
    if str(frontend_type) == "funasr_nano_encoder":
        return None
    return torch.bfloat16


def _load_train_config(args: argparse.Namespace) -> DeepSpeedTrainConfig:
    config_data = dict(load_yaml(args.train_config_yaml))
    valid_keys = {field.name for field in fields(DeepSpeedTrainConfig)}
    config_data = {key: value for key, value in config_data.items() if key in valid_keys}
    config_data["step_eval_split"] = str(args.split)
    config_data["step_eval_samples"] = int(args.samples)
    config_data["step_eval_batch_size"] = int(args.batch_size)
    config_data["eval_batch_size"] = int(args.batch_size)
    config_data["num_workers"] = int(args.num_workers)
    config_data["device"] = str(args.device)
    config_data["wandb_enabled"] = False
    config_data["specaugment_enabled"] = False
    if args.teacher_device is not None:
        config_data["ctc_teacher_online_device"] = str(args.teacher_device)
    if args.audio_cache_dir is not None:
        config_data["ctc_teacher_online_audio_cache_dir"] = str(args.audio_cache_dir)
    return DeepSpeedTrainConfig(**config_data)


def _online_objective_enabled(config: DeepSpeedTrainConfig) -> bool:
    weights = (
        config.ctc_teacher_online_loss_weight,
        config.ctc_teacher_online_blank_loss_weight,
        config.ctc_teacher_online_mass_loss_weight,
        config.ctc_teacher_online_full_loss_weight,
        config.ctc_teacher_online_encoder_loss_weight,
        config.ctc_teacher_online_sequence_loss_weight,
        config.ctc_teacher_online_sequence_presence_loss_weight,
        config.ctc_teacher_online_sequence_window_loss_weight,
        config.ctc_teacher_online_nonblank_hard_loss_weight,
        config.ctc_teacher_online_nonblank_margin_loss_weight,
        config.ctc_teacher_online_nonblank_window_loss_weight,
        config.ctc_teacher_online_nonblank_window_margin_loss_weight,
        config.ctc_teacher_online_nonblank_window_topk_loss_weight,
    )
    return any(float(weight) > 0.0 for weight in weights)


def _build_online_teacher(
    *,
    config: DeepSpeedTrainConfig,
    model_config: RWKVCTCModelConfig,
    device: torch.device,
    audio_cache_dir: str,
) -> FunASRNanoCTCTopKOnlineTeacher:
    if config.ctc_teacher_online_model_path is None:
        raise ValueError("ctc_teacher_online_model_path is required for online CTC distillation eval.")
    teacher_device = config.ctc_teacher_online_device
    if teacher_device is None:
        teacher_device = str(device)
    audio_index_path = config.ctc_teacher_online_audio_index_path
    if audio_index_path is None and config.webdataset_bucket_manifest_path is None:
        audio_index_path = config.webdataset_length_index_path or config.manifest_path
    return FunASRNanoCTCTopKOnlineTeacher(
        FunASROnlineCTCTeacherConfig(
            model_path=str(config.ctc_teacher_online_model_path),
            audio_index_path=str(audio_index_path) if audio_index_path is not None else None,
            webdataset_index_path=(
                config.ctc_teacher_online_webdataset_index_path or config.webdataset_index_path
            ),
            webdataset_root=config.webdataset_root,
            audio_cache_dir=audio_cache_dir,
            keep_audio_cache=bool(config.ctc_teacher_online_keep_audio_cache),
            device=str(teacher_device),
            split=str(config.webdataset_split or "train"),
            top_k=int(config.ctc_teacher_online_top_k),
            project_blank_id=int(config.blank_id),
            project_vocab_size=int(model_config.ctc_vocab_size),
            project_ignored_token_ids=tuple(
                int(value) for value in config.ctc_teacher_online_project_ignored_token_ids
            ),
            return_full_log_probs=float(config.ctc_teacher_online_full_loss_weight) > 0.0,
            return_encoder_out=float(config.ctc_teacher_online_encoder_loss_weight) > 0.0,
        )
    )


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def _component_losses(
    *,
    config: DeepSpeedTrainConfig,
    losses: dict[str, Any],
    utt_ids: list[str],
    records: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    student_logits = losses.get("logits")
    if not isinstance(student_logits, torch.Tensor):
        raise RuntimeError("joint_losses did not return logits tensor.")
    student_lengths = losses.get("logit_lengths")
    if student_lengths is not None and not isinstance(student_lengths, torch.Tensor):
        raise RuntimeError("joint_losses returned non-tensor logit_lengths.")

    output: dict[str, float | int] = {}
    if float(config.ctc_teacher_online_loss_weight) > 0.0:
        loss, matched, missing = _ctc_teacher_topk_loss(
            student_logits,
            student_lengths,
            utt_ids,
            records,
            blank_id=int(config.blank_id),
            time_map=str(config.ctc_teacher_topk_time_map),
            frame_filter=str(config.ctc_teacher_frame_filter or "all"),
            frame_filter_neighbor_radius=int(config.ctc_teacher_frame_filter_neighbor_radius),
            frame_filter_min_nonblank_prob=float(config.ctc_teacher_frame_filter_min_nonblank_prob),
            missing_policy=str(config.ctc_teacher_topk_missing_policy),
        )
        output.update(
            {
                "online_ctc_teacher_loss": _to_float(loss),
                "online_ctc_teacher_matched": int(matched),
                "online_ctc_teacher_missing": int(missing),
            }
        )
    if float(config.ctc_teacher_online_blank_loss_weight) > 0.0:
        loss, matched, missing = _ctc_teacher_blank_loss(
            student_logits,
            student_lengths,
            utt_ids,
            records,
            blank_id=int(config.blank_id),
            time_map=str(config.ctc_teacher_topk_time_map),
            missing_policy=str(config.ctc_teacher_topk_missing_policy),
        )
        output.update(
            {
                "online_ctc_blank_loss": _to_float(loss),
                "online_ctc_blank_matched": int(matched),
                "online_ctc_blank_missing": int(missing),
            }
        )
    if float(config.ctc_teacher_online_mass_loss_weight) > 0.0:
        loss, matched, missing = _ctc_teacher_mass_loss(
            student_logits,
            student_lengths,
            utt_ids,
            records,
            blank_id=int(config.blank_id),
            time_map=str(config.ctc_teacher_topk_time_map),
            missing_policy=str(config.ctc_teacher_topk_missing_policy),
        )
        output.update(
            {
                "online_ctc_mass_loss": _to_float(loss),
                "online_ctc_mass_matched": int(matched),
                "online_ctc_mass_missing": int(missing),
            }
        )
    return output


def _add_weighted(acc: dict[str, float], counts: dict[str, int], values: dict[str, float | int], batch_size: int) -> None:
    for key, value in values.items():
        if key.endswith("_matched") or key.endswith("_missing"):
            acc[key] = acc.get(key, 0.0) + float(value)
            continue
        acc[key] = acc.get(key, 0.0) + float(value) * float(batch_size)
        counts[key] = counts.get(key, 0) + int(batch_size)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.samples) <= 0:
        raise ValueError("--samples must be positive.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")

    config = _load_train_config(args)
    if not _online_objective_enabled(config):
        raise ValueError("No online CTC teacher loss weights are enabled in the train config.")

    device = _resolve_device(str(args.device))
    model_config_path = _resolve_model_config_path(args)
    model_config = RWKVCTCModelConfig(**load_yaml(model_config_path))
    student_dtype = _resolve_student_dtype(
        str(args.student_dtype),
        device=device,
        frontend_type=str(model_config.frontend_type),
    )
    feature_dtype = student_dtype
    if str(model_config.frontend_type) == "funasr_nano_encoder":
        feature_dtype = None

    model = RWKVCTCModel(model_config)
    restored = load_checkpoint(args.checkpoint_path, model=model, map_location="cpu", strict=True)
    model.to(device)
    if student_dtype is not None:
        model.to(dtype=student_dtype)
    model.eval()

    loader, sampler = _build_eval_loader(config, shuffle_shards=bool(args.shuffle_shards), step_subset=True)
    if loader is None:
        raise ValueError("No eval loader could be built from the train config.")
    _set_loader_epoch(loader, sampler, int(args.epoch))

    cache_context = None
    audio_cache_dir = config.ctc_teacher_online_audio_cache_dir
    if audio_cache_dir is None:
        cache_context = tempfile.TemporaryDirectory(prefix="rwkvasr_online_ctc_eval_", dir="/tmp")
        audio_cache_dir = cache_context.name
    try:
        teacher = _build_online_teacher(
            config=config,
            model_config=model_config,
            device=device,
            audio_cache_dir=str(audio_cache_dir),
        )
        trainer = RWKVDualModeCTCTrainer(model)
        total_loss_sum = 0.0
        sample_count = 0
        batch_count = 0
        component_sums: dict[str, float] = {}
        component_counts: dict[str, int] = {}
        for batch in loader:
            remaining = int(args.samples) - sample_count
            if remaining <= 0:
                break
            if int(batch.features.size(0)) > remaining:
                batch = batch.prefix(remaining)
            batch = batch.to(device, feature_dtype=feature_dtype)
            records = teacher.topk_records(batch.utt_ids, batch.ctc_teacher_audio_rows)
            mask = trainer.eval_direction_mask(str(config.eval_mode), device=batch.features.device)
            losses = model.joint_losses(
                batch.features,
                batch.feature_lengths,
                batch.targets,
                batch.target_lengths,
                decoder_targets=batch.decoder_targets,
                decoder_target_lengths=batch.decoder_target_lengths,
                decoder_prompt_before_audio=batch.decoder_prompt_before_audio,
                decoder_prompt_before_audio_lengths=batch.decoder_prompt_before_audio_lengths,
                direction_mask=mask,
            )
            total_loss = _online_ctc_teacher_distillation_loss(
                config=config,
                losses=losses,
                batch=batch,
                ctc_teacher_online_records=records,
            )
            batch_size = int(batch.features.size(0))
            total_loss_sum += _to_float(total_loss) * float(batch_size)
            component_values = _component_losses(
                config=config,
                losses=losses,
                utt_ids=batch.utt_ids,
                records=records,
            )
            _add_weighted(component_sums, component_counts, component_values, batch_size)
            sample_count += batch_size
            batch_count += 1
            if int(args.log_every) > 0 and batch_count % int(args.log_every) == 0:
                running_loss = total_loss_sum / float(max(1, sample_count))
                print(
                    "eval_online_ctc_distill "
                    f"batches={batch_count} samples={sample_count}/{int(args.samples)} "
                    f"loss={running_loss:.6f}",
                    flush=True,
                )
    finally:
        if cache_context is not None:
            cache_context.cleanup()

    if sample_count <= 0:
        raise RuntimeError("No samples were evaluated.")

    components: dict[str, float | int] = {}
    for key, value in sorted(component_sums.items()):
        if key.endswith("_matched") or key.endswith("_missing"):
            components[key] = int(round(value))
        else:
            components[key] = value / float(max(1, component_counts.get(key, 0)))

    return {
        "version": 1,
        "checkpoint_path": str(Path(args.checkpoint_path)),
        "checkpoint_step": int(restored.get("step", 0)),
        "train_config_yaml": str(Path(args.train_config_yaml)),
        "model_config_yaml": str(model_config_path),
        "split": str(args.split),
        "epoch": int(args.epoch),
        "shuffle_shards": bool(args.shuffle_shards),
        "requested_samples": int(args.samples),
        "eval_samples": int(sample_count),
        "eval_batches": int(batch_count),
        "eval_loss": total_loss_sum / float(sample_count),
        "components": components,
        "weights": {
            "online_ctc_teacher": float(config.ctc_teacher_online_loss_weight),
            "online_ctc_blank": float(config.ctc_teacher_online_blank_loss_weight),
            "online_ctc_mass": float(config.ctc_teacher_online_mass_loss_weight),
            "online_ctc_full": float(config.ctc_teacher_online_full_loss_weight),
            "online_ctc_encoder": float(config.ctc_teacher_online_encoder_loss_weight),
            "online_ctc_sequence": float(config.ctc_teacher_online_sequence_loss_weight),
            "online_ctc_sequence_presence": float(config.ctc_teacher_online_sequence_presence_loss_weight),
            "online_ctc_sequence_window": float(config.ctc_teacher_online_sequence_window_loss_weight),
            "online_ctc_nonblank_hard": float(config.ctc_teacher_online_nonblank_hard_loss_weight),
            "online_ctc_nonblank_margin": float(config.ctc_teacher_online_nonblank_margin_loss_weight),
            "online_ctc_nonblank_window": float(config.ctc_teacher_online_nonblank_window_loss_weight),
            "online_ctc_nonblank_window_margin": float(config.ctc_teacher_online_nonblank_window_margin_loss_weight),
            "online_ctc_nonblank_window_topk": float(config.ctc_teacher_online_nonblank_window_topk_loss_weight),
        },
        "student_dtype": str(student_dtype).replace("torch.", "") if student_dtype is not None else "fp32",
        "feature_dtype": str(feature_dtype).replace("torch.", "") if feature_dtype is not None else "fp32",
        "device": str(device),
        "teacher_device": str(config.ctc_teacher_online_device or device),
        "teacher_top_k": int(config.ctc_teacher_online_top_k),
        "audio_cache_dir": str(audio_cache_dir),
    }


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "eval_online_ctc_distill "
        f"samples={result['eval_samples']} loss={float(result['eval_loss']):.6f} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
