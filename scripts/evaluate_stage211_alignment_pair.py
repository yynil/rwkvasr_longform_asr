from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from rwkvasr.cli.eval_online_ctc_distill import (
    _build_online_teacher,
    _resolve_student_dtype,
)
from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_ALIGNMENT_CHECKPOINT_EVAL_ARTIFACT,
    STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    resolve_stage211_nano_teacher_checkpoint,
    sha256_file,
    validate_stage211_phase_train_config,
)
from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig
from rwkvasr.training.checkpoint import load_checkpoint
from rwkvasr.training.deepspeed_loop import (
    DeepSpeedTrainConfig,
    _build_eval_loader,
    _build_step_eval_provenance,
    _evaluate_epoch_loss,
    _materialize_step_eval_batches,
)


PHASES = ("mixer", "block", "logits")
LAYER_IDS = {str(index) for index in range(70)}
LOGIT_REQUIRED_METRICS = (
    "full_kl",
    "conditional_nonblank_kl",
    "conditional_nonblank_hard_ce",
    "blank_binary_kl",
    "selected_top1_agreement",
    "all_top1_agreement",
    "active_top1_agreement",
    "blank_prob_mae",
    "teacher_nonblank_rate",
    "student_nonblank_rate",
    "nonblank_rate_ratio",
    "ctc_token_error_rate",
    "ctc_token_deletion_rate",
    "collapsed_length_ratio",
    "sequence_exact_rate",
    "mean_frame_delta",
    "matched_utterances",
    "missing_utterances",
)


def _resolved_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def _load_phase_config(
    *,
    phase: str,
    train_config_path: Path,
    samples: int,
    batch_size: int,
    num_workers: int,
    feature_seed: int,
    device: str,
    teacher_device: str | None,
    audio_cache_dir: Path,
) -> DeepSpeedTrainConfig:
    raw = dict(load_yaml(train_config_path))
    validate_stage211_phase_train_config(raw, phase=phase)
    valid_keys = {field.name for field in fields(DeepSpeedTrainConfig)}
    config_data = {key: value for key, value in raw.items() if key in valid_keys}
    config_data.update(
        {
            "step_eval_split": "eval",
            "step_eval_samples": int(samples),
            "step_eval_batch_size": int(batch_size),
            "eval_batch_size": int(batch_size),
            "step_eval_shuffle": False,
            "step_eval_at_start": False,
            "step_eval_cache_batches": True,
            "step_eval_feature_seed": int(feature_seed),
            "num_workers": int(num_workers),
            "device": str(device),
            "wandb_enabled": False,
            "specaugment_enabled": False,
            "ctc_teacher_online_audio_cache_dir": str(audio_cache_dir),
        }
    )
    if teacher_device is not None:
        config_data["ctc_teacher_online_device"] = str(teacher_device)
    return DeepSpeedTrainConfig(**config_data)


def _validate_report_metrics(
    *,
    phase: str,
    eval_samples: int,
    layer_components: dict[str, dict[int, dict[str, float]]],
    logit_metrics: dict[str, float],
    decoder_hidden_metrics: dict[str, float],
) -> None:
    if eval_samples != STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES:
        raise RuntimeError(
            "Stage211 alignment pair eval coverage mismatch: "
            f"{eval_samples} != {STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES}."
        )
    if phase in {"mixer", "block"}:
        component = layer_components.get(phase)
        if not isinstance(component, dict) or {
            str(layer_id) for layer_id in component
        } != LAYER_IDS:
            raise RuntimeError(
                f"Stage211 {phase} pair eval did not cover exactly 70 layers."
            )
        for layer_id, metrics in component.items():
            for name in ("loss", "cosine", "rms_ratio"):
                value = float(metrics.get(name, float("nan")))
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"Stage211 {phase} layer {layer_id} metric {name} is not finite."
                    )
    if phase == "block":
        decoder_loss = float(decoder_hidden_metrics.get("loss", float("nan")))
        if not math.isfinite(decoder_loss):
            raise RuntimeError(
                "Stage211 block pair eval lacks finite decoder-hidden loss."
            )
    if phase == "logits":
        for name in LOGIT_REQUIRED_METRICS:
            value = float(logit_metrics.get(name, float("nan")))
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Stage211 logits pair eval metric {name} is not finite."
                )
        if (
            int(round(float(logit_metrics["matched_utterances"])))
            != STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES
            or float(logit_metrics["missing_utterances"]) != 0.0
            or float(logit_metrics["mean_frame_delta"]) != 0.0
        ):
            raise RuntimeError(
                "Stage211 logits pair eval lacks exact teacher coverage/frame parity."
            )


def _pair_id(binding: dict[str, Any]) -> str:
    encoded = json.dumps(
        binding,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluate_checkpoint(
    *,
    role: str,
    logical_step: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    model_config: RWKVCTCModelConfig,
    config: DeepSpeedTrainConfig,
    cached_batches: list[Any],
    teacher: Any,
    device: torch.device,
    student_dtype: torch.dtype | None,
    feature_dtype: torch.dtype | None,
    eval_provenance: dict[str, Any],
    pair_binding: dict[str, Any],
) -> dict[str, Any]:
    model = RWKVCTCModel(model_config)
    restored = load_checkpoint(
        checkpoint_path,
        model=model,
        map_location="cpu",
        strict=True,
    )
    checkpoint_step = int(restored.get("step", 0))
    if role == "candidate" and checkpoint_step != int(logical_step):
        raise ValueError(
            "Stage211 candidate checkpoint payload step mismatch: "
            f"{checkpoint_step} != {logical_step}."
        )
    model.to(device)
    if student_dtype is not None:
        model.to(dtype=student_dtype)
    model.eval()

    layer_metrics: dict[int, dict[str, float]] = {}
    layer_components: dict[str, dict[int, dict[str, float]]] = {}
    logit_metrics: dict[str, float] = {}
    decoder_hidden_metrics: dict[str, float] = {}
    try:
        eval_loss, eval_samples = _evaluate_epoch_loss(
            model=model,
            loader=cached_batches,
            sampler=None,
            epoch=0,
            device=device,
            feature_dtype=feature_dtype,
            mode=str(config.eval_mode),
            max_eval_samples=STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
            config=config,
            ctc_teacher_online=teacher,
            layer_metrics_output=layer_metrics,
            layer_component_metrics_output=layer_components,
            logit_metrics_output=logit_metrics,
            decoder_hidden_metrics_output=decoder_hidden_metrics,
        )
    finally:
        model.to("cpu")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not math.isfinite(float(eval_loss)):
        raise RuntimeError(
            f"Stage211 {role} alignment pair eval loss is not finite."
        )
    _validate_report_metrics(
        phase=str(pair_binding["phase"]),
        eval_samples=int(eval_samples),
        layer_components=layer_components,
        logit_metrics=logit_metrics,
        decoder_hidden_metrics=decoder_hidden_metrics,
    )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": STAGE211_ALIGNMENT_CHECKPOINT_EVAL_ARTIFACT,
        "phase": str(pair_binding["phase"]),
        "role": role,
        "pair_eval_id": _pair_id(pair_binding),
        "step": int(logical_step),
        "checkpoint_step": checkpoint_step,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "train_config_path": str(pair_binding["train_config_path"]),
        "train_config_sha256": str(pair_binding["train_config_sha256"]),
        "model_config_path": str(pair_binding["model_config_path"]),
        "model_config_sha256": str(pair_binding["model_config_sha256"]),
        "nano_checkpoint_path": str(pair_binding["nano_checkpoint_path"]),
        "nano_checkpoint_sha256": str(pair_binding["nano_checkpoint_sha256"]),
        "feature_seed": int(pair_binding["feature_seed"]),
        "student_dtype": (
            str(student_dtype).replace("torch.", "")
            if student_dtype is not None
            else "fp32"
        ),
        "feature_dtype": (
            str(feature_dtype).replace("torch.", "")
            if feature_dtype is not None
            else "fp32"
        ),
        "device": str(device),
        "eval_loss": float(eval_loss),
        "eval_samples": int(eval_samples),
        "eval_batches": len(cached_batches),
        "eval_provenance": eval_provenance,
        "layer_metrics": {
            str(layer_id): metrics
            for layer_id, metrics in layer_metrics.items()
        },
        "layer_components": {
            component: {
                str(layer_id): metrics
                for layer_id, metrics in component_metrics.items()
            }
            for component, component_metrics in layer_components.items()
        },
        "logit_metrics": logit_metrics,
        "decoder_hidden_metrics": decoder_hidden_metrics,
    }


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(
            f"Refusing to overwrite a different Stage211 alignment eval: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def evaluate_pair(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = str(args.phase)
    samples = int(args.samples)
    feature_seed = int(args.feature_seed)
    if phase not in PHASES:
        raise ValueError(f"Unsupported Stage211 alignment phase: {phase!r}")
    if samples != STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES:
        raise ValueError(
            "Stage211 alignment pair eval requires exactly "
            f"{STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES} samples."
        )
    if feature_seed != 0:
        raise ValueError("Stage211 alignment pair eval requires --feature-seed=0.")
    if int(args.batch_size) <= 0 or int(args.num_workers) < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative.")

    train_config_path = _resolved_file(
        Path(args.train_config),
        label="Stage211 train config",
    )
    model_config_path = _resolved_file(
        Path(args.model_config),
        label="Stage211 model config",
    )
    baseline_checkpoint = _resolved_file(
        Path(args.baseline_checkpoint),
        label="Stage211 baseline checkpoint",
    )
    candidate_checkpoint = _resolved_file(
        Path(args.candidate_checkpoint),
        label="Stage211 candidate checkpoint",
    )
    if baseline_checkpoint == candidate_checkpoint:
        raise ValueError(
            "Stage211 baseline and candidate checkpoints must be different files."
        )
    checkpoint_match = re.fullmatch(
        r"step-([0-9]+)\.pt",
        candidate_checkpoint.name,
    )
    if checkpoint_match is None:
        raise ValueError(
            "Stage211 candidate checkpoint must be named step-N.pt."
        )
    candidate_step = int(checkpoint_match.group(1))
    if candidate_step <= 0:
        raise ValueError(
            "Stage211 candidate checkpoint step must be positive."
        )
    baseline_output = Path(args.baseline_output).expanduser().resolve()
    candidate_output = Path(args.candidate_output).expanduser().resolve()
    if baseline_output == candidate_output:
        raise ValueError(
            "Stage211 baseline and candidate reports must use different files."
        )
    audio_cache_dir = (
        Path(args.audio_cache_dir).expanduser().resolve()
        if args.audio_cache_dir is not None
        else candidate_output.parent / "teacher_audio_cache"
    )
    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device={str(device)!r}, but CUDA is unavailable."
        )
    config = _load_phase_config(
        phase=phase,
        train_config_path=train_config_path,
        samples=samples,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        feature_seed=feature_seed,
        device=str(device),
        teacher_device=args.teacher_device,
        audio_cache_dir=audio_cache_dir,
    )
    model_config = RWKVCTCModelConfig(**load_yaml(model_config_path))
    student_dtype = _resolve_student_dtype(
        str(args.student_dtype),
        device=device,
        frontend_type=str(model_config.frontend_type),
    )
    feature_dtype = (
        None
        if str(model_config.frontend_type) == "funasr_nano_encoder"
        else student_dtype
    )

    loader, sampler = _build_eval_loader(
        config,
        shuffle_shards=False,
        step_subset=True,
    )
    if loader is None:
        raise RuntimeError("Stage211 alignment pair eval could not build the fixed loader.")
    cached_batches, cached_samples = _materialize_step_eval_batches(
        loader,
        sampler,
        epoch=0,
        max_eval_samples=samples,
        feature_seed=feature_seed,
    )
    if cached_samples != samples:
        raise RuntimeError(
            "Stage211 alignment pair materialization coverage mismatch: "
            f"{cached_samples} != {samples}."
        )

    bucket_manifest = _resolved_file(
        Path(str(config.webdataset_bucket_manifest_path or "")),
        label="Stage211 fixed-eval bucket manifest",
    )
    eval_provenance = _build_step_eval_provenance(
        config=config,
        bucket_manifest_path=bucket_manifest,
    )
    nano_checkpoint = resolve_stage211_nano_teacher_checkpoint(
        dict(load_yaml(train_config_path))
    )
    baseline_sha256 = sha256_file(baseline_checkpoint)
    candidate_sha256 = sha256_file(candidate_checkpoint)
    pair_binding = {
        "schema_version": 1,
        "phase": phase,
        "baseline_checkpoint_path": str(baseline_checkpoint),
        "baseline_checkpoint_sha256": baseline_sha256,
        "candidate_checkpoint_path": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": candidate_sha256,
        "train_config_path": str(train_config_path),
        "train_config_sha256": sha256_file(train_config_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "nano_checkpoint_path": str(nano_checkpoint),
        "nano_checkpoint_sha256": sha256_file(nano_checkpoint),
        "eval_provenance": eval_provenance,
        "feature_seed": feature_seed,
        "samples": samples,
    }

    teacher = _build_online_teacher(
        config=config,
        model_config=model_config,
        device=device,
        audio_cache_dir=str(audio_cache_dir),
    )
    try:
        baseline_report = _evaluate_checkpoint(
            role="baseline",
            logical_step=0,
            checkpoint_path=baseline_checkpoint,
            checkpoint_sha256=baseline_sha256,
            model_config=model_config,
            config=config,
            cached_batches=cached_batches,
            teacher=teacher,
            device=device,
            student_dtype=student_dtype,
            feature_dtype=feature_dtype,
            eval_provenance=eval_provenance,
            pair_binding=pair_binding,
        )
        candidate_report = _evaluate_checkpoint(
            role="candidate",
            logical_step=candidate_step,
            checkpoint_path=candidate_checkpoint,
            checkpoint_sha256=candidate_sha256,
            model_config=model_config,
            config=config,
            cached_batches=cached_batches,
            teacher=teacher,
            device=device,
            student_dtype=student_dtype,
            feature_dtype=feature_dtype,
            eval_provenance=eval_provenance,
            pair_binding=pair_binding,
        )
    finally:
        del teacher
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_immutable_json(baseline_output, baseline_report)
    _write_immutable_json(candidate_output, candidate_report)
    return baseline_report, candidate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Stage211 phase-initial and final checkpoints over one "
            "materialized fixed-feature batch set."
        )
    )
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument(
        "--samples",
        type=int,
        default=STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    )
    parser.add_argument("--feature-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--audio-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--student-dtype",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    baseline, candidate = evaluate_pair(args)
    print(
        "stage211_alignment_pair "
        f"phase={args.phase} pair={baseline['pair_eval_id']} "
        f"samples={baseline['eval_samples']} "
        f"baseline_loss={baseline['eval_loss']:.6f} "
        f"candidate_loss={candidate['eval_loss']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
