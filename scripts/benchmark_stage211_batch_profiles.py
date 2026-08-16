from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.data import (
    estimate_bucket_manifest_steps,
    estimate_bucket_manifest_tail_padding_samples,
    load_webdataset_bucket_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+):(?P<batch>[1-9][0-9]*):(?P<frames>[1-9][0-9]*)$")
TRAIN_LINE_MARKER = "[deepspeed-train]"
STEP_PATTERN = re.compile(r"\bstep=([0-9]+)\b")
LOSS_PATTERN = re.compile(r"\bloss=([^ ]+)")
ONLINE_MATCH_PATTERN = re.compile(
    r"\b(?P<name>online_[A-Za-z0-9_]+_match)=(?P<count>[0-9]+)/(?P<total>[0-9]+)\b"
)
COSINE_PATTERNS = (
    re.compile(r"\bonline_layer_mixer_cosine=([^ ]+)"),
    re.compile(r"\bonline_layer_block_cosine=([^ ]+)"),
    re.compile(r"\bonline_layer_ffn_cosine=([^ ]+)"),
)
MISSING_PATTERN = re.compile(r"\b(?:[A-Za-z0-9_]*missing)=([0-9]+)\b")
FRAME_DELTA_PATTERN = re.compile(r"\b(?:[A-Za-z0-9_]*frame_delta)=(-?[0-9]+)\b")


@dataclass(frozen=True)
class BatchProfile:
    name: str
    batch_size: int
    frame_budget: int


DEFAULT_PROFILES = (
    BatchProfile("baseline", 36, 24_000),
    BatchProfile("batch48_frames42k", 48, 42_000),
    BatchProfile("batch64_frames56k", 64, 56_000),
    BatchProfile("batch80_frames70k", 80, 70_000),
    BatchProfile("batch96_frames84k", 96, 84_000),
    BatchProfile("batch128_frames112k", 128, 112_000),
    BatchProfile("batch160_frames140k", 160, 140_000),
    BatchProfile("batch192_frames168k", 192, 168_000),
)


@dataclass(frozen=True)
class TrainTelemetry:
    step: int
    elapsed_seconds: float
    loss: float
    cosine: float | None
    match_count: int
    match_total: int
    missing_total: int
    max_abs_frame_delta: int
    match_fields: dict[str, tuple[int, int]] = field(default_factory=dict)


ONLINE_MATCH_WEIGHT_FIELDS = (
    ("ctc_teacher_online_loss_weight", "online_teacher_match"),
    ("ctc_teacher_online_blank_loss_weight", "online_blank_match"),
    ("ctc_teacher_online_mass_loss_weight", "online_mass_match"),
    ("ctc_teacher_online_full_loss_weight", "online_full_match"),
    (
        "ctc_teacher_online_conditional_nonblank_loss_weight",
        "online_conditional_nonblank_match",
    ),
    (
        "ctc_teacher_online_conditional_nonblank_hard_loss_weight",
        "online_conditional_nonblank_hard_match",
    ),
    ("ctc_teacher_online_encoder_loss_weight", "online_encoder_match"),
    ("ctc_teacher_online_decoder_hidden_loss_weight", "online_decoder_hidden_match"),
    ("ctc_teacher_online_sequence_loss_weight", "online_sequence_match"),
    (
        "ctc_teacher_online_sequence_presence_loss_weight",
        "online_sequence_presence_match",
    ),
    ("ctc_teacher_online_sequence_window_loss_weight", "online_sequence_window_match"),
    ("ctc_teacher_online_nonblank_window_topk_loss_weight", "online_nonblank_window_topk_match"),
)


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_profile(value: str) -> BatchProfile:
    match = PROFILE_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("profile must use NAME:BATCH_SIZE:FRAME_BUDGET")
    return BatchProfile(
        name=match.group("name"),
        batch_size=int(match.group("batch")),
        frame_budget=int(match.group("frames")),
    )


def required_online_match_fields(config: dict[str, Any]) -> tuple[str, ...]:
    required = [
        match_field
        for weight_field, match_field in ONLINE_MATCH_WEIGHT_FIELDS
        if float(config.get(weight_field, 0.0) or 0.0) > 0.0
    ]
    if any(
        float(config.get(field_name, 0.0) or 0.0) > 0.0
        for field_name in (
            "ctc_teacher_online_nonblank_hard_loss_weight",
            "ctc_teacher_online_nonblank_margin_loss_weight",
        )
    ):
        required.append("online_nonblank_match")
    if any(
        float(config.get(field_name, 0.0) or 0.0) > 0.0
        for field_name in (
            "ctc_teacher_online_nonblank_window_loss_weight",
            "ctc_teacher_online_nonblank_window_margin_loss_weight",
        )
    ):
        required.append("online_nonblank_window_match")
    if any(
        float(config.get(field_name, 0.0) or 0.0) > 0.0
        for field_name in (
            "ctc_teacher_online_layer_mixer_loss_weight",
            "ctc_teacher_online_layer_ffn_loss_weight",
            "ctc_teacher_online_layer_block_loss_weight",
        )
    ):
        required.append("online_layer_match")
    return tuple(required)


def parse_train_telemetry(line: str, *, elapsed_seconds: float) -> TrainTelemetry | None:
    if TRAIN_LINE_MARKER not in line:
        return None
    step_match = STEP_PATTERN.search(line)
    loss_match = LOSS_PATTERN.search(line)
    match_fields = {
        match.group("name"): (int(match.group("count")), int(match.group("total")))
        for match in ONLINE_MATCH_PATTERN.finditer(line)
    }
    if step_match is None or loss_match is None or not match_fields:
        return None
    primary_match = match_fields.get("online_layer_match") or next(iter(match_fields.values()))
    cosine: float | None = None
    for pattern in COSINE_PATTERNS:
        cosine_match = pattern.search(line)
        if cosine_match is not None:
            cosine = float(cosine_match.group(1))
            break
    missing_values = [int(value) for value in MISSING_PATTERN.findall(line)]
    frame_deltas = [abs(int(value)) for value in FRAME_DELTA_PATTERN.findall(line)]
    return TrainTelemetry(
        step=int(step_match.group(1)),
        elapsed_seconds=float(elapsed_seconds),
        loss=float(loss_match.group(1)),
        cosine=cosine,
        match_count=int(primary_match[0]),
        match_total=int(primary_match[1]),
        missing_total=sum(missing_values),
        max_abs_frame_delta=max(frame_deltas, default=0),
        match_fields=match_fields,
    )


def build_probe_config(
    base_config: dict[str, Any],
    *,
    profile: BatchProfile,
    output_dir: Path,
    init_checkpoint: Path,
    max_steps: int,
    world_size: int,
) -> dict[str, Any]:
    if max_steps <= 1:
        raise ValueError("probe max_steps must be greater than one")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    config = copy.deepcopy(base_config)
    config.update(
        {
            "output_dir": str(output_dir.resolve()),
            "init_checkpoint_path": str(init_checkpoint.resolve()),
            "resume_from": None,
            "resume_tag": None,
            "max_steps": int(max_steps),
            "epochs": None,
            "save_every": int(max_steps),
            "save_deepspeed_sharded_checkpoints": False,
            "step_eval_at_start": False,
            "step_eval_every": None,
            "step_eval_samples": None,
            "max_eval_samples": 0,
            "top_k_step_checkpoints": 1,
            "periodic_checkpoint_keep_last": 1,
            "wandb_enabled": False,
            "wandb_run_name": None,
            "log_every": 1,
            "batch_size": int(profile.batch_size),
            "batch_token_budget": int(profile.frame_budget),
            "length_bucket_frame_budget": int(profile.frame_budget),
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
        }
    )
    deepspeed = copy.deepcopy(config.get("deepspeed"))
    if not isinstance(deepspeed, dict):
        raise ValueError("base config lacks a DeepSpeed mapping")
    gradient_accumulation = int(deepspeed.get("gradient_accumulation_steps", 1))
    deepspeed.update(
        {
            "train_micro_batch_size_per_gpu": int(profile.batch_size),
            "train_batch_size": int(profile.batch_size)
            * int(world_size)
            * gradient_accumulation,
        }
    )
    config["deepspeed"] = deepspeed
    return config


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_worktree_changes() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line)


def _gpu_indices() -> tuple[int, ...]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


def _active_gpu_processes() -> tuple[str, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _sample_gpu_memory() -> dict[int, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[int, float] = {}
    for line in result.stdout.splitlines():
        index, separator, memory = line.partition(",")
        if separator:
            values[int(index.strip())] = float(memory.strip()) / 1024.0
    return values


def _monitor_gpu_memory(
    stop: threading.Event,
    peaks: dict[int, float],
    errors: list[str],
    *,
    poll_seconds: float,
) -> None:
    while not stop.wait(poll_seconds):
        try:
            for index, memory_gib in _sample_gpu_memory().items():
                peaks[index] = max(peaks.get(index, 0.0), memory_gib)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            errors.append(str(error))
            return


def _probe_command(config_path: Path, *, world_size: int, master_port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master-port",
        str(master_port),
        "--nproc_per_node",
        str(world_size),
        "-m",
        "rwkvasr.cli.train_ctc_deepspeed",
        "--config-yaml",
        str(config_path.resolve()),
    ]


def run_profile(
    *,
    profile: BatchProfile,
    config_path: Path,
    log_path: Path,
    world_size: int,
    master_port: int,
    memory_poll_seconds: float,
) -> dict[str, Any]:
    command = _probe_command(config_path, world_size=world_size, master_port=master_port)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
        }
    )
    telemetry: list[TrainTelemetry] = []
    tail: deque[str] = deque(maxlen=80)
    peaks: dict[int, float] = {}
    monitor_errors: list[str] = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor_gpu_memory,
        args=(stop, peaks, monitor_errors),
        kwargs={"poll_seconds": memory_poll_seconds},
        daemon=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = datetime.now(UTC)
    started = time.monotonic()
    monitor.start()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                elapsed = time.monotonic() - started
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                point = parse_train_telemetry(line, elapsed_seconds=elapsed)
                if point is not None:
                    telemetry.append(point)
                    if point.step == 1 or point.step % 20 == 0:
                        print(
                            f"[stage211-throughput] profile={profile.name} "
                            f"step={point.step} loss={point.loss:.6f}",
                            flush=True,
                        )
            return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            stop.set()
            monitor.join(timeout=max(5.0, memory_poll_seconds * 4.0))
    finished = datetime.now(UTC)
    return {
        "profile": asdict(profile),
        "command": command,
        "started_at": started_wall.isoformat(),
        "finished_at": finished.isoformat(),
        "return_code": int(return_code),
        "log_path": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "gpu_peak_memory_gib": {str(index): value for index, value in sorted(peaks.items())},
        "gpu_memory_monitor_errors": monitor_errors,
        "telemetry": [asdict(point) for point in telemetry],
        "error_tail": list(tail) if return_code != 0 else [],
    }


def summarize_profile(
    result: dict[str, Any],
    *,
    warmup_steps: int,
    max_steps: int,
    world_size: int,
    full_steps: int,
    max_peak_memory_gib: float,
    required_match_fields: tuple[str, ...] = ("online_layer_match",),
) -> dict[str, Any]:
    points = [TrainTelemetry(**row) for row in result.get("telemetry", [])]
    by_step = {point.step: point for point in points}
    expected_steps = set(range(1, max_steps + 1))
    missing_steps = sorted(expected_steps - set(by_step))
    ordered = [by_step[step] for step in sorted(by_step) if step <= max_steps]
    anchor = by_step.get(warmup_steps)
    measured = [point for point in ordered if point.step > warmup_steps]
    elapsed = 0.0
    step_rate = 0.0
    if anchor is not None and measured:
        elapsed = measured[-1].elapsed_seconds - anchor.elapsed_seconds
        if elapsed > 0.0:
            step_rate = (measured[-1].step - anchor.step) / elapsed
    finite_losses = bool(measured) and all(math.isfinite(point.loss) for point in measured)
    finite_cosines = all(
        point.cosine is None or math.isfinite(point.cosine) for point in measured
    )
    if not required_match_fields:
        raise ValueError("throughput preflight requires at least one online match field")
    primary_match_field = (
        "online_layer_match"
        if "online_layer_match" in required_match_fields
        else required_match_fields[0]
    )

    def point_match_fields(point: TrainTelemetry) -> dict[str, tuple[int, int]]:
        if point.match_fields:
            return {
                name: (int(values[0]), int(values[1]))
                for name, values in point.match_fields.items()
            }
        return {"online_layer_match": (point.match_count, point.match_total)}

    missing_required_match_fields: dict[str, list[str]] = {}
    incomplete_required_match_fields: dict[str, dict[str, list[int]]] = {}
    for point in measured:
        matches = point_match_fields(point)
        missing_fields = [name for name in required_match_fields if name not in matches]
        if missing_fields:
            missing_required_match_fields[str(point.step)] = missing_fields
        incomplete = {
            name: [matches[name][0], matches[name][1]]
            for name in required_match_fields
            if name in matches and (matches[name][1] <= 0 or matches[name][0] != matches[name][1])
        }
        if incomplete:
            incomplete_required_match_fields[str(point.step)] = incomplete
    complete_matches = bool(measured) and not (
        missing_required_match_fields or incomplete_required_match_fields
    )
    missing_total = sum(point.missing_total for point in measured)
    max_abs_frame_delta = max((point.max_abs_frame_delta for point in measured), default=0)
    peak_values = [float(value) for value in result.get("gpu_peak_memory_gib", {}).values()]
    peak_memory = max(peak_values, default=float("inf"))
    memory_monitor_ok = not result.get("gpu_memory_monitor_errors") and bool(peak_values)
    process_ok = int(result.get("return_code", -1)) == 0
    complete_steps = not missing_steps and len(ordered) == max_steps
    safety_pass = all(
        (
            process_ok,
            complete_steps,
            finite_losses,
            finite_cosines,
            complete_matches,
            missing_total == 0,
            max_abs_frame_delta == 0,
            memory_monitor_ok,
            peak_memory <= max_peak_memory_gib,
            step_rate > 0.0,
        )
    )
    sample_weights = [
        point_match_fields(point).get(primary_match_field, (0, 0))[1] for point in measured
    ]
    local_samples = sum(sample_weights)
    mean_loss = (
        sum(point.loss * weight for point, weight in zip(measured, sample_weights, strict=True))
        / local_samples
        if local_samples > 0
        else None
    )
    cosine_weight = sum(
        weight
        for point, weight in zip(measured, sample_weights, strict=True)
        if point.cosine is not None
    )
    mean_cosine = (
        sum(
            float(point.cosine) * weight
            for point, weight in zip(measured, sample_weights, strict=True)
            if point.cosine is not None
        )
        / cosine_weight
        if cosine_weight > 0
        else None
    )
    return {
        "process_ok": process_ok,
        "complete_steps": complete_steps,
        "missing_steps": missing_steps,
        "measurement_steps": len(measured),
        "measurement_elapsed_seconds": elapsed,
        "steps_per_second": step_rate,
        "estimated_global_samples_per_second": (
            local_samples * world_size / elapsed if elapsed > 0.0 else 0.0
        ),
        "mean_loss": mean_loss,
        "mean_cosine": mean_cosine,
        "alignment_mean_aggregation": "primary_match_sample_weighted",
        "required_match_fields": list(required_match_fields),
        "primary_match_field": primary_match_field,
        "missing_required_match_fields": missing_required_match_fields,
        "incomplete_required_match_fields": incomplete_required_match_fields,
        "complete_teacher_matches": complete_matches,
        "missing_total": missing_total,
        "max_abs_frame_delta": max_abs_frame_delta,
        "peak_memory_gib": peak_memory,
        "max_peak_memory_gib": max_peak_memory_gib,
        "memory_monitor_ok": memory_monitor_ok,
        "full_coverage_steps": int(full_steps),
        "projected_full_coverage_seconds": (
            float(full_steps) / step_rate if step_rate > 0.0 else None
        ),
        "safety_pass": safety_pass,
    }


def select_profile(
    rows: list[dict[str, Any]],
    *,
    baseline_name: str,
    min_improvement_ratio: float,
    max_loss_regression_ratio: float = 0.05,
    max_cosine_regression: float = 0.005,
) -> dict[str, Any]:
    by_name = {str(row["profile"]["name"]): row for row in rows}
    if baseline_name not in by_name:
        raise ValueError(f"baseline profile is absent: {baseline_name}")
    baseline = by_name[baseline_name]
    baseline_summary = baseline["summary"]
    baseline_seconds = baseline_summary.get("projected_full_coverage_seconds")
    baseline_loss = baseline_summary.get("mean_loss")
    baseline_cosine = baseline_summary.get("mean_cosine")
    if (
        baseline_summary.get("safety_pass") is not True
        or not isinstance(baseline_seconds, (float, int))
        or not isinstance(baseline_loss, (float, int))
        or not isinstance(baseline_cosine, (float, int))
    ):
        return {
            "decision": "baseline_failed",
            "baseline_profile": baseline_name,
            "recommended_profile": None,
            "min_improvement_ratio": min_improvement_ratio,
            "max_loss_regression_ratio": max_loss_regression_ratio,
            "max_cosine_regression": max_cosine_regression,
        }
    candidates: list[tuple[float, str, float]] = []
    comparisons: list[dict[str, Any]] = []
    for name, row in by_name.items():
        if name == baseline_name:
            continue
        candidate_seconds = row["summary"].get("projected_full_coverage_seconds")
        candidate_loss = row["summary"].get("mean_loss")
        candidate_cosine = row["summary"].get("mean_cosine")
        improvement = None
        if isinstance(candidate_seconds, (float, int)) and baseline_seconds > 0:
            improvement = (float(baseline_seconds) - float(candidate_seconds)) / float(
                baseline_seconds
            )
        loss_regression_ratio = None
        if isinstance(candidate_loss, (float, int)):
            loss_regression_ratio = (float(candidate_loss) - float(baseline_loss)) / max(
                abs(float(baseline_loss)), 1.0e-12
            )
        cosine_regression = None
        if isinstance(candidate_cosine, (float, int)):
            cosine_regression = float(baseline_cosine) - float(candidate_cosine)
        quality_pass = (
            loss_regression_ratio is not None
            and loss_regression_ratio <= max_loss_regression_ratio
            and cosine_regression is not None
            and cosine_regression <= max_cosine_regression
        )
        admissible = (
            row["summary"].get("safety_pass") is True
            and improvement is not None
            and improvement >= min_improvement_ratio
            and quality_pass
        )
        comparisons.append(
            {
                "profile": name,
                "projected_full_coverage_seconds": candidate_seconds,
                "improvement_ratio": improvement,
                "mean_loss": candidate_loss,
                "loss_regression_ratio": loss_regression_ratio,
                "mean_cosine": candidate_cosine,
                "cosine_regression": cosine_regression,
                "quality_pass": quality_pass,
                "admissible": admissible,
            }
        )
        if admissible:
            candidates.append((float(candidate_seconds), name, float(improvement)))
    candidates.sort()
    return {
        "decision": "candidate_recommended" if candidates else "keep_baseline",
        "baseline_profile": baseline_name,
        "recommended_profile": candidates[0][1] if candidates else baseline_name,
        "recommended_improvement_ratio": candidates[0][2] if candidates else 0.0,
        "min_improvement_ratio": min_improvement_ratio,
        "baseline_mean_loss": float(baseline_loss),
        "baseline_mean_cosine": float(baseline_cosine),
        "max_loss_regression_ratio": max_loss_regression_ratio,
        "max_cosine_regression": max_cosine_regression,
        "comparisons": comparisons,
        "formal_admission": False,
    }


def _profile_full_coverage(
    config: dict[str, Any],
    profile: BatchProfile,
    *,
    world_size: int,
    formal_epochs: int,
) -> dict[str, int]:
    manifest_path = Path(str(config.get("webdataset_bucket_manifest_path") or "")).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bucket manifest is unavailable: {manifest_path}")
    manifest = load_webdataset_bucket_manifest(manifest_path)
    steps_per_epoch = estimate_bucket_manifest_steps(
        manifest,
        split=str(config.get("webdataset_split") or "train"),
        batch_size=profile.batch_size,
        world_size=world_size,
        frame_budget=profile.frame_budget,
        drop_last=False,
    )
    tail_padding = estimate_bucket_manifest_tail_padding_samples(
        manifest,
        split=str(config.get("webdataset_split") or "train"),
        batch_size=profile.batch_size,
        world_size=world_size,
        frame_budget=profile.frame_budget,
    )
    return {
        "steps_per_epoch": int(steps_per_epoch),
        "formal_epochs": int(formal_epochs),
        "full_coverage_steps": int(steps_per_epoch) * int(formal_epochs),
        "tail_padding_samples_per_epoch": int(tail_padding),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Stage211 batch profiles without changing formal coverage receipts."
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("mixer", "block", "logits"), required=True)
    parser.add_argument("--profile", type=parse_profile, action="append", default=[])
    parser.add_argument("--baseline-profile", default="baseline")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--formal-epochs", type=int, default=3)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29731)
    parser.add_argument("--max-peak-memory-gib", type=float, default=22.0)
    parser.add_argument("--min-improvement-ratio", type=float, default=0.10)
    parser.add_argument("--max-loss-regression-ratio", type=float, default=0.05)
    parser.add_argument("--max-cosine-regression", type=float, default=0.005)
    parser.add_argument("--memory-poll-seconds", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profiles = args.profile or list(DEFAULT_PROFILES)
    if len({profile.name for profile in profiles}) != len(profiles):
        parser.error("profile names must be unique")
    if args.baseline_profile not in {profile.name for profile in profiles}:
        parser.error("--baseline-profile must name one of the requested profiles")
    if args.warmup_steps <= 0 or args.measure_steps <= 0:
        parser.error("warmup and measured steps must be positive")
    if args.formal_epochs <= 0 or args.world_size <= 0:
        parser.error("formal epochs and world size must be positive")
    if args.max_peak_memory_gib <= 0.0 or not 0.0 <= args.min_improvement_ratio < 1.0:
        parser.error("memory limit must be positive and improvement ratio must be in [0, 1)")
    if args.max_loss_regression_ratio < 0.0 or args.max_cosine_regression < 0.0:
        parser.error("quality regression limits must be non-negative")
    if args.memory_poll_seconds <= 0.0:
        parser.error("memory poll interval must be positive")

    base_config_path = args.base_config.expanduser().resolve()
    init_checkpoint = args.init_checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(str(base_config_path))
    if not init_checkpoint.is_file():
        raise FileNotFoundError(str(init_checkpoint))
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"probe output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(base_config_path)
    manifest_path = Path(
        str(base_config.get("webdataset_bucket_manifest_path") or "")
    ).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(str(manifest_path))
    max_steps = int(args.warmup_steps) + int(args.measure_steps)
    worktree_changes = _git_worktree_changes()
    if not args.dry_run and worktree_changes:
        raise RuntimeError(
            "Stage211 throughput preflight requires a clean git worktree: "
            + "; ".join(worktree_changes)
        )
    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "batch_throughput_preflight",
        "phase": str(args.phase),
        "complete": False,
        "formal_admission": False,
        "dry_run": bool(args.dry_run),
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "git_worktree_clean": not worktree_changes,
        "git_worktree_changes": list(worktree_changes),
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "base_config_path": str(base_config_path),
        "base_config_sha256": sha256_file(base_config_path),
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": sha256_file(manifest_path),
        "warmup_steps": int(args.warmup_steps),
        "measure_steps": int(args.measure_steps),
        "formal_epochs": int(args.formal_epochs),
        "world_size": int(args.world_size),
        "max_peak_memory_gib": float(args.max_peak_memory_gib),
        "min_improvement_ratio": float(args.min_improvement_ratio),
        "max_loss_regression_ratio": float(args.max_loss_regression_ratio),
        "max_cosine_regression": float(args.max_cosine_regression),
        "profiles": [],
    }
    report_path = output_root / "batch_throughput_preflight.json"
    _atomic_write_json(report_path, report)

    if not args.dry_run:
        gpu_indices = _gpu_indices()
        if len(gpu_indices) < args.world_size:
            raise RuntimeError(
                f"requested {args.world_size} GPUs but nvidia-smi exposes {len(gpu_indices)}"
            )
        active_processes = _active_gpu_processes()
        if active_processes:
            raise RuntimeError(
                "Stage211 throughput preflight refuses busy GPUs: " + "; ".join(active_processes)
            )
        report["gpu_indices"] = list(gpu_indices[: args.world_size])

    for profile_index, profile in enumerate(profiles):
        profile_root = output_root / profile.name
        run_dir = profile_root / "run"
        config_path = profile_root / "train_config.yaml"
        profile_root.mkdir(parents=True, exist_ok=False)
        config = build_probe_config(
            base_config,
            profile=profile,
            output_dir=run_dir,
            init_checkpoint=init_checkpoint,
            max_steps=max_steps,
            world_size=int(args.world_size),
        )
        config["stage211_batch_profile_probe_phase"] = str(args.phase)
        required_match_fields = required_online_match_fields(config)
        if not required_match_fields:
            raise ValueError(
                "Stage211 throughput preflight config enables no online teacher match fields."
            )
        save_yaml(config_path, config)
        coverage = _profile_full_coverage(
            config,
            profile,
            world_size=int(args.world_size),
            formal_epochs=int(args.formal_epochs),
        )
        row: dict[str, Any] = {
            "profile": asdict(profile),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "coverage": coverage,
            "required_match_fields": list(required_match_fields),
            "command": _probe_command(
                config_path,
                world_size=int(args.world_size),
                master_port=int(args.master_port) + profile_index,
            ),
        }
        if args.dry_run:
            row["status"] = "dry_run"
        else:
            result = run_profile(
                profile=profile,
                config_path=config_path,
                log_path=profile_root / "train.log",
                world_size=int(args.world_size),
                master_port=int(args.master_port) + profile_index,
                memory_poll_seconds=float(args.memory_poll_seconds),
            )
            row.update(result)
            row["summary"] = summarize_profile(
                result,
                warmup_steps=int(args.warmup_steps),
                max_steps=max_steps,
                world_size=int(args.world_size),
                full_steps=int(coverage["full_coverage_steps"]),
                max_peak_memory_gib=float(args.max_peak_memory_gib),
                required_match_fields=required_match_fields,
            )
        report["profiles"].append(row)
        _atomic_write_json(report_path, report)

    if args.dry_run:
        report["complete"] = True
        report["selection"] = {
            "decision": "dry_run_only",
            "formal_admission": False,
        }
    else:
        report["selection"] = select_profile(
            report["profiles"],
            baseline_name=str(args.baseline_profile),
            min_improvement_ratio=float(args.min_improvement_ratio),
            max_loss_regression_ratio=float(args.max_loss_regression_ratio),
            max_cosine_regression=float(args.max_cosine_regression),
        )
        report["complete"] = True
    report["finished_at"] = datetime.now(UTC).isoformat()
    _atomic_write_json(report_path, report)
    print(f"[stage211-throughput] report={report_path}", flush=True)
    print(
        f"[stage211-throughput] decision={report['selection']['decision']} "
        f"recommended={report['selection'].get('recommended_profile')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
