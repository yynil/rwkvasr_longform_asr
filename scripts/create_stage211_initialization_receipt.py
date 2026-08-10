#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from rwkvasr.eval.stage211_gate import sha256_file
from rwkvasr.eval.stage211_initialization import (
    DEFAULT_STAGE211_INITIALIZATION_RECEIPT,
    validate_stage211_initialization_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_RECEIPT = (
    Path.home()
    / "rwkvasr_eval"
    / "stage211_calibration_selected_full"
    / "public"
    / "reuse_receipt.json"
)
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
LOADER_SOURCE_PATHS = (
    REPO_ROOT / "src" / "rwkvasr" / "modules" / "rwkv_asr_ctc.py",
    REPO_ROOT / "src" / "rwkvasr" / "modules" / "sensevoice_rwkv_encoder.py",
)
NON_ATTENTION_SUFFIXES = (
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "feed_forward.w_1.weight",
    "feed_forward.w_1.bias",
    "feed_forward.w_2.weight",
    "feed_forward.w_2.bias",
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _state_dict(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no state dict: {path}")
    return state


def _source_layer_prefix(layer_index: int) -> str:
    if layer_index == 0:
        return "encoders0.0"
    if layer_index < 50:
        return f"encoders.{layer_index - 1}"
    return f"tp_encoders.{layer_index - 50}"


def _equal_after_projection(student: Tensor, teacher: Tensor) -> bool:
    return tuple(student.shape) == tuple(teacher.shape) and torch.equal(
        student,
        teacher.to(dtype=student.dtype),
    )


def _audit_frozen_tensors(
    *,
    stage210_checkpoint: Path,
    nano_checkpoint: Path,
    encoder_layers: int,
    project_blank_id: int,
    teacher_blank_id: int,
    ignored_token_ids: tuple[int, ...],
) -> dict[str, Any]:
    student = _state_dict(stage210_checkpoint)
    teacher = _state_dict(nano_checkpoint)
    mismatches: list[str] = []
    non_attention_count = 0
    for layer_index in range(encoder_layers):
        source_prefix = _source_layer_prefix(layer_index)
        for suffix in NON_ATTENTION_SUFFIXES:
            student_key = f"encoder.sensevoice_encoder.layers.{layer_index}.{suffix}"
            teacher_key = f"audio_encoder.{source_prefix}.{suffix}"
            student_value = student.get(student_key)
            teacher_value = teacher.get(teacher_key)
            non_attention_count += 1
            if not isinstance(student_value, Tensor) or not isinstance(teacher_value, Tensor):
                mismatches.append(f"missing:{student_key}:{teacher_key}")
            elif not _equal_after_projection(student_value, teacher_value):
                mismatches.append(f"value:{student_key}:{teacher_key}")
    for suffix in ("after_norm.weight", "after_norm.bias", "tp_norm.weight", "tp_norm.bias"):
        student_key = f"encoder.sensevoice_encoder.{suffix}"
        teacher_key = f"audio_encoder.{suffix}"
        student_value = student.get(student_key)
        teacher_value = teacher.get(teacher_key)
        non_attention_count += 1
        if not isinstance(student_value, Tensor) or not isinstance(teacher_value, Tensor):
            mismatches.append(f"missing:{student_key}:{teacher_key}")
        elif not _equal_after_projection(student_value, teacher_value):
            mismatches.append(f"value:{student_key}:{teacher_key}")

    decoder_keys = sorted(key for key in teacher if key.startswith("ctc_decoder."))
    for key in decoder_keys:
        student_value = student.get(key)
        teacher_value = teacher[key]
        if not isinstance(student_value, Tensor) or not _equal_after_projection(
            student_value, teacher_value
        ):
            mismatches.append(f"decoder:{key}")

    student_weight = student.get("ctc_head.weight")
    student_bias = student.get("ctc_head.bias")
    teacher_weight = teacher.get("ctc.ctc_lo.weight")
    teacher_bias = teacher.get("ctc.ctc_lo.bias")
    if not all(
        isinstance(value, Tensor)
        for value in (student_weight, student_bias, teacher_weight, teacher_bias)
    ):
        raise ValueError("Stage211 initialization checkpoint lacks CTC head tensors.")
    assert isinstance(student_weight, Tensor)
    assert isinstance(student_bias, Tensor)
    assert isinstance(teacher_weight, Tensor)
    assert isinstance(teacher_bias, Tensor)
    if int(student_weight.size(0)) != int(teacher_weight.size(0)) + 1:
        raise ValueError("Stage211 projected CTC head row count is invalid.")

    mismatched_head_rows = 0
    ignored = set(int(token_id) for token_id in ignored_token_ids)
    for start in range(0, int(teacher_weight.size(0)), 2048):
        stop = min(start + 2048, int(teacher_weight.size(0)))
        teacher_ids = torch.arange(start, stop, dtype=torch.long)
        project_ids = teacher_ids.clone()
        project_ids[teacher_ids == teacher_blank_id] = project_blank_id
        expected_weight = teacher_weight[start:stop].to(dtype=student_weight.dtype)
        expected_bias = teacher_bias[start:stop].to(dtype=student_bias.dtype)
        weight_match = torch.eq(student_weight[project_ids], expected_weight).flatten(1).all(dim=1)
        bias_match = torch.eq(student_bias[project_ids], expected_bias)
        mismatched_head_rows += int((~(weight_match & bias_match)).sum().item())
    for ignored_id in ignored:
        if ignored_id == project_blank_id:
            continue
        if not torch.count_nonzero(student_weight[ignored_id]).item() == 0:
            mismatched_head_rows += 1
        if float(student_bias[ignored_id].item()) != float(
            student_bias.new_tensor(-1.0e4).item()
        ):
            mismatched_head_rows += 1

    return {
        "complete": not mismatches and mismatched_head_rows == 0,
        "non_attention_mlp_norm_tensors": non_attention_count,
        "ctc_decoder_tensors": len(decoder_keys),
        "ctc_head_teacher_rows": int(teacher_weight.size(0)),
        "ctc_head_project_rows": int(student_weight.size(0)),
        "mismatched_tensors": len(mismatches),
        "mismatched_head_rows": mismatched_head_rows,
        "mismatch_examples": mismatches[:20],
    }


def _parse_runtime_reports(log_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    init_lines = [line for line in text.splitlines() if "Loaded FunASR-Nano CTC init:" in line]
    freeze_lines = [line for line in text.splitlines() if "Applied parameter freeze:" in line]
    if len(init_lines) != 1 or len(freeze_lines) != 1:
        raise ValueError("Stage210 log must contain exactly one Nano init and freeze report.")
    init_line = init_lines[0]
    freeze_line = freeze_lines[0]

    def integer(name: str, line: str) -> int:
        match = re.search(rf"\b{re.escape(name)}=([0-9]+)\b", line)
        if match is None:
            raise ValueError(f"Stage210 runtime report lacks {name}.")
        return int(match.group(1))

    errors_match = re.search(r"rwkv_first_layer_errors=(\{[^}]+\})", init_line)
    ignored_match = re.search(r"ignored_rows=(\[[^]]*\])", init_line)
    scale_match = re.search(r"rwkv_qkv_scale_mode=([a-z0-9_]+)", init_line)
    if errors_match is None or ignored_match is None or scale_match is None:
        raise ValueError("Stage210 runtime Nano initialization report is incomplete.")
    errors = ast.literal_eval(errors_match.group(1))
    ignored_rows = ast.literal_eval(ignored_match.group(1))
    runtime = {
        "qkv_mapped_to_both_directions": True,
        "qkv_projection_scale_mode": scale_match.group(1),
        "encoder_layers": 70,
        "rwkv_encoder_loaded_tensors": integer("rwkv_encoder_tensors", init_line),
        "expected_non_attention_mlp_norm_tensors": 564,
        "expected_bidirectional_qkvo_and_input_projection_tensors": 561,
        "first_layer_reconstruction_errors": {
            name: float(errors[name]) for name in ("q", "k", "v")
        },
        "ctc_decoder_layers": 5,
        "ctc_decoder_loaded_tensors": integer("decoder_tensors", init_line),
        "ctc_head_loaded_rows": integer("head_rows", init_line),
        "ctc_head_ignored_rows": [int(value) for value in ignored_rows],
    }
    freeze = {
        "freeze_encoder_except_time_mixer": "freeze_encoder_except_time_mixer=True" in freeze_line,
        "freeze_ctc_decoder": "freeze_ctc_decoder=True" in freeze_line,
        "freeze_ctc_head": "freeze_ctc_head=True" in freeze_line,
        "frozen_tensors": integer("frozen_tensors", freeze_line),
        "trainable_params": integer("trainable_params", freeze_line),
    }
    return runtime, freeze


def build_receipt(
    *,
    calibration_reuse_receipt_path: Path,
    nano_checkpoint: Path,
) -> dict[str, Any]:
    calibration_reuse_receipt_path = calibration_reuse_receipt_path.expanduser().resolve()
    nano_checkpoint = nano_checkpoint.expanduser().resolve()
    calibration = _load_json(
        calibration_reuse_receipt_path,
        label="Stage211 calibration reuse receipt",
    )
    calibration_checkpoint = Path(str(calibration.get("checkpoint_path") or "")).resolve()
    if calibration.get("checkpoint_sha256") != sha256_file(calibration_checkpoint):
        raise ValueError("Stage211 calibration checkpoint binding is invalid.")
    selection_path = Path(str(calibration.get("selection_report_path") or "")).resolve()
    if calibration.get("selection_report_sha256") != sha256_file(selection_path):
        raise ValueError("Stage211 calibration selection binding is invalid.")
    selection = _load_json(selection_path, label="Stage211 calibration selection report")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("Stage211 calibration selection lacks the selected checkpoint.")
    if (
        Path(str(selected.get("checkpoint_path") or "")).resolve() != calibration_checkpoint
        or selected.get("checkpoint_sha256") != calibration["checkpoint_sha256"]
        or selected.get("eligible") is not True
    ):
        raise ValueError("Stage211 calibration selection checkpoint mismatch.")

    calibration_provenance = calibration_checkpoint.parent / "stage211_provenance.json"
    provenance = _load_json(calibration_provenance, label="Stage211 calibration provenance")
    stage210_checkpoint = Path(str(provenance.get("init_checkpoint_path") or "")).resolve()
    if provenance.get("init_checkpoint_sha256") != sha256_file(stage210_checkpoint):
        raise ValueError("Stage211 calibration provenance initialization changed.")
    stage210_run = stage210_checkpoint.parent
    train_config_path = stage210_run / "train_config.yaml"
    log_path = stage210_run / "logs" / "subblock.log"
    config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage210 training config is invalid.")
    expected_config = {
        "num_layers": 70,
        "ctc_decoder_type": "funasr_nano_transformer",
        "ctc_decoder_num_layers": 5,
        "funasr_nano_ctc_init_load_rwkv_encoder_from_qkv": True,
        "funasr_nano_ctc_init_rwkv_qkv_scale_mode": "rwkv_norm",
        "funasr_nano_ctc_init_load_decoder": True,
        "funasr_nano_ctc_init_load_head": True,
        "funasr_nano_ctc_teacher_blank_id": 60514,
        "ctc_teacher_online_project_ignored_token_ids": [60514],
        "blank_id": 60515,
        "freeze_encoder_except_time_mixer": True,
        "freeze_ctc_decoder": True,
        "freeze_ctc_head": True,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Stage210 Nano initialization/freeze config contract mismatch.")
    configured_nano = Path(str(config.get("funasr_nano_ctc_init_checkpoint_path") or "")).resolve()
    if configured_nano != nano_checkpoint:
        raise ValueError("Stage210 initialization config binds another Nano checkpoint.")

    runtime_report, freeze_report = _parse_runtime_reports(log_path)
    ignored_ids = tuple(
        int(value) for value in config.get("ctc_teacher_online_project_ignored_token_ids", [])
    )
    frozen_tensor_audit = _audit_frozen_tensors(
        stage210_checkpoint=stage210_checkpoint,
        nano_checkpoint=nano_checkpoint,
        encoder_layers=int(config["num_layers"]),
        project_blank_id=int(config["blank_id"]),
        teacher_blank_id=int(config["funasr_nano_ctc_teacher_blank_id"]),
        ignored_token_ids=ignored_ids,
    )
    if not frozen_tensor_audit["complete"]:
        raise ValueError(
            "Stage210 frozen Nano tensor audit failed: "
            f"{frozen_tensor_audit['mismatch_examples']}"
        )

    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "nano_initialization_proof",
        "complete": True,
        "calibration_reuse_receipt_path": str(calibration_reuse_receipt_path),
        "calibration_reuse_receipt_sha256": sha256_file(calibration_reuse_receipt_path),
        "calibration_selection_report_path": str(selection_path),
        "calibration_selection_report_sha256": sha256_file(selection_path),
        "calibration_checkpoint_path": str(calibration_checkpoint),
        "calibration_checkpoint_sha256": sha256_file(calibration_checkpoint),
        "calibration_provenance_path": str(calibration_provenance),
        "calibration_provenance_sha256": sha256_file(calibration_provenance),
        "stage210_checkpoint_path": str(stage210_checkpoint),
        "stage210_checkpoint_sha256": sha256_file(stage210_checkpoint),
        "stage210_train_config_path": str(train_config_path),
        "stage210_train_config_sha256": sha256_file(train_config_path),
        "stage210_log_path": str(log_path),
        "stage210_log_sha256": sha256_file(log_path),
        "nano_checkpoint_path": str(nano_checkpoint),
        "nano_checkpoint_sha256": sha256_file(nano_checkpoint),
        "loader_source_bindings": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in LOADER_SOURCE_PATHS
        ],
        "runtime_load_report": runtime_report,
        "freeze_report": freeze_report,
        "frozen_tensor_audit": frozen_tensor_audit,
    }


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different Stage211 initialization receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the immutable Stage211 Nano QKV/MLP/decoder/head initialization proof."
    )
    parser.add_argument(
        "--calibration-reuse-receipt",
        type=Path,
        default=DEFAULT_CALIBRATION_RECEIPT,
    )
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_STAGE211_INITIALIZATION_RECEIPT)
    args = parser.parse_args()
    receipt = build_receipt(
        calibration_reuse_receipt_path=args.calibration_reuse_receipt,
        nano_checkpoint=args.nano_checkpoint,
    )
    _write_immutable(args.output, receipt)
    validate_stage211_initialization_receipt(
        args.output,
        expected_calibration_checkpoint=Path(receipt["calibration_checkpoint_path"]),
        expected_nano_checkpoint_sha256=receipt["nano_checkpoint_sha256"],
    )
    print(
        f"initialization_receipt={args.output.expanduser().resolve()} "
        f"nano_sha256={receipt['nano_checkpoint_sha256']} "
        f"exact_audited_tensors={receipt['frozen_tensor_audit']['non_attention_mlp_norm_tensors'] + receipt['frozen_tensor_audit']['ctc_decoder_tensors']} "
        f"head_rows={receipt['frozen_tensor_audit']['ctc_head_project_rows']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
