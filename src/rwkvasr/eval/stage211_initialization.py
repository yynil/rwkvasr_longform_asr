from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import sha256_file


DEFAULT_STAGE211_INITIALIZATION_RECEIPT = (
    Path.home() / "rwkvasr_eval" / "stage211_initialization" / "nano_initialization_receipt.json"
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_bound_file(
    receipt: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    label: str,
) -> Path:
    path = Path(str(receipt.get(path_key) or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    digest = str(receipt.get(sha256_key) or "")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None or sha256_file(path) != digest:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def validate_stage211_initialization_receipt(
    receipt_path: Path,
    *,
    expected_calibration_checkpoint: Path | None = None,
    expected_nano_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 Nano initialization receipt")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "nano_initialization_proof",
        "complete": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 Nano initialization receipt contract mismatch.")

    bound_paths: dict[str, Path] = {}
    for stem, label in (
        ("calibration_reuse_receipt", "calibration reuse receipt"),
        ("calibration_selection_report", "calibration selection report"),
        ("calibration_checkpoint", "calibration checkpoint"),
        ("calibration_provenance", "calibration provenance"),
        ("stage210_checkpoint", "Stage210 initialization checkpoint"),
        ("stage210_train_config", "Stage210 train config"),
        ("stage210_log", "Stage210 initialization log"),
        ("nano_checkpoint", "Nano checkpoint"),
    ):
        bound_paths[stem] = _validate_bound_file(
            receipt,
            path_key=f"{stem}_path",
            sha256_key=f"{stem}_sha256",
            label=f"Stage211 {label}",
        )

    if expected_calibration_checkpoint is not None:
        calibration_checkpoint = expected_calibration_checkpoint.expanduser().resolve()
        if bound_paths["calibration_checkpoint"] != calibration_checkpoint:
            raise ValueError("Stage211 initialization receipt binds another calibration checkpoint.")
        if receipt["calibration_checkpoint_sha256"] != sha256_file(calibration_checkpoint):
            raise ValueError("Stage211 initialization calibration checkpoint changed.")
    if (
        expected_nano_checkpoint_sha256 is not None
        and receipt["nano_checkpoint_sha256"] != expected_nano_checkpoint_sha256
    ):
        raise ValueError("Stage211 initialization receipt binds another Nano checkpoint.")

    source_bindings = receipt.get("loader_source_bindings")
    if not isinstance(source_bindings, list) or len(source_bindings) != 2:
        raise ValueError("Stage211 initialization receipt lacks loader source bindings.")
    for index, binding in enumerate(source_bindings):
        if not isinstance(binding, dict):
            raise ValueError("Stage211 initialization loader source binding is invalid.")
        _validate_bound_file(
            binding,
            path_key="path",
            sha256_key="sha256",
            label=f"Stage211 initialization loader source {index}",
        )

    runtime = receipt.get("runtime_load_report")
    if not isinstance(runtime, dict):
        raise ValueError("Stage211 initialization receipt lacks a runtime load report.")
    errors = runtime.get("first_layer_reconstruction_errors")
    if not isinstance(errors, dict) or set(errors) != {"q", "k", "v"}:
        raise ValueError("Stage211 initialization receipt lacks Q/K/V reconstruction errors.")
    for name, value in errors.items():
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 0.25:
            raise ValueError(
                f"Stage211 initialization {name.upper()} reconstruction error is invalid."
            )
    expected_runtime = {
        "qkv_mapped_to_both_directions": True,
        "qkv_projection_scale_mode": "rwkv_norm",
        "encoder_layers": 70,
        "rwkv_encoder_loaded_tensors": 1125,
        "expected_non_attention_mlp_norm_tensors": 564,
        "expected_bidirectional_qkvo_and_input_projection_tensors": 561,
        "ctc_decoder_layers": 5,
        "ctc_decoder_loaded_tensors": 84,
        "ctc_head_loaded_rows": 60515,
        "ctc_head_ignored_rows": [60514],
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise ValueError("Stage211 initialization runtime load report is incomplete.")

    freeze = receipt.get("freeze_report")
    expected_freeze = {
        "freeze_encoder_except_time_mixer": True,
        "freeze_ctc_decoder": True,
        "freeze_ctc_head": True,
        "frozen_tensors": 650,
        "trainable_params": 189378560,
    }
    if not isinstance(freeze, dict) or any(
        freeze.get(key) != value for key, value in expected_freeze.items()
    ):
        raise ValueError("Stage211 initialization freeze report is invalid.")

    tensor_audit = receipt.get("frozen_tensor_audit")
    expected_audit = {
        "complete": True,
        "non_attention_mlp_norm_tensors": 564,
        "ctc_decoder_tensors": 84,
        "ctc_head_teacher_rows": 60515,
        "ctc_head_project_rows": 60516,
        "mismatched_tensors": 0,
        "mismatched_head_rows": 0,
    }
    if not isinstance(tensor_audit, dict) or any(
        tensor_audit.get(key) != value for key, value in expected_audit.items()
    ):
        raise ValueError("Stage211 initialization frozen-tensor audit is invalid.")
    if tensor_audit.get("mismatch_examples") != []:
        raise ValueError("Stage211 initialization frozen-tensor audit contains mismatches.")

    return dict(receipt)
