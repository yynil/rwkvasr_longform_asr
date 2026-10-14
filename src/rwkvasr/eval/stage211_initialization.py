from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import sha256_file


DEFAULT_STAGE211_INITIALIZATION_RECEIPT = (
    Path.home() / "rwkvasr_eval" / "stage211_initialization" / "nano_initialization_receipt.json"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVOLVED_LOADER_PATH = (
    _REPO_ROOT / "src" / "rwkvasr" / "modules" / "rwkv_asr_ctc.py"
).resolve()
_EVOLVED_LOADER_NORMALIZATION = "strip_ctc_target_validation_and_projection_elision_v2"


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_approved_ctc_target_validation_call(statement: ast.stmt) -> bool:
    call = statement.value if isinstance(statement, ast.Expr) else None
    function = call.func if isinstance(call, ast.Call) else None
    return bool(
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "self"
        and function.attr == "_validate_ctc_targets"
        and len(call.args) == 2
        and all(isinstance(argument, ast.Name) for argument in call.args)
        and [argument.id for argument in call.args if isinstance(argument, ast.Name)]
        == ["targets", "target_lengths"]
        and not call.keywords
    )


def _ast_shape(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _parsed_statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _normalize_approved_ctc_projection_elision(method: ast.FunctionDef) -> bool:
    compute_arguments = [
        index
        for index, argument in enumerate(method.args.kwonlyargs)
        if argument.arg == "compute_ctc_logits"
    ]
    compute_references = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Name) and node.id == "compute_ctc_logits"
    ]
    if not compute_arguments and not compute_references:
        return False
    if compute_arguments != [len(method.args.kwonlyargs) - 1] or len(compute_references) != 2:
        return False
    argument_index = compute_arguments[0]
    argument = method.args.kwonlyargs[argument_index]
    default = method.args.kw_defaults[argument_index]
    if (
        not isinstance(argument.annotation, ast.Name)
        or argument.annotation.id != "bool"
        or not isinstance(default, ast.Constant)
        or default.value is not True
    ):
        return False

    approved_current = _parsed_statements(
        """
if not compute_ctc_logits and ctc_loss_weight > 0.0:
    raise ValueError("compute_ctc_logits=False requires ctc_loss_weight=0.")
logits = (
    self.apply_ctc_logit_mask(self.ctc_head(ctc_features))
    if compute_ctc_logits
    else None
)
zero_source = logits if isinstance(logits, Tensor) else ctc_features
zero_loss = zero_source.float().sum() * 0.0
"""
    )
    current_shapes = [_ast_shape(statement) for statement in approved_current]
    matching_offsets = [
        index
        for index in range(len(method.body) - len(current_shapes) + 1)
        if [
            _ast_shape(statement)
            for statement in method.body[index : index + len(current_shapes)]
        ]
        == current_shapes
    ]
    if len(matching_offsets) != 1:
        return False

    logit_length_guard = _parsed_statements(
        """
if logit_lengths is None:
    raise ValueError("CTC training/distillation requires feature lengths.")
"""
    )[0]
    guard_offsets = [
        index
        for index, statement in enumerate(method.body)
        if _ast_shape(statement) == _ast_shape(logit_length_guard)
    ]
    if len(guard_offsets) != 1 or guard_offsets[0] >= matching_offsets[0]:
        return False

    historical_logits, historical_zero_loss = _parsed_statements(
        """
logits = self.apply_ctc_logit_mask(self.ctc_head(ctc_features))
zero_loss = logits.float().sum() * 0.0
"""
    )
    offset = matching_offsets[0]
    method.body[offset : offset + len(current_shapes)] = [historical_zero_loss]
    guard_offset = next(
        index
        for index, statement in enumerate(method.body)
        if _ast_shape(statement) == _ast_shape(logit_length_guard)
    )
    method.body.insert(guard_offset, historical_logits)
    del method.args.kwonlyargs[argument_index]
    del method.args.kw_defaults[argument_index]
    return True


def _normalized_loader_ast(source: bytes) -> tuple[str, int, int, int]:
    tree = ast.parse(source.decode("utf-8"))
    removed_methods = 0
    removed_calls = 0
    normalized_projection_elisions = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "RWKVCTCModel":
            continue
        retained: list[ast.stmt] = []
        for item in node.body:
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == "_validate_ctc_targets"
            ):
                removed_methods += 1
                continue
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "ctc_loss":
                filtered: list[ast.stmt] = []
                for statement in item.body:
                    if _is_approved_ctc_target_validation_call(statement):
                        removed_calls += 1
                        continue
                    filtered.append(statement)
                item.body = filtered
            if isinstance(item, ast.FunctionDef) and item.name == "joint_losses":
                normalized_projection_elisions += int(
                    _normalize_approved_ctc_projection_elision(item)
                )
            retained.append(item)
        node.body = retained
    rendered = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return (
        hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        removed_methods,
        removed_calls,
        normalized_projection_elisions,
    )


def _git_historical_source(path: Path, *, expected_sha256: str) -> tuple[str, bytes]:
    try:
        relative = path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Stage211 evolved initialization loader is outside the repository: {path}"
        ) from error
    commits = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "log", "--format=%H", "--", relative],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for commit in commits:
        source = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "show", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if _sha256_bytes(source) != expected_sha256:
            continue
        ancestor = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            check=False,
        )
        if ancestor.returncode != 0:
            raise ValueError(
                "Stage211 receipt-bound initialization loader commit is not an ancestor of HEAD."
            )
        return commit, source
    raise ValueError(
        "Stage211 initialization loader source changed and its receipt-bound bytes "
        "are unavailable from repository history."
    )


def _validate_loader_source_binding(
    binding: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    path = Path(str(binding.get("path") or "")).expanduser().resolve()
    expected_sha256 = str(binding.get("sha256") or "")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Stage211 initialization loader source {index} is missing: {path}")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError(f"Stage211 initialization loader source {index} SHA-256 is invalid.")
    current_source = path.read_bytes()
    current_sha256 = _sha256_bytes(current_source)
    if current_sha256 == expected_sha256:
        return {
            "path": str(path),
            "mode": "exact_file_sha256",
            "receipt_sha256": expected_sha256,
            "current_sha256": current_sha256,
        }
    if path != _EVOLVED_LOADER_PATH:
        raise ValueError(f"Stage211 initialization loader source {index} SHA-256 mismatch: {path}")

    historical_commit, historical_source = _git_historical_source(
        path,
        expected_sha256=expected_sha256,
    )
    (
        historical_ast_sha256,
        historical_methods,
        historical_calls,
        historical_projection_elisions,
    ) = _normalized_loader_ast(historical_source)
    (
        current_ast_sha256,
        current_methods,
        current_calls,
        current_projection_elisions,
    ) = _normalized_loader_ast(current_source)
    if (
        historical_methods != 0
        or historical_calls != 0
        or historical_projection_elisions != 0
        or current_methods != 1
        or current_calls != 1
        or current_projection_elisions != 1
        or current_ast_sha256 != historical_ast_sha256
    ):
        raise ValueError(
            "Stage211 initialization loader changed outside the approved CTC-target "
            "validation and hidden-only projection-elision evolution."
        )
    return {
        "path": str(path),
        "mode": "git_history_non_initialization_ast_equivalent",
        "receipt_sha256": expected_sha256,
        "current_sha256": current_sha256,
        "historical_commit": historical_commit,
        "normalization": _EVOLVED_LOADER_NORMALIZATION,
        "normalized_ast_sha256": current_ast_sha256,
        "removed_current_methods": current_methods,
        "removed_current_calls": current_calls,
        "normalized_current_projection_elisions": current_projection_elisions,
    }


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
    loader_source_validation: list[dict[str, Any]] = []
    for index, binding in enumerate(source_bindings):
        if not isinstance(binding, dict):
            raise ValueError("Stage211 initialization loader source binding is invalid.")
        loader_source_validation.append(
            _validate_loader_source_binding(binding, index=index)
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

    validated = dict(receipt)
    validated["loader_source_validation"] = loader_source_validation
    validated["loader_source_chain_passed"] = True
    return validated
