from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from rwkvasr.eval.stage211_gate import sha256_file
from rwkvasr.eval import stage211_initialization
from rwkvasr.eval.stage211_initialization import (
    _normalized_loader_ast,
    _validate_loader_source_binding,
    validate_stage211_initialization_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
receipt_builder = importlib.import_module("scripts.create_stage211_initialization_receipt")


def _source_prefix(layer_index: int) -> str:
    if layer_index == 0:
        return "encoders0.0"
    if layer_index < 50:
        return f"encoders.{layer_index - 1}"
    return f"tp_encoders.{layer_index - 50}"


def _write_model_pair(tmp_path: Path) -> tuple[Path, Path]:
    teacher: dict[str, torch.Tensor] = {}
    student: dict[str, torch.Tensor] = {}
    for layer_index in range(70):
        for suffix_index, suffix in enumerate(receipt_builder.NON_ATTENTION_SUFFIXES):
            value = torch.tensor([layer_index + suffix_index / 10.0])
            teacher[f"audio_encoder.{_source_prefix(layer_index)}.{suffix}"] = value
            student[f"encoder.sensevoice_encoder.layers.{layer_index}.{suffix}"] = value.clone()
    for index, suffix in enumerate(
        ("after_norm.weight", "after_norm.bias", "tp_norm.weight", "tp_norm.bias")
    ):
        value = torch.tensor([100.0 + index])
        teacher[f"audio_encoder.{suffix}"] = value
        student[f"encoder.sensevoice_encoder.{suffix}"] = value.clone()
    for index in range(84):
        value = torch.tensor([200.0 + index])
        teacher[f"ctc_decoder.tensor_{index}"] = value
        student[f"ctc_decoder.tensor_{index}"] = value.clone()

    teacher_weight = torch.arange(60515, dtype=torch.float32).unsqueeze(1)
    teacher_bias = torch.arange(60515, dtype=torch.float32)
    student_weight = torch.zeros((60516, 1), dtype=torch.float32)
    student_bias = torch.full((60516,), -1.0e4, dtype=torch.float32)
    student_weight[:60514] = teacher_weight[:60514]
    student_bias[:60514] = teacher_bias[:60514]
    student_weight[60515] = teacher_weight[60514]
    student_bias[60515] = teacher_bias[60514]
    teacher["ctc.ctc_lo.weight"] = teacher_weight
    teacher["ctc.ctc_lo.bias"] = teacher_bias
    student["ctc_head.weight"] = student_weight
    student["ctc_head.bias"] = student_bias

    nano_checkpoint = tmp_path / "nano" / "model.pt"
    nano_checkpoint.parent.mkdir()
    torch.save({"state_dict": teacher}, nano_checkpoint)
    stage210_checkpoint = tmp_path / "stage210" / "step-30000.pt"
    stage210_checkpoint.parent.mkdir()
    torch.save({"model": student, "step": 30000}, stage210_checkpoint)
    return stage210_checkpoint, nano_checkpoint


def _write_production_shaped_inputs(tmp_path: Path) -> tuple[Path, Path]:
    stage210_checkpoint, nano_checkpoint = _write_model_pair(tmp_path)
    stage210_run = stage210_checkpoint.parent
    config = {
        "num_layers": 70,
        "ctc_decoder_type": "funasr_nano_transformer",
        "ctc_decoder_num_layers": 5,
        "funasr_nano_ctc_init_checkpoint_path": str(nano_checkpoint.resolve()),
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
    (stage210_run / "train_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    log_dir = stage210_run / "logs"
    log_dir.mkdir()
    (log_dir / "subblock.log").write_text(
        "[rwkvasr] Loaded FunASR-Nano CTC init: "
        "rwkv_encoder_tensors=1125 "
        "rwkv_first_layer_errors={'q': 0.05, 'k': 0.04, 'v': 0.03} "
        "rwkv_qkv_scale_mode=rwkv_norm decoder_tensors=84 "
        "head_rows=60515 ignored_rows=[60514]\n"
        "[rwkvasr] Applied parameter freeze: freeze_encoder=False "
        "freeze_encoder_except_time_mixer=True freeze_ctc_decoder=True "
        "freeze_ctc_head=True frozen_tensors=650 trainable_params=189378560\n",
        encoding="utf-8",
    )

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    calibration_checkpoint = calibration_dir / "step-30000.pt"
    calibration_checkpoint.write_bytes(b"calibration")
    (calibration_dir / "stage211_provenance.json").write_text(
        json.dumps(
            {
                "init_checkpoint_path": str(stage210_checkpoint.resolve()),
                "init_checkpoint_sha256": sha256_file(stage210_checkpoint),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected": {
                    "checkpoint_path": str(calibration_checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(calibration_checkpoint),
                    "eligible": True,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calibration_receipt = tmp_path / "calibration-reuse.json"
    calibration_receipt.write_text(
        json.dumps(
            {
                "checkpoint_path": str(calibration_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(calibration_checkpoint),
                "selection_report_path": str(selection.resolve()),
                "selection_report_sha256": sha256_file(selection),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return calibration_receipt, nano_checkpoint


def test_initialization_receipt_audits_exact_nano_frozen_tensors(tmp_path: Path) -> None:
    calibration_receipt, nano_checkpoint = _write_production_shaped_inputs(tmp_path)
    receipt = receipt_builder.build_receipt(
        calibration_reuse_receipt_path=calibration_receipt,
        nano_checkpoint=nano_checkpoint,
    )
    output = tmp_path / "initialization-receipt.json"
    receipt_builder._write_immutable(output, receipt)

    validated = validate_stage211_initialization_receipt(
        output,
        expected_calibration_checkpoint=Path(receipt["calibration_checkpoint_path"]),
        expected_nano_checkpoint_sha256=sha256_file(nano_checkpoint),
    )

    assert validated["runtime_load_report"]["qkv_mapped_to_both_directions"] is True
    assert validated["loader_source_chain_passed"] is True
    assert [row["mode"] for row in validated["loader_source_validation"]] == [
        "exact_file_sha256",
        "exact_file_sha256",
    ]
    assert validated["frozen_tensor_audit"] == {
        "complete": True,
        "non_attention_mlp_norm_tensors": 564,
        "ctc_decoder_tensors": 84,
        "ctc_head_teacher_rows": 60515,
        "ctc_head_project_rows": 60516,
        "mismatched_tensors": 0,
        "mismatched_head_rows": 0,
        "mismatch_examples": [],
    }

    receipt["runtime_load_report"]["qkv_mapped_to_both_directions"] = False
    output.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime load report"):
        validate_stage211_initialization_receipt(output)


def test_initialization_loader_source_accepts_only_recorded_git_evolution() -> None:
    source = REPO_ROOT / "src" / "rwkvasr" / "modules" / "rwkv_asr_ctc.py"

    evidence = _validate_loader_source_binding(
        {
            "path": str(source.resolve()),
            "sha256": "ed555d921292bfb4fc4efcf4dbaf24c8ca6fe8c0dc31cde968bba78156e4bc56",
        },
        index=0,
    )

    assert evidence["mode"] == "git_history_non_initialization_ast_equivalent"
    assert evidence["historical_commit"] == "c4fee65421617b29ab70d415e5f3721f0ca64cbe"
    assert evidence["normalization"] == "strip_ctc_target_validation_v1"
    assert evidence["removed_current_methods"] == 1
    assert evidence["removed_current_calls"] == 1
    assert len(evidence["normalized_ast_sha256"]) == 64


def test_initialization_loader_source_rejects_other_ast_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical_source = b"""\
class RWKVCTCModel:
    def ctc_loss(self, logits, targets, target_lengths):
        return logits
"""
    changed_source = b"""\
class RWKVCTCModel:
    def ctc_loss(self, logits, targets, target_lengths):
        self._validate_ctc_targets(targets, target_lengths)
        return logits + 1

    def _validate_ctc_targets(self, targets, target_lengths):
        if len(targets) != sum(target_lengths):
            raise ValueError
"""
    source = tmp_path / "rwkv_asr_ctc.py"
    source.write_bytes(changed_source)
    historical_sha256 = stage211_initialization._sha256_bytes(historical_source)
    monkeypatch.setattr(stage211_initialization, "_EVOLVED_LOADER_PATH", source.resolve())
    monkeypatch.setattr(
        stage211_initialization,
        "_git_historical_source",
        lambda path, *, expected_sha256: ("a" * 40, historical_source),
    )

    historical_ast, _, _ = _normalized_loader_ast(historical_source)
    current_ast, methods, calls = _normalized_loader_ast(changed_source)
    assert methods == 1
    assert calls == 1
    assert current_ast != historical_ast
    with pytest.raises(ValueError, match="outside the approved"):
        _validate_loader_source_binding(
            {"path": str(source.resolve()), "sha256": historical_sha256},
            index=0,
        )


def test_initialization_loader_source_rejects_changed_validation_call_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical_source = b"""\
class RWKVCTCModel:
    def ctc_loss(self, logits, targets, target_lengths):
        return logits
"""
    changed_source = b"""\
class RWKVCTCModel:
    def ctc_loss(self, logits, targets, target_lengths):
        self._validate_ctc_targets(target_lengths, targets)
        return logits

    def _validate_ctc_targets(self, targets, target_lengths):
        return None
"""
    source = tmp_path / "rwkv_asr_ctc.py"
    source.write_bytes(changed_source)
    monkeypatch.setattr(stage211_initialization, "_EVOLVED_LOADER_PATH", source.resolve())
    monkeypatch.setattr(
        stage211_initialization,
        "_git_historical_source",
        lambda path, *, expected_sha256: ("a" * 40, historical_source),
    )

    _, methods, calls = _normalized_loader_ast(changed_source)
    assert methods == 1
    assert calls == 0
    with pytest.raises(ValueError, match="outside the approved"):
        _validate_loader_source_binding(
            {
                "path": str(source.resolve()),
                "sha256": stage211_initialization._sha256_bytes(historical_source),
            },
            index=0,
        )


def test_initialization_frozen_tensor_audit_rejects_changed_mlp(tmp_path: Path) -> None:
    stage210_checkpoint, nano_checkpoint = _write_model_pair(tmp_path)
    payload = torch.load(stage210_checkpoint, weights_only=True)
    payload["model"]["encoder.sensevoice_encoder.layers.3.feed_forward.w_1.weight"] += 1.0
    torch.save(payload, stage210_checkpoint)

    audit = receipt_builder._audit_frozen_tensors(
        stage210_checkpoint=stage210_checkpoint,
        nano_checkpoint=nano_checkpoint,
        encoder_layers=70,
        project_blank_id=60515,
        teacher_blank_id=60514,
        ignored_token_ids=(60514,),
    )

    assert audit["complete"] is False
    assert audit["mismatched_tensors"] == 1
    assert "feed_forward.w_1.weight" in audit["mismatch_examples"][0]
