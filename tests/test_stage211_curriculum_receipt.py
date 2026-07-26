import importlib
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage211_receipt = importlib.import_module("scripts.create_stage211_curriculum_receipt")
audit_stage211_checkpoint_delta = stage211_receipt.audit_stage211_checkpoint_delta


def _write_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> None:
    torch.save({"model": state, "step": 1}, path)


def _base_state() -> dict[str, torch.Tensor]:
    return {
        "encoder.layers.0.time_mixer.weight": torch.zeros(2, 2),
        "encoder.layers.0.input_proj.weight": torch.zeros(2, 3),
        "encoder.layers.0.feed_forward.weight": torch.ones(2, 2),
        "ctc_decoder.layers.0.weight": torch.full((2, 2), 2.0),
        "ctc_head.weight": torch.full((2, 2), 3.0),
    }


def test_checkpoint_delta_allows_only_stage211_operator_path(tmp_path: Path) -> None:
    initial = _base_state()
    completion = {key: value.clone() for key, value in initial.items()}
    completion["encoder.layers.0.time_mixer.weight"].add_(1.0)
    completion["encoder.layers.0.input_proj.weight"].add_(2.0)
    init_path = tmp_path / "init.pt"
    completion_path = tmp_path / "completion.pt"
    _write_checkpoint(init_path, initial)
    _write_checkpoint(completion_path, completion)

    audit = audit_stage211_checkpoint_delta(
        init_checkpoint_path=init_path,
        completion_checkpoint_path=completion_path,
    )

    assert audit["complete"] is True
    assert audit["initial_tensor_count"] == 5
    assert audit["completion_tensor_count"] == 5
    assert audit["allowed_changed_tensors"] == 2
    assert audit["allowed_changed_numel"] == 10
    assert audit["allowed_unchanged_tensors"] == 0
    assert audit["frozen_unchanged_tensors"] == 3
    assert audit["forbidden_changed_tensors"] == 0


def test_checkpoint_delta_rejects_frozen_path_change(tmp_path: Path) -> None:
    initial = _base_state()
    completion = {key: value.clone() for key, value in initial.items()}
    completion["encoder.layers.0.time_mixer.weight"].add_(1.0)
    completion["ctc_head.weight"].add_(1.0)
    init_path = tmp_path / "init.pt"
    completion_path = tmp_path / "completion.pt"
    _write_checkpoint(init_path, initial)
    _write_checkpoint(completion_path, completion)

    with pytest.raises(ValueError, match="frozen parameter path changed"):
        audit_stage211_checkpoint_delta(
            init_checkpoint_path=init_path,
            completion_checkpoint_path=completion_path,
        )


def test_checkpoint_delta_rejects_no_operator_update(tmp_path: Path) -> None:
    initial = _base_state()
    init_path = tmp_path / "init.pt"
    completion_path = tmp_path / "completion.pt"
    _write_checkpoint(init_path, initial)
    _write_checkpoint(
        completion_path,
        {key: value.clone() for key, value in initial.items()},
    )

    with pytest.raises(ValueError, match="changed no TimeMixer/input-projection"):
        audit_stage211_checkpoint_delta(
            init_checkpoint_path=init_path,
            completion_checkpoint_path=completion_path,
        )
