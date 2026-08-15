from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

import rwkvasr.eval.stage211_storage as stage211_storage
from rwkvasr.eval.stage211_gate import sha256_file


def _write_checkpoint(path: Path, *, step: int, epoch: int) -> None:
    torch.save(
        {
            "step": step,
            "extra": {
                "epoch": epoch,
                "epoch_batch_offset": 0,
                "completed_epoch_batch_count": 4,
            },
        },
        path,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    ds_root = run_dir / "ds_checkpoints"
    ds_root.mkdir(parents=True)
    for tag in ("best", "epoch-1", "epoch-2", "epoch-3", "step-4", "step-12"):
        tag_dir = ds_root / tag
        tag_dir.mkdir()
        (tag_dir / "state.pt").write_bytes(tag.encode("ascii"))
    (ds_root / "latest").write_text("epoch-3\n", encoding="utf-8")
    (ds_root / "zero_to_fp32.py").write_text("# retained\n", encoding="utf-8")

    records = []
    for epoch in range(1, 4):
        checkpoint = run_dir / f"epoch-{epoch}.pt"
        _write_checkpoint(checkpoint, step=epoch * 4, epoch=epoch)
        records.append(
            {
                "epoch": epoch,
                "step": epoch * 4,
                "epoch_batch_offset": 0,
                "completed_epoch_batch_count": 4,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    completion = run_dir / "step-12.pt"
    _write_checkpoint(completion, step=12, epoch=3)
    best = run_dir / "best.pt"
    _write_checkpoint(best, step=12, epoch=3)
    (run_dir / "latest_checkpoint.yaml").write_text(
        yaml.safe_dump(
            {
                "checkpoint_type": "deepspeed",
                "step": 12,
                "epoch": 3,
                "epoch_batch_offset": 0,
                "checkpoint_path": str((run_dir / "epoch-3.pt").resolve()),
                "deepspeed_checkpoint_dir": str((ds_root / "epoch-3").resolve()),
                "resume_tag": "epoch-3",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "best_checkpoint.yaml").write_text(
        yaml.safe_dump(
            {
                "epoch": 3,
                "step": 12,
                "checkpoint_path": str(best.resolve()),
                "deepspeed_checkpoint_dir": str(ds_root.resolve()),
                "resume_tag": "best",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "curriculum_coverage",
        "complete": True,
        "full_data_profile": True,
        "phase": "mixer",
        "difficulty": "easy",
        "epochs": 3,
        "steps_per_epoch": 4,
        "steps": 12,
        "run_dir": str(run_dir.resolve()),
        "completion_checkpoint_path": str(completion.resolve()),
        "completion_checkpoint_sha256": sha256_file(completion),
        "runtime_epoch_coverage": {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "runtime_epoch_coverage",
            "complete": True,
            "epochs": 3,
            "steps_per_epoch": 4,
            "total_steps": 12,
            "records": records,
        },
        "parameter_delta_audit": {
            "complete": True,
            "policy": "stage211_timemixer_and_input_projection_only",
            "forbidden_changed_tensors": 0,
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, run_dir


def test_completed_segment_compaction_preserves_evidence_and_resume_tags(
    tmp_path: Path,
) -> None:
    receipt_path, run_dir = _fixture(tmp_path)

    plan = stage211_storage.compact_stage211_completed_segment(
        curriculum_receipt_path=receipt_path,
        execute=False,
    )
    assert plan["complete"] is False
    assert len(plan["remove"]) == 4
    assert all((run_dir / "ds_checkpoints" / record["tag"]).is_dir() for record in plan["remove"])

    result = stage211_storage.compact_stage211_completed_segment(
        curriculum_receipt_path=receipt_path
    )

    assert result["complete"] is True
    assert result["preserved_tags"] == ["best", "epoch-3"]
    assert result["removed_tags"] == ["epoch-1", "epoch-2", "step-12", "step-4"]
    assert result["bytes_planned"] > 0
    assert sorted(
        path.name for path in (run_dir / "ds_checkpoints").iterdir() if path.is_dir()
    ) == ["best", "epoch-3"]
    for name in ("best.pt", "epoch-1.pt", "epoch-2.pt", "epoch-3.pt", "step-12.pt"):
        assert (run_dir / name).is_file()
    assert (run_dir / "stage211_storage_compaction_plan.json").is_file()
    assert (run_dir / "stage211_storage_compaction_receipt.json").is_file()

    repeated = stage211_storage.compact_stage211_completed_segment(
        curriculum_receipt_path=receipt_path
    )
    assert repeated == result

    restored = run_dir / "ds_checkpoints" / result["removed_tags"][0]
    restored.mkdir()
    with pytest.raises(ValueError, match="compacted checkpoint tag reappeared"):
        stage211_storage.compact_stage211_completed_segment(curriculum_receipt_path=receipt_path)


def test_completed_segment_compaction_resumes_an_interrupted_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, run_dir = _fixture(tmp_path)
    real_rmtree = shutil.rmtree
    calls = 0

    def interrupt_after_first(path: Path) -> None:
        nonlocal calls
        calls += 1
        real_rmtree(path)
        if calls == 1:
            raise RuntimeError("controlled interruption")

    monkeypatch.setattr(stage211_storage.shutil, "rmtree", interrupt_after_first)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        stage211_storage.compact_stage211_completed_segment(curriculum_receipt_path=receipt_path)
    assert (run_dir / "stage211_storage_compaction_plan.json").is_file()
    assert not (run_dir / "stage211_storage_compaction_receipt.json").exists()

    monkeypatch.setattr(stage211_storage.shutil, "rmtree", real_rmtree)
    result = stage211_storage.compact_stage211_completed_segment(
        curriculum_receipt_path=receipt_path
    )
    assert result["complete"] is True
    assert all(not (run_dir / "ds_checkpoints" / tag).exists() for tag in result["removed_tags"])


def test_completed_segment_compaction_rejects_unknown_or_active_state(
    tmp_path: Path,
) -> None:
    receipt_path, run_dir = _fixture(tmp_path)
    unknown = run_dir / "ds_checkpoints" / "manual-copy"
    unknown.mkdir()
    with pytest.raises(ValueError, match="unsafe checkpoint tag"):
        stage211_storage.compact_stage211_completed_segment(curriculum_receipt_path=receipt_path)

    unknown.rmdir()
    latest_path = run_dir / "latest_checkpoint.yaml"
    latest = yaml.safe_load(latest_path.read_text(encoding="utf-8"))
    latest["step"] = 11
    latest_path.write_text(yaml.safe_dump(latest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="nonterminal latest checkpoint"):
        stage211_storage.compact_stage211_completed_segment(curriculum_receipt_path=receipt_path)
