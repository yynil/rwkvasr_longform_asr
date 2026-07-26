from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
comparison = importlib.import_module("scripts.compare_public_ctc_with_nano")


def _write_predictions(
    path: Path,
    *,
    rows: list[tuple[str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"utt_id": utt_id, "pred_text": pred, "ref_text": ref}) + "\n"
            for utt_id, pred, ref in rows
        ),
        encoding="utf-8",
    )


def test_compare_dataset_reports_real_reference_gap_and_gate(tmp_path: Path) -> None:
    nano = tmp_path / "nano.jsonl"
    student = tmp_path / "student.jsonl"
    ref = "one two three four five"
    _write_predictions(
        nano,
        rows=[
            ("utt-1", "one two three four five", ref),
            ("utt-2", "one two three four", ref),
        ],
    )
    _write_predictions(
        student,
        rows=[
            ("utt-1", "one two three four", ref),
            ("utt-2", "one two three", ref),
        ],
    )

    result = comparison.compare_dataset(
        dataset="librispeech_test_clean",
        nano_path=nano,
        student_path=student,
        normalization="ctc",
        max_relative_ratio=1.20,
        max_absolute_gap_points=3.0,
    )

    assert result["sample_count"] == 2
    assert result["nano_error_rate"] == pytest.approx(0.1)
    assert result["student_error_rate"] == pytest.approx(0.3)
    assert result["absolute_gap_points"] == pytest.approx(20.0)
    assert result["relative_ratio"] == pytest.approx(3.0)
    assert result["identical_utt_coverage"] is True
    assert result["gate_pass"] is False


def test_compare_dataset_rejects_reference_or_coverage_mismatch(tmp_path: Path) -> None:
    nano = tmp_path / "nano.jsonl"
    student = tmp_path / "student.jsonl"
    _write_predictions(nano, rows=[("utt-1", "hello", "hello world")])
    _write_predictions(student, rows=[("utt-1", "hello", "different reference")])

    with pytest.raises(ValueError, match="normalized references differ"):
        comparison.compare_dataset(
            dataset="librispeech_test_clean",
            nano_path=nano,
            student_path=student,
            normalization="ctc",
            max_relative_ratio=1.20,
            max_absolute_gap_points=3.0,
        )

    _write_predictions(student, rows=[("utt-2", "hello", "hello world")])
    with pytest.raises(ValueError, match="prediction coverage differs"):
        comparison.compare_dataset(
            dataset="librispeech_test_clean",
            nano_path=nano,
            student_path=student,
            normalization="ctc",
            max_relative_ratio=1.20,
            max_absolute_gap_points=3.0,
        )


def test_build_report_requires_all_public_datasets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Nano prediction map mismatch"):
        comparison.build_report(
            student_prediction_dir=tmp_path,
            nano_predictions={},
            normalization="ctc",
            max_relative_ratio=1.20,
            max_absolute_gap_points=3.0,
        )


def test_build_report_binds_student_checkpoint_when_requested(tmp_path: Path) -> None:
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"student checkpoint")
    nano_predictions: dict[str, Path] = {}
    student_dir = tmp_path / "student"
    for dataset in comparison.DATASETS:
        nano = tmp_path / "nano" / f"{dataset}.jsonl"
        student = student_dir / f"{dataset}.ctc.jsonl"
        rows = [("utt-1", "hello", "hello")]
        _write_predictions(nano, rows=rows)
        _write_predictions(student, rows=rows)
        nano_predictions[dataset] = nano

    report = comparison.build_report(
        student_prediction_dir=student_dir,
        nano_predictions=nano_predictions,
        normalization="ctc",
        max_relative_ratio=1.20,
        max_absolute_gap_points=3.0,
        student_checkpoint=checkpoint,
    )

    assert report["student_checkpoint_path"] == str(checkpoint.resolve())
    assert len(report["student_checkpoint_sha256"]) == 64
