from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
loop = importlib.import_module("scripts.run_stage211_mixer_retention_loop")


def _args(tmp_path: Path) -> argparse.Namespace:
    phase_root = tmp_path / "phase"
    original_gate_dir = tmp_path / "gates" / "mixer"
    correction_run_root = tmp_path / "runs" / "retention"
    correction_gate_root = tmp_path / "gates" / "retention"
    for path in (
        phase_root,
        original_gate_dir,
        correction_run_root,
        correction_gate_root,
    ):
        path.mkdir(parents=True)
    replay = tmp_path / "replay.json"
    replay.write_text("{}\n", encoding="utf-8")
    nano = tmp_path / "model.pt"
    nano.write_bytes(b"nano")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n", encoding="utf-8")
    manifests = tmp_path / "manifests"
    predictions = tmp_path / "predictions"
    configs = tmp_path / "configs"
    for path in (manifests, predictions, configs):
        path.mkdir()
    return argparse.Namespace(
        phase_root=phase_root.resolve(),
        original_gate_dir=original_gate_dir.resolve(),
        correction_run_root=correction_run_root.resolve(),
        correction_gate_root=correction_gate_root.resolve(),
        replay_receipt=replay.resolve(),
        public_manifest_dir=manifests.resolve(),
        nano_prediction_dir=predictions.resolve(),
        nano_checkpoint=nano.resolve(),
        baseline_public_comparison_report=baseline.resolve(),
        config_dir=configs.resolve(),
        selection=(tmp_path / "selected.json").resolve(),
        max_rounds=3,
        master_port=29641,
        devices="0,1,2,3",
        dry_run=False,
    )


def test_validate_selection_only_uses_deep_validator_without_running_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selection = tmp_path / "block-selected.json"
    selection.write_text("{}\n", encoding="utf-8")
    nano = tmp_path / "nano.pt"
    nano.write_bytes(b"nano")
    calls: list[tuple[Path, Path, str]] = []

    def fake_validate(
        selection_path: Path,
        *,
        nano_checkpoint: Path,
        phase: str,
    ) -> dict[str, object]:
        calls.append((selection_path, nano_checkpoint, phase))
        return {
            "phase": phase,
            "correction_round": 2,
            "checkpoint_path": str(tmp_path / "block.pt"),
        }

    monkeypatch.setattr(loop, "_validate_existing_selection", fake_validate)
    monkeypatch.setattr(
        loop,
        "run_retention_loop",
        lambda args: pytest.fail(f"training loop invoked in validate-only mode: {args}"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(loop.__file__),
            "--phase",
            "block",
            "--selection",
            str(selection),
            "--nano-checkpoint",
            str(nano),
            "--validate-selection-only",
        ],
    )

    assert loop.main() == 0
    assert calls == [(selection.resolve(), nano.resolve(), "block")]
    assert "selection validation passed phase=block round=2" in capsys.readouterr().out


def test_retention_finalizer_command_binds_all_prior_receipts(tmp_path: Path) -> None:
    args = _args(tmp_path)
    receipts = [tmp_path / "round1.json", tmp_path / "round2.json"]

    command = loop._finalizer_command(
        args,
        output_dir=tmp_path / "gate",
        correction_receipts=receipts,
    )

    indices = [
        index
        for index, value in enumerate(command)
        if value == "--post-coverage-correction-receipt"
    ]
    assert [command[index + 1] for index in indices] == [str(path) for path in receipts]
    assert command[command.index("--baseline-public-comparison-report") + 1] == str(
        args.baseline_public_comparison_report
    )


@pytest.mark.parametrize("phase", ("block", "logits"))
def test_phase_correction_commands_preserve_phase_objective(
    tmp_path: Path,
    phase: str,
) -> None:
    args = _args(tmp_path)
    args.phase = phase
    if phase == "logits":
        args.baseline_public_comparison_report = None
    receipt = tmp_path / "round1.json"

    finalizer = loop._finalizer_command(
        args,
        output_dir=tmp_path / "gate",
        correction_receipts=[receipt],
    )
    correction = loop._correction_command(
        args,
        round_index=1,
        admission_gate=tmp_path / "failed.json",
        init_checkpoint=tmp_path / "init.pt",
        run_dir=tmp_path / "run",
    )

    assert finalizer[finalizer.index("--phase") + 1] == phase
    assert correction[correction.index("--phase") + 1] == phase
    assert "--post-coverage-correction-receipt" in finalizer
    if phase == "block":
        assert "--baseline-public-comparison-report" in finalizer
    else:
        assert "--baseline-public-comparison-report" not in finalizer

    gate = tmp_path / "gate" / "phase_gate.json"
    checkpoint = tmp_path / f"{phase}.pt"
    promotion = tmp_path / f"{phase}-promotion.json"
    gate.parent.mkdir(exist_ok=True)
    gate.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(phase.encode())
    promotion.write_text("{}\n", encoding="utf-8")
    selection = loop._selection_payload(
        round_index=1,
        gate_dir=gate.parent,
        gate={"checkpoint_path": str(checkpoint.resolve())},
        promotion=promotion,
        phase=phase,
    )
    assert selection["artifact"] == "phase_gate_selection"
    assert selection["phase"] == phase
    assert selection["checkpoint_sha256"] == loop.sha256_file(checkpoint)


def test_retention_loop_runs_failed_round_then_passing_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    init_checkpoint = tmp_path / "long.pt"
    round1_checkpoint = tmp_path / "round1.pt"
    round2_checkpoint = tmp_path / "round2.pt"
    for path in (init_checkpoint, round1_checkpoint, round2_checkpoint):
        path.write_bytes(path.name.encode())
    original_gate = args.original_gate_dir / "phase_gate.json"
    original_gate.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_validate_gate(path: Path) -> dict[str, object]:
        if path == original_gate:
            return {"gate_passed": False, "checkpoint_path": str(init_checkpoint)}
        if path.parent.name == "round_01":
            return {"gate_passed": False, "checkpoint_path": str(round1_checkpoint)}
        if path.parent.name == "round_02":
            return {"gate_passed": True, "checkpoint_path": str(round2_checkpoint)}
        raise AssertionError(path)

    def fake_run(
        command: list[str],
        *,
        dry_run: bool,
        allow_failure: bool = False,
    ) -> int:
        assert dry_run is False
        commands.append(command)
        if str(loop.CORRECTION_RUNNER) in command:
            run_dir = Path(command[command.index("--output-dir") + 1])
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "correction_receipt.json").write_text("{}\n", encoding="utf-8")
            return 0
        if str(loop.FINALIZER) in command:
            gate_dir = Path(command[command.index("--output-dir") + 1])
            gate_dir.mkdir(parents=True, exist_ok=True)
            (gate_dir / "phase_gate.json").write_text("{}\n", encoding="utf-8")
            return 1 if gate_dir.name == "round_01" else 0
        raise AssertionError(command)

    def fake_ensure_promotion(**kwargs: object) -> Path:
        gate_dir = Path(str(kwargs["gate_dir"]))
        promotion = gate_dir / "mixer_promotion_receipt.json"
        promotion.write_text("{}\n", encoding="utf-8")
        return promotion

    monkeypatch.setattr(loop, "_validate_gate", fake_validate_gate)
    monkeypatch.setattr(loop, "_run", fake_run)
    monkeypatch.setattr(loop, "_ensure_promotion", fake_ensure_promotion)

    selected = loop.run_retention_loop(args)

    assert selected == args.selection
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    assert selection["correction_round"] == 2
    assert selection["checkpoint_path"] == str(round2_checkpoint.resolve())
    correction_commands = [
        command for command in commands if str(loop.CORRECTION_RUNNER) in command
    ]
    assert len(correction_commands) == 2
    finalizer_commands = [command for command in commands if str(loop.FINALIZER) in command]
    assert len(finalizer_commands) == 2
    round2_finalizer = finalizer_commands[-1]
    assert round2_finalizer.count("--post-coverage-correction-receipt") == 2


def test_retention_loop_rejects_failed_gate_with_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    checkpoint = tmp_path / "long.pt"
    checkpoint.write_bytes(b"long")
    gate = args.original_gate_dir / "phase_gate.json"
    gate.write_text("{}\n", encoding="utf-8")
    (args.original_gate_dir / "mixer_promotion_receipt.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loop,
        "_validate_gate",
        lambda path: {"gate_passed": False, "checkpoint_path": str(checkpoint)},
    )

    with pytest.raises(ValueError, match="failed Mixer gate must not have a promotion"):
        loop.run_retention_loop(args)
