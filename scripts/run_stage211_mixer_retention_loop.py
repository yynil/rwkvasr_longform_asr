from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS,
    STAGE211_RETENTION_CORRECTION_MAX_ROUNDS,
    build_stage211_correction_extension_decision,
    sha256_file,
    stage211_correction_admission_mode,
    validate_stage211_correction_extension_decision,
    validate_stage211_phase_gate_report,
)

try:
    from scripts.run_stage211_strict_chained_alignment import (
        _validate_promotion_receipt,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage211_strict_chained_alignment import (
        _validate_promotion_receipt,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
FINALIZER = REPO_ROOT / "scripts" / "finalize_stage211_phase.py"
CORRECTION_RUNNER = REPO_ROOT / "scripts" / "run_stage211_retention_correction.py"
PROMOTION_BUILDER = REPO_ROOT / "scripts" / "create_stage211_promotion_receipt.py"
DEFAULT_PHASE_ROOT = (
    Path.home() / "rwkvasr_runs" / "stage211_full_alignment" / "stage211a_mixer_full_data_3ep"
)
DEFAULT_ORIGINAL_GATE_DIR = Path.home() / "rwkvasr_eval" / "stage211_phase_gates" / "mixer"
DEFAULT_CORRECTION_RUN_ROOT = (
    Path.home()
    / "rwkvasr_runs"
    / "stage211_full_alignment"
    / "stage211a_mixer_retention_correction"
)
DEFAULT_CORRECTION_GATE_ROOT = (
    Path.home() / "rwkvasr_eval" / "stage211_phase_gates" / "mixer_retention"
)
DEFAULT_REPLAY_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "retention_replay_v3"
    / "receipt.json"
)
DEFAULT_STRATIFIED_HIDDEN_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "stratified_hidden_eval_v3"
    / "receipt.json"
)
DEFAULT_PUBLIC_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"
DEFAULT_NANO_EVAL_DIR = Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512"
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
DEFAULT_SELECTION = Path.home() / "rwkvasr_eval" / "stage211_phase_gates" / "mixer_selected.json"
PHASE_TARGETS = {"mixer": "block", "block": "logits", "logits": "sft"}


def _phase(args: argparse.Namespace) -> str:
    phase = str(getattr(args, "phase", "mixer"))
    if phase not in PHASE_TARGETS:
        raise ValueError(f"Unsupported Stage211 correction-loop phase: {phase!r}")
    return phase


def _selection_artifact(phase: str) -> str:
    return "mixer_gate_selection" if phase == "mixer" else "phase_gate_selection"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different Stage211 selection: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _run(command: list[str], *, dry_run: bool, allow_failure: bool = False) -> int:
    print(f"[stage211-retention-loop] command={shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"Stage211 retention-loop command failed with code={result.returncode}: "
            f"{shlex.join(command)}"
        )
    return int(result.returncode)


def _finalizer_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    correction_receipts: list[Path],
) -> list[str]:
    phase = _phase(args)
    command = [
        str(PYTHON),
        str(FINALIZER),
        "--phase",
        phase,
        "--phase-root",
        str(args.phase_root),
        "--output-dir",
        str(output_dir),
        "--public-manifest-dir",
        str(args.public_manifest_dir),
        "--nano-prediction-dir",
        str(args.nano_prediction_dir),
        "--stratified-hidden-receipt",
        str(
            getattr(
                args,
                "stratified_hidden_receipt",
                DEFAULT_STRATIFIED_HIDDEN_RECEIPT,
            )
        ),
        "--devices",
        str(args.devices),
    ]
    if args.baseline_public_comparison_report is None:
        raise ValueError(f"Stage211 {phase} correction loop requires a public baseline.")
    command.extend(
        (
            "--baseline-public-comparison-report",
            str(args.baseline_public_comparison_report),
        )
    )
    for receipt in correction_receipts:
        command.extend(("--post-coverage-correction-receipt", str(receipt)))
    return command


def _correction_command(
    args: argparse.Namespace,
    *,
    round_index: int,
    admission_gate: Path,
    init_checkpoint: Path,
    run_dir: Path,
) -> list[str]:
    return [
        str(PYTHON),
        str(CORRECTION_RUNNER),
        "--phase",
        _phase(args),
        "--round",
        str(round_index),
        "--replay-receipt",
        str(args.replay_receipt),
        "--admission-gate",
        str(admission_gate),
        "--init-checkpoint",
        str(init_checkpoint),
        "--nano-checkpoint",
        str(args.nano_checkpoint),
        "--output-dir",
        str(run_dir),
        "--config-dir",
        str(args.config_dir),
        "--master-port",
        str(args.master_port),
        "--auto-batch-profile",
        "--batch-profile-master-port",
        str(args.master_port + 100),
    ]


def _validate_gate(gate_path: Path, *, phase: str = "mixer") -> dict[str, Any]:
    raw = _load_json(gate_path, label=f"Stage211 {phase} phase gate")
    checkpoint = Path(str(raw.get("checkpoint_path") or "")).resolve()
    return validate_stage211_phase_gate_report(
        gate_path,
        expected_phase=phase,
        checkpoint_path=checkpoint,
        require_passed=False,
    )


def _validate_gate_for_phase(gate_path: Path, *, phase: str) -> dict[str, Any]:
    if phase == "mixer":
        return _validate_gate(gate_path)
    return _validate_gate(gate_path, phase=phase)


def _correction_extension_decision(
    *,
    phase: str,
    completed_round: int,
    prior_gate_path: Path,
    prior_gate: dict[str, Any],
    current_gate_path: Path,
    current_gate: dict[str, Any],
    max_rounds: int,
) -> dict[str, Any]:
    prior_extension_decision_path: Path | None = None
    prior_extension_decision: dict[str, Any] | None = None
    if completed_round > STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS:
        prior_extension_decision_path = (
            prior_gate_path.parent / "correction_extension_decision.json"
        )
        prior_extension_decision = validate_stage211_correction_extension_decision(
            prior_extension_decision_path,
            phase=phase,
            next_round=completed_round,
            admission_gate_path=prior_gate_path,
            admission_gate=prior_gate,
        )
    return build_stage211_correction_extension_decision(
        phase=phase,
        completed_round=completed_round,
        prior_gate_path=prior_gate_path,
        prior_gate=prior_gate,
        current_gate_path=current_gate_path,
        current_gate=current_gate,
        max_rounds=max_rounds,
        prior_extension_decision_path=prior_extension_decision_path,
        prior_extension_decision=prior_extension_decision,
    )


def _ensure_promotion(
    *,
    gate_dir: Path,
    gate: dict[str, Any],
    nano_checkpoint: Path,
    dry_run: bool,
    phase: str = "mixer",
) -> Path:
    checkpoint = Path(str(gate["checkpoint_path"])).resolve()
    gate_path = gate_dir / "phase_gate.json"
    promotion = gate_dir / f"{phase}_promotion_receipt.json"
    if not promotion.is_file():
        _run(
            [
                str(PYTHON),
                str(PROMOTION_BUILDER),
                "--source-phase",
                phase,
                "--checkpoint",
                str(checkpoint),
                "--gate-report",
                str(gate_path),
                "--output",
                str(promotion),
                "--confirm-gate-passed",
            ],
            dry_run=dry_run,
        )
    if dry_run:
        return promotion
    _validate_promotion_receipt(
        receipt_path=promotion,
        target_phase=PHASE_TARGETS[phase],
        checkpoint_path=checkpoint,
        nano_checkpoint_path=nano_checkpoint,
    )
    return promotion


def _selection_payload(
    *,
    round_index: int,
    gate_dir: Path,
    gate: dict[str, Any],
    promotion: Path,
    phase: str = "mixer",
) -> dict[str, Any]:
    gate_path = gate_dir / "phase_gate.json"
    checkpoint = Path(str(gate["checkpoint_path"])).resolve()
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": _selection_artifact(phase),
        "phase": phase,
        "correction_round": round_index,
        "gate_dir": str(gate_dir),
        "gate_path": str(gate_path),
        "gate_sha256": sha256_file(gate_path),
        "promotion_receipt_path": str(promotion),
        "promotion_receipt_sha256": sha256_file(promotion),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _validate_existing_selection(
    selection_path: Path,
    *,
    nano_checkpoint: Path,
    phase: str = "mixer",
) -> dict[str, Any]:
    selection = _load_json(selection_path, label=f"Stage211 selected {phase} gate")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": _selection_artifact(phase),
        "phase": phase,
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 selected {phase} gate contract mismatch.")
    gate_dir = Path(str(selection.get("gate_dir") or "")).resolve()
    gate = _validate_gate_for_phase(gate_dir / "phase_gate.json", phase=phase)
    if gate.get("gate_passed") is not True:
        raise ValueError(f"Stage211 selected {phase} gate is not passing.")
    promotion = Path(str(selection.get("promotion_receipt_path") or "")).resolve()
    checkpoint = Path(str(gate["checkpoint_path"])).resolve()
    _validate_promotion_receipt(
        receipt_path=promotion,
        target_phase=PHASE_TARGETS[phase],
        checkpoint_path=checkpoint,
        nano_checkpoint_path=nano_checkpoint,
    )
    rebuilt = _selection_payload(
        round_index=int(selection.get("correction_round", -1)),
        gate_dir=gate_dir,
        gate=gate,
        promotion=promotion,
        phase=phase,
    )
    if rebuilt != selection:
        raise ValueError(f"Stage211 selected {phase} gate binding changed.")
    return selection


def run_retention_loop(args: argparse.Namespace) -> Path | None:
    phase = _phase(args)
    if args.selection.is_file():
        selected = _validate_existing_selection(
            args.selection,
            nano_checkpoint=args.nano_checkpoint,
            phase=phase,
        )
        print(
            "[stage211-correction-loop] reused selection "
            f"phase={phase} "
            f"round={selected['correction_round']} "
            f"checkpoint={selected['checkpoint_path']}",
            flush=True,
        )
        return args.selection

    original_gate_path = args.original_gate_dir / "phase_gate.json"
    if not original_gate_path.is_file():
        code = _run(
            _finalizer_command(
                args,
                output_dir=args.original_gate_dir,
                correction_receipts=[],
            ),
            dry_run=bool(args.dry_run),
            allow_failure=True,
        )
        if args.dry_run:
            return None
        if code == 0 and not original_gate_path.is_file():
            raise ValueError(f"Stage211 original {phase} finalizer produced no phase gate.")

    prior_gate_path = original_gate_path
    last_failed_gate_path: Path | None = None
    correction_receipts: list[Path] = []
    for round_index in range(0, int(args.max_rounds) + 1):
        gate_dir = (
            args.original_gate_dir
            if round_index == 0
            else args.correction_gate_root / f"round_{round_index:02d}"
        )
        gate_path = gate_dir / "phase_gate.json"
        if round_index > 0:
            run_dir = args.correction_run_root / f"round_{round_index:02d}"
            receipt = run_dir / "correction_receipt.json"
            if not receipt.is_file():
                prior_gate = _validate_gate_for_phase(prior_gate_path, phase=phase)
                admission_mode = stage211_correction_admission_mode(
                    gate_passed=prior_gate.get("gate_passed"),
                    round_index=round_index,
                )
                init_checkpoint = Path(str(prior_gate["checkpoint_path"])).resolve()
                print(
                    "[stage211-correction-loop] correction admission "
                    f"phase={phase} round={round_index} mode={admission_mode} "
                    f"gate={prior_gate_path}",
                    flush=True,
                )
                _run(
                    _correction_command(
                        args,
                        round_index=round_index,
                        admission_gate=prior_gate_path,
                        init_checkpoint=init_checkpoint,
                        run_dir=run_dir,
                    ),
                    dry_run=bool(args.dry_run),
                )
            correction_receipts.append(receipt)
            if not gate_path.is_file():
                code = _run(
                    _finalizer_command(
                        args,
                        output_dir=gate_dir,
                        correction_receipts=correction_receipts,
                    ),
                    dry_run=bool(args.dry_run),
                    allow_failure=True,
                )
                if args.dry_run:
                    return None
                if code == 0 and not gate_path.is_file():
                    raise ValueError(
                        f"Stage211 correction round {round_index} produced no phase gate."
                    )

        gate = _validate_gate_for_phase(gate_path, phase=phase)
        if gate.get("gate_passed") is True:
            if 0 < round_index < STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS:
                print(
                    "[stage211-correction-loop] early gate pass requires mandatory continuation "
                    f"phase={phase} completed_round={round_index} "
                    f"guaranteed_rounds={STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS}",
                    flush=True,
                )
                prior_gate_path = gate_path
                continue
            promotion = _ensure_promotion(
                gate_dir=gate_dir,
                gate=gate,
                nano_checkpoint=args.nano_checkpoint,
                dry_run=bool(args.dry_run),
                phase=phase,
            )
            if args.dry_run:
                return None
            selection = _selection_payload(
                round_index=round_index,
                gate_dir=gate_dir,
                gate=gate,
                promotion=promotion,
                phase=phase,
            )
            _write_immutable_json(args.selection, selection)
            print(
                "[stage211-correction-loop] passed "
                f"phase={phase} round={round_index} checkpoint={selection['checkpoint_path']} "
                f"selection={args.selection}",
                flush=True,
            )
            return args.selection
        if (gate_dir / f"{phase}_promotion_receipt.json").exists():
            phase_label = "Mixer" if phase == "mixer" else phase.capitalize()
            raise ValueError(
                f"Stage211 failed {phase_label} gate must not have a promotion receipt."
            )
        if round_index >= STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS and round_index < int(
            args.max_rounds
        ):
            if last_failed_gate_path is None:
                raise ValueError("Stage211 correction extension lacks a prior failed gate.")
            progress_gate_path = last_failed_gate_path
            prior_gate = _validate_gate_for_phase(progress_gate_path, phase=phase)
            decision = _correction_extension_decision(
                phase=phase,
                completed_round=round_index,
                prior_gate_path=progress_gate_path,
                prior_gate=prior_gate,
                current_gate_path=gate_path,
                current_gate=gate,
                max_rounds=int(args.max_rounds),
            )
            decision_path = gate_dir / "correction_extension_decision.json"
            _write_immutable_json(decision_path, decision)
            print(
                "[stage211-correction-loop] extension decision "
                f"phase={phase} completed_round={round_index} "
                f"continue={str(decision['continue_training']).lower()} "
                f"improved={','.join(decision['improved_metrics']) or '-'} "
                f"receipt={decision_path}",
                flush=True,
            )
            if decision["continue_training"] is not True:
                raise ValueError(
                    f"Stage211 {phase} correction stalled after round {round_index}; "
                    f"extension evidence is preserved at {decision_path}."
                )
        last_failed_gate_path = gate_path
        prior_gate_path = gate_path

    raise ValueError(
        f"Stage211 {phase} gate failed after {args.max_rounds} complete correction rounds."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run a restart-safe Stage211 post-coverage correction/evaluation loop.")
    )
    parser.add_argument("--phase", choices=tuple(PHASE_TARGETS), default="mixer")
    parser.add_argument("--phase-root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--original-gate-dir", type=Path, default=DEFAULT_ORIGINAL_GATE_DIR)
    parser.add_argument(
        "--correction-run-root",
        type=Path,
        default=DEFAULT_CORRECTION_RUN_ROOT,
    )
    parser.add_argument(
        "--correction-gate-root",
        type=Path,
        default=DEFAULT_CORRECTION_GATE_ROOT,
    )
    parser.add_argument("--replay-receipt", type=Path, default=DEFAULT_REPLAY_RECEIPT)
    parser.add_argument(
        "--stratified-hidden-receipt",
        type=Path,
        default=DEFAULT_STRATIFIED_HIDDEN_RECEIPT,
    )
    parser.add_argument(
        "--public-manifest-dir",
        type=Path,
        default=DEFAULT_PUBLIC_MANIFEST_DIR,
    )
    parser.add_argument(
        "--nano-prediction-dir",
        type=Path,
        default=DEFAULT_NANO_EVAL_DIR / "predictions",
    )
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    parser.add_argument(
        "--baseline-public-comparison-report",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path.home() / "rwkvasr_configs" / "stage211_full_alignment",
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--max-rounds",
        type=int,
        choices=range(1, STAGE211_RETENTION_CORRECTION_MAX_ROUNDS + 1),
        default=STAGE211_RETENTION_CORRECTION_MAX_ROUNDS,
    )
    parser.add_argument("--master-port", type=int, default=29641)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument(
        "--validate-selection-only",
        action="store_true",
        help="Deep-validate an existing selection without running evaluation or training.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in (
        "phase_root",
        "original_gate_dir",
        "correction_run_root",
        "correction_gate_root",
        "replay_receipt",
        "stratified_hidden_receipt",
        "public_manifest_dir",
        "nano_prediction_dir",
        "nano_checkpoint",
        "baseline_public_comparison_report",
        "config_dir",
        "selection",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.validate_selection_only:
        if not args.selection.is_file():
            raise ValueError(f"Stage211 selected {_phase(args)} gate is missing: {args.selection}")
        selected = _validate_existing_selection(
            args.selection,
            nano_checkpoint=args.nano_checkpoint,
            phase=_phase(args),
        )
        print(
            "[stage211-correction-loop] selection validation passed "
            f"phase={selected['phase']} "
            f"round={selected['correction_round']} "
            f"checkpoint={selected['checkpoint_path']}",
            flush=True,
        )
        return 0
    run_retention_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
