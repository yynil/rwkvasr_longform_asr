from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    STAGE211_RETENTION_CORRECTION_MAX_ROUNDS,
    sha256_file,
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
    / "retention_replay_v1"
    / "receipt.json"
)
DEFAULT_PUBLIC_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"
DEFAULT_NANO_EVAL_DIR = Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512"
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
DEFAULT_BASELINE_PUBLIC_REPORT = (
    Path.home()
    / "rwkvasr_eval"
    / "stage211_calibration_selected_full"
    / "public"
    / "nano_comparison.json"
)
DEFAULT_SELECTION = Path.home() / "rwkvasr_eval" / "stage211_phase_gates" / "mixer_selected.json"


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
    command = [
        str(PYTHON),
        str(FINALIZER),
        "--phase",
        "mixer",
        "--phase-root",
        str(args.phase_root),
        "--output-dir",
        str(output_dir),
        "--public-manifest-dir",
        str(args.public_manifest_dir),
        "--nano-prediction-dir",
        str(args.nano_prediction_dir),
        "--baseline-public-comparison-report",
        str(args.baseline_public_comparison_report),
        "--devices",
        str(args.devices),
    ]
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
    ]


def _validate_gate(gate_path: Path) -> dict[str, Any]:
    raw = _load_json(gate_path, label="Stage211 Mixer phase gate")
    checkpoint = Path(str(raw.get("checkpoint_path") or "")).resolve()
    return validate_stage211_phase_gate_report(
        gate_path,
        expected_phase="mixer",
        checkpoint_path=checkpoint,
        require_passed=False,
    )


def _ensure_promotion(
    *,
    gate_dir: Path,
    gate: dict[str, Any],
    nano_checkpoint: Path,
    dry_run: bool,
) -> Path:
    checkpoint = Path(str(gate["checkpoint_path"])).resolve()
    gate_path = gate_dir / "phase_gate.json"
    promotion = gate_dir / "mixer_promotion_receipt.json"
    if not promotion.is_file():
        _run(
            [
                str(PYTHON),
                str(PROMOTION_BUILDER),
                "--source-phase",
                "mixer",
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
        target_phase="block",
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
) -> dict[str, Any]:
    gate_path = gate_dir / "phase_gate.json"
    checkpoint = Path(str(gate["checkpoint_path"])).resolve()
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "mixer_gate_selection",
        "phase": "mixer",
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
) -> dict[str, Any]:
    selection = _load_json(selection_path, label="Stage211 selected Mixer gate")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "mixer_gate_selection",
        "phase": "mixer",
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 selected Mixer gate contract mismatch.")
    gate_dir = Path(str(selection.get("gate_dir") or "")).resolve()
    gate = _validate_gate(gate_dir / "phase_gate.json")
    if gate.get("gate_passed") is not True:
        raise ValueError("Stage211 selected Mixer gate is not passing.")
    promotion = Path(str(selection.get("promotion_receipt_path") or "")).resolve()
    checkpoint = Path(str(gate["checkpoint_path"])).resolve()
    _validate_promotion_receipt(
        receipt_path=promotion,
        target_phase="block",
        checkpoint_path=checkpoint,
        nano_checkpoint_path=nano_checkpoint,
    )
    rebuilt = _selection_payload(
        round_index=int(selection.get("correction_round", -1)),
        gate_dir=gate_dir,
        gate=gate,
        promotion=promotion,
    )
    if rebuilt != selection:
        raise ValueError("Stage211 selected Mixer gate binding changed.")
    return selection


def run_retention_loop(args: argparse.Namespace) -> Path | None:
    if args.selection.is_file():
        selected = _validate_existing_selection(
            args.selection,
            nano_checkpoint=args.nano_checkpoint,
        )
        print(
            "[stage211-retention-loop] reused selection "
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
            raise ValueError("Stage211 original Mixer finalizer produced no phase gate.")

    prior_gate_path = original_gate_path
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
                prior_gate = _validate_gate(prior_gate_path)
                if prior_gate.get("gate_passed") is not False:
                    raise ValueError("Stage211 correction admission gate unexpectedly passed.")
                init_checkpoint = Path(str(prior_gate["checkpoint_path"])).resolve()
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

        gate = _validate_gate(gate_path)
        if gate.get("gate_passed") is True:
            promotion = _ensure_promotion(
                gate_dir=gate_dir,
                gate=gate,
                nano_checkpoint=args.nano_checkpoint,
                dry_run=bool(args.dry_run),
            )
            if args.dry_run:
                return None
            selection = _selection_payload(
                round_index=round_index,
                gate_dir=gate_dir,
                gate=gate,
                promotion=promotion,
            )
            _write_immutable_json(args.selection, selection)
            print(
                "[stage211-retention-loop] passed "
                f"round={round_index} checkpoint={selection['checkpoint_path']} "
                f"selection={args.selection}",
                flush=True,
            )
            return args.selection
        if (gate_dir / "mixer_promotion_receipt.json").exists():
            raise ValueError("Stage211 failed Mixer gate must not have a promotion receipt.")
        prior_gate_path = gate_path

    raise ValueError(
        f"Stage211 Mixer retention gate failed after {args.max_rounds} complete rounds."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run the restart-safe Stage211A Mixer retention correction/evaluation loop.")
    )
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
        default=DEFAULT_BASELINE_PUBLIC_REPORT,
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
        "public_manifest_dir",
        "nano_prediction_dir",
        "nano_checkpoint",
        "baseline_public_comparison_report",
        "config_dir",
        "selection",
    ):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())
    run_retention_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
