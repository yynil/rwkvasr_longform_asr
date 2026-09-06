#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from scripts.build_stage211_sft_correction_profile import (
        DEFAULT_OUTPUT_ROOT as DEFAULT_PROFILE_ROOT,
        build_correction_profile,
        validate_correction_profile,
    )
    from scripts.evaluate_stage211_sft_correction import (
        validate_correction_evaluation_report,
    )
    from scripts.run_stage211_labeled_sft import (
        _validate_completion as validate_full_sft_completion,
    )
    from scripts.run_stage211_sft_correction import validate_completion_receipt
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from build_stage211_sft_correction_profile import (
        DEFAULT_OUTPUT_ROOT as DEFAULT_PROFILE_ROOT,
        build_correction_profile,
        validate_correction_profile,
    )
    from evaluate_stage211_sft_correction import validate_correction_evaluation_report
    from run_stage211_labeled_sft import (
        _validate_completion as validate_full_sft_completion,
    )
    from run_stage211_sft_correction import validate_completion_receipt


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MAX_ROUNDS = 3
DEFAULT_RUN_ROOT = (
    Path.home() / "rwkvasr_runs" / "stage211_full_alignment" / "stage211d_sft_correction"
)
DEFAULT_EVAL_ROOT = Path.home() / "rwkvasr_eval" / "stage211_phase_gates" / "sft_correction"
DEFAULT_CONFIG_ROOT = Path.home() / "rwkvasr_configs" / "stage211_full_alignment"
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"


def _run(command: list[str], *, dry_run: bool) -> None:
    print(f"[stage211-sft-correction-loop] command={shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _evaluation_command(
    args: argparse.Namespace,
    *,
    completion_receipt: Path,
    eval_dir: Path,
) -> list[str]:
    return [
        str(PYTHON),
        str(REPO_ROOT / "scripts" / "evaluate_stage211_sft_correction.py"),
        "--completion-receipt",
        str(completion_receipt),
        "--full-sft-failed-report",
        str(args.full_sft_failed_report.expanduser().resolve()),
        "--output-dir",
        str(eval_dir),
        "--final-output-dir",
        str(args.final_output_dir.expanduser().resolve()),
        "--public-manifest-dir",
        str(args.public_manifest_dir.expanduser().resolve()),
        "--nano-prediction-dir",
        str(args.nano_prediction_dir.expanduser().resolve()),
        "--public-overlap-receipt",
        str(args.public_overlap_receipt.expanduser().resolve()),
        "--phase-gate-root",
        str(args.phase_gate_root.expanduser().resolve()),
        "--mixer-gate-selection",
        str(args.mixer_gate_selection.expanduser().resolve()),
        "--block-gate-selection",
        str(args.block_gate_selection.expanduser().resolve()),
        "--logits-gate-selection",
        str(args.logits_gate_selection.expanduser().resolve()),
        "--initialization-receipt",
        str(args.initialization_receipt.expanduser().resolve()),
        "--calibration-reuse-receipt",
        str(args.calibration_reuse_receipt.expanduser().resolve()),
        "--devices",
        str(args.devices),
    ]


def run_loop(args: argparse.Namespace) -> Path | None:
    full_completion_path = args.full_sft_completion.expanduser().resolve()
    full_failed_report_path = args.full_sft_failed_report.expanduser().resolve()
    full_completion, full_checkpoint = validate_full_sft_completion(
        full_completion_path,
        require_full_profile=True,
    )
    full_profile_path = Path(
        str(full_completion.get("labeled_profile_receipt_path") or "")
    ).resolve()
    if not full_profile_path.is_file():
        raise ValueError("Full Stage211D completion does not bind its schema-v2 profile.")
    profile_root = args.correction_profile_root.expanduser().resolve()
    profile_path = profile_root / "stage211_sft_correction_profile.json"
    if profile_path.is_file():
        validate_correction_profile(
            profile_path,
            expected_full_profile_path=full_profile_path,
        )
    elif args.dry_run:
        print(
            f"[stage211-sft-correction-loop] dry-run would build profile={profile_path}",
            flush=True,
        )
    else:
        build_correction_profile(
            full_profile_path=full_profile_path,
            output_root=profile_root,
        )
    if args.dry_run:
        return None

    current_checkpoint = full_checkpoint
    current_admission = full_failed_report_path
    run_root = args.run_root.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    for round_index in range(1, MAX_ROUNDS + 1):
        run_dir = run_root / f"round_{round_index:02d}"
        completion_receipt = run_dir / "sft_correction_complete.json"
        if completion_receipt.is_file():
            completion = validate_completion_receipt(
                completion_receipt,
                expected_round=round_index,
            )
        else:
            command = [
                str(PYTHON),
                str(REPO_ROOT / "scripts" / "run_stage211_sft_correction.py"),
                "--round",
                str(round_index),
                "--correction-profile",
                str(profile_path),
                "--full-sft-completion",
                str(full_completion_path),
                "--admission-report",
                str(current_admission),
                "--init-checkpoint",
                str(current_checkpoint),
                "--nano-checkpoint",
                str(args.nano_checkpoint.expanduser().resolve()),
                "--output-dir",
                str(run_dir),
                "--config-dir",
                str(args.config_dir.expanduser().resolve()),
                "--master-port",
                str(int(args.master_port) + round_index - 1),
                "--max-peak-reserved-gib",
                str(float(args.max_peak_reserved_gib)),
            ]
            _run(command, dry_run=False)
            completion = validate_completion_receipt(
                completion_receipt,
                expected_round=round_index,
            )

        eval_dir = eval_root / f"round_{round_index:02d}"
        evaluation_path = eval_dir / "correction_evaluation.json"
        evaluation_command = _evaluation_command(
            args,
            completion_receipt=completion_receipt,
            eval_dir=eval_dir,
        )
        if evaluation_path.is_file():
            evaluation = validate_correction_evaluation_report(
                evaluation_path,
                expected_round=round_index,
                expected_full_completion_path=full_completion_path,
                expected_correction_profile_path=profile_path,
            )
            if evaluation.get("gate_passed") is True:
                _run(evaluation_command, dry_run=False)
                evaluation = validate_correction_evaluation_report(
                    evaluation_path,
                    expected_round=round_index,
                    expected_full_completion_path=full_completion_path,
                    expected_correction_profile_path=profile_path,
                    require_passed=True,
                )
        else:
            _run(evaluation_command, dry_run=False)
            evaluation = validate_correction_evaluation_report(
                evaluation_path,
                expected_round=round_index,
                expected_full_completion_path=full_completion_path,
                expected_correction_profile_path=profile_path,
            )
        if evaluation.get("gate_passed") is True:
            final_path = args.final_output_dir.expanduser().resolve() / "stage211_complete.json"
            if not final_path.is_file():
                raise ValueError("Passed Stage211D correction did not produce final report.")
            print(
                f"[stage211-sft-correction-loop] passed round={round_index} final={final_path}",
                flush=True,
            )
            return final_path
        current_checkpoint = Path(str(completion["completion_checkpoint_path"])).resolve()
        current_admission = evaluation_path

    raise ValueError(
        "Stage211D remained below the strict five-dataset bilingual/Nano gate after "
        f"{MAX_ROUNDS} complete correction rounds."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run up to three strict Stage211D balanced-label correction rounds."
    )
    parser.add_argument("--full-sft-completion", type=Path, required=True)
    parser.add_argument("--full-sft-failed-report", type=Path, required=True)
    parser.add_argument("--correction-profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--final-output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    parser.add_argument("--public-manifest-dir", type=Path, required=True)
    parser.add_argument("--nano-prediction-dir", type=Path, required=True)
    parser.add_argument("--public-overlap-receipt", type=Path, required=True)
    parser.add_argument("--phase-gate-root", type=Path, required=True)
    parser.add_argument("--mixer-gate-selection", type=Path, required=True)
    parser.add_argument("--block-gate-selection", type=Path, required=True)
    parser.add_argument("--logits-gate-selection", type=Path, required=True)
    parser.add_argument("--initialization-receipt", type=Path, required=True)
    parser.add_argument("--calibration-reuse-receipt", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=29651)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=22.0)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_peak_reserved_gib <= 0:
        parser.error("--max-peak-reserved-gib must be positive")
    run_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
