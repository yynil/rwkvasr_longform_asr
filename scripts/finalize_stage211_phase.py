from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    STAGE211_AUDIO_CURRICULUM,
    sha256_file,
    validate_stage211_full_data_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
PUBLIC_EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_public_eval_benchmarks.sh"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_public_ctc_with_nano.py"
HIDDEN_GATE_SCRIPT = REPO_ROOT / "scripts" / "create_stage211_hidden_alignment_gate.py"
PHASE_GATE_SCRIPT = REPO_ROOT / "scripts" / "create_stage211_phase_gate.py"
PROMOTION_SCRIPT = REPO_ROOT / "scripts" / "create_stage211_promotion_receipt.py"
DEFAULT_PUBLIC_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"
DEFAULT_NANO_PREDICTION_DIR = (
    Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512" / "predictions"
)
DEFAULT_EVAL_ROOT = Path.home() / "rwkvasr_eval" / "stage211_phase_gates"
DATASETS = (
    "aishell1_test",
    "librispeech_test_clean",
    "librispeech_test_other",
    "commonvoice_en_test",
    "wenetspeech_test_net",
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(f"[stage211-finalize] command={shlex.join(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def _resolve_curriculum(
    *,
    phase: str,
    phase_root: Path,
) -> tuple[dict[str, Any], Path, list[Path]]:
    summary_path = phase_root / "curriculum_complete.json"
    summary = _load_json(summary_path, label="Stage211 full-phase curriculum summary")
    if (
        summary.get("pipeline") != "stage211"
        or summary.get("artifact") != "full_phase_curriculum"
        or summary.get("phase") != phase
        or summary.get("complete") is not True
    ):
        raise ValueError(f"Invalid Stage211 curriculum summary: {summary_path}")
    coverage = summary.get("full_data_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Stage211 curriculum summary lacks full_data_coverage.")
    checkpoint = Path(str(coverage.get("final_checkpoint_path") or "")).resolve()
    validate_stage211_full_data_coverage(
        coverage,
        phase=phase,
        checkpoint_path=checkpoint,
    )
    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 curriculum summary lacks segment receipts.")
    by_difficulty = {
        str(segment.get("difficulty")): segment for segment in segments if isinstance(segment, dict)
    }
    receipt_paths = [
        Path(str(by_difficulty[difficulty]["receipt_path"])).resolve()
        for difficulty in STAGE211_AUDIO_CURRICULUM
    ]
    return coverage, checkpoint, receipt_paths


def _run_public_eval(
    *,
    checkpoint: Path,
    output_dir: Path,
    manifest_dir: Path,
    devices: str,
    dry_run: bool,
) -> None:
    env = {
        **os.environ,
        "CHECKPOINT_PATH": str(checkpoint),
        "OUTPUT_DIR": str(output_dir),
        "MANIFEST_DIR": str(manifest_dir),
        "PREPARE_MANIFESTS": "0",
        "DEVICES": devices,
        "CTC_BATCH_SIZE": "4",
        "CTC_NUM_WORKERS": "0",
        "CTC_SHARD_STAGE2": "1",
        "CTC_LIMIT": "0",
        "RUN_AR": "0",
        "METRIC_NORMALIZATION": "ctc",
    }
    _run(
        ["bash", str(PUBLIC_EVAL_SCRIPT)],
        dry_run=dry_run,
        env=env,
    )


def _run_nano_comparison(
    *,
    checkpoint: Path,
    public_output_dir: Path,
    nano_prediction_dir: Path,
    comparison_json: Path,
    comparison_md: Path,
    dry_run: bool,
) -> None:
    command = [
        str(PYTHON),
        str(COMPARE_SCRIPT),
        "--student-prediction-dir",
        str(public_output_dir / "predictions"),
    ]
    for dataset in DATASETS:
        command.extend(
            (
                "--nano-prediction",
                f"{dataset}={nano_prediction_dir / f'{dataset}.ctc.jsonl'}",
            )
        )
    command.extend(
        (
            "--student-checkpoint",
            str(checkpoint),
            "--output-json",
            str(comparison_json),
            "--output-md",
            str(comparison_md),
            "--normalization",
            "ctc",
            "--max-relative-ratio",
            "1.20",
            "--max-absolute-gap-points",
            "3.0",
        )
    )
    _run(command, dry_run=dry_run)


def finalize_phase(args: argparse.Namespace) -> Path:
    phase = str(args.phase)
    phase_root = args.phase_root.expanduser().resolve()
    manifest_dir = args.public_manifest_dir.expanduser().resolve()
    nano_prediction_dir = args.nano_prediction_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (DEFAULT_EVAL_ROOT / phase).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _, checkpoint, receipt_paths = _resolve_curriculum(
        phase=phase,
        phase_root=phase_root,
    )
    public_output = output_dir / "public"
    comparison_json = output_dir / "nano_comparison.json"
    comparison_md = output_dir / "nano_comparison.md"
    _run_public_eval(
        checkpoint=checkpoint,
        output_dir=public_output,
        manifest_dir=manifest_dir,
        devices=str(args.devices),
        dry_run=bool(args.dry_run),
    )
    _run_nano_comparison(
        checkpoint=checkpoint,
        public_output_dir=public_output,
        nano_prediction_dir=nano_prediction_dir,
        comparison_json=comparison_json,
        comparison_md=comparison_md,
        dry_run=bool(args.dry_run),
    )

    hidden_gate_path: Path | None = None
    if phase in {"mixer", "block"}:
        hidden_gate_path = output_dir / "hidden_gate.json"
        baseline_report = phase_root / "easy" / "step_eval_baseline.yaml"
        long_step = int(STAGE211_AUDIO_CURRICULUM["long"]["steps"])
        candidate_report = phase_root / "long" / f"step_eval_layers_step-{long_step}.yaml"
        _run(
            [
                str(PYTHON),
                str(HIDDEN_GATE_SCRIPT),
                "--phase",
                phase,
                "--baseline-report",
                str(baseline_report),
                "--candidate-report",
                str(candidate_report),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(hidden_gate_path),
            ],
            dry_run=bool(args.dry_run),
        )

    phase_gate_path = output_dir / "phase_gate.json"
    phase_gate_command = [
        str(PYTHON),
        str(PHASE_GATE_SCRIPT),
        "--phase",
        phase,
        "--checkpoint",
        str(checkpoint),
        "--public-comparison-report",
        str(comparison_json),
        "--manifest-dir",
        str(manifest_dir),
        "--output",
        str(phase_gate_path),
    ]
    for receipt_path in receipt_paths:
        phase_gate_command.extend(("--coverage-receipt", str(receipt_path)))
    if hidden_gate_path is not None:
        phase_gate_command.extend(("--alignment-report", str(hidden_gate_path)))
    if phase in {"mixer", "block"}:
        if args.baseline_public_comparison_report is None:
            raise ValueError(
                f"Stage211 {phase} finalization requires --baseline-public-comparison-report."
            )
        phase_gate_command.extend(
            (
                "--baseline-public-comparison-report",
                str(args.baseline_public_comparison_report.expanduser().resolve()),
            )
        )
    _run(phase_gate_command, dry_run=bool(args.dry_run))

    promotion_receipt = output_dir / f"{phase}_promotion_receipt.json"
    _run(
        [
            str(PYTHON),
            str(PROMOTION_SCRIPT),
            "--source-phase",
            phase,
            "--checkpoint",
            str(checkpoint),
            "--gate-report",
            str(phase_gate_path),
            "--output",
            str(promotion_receipt),
            "--confirm-gate-passed",
        ],
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run:
        report = _load_json(phase_gate_path, label="Stage211 phase gate")
        if report.get("gate_passed") is not True or report.get("checkpoint_sha256") != sha256_file(
            checkpoint
        ):
            raise ValueError(f"Stage211 {phase} phase gate did not pass.")
    print(
        f"[stage211-finalize] complete phase={phase} checkpoint={checkpoint} "
        f"gate={phase_gate_path} promotion={promotion_receipt}",
        flush=True,
    )
    return promotion_receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and gate one completed Stage211 A/B/C full-data phase on "
            "the complete normalized English WER and Chinese CER suite."
        )
    )
    parser.add_argument("--phase", choices=("mixer", "block", "logits"), required=True)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--public-manifest-dir",
        type=Path,
        default=DEFAULT_PUBLIC_MANIFEST_DIR,
    )
    parser.add_argument(
        "--nano-prediction-dir",
        type=Path,
        default=DEFAULT_NANO_PREDICTION_DIR,
    )
    parser.add_argument(
        "--baseline-public-comparison-report",
        type=Path,
        default=None,
    )
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    finalize_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
