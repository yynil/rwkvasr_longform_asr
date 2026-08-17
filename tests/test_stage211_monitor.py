from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"


def _emit_readiness(tmp_path: Path) -> str:
    labeled_root = tmp_path / "labeled"
    combined_root = tmp_path / "combined"
    labeled_root.mkdir(exist_ok=True)
    combined_root.mkdir(exist_ok=True)
    environment = {
        **os.environ,
        "SFT_LABELED_ROOT": str(labeled_root),
        "SFT_PREPARATION_LOG": str(labeled_root / "prepare.log"),
        "SFT_FINALIZER_LOG": str(labeled_root / "finalize.log"),
        "SFT_PROFILE_RECEIPT": str(labeled_root / "profile.json"),
        "SFT_EXPECTED_INPUT_SAMPLES": "100",
        "SUPPLEMENTAL_COMBINED_ROOT": str(combined_root),
        "SUPPLEMENTAL_COMBINED_INVENTORY": str(combined_root / "inventory.json"),
        "SUPPLEMENTAL_COMBINED_PROFILE": str(combined_root / "profile.json"),
        "SUPPLEMENTAL_NINE_CELL_RECEIPT": str(combined_root / "nine-cell.json"),
        "SUPPLEMENTAL_REPLAY_RECEIPT": str(combined_root / "replay.json"),
    }
    command = (
        f"source {shlex.quote(str(MONITOR))}; "
        "stage211_emit_supplemental_readiness; "
        "stage211_emit_sft_readiness"
    )
    return subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
    ).stdout


def test_stage211_monitor_reports_full_readiness_without_data_scan(tmp_path: Path) -> None:
    labeled_root = tmp_path / "labeled"
    combined_root = tmp_path / "combined"
    labeled_root.mkdir()
    combined_root.mkdir()
    preparation_log = labeled_root / "prepare.log"
    preparation_log.write_text(
        "[rwkvasr] ctc-align progress processed=75 kept=70 elapsed=1.0s current=a.tar\n"
        "[rwkvasr] ctc-align lengths complete processed=100 kept=92 lengths=index.jsonl\n"
        "CTC-aligned clean preprocessing complete\n",
        encoding="utf-8",
    )
    (labeled_root / "finalize.log").write_text("profile validated\n", encoding="utf-8")
    artifacts = (
        labeled_root / "profile.json",
        combined_root / "inventory.json",
        combined_root / "profile.json",
        combined_root / "nine-cell.json",
        combined_root / "replay.json",
    )
    for index, path in enumerate(artifacts):
        path.write_text(f"artifact-{index}\n", encoding="utf-8")

    output = _emit_readiness(tmp_path)

    assert (
        "sft_labeled_preparation=complete processed=100 expected=100 progress_pct=100.0000 kept=92"
    ) in output
    assert "sft_labeled_finalizer=profile validated" in output
    for label, path in (
        ("sft_labeled_profile", artifacts[0]),
        ("supplemental_combined_inventory", artifacts[1]),
        ("supplemental_combined_profile", artifacts[2]),
        ("supplemental_nine_cell_receipt", artifacts[3]),
        ("supplemental_replay_receipt", artifacts[4]),
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert f"{label}=ready path={path} sha256={digest}" in output


def test_stage211_monitor_reports_missing_readiness_as_pending(tmp_path: Path) -> None:
    output = _emit_readiness(tmp_path)

    assert (
        "sft_labeled_preparation=incomplete processed=0 expected=100 progress_pct=0.0000 kept=0"
    ) in output
    assert "sft_labeled_profile=pending" in output
    assert "sft_labeled_finalizer=pending" in output
    assert "supplemental_combined_inventory=pending" in output
    assert "supplemental_combined_profile=pending" in output
    assert "supplemental_nine_cell_receipt=pending" in output
    assert "supplemental_replay_receipt=pending" in output


def test_stage211_monitor_supersedes_stale_sft_finalizer_failure(
    tmp_path: Path,
) -> None:
    labeled_root = tmp_path / "labeled"
    labeled_root.mkdir()
    finalizer_log = labeled_root / "finalize.log"
    finalizer_log.write_text(
        "ValueError: Stage211 SFT full input language counts mismatch.\n",
        encoding="utf-8",
    )
    profile = labeled_root / "profile.json"
    profile.write_text("validated-profile\n", encoding="utf-8")
    os.utime(finalizer_log, (1_000_000, 1_000_000))
    os.utime(profile, (1_000_001, 1_000_001))

    output = _emit_readiness(tmp_path)

    assert "sft_labeled_finalizer=profile validated stale_failure_superseded=true" in output


def test_stage211_monitor_reports_only_authoritative_corrected_public_coverage(
    tmp_path: Path,
) -> None:
    corrected_root = tmp_path / "corrected"
    phase_gate_root = tmp_path / "phase-gates"
    stale_root = tmp_path / "stale-calibration"
    expected_rows = {
        "librispeech_test_clean": 2620,
        "librispeech_test_other": 2939,
        "commonvoice_en_test": 14927,
        "aishell1_test": 7176,
        "wenetspeech_test_net": 24774,
    }
    for role_root in (
        corrected_root / "calibration" / "predictions",
        corrected_root / "nano_2512" / "predictions",
    ):
        role_root.mkdir(parents=True)
        for dataset, rows in expected_rows.items():
            (role_root / f"{dataset}.ctc.jsonl").write_text(
                "{}\n" * rows,
                encoding="utf-8",
            )
    candidate = phase_gate_root / "mixer" / "predictions" / "commonvoice_en_test.ctc.jsonl"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("{}\n" * 14926, encoding="utf-8")
    stale = stale_root / "predictions" / "commonvoice_en_test.ctc.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}\n" * 14922, encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_public_evaluation_coverage',
            "stage211-monitor-test",
            str(MONITOR),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CORRECTED_PUBLIC_ROOT": str(corrected_root),
            "PHASE_GATE_ROOT": str(phase_gate_root),
            "CALIBRATION_EVAL_ROOT": str(stale_root),
        },
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 11
    assert sum("status=complete" in line for line in lines) == 10
    assert (
        "role=corrected_calibration dataset=commonvoice_en_test rows=14927 "
        "expected_rows=14927 status=complete"
    ) in result.stdout
    assert (
        "role=corrected_nano dataset=wenetspeech_test_net rows=24774 "
        "expected_rows=24774 status=complete"
    ) in result.stdout
    assert (
        "role=phase_gate dataset=commonvoice_en_test rows=14926 expected_rows=14927 status=mismatch"
    ) in result.stdout
    assert str(stale) not in result.stdout
