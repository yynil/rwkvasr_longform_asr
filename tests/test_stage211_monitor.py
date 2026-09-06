from __future__ import annotations

import hashlib
import os
import signal
import shlex
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"


def test_stage211_monitor_parses_progress_and_errors_without_ripgrep(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("dirname", "grep", "tail", "awk"):
        (bin_dir / command).symlink_to(shutil.which(command))
    log = tmp_path / "train.log"
    log.write_text(
        "[rwkvasr] Distributed init complete.\n"
        "Traceback: stale failure\n"
        "[rwkvasr] Distributed init complete.\n"
        "[deepspeed-train] step=10 loss=0.2\n"
        "[deepspeed-train] step=20 loss=0.1\n"
        "CUDA OOM: current failure\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            shutil.which("bash"),
            "-c",
            'source "$1"; stage211_latest_training_record "$2"; '
            'stage211_current_attempt_errors "$2"',
            "monitor-test",
            str(MONITOR),
            str(log),
        ],
        env={**os.environ, "PATH": str(bin_dir)},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=5.0,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == [
        "[deepspeed-train] step=20 loss=0.1",
        "4:CUDA OOM: current failure",
    ]


def test_stage211_monitor_daemon_is_singleton_but_one_shot_remains_available(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    monitor_log = output_root / "monitor.log"
    daemon_lock = output_root / "daemon.lock"
    snapshot_lock = output_root / "snapshot.lock"
    environment = {
        **os.environ,
        "OUTPUT_ROOT": str(output_root),
        "MONITOR_LOG": str(monitor_log),
        "MONITOR_DAEMON_LOCK": str(daemon_lock),
        "MONITOR_SNAPSHOT_LOCK": str(snapshot_lock),
        "POLL_SECONDS": "60",
    }
    command = (
        'source "$1"; '
        'stage211_emit_snapshot() { printf "snapshot\\n"; }; '
        "tmux() { return 0; }; "
        "stage211_main"
    )
    arguments = [
        "bash",
        "-c",
        command,
        "stage211-monitor-test",
        str(MONITOR),
    ]
    daemon = subprocess.Popen(
        arguments,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if monitor_log.is_file() and monitor_log.read_text(encoding="utf-8") == "snapshot\n":
                break
            if daemon.poll() is not None:
                stdout, stderr = daemon.communicate()
                raise AssertionError(
                    f"monitor daemon exited before acquiring its lock: {stdout=} {stderr=}"
                )
            time.sleep(0.05)
        else:
            raise AssertionError("monitor daemon did not publish its initial snapshot")

        duplicate = subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        assert duplicate.returncode == 0
        assert "hourly monitor already active" in duplicate.stderr
        assert monitor_log.read_text(encoding="utf-8") == "snapshot\n"

        one_shot = subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            env={**environment, "MONITOR_ONCE": "1"},
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        assert one_shot.returncode == 0, one_shot.stderr
        assert monitor_log.read_text(encoding="utf-8") == "snapshot\nsnapshot\n"
    finally:
        if daemon.poll() is None:
            os.killpg(daemon.pid, signal.SIGTERM)
            try:
                daemon.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(daemon.pid, signal.SIGKILL)
                daemon.wait(timeout=5.0)


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


def test_stage211_monitor_uses_profile_total_after_metadata_exclusion(
    tmp_path: Path,
) -> None:
    labeled_root = tmp_path / "labeled"
    labeled_root.mkdir()
    (labeled_root / "prepare.log").write_text(
        "[rwkvasr] ctc-align lengths complete processed=100 kept=92 lengths=index.jsonl\n"
        "CTC-aligned clean preprocessing complete\n",
        encoding="utf-8",
    )
    (labeled_root / "profile.json").write_text(
        '{"expected":{"total_samples":90}}\n',
        encoding="utf-8",
    )

    output = _emit_readiness(tmp_path)

    assert (
        "sft_labeled_preparation=complete processed=100 expected=100 progress_pct=100.0000 kept=90"
    ) in output


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


def test_stage211_monitor_formal_errors_exclude_archived_and_probe_logs(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "stage211"
    current = output_root / "stage211a_mixer_full_data_3ep/supplemental_natural/logs/train.log"
    failed = output_root / (
        "stage211a_mixer_full_data_3ep/"
        "supplemental_natural.failed_parquet_io_step710/logs/train.log"
    )
    probe = output_root / (
        "stage211a_mixer_full_data_3ep/batch_profile_preflight/profile/train.log"
    )
    nonrepresentative = output_root / (
        "stage211a_mixer_full_data_3ep/attempt.nonrepresentative_stopped/logs/train.log"
    )
    for path in (current, failed, probe, nonrepresentative):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "[rwkvasr] Distributed init complete.\nTraceback: synthetic failure\n",
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; '
                'while IFS= read -r -d "" path; do printf "%s\\n" "$path"; done '
                '< <(stage211_recent_formal_training_logs "$2")'
            ),
            "stage211-monitor-test",
            str(MONITOR),
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "RECENT_LOG_MINUTES": "90"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(current)]
