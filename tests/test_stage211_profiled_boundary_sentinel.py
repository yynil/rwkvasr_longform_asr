from __future__ import annotations

import argparse
import importlib
import signal
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sentinel = importlib.import_module("scripts.watch_stage211_profiled_boundary")


def _write_process(
    proc_root: Path,
    *,
    pid: int,
    command: tuple[str, ...],
    state: str = "S",
    children: tuple[int, ...] = (),
) -> None:
    process = proc_root / str(pid)
    task = process / "task" / str(pid)
    task.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in command) + b"\0")
    (process / "status").write_text(
        f"Name:\tpython\nState:\t{state} (test)\n",
        encoding="utf-8",
    )
    (task / "children").write_text(
        " ".join(str(value) for value in children),
        encoding="utf-8",
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        legacy_controller_pid=100,
        output_root=tmp_path / "runs",
        watch_log=tmp_path / "watch.log",
        handoff_log=tmp_path / "handoff.log",
        poll_seconds=0.01,
        readiness_poll_seconds=300,
        expected_long_step=105,
    )


def test_boundary_sentinel_validates_controller_and_finds_only_long_child(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        pid=100,
        command=(
            "python",
            "scripts/run_stage211_full_phase_curriculum.py",
            "--phase",
            "mixer",
        ),
        children=(101, 102),
    )
    _write_process(
        proc_root,
        pid=101,
        command=(
            "python",
            "scripts/run_stage211_strict_chained_alignment.py",
            "--phase",
            "mixer",
            "--difficulty",
            "hard",
        ),
    )
    _write_process(
        proc_root,
        pid=102,
        command=(
            "python",
            "scripts/run_stage211_strict_chained_alignment.py",
            "--phase",
            "mixer",
            "--difficulty",
            "long",
        ),
    )

    command = sentinel._validate_controller(100, proc_root=proc_root)

    assert sentinel._argument_value(command, "--phase") == "mixer"
    assert sentinel._find_long_child(100, proc_root=proc_root) == 102


def test_boundary_sentinel_rejects_ambiguous_long_children(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        pid=100,
        command=("python", "run_stage211_full_phase_curriculum.py", "--phase", "mixer"),
        children=(101, 102),
    )
    for pid in (101, 102):
        _write_process(
            proc_root,
            pid=pid,
            command=(
                "python",
                "run_stage211_strict_chained_alignment.py",
                "--phase",
                "mixer",
                "--difficulty",
                "long",
            ),
        )

    with pytest.raises(ValueError, match="multiple Long children"):
        sentinel._find_long_child(100, proc_root=proc_root)


def test_boundary_sentinel_requires_exact_long_checkpoint_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-105.pt"
    torch.save({"step": 105}, checkpoint)

    sentinel._validate_long_checkpoint(checkpoint, expected_step=105)
    with pytest.raises(ValueError, match="step mismatch"):
        sentinel._validate_long_checkpoint(checkpoint, expected_step=104)


def test_boundary_sentinel_stops_only_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    _write_process(
        proc_root,
        pid=100,
        command=("python", "run_stage211_full_phase_curriculum.py", "--phase", "mixer"),
    )
    signals: list[tuple[int, signal.Signals]] = []

    def fake_kill(pid: int, value: signal.Signals) -> None:
        signals.append((pid, value))
        (proc_root / str(pid) / "status").write_text(
            "Name:\tpython\nState:\tT (stopped)\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(sentinel.os, "kill", fake_kill)

    assert sentinel._stop_controller(100, proc_root=proc_root) == "T"
    assert signals == [(100, signal.SIGSTOP)]


def test_boundary_sentinel_handoff_command_uses_current_python(tmp_path: Path) -> None:
    command = sentinel._handoff_command(
        controller_pid=100,
        output_root=tmp_path / "runs",
        handoff_log=tmp_path / "handoff.log",
        readiness_poll_seconds=300,
    )

    assert command[0] == sys.executable
    assert command[1] == str(sentinel.HANDOFF)
    assert command[command.index("--legacy-controller-pid") + 1] == "100"
    assert command[command.index("--readiness-poll-seconds") + 1] == "300"


@pytest.mark.parametrize(
    ("long_child", "expected_trigger"),
    ((123, "live_long_child"), (None, "terminal_long_checkpoint")),
)
def test_boundary_sentinel_stops_parent_then_execs_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    long_child: int | None,
    expected_trigger: str,
) -> None:
    args = _args(tmp_path)
    checkpoint = args.output_root / "stage211a_mixer_full_data_3ep" / "long" / "step-105.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    logs: list[str] = []
    waited: list[int] = []
    validated: list[Path] = []

    monkeypatch.setattr(
        sentinel,
        "_validate_controller",
        lambda pid: ("python", "run_stage211_full_phase_curriculum.py", "--phase", "mixer"),
    )
    monkeypatch.setattr(sentinel, "_process_exists", lambda pid: True)
    monkeypatch.setattr(sentinel, "_find_long_child", lambda pid: long_child)
    monkeypatch.setattr(sentinel, "_stop_controller", lambda pid: "T")
    monkeypatch.setattr(
        sentinel,
        "_wait_for_exit",
        lambda pid, **kwargs: waited.append(pid),
    )
    monkeypatch.setattr(
        sentinel,
        "_validate_long_checkpoint",
        lambda path, **kwargs: validated.append(path),
    )
    monkeypatch.setattr(sentinel, "_append_log", lambda path, message: logs.append(message))

    class ExecCalled(RuntimeError):
        pass

    monkeypatch.setattr(
        sentinel.os,
        "execv",
        lambda executable, command: (_ for _ in ()).throw(ExecCalled(command)),
    )

    with pytest.raises(ExecCalled):
        sentinel.run(args)

    assert any(f"trigger={expected_trigger}" in row for row in logs)
    assert waited == ([123] if long_child is not None else [])
    assert validated == [checkpoint]
