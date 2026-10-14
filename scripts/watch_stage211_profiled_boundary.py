from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOFF = REPO_ROOT / "scripts" / "run_stage211_profiled_boundary_handoff.py"
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_runs" / "stage211_full_alignment"
DEFAULT_WATCH_LOG = DEFAULT_OUTPUT_ROOT / "batch_profile_boundary.log"
DEFAULT_HANDOFF_LOG = DEFAULT_OUTPUT_ROOT / "profiled_boundary_handoff.log"
EXPECTED_LONG_STEP = 105


def _argument_value(command: tuple[str, ...], name: str) -> str | None:
    try:
        index = command.index(name)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _read_cmdline(pid: int, *, proc_root: Path = Path("/proc")) -> tuple[str, ...]:
    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    return tuple(part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part)


def _read_state(pid: int, *, proc_root: Path = Path("/proc")) -> str:
    for line in (proc_root / str(pid) / "status").read_text(encoding="utf-8").splitlines():
        if line.startswith("State:"):
            fields = line.split()
            if len(fields) >= 2:
                return fields[1]
    raise ValueError(f"Process {pid} has no readable Linux state.")


def _validate_controller(pid: int, *, proc_root: Path = Path("/proc")) -> tuple[str, ...]:
    command = _read_cmdline(pid, proc_root=proc_root)
    if (
        not any(token.endswith("run_stage211_full_phase_curriculum.py") for token in command)
        or _argument_value(command, "--phase") != "mixer"
    ):
        raise ValueError(f"PID {pid} is not the expected Stage211 Mixer controller.")
    state = _read_state(pid, proc_root=proc_root)
    if state in {"X", "Z"}:
        raise ValueError(f"Stage211 Mixer controller {pid} is not live: state={state}")
    return command


def _direct_children(pid: int, *, proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    path = proc_root / str(pid) / "task" / str(pid) / "children"
    if not path.is_file():
        return ()
    return tuple(int(value) for value in path.read_text(encoding="utf-8").split())


def _is_long_child(pid: int, *, proc_root: Path = Path("/proc")) -> bool:
    try:
        command = _read_cmdline(pid, proc_root=proc_root)
    except FileNotFoundError:
        return False
    return (
        any(token.endswith("run_stage211_strict_chained_alignment.py") for token in command)
        and _argument_value(command, "--phase") == "mixer"
        and _argument_value(command, "--difficulty") == "long"
    )


def _find_long_child(pid: int, *, proc_root: Path = Path("/proc")) -> int | None:
    matches = [
        child
        for child in _direct_children(pid, proc_root=proc_root)
        if _is_long_child(child, proc_root=proc_root)
    ]
    if len(matches) > 1:
        raise ValueError(f"Stage211 Mixer controller has multiple Long children: {matches}")
    return matches[0] if matches else None


def _checkpoint_step(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return int(payload.get("step", 0))
    finally:
        del payload


def _validate_long_checkpoint(path: Path, *, expected_step: int) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Stage211 Long terminal checkpoint is missing or empty: {path}")
    actual_step = _checkpoint_step(path)
    if actual_step != expected_step:
        raise ValueError(
            f"Stage211 Long terminal checkpoint step mismatch: {actual_step}/{expected_step}"
        )


def _stop_controller(pid: int, *, proc_root: Path = Path("/proc")) -> str:
    state = _read_state(pid, proc_root=proc_root)
    if state != "T":
        os.kill(pid, signal.SIGSTOP)
    for _ in range(100):
        state = _read_state(pid, proc_root=proc_root)
        if state == "T":
            return state
        time.sleep(0.01)
    raise ValueError(f"Stage211 Mixer controller {pid} did not enter stopped state.")


def _process_exists(pid: int, *, proc_root: Path = Path("/proc")) -> bool:
    return (proc_root / str(pid)).exists()


def _wait_for_exit(pid: int, *, poll_seconds: float, proc_root: Path = Path("/proc")) -> None:
    while _process_exists(pid, proc_root=proc_root):
        try:
            state = _read_state(pid, proc_root=proc_root)
        except FileNotFoundError:
            if _process_exists(pid, proc_root=proc_root):
                raise
            return
        if state in {"X", "Z"}:
            return
        time.sleep(poll_seconds)


def _handoff_command(
    *,
    controller_pid: int,
    output_root: Path,
    handoff_log: Path,
    readiness_poll_seconds: int,
) -> list[str]:
    return [
        sys.executable,
        str(HANDOFF),
        "--legacy-controller-pid",
        str(controller_pid),
        "--output-root",
        str(output_root),
        "--handoff-log",
        str(handoff_log),
        "--readiness-poll-seconds",
        str(readiness_poll_seconds),
    ]


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[stage211-boundary-sentinel] {timestamp} {message}\n")
        stream.flush()


def run(args: argparse.Namespace) -> None:
    controller_pid = int(args.legacy_controller_pid)
    output_root = args.output_root.expanduser().resolve()
    watch_log = args.watch_log.expanduser().resolve()
    handoff_log = args.handoff_log.expanduser().resolve()
    long_checkpoint = (
        output_root
        / "stage211a_mixer_full_data_3ep"
        / "long"
        / f"step-{int(args.expected_long_step)}.pt"
    )
    command = _validate_controller(controller_pid)
    _append_log(
        watch_log,
        f"armed controller={controller_pid} poll={args.poll_seconds}s command={' '.join(command)}",
    )

    trigger: str | None = None
    long_child: int | None = None
    while _process_exists(controller_pid):
        long_child = _find_long_child(controller_pid)
        if long_child is not None:
            trigger = "live_long_child"
            break
        if long_checkpoint.is_file():
            _validate_long_checkpoint(
                long_checkpoint,
                expected_step=int(args.expected_long_step),
            )
            trigger = "terminal_long_checkpoint"
            break
        if _read_state(controller_pid) == "T":
            raise ValueError(
                "Stage211 Mixer controller is already stopped without a valid Long boundary."
            )
        time.sleep(float(args.poll_seconds))

    if trigger is None:
        _append_log(watch_log, f"controller_exited_before_long controller={controller_pid}")
        raise ValueError("Stage211 Mixer controller exited before a valid Long boundary.")

    state = _stop_controller(controller_pid)
    _append_log(
        watch_log,
        f"controller_stopped={controller_pid} state={state} trigger={trigger} "
        f"long_child={long_child or '-'}",
    )
    if long_child is not None:
        _wait_for_exit(long_child, poll_seconds=float(args.poll_seconds))
        _validate_long_checkpoint(
            long_checkpoint,
            expected_step=int(args.expected_long_step),
        )
        _append_log(watch_log, f"long_child_complete pid={long_child}")

    handoff_command = _handoff_command(
        controller_pid=controller_pid,
        output_root=output_root,
        handoff_log=handoff_log,
        readiness_poll_seconds=int(args.readiness_poll_seconds),
    )
    _append_log(watch_log, f"exec_handoff command={' '.join(handoff_command)}")
    os.execv(handoff_command[0], handoff_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stop a legacy Stage211 controller exactly after its short Long segment."
    )
    parser.add_argument("--legacy-controller-pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--watch-log", type=Path, default=DEFAULT_WATCH_LOG)
    parser.add_argument("--handoff-log", type=Path, default=DEFAULT_HANDOFF_LOG)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--readiness-poll-seconds", type=int, default=300)
    parser.add_argument("--expected-long-step", type=int, default=EXPECTED_LONG_STEP)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.legacy_controller_pid <= 0
        or args.poll_seconds <= 0.0
        or args.readiness_poll_seconds <= 0
        or args.expected_long_step <= 0
    ):
        raise SystemExit("controller PID, poll intervals, and Long step must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
