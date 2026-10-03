from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_batch_profile import (
    validate_stage211_batch_profile_admission,
    validate_stage211_batch_profile_preflight,
)
from rwkvasr.eval.stage211_gate import STAGE211_AUDIO_CURRICULUM, sha256_file
from rwkvasr.eval.stage211_supplemental import (
    STAGE211_SUPPLEMENTAL_DIFFICULTY,
    stage211_supplemental_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
STRICT_RUNNER = REPO_ROOT / "scripts" / "run_stage211_strict_chained_alignment.py"
RECEIPT_CREATOR = REPO_ROOT / "scripts" / "create_stage211_curriculum_receipt.py"
PROFILE_RECEIPT_CREATOR = REPO_ROOT / "scripts" / "create_stage211_supplemental_profile_receipt.py"
PROFILE_BENCHMARK = REPO_ROOT / "scripts" / "benchmark_stage211_batch_profiles.py"
ADMISSION_CREATOR = REPO_ROOT / "scripts" / "create_stage211_batch_profile_admission.py"
FULL_PHASE_CONTROLLER = REPO_ROOT / "scripts" / "run_stage211_full_phase_curriculum.py"
CURRICULUM_VALIDATOR = REPO_ROOT / "scripts" / "finalize_stage211_phase.py"
SUPPLEMENTAL_RETENTION_VALIDATOR = (
    REPO_ROOT / "scripts" / "validate_stage211_supplemental_retention.py"
)
RETENTION_REPLAY_VALIDATOR = REPO_ROOT / "scripts" / "validate_stage211_retention_replay.py"
BOOTSTRAP = REPO_ROOT / "scripts" / "start_stage211_abcd_after_calibration.sh"

DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_runs" / "stage211_full_alignment"
DEFAULT_CONFIG_ROOT = Path.home() / "rwkvasr_configs" / "stage211_full_alignment"
DEFAULT_METADATA_ROOT = Path.home() / "rwkvasr_data" / "stage211_full_curriculum"
DEFAULT_PHASE_ROOT = DEFAULT_OUTPUT_ROOT / "stage211a_mixer_full_data_3ep"
DEFAULT_INITIAL_CHECKPOINT = (
    Path.home() / "rwkvasr_runs" / "sensevoice_rwkv_stage211a_recovery_stage210a30000_"
    "nanomlpfrozen_teacherforced_mixeronly_easy1490h_1ep_lr3e6_wd0_4x4090" / "step-30000.pt"
)
DEFAULT_EASY_MANIFEST = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_easy_source_grouped_buckets"
    / "manifest_stage211_fixed_eval.json"
)
DEFAULT_LONG_MANIFEST = (
    DEFAULT_METADATA_ROOT
    / "stage179d_long_dedup_audio_only_online_ctc"
    / "webdataset_buckets_audio_text"
    / "manifest_stage211_fixed_eval.json"
)
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
DEFAULT_SUPPLEMENTAL_ROOT = (
    Path.home() / "rwkvasr_data" / "stage211_supplemental_combined_v4_locality"
)
DEFAULT_SUPPLEMENTAL_INVENTORY = DEFAULT_SUPPLEMENTAL_ROOT / "supplemental_inventory.json"
DEFAULT_SUPPLEMENTAL_PROFILE_RECEIPT = (
    DEFAULT_SUPPLEMENTAL_ROOT / "supplemental_profile_receipt.json"
)
DEFAULT_SUPERVISOR_LOG = DEFAULT_OUTPUT_ROOT / "supervisor.log"
DEFAULT_HANDOFF_LOG = DEFAULT_OUTPUT_ROOT / "profiled_boundary_handoff.log"
DEFAULT_SUPERVISOR_SESSION = "rwkvasr_stage211_abcd_strict_supervisor"


def _run(command: list[str], *, log_path: Path | None = None) -> None:
    rendered = shlex.join(command)
    print(f"[stage211-profiled-handoff] command={rendered}", flush=True)
    if log_path is None:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[stage211-profiled-handoff] command={rendered}\n")
        log.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return int(payload.get("step", 0))
    finally:
        del payload


def _wait_for_files(paths: tuple[Path, ...], *, poll_seconds: int) -> None:
    while True:
        missing = [path for path in paths if not path.is_file() or path.stat().st_size <= 0]
        if not missing:
            return
        print(
            "[stage211-profiled-handoff] waiting for immutable readiness files "
            f"poll_seconds={poll_seconds} missing={','.join(map(str, missing))}",
            flush=True,
        )
        time.sleep(poll_seconds)


def _retention_validation_commands(
    *,
    stratified_hidden_receipt: Path,
    retention_replay_receipt: Path,
) -> tuple[list[str], list[str]]:
    return (
        [
            str(PYTHON),
            str(SUPPLEMENTAL_RETENTION_VALIDATOR),
            "--stratified-receipt",
            str(stratified_hidden_receipt),
        ],
        [
            str(PYTHON),
            str(RETENTION_REPLAY_VALIDATOR),
            "--receipt",
            str(retention_replay_receipt),
        ],
    )


def _legacy_controller_state(pid: int) -> str:
    status_path = Path(f"/proc/{pid}/status")
    command_path = Path(f"/proc/{pid}/cmdline")
    if not status_path.is_file() or not command_path.is_file():
        raise ValueError(f"Legacy Stage211 controller is unavailable: pid={pid}")
    command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    if "run_stage211_full_phase_curriculum.py --phase mixer" not in command:
        raise ValueError(f"PID {pid} is not the legacy Stage211 Mixer controller.")
    state_line = next(
        (
            line
            for line in status_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("State:")
        ),
        "",
    )
    state = state_line.partition(":")[2].strip()
    if not state.startswith("T"):
        raise ValueError(
            f"Legacy Stage211 controller is not stopped at the Long boundary: pid={pid} state={state!r}"
        )
    return state


def _long_receipt_command(
    *,
    phase_root: Path,
    long_manifest: Path,
) -> list[str]:
    hard_step = int(STAGE211_AUDIO_CURRICULUM["hard"]["steps"])
    long_step = int(STAGE211_AUDIO_CURRICULUM["long"]["steps"])
    return [
        str(PYTHON),
        str(RECEIPT_CREATOR),
        "--phase",
        "mixer",
        "--difficulty",
        "long",
        "--run-dir",
        str(phase_root / "long"),
        "--bucket-manifest",
        str(long_manifest),
        "--init-checkpoint",
        str(phase_root / "hard" / f"step-{hard_step}.pt"),
        "--completion-checkpoint",
        str(phase_root / "long" / f"step-{long_step}.pt"),
        "--output",
        str(phase_root / "receipts" / "long.json"),
    ]


def _template_command(
    *,
    phase_root: Path,
    config_root: Path,
    manifest: Path,
    inventory: Path,
    nano_checkpoint: Path,
    master_port: int,
) -> list[str]:
    long_step = int(STAGE211_AUDIO_CURRICULUM["long"]["steps"])
    return [
        str(PYTHON),
        str(STRICT_RUNNER),
        "--phase",
        "mixer",
        "--difficulty",
        STAGE211_SUPPLEMENTAL_DIFFICULTY,
        "--full-data-profile",
        "--output-dir",
        str(phase_root / STAGE211_SUPPLEMENTAL_DIFFICULTY),
        "--config-dir",
        str(config_root),
        "--bucket-manifest",
        str(manifest),
        "--supplemental-inventory",
        str(inventory),
        "--nano-checkpoint",
        str(nano_checkpoint),
        "--init-checkpoint",
        str(phase_root / "long" / f"step-{long_step}.pt"),
        "--curriculum-receipt",
        str(phase_root / "receipts" / "long.json"),
        "--master-port",
        str(master_port),
        "--dry-run",
        "--skip-nano-weight-audit",
    ]


def _find_template_config(config_root: Path, *, expected_steps: int) -> Path:
    root = config_root / "mixer" / "full_supplemental_natural"
    candidates = sorted(root.glob("*.yaml"))
    matches = [
        path
        for path in candidates
        if int(load_yaml(path).get("max_steps", -1)) == int(expected_steps)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Stage211 Supplemental template config is ambiguous: "
            f"expected_steps={expected_steps} matches={matches}"
        )
    return matches[0].resolve()


def _ensure_template_config(
    *,
    phase_root: Path,
    config_root: Path,
    manifest: Path,
    inventory: Path,
    nano_checkpoint: Path,
    master_port: int,
    expected_steps: int,
    log_path: Path,
) -> Path:
    root = config_root / "mixer" / "full_supplemental_natural"
    matches = [
        path
        for path in sorted(root.glob("*.yaml"))
        if int(load_yaml(path).get("max_steps", -1)) == int(expected_steps)
    ]
    if len(matches) == 1:
        template = matches[0].resolve()
        print(
            "[stage211-profiled-handoff] reusing Supplemental template "
            f"path={template} expected_steps={expected_steps}",
            flush=True,
        )
        return template
    if len(matches) > 1:
        raise ValueError(
            "Stage211 Supplemental template config is ambiguous: "
            f"expected_steps={expected_steps} matches={matches}"
        )
    provenance = phase_root / STAGE211_SUPPLEMENTAL_DIFFICULTY / "stage211_provenance.json"
    if provenance.is_file():
        raise ValueError(
            "Stage211 Supplemental template is missing after formal provenance was recorded; "
            "refusing fresh initialization during recovery."
        )
    _run(
        _template_command(
            phase_root=phase_root,
            config_root=config_root,
            manifest=manifest,
            inventory=inventory,
            nano_checkpoint=nano_checkpoint,
            master_port=master_port,
        ),
        log_path=log_path,
    )
    return _find_template_config(config_root, expected_steps=expected_steps)


def _preflight_command(
    *,
    base_config: Path,
    init_checkpoint: Path,
    output_root: Path,
    master_port: int,
) -> list[str]:
    return [
        str(PYTHON),
        str(PROFILE_BENCHMARK),
        "--phase",
        "mixer",
        "--base-config",
        str(base_config),
        "--init-checkpoint",
        str(init_checkpoint),
        "--output-root",
        str(output_root),
        "--master-port",
        str(master_port),
        "--max-peak-memory-gib",
        "22.0",
        "--min-improvement-ratio",
        "0.10",
        "--max-loss-regression-ratio",
        "0.05",
        "--max-cosine-regression",
        "0.005",
    ]


def _admission_command(
    *,
    report_path: Path,
    admission_path: Path,
    admitted_by: str,
    reason: str,
) -> list[str]:
    return [
        str(PYTHON),
        str(ADMISSION_CREATOR),
        "--preflight-report",
        str(report_path),
        "--phase",
        "mixer",
        "--admitted-by",
        admitted_by,
        "--reason",
        reason,
        "--admit-recommended-profile",
        "--output",
        str(admission_path),
    ]


def _profiled_controller_command(
    *,
    initial_checkpoint: Path,
    output_root: Path,
    config_root: Path,
    metadata_root: Path,
    easy_manifest: Path,
    nano_checkpoint: Path,
    inventory: Path,
    profile_receipt: Path,
    admission_path: Path | None,
    master_port: int,
) -> list[str]:
    command = [
        str(PYTHON),
        str(FULL_PHASE_CONTROLLER),
        "--phase",
        "mixer",
        "--init-checkpoint",
        str(initial_checkpoint),
        "--output-root",
        str(output_root),
        "--config-root",
        str(config_root),
        "--metadata-root",
        str(metadata_root),
        "--easy-manifest",
        str(easy_manifest),
        "--nano-checkpoint",
        str(nano_checkpoint),
        "--supplemental-inventory",
        str(inventory),
        "--supplemental-profile-receipt",
        str(profile_receipt),
    ]
    if admission_path is not None:
        command.extend(
            (
                "--batch-profile-admission",
                f"{STAGE211_SUPPLEMENTAL_DIFFICULTY}={admission_path}",
            )
        )
    command.extend(
        (
            "--master-port",
            str(master_port),
            "--final-checkpoint-path-output",
            str(
                output_root
                / "stage211a_mixer_full_data_3ep"
                / "final_checkpoint_with_supplemental.txt"
            ),
        )
    )
    return command


def _supplemental_profile_requires_admission(report: dict[str, object]) -> bool:
    if report["selection_decision"] != "keep_baseline":
        return True
    selected = report["selected_profile_row"]
    if not isinstance(selected, dict) or not isinstance(selected.get("profile"), dict):
        raise ValueError("Stage211 Supplemental preflight selected profile is invalid.")
    profile = selected["profile"]
    if (
        int(profile["batch_size"]) != 36
        or int(profile["frame_budget"]) != 24_000
        or int(profile["num_workers"]) != 8
        or profile["gradient_checkpointing"] is not False
    ):
        raise ValueError("Stage211 Supplemental retained baseline differs from the formal default.")
    return False


def _supervisor_command(
    *,
    session: str,
    supervisor_log: Path,
) -> list[str]:
    return [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session,
        "-c",
        str(REPO_ROOT),
        "env",
        "START_STAGE=post_mixer",
        "REUSE_COMPLETED_CALIBRATION_EVAL=1",
        "bash",
        "-c",
        'exec bash "$1" >>"$2" 2>&1',
        "_",
        str(BOOTSTRAP),
        str(supervisor_log),
    ]


def _replace_supervisor(
    *,
    legacy_controller_pid: int,
    session: str,
    supervisor_log: Path,
) -> None:
    _legacy_controller_state(legacy_controller_pid)
    active = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if active.returncode != 0:
        raise ValueError(f"Legacy Stage211 supervisor session is unavailable: {session}")
    subprocess.run(["tmux", "kill-session", "-t", session], check=True)
    subprocess.run(
        _supervisor_command(session=session, supervisor_log=supervisor_log),
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure and admit a fresh Stage211 Mixer Supplemental batch profile after "
            "the legacy controller has stopped at the completed Long boundary."
        )
    )
    parser.add_argument("--legacy-controller-pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--easy-manifest", type=Path, default=DEFAULT_EASY_MANIFEST)
    parser.add_argument("--long-manifest", type=Path, default=DEFAULT_LONG_MANIFEST)
    parser.add_argument(
        "--supplemental-inventory", type=Path, default=DEFAULT_SUPPLEMENTAL_INVENTORY
    )
    parser.add_argument(
        "--supplemental-profile-receipt",
        type=Path,
        default=DEFAULT_SUPPLEMENTAL_PROFILE_RECEIPT,
    )
    parser.add_argument(
        "--stratified-hidden-receipt",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--retention-replay-receipt",
        type=Path,
        default=None,
    )
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--preflight-master-port", type=int, default=29731)
    parser.add_argument("--readiness-poll-seconds", type=int, default=3600)
    parser.add_argument("--admitted-by", default="codex-stage211-boundary-handoff")
    parser.add_argument(
        "--reason",
        default=(
            "phase-specific four-GPU profile passed memory, complete-match, "
            "quality-equivalence, and full-coverage wall-time gates"
        ),
    )
    parser.add_argument("--supervisor-session", default=DEFAULT_SUPERVISOR_SESSION)
    parser.add_argument("--supervisor-log", type=Path, default=DEFAULT_SUPERVISOR_LOG)
    parser.add_argument("--handoff-log", type=Path, default=DEFAULT_HANDOFF_LOG)
    args = parser.parse_args()
    if args.legacy_controller_pid <= 0 or args.readiness_poll_seconds <= 0:
        parser.error("controller PID and readiness poll interval must be positive")

    output_root = args.output_root.expanduser().resolve()
    config_root = args.config_root.expanduser().resolve()
    metadata_root = args.metadata_root.expanduser().resolve()
    phase_root = output_root / "stage211a_mixer_full_data_3ep"
    initial_checkpoint = args.initial_checkpoint.expanduser().resolve()
    easy_manifest = args.easy_manifest.expanduser().resolve()
    long_manifest = args.long_manifest.expanduser().resolve()
    inventory = args.supplemental_inventory.expanduser().resolve()
    profile_receipt = args.supplemental_profile_receipt.expanduser().resolve()
    stratified_hidden_receipt = (
        args.stratified_hidden_receipt.expanduser().resolve()
        if args.stratified_hidden_receipt is not None
        else metadata_root / "stratified_hidden_eval_v3" / "receipt.json"
    )
    retention_replay_receipt = (
        args.retention_replay_receipt.expanduser().resolve()
        if args.retention_replay_receipt is not None
        else metadata_root / "retention_replay_v4_locality" / "receipt.json"
    )
    nano_checkpoint = args.nano_checkpoint.expanduser().resolve()
    handoff_log = args.handoff_log.expanduser().resolve()

    state = _legacy_controller_state(args.legacy_controller_pid)
    print(
        "[stage211-profiled-handoff] legacy controller boundary validated "
        f"pid={args.legacy_controller_pid} state={state}",
        flush=True,
    )
    long_step = int(STAGE211_AUDIO_CURRICULUM["long"]["steps"])
    long_checkpoint = phase_root / "long" / f"step-{long_step}.pt"
    if not long_checkpoint.is_file() or _checkpoint_step(long_checkpoint) != long_step:
        raise ValueError(f"Stage211 Long terminal checkpoint is unavailable: {long_checkpoint}")
    _run(
        _long_receipt_command(phase_root=phase_root, long_manifest=long_manifest),
        log_path=handoff_log,
    )
    long_receipt = phase_root / "receipts" / "long.json"
    if not long_receipt.is_file() or long_receipt.stat().st_size <= 0:
        raise ValueError(f"Stage211 Long receipt was not created: {long_receipt}")

    _wait_for_files(
        (inventory, profile_receipt),
        poll_seconds=int(args.readiness_poll_seconds),
    )
    _run(
        [
            str(PYTHON),
            str(PROFILE_RECEIPT_CREATOR),
            "--inventory",
            str(inventory),
            "--output",
            str(profile_receipt),
        ],
        log_path=handoff_log,
    )
    supplemental = stage211_supplemental_profile(
        inventory,
        epochs=3,
        batch_size=36,
        world_size=4,
        frame_budget=24_000,
        require_training_ready=True,
        verify_part_sha256=False,
    )
    _wait_for_files(
        (stratified_hidden_receipt, retention_replay_receipt),
        poll_seconds=int(args.readiness_poll_seconds),
    )
    for command in _retention_validation_commands(
        stratified_hidden_receipt=stratified_hidden_receipt,
        retention_replay_receipt=retention_replay_receipt,
    ):
        _run(command, log_path=handoff_log)
    print(
        "[stage211-profiled-handoff] supplemental retention barrier passed "
        f"stratified={stratified_hidden_receipt} replay={retention_replay_receipt}",
        flush=True,
    )
    manifest = Path(str(supplemental["bucket_manifest_path"])).resolve()
    base_config = _ensure_template_config(
        phase_root=phase_root,
        config_root=config_root,
        manifest=manifest,
        inventory=inventory,
        nano_checkpoint=nano_checkpoint,
        master_port=int(args.master_port),
        expected_steps=int(supplemental["steps"]),
        log_path=handoff_log,
    )
    init_sha = sha256_file(long_checkpoint)
    manifest_sha = sha256_file(manifest)
    profile_scope = f"supplemental-{init_sha[:16]}-{manifest_sha[:16]}"
    preflight_root = phase_root / "batch_profile_preflight" / profile_scope
    report_path = preflight_root / "batch_throughput_preflight.json"
    if report_path.is_file():
        report = validate_stage211_batch_profile_preflight(
            report_path,
            phase="mixer",
            require_candidate=False,
        )
        if (
            Path(str(report["init_checkpoint_path"])).resolve() != long_checkpoint.resolve()
            or Path(str(report["bucket_manifest_path"])).resolve() != manifest
        ):
            raise ValueError("Existing Stage211 Supplemental preflight binds other inputs.")
    else:
        if preflight_root.exists() and any(preflight_root.iterdir()):
            raise ValueError(f"Incomplete Stage211 Supplemental preflight exists: {preflight_root}")
        _run(
            _preflight_command(
                base_config=base_config,
                init_checkpoint=long_checkpoint,
                output_root=preflight_root,
                master_port=int(args.preflight_master_port),
            ),
            log_path=handoff_log,
        )
        report = validate_stage211_batch_profile_preflight(
            report_path,
            phase="mixer",
            require_candidate=False,
        )

    admission_path: Path | None = None
    if _supplemental_profile_requires_admission(report):
        admission_path = (
            phase_root / "batch_profile_preflight" / f"{profile_scope}-admission.json"
        )
        _run(
            _admission_command(
                report_path=report_path,
                admission_path=admission_path,
                admitted_by=str(args.admitted_by),
                reason=str(args.reason),
            ),
            log_path=handoff_log,
        )
        admission = validate_stage211_batch_profile_admission(
            admission_path,
            phase="mixer",
            expected_init_checkpoint=long_checkpoint,
            expected_bucket_manifest=manifest,
        )
        print(
            "[stage211-profiled-handoff] admitted Supplemental profile "
            f"name={admission['selected_profile']['name']} "
            f"steps={admission['selected_coverage']['full_coverage_steps']}",
            flush=True,
        )
    else:
        print(
            "[stage211-profiled-handoff] retained measured Supplemental baseline "
            "profile=baseline batch_size=36 frame_budget=24000",
            flush=True,
        )

    _run(
        _profiled_controller_command(
            initial_checkpoint=initial_checkpoint,
            output_root=output_root,
            config_root=config_root,
            metadata_root=metadata_root,
            easy_manifest=easy_manifest,
            nano_checkpoint=nano_checkpoint,
            inventory=inventory,
            profile_receipt=profile_receipt,
            admission_path=admission_path,
            master_port=int(args.master_port),
        ),
        log_path=handoff_log,
    )
    _run(
        [
            str(PYTHON),
            str(CURRICULUM_VALIDATOR),
            "--phase",
            "mixer",
            "--phase-root",
            str(phase_root),
            "--validate-curriculum-only",
        ],
        log_path=handoff_log,
    )
    _replace_supervisor(
        legacy_controller_pid=int(args.legacy_controller_pid),
        session=str(args.supervisor_session),
        supervisor_log=args.supervisor_log.expanduser().resolve(),
    )
    print(
        "[stage211-profiled-handoff] Mixer curriculum validated; "
        "strict supervisor resumed at post_mixer",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
