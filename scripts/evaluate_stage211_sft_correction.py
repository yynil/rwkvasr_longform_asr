#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_nano_public_baseline_receipt,
    validate_stage211_public_benchmark,
    validate_stage211_public_overlap_binding,
)
from rwkvasr.eval.stage211_initialization import DEFAULT_STAGE211_INITIALIZATION_RECEIPT
from rwkvasr.eval.stage211_public_metrics import (
    _validate_replayed_value,
    build_stage211_sft_correction_public_progress,
    build_stage211_sft_public_progress,
    replay_stage211_public_comparison,
)

try:
    from scripts.build_stage211_sft_correction_profile import validate_correction_profile
    from scripts.create_stage211_stepwise_report import create_stepwise_report
    from scripts.finalize_stage211_labeled_sft import (
        DEFAULT_NANO_PREDICTION_DIR,
        DEFAULT_PUBLIC_MANIFEST_DIR,
        _enrich_public_benchmark,
        _resolve_mixer_gate,
        _resolve_phase_gate,
        _run_nano_comparison,
        _run_public_eval,
        _validate_final_report,
        _validate_public_eval_inputs,
    )
    from scripts.run_stage211_labeled_sft import (
        _validate_completion as validate_full_sft_completion,
    )
    from scripts.run_stage211_sft_correction import (
        _validate_failed_full_sft_report,
        validate_completion_receipt,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from build_stage211_sft_correction_profile import validate_correction_profile
    from create_stage211_stepwise_report import create_stepwise_report
    from finalize_stage211_labeled_sft import (
        DEFAULT_NANO_PREDICTION_DIR,
        DEFAULT_PUBLIC_MANIFEST_DIR,
        _enrich_public_benchmark,
        _resolve_mixer_gate,
        _resolve_phase_gate,
        _run_nano_comparison,
        _run_public_eval,
        _validate_final_report,
        _validate_public_eval_inputs,
    )
    from run_stage211_labeled_sft import (
        _validate_completion as validate_full_sft_completion,
    )
    from run_stage211_sft_correction import (
        _validate_failed_full_sft_report,
        validate_completion_receipt,
    )


DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_eval" / "stage211_phase_gates" / "sft_correction"
SFT_CORRECTION_EVALUATION_SCHEMA_VERSION = 2
SFT_CORRECTION_COVERAGE_SCHEMA_VERSION = 1


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Refusing to overwrite a different Stage211D evaluation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _bound_path(
    payload: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    label: str,
) -> Path:
    path = Path(str(payload.get(path_key) or "")).resolve()
    if not path.is_file() or payload.get(sha_key) != sha256_file(path):
        raise ValueError(f"{label} is unavailable or changed: {path}")
    return path


def _ordered_completion_bindings(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_round: int,
    full_completion_path: Path,
    correction_profile_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) != expected_round:
        raise ValueError("Stage211D correction evaluation has incomplete ordered rounds.")
    bindings: list[dict[str, Any]] = []
    completions: list[dict[str, Any]] = []
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping) or int(raw.get("round", -1)) != index:
            raise ValueError("Stage211D correction completion order is invalid.")
        path = _bound_path(
            raw,
            path_key="path",
            sha_key="sha256",
            label=f"Stage211D correction round {index} completion",
        )
        completion = validate_completion_receipt(path, expected_round=index)
        if (
            Path(str(completion["full_sft_completion_path"])).resolve()
            != full_completion_path
            or Path(str(completion["correction_profile_path"])).resolve()
            != correction_profile_path
        ):
            raise ValueError("Stage211D correction rounds do not share full/profile bindings.")
        bindings.append({"round": index, "path": str(path), "sha256": sha256_file(path)})
        completions.append(completion)
    return bindings, completions


def _correction_coverage(
    *,
    bindings: Sequence[Mapping[str, Any]],
    completions: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    correction_profile_path: Path,
    full_completion_path: Path,
    full_completion: Mapping[str, Any],
    full_checkpoint: Path,
    candidate_checkpoint: Path,
) -> dict[str, Any]:
    if not 1 <= len(completions) <= 3 or len(bindings) != len(completions):
        raise ValueError("Stage211D correction coverage has an invalid round count.")
    previous_checkpoint = full_checkpoint.resolve()
    row_exposures = 0
    hour_exposures = 0.0
    steps = 0
    tail_padding = 0
    executed_exposures = 0
    language_exposures: dict[str, int] = {}
    source_exposures: dict[str, int] = {}
    round_receipts: list[dict[str, Any]] = []
    expected_rows = int(profile["train_samples"])
    expected_hours = float(profile["total_train_hours"])
    expected_steps = int(profile["estimated_train_steps"])
    expected_tail = int(profile["tail_padding_samples"])
    expected_executed = int(profile["executed_sample_exposures"])
    expected_languages = dict(profile["language_counts"])
    expected_sources = dict(profile["source_counts"])
    expected_nano_path = Path(
        str(full_completion["nano_teacher_checkpoint_path"])
    ).resolve()
    expected_nano_sha256 = str(full_completion["nano_teacher_checkpoint_sha256"])
    for index, (binding, completion) in enumerate(
        zip(bindings, completions, strict=True),
        start=1,
    ):
        init_checkpoint = Path(str(completion["init_checkpoint_path"])).resolve()
        completion_checkpoint = Path(
            str(completion["completion_checkpoint_path"])
        ).resolve()
        if init_checkpoint != previous_checkpoint:
            raise ValueError("Stage211D correction checkpoint chain is not contiguous.")
        if (
            int(completion.get("round", -1)) != index
            or int(completion.get("rows", -1)) != expected_rows
            or int(completion.get("row_exposures", -1)) != expected_rows
            or not math.isclose(
                float(completion.get("hours", float("nan"))),
                expected_hours,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or int(completion.get("steps", -1)) != expected_steps
            or int(completion.get("tail_padding_samples", -1)) != expected_tail
            or int(completion.get("executed_sample_exposures", -1))
            != expected_executed
            or completion.get("language_counts") != expected_languages
            or completion.get("source_counts") != expected_sources
            or Path(str(completion.get("nano_teacher_checkpoint_path") or "")).resolve()
            != expected_nano_path
            or completion.get("nano_teacher_checkpoint_sha256")
            != expected_nano_sha256
        ):
            raise ValueError("Stage211D correction receipt differs from its shared profile.")
        row_exposures += expected_rows
        hour_exposures += expected_hours
        steps += expected_steps
        tail_padding += expected_tail
        executed_exposures += expected_executed
        for language, count in expected_languages.items():
            language_exposures[str(language)] = (
                language_exposures.get(str(language), 0) + int(count)
            )
        for source, count in expected_sources.items():
            source_exposures[str(source)] = source_exposures.get(str(source), 0) + int(
                count
            )
        round_receipts.append(
            {
                "round": index,
                "receipt_path": str(Path(str(binding["path"])).resolve()),
                "receipt_sha256": str(binding["sha256"]),
                "init_checkpoint_path": str(init_checkpoint),
                "init_checkpoint_sha256": str(completion["init_checkpoint_sha256"]),
                "completion_checkpoint_path": str(completion_checkpoint),
                "completion_checkpoint_sha256": str(
                    completion["completion_checkpoint_sha256"]
                ),
                "row_exposures": expected_rows,
                "hour_exposures": expected_hours,
                "steps": expected_steps,
                "tail_padding_sample_exposures": expected_tail,
                "executed_sample_exposures": expected_executed,
            }
        )
        previous_checkpoint = completion_checkpoint
    if previous_checkpoint != candidate_checkpoint.resolve():
        raise ValueError("Stage211D correction coverage does not reach the final checkpoint.")
    if sum(language_exposures.values()) != row_exposures or sum(
        source_exposures.values()
    ) != row_exposures:
        raise ValueError("Stage211D correction exposure subtotals are inconsistent.")
    return {
        "schema_version": SFT_CORRECTION_COVERAGE_SCHEMA_VERSION,
        "applied": True,
        "rounds": len(completions),
        "unique_rows_per_round": expected_rows,
        "row_exposures": row_exposures,
        "hour_exposures": hour_exposures,
        "steps": steps,
        "tail_padding_sample_exposures": tail_padding,
        "executed_sample_exposures": executed_exposures,
        "language_row_exposures": dict(sorted(language_exposures.items())),
        "source_row_exposures": dict(sorted(source_exposures.items())),
        "full_sft_completion_path": str(full_completion_path.resolve()),
        "full_sft_completion_sha256": sha256_file(full_completion_path),
        "correction_profile_path": str(correction_profile_path.resolve()),
        "correction_profile_sha256": sha256_file(correction_profile_path),
        "initial_checkpoint_path": str(full_checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256_file(full_checkpoint),
        "final_checkpoint_path": str(candidate_checkpoint.resolve()),
        "final_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "round_receipts": round_receipts,
    }


def validate_correction_evaluation_report(
    report_path: Path,
    *,
    expected_round: int | None = None,
    expected_full_completion_path: Path | None = None,
    expected_correction_profile_path: Path | None = None,
    require_passed: bool | None = None,
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    report = _load_json(report_path, label="Stage211D correction evaluation")
    required = {
        "schema_version": SFT_CORRECTION_EVALUATION_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "sft_correction_evaluation",
        "phase": "sft",
        "complete": True,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Stage211D correction evaluation contract mismatch.")
    round_index = int(report.get("round", -1))
    if not 1 <= round_index <= 3 or (
        expected_round is not None and round_index != expected_round
    ):
        raise ValueError("Stage211D correction evaluation round mismatch.")
    full_completion_path = _bound_path(
        report,
        path_key="full_sft_completion_path",
        sha_key="full_sft_completion_sha256",
        label="Stage211D full-SFT completion",
    )
    correction_profile_path = _bound_path(
        report,
        path_key="correction_profile_path",
        sha_key="correction_profile_sha256",
        label="Stage211D correction profile",
    )
    for requested, actual, label in (
        (expected_full_completion_path, full_completion_path, "full completion"),
        (expected_correction_profile_path, correction_profile_path, "profile"),
    ):
        if requested is not None and requested.expanduser().resolve() != actual:
            raise ValueError(f"Stage211D requested correction {label} differs from report.")
    profile = validate_correction_profile(correction_profile_path)
    full_completion, full_checkpoint = validate_full_sft_completion(full_completion_path)
    full_failed_report_path = _bound_path(
        report,
        path_key="full_sft_failed_report_path",
        sha_key="full_sft_failed_report_sha256",
        label="Stage211D failed full-SFT report",
    )
    full_failed_report = _validate_failed_full_sft_report(
        full_failed_report_path,
        full_completion_path=full_completion_path,
        full_completion=full_completion,
        full_checkpoint=full_checkpoint,
    )
    records = report.get("correction_completion_receipts")
    if not isinstance(records, list):
        raise ValueError("Stage211D correction evaluation lacks ordered completions.")
    bindings, completions = _ordered_completion_bindings(
        records,
        expected_round=round_index,
        full_completion_path=full_completion_path,
        correction_profile_path=correction_profile_path,
    )
    if bindings != records:
        raise ValueError("Stage211D correction completion bindings are non-canonical.")
    candidate_checkpoint = Path(str(report.get("checkpoint_path") or "")).resolve()
    last_checkpoint = Path(str(completions[-1]["completion_checkpoint_path"])).resolve()
    if (
        candidate_checkpoint != last_checkpoint
        or not candidate_checkpoint.is_file()
        or report.get("checkpoint_sha256") != sha256_file(candidate_checkpoint)
    ):
        raise ValueError("Stage211D correction evaluation checkpoint binding mismatch.")
    if report.get("correction_profile_expected") != {
        "train_samples": profile["train_samples"],
        "language_counts": profile["language_counts"],
        "estimated_train_steps": profile["estimated_train_steps"],
    }:
        raise ValueError("Stage211D correction evaluation profile summary mismatch.")
    correction_coverage = _correction_coverage(
        bindings=bindings,
        completions=completions,
        profile=profile,
        correction_profile_path=correction_profile_path,
        full_completion_path=full_completion_path,
        full_completion=full_completion,
        full_checkpoint=full_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
    )
    if report.get("correction_coverage") != correction_coverage:
        raise ValueError("Stage211D correction coverage differs from deep replay.")

    embedded_candidate = report.get("public_benchmark")
    if not isinstance(embedded_candidate, dict):
        raise ValueError("Stage211D correction evaluation lacks public benchmark.")
    raw_results = embedded_candidate.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Stage211D correction public results are invalid.")
    by_dataset = {
        str(result.get("dataset")): result for result in raw_results if isinstance(result, dict)
    }
    if set(by_dataset) != set(STAGE211_PUBLIC_BENCHMARKS):
        raise ValueError("Stage211D correction public dataset coverage mismatch.")
    manifest_paths = {
        dataset: Path(str(result.get("manifest_path") or "")).resolve()
        for dataset, result in by_dataset.items()
    }
    baseline_source_path = _bound_path(
        report,
        path_key="baseline_public_comparison_report_path",
        sha_key="baseline_public_comparison_report_sha256",
        label="Stage211D correction Logits baseline comparison",
    )
    if baseline_source_path != Path(
        str(full_failed_report["baseline_public_comparison_report_path"])
    ).resolve():
        raise ValueError("Stage211D correction changed the fixed Logits baseline.")
    candidate_source_path = _bound_path(
        report,
        path_key="public_comparison_report_path",
        sha_key="public_comparison_report_sha256",
        label="Stage211D correction candidate comparison",
    )
    logits_checkpoint = Path(str(full_completion["init_checkpoint_path"])).resolve()
    replayed_baseline = replay_stage211_public_comparison(
        _load_json(baseline_source_path, label="Logits public comparison"),
        manifest_paths=manifest_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        expected_checkpoint=logits_checkpoint,
    )
    replayed_candidate = replay_stage211_public_comparison(
        _load_json(candidate_source_path, label="Correction public comparison"),
        manifest_paths=manifest_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        expected_checkpoint=candidate_checkpoint,
        require_student_prediction_receipt=True,
    )
    progress = build_stage211_sft_correction_public_progress(
        baseline=replayed_baseline,
        candidate=replayed_candidate,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    _validate_replayed_value(
        report.get("baseline_public_benchmark"),
        replayed_baseline,
        label="SFT correction baseline public benchmark",
    )
    _validate_replayed_value(
        embedded_candidate,
        replayed_candidate,
        label="SFT correction candidate public benchmark",
    )
    _validate_replayed_value(
        report.get("public_progress"),
        progress,
        label="SFT correction public progress",
    )
    benchmark = validate_stage211_public_benchmark(
        replayed_candidate,
        require_metric_source_recomputed=True,
    )
    validate_stage211_public_overlap_binding(
        report.get("public_overlap"),
        public_benchmark=benchmark,
    )
    nano_receipt_path = _bound_path(
        report,
        path_key="nano_public_baseline_receipt_path",
        sha_key="nano_public_baseline_receipt_sha256",
        label="Stage211D correction Nano public baseline",
    )
    validate_stage211_nano_public_baseline_receipt(
        nano_receipt_path,
        expected_nano_checkpoint_sha256=str(full_completion["nano_teacher_checkpoint_sha256"]),
        public_benchmark=benchmark,
    )
    gate_passed = bool(progress["gate_passed"]) and bool(benchmark["all_datasets_pass"])
    if report.get("gate_passed") is not gate_passed:
        raise ValueError("Stage211D correction gate decision differs from replayed evidence.")
    if require_passed is not None and gate_passed is not require_passed:
        raise ValueError(
            f"Stage211D correction gate_passed={gate_passed} expected={require_passed}."
        )
    return report


def _collect_completion_bindings(
    *,
    completion_path: Path,
    completion: Mapping[str, Any],
    round_index: int,
) -> list[dict[str, Any]]:
    if round_index == 1:
        return [
            {
                "round": 1,
                "path": str(completion_path),
                "sha256": sha256_file(completion_path),
            }
        ]
    admission_path = Path(str(completion["admission_report_path"])).resolve()
    previous = validate_correction_evaluation_report(
        admission_path,
        expected_round=round_index - 1,
        require_passed=False,
    )
    records = previous.get("correction_completion_receipts")
    if not isinstance(records, list):
        raise ValueError("Previous correction evaluation lacks completion history.")
    return [
        *records,
        {
            "round": round_index,
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
        },
    ]


def _write_passed_final_report(
    *,
    args: argparse.Namespace,
    evaluation_path: Path,
    evaluation: Mapping[str, Any],
    full_completion_path: Path,
    full_completion: Mapping[str, Any],
    candidate_checkpoint: Path,
    baseline_report_path: Path,
    comparison_json: Path,
    baseline_benchmark: Mapping[str, Any],
    candidate_benchmark: Mapping[str, Any],
    public_overlap_binding: Mapping[str, Any],
    nano_receipt_path: Path,
) -> Path:
    nano_checkpoint = Path(str(full_completion["nano_teacher_checkpoint_path"])).resolve()
    phase_gate_root = args.phase_gate_root.expanduser().resolve()
    mixer_gate, mixer_selection = _resolve_mixer_gate(
        phase_gate_root=phase_gate_root,
        selection_path=args.mixer_gate_selection,
        nano_teacher_checkpoint=nano_checkpoint,
    )
    block_gate, block_selection = _resolve_phase_gate(
        phase="block",
        phase_gate_root=phase_gate_root,
        selection_path=args.block_gate_selection,
        nano_teacher_checkpoint=nano_checkpoint,
    )
    logits_gate, logits_selection = _resolve_phase_gate(
        phase="logits",
        phase_gate_root=phase_gate_root,
        selection_path=args.logits_gate_selection,
        nano_teacher_checkpoint=nano_checkpoint,
    )
    standard_progress = build_stage211_sft_public_progress(
        baseline=dict(baseline_benchmark),
        candidate=dict(candidate_benchmark),
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    if standard_progress["gate_passed"] is not True:
        raise ValueError("Passed correction does not satisfy the standard SFT progress gate.")
    output_dir = args.final_output_dir.expanduser().resolve()
    final_path = output_dir / "stage211_complete.json"
    report = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "final_completion",
        "phase": "sft",
        "complete": True,
        "gate_passed": True,
        "checkpoint_path": str(candidate_checkpoint),
        "checkpoint_sha256": sha256_file(candidate_checkpoint),
        "sft_completion_path": str(full_completion_path),
        "sft_completion_sha256": sha256_file(full_completion_path),
        "labeled_data_coverage": dict(full_completion),
        "logits_promotion_receipt_path": full_completion["logits_promotion_receipt_path"],
        "logits_promotion_receipt_sha256": full_completion[
            "logits_promotion_receipt_sha256"
        ],
        "logits_phase_gate_path": str(logits_gate),
        "logits_phase_gate_sha256": sha256_file(logits_gate),
        "nano_teacher_checkpoint_path": str(nano_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        "nano_public_baseline_receipt_path": str(nano_receipt_path),
        "nano_public_baseline_receipt_sha256": sha256_file(nano_receipt_path),
        "nano_public_baseline_checkpoint_sha256": sha256_file(nano_checkpoint),
        "public_overlap": dict(public_overlap_binding),
        "baseline_public_comparison_report_path": str(baseline_report_path),
        "baseline_public_comparison_report_sha256": sha256_file(baseline_report_path),
        "public_comparison_report_path": str(comparison_json),
        "public_comparison_report_sha256": sha256_file(comparison_json),
        "baseline_public_benchmark": dict(baseline_benchmark),
        "public_benchmark": dict(candidate_benchmark),
        "public_progress": standard_progress,
        "mixer_phase_gate_path": str(mixer_gate),
        "mixer_phase_gate_sha256": sha256_file(mixer_gate),
        "mixer_gate_selection_path": str(mixer_selection) if mixer_selection else None,
        "mixer_gate_selection_sha256": (
            sha256_file(mixer_selection) if mixer_selection else None
        ),
        "block_phase_gate_path": str(block_gate),
        "block_phase_gate_sha256": sha256_file(block_gate),
        "block_gate_selection_path": str(block_selection) if block_selection else None,
        "block_gate_selection_sha256": (
            sha256_file(block_selection) if block_selection else None
        ),
        "logits_gate_selection_path": str(logits_selection) if logits_selection else None,
        "logits_gate_selection_sha256": (
            sha256_file(logits_selection) if logits_selection else None
        ),
        "sft_correction_evaluation_path": str(evaluation_path),
        "sft_correction_evaluation_sha256": sha256_file(evaluation_path),
        "sft_correction_profile_path": evaluation["correction_profile_path"],
        "sft_correction_profile_sha256": evaluation["correction_profile_sha256"],
        "sft_correction_completion_receipts": evaluation[
            "correction_completion_receipts"
        ],
        "sft_correction_coverage": evaluation["correction_coverage"],
        "sft_correction_public_progress": evaluation["public_progress"],
    }
    _write_immutable_json(final_path, report)
    _validate_final_report(final_path, checkpoint=candidate_checkpoint)
    create_stepwise_report(
        initialization_receipt_path=args.initialization_receipt,
        calibration_receipt_path=args.calibration_reuse_receipt,
        mixer_gate_path=mixer_gate,
        block_gate_path=block_gate,
        logits_gate_path=logits_gate,
        sft_final_report_path=final_path,
        output_json=output_dir / "stage211_stepwise_results.json",
        output_markdown=output_dir / "stage211_stepwise_results.md",
    )
    return final_path


def evaluate_correction(args: argparse.Namespace) -> Path:
    completion_path = args.completion_receipt.expanduser().resolve()
    completion = validate_completion_receipt(completion_path)
    round_index = int(completion["round"])
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.output_root.expanduser().resolve() / f"round_{round_index:02d}"
    )
    report_path = output_dir / "correction_evaluation.json"
    if report_path.is_file():
        validate_correction_evaluation_report(
            report_path,
            expected_round=round_index,
        )
        return report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    full_completion_path = Path(str(completion["full_sft_completion_path"])).resolve()
    correction_profile_path = Path(str(completion["correction_profile_path"])).resolve()
    full_completion, full_checkpoint = validate_full_sft_completion(full_completion_path)
    full_failed_report_path = args.full_sft_failed_report.expanduser().resolve()
    full_failed_report = _validate_failed_full_sft_report(
        full_failed_report_path,
        full_completion_path=full_completion_path,
        full_completion=full_completion,
        full_checkpoint=full_checkpoint,
    )
    records = _collect_completion_bindings(
        completion_path=completion_path,
        completion=completion,
        round_index=round_index,
    )
    candidate_checkpoint = Path(str(completion["completion_checkpoint_path"])).resolve()
    public_output = output_dir / "public"
    comparison_json = output_dir / "nano_comparison.json"
    comparison_md = output_dir / "nano_comparison.md"
    manifest_dir = args.public_manifest_dir.expanduser().resolve()
    nano_prediction_dir = args.nano_prediction_dir.expanduser().resolve()
    nano_receipt_path = Path(
        str(full_failed_report["nano_public_baseline_receipt_path"])
    ).resolve()
    if not args.dry_run:
        _validate_public_eval_inputs(
            manifest_dir=manifest_dir,
            nano_prediction_dir=nano_prediction_dir,
            nano_public_baseline_receipt=nano_receipt_path,
        )
    student_prediction_receipt = _run_public_eval(
        checkpoint=candidate_checkpoint,
        output_dir=public_output,
        manifest_dir=manifest_dir,
        devices=str(args.devices),
        dry_run=bool(args.dry_run),
    )
    _run_nano_comparison(
        checkpoint=candidate_checkpoint,
        public_output_dir=public_output,
        nano_prediction_dir=nano_prediction_dir,
        comparison_json=comparison_json,
        comparison_md=comparison_md,
        student_prediction_receipt=student_prediction_receipt,
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        print(f"[stage211-sft-correction-eval] dry-run report={report_path}", flush=True)
        return report_path
    baseline_report_path = Path(
        str(full_failed_report["baseline_public_comparison_report_path"])
    ).resolve()
    baseline_benchmark = _enrich_public_benchmark(
        _load_json(baseline_report_path, label="Logits public comparison"),
        manifest_dir=manifest_dir,
    )
    candidate_benchmark = _enrich_public_benchmark(
        _load_json(comparison_json, label="Correction public comparison"),
        manifest_dir=manifest_dir,
        require_student_prediction_receipt=True,
    )
    progress = build_stage211_sft_correction_public_progress(
        baseline=baseline_benchmark,
        candidate=candidate_benchmark,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    overlap_path = Path(
        getattr(args, "public_overlap_receipt", DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT)
    ).expanduser().resolve()
    public_overlap = {"receipt_path": str(overlap_path), "receipt_sha256": sha256_file(overlap_path)}
    validate_stage211_public_overlap_binding(
        public_overlap,
        public_benchmark=candidate_benchmark,
    )
    validate_stage211_nano_public_baseline_receipt(
        nano_receipt_path,
        expected_nano_checkpoint_sha256=str(full_completion["nano_teacher_checkpoint_sha256"]),
        public_benchmark=candidate_benchmark,
    )
    gate_passed = bool(progress["gate_passed"]) and bool(
        candidate_benchmark.get("all_datasets_pass")
    )
    profile = validate_correction_profile(correction_profile_path)
    correction_bindings, correction_completions = _ordered_completion_bindings(
        records,
        expected_round=round_index,
        full_completion_path=full_completion_path,
        correction_profile_path=correction_profile_path,
    )
    correction_coverage = _correction_coverage(
        bindings=correction_bindings,
        completions=correction_completions,
        profile=profile,
        correction_profile_path=correction_profile_path,
        full_completion_path=full_completion_path,
        full_completion=full_completion,
        full_checkpoint=full_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
    )
    report = {
        "schema_version": SFT_CORRECTION_EVALUATION_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "sft_correction_evaluation",
        "phase": "sft",
        "complete": True,
        "round": round_index,
        "gate_passed": gate_passed,
        "checkpoint_path": str(candidate_checkpoint),
        "checkpoint_sha256": sha256_file(candidate_checkpoint),
        "full_sft_completion_path": str(full_completion_path),
        "full_sft_completion_sha256": sha256_file(full_completion_path),
        "full_sft_failed_report_path": str(full_failed_report_path),
        "full_sft_failed_report_sha256": sha256_file(full_failed_report_path),
        "correction_profile_path": str(correction_profile_path),
        "correction_profile_sha256": sha256_file(correction_profile_path),
        "correction_profile_expected": {
            "train_samples": profile["train_samples"],
            "language_counts": profile["language_counts"],
            "estimated_train_steps": profile["estimated_train_steps"],
        },
        "correction_completion_receipts": records,
        "correction_coverage": correction_coverage,
        "baseline_public_comparison_report_path": str(baseline_report_path),
        "baseline_public_comparison_report_sha256": sha256_file(baseline_report_path),
        "public_comparison_report_path": str(comparison_json),
        "public_comparison_report_sha256": sha256_file(comparison_json),
        "baseline_public_benchmark": baseline_benchmark,
        "public_benchmark": candidate_benchmark,
        "public_progress": progress,
        "public_overlap": public_overlap,
        "nano_public_baseline_receipt_path": str(nano_receipt_path),
        "nano_public_baseline_receipt_sha256": sha256_file(nano_receipt_path),
    }
    _write_immutable_json(report_path, report)
    validate_correction_evaluation_report(
        report_path,
        expected_round=round_index,
        expected_full_completion_path=full_completion_path,
        expected_correction_profile_path=correction_profile_path,
        require_passed=gate_passed,
    )
    if gate_passed:
        final_path = _write_passed_final_report(
            args=args,
            evaluation_path=report_path,
            evaluation=report,
            full_completion_path=full_completion_path,
            full_completion=full_completion,
            candidate_checkpoint=candidate_checkpoint,
            baseline_report_path=baseline_report_path,
            comparison_json=comparison_json,
            baseline_benchmark=baseline_benchmark,
            candidate_benchmark=candidate_benchmark,
            public_overlap_binding=public_overlap,
            nano_receipt_path=nano_receipt_path,
        )
        print(
            f"[stage211-sft-correction-eval] passed round={round_index} final={final_path}",
            flush=True,
        )
    else:
        print(
            f"[stage211-sft-correction-eval] failed round={round_index} report={report_path}",
            flush=True,
        )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one Stage211D labeled correction round on all public sets."
    )
    parser.add_argument("--completion-receipt", type=Path, required=True)
    parser.add_argument("--full-sft-failed-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--final-output-dir", type=Path, required=True)
    parser.add_argument("--public-manifest-dir", type=Path, default=DEFAULT_PUBLIC_MANIFEST_DIR)
    parser.add_argument("--nano-prediction-dir", type=Path, default=DEFAULT_NANO_PREDICTION_DIR)
    parser.add_argument(
        "--public-overlap-receipt",
        type=Path,
        default=DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    )
    parser.add_argument("--phase-gate-root", type=Path, required=True)
    parser.add_argument("--mixer-gate-selection", type=Path, required=True)
    parser.add_argument("--block-gate-selection", type=Path, required=True)
    parser.add_argument("--logits-gate-selection", type=Path, required=True)
    parser.add_argument(
        "--initialization-receipt",
        type=Path,
        default=DEFAULT_STAGE211_INITIALIZATION_RECEIPT,
    )
    parser.add_argument("--calibration-reuse-receipt", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluate_correction(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
