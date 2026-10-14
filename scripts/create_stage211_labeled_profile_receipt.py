#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import sha256_file


DEFAULT_LABELED_ROOT = (
    Path.home() / "rwkvasr_data" / "stage211_sft_full_labeled_v3_public_clean"
)
PROFILE_METRIC_KEYS = (
    "train_samples",
    "eval_samples",
    "total_samples",
    "total_hours",
    "ctc_tokens",
    "ctc_unk_tokens",
    "ctc_forbidden_tokens",
    "unique_utterance_ids",
    "pronunciation_target_samples",
    "ctc_feasible_samples",
    "estimated_train_steps",
    "tail_padding_samples_per_epoch",
    "tail_padding_sample_exposures",
    "executed_sample_exposures",
    "source_counts",
    "language_counts",
    "interleave_lane_counts",
)


def _audit_labeled_data(
    *,
    webdataset_root: Path,
    length_index_path: Path,
    bucket_manifest_path: Path,
) -> dict[str, Any]:
    try:
        from scripts.run_stage211_strict_chained_alignment import (
            _audit_labeled_data as audit,
        )
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        from run_stage211_strict_chained_alignment import (  # type: ignore[no-redef]
            _audit_labeled_data as audit,
        )

    return audit(
        webdataset_root=webdataset_root,
        length_index_path=length_index_path,
        bucket_manifest_path=bucket_manifest_path,
    )


def build_receipt(
    *,
    labeled_root: Path,
    length_index: Path | None = None,
    bucket_manifest: Path | None = None,
) -> dict[str, Any]:
    labeled_root = labeled_root.expanduser().resolve()
    length_index = (
        length_index.expanduser().resolve()
        if length_index is not None
        else (labeled_root / "webdataset_lengths.jsonl").resolve()
    )
    bucket_manifest = (
        bucket_manifest.expanduser().resolve()
        if bucket_manifest is not None
        else (labeled_root / "webdataset_buckets_audio_text" / "manifest.json").resolve()
    )
    audit = _audit_labeled_data(
        webdataset_root=labeled_root,
        length_index_path=length_index,
        bucket_manifest_path=bucket_manifest,
    )
    preparation = audit.get("label_preparation")
    if (
        not isinstance(preparation, dict)
        or preparation.get("profile_schema_version") != 2
        or not isinstance(preparation.get("source_filter"), dict)
    ):
        raise ValueError(
            "The formal Stage211D profile receipt requires schema-v2 full-label preparation."
        )
    expected = {key: audit[key] for key in PROFILE_METRIC_KEYS}
    return {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "stage211_labeled_profile_receipt",
        "phase": "sft",
        "complete": True,
        "epochs": 1,
        "batch_size": 12,
        "world_size": 4,
        "frame_budget": 8_000,
        "all_accepted_unique_rows_required": True,
        "source_language_interleave_required": True,
        "interleave_source_field": preparation["interleave_source_field"],
        "interleave_source_lanes": preparation["interleave_source_lanes"],
        "labeled_webdataset_root": str(labeled_root),
        "length_index_path": str(length_index),
        "length_index_sha256": sha256_file(length_index),
        "bucket_manifest_path": str(bucket_manifest),
        "bucket_manifest_sha256": sha256_file(bucket_manifest),
        "preparation_summary_path": preparation["summary_path"],
        "preparation_summary_sha256": preparation["summary_sha256"],
        "preparation_log_path": preparation["log_path"],
        "preparation_log_sha256": preparation["log_sha256"],
        "source_filter": preparation["source_filter"],
        "expected": expected,
        "labeled_data_audit": audit,
    }


def write_immutable_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    rendered = json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Refusing to replace a different Stage211D profile: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def validate_receipt(
    path: Path,
    *,
    labeled_root: Path | None = None,
    length_index: Path | None = None,
    bucket_manifest: Path | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Stage211D labeled profile receipt is unavailable: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict):
        raise ValueError("Stage211D labeled profile receipt must be a JSON object.")
    recorded_root = Path(str(actual.get("labeled_webdataset_root") or "")).resolve()
    recorded_index = Path(str(actual.get("length_index_path") or "")).resolve()
    recorded_manifest = Path(str(actual.get("bucket_manifest_path") or "")).resolve()
    for requested, recorded, label in (
        (labeled_root, recorded_root, "root"),
        (length_index, recorded_index, "length index"),
        (bucket_manifest, recorded_manifest, "bucket manifest"),
    ):
        if requested is not None and requested.expanduser().resolve() != recorded:
            raise ValueError(f"Stage211D requested labeled {label} differs from its receipt.")
    expected = build_receipt(
        labeled_root=recorded_root,
        length_index=recorded_index,
        bucket_manifest=recorded_manifest,
    )
    if actual != expected:
        raise ValueError("Stage211D labeled profile receipt differs from deep recomputation.")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or validate the immutable full-labeled Stage211D profile receipt."
    )
    parser.add_argument("--labeled-root", type=Path, default=DEFAULT_LABELED_ROOT)
    parser.add_argument("--length-index", type=Path, default=None)
    parser.add_argument("--bucket-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    labeled_root = args.labeled_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else labeled_root / "stage211_labeled_profile_receipt.json"
    )
    if args.validate_only:
        receipt = validate_receipt(
            output,
            labeled_root=labeled_root,
            length_index=args.length_index,
            bucket_manifest=args.bucket_manifest,
        )
    else:
        receipt = build_receipt(
            labeled_root=labeled_root,
            length_index=args.length_index,
            bucket_manifest=args.bucket_manifest,
        )
        write_immutable_receipt(output, receipt)
        validate_receipt(output, labeled_root=labeled_root)
    expected = receipt["expected"]
    print(
        "[stage211-labeled-profile] "
        f"rows={expected['total_samples']} hours={expected['total_hours']:.6f} "
        f"steps={expected['estimated_train_steps']} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
