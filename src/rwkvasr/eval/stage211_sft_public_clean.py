from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from rwkvasr.eval.stage211_gate import sha256_file
from rwkvasr.eval.stage211_sft_public_overlap import (
    validate_stage211_sft_public_overlap_receipt,
)


SCHEMA_VERSION = 1
ARTIFACT = "stage211_sft_public_overlap_exclusion_rebuild"
EXCLUSION_REASON = "public_evaluation_overlap_exact_encoded_audio"
DEFAULT_BUCKET_WIDTH = 80
DEFAULT_ENTRIES_PER_PART = 100_000
INTERLEAVE_FIELD = "stage211_sft_interleave_lane"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _iter_jsonl(path: Path, *, label: str) -> Iterable[tuple[int, dict[str, Any], str]]:
    if not path.is_file():
        raise ValueError(f"{label} is unavailable: {path}")
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} row {line_number} is not a JSON object.")
            yield line_number, value, line


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _immutable_write(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_replace(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    temporary = path.with_name(f".{path.name}.rewrite.{os.getpid()}")
    try:
        with temporary.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _identity(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("shard_name") or ""),
        int(row.get("audio_offset", -1)),
        str(row.get("key") or ""),
    )


def load_exclusion_map(
    overlap_receipt_path: Path,
    *,
    expected_labeled_profile: Path,
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], dict[str, Any]]:
    overlap = validate_stage211_sft_public_overlap_receipt(
        overlap_receipt_path,
        expected_labeled_profile=expected_labeled_profile,
        require_training_ready=False,
    )
    if int(overlap["overlap"]["pair_count"]) <= 0:
        raise ValueError("Stage211D public-clean rebuild requires at least one exact match.")
    matches_path = Path(str(overlap["matches_output"]["path"])).resolve()
    exclusions: dict[tuple[str, int, str], dict[str, Any]] = {}
    for line_number, match, _ in _iter_jsonl(
        matches_path,
        label="Stage211D exact public match output",
    ):
        identity = _identity(match)
        if not identity[0] or identity[1] < 0 or not identity[2]:
            raise ValueError(f"Stage211D public match {line_number} has invalid identity.")
        row_binding = {
            "audio_member": str(match.get("audio_member") or ""),
            "audio_offset": int(match.get("audio_offset", -1)),
            "audio_size": int(match.get("audio_size", -1)),
            "key": str(match.get("key") or ""),
            "shard_name": str(match.get("shard_name") or ""),
            "source_dataset": str(match.get("source_dataset") or ""),
            "split": str(match.get("split") or ""),
            "utt_id": str(match.get("utt_id") or ""),
        }
        public_match = {
            "audio_sha256": str(match.get("audio_sha256") or ""),
            "dataset": str(match.get("dataset") or ""),
            "public_utt_id": str(match.get("public_utt_id") or ""),
        }
        existing = exclusions.get(identity)
        if existing is None:
            exclusions[identity] = {**row_binding, "public_matches": [public_match]}
        else:
            if any(existing[key] != value for key, value in row_binding.items()):
                raise ValueError("Stage211D public matches disagree on a training-row identity.")
            if public_match in existing["public_matches"]:
                raise ValueError("Stage211D public match output contains a duplicate pair.")
            existing["public_matches"].append(public_match)
    if len(exclusions) != (
        int(overlap["overlap"]["training_rows"])
        + int(overlap["overlap"]["internal_eval_rows"])
    ):
        raise ValueError("Stage211D public match unique-row accounting changed.")
    for exclusion in exclusions.values():
        exclusion["public_matches"] = sorted(
            exclusion["public_matches"], key=lambda row: (row["dataset"], row["public_utt_id"])
        )
    return exclusions, overlap


def filter_length_index(
    *,
    source_path: Path,
    output_path: Path,
    exclusions: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    seen_exclusions: set[tuple[str, int, str]] = set()
    retained_rows = 0
    source_rows = 0
    excluded_rows: list[dict[str, Any]] = []
    retained_split_counts: Counter[str] = Counter()
    try:
        with temporary.open("x", encoding="utf-8") as target:
            for line_number, row, original_line in _iter_jsonl(
                source_path,
                label="Stage211D source labeled length index",
            ):
                source_rows += 1
                identity = _identity(row)
                exclusion = exclusions.get(identity)
                if exclusion is None:
                    target.write(original_line if original_line.endswith("\n") else original_line + "\n")
                    retained_rows += 1
                    retained_split_counts[str(row.get("split") or "")] += 1
                    continue
                if identity in seen_exclusions:
                    raise ValueError("Stage211D exclusion identity appears more than once in the index.")
                required = (
                    "audio_member",
                    "audio_offset",
                    "audio_size",
                    "key",
                    "shard_name",
                    "source_dataset",
                    "split",
                    "utt_id",
                )
                if any(row.get(key) != exclusion.get(key) for key in required):
                    raise ValueError(
                        f"Stage211D exclusion binding differs from index row {line_number}."
                    )
                seen_exclusions.add(identity)
                excluded_rows.append(
                    {
                        "ctc_num_tokens": int(row["ctc_num_tokens"]),
                        "language": str(row.get("language") or ""),
                        "line_number": line_number,
                        "num_frames": int(row["num_frames"]),
                        "stage211_sft_interleave_lane": str(row.get(INTERLEAVE_FIELD) or ""),
                        **dict(exclusion),
                    }
                )
            target.flush()
            os.fsync(target.fileno())
        missing = set(exclusions).difference(seen_exclusions)
        if missing:
            raise ValueError(
                "Stage211D public exclusions were not found exactly once in the source index: "
                f"missing={len(missing)}"
            )
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    excluded_rows.sort(key=lambda row: (row["shard_name"], row["audio_offset"], row["key"]))
    return {
        "source_rows": source_rows,
        "retained_rows": retained_rows,
        "excluded_rows": len(excluded_rows),
        "retained_split_counts": dict(sorted(retained_split_counts.items())),
        "excluded_records": excluded_rows,
    }


def _decrement(mapping: dict[str, Any], key: str, amount: int) -> None:
    value = int(mapping.get(key, 0)) - int(amount)
    if value < 0:
        raise ValueError(f"Stage211D public-clean summary underflow for {key!r}.")
    mapping[key] = value


def updated_summary(
    source_summary: Mapping[str, Any],
    *,
    output_root: Path,
    excluded_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = deepcopy(dict(source_summary))
    records = list(excluded_records)
    root = output_root.expanduser().resolve()
    summary["output_dir"] = str(root)
    summary["length_index_path"] = str(root / "webdataset_lengths.jsonl")
    summary["index_path"] = str(root / "webdataset_index.json")
    summary["num_kept_samples"] = int(summary["num_kept_samples"]) - len(records)
    summary["num_dropped_samples"] = int(summary["num_dropped_samples"]) + len(records)
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Stage211D source summary lacks count accounting.")
    dropped_reason = counts.setdefault("dropped_by_reason", {})
    dropped_reason[EXCLUSION_REASON] = int(dropped_reason.get(EXCLUSION_REASON, 0)) + len(records)
    for record in records:
        split = str(record["split"])
        source = str(record["source_dataset"])
        language = str(record["language"])
        lane = str(record[INTERLEAVE_FIELD])
        for mapping_name, key in (
            ("kept_by_split", split),
            ("kept_by_source", source),
            ("kept_by_language", language),
            ("kept_by_label_source", "metadata"),
            ("kept_by_split_source", f"{split}/{source}"),
            ("kept_by_split_language", f"{split}/{language}"),
            ("kept_by_interleave_lane", lane),
        ):
            mapping = counts.get(mapping_name)
            if not isinstance(mapping, dict):
                raise ValueError(f"Stage211D source summary lacks {mapping_name}.")
            _decrement(mapping, key, 1)
        for mapping_name, key in (
            ("dropped_by_source", source),
            ("dropped_by_language", language),
        ):
            mapping = counts.setdefault(mapping_name, {})
            mapping[key] = int(mapping.get(key, 0)) + 1
    if sum(int(value) for value in counts["dropped_by_reason"].values()) != int(
        summary["num_dropped_samples"]
    ):
        raise ValueError("Stage211D public-clean dropped-reason accounting changed.")
    return summary


def updated_webdataset_index(
    source_index: Mapping[str, Any],
    *,
    output_root: Path,
    excluded_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    index = deepcopy(dict(source_index))
    records = list(excluded_records)
    index["root"] = str(output_root.expanduser().resolve())
    index["num_samples"] = int(index["num_samples"]) - len(records)
    shards = {
        str(row.get("name") or ""): row
        for row in index.get("shards", [])
        if isinstance(row, dict)
    }
    for record in records:
        split = str(record["split"])
        shard_name = str(record["shard_name"])
        _decrement(index["splits"][split], "num_samples", 1)
        shard = shards.get(shard_name)
        if shard is None:
            raise ValueError(f"Stage211D source index lacks shard {shard_name!r}.")
        _decrement(shard, "num_samples", 1)
        _decrement(shard["splits"][split], "num_samples", 1)
    return index


def _run_bucket_builder(
    *,
    repo_root: Path,
    builder: Path,
    staging_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    bucket_dir = staging_root / "webdataset_buckets_audio_text"
    manifest_path = bucket_dir / "manifest.json"
    bucket_dir.mkdir(parents=True)
    subprocess.run(
        [
            str(builder),
            "--shard-root",
            str(staging_root),
            "--length-index-path",
            str(staging_root / "webdataset_lengths.jsonl"),
            "--output-dir",
            str(bucket_dir),
            "--manifest-path",
            str(manifest_path),
            "--bucket-width",
            str(DEFAULT_BUCKET_WIDTH),
            "--text-cost-source",
            "num_text_tokens",
            "--text-cost-weight",
            "4",
            "--json-size-text-offset",
            "256",
            "--json-size-bytes-per-token",
            "4.0",
            "--entries-per-part",
            str(DEFAULT_ENTRIES_PER_PART),
            "--source-field",
            INTERLEAVE_FIELD,
        ],
        cwd=repo_root,
        check=True,
    )
    manifest = _load_json(manifest_path, label="Stage211D public-clean bucket manifest")
    manifest["root"] = str(final_root)
    manifest["source_length_index_path"] = str(final_root / "webdataset_lengths.jsonl")
    _atomic_replace(manifest_path, _json_bytes(manifest))
    return manifest


def build_stage211_sft_public_clean_profile(
    *,
    source_profile_path: Path,
    overlap_receipt_path: Path,
    output_root: Path,
    repo_root: Path,
    builder_path: Path,
) -> dict[str, Any]:
    try:
        from scripts.create_stage211_labeled_profile_receipt import (
            build_receipt as build_profile,
            validate_receipt as validate_profile,
            write_immutable_receipt as write_profile,
        )
    except ModuleNotFoundError as error:  # pragma: no cover - direct script fallback
        if error.name != "scripts":
            raise
        from create_stage211_labeled_profile_receipt import (  # type: ignore[no-redef]
            build_receipt as build_profile,
            validate_receipt as validate_profile,
            write_immutable_receipt as write_profile,
        )

    source_profile_path = source_profile_path.expanduser().resolve()
    overlap_receipt_path = overlap_receipt_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    builder_path = builder_path.expanduser().resolve()
    receipt_path = output_root / "public_overlap_exclusion_rebuild_receipt.json"
    if receipt_path.is_file():
        return validate_stage211_sft_public_clean_rebuild_receipt(receipt_path)
    if output_root.exists():
        raise ValueError(f"Stage211D public-clean output exists without a receipt: {output_root}")
    if not builder_path.is_file() or not os.access(builder_path, os.X_OK):
        raise ValueError(f"Stage211D bucket builder is unavailable: {builder_path}")

    source_profile = validate_profile(source_profile_path)
    exclusions, overlap = load_exclusion_map(
        overlap_receipt_path,
        expected_labeled_profile=source_profile_path,
    )
    source_root = Path(str(source_profile["labeled_webdataset_root"])).resolve()
    source_length_index = Path(str(source_profile["length_index_path"])).resolve()
    source_summary_path = Path(str(source_profile["preparation_summary_path"])).resolve()
    source_log_path = Path(str(source_profile["preparation_log_path"])).resolve()
    source_index_path = source_root / "webdataset_index.json"
    staging = output_root.with_name(f".{output_root.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise ValueError(f"Stage211D public-clean staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for shard in sorted(source_root.glob("*.tar")):
            target = shard.resolve()
            if not target.is_file():
                raise ValueError(f"Stage211D source shard is unavailable: {shard}")
            (staging / shard.name).symlink_to(target)
        filter_result = filter_length_index(
            source_path=source_length_index,
            output_path=staging / "webdataset_lengths.jsonl",
            exclusions=exclusions,
        )
        excluded_records = list(filter_result.pop("excluded_records"))
        source_summary = _load_json(source_summary_path, label="Stage211D source summary")
        summary = updated_summary(
            source_summary,
            output_root=output_root,
            excluded_records=excluded_records,
        )
        _immutable_write(staging / "webdataset_lengths.summary.json", _json_bytes(summary))
        source_index = _load_json(source_index_path, label="Stage211D source WebDataset index")
        index = updated_webdataset_index(
            source_index,
            output_root=output_root,
            excluded_records=excluded_records,
        )
        _immutable_write(staging / "webdataset_index.json", _json_bytes(index))
        log_text = source_log_path.read_text(encoding="utf-8", errors="strict")
        clean_log = (
            log_text.rstrip("\n")
            + "\n"
            + f"[stage211-sft-public-clean] source_profile={source_profile_path}\n"
            + f"[stage211-sft-public-clean] rejected_overlap_receipt={overlap_receipt_path}\n"
            + f"[stage211-sft-public-clean] exclusion_reason={EXCLUSION_REASON} rows={len(excluded_records)}\n"
            + "[stage211-sft-public-clean] CTC-aligned clean preprocessing complete\n"
        )
        _immutable_write(staging / "prepare_ctc_aligned.log", clean_log.encode())
        _immutable_write(
            staging / "public_overlap_exclusions.jsonl",
            _jsonl_bytes(excluded_records),
        )
        _run_bucket_builder(
            repo_root=repo_root,
            builder=builder_path,
            staging_root=staging,
            final_root=output_root,
        )
        staging.replace(output_root)
    except BaseException:
        if staging.exists():
            failed = staging.with_name(f"{staging.name}.failed")
            if not failed.exists():
                staging.replace(failed)
        raise

    try:
        profile_path = output_root / "stage211_labeled_profile_receipt.json"
        profile = build_profile(labeled_root=output_root)
        write_profile(profile_path, profile)
        validate_profile(profile_path, labeled_root=output_root)
        exclusions_path = output_root / "public_overlap_exclusions.jsonl"
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": "stage211",
            "artifact": ARTIFACT,
            "complete": True,
            "exclusion_reason": EXCLUSION_REASON,
            "source_profile_path": str(source_profile_path),
            "source_profile_sha256": sha256_file(source_profile_path),
            "rejected_overlap_receipt_path": str(overlap_receipt_path),
            "rejected_overlap_receipt_sha256": sha256_file(overlap_receipt_path),
            "rejected_overlap_training_rows": int(overlap["overlap"]["training_rows"]),
            "rejected_overlap_internal_eval_rows": int(
                overlap["overlap"]["internal_eval_rows"]
            ),
            "output_root": str(output_root),
            "output_profile_path": str(profile_path),
            "output_profile_sha256": sha256_file(profile_path),
            "output_length_index_path": str(output_root / "webdataset_lengths.jsonl"),
            "output_length_index_sha256": sha256_file(
                output_root / "webdataset_lengths.jsonl"
            ),
            "output_bucket_manifest_path": str(
                output_root / "webdataset_buckets_audio_text/manifest.json"
            ),
            "output_bucket_manifest_sha256": sha256_file(
                output_root / "webdataset_buckets_audio_text/manifest.json"
            ),
            "exclusions_path": str(exclusions_path),
            "exclusions_sha256": sha256_file(exclusions_path),
            "exclusions_rows": len(excluded_records),
            "coverage": filter_result,
            "output_expected": profile["expected"],
            "builder_path": str(builder_path),
            "builder_sha256": sha256_file(builder_path),
            "builder_source_path": str(
                repo_root / "tools/src/bin/build_bucket_index.rs"
            ),
            "builder_source_sha256": sha256_file(
                repo_root / "tools/src/bin/build_bucket_index.rs"
            ),
            "module_source_path": str(Path(__file__).resolve()),
            "module_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        _immutable_write(receipt_path, _json_bytes(receipt))
        return validate_stage211_sft_public_clean_rebuild_receipt(receipt_path)
    except BaseException:
        if output_root.exists():
            failed = output_root.with_name(f"{output_root.name}.failed_profile.{os.getpid()}")
            if not failed.exists():
                output_root.replace(failed)
        raise


def _checked_file(path_value: Any, sha_value: Any, *, label: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != str(sha_value or ""):
        raise ValueError(f"{label} is unavailable or changed: {path}")
    return path


def validate_stage211_sft_public_clean_rebuild_receipt(
    receipt_path: Path,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211D public-clean rebuild receipt")
    expected_contract = {
        "artifact": ARTIFACT,
        "complete": True,
        "exclusion_reason": EXCLUSION_REASON,
        "pipeline": "stage211",
        "schema_version": SCHEMA_VERSION,
    }
    if any(receipt.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("Stage211D public-clean rebuild contract mismatch.")
    source_profile = _checked_file(
        receipt.get("source_profile_path"),
        receipt.get("source_profile_sha256"),
        label="Stage211D source profile",
    )
    rejected_overlap = _checked_file(
        receipt.get("rejected_overlap_receipt_path"),
        receipt.get("rejected_overlap_receipt_sha256"),
        label="Stage211D rejected overlap receipt",
    )
    overlap = validate_stage211_sft_public_overlap_receipt(
        rejected_overlap,
        expected_labeled_profile=source_profile,
        require_training_ready=False,
    )
    if (
        int(receipt.get("rejected_overlap_training_rows", -1))
        != int(overlap["overlap"]["training_rows"])
        or int(receipt.get("rejected_overlap_internal_eval_rows", -1))
        != int(overlap["overlap"]["internal_eval_rows"])
    ):
        raise ValueError("Stage211D rejected-overlap accounting changed.")
    output_root = Path(str(receipt.get("output_root") or "")).resolve()
    profile_path = _checked_file(
        receipt.get("output_profile_path"),
        receipt.get("output_profile_sha256"),
        label="Stage211D public-clean output profile",
    )
    try:
        from scripts.create_stage211_labeled_profile_receipt import validate_receipt
    except ModuleNotFoundError as error:  # pragma: no cover
        if error.name != "scripts":
            raise
        from create_stage211_labeled_profile_receipt import validate_receipt  # type: ignore[no-redef]

    profile = validate_receipt(profile_path, labeled_root=output_root)
    for path_key, sha_key, label in (
        ("output_length_index_path", "output_length_index_sha256", "length index"),
        ("output_bucket_manifest_path", "output_bucket_manifest_sha256", "bucket manifest"),
        ("exclusions_path", "exclusions_sha256", "exclusions"),
        ("builder_path", "builder_sha256", "builder"),
        ("builder_source_path", "builder_source_sha256", "builder source"),
        ("module_source_path", "module_source_sha256", "module source"),
    ):
        _checked_file(receipt.get(path_key), receipt.get(sha_key), label=f"Stage211D {label}")
    if receipt.get("output_expected") != profile["expected"]:
        raise ValueError("Stage211D public-clean output expected coverage changed.")
    exclusions_path = Path(str(receipt["exclusions_path"])).resolve()
    exclusion_rows = sum(1 for _ in _iter_jsonl(exclusions_path, label="exclusions"))
    expected_exclusions = int(receipt["rejected_overlap_training_rows"]) + int(
        receipt["rejected_overlap_internal_eval_rows"]
    )
    coverage = receipt.get("coverage")
    if (
        not isinstance(coverage, dict)
        or exclusion_rows != int(receipt.get("exclusions_rows", -1))
        or exclusion_rows != expected_exclusions
        or int(coverage.get("excluded_rows", -1)) != expected_exclusions
        or int(coverage.get("source_rows", -1))
        != int(coverage.get("retained_rows", -1)) + expected_exclusions
        or int(coverage.get("retained_rows", -1)) != int(profile["expected"]["total_samples"])
    ):
        raise ValueError("Stage211D public-clean rebuild coverage changed.")
    return receipt
