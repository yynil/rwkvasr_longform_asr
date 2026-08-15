from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rwkvasr.data import load_webdataset_bucket_manifest


EXPECTED_CELLS = (
    "easy_en",
    "easy_zh",
    "medium_en",
    "medium_zh",
    "hard_en",
    "hard_zh",
    "long_zh",
)
DEFAULT_CELL_TARGETS: dict[str, int | None] = {
    "easy_en": 130_000,
    "easy_zh": 130_000,
    "medium_en": 270_000,
    "medium_zh": 270_000,
    "hard_en": 100_000,
    "hard_zh": 100_000,
    "long_zh": None,
}
REPLAY_METADATA_KEYS = (
    "_stage211_replay_cell",
    "_stage211_replay_source",
    "_stage211_replay_bucket_id",
    "_stage211_replay_row_sha256",
    "_stage211_replay_score",
    "_stage211_replay_seed",
    "_stage211_replay_source_part",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _keys_sha256(keys: Sequence[str] | set[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("key") or row.get("id") or "")


def _canonical_language(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith(("zh", "cmn")):
        return "zh"
    return None


def _source_dataset(row: Mapping[str, Any]) -> str:
    return str(
        row.get("source_dataset")
        or row.get("_stage179_source")
        or row.get("_stage179_input")
        or "unknown"
    )


def _score_candidate(*, seed: int, cell: str, source_dataset: str, bucket_id: int, key: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{cell}\0{source_dataset}\0{bucket_id}\0{key}".encode("utf-8")
    ).hexdigest()


def _read_jsonl_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = _row_key(row)
            if not key:
                raise ValueError(f"Missing evaluation key at {path}:{line_number}")
            keys.add(key)
    return keys


def _validate_bound_path(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    expected_sha256 = str(record.get("sha256") or "")
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def _load_exclusions(receipt: Mapping[str, Any]) -> tuple[set[str], Path, int]:
    records = receipt.get("exclusions")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("Replay receipt must bind exactly two evaluation exclusions.")
    by_kind = {str(record.get("kind")): record for record in records if isinstance(record, dict)}
    expected_kinds = {
        "stratified_hidden_eval",
        "historical_fixed_hidden_eval",
    }
    if set(by_kind) != expected_kinds:
        raise ValueError("Replay exclusion kinds are incomplete or unexpected.")

    sidecar_record = by_kind["stratified_hidden_eval"]
    sidecar_path = _validate_bound_path(
        sidecar_record,
        label="Stage211 stratified hidden-eval receipt",
    )
    sidecar = _load_json(sidecar_path, label="Stage211 stratified hidden-eval receipt")
    if sidecar.get("artifact") != "stratified_hidden_eval_manifest":
        raise ValueError("Replay sidecar exclusion artifact is invalid.")
    sidecar_keys = {str(row.get("key") or "") for row in sidecar.get("selected_rows", [])}
    sidecar_keys.discard("")

    fixed_record = by_kind["historical_fixed_hidden_eval"]
    fixed_path = _validate_bound_path(
        fixed_record,
        label="Stage211 historical fixed hidden-eval part",
    )
    fixed_keys = _read_jsonl_keys(fixed_path)
    for record, keys in (
        (sidecar_record, sidecar_keys),
        (fixed_record, fixed_keys),
    ):
        if int(record.get("unique_keys", -1)) != len(keys) or record.get(
            "keys_sha256"
        ) != _keys_sha256(keys):
            raise ValueError("Replay exclusion key binding mismatch.")

    exclusions = sidecar_keys | fixed_keys
    union = receipt.get("exclusion_union")
    if not isinstance(union, dict) or (
        int(union.get("unique_keys", -1)) != len(exclusions)
        or union.get("keys_sha256") != _keys_sha256(exclusions)
    ):
        raise ValueError("Replay exclusion union binding mismatch.")
    return exclusions, fixed_path, len(fixed_keys)


def _flatten_nested_counts(payload: Any, *, label: str) -> Counter[tuple[str, str, int]]:
    if not isinstance(payload, dict):
        raise ValueError(f"Replay {label} must be an object.")
    output: Counter[tuple[str, str, int]] = Counter()
    for cell, sources in payload.items():
        if cell not in EXPECTED_CELLS or not isinstance(sources, dict):
            raise ValueError(f"Replay {label} has an invalid cell: {cell}")
        for source_dataset, buckets in sources.items():
            if not isinstance(buckets, dict):
                raise ValueError(f"Replay {label} has invalid source buckets.")
            for bucket_id, count in buckets.items():
                value = int(count)
                if value <= 0:
                    raise ValueError(f"Replay {label} counts must be positive.")
                output[(str(cell), str(source_dataset), int(bucket_id))] += value
    return output


def _validate_retention_replay_v1(
    receipt_path: Path,
    *,
    expected_cell_targets: Mapping[str, int | None] = DEFAULT_CELL_TARGETS,
    expected_fixed_eval_samples: int = 256,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 retention replay receipt")
    expected_fields = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "retention_replay_manifest",
        "unique_keys": receipt.get("samples"),
    }
    if any(receipt.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Stage211 retention replay receipt contract mismatch.")
    if set(expected_cell_targets) != set(EXPECTED_CELLS):
        raise ValueError("Expected cell targets do not cover the seven replay cells.")

    builder = receipt.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("Replay receipt lacks a builder binding.")
    _validate_bound_path(builder, label="Stage211 replay builder")
    source_manifests = receipt.get("source_manifests")
    if not isinstance(source_manifests, dict) or set(source_manifests) != {
        "easy",
        "medium",
        "hard",
        "long",
    }:
        raise ValueError("Replay source-manifest coverage mismatch.")
    for difficulty, record in source_manifests.items():
        if not isinstance(record, dict):
            raise ValueError(f"Replay source manifest record is invalid: {difficulty}")
        _validate_bound_path(record, label=f"Stage211 replay {difficulty} source manifest")
        if (
            int(record.get("declared_train_samples", -1)) <= 0
            or int(record.get("part_count", -1)) <= 0
            or len(str(record.get("part_inventory_sha256") or "")) != 64
        ):
            raise ValueError(f"Replay source manifest inventory is invalid: {difficulty}")

    exclusions, fixed_eval_path, fixed_eval_samples = _load_exclusions(receipt)
    if fixed_eval_samples != expected_fixed_eval_samples:
        raise ValueError(
            "Replay fixed-eval sample count mismatch: "
            f"actual={fixed_eval_samples} expected={expected_fixed_eval_samples}"
        )
    preflight_path = Path(str(receipt.get("capacity_preflight_path") or "")).resolve()
    if not preflight_path.is_file() or receipt.get("capacity_preflight_sha256") != sha256_file(
        preflight_path
    ):
        raise ValueError("Replay capacity preflight is missing or changed.")
    preflight = _load_json(preflight_path, label="Stage211 replay capacity preflight")
    expected_preflight_fields = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "retention_replay_capacity_preflight",
        "bucket_width": 80,
        "source_manifests": source_manifests,
        "exclusions": receipt.get("exclusions"),
        "exclusion_union": receipt.get("exclusion_union"),
    }
    if any(preflight.get(key) != value for key, value in expected_preflight_fields.items()):
        raise ValueError("Replay capacity preflight contract mismatch.")

    selection = receipt.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Replay receipt lacks selection metadata.")
    seed = int(selection.get("seed", -1))
    bucket_width = int(selection.get("bucket_width", -1))
    if seed < 0 or bucket_width != 80:
        raise ValueError("Replay selection seed or bucket width is invalid.")
    expected_algorithm = (
        "two_pass_equal_source_then_equal_80_frame_bucket_"
        "deterministic_priority_reservoir_without_replacement"
    )
    if selection.get("algorithm") != expected_algorithm:
        raise ValueError("Replay selection algorithm mismatch.")
    requested_targets = selection.get("requested_cell_targets")
    realized_targets = selection.get("realized_cell_targets")
    if not isinstance(requested_targets, dict) or not isinstance(realized_targets, dict):
        raise ValueError("Replay receipt lacks cell targets.")
    expected_requested = {
        cell: ("all" if target is None else int(target))
        for cell, target in expected_cell_targets.items()
    }
    if requested_targets != expected_requested or set(realized_targets) != set(EXPECTED_CELLS):
        raise ValueError("Replay requested or realized cell targets mismatch.")

    manifest_path = Path(str(receipt.get("manifest_path") or "")).resolve()
    if not manifest_path.is_file() or receipt.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Replay manifest is missing or changed.")
    manifest = load_webdataset_bucket_manifest(manifest_path)
    if manifest.bucket_width != 80:
        raise ValueError("Replay manifest must use 80-frame buckets.")
    train_samples = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
    eval_samples = sum(bucket.num_samples for bucket in manifest.splits.get("eval", ()))
    if (
        train_samples != int(receipt.get("samples", -1))
        or train_samples != int(receipt.get("manifest_train_samples", -1))
        or eval_samples != expected_fixed_eval_samples
        or eval_samples != int(receipt.get("manifest_eval_samples", -1))
    ):
        raise ValueError("Replay manifest split counts mismatch.")
    eval_parts = [
        Path(part.path).resolve()
        for bucket in manifest.splits.get("eval", ())
        for part in bucket.parts
    ]
    if eval_parts != [fixed_eval_path]:
        raise ValueError("Replay manifest does not bind the exact fixed-eval part.")

    raw_output_parts = receipt.get("output_parts")
    if not isinstance(raw_output_parts, list) or not raw_output_parts:
        raise ValueError("Replay receipt lacks output parts.")
    receipt_parts: dict[Path, dict[str, Any]] = {}
    for record in raw_output_parts:
        if not isinstance(record, dict):
            raise ValueError("Replay output-part record is invalid.")
        part_path = _validate_bound_path(record, label="Stage211 replay output part")
        if part_path in receipt_parts:
            raise ValueError(f"Replay output part is duplicated: {part_path}")
        receipt_parts[part_path] = record
    manifest_parts: dict[Path, tuple[int, int, str | None]] = {}
    for bucket in manifest.splits.get("train", ()):
        for part in bucket.parts:
            path = Path(part.path).resolve()
            if path in manifest_parts:
                raise ValueError(f"Replay manifest part is duplicated: {path}")
            manifest_parts[path] = (bucket.bucket_id, part.num_samples, part.source_label)
    if set(receipt_parts) != set(manifest_parts):
        raise ValueError("Replay receipt and manifest output-part sets differ.")

    selected_keys: set[str] = set()
    selected_row_records: list[str] = []
    actual_strata: Counter[tuple[str, str, int]] = Counter()
    cell_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts: dict[str, Counter[int]] = defaultdict(Counter)
    frame_ranges: dict[str, list[int]] = {}
    total_frames = 0
    for part_path, (manifest_bucket, manifest_count, source_label) in manifest_parts.items():
        record = receipt_parts[part_path]
        if (
            int(record.get("bucket_id", -1)) != manifest_bucket
            or int(record.get("num_samples", -1)) != manifest_count
            or record.get("source_label") != source_label
        ):
            raise ValueError(f"Replay output-part manifest binding mismatch: {part_path}")
        expected_cell = str(record.get("cell") or "")
        expected_source = str(record.get("source_dataset") or "")
        if source_label != f"{expected_cell}:{expected_source}":
            raise ValueError(f"Replay output-part source label mismatch: {part_path}")
        part_rows = 0
        with part_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                part_rows += 1
                row = json.loads(line)
                key = _row_key(row)
                if not key or key in selected_keys:
                    raise ValueError(
                        f"Replay row key is missing or duplicated at {part_path}:{line_number}"
                    )
                if key in exclusions:
                    raise ValueError(f"Evaluation key leaked into replay: {key}")
                selected_keys.add(key)
                cell = str(row.get("_stage211_replay_cell") or "")
                source_dataset = str(row.get("_stage211_replay_source") or "")
                bucket_id = int(row.get("_stage211_replay_bucket_id", -1))
                row_seed = int(row.get("_stage211_replay_seed", -1))
                row_sha256 = str(row.get("_stage211_replay_row_sha256") or "")
                score = str(row.get("_stage211_replay_score") or "")
                language = _canonical_language(row.get("language"))
                num_frames = int(row.get("num_frames") or 0)
                source_part = Path(str(row.get("_stage211_replay_source_part") or "")).resolve()
                if (
                    cell not in EXPECTED_CELLS
                    or language not in {"en", "zh"}
                    or cell != f"{cell.rsplit('_', 1)[0]}_{language}"
                    or source_dataset != _source_dataset(row)
                    or source_dataset != expected_source
                    or cell != expected_cell
                    or num_frames <= 0
                    or bucket_id != num_frames // 80
                    or bucket_id != manifest_bucket
                    or row_seed != seed
                    or not source_part.is_file()
                ):
                    raise ValueError(f"Replay row provenance mismatch for key={key}")
                original_row = dict(row)
                for metadata_key in REPLAY_METADATA_KEYS:
                    original_row.pop(metadata_key, None)
                if row_sha256 != _payload_sha256(original_row):
                    raise ValueError(f"Replay original-row SHA-256 mismatch for key={key}")
                if score != _score_candidate(
                    seed=seed,
                    cell=cell,
                    source_dataset=source_dataset,
                    bucket_id=bucket_id,
                    key=key,
                ):
                    raise ValueError(f"Replay deterministic score mismatch for key={key}")
                actual_strata[(cell, source_dataset, bucket_id)] += 1
                total_frames += num_frames
                cell_counts[cell] += 1
                language_counts[language] += 1
                source_counts[cell][source_dataset] += 1
                bucket_counts[cell][bucket_id] += 1
                bounds = frame_ranges.setdefault(cell, [num_frames, num_frames])
                bounds[0] = min(bounds[0], num_frames)
                bounds[1] = max(bounds[1], num_frames)
                selected_row_records.append(f"{cell}\0{key}\0{row_sha256}")
        if part_rows != manifest_count:
            raise ValueError(
                f"Replay output-part row count mismatch: {part_path} "
                f"actual={part_rows} expected={manifest_count}"
            )

    if len(selected_keys) != train_samples:
        raise ValueError("Replay globally unique row count mismatch.")
    if (
        int(receipt.get("total_frames", -1)) != total_frames
        or abs(float(receipt.get("total_hours", -1.0)) - total_frames / 100.0 / 3600.0) > 1e-9
    ):
        raise ValueError("Replay total frame/hour exposure mismatch.")
    if receipt.get("selected_keys_sha256") != _keys_sha256(selected_keys):
        raise ValueError("Replay selected-key digest mismatch.")
    selected_row_records.sort()
    rows_sha256 = hashlib.sha256(
        "\n".join(selected_row_records).encode("utf-8") + b"\n"
    ).hexdigest()
    if receipt.get("selected_rows_sha256") != rows_sha256:
        raise ValueError("Replay selected-row digest mismatch.")

    quotas = _flatten_nested_counts(receipt.get("quotas"), label="quotas")
    capacities = _flatten_nested_counts(receipt.get("capacities"), label="capacities")
    preflight_capacities = _flatten_nested_counts(
        preflight.get("capacities"),
        label="capacity preflight",
    )
    if quotas != actual_strata or any(
        capacities[stratum] < count for stratum, count in quotas.items()
    ):
        raise ValueError("Replay quota/capacity audit mismatch.")
    if capacities != preflight_capacities or receipt.get("scan") != preflight.get("scan"):
        raise ValueError("Replay receipt differs from its capacity preflight.")
    cells = receipt.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(EXPECTED_CELLS):
        raise ValueError("Replay cell summaries are incomplete.")
    for cell in EXPECTED_CELLS:
        requested = expected_cell_targets[cell]
        realized = int(realized_targets[cell])
        if requested is not None and realized != int(requested):
            raise ValueError(f"Replay finite target mismatch for {cell}.")
        if realized != cell_counts[cell]:
            raise ValueError(f"Replay realized target mismatch for {cell}.")
        expected_cell_summary = {
            "samples": cell_counts[cell],
            "sources": dict(sorted(source_counts[cell].items())),
            "buckets": {
                str(bucket_id): count for bucket_id, count in sorted(bucket_counts[cell].items())
            },
            "min_frames": frame_ranges[cell][0],
            "max_frames": frame_ranges[cell][1],
        }
        if cells[cell] != expected_cell_summary:
            raise ValueError(f"Replay cell summary mismatch for {cell}.")
    if receipt.get("language_counts") != dict(sorted(language_counts.items())):
        raise ValueError("Replay language-count summary mismatch.")
    expected_en = sum(
        int(target)
        for cell, target in expected_cell_targets.items()
        if cell.endswith("_en") and target is not None
    )
    expected_zh_without_long = sum(
        int(target)
        for cell, target in expected_cell_targets.items()
        if cell.endswith("_zh") and cell != "long_zh" and target is not None
    )
    if (
        language_counts["en"] != expected_en
        or language_counts["zh"] != expected_zh_without_long + cell_counts["long_zh"]
    ):
        raise ValueError("Replay English/Chinese balance contract mismatch.")
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "validated_unique_keys": len(selected_keys),
    }


def validate_retention_replay(
    receipt_path: Path,
    *,
    expected_cell_targets: Mapping[str, int | None] = DEFAULT_CELL_TARGETS,
    expected_fixed_eval_samples: int = 256,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 retention replay receipt")
    if int(receipt.get("schema_version", -1)) == 2:
        if expected_cell_targets != DEFAULT_CELL_TARGETS or expected_fixed_eval_samples != 256:
            raise ValueError("Stage211 replay v2 uses its receipt-bound production targets.")
        try:
            from scripts.validate_stage211_supplemental_retention import (
                validate_supplemental_retention_replay,
            )
        except ModuleNotFoundError:
            from validate_stage211_supplemental_retention import (  # type: ignore[no-redef]
                validate_supplemental_retention_replay,
            )

        return validate_supplemental_retention_replay(receipt_path)
    return _validate_retention_replay_v1(
        receipt_path,
        expected_cell_targets=expected_cell_targets,
        expected_fixed_eval_samples=expected_fixed_eval_samples,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate a production Stage211 retention replay receipt."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validated = validate_retention_replay(args.receipt)
    print(
        "stage211_retention_replay_valid "
        f"samples={validated['validated_unique_keys']} "
        f"receipt_sha256={validated['receipt_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
