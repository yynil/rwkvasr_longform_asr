from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.build_stage211_retention_replay import DEFAULT_CELL_TARGETS
    from scripts.create_stage211_supplemental_profile_receipt import (
        build_receipt as build_supplemental_profile_receipt,
    )
    from scripts.validate_stage211_retention_replay import validate_retention_replay
except ModuleNotFoundError:
    from build_stage211_retention_replay import DEFAULT_CELL_TARGETS  # type: ignore[no-redef]
    from create_stage211_supplemental_profile_receipt import (  # type: ignore[no-redef]
        build_receipt as build_supplemental_profile_receipt,
    )
    from validate_stage211_retention_replay import (  # type: ignore[no-redef]
        validate_retention_replay,
    )

from rwkvasr.eval.stage211_supplemental import validate_stage211_supplemental_inventory


BASE_CELLS = tuple(DEFAULT_CELL_TARGETS)
SUPPLEMENTAL_CELLS = ("supplemental_en", "supplemental_zh")
ALL_CELLS = (*BASE_CELLS, *SUPPLEMENTAL_CELLS)
REPLAY_METADATA_KEYS = {
    "_stage211_replay_cell",
    "_stage211_replay_source",
    "_stage211_replay_bucket_id",
    "_stage211_replay_row_sha256",
    "_stage211_replay_score",
    "_stage211_replay_seed",
    "_stage211_replay_source_part",
}
SIDECAR_METADATA_KEYS = {
    "_stage211_sidecar_cell",
    "_stage211_sidecar_row_sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
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


def _keys_sha256(keys: set[str] | list[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _bound_file(
    record: Mapping[str, Any],
    *,
    path_key: str = "path",
    sha256_key: str = "sha256",
    label: str,
) -> Path:
    path = Path(str(record.get(path_key) or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    if record.get(sha256_key) != sha256_file(path):
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("key") or row.get("id") or "")


def _canonical_language(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith(("zh", "cmn")):
        return "zh"
    return None


def _source_dataset(row: Mapping[str, Any], fallback: str | None = None) -> str:
    return str(
        row.get("source_dataset")
        or row.get("_stage179_source")
        or row.get("_stage179_input")
        or fallback
        or "unknown"
    )


def _validate_supplemental_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    inventory_path = _bound_file(
        record,
        path_key="inventory_path",
        sha256_key="inventory_sha256",
        label="Stage211 Supplemental combined inventory",
    )
    profile_path = _bound_file(
        record,
        path_key="profile_receipt_path",
        sha256_key="profile_receipt_sha256",
        label="Stage211 Supplemental profile receipt",
    )
    manifest_path = _bound_file(
        record,
        path_key="manifest_path",
        sha256_key="manifest_sha256",
        label="Stage211 Supplemental manifest",
    )
    validated = validate_stage211_supplemental_inventory(
        inventory_path,
        require_training_ready=True,
        verify_part_sha256=False,
    )
    inventory = validated["inventory"]
    profile = _load_json(profile_path, label="Stage211 Supplemental profile receipt")
    if profile != build_supplemental_profile_receipt(inventory_path):
        raise ValueError("Stage211 Supplemental profile receipt is stale or changed.")
    if (
        inventory.get("schema_version") != 2
        or inventory.get("artifact") != "stage211_supplemental_combined_inventory"
        or set(inventory.get("language") or []) != {"en", "zh"}
        or Path(str(validated["bucket_manifest_path"])).resolve() != manifest_path
        or int(record.get("rows", -1)) != int(validated["rows"])
        or not math.isclose(
            float(record.get("hours", float("nan"))),
            float(validated["hours"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Stage211 Supplemental input binding changed.")
    return validated


def _manifest_parts(
    manifest: Mapping[str, Any], *, split: str
) -> dict[Path, tuple[int, int, str | None]]:
    parts: dict[Path, tuple[int, int, str | None]] = {}
    split_record = manifest.get("splits", {}).get(split, {})
    for bucket in split_record.get("buckets", []):
        bucket_id = int(bucket["bucket_id"])
        if int(bucket.get("num_samples", -1)) != sum(
            int(part["num_samples"]) for part in bucket.get("parts", [])
        ):
            raise ValueError(f"Manifest {split} bucket count mismatch: {bucket_id}")
        for part in bucket.get("parts", []):
            path = Path(str(part["path"])).expanduser()
            if not path.is_absolute():
                raise ValueError(f"Manifest part path is not absolute: {path}")
            path = path.resolve()
            if path in parts:
                raise ValueError(f"Manifest part is duplicated: {path}")
            parts[path] = (
                bucket_id,
                int(part["num_samples"]),
                str(part["source_label"]) if part.get("source_label") is not None else None,
            )
    declared = int(split_record.get("num_samples", 0))
    if declared != sum(count for _, count, _ in parts.values()):
        raise ValueError(f"Manifest {split} total count mismatch.")
    return parts


def validate_stratified_hidden_eval_v2(receipt_path: Path) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 nine-cell stratified receipt")
    expected = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "stratified_hidden_eval_manifest",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 nine-cell stratified receipt contract mismatch.")
    builder = receipt.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("Stage211 nine-cell stratified receipt lacks builder binding.")
    _bound_file(builder, label="Stage211 Supplemental retention builder")

    base_record = receipt.get("base_stratified_receipt")
    if not isinstance(base_record, dict):
        raise ValueError("Stage211 nine-cell receipt lacks its seven-cell base binding.")
    base_path = _bound_file(base_record, label="Stage211 seven-cell stratified receipt")
    base = _load_json(base_path, label="Stage211 seven-cell stratified receipt")
    if (
        base.get("schema_version") != 1
        or base.get("artifact") != "stratified_hidden_eval_manifest"
        or set(base.get("cells") or {}) != set(BASE_CELLS)
        or int(base_record.get("samples", -1)) != int(base.get("combined_samples", -2))
    ):
        raise ValueError("Stage211 seven-cell base receipt contract changed.")
    supplemental_record = receipt.get("supplemental_inputs")
    if not isinstance(supplemental_record, dict):
        raise ValueError("Stage211 nine-cell receipt lacks Supplemental bindings.")
    validated_supplemental = _validate_supplemental_inputs(supplemental_record)

    selection = receipt.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("cells") != list(ALL_CELLS)
        or int(selection.get("per_cell", -1)) <= 0
        or int(selection.get("seed", -1)) < 0
        or selection.get("algorithm")
        != "source_then_80_frame_bucket_balanced_deterministic_round_robin"
    ):
        raise ValueError("Stage211 nine-cell selection contract mismatch.")
    per_cell = int(selection["per_cell"])
    expected_base_per_cell = int(base.get("selection", {}).get("per_cell", -1))
    if expected_base_per_cell != per_cell:
        raise ValueError("Stage211 v1/v2 stratified per-cell counts differ.")
    cells = receipt.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(ALL_CELLS):
        raise ValueError("Stage211 nine-cell stratified coverage mismatch.")
    if any(cells[cell] != base["cells"][cell] for cell in BASE_CELLS):
        raise ValueError("Stage211 nine-cell receipt rewrites a seven-cell record.")

    selected_rows = receipt.get("selected_rows")
    if not isinstance(selected_rows, list):
        raise ValueError("Stage211 nine-cell receipt lacks selected rows.")
    selected_by_key: dict[str, dict[str, Any]] = {}
    for record in selected_rows:
        if not isinstance(record, dict):
            raise ValueError("Stage211 nine-cell selected-row record is invalid.")
        key = str(record.get("key") or "")
        if not key or key in selected_by_key or str(record.get("cell")) not in ALL_CELLS:
            raise ValueError("Stage211 nine-cell selected-row key is missing or duplicated.")
        selected_by_key[key] = record
    base_rows = base.get("selected_rows")
    if not isinstance(base_rows, list) or selected_rows[: len(base_rows)] != sorted(
        base_rows,
        key=lambda row: (str(row["cell"]), str(row["key"])),
    ):
        raise ValueError("Stage211 nine-cell receipt does not preserve v1 selected rows.")

    supplemental_manifest = _load_json(
        Path(str(validated_supplemental["bucket_manifest_path"])).resolve(),
        label="Stage211 Supplemental source manifest",
    )
    supplemental_source_parts = set(_manifest_parts(supplemental_manifest, split="train"))
    seen_keys: set[str] = set()
    all_manifest_parts: dict[Path, tuple[int, int, str | None]] = {}
    for cell_name in ALL_CELLS:
        cell = cells[cell_name]
        if not isinstance(cell, dict) or int(cell.get("samples", -1)) != per_cell:
            raise ValueError(f"Stage211 stratified cell is invalid: {cell_name}")
        manifest_path = _bound_file(
            cell,
            path_key="manifest_path",
            sha256_key="manifest_sha256",
            label=f"Stage211 stratified manifest {cell_name}",
        )
        manifest = _load_json(manifest_path, label=f"Stage211 stratified manifest {cell_name}")
        if _manifest_parts(manifest, split="train"):
            raise ValueError(f"Stage211 stratified manifest has a train split: {cell_name}")
        parts = _manifest_parts(manifest, split="eval")
        if sum(count for _, count, _ in parts.values()) != per_cell:
            raise ValueError(f"Stage211 stratified manifest count mismatch: {cell_name}")
        for part_path, (_, expected_rows, _) in parts.items():
            if part_path in all_manifest_parts:
                raise ValueError(f"Stage211 stratified output part is reused: {part_path}")
            all_manifest_parts[part_path] = parts[part_path]
            actual_rows = 0
            with part_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    actual_rows += 1
                    row = json.loads(line)
                    key = _row_key(row)
                    if not key or key in seen_keys:
                        raise ValueError("Stage211 stratified output key is missing or duplicated.")
                    seen_keys.add(key)
                    selected = selected_by_key.get(key)
                    if selected is None or selected.get("cell") != cell_name:
                        raise ValueError(f"Stage211 stratified row is not selected: {key}")
                    row_sha256 = str(row.get("_stage211_sidecar_row_sha256") or "")
                    original = dict(row)
                    for metadata_key in SIDECAR_METADATA_KEYS:
                        original.pop(metadata_key, None)
                    if (
                        row.get("_stage211_sidecar_cell") != cell_name
                        or row_sha256 != _payload_sha256(original)
                        or row_sha256 != selected.get("row_sha256")
                    ):
                        raise ValueError(f"Stage211 stratified row identity mismatch: {key}")
                    source_part = Path(str(selected.get("source_part") or "")).resolve()
                    if cell_name in SUPPLEMENTAL_CELLS and source_part not in (
                        supplemental_source_parts
                    ):
                        raise ValueError(
                            f"Stage211 Supplemental eval row uses an unbound source part: {key}"
                        )
                    source_parts = receipt.get("source_parts")
                    if not isinstance(source_parts, dict) or source_parts.get(
                        str(source_part)
                    ) != selected.get("source_part_sha256"):
                        raise ValueError(f"Stage211 stratified source binding mismatch: {key}")
            if actual_rows != expected_rows:
                raise ValueError(f"Stage211 stratified part row count mismatch: {part_path}")
    expected_samples = per_cell * len(ALL_CELLS)
    if (
        len(seen_keys) != expected_samples
        or len(selected_by_key) != expected_samples
        or int(receipt.get("combined_samples", -1)) != expected_samples
        or receipt.get("selected_keys_sha256") != _keys_sha256(seen_keys)
    ):
        raise ValueError("Stage211 nine-cell selected-key coverage mismatch.")
    row_records = sorted(
        f"{record['cell']}\0{record['key']}\0{record['row_sha256']}" for record in selected_rows
    )
    if (
        receipt.get("selected_rows_sha256")
        != hashlib.sha256(("\n".join(row_records) + "\n").encode("utf-8")).hexdigest()
    ):
        raise ValueError("Stage211 nine-cell selected-row digest mismatch.")

    combined_path = _bound_file(
        receipt,
        path_key="combined_manifest_path",
        sha256_key="combined_manifest_sha256",
        label="Stage211 nine-cell combined manifest",
    )
    combined = _load_json(combined_path, label="Stage211 nine-cell combined manifest")
    if _manifest_parts(combined, split="train"):
        raise ValueError("Stage211 nine-cell combined manifest has a train split.")
    if _manifest_parts(combined, split="eval") != all_manifest_parts:
        raise ValueError("Stage211 nine-cell combined manifest part coverage mismatch.")
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "validated_unique_keys": len(seen_keys),
    }


def _flatten_counts(
    payload: Any, *, expected_cells: set[str], label: str
) -> Counter[tuple[str, str, int]]:
    if not isinstance(payload, dict) or set(payload) != expected_cells:
        raise ValueError(f"Stage211 Supplemental replay {label} cell coverage mismatch.")
    output: Counter[tuple[str, str, int]] = Counter()
    for cell, sources in payload.items():
        if not isinstance(sources, dict):
            raise ValueError(f"Stage211 Supplemental replay {label} sources are invalid.")
        for source_dataset, buckets in sources.items():
            if not isinstance(buckets, dict):
                raise ValueError(f"Stage211 Supplemental replay {label} buckets are invalid.")
            for bucket, count in buckets.items():
                value = int(count)
                if value <= 0:
                    raise ValueError(f"Stage211 Supplemental replay {label} count is invalid.")
                output[(str(cell), str(source_dataset), int(bucket))] = value
    return output


def _base_replay_records(
    replay: Mapping[str, Any],
) -> tuple[set[str], list[str], int, set[Path]]:
    manifest_path = Path(str(replay["manifest_path"])).resolve()
    manifest = _load_json(manifest_path, label="Stage211 v1 replay manifest")
    keys: set[str] = set()
    rows: list[str] = []
    total_frames = 0
    parts = set(_manifest_parts(manifest, split="train"))
    for part_path in parts:
        with part_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = _row_key(row)
                if not key or key in keys:
                    raise ValueError("Stage211 v1 replay row is missing or duplicated.")
                keys.add(key)
                rows.append(
                    f"{row['_stage211_replay_cell']}\0{key}\0{row['_stage211_replay_row_sha256']}"
                )
                total_frames += int(row.get("num_frames") or 0)
    rows.sort()
    return keys, rows, total_frames, parts


def validate_supplemental_retention_replay(receipt_path: Path) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 Supplemental replay v2 receipt")
    expected = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "retention_replay_manifest",
        "unique_keys": receipt.get("samples"),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 Supplemental replay v2 contract mismatch.")
    builder = receipt.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("Stage211 Supplemental replay v2 lacks builder binding.")
    _bound_file(builder, label="Stage211 Supplemental retention builder")

    base_record = receipt.get("base_replay_receipt")
    if not isinstance(base_record, dict):
        raise ValueError("Stage211 Supplemental replay lacks v1 replay binding.")
    base_path = _bound_file(base_record, label="Stage211 retention replay v1 receipt")
    raw_targets = base_record.get("cell_targets")
    if not isinstance(raw_targets, dict) or set(raw_targets) != set(BASE_CELLS):
        raise ValueError("Stage211 Supplemental replay lacks v1 target bindings.")
    base_targets = {
        cell: None if target == "all" else int(target) for cell, target in raw_targets.items()
    }
    base = validate_retention_replay(
        base_path,
        expected_cell_targets=base_targets,
        expected_fixed_eval_samples=int(base_record.get("fixed_eval_samples", 256)),
    )
    if int(base_record.get("samples", -1)) != int(base["samples"]) or not math.isclose(
        float(base_record.get("hours", float("nan"))),
        float(base["total_hours"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Stage211 Supplemental replay v1 summary changed.")
    stratified_record = receipt.get("stratified_hidden_eval")
    if not isinstance(stratified_record, dict):
        raise ValueError("Stage211 Supplemental replay lacks nine-cell exclusion binding.")
    stratified_path = _bound_file(
        stratified_record,
        label="Stage211 nine-cell stratified receipt",
    )
    stratified = validate_stratified_hidden_eval_v2(stratified_path)
    if int(stratified_record.get("samples", -1)) != int(
        stratified["combined_samples"]
    ) or stratified_record.get("cells") != list(ALL_CELLS):
        raise ValueError("Stage211 Supplemental replay nine-cell summary changed.")
    supplemental_record = receipt.get("supplemental_inputs")
    if not isinstance(supplemental_record, dict):
        raise ValueError("Stage211 Supplemental replay lacks input bindings.")
    validated_supplemental = _validate_supplemental_inputs(supplemental_record)
    supplemental_source_manifest = _load_json(
        Path(str(validated_supplemental["bucket_manifest_path"])).resolve(),
        label="Stage211 Supplemental source manifest",
    )
    supplemental_source_parts = _manifest_parts(
        supplemental_source_manifest,
        split="train",
    )

    preflight_path = _bound_file(
        receipt,
        path_key="capacity_preflight_path",
        sha256_key="capacity_preflight_sha256",
        label="Stage211 Supplemental replay capacity preflight",
    )
    preflight = _load_json(preflight_path, label="Stage211 Supplemental replay preflight")
    if (
        preflight.get("schema_version") != 2
        or preflight.get("artifact") != "supplemental_retention_capacity_preflight"
        or preflight.get("base_replay_receipt", {}).get("path") != str(base_path)
        or preflight.get("base_stratified_receipt", {}).get("path")
        != str(Path(str(stratified["base_stratified_receipt"]["path"])).resolve())
        or preflight.get("supplemental_inputs") != supplemental_record
    ):
        raise ValueError("Stage211 Supplemental replay preflight binding mismatch.")
    selection = receipt.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Stage211 Supplemental replay lacks selection metadata.")
    targets = selection.get("supplemental_cell_targets")
    if (
        not isinstance(targets, dict)
        or set(targets) != set(SUPPLEMENTAL_CELLS)
        or any(int(value) <= 0 for value in targets.values())
        or int(selection.get("bucket_width", -1)) != 80
        or selection.get("base_cell_targets") != raw_targets
    ):
        raise ValueError("Stage211 Supplemental replay selection contract mismatch.")
    replay_seed = int(selection.get("supplemental_seed", -1))
    if replay_seed < 0:
        raise ValueError("Stage211 Supplemental replay seed is invalid.")
    capacities = _flatten_counts(
        receipt.get("capacities"),
        expected_cells=set(SUPPLEMENTAL_CELLS),
        label="capacities",
    )
    quotas = _flatten_counts(
        receipt.get("quotas"),
        expected_cells=set(SUPPLEMENTAL_CELLS),
        label="quotas",
    )
    if capacities != _flatten_counts(
        preflight.get("capacities"),
        expected_cells=set(SUPPLEMENTAL_CELLS),
        label="preflight capacities",
    ) or any(capacities[stratum] < count for stratum, count in quotas.items()):
        raise ValueError("Stage211 Supplemental replay quota/capacity binding mismatch.")

    manifest_path = _bound_file(
        receipt,
        path_key="manifest_path",
        sha256_key="manifest_sha256",
        label="Stage211 Supplemental replay v2 manifest",
    )
    manifest = _load_json(manifest_path, label="Stage211 Supplemental replay v2 manifest")
    manifest_train_parts = _manifest_parts(manifest, split="train")
    manifest_eval_parts = _manifest_parts(manifest, split="eval")
    component_path = _bound_file(
        receipt,
        path_key="supplemental_component_manifest_path",
        sha256_key="supplemental_component_manifest_sha256",
        label="Stage211 Supplemental replay component manifest",
    )
    component = _load_json(component_path, label="Stage211 Supplemental component manifest")
    component_parts = _manifest_parts(component, split="train")
    base_keys, base_rows, base_frames, base_parts = _base_replay_records(base)
    if set(manifest_train_parts) != base_parts | set(component_parts):
        raise ValueError("Stage211 Supplemental replay manifest component coverage mismatch.")
    base_manifest = _load_json(
        Path(str(base["manifest_path"])).resolve(),
        label="Stage211 replay v1 manifest",
    )
    if manifest_eval_parts != _manifest_parts(base_manifest, split="eval"):
        raise ValueError("Stage211 Supplemental replay changed the fixed eval split.")

    raw_output_parts = receipt.get("supplemental_output_parts")
    if not isinstance(raw_output_parts, list) or not raw_output_parts:
        raise ValueError("Stage211 Supplemental replay lacks component output parts.")
    output_records: dict[Path, dict[str, Any]] = {}
    for record in raw_output_parts:
        if not isinstance(record, dict):
            raise ValueError("Stage211 Supplemental replay output record is invalid.")
        path = _bound_file(record, label="Stage211 Supplemental replay output part")
        if path in output_records:
            raise ValueError(f"Stage211 Supplemental output part is duplicated: {path}")
        output_records[path] = record
    if set(output_records) != set(component_parts):
        raise ValueError("Stage211 Supplemental component receipt/manifest parts differ.")

    eval_keys = {str(row["key"]) for row in stratified["selected_rows"]}
    fixed_keys: set[str] = set()
    for part_path in manifest_eval_parts:
        with part_path.open("r", encoding="utf-8") as source:
            fixed_keys.update(_row_key(json.loads(line)) for line in source if line.strip())
    supplemental_keys: set[str] = set()
    supplemental_rows: list[str] = []
    actual_strata: Counter[tuple[str, str, int]] = Counter()
    cell_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts: dict[str, Counter[int]] = defaultdict(Counter)
    frame_ranges: dict[str, list[int]] = {}
    supplemental_frames = 0
    for part_path, (manifest_bucket, expected_rows, source_label) in component_parts.items():
        record = output_records[part_path]
        expected_cell = str(record.get("cell") or "")
        expected_source = str(record.get("source_dataset") or "")
        if (
            int(record.get("bucket_id", -1)) != manifest_bucket
            or int(record.get("num_samples", -1)) != expected_rows
            or record.get("source_label") != source_label
            or source_label != f"{expected_cell}:{expected_source}"
        ):
            raise ValueError(f"Stage211 Supplemental output binding mismatch: {part_path}")
        actual_rows = 0
        with part_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                actual_rows += 1
                row = json.loads(line)
                key = _row_key(row)
                if (
                    not key
                    or key in supplemental_keys
                    or key in base_keys
                    or key in eval_keys
                    or key in fixed_keys
                ):
                    raise ValueError(f"Stage211 Supplemental replay key leaked/duplicated: {key}")
                supplemental_keys.add(key)
                cell = str(row.get("_stage211_replay_cell") or "")
                language = _canonical_language(row.get("language"))
                source_dataset = str(row.get("_stage211_replay_source") or "")
                bucket_id = int(row.get("_stage211_replay_bucket_id", -1))
                num_frames = int(row.get("num_frames") or 0)
                source_part = Path(str(row.get("_stage211_replay_source_part") or "")).resolve()
                source_fallback = supplemental_source_parts.get(source_part, (-1, -1, None))[2]
                original = dict(row)
                for metadata_key in REPLAY_METADATA_KEYS:
                    original.pop(metadata_key, None)
                row_sha256 = str(row.get("_stage211_replay_row_sha256") or "")
                score = hashlib.sha256(
                    (f"{replay_seed}\0{cell}\0{source_dataset}\0{bucket_id}\0{key}").encode("utf-8")
                ).hexdigest()
                if (
                    cell not in SUPPLEMENTAL_CELLS
                    or language not in {"en", "zh"}
                    or cell != f"supplemental_{language}"
                    or cell != expected_cell
                    or source_dataset != expected_source
                    or source_dataset != _source_dataset(row, source_fallback)
                    or num_frames <= 0
                    or bucket_id != num_frames // 80
                    or bucket_id != manifest_bucket
                    or int(row.get("_stage211_replay_seed", -1)) != replay_seed
                    or source_part not in supplemental_source_parts
                    or row_sha256 != _payload_sha256(original)
                    or row.get("_stage211_replay_score") != score
                ):
                    raise ValueError(f"Stage211 Supplemental replay provenance mismatch: {key}")
                actual_strata[(cell, source_dataset, bucket_id)] += 1
                cell_counts[cell] += 1
                language_counts[language] += 1
                source_counts[cell][source_dataset] += 1
                bucket_counts[cell][bucket_id] += 1
                bounds = frame_ranges.setdefault(cell, [num_frames, num_frames])
                bounds[0] = min(bounds[0], num_frames)
                bounds[1] = max(bounds[1], num_frames)
                supplemental_frames += num_frames
                supplemental_rows.append(f"{cell}\0{key}\0{row_sha256}")
        if actual_rows != expected_rows:
            raise ValueError(f"Stage211 Supplemental output row count mismatch: {part_path}")
    if actual_strata != quotas:
        raise ValueError("Stage211 Supplemental replay realized quotas changed.")
    for cell in SUPPLEMENTAL_CELLS:
        if cell_counts[cell] != int(targets[cell]):
            raise ValueError(f"Stage211 Supplemental replay target mismatch: {cell}")
        expected_cell = {
            "samples": cell_counts[cell],
            "sources": dict(sorted(source_counts[cell].items())),
            "buckets": {
                str(bucket): count for bucket, count in sorted(bucket_counts[cell].items())
            },
            "min_frames": frame_ranges[cell][0],
            "max_frames": frame_ranges[cell][1],
        }
        if receipt.get("cells", {}).get(cell) != expected_cell:
            raise ValueError(f"Stage211 Supplemental replay cell summary changed: {cell}")
    if any(receipt.get("cells", {}).get(cell) != base["cells"][cell] for cell in BASE_CELLS):
        raise ValueError("Stage211 Supplemental replay rewrites a v1 cell summary.")

    all_keys = base_keys | supplemental_keys
    all_rows = sorted([*base_rows, *supplemental_rows])
    total_frames = base_frames + supplemental_frames
    expected_language_counts = {
        "en": int(base["language_counts"]["en"]) + language_counts["en"],
        "zh": int(base["language_counts"]["zh"]) + language_counts["zh"],
    }
    if (
        len(all_keys) != int(receipt.get("samples", -1))
        or len(all_keys) != int(receipt.get("unique_keys", -1))
        or len(supplemental_keys) != int(receipt.get("supplemental_samples", -1))
        or total_frames != int(receipt.get("total_frames", -1))
        or supplemental_frames != int(receipt.get("supplemental_frames", -1))
        or not math.isclose(
            float(receipt.get("total_hours", float("nan"))),
            total_frames / 100.0 / 3600.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(receipt.get("supplemental_hours", float("nan"))),
            supplemental_frames / 100.0 / 3600.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or receipt.get("language_counts") != expected_language_counts
        or receipt.get("selected_keys_sha256") != _keys_sha256(all_keys)
        or receipt.get("selected_rows_sha256")
        != hashlib.sha256(("\n".join(all_rows) + "\n").encode("utf-8")).hexdigest()
        or int(receipt.get("manifest_train_samples", -1)) != len(all_keys)
        or int(receipt.get("manifest_eval_samples", -1)) != len(fixed_keys)
    ):
        raise ValueError("Stage211 Supplemental replay aggregate binding mismatch.")
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "validated_unique_keys": len(all_keys),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate Stage211 Supplemental retention v2 artifacts."
    )
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--stratified-receipt", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.receipt is None) == (args.stratified_receipt is None):
        raise ValueError("Specify exactly one of --receipt or --stratified-receipt.")
    if args.receipt is not None:
        validated = validate_supplemental_retention_replay(args.receipt)
        label = "stage211_supplemental_retention_valid"
    else:
        validated = validate_stratified_hidden_eval_v2(args.stratified_receipt)
        label = "stage211_supplemental_stratified_valid"
    print(
        f"{label} samples={validated['validated_unique_keys']} "
        f"receipt_sha256={validated['receipt_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
