from __future__ import annotations

import argparse
import fcntl
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.build_stage211_retention_replay import (
        DEFAULT_CELL_TARGETS as BASE_CELL_TARGETS,
        ReplayCandidate,
        SourcePart,
        _allocate_even_with_capacity,
        _iter_part_rows,
        _keys_sha256,
        _load_source_parts,
        _make_candidate,
        _nested_counts,
        _payload_sha256,
        _row_key,
        _source_dataset,
        _write_immutable,
        _write_outputs,
        sha256_file,
    )
    from scripts.create_stage211_supplemental_profile_receipt import (
        build_receipt as build_supplemental_profile_receipt,
    )
    from scripts.validate_stage211_retention_replay import validate_retention_replay
except ModuleNotFoundError:
    from build_stage211_retention_replay import (  # type: ignore[no-redef]
        DEFAULT_CELL_TARGETS as BASE_CELL_TARGETS,
        ReplayCandidate,
        SourcePart,
        _allocate_even_with_capacity,
        _iter_part_rows,
        _keys_sha256,
        _load_source_parts,
        _make_candidate,
        _nested_counts,
        _payload_sha256,
        _row_key,
        _source_dataset,
        _write_immutable,
        _write_outputs,
        sha256_file,
    )
    from create_stage211_supplemental_profile_receipt import (  # type: ignore[no-redef]
        build_receipt as build_supplemental_profile_receipt,
    )
    from validate_stage211_retention_replay import (  # type: ignore[no-redef]
        validate_retention_replay,
    )

from rwkvasr.eval.stage211_supplemental import validate_stage211_supplemental_inventory


BASE_CELLS = tuple(BASE_CELL_TARGETS)
SUPPLEMENTAL_CELLS = ("supplemental_en", "supplemental_zh")
ALL_CELLS = (*BASE_CELLS, *SUPPLEMENTAL_CELLS)
DEFAULT_REPLAY_PER_LANGUAGE = 130_000
DEFAULT_EVAL_PER_CELL = 256
DEFAULT_REPLAY_SEED = 2112
DEFAULT_EVAL_SEED = 212
DEFAULT_BUCKET_WIDTH = 80
DEFAULT_MAX_ROWS_PER_PART = 100_000
DEFAULT_CANDIDATES_PER_PART = 64
DEFAULT_MAX_EVAL_SCAN_ROWS_PER_PART = 8192


def _default_paths() -> dict[str, Path]:
    metadata = Path.home() / "rwkvasr_data" / "stage211_full_curriculum"
    supplemental = Path.home() / "rwkvasr_data" / "stage211_supplemental_combined_v3"
    return {
        "base_replay_receipt": metadata / "retention_replay_v1" / "receipt.json",
        "base_stratified_receipt": metadata / "stratified_hidden_eval_v1" / "receipt.json",
        "supplemental_inventory": supplemental / "supplemental_inventory.json",
        "supplemental_profile_receipt": supplemental / "supplemental_profile_receipt.json",
        "output_root": metadata,
    }


@dataclass(frozen=True, slots=True)
class EvalCandidate:
    language: str
    cell: str
    source_dataset: str
    bucket_id: int
    key: str
    num_frames: int
    score: str
    row_sha256: str
    source_part: Path
    row: dict[str, Any]


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _canonical_language(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith(("zh", "cmn")):
        return "zh"
    return None


def _bound_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_supplemental_inputs(
    *, inventory_path: Path, profile_receipt_path: Path
) -> dict[str, Any]:
    inventory_path = inventory_path.expanduser().resolve()
    profile_receipt_path = profile_receipt_path.expanduser().resolve()
    validated = validate_stage211_supplemental_inventory(
        inventory_path,
        require_training_ready=True,
        verify_part_sha256=True,
    )
    inventory = validated["inventory"]
    if (
        inventory.get("schema_version") != 2
        or inventory.get("artifact") != "stage211_supplemental_combined_inventory"
        or set(inventory.get("language") or []) != {"en", "zh"}
    ):
        raise ValueError("Supplemental retention requires the combined-v2 EN/ZH inventory.")
    profile = _load_json(
        profile_receipt_path,
        label="Stage211 Supplemental profile receipt",
    )
    if profile != build_supplemental_profile_receipt(inventory_path):
        raise ValueError("Stage211 Supplemental profile receipt is stale or changed.")
    manifest_path = Path(str(validated["bucket_manifest_path"])).resolve()
    return {
        "inventory": inventory,
        "inventory_path": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "profile_receipt": profile,
        "profile_receipt_path": str(profile_receipt_path),
        "profile_receipt_sha256": sha256_file(profile_receipt_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": int(validated["rows"]),
        "hours": float(validated["hours"]),
    }


def _base_selected_rows(
    replay: Mapping[str, Any],
) -> tuple[set[str], list[str], int]:
    manifest = _load_json(
        Path(str(replay["manifest_path"])).resolve(),
        label="Stage211 base replay manifest",
    )
    keys: set[str] = set()
    records: list[str] = []
    total_frames = 0
    for bucket in manifest.get("splits", {}).get("train", {}).get("buckets", []):
        for part in bucket.get("parts", []):
            part_path = Path(str(part["path"])).resolve()
            with part_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = _row_key(row)
                    cell = str(row.get("_stage211_replay_cell") or "")
                    row_sha256 = str(row.get("_stage211_replay_row_sha256") or "")
                    if not key or key in keys or cell not in BASE_CELLS:
                        raise ValueError("Base replay row identity is invalid.")
                    keys.add(key)
                    records.append(f"{cell}\0{key}\0{row_sha256}")
                    total_frames += int(row.get("num_frames") or 0)
    records.sort()
    if (
        len(keys) != int(replay["samples"])
        or _keys_sha256(keys) != replay["selected_keys_sha256"]
        or hashlib.sha256(("\n".join(records) + "\n").encode("utf-8")).hexdigest()
        != replay["selected_rows_sha256"]
        or total_frames != int(replay["total_frames"])
    ):
        raise ValueError("Base replay selected-row binding changed.")
    return keys, records, total_frames


def _base_stratified_keys(receipt: Mapping[str, Any]) -> set[str]:
    rows = receipt.get("selected_rows")
    if not isinstance(rows, list):
        raise ValueError("Base stratified receipt lacks selected rows.")
    keys = {str(row.get("key") or "") for row in rows if isinstance(row, dict)}
    keys.discard("")
    if len(keys) != int(receipt.get("combined_samples", -1)):
        raise ValueError("Base stratified selected keys are incomplete.")
    return keys


def _eval_score(*, seed: int, language: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\0supplemental\0{language}\0{key}".encode("utf-8")).hexdigest()


def _replay_score(
    *, seed: int, cell: str, source_dataset: str, bucket_id: int, key: str
) -> tuple[str, int]:
    value = hashlib.sha256(
        f"{seed}\0{cell}\0{source_dataset}\0{bucket_id}\0{key}".encode("utf-8")
    ).hexdigest()
    return value, int(value, 16)


def _classify(
    *, part: SourcePart, row: Mapping[str, Any], bucket_width: int
) -> tuple[str, str, str, int, str, int] | None:
    language = _canonical_language(row.get("language"))
    key = _row_key(row)
    try:
        num_frames = int(row.get("num_frames") or 0)
    except (TypeError, ValueError):
        return None
    if language not in {"en", "zh"} or not key or num_frames <= 0:
        return None
    return (
        f"supplemental_{language}",
        language,
        _source_dataset(row, part.source_label),
        num_frames // bucket_width,
        key,
        num_frames,
    )


def _select_balanced(
    candidates: Sequence[EvalCandidate], *, count: int, seed: int, scope: str
) -> list[EvalCandidate]:
    unique: dict[str, EvalCandidate] = {}
    for candidate in candidates:
        previous = unique.get(candidate.key)
        if previous is None or (candidate.score, candidate.key) < (
            previous.score,
            previous.key,
        ):
            unique[candidate.key] = candidate
    strata: dict[tuple[str, int], list[EvalCandidate]] = defaultdict(list)
    for candidate in unique.values():
        strata[(candidate.source_dataset, candidate.bucket_id)].append(candidate)
    for values in strata.values():
        values.sort(key=lambda item: (item.score, item.key))
    order = sorted(
        strata,
        key=lambda label: hashlib.sha256(
            f"{seed}\0{scope}\0{label[0]}\0{label[1]}".encode("utf-8")
        ).hexdigest(),
    )
    offsets = {label: 0 for label in order}
    selected: list[EvalCandidate] = []
    while len(selected) < count:
        added = False
        for label in order:
            offset = offsets[label]
            if offset >= len(strata[label]):
                continue
            selected.append(strata[label][offset])
            offsets[label] = offset + 1
            added = True
            if len(selected) == count:
                break
        if not added:
            break
    if len(selected) != count:
        raise ValueError(
            f"Supplemental stratified cell {scope} has {len(selected)} rows; required {count}."
        )
    return selected


def _scan_capacities_and_eval(
    *,
    parts: Sequence[SourcePart],
    excluded_keys: set[str],
    bucket_width: int,
    eval_per_cell: int,
    candidates_per_part: int,
    max_eval_scan_rows_per_part: int,
    eval_seed: int,
) -> tuple[Counter[tuple[str, str, int]], list[EvalCandidate], dict[str, Any]]:
    capacities: Counter[tuple[str, str, int]] = Counter()
    candidate_pool: dict[str, list[EvalCandidate]] = {"en": [], "zh": []}
    scan: dict[str, Counter[str]] = {
        "en": Counter(),
        "zh": Counter(),
        "all": Counter(),
    }
    for part_index, part in enumerate(parts, start=1):
        per_part: dict[str, list[EvalCandidate]] = {"en": [], "zh": []}
        for row_index, row in enumerate(_iter_part_rows(part)):
            scan["all"]["source_rows"] += 1
            classified = _classify(part=part, row=row, bucket_width=bucket_width)
            if classified is None:
                scan["all"]["ignored_invalid_or_language"] += 1
                continue
            cell, language, source_dataset, bucket_id, key, num_frames = classified
            if key in excluded_keys:
                scan[language]["excluded_existing_key"] += 1
                continue
            capacities[(cell, source_dataset, bucket_id)] += 1
            scan[language]["eligible_before_eval"] += 1
            if row_index >= max_eval_scan_rows_per_part:
                continue
            per_part[language].append(
                EvalCandidate(
                    language=language,
                    cell=cell,
                    source_dataset=source_dataset,
                    bucket_id=bucket_id,
                    key=key,
                    num_frames=num_frames,
                    score=_eval_score(seed=eval_seed, language=language, key=key),
                    row_sha256=_payload_sha256(row),
                    source_part=part.path,
                    row=dict(row),
                )
            )
        for language, values in per_part.items():
            values.sort(key=lambda item: (item.score, item.key))
            candidate_pool[language].extend(values[:candidates_per_part])
        if part_index % 100 == 0:
            print(
                "stage211_supplemental_retention "
                f"scan=capacity parts={part_index}/{len(parts)} "
                f"rows={scan['all']['source_rows']}",
                flush=True,
            )
    selected: list[EvalCandidate] = []
    for language in ("en", "zh"):
        values = _select_balanced(
            candidate_pool[language],
            count=eval_per_cell,
            seed=eval_seed,
            scope=f"supplemental_{language}",
        )
        selected.extend(values)
        for candidate in values:
            stratum = (candidate.cell, candidate.source_dataset, candidate.bucket_id)
            capacities[stratum] -= 1
            if capacities[stratum] <= 0:
                del capacities[stratum]
            scan[language]["withheld_eval"] += 1
    if len({candidate.key for candidate in selected}) != len(selected):
        raise ValueError("Supplemental EN/ZH stratified cells share duplicate keys.")
    return (
        capacities,
        selected,
        {
            "total_source_rows": scan["all"]["source_rows"],
            "ignored_invalid_or_language": scan["all"]["ignored_invalid_or_language"],
            "by_language": {
                language: dict(sorted(scan[language].items())) for language in ("en", "zh")
            },
        },
    )


def _build_supplemental_quotas(
    *, capacities: Counter[tuple[str, str, int]], target_per_language: int, seed: int
) -> dict[tuple[str, str, int], int]:
    output: dict[tuple[str, str, int], int] = {}
    for cell in SUPPLEMENTAL_CELLS:
        sources: dict[str, dict[int, int]] = defaultdict(dict)
        for (candidate_cell, source, bucket), capacity in capacities.items():
            if candidate_cell == cell:
                sources[source][bucket] = capacity
        source_quotas = _allocate_even_with_capacity(
            {source: sum(buckets.values()) for source, buckets in sources.items()},
            target=target_per_language,
            seed=seed,
            scope=f"{cell}:sources",
        )
        for source, source_quota in source_quotas.items():
            bucket_quotas = _allocate_even_with_capacity(
                sources[source],
                target=source_quota,
                seed=seed,
                scope=f"{cell}:{source}:buckets",
            )
            for bucket, quota in bucket_quotas.items():
                output[(cell, source, bucket)] = quota
    return output


def _sample_replay(
    *,
    parts: Sequence[SourcePart],
    excluded_keys: set[str],
    quotas: Mapping[tuple[str, str, int], int],
    bucket_width: int,
    seed: int,
) -> list[ReplayCandidate]:
    heaps: dict[tuple[str, str, int], list[tuple[int, int, ReplayCandidate]]] = defaultdict(list)
    heap_keys: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    serial = 0
    total_rows = 0
    for part_index, part in enumerate(parts, start=1):
        for row in _iter_part_rows(part):
            total_rows += 1
            classified = _classify(part=part, row=row, bucket_width=bucket_width)
            if classified is None:
                continue
            cell, language, source_dataset, bucket_id, key, num_frames = classified
            if key in excluded_keys:
                continue
            stratum = (cell, source_dataset, bucket_id)
            quota = int(quotas.get(stratum, 0))
            if quota <= 0 or key in heap_keys[stratum]:
                continue
            score_hex, score_int = _replay_score(
                seed=seed,
                cell=cell,
                source_dataset=source_dataset,
                bucket_id=bucket_id,
                key=key,
            )
            heap = heaps[stratum]
            if len(heap) >= quota and score_int >= -heap[0][0]:
                continue
            candidate = _make_candidate(
                row=row,
                part=part,
                cell=cell,
                language=language,
                source_dataset=source_dataset,
                bucket_id=bucket_id,
                key=key,
                num_frames=num_frames,
                seed=seed,
                score_hex=score_hex,
                score_int=score_int,
            )
            serial += 1
            entry = (-score_int, serial, candidate)
            if len(heap) < quota:
                heapq.heappush(heap, entry)
                heap_keys[stratum].add(key)
            elif score_int < -heap[0][0]:
                removed = heapq.heapreplace(heap, entry)[2]
                heap_keys[stratum].remove(removed.key)
                heap_keys[stratum].add(key)
        if part_index % 100 == 0:
            print(
                "stage211_supplemental_retention "
                f"scan=reservoir parts={part_index}/{len(parts)} rows={total_rows}",
                flush=True,
            )
    selected: list[ReplayCandidate] = []
    for stratum, quota in quotas.items():
        if len(heaps[stratum]) != quota:
            raise ValueError(
                f"Supplemental replay reservoir underfilled for {stratum}: "
                f"required={quota} actual={len(heaps[stratum])}"
            )
        selected.extend(entry[2] for entry in heaps[stratum])
    selected.sort(
        key=lambda item: (
            item.bucket_id,
            item.cell,
            item.source_dataset,
            item.score_hex,
            item.key,
        )
    )
    keys = [candidate.key for candidate in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Supplemental replay selection contains duplicate keys.")
    return selected


def _render_eval_manifest(
    *, bucket_parts: Mapping[int, list[dict[str, Any]]], samples: int
) -> dict[str, Any]:
    return {
        "bucket_metric": "audio_frames_plus_text_cost",
        "bucket_width": DEFAULT_BUCKET_WIDTH,
        "entries_per_part": samples,
        "root": "/",
        "source_length_index_path": "/dev/null",
        "splits": {
            "eval": {
                "num_samples": samples,
                "buckets": [
                    {
                        "bucket_id": bucket_id,
                        "num_samples": sum(int(part["num_samples"]) for part in parts),
                        "parts": parts,
                    }
                    for bucket_id, parts in sorted(bucket_parts.items())
                ],
            }
        },
    }


def _write_stratified_v2(
    *,
    base_receipt_path: Path,
    base_receipt: Mapping[str, Any],
    selected: Sequence[EvalCandidate],
    supplemental: Mapping[str, Any],
    output_dir: Path,
    eval_seed: int,
    eval_per_cell: int,
    candidates_per_part: int,
    max_eval_scan_rows_per_part: int,
) -> tuple[Path, dict[str, Any]]:
    new_cells: dict[str, dict[str, Any]] = {}
    combined_new_parts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    selected_rows: list[dict[str, Any]] = list(base_receipt["selected_rows"])
    source_parts = dict(base_receipt.get("source_parts") or {})
    for cell in SUPPLEMENTAL_CELLS:
        by_bucket: dict[int, list[EvalCandidate]] = defaultdict(list)
        for candidate in selected:
            if candidate.cell == cell:
                by_bucket[candidate.bucket_id].append(candidate)
        cell_parts: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for bucket_id, values in sorted(by_bucket.items()):
            values.sort(key=lambda item: (item.score, item.key))
            part_path = output_dir / "parts" / cell / f"bucket_{bucket_id:04d}.jsonl"
            rendered: list[str] = []
            for candidate in values:
                row = dict(candidate.row)
                row["_stage211_sidecar_cell"] = cell
                row["_stage211_sidecar_row_sha256"] = candidate.row_sha256
                rendered.append(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                part_sha256 = source_parts.get(str(candidate.source_part))
                if part_sha256 is None:
                    part_sha256 = sha256_file(candidate.source_part)
                    source_parts[str(candidate.source_part)] = part_sha256
                selected_rows.append(
                    {
                        "cell": cell,
                        "key": candidate.key,
                        "row_sha256": candidate.row_sha256,
                        "source_part": str(candidate.source_part),
                        "source_part_sha256": part_sha256,
                    }
                )
            _write_immutable(part_path, "".join(rendered))
            record = {
                "path": str(part_path),
                "num_samples": len(values),
                "source_label": cell,
            }
            cell_parts[bucket_id].append(record)
            combined_new_parts[bucket_id].append(record)
        manifest = _render_eval_manifest(bucket_parts=cell_parts, samples=eval_per_cell)
        manifest_path = output_dir / f"manifest_{cell}.json"
        _write_immutable(
            manifest_path,
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        )
        values = [candidate for candidate in selected if candidate.cell == cell]
        new_cells[cell] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "samples": len(values),
            "source_datasets": sorted({candidate.source_dataset for candidate in values}),
            "bucket_ids": sorted({candidate.bucket_id for candidate in values}),
            "min_frames": min(candidate.num_frames for candidate in values),
            "max_frames": max(candidate.num_frames for candidate in values),
        }

    base_combined = _load_json(
        Path(str(base_receipt["combined_manifest_path"])).resolve(),
        label="Stage211 base stratified combined manifest",
    )
    combined_parts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for bucket in base_combined.get("splits", {}).get("eval", {}).get("buckets", []):
        combined_parts[int(bucket["bucket_id"])].extend(bucket.get("parts", []))
    for bucket_id, parts in combined_new_parts.items():
        combined_parts[bucket_id].extend(parts)
    for parts in combined_parts.values():
        parts.sort(key=lambda item: (str(item.get("source_label") or ""), str(item["path"])))
    combined_samples = int(base_receipt["combined_samples"]) + len(selected)
    combined_manifest = _render_eval_manifest(
        bucket_parts=combined_parts,
        samples=combined_samples,
    )
    combined_manifest_path = output_dir / "manifest_all.json"
    _write_immutable(
        combined_manifest_path,
        json.dumps(combined_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    cells = {**base_receipt["cells"], **new_cells}
    selected_rows.sort(key=lambda row: (str(row["cell"]), str(row["key"])))
    selected_keys = [str(row["key"]) for row in selected_rows]
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("Nine-cell stratified receipt contains duplicate keys.")
    receipt = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "stratified_hidden_eval_manifest",
        "builder": _bound_record(Path(__file__).resolve()),
        "base_stratified_receipt": {
            **_bound_record(base_receipt_path),
            "schema_version": int(base_receipt["schema_version"]),
            "samples": int(base_receipt["combined_samples"]),
        },
        "supplemental_inputs": {
            "inventory_path": supplemental["inventory_path"],
            "inventory_sha256": supplemental["inventory_sha256"],
            "profile_receipt_path": supplemental["profile_receipt_path"],
            "profile_receipt_sha256": supplemental["profile_receipt_sha256"],
            "manifest_path": supplemental["manifest_path"],
            "manifest_sha256": supplemental["manifest_sha256"],
            "rows": supplemental["rows"],
            "hours": supplemental["hours"],
        },
        "selection": {
            "seed": eval_seed,
            "per_cell": eval_per_cell,
            "candidates_per_part": candidates_per_part,
            "max_rows_per_part": max_eval_scan_rows_per_part,
            "cells": list(ALL_CELLS),
            "algorithm": "source_then_80_frame_bucket_balanced_deterministic_round_robin",
        },
        "source_manifests": {
            **base_receipt["source_manifests"],
            "supplemental": {
                "path": supplemental["manifest_path"],
                "sha256": supplemental["manifest_sha256"],
            },
        },
        "source_parts": dict(sorted(source_parts.items())),
        "cells": cells,
        "selected_rows": selected_rows,
        "selected_keys_sha256": _keys_sha256(selected_keys),
        "selected_rows_sha256": hashlib.sha256(
            (
                "\n".join(
                    f"{row['cell']}\0{row['key']}\0{row['row_sha256']}" for row in selected_rows
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest(),
        "combined_samples": combined_samples,
        "combined_manifest_path": str(combined_manifest_path),
        "combined_manifest_sha256": sha256_file(combined_manifest_path),
    }
    receipt_path = output_dir / "receipt.json"
    _write_immutable(
        receipt_path,
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    return receipt_path, receipt


def _merge_replay_manifests(
    *, base_manifest_path: Path, supplemental_manifest_path: Path, output_path: Path
) -> dict[str, Any]:
    base = _load_json(base_manifest_path, label="Stage211 base replay manifest")
    supplemental = _load_json(
        supplemental_manifest_path,
        label="Stage211 Supplemental replay component manifest",
    )
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for manifest in (base, supplemental):
        for bucket in manifest.get("splits", {}).get("train", {}).get("buckets", []):
            buckets[int(bucket["bucket_id"])].extend(bucket.get("parts", []))
    train_buckets = []
    for bucket_id, parts in sorted(buckets.items()):
        parts.sort(key=lambda item: (str(item.get("source_label") or ""), str(item["path"])))
        train_buckets.append(
            {
                "bucket_id": bucket_id,
                "num_samples": sum(int(part["num_samples"]) for part in parts),
                "parts": parts,
            }
        )
    manifest = {
        "bucket_metric": "audio_frames_plus_text_cost",
        "bucket_width": DEFAULT_BUCKET_WIDTH,
        "entries_per_part": DEFAULT_MAX_ROWS_PER_PART,
        "root": "/",
        "source_length_index_path": "/dev/null",
        "splits": {
            "train": {
                "num_samples": sum(bucket["num_samples"] for bucket in train_buckets),
                "buckets": train_buckets,
            },
            "eval": base["splits"]["eval"],
        },
    }
    _write_immutable(
        output_path,
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def build_supplemental_retention(
    *,
    base_replay_receipt_path: Path,
    base_stratified_receipt_path: Path,
    supplemental_inventory_path: Path,
    supplemental_profile_receipt_path: Path,
    output_root: Path,
    base_cell_targets: Mapping[str, int | None] = BASE_CELL_TARGETS,
    base_fixed_eval_samples: int = DEFAULT_EVAL_PER_CELL,
    replay_per_language: int = DEFAULT_REPLAY_PER_LANGUAGE,
    eval_per_cell: int = DEFAULT_EVAL_PER_CELL,
    replay_seed: int = DEFAULT_REPLAY_SEED,
    eval_seed: int = DEFAULT_EVAL_SEED,
    candidates_per_part: int = DEFAULT_CANDIDATES_PER_PART,
    max_eval_scan_rows_per_part: int = DEFAULT_MAX_EVAL_SCAN_ROWS_PER_PART,
    max_rows_per_part: int = DEFAULT_MAX_ROWS_PER_PART,
    replay_dir_name: str = "retention_replay_v3",
    stratified_dir_name: str = "stratified_hidden_eval_v3",
) -> dict[str, Any]:
    if (
        replay_per_language <= 0
        or eval_per_cell <= 0
        or candidates_per_part <= 0
        or max_eval_scan_rows_per_part <= 0
        or max_rows_per_part <= 0
    ):
        raise ValueError("Supplemental replay and evaluation limits must be positive.")
    for label, name in (
        ("replay", replay_dir_name),
        ("stratified", stratified_dir_name),
    ):
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Supplemental {label} directory name is invalid: {name!r}")
    output_root = output_root.expanduser().resolve()
    replay_dir = output_root / replay_dir_name
    stratified_dir = output_root / stratified_dir_name
    base_replay_receipt_path = base_replay_receipt_path.expanduser().resolve()
    base_stratified_receipt_path = base_stratified_receipt_path.expanduser().resolve()
    base_replay = validate_retention_replay(
        base_replay_receipt_path,
        expected_cell_targets=base_cell_targets,
        expected_fixed_eval_samples=base_fixed_eval_samples,
    )
    base_stratified = _load_json(
        base_stratified_receipt_path,
        label="Stage211 base stratified receipt",
    )
    if (
        base_stratified.get("schema_version") != 1
        or base_stratified.get("artifact") != "stratified_hidden_eval_manifest"
        or set(base_stratified.get("cells") or {}) != set(BASE_CELLS)
    ):
        raise ValueError("Stage211 base stratified receipt is not the seven-cell v1 artifact.")
    supplemental = _validate_supplemental_inputs(
        inventory_path=supplemental_inventory_path,
        profile_receipt_path=supplemental_profile_receipt_path,
    )
    source_parts, source_manifest_record = _load_source_parts(
        difficulty="supplemental",
        manifest_path=Path(supplemental["manifest_path"]),
    )
    base_keys, base_row_records, base_frames = _base_selected_rows(base_replay)
    base_eval_keys = _base_stratified_keys(base_stratified)
    fixed_keys: set[str] = set()
    for exclusion in base_replay["exclusions"]:
        if exclusion["kind"] == "historical_fixed_hidden_eval":
            path = Path(str(exclusion["path"])).resolve()
            with path.open("r", encoding="utf-8") as source:
                fixed_keys = {_row_key(json.loads(line)) for line in source if line.strip()}
    excluded_existing = base_keys | base_eval_keys | fixed_keys
    preflight_path = replay_dir / "capacity_preflight.json"
    preflight_contract = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "supplemental_retention_capacity_preflight",
        "builder": _bound_record(Path(__file__).resolve()),
        "base_replay_receipt": {
            **_bound_record(base_replay_receipt_path),
            "samples": int(base_replay["samples"]),
            "fixed_eval_samples": base_fixed_eval_samples,
            "cell_targets": {
                cell: ("all" if target is None else int(target))
                for cell, target in base_cell_targets.items()
            },
        },
        "base_stratified_receipt": {
            **_bound_record(base_stratified_receipt_path),
            "samples": int(base_stratified["combined_samples"]),
        },
        "supplemental_inputs": {
            key: supplemental[key]
            for key in (
                "inventory_path",
                "inventory_sha256",
                "profile_receipt_path",
                "profile_receipt_sha256",
                "manifest_path",
                "manifest_sha256",
                "rows",
                "hours",
            )
        },
        "source_manifest": source_manifest_record,
        "selection": {
            "bucket_width": DEFAULT_BUCKET_WIDTH,
            "replay_per_language": replay_per_language,
            "eval_per_cell": eval_per_cell,
            "replay_seed": replay_seed,
            "eval_seed": eval_seed,
            "candidates_per_part": candidates_per_part,
            "max_eval_scan_rows_per_part": max_eval_scan_rows_per_part,
            "max_rows_per_part": max_rows_per_part,
        },
        "existing_exclusions": {
            "unique_keys": len(excluded_existing),
            "keys_sha256": _keys_sha256(excluded_existing),
        },
    }
    if preflight_path.is_file():
        preflight = _load_json(preflight_path, label="Supplemental retention preflight")
        if any(preflight.get(key) != value for key, value in preflight_contract.items()):
            raise ValueError("Supplemental retention preflight input binding changed.")
        capacities = Counter(
            {
                (cell, source, int(bucket)): int(count)
                for cell, sources in preflight["capacities"].items()
                for source, buckets in sources.items()
                for bucket, count in buckets.items()
            }
        )
        eval_selected = [
            EvalCandidate(
                language=str(row["language"]),
                cell=str(row["cell"]),
                source_dataset=str(row["source_dataset"]),
                bucket_id=int(row["bucket_id"]),
                key=str(row["key"]),
                num_frames=int(row["num_frames"]),
                score=str(row["score"]),
                row_sha256=str(row["row_sha256"]),
                source_part=Path(str(row["source_part"])).resolve(),
                row=dict(row["row"]),
            )
            for row in preflight["selected_eval_rows"]
        ]
        scan = dict(preflight["scan"])
        print(f"stage211_supplemental_retention reused_preflight={preflight_path}", flush=True)
    else:
        capacities, eval_selected, scan = _scan_capacities_and_eval(
            parts=source_parts,
            excluded_keys=excluded_existing,
            bucket_width=DEFAULT_BUCKET_WIDTH,
            eval_per_cell=eval_per_cell,
            candidates_per_part=candidates_per_part,
            max_eval_scan_rows_per_part=max_eval_scan_rows_per_part,
            eval_seed=eval_seed,
        )
        preflight = {
            **preflight_contract,
            "scan": scan,
            "capacities": _nested_counts(capacities),
            "selected_eval_rows": [
                {
                    "language": candidate.language,
                    "cell": candidate.cell,
                    "source_dataset": candidate.source_dataset,
                    "bucket_id": candidate.bucket_id,
                    "key": candidate.key,
                    "num_frames": candidate.num_frames,
                    "score": candidate.score,
                    "row_sha256": candidate.row_sha256,
                    "source_part": str(candidate.source_part),
                    "row": candidate.row,
                }
                for candidate in eval_selected
            ],
        }
        _write_immutable(
            preflight_path,
            json.dumps(preflight, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        )
    eval_keys = {candidate.key for candidate in eval_selected}
    stratified_receipt_path, stratified_receipt = _write_stratified_v2(
        base_receipt_path=base_stratified_receipt_path,
        base_receipt=base_stratified,
        selected=eval_selected,
        supplemental=supplemental,
        output_dir=stratified_dir,
        eval_seed=eval_seed,
        eval_per_cell=eval_per_cell,
        candidates_per_part=candidates_per_part,
        max_eval_scan_rows_per_part=max_eval_scan_rows_per_part,
    )
    quotas = _build_supplemental_quotas(
        capacities=capacities,
        target_per_language=replay_per_language,
        seed=replay_seed,
    )
    replay_selected = _sample_replay(
        parts=source_parts,
        excluded_keys=excluded_existing | eval_keys,
        quotas=quotas,
        bucket_width=DEFAULT_BUCKET_WIDTH,
        seed=replay_seed,
    )
    supplemental_manifest_path, supplemental_output_parts, _ = _write_outputs(
        selected=replay_selected,
        output_dir=replay_dir / "supplemental_component",
        fixed_eval_part=Path(
            next(
                record["path"]
                for record in base_replay["exclusions"]
                if record["kind"] == "historical_fixed_hidden_eval"
            )
        ).resolve(),
        fixed_eval_samples=int(base_replay["manifest_eval_samples"]),
        bucket_width=DEFAULT_BUCKET_WIDTH,
        max_rows_per_part=max_rows_per_part,
    )
    manifest_path = replay_dir / "manifest.json"
    manifest = _merge_replay_manifests(
        base_manifest_path=Path(str(base_replay["manifest_path"])).resolve(),
        supplemental_manifest_path=supplemental_manifest_path,
        output_path=manifest_path,
    )
    supplemental_keys = {candidate.key for candidate in replay_selected}
    if supplemental_keys & (base_keys | eval_keys | base_eval_keys | fixed_keys):
        raise RuntimeError("Supplemental replay leaked an excluded or base key.")
    all_keys = base_keys | supplemental_keys
    supplemental_row_records = sorted(
        f"{candidate.cell}\0{candidate.key}\0{candidate.row_sha256}"
        for candidate in replay_selected
    )
    all_row_records = sorted([*base_row_records, *supplemental_row_records])
    cell_counts = Counter(candidate.cell for candidate in replay_selected)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts: dict[str, Counter[int]] = defaultdict(Counter)
    frame_ranges: dict[str, list[int]] = {}
    for candidate in replay_selected:
        source_counts[candidate.cell][candidate.source_dataset] += 1
        bucket_counts[candidate.cell][candidate.bucket_id] += 1
        bounds = frame_ranges.setdefault(
            candidate.cell,
            [candidate.num_frames, candidate.num_frames],
        )
        bounds[0] = min(bounds[0], candidate.num_frames)
        bounds[1] = max(bounds[1], candidate.num_frames)
    cells = dict(base_replay["cells"])
    for cell in SUPPLEMENTAL_CELLS:
        cells[cell] = {
            "samples": cell_counts[cell],
            "sources": dict(sorted(source_counts[cell].items())),
            "buckets": {
                str(bucket): count for bucket, count in sorted(bucket_counts[cell].items())
            },
            "min_frames": frame_ranges[cell][0],
            "max_frames": frame_ranges[cell][1],
        }
    supplemental_frames = sum(candidate.num_frames for candidate in replay_selected)
    language_counts = {
        "en": int(base_replay["language_counts"]["en"]) + replay_per_language,
        "zh": int(base_replay["language_counts"]["zh"]) + replay_per_language,
    }
    receipt = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "retention_replay_manifest",
        "builder": _bound_record(Path(__file__).resolve()),
        "contract": {
            "admission": "post_original_and_supplemental_five_segment_coverage_only",
            "architecture": "BiRWKV_TimeMixer_unchanged",
            "learning_rate": 1e-6,
            "trainable_boundary": "mixer_only",
            "early_stopping": False,
        },
        "base_replay_receipt": {
            **_bound_record(base_replay_receipt_path),
            "samples": int(base_replay["samples"]),
            "hours": float(base_replay["total_hours"]),
            "fixed_eval_samples": base_fixed_eval_samples,
            "cell_targets": preflight_contract["base_replay_receipt"]["cell_targets"],
        },
        "stratified_hidden_eval": {
            **_bound_record(stratified_receipt_path),
            "samples": int(stratified_receipt["combined_samples"]),
            "cells": list(ALL_CELLS),
        },
        "supplemental_inputs": preflight_contract["supplemental_inputs"],
        "source_manifest": source_manifest_record,
        "capacity_preflight_path": str(preflight_path),
        "capacity_preflight_sha256": sha256_file(preflight_path),
        "selection": {
            "algorithm": (
                "compose_validated_v1_with_two_pass_equal_source_then_equal_80_frame_"
                "bucket_deterministic_priority_reservoir_without_replacement"
            ),
            "bucket_width": DEFAULT_BUCKET_WIDTH,
            "supplemental_seed": replay_seed,
            "supplemental_cell_targets": {cell: replay_per_language for cell in SUPPLEMENTAL_CELLS},
            "base_cell_targets": preflight_contract["base_replay_receipt"]["cell_targets"],
        },
        "exclusions": {
            "stratified_receipt_path": str(stratified_receipt_path),
            "stratified_receipt_sha256": sha256_file(stratified_receipt_path),
            "stratified_unique_keys": len(base_eval_keys | eval_keys),
            "fixed_unique_keys": len(fixed_keys),
            "base_replay_unique_keys": len(base_keys),
            "all_excluded_keys_sha256": _keys_sha256(
                base_keys | base_eval_keys | eval_keys | fixed_keys
            ),
        },
        "scan": scan,
        "capacities": _nested_counts(capacities),
        "quotas": _nested_counts(quotas),
        "samples": len(all_keys),
        "unique_keys": len(all_keys),
        "total_frames": base_frames + supplemental_frames,
        "total_hours": (base_frames + supplemental_frames) / 100.0 / 3600.0,
        "supplemental_samples": len(replay_selected),
        "supplemental_frames": supplemental_frames,
        "supplemental_hours": supplemental_frames / 100.0 / 3600.0,
        "selected_keys_sha256": _keys_sha256(all_keys),
        "selected_rows_sha256": hashlib.sha256(
            ("\n".join(all_row_records) + "\n").encode("utf-8")
        ).hexdigest(),
        "language_counts": language_counts,
        "cells": cells,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_train_samples": int(manifest["splits"]["train"]["num_samples"]),
        "manifest_eval_samples": int(manifest["splits"]["eval"]["num_samples"]),
        "supplemental_component_manifest_path": str(supplemental_manifest_path),
        "supplemental_component_manifest_sha256": sha256_file(supplemental_manifest_path),
        "supplemental_output_parts": supplemental_output_parts,
    }
    receipt_path = replay_dir / "receipt.json"
    _write_immutable(
        receipt_path,
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description="Build Supplemental-aware Stage211 nine-cell eval and replay v2 artifacts."
    )
    parser.add_argument("--base-replay-receipt", type=Path, default=defaults["base_replay_receipt"])
    parser.add_argument(
        "--base-stratified-receipt",
        type=Path,
        default=defaults["base_stratified_receipt"],
    )
    parser.add_argument(
        "--supplemental-inventory",
        type=Path,
        default=defaults["supplemental_inventory"],
    )
    parser.add_argument(
        "--supplemental-profile-receipt",
        type=Path,
        default=defaults["supplemental_profile_receipt"],
    )
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--replay-per-language", type=int, default=DEFAULT_REPLAY_PER_LANGUAGE)
    parser.add_argument("--eval-per-cell", type=int, default=DEFAULT_EVAL_PER_CELL)
    parser.add_argument("--replay-dir-name", default="retention_replay_v3")
    parser.add_argument("--stratified-dir-name", default="stratified_hidden_eval_v3")
    return parser


def _reuse_validated_supplemental_retention(
    *,
    base_replay_receipt_path: Path,
    base_stratified_receipt_path: Path,
    supplemental_inventory_path: Path,
    supplemental_profile_receipt_path: Path,
    output_root: Path,
    replay_per_language: int,
    eval_per_cell: int,
    replay_dir_name: str = "retention_replay_v3",
    stratified_dir_name: str = "stratified_hidden_eval_v3",
) -> dict[str, Any] | None:
    replay_receipt_path = output_root / replay_dir_name / "receipt.json"
    stratified_receipt_path = output_root / stratified_dir_name / "receipt.json"
    if not replay_receipt_path.is_file() or not stratified_receipt_path.is_file():
        return None
    try:
        try:
            from scripts.validate_stage211_supplemental_retention import (
                validate_stratified_hidden_eval_v2,
                validate_supplemental_retention_replay,
            )
        except ModuleNotFoundError:
            from validate_stage211_supplemental_retention import (  # type: ignore[no-redef]
                validate_stratified_hidden_eval_v2,
                validate_supplemental_retention_replay,
            )

        stratified = validate_stratified_hidden_eval_v2(stratified_receipt_path)
        replay = validate_supplemental_retention_replay(replay_receipt_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(
            "stage211_supplemental_retention_reuse_rejected "
            f"reason={type(error).__name__}",
            flush=True,
        )
        return None

    base_replay_receipt_path = base_replay_receipt_path.expanduser().resolve()
    base_stratified_receipt_path = base_stratified_receipt_path.expanduser().resolve()
    supplemental_inventory_path = supplemental_inventory_path.expanduser().resolve()
    supplemental_profile_receipt_path = supplemental_profile_receipt_path.expanduser().resolve()
    expected_supplemental = {
        "inventory_path": str(supplemental_inventory_path),
        "inventory_sha256": sha256_file(supplemental_inventory_path),
        "profile_receipt_path": str(supplemental_profile_receipt_path),
        "profile_receipt_sha256": sha256_file(supplemental_profile_receipt_path),
    }
    expected_base_replay = {
        "path": str(base_replay_receipt_path),
        "sha256": sha256_file(base_replay_receipt_path),
    }
    expected_base_stratified = {
        "path": str(base_stratified_receipt_path),
        "sha256": sha256_file(base_stratified_receipt_path),
    }

    def bound_matches(record: Any, expected: Mapping[str, str]) -> bool:
        return isinstance(record, Mapping) and all(
            record.get(key) == value for key, value in expected.items()
        )

    replay_selection = replay.get("selection")
    stratified_selection = stratified.get("selection")
    reusable = (
        bound_matches(replay.get("base_replay_receipt"), expected_base_replay)
        and bound_matches(
            stratified.get("base_stratified_receipt"),
            expected_base_stratified,
        )
        and bound_matches(replay.get("supplemental_inputs"), expected_supplemental)
        and bound_matches(stratified.get("supplemental_inputs"), expected_supplemental)
        and bound_matches(
            replay.get("stratified_hidden_eval"),
            {
                "path": str(stratified_receipt_path.resolve()),
                "sha256": sha256_file(stratified_receipt_path),
            },
        )
        and isinstance(replay_selection, Mapping)
        and replay_selection.get("supplemental_cell_targets")
        == {cell: replay_per_language for cell in SUPPLEMENTAL_CELLS}
        and isinstance(stratified_selection, Mapping)
        and int(stratified_selection.get("per_cell", -1)) == eval_per_cell
    )
    if not reusable:
        print(
            "stage211_supplemental_retention_reuse_rejected reason=input_binding",
            flush=True,
        )
        return None
    return replay


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".stage211_supplemental_retention_v2.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt = _reuse_validated_supplemental_retention(
            base_replay_receipt_path=args.base_replay_receipt,
            base_stratified_receipt_path=args.base_stratified_receipt,
            supplemental_inventory_path=args.supplemental_inventory,
            supplemental_profile_receipt_path=args.supplemental_profile_receipt,
            output_root=output_root,
            replay_per_language=args.replay_per_language,
            eval_per_cell=args.eval_per_cell,
            replay_dir_name=args.replay_dir_name,
            stratified_dir_name=args.stratified_dir_name,
        )
        reused = receipt is not None
        if receipt is None:
            receipt = build_supplemental_retention(
                base_replay_receipt_path=args.base_replay_receipt,
                base_stratified_receipt_path=args.base_stratified_receipt,
                supplemental_inventory_path=args.supplemental_inventory,
                supplemental_profile_receipt_path=args.supplemental_profile_receipt,
                output_root=output_root,
                replay_per_language=args.replay_per_language,
                eval_per_cell=args.eval_per_cell,
                replay_dir_name=args.replay_dir_name,
                stratified_dir_name=args.stratified_dir_name,
            )
    print(
        "stage211_supplemental_retention_complete "
        f"reused={str(reused).lower()} "
        f"samples={receipt['samples']} hours={receipt['total_hours']:.6f} "
        f"manifest={receipt['manifest_path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
