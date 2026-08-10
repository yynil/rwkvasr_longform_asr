from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DIFFICULTIES = ("easy", "medium", "hard", "long")
CELL_LANGUAGES = {
    "easy": ("en", "zh"),
    "medium": ("en", "zh"),
    "hard": ("en", "zh"),
    "long": ("zh",),
}
CELLS = tuple(
    f"{difficulty}_{language}"
    for difficulty in DIFFICULTIES
    for language in CELL_LANGUAGES[difficulty]
)
DEFAULT_CELL_TARGETS: dict[str, int | None] = {
    "easy_en": 200_000,
    "easy_zh": 200_000,
    "medium_en": 200_000,
    "medium_zh": 200_000,
    "hard_en": 100_000,
    "hard_zh": 100_000,
    "long_zh": None,
}
DEFAULT_SEED = 2111
DEFAULT_BUCKET_WIDTH = 80
DEFAULT_MAX_ROWS_PER_PART = 100_000


def _default_source_manifests() -> dict[str, Path]:
    root = Path.home() / "rwkvasr_data" / "stage211_full_curriculum"
    return {
        "easy": (
            Path.home()
            / "rwkvasr_data"
            / "stage211_easy_source_grouped_buckets"
            / "manifest_stage211_fixed_eval.json"
        ),
        "medium": (
            root
            / "stage179b_medium_dedup_audio_only_online_ctc"
            / "webdataset_buckets_audio_text"
            / "manifest_stage211_fixed_eval.json"
        ),
        "hard": (
            root
            / "stage179c_hard_dedup_audio_only_online_ctc"
            / "webdataset_buckets_audio_text"
            / "manifest_stage211_fixed_eval.json"
        ),
        "long": (
            root
            / "stage179d_long_dedup_audio_only_online_ctc"
            / "webdataset_buckets_audio_text"
            / "manifest_stage211_fixed_eval.json"
        ),
    }


def _default_stratified_receipt() -> Path:
    return (
        Path.home()
        / "rwkvasr_data"
        / "stage211_full_curriculum"
        / "stratified_hidden_eval_v1"
        / "receipt.json"
    )


def _default_fixed_eval_part() -> Path:
    return (
        Path.home()
        / "rwkvasr_data"
        / "stage211_full_curriculum"
        / "fixed_hidden_eval"
        / "part_000000.jsonl"
    )


@dataclass(frozen=True, slots=True)
class SourcePart:
    difficulty: str
    input_bucket_id: int
    path: Path
    expected_samples: int
    source_label: str | None


@dataclass(frozen=True, slots=True)
class ReplayCandidate:
    cell: str
    language: str
    source_dataset: str
    bucket_id: int
    key: str
    num_frames: int
    score_hex: str
    score_int: int
    row_sha256: str
    rendered_row: str


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


def _keys_sha256(keys: Sequence[str] | set[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_language(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith(("zh", "cmn")):
        return "zh"
    return None


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("key") or row.get("id") or "")


def _source_dataset(row: Mapping[str, Any], fallback: str | None) -> str:
    return str(
        row.get("source_dataset")
        or row.get("_stage179_source")
        or row.get("_stage179_input")
        or fallback
        or "unknown"
    )


def _resolve_part_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _load_source_parts(
    *, difficulty: str, manifest_path: Path
) -> tuple[list[SourcePart], dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, dict):
        raise ValueError(f"Source manifest has no train split: {manifest_path}")
    parts: list[SourcePart] = []
    inventory: list[dict[str, Any]] = []
    for bucket in train.get("buckets", []):
        input_bucket_id = int(bucket["bucket_id"])
        for raw_part in bucket.get("parts", []):
            part = SourcePart(
                difficulty=difficulty,
                input_bucket_id=input_bucket_id,
                path=_resolve_part_path(manifest_path, str(raw_part["path"])),
                expected_samples=int(raw_part["num_samples"]),
                source_label=(
                    str(raw_part["source_label"])
                    if raw_part.get("source_label") is not None
                    else None
                ),
            )
            if not part.path.is_file():
                raise FileNotFoundError(str(part.path))
            parts.append(part)
            inventory.append(
                {
                    "input_bucket_id": part.input_bucket_id,
                    "num_samples": part.expected_samples,
                    "path": str(part.path),
                    "source_label": part.source_label,
                }
            )
    declared_samples = int(train.get("num_samples", 0))
    part_samples = sum(part.expected_samples for part in parts)
    if declared_samples != part_samples:
        raise ValueError(
            f"Manifest train count mismatch for {manifest_path}: "
            f"declared={declared_samples} parts={part_samples}"
        )
    return parts, {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "declared_train_samples": declared_samples,
        "part_count": len(parts),
        "part_inventory_sha256": _payload_sha256(inventory),
    }


def _iter_part_rows(part: SourcePart) -> Iterator[dict[str, Any]]:
    seen = 0
    with part.path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {part.path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {part.path}:{line_number}")
            yield row
    if seen != part.expected_samples:
        raise ValueError(
            f"Part row count mismatch for {part.path}: "
            f"expected={part.expected_samples} actual={seen}"
        )


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


def _load_exclusions(
    *, stratified_receipt: Path, fixed_eval_part: Path
) -> tuple[set[str], list[dict[str, Any]], int]:
    stratified_receipt = stratified_receipt.expanduser().resolve()
    fixed_eval_part = fixed_eval_part.expanduser().resolve()
    receipt = json.loads(stratified_receipt.read_text(encoding="utf-8"))
    if receipt.get("artifact") != "stratified_hidden_eval_manifest":
        raise ValueError(
            f"Expected a stratified_hidden_eval_manifest receipt at {stratified_receipt}"
        )
    sidecar_keys = {str(row.get("key") or "") for row in receipt.get("selected_rows", [])}
    sidecar_keys.discard("")
    expected_sidecar = int(receipt.get("combined_samples", len(sidecar_keys)))
    if len(sidecar_keys) != expected_sidecar:
        raise ValueError(
            "Stratified receipt selected key count mismatch: "
            f"expected={expected_sidecar} unique={len(sidecar_keys)}"
        )
    fixed_keys = _read_jsonl_keys(fixed_eval_part)
    exclusions = sidecar_keys | fixed_keys
    records = [
        {
            "kind": "stratified_hidden_eval",
            "path": str(stratified_receipt),
            "sha256": sha256_file(stratified_receipt),
            "unique_keys": len(sidecar_keys),
            "keys_sha256": _keys_sha256(sidecar_keys),
        },
        {
            "kind": "historical_fixed_hidden_eval",
            "path": str(fixed_eval_part),
            "sha256": sha256_file(fixed_eval_part),
            "unique_keys": len(fixed_keys),
            "keys_sha256": _keys_sha256(fixed_keys),
        },
    ]
    return exclusions, records, len(fixed_keys)


def _classify_row(
    *,
    part: SourcePart,
    row: Mapping[str, Any],
    bucket_width: int,
) -> tuple[str, str, str, int, str, int] | None:
    language = _canonical_language(row.get("language"))
    if language not in CELL_LANGUAGES[part.difficulty]:
        return None
    key = _row_key(row)
    try:
        num_frames = int(row.get("num_frames") or 0)
    except (TypeError, ValueError):
        return None
    if not key or num_frames <= 0:
        return None
    cell = f"{part.difficulty}_{language}"
    source_dataset = _source_dataset(row, part.source_label)
    bucket_id = num_frames // bucket_width
    return cell, language, source_dataset, bucket_id, key, num_frames


def _allocation_order(*, labels: Sequence[Any], seed: int, scope: str) -> list[Any]:
    return sorted(
        labels,
        key=lambda label: hashlib.sha256(f"{seed}\0{scope}\0{label}".encode("utf-8")).hexdigest(),
    )


def _allocate_even_with_capacity(
    capacities: Mapping[Any, int], *, target: int, seed: int, scope: str
) -> dict[Any, int]:
    if target < 0:
        raise ValueError("Replay target cannot be negative.")
    normalized = {label: int(capacity) for label, capacity in capacities.items()}
    if any(capacity < 0 for capacity in normalized.values()):
        raise ValueError("Replay capacity cannot be negative.")
    total_capacity = sum(normalized.values())
    if target > total_capacity:
        raise ValueError(
            f"Insufficient replay capacity for {scope}: target={target} eligible={total_capacity}"
        )
    quotas = {label: 0 for label in normalized}
    remaining = target
    while remaining:
        active = [label for label, capacity in normalized.items() if quotas[label] < capacity]
        if not active:
            raise RuntimeError(f"Replay allocation stalled for {scope}.")
        active = _allocation_order(labels=active, seed=seed, scope=scope)
        base, extra = divmod(remaining, len(active))
        allocated = 0
        for index, label in enumerate(active):
            requested = base + (1 if index < extra else 0)
            if requested <= 0:
                continue
            grant = min(requested, normalized[label] - quotas[label])
            quotas[label] += grant
            allocated += grant
        if allocated <= 0:
            raise RuntimeError(f"Replay allocation made no progress for {scope}.")
        remaining -= allocated
    return {label: quota for label, quota in quotas.items() if quota}


def _build_quotas(
    *,
    capacities: Counter[tuple[str, str, int]],
    cell_targets: Mapping[str, int | None],
    seed: int,
) -> tuple[dict[tuple[str, str, int], int], dict[str, int]]:
    by_cell_source_bucket: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (cell, source_dataset, bucket_id), capacity in capacities.items():
        by_cell_source_bucket[cell][source_dataset][bucket_id] = capacity

    quotas: dict[tuple[str, str, int], int] = {}
    realized_targets: dict[str, int] = {}
    for cell in CELLS:
        source_buckets = by_cell_source_bucket.get(cell, {})
        source_capacities = {
            source: sum(buckets.values()) for source, buckets in source_buckets.items()
        }
        requested_target = cell_targets[cell]
        target = (
            sum(source_capacities.values()) if requested_target is None else int(requested_target)
        )
        source_quotas = _allocate_even_with_capacity(
            source_capacities,
            target=target,
            seed=seed,
            scope=f"{cell}:sources",
        )
        for source_dataset, source_quota in source_quotas.items():
            bucket_quotas = _allocate_even_with_capacity(
                source_buckets[source_dataset],
                target=source_quota,
                seed=seed,
                scope=f"{cell}:{source_dataset}:buckets",
            )
            for bucket_id, bucket_quota in bucket_quotas.items():
                quotas[(cell, source_dataset, bucket_id)] = bucket_quota
        realized_targets[cell] = target
    return quotas, realized_targets


def _score_candidate(
    *, seed: int, cell: str, source_dataset: str, bucket_id: int, key: str
) -> tuple[str, int]:
    score_hex = hashlib.sha256(
        f"{seed}\0{cell}\0{source_dataset}\0{bucket_id}\0{key}".encode("utf-8")
    ).hexdigest()
    return score_hex, int(score_hex, 16)


def _make_candidate(
    *,
    row: dict[str, Any],
    part: SourcePart,
    cell: str,
    language: str,
    source_dataset: str,
    bucket_id: int,
    key: str,
    num_frames: int,
    seed: int,
) -> ReplayCandidate:
    row_sha256 = _payload_sha256(row)
    score_hex, score_int = _score_candidate(
        seed=seed,
        cell=cell,
        source_dataset=source_dataset,
        bucket_id=bucket_id,
        key=key,
    )
    output_row = dict(row)
    output_row["_stage211_replay_cell"] = cell
    output_row["_stage211_replay_source"] = source_dataset
    output_row["_stage211_replay_bucket_id"] = bucket_id
    output_row["_stage211_replay_row_sha256"] = row_sha256
    output_row["_stage211_replay_score"] = score_hex
    output_row["_stage211_replay_seed"] = seed
    output_row["_stage211_replay_source_part"] = str(part.path)
    return ReplayCandidate(
        cell=cell,
        language=language,
        source_dataset=source_dataset,
        bucket_id=bucket_id,
        key=key,
        num_frames=num_frames,
        score_hex=score_hex,
        score_int=score_int,
        row_sha256=row_sha256,
        rendered_row=json.dumps(output_row, ensure_ascii=True, sort_keys=True) + "\n",
    )


def _scan_capacities(
    *,
    source_parts: Mapping[str, Sequence[SourcePart]],
    exclusions: set[str],
    bucket_width: int,
) -> tuple[Counter[tuple[str, str, int]], dict[str, Any]]:
    capacities: Counter[tuple[str, str, int]] = Counter()
    per_cell: dict[str, Counter[str]] = defaultdict(Counter)
    per_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    total_rows = 0
    for difficulty in DIFFICULTIES:
        for part in source_parts[difficulty]:
            for row in _iter_part_rows(part):
                total_rows += 1
                per_difficulty[difficulty]["rows"] += 1
                classified = _classify_row(
                    part=part,
                    row=row,
                    bucket_width=bucket_width,
                )
                if classified is None:
                    per_difficulty[difficulty]["ignored_invalid_or_language"] += 1
                    continue
                cell, _, source_dataset, bucket_id, key, _ = classified
                if key in exclusions:
                    per_cell[cell]["excluded_eval"] += 1
                    continue
                capacities[(cell, source_dataset, bucket_id)] += 1
                per_cell[cell]["eligible"] += 1
                if total_rows % 1_000_000 == 0:
                    print(
                        f"stage211_retention_replay scan=capacity rows={total_rows}",
                        flush=True,
                    )
    return capacities, {
        "total_source_rows": total_rows,
        "by_difficulty": {
            difficulty: dict(sorted(per_difficulty[difficulty].items()))
            for difficulty in DIFFICULTIES
        },
        "by_cell": {cell: dict(sorted(per_cell[cell].items())) for cell in CELLS},
    }


def _sample_candidates(
    *,
    source_parts: Mapping[str, Sequence[SourcePart]],
    exclusions: set[str],
    quotas: Mapping[tuple[str, str, int], int],
    bucket_width: int,
    seed: int,
) -> list[ReplayCandidate]:
    heaps: dict[tuple[str, str, int], list[tuple[int, int, ReplayCandidate]]] = defaultdict(list)
    heap_keys: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    serial = 0
    total_rows = 0
    for difficulty in DIFFICULTIES:
        for part in source_parts[difficulty]:
            for row in _iter_part_rows(part):
                total_rows += 1
                classified = _classify_row(
                    part=part,
                    row=row,
                    bucket_width=bucket_width,
                )
                if classified is None:
                    continue
                cell, language, source_dataset, bucket_id, key, num_frames = classified
                if key in exclusions:
                    continue
                stratum = (cell, source_dataset, bucket_id)
                quota = quotas.get(stratum, 0)
                if quota <= 0 or key in heap_keys[stratum]:
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
                )
                heap = heaps[stratum]
                serial += 1
                entry = (-candidate.score_int, serial, candidate)
                if len(heap) < quota:
                    heapq.heappush(heap, entry)
                    heap_keys[stratum].add(key)
                elif candidate.score_int < -heap[0][0]:
                    removed = heapq.heapreplace(heap, entry)[2]
                    heap_keys[stratum].remove(removed.key)
                    heap_keys[stratum].add(key)
                if total_rows % 1_000_000 == 0:
                    print(
                        f"stage211_retention_replay scan=reservoir rows={total_rows}",
                        flush=True,
                    )

    selected: list[ReplayCandidate] = []
    for stratum, quota in quotas.items():
        actual = len(heaps[stratum])
        if actual != quota:
            raise ValueError(
                f"Replay reservoir underfilled for {stratum}: required={quota} actual={actual}"
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
    selected_keys = [candidate.key for candidate in selected]
    if len(set(selected_keys)) != len(selected_keys):
        duplicates = [key for key, count in Counter(selected_keys).items() if count > 1]
        raise ValueError(
            f"Replay inputs are not globally deduplicated; duplicate keys include {duplicates[:20]}"
        )
    if set(selected_keys) & exclusions:
        raise RuntimeError("Evaluation exclusions leaked into replay selection.")
    return selected


def _write_immutable(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Refusing to replace a different immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _source_slug(source_dataset: str) -> str:
    return hashlib.sha256(source_dataset.encode("utf-8")).hexdigest()[:16]


def _write_outputs(
    *,
    selected: Sequence[ReplayCandidate],
    output_dir: Path,
    fixed_eval_part: Path,
    fixed_eval_samples: int,
    bucket_width: int,
    max_rows_per_part: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[ReplayCandidate]] = defaultdict(list)
    for candidate in selected:
        grouped[(candidate.bucket_id, candidate.cell, candidate.source_dataset)].append(candidate)

    manifest_buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    output_parts: list[dict[str, Any]] = []
    for (bucket_id, cell, source_dataset), candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: (item.score_hex, item.key))
        source_label = f"{cell}:{source_dataset}"
        for part_index, offset in enumerate(range(0, len(candidates), max_rows_per_part)):
            rows = candidates[offset : offset + max_rows_per_part]
            part_path = (
                output_dir
                / "train"
                / f"bucket_{bucket_id:04d}"
                / cell
                / f"source_{_source_slug(source_dataset)}"
                / f"part_{part_index:06d}.jsonl"
            )
            _write_immutable(part_path, "".join(row.rendered_row for row in rows))
            part_record = {
                "path": str(part_path),
                "num_samples": len(rows),
                "source_label": source_label,
            }
            manifest_buckets[bucket_id].append(part_record)
            output_parts.append(
                {
                    **part_record,
                    "bucket_id": bucket_id,
                    "cell": cell,
                    "source_dataset": source_dataset,
                    "sha256": sha256_file(part_path),
                }
            )

    train_buckets = []
    for bucket_id, parts in sorted(manifest_buckets.items()):
        parts.sort(key=lambda item: (str(item["source_label"]), str(item["path"])))
        train_buckets.append(
            {
                "bucket_id": bucket_id,
                "num_samples": sum(int(part["num_samples"]) for part in parts),
                "parts": parts,
            }
        )
    manifest = {
        "bucket_metric": "audio_frames_plus_text_cost",
        "bucket_width": bucket_width,
        "entries_per_part": max_rows_per_part,
        "root": "/",
        "source_length_index_path": "/dev/null",
        "splits": {
            "train": {
                "num_samples": len(selected),
                "buckets": train_buckets,
            },
            "eval": {
                "num_samples": fixed_eval_samples,
                "buckets": [
                    {
                        "bucket_id": 0,
                        "num_samples": fixed_eval_samples,
                        "parts": [
                            {
                                "path": str(fixed_eval_part),
                                "num_samples": fixed_eval_samples,
                            }
                        ],
                    }
                ],
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_immutable(
        manifest_path,
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    return manifest_path, output_parts, manifest


def _nested_counts(
    values: Mapping[tuple[str, str, int], int],
) -> dict[str, dict[str, dict[str, int]]]:
    nested: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (cell, source_dataset, bucket_id), count in sorted(values.items()):
        nested[cell][source_dataset][str(bucket_id)] = int(count)
    return {
        cell: {source: dict(buckets) for source, buckets in sorted(sources.items())}
        for cell, sources in sorted(nested.items())
    }


def build_retention_replay(
    *,
    source_manifests: Mapping[str, Path],
    output_dir: Path,
    stratified_receipt: Path,
    fixed_eval_part: Path,
    cell_targets: Mapping[str, int | None] | None = None,
    seed: int = DEFAULT_SEED,
    bucket_width: int = DEFAULT_BUCKET_WIDTH,
    max_rows_per_part: int = DEFAULT_MAX_ROWS_PER_PART,
) -> dict[str, Any]:
    if set(source_manifests) != set(DIFFICULTIES):
        raise ValueError(f"Expected source manifests for {DIFFICULTIES}.")
    targets = dict(DEFAULT_CELL_TARGETS if cell_targets is None else cell_targets)
    if set(targets) != set(CELLS):
        raise ValueError(f"Expected replay targets for {CELLS}.")
    if any(target is not None and int(target) <= 0 for target in targets.values()):
        raise ValueError("Finite replay cell targets must be positive.")
    if bucket_width <= 0 or max_rows_per_part <= 0:
        raise ValueError("Bucket width and max rows per part must be positive.")

    output_dir = output_dir.expanduser().resolve()
    fixed_eval_part = fixed_eval_part.expanduser().resolve()
    exclusions, exclusion_records, fixed_eval_samples = _load_exclusions(
        stratified_receipt=stratified_receipt,
        fixed_eval_part=fixed_eval_part,
    )
    parts_by_difficulty: dict[str, list[SourcePart]] = {}
    source_records: dict[str, dict[str, Any]] = {}
    for difficulty in DIFFICULTIES:
        parts, record = _load_source_parts(
            difficulty=difficulty,
            manifest_path=source_manifests[difficulty],
        )
        parts_by_difficulty[difficulty] = parts
        source_records[difficulty] = record

    capacities, scan = _scan_capacities(
        source_parts=parts_by_difficulty,
        exclusions=exclusions,
        bucket_width=bucket_width,
    )
    quotas, realized_targets = _build_quotas(
        capacities=capacities,
        cell_targets=targets,
        seed=seed,
    )
    selected = _sample_candidates(
        source_parts=parts_by_difficulty,
        exclusions=exclusions,
        quotas=quotas,
        bucket_width=bucket_width,
        seed=seed,
    )
    if len(selected) != sum(realized_targets.values()):
        raise RuntimeError("Replay selection total does not match realized targets.")

    manifest_path, output_parts, manifest = _write_outputs(
        selected=selected,
        output_dir=output_dir,
        fixed_eval_part=fixed_eval_part,
        fixed_eval_samples=fixed_eval_samples,
        bucket_width=bucket_width,
        max_rows_per_part=max_rows_per_part,
    )
    cell_counts = Counter(candidate.cell for candidate in selected)
    language_counts = Counter(candidate.language for candidate in selected)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts: dict[str, Counter[int]] = defaultdict(Counter)
    frame_ranges: dict[str, list[int]] = {}
    for candidate in selected:
        source_counts[candidate.cell][candidate.source_dataset] += 1
        bucket_counts[candidate.cell][candidate.bucket_id] += 1
        bounds = frame_ranges.setdefault(
            candidate.cell, [candidate.num_frames, candidate.num_frames]
        )
        bounds[0] = min(bounds[0], candidate.num_frames)
        bounds[1] = max(bounds[1], candidate.num_frames)

    selected_key_records = [
        f"{candidate.cell}\0{candidate.key}\0{candidate.row_sha256}" for candidate in selected
    ]
    selected_key_records.sort()
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "retention_replay_manifest",
        "contract": {
            "admission": "post_original_easy_medium_hard_long_coverage_only",
            "architecture": "BiRWKV_TimeMixer_unchanged",
            "phase": "mixer",
            "learning_rate": 1e-6,
            "trainable_boundary": "mixer_only",
            "early_stopping": False,
        },
        "selection": {
            "seed": seed,
            "bucket_width": bucket_width,
            "max_rows_per_part": max_rows_per_part,
            "requested_cell_targets": {
                cell: ("all" if target is None else int(target)) for cell, target in targets.items()
            },
            "realized_cell_targets": realized_targets,
            "algorithm": (
                "two_pass_equal_source_then_equal_80_frame_bucket_"
                "deterministic_priority_reservoir_without_replacement"
            ),
        },
        "source_manifests": source_records,
        "exclusions": exclusion_records,
        "exclusion_union": {
            "unique_keys": len(exclusions),
            "keys_sha256": _keys_sha256(exclusions),
        },
        "scan": scan,
        "capacities": _nested_counts(capacities),
        "quotas": _nested_counts(quotas),
        "samples": len(selected),
        "unique_keys": len(selected),
        "selected_keys_sha256": _keys_sha256([candidate.key for candidate in selected]),
        "selected_rows_sha256": hashlib.sha256(
            "\n".join(selected_key_records).encode("utf-8") + b"\n"
        ).hexdigest(),
        "language_counts": dict(sorted(language_counts.items())),
        "cells": {
            cell: {
                "samples": cell_counts[cell],
                "sources": dict(sorted(source_counts[cell].items())),
                "buckets": {
                    str(bucket_id): count
                    for bucket_id, count in sorted(bucket_counts[cell].items())
                },
                "min_frames": frame_ranges[cell][0],
                "max_frames": frame_ranges[cell][1],
            }
            for cell in CELLS
        },
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_train_samples": int(manifest["splits"]["train"]["num_samples"]),
        "manifest_eval_samples": int(manifest["splits"]["eval"]["num_samples"]),
        "output_parts": output_parts,
    }
    receipt_path = output_dir / "receipt.json"
    _write_immutable(
        receipt_path,
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def _parse_source_manifest(value: str) -> tuple[str, Path]:
    difficulty, separator, raw_path = value.partition("=")
    if separator != "=" or difficulty not in DIFFICULTIES or not raw_path:
        raise argparse.ArgumentTypeError(
            "--source-manifest must use difficulty=/absolute/or/relative/path"
        )
    return difficulty, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build the immutable post-coverage Stage211A retention replay manifest.")
    )
    parser.add_argument(
        "--source-manifest",
        action="append",
        default=[],
        metavar="DIFFICULTY=PATH",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(Path.home() / "rwkvasr_data" / "stage211_full_curriculum" / "retention_replay_v1"),
    )
    parser.add_argument(
        "--stratified-receipt",
        type=Path,
        default=_default_stratified_receipt(),
    )
    parser.add_argument(
        "--fixed-eval-part",
        type=Path,
        default=_default_fixed_eval_part(),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bucket-width", type=int, default=DEFAULT_BUCKET_WIDTH)
    parser.add_argument(
        "--max-rows-per-part",
        type=int,
        default=DEFAULT_MAX_ROWS_PER_PART,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_manifests = _default_source_manifests()
    for raw_value in args.source_manifest:
        difficulty, path = _parse_source_manifest(raw_value)
        source_manifests[difficulty] = path
    receipt = build_retention_replay(
        source_manifests=source_manifests,
        output_dir=args.output_dir,
        stratified_receipt=args.stratified_receipt,
        fixed_eval_part=args.fixed_eval_part,
        seed=int(args.seed),
        bucket_width=int(args.bucket_width),
        max_rows_per_part=int(args.max_rows_per_part),
    )
    print(
        "stage211_retention_replay "
        f"samples={receipt['samples']} manifest={receipt['manifest_path']} ",
        f"receipt={args.output_dir.expanduser().resolve() / 'receipt.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
