from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DIFFICULTIES = ("easy", "medium", "hard", "long")
CELL_LANGUAGES = {
    "easy": ("en", "zh"),
    "medium": ("en", "zh"),
    "hard": ("en", "zh"),
    "long": ("zh",),
}
DEFAULT_PER_CELL = 256
DEFAULT_CANDIDATES_PER_PART = 64
DEFAULT_MAX_ROWS_PER_PART = 8192
DEFAULT_SEED = 211


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


@dataclass(frozen=True, slots=True)
class Candidate:
    difficulty: str
    language: str
    bucket_id: int
    source_dataset: str
    source_manifest: Path
    source_part: Path
    key: str
    score: str
    row_sha256: str
    row: dict[str, Any]


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


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _canonical_language(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith(("zh", "cmn")):
        return "zh"
    return None


def _resolve_part_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _iter_train_parts(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> Iterator[tuple[int, Path]]:
    train = manifest.get("splits", {}).get("train", {})
    for bucket in train.get("buckets", []):
        bucket_id = int(bucket["bucket_id"])
        for part in bucket.get("parts", []):
            yield bucket_id, _resolve_part_path(manifest_path, str(part["path"]))


def _candidate_score(
    *,
    seed: int,
    difficulty: str,
    language: str,
    key: str,
) -> str:
    return hashlib.sha256(
        f"{seed}\0{difficulty}\0{language}\0{key}".encode("utf-8")
    ).hexdigest()


def _candidate_from_row(
    *,
    seed: int,
    difficulty: str,
    bucket_id: int,
    source_manifest: Path,
    source_part: Path,
    row: dict[str, Any],
) -> Candidate | None:
    language = _canonical_language(row.get("language"))
    if language not in CELL_LANGUAGES[difficulty]:
        return None
    key = str(row.get("utt_id") or row.get("key") or row.get("id") or "")
    if not key:
        return None
    num_frames = int(row.get("num_frames") or 0)
    if num_frames <= 0:
        return None
    source_dataset = str(
        row.get("source_dataset")
        or row.get("_stage179_source")
        or row.get("_stage179_input")
        or "unknown"
    )
    return Candidate(
        difficulty=difficulty,
        language=language,
        bucket_id=bucket_id,
        source_dataset=source_dataset,
        source_manifest=source_manifest,
        source_part=source_part,
        key=key,
        score=_candidate_score(
            seed=seed,
            difficulty=difficulty,
            language=language,
            key=key,
        ),
        row_sha256=_sha256_payload(row),
        row=row,
    )


def collect_candidates(
    *,
    difficulty: str,
    manifest_path: Path,
    candidates_per_part: int,
    max_rows_per_part: int,
    seed: int,
) -> dict[str, list[Candidate]]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = {language: [] for language in CELL_LANGUAGES[difficulty]}
    for bucket_id, part_path in _iter_train_parts(manifest_path, manifest):
        if not part_path.is_file():
            raise FileNotFoundError(str(part_path))
        part_candidates = {language: [] for language in CELL_LANGUAGES[difficulty]}
        with part_path.open("r", encoding="utf-8") as source:
            for row_index, line in enumerate(source):
                if row_index >= max_rows_per_part:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                candidate = _candidate_from_row(
                    seed=seed,
                    difficulty=difficulty,
                    bucket_id=bucket_id,
                    source_manifest=manifest_path,
                    source_part=part_path,
                    row=row,
                )
                if candidate is None:
                    continue
                part_candidates[candidate.language].append(candidate)
        for language, candidates in part_candidates.items():
            candidates.sort(key=lambda item: (item.score, item.key))
            output[language].extend(candidates[:candidates_per_part])
    return output


def select_balanced_candidates(
    candidates: list[Candidate],
    *,
    count: int,
    seed: int,
) -> list[Candidate]:
    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        previous = unique.get(candidate.key)
        if previous is None or candidate.score < previous.score:
            unique[candidate.key] = candidate
    strata: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
    for candidate in unique.values():
        strata[(candidate.source_dataset, candidate.bucket_id)].append(candidate)
    for values in strata.values():
        values.sort(key=lambda item: (item.score, item.key))
    stratum_order = sorted(
        strata,
        key=lambda value: hashlib.sha256(
            f"{seed}\0{value[0]}\0{value[1]}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[Candidate] = []
    offsets = {stratum: 0 for stratum in stratum_order}
    while len(selected) < count:
        added = False
        for stratum in stratum_order:
            offset = offsets[stratum]
            values = strata[stratum]
            if offset >= len(values):
                continue
            selected.append(values[offset])
            offsets[stratum] = offset + 1
            added = True
            if len(selected) >= count:
                break
        if not added:
            break
    if len(selected) != count:
        raise ValueError(
            f"Stratified cell has only {len(selected)} usable rows; required {count}."
        )
    return selected


def _write_immutable(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Refusing to replace a different immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _render_manifest(
    *,
    bucket_parts: dict[int, list[tuple[Path, int, str]]],
    total_samples: int,
) -> dict[str, Any]:
    buckets = []
    for bucket_id, parts in sorted(bucket_parts.items()):
        buckets.append(
            {
                "bucket_id": bucket_id,
                "num_samples": sum(count for _, count, _ in parts),
                "parts": [
                    {
                        "path": str(path),
                        "num_samples": count,
                        "source_label": source_label,
                    }
                    for path, count, source_label in parts
                ],
            }
        )
    return {
        "bucket_metric": "audio_frames_plus_text_cost",
        "bucket_width": 80,
        "entries_per_part": total_samples,
        "root": "/",
        "source_length_index_path": "/dev/null",
        "splits": {
            "eval": {
                "num_samples": total_samples,
                "buckets": buckets,
            }
        },
    }


def build_stratified_eval(
    *,
    source_manifests: dict[str, Path],
    output_dir: Path,
    per_cell: int,
    candidates_per_part: int,
    max_rows_per_part: int,
    seed: int,
) -> dict[str, Any]:
    if set(source_manifests) != set(DIFFICULTIES):
        raise ValueError(f"Expected source manifests for {DIFFICULTIES}.")
    if min(per_cell, candidates_per_part, max_rows_per_part) <= 0:
        raise ValueError("Sampling limits must be positive.")
    output_dir = output_dir.expanduser().resolve()
    source_manifests = {
        difficulty: path.expanduser().resolve()
        for difficulty, path in source_manifests.items()
    }
    selected_cells: dict[tuple[str, str], list[Candidate]] = {}
    for difficulty in DIFFICULTIES:
        candidates = collect_candidates(
            difficulty=difficulty,
            manifest_path=source_manifests[difficulty],
            candidates_per_part=candidates_per_part,
            max_rows_per_part=max_rows_per_part,
            seed=seed,
        )
        for language in CELL_LANGUAGES[difficulty]:
            selected_cells[(difficulty, language)] = select_balanced_candidates(
                candidates[language],
                count=per_cell,
                seed=seed,
            )

    combined_parts: dict[int, list[tuple[Path, int, str]]] = defaultdict(list)
    cell_manifests: dict[str, dict[str, Any]] = {}
    selected_source_parts: set[Path] = set()
    selected_rows: list[dict[str, Any]] = []
    for (difficulty, language), selected in selected_cells.items():
        cell_name = f"{difficulty}_{language}"
        by_bucket: dict[int, list[Candidate]] = defaultdict(list)
        for candidate in selected:
            by_bucket[candidate.bucket_id].append(candidate)
            selected_source_parts.add(candidate.source_part)
            selected_rows.append(
                {
                    "cell": cell_name,
                    "key": candidate.key,
                    "row_sha256": candidate.row_sha256,
                    "source_part": str(candidate.source_part),
                }
            )
        cell_parts: dict[int, list[tuple[Path, int, str]]] = defaultdict(list)
        for bucket_id, bucket_rows in sorted(by_bucket.items()):
            bucket_rows.sort(key=lambda item: (item.score, item.key))
            part_path = (
                output_dir
                / "parts"
                / difficulty
                / language
                / f"bucket_{bucket_id:04d}.jsonl"
            )
            rendered_rows = []
            for candidate in bucket_rows:
                row = dict(candidate.row)
                row["_stage211_sidecar_cell"] = cell_name
                row["_stage211_sidecar_row_sha256"] = candidate.row_sha256
                rendered_rows.append(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            _write_immutable(part_path, "".join(rendered_rows))
            part_record = (part_path, len(bucket_rows), cell_name)
            cell_parts[bucket_id].append(part_record)
            combined_parts[bucket_id].append(part_record)
        manifest = _render_manifest(
            bucket_parts=cell_parts,
            total_samples=per_cell,
        )
        manifest_path = output_dir / f"manifest_{cell_name}.json"
        _write_immutable(
            manifest_path,
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        )
        cell_manifests[cell_name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "samples": per_cell,
            "source_datasets": sorted({item.source_dataset for item in selected}),
            "bucket_ids": sorted({item.bucket_id for item in selected}),
            "min_frames": min(int(item.row["num_frames"]) for item in selected),
            "max_frames": max(int(item.row["num_frames"]) for item in selected),
        }

    combined_count = sum(len(values) for values in selected_cells.values())
    combined_manifest = _render_manifest(
        bucket_parts=combined_parts,
        total_samples=combined_count,
    )
    combined_manifest_path = output_dir / "manifest_all.json"
    _write_immutable(
        combined_manifest_path,
        json.dumps(combined_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )

    source_part_hashes = {
        str(path): sha256_file(path)
        for path in sorted(selected_source_parts)
    }
    for row in selected_rows:
        row["source_part_sha256"] = source_part_hashes[row["source_part"]]
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stratified_hidden_eval_manifest",
        "selection": {
            "seed": seed,
            "per_cell": per_cell,
            "candidates_per_part": candidates_per_part,
            "max_rows_per_part": max_rows_per_part,
            "cells": [
                f"{difficulty}_{language}"
                for difficulty in DIFFICULTIES
                for language in CELL_LANGUAGES[difficulty]
            ],
        },
        "source_manifests": {
            difficulty: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for difficulty, path in source_manifests.items()
        },
        "source_parts": source_part_hashes,
        "cells": cell_manifests,
        "combined_manifest_path": str(combined_manifest_path),
        "combined_manifest_sha256": sha256_file(combined_manifest_path),
        "combined_samples": combined_count,
        "selected_rows": sorted(
            selected_rows,
            key=lambda row: (row["cell"], row["key"]),
        ),
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
        description=(
            "Build immutable Stage211 hidden-eval cells across difficulty and language."
        )
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
        default=(
            Path.home()
            / "rwkvasr_data"
            / "stage211_full_curriculum"
            / "stratified_hidden_eval_v1"
        ),
    )
    parser.add_argument("--per-cell", type=int, default=DEFAULT_PER_CELL)
    parser.add_argument(
        "--candidates-per-part",
        type=int,
        default=DEFAULT_CANDIDATES_PER_PART,
    )
    parser.add_argument(
        "--max-rows-per-part",
        type=int,
        default=DEFAULT_MAX_ROWS_PER_PART,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_manifests = _default_source_manifests()
    for raw_value in args.source_manifest:
        difficulty, path = _parse_source_manifest(raw_value)
        source_manifests[difficulty] = path
    receipt = build_stratified_eval(
        source_manifests=source_manifests,
        output_dir=args.output_dir,
        per_cell=int(args.per_cell),
        candidates_per_part=int(args.candidates_per_part),
        max_rows_per_part=int(args.max_rows_per_part),
        seed=int(args.seed),
    )
    print(
        "stage211_stratified_hidden_eval "
        f"cells={len(receipt['cells'])} samples={receipt['combined_samples']} "
        f"manifest={receipt['combined_manifest_path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
