#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from rwkvasr.data import (
    estimate_bucket_manifest_steps,
    estimate_bucket_manifest_tail_padding_samples,
    load_webdataset_bucket_manifest,
)
from rwkvasr.eval.stage211_gate import sha256_file

try:
    from scripts.build_stage211_retention_replay import _allocate_even_with_capacity
    from scripts.create_stage211_labeled_profile_receipt import (
        validate_receipt as validate_full_labeled_profile_receipt,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from build_stage211_retention_replay import _allocate_even_with_capacity
    from create_stage211_labeled_profile_receipt import (
        validate_receipt as validate_full_labeled_profile_receipt,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_ROOT = Path.home() / "rwkvasr_data" / "stage211_sft_full_labeled_v2"
DEFAULT_FULL_PROFILE = DEFAULT_FULL_ROOT / "stage211_labeled_profile_receipt.json"
DEFAULT_OUTPUT_ROOT = (
    Path.home() / "rwkvasr_data" / "stage211_sft_source_balanced_correction_v1"
)
DEFAULT_SEED = 2114
DEFAULT_BUCKET_WIDTH = 80
DEFAULT_ENTRIES_PER_PART = 100_000
TRAIN_BATCH_SIZE = 12
TRAIN_WORLD_SIZE = 4
TRAIN_FRAME_BUDGET = 8_000
CORRECTION_FIELD = "stage211_sft_correction_lane"
CORRECTION_BUCKET_FIELD = "stage211_sft_correction_acoustic_bucket_id"
SOURCE_LANGUAGES = {
    "aishell3": "zh",
    "commonvoice_cn": "zh",
    "commonvoice_en": "en",
    "librispeech": "en",
}
ENGLISH_SOURCES = ("commonvoice_en", "librispeech")
CHINESE_SOURCES = ("aishell3", "commonvoice_cn")


@dataclass(frozen=True, slots=True)
class CorrectionCandidate:
    key: str
    language: str
    source_dataset: str
    acoustic_bucket_id: int
    num_frames: int
    ctc_num_tokens: int
    score_hex: str
    score_int: int
    source_row_sha256: str
    rendered_row: str


@dataclass(frozen=True, slots=True)
class CorrectionSelection:
    candidates: tuple[CorrectionCandidate, ...]
    scan: dict[str, Any]
    source_quotas: dict[str, int]
    bucket_quotas: dict[tuple[str, int], int]


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


def _records_sha256(records: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records):
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_immutable(path: Path, content: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace a different immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _iter_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid correction-source JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Correction-source row is not an object at {path}:{line_number}")
            yield line_number, row


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("key") or row.get("id") or "").strip()


def _validate_source_row(row: Mapping[str, Any], *, line_number: int) -> tuple[str, str, str]:
    split = str(row.get("split") or "").strip()
    if split not in {"train", "eval"}:
        raise ValueError(f"Correction-source row {line_number} has unsupported split {split!r}.")
    key = _row_key(row)
    if not key:
        raise ValueError(f"Correction-source row {line_number} has no utterance id.")
    source_dataset = str(row.get("source_dataset") or "").strip()
    expected_language = SOURCE_LANGUAGES.get(source_dataset)
    language = str(row.get("language") or "").strip().lower().replace("_", "-")
    if language.startswith("en"):
        language = "en"
    elif language.startswith(("zh", "cmn")):
        language = "zh"
    if expected_language is None or language != expected_language:
        raise ValueError(
            f"Correction-source row {line_number} has invalid source/language "
            f"{source_dataset!r}/{language!r}."
        )
    required_strings = ("shard_name", "audio_member", "json_member")
    if any(not str(row.get(field) or "").strip() for field in required_strings):
        raise ValueError(f"Correction-source row {line_number} lacks an audio/JSON binding.")
    try:
        normalized_chars = int(row["normalized_text_chars"])
        ctc_num_tokens = int(row["ctc_num_tokens"])
        ctc_unk_tokens = int(row["ctc_unk_tokens"])
        ctc_forbidden_tokens = int(row["ctc_forbidden_tokens"])
        ctc_required_frames = int(row["ctc_required_frames"])
        ctc_logit_frames = int(row["ctc_logit_frames"])
        num_frames = int(row["num_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Correction-source row {line_number} lacks valid CTC/frame metadata."
        ) from error
    if (
        normalized_chars <= 0
        or ctc_num_tokens <= 0
        or ctc_unk_tokens != 0
        or ctc_forbidden_tokens != 0
        or ctc_required_frames <= 0
        or ctc_logit_frames <= 0
        or ctc_required_frames > ctc_logit_frames
        or num_frames <= 0
    ):
        raise ValueError(
            f"Correction-source row {line_number} has an invalid pronunciation CTC target."
        )
    if not str(row.get("stage211_sft_interleave_lane") or "").strip():
        raise ValueError(f"Correction-source row {line_number} lacks its full-profile lane.")
    return split, key, language


def _score_candidate(
    *,
    seed: int,
    language: str,
    source_dataset: str,
    bucket_id: int,
    key: str,
) -> tuple[str, int]:
    payload = (
        f"{seed}\0stage211_sft_correction\0{language}\0"
        f"{source_dataset}\0{bucket_id}\0{key}"
    ).encode("utf-8")
    score_hex = hashlib.sha256(payload).hexdigest()
    return score_hex, int(score_hex, 16)


def _make_candidate(
    *,
    row: Mapping[str, Any],
    key: str,
    language: str,
    source_dataset: str,
    bucket_id: int,
    seed: int,
) -> CorrectionCandidate:
    score_hex, score_int = _score_candidate(
        seed=seed,
        language=language,
        source_dataset=source_dataset,
        bucket_id=bucket_id,
        key=key,
    )
    output_row = dict(row)
    output_row[CORRECTION_FIELD] = language
    output_row[CORRECTION_BUCKET_FIELD] = bucket_id
    return CorrectionCandidate(
        key=key,
        language=language,
        source_dataset=source_dataset,
        acoustic_bucket_id=bucket_id,
        num_frames=int(row["num_frames"]),
        ctc_num_tokens=int(row["ctc_num_tokens"]),
        score_hex=score_hex,
        score_int=score_int,
        source_row_sha256=_payload_sha256(row),
        rendered_row=_json_bytes(output_row).decode("utf-8") + "\n",
    )


def _nested_bucket_counts(values: Mapping[tuple[str, int], int]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for (source_dataset, bucket_id), count in sorted(values.items()):
        nested[source_dataset][str(bucket_id)] = int(count)
    return {source: dict(buckets) for source, buckets in sorted(nested.items())}


def select_correction_rows(
    source_index: Path,
    *,
    seed: int = DEFAULT_SEED,
    bucket_width: int = DEFAULT_BUCKET_WIDTH,
) -> CorrectionSelection:
    source_index = source_index.expanduser().resolve()
    if not source_index.is_file() or source_index.stat().st_size <= 0:
        raise ValueError(f"Full labeled length index is unavailable: {source_index}")
    if bucket_width <= 0:
        raise ValueError("Correction bucket width must be positive.")

    seen_keys: set[str] = set()
    eval_keys: set[str] = set()
    chinese: list[CorrectionCandidate] = []
    english_capacities: Counter[tuple[str, int]] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    split_source_counts: Counter[tuple[str, str]] = Counter()
    split_language_counts: Counter[tuple[str, str]] = Counter()
    input_frames = 0
    input_ctc_tokens = 0

    for line_number, row in _iter_rows(source_index):
        split, key, language = _validate_source_row(row, line_number=line_number)
        if key in seen_keys:
            raise ValueError(f"Correction-source utterance {key!r} is duplicated.")
        seen_keys.add(key)
        source_dataset = str(row["source_dataset"])
        num_frames = int(row["num_frames"])
        bucket_id = num_frames // bucket_width
        split_counts[split] += 1
        source_counts[source_dataset] += 1
        language_counts[language] += 1
        split_source_counts[(split, source_dataset)] += 1
        split_language_counts[(split, language)] += 1
        input_frames += num_frames
        input_ctc_tokens += int(row["ctc_num_tokens"])
        if split == "eval":
            eval_keys.add(key)
            continue
        if language == "zh":
            chinese.append(
                _make_candidate(
                    row=row,
                    key=key,
                    language=language,
                    source_dataset=source_dataset,
                    bucket_id=bucket_id,
                    seed=seed,
                )
            )
        else:
            english_capacities[(source_dataset, bucket_id)] += 1

    if set(source_counts) != set(SOURCE_LANGUAGES):
        raise ValueError(
            "Correction source does not contain exactly the four full-profile datasets: "
            f"{dict(sorted(source_counts.items()))}"
        )
    if any(split_counts[split] <= 0 for split in ("train", "eval")):
        raise ValueError(f"Correction source requires non-empty train/eval splits: {split_counts}")
    chinese_target = len(chinese)
    if chinese_target <= 0:
        raise ValueError("Correction source has no accepted Chinese train rows.")
    source_capacities = {
        source_dataset: sum(
            count
            for (candidate_source, _), count in english_capacities.items()
            if candidate_source == source_dataset
        )
        for source_dataset in ENGLISH_SOURCES
    }
    source_quotas = _allocate_even_with_capacity(
        source_capacities,
        target=chinese_target,
        seed=seed,
        scope="stage211_sft_correction_en_sources",
    )
    bucket_quotas: dict[tuple[str, int], int] = {}
    for source_dataset in ENGLISH_SOURCES:
        capacities = {
            bucket_id: count
            for (candidate_source, bucket_id), count in english_capacities.items()
            if candidate_source == source_dataset
        }
        quotas = _allocate_even_with_capacity(
            capacities,
            target=int(source_quotas.get(source_dataset, 0)),
            seed=seed,
            scope=f"stage211_sft_correction_en_buckets:{source_dataset}",
        )
        bucket_quotas.update(
            ((source_dataset, int(bucket_id)), int(count))
            for bucket_id, count in quotas.items()
        )

    heaps: dict[tuple[str, int], list[tuple[int, int, CorrectionCandidate]]] = defaultdict(list)
    serial = 0
    second_pass_rows = 0
    for line_number, row in _iter_rows(source_index):
        split = str(row.get("split") or "")
        source_dataset = str(row.get("source_dataset") or "")
        if split != "train" or source_dataset not in ENGLISH_SOURCES:
            continue
        second_pass_rows += 1
        key = _row_key(row)
        bucket_id = int(row["num_frames"]) // bucket_width
        stratum = (source_dataset, bucket_id)
        quota = int(bucket_quotas.get(stratum, 0))
        if quota <= 0:
            continue
        candidate = _make_candidate(
            row=row,
            key=key,
            language="en",
            source_dataset=source_dataset,
            bucket_id=bucket_id,
            seed=seed,
        )
        heap = heaps[stratum]
        serial += 1
        entry = (-candidate.score_int, serial, candidate)
        if len(heap) < quota:
            heapq.heappush(heap, entry)
        elif candidate.score_int < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    english: list[CorrectionCandidate] = []
    for stratum, quota in sorted(bucket_quotas.items()):
        actual = len(heaps[stratum])
        if actual != quota:
            raise ValueError(
                f"Correction English reservoir underfilled for {stratum}: "
                f"required={quota} actual={actual}"
            )
        english.extend(entry[2] for entry in heaps[stratum])
    if len(english) != chinese_target:
        raise RuntimeError(
            f"Correction language balance failed: en={len(english)} zh={chinese_target}"
        )
    candidates = chinese + english
    candidates.sort(
        key=lambda item: (
            item.acoustic_bucket_id,
            item.language,
            item.source_dataset,
            item.score_hex,
            item.key,
        )
    )
    selected_keys = [candidate.key for candidate in candidates]
    if len(set(selected_keys)) != len(selected_keys):
        raise RuntimeError("Correction selection contains duplicate utterance IDs.")
    if set(selected_keys) & eval_keys:
        raise RuntimeError("Full-profile eval rows leaked into correction training selection.")

    scan = {
        "rows": sum(split_counts.values()),
        "unique_keys": len(seen_keys),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "split_source_counts": {
            f"{split}/{source}": count
            for (split, source), count in sorted(split_source_counts.items())
        },
        "split_language_counts": {
            f"{split}/{language}": count
            for (split, language), count in sorted(split_language_counts.items())
        },
        "total_frames": input_frames,
        "ctc_tokens": input_ctc_tokens,
        "eval_unique_keys": len(eval_keys),
        "eval_keys_sha256": _keys_sha256(eval_keys),
        "english_capacity_rows": sum(english_capacities.values()),
        "english_capacities": _nested_bucket_counts(english_capacities),
        "english_second_pass_rows": second_pass_rows,
        "chinese_train_rows": chinese_target,
    }
    return CorrectionSelection(
        candidates=tuple(candidates),
        scan=scan,
        source_quotas=dict(sorted(source_quotas.items())),
        bucket_quotas=dict(sorted(bucket_quotas.items())),
    )


def _resolve_part_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _normalized_split(
    manifest_path: Path,
    split: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = {"num_samples": int(split.get("num_samples", 0)), "buckets": []}
    inventory: list[dict[str, Any]] = []
    for bucket in split.get("buckets", []):
        bucket_id = int(bucket["bucket_id"])
        parts = []
        for raw_part in bucket.get("parts", []):
            path = _resolve_part_path(manifest_path, str(raw_part["path"]))
            if not path.is_file():
                raise FileNotFoundError(str(path))
            part = {
                "path": str(path),
                "num_samples": int(raw_part["num_samples"]),
            }
            for optional in ("first_shard", "last_shard", "source_label"):
                if raw_part.get(optional) is not None:
                    part[optional] = raw_part[optional]
            parts.append(part)
            inventory.append(
                {
                    "bucket_id": bucket_id,
                    **part,
                    "sha256": sha256_file(path),
                }
            )
        output["buckets"].append(
            {
                "bucket_id": bucket_id,
                "num_samples": sum(int(part["num_samples"]) for part in parts),
                "parts": parts,
            }
        )
    actual = sum(int(bucket["num_samples"]) for bucket in output["buckets"])
    if actual != output["num_samples"]:
        raise ValueError(
            f"Bucket split sample mismatch at {manifest_path}: "
            f"declared={output['num_samples']} parts={actual}"
        )
    return output, inventory


def _bucket_builder_binary() -> Path:
    binary = REPO_ROOT / "tools" / "target" / "release" / "build_bucket_index"
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(REPO_ROOT / "tools" / "Cargo.toml"),
            "--bin",
            "build_bucket_index",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    if not binary.is_file():
        raise FileNotFoundError(str(binary))
    return binary


def _run_bucket_builder(
    *,
    labeled_root: Path,
    length_index: Path,
    bucket_dir: Path,
    manifest_path: Path,
    bucket_width: int,
    entries_per_part: int,
) -> Path:
    binary = _bucket_builder_binary()
    subprocess.run(
        [
            str(binary),
            "--shard-root",
            str(labeled_root),
            "--length-index-path",
            str(length_index),
            "--output-dir",
            str(bucket_dir),
            "--manifest-path",
            str(manifest_path),
            "--bucket-width",
            str(bucket_width),
            "--entries-per-part",
            str(entries_per_part),
            "--text-cost-source",
            "auto",
            "--text-cost-weight",
            "4",
            "--json-size-text-offset",
            "256",
            "--source-field",
            CORRECTION_FIELD,
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return binary


def _inject_full_eval_split(
    *,
    correction_manifest_path: Path,
    full_manifest_path: Path,
) -> list[dict[str, Any]]:
    correction = json.loads(correction_manifest_path.read_text(encoding="utf-8"))
    full = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(correction, dict) or not isinstance(full, dict):
        raise ValueError("Correction/full bucket manifests must be JSON objects.")
    if set(correction.get("splits", {})) != {"train"}:
        raise ValueError("Correction bucket builder unexpectedly emitted non-train splits.")
    full_eval = full.get("splits", {}).get("eval")
    if not isinstance(full_eval, dict):
        raise ValueError("Full labeled bucket manifest has no eval split.")
    normalized_eval, inventory = _normalized_split(full_manifest_path, full_eval)
    correction["splits"]["eval"] = normalized_eval
    _replace_json(correction_manifest_path, correction)
    return inventory


def _auto_text_units(row: Mapping[str, Any]) -> float:
    for field in ("num_text_tokens", "num_text_chars"):
        value = row.get(field)
        if value is not None:
            return float(value)
    if row.get("text_bytes") is not None:
        return float(row["text_bytes"]) / 4.0
    if row.get("json_size") is not None:
        return max(0.0, float(row["json_size"]) - 256.0) / 4.0
    return 0.0


def _manifest_inventory_and_audit(
    *,
    manifest_path: Path,
    expected_candidates: Sequence[CorrectionCandidate],
    expected_eval_keys_sha256: str,
    expected_eval_samples: int,
    full_eval_inventory: Sequence[Mapping[str, Any]],
    bucket_width: int,
) -> dict[str, Any]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("source_field") != CORRECTION_FIELD:
        raise ValueError("Correction manifest source field mismatch.")
    if raw.get("bucket_metric") != "audio_frames_plus_text_cost":
        raise ValueError("Correction manifest does not use audio plus text-cost bucketing.")
    if raw.get("text_cost_source") != "auto" or float(raw.get("text_cost_weight", -1)) != 4.0:
        raise ValueError("Correction manifest text-cost contract mismatch.")
    manifest = load_webdataset_bucket_manifest(manifest_path)
    if manifest.bucket_width != bucket_width:
        raise ValueError("Correction manifest bucket width mismatch.")
    expected_by_key = {candidate.key: candidate for candidate in expected_candidates}
    seen_train: set[str] = set()
    train_source_labels: Counter[str] = Counter()
    output_parts: list[dict[str, Any]] = []
    for bucket in manifest.splits.get("train", ()):
        for part in bucket.parts:
            if part.source_label not in {"en", "zh"}:
                raise ValueError(f"Correction train part has invalid lane {part.source_label!r}.")
            part_path = _resolve_part_path(manifest_path, part.path)
            rows = 0
            with part_path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    rows += 1
                    row = json.loads(line)
                    key = _row_key(row)
                    candidate = expected_by_key.get(key)
                    if candidate is None:
                        raise ValueError(f"Correction manifest contains unselected key {key!r}.")
                    if key in seen_train:
                        raise ValueError(f"Correction manifest duplicates train key {key!r}.")
                    if line != candidate.rendered_row:
                        raise ValueError(
                            f"Correction manifest changed selected row {key!r} at "
                            f"{part_path}:{line_number}."
                        )
                    expected_bucket = int(
                        (int(row["num_frames"]) + round(4.0 * _auto_text_units(row)))
                        // bucket_width
                    )
                    if expected_bucket != bucket.bucket_id:
                        raise ValueError(
                            f"Correction manifest bucket mismatch for {key!r}: "
                            f"expected={expected_bucket} actual={bucket.bucket_id}"
                        )
                    if row.get(CORRECTION_FIELD) != part.source_label:
                        raise ValueError(f"Correction row/part lane mismatch for {key!r}.")
                    seen_train.add(key)
                    train_source_labels[str(part.source_label)] += 1
            if rows != part.num_samples:
                raise ValueError(
                    f"Correction part count mismatch for {part_path}: "
                    f"declared={part.num_samples} actual={rows}"
                )
            output_parts.append(
                {
                    "bucket_id": bucket.bucket_id,
                    "path": str(part_path),
                    "num_samples": rows,
                    "source_label": part.source_label,
                    "sha256": sha256_file(part_path),
                }
            )
    if seen_train != set(expected_by_key):
        raise ValueError("Correction manifest train coverage differs from selected rows.")
    expected_per_language = len(expected_candidates) // 2
    if dict(sorted(train_source_labels.items())) != {
        "en": expected_per_language,
        "zh": expected_per_language,
    }:
        raise ValueError("Correction manifest is not exactly language balanced.")

    normalized_full_inventory = [dict(record) for record in full_eval_inventory]
    actual_eval_inventory: list[dict[str, Any]] = []
    eval_keys: set[str] = set()
    for bucket in manifest.splits.get("eval", ()):
        for part in bucket.parts:
            part_path = _resolve_part_path(manifest_path, part.path)
            rows = 0
            with part_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    rows += 1
                    row = json.loads(line)
                    key = _row_key(row)
                    if not key or key in eval_keys:
                        raise ValueError("Correction eval monitor has a missing/duplicate key.")
                    if str(row.get("split") or "") != "eval":
                        raise ValueError("Correction eval monitor contains a non-eval source row.")
                    eval_keys.add(key)
            if rows != part.num_samples:
                raise ValueError(f"Correction eval-monitor part count mismatch: {part_path}")
            record = {
                "bucket_id": bucket.bucket_id,
                "path": str(part_path),
                "num_samples": rows,
            }
            if part.first_shard is not None:
                record["first_shard"] = part.first_shard
            if part.last_shard is not None:
                record["last_shard"] = part.last_shard
            if part.source_label is not None:
                record["source_label"] = part.source_label
            record["sha256"] = sha256_file(part_path)
            actual_eval_inventory.append(record)
    if actual_eval_inventory != normalized_full_inventory:
        raise ValueError("Correction eval monitor differs from full-profile eval parts.")
    if len(eval_keys) != expected_eval_samples or _keys_sha256(eval_keys) != (
        expected_eval_keys_sha256
    ):
        raise ValueError("Correction eval monitor key binding mismatch.")
    if eval_keys & seen_train:
        raise ValueError("Correction train/eval key sets overlap.")

    train_samples = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
    eval_samples = sum(bucket.num_samples for bucket in manifest.splits.get("eval", ()))
    if train_samples != len(expected_candidates) or eval_samples != expected_eval_samples:
        raise ValueError("Correction manifest split counts mismatch.")
    steps = estimate_bucket_manifest_steps(
        manifest,
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
        world_size=TRAIN_WORLD_SIZE,
        frame_budget=TRAIN_FRAME_BUDGET,
        drop_last=False,
    )
    tail_padding = estimate_bucket_manifest_tail_padding_samples(
        manifest,
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
        world_size=TRAIN_WORLD_SIZE,
        frame_budget=TRAIN_FRAME_BUDGET,
    )
    return {
        "train_samples": train_samples,
        "eval_monitor_samples": eval_samples,
        "train_source_labels": dict(sorted(train_source_labels.items())),
        "train_output_parts": output_parts,
        "full_eval_parts": actual_eval_inventory,
        "estimated_train_steps": steps,
        "tail_padding_samples": tail_padding,
    }


def _assert_full_profile_matches_scan(
    full_profile: Mapping[str, Any],
    scan: Mapping[str, Any],
) -> None:
    expected = full_profile.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("Full labeled profile lacks expected coverage.")
    required = {
        "train_samples": int(scan["split_counts"]["train"]),
        "eval_samples": int(scan["split_counts"]["eval"]),
        "total_samples": int(scan["rows"]),
        "unique_utterance_ids": int(scan["unique_keys"]),
        "source_counts": dict(scan["source_counts"]),
        "language_counts": dict(scan["language_counts"]),
    }
    for key, value in required.items():
        if expected.get(key) != value:
            raise ValueError(
                f"Correction source scan differs from full profile for {key}: "
                f"expected={expected.get(key)!r} actual={value!r}"
            )


def _compose_receipt(
    *,
    full_profile_path: Path,
    full_profile: Mapping[str, Any],
    output_root: Path,
    selection: CorrectionSelection,
    seed: int,
    bucket_width: int,
    entries_per_part: int,
) -> dict[str, Any]:
    full_root = Path(str(full_profile["labeled_webdataset_root"])).resolve()
    full_index = Path(str(full_profile["length_index_path"])).resolve()
    full_manifest = Path(str(full_profile["bucket_manifest_path"])).resolve()
    correction_index = output_root / "webdataset_lengths.jsonl"
    correction_manifest = output_root / "webdataset_buckets_audio_text" / "manifest.json"
    raw_full = json.loads(full_manifest.read_text(encoding="utf-8"))
    full_eval, full_eval_inventory = _normalized_split(
        full_manifest,
        raw_full.get("splits", {}).get("eval", {}),
    )
    manifest_audit = _manifest_inventory_and_audit(
        manifest_path=correction_manifest,
        expected_candidates=selection.candidates,
        expected_eval_keys_sha256=str(selection.scan["eval_keys_sha256"]),
        expected_eval_samples=int(selection.scan["eval_unique_keys"]),
        full_eval_inventory=full_eval_inventory,
        bucket_width=bucket_width,
    )
    if manifest_audit["eval_monitor_samples"] != int(full_eval["num_samples"]):
        raise ValueError("Correction eval-monitor count differs from full-profile manifest.")

    language_counts = Counter(candidate.language for candidate in selection.candidates)
    source_counts = Counter(candidate.source_dataset for candidate in selection.candidates)
    acoustic_bucket_counts: dict[str, Counter[int]] = defaultdict(Counter)
    total_frames = 0
    total_ctc_tokens = 0
    selected_records: list[str] = []
    for candidate in selection.candidates:
        acoustic_bucket_counts[candidate.language][candidate.acoustic_bucket_id] += 1
        total_frames += candidate.num_frames
        total_ctc_tokens += candidate.ctc_num_tokens
        selected_records.append(
            f"{candidate.language}\0{candidate.key}\0{candidate.source_row_sha256}"
        )
    selected_keys = [candidate.key for candidate in selection.candidates]
    label_preparation = full_profile.get("labeled_data_audit", {}).get("label_preparation", {})
    if not isinstance(label_preparation, dict):
        raise ValueError("Full labeled profile lacks label-preparation support proof.")
    binary = REPO_ROOT / "tools" / "target" / "release" / "build_bucket_index"
    if not binary.is_file():
        raise FileNotFoundError(str(binary))
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stage211_sft_correction_profile",
        "phase": "sft_correction",
        "complete": True,
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "contract": {
            "admission": "only_after_failed_complete_full_labeled_sft_public_gate",
            "mandatory_full_sft_not_replaced": True,
            "architecture": "BiRWKV_TimeMixer_unchanged",
            "trainable_boundary": "mixer_only",
            "learning_rate": 1e-7,
            "epochs_per_round": 1,
            "max_rounds": 3,
            "early_stopping_within_round": False,
            "ctc_plus_nano_alignment": True,
            "full_profile_eval_excluded_from_training": True,
            "full_profile_eval_reused_read_only_for_step_eval": True,
        },
        "full_labeled_profile": {
            "receipt_path": str(full_profile_path),
            "receipt_sha256": sha256_file(full_profile_path),
            "labeled_root": str(full_root),
            "length_index_path": str(full_index),
            "length_index_sha256": sha256_file(full_index),
            "bucket_manifest_path": str(full_manifest),
            "bucket_manifest_sha256": sha256_file(full_manifest),
            "expected": full_profile["expected"],
            "source_filter": full_profile["source_filter"],
        },
        "ctc_label_support": {
            "tokenizer_type": label_preparation.get("tokenizer_type"),
            "tokenizer_model_path": label_preparation.get("tokenizer_model_path"),
            "tokenizer_model_sha256": label_preparation.get("tokenizer_model_sha256"),
            "text_normalization": label_preparation.get("text_normalization"),
            "frontend_downsample": label_preparation.get("frontend_downsample"),
            "drop_unk_token": label_preparation.get("drop_unk_token"),
            "ctc_suppressed_token_ids_count": label_preparation.get(
                "ctc_suppressed_token_ids_count"
            ),
            "ctc_suppressed_token_ids_sha256": label_preparation.get(
                "ctc_suppressed_token_ids_sha256"
            ),
        },
        "selection": {
            "seed": seed,
            "acoustic_bucket_width": bucket_width,
            "algorithm": (
                "all_chinese_train_plus_equal_english_two_pass_equal_source_then_"
                "equal_80_frame_bucket_sha256_priority_reservoir_without_replacement"
            ),
            "all_chinese_train_rows_required": True,
            "english_without_replacement": True,
            "source_quotas": selection.source_quotas,
            "bucket_quotas": _nested_bucket_counts(selection.bucket_quotas),
        },
        "source_scan": selection.scan,
        "train_samples": len(selection.candidates),
        "unique_train_keys": len(selected_keys),
        "selected_keys_sha256": _keys_sha256(selected_keys),
        "selected_rows_sha256": _records_sha256(selected_records),
        "language_counts": dict(sorted(language_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "acoustic_bucket_counts": {
            language: {
                str(bucket_id): count
                for bucket_id, count in sorted(buckets.items())
            }
            for language, buckets in sorted(acoustic_bucket_counts.items())
        },
        "total_train_frames": total_frames,
        "total_train_hours": total_frames / 100.0 / 3600.0,
        "ctc_train_tokens": total_ctc_tokens,
        "batch_size": TRAIN_BATCH_SIZE,
        "world_size": TRAIN_WORLD_SIZE,
        "frame_budget": TRAIN_FRAME_BUDGET,
        "estimated_train_steps": manifest_audit["estimated_train_steps"],
        "tail_padding_samples": manifest_audit["tail_padding_samples"],
        "executed_sample_exposures": (
            len(selection.candidates) + int(manifest_audit["tail_padding_samples"])
        ),
        "eval_monitor_samples": manifest_audit["eval_monitor_samples"],
        "eval_monitor_keys_sha256": selection.scan["eval_keys_sha256"],
        "eval_monitor_excluded_from_train_accounting": True,
        "length_index_path": str(correction_index),
        "length_index_sha256": sha256_file(correction_index),
        "bucket_manifest_path": str(correction_manifest),
        "bucket_manifest_sha256": sha256_file(correction_manifest),
        "bucket_manifest_train_source_labels": manifest_audit["train_source_labels"],
        "bucket_output_parts": manifest_audit["train_output_parts"],
        "full_eval_parts": manifest_audit["full_eval_parts"],
        "bucket_builder": {
            "binary_path": str(binary),
            "binary_sha256": sha256_file(binary),
            "source_path": str(
                REPO_ROOT / "tools" / "src" / "bin" / "build_bucket_index.rs"
            ),
            "source_sha256": sha256_file(
                REPO_ROOT / "tools" / "src" / "bin" / "build_bucket_index.rs"
            ),
            "bucket_width": bucket_width,
            "entries_per_part": entries_per_part,
            "text_cost_source": "auto",
            "text_cost_weight": 4.0,
            "json_size_text_offset": 256,
            "source_field": CORRECTION_FIELD,
        },
    }


def validate_correction_profile(
    receipt_path: Path,
    *,
    expected_full_profile_path: Path | None = None,
    expected_seed: int | None = None,
    expected_bucket_width: int | None = None,
    expected_entries_per_part: int | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    if not receipt_path.is_file() or receipt_path.stat().st_size <= 0:
        raise ValueError(f"Stage211D correction profile is unavailable: {receipt_path}")
    actual = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict):
        raise ValueError("Stage211D correction profile must be a JSON object.")
    required = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stage211_sft_correction_profile",
        "phase": "sft_correction",
        "complete": True,
    }
    if any(actual.get(key) != value for key, value in required.items()):
        raise ValueError("Stage211D correction profile contract mismatch.")
    full_binding = actual.get("full_labeled_profile")
    if not isinstance(full_binding, dict):
        raise ValueError("Stage211D correction profile lacks its full-profile binding.")
    full_profile_path = Path(str(full_binding.get("receipt_path") or "")).resolve()
    if expected_full_profile_path is not None and full_profile_path != (
        expected_full_profile_path.expanduser().resolve()
    ):
        raise ValueError("Requested full labeled profile differs from correction receipt.")
    if not full_profile_path.is_file() or full_binding.get("receipt_sha256") != sha256_file(
        full_profile_path
    ):
        raise ValueError("Bound full labeled profile is unavailable or changed.")
    full_profile = validate_full_labeled_profile_receipt(full_profile_path)
    seed = int(actual.get("selection", {}).get("seed", -1))
    bucket_width = int(actual.get("selection", {}).get("acoustic_bucket_width", -1))
    entries_per_part = int(actual.get("bucket_builder", {}).get("entries_per_part", -1))
    for expected, observed, label in (
        (expected_seed, seed, "seed"),
        (expected_bucket_width, bucket_width, "bucket width"),
        (expected_entries_per_part, entries_per_part, "entries per part"),
    ):
        if expected is not None and int(expected) != observed:
            raise ValueError(f"Requested correction {label} differs from its receipt.")
    full_index = Path(str(full_profile["length_index_path"])).resolve()
    selection = select_correction_rows(full_index, seed=seed, bucket_width=bucket_width)
    _assert_full_profile_matches_scan(full_profile, selection.scan)
    output_root = receipt_path.parent
    expected_index = output_root / "webdataset_lengths.jsonl"
    rendered_index = "".join(candidate.rendered_row for candidate in selection.candidates)
    if not expected_index.is_file() or expected_index.read_text(encoding="utf-8") != rendered_index:
        raise ValueError("Correction length index differs from deterministic selection.")
    expected = _compose_receipt(
        full_profile_path=full_profile_path,
        full_profile=full_profile,
        output_root=output_root,
        selection=selection,
        seed=seed,
        bucket_width=bucket_width,
        entries_per_part=entries_per_part,
    )
    if actual != expected:
        raise ValueError("Stage211D correction profile differs from deep recomputation.")
    return actual


def build_correction_profile(
    *,
    full_profile_path: Path,
    output_root: Path,
    seed: int = DEFAULT_SEED,
    bucket_width: int = DEFAULT_BUCKET_WIDTH,
    entries_per_part: int = DEFAULT_ENTRIES_PER_PART,
) -> dict[str, Any]:
    full_profile_path = full_profile_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    receipt_path = output_root / "stage211_sft_correction_profile.json"
    if receipt_path.is_file():
        return validate_correction_profile(
            receipt_path,
            expected_full_profile_path=full_profile_path,
            expected_seed=seed,
            expected_bucket_width=bucket_width,
            expected_entries_per_part=entries_per_part,
        )
    if bucket_width <= 0 or entries_per_part <= 0:
        raise ValueError("Correction bucket width and entries per part must be positive.")
    full_profile = validate_full_labeled_profile_receipt(full_profile_path)
    full_root = Path(str(full_profile["labeled_webdataset_root"])).resolve()
    full_index = Path(str(full_profile["length_index_path"])).resolve()
    full_manifest = Path(str(full_profile["bucket_manifest_path"])).resolve()
    selection = select_correction_rows(full_index, seed=seed, bucket_width=bucket_width)
    _assert_full_profile_matches_scan(full_profile, selection.scan)
    output_root.mkdir(parents=True, exist_ok=True)
    correction_index = output_root / "webdataset_lengths.jsonl"
    _write_immutable(
        correction_index,
        "".join(candidate.rendered_row for candidate in selection.candidates),
    )
    bucket_dir = output_root / "webdataset_buckets_audio_text"
    manifest_path = bucket_dir / "manifest.json"
    _run_bucket_builder(
        labeled_root=full_root,
        length_index=correction_index,
        bucket_dir=bucket_dir,
        manifest_path=manifest_path,
        bucket_width=bucket_width,
        entries_per_part=entries_per_part,
    )
    _inject_full_eval_split(
        correction_manifest_path=manifest_path,
        full_manifest_path=full_manifest,
    )
    receipt = _compose_receipt(
        full_profile_path=full_profile_path,
        full_profile=full_profile,
        output_root=output_root,
        selection=selection,
        seed=seed,
        bucket_width=bucket_width,
        entries_per_part=entries_per_part,
    )
    _write_immutable(
        receipt_path,
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    return validate_correction_profile(
        receipt_path,
        expected_full_profile_path=full_profile_path,
        expected_seed=seed,
        expected_bucket_width=bucket_width,
        expected_entries_per_part=entries_per_part,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or deeply validate the deterministic bilingual Stage211D labeled "
            "correction profile."
        )
    )
    parser.add_argument("--full-profile", type=Path, default=DEFAULT_FULL_PROFILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bucket-width", type=int, default=DEFAULT_BUCKET_WIDTH)
    parser.add_argument("--entries-per-part", type=int, default=DEFAULT_ENTRIES_PER_PART)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    receipt_path = args.output_root.expanduser().resolve() / (
        "stage211_sft_correction_profile.json"
    )
    if args.validate_only:
        receipt = validate_correction_profile(
            receipt_path,
            expected_full_profile_path=args.full_profile,
            expected_seed=args.seed,
            expected_bucket_width=args.bucket_width,
            expected_entries_per_part=args.entries_per_part,
        )
    else:
        receipt = build_correction_profile(
            full_profile_path=args.full_profile,
            output_root=args.output_root,
            seed=args.seed,
            bucket_width=args.bucket_width,
            entries_per_part=args.entries_per_part,
        )
    print(
        "[stage211-sft-correction-profile] "
        f"train={receipt['train_samples']} en={receipt['language_counts']['en']} "
        f"zh={receipt['language_counts']['zh']} steps={receipt['estimated_train_steps']} "
        f"output={receipt_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
