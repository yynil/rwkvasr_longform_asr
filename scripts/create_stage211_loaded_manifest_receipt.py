from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from rwkvasr.data import (
    estimate_bucket_manifest_steps,
    estimate_bucket_manifest_tail_padding_samples,
    load_webdataset_bucket_manifest,
)
from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST,
    DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_AUDIO_TOTAL_ROWS,
    STAGE211_AUDIO_TRAIN_PART_COUNTS,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    sha256_file,
    validate_stage211_global_dedup_manifest,
    validate_stage211_loaded_manifest_receipt,
)


DEFAULT_METADATA_ROOT = Path.home() / "rwkvasr_data" / "stage211_full_curriculum"
DEFAULT_RUNTIME_MANIFESTS = {
    "easy": (
        Path.home()
        / "rwkvasr_data"
        / "stage211_easy_source_grouped_buckets"
        / "manifest_stage211_fixed_eval.json"
    ),
    "medium": (
        DEFAULT_METADATA_ROOT
        / "stage179b_medium_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest_stage211_fixed_eval.json"
    ),
    "hard": (
        DEFAULT_METADATA_ROOT
        / "stage179c_hard_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest_stage211_fixed_eval.json"
    ),
    "long": (
        DEFAULT_METADATA_ROOT
        / "stage179d_long_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest_stage211_fixed_eval.json"
    ),
}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(rendered).hexdigest()


def _resolve_part_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _iter_parts(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    split: str,
) -> Iterator[tuple[Path, int]]:
    split_record = manifest.get("splits", {}).get(split)
    if not isinstance(split_record, dict):
        raise ValueError(f"Manifest lacks split {split!r}: {manifest_path}")
    for bucket in split_record.get("buckets", []):
        for part in bucket.get("parts", []):
            yield (
                _resolve_part_path(manifest_path, str(part.get("path") or "")),
                int(part.get("num_samples", -1)),
            )


def _sha256_and_rows(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    last_byte = b""
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            rows += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        rows += 1
    return digest.hexdigest(), rows


def _resolve_tar_path(row: dict[str, Any], *, root: Path) -> Path:
    raw = str(row.get("tar_path") or row.get("shard_path") or "").strip()
    if raw:
        path = Path(raw)
    else:
        shard = str(row.get("shard_name") or row.get("shard") or "").strip()
        if not shard:
            raise ValueError("Stage211 row lacks tar/shard path.")
        path = Path(shard)
    return path if path.is_absolute() else root / path


def _audio_key(row: dict[str, Any], *, root: Path) -> bytes:
    audio_member = str(
        row.get("audio_member") or row.get("wav_member") or row.get("audio") or ""
    ).strip()
    if not audio_member:
        audio_member = str(row.get("utt_id") or row.get("key") or row.get("id") or "")
    payload = "\0".join(
        (
            str(_resolve_tar_path(row, root=root)),
            audio_member,
            str(row.get("audio_offset") or ""),
            str(row.get("audio_size") or ""),
        )
    )
    return hashlib.blake2b(
        payload.encode("utf-8", errors="surrogatepass"),
        digest_size=16,
    ).digest()


def _iter_rows(manifest_path: Path, manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for part_path, _ in _iter_parts(manifest_path, manifest, split="train"):
        with part_path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Non-object row in {part_path}")
                    yield row


def _sorted_key_set_sha256(keys: set[bytes]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key)
    return digest.hexdigest()


def _prove_repartition_identity(
    *,
    runtime_manifest_path: Path,
    runtime_manifest: dict[str, Any],
    global_manifest_path: Path,
    global_manifest: dict[str, Any],
    expected_rows: int,
) -> str:
    runtime_root = Path(str(runtime_manifest.get("root") or "/"))
    keys: set[bytes] = set()
    for row in _iter_rows(runtime_manifest_path, runtime_manifest):
        key = _audio_key(row, root=runtime_root)
        if key in keys:
            raise ValueError("Repartitioned Stage211 runtime manifest contains duplicate audio.")
        keys.add(key)
    if len(keys) != expected_rows:
        raise ValueError(
            f"Repartitioned Stage211 runtime rows mismatch: {len(keys)}/{expected_rows}"
        )
    key_set_sha256 = _sorted_key_set_sha256(keys)

    global_root = Path(str(global_manifest.get("root") or "/"))
    global_rows = 0
    for row in _iter_rows(global_manifest_path, global_manifest):
        global_rows += 1
        key = _audio_key(row, root=global_root)
        if key not in keys:
            raise ValueError("Repartitioned Stage211 runtime data differs from Stage179 source.")
        keys.remove(key)
    if global_rows != expected_rows or keys:
        raise ValueError(
            "Repartitioned Stage211 source/runtime key sets are incomplete: "
            f"source_rows={global_rows}/{expected_rows} missing={len(keys)}"
        )
    return key_set_sha256


def _load_work_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"part_records": {}}
    state = _load_json(path, label="Stage211 loaded-manifest work state")
    if not isinstance(state.get("part_records"), dict):
        raise ValueError(f"Invalid Stage211 loaded-manifest work state: {path}")
    return state


def _save_work_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _part_record(
    path: Path,
    *,
    declared_rows: int,
    work_state_path: Path,
    work_state: dict[str, Any],
) -> dict[str, Any]:
    stat = path.stat()
    cache_key = str(path)
    cached = work_state["part_records"].get(cache_key)
    if (
        isinstance(cached, dict)
        and int(cached.get("size_bytes", -1)) == stat.st_size
        and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
        and int(cached.get("rows", -1)) == declared_rows
        and len(str(cached.get("sha256") or "")) == 64
    ):
        return dict(cached)
    digest, actual_rows = _sha256_and_rows(path)
    if actual_rows != declared_rows:
        raise ValueError(f"Stage211 part row mismatch: {path} {actual_rows}/{declared_rows}")
    record = {
        "path": str(path),
        "rows": declared_rows,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }
    work_state["part_records"][cache_key] = record
    _save_work_state(work_state_path, work_state)
    return record


def build_receipt(
    *,
    global_dedup_manifest_path: Path,
    runtime_manifests: dict[str, Path],
    output_path: Path,
    work_state_path: Path,
) -> dict[str, Any]:
    global_dedup_manifest_path = global_dedup_manifest_path.expanduser().resolve()
    global_manifest = validate_stage211_global_dedup_manifest(global_dedup_manifest_path)
    global_stages = global_manifest["stages"]
    work_state = _load_work_state(work_state_path)
    segments = []
    fixed_eval_path: Path | None = None
    fixed_eval_sha256: str | None = None

    for difficulty, expected in STAGE211_AUDIO_CURRICULUM.items():
        runtime_manifest_path = runtime_manifests[difficulty].expanduser().resolve()
        runtime_manifest = _load_json(
            runtime_manifest_path,
            label=f"Stage211 {difficulty} runtime manifest",
        )
        source_manifest_path = runtime_manifest_path.with_name("manifest.json")
        source_manifest = _load_json(
            source_manifest_path,
            label=f"Stage211 {difficulty} source manifest",
        )
        runtime_without_eval = json.loads(json.dumps(runtime_manifest))
        runtime_without_eval.get("splits", {}).pop("eval", None)
        if runtime_without_eval != source_manifest:
            raise ValueError(f"Stage211 {difficulty} runtime manifest rewrites train data.")

        global_stage_name, global_stage = next(
            (
                (name, stage)
                for name, stage in global_stages.items()
                if stage.get("difficulty") == difficulty
            ),
            (None, None),
        )
        if global_stage_name is None or not isinstance(global_stage, dict):
            raise ValueError(f"Stage211 global manifest lacks {difficulty}.")
        global_source_manifest_path = Path(global_stage["bucket_manifest_path"]).resolve()
        global_source_manifest = _load_json(
            global_source_manifest_path,
            label=f"Stage211 {difficulty} global source manifest",
        )
        source_length_index_path = str(runtime_manifest.get("source_length_index_path") or "")
        if source_length_index_path != str(global_stage.get("length_index_path") or ""):
            raise ValueError(f"Stage211 {difficulty} length-index path mismatch.")

        eval_parts = list(_iter_parts(runtime_manifest_path, runtime_manifest, split="eval"))
        if len(eval_parts) != 1 or eval_parts[0][1] != 256:
            raise ValueError(f"Stage211 {difficulty} fixed eval split mismatch.")
        current_eval_path = eval_parts[0][0]
        current_eval_sha256 = sha256_file(current_eval_path)
        if fixed_eval_path is None:
            fixed_eval_path = current_eval_path
            fixed_eval_sha256 = current_eval_sha256
        elif current_eval_path != fixed_eval_path or current_eval_sha256 != fixed_eval_sha256:
            raise ValueError("Stage211 runtime manifests do not share one fixed eval part.")

        part_records = []
        declared_rows = 0
        runtime_parts = list(_iter_parts(runtime_manifest_path, runtime_manifest, split="train"))
        for index, (part_path, part_rows) in enumerate(runtime_parts, start=1):
            record = _part_record(
                part_path,
                declared_rows=part_rows,
                work_state_path=work_state_path,
                work_state=work_state,
            )
            part_records.append(record)
            declared_rows += part_rows
            if index == 1 or index % 25 == 0 or index == len(runtime_parts):
                print(
                    f"[loaded-manifest] difficulty={difficulty} hashed_parts="
                    f"{index}/{len(runtime_parts)} rows={declared_rows}",
                    flush=True,
                )
        if (
            declared_rows != int(expected["rows"])
            or len(part_records) != STAGE211_AUDIO_TRAIN_PART_COUNTS[difficulty]
        ):
            raise ValueError(f"Stage211 {difficulty} runtime part coverage mismatch.")
        parsed_manifest = load_webdataset_bucket_manifest(runtime_manifest_path)
        steps_per_epoch = estimate_bucket_manifest_steps(
            parsed_manifest,
            split="train",
            batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
            world_size=STAGE211_FULL_DATA_WORLD_SIZE,
            frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
            drop_last=False,
        )
        tail_padding = estimate_bucket_manifest_tail_padding_samples(
            parsed_manifest,
            split="train",
            batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
            world_size=STAGE211_FULL_DATA_WORLD_SIZE,
            frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        )
        if steps_per_epoch != int(expected["steps_per_epoch"]) or tail_padding != int(
            expected["tail_padding_samples_per_epoch"]
        ):
            raise ValueError(f"Stage211 {difficulty} runtime batching contract mismatch.")

        global_source_sha256 = sha256_file(global_source_manifest_path)
        source_sha256 = sha256_file(source_manifest_path)
        repartitioned = global_source_sha256 != source_sha256
        loaded_key_set_sha256 = None
        global_key_set_sha256 = None
        if repartitioned:
            cache = work_state.get("repartition_identity")
            cache_key = {
                "difficulty": difficulty,
                "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
                "global_source_manifest_sha256": global_source_sha256,
            }
            if isinstance(cache, dict) and all(
                cache.get(key) == value for key, value in cache_key.items()
            ):
                loaded_key_set_sha256 = str(cache.get("audio_key_set_sha256") or "")
            else:
                loaded_key_set_sha256 = _prove_repartition_identity(
                    runtime_manifest_path=runtime_manifest_path,
                    runtime_manifest=runtime_manifest,
                    global_manifest_path=global_source_manifest_path,
                    global_manifest=global_source_manifest,
                    expected_rows=int(expected["rows"]),
                )
                work_state["repartition_identity"] = {
                    **cache_key,
                    "audio_key_set_sha256": loaded_key_set_sha256,
                }
                _save_work_state(work_state_path, work_state)
            global_key_set_sha256 = loaded_key_set_sha256

        train_split_sha256 = _canonical_json_sha256(runtime_manifest.get("splits", {}).get("train"))
        segments.append(
            {
                "difficulty": difficulty,
                "global_stage_name": global_stage_name,
                "train_rows": int(expected["rows"]),
                "train_hours": float(global_stage["selected_hours"]),
                "train_parts": len(part_records),
                "steps_per_epoch": steps_per_epoch,
                "epochs": STAGE211_FULL_DATA_EPOCHS,
                "total_steps": steps_per_epoch * STAGE211_FULL_DATA_EPOCHS,
                "tail_padding_samples_per_epoch": tail_padding,
                "source_length_index_path": source_length_index_path,
                "global_source_manifest_path": str(global_source_manifest_path),
                "global_source_manifest_sha256": global_source_sha256,
                "source_manifest_path": str(source_manifest_path),
                "source_manifest_sha256": source_sha256,
                "runtime_manifest_path": str(runtime_manifest_path),
                "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
                "train_split_sha256": train_split_sha256,
                "repartitioned": repartitioned,
                "loaded_audio_key_set_sha256": loaded_key_set_sha256,
                "global_audio_key_set_sha256": global_key_set_sha256,
                "part_records_sha256": _canonical_json_sha256(part_records),
                "part_records": part_records,
            }
        )

    if fixed_eval_path is None or fixed_eval_sha256 is None:
        raise ValueError("Stage211 loaded-manifest receipt lacks fixed eval data.")
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "loaded_manifest_chain",
        "complete": True,
        "part_hash_algorithm": "sha256",
        "part_hash_audit_complete": True,
        "global_dedup_manifest_path": str(global_dedup_manifest_path),
        "global_dedup_manifest_sha256": sha256_file(global_dedup_manifest_path),
        "dedupe_key": global_manifest["dedupe_key"],
        "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
        "total_hours": float(global_manifest["total_unique_hours"]),
        "total_train_parts": sum(STAGE211_AUDIO_TRAIN_PART_COUNTS.values()),
        "fixed_eval": {
            "path": str(fixed_eval_path),
            "sha256": fixed_eval_sha256,
            "rows": 256,
        },
        "segments": segments,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if output_path.is_file() and output_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different receipt: {output_path}")
    output_path.write_text(rendered, encoding="utf-8")
    validate_stage211_loaded_manifest_receipt(
        output_path,
        expected_global_dedup_manifest=global_dedup_manifest_path,
        verify_part_sha256=False,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hash and bind the four Stage211 runtime manifests to the global Stage179 "
            "deduplicated curriculum."
        )
    )
    parser.add_argument(
        "--global-dedup-manifest",
        type=Path,
        default=DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST,
    )
    parser.add_argument(
        "--runtime-manifest",
        action="append",
        default=[],
        help="Override one runtime manifest as difficulty=/absolute/path.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT)
    parser.add_argument("--work-state", type=Path, default=None)
    args = parser.parse_args()

    runtime_manifests = dict(DEFAULT_RUNTIME_MANIFESTS)
    for raw in args.runtime_manifest:
        difficulty, separator, path = str(raw).partition("=")
        if separator != "=" or difficulty not in runtime_manifests or not path:
            parser.error("--runtime-manifest must use easy|medium|hard|long=/absolute/path")
        runtime_manifests[difficulty] = Path(path)
    output_path = args.output.expanduser().resolve()
    work_state_path = (
        args.work_state.expanduser().resolve()
        if args.work_state is not None
        else output_path.with_suffix(".work.json")
    )
    receipt = build_receipt(
        global_dedup_manifest_path=args.global_dedup_manifest,
        runtime_manifests=runtime_manifests,
        output_path=output_path,
        work_state_path=work_state_path,
    )
    print(
        f"receipt={output_path} sha256={sha256_file(output_path)} "
        f"rows={receipt['total_unique_rows']} parts={receipt['total_train_parts']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
