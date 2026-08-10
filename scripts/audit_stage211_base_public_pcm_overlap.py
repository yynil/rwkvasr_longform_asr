#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rwkvasr.data.webdataset_lengths import (
    _make_shard_reader,
    _read_indexed_entry_payload,
    parse_webdataset_length_entry,
)
from rwkvasr.eval.stage211_supplemental import (
    validate_stage211_supplemental_inventory,
)

try:
    from scripts.filter_stage211_social_pcm_overlap import (
        DEFAULT_PUBLIC_MANIFESTS,
        EXPECTED_PUBLIC_ROWS,
        FINGERPRINT_ALGORITHM,
        _load_public_fingerprint_map,
        build_public_fingerprints,
        canonical_pcm_fingerprint,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from filter_stage211_social_pcm_overlap import (
        DEFAULT_PUBLIC_MANIFESTS,
        EXPECTED_PUBLIC_ROWS,
        FINGERPRINT_ALGORITHM,
        _load_public_fingerprint_map,
        build_public_fingerprints,
        canonical_pcm_fingerprint,
    )


DEFAULT_BASE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_supplemental_natural_v1/supplemental_inventory.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_v1"
)
SCHEMA_VERSION = 1
PART_ARTIFACT = "stage211_base_public_pcm_part_fingerprints"
AUDIT_ARTIFACT = "stage211_base_public_pcm_overlap_audit"


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_line(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            yield row


def _immutable_bytes(path: Path, payload: bytes) -> str:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace different Stage211 evidence: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def _immutable_json(path: Path, payload: dict[str, Any]) -> str:
    rendered = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return _immutable_bytes(path, rendered)


def _resolve_part_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def validate_base_inventory(path: str | Path) -> dict[str, Any]:
    validated = validate_stage211_supplemental_inventory(
        path,
        require_training_ready=True,
        verify_part_sha256=True,
    )
    inventory = validated["inventory"]
    if (
        inventory.get("schema_version") != 1
        or inventory.get("artifact") != "stage211_supplemental_natural_inventory"
        or inventory.get("storage_kinds") != ["parquet", "zip"]
        or inventory.get("language") != "en"
    ):
        raise ValueError("Stage211 base public-overlap audit requires natural inventory v1.")
    manifest_path = Path(validated["bucket_manifest_path"])
    manifest = _load_json(manifest_path, label="Stage211 base bucket manifest")
    train = (manifest.get("splits") or {}).get("train")
    if not isinstance(train, dict) or not isinstance(train.get("buckets"), list):
        raise ValueError("Stage211 base bucket manifest has no train buckets.")
    parts: list[dict[str, Any]] = []
    for bucket in train["buckets"]:
        if not isinstance(bucket, dict) or not isinstance(bucket.get("parts"), list):
            raise ValueError("Stage211 base bucket manifest contains an invalid bucket.")
        bucket_id = int(bucket.get("bucket_id", -1))
        for part in bucket["parts"]:
            if not isinstance(part, dict):
                raise ValueError("Stage211 base bucket manifest contains an invalid part.")
            part_path = _resolve_part_path(manifest_path, str(part.get("path") or ""))
            expected_sha256 = str(part.get("sha256") or "")
            if (
                bucket_id < 0
                or not part_path.is_file()
                or part_path.stat().st_size != int(part.get("size_bytes", -1))
                or _sha256(part_path) != expected_sha256
                or int(part.get("num_samples", -1)) <= 0
                or not str(part.get("source_label") or "")
            ):
                raise ValueError(f"Stage211 base train part changed: {part_path}")
            parts.append(
                {
                    "part_index": len(parts),
                    "bucket_id": bucket_id,
                    "path": str(part_path),
                    "size_bytes": part_path.stat().st_size,
                    "sha256": expected_sha256,
                    "rows": int(part["num_samples"]),
                    "source_label": str(part["source_label"]),
                }
            )
    if sum(int(part["rows"]) for part in parts) != int(validated["rows"]):
        raise ValueError("Stage211 base public-overlap part coverage changed.")
    return {**validated, "parts": parts}


def _part_paths(output_root: Path, part_index: int) -> tuple[Path, Path]:
    root = output_root / "part_fingerprints"
    stem = f"part_{part_index:06d}"
    return root / f"{stem}.jsonl", root / f"{stem}.receipt.json"


def _validate_part_receipt(
    receipt_path: Path,
    *,
    base: dict[str, Any],
    part: dict[str, Any],
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, label="Stage211 base PCM part receipt")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": PART_ARTIFACT,
        "complete": True,
        "decode_failures": 0,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "base_inventory_path": base["inventory_path"],
        "base_inventory_sha256": base["inventory_sha256"],
        "base_bucket_manifest_path": base["bucket_manifest_path"],
        "base_bucket_manifest_sha256": base["bucket_manifest_sha256"],
        "part_index": int(part["part_index"]),
        "bucket_id": int(part["bucket_id"]),
        "input_part_path": str(Path(part["path"]).resolve()),
        "input_part_size_bytes": int(part["size_bytes"]),
        "input_part_sha256": part["sha256"],
        "source_label": part["source_label"],
        "rows": int(part["rows"]),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 base PCM part receipt changed: {receipt_path}")
    fingerprint_path = Path(str(receipt.get("fingerprint_part_path") or "")).resolve()
    if (
        not fingerprint_path.is_file()
        or fingerprint_path.stat().st_size
        != int(receipt.get("fingerprint_part_size_bytes", -1))
        or _sha256(fingerprint_path) != receipt.get("fingerprint_part_sha256")
    ):
        raise ValueError(f"Stage211 base PCM fingerprint part changed: {fingerprint_path}")
    with fingerprint_path.open("rb") as source:
        rows = sum(chunk.count(b"\n") for chunk in iter(lambda: source.read(1 << 20), b""))
    if rows != int(part["rows"]):
        raise ValueError(f"Stage211 base PCM fingerprint row count changed: {fingerprint_path}")
    return receipt


def fingerprint_base_part(
    *,
    base: dict[str, Any],
    output_root: Path,
    part_index: int,
) -> str:
    parts = base["parts"]
    if not 0 <= part_index < len(parts):
        raise ValueError(f"Stage211 base PCM part index is out of range: {part_index}")
    part = parts[part_index]
    fingerprint_path, receipt_path = _part_paths(output_root, part_index)
    if receipt_path.is_file():
        _validate_part_receipt(receipt_path, base=base, part=part)
        return "reused"

    input_path = Path(part["path"])
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = fingerprint_path.with_name(f"{fingerprint_path.name}.tmp.{os.getpid()}")
    duration_ms = 0
    pcm_samples = 0
    source_counts: Counter[str] = Counter()
    source_duration_ms: Counter[str] = Counter()
    reader: Any | None = None
    reader_key: tuple[str, str] | None = None
    row_count = 0
    try:
        with input_path.open("r", encoding="utf-8") as source, temporary.open("wb") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"Stage211 base row is not an object: {input_path}:{line_number}")
                entry = parse_webdataset_length_entry(raw)
                source_label = str(raw.get("source_dataset") or "")
                if entry.split != "train" or source_label != part["source_label"]:
                    raise ValueError(
                        f"Stage211 base row source/split changed: {input_path}:{line_number}"
                    )
                shard_path = Path(entry.shard_name).expanduser()
                if not shard_path.is_absolute():
                    shard_path = Path("/") / shard_path
                shard_path = shard_path.resolve()
                next_reader_key = (entry.storage_kind, str(shard_path))
                if next_reader_key != reader_key:
                    if reader is not None:
                        reader.close()
                    reader = _make_shard_reader(entry.storage_kind, shard_path)
                    reader_key = next_reader_key
                audio_bytes, _ = _read_indexed_entry_payload(reader, entry)
                fingerprint, samples = canonical_pcm_fingerprint(io.BytesIO(audio_bytes))
                row_duration_ms = int(raw.get("duration_ms") or 0)
                if row_duration_ms <= 0 or samples <= 0:
                    raise ValueError(
                        f"Stage211 base PCM duration changed: {input_path}:{line_number}"
                    )
                target.write(
                    _canonical_line(
                        {
                            "audio_member_sha256": _sha256_bytes(audio_bytes),
                            "canonical_pcm_sha256": fingerprint,
                            "duration_ms": row_duration_ms,
                            "input_line_number": line_number,
                            "input_part_index": part_index,
                            "key": entry.key,
                            "pcm_samples": samples,
                            "source_dataset": source_label,
                            "storage_kind": entry.storage_kind,
                            "utt_id": entry.utt_id,
                        }
                    )
                )
                row_count += 1
                duration_ms += row_duration_ms
                pcm_samples += samples
                source_counts[source_label] += 1
                source_duration_ms[source_label] += row_duration_ms
        if row_count != int(part["rows"]):
            raise ValueError(
                f"Stage211 base PCM part row count changed: {row_count}/{part['rows']}"
            )
        if fingerprint_path.is_file():
            if (
                fingerprint_path.stat().st_size != temporary.stat().st_size
                or _sha256(fingerprint_path) != _sha256(temporary)
            ):
                raise ValueError(
                    f"Existing Stage211 base PCM part differs from replay: {fingerprint_path}"
                )
            temporary.unlink()
        else:
            temporary.replace(fingerprint_path)
    finally:
        if reader is not None:
            reader.close()
        temporary.unlink(missing_ok=True)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": PART_ARTIFACT,
        "complete": True,
        "decode_failures": 0,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "base_inventory_path": base["inventory_path"],
        "base_inventory_sha256": base["inventory_sha256"],
        "base_bucket_manifest_path": base["bucket_manifest_path"],
        "base_bucket_manifest_sha256": base["bucket_manifest_sha256"],
        "part_index": part_index,
        "bucket_id": int(part["bucket_id"]),
        "input_part_path": str(input_path),
        "input_part_size_bytes": int(part["size_bytes"]),
        "input_part_sha256": part["sha256"],
        "source_label": part["source_label"],
        "rows": row_count,
        "duration_ms": duration_ms,
        "pcm_samples": pcm_samples,
        "source_counts": dict(sorted(source_counts.items())),
        "source_duration_ms": dict(sorted(source_duration_ms.items())),
        "fingerprint_part_path": str(fingerprint_path.resolve()),
        "fingerprint_part_size_bytes": fingerprint_path.stat().st_size,
        "fingerprint_part_sha256": _sha256(fingerprint_path),
    }
    _immutable_json(receipt_path, receipt)
    return "created"


def run_fingerprint_worker(
    *,
    base: dict[str, Any],
    output_root: Path,
    worker_index: int,
    num_workers: int,
    max_parts: int | None,
) -> dict[str, int]:
    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError("worker-index must satisfy 0 <= worker-index < num-workers")
    assigned = [
        int(part["part_index"])
        for part in base["parts"]
        if int(part["part_index"]) % num_workers == worker_index
    ]
    if max_parts is not None:
        assigned = assigned[: int(max_parts)]
    counts: Counter[str] = Counter()
    for position, part_index in enumerate(assigned, start=1):
        status = fingerprint_base_part(
            base=base,
            output_root=output_root,
            part_index=part_index,
        )
        counts[status] += 1
        print(
            f"[stage211-base-public-pcm] worker={worker_index}/{num_workers} "
            f"parts={position}/{len(assigned)} part_index={part_index} status={status}",
            flush=True,
        )
    return {"assigned": len(assigned), **dict(sorted(counts.items()))}


def _part_receipts(base: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for part in base["parts"]:
        _, receipt_path = _part_paths(output_root, int(part["part_index"]))
        receipt = _validate_part_receipt(receipt_path, base=base, part=part)
        records.append(
            {
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": _sha256(receipt_path),
                **receipt,
            }
        )
    return records


def _overlap_payload(
    *,
    part_receipts: list[dict[str, Any]],
    public_by_fingerprint: dict[str, list[dict[str, str]]],
) -> tuple[bytes, int, dict[str, int]]:
    rendered = bytearray()
    overlap_rows = 0
    overlap_by_dataset: Counter[str] = Counter()
    for receipt in part_receipts:
        fingerprint_path = Path(str(receipt["fingerprint_part_path"]))
        for row in _iter_jsonl(fingerprint_path):
            matches = public_by_fingerprint.get(str(row["canonical_pcm_sha256"]))
            if not matches:
                continue
            rendered.extend(_canonical_line({**row, "public_matches": matches}))
            overlap_rows += 1
            overlap_by_dataset.update(match["dataset"] for match in matches)
    return bytes(rendered), overlap_rows, dict(sorted(overlap_by_dataset.items()))


def validate_audit_receipt(
    path: str | Path,
    *,
    require_training_ready: bool = False,
    verify_overlap_replay: bool = True,
) -> dict[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    audit = _load_json(receipt_path, label="Stage211 base public PCM audit")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": AUDIT_ARTIFACT,
        "complete": True,
        "comparison_mode": "normalized_pcm_exact",
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "near_duplicate_complete": False,
        "decode_failures": 0,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 base public PCM audit contract changed.")
    if require_training_ready and audit.get("training_ready") is not True:
        raise ValueError("Stage211 base public PCM audit found public-evaluation overlap.")
    base_path = Path(str(audit.get("base_inventory_path") or "")).resolve()
    base = validate_base_inventory(base_path)
    if (
        audit.get("base_inventory_sha256") != base["inventory_sha256"]
        or audit.get("base_bucket_manifest_path") != base["bucket_manifest_path"]
        or audit.get("base_bucket_manifest_sha256") != base["bucket_manifest_sha256"]
    ):
        raise ValueError("Stage211 base public PCM audit base binding changed.")
    part_receipts = _part_receipts(base, receipt_path.parent)
    if audit.get("part_fingerprint_receipts") != part_receipts:
        raise ValueError("Stage211 base public PCM audit part receipt set changed.")
    scanned_rows = sum(int(record["rows"]) for record in part_receipts)
    scanned_duration_ms = sum(int(record["duration_ms"]) for record in part_receipts)
    source_counts: Counter[str] = Counter()
    source_duration_ms: Counter[str] = Counter()
    for record in part_receipts:
        source_counts.update(record["source_counts"])
        source_duration_ms.update(record["source_duration_ms"])
    if (
        scanned_rows != int(base["rows"])
        or int(audit.get("scanned_rows", -1)) != scanned_rows
        or int(audit.get("scanned_duration_ms", -1)) != scanned_duration_ms
        or not math.isclose(
            scanned_duration_ms / 3_600_000.0,
            float(base["hours"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or audit.get("scanned_counts_by_source") != dict(sorted(source_counts.items()))
        or audit.get("scanned_duration_ms_by_source")
        != dict(sorted(source_duration_ms.items()))
        or source_counts != Counter(base["inventory"]["selected_counts_by_source"])
    ):
        raise ValueError("Stage211 base public PCM audit coverage changed.")
    recorded_public_receipts = audit.get("public_fingerprint_receipts")
    if not isinstance(recorded_public_receipts, list) or not recorded_public_receipts:
        raise ValueError("Stage211 base public PCM audit public receipt set is missing.")
    public_manifests = {
        str(record["dataset"]): Path(str(record["manifest_path"])).resolve()
        for record in recorded_public_receipts
    }
    expected_public_rows = {
        str(record["dataset"]): int(record["rows"])
        for record in recorded_public_receipts
    }
    public_by_fingerprint, public_receipts = _load_public_fingerprint_map(
        public_manifests=public_manifests,
        output_root=receipt_path.parent,
        expected_rows=expected_public_rows,
    )
    if audit.get("public_fingerprint_receipts") != public_receipts:
        raise ValueError("Stage211 base public PCM audit public receipt set changed.")
    overlap_path = Path(str(audit.get("overlap_output", {}).get("path") or "")).resolve()
    if (
        not overlap_path.is_file()
        or overlap_path.stat().st_size
        != int(audit.get("overlap_output", {}).get("size_bytes", -1))
        or _sha256(overlap_path) != audit.get("overlap_output", {}).get("sha256")
    ):
        raise ValueError("Stage211 base public PCM overlap output changed.")
    if verify_overlap_replay:
        rendered, overlap_rows, overlap_by_dataset = _overlap_payload(
            part_receipts=part_receipts,
            public_by_fingerprint=public_by_fingerprint,
        )
        if overlap_path.read_bytes() != rendered:
            raise ValueError("Stage211 base public PCM overlap replay changed.")
    else:
        overlap_rows = sum(1 for _ in _iter_jsonl(overlap_path))
        overlap_by_dataset = dict(audit.get("public_overlap_rows_by_dataset") or {})
    if (
        int(audit.get("public_overlap_rows", -1)) != overlap_rows
        or audit.get("public_overlap_rows_by_dataset") != overlap_by_dataset
        or (audit.get("training_ready") is True) != (overlap_rows == 0)
        or audit.get("admission_state")
        != ("normalized_pcm_exact_public_clear" if overlap_rows == 0 else "public_overlap_detected")
    ):
        raise ValueError("Stage211 base public PCM overlap totals changed.")
    return audit


def finalize_audit(
    *,
    base: dict[str, Any],
    public_manifests: dict[str, Path],
    output_root: Path,
    expected_public_rows: dict[str, int] | None,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "audit_receipt.json"
    if receipt_path.is_file():
        return validate_audit_receipt(receipt_path, verify_overlap_replay=True)
    part_receipts = _part_receipts(base, output_root)
    public_by_fingerprint, public_receipts = _load_public_fingerprint_map(
        public_manifests=public_manifests,
        output_root=output_root,
        expected_rows=expected_public_rows,
    )
    rendered, overlap_rows, overlap_by_dataset = _overlap_payload(
        part_receipts=part_receipts,
        public_by_fingerprint=public_by_fingerprint,
    )
    overlap_path = output_root / "public_overlap_exclusions.jsonl"
    _immutable_bytes(overlap_path, rendered)
    source_counts: Counter[str] = Counter()
    source_duration_ms: Counter[str] = Counter()
    for record in part_receipts:
        source_counts.update(record["source_counts"])
        source_duration_ms.update(record["source_duration_ms"])
    scanned_rows = sum(int(record["rows"]) for record in part_receipts)
    scanned_duration_ms = sum(int(record["duration_ms"]) for record in part_receipts)
    training_ready = overlap_rows == 0
    audit = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": AUDIT_ARTIFACT,
        "complete": True,
        "training_ready": training_ready,
        "admission_state": (
            "normalized_pcm_exact_public_clear"
            if training_ready
            else "public_overlap_detected"
        ),
        "comparison_mode": "normalized_pcm_exact",
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "near_duplicate_complete": False,
        "decode_failures": 0,
        "base_inventory_path": base["inventory_path"],
        "base_inventory_sha256": base["inventory_sha256"],
        "base_bucket_manifest_path": base["bucket_manifest_path"],
        "base_bucket_manifest_sha256": base["bucket_manifest_sha256"],
        "scanned_rows": scanned_rows,
        "scanned_duration_ms": scanned_duration_ms,
        "scanned_hours": scanned_duration_ms / 3_600_000.0,
        "scanned_counts_by_source": dict(sorted(source_counts.items())),
        "scanned_duration_ms_by_source": dict(sorted(source_duration_ms.items())),
        "public_overlap_rows": overlap_rows,
        "public_overlap_rows_by_dataset": overlap_by_dataset,
        "part_fingerprint_receipts": part_receipts,
        "public_fingerprint_receipts": public_receipts,
        "overlap_output": {
            "path": str(overlap_path.resolve()),
            "size_bytes": overlap_path.stat().st_size,
            "sha256": _sha256(overlap_path),
        },
    }
    _immutable_json(receipt_path, audit)
    return validate_audit_receipt(receipt_path, verify_overlap_replay=True)


def _parse_public_manifest(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_PUBLIC_MANIFESTS)
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--public-manifest must use DATASET=/absolute/path.jsonl")
        parsed[name] = Path(raw_path).expanduser().resolve()
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Stage211 base supplemental audio against public evaluation PCM."
    )
    parser.add_argument("--base-inventory", type=Path, default=DEFAULT_BASE_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--public-manifest",
        action="append",
        default=[],
        metavar="DATASET=PATH",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("fingerprint-worker")
    worker.add_argument("--worker-index", type=int, default=0)
    worker.add_argument("--num-workers", type=int, default=1)
    worker.add_argument("--max-parts", type=int, default=None)
    subparsers.add_parser("public")
    subparsers.add_parser("finalize")
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--worker-index", type=int, default=0)
    all_parser.add_argument("--num-workers", type=int, default=1)
    all_parser.add_argument("--max-parts", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    public_manifests = _parse_public_manifest(args.public_manifest)
    expected_rows = (
        EXPECTED_PUBLIC_ROWS if public_manifests == DEFAULT_PUBLIC_MANIFESTS else None
    )
    base = validate_base_inventory(args.base_inventory)
    if args.command in {"public", "all"}:
        build_public_fingerprints(
            public_manifests=public_manifests,
            output_root=output_root,
            expected_rows=expected_rows,
        )
    if args.command in {"fingerprint-worker", "all"}:
        result = run_fingerprint_worker(
            base=base,
            output_root=output_root,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
            max_parts=args.max_parts,
        )
        print(f"[stage211-base-public-pcm] fingerprint_result={result}", flush=True)
    if args.command in {"finalize", "all"}:
        audit = finalize_audit(
            base=base,
            public_manifests=public_manifests,
            output_root=output_root,
            expected_public_rows=expected_rows,
        )
        print(
            f"[stage211-base-public-pcm] complete rows={audit['scanned_rows']} "
            f"overlap={audit['public_overlap_rows']} "
            f"training_ready={audit['training_ready']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
