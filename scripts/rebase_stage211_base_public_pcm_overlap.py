#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.audit_stage211_base_public_pcm_overlap import (
        EXPECTED_PUBLIC_ROWS,
        _archive_paths,
        _immutable_json,
        _index_paths,
        _sha256,
        build_public_fingerprints,
        finalize_audit,
        rebase_archive_fingerprints,
        validate_audit_receipt,
        validate_base_inventory,
        validate_location_index,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from audit_stage211_base_public_pcm_overlap import (
        EXPECTED_PUBLIC_ROWS,
        _archive_paths,
        _immutable_json,
        _index_paths,
        _sha256,
        build_public_fingerprints,
        finalize_audit,
        rebase_archive_fingerprints,
        validate_audit_receipt,
        validate_base_inventory,
        validate_location_index,
    )


DEFAULT_BASE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_supplemental_natural_v1/supplemental_inventory.json"
)
DEFAULT_SOURCE_AUDIT = (
    Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_v1/audit_receipt.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_v2"
DEFAULT_CLEAN_MANIFEST_ROOT = Path.home() / "rwkvasr_eval/stage211_public_clean_v2/manifests"
DEFAULT_PUBLIC_MANIFESTS = {
    dataset: DEFAULT_CLEAN_MANIFEST_ROOT / f"{dataset}.jsonl"
    for dataset in EXPECTED_PUBLIC_ROWS
}
REBASE_ARTIFACT = "stage211_base_public_pcm_fingerprint_rebase"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _receipt_set_sha256(root: Path, archive_count: int) -> str:
    digest = hashlib.sha256()
    for archive_index in range(archive_count):
        _, receipt_path = _archive_paths(root, archive_index)
        digest.update(bytes.fromhex(_sha256(receipt_path)))
    return digest.hexdigest()


def validate_rebase_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 base PCM rebase receipt")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": REBASE_ARTIFACT,
        "complete": True,
        "audio_redecoded": False,
        "reuse_mode": "same_filesystem_hardlink_plus_rebound_receipts",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 base PCM rebase receipt contract changed.")
    source_audit_path = Path(str(receipt.get("source_audit_receipt_path") or "")).resolve()
    destination_audit_path = Path(
        str(receipt.get("destination_audit_receipt_path") or "")
    ).resolve()
    if (
        _sha256(source_audit_path) != receipt.get("source_audit_receipt_sha256")
        or _sha256(destination_audit_path)
        != receipt.get("destination_audit_receipt_sha256")
    ):
        raise ValueError("Stage211 base PCM rebase audit binding changed.")
    source_audit = validate_audit_receipt(source_audit_path, verify_overlap_replay=True)
    destination_audit = validate_audit_receipt(
        destination_audit_path,
        require_training_ready=True,
        verify_overlap_replay=True,
    )
    if (
        source_audit.get("base_inventory_sha256")
        != destination_audit.get("base_inventory_sha256")
        or int(destination_audit.get("public_overlap_rows", -1)) != 0
    ):
        raise ValueError("Stage211 base PCM rebase changed base coverage.")

    source_root = source_audit_path.parent
    destination_root = destination_audit_path.parent
    archive_count = int(receipt.get("archive_count", -1))
    if archive_count != len(source_audit.get("archive_fingerprint_receipts") or []):
        raise ValueError("Stage211 base PCM rebase archive count changed.")
    source_database, _ = _index_paths(source_root)
    destination_database, _ = _index_paths(destination_root)
    if not source_database.samefile(destination_database):
        raise ValueError("Stage211 base PCM rebased location index is not the source hardlink.")
    for archive_index in range(archive_count):
        source_fingerprint, _ = _archive_paths(source_root, archive_index)
        destination_fingerprint, _ = _archive_paths(destination_root, archive_index)
        if not source_fingerprint.samefile(destination_fingerprint):
            raise ValueError(
                "Stage211 base PCM rebased fingerprint is not the source hardlink: "
                f"{archive_index}"
            )
    if (
        _receipt_set_sha256(source_root, archive_count)
        != receipt.get("source_archive_receipt_set_sha256")
        or _receipt_set_sha256(destination_root, archive_count)
        != receipt.get("destination_archive_receipt_set_sha256")
    ):
        raise ValueError("Stage211 base PCM rebased archive receipt set changed.")
    return receipt


def _validate_reusable_rebase_receipt(
    *,
    receipt_path: Path,
    base_inventory: Path,
    source_audit_receipt: Path,
    public_manifests: dict[str, Path],
    expected_public_rows: dict[str, int],
) -> dict[str, Any]:
    receipt = validate_rebase_receipt(receipt_path)
    if (
        receipt.get("source_audit_receipt_path") != str(source_audit_receipt)
        or receipt.get("source_audit_receipt_sha256") != _sha256(source_audit_receipt)
    ):
        raise ValueError(
            "Stage211 existing base PCM rebase does not bind the requested source audit."
        )
    source_audit = _load_json(
        source_audit_receipt,
        label="Stage211 requested base PCM source audit",
    )
    if (
        source_audit.get("base_inventory_path") != str(base_inventory)
        or source_audit.get("base_inventory_sha256") != _sha256(base_inventory)
    ):
        raise ValueError(
            "Stage211 existing base PCM rebase does not bind the requested base inventory."
        )
    expected_rows = {str(key): int(value) for key, value in expected_public_rows.items()}
    if receipt.get("public_manifest_rows") != dict(sorted(expected_rows.items())):
        raise ValueError(
            "Stage211 existing base PCM rebase does not bind the requested public row map."
        )
    destination_audit = _load_json(
        Path(str(receipt.get("destination_audit_receipt_path") or "")).resolve(),
        label="Stage211 rebased destination audit",
    )
    records = destination_audit.get("public_fingerprint_receipts")
    by_dataset = {
        str(record.get("dataset")): record
        for record in records or []
        if isinstance(record, dict)
    }
    if (
        not isinstance(records, list)
        or len(records) != len(expected_rows)
        or set(public_manifests) != set(expected_rows)
        or set(by_dataset) != set(expected_rows)
    ):
        raise ValueError(
            "Stage211 existing base PCM rebase public manifest set does not match the request."
        )
    for dataset, manifest_path in public_manifests.items():
        record = by_dataset[dataset]
        if (
            record.get("manifest_path") != str(manifest_path)
            or record.get("manifest_sha256") != _sha256(manifest_path)
            or int(record.get("rows", -1)) != expected_rows[dataset]
        ):
            raise ValueError(
                "Stage211 existing base PCM rebase does not bind the requested "
                f"public manifest: {dataset}."
            )
    return receipt


def rebase_and_finalize(
    *,
    base_inventory: Path,
    source_audit_receipt: Path,
    output_root: Path,
    public_manifests: dict[str, Path],
    expected_public_rows: dict[str, int],
) -> dict[str, Any]:
    base_inventory = base_inventory.expanduser().resolve()
    source_audit_receipt = source_audit_receipt.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    public_manifests = {
        str(dataset): path.expanduser().resolve()
        for dataset, path in public_manifests.items()
    }
    receipt_path = output_root / "rebase_receipt.json"
    if receipt_path.is_file():
        return _validate_reusable_rebase_receipt(
            receipt_path=receipt_path,
            base_inventory=base_inventory,
            source_audit_receipt=source_audit_receipt,
            public_manifests=public_manifests,
            expected_public_rows=expected_public_rows,
        )
    base = validate_base_inventory(base_inventory)
    rebase = rebase_archive_fingerprints(
        base=base,
        source_audit_receipt=source_audit_receipt,
        output_root=output_root,
    )
    build_public_fingerprints(
        public_manifests=public_manifests,
        output_root=output_root,
        expected_rows=expected_public_rows,
    )
    location_index = validate_location_index(output_root, base=base)
    audit = finalize_audit(
        base=base,
        public_manifests=public_manifests,
        output_root=output_root,
        expected_public_rows=expected_public_rows,
        location_index=location_index,
    )
    if audit.get("training_ready") is not True:
        raise ValueError("Stage211 corrected public suite overlaps the base supplemental pool.")
    destination_audit = output_root / "audit_receipt.json"
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": REBASE_ARTIFACT,
        "complete": True,
        "audio_redecoded": False,
        "reuse_mode": "same_filesystem_hardlink_plus_rebound_receipts",
        **rebase,
        "destination_audit_receipt_path": str(destination_audit.resolve()),
        "destination_audit_receipt_sha256": _sha256(destination_audit),
        "public_manifest_rows": dict(sorted(expected_public_rows.items())),
    }
    _immutable_json(receipt_path, receipt)
    return validate_rebase_receipt(receipt_path)


def _parse_public_manifests(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_PUBLIC_MANIFESTS)
    parsed: dict[str, Path] = {}
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        if not separator or not dataset or not raw_path:
            raise ValueError("--public-manifest must use DATASET=/absolute/path.jsonl")
        parsed[dataset] = Path(raw_path).expanduser().resolve()
    if set(parsed) != set(EXPECTED_PUBLIC_ROWS):
        raise ValueError("Stage211 corrected public manifest dataset set changed.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind completed Stage211 base PCM fingerprints to the corrected public suite "
            "without decoding base audio again."
        )
    )
    parser.add_argument("--base-inventory", type=Path, default=DEFAULT_BASE_INVENTORY)
    parser.add_argument("--source-audit-receipt", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--public-manifest", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = rebase_and_finalize(
        base_inventory=args.base_inventory,
        source_audit_receipt=args.source_audit_receipt,
        output_root=args.output_root,
        public_manifests=_parse_public_manifests(args.public_manifest),
        expected_public_rows=EXPECTED_PUBLIC_ROWS,
    )
    print(
        "[stage211-base-public-pcm-rebase] "
        f"archives={receipt['archive_count']} audio_redecoded={receipt['audio_redecoded']} "
        f"receipt={args.output_root.expanduser().resolve() / 'rebase_receipt.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
