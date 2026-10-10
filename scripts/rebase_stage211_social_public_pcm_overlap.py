#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.filter_stage211_social_pcm_overlap import (
        EXPECTED_PUBLIC_ROWS,
        _immutable_hardlink,
        _immutable_json,
        _public_fingerprint_paths,
        _sha256,
        _source_fingerprint_paths,
        _validate_public_fingerprint_receipt,
        build_public_fingerprints,
        finalize_filtered_inventory,
        rebase_social_source_fingerprints,
        validate_filtered_inventory,
        validate_materialized_inventory,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from filter_stage211_social_pcm_overlap import (
        EXPECTED_PUBLIC_ROWS,
        _immutable_hardlink,
        _immutable_json,
        _public_fingerprint_paths,
        _sha256,
        _source_fingerprint_paths,
        _validate_public_fingerprint_receipt,
        build_public_fingerprints,
        finalize_filtered_inventory,
        rebase_social_source_fingerprints,
        validate_filtered_inventory,
        validate_materialized_inventory,
    )


DEFAULT_SOURCE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_social_vad_filtered_v1/filtered_inventory.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_social_vad_filtered_v2"
DEFAULT_PUBLIC_FINGERPRINT_SOURCE_ROOT = (
    Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_v2"
)
DEFAULT_CLEAN_MANIFEST_ROOT = Path.home() / "rwkvasr_eval/stage211_public_clean_v2/manifests"
DEFAULT_PUBLIC_MANIFESTS = {
    dataset: DEFAULT_CLEAN_MANIFEST_ROOT / f"{dataset}.jsonl"
    for dataset in EXPECTED_PUBLIC_ROWS
}
REBASE_ARTIFACT = "stage211_social_public_pcm_fingerprint_rebase"


def rebind_corrected_public_fingerprints(
    *,
    source_root: Path,
    output_root: Path,
    public_manifests: dict[str, Path],
    expected_public_rows: dict[str, int],
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root:
        raise ValueError(
            "Stage211 corrected-public fingerprint source and destination must differ."
        )
    if set(public_manifests) != set(expected_public_rows):
        raise ValueError("Stage211 corrected-public fingerprint reuse dataset set changed.")
    statuses: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for dataset, manifest_path in sorted(public_manifests.items()):
        manifest_path = manifest_path.expanduser().resolve()
        source_part, source_receipt_path = _public_fingerprint_paths(source_root, dataset)
        source_receipt = _validate_public_fingerprint_receipt(
            source_receipt_path,
            dataset=dataset,
            manifest_path=manifest_path,
            expected_rows=int(expected_public_rows[dataset]),
        )
        destination_part, destination_receipt_path = _public_fingerprint_paths(
            output_root,
            dataset,
        )
        statuses[dataset] = _immutable_hardlink(source_part, destination_part)
        destination_receipt = {
            **source_receipt,
            "part_path": str(destination_part.resolve()),
        }
        _immutable_json(destination_receipt_path, destination_receipt)
        _validate_public_fingerprint_receipt(
            destination_receipt_path,
            dataset=dataset,
            manifest_path=manifest_path,
            expected_rows=int(expected_public_rows[dataset]),
        )
        if not source_part.samefile(destination_part):
            raise ValueError(
                f"Stage211 corrected-public fingerprint hardlink changed: {dataset}"
            )
        records.append(
            {
                "dataset": dataset,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "rows": int(expected_public_rows[dataset]),
                "source_receipt_path": str(source_receipt_path.resolve()),
                "source_receipt_sha256": _sha256(source_receipt_path),
                "destination_receipt_path": str(destination_receipt_path.resolve()),
                "destination_receipt_sha256": _sha256(destination_receipt_path),
                "part_sha256": source_receipt["part_sha256"],
            }
        )
    return {
        "mode": "same_filesystem_hardlink_rebound_receipts_v1",
        "source_root": str(source_root),
        "destination_root": str(output_root),
        "datasets": records,
        "hardlink_status": dict(sorted(statuses.items())),
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _receipt_set_sha256(root: Path, source_count: int) -> str:
    digest = hashlib.sha256()
    for source_index in range(source_count):
        _, receipt_path = _source_fingerprint_paths(root, source_index)
        digest.update(bytes.fromhex(_sha256(receipt_path)))
    return digest.hexdigest()


def validate_rebase_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 social PCM rebase receipt")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": REBASE_ARTIFACT,
        "complete": True,
        "audio_redecoded": False,
        "reuse_mode": "same_filesystem_hardlink_plus_rebound_receipts",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 social PCM rebase receipt contract changed.")
    source_inventory_path = Path(
        str(receipt.get("source_filtered_inventory_path") or "")
    ).resolve()
    destination_inventory_path = Path(
        str(receipt.get("destination_filtered_inventory_path") or "")
    ).resolve()
    if (
        _sha256(source_inventory_path) != receipt.get("source_filtered_inventory_sha256")
        or _sha256(destination_inventory_path)
        != receipt.get("destination_filtered_inventory_sha256")
    ):
        raise ValueError("Stage211 social PCM rebase inventory binding changed.")
    source = validate_filtered_inventory(source_inventory_path, verify_part_sha256=True)
    destination = validate_filtered_inventory(
        destination_inventory_path,
        verify_part_sha256=True,
    )
    if source["inventory"].get("materialized_inventory_sha256") != destination[
        "inventory"
    ].get("materialized_inventory_sha256"):
        raise ValueError("Stage211 social PCM rebase changed materialized coverage.")

    source_root = source_inventory_path.parent
    destination_root = destination_inventory_path.parent
    source_count = int(receipt.get("source_count", -1))
    source_records = source["inventory"].get("source_fingerprint_receipts") or []
    if source_count != len(source_records):
        raise ValueError("Stage211 social PCM rebase source count changed.")
    for source_index in range(source_count):
        source_part, _ = _source_fingerprint_paths(source_root, source_index)
        destination_part, _ = _source_fingerprint_paths(destination_root, source_index)
        if not source_part.samefile(destination_part):
            raise ValueError(
                "Stage211 social PCM rebased fingerprint is not the source hardlink: "
                f"{source_index}"
            )
    if (
        _receipt_set_sha256(source_root, source_count)
        != receipt.get("source_receipt_set_sha256")
        or _receipt_set_sha256(destination_root, source_count)
        != receipt.get("destination_receipt_set_sha256")
    ):
        raise ValueError("Stage211 social PCM rebased receipt set changed.")
    public_manifest_rows = receipt.get("public_manifest_rows")
    if (
        not isinstance(public_manifest_rows, dict)
        or int(destination["inventory"].get("public_overlap", {}).get("public_rows", -1))
        != sum(int(value) for value in public_manifest_rows.values())
    ):
        raise ValueError("Stage211 social PCM corrected public coverage changed.")
    public_reuse = receipt.get("corrected_public_fingerprint_reuse")
    if public_reuse is not None:
        if (
            not isinstance(public_reuse, dict)
            or public_reuse.get("mode")
            != "same_filesystem_hardlink_rebound_receipts_v1"
            or Path(str(public_reuse.get("destination_root") or "")).resolve()
            != destination_root
        ):
            raise ValueError("Stage211 corrected-public fingerprint reuse contract changed.")
        source_public_root = Path(str(public_reuse.get("source_root") or "")).resolve()
        records = public_reuse.get("datasets")
        if not isinstance(records, list) or len(records) != len(public_manifest_rows):
            raise ValueError("Stage211 corrected-public fingerprint reuse coverage changed.")
        by_dataset = {
            str(record.get("dataset")): record
            for record in records
            if isinstance(record, dict)
        }
        if set(by_dataset) != set(public_manifest_rows):
            raise ValueError("Stage211 corrected-public fingerprint reuse dataset set changed.")
        for dataset, expected_rows in public_manifest_rows.items():
            record = by_dataset[dataset]
            manifest_path = Path(str(record.get("manifest_path") or "")).resolve()
            source_part, source_receipt_path = _public_fingerprint_paths(
                source_public_root,
                dataset,
            )
            destination_part, destination_receipt_path = _public_fingerprint_paths(
                destination_root,
                dataset,
            )
            source_receipt = _validate_public_fingerprint_receipt(
                source_receipt_path,
                dataset=dataset,
                manifest_path=manifest_path,
                expected_rows=int(expected_rows),
            )
            _validate_public_fingerprint_receipt(
                destination_receipt_path,
                dataset=dataset,
                manifest_path=manifest_path,
                expected_rows=int(expected_rows),
            )
            if (
                record.get("manifest_sha256") != _sha256(manifest_path)
                or record.get("source_receipt_path") != str(source_receipt_path.resolve())
                or record.get("source_receipt_sha256") != _sha256(source_receipt_path)
                or record.get("destination_receipt_path")
                != str(destination_receipt_path.resolve())
                or record.get("destination_receipt_sha256")
                != _sha256(destination_receipt_path)
                or record.get("part_sha256") != source_receipt.get("part_sha256")
                or not source_part.samefile(destination_part)
            ):
                raise ValueError(
                    f"Stage211 corrected-public fingerprint reuse changed: {dataset}"
                )
    return receipt


def _validate_reusable_rebase_receipt(
    *,
    receipt_path: Path,
    source_filtered_inventory: Path,
    public_manifests: dict[str, Path],
    expected_public_rows: dict[str, int],
    public_fingerprint_source_root: Path | None,
) -> dict[str, Any]:
    receipt = validate_rebase_receipt(receipt_path)
    if (
        receipt.get("source_filtered_inventory_path") != str(source_filtered_inventory)
        or receipt.get("source_filtered_inventory_sha256")
        != _sha256(source_filtered_inventory)
    ):
        raise ValueError(
            "Stage211 existing social PCM rebase does not bind the requested source inventory."
        )
    expected_rows = {str(key): int(value) for key, value in expected_public_rows.items()}
    if receipt.get("public_manifest_rows") != dict(sorted(expected_rows.items())):
        raise ValueError(
            "Stage211 existing social PCM rebase does not bind the requested public row map."
        )
    public_reuse = receipt.get("corrected_public_fingerprint_reuse")
    expected_reuse_root = (
        None
        if public_fingerprint_source_root is None
        else str(public_fingerprint_source_root.expanduser().resolve())
    )
    recorded_reuse_root = (
        None if public_reuse is None else str(public_reuse.get("source_root") or "")
    )
    if recorded_reuse_root != expected_reuse_root:
        raise ValueError(
            "Stage211 existing social PCM rebase corrected-public reuse source changed."
        )
    destination_inventory = _load_json(
        Path(str(receipt.get("destination_filtered_inventory_path") or "")).resolve(),
        label="Stage211 rebased destination social inventory",
    )
    records = destination_inventory.get("public_fingerprint_receipts")
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
            "Stage211 existing social PCM rebase public manifest set does not match the request."
        )
    for dataset, manifest_path in public_manifests.items():
        record = by_dataset[dataset]
        if (
            record.get("manifest_path") != str(manifest_path)
            or record.get("manifest_sha256") != _sha256(manifest_path)
            or int(record.get("rows", -1)) != expected_rows[dataset]
        ):
            raise ValueError(
                "Stage211 existing social PCM rebase does not bind the requested "
                f"public manifest: {dataset}."
            )
    return receipt


def rebase_and_finalize(
    *,
    source_filtered_inventory: Path,
    output_root: Path,
    public_manifests: dict[str, Path],
    expected_public_rows: dict[str, int],
    public_fingerprint_source_root: Path | None = None,
) -> dict[str, Any]:
    source_filtered_inventory = source_filtered_inventory.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    public_manifests = {
        str(dataset): path.expanduser().resolve()
        for dataset, path in public_manifests.items()
    }
    receipt_path = output_root / "rebase_receipt.json"
    if receipt_path.is_file():
        return _validate_reusable_rebase_receipt(
            receipt_path=receipt_path,
            source_filtered_inventory=source_filtered_inventory,
            public_manifests=public_manifests,
            expected_public_rows=expected_public_rows,
            public_fingerprint_source_root=public_fingerprint_source_root,
        )
    source = validate_filtered_inventory(
        source_filtered_inventory,
        verify_part_sha256=True,
    )
    materialized = validate_materialized_inventory(
        Path(str(source["inventory"]["materialized_inventory_path"]))
    )
    rebase = rebase_social_source_fingerprints(
        materialized=materialized,
        source_root=source_filtered_inventory.parent,
        output_root=output_root,
    )
    public_reuse = None
    if public_fingerprint_source_root is None:
        build_public_fingerprints(
            public_manifests=public_manifests,
            output_root=output_root,
            expected_rows=expected_public_rows,
        )
    else:
        public_reuse = rebind_corrected_public_fingerprints(
            source_root=public_fingerprint_source_root,
            output_root=output_root,
            public_manifests=public_manifests,
            expected_public_rows=expected_public_rows,
        )
    destination = finalize_filtered_inventory(
        materialized=materialized,
        public_manifests=public_manifests,
        output_root=output_root,
        expected_public_rows=expected_public_rows,
    )
    destination_inventory = output_root / "filtered_inventory.json"
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": REBASE_ARTIFACT,
        "complete": True,
        "audio_redecoded": False,
        "reuse_mode": "same_filesystem_hardlink_plus_rebound_receipts",
        **rebase,
        "source_filtered_inventory_path": str(source_filtered_inventory),
        "source_filtered_inventory_sha256": _sha256(source_filtered_inventory),
        "destination_filtered_inventory_path": str(destination_inventory.resolve()),
        "destination_filtered_inventory_sha256": _sha256(destination_inventory),
        "destination_selected_rows": int(destination["selected_rows"]),
        "destination_selected_hours": float(destination["selected_hours"]),
        "public_manifest_rows": dict(sorted(expected_public_rows.items())),
    }
    if public_reuse is not None:
        receipt["corrected_public_fingerprint_reuse"] = public_reuse
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
            "Rebind completed Stage211 social PCM fingerprints to the corrected public "
            "suite without decoding social audio again."
        )
    )
    parser.add_argument(
        "--source-filtered-inventory",
        type=Path,
        default=DEFAULT_SOURCE_INVENTORY,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--public-manifest", action="append", default=[])
    parser.add_argument(
        "--public-fingerprint-source-root",
        type=Path,
        default=DEFAULT_PUBLIC_FINGERPRINT_SOURCE_ROOT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = rebase_and_finalize(
        source_filtered_inventory=args.source_filtered_inventory,
        output_root=args.output_root,
        public_manifests=_parse_public_manifests(args.public_manifest),
        expected_public_rows=EXPECTED_PUBLIC_ROWS,
        public_fingerprint_source_root=args.public_fingerprint_source_root,
    )
    print(
        "[stage211-social-public-pcm-rebase] "
        f"sources={receipt['source_count']} audio_redecoded={receipt['audio_redecoded']} "
        f"receipt={args.output_root.expanduser().resolve() / 'rebase_receipt.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
