#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MATERIALIZED_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_social_vad_materialized_v1/materialized_inventory.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_social_vad_filtered_v1"
MATERIALIZED_INVENTORY_ARTIFACT = "stage211_social_vad_materialized_inventory"
SOURCE_FINGERPRINT_ARTIFACT = "stage211_social_pcm_source_fingerprints"
PROGRESS_ARTIFACT = "stage211_social_pcm_progress"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return raw


def _source_rows(inventory_path: Path) -> list[dict[str, Any]]:
    inventory = _load_json(
        inventory_path,
        label="Stage211 social materialized inventory",
    )
    expected = {
        "schema_version": 1,
        "artifact": MATERIALIZED_INVENTORY_ARTIFACT,
        "complete": True,
        "training_ready": False,
    }
    if any(inventory.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 social materialized inventory contract mismatch.")
    rows = inventory.get("source_receipts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage211 social materialized source receipts are missing.")
    if [int(row.get("source_index", -1)) for row in rows] != list(range(len(rows))):
        raise ValueError("Stage211 social materialized source order is not contiguous.")
    for row in rows:
        segments = int(row.get("segments", -1))
        duration_ms = int(row.get("duration_ms", -1))
        tar_size_bytes = int(row.get("tar_size_bytes") or 0)
        empty_source = segments == 0
        invalid_empty = empty_source and (
            duration_ms != 0
            or row.get("tar_path") is not None
            or row.get("tar_sha256") is not None
            or row.get("tar_size_bytes") is not None
        )
        invalid_materialized = not empty_source and (
            duration_ms <= 0
            or tar_size_bytes <= 0
            or not row.get("tar_path")
            or len(str(row.get("tar_sha256") or "")) != 64
        )
        if (
            segments < 0
            or duration_ms < 0
            or len(str(row.get("sha256") or "")) != 64
            or invalid_empty
            or invalid_materialized
        ):
            raise ValueError(
                "Stage211 social materialized source metadata is invalid: "
                f"{row.get('source_index')}"
            )
    if sum(int(row["segments"]) for row in rows) != int(inventory.get("selected_rows", -1)):
        raise ValueError("Stage211 social materialized row total changed.")
    return rows


def _completed_receipts(
    receipt_dir: Path,
    *,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    indices: list[int] = []
    for path in sorted(receipt_dir.glob("source_*.receipt.json")):
        stem = path.name.removeprefix("source_").removesuffix(".receipt.json")
        if not stem.isdigit():
            raise ValueError(f"Stage211 social PCM receipt name is invalid: {path}")
        source_index = int(stem)
        if source_index >= len(sources):
            raise ValueError(f"Stage211 social PCM receipt is out of range: {path}")
        source = sources[source_index]
        receipt = _load_json(path, label="Stage211 social PCM source receipt")
        expected_part_path = receipt_dir / f"source_{source_index:06d}.jsonl"
        expected_tar_path = (
            str(Path(str(source["tar_path"])).resolve())
            if source.get("tar_path") is not None
            else None
        )
        expected = {
            "schema_version": 1,
            "artifact": SOURCE_FINGERPRINT_ARTIFACT,
            "complete": True,
            "source_index": source_index,
            "source_receipt_path": str(Path(str(source["path"])).resolve()),
            "source_receipt_sha256": source["sha256"],
            "tar_path": expected_tar_path,
            "tar_sha256": source["tar_sha256"],
            "rows": int(source["segments"]),
            "duration_ms": int(source["duration_ms"]),
            "part_path": str(expected_part_path.resolve()),
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Stage211 social PCM source receipt changed: {path}")
        part_size = int(receipt.get("part_size_bytes", -1))
        expected_part_size_is_valid = (
            part_size == 0 if int(source["segments"]) == 0 else part_size > 0
        )
        if (
            not expected_part_path.is_file()
            or not expected_part_size_is_valid
            or expected_part_path.stat().st_size != part_size
            or len(str(receipt.get("part_sha256") or "")) != 64
        ):
            raise ValueError(f"Stage211 social PCM fingerprint part changed: {path}")
        indices.append(source_index)
        receipts.append(receipt)
    if indices != list(range(len(indices))):
        raise ValueError(
            "Stage211 social PCM completed receipts are not the exact contiguous prefix."
        )
    return receipts


def _totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "sources": len(rows),
        "bytes": sum(int(row.get("tar_size_bytes") or 0) for row in rows),
        "rows": sum(int(row["segments"]) for row in rows),
        "duration_ms": sum(int(row["duration_ms"]) for row in rows),
    }


def _with_progress(covered: dict[str, int], total: dict[str, int]) -> dict[str, Any]:
    return {
        key: {
            "covered": int(covered[key]),
            "total": int(total[key]),
            "remaining": int(total[key] - covered[key]),
            "percent": 100.0 * float(covered[key]) / float(total[key]),
        }
        for key in ("sources", "bytes", "rows", "duration_ms")
    }


def build_progress(
    materialized_inventory: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    inventory_path = Path(materialized_inventory).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    sources = _source_rows(inventory_path)
    receipt_dir = root / "source_fingerprints"
    receipts = _completed_receipts(receipt_dir, sources=sources)
    completed_count = len(receipts)
    covered = _totals(sources[:completed_count])
    total = _totals(sources)
    next_source = sources[completed_count] if completed_count < len(sources) else None
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": PROGRESS_ARTIFACT,
        "observational": True,
        "complete": completed_count == len(sources),
        "receipts_contiguous": True,
        "materialized_inventory_path": str(inventory_path),
        "output_root": str(root),
        "receipt_directory": str(receipt_dir.resolve()),
        "latest_completed_source_index": completed_count - 1,
        "progress": _with_progress(covered, total),
        "covered_hours": covered["duration_ms"] / 3_600_000.0,
        "total_hours": total["duration_ms"] / 3_600_000.0,
        "remaining_hours": (total["duration_ms"] - covered["duration_ms"]) / 3_600_000.0,
        "next_source": (
            {
                "source_index": int(next_source["source_index"]),
                "tar_size_bytes": int(next_source.get("tar_size_bytes") or 0),
                "rows": int(next_source["segments"]),
                "duration_ms": int(next_source["duration_ms"]),
            }
            if next_source is not None
            else None
        ),
    }


def render_text(progress: dict[str, Any]) -> str:
    values = progress["progress"]
    lines = [
        "stage211_social_pcm_progress "
        f"complete={str(progress['complete']).lower()} "
        f"receipts_contiguous={str(progress['receipts_contiguous']).lower()} "
        f"latest_source_index={progress['latest_completed_source_index']}",
        "sources={covered}/{total} percent={percent:.4f}".format(**values["sources"]),
        "bytes={covered}/{total} percent={percent:.4f}".format(**values["bytes"]),
        "rows={covered}/{total} percent={percent:.4f}".format(**values["rows"]),
        "duration_hours={covered:.6f}/{total:.6f} percent={percent:.4f}".format(
            covered=progress["covered_hours"],
            total=progress["total_hours"],
            percent=values["duration_ms"]["percent"],
        ),
    ]
    next_source = progress["next_source"]
    if next_source is None:
        lines.append("next_source=none")
    else:
        lines.append(
            "next_source={source_index} bytes={tar_size_bytes} rows={rows} "
            "duration_ms={duration_ms}".format(**next_source)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report read-only weighted progress for Stage211 social PCM filtering."
    )
    parser.add_argument(
        "--materialized-inventory",
        type=Path,
        default=DEFAULT_MATERIALIZED_INVENTORY,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    progress = build_progress(args.materialized_inventory, args.output_root)
    if args.format == "json":
        print(json.dumps(progress, indent=2, sort_keys=True))
    else:
        print(render_text(progress), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
