#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_v1"
ARCHIVE_RECEIPT_ARTIFACT = "stage211_base_public_pcm_archive_fingerprints"
PROGRESS_ARTIFACT = "stage211_base_public_pcm_progress"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} is not an object: {path}")
    return raw


def _archive_rows(database_path: Path) -> list[dict[str, Any]]:
    if not database_path.is_file():
        raise ValueError(f"Stage211 base PCM location index is missing: {database_path}")
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT archive_index, storage_kind, shard_path, source_dataset, rows, "
                "duration_ms, archive_size_bytes, archive_mtime_ns, archive_sha256 "
                "FROM archives ORDER BY archive_index"
            )
        ]
    finally:
        connection.close()
    if not rows:
        raise ValueError("Stage211 base PCM location index contains no archives.")
    if [int(row["archive_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Stage211 base PCM location-index archive order is not contiguous.")
    for row in rows:
        if (
            any(
                int(row[key]) <= 0
                for key in ("rows", "duration_ms", "archive_size_bytes", "archive_mtime_ns")
            )
            or len(str(row["archive_sha256"])) != 64
        ):
            raise ValueError(
                "Stage211 base PCM location-index archive metadata is invalid: "
                f"{row['archive_index']}"
            )
    return rows


def _completed_receipts(
    receipt_dir: Path,
    *,
    archives: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    receipt_paths = sorted(receipt_dir.glob("archive_*.receipt.json"))
    receipts: list[dict[str, Any]] = []
    indices: list[int] = []
    for path in receipt_paths:
        stem = path.name.removeprefix("archive_").removesuffix(".receipt.json")
        if not stem.isdigit():
            raise ValueError(f"Stage211 base PCM archive receipt name is invalid: {path}")
        index = int(stem)
        if index >= len(archives):
            raise ValueError(f"Stage211 base PCM archive receipt is out of range: {path}")
        receipt = _load_json(path, label="Stage211 base PCM archive receipt")
        expected = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": ARCHIVE_RECEIPT_ARTIFACT,
            "complete": True,
            "decode_failures": 0,
            "archive_index": index,
            "storage_kind": archives[index]["storage_kind"],
            "shard_path": archives[index]["shard_path"],
            "source_dataset": archives[index]["source_dataset"],
            "rows": int(archives[index]["rows"]),
            "duration_ms": int(archives[index]["duration_ms"]),
            "archive_size_bytes": int(archives[index]["archive_size_bytes"]),
            "archive_mtime_ns": int(archives[index]["archive_mtime_ns"]),
            "archive_sha256": archives[index]["archive_sha256"],
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Stage211 base PCM archive receipt changed: {path}")
        indices.append(index)
        receipts.append(receipt)
    if indices != list(range(len(indices))):
        raise ValueError(
            "Stage211 base PCM completed receipts are not the exact contiguous prefix."
        )
    return receipts


def _totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "archives": len(rows),
        "bytes": sum(int(row["archive_size_bytes"]) for row in rows),
        "rows": sum(int(row["rows"]) for row in rows),
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
        for key in ("archives", "bytes", "rows", "duration_ms")
    }


def build_progress(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    database_path = root / "manifest_location_index.sqlite"
    archives = _archive_rows(database_path)
    receipts = _completed_receipts(root / "archive_fingerprints", archives=archives)
    completed_count = len(receipts)
    covered_rows = archives[:completed_count]
    remaining_rows = archives[completed_count:]
    total = _totals(archives)
    covered = _totals(covered_rows)
    remaining_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in remaining_rows:
        remaining_by_source[str(row["source_dataset"])].append(row)
    source_records = []
    for source, rows in sorted(
        remaining_by_source.items(), key=lambda item: int(item[1][0]["archive_index"])
    ):
        source_records.append(
            {
                "source_dataset": source,
                "first_archive_index": min(int(row["archive_index"]) for row in rows),
                "last_archive_index": max(int(row["archive_index"]) for row in rows),
                **_totals(rows),
            }
        )
    next_archive = remaining_rows[0] if remaining_rows else None
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": PROGRESS_ARTIFACT,
        "observational": True,
        "complete": completed_count == len(archives),
        "receipts_contiguous": True,
        "output_root": str(root),
        "database_path": str(database_path),
        "receipt_directory": str((root / "archive_fingerprints").resolve()),
        "latest_completed_archive_index": completed_count - 1,
        "progress": _with_progress(covered, total),
        "covered_hours": covered["duration_ms"] / 3_600_000.0,
        "total_hours": total["duration_ms"] / 3_600_000.0,
        "remaining_hours": (total["duration_ms"] - covered["duration_ms"]) / 3_600_000.0,
        "next_archive": (
            {
                "archive_index": int(next_archive["archive_index"]),
                "source_dataset": str(next_archive["source_dataset"]),
                "storage_kind": str(next_archive["storage_kind"]),
                "archive_size_bytes": int(next_archive["archive_size_bytes"]),
                "rows": int(next_archive["rows"]),
                "duration_ms": int(next_archive["duration_ms"]),
            }
            if next_archive is not None
            else None
        ),
        "remaining_by_source": source_records,
    }


def render_text(progress: dict[str, Any]) -> str:
    values = progress["progress"]
    lines = [
        "stage211_base_public_pcm_progress "
        f"complete={str(progress['complete']).lower()} "
        f"receipts_contiguous={str(progress['receipts_contiguous']).lower()} "
        f"latest_archive_index={progress['latest_completed_archive_index']}",
        "archives={covered}/{total} percent={percent:.4f}".format(**values["archives"]),
        "bytes={covered}/{total} percent={percent:.4f}".format(**values["bytes"]),
        "rows={covered}/{total} percent={percent:.4f}".format(**values["rows"]),
        "duration_hours={covered:.6f}/{total:.6f} percent={percent:.4f}".format(
            covered=progress["covered_hours"],
            total=progress["total_hours"],
            percent=values["duration_ms"]["percent"],
        ),
    ]
    next_archive = progress["next_archive"]
    if next_archive is None:
        lines.append("next_archive=none")
    else:
        lines.append(
            "next_archive={archive_index} source={source_dataset} storage={storage_kind} "
            "bytes={archive_size_bytes} rows={rows} duration_ms={duration_ms}".format(
                **next_archive
            )
        )
    for source in progress["remaining_by_source"]:
        lines.append(
            "remaining_source={source_dataset} archives={archives} bytes={bytes} rows={rows} "
            "duration_hours={hours:.6f} first_index={first_archive_index} "
            "last_index={last_archive_index}".format(
                **source,
                hours=int(source["duration_ms"]) / 3_600_000.0,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report read-only weighted progress for the Stage211 base/public PCM audit."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    progress = build_progress(args.output_root)
    if args.format == "json":
        print(json.dumps(progress, indent=2, sort_keys=True))
    else:
        print(render_text(progress), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
