#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
)
from rwkvasr.eval.stage211_supplemental import (
    DEFAULT_STAGE211_SUPPLEMENTAL_INVENTORY,
    stage211_supplemental_profile,
)


def build_receipt(inventory_path: Path) -> dict[str, Any]:
    profile = stage211_supplemental_profile(
        inventory_path,
        epochs=STAGE211_FULL_DATA_EPOCHS,
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        require_training_ready=True,
        verify_part_sha256=True,
    )
    inventory = profile["inventory"]
    archive_status_counts: Counter[str] = Counter()
    for record in inventory["archive_records"]:
        archive_status_counts[
            f"{record.get('source', 'unknown')}:{record.get('status', 'unknown')}"
        ] += 1
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "supplemental_natural_profile_receipt",
        "complete": True,
        "inventory_path": profile["inventory_path"],
        "inventory_sha256": profile["inventory_sha256"],
        "bucket_manifest_path": profile["bucket_manifest_path"],
        "bucket_manifest_sha256": profile["bucket_manifest_sha256"],
        "rows": int(profile["rows"]),
        "hours": float(profile["hours"]),
        "epochs": int(profile["epochs"]),
        "steps_per_epoch": int(profile["steps_per_epoch"]),
        "steps": int(profile["steps"]),
        "tail_padding_samples_per_epoch": int(
            profile["tail_padding_samples_per_epoch"]
        ),
        "row_exposures": int(profile["row_exposures"]),
        "hour_exposures": float(profile["hour_exposures"]),
        "tail_padding_sample_exposures": int(
            profile["tail_padding_sample_exposures"]
        ),
        "executed_sample_exposures": int(profile["executed_sample_exposures"]),
        "selected_counts_by_source": inventory["selected_counts_by_source"],
        "selected_hours_by_source": inventory["selected_hours_by_source"],
        "train_part_count": len(inventory["part_records"]),
        "archive_status_counts": dict(sorted(archive_status_counts.items())),
    }


def write_immutable_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Refusing to replace a different Stage211 receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and receipt the Stage211 supplemental natural-audio profile."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_STAGE211_SUPPLEMENTAL_INVENTORY,
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    inventory_path = args.inventory.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else inventory_path.parent / "supplemental_profile_receipt.json"
    )
    receipt = build_receipt(inventory_path)
    write_immutable_receipt(output_path, receipt)
    print(
        "[stage211-supplemental-profile] "
        f"rows={receipt['rows']} hours={receipt['hours']:.6f} "
        f"steps={receipt['steps']} output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
