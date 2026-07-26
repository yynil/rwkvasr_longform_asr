from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rwkvasr.data import estimate_bucket_manifest_steps, load_webdataset_bucket_manifest
from rwkvasr.eval.stage211_gate import (
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    sha256_file,
)


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return int(payload.get("step", 0))
    finally:
        del payload


def build_receipt(
    *,
    phase: str,
    difficulty: str,
    run_dir: Path,
    bucket_manifest_path: Path,
    init_checkpoint_path: Path,
    completion_checkpoint_path: Path,
) -> dict[str, Any]:
    expected = STAGE211_AUDIO_CURRICULUM[difficulty]
    run_dir = run_dir.resolve()
    bucket_manifest_path = bucket_manifest_path.resolve()
    init_checkpoint_path = init_checkpoint_path.resolve()
    completion_checkpoint_path = completion_checkpoint_path.resolve()
    for label, path in (
        ("run directory", run_dir),
        ("bucket manifest", bucket_manifest_path),
        ("initial checkpoint", init_checkpoint_path),
        ("completion checkpoint", completion_checkpoint_path),
    ):
        exists = path.is_dir() if label == "run directory" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"Stage211 {difficulty} {label} is unavailable: {path}")

    manifest = load_webdataset_bucket_manifest(bucket_manifest_path)
    rows = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
    steps_per_epoch = estimate_bucket_manifest_steps(
        manifest,
        split="train",
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        drop_last=True,
    )
    steps = steps_per_epoch * STAGE211_FULL_DATA_EPOCHS
    if (
        rows != int(expected["rows"])
        or steps_per_epoch != int(expected["steps_per_epoch"])
        or steps != int(expected["steps"])
    ):
        raise ValueError(
            f"Stage211 {difficulty} manifest coverage mismatch: "
            f"rows={rows}/{expected['rows']} "
            f"steps_per_epoch={steps_per_epoch}/{expected['steps_per_epoch']} "
            f"steps={steps}/{expected['steps']}"
        )
    checkpoint_step = _checkpoint_step(completion_checkpoint_path)
    if checkpoint_step != int(expected["steps"]):
        raise ValueError(
            f"Stage211 {difficulty} completion checkpoint step mismatch: "
            f"actual={checkpoint_step} expected={expected['steps']}"
        )

    provenance_path = run_dir / "stage211_provenance.json"
    if not provenance_path.is_file():
        raise ValueError(f"Stage211 curriculum run lacks immutable provenance: {provenance_path}")
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "curriculum_coverage",
        "phase": phase,
        "difficulty": difficulty,
        "complete": True,
        "full_data_profile": True,
        "epochs": STAGE211_FULL_DATA_EPOCHS,
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "rows": rows,
        "row_exposures": rows * STAGE211_FULL_DATA_EPOCHS,
        "hours": float(expected["hours"]),
        "hour_exposures": float(expected["hours"]) * STAGE211_FULL_DATA_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "steps": steps,
        "run_dir": str(run_dir),
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "bucket_manifest_path": str(bucket_manifest_path),
        "bucket_manifest_sha256": sha256_file(bucket_manifest_path),
        "init_checkpoint_path": str(init_checkpoint_path),
        "init_checkpoint_sha256": sha256_file(init_checkpoint_path),
        "completion_checkpoint_path": str(completion_checkpoint_path),
        "completion_checkpoint_sha256": sha256_file(completion_checkpoint_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an immutable Stage211 full-data curriculum-segment receipt."
    )
    parser.add_argument("--phase", choices=("mixer", "block", "logits"), required=True)
    parser.add_argument(
        "--difficulty",
        choices=tuple(STAGE211_AUDIO_CURRICULUM),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bucket-manifest", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--completion-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        phase=str(args.phase),
        difficulty=str(args.difficulty),
        run_dir=args.run_dir,
        bucket_manifest_path=args.bucket_manifest,
        init_checkpoint_path=args.init_checkpoint,
        completion_checkpoint_path=args.completion_checkpoint,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if output_path.is_file() and output_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different coverage receipt: {output_path}")
    output_path.write_text(rendered, encoding="utf-8")
    print(
        f"coverage_receipt={output_path} phase={receipt['phase']} "
        f"difficulty={receipt['difficulty']} rows={receipt['rows']} steps={receipt['steps']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
