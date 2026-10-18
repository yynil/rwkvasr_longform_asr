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
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
)


DEFAULT_EASY_MANIFEST = (
    Path.home() / "rwkvasr_data" / "stage211_easy_source_grouped_buckets" / "manifest.json"
)
DEFAULT_METADATA_ROOT = Path.home() / "rwkvasr_data" / "stage211_full_curriculum"
STAGES = (
    "stage179b_medium_dedup_audio_only_online_ctc",
    "stage179c_hard_dedup_audio_only_online_ctc",
    "stage179d_long_dedup_audio_only_online_ctc",
)
FIXED_EVAL_SAMPLES = 256


def _resolve_recorded_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _iter_part_paths(manifest_path: Path, manifest: dict[str, Any]) -> Iterator[Path]:
    train = manifest.get("splits", {}).get("train", {})
    buckets = train.get("buckets", []) if isinstance(train, dict) else train
    for bucket in buckets:
        for part in bucket.get("parts", []):
            yield _resolve_recorded_path(manifest_path, str(part["path"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixed_eval(
    *,
    easy_manifest_path: Path,
    metadata_root: Path,
) -> tuple[Path, list[Path]]:
    easy_manifest_path = easy_manifest_path.resolve()
    metadata_root = metadata_root.resolve()
    easy_manifest = json.loads(easy_manifest_path.read_text(encoding="utf-8"))
    fixed_dir = metadata_root / "fixed_hidden_eval"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    fixed_part = fixed_dir / "part_000000.jsonl"
    selected_lines: list[str] = []
    seen_ids: set[str] = set()
    for part_path in _iter_part_paths(easy_manifest_path, easy_manifest):
        with part_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                utt_id = str(row.get("utt_id") or row.get("key") or row.get("id") or "")
                if not utt_id or utt_id in seen_ids:
                    continue
                selected_lines.append(json.dumps(row, ensure_ascii=True) + "\n")
                seen_ids.add(utt_id)
                if len(selected_lines) >= FIXED_EVAL_SAMPLES:
                    break
        if len(selected_lines) >= FIXED_EVAL_SAMPLES:
            break
    if len(selected_lines) != FIXED_EVAL_SAMPLES:
        raise ValueError(f"Could not materialize {FIXED_EVAL_SAMPLES} unique fixed-eval rows.")
    rendered_part = "".join(selected_lines)
    if fixed_part.is_file() and fixed_part.read_text(encoding="utf-8") != rendered_part:
        raise ValueError(f"Refusing to replace a different fixed-eval part: {fixed_part}")
    fixed_part.write_text(rendered_part, encoding="utf-8")

    outputs: list[Path] = []
    source_manifests = [
        easy_manifest_path,
        *(
            metadata_root / stage / "webdataset_buckets_audio_text" / "manifest.json"
            for stage in STAGES
        ),
    ]
    for source_manifest in source_manifests:
        if not source_manifest.is_file():
            raise FileNotFoundError(str(source_manifest))
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        splits = manifest.setdefault("splits", {})
        splits["eval"] = {
            "num_samples": FIXED_EVAL_SAMPLES,
            "buckets": [
                {
                    "bucket_id": 0,
                    "num_samples": FIXED_EVAL_SAMPLES,
                    "parts": [
                        {
                            "path": str(fixed_part),
                            "num_samples": FIXED_EVAL_SAMPLES,
                        }
                    ],
                }
            ],
        }
        output = source_manifest.with_name("manifest_stage211_fixed_eval.json")
        rendered = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        if output.is_file() and output.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Refusing to replace a different Stage211 manifest: {output}")
        output.write_text(rendered, encoding="utf-8")
        outputs.append(output)
    return fixed_part, outputs


def validate_fixed_eval_outputs(
    *,
    fixed_part: Path,
    outputs: list[Path],
) -> list[dict[str, int | str]]:
    if len(outputs) != len(STAGE211_AUDIO_CURRICULUM):
        raise ValueError("Stage211 fixed-eval builder must produce four manifests.")
    fixed_part = fixed_part.resolve()
    fixed_part_sha256 = _sha256(fixed_part)
    records: list[dict[str, int | str]] = []
    for difficulty, output in zip(
        STAGE211_AUDIO_CURRICULUM,
        outputs,
        strict=True,
    ):
        manifest = load_webdataset_bucket_manifest(output)
        train_rows = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
        eval_buckets = manifest.splits.get("eval", ())
        eval_rows = sum(bucket.num_samples for bucket in eval_buckets)
        steps_per_epoch = estimate_bucket_manifest_steps(
            manifest,
            split="train",
            batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
            world_size=STAGE211_FULL_DATA_WORLD_SIZE,
            frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
            drop_last=False,
        )
        tail_padding_samples_per_epoch = estimate_bucket_manifest_tail_padding_samples(
            manifest,
            split="train",
            batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
            world_size=STAGE211_FULL_DATA_WORLD_SIZE,
            frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        )
        expected = STAGE211_AUDIO_CURRICULUM[difficulty]
        if (
            train_rows != int(expected["rows"])
            or eval_rows != FIXED_EVAL_SAMPLES
            or steps_per_epoch != int(expected["steps_per_epoch"])
            or tail_padding_samples_per_epoch
            != int(expected["tail_padding_samples_per_epoch"])
        ):
            raise ValueError(
                f"Stage211 {difficulty} fixed-eval manifest mismatch: "
                f"train={train_rows}/{expected['rows']} "
                f"eval={eval_rows}/{FIXED_EVAL_SAMPLES} "
                f"steps={steps_per_epoch}/{expected['steps_per_epoch']} "
                "tail_padding_samples_per_epoch="
                f"{tail_padding_samples_per_epoch}/"
                f"{expected['tail_padding_samples_per_epoch']}"
            )
        eval_parts = [
            _resolve_recorded_path(output, part.path)
            for bucket in eval_buckets
            for part in bucket.parts
        ]
        if eval_parts != [fixed_part] or _sha256(eval_parts[0]) != fixed_part_sha256:
            raise ValueError(f"Stage211 {difficulty} does not bind the shared fixed-eval part.")
        records.append(
            {
                "difficulty": difficulty,
                "train_rows": train_rows,
                "eval_rows": eval_rows,
                "steps_per_epoch": steps_per_epoch,
                "tail_padding_samples_per_epoch": tail_padding_samples_per_epoch,
                "manifest_sha256": _sha256(output),
                "fixed_eval_sha256": fixed_part_sha256,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach one immutable 256-row hidden-eval split to Stage211 curricula."
    )
    parser.add_argument("--easy-manifest", type=Path, default=DEFAULT_EASY_MANIFEST)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    args = parser.parse_args()

    fixed_part, outputs = build_fixed_eval(
        easy_manifest_path=args.easy_manifest,
        metadata_root=args.metadata_root,
    )
    records = validate_fixed_eval_outputs(
        fixed_part=fixed_part,
        outputs=outputs,
    )
    print(
        f"fixed_eval_part={fixed_part} samples={FIXED_EVAL_SAMPLES} sha256={_sha256(fixed_part)}",
        flush=True,
    )
    for output, record in zip(outputs, records, strict=True):
        print(
            f"manifest={output} difficulty={record['difficulty']} "
            f"train={record['train_rows']} eval={record['eval_rows']} "
            f"steps_per_epoch={record['steps_per_epoch']} "
            f"tail_padding_samples_per_epoch={record['tail_padding_samples_per_epoch']} "
            f"sha256={record['manifest_sha256']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
