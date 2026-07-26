from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


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
        raise ValueError(
            f"Could not materialize {FIXED_EVAL_SAMPLES} unique fixed-eval rows."
        )
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
    print(
        f"fixed_eval_part={fixed_part} samples={FIXED_EVAL_SAMPLES} "
        f"sha256={_sha256(fixed_part)}",
        flush=True,
    )
    for output in outputs:
        print(f"manifest={output} sha256={_sha256(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
