from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rwkvasr.config import save_yaml

try:
    from scripts.run_stage210g_conditional_hard_ctc_distill import (
        DEFAULT_OUTPUT_ROOT,
        _config as _stage210g_config,
        _latest_step,
        _run,
        _segments as _stage210g_segments,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage210g_conditional_hard_ctc_distill import (
        DEFAULT_OUTPUT_ROOT,
        _config as _stage210g_config,
        _latest_step,
        _run,
        _segments as _stage210g_segments,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage210h_source_balanced_ctc_distill"
DEFAULT_BUCKET_MANIFEST = Path(
    "/tmp/rwkvasr_data/stage210h_easy_source_grouped_buckets/manifest.json"
)


def _default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / (
        "sensevoice_rwkv_stage210h_stage210g1000_sourcebalanced_easy_"
        "factorized_hard_admission2k_lr2e7_wd0_4x4090"
    )


def _segments(*, smoke: bool) -> list[dict[str, Any]]:
    return _stage210g_segments("admission", smoke=smoke)


def _config(
    *,
    segment: dict[str, Any],
    output_dir: Path,
    init_checkpoint: Path,
    bucket_manifest: Path,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    config = _stage210g_config(
        phase="admission",
        segment=segment,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        resume=resume,
        smoke=smoke,
    )
    config.update(
        {
            "webdataset_bucket_manifest_path": str(bucket_manifest),
            "bucket_source_interleave": True,
            "wandb_run_name": f"{output_dir.name}_{segment['name']}",
        }
    )
    return config


def _write_config(
    *,
    segment: dict[str, Any],
    config: dict[str, Any],
    config_dir: Path,
    smoke: bool,
) -> Path:
    target_dir = config_dir / ("smoke" if smoke else "admission")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"stage210h_{segment['name']}.yaml"
    save_yaml(path, config)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage210g objective on source-balanced easy buckets."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--bucket-manifest", type=Path, default=DEFAULT_BUCKET_MANIFEST)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_output_dir = args.output_dir or _default_output_dir()
    output_dir = Path(f"{base_output_dir}_smoke") if args.smoke else base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_step = _latest_step(output_dir)
    init_checkpoint = args.init_checkpoint
    if not args.bucket_manifest.is_file():
        raise FileNotFoundError(str(args.bucket_manifest))
    if latest_step <= 0:
        if init_checkpoint is None:
            parser.error("a fresh Stage210h run requires --init-checkpoint from Stage210g")
        if not init_checkpoint.is_file():
            raise FileNotFoundError(str(init_checkpoint))
    if init_checkpoint is None:
        init_checkpoint = output_dir / "resume-placeholder.pt"

    print("phase=admission", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"bucket_manifest={args.bucket_manifest}", flush=True)
    print(f"latest_step={latest_step}", flush=True)
    print(f"init_checkpoint={init_checkpoint}", flush=True)
    for segment in _segments(smoke=bool(args.smoke)):
        target_step = int(segment["target_step"])
        print(
            f"segment={segment['name']} split={segment['split'].name} target_step={target_step}",
            flush=True,
        )
        if latest_step >= target_step:
            print(f"skip {segment['name']} target_step={target_step}", flush=True)
            continue
        config = _config(
            segment=segment,
            output_dir=output_dir,
            init_checkpoint=init_checkpoint,
            bucket_manifest=args.bucket_manifest,
            resume=latest_step > 0,
            smoke=bool(args.smoke),
        )
        config_path = _write_config(
            segment=segment,
            config=config,
            config_dir=args.config_dir,
            smoke=bool(args.smoke),
        )
        code = _run(
            config_path,
            output_dir / "logs" / f"{segment['name']}.log",
            dry_run=bool(args.dry_run),
            master_port=int(args.master_port),
        )
        if code != 0 or args.dry_run:
            return code
        latest_step = _latest_step(output_dir)
        if latest_step < target_step:
            raise RuntimeError(
                f"Stage210h exited early: latest_step={latest_step} target_step={target_step}"
            )
    print(f"complete latest_step={latest_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
