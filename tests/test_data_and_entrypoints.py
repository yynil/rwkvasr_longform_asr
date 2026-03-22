import io
import json
import math
import tarfile
import wave
from pathlib import Path

import torch

from rwkvasr.cli.train_ctc import _resolve_train_config, build_parser
from rwkvasr.config import load_yaml
from rwkvasr.data import (
    ASRManifestDataset,
    FeatureCollator,
    StableHashSplitConfig,
    accumulate_webdataset_global_cmvn_stats,
    inspect_webdataset_lengths,
    compute_manifest_global_cmvn,
    inspect_webdataset,
)
from rwkvasr.modules import load_wenet_cmvn
from rwkvasr.training.checkpoint import load_checkpoint, save_checkpoint
from rwkvasr.training.train_loop import TrainConfig, train_ctc_model
from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig
from rwkvasr.training.optimizer import build_rwkv_optimizer, RWKVOptimizerConfig


def _write_manifest(tmp_path: Path, num_examples: int = 3, base_frames: int = 12) -> Path:
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx in range(num_examples):
            features = torch.randn(base_frames + idx, 80)
            feat_path = tmp_path / f"feat-{idx}.pt"
            torch.save(features, feat_path)
            entry = {
                "utt_id": f"utt-{idx}",
                "feature_path": feat_path.name,
                "token_ids": [1, 2, 3, 4],
            }
            handle.write(json.dumps(entry) + "\n")
    return manifest_path


def _make_wav_bytes(num_frames: int = 16000) -> bytes:
    time = torch.linspace(0.0, 1.0, steps=num_frames)
    waveform = (torch.sin(2.0 * torch.pi * 220.0 * time).clamp(-1.0, 1.0) * 32767.0).to(
        dtype=torch.int16
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(waveform.numpy().tobytes())
    return buffer.getvalue()


def _write_webdataset_root(tmp_path: Path, num_examples: int = 2) -> Path:
    root = tmp_path / "webdataset"
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / "shard_00000000.tar"
    with tarfile.open(shard_path, "w") as archive:
        for idx in range(num_examples):
            key = f"{idx + 1:010d}"
            wav_bytes = _make_wav_bytes()
            wav_info = tarfile.TarInfo(name=f"{key}.wav")
            wav_info.size = len(wav_bytes)
            archive.addfile(wav_info, io.BytesIO(wav_bytes))

            json_bytes = json.dumps(
                {
                    "sid": f"utt-{idx}",
                    "text": "你好世界",
                    "token_ids": [1, 2, 3, 4],
                    "sample_rate": 16000,
                    "format": "wav",
                    "begin_time": float(idx),
                    "end_time": float(idx) + 1.0,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            archive.addfile(json_info, io.BytesIO(json_bytes))
    return root


def _write_webdataset_root_for_split(tmp_path: Path, num_examples: int = 8) -> Path:
    root = tmp_path / "webdataset_split"
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / "shard_00000000.tar"
    with tarfile.open(shard_path, "w") as archive:
        for idx in range(num_examples):
            key = f"{idx + 1:010d}"
            wav_bytes = _make_wav_bytes()
            wav_info = tarfile.TarInfo(name=f"{key}.wav")
            wav_info.size = len(wav_bytes)
            archive.addfile(wav_info, io.BytesIO(wav_bytes))

            json_bytes = json.dumps(
                {
                    "sid": f"utt-{idx}",
                    "text": f"你好世界{idx}",
                    "token_ids": [1, 2, 3, 4],
                    "sample_rate": 16000,
                    "format": "wav",
                    "begin_time": float(idx),
                    "end_time": float(idx) + 1.0,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            archive.addfile(json_info, io.BytesIO(json_bytes))
    return root


def test_manifest_dataset_and_collator(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    dataset = ASRManifestDataset(manifest)
    collator = FeatureCollator()

    batch = collator([dataset[0], dataset[1]])
    assert batch.features.shape == (2, 13, 80)
    assert batch.feature_lengths.tolist() == [12, 13]
    assert batch.target_lengths.tolist() == [4, 4]
    assert batch.utt_ids == ["utt-0", "utt-1"]


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            vocab_size=16,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
        )
    )
    optimizer = build_rwkv_optimizer(model, RWKVOptimizerConfig(lr=1e-3, weight_decay=0.1))
    checkpoint = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, step=7, extra={"tag": "x"})

    restored = load_checkpoint(checkpoint, model=model, optimizer=optimizer)
    assert restored["step"] == 7
    assert restored["extra"]["tag"] == "x"


def test_train_ctc_model_smoke(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, num_examples=4, base_frames=48)
    out_dir = tmp_path / "out"

    result = train_ctc_model(
        TrainConfig(
            manifest_path=str(manifest),
            output_dir=str(out_dir),
            vocab_size=8,
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            batch_size=2,
            max_steps=2,
            save_every=1,
            device="cpu",
            p_start=0.0,
            p_max=0.0,
        )
    )

    assert result["steps"] == 2
    assert (out_dir / "global_cmvn.json").exists()
    assert (out_dir / "model_config.yaml").exists()
    assert (out_dir / "train_config.yaml").exists()
    assert (out_dir / "step-1.pt").exists()
    assert (out_dir / "step-2.pt").exists()
    assert load_yaml(out_dir / "model_config.yaml")["vocab_size"] == 8


def test_compute_manifest_global_cmvn_matches_expected_stats(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, num_examples=2, base_frames=4)
    cmvn_path = tmp_path / "global_cmvn.json"
    compute_manifest_global_cmvn(manifest, cmvn_path)

    mean, istd = load_wenet_cmvn(cmvn_path)
    stacked = torch.cat([torch.load(tmp_path / "feat-0.pt"), torch.load(tmp_path / "feat-1.pt")], dim=0)
    expected_mean = stacked.mean(dim=0)
    expected_var = stacked.to(dtype=torch.float64).var(dim=0, unbiased=False).to(dtype=torch.float32)
    expected_istd = torch.tensor(
        [1.0 / math.sqrt(max(float(value), 1.0e-20)) for value in expected_var],
        dtype=torch.float32,
    )

    assert torch.allclose(mean, expected_mean, atol=1e-5, rtol=1e-5)
    assert torch.allclose(istd, expected_istd, atol=1e-5, rtol=1e-5)


def test_compute_webdataset_global_cmvn_ignores_rank_partition(tmp_path: Path, monkeypatch) -> None:
    webdataset_root = _write_webdataset_root(tmp_path, num_examples=4)
    full_stats = accumulate_webdataset_global_cmvn_stats(webdataset_root)

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    partitioned_env_stats = accumulate_webdataset_global_cmvn_stats(webdataset_root)

    assert partitioned_env_stats.frame_num == full_stats.frame_num


def test_train_ctc_model_supports_webdataset_root(tmp_path: Path) -> None:
    webdataset_root = _write_webdataset_root(tmp_path)
    out_dir = tmp_path / "out_webdataset"

    result = train_ctc_model(
        TrainConfig(
            output_dir=str(out_dir),
            vocab_size=8,
            webdataset_root=str(webdataset_root),
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            batch_size=2,
            max_steps=1,
            save_every=1,
            num_workers=0,
            device="cpu",
            p_start=0.0,
            p_max=0.0,
        )
    )

    assert result["steps"] == 1
    assert (out_dir / "global_cmvn.json").exists()
    assert (out_dir / "model_config.yaml").exists()
    assert (out_dir / "train_config.yaml").exists()
    assert (out_dir / "step-1.pt").exists()
    assert load_yaml(out_dir / "train_config.yaml")["webdataset_root"] == str(webdataset_root)


def test_train_ctc_model_supports_length_bucketed_webdataset_root(tmp_path: Path) -> None:
    webdataset_root = _write_webdataset_root(tmp_path, num_examples=4)
    length_index_path = tmp_path / "webdataset_lengths.jsonl"
    inspect_webdataset_lengths(webdataset_root, output_path=length_index_path)
    out_dir = tmp_path / "out_webdataset_bucketed"

    result = train_ctc_model(
        TrainConfig(
            output_dir=str(out_dir),
            vocab_size=8,
            webdataset_root=str(webdataset_root),
            webdataset_length_index_path=str(length_index_path),
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            batch_size=2,
            max_steps=1,
            save_every=1,
            num_workers=0,
            device="cpu",
            p_start=0.0,
            p_max=0.0,
        )
    )

    assert result["steps"] == 1
    assert (out_dir / "step-1.pt").exists()


def test_train_ctc_model_records_epoch_metrics_and_supports_resume(tmp_path: Path) -> None:
    webdataset_root = _write_webdataset_root_for_split(tmp_path, num_examples=8)
    split_config = StableHashSplitConfig(eval_ratio=0.25, hash_seed=5, split_by="sample_id")
    index_path = tmp_path / "webdataset_index.json"
    inspect_webdataset(
        webdataset_root,
        output_path=index_path,
        split_config=split_config,
    )
    length_index_path = tmp_path / "webdataset_lengths.jsonl"
    inspect_webdataset_lengths(
        webdataset_root,
        output_path=length_index_path,
        split_config=split_config,
    )
    out_dir = tmp_path / "out_epoch_eval"

    first = train_ctc_model(
        TrainConfig(
            output_dir=str(out_dir),
            vocab_size=8,
            webdataset_root=str(webdataset_root),
            webdataset_index_path=str(index_path),
            webdataset_length_index_path=str(length_index_path),
            webdataset_split="train",
            webdataset_eval_ratio=0.25,
            webdataset_hash_seed=5,
            webdataset_split_by="sample_id",
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            batch_size=2,
            epochs=1,
            save_every=10,
            num_workers=0,
            device="cpu",
            p_start=0.0,
            p_max=0.0,
        )
    )

    assert first["best_epoch"] == 1
    assert (out_dir / "epoch-1.pt").exists()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "best_checkpoint.yaml").exists()
    metrics = load_yaml(out_dir / "epoch_metrics.yaml")
    assert len(metrics["epochs"]) == 1
    assert "train_loss" in metrics["epochs"][0]
    assert "eval_loss" in metrics["epochs"][0]

    resumed = train_ctc_model(
        TrainConfig(
            output_dir=str(out_dir),
            vocab_size=8,
            webdataset_root=str(webdataset_root),
            webdataset_index_path=str(index_path),
            webdataset_length_index_path=str(length_index_path),
            webdataset_split="train",
            webdataset_eval_ratio=0.25,
            webdataset_hash_seed=5,
            webdataset_split_by="sample_id",
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            batch_size=2,
            epochs=2,
            save_every=10,
            num_workers=0,
            device="cpu",
            resume_from=str(out_dir / "epoch-1.pt"),
            p_start=0.0,
            p_max=0.0,
        )
    )

    assert resumed["steps"] == resumed["steps_per_epoch"] * 2
    assert (out_dir / "epoch-2.pt").exists()
    resumed_metrics = load_yaml(out_dir / "epoch_metrics.yaml")
    assert len(resumed_metrics["epochs"]) == 2


def test_train_ctc_model_resolves_epochs_from_webdataset_index(tmp_path: Path) -> None:
    webdataset_root = _write_webdataset_root_for_split(tmp_path, num_examples=12)
    index_path = tmp_path / "webdataset_index.json"
    index_data = inspect_webdataset(
        webdataset_root,
        output_path=index_path,
        split_config=StableHashSplitConfig(eval_ratio=0.25, hash_seed=5, split_by="sample_id"),
    )
    train_samples = int(index_data["splits"]["train"]["num_samples"])
    expected_steps_per_epoch = math.ceil(train_samples / 2)
    out_dir = tmp_path / "out_webdataset_epochs"

    result = train_ctc_model(
        TrainConfig(
            output_dir=str(out_dir),
            vocab_size=8,
            webdataset_root=str(webdataset_root),
            webdataset_index_path=str(index_path),
            webdataset_split="train",
            webdataset_eval_ratio=0.25,
            webdataset_hash_seed=5,
            webdataset_split_by="sample_id",
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            batch_size=2,
            epochs=2,
            save_every=10,
            num_workers=0,
            device="cpu",
            p_start=0.0,
            p_max=0.0,
        )
    )

    assert result["steps_per_epoch"] == expected_steps_per_epoch
    assert result["steps"] == expected_steps_per_epoch * 2
    assert load_yaml(out_dir / "train_config.yaml")["steps_per_epoch"] == expected_steps_per_epoch


def test_train_cli_config_can_be_loaded_from_yaml_and_overridden(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {tmp_path / 'out'}",
                f"manifest_path: {tmp_path / 'manifest.jsonl'}",
                "batch_size: 8",
                "max_steps: 100",
                "frontend_type: conv2d6",
            ]
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "--config-yaml",
            str(config_path),
            "--batch-size",
            "2",
            "--max-steps",
            "4",
        ]
    )
    resolved = _resolve_train_config(args)

    assert resolved == TrainConfig(
        output_dir=str(tmp_path / "out"),
        manifest_path=str(tmp_path / "manifest.jsonl"),
        batch_size=2,
        max_steps=4,
        frontend_type="conv2d6",
    )
