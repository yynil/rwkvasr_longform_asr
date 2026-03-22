import io
import json
import tarfile
import wave
from pathlib import Path
from types import SimpleNamespace

import torch

from rwkvasr.cli.eval_ctc import _resolve_model_config
from rwkvasr.config import save_yaml
from rwkvasr.data import StableHashSplitConfig, inspect_webdataset
from rwkvasr.eval import ctc_greedy_decode, edit_distance, evaluate_ctc_modes, token_error_rate
from rwkvasr.eval.ctc import EvalConfig
from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig
from rwkvasr.training.checkpoint import save_checkpoint


def _write_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "eval_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for idx in range(2):
            feat = torch.randn(48 + idx, 80)
            feat_path = tmp_path / f"eval-feat-{idx}.pt"
            torch.save(feat, feat_path)
            handle.write(
                json.dumps(
                    {
                        "utt_id": f"utt-{idx}",
                        "feature_path": feat_path.name,
                        "token_ids": [1, 2, 3],
                    }
                )
                + "\n"
            )
    return manifest


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


def _write_webdataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "eval_webdataset"
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / "shard_00000000.tar"
    wav_bytes = _make_wav_bytes()
    with tarfile.open(shard_path, "w") as archive:
        for idx in range(2):
            key = f"{idx + 1:010d}"
            wav_info = tarfile.TarInfo(name=f"{key}.wav")
            wav_info.size = len(wav_bytes)
            archive.addfile(wav_info, io.BytesIO(wav_bytes))
            sample_bytes = json.dumps(
                {
                    "sid": f"utt-{idx}",
                    "token_ids": [1, 2, 3],
                    "sample_rate": 16000,
                    "format": "wav",
                }
            ).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(sample_bytes)
            archive.addfile(json_info, io.BytesIO(sample_bytes))
    return root


def test_ctc_greedy_decode_collapses_repeats_and_blanks() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 4.0],
            ]
        ]
    )
    decoded = ctc_greedy_decode(logits, torch.tensor([5]), blank_id=0)
    assert decoded == [[1, 2]]


def test_edit_distance_and_token_error_rate() -> None:
    assert edit_distance([1, 2, 3], [1, 3]) == 1
    ter = token_error_rate([[1, 2, 3], [1, 2]], [[1, 3], [1, 2]])
    assert ter == 1 / 5


def test_evaluate_ctc_modes_smoke(tmp_path: Path) -> None:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            vocab_size=8,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
        )
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    manifest = _write_manifest(tmp_path)

    metrics = evaluate_ctc_modes(
        EvalConfig(
            manifest_path=str(manifest),
            checkpoint_path=str(checkpoint),
            batch_size=2,
            device="cpu",
            modes=("bi", "l2r", "alt"),
            model_config=model.config,
        )
    )

    assert set(metrics.keys()) == {"bi", "l2r", "alt"}
    for mode_metrics in metrics.values():
        assert "token_error_rate" in mode_metrics
        assert "exact_match" in mode_metrics


def test_evaluate_ctc_modes_supports_webdataset_root(tmp_path: Path) -> None:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            vocab_size=8,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
        )
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    webdataset_root = _write_webdataset_root(tmp_path)

    metrics = evaluate_ctc_modes(
        EvalConfig(
            checkpoint_path=str(checkpoint),
            batch_size=2,
            device="cpu",
            modes=("bi", "l2r"),
            webdataset_root=str(webdataset_root),
            model_config=model.config,
        )
    )

    assert set(metrics.keys()) == {"bi", "l2r"}


def test_evaluate_ctc_modes_supports_webdataset_sample_split(tmp_path: Path) -> None:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            vocab_size=8,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
        )
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    webdataset_root = _write_webdataset_root(tmp_path)
    index_data = inspect_webdataset(
        webdataset_root,
        output_path=tmp_path / "webdataset_index.json",
        split_config=StableHashSplitConfig(eval_ratio=0.5, hash_seed=11, split_by="sample_id"),
    )
    eval_utts = float(index_data["splits"]["eval"]["num_samples"])

    metrics = evaluate_ctc_modes(
        EvalConfig(
            checkpoint_path=str(checkpoint),
            batch_size=2,
            device="cpu",
            modes=("bi",),
            webdataset_root=str(webdataset_root),
            webdataset_split="eval",
            webdataset_eval_ratio=0.5,
            webdataset_hash_seed=11,
            webdataset_split_by="sample_id",
            model_config=model.config,
        )
    )

    assert metrics["bi"]["num_utts"] == eval_utts


def test_eval_cli_model_config_can_be_loaded_from_yaml(tmp_path: Path) -> None:
    model_config = RWKVCTCModelConfig(
        input_dim=80,
        n_embd=128,
        dim_att=128,
        dim_ff=256,
        num_layers=2,
        vocab_size=8,
        head_size=32,
        conv_kernel_size=5,
        dropout=0.0,
    )
    checkpoint = tmp_path / "model.pt"
    save_yaml(tmp_path / "model_config.yaml", model_config)

    resolved = _resolve_model_config(
        SimpleNamespace(
            config_yaml=None,
            checkpoint_path=str(checkpoint),
            vocab_size=None,
            input_dim=80,
            n_embd=0,
            dim_att=0,
            dim_ff=0,
            num_layers=0,
            head_size=0,
            conv_kernel_size=0,
            dropout=0.0,
            blank_id=0,
            frontend_type="conv2d6",
            cmvn_file=None,
            cmvn_is_json=True,
        )
    )

    assert resolved == model_config
