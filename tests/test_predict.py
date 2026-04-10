import io
import json
import sys
import tarfile
import types
import wave
from pathlib import Path

import torch
from safetensors.torch import load_file

from rwkvasr.eval import ctc_greedy_decode
from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig
from rwkvasr.predict import (
    CTCDecodeDebug,
    CTCLabeledPrediction,
    PredictionConfig,
    batched_ctc_prefix_beam_search,
    build_token_alignments,
    ctc_forced_align,
    export_ctc_logits,
    predict_ctc_labeled,
    predict_ctc,
    write_labeled_predictions_jsonl,
    write_predictions_jsonl,
)
from rwkvasr.training.checkpoint import export_checkpoint_to_safetensors, save_checkpoint


class _FakeWhisperProcessor:
    eot = 50000

    class _FakeEncoding:
        @staticmethod
        def decode_bytes(token_ids: list[int]) -> bytes:
            return f"decoded:{','.join(str(token_id) for token_id in token_ids)}".encode("utf-8")

    encoding = _FakeEncoding()

    def encode(self, text: str) -> list[int]:
        return [101, 102]

    def decode(self, token_ids: list[int]) -> str:
        return f"decoded:{','.join(str(token_id) for token_id in token_ids)}"


def _install_fake_whisper(monkeypatch) -> None:
    fake_tokenizer_module = types.ModuleType("whisper.tokenizer")

    def fake_get_tokenizer(*, multilingual: bool, language: str | None = None, task: str | None = None):
        assert multilingual is True
        return _FakeWhisperProcessor()

    fake_tokenizer_module.get_tokenizer = fake_get_tokenizer  # type: ignore[attr-defined]
    fake_whisper_module = types.ModuleType("whisper")
    fake_whisper_module.tokenizer = fake_tokenizer_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper_module)
    monkeypatch.setitem(sys.modules, "whisper.tokenizer", fake_tokenizer_module)


def _build_constant_ctc_model() -> RWKVCTCModel:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=80,
            dim_att=80,
            dim_ff=160,
            num_layers=1,
            vocab_size=4,
            head_size=8,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.ctc_head.bias[0] = 1.0
        model.ctc_head.bias[1] = 5.0
        model.ctc_head.bias[2] = -2.0
        model.ctc_head.bias[3] = -2.0
    return model


def _write_unlabeled_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "predict_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for idx in range(2):
            feat = torch.randn(12 + idx, 80)
            feat_path = tmp_path / f"predict-feat-{idx}.pt"
            torch.save(feat, feat_path)
            handle.write(json.dumps({"utt_id": f"utt-{idx}", "feature_path": feat_path.name}) + "\n")
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


def _write_unlabeled_webdataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "predict_webdataset"
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
                    "sample_rate": 16000,
                    "format": "wav",
                }
            ).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(sample_bytes)
            archive.addfile(json_info, io.BytesIO(sample_bytes))
    return root


def test_ctc_prefix_beam_search_aggregates_paths_beyond_greedy() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [
                    [0.6, 0.4],
                    [0.6, 0.4],
                ]
            ],
            dtype=torch.float32,
        )
    )

    greedy = ctc_greedy_decode(logits, torch.tensor([2]), blank_id=0)
    beam1 = batched_ctc_prefix_beam_search(logits, torch.tensor([2]), blank_id=0, beam_size=1)
    beam2 = batched_ctc_prefix_beam_search(logits, torch.tensor([2]), blank_id=0, beam_size=2)

    assert greedy == [[]]
    assert beam1[0][0].token_ids == ()
    assert beam2[0][0].token_ids == (1,)


def test_ctc_prefix_beam_search_length_bonus_can_reduce_deletion_bias() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [
                    [0.9, 0.1],
                    [0.9, 0.1],
                ]
            ],
            dtype=torch.float32,
        )
    )

    no_bonus = batched_ctc_prefix_beam_search(logits, torch.tensor([2]), blank_id=0, beam_size=2)
    with_bonus = batched_ctc_prefix_beam_search(
        logits,
        torch.tensor([2]),
        blank_id=0,
        beam_size=2,
        length_bonus=5.0,
    )

    assert no_bonus[0][0].token_ids == ()
    assert with_bonus[0][0].token_ids == (1,)


def test_ctc_forced_align_returns_monotonic_token_spans() -> None:
    log_probs = torch.log(
        torch.tensor(
            [
                [0.9, 0.1, 0.0 + 1e-6],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.1],
                [0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8],
            ],
            dtype=torch.float32,
        )
    )

    spans = ctc_forced_align(log_probs, [1, 2], blank_id=0)
    alignments = build_token_alignments(
        log_probs,
        [1, 2],
        blank_id=0,
        frontend_type="linear",
        frame_shift_ms=10.0,
    )

    assert spans == [(1, 2), (4, 4)]
    assert alignments[0].start_ms == 10.0
    assert alignments[0].end_ms == 30.0
    assert alignments[1].start_ms == 40.0
    assert alignments[1].end_ms == 50.0


def test_predict_ctc_supports_unlabeled_manifest(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    model = _build_constant_ctc_model()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    manifest = _write_unlabeled_manifest(tmp_path)

    predictions = predict_ctc(
        PredictionConfig(
            checkpoint_path=str(checkpoint),
            batch_size=2,
            manifest_path=str(manifest),
            device="cpu",
            mode="bi",
            beam_size=4,
            model_config=model.config,
        )
    )

    assert [prediction.utt_id for prediction in predictions] == ["utt-0", "utt-1"]
    assert all(prediction.token_ids and set(prediction.token_ids) == {1} for prediction in predictions)
    assert all(prediction.text and prediction.text.startswith("decoded:") for prediction in predictions)
    assert all(len(prediction.alignments) == len(prediction.token_ids) for prediction in predictions)
    assert all(prediction.alignments[0].start_ms >= 0.0 for prediction in predictions)

    output_path = write_predictions_jsonl(tmp_path / "predictions.jsonl", predictions)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_predict_ctc_accepts_safetensors_checkpoint(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    model = _build_constant_ctc_model()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    safetensors_path = tmp_path / "model.safetensors"
    export_checkpoint_to_safetensors(checkpoint, safetensors_path)
    manifest = _write_unlabeled_manifest(tmp_path)

    predictions = predict_ctc(
        PredictionConfig(
            checkpoint_path=str(safetensors_path),
            batch_size=2,
            manifest_path=str(manifest),
            device="cpu",
            mode="bi",
            beam_size=4,
            model_config=model.config,
        )
    )

    assert [prediction.utt_id for prediction in predictions] == ["utt-0", "utt-1"]
    assert all(prediction.token_ids and set(prediction.token_ids) == {1} for prediction in predictions)
    assert all(len(prediction.alignments) == len(prediction.token_ids) for prediction in predictions)


def test_predict_ctc_supports_unlabeled_webdataset(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    model = _build_constant_ctc_model()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    webdataset_root = _write_unlabeled_webdataset_root(tmp_path)

    predictions = predict_ctc(
        PredictionConfig(
            checkpoint_path=str(checkpoint),
            batch_size=2,
            webdataset_root=str(webdataset_root),
            device="cpu",
            mode="bi",
            beam_size=4,
            model_config=model.config,
        )
    )

    assert [prediction.utt_id for prediction in predictions] == ["utt-0", "utt-1"]
    assert all(prediction.token_ids and set(prediction.token_ids) == {1} for prediction in predictions)
    assert all(prediction.text and prediction.text.startswith("decoded:") for prediction in predictions)
    assert all(len(prediction.alignments) == len(prediction.token_ids) for prediction in predictions)


def test_predict_ctc_labeled_supports_manifest(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    model = _build_constant_ctc_model()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    manifest = tmp_path / "predict_manifest_labeled.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for idx in range(2):
            feat = torch.randn(12 + idx, 80)
            feat_path = tmp_path / f"predict-labeled-feat-{idx}.pt"
            torch.save(feat, feat_path)
            handle.write(
                json.dumps(
                    {
                        "utt_id": f"utt-{idx}",
                        "feature_path": feat_path.name,
                        "text": f"ref-text-{idx}",
                        "token_ids": [1],
                    }
                )
                + "\n"
            )

    predictions = predict_ctc_labeled(
        PredictionConfig(
            checkpoint_path=str(checkpoint),
            batch_size=2,
            manifest_path=str(manifest),
            device="cpu",
            mode="bi",
            beam_size=4,
            save_debug_lengths=True,
            model_config=model.config,
        ),
        limit=2,
    )

    assert [prediction.utt_id for prediction in predictions] == ["utt-0", "utt-1"]
    assert [prediction.ref_text for prediction in predictions] == ["ref-text-0", "ref-text-1"]
    assert all(prediction.pred_text and prediction.pred_text.startswith("decoded:") for prediction in predictions)
    assert all(len(prediction.alignments) == len(prediction.pred_token_ids) for prediction in predictions)
    assert all(prediction.debug is not None for prediction in predictions)
    assert all(prediction.debug and prediction.debug.feature_length > 0 for prediction in predictions)

    output_path = write_labeled_predictions_jsonl(tmp_path / "labeled_predictions.jsonl", predictions)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_write_labeled_predictions_jsonl_serializes_pairs(tmp_path: Path) -> None:
    predictions = [
        CTCLabeledPrediction(
            utt_id="utt-0",
            pred_token_ids=[1, 2],
            ref_token_ids=[1, 3],
            pred_text="pred",
            ref_text="ref",
            score=-1.0,
            mode="bi",
            alignments=[],
            debug=CTCDecodeDebug(
                feature_length=100,
                logit_length=16,
                pred_token_count=2,
                ref_token_count=2,
                blank_top1_ratio=0.75,
                avg_blank_prob=0.6,
            ),
        )
    ]
    output_path = write_labeled_predictions_jsonl(tmp_path / "pairs.jsonl", predictions)
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["pred_text"] == "pred"
    assert payload["ref_text"] == "ref"
    assert payload["pred_token_ids"] == [1, 2]
    assert payload["ref_token_ids"] == [1, 3]
    assert payload["debug"]["feature_length"] == 100


def test_export_ctc_logits_writes_rust_readable_parts(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    model = _build_constant_ctc_model()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    safetensors_path = tmp_path / "model.safetensors"
    export_checkpoint_to_safetensors(checkpoint, safetensors_path)
    manifest = _write_unlabeled_manifest(tmp_path)

    export_index = export_ctc_logits(
        PredictionConfig(
            checkpoint_path=str(safetensors_path),
            batch_size=2,
            manifest_path=str(manifest),
            device="cpu",
            mode="bi",
            model_config=model.config,
        ),
        tmp_path / "decode_export",
        max_batches=1,
    )

    assert len(export_index.parts) == 1
    part = export_index.parts[0]
    assert Path(part.tensors_path).exists()
    assert Path(part.utt_ids_path).exists()
    tensors = load_file(part.tensors_path)
    assert tuple(tensors["logits"].shape) == (2, 13, 4)
    assert tensors["lengths"].dtype == torch.int32
    assert tensors["lengths"].tolist() == [12, 13]

    beams = batched_ctc_prefix_beam_search(
        tensors["logits"],
        tensors["lengths"],
        blank_id=model.config.blank_id,
        beam_size=4,
    )
    assert all(hypotheses[0].token_ids and set(hypotheses[0].token_ids) == {1} for hypotheses in beams)

    exported_ids = Path(part.utt_ids_path).read_text(encoding="utf-8").strip().splitlines()
    assert exported_ids == ["utt-0", "utt-1"]
    index_payload = json.loads((tmp_path / "decode_export" / "export_index.json").read_text(encoding="utf-8"))
    assert index_payload["blank_id"] == 0
    assert index_payload["subsampling_rate"] == 1
    assert index_payload["right_context"] == 0
