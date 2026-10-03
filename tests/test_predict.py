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
    CTCHotword,
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
from rwkvasr.predict.ctc import load_hotwords
import rwkvasr.predict.ctc as predict_ctc_module
from rwkvasr.predict.rwkv_decoder import _ctc_draft_fallback_decision
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


class _FakePredictionBatch:
    def __init__(self, *, labeled: bool = False) -> None:
        self.features = torch.zeros(1, 3, 80)
        self.feature_lengths = torch.tensor([3], dtype=torch.long)
        self.utt_ids = ["utt-0"]
        self.decoder_prompt_before_audio = None
        self.decoder_prompt_before_audio_lengths = None
        if labeled:
            self.targets = torch.tensor([2], dtype=torch.long)
            self.target_lengths = torch.tensor([1], dtype=torch.long)
            self.texts = ["ref"]

    def to(self, device: torch.device, *, feature_dtype: torch.dtype | None = None):
        dtype = feature_dtype or self.features.dtype
        self.features = self.features.to(device=device, dtype=dtype)
        self.feature_lengths = self.feature_lengths.to(device=device)
        if hasattr(self, "targets"):
            self.targets = self.targets.to(device=device)
            self.target_lengths = self.target_lengths.to(device=device)
        return self


class _FakeDecoderAwareCTCModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(num_layers=1, blank_id=0, frontend_type="linear")
        self.decoder = None
        self.encoder = self
        self.ctc_head = self._bypass_head
        self.used_decoder_aware_logits = False

    def to(self, device: torch.device):
        return self

    def eval(self):
        return self

    def __call__(self, features, feature_lengths, *, direction_mask=None):
        encoded = torch.zeros(features.size(0), features.size(1), 4, device=features.device, dtype=features.dtype)
        return encoded, feature_lengths, None

    def _bypass_head(self, encoded: torch.Tensor) -> torch.Tensor:
        logits = encoded.new_full((encoded.size(0), encoded.size(1), 3), -5.0)
        logits[..., 1] = 5.0
        return logits

    def ctc_logits_from_encoded(
        self,
        encoded: torch.Tensor,
        encoded_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.used_decoder_aware_logits = True
        logits = encoded.new_full((encoded.size(0), encoded.size(1), 3), -5.0)
        logits[..., 2] = 5.0
        return logits, encoded_lengths


def _build_sampling_rwkv_model() -> RWKVCTCModel:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=64,
            dim_att=64,
            dim_ff=128,
            num_layers=1,
            vocab_size=4,
            head_size=64,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
            decoder_enabled=True,
            decoder_num_layers=1,
            decoder_n_embd=64,
            decoder_ffn_hidden_size=256,
            decoder_audio_conditioning="full",
        )
    )
    if model.decoder is None:
        raise RuntimeError("Failed to construct RWKV decoder path for sampling test.")

    hidden_size = int(model.decoder.hidden_size)
    logits_base = torch.tensor([0.0, 2.0, 1.9, -2.0], dtype=torch.float32)

    class _FixedHead(torch.nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            if hidden.ndim == 2:
                return self.logits.view(1, -1).expand(int(hidden.size(0)), -1).to(device=hidden.device, dtype=hidden.dtype)
            if hidden.ndim == 3:
                return self.logits.view(1, 1, -1).expand(
                    int(hidden.size(0)),
                    int(hidden.size(1)),
                    -1,
                ).to(device=hidden.device, dtype=hidden.dtype)
            raise ValueError(f"Unexpected hidden shape for fixed head: {tuple(hidden.shape)}")

    fixed_head = _FixedHead(logits_base)

    def _fixed_hidden_embeds(x: torch.Tensor, state=None) -> tuple[torch.Tensor, object]:
        return torch.zeros((x.size(0), x.size(1), hidden_size), device=x.device, dtype=x.dtype), state

    def _fixed_forward_tokens(
        token_ids: torch.Tensor,
        state=None,
    ) -> tuple[torch.Tensor, object]:
        zeros = torch.zeros((token_ids.size(0), token_ids.size(1), hidden_size), device=token_ids.device, dtype=torch.float32)
        return fixed_head(zeros), state

    model.decoder.head = fixed_head
    model.decoder.forward_hidden_embeds = _fixed_hidden_embeds
    model.decoder.forward_tokens = _fixed_forward_tokens
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
                    "text": f"ref-text-{idx}",
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


def test_ctc_prefix_beam_search_blank_logit_bias_can_reduce_blank_bias() -> None:
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

    no_bias = batched_ctc_prefix_beam_search(logits, torch.tensor([2]), blank_id=0, beam_size=2)
    zero_bias = batched_ctc_prefix_beam_search(
        logits,
        torch.tensor([2]),
        blank_id=0,
        beam_size=2,
        blank_logit_bias=0.0,
    )
    blank_penalty = batched_ctc_prefix_beam_search(
        logits,
        torch.tensor([2]),
        blank_id=0,
        beam_size=2,
        blank_logit_bias=-3.0,
    )

    assert no_bias[0][0].token_ids == ()
    assert zero_bias[0][0].token_ids == no_bias[0][0].token_ids
    assert blank_penalty[0][0].token_ids == (1,)


def test_ctc_prefix_beam_search_hotword_bonus_can_flip_ambiguous_choice() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [
                    [0.01, 0.51, 0.48],
                ]
            ],
            dtype=torch.float32,
        )
    )

    no_hotword = batched_ctc_prefix_beam_search(
        logits,
        torch.tensor([1]),
        blank_id=0,
        beam_size=2,
    )
    with_hotword = batched_ctc_prefix_beam_search(
        logits,
        torch.tensor([1]),
        blank_id=0,
        beam_size=2,
        hotwords=(CTCHotword(text="preferred", token_ids=(2,), weight=1.0),),
    )

    assert no_hotword[0][0].token_ids == (1,)
    assert with_hotword[0][0].token_ids == (2,)


def test_load_hotwords_supports_default_and_per_line_weights(tmp_path: Path) -> None:
    class _Tokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(ch) for ch in text]

    hotwords_path = tmp_path / "hotwords.txt"
    hotwords_path.write_text(
        "# comment\nwizard\n搜狐体育\t6.5\n\n",
        encoding="utf-8",
    )

    hotwords = load_hotwords(
        hotwords_path,
        tokenizer=_Tokenizer(),
        default_weight=3.0,
    )

    assert [(item.text, item.weight) for item in hotwords] == [("wizard", 3.0), ("搜狐体育", 6.5)]
    assert hotwords[0].token_ids
    assert hotwords[1].token_ids


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


def test_rwkv_decoder_sampling_top_k_matches_greedy_when_top_k_is_one() -> None:
    model = _build_sampling_rwkv_model()
    encoded = torch.zeros(1, 4, int(model.config.n_embd))
    encoded_lengths = torch.tensor([4], dtype=torch.long)

    greedy, _, _ = model.decoder_greedy_decode(
        encoded,
        encoded_lengths,
        max_new_tokens=6,
        do_sample=False,
    )
    sampled, _, _ = model.decoder_greedy_decode(
        encoded,
        encoded_lengths,
        max_new_tokens=6,
        do_sample=True,
        top_k=1,
    )
    assert sampled == greedy


def test_rwkv_decoder_sampling_can_stochastically_choose_non_argmax_token() -> None:
    model = _build_sampling_rwkv_model()
    encoded = torch.zeros(1, 4, int(model.config.n_embd))
    encoded_lengths = torch.tensor([4], dtype=torch.long)

    saw_non_argmax = False
    for seed in range(100):
        torch.manual_seed(seed)
        generated_batch, _, _ = model.decoder_greedy_decode(
            encoded,
            encoded_lengths,
            max_new_tokens=1,
            do_sample=True,
            top_k=2,
        )
        generated = generated_batch[0][0]
        if generated == 2:
            saw_non_argmax = True
            break
    assert saw_non_argmax


def test_ctc_draft_fallback_decision_rejects_repetition_and_large_drift() -> None:
    fallback, reason, cer, ratio = _ctc_draft_fallback_decision(
        pred_text="This is English text. she is in the sleep and she is in the sleep and she is in the sleep",
        draft_text="she is in the sleep and cognition department",
        max_cer=0.25,
        min_length_ratio=0.75,
        max_length_ratio=1.25,
        reject_repetition=True,
        metric_normalization="ctc",
    )
    assert fallback is True
    assert reason == "repeated_span"
    assert cer is not None and cer > 0
    assert ratio is not None and ratio > 0

    fallback, reason, _, _ = _ctc_draft_fallback_decision(
        pred_text="This is English text. thank you very much.",
        draft_text="thank you very much",
        max_cer=0.25,
        min_length_ratio=0.75,
        max_length_ratio=1.25,
        reject_repetition=True,
        metric_normalization="ctc",
    )
    assert fallback is False
    assert reason is None


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


def test_predict_ctc_uses_decoder_aware_ctc_logits(monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    model = _FakeDecoderAwareCTCModel()
    monkeypatch.setattr(predict_ctc_module, "_load_prediction_model", lambda config, device: (model, None))
    monkeypatch.setattr(predict_ctc_module, "_build_prediction_loader", lambda config: [_FakePredictionBatch()])

    predictions = predict_ctc(
        PredictionConfig(
            checkpoint_path="unused.pt",
            batch_size=1,
            manifest_path="unused.jsonl",
            device="cpu",
            mode="bi",
            beam_size=1,
            model_config=RWKVCTCModelConfig(input_dim=80, n_embd=4, dim_att=4, dim_ff=8, num_layers=1, vocab_size=3),
        )
    )

    assert model.used_decoder_aware_logits
    assert predictions[0].token_ids == [2]


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


def test_predict_ctc_labeled_supports_webdataset_length_index(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    model = _build_constant_ctc_model()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model=model, step=0)
    webdataset_root = _write_unlabeled_webdataset_root(tmp_path)
    length_index = tmp_path / "predict_webdataset_lengths.jsonl"
    with length_index.open("w", encoding="utf-8") as handle:
        for idx in range(2):
            key = f"{idx + 1:010d}"
            handle.write(
                json.dumps(
                    {
                        "shard_name": "shard_00000000.tar",
                        "key": key,
                        "utt_id": f"utt-{idx}",
                        "split": "eval",
                        "num_frames": 100,
                        "audio_member": f"{key}.wav",
                        "audio_format": "wav",
                        "json_member": f"{key}.json",
                        "audio_offset": None,
                        "audio_size": None,
                        "json_offset": None,
                        "json_size": None,
                    }
                )
                + "\n"
            )

    predictions = predict_ctc_labeled(
        PredictionConfig(
            checkpoint_path=str(checkpoint),
            batch_size=2,
            webdataset_root=str(webdataset_root),
            webdataset_length_index_path=str(length_index),
            webdataset_split="eval",
            device="cpu",
            mode="bi",
            beam_size=4,
            model_config=model.config,
        )
    )

    assert [prediction.utt_id for prediction in predictions] == ["utt-0", "utt-1"]
    assert [prediction.ref_text for prediction in predictions] == ["ref-text-0", "ref-text-1"]
    assert all(prediction.pred_text and prediction.pred_text.startswith("decoded:") for prediction in predictions)


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


def test_predict_ctc_labeled_uses_decoder_aware_ctc_logits(monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    model = _FakeDecoderAwareCTCModel()
    monkeypatch.setattr(predict_ctc_module, "_load_prediction_model", lambda config, device: (model, None))
    monkeypatch.setattr(
        predict_ctc_module,
        "_build_labeled_prediction_loader",
        lambda config, *, tokenizer=None: [_FakePredictionBatch(labeled=True)],
    )

    predictions = predict_ctc_labeled(
        PredictionConfig(
            checkpoint_path="unused.pt",
            batch_size=1,
            manifest_path="unused.jsonl",
            device="cpu",
            mode="bi",
            beam_size=1,
            model_config=RWKVCTCModelConfig(input_dim=80, n_embd=4, dim_att=4, dim_ff=8, num_layers=1, vocab_size=3),
        )
    )

    assert model.used_decoder_aware_logits
    assert predictions[0].pred_token_ids == [2]


def test_prediction_model_config_uses_native_backend_on_cpu() -> None:
    config = RWKVCTCModelConfig(
        input_dim=80,
        n_embd=4,
        dim_att=4,
        dim_ff=8,
        num_layers=1,
        vocab_size=3,
        backend="cuda_clampw",
    )

    resolved = predict_ctc_module._prediction_model_config_for_device(
        config,
        device=torch.device("cpu"),
    )

    assert resolved.backend == "native"
    assert config.backend == "cuda_clampw"


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
