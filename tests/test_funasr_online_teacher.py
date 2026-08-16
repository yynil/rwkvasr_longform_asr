from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from rwkvasr.training.funasr_online_teacher import (
    FunASRNanoCTCTopKOnlineTeacher,
    FunASROnlineCTCTeacherConfig,
    _capture_funasr_ctc_decoder_hiddens,
    _capture_funasr_encoder_hiddens,
    _resolve_audio_path,
)


class _FakeEncoderLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim, bias=False)
        self.feed_forward = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        x = x + self.self_attn(x)
        x = x + self.feed_forward(x)
        return x, mask


class _FakeNanoEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoders0 = nn.ModuleList([_FakeEncoderLayer(dim)])
        self.encoders = nn.ModuleList([_FakeEncoderLayer(dim)])
        self.tp_encoders = nn.ModuleList([_FakeEncoderLayer(dim)])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        mask = torch.ones(x.size(0), 1, x.size(1), device=x.device)
        for layer in (*self.encoders0, *self.encoders, *self.tp_encoders):
            x, mask = layer(x, mask)
        return x, lengths


class _InplaceFakeNanoEncoder(_FakeNanoEncoder):
    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x.mul_(2.0)
        return super().forward(x, lengths)


class _NativeNanoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from funasr.models.sense_voice.model import (
            EncoderLayerSANM,
            MultiHeadedAttentionSANM,
            PositionwiseFeedForward,
        )

        first = EncoderLayerSANM(
            8,
            4,
            MultiHeadedAttentionSANM(2, 8, 4, 0.0, 3, 0),
            PositionwiseFeedForward(4, 12, 0.0),
            0.0,
        )
        second = EncoderLayerSANM(
            4,
            4,
            MultiHeadedAttentionSANM(2, 4, 4, 0.0, 3, 0),
            PositionwiseFeedForward(4, 12, 0.0),
            0.0,
        )
        self.encoders0 = nn.ModuleList([first])
        self.encoders = nn.ModuleList([second])
        self.tp_encoders = nn.ModuleList()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        steps = torch.arange(x.size(1), device=x.device)
        mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1)
        for layer in (*self.encoders0, *self.encoders):
            x, mask = layer(x, mask)[:2]
        return x, lengths


class _FakeCTCDecoder(nn.Module):
    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        return x, lengths


class _FakeLayeredCTCDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = self.linear2(x)
        for block in self.blocks:
            x = block(x)
        return x, lengths


class _FakeCTC(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.ctc_lo = nn.Linear(dim, vocab_size)

    def log_softmax(self, x: torch.Tensor) -> torch.Tensor:
        return self.ctc_lo(x).log_softmax(dim=-1)


class _FakeNanoModel(nn.Module):
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.audio_encoder = _InplaceFakeNanoEncoder(dim)
        self.ctc_decoder = _FakeCTCDecoder()
        self.ctc = _FakeCTC(dim, vocab_size)
        self.blank_id = vocab_size - 1


def test_resolve_audio_path_materializes_embedded_archive_bytes(tmp_path) -> None:
    payload = b"embedded-audio-payload"
    resolved = _resolve_audio_path(
        {
            "audio_member": "MLS/sample.wav",
            "audio_size": len(payload),
            "_audio_bytes": payload,
        },
        "utt-embedded",
        shard_paths={},
        audio_cache_dir=tmp_path,
        keep_audio_cache=False,
    )

    assert resolved.temporary is True
    assert resolved.path != ""
    assert Path(resolved.path).read_bytes() == payload


def test_capture_funasr_encoder_hiddens_collects_requested_components() -> None:
    torch.manual_seed(2701)
    encoder = _FakeNanoEncoder(dim=6)
    features = torch.randn(2, 5, 6)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    with _capture_funasr_encoder_hiddens(
        encoder,
        (0, 2),
        capture_input=True,
    ) as captured:
        encoder(features, lengths)

    assert torch.equal(captured["encoder_input"], features)
    assert torch.equal(captured["encoder_input_lengths"], lengths)
    assert set(captured["layers"]) == {0, 2}
    for layer_id in (0, 2):
        assert set(captured["layers"][layer_id]) == {"input", "mixer", "ffn", "block"}
        for component in ("input", "mixer", "ffn", "block"):
            assert captured["layers"][layer_id][component].shape == (2, 5, 6)
            assert not captured["layers"][layer_id][component].requires_grad
    assert torch.equal(captured["layers"][0]["input"], features)


def test_capture_funasr_encoder_hiddens_matches_native_sanm_component_semantics() -> None:
    torch.manual_seed(2706)
    encoder = _NativeNanoEncoder().eval()
    features = torch.randn(2, 6, 8)
    lengths = torch.tensor([6, 4], dtype=torch.long)

    with _capture_funasr_encoder_hiddens(
        encoder,
        (0, 1),
        capture_input=False,
    ) as captured:
        encoder(features, lengths)

    steps = torch.arange(features.size(1))
    mask = (steps.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
    layers = (*encoder.encoders0, *encoder.encoders)
    for layer_id, layer in enumerate(layers):
        components = captured["layers"][layer_id]
        layer_input = components["input"]
        normalized = layer.norm1(layer_input)
        expected_mixer = layer.self_attn(normalized, mask)
        expected_post_mixer = expected_mixer
        if layer.in_size == layer.size:
            expected_post_mixer = layer_input + expected_mixer
        expected_ffn = layer.feed_forward(layer.norm2(expected_post_mixer))
        expected_block = expected_post_mixer + expected_ffn

        torch.testing.assert_close(components["mixer"], expected_mixer, rtol=0.0, atol=0.0)
        torch.testing.assert_close(components["ffn"], expected_ffn, rtol=0.0, atol=0.0)
        torch.testing.assert_close(components["block"], expected_block, rtol=0.0, atol=0.0)


def test_capture_funasr_encoder_hiddens_rejects_invalid_layer() -> None:
    encoder = _FakeNanoEncoder(dim=4)

    with pytest.raises(ValueError, match="out of range"):
        with _capture_funasr_encoder_hiddens(encoder, (3,), capture_input=False):
            pass


def test_capture_funasr_ctc_decoder_hiddens_collects_projection_and_blocks() -> None:
    torch.manual_seed(2703)
    decoder = _FakeLayeredCTCDecoder(dim=6)
    features = torch.randn(2, 5, 6)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    with _capture_funasr_ctc_decoder_hiddens(decoder, enabled=True) as captured:
        decoder(features, lengths)

    assert set(captured) == {"input", "layer_0", "layer_1"}
    for value in captured.values():
        assert value.shape == (2, 5, 6)
        assert not value.requires_grad


def test_feature_records_batches_teacher_forward_and_preserves_student_features() -> None:
    torch.manual_seed(2702)
    teacher = object.__new__(FunASRNanoCTCTopKOnlineTeacher)
    teacher.config = FunASROnlineCTCTeacherConfig(
        model_path="unused",
        device="cpu",
        top_k=3,
        project_blank_id=7,
        project_vocab_size=8,
        return_encoder_input=True,
        return_layer_hiddens=True,
    )
    teacher.model = _FakeNanoModel(dim=6, vocab_size=7)
    teacher.audio_rows = {}

    features = torch.randn(2, 5, 6)
    original = features.clone()
    lengths = torch.tensor([5, 3], dtype=torch.long)
    records = teacher.feature_records(
        ["utt-a", "utt-b"],
        features,
        lengths,
        layer_ids=[0, 2],
    )

    assert torch.equal(features, original)
    assert set(records) == {"utt-a", "utt-b"}
    assert records["utt-a"]["num_frames"] == 5
    assert records["utt-b"]["num_frames"] == 3
    assert torch.allclose(records["utt-a"]["encoder_input"].float(), original[0], atol=1e-3)
    assert records["utt-b"]["encoder_layer_hiddens"]["2"]["block"].shape == (3, 6)

    hidden_only = teacher.feature_records(
        ["utt-a", "utt-b"],
        features,
        lengths,
        layer_ids=[0],
        include_ctc_outputs=False,
    )
    assert hidden_only["utt-a"]["format"] == "funasr_nano_encoder_hidden_online_v1"
    assert "num_frames" not in hidden_only["utt-a"]
    assert hidden_only["utt-b"]["encoder_layer_hiddens"]["0"]["input"].shape == (3, 6)
    assert hidden_only["utt-b"]["encoder_layer_hiddens"]["0"]["mixer"].shape == (3, 6)
    assert hidden_only["utt-b"]["encoder_layer_hiddens"]["0"]["mixer"].device.type == "cpu"
    assert hidden_only["utt-b"]["encoder_layer_hiddens"]["0"]["mixer"].dtype == torch.float16

    teacher.config = replace(teacher.config, keep_layer_hiddens_on_device=True)
    device_resident = teacher.feature_records(
        ["utt-a", "utt-b"],
        features,
        lengths,
        layer_ids=[0],
        include_ctc_outputs=False,
    )
    resident_mixer = device_resident["utt-b"]["encoder_layer_hiddens"]["0"]["mixer"]
    assert resident_mixer.device == features.device
    assert resident_mixer.dtype == torch.float16

    teacher.config = replace(
        teacher.config,
        return_full_log_probs=True,
        keep_full_log_probs_on_device=True,
        project_ignored_token_ids=(),
    )
    full_records = teacher.feature_records(["utt-a", "utt-b"], features, lengths)
    full_log_probs = full_records["utt-b"]["full_log_probs"]
    assert full_log_probs.shape == (3, 8)
    assert full_log_probs.device == features.device
    assert full_log_probs.dtype == torch.float16

    teacher.model.ctc_decoder = _FakeLayeredCTCDecoder(dim=6)
    teacher.config = replace(
        teacher.config,
        return_ctc_decoder_hiddens=True,
        keep_layer_hiddens_on_device=False,
    )
    decoder_records = teacher.feature_records(["utt-a", "utt-b"], features, lengths)
    decoder_hiddens = decoder_records["utt-b"]["ctc_decoder_hiddens"]
    assert set(decoder_hiddens) == {"input", "layer_0", "layer_1"}
    assert decoder_hiddens["layer_1"].shape == (3, 6)
    assert decoder_hiddens["layer_1"].dtype == torch.float16


def test_feature_records_project_ignored_tokens_before_ctc_distribution_outputs() -> None:
    teacher = object.__new__(FunASRNanoCTCTopKOnlineTeacher)
    teacher.config = FunASROnlineCTCTeacherConfig(
        model_path="unused",
        device="cpu",
        top_k=3,
        project_blank_id=7,
        project_vocab_size=8,
        project_ignored_token_ids=(0, 6),
        return_full_log_probs=True,
        keep_full_log_probs_on_device=True,
    )
    teacher.model = _FakeNanoModel(dim=4, vocab_size=7)
    teacher.audio_rows = {}
    with torch.no_grad():
        teacher.model.ctc.ctc_lo.weight.zero_()
        teacher.model.ctc.ctc_lo.bias.copy_(torch.tensor([20.0, 5.0, 2.0, 1.0, 0.0, -1.0, 4.0]))

    record = teacher.feature_records(
        ["utt-projected"],
        torch.zeros(1, 3, 4),
        torch.tensor([3], dtype=torch.long),
    )["utt-projected"]

    full_log_probs = record["full_log_probs"].float()
    expected_allowed = torch.log_softmax(
        torch.tensor([5.0, 2.0, 1.0, 0.0, -1.0, 4.0]),
        dim=-1,
    )
    assert record["projected_distribution_normalized"] is True
    assert torch.isneginf(full_log_probs[:, 0]).all()
    assert torch.isneginf(full_log_probs[:, 6]).all()
    assert torch.allclose(
        torch.logsumexp(full_log_probs, dim=-1),
        torch.zeros(3),
        atol=2.0e-3,
    )
    assert torch.allclose(
        torch.as_tensor(record["blank_log_probs"]),
        expected_allowed[5].expand(3),
        atol=1.0e-6,
    )
    assert torch.allclose(
        full_log_probs[:, 7],
        expected_allowed[5].expand(3),
        atol=2.0e-3,
    )
    assert not torch.as_tensor(record["topk_token_ids"]).eq(0).any()
    assert not torch.as_tensor(record["topk_token_ids"]).eq(6).any()
    assert torch.as_tensor(record["argmax_token_ids"]).tolist() == [1]
