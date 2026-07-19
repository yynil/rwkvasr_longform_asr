from __future__ import annotations

import pytest
import torch
from torch import nn

from rwkvasr.training.funasr_online_teacher import (
    FunASRNanoCTCTopKOnlineTeacher,
    FunASROnlineCTCTeacherConfig,
    _capture_funasr_encoder_hiddens,
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


class _FakeCTCDecoder(nn.Module):
    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
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


def test_capture_funasr_encoder_hiddens_rejects_invalid_layer() -> None:
    encoder = _FakeNanoEncoder(dim=4)

    with pytest.raises(ValueError, match="out of range"):
        with _capture_funasr_encoder_hiddens(encoder, (3,), capture_input=False):
            pass


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
