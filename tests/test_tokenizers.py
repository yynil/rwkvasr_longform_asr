import json
import sys
import types
from pathlib import Path

import pytest

from rwkvasr.data import (
    ASRManifestDataset,
    QwenTokenizer,
    RWKVTokenizer,
    WhisperMultilingualTokenizer,
    build_text_tokenizer,
    maybe_append_eos_token_ids,
)


class _FakeWhisperProcessor:
    eot = 50000

    class _FakeEncoding:
        @staticmethod
        def decode_bytes(token_ids: list[int]) -> bytes:
            return f"decoded:{','.join(str(token_id) for token_id in token_ids)}".encode("utf-8")

    encoding = _FakeEncoding()

    def encode(self, text: str) -> list[int]:
        if text == "<special>":
            return [50001]
        return [101, 102, 103]

    def decode(self, token_ids: list[int]) -> str:
        return f"decoded:{','.join(str(token_id) for token_id in token_ids)}"


def _install_fake_whisper(monkeypatch) -> None:
    fake_tokenizer_module = types.ModuleType("whisper.tokenizer")

    def fake_get_tokenizer(*, multilingual: bool, language: str | None = None, task: str | None = None):
        assert multilingual is True
        assert language in {None, "zh"}
        assert task in {None, "transcribe"}
        return _FakeWhisperProcessor()

    fake_tokenizer_module.get_tokenizer = fake_get_tokenizer  # type: ignore[attr-defined]
    fake_whisper_module = types.ModuleType("whisper")
    fake_whisper_module.tokenizer = fake_tokenizer_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper_module)
    monkeypatch.setitem(sys.modules, "whisper.tokenizer", fake_tokenizer_module)


class _FakeQwenEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class _FakeQwenProcessor:
    @staticmethod
    def from_file(_path: str):
        return _FakeQwenProcessor()

    def encode(self, text: str) -> _FakeQwenEncoding:
        if text == "你好":
            return _FakeQwenEncoding([11, 12])
        return _FakeQwenEncoding([21, 22, 23])

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return f"qwen:{','.join(str(token_id) for token_id in token_ids)}"

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        assert with_added_tokens is True
        return 151643


def _install_fake_qwen(monkeypatch) -> None:
    fake_module = types.ModuleType("tokenizers")
    fake_module.Tokenizer = _FakeQwenProcessor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenizers", fake_module)


def test_whisper_multilingual_tokenizer_uses_text_only_vocab(monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    tokenizer = WhisperMultilingualTokenizer(language="zh", task="transcribe")

    assert tokenizer.vocab_size == 50000
    assert tokenizer.encode("你好") == [101, 102, 103]
    assert tokenizer.decode([1, 2]) == "decoded:1,2"

    with pytest.raises(ValueError):
        tokenizer.encode("<special>")


def test_build_text_tokenizer_creates_whisper_default(monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)

    tokenizer = build_text_tokenizer("whisper_multilingual")

    assert isinstance(tokenizer, WhisperMultilingualTokenizer)
    assert tokenizer.vocab_size == 50000


def test_manifest_dataset_defaults_to_whisper_tokenizer_for_text(monkeypatch, tmp_path: Path) -> None:
    _install_fake_whisper(monkeypatch)
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"utt_id": "utt-0", "text": "你好", "feature_path": "feat.pt"}) + "\n")

    dataset = ASRManifestDataset(manifest_path)

    assert isinstance(dataset.tokenizer, WhisperMultilingualTokenizer)
    assert dataset.entries[0].token_ids == [101, 102, 103]


def test_qwen_tokenizer_uses_tokenizer_json(monkeypatch) -> None:
    _install_fake_qwen(monkeypatch)

    tokenizer = QwenTokenizer("fake-tokenizer.json")

    assert tokenizer.vocab_size == 151643
    assert tokenizer.encode("你好") == [11, 12]
    assert tokenizer.decode([1, 2]) == "qwen:1,2"


def test_build_text_tokenizer_creates_qwen_tokenizer(monkeypatch) -> None:
    _install_fake_qwen(monkeypatch)

    tokenizer = build_text_tokenizer("qwen3", model_path="fake-tokenizer.json")

    assert isinstance(tokenizer, QwenTokenizer)
    assert tokenizer.vocab_size == 151643


def test_real_whisper_multilingual_tokenizer_encodes_text_when_dependency_is_available() -> None:
    pytest.importorskip("whisper.tokenizer")

    tokenizer = build_text_tokenizer("whisper_multilingual")
    token_ids = tokenizer.encode("hello 世界")

    assert token_ids
    assert all(int(token_id) < tokenizer.vocab_size for token_id in token_ids)


def test_real_whisper_multilingual_tokenizer_decode_strips_invalid_utf8_fragments() -> None:
    pytest.importorskip("whisper.tokenizer")

    tokenizer = build_text_tokenizer("whisper_multilingual")
    decoded = tokenizer.decode([126, 220])

    assert "\ufffd" not in decoded


def test_rwkv_tokenizer_uses_official_vocab_file_roundtrip() -> None:
    vocab_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "RWKV-LM"
        / "RWKV-v7"
        / "rwkv_vocab_v20230424.txt"
    )
    if not vocab_path.exists():
        pytest.skip(f"RWKV vocab not available at {vocab_path}")

    tokenizer = RWKVTokenizer(str(vocab_path))
    token_ids = tokenizer.encode("Hello 你好")

    assert tokenizer.decode(token_ids) == "Hello 你好"
    assert tokenizer.eos_token_id == 0
    assert tokenizer.decode([0, *token_ids, 65535]) == "Hello 你好"
    assert tokenizer.vocab_size == 65536


def test_build_text_tokenizer_creates_rwkv_tokenizer() -> None:
    vocab_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "RWKV-LM"
        / "RWKV-v7"
        / "rwkv_vocab_v20230424.txt"
    )
    if not vocab_path.exists():
        pytest.skip(f"RWKV vocab not available at {vocab_path}")

    tokenizer = build_text_tokenizer("rwkv", model_path=str(vocab_path))

    assert isinstance(tokenizer, RWKVTokenizer)
    assert tokenizer.vocab_size == 65536


def test_maybe_append_eos_token_ids_is_idempotent_for_rwkv() -> None:
    vocab_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "RWKV-LM"
        / "RWKV-v7"
        / "rwkv_vocab_v20230424.txt"
    )
    if not vocab_path.exists():
        pytest.skip(f"RWKV vocab not available at {vocab_path}")

    tokenizer = RWKVTokenizer(str(vocab_path))
    token_ids = tokenizer.encode("Hello")

    appended = maybe_append_eos_token_ids(token_ids, append_eos=True, tokenizer=tokenizer)
    appended_twice = maybe_append_eos_token_ids(appended, append_eos=True, tokenizer=tokenizer)

    assert appended[-1] == 0
    assert appended_twice == appended


def test_manifest_dataset_can_append_rwkv_eos(tmp_path: Path) -> None:
    vocab_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "RWKV-LM"
        / "RWKV-v7"
        / "rwkv_vocab_v20230424.txt"
    )
    if not vocab_path.exists():
        pytest.skip(f"RWKV vocab not available at {vocab_path}")

    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"utt_id": "utt-0", "text": "你好", "feature_path": "feat.pt"}) + "\n")

    dataset = ASRManifestDataset(
        manifest_path,
        tokenizer=RWKVTokenizer(str(vocab_path)),
        append_eos=True,
    )

    assert dataset.entries[0].token_ids[-1] == 0
