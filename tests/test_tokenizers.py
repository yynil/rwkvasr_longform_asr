import base64
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
    SenseVoiceTiktokenTokenizer,
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


class _EchoCharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    @property
    def eos_token_id(self) -> int | None:
        return None


def _install_fake_whisper(monkeypatch) -> None:
    fake_tokenizer_module = types.ModuleType("whisper.tokenizer")

    def fake_get_tokenizer(
        *, multilingual: bool, language: str | None = None, task: str | None = None
    ):
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


def test_manifest_dataset_defaults_to_whisper_tokenizer_for_text(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_whisper(monkeypatch)
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"utt_id": "utt-0", "text": "你好", "feature_path": "feat.pt"}) + "\n"
        )

    dataset = ASRManifestDataset(manifest_path)

    assert isinstance(dataset.tokenizer, WhisperMultilingualTokenizer)
    assert dataset.entries[0].token_ids == [101, 102, 103]


def test_manifest_dataset_ctc_normalization_reencodes_text_token_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "utt_id": "utt-0",
                    "language": "en",
                    "text": "Yorke's house, wasn't quiet!",
                    "token_ids": [999],
                    "feature_path": "feat.pt",
                }
            )
            + "\n"
        )

    dataset = ASRManifestDataset(
        manifest_path,
        tokenizer=_EchoCharTokenizer(),
        text_normalization="ctc",
    )

    assert dataset.entries[0].text == "yorkes house wasnt quiet"
    assert dataset.entries[0].token_ids == [ord(ch) for ch in "yorkes house wasnt quiet"]


def test_manifest_dataset_decoder_target_preserves_runtime_punctuation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "utt_id": "utt-0",
                    "language": "en",
                    "text": "Yorke's house, wasn't quiet!",
                    "feature_path": "feat.pt",
                }
            )
            + "\n"
        )

    dataset = ASRManifestDataset(
        manifest_path,
        tokenizer=_EchoCharTokenizer(),
        decoder_tokenizer=_EchoCharTokenizer(),
        text_normalization="ctc",
        decoder_text_normalization="runtime",
    )

    assert dataset.entries[0].text == "yorkes house wasnt quiet"
    assert dataset.entries[0].token_ids == [ord(ch) for ch in "yorkes house wasnt quiet"]
    assert dataset.entries[0].decoder_token_ids == [
        ord(ch) for ch in "yorke's house, wasn't quiet!"
    ]


def test_manifest_dataset_decoder_target_language_confirmation_can_correct_metadata(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "utt_id": "utt-0",
                    "language": "en",
                    "text": "你好，世界！",
                    "feature_path": "feat.pt",
                }
            )
            + "\n"
        )

    dataset = ASRManifestDataset(
        manifest_path,
        tokenizer=_EchoCharTokenizer(),
        decoder_tokenizer=_EchoCharTokenizer(),
        text_normalization="ctc",
        decoder_text_normalization="runtime",
        decoder_prompt_before_audio="User: The language label says this audio is {language_name}.\n",
        decoder_prompt_before_audio_use_language=True,
        decoder_target_prefix="{language_confirmation} ",
        decoder_target_prefix_use_language=True,
    )

    prompt = "".join(
        chr(token_id) for token_id in dataset.entries[0].decoder_prompt_before_audio_token_ids or []
    )
    target = "".join(chr(token_id) for token_id in dataset.entries[0].decoder_token_ids or [])

    assert "English" in prompt
    assert target == "这是中文文字。 你好，世界！"


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


def _write_byte_tiktoken_vocab(vocab_path: Path) -> None:
    with vocab_path.open("w", encoding="utf-8") as handle:
        for idx in range(256):
            token = base64.b64encode(bytes([idx])).decode("ascii")
            handle.write(f"{token} {idx}\n")


def test_sensevoice_tiktoken_tokenizer_uses_fun_asr_special_layout(tmp_path: Path) -> None:
    pytest.importorskip("tiktoken")
    vocab_path = tmp_path / "multilingual.tiktoken"
    _write_byte_tiktoken_vocab(vocab_path)

    tokenizer = SenseVoiceTiktokenTokenizer(str(vocab_path))
    token_ids = tokenizer.encode("Hello 你好")

    assert tokenizer.vocab_size == 1935
    assert tokenizer.eos_token_id is None
    assert token_ids
    assert all(0 <= int(token_id) < tokenizer.vocab_size for token_id in token_ids)
    assert tokenizer.decode(token_ids) == "Hello 你好"


def test_sensevoice_tiktoken_ctc_suppresses_non_pronunciation_units(tmp_path: Path) -> None:
    pytest.importorskip("tiktoken")
    vocab_path = tmp_path / "multilingual.tiktoken"
    _write_byte_tiktoken_vocab(vocab_path)

    tokenizer = SenseVoiceTiktokenTokenizer(str(vocab_path))
    special_tokens = tokenizer.processor._special_tokens
    nospeech_id = int(special_tokens["<|nospeech|>"])
    timestamp_id = int(special_tokens["<|0.00|>"])
    text_id = tokenizer.encode("a")[0]

    suppressed = set(tokenizer.ctc_suppressed_token_ids(blank_id=timestamp_id))

    assert nospeech_id in suppressed
    assert timestamp_id not in suppressed
    assert ord(" ") in suppressed
    assert 255 not in suppressed
    assert tokenizer.decode([255]) == ""
    assert text_id not in suppressed

    project_blank_id = tokenizer.vocab_size
    project_suppressed = set(tokenizer.ctc_suppressed_token_ids(blank_id=project_blank_id))
    assert project_blank_id not in project_suppressed
    assert all(int(token_id) in project_suppressed for token_id in special_tokens.values())
    assert timestamp_id in project_suppressed
    assert text_id not in project_suppressed


def test_real_sensevoice_tiktoken_preserves_composable_rare_chinese_bytes() -> None:
    pytest.importorskip("tiktoken")
    vocab_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "fun-asr-nano-2512"
        / "multilingual.tiktoken"
    )
    if not vocab_path.is_file():
        pytest.skip(f"canonical SenseVoice tokenizer is unavailable: {vocab_path}")
    tokenizer = SenseVoiceTiktokenTokenizer(str(vocab_path))
    text = "溦捘"
    token_ids = tokenizer.encode(text)
    suppressed = set(tokenizer.ctc_suppressed_token_ids(blank_id=60_515))

    assert tokenizer.decode(token_ids) == text
    assert not suppressed.intersection(token_ids)


def test_build_text_tokenizer_creates_sensevoice_tiktoken(tmp_path: Path) -> None:
    pytest.importorskip("tiktoken")
    vocab_path = tmp_path / "multilingual.tiktoken"
    _write_byte_tiktoken_vocab(vocab_path)

    tokenizer = build_text_tokenizer("sensevoice_tiktoken", model_path=str(vocab_path))

    assert isinstance(tokenizer, SenseVoiceTiktokenTokenizer)
    assert tokenizer.vocab_size == 1935


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
        handle.write(
            json.dumps({"utt_id": "utt-0", "text": "你好", "feature_path": "feat.pt"}) + "\n"
        )

    dataset = ASRManifestDataset(
        manifest_path,
        tokenizer=RWKVTokenizer(str(vocab_path)),
        append_eos=True,
    )

    assert dataset.entries[0].token_ids[-1] == 0
