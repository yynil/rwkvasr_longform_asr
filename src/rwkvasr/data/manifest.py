from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from rwkvasr.modules import WenetFbankConfig, compute_wenet_fbank


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    token_ids: list[int]
    text: str | None = None
    feature_path: str | None = None
    audio_filepath: str | None = None
    num_frames: int | None = None


@dataclass
class ASRBatch:
    features: Tensor
    feature_lengths: Tensor
    targets: Tensor
    target_lengths: Tensor
    utt_ids: list[str]

    def prefix(self, num_samples: int) -> "ASRBatch":
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        batch_size = int(self.features.size(0))
        if num_samples >= batch_size:
            return self
        feature_lengths = self.feature_lengths[:num_samples]
        max_feature_length = int(feature_lengths.max().item())
        target_lengths = self.target_lengths[:num_samples]
        total_targets = int(target_lengths.sum().item())
        return ASRBatch(
            features=self.features[:num_samples, :max_feature_length].contiguous(),
            feature_lengths=feature_lengths.contiguous(),
            targets=self.targets[:total_targets].contiguous(),
            target_lengths=target_lengths.contiguous(),
            utt_ids=self.utt_ids[:num_samples],
        )

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype | None = None,
    ) -> "ASRBatch":
        features = self.features.to(device)
        if feature_dtype is not None and features.is_floating_point():
            features = features.to(dtype=feature_dtype)
        return ASRBatch(
            features=features,
            feature_lengths=self.feature_lengths.to(device),
            targets=self.targets.to(device),
            target_lengths=self.target_lengths.to(device),
            utt_ids=self.utt_ids,
        )


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_token_id(self) -> int | None: ...


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.processor = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, token_ids: list[int]) -> str:
        return str(self.processor.decode(token_ids))

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    @property
    def eos_token_id(self) -> int | None:
        eos_id = int(self.processor.eos_id())
        return None if eos_id < 0 else eos_id


class WhisperMultilingualTokenizer:
    def __init__(self, *, language: str | None = None, task: str | None = None):
        try:
            from whisper.tokenizer import get_tokenizer
        except ImportError as exc:
            raise ImportError(
                "Whisper tokenizer support requires the `openai-whisper` package. "
                "Run `uv sync` after updating dependencies."
            ) from exc

        self.processor = get_tokenizer(multilingual=True, language=language, task=task)
        self._text_vocab_size = int(self.processor.eot)

    def encode(self, text: str) -> list[int]:
        token_ids = list(self.processor.encode(text))
        if any(int(token_id) >= self._text_vocab_size for token_id in token_ids):
            raise ValueError("Text encoded to a Whisper special token outside the CTC text vocabulary.")
        return [int(token_id) for token_id in token_ids]

    def decode(self, token_ids: list[int]) -> str:
        token_ids = [int(token_id) for token_id in token_ids if int(token_id) < self._text_vocab_size]
        if hasattr(self.processor.encoding, "decode_bytes"):
            raw_bytes = self.processor.encoding.decode_bytes(token_ids)
            return raw_bytes.decode("utf-8", errors="ignore")
        return str(self.processor.decode(token_ids)).replace("\ufffd", "")

    @property
    def vocab_size(self) -> int:
        return self._text_vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return None


class QwenTokenizer:
    def __init__(self, model_path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "Qwen tokenizer support requires the `tokenizers` package. "
                "Run `uv sync` after updating dependencies."
            ) from exc

        self.processor = Tokenizer.from_file(model_path)
        self._vocab_size = int(self.processor.get_vocab_size(with_added_tokens=True))

    def encode(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.processor.encode(text).ids]

    def decode(self, token_ids: list[int]) -> str:
        return str(
            self.processor.decode([int(token_id) for token_id in token_ids], skip_special_tokens=True)
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return None


class RWKVTokenizer:
    OFFICIAL_VOCAB_SIZE = 65536
    OFFICIAL_EOS_TOKEN_ID = 0

    def __init__(self, model_path: str):
        self.idx2token: dict[int, bytes] = {}
        sorted_tokens: list[bytes] = []
        lines = Path(model_path).read_text(encoding="utf-8").splitlines()
        for line in lines:
            idx = int(line[: line.index(" ")])
            token = eval(line[line.index(" ") : line.rindex(" ")])
            token = token.encode("utf-8") if isinstance(token, str) else token
            if not isinstance(token, bytes):
                raise TypeError(f"RWKV vocab entry {idx} is not bytes: {type(token)}")
            sorted_tokens.append(token)
            self.idx2token[idx] = token

        self.token2idx = {token: int(idx) for idx, token in self.idx2token.items()}
        self._vocab_size = max(self.OFFICIAL_VOCAB_SIZE, (max(self.idx2token) + 1) if self.idx2token else 0)
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for token in reversed(sorted_tokens):
            if len(token) < 2:
                continue
            s0 = int(token[0])
            s1 = int(token[1])
            self.table[s0][s1].append(token)
            self.wlen[s0] = max(self.wlen[s0], len(token))
            self.good[s0].add(s1)

    def encode_bytes(self, src: bytes) -> list[int]:
        src_len = len(src)
        tokens: list[int] = []
        i = 0
        while i < src_len:
            token = src[i : i + 1]
            if i < src_len - 1:
                s0 = int(src[i])
                s1 = int(src[i + 1])
                if s1 in self.good[s0]:
                    candidate = src[i : i + self.wlen[s0]]
                    try:
                        token = next(filter(candidate.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[token])
            i += len(token)
        return tokens

    def decode_bytes(self, token_ids: list[int]) -> bytes:
        eos_id = self.eos_token_id
        parts: list[bytes] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if eos_id is not None and token_id == eos_id:
                continue
            parts.append(self.idx2token.get(token_id, b""))
        return b"".join(parts)

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return self.decode_bytes(token_ids).decode("utf-8", errors="ignore")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self.OFFICIAL_EOS_TOKEN_ID


def tokenizer_eos_token_id(tokenizer: TokenizerLike | None) -> int | None:
    if tokenizer is None:
        return None
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return None
    return int(eos_token_id)


def maybe_append_eos_token_ids(
    token_ids: list[int] | tuple[int, ...],
    *,
    append_eos: bool,
    tokenizer: TokenizerLike | None = None,
) -> list[int]:
    normalized = [int(token_id) for token_id in token_ids]
    if not append_eos:
        return normalized
    eos_token_id = tokenizer_eos_token_id(tokenizer)
    if eos_token_id is None:
        raise ValueError("append_eos=True requires a tokenizer with a defined eos_token_id.")
    if normalized and normalized[-1] == eos_token_id:
        return normalized
    return [*normalized, eos_token_id]


def build_text_tokenizer(
    tokenizer_type: str,
    *,
    model_path: str | None = None,
    language: str | None = None,
    task: str | None = None,
) -> TokenizerLike:
    if tokenizer_type == "sentencepiece":
        if model_path is None:
            raise ValueError("SentencePiece tokenizer requires model_path.")
        return SentencePieceTokenizer(model_path)
    if tokenizer_type in {"whisper", "whisper_multilingual"}:
        return WhisperMultilingualTokenizer(language=language, task=task)
    if tokenizer_type in {"qwen", "qwen3"}:
        if model_path is None:
            raise ValueError("Qwen tokenizer requires model_path.")
        return QwenTokenizer(model_path)
    if tokenizer_type in {"rwkv", "rwkv_v20230424"}:
        if model_path is None:
            raise ValueError("RWKV tokenizer requires model_path.")
        return RWKVTokenizer(model_path)
    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")


class LogMelFeatureExtractor:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        dither: float = 0.0,
    ):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.dither = dither

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        import torchaudio

        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate
        features = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            sample_frequency=sample_rate,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=self.dither,
            use_energy=False,
        )
        return features


class WenetFbankFeatureExtractor:
    def __init__(self, config: WenetFbankConfig | None = None):
        self.config = config or WenetFbankConfig()

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        import torchaudio

        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.config.sample_rate)
            sample_rate = self.config.sample_rate
        return compute_wenet_fbank(waveform, sample_rate, self.config).float()


class ASRManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        tokenizer: TokenizerLike | None = None,
        feature_extractor: LogMelFeatureExtractor | None = None,
        append_eos: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.append_eos = bool(append_eos)
        self.entries = self._load_entries()

    def _load_entries(self) -> list[ManifestEntry]:
        entries: list[ManifestEntry] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                utt_id = raw.get("utt_id") or raw.get("id") or raw.get("audio_id")
                if utt_id is None:
                    raise ValueError("Manifest entry must contain utt_id, id, or audio_id.")
                token_ids = raw.get("token_ids")
                text = raw.get("text")
                if token_ids is None:
                    if text is None:
                        raise ValueError("Manifest entry needs token_ids or text with a tokenizer.")
                    if self.tokenizer is None:
                        self.tokenizer = build_text_tokenizer("whisper_multilingual")
                    token_ids = self.tokenizer.encode(text)
                token_ids = maybe_append_eos_token_ids(
                    token_ids,
                    append_eos=self.append_eos,
                    tokenizer=self.tokenizer,
                )
                entries.append(
                    ManifestEntry(
                        utt_id=str(utt_id),
                        token_ids=token_ids,
                        text=text,
                        feature_path=raw.get("feature_path"),
                        audio_filepath=raw.get("audio_filepath"),
                        num_frames=raw.get("num_frames"),
                    )
                )
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def _load_features(self, entry: ManifestEntry) -> Tensor:
        if entry.feature_path is not None:
            feature_path = Path(entry.feature_path)
            if not feature_path.is_absolute():
                feature_path = self.root / feature_path
            features = torch.load(feature_path, map_location="cpu")
            if not isinstance(features, Tensor):
                raise TypeError(f"Expected Tensor features in {feature_path}, got {type(features)}")
            return features.float()

        if entry.audio_filepath is not None:
            import torchaudio

            audio_path = Path(entry.audio_filepath)
            if not audio_path.is_absolute():
                audio_path = self.root / audio_path
            waveform, sample_rate = torchaudio.load(audio_path)
            return self.feature_extractor(waveform, sample_rate).float()

        raise ValueError("Manifest entry must contain either feature_path or audio_filepath.")

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        features = self._load_features(entry)
        targets = torch.tensor(entry.token_ids, dtype=torch.long)
        return {
            "utt_id": entry.utt_id,
            "features": features,
            "feature_length": features.size(0),
            "targets": targets,
            "target_length": targets.numel(),
            "text": entry.text,
        }


class FeatureCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> ASRBatch:
        if not samples:
            raise ValueError("samples must not be empty")

        batch_size = len(samples)
        feat_dim = samples[0]["features"].size(-1)
        max_frames = max(int(sample["feature_length"]) for sample in samples)
        total_targets = sum(int(sample["target_length"]) for sample in samples)

        features = torch.zeros(batch_size, max_frames, feat_dim, dtype=samples[0]["features"].dtype)
        feature_lengths = torch.zeros(batch_size, dtype=torch.long)
        targets = torch.zeros(total_targets, dtype=torch.long)
        target_lengths = torch.zeros(batch_size, dtype=torch.long)
        utt_ids: list[str] = []

        offset = 0
        for idx, sample in enumerate(samples):
            feat = sample["features"]
            feature_len = int(sample["feature_length"])
            target = sample["targets"]
            target_len = int(sample["target_length"])

            features[idx, :feature_len] = feat
            feature_lengths[idx] = feature_len
            targets[offset : offset + target_len] = target
            target_lengths[idx] = target_len
            utt_ids.append(str(sample["utt_id"]))
            offset += target_len

        return ASRBatch(
            features=features,
            feature_lengths=feature_lengths,
            targets=targets,
            target_lengths=target_lengths,
            utt_ids=utt_ids,
        )
