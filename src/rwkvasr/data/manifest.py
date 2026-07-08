from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from rwkvasr.modules import (
    Qwen3ASRFeatureConfig,
    WenetFbankConfig,
    compute_qwen3_asr_log_mel,
    compute_wenet_fbank,
)

from .text_normalization import normalize_asr_text


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    token_ids: list[int]
    text: str | None = None
    language: str | None = None
    decoder_token_ids: list[int] | None = None
    decoder_prompt_before_audio_token_ids: list[int] | None = None
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
    decoder_targets: Tensor | None = None
    decoder_target_lengths: Tensor | None = None
    decoder_prompt_before_audio: Tensor | None = None
    decoder_prompt_before_audio_lengths: Tensor | None = None
    ctc_teacher_audio_rows: list[dict[str, Any] | None] | None = None

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
        decoder_targets = None
        decoder_target_lengths = None
        decoder_prompt_before_audio = None
        decoder_prompt_before_audio_lengths = None
        if self.decoder_targets is not None and self.decoder_target_lengths is not None:
            decoder_target_lengths = self.decoder_target_lengths[:num_samples]
            total_decoder_targets = int(decoder_target_lengths.sum().item())
            decoder_targets = self.decoder_targets[:total_decoder_targets].contiguous()
        if (
            self.decoder_prompt_before_audio is not None
            and self.decoder_prompt_before_audio_lengths is not None
        ):
            decoder_prompt_before_audio_lengths = self.decoder_prompt_before_audio_lengths[:num_samples]
            total_decoder_prompt_tokens = int(decoder_prompt_before_audio_lengths.sum().item())
            decoder_prompt_before_audio = self.decoder_prompt_before_audio[
                :total_decoder_prompt_tokens
            ].contiguous()
        return ASRBatch(
            features=self.features[:num_samples, :max_feature_length].contiguous(),
            feature_lengths=feature_lengths.contiguous(),
            targets=self.targets[:total_targets].contiguous(),
            target_lengths=target_lengths.contiguous(),
            utt_ids=self.utt_ids[:num_samples],
            decoder_targets=decoder_targets,
            decoder_target_lengths=(
                decoder_target_lengths.contiguous() if decoder_target_lengths is not None else None
            ),
            decoder_prompt_before_audio=decoder_prompt_before_audio,
            decoder_prompt_before_audio_lengths=(
                decoder_prompt_before_audio_lengths.contiguous()
                if decoder_prompt_before_audio_lengths is not None
                else None
            ),
            ctc_teacher_audio_rows=(
                self.ctc_teacher_audio_rows[:num_samples]
                if self.ctc_teacher_audio_rows is not None
                else None
            ),
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
            decoder_targets=(
                self.decoder_targets.to(device) if self.decoder_targets is not None else None
            ),
            decoder_target_lengths=(
                self.decoder_target_lengths.to(device)
                if self.decoder_target_lengths is not None
                else None
            ),
            decoder_prompt_before_audio=(
                self.decoder_prompt_before_audio.to(device)
                if self.decoder_prompt_before_audio is not None
                else None
            ),
            decoder_prompt_before_audio_lengths=(
                self.decoder_prompt_before_audio_lengths.to(device)
                if self.decoder_prompt_before_audio_lengths is not None
                else None
            ),
            ctc_teacher_audio_rows=self.ctc_teacher_audio_rows,
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


class SenseVoiceTiktokenTokenizer:
    LANGUAGE_CODES = (
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "tr",
        "pl",
        "ca",
        "nl",
        "ar",
        "sv",
        "it",
        "id",
        "hi",
        "fi",
        "vi",
        "he",
        "uk",
        "el",
        "ms",
        "cs",
        "ro",
        "da",
        "hu",
        "ta",
        "no",
        "th",
        "ur",
        "hr",
        "bg",
        "lt",
        "la",
        "mi",
        "ml",
        "cy",
        "sk",
        "te",
        "fa",
        "lv",
        "bn",
        "sr",
        "az",
        "sl",
        "kn",
        "et",
        "mk",
        "br",
        "eu",
        "is",
        "hy",
        "ne",
        "mn",
        "bs",
        "kk",
        "sq",
        "sw",
        "gl",
        "mr",
        "pa",
        "si",
        "km",
        "sn",
        "yo",
        "so",
        "af",
        "oc",
        "ka",
        "be",
        "tg",
        "sd",
        "gu",
        "am",
        "yi",
        "lo",
        "uz",
        "fo",
        "ht",
        "ps",
        "tk",
        "nn",
        "mt",
        "sa",
        "lb",
        "my",
        "bo",
        "tl",
        "mg",
        "as",
        "tt",
        "haw",
        "ln",
        "ha",
        "ba",
        "jw",
        "su",
        "yue",
        "minnan",
        "wuyu",
        "dialect",
        "zh/en",
        "en/zh",
    )
    AUDIO_EVENTS = (
        "ASR",
        "AED",
        "SER",
        "Speech",
        "/Speech",
        "BGM",
        "/BGM",
        "Laughter",
        "/Laughter",
        "Applause",
        "/Applause",
    )
    EMOTIONS = ("HAPPY", "SAD", "ANGRY", "NEUTRAL")
    PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, model_path: str, *, num_languages: int = 8749):
        try:
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe
        except ImportError as exc:
            raise ImportError(
                "SenseVoice tiktoken support requires the `tiktoken` package. "
                "Run `uv sync` after updating dependencies."
            ) from exc

        mergeable_ranks = load_tiktoken_bpe(model_path)
        special_tokens: dict[str, int] = {}
        next_token_id = len(mergeable_ranks)
        specials = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            *[f"<|{language}|>" for language in self.LANGUAGE_CODES[:num_languages]],
            *[f"<|{audio_event}|>" for audio_event in self.AUDIO_EVENTS],
            *[f"<|{emotion}|>" for emotion in self.EMOTIONS],
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
            *[f"<|SPECIAL_TOKEN_{idx}|>" for idx in range(1, 51)],
            *[f"<|{idx * 0.02:.2f}|>" for idx in range(1501)],
        ]
        for token in specials:
            special_tokens[token] = next_token_id
            next_token_id += 1
        self._special_token_ids = frozenset(int(token_id) for token_id in special_tokens.values())

        self.processor = tiktoken.Encoding(
            name=Path(model_path).name,
            explicit_n_vocab=next_token_id,
            pat_str=self.PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        self._vocab_size = int(self.processor.n_vocab)
        self._timestamp_begin = int(special_tokens["<|0.00|>"])

    def encode(self, text: str) -> list[int]:
        return [
            int(token_id)
            for token_id in self.processor.encode(
                text,
                allowed_special=set(),
                disallowed_special=(),
            )
        ]

    def decode(self, token_ids: list[int]) -> str:
        filtered = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < self._timestamp_begin]
        return str(self.processor.decode(filtered))

    def ctc_suppressed_token_ids(self, *, blank_id: int | None = None) -> tuple[int, ...]:
        suppressed: list[int] = []
        blank = None if blank_id is None else int(blank_id)
        for token_id in range(int(self._vocab_size)):
            if blank is not None and token_id == blank:
                continue
            if token_id in self._special_token_ids:
                suppressed.append(token_id)
                continue
            try:
                text = str(self.processor.decode([token_id]))
            except Exception:
                suppressed.append(token_id)
                continue
            if "\ufffd" in text:
                suppressed.append(token_id)
                continue
            if not normalize_asr_text(text, mode="ctc").strip():
                suppressed.append(token_id)
        return tuple(suppressed)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

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
    if tokenizer_type in {"sensevoice_tiktoken", "sensevoice", "funasr_tiktoken"}:
        if model_path is None:
            raise ValueError("SenseVoice tiktoken tokenizer requires model_path.")
        return SenseVoiceTiktokenTokenizer(model_path)
    if tokenizer_type in {"qwen", "qwen3"}:
        if model_path is None:
            raise ValueError("Qwen tokenizer requires model_path.")
        return QwenTokenizer(model_path)
    if tokenizer_type in {"rwkv", "rwkv_v20230424"}:
        if model_path is None:
            raise ValueError("RWKV tokenizer requires model_path.")
        return RWKVTokenizer(model_path)
    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")


def ctc_suppressed_token_ids_for_tokenizer(
    tokenizer: TokenizerLike,
    *,
    blank_id: int | None = None,
) -> tuple[int, ...]:
    resolver = getattr(tokenizer, "ctc_suppressed_token_ids", None)
    if not callable(resolver):
        return ()
    return tuple(int(token_id) for token_id in resolver(blank_id=blank_id))


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


def apply_lfr_stacking(features: Tensor, *, lfr_m: int = 7, lfr_n: int = 6) -> Tensor:
    if features.dim() != 2:
        raise ValueError(f"LFR stacking expects [frames, bins] features, got {tuple(features.shape)}.")
    lfr_m = int(lfr_m)
    lfr_n = int(lfr_n)
    if lfr_m <= 0 or lfr_n <= 0:
        raise ValueError("lfr_m and lfr_n must be positive.")
    num_frames = int(features.size(0))
    if num_frames <= 0:
        return features.new_zeros(0, int(features.size(1)) * lfr_m)
    out_frames = (num_frames + lfr_n - 1) // lfr_n
    stacked: list[Tensor] = []
    for frame_idx in range(out_frames):
        start = frame_idx * lfr_n
        end = start + lfr_m
        if end <= num_frames:
            chunk = features[start:end]
        else:
            pad = features[-1:].expand(end - num_frames, -1)
            chunk = torch.cat([features[start:num_frames], pad], dim=0)
        stacked.append(chunk.reshape(-1))
    return torch.stack(stacked, dim=0)


class SenseVoiceLFRFbankFeatureExtractor:
    def __init__(
        self,
        *,
        num_mel_bins: int = 80,
        lfr_m: int = 7,
        lfr_n: int = 6,
    ):
        self.base = WenetFbankFeatureExtractor(
            WenetFbankConfig(
                num_mel_bins=int(num_mel_bins),
                dither=0.0,
                window_type="hamming",
            )
        )
        self.lfr_m = int(lfr_m)
        self.lfr_n = int(lfr_n)

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        features = self.base(waveform, sample_rate)
        return apply_lfr_stacking(features, lfr_m=self.lfr_m, lfr_n=self.lfr_n).float()


class FunASRWavFrontendFeatureExtractor:
    """Feature extractor matching FunASR-Nano's WavFrontend, including its LFR padding."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 1.0,
    ):
        try:
            from funasr.frontends.wav_frontend import WavFrontend
        except ImportError as exc:  # pragma: no cover - dependency is present in the project env.
            raise ImportError("funasr is required for feature_extractor_type='funasr_wav_frontend'.") from exc
        self.sample_rate = int(sample_rate)
        self.frontend = WavFrontend(
            fs=int(sample_rate),
            window="hamming",
            n_mels=int(num_mel_bins),
            frame_length=25,
            frame_shift=10,
            lfr_m=int(lfr_m),
            lfr_n=int(lfr_n),
            cmvn_file=None,
            dither=float(dither),
            snip_edges=True,
            upsacle_samples=True,
        )

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        import torchaudio

        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        elif waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        if waveform.dim() != 1:
            raise ValueError(f"Expected audio waveform with shape [time] or [channels, time], got {tuple(waveform.shape)}.")
        if int(sample_rate) != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                int(sample_rate),
                self.sample_rate,
            ).squeeze(0)
        lengths = torch.tensor([int(waveform.numel())], dtype=torch.long)
        features, feature_lengths = self.frontend(waveform.unsqueeze(0), lengths)
        return features[0, : int(feature_lengths[0].item())].float()


class Qwen3ASRFeatureExtractor:
    def __init__(self, config: Qwen3ASRFeatureConfig | None = None):
        self.config = config or Qwen3ASRFeatureConfig()

    def __call__(self, waveform: Tensor, sample_rate: int) -> Tensor:
        import torchaudio

        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        elif waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        if waveform.dim() != 1:
            raise ValueError(f"Expected audio waveform with shape [time] or [channels, time], got {tuple(waveform.shape)}.")
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sample_rate,
                self.config.sample_rate,
            ).squeeze(0)
            sample_rate = self.config.sample_rate
        return compute_qwen3_asr_log_mel(waveform, sample_rate, self.config).float()


def build_audio_feature_extractor(
    feature_extractor_type: str,
    *,
    input_dim: int,
) -> WenetFbankFeatureExtractor | Qwen3ASRFeatureExtractor | SenseVoiceLFRFbankFeatureExtractor | FunASRWavFrontendFeatureExtractor:
    normalized = feature_extractor_type.lower().replace("-", "_")
    if normalized in {"wenet", "wenet_fbank", "fbank"}:
        return WenetFbankFeatureExtractor(WenetFbankConfig(num_mel_bins=int(input_dim)))
    if normalized in {"qwen3", "qwen3_asr", "whisper", "whisper_log_mel"}:
        return Qwen3ASRFeatureExtractor(Qwen3ASRFeatureConfig(feature_size=int(input_dim)))
    if normalized in {"sensevoice", "sensevoice_fbank", "sensevoice_lfr_fbank"}:
        if int(input_dim) != 560:
            raise ValueError("sensevoice_lfr_fbank emits 560-dimensional features; set input_dim=560.")
        return SenseVoiceLFRFbankFeatureExtractor(num_mel_bins=80, lfr_m=7, lfr_n=6)
    if normalized in {"funasr_wav_frontend", "funasr_nano_wav_frontend"}:
        if int(input_dim) != 560:
            raise ValueError("funasr_wav_frontend emits 560-dimensional features; set input_dim=560.")
        return FunASRWavFrontendFeatureExtractor(num_mel_bins=80, lfr_m=7, lfr_n=6)
    raise ValueError(f"Unsupported feature_extractor_type: {feature_extractor_type}")


def load_audio_waveform(audio_path: str | Path) -> tuple[Tensor, int]:
    path = Path(audio_path)
    try:
        import torchaudio

        return torchaudio.load(path)
    except (ImportError, RuntimeError, OSError):
        pass

    try:
        import soundfile as sf

        audio_array, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(audio_array.T.copy()), int(sample_rate)
    except Exception:
        pass

    return _load_audio_with_ffmpeg(path)


def _load_audio_with_ffmpeg(audio_path: Path) -> tuple[Tensor, int]:
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"Could not load audio {audio_path}: torchaudio/soundfile failed and ffmpeg is unavailable."
        )

    sample_rate = 16000
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    env = os.environ.copy()
    loader_paths = ["/usr/lib/x86_64-linux-gnu/blas", "/usr/lib/x86_64-linux-gnu/lapack"]
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join([*loader_paths, existing] if existing else loader_paths)
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {audio_path}: {message}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg decoded no samples from {audio_path}")
    return torch.from_numpy(audio).unsqueeze(0), sample_rate


_LANGUAGE_NAME_BY_CODE = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh_cn": "Chinese",
    "zh-hans": "Chinese",
    "zh_hans": "Chinese",
    "zh-tw": "Chinese",
    "zh_tw": "Chinese",
    "zh-hant": "Chinese",
    "zh_hant": "Chinese",
}


def _normalize_language_code(language: str | None) -> str:
    return (language or "").strip().lower()


def decoder_language_name(language: str | None) -> str:
    code = _normalize_language_code(language)
    if code in _LANGUAGE_NAME_BY_CODE:
        return _LANGUAGE_NAME_BY_CODE[code]
    prefix = code.split("-", 1)[0].split("_", 1)[0]
    if prefix in _LANGUAGE_NAME_BY_CODE:
        return _LANGUAGE_NAME_BY_CODE[prefix]
    return code or "the same language as the audio"


def decoder_language_confirmation(
    language: str | None,
    *,
    english: str = "This is English text.",
    chinese: str = "这是中文文字。",
) -> str:
    language_name = decoder_language_name(language)
    normalized_name = language_name.lower()
    if normalized_name == "chinese":
        return chinese
    if normalized_name == "english":
        return english
    return f"This is {language_name} text."


def infer_decoder_language_from_text(text: str | None, *, fallback: str | None = None) -> str | None:
    if text is None:
        return fallback
    cjk_count, latin_count = _count_text_scripts(str(text))
    if cjk_count > 0 and cjk_count >= latin_count:
        return "zh"
    if latin_count > 0:
        return "en"
    return fallback


def maybe_flip_decoder_prompt_language_label(
    language: str | None,
    *,
    sample_id: str,
    probability: float,
    seed: int,
) -> str | None:
    probability = float(probability)
    if probability <= 0.0:
        return language
    if probability > 1.0:
        raise ValueError("decoder_prompt_language_label_noise_prob must be <= 1.0")
    language_name = decoder_language_name(language)
    normalized_name = language_name.lower()
    if normalized_name not in {"english", "chinese"}:
        return language
    digest = hashlib.sha1(f"{int(seed)}:{sample_id}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)
    if draw >= probability:
        return language
    return "zh" if normalized_name == "english" else "en"


def _stable_unit_interval(*, seed: int, sample_id: str, salt: str) -> float:
    digest = hashlib.sha1(f"{int(seed)}:{salt}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def _decoder_language_family(language: str | None) -> str | None:
    name = decoder_language_name(language).lower()
    if name == "english":
        return "en"
    if name == "chinese":
        return "zh"
    return None


def _count_text_scripts(text: str) -> tuple[int, int]:
    cjk_count = 0
    latin_count = 0
    for ch in text:
        codepoint = ord(ch)
        if 0x4E00 <= codepoint <= 0x9FFF:
            cjk_count += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin_count += 1
    return cjk_count, latin_count


def _ctc_draft_has_reference_language_mismatch(draft: str, *, reference_family: str | None) -> bool:
    if reference_family is None:
        return False
    draft_cjk_count, draft_latin_count = _count_text_scripts(draft)
    if reference_family == "en":
        return draft_cjk_count > 0
    if reference_family == "zh":
        return draft_cjk_count == 0 and draft_latin_count > 0
    return False


def maybe_dropout_ctc_draft_text(
    ctc_draft: str,
    *,
    sample_id: str,
    reference_text: str | None,
    reference_language: str | None,
    dropout_prob: float = 0.0,
    language_mismatch_dropout_prob: float = 0.0,
    seed: int = 0,
) -> str:
    """Deterministically blank unreliable draft hints before prompt rendering."""

    draft = str(ctc_draft or "")
    if not draft.strip():
        return draft
    dropout_prob = float(dropout_prob)
    mismatch_prob = float(language_mismatch_dropout_prob)
    if not 0.0 <= dropout_prob <= 1.0:
        raise ValueError("decoder_ctc_draft_dropout_prob must be between 0.0 and 1.0")
    if not 0.0 <= mismatch_prob <= 1.0:
        raise ValueError("decoder_ctc_draft_language_mismatch_dropout_prob must be between 0.0 and 1.0")

    if dropout_prob > 0.0 and _stable_unit_interval(
        seed=seed,
        sample_id=sample_id,
        salt="ctc_draft_dropout",
    ) < dropout_prob:
        return ""

    if mismatch_prob <= 0.0:
        return draft
    reference_family = _decoder_language_family(
        infer_decoder_language_from_text(reference_text, fallback=reference_language)
    )
    if reference_family in {"en", "zh"}:
        if _ctc_draft_has_reference_language_mismatch(draft, reference_family=reference_family):
            if _stable_unit_interval(
                seed=seed,
                sample_id=sample_id,
                salt="ctc_draft_language_mismatch_dropout",
            ) < mismatch_prob:
                return ""
            return draft
        return draft
    draft_family = _decoder_language_family(infer_decoder_language_from_text(draft, fallback=None))
    if reference_family is None or draft_family is None or reference_family == draft_family:
        return draft
    if _stable_unit_interval(
        seed=seed,
        sample_id=sample_id,
        salt="ctc_draft_language_mismatch_dropout",
    ) < mismatch_prob:
        return ""
    return draft


def render_decoder_language_template(
    template: str,
    *,
    language: str | None,
    english_confirmation: str = "This is English text.",
    chinese_confirmation: str = "这是中文文字。",
    ctc_draft: str = "",
) -> str:
    if not template:
        return ""
    code = _normalize_language_code(language)
    language_name = decoder_language_name(language)
    language_confirmation = decoder_language_confirmation(
        language,
        english=english_confirmation,
        chinese=chinese_confirmation,
    )
    return str(template).format(
        language=code or language_name,
        language_name=language_name,
        output_language_name=language_name,
        language_confirmation=language_confirmation,
        ctc_draft=str(ctc_draft or ""),
        ctc_draft_text=str(ctc_draft or ""),
    )


@lru_cache(maxsize=16)
def load_ctc_draft_cache(path: str, *, text_key: str = "pred_text") -> dict[str, str]:
    cache_path = Path(path)
    drafts: dict[str, str] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in CTC draft cache {cache_path}:{line_number}") from exc
            draft = raw.get(text_key)
            if draft is None and text_key != "pred_text":
                draft = raw.get("pred_text")
            if draft is None:
                draft = raw.get("ctc_draft") or raw.get("text")
            if draft is None:
                continue
            for key_name in ("utt_id", "id", "audio_id", "sid", "key"):
                key_value = raw.get(key_name)
                if key_value is not None:
                    drafts[str(key_value)] = str(draft)
    return drafts


def resolve_ctc_draft_text(
    *,
    cache_path: str | None,
    sample_id: str,
    metadata: dict[str, Any] | None = None,
    text_key: str = "pred_text",
    missing_policy: str = "empty",
) -> str:
    if not cache_path:
        return ""
    cache = load_ctc_draft_cache(str(cache_path), text_key=str(text_key or "pred_text"))
    candidate_keys = [str(sample_id)]
    if metadata:
        for key_name in ("utt_id", "id", "audio_id", "sid", "key"):
            key_value = metadata.get(key_name)
            if key_value is not None:
                candidate_keys.append(str(key_value))
    for candidate in candidate_keys:
        if candidate in cache:
            return cache[candidate]
    if str(missing_policy) == "error":
        raise KeyError(f"Missing CTC draft for sample_id={sample_id!r} in {cache_path}")
    if str(missing_policy) != "empty":
        raise ValueError("decoder_ctc_draft_missing_policy must be 'empty' or 'error'")
    return ""


def combine_decoder_prompt_templates(prompt_before_audio: str, ctc_draft_prompt_template: str) -> str:
    base = str(prompt_before_audio or "")
    draft_template = str(ctc_draft_prompt_template or "")
    if not draft_template:
        return base
    if not base:
        return draft_template
    return base.rstrip() + "\n" + draft_template.lstrip()


class ASRManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        tokenizer: TokenizerLike | None = None,
        decoder_tokenizer: TokenizerLike | None = None,
        feature_extractor: Any | None = None,
        append_eos: bool = False,
        text_normalization: str = "none",
        decoder_append_eos: bool = False,
        decoder_text_normalization: str = "none",
        decoder_prompt_before_audio: str = "",
        decoder_prompt_before_audio_use_language: bool = False,
        decoder_ctc_draft_cache_path: str | None = None,
        decoder_ctc_draft_prompt_template: str = "",
        decoder_ctc_draft_text_key: str = "pred_text",
        decoder_ctc_draft_missing_policy: str = "empty",
        decoder_ctc_draft_dropout_prob: float = 0.0,
        decoder_ctc_draft_language_mismatch_dropout_prob: float = 0.0,
        decoder_ctc_draft_dropout_seed: int = 0,
        decoder_target_prefix: str = "",
        decoder_target_prefix_use_language: bool = False,
        decoder_language_confirmation_en: str = "This is English text.",
        decoder_language_confirmation_zh: str = "这是中文文字。",
        decoder_prompt_language_label_noise_prob: float = 0.0,
        decoder_prompt_language_label_noise_seed: int = 0,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.append_eos = bool(append_eos)
        self.text_normalization = str(text_normalization)
        self.decoder_append_eos = bool(decoder_append_eos)
        self.decoder_text_normalization = str(decoder_text_normalization)
        self.decoder_prompt_before_audio = str(decoder_prompt_before_audio)
        self.decoder_prompt_before_audio_use_language = bool(decoder_prompt_before_audio_use_language)
        self.decoder_ctc_draft_cache_path = decoder_ctc_draft_cache_path
        self.decoder_ctc_draft_prompt_template = str(decoder_ctc_draft_prompt_template or "")
        self.decoder_ctc_draft_text_key = str(decoder_ctc_draft_text_key or "pred_text")
        self.decoder_ctc_draft_missing_policy = str(decoder_ctc_draft_missing_policy or "empty")
        self.decoder_ctc_draft_dropout_prob = float(decoder_ctc_draft_dropout_prob)
        self.decoder_ctc_draft_language_mismatch_dropout_prob = float(
            decoder_ctc_draft_language_mismatch_dropout_prob
        )
        self.decoder_ctc_draft_dropout_seed = int(decoder_ctc_draft_dropout_seed)
        if self.decoder_ctc_draft_cache_path:
            load_ctc_draft_cache(
                str(self.decoder_ctc_draft_cache_path),
                text_key=self.decoder_ctc_draft_text_key,
            )
        self.decoder_target_prefix = str(decoder_target_prefix)
        self.decoder_target_prefix_use_language = bool(decoder_target_prefix_use_language)
        self.decoder_language_confirmation_en = str(decoder_language_confirmation_en)
        self.decoder_language_confirmation_zh = str(decoder_language_confirmation_zh)
        self.decoder_prompt_language_label_noise_prob = float(decoder_prompt_language_label_noise_prob)
        self.decoder_prompt_language_label_noise_seed = int(decoder_prompt_language_label_noise_seed)
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
                raw_text = raw.get("text")
                language = raw.get("language")
                text = None
                if raw_text is not None:
                    text = normalize_asr_text(
                        str(raw_text),
                        language=str(language) if language is not None else None,
                        mode=self.text_normalization,
                    )
                if token_ids is None or (raw_text is not None and self.text_normalization != "none"):
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
                decoder_token_ids = None
                decoder_prompt_before_audio_token_ids = None
                if self.decoder_tokenizer is not None:
                    if raw_text is None:
                        raise ValueError("Manifest entry needs text when decoder_tokenizer is set.")
                    prompt_template = combine_decoder_prompt_templates(
                        self.decoder_prompt_before_audio,
                        self.decoder_ctc_draft_prompt_template,
                    )
                    if (
                        prompt_template
                        and (
                            self.decoder_prompt_before_audio_use_language
                            or self.decoder_ctc_draft_prompt_template
                        )
                    ):
                        prompt_language = maybe_flip_decoder_prompt_language_label(
                            str(language) if language is not None else None,
                            sample_id=str(utt_id),
                            probability=self.decoder_prompt_language_label_noise_prob,
                            seed=self.decoder_prompt_language_label_noise_seed,
                        )
                        ctc_draft = resolve_ctc_draft_text(
                            cache_path=self.decoder_ctc_draft_cache_path,
                            sample_id=str(utt_id),
                            metadata=raw,
                            text_key=self.decoder_ctc_draft_text_key,
                            missing_policy=self.decoder_ctc_draft_missing_policy,
                        )
                        ctc_draft = maybe_dropout_ctc_draft_text(
                            ctc_draft,
                            sample_id=str(utt_id),
                            reference_text=str(raw_text),
                            reference_language=str(language) if language is not None else None,
                            dropout_prob=self.decoder_ctc_draft_dropout_prob,
                            language_mismatch_dropout_prob=(
                                self.decoder_ctc_draft_language_mismatch_dropout_prob
                            ),
                            seed=self.decoder_ctc_draft_dropout_seed,
                        )
                        prompt_text = render_decoder_language_template(
                            prompt_template,
                            language=prompt_language,
                            english_confirmation=self.decoder_language_confirmation_en,
                            chinese_confirmation=self.decoder_language_confirmation_zh,
                            ctc_draft=ctc_draft,
                        )
                        decoder_prompt_before_audio_token_ids = self.decoder_tokenizer.encode(prompt_text)
                    decoder_text = normalize_asr_text(
                        str(raw_text),
                        language=str(language) if language is not None else None,
                        mode=self.decoder_text_normalization,
                    )
                    if self.decoder_target_prefix:
                        target_prefix_language = infer_decoder_language_from_text(
                            str(raw_text),
                            fallback=str(language) if language is not None else None,
                        )
                        if self.decoder_target_prefix_use_language:
                            decoder_text = (
                                render_decoder_language_template(
                                    self.decoder_target_prefix,
                                    language=target_prefix_language,
                                    english_confirmation=self.decoder_language_confirmation_en,
                                    chinese_confirmation=self.decoder_language_confirmation_zh,
                                )
                                + decoder_text
                            )
                        else:
                            decoder_text = self.decoder_target_prefix + decoder_text
                    decoder_token_ids = self.decoder_tokenizer.encode(decoder_text)
                    decoder_token_ids = maybe_append_eos_token_ids(
                        decoder_token_ids,
                        append_eos=self.decoder_append_eos,
                        tokenizer=self.decoder_tokenizer,
                    )
                entries.append(
                    ManifestEntry(
                        utt_id=str(utt_id),
                        token_ids=token_ids,
                        text=text,
                        language=str(language) if language is not None else None,
                        decoder_token_ids=decoder_token_ids,
                        decoder_prompt_before_audio_token_ids=decoder_prompt_before_audio_token_ids,
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
            audio_path = Path(entry.audio_filepath)
            if not audio_path.is_absolute():
                audio_path = self.root / audio_path
            waveform, sample_rate = load_audio_waveform(audio_path)
            return self.feature_extractor(waveform, sample_rate).float()

        raise ValueError("Manifest entry must contain either feature_path or audio_filepath.")

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        features = self._load_features(entry)
        targets = torch.tensor(entry.token_ids, dtype=torch.long)
        sample = {
            "utt_id": entry.utt_id,
            "features": features,
            "feature_length": features.size(0),
            "targets": targets,
            "target_length": targets.numel(),
            "text": entry.text,
        }
        if entry.decoder_token_ids is not None:
            decoder_targets = torch.tensor(entry.decoder_token_ids, dtype=torch.long)
            sample["decoder_targets"] = decoder_targets
            sample["decoder_target_length"] = decoder_targets.numel()
        if entry.decoder_prompt_before_audio_token_ids is not None:
            decoder_prompt_before_audio = torch.tensor(
                entry.decoder_prompt_before_audio_token_ids,
                dtype=torch.long,
            )
            sample["decoder_prompt_before_audio"] = decoder_prompt_before_audio
            sample["decoder_prompt_before_audio_length"] = decoder_prompt_before_audio.numel()
        return sample


class FeatureCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> ASRBatch:
        if not samples:
            raise ValueError("samples must not be empty")

        batch_size = len(samples)
        feat_dim = samples[0]["features"].size(-1)
        max_frames = max(int(sample["feature_length"]) for sample in samples)
        total_targets = sum(int(sample["target_length"]) for sample in samples)
        has_decoder_targets = any("decoder_targets" in sample for sample in samples)
        has_decoder_prompt_before_audio = any("decoder_prompt_before_audio" in sample for sample in samples)
        total_decoder_targets = (
            sum(int(sample.get("decoder_target_length", sample["target_length"])) for sample in samples)
            if has_decoder_targets
            else 0
        )
        total_decoder_prompt_before_audio = (
            sum(int(sample.get("decoder_prompt_before_audio_length", 0)) for sample in samples)
            if has_decoder_prompt_before_audio
            else 0
        )

        features = torch.zeros(batch_size, max_frames, feat_dim, dtype=samples[0]["features"].dtype)
        feature_lengths = torch.zeros(batch_size, dtype=torch.long)
        targets = torch.zeros(total_targets, dtype=torch.long)
        target_lengths = torch.zeros(batch_size, dtype=torch.long)
        decoder_targets = (
            torch.zeros(total_decoder_targets, dtype=torch.long) if has_decoder_targets else None
        )
        decoder_target_lengths = (
            torch.zeros(batch_size, dtype=torch.long) if has_decoder_targets else None
        )
        decoder_prompt_before_audio = (
            torch.zeros(total_decoder_prompt_before_audio, dtype=torch.long)
            if has_decoder_prompt_before_audio
            else None
        )
        decoder_prompt_before_audio_lengths = (
            torch.zeros(batch_size, dtype=torch.long) if has_decoder_prompt_before_audio else None
        )
        utt_ids: list[str] = []
        ctc_teacher_audio_rows = (
            [
                dict(sample["ctc_teacher_audio_row"])
                if isinstance(sample.get("ctc_teacher_audio_row"), dict)
                else None
                for sample in samples
            ]
            if any("ctc_teacher_audio_row" in sample for sample in samples)
            else None
        )

        offset = 0
        decoder_offset = 0
        decoder_prompt_offset = 0
        for idx, sample in enumerate(samples):
            feat = sample["features"]
            feature_len = int(sample["feature_length"])
            target = sample["targets"]
            target_len = int(sample["target_length"])
            decoder_target = sample.get("decoder_targets", target)
            decoder_target_len = int(sample.get("decoder_target_length", target_len))
            decoder_prompt = sample.get("decoder_prompt_before_audio")
            decoder_prompt_len = int(sample.get("decoder_prompt_before_audio_length", 0))

            features[idx, :feature_len] = feat
            feature_lengths[idx] = feature_len
            targets[offset : offset + target_len] = target
            target_lengths[idx] = target_len
            if decoder_targets is not None and decoder_target_lengths is not None:
                decoder_targets[decoder_offset : decoder_offset + decoder_target_len] = decoder_target
                decoder_target_lengths[idx] = decoder_target_len
                decoder_offset += decoder_target_len
            if (
                decoder_prompt_before_audio is not None
                and decoder_prompt_before_audio_lengths is not None
                and decoder_prompt is not None
            ):
                decoder_prompt_before_audio[
                    decoder_prompt_offset : decoder_prompt_offset + decoder_prompt_len
                ] = decoder_prompt
                decoder_prompt_before_audio_lengths[idx] = decoder_prompt_len
                decoder_prompt_offset += decoder_prompt_len
            utt_ids.append(str(sample["utt_id"]))
            offset += target_len

        return ASRBatch(
            features=features,
            feature_lengths=feature_lengths,
            targets=targets,
            target_lengths=target_lengths,
            utt_ids=utt_ids,
            decoder_targets=decoder_targets,
            decoder_target_lengths=decoder_target_lengths,
            decoder_prompt_before_audio=decoder_prompt_before_audio,
            decoder_prompt_before_audio_lengths=decoder_prompt_before_audio_lengths,
            ctc_teacher_audio_rows=ctc_teacher_audio_rows,
        )
