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

    @property
    def vocab_size(self) -> int: ...


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
        return str(self.processor.decode(token_ids))

    @property
    def vocab_size(self) -> int:
        return self._text_vocab_size


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
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
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
                entries.append(
                    ManifestEntry(
                        utt_id=str(utt_id),
                        token_ids=[int(token) for token in token_ids],
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
