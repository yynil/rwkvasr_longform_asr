from __future__ import annotations

import io
import json
import math
import os
import random
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from rwkvasr.data import ASRManifestDataset, WenetFbankFeatureExtractor, WebDatasetASRIterableDataset, WebDatasetConfig, build_text_tokenizer
from rwkvasr.data.webdataset_common import AUDIO_SUFFIXES
from rwkvasr.data.webdataset_index import StableHashSplitConfig, resolve_sample_id, sample_in_split, shard_in_split
from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig, build_inference_direction_mask
from rwkvasr.training.checkpoint import load_checkpoint

_LOG_ZERO = float("-inf")


def _log_addexp(a: float, b: float) -> float:
    if a == _LOG_ZERO:
        return b
    if b == _LOG_ZERO:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _hypothesis_sort_score(
    blank_score: float,
    non_blank_score: float,
    *,
    token_count: int,
    length_bonus: float,
    insertion_bonus: float,
) -> float:
    return _log_addexp(blank_score, non_blank_score) + (length_bonus + insertion_bonus) * float(token_count)


@dataclass(frozen=True)
class CTCPrefixBeamHypothesis:
    token_ids: tuple[int, ...]
    score: float
    blank_score: float
    non_blank_score: float


@dataclass(frozen=True)
class CTCPrediction:
    utt_id: str
    token_ids: list[int]
    text: str | None
    score: float
    mode: str
    alignments: list["CTCTokenAlignment"]
    debug: "CTCDecodeDebug | None" = None


@dataclass(frozen=True)
class CTCTokenAlignment:
    token_id: int
    token_text: str | None
    start_encoder_t: int
    end_encoder_t: int
    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class CTCLabeledPrediction:
    utt_id: str
    pred_token_ids: list[int]
    ref_token_ids: list[int]
    pred_text: str | None
    ref_text: str | None
    score: float
    mode: str
    alignments: list[CTCTokenAlignment]
    debug: "CTCDecodeDebug | None" = None


@dataclass(frozen=True)
class CTCDecodeDebug:
    feature_length: int
    logit_length: int
    pred_token_count: int
    ref_token_count: int | None
    blank_top1_ratio: float
    avg_blank_prob: float


@dataclass(frozen=True)
class PredictionConfig:
    checkpoint_path: str
    batch_size: int
    model_config: RWKVCTCModelConfig
    manifest_path: str | None = None
    webdataset_root: str | None = None
    webdataset_split: str = "all"
    webdataset_eval_ratio: float = 0.0
    webdataset_hash_seed: int = 0
    webdataset_split_by: str = "shard_name"
    device: str = "cpu"
    mode: str = "bi"
    beam_size: int = 8
    token_prune_topk: int | None = None
    tokenizer_type: str = "whisper_multilingual"
    tokenizer_model_path: str | None = None
    tokenizer_language: str | None = None
    tokenizer_task: str | None = None
    num_workers: int = 0
    frame_shift_ms: float = 10.0
    length_bonus: float = 0.0
    insertion_bonus: float = 0.0
    save_debug_lengths: bool = False


@dataclass(frozen=True)
class ExportedLogitsPart:
    part_index: int
    tensors_path: str
    utt_ids_path: str
    num_samples: int
    max_time: int
    vocab_size: int


@dataclass(frozen=True)
class ExportedLogitsIndex:
    checkpoint_path: str
    mode: str
    blank_id: int
    frontend_type: str
    subsampling_rate: int
    right_context: int
    frame_shift_ms: float
    logits_key: str
    lengths_key: str
    parts: list[ExportedLogitsPart]


def _build_decode_debug(
    frame_log_probs: Tensor,
    *,
    blank_id: int,
    feature_length: int,
    logit_length: int,
    pred_token_count: int,
    ref_token_count: int | None,
) -> CTCDecodeDebug:
    if logit_length <= 0:
        return CTCDecodeDebug(
            feature_length=int(feature_length),
            logit_length=int(logit_length),
            pred_token_count=int(pred_token_count),
            ref_token_count=None if ref_token_count is None else int(ref_token_count),
            blank_top1_ratio=0.0,
            avg_blank_prob=0.0,
        )

    frame_top = frame_log_probs.argmax(dim=-1)
    blank_top1_ratio = float((frame_top == blank_id).float().mean().item())
    avg_blank_prob = float(frame_log_probs[:, blank_id].exp().mean().item())
    return CTCDecodeDebug(
        feature_length=int(feature_length),
        logit_length=int(logit_length),
        pred_token_count=int(pred_token_count),
        ref_token_count=None if ref_token_count is None else int(ref_token_count),
        blank_top1_ratio=blank_top1_ratio,
        avg_blank_prob=avg_blank_prob,
    )


@dataclass
class PredictionBatch:
    features: Tensor
    feature_lengths: Tensor
    utt_ids: list[str]

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype | None = None,
    ) -> "PredictionBatch":
        features = self.features.to(device)
        if feature_dtype is not None and features.is_floating_point():
            features = features.to(dtype=feature_dtype)
        return PredictionBatch(
            features=features,
            feature_lengths=self.feature_lengths.to(device),
            utt_ids=self.utt_ids,
        )


@dataclass
class LabeledPredictionBatch:
    features: Tensor
    feature_lengths: Tensor
    targets: Tensor
    target_lengths: Tensor
    utt_ids: list[str]
    texts: list[str | None]

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype | None = None,
    ) -> "LabeledPredictionBatch":
        features = self.features.to(device)
        if feature_dtype is not None and features.is_floating_point():
            features = features.to(dtype=feature_dtype)
        return LabeledPredictionBatch(
            features=features,
            feature_lengths=self.feature_lengths.to(device),
            targets=self.targets.to(device),
            target_lengths=self.target_lengths.to(device),
            utt_ids=self.utt_ids,
            texts=self.texts,
        )


class PredictionCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> PredictionBatch:
        if not samples:
            raise ValueError("samples must not be empty")

        batch_size = len(samples)
        feat_dim = samples[0]["features"].size(-1)
        max_frames = max(int(sample["feature_length"]) for sample in samples)
        features = torch.zeros(batch_size, max_frames, feat_dim, dtype=samples[0]["features"].dtype)
        feature_lengths = torch.zeros(batch_size, dtype=torch.long)
        utt_ids: list[str] = []

        for idx, sample in enumerate(samples):
            feat = sample["features"]
            feature_len = int(sample["feature_length"])
            features[idx, :feature_len] = feat
            feature_lengths[idx] = feature_len
            utt_ids.append(str(sample["utt_id"]))

        return PredictionBatch(
            features=features,
            feature_lengths=feature_lengths,
            utt_ids=utt_ids,
        )


class LabeledPredictionCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> LabeledPredictionBatch:
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
        texts: list[str | None] = []

        target_offset = 0
        for idx, sample in enumerate(samples):
            feat = sample["features"]
            feature_len = int(sample["feature_length"])
            features[idx, :feature_len] = feat
            feature_lengths[idx] = feature_len

            target = sample["targets"]
            target_len = int(sample["target_length"])
            targets[target_offset : target_offset + target_len] = target
            target_lengths[idx] = target_len
            target_offset += target_len

            utt_ids.append(str(sample["utt_id"]))
            texts.append(sample.get("text"))

        return LabeledPredictionBatch(
            features=features,
            feature_lengths=feature_lengths,
            targets=targets,
            target_lengths=target_lengths,
            utt_ids=utt_ids,
            texts=texts,
        )


class PredictionManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        feature_extractor: WenetFbankFeatureExtractor | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.entries = self._load_entries()

    def _load_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                utt_id = raw.get("utt_id") or raw.get("id") or raw.get("audio_id")
                if utt_id is None:
                    raise ValueError("Prediction manifest entry must contain utt_id, id, or audio_id.")
                if raw.get("feature_path") is None and raw.get("audio_filepath") is None:
                    raise ValueError("Prediction manifest entry must contain feature_path or audio_filepath.")
                entries.append(
                    {
                        "utt_id": str(utt_id),
                        "feature_path": raw.get("feature_path"),
                        "audio_filepath": raw.get("audio_filepath"),
                    }
                )
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def _load_features(self, entry: dict[str, Any]) -> Tensor:
        feature_path = entry.get("feature_path")
        if feature_path is not None:
            resolved = Path(feature_path)
            if not resolved.is_absolute():
                resolved = self.root / resolved
            features = torch.load(resolved, map_location="cpu")
            if not isinstance(features, Tensor):
                raise TypeError(f"Expected Tensor features in {resolved}, got {type(features)}")
            return features.float()

        audio_filepath = entry.get("audio_filepath")
        if audio_filepath is None:
            raise ValueError("Prediction manifest entry must contain feature_path or audio_filepath.")
        import torchaudio

        resolved = Path(audio_filepath)
        if not resolved.is_absolute():
            resolved = self.root / resolved
        waveform, sample_rate = torchaudio.load(resolved)
        return self.feature_extractor(waveform, sample_rate).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        features = self._load_features(entry)
        return {
            "utt_id": entry["utt_id"],
            "features": features,
            "feature_length": features.size(0),
        }


def decode_prediction_webdataset_sample(
    *,
    key: str,
    audio_bytes: bytes,
    metadata_bytes: bytes,
    feature_extractor: WenetFbankFeatureExtractor,
    utt_id_key: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import soundfile as sf

    metadata = metadata or json.loads(metadata_bytes.decode("utf-8"))
    audio_buffer = io.BytesIO(audio_bytes)
    audio_array, sample_rate = sf.read(audio_buffer, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_array).transpose(0, 1)
    features = feature_extractor(waveform, sample_rate).float()
    utt_id = str(metadata.get(utt_id_key) or key)
    return {
        "utt_id": utt_id,
        "features": features,
        "feature_length": features.size(0),
    }


class PredictionWebDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        shard_root: str | Path,
        *,
        feature_extractor: WenetFbankFeatureExtractor | None = None,
        config: WebDatasetConfig | None = None,
    ):
        super().__init__()
        self.shard_root = Path(shard_root)
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.config = config or WebDatasetConfig()
        self.epoch = 0
        self.shards = self._resolve_shards()
        self.split_config = StableHashSplitConfig(
            eval_ratio=self.config.eval_ratio,
            hash_seed=self.config.hash_seed,
            utt_id_key=self.config.utt_id_key,
            split_by=self.config.split_by,
        )

    def _resolve_shards(self) -> list[Path]:
        if self.shard_root.is_file():
            shards = [self.shard_root]
        else:
            shards = sorted(self.shard_root.glob(self.config.shard_pattern))
        if not shards:
            raise FileNotFoundError(
                f"No shard files matching {self.config.shard_pattern!r} under {self.shard_root}"
            )
        return shards

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _partition_shards(self) -> list[Path]:
        shards = list(self.shards)
        if self.config.shuffle_shards:
            random.Random(self.config.seed + self.epoch).shuffle(shards)
        if self.config.split != "all" and self.split_config.split_by == "shard_name":
            shards = [shard for shard in shards if shard_in_split(shard.name, self.config.split, self.split_config)]

        rank = int(os.environ.get("RANK", "0"))
        world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        shards = shards[rank::world_size]

        worker = get_worker_info()
        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
        return shards

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for shard_path in self._partition_shards():
            yield from self._iter_shard(shard_path)

    def _iter_shard(self, shard_path: Path) -> Iterable[dict[str, Any]]:
        pending: dict[str, dict[str, bytes]] = {}
        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if "." not in name:
                    continue
                key, suffix = name.rsplit(".", 1)
                suffix = suffix.lower()
                if suffix != "json" and suffix not in AUDIO_SUFFIXES:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                sample = pending.setdefault(key, {})
                if suffix == "json":
                    sample["json"] = extracted.read()
                else:
                    sample["audio"] = extracted.read()
                if "audio" not in sample or "json" not in sample:
                    continue
                metadata = json.loads(sample["json"].decode("utf-8"))
                include_sample = True
                if self.split_config.split_by == "sample_id":
                    sample_id = resolve_sample_id(key, metadata, utt_id_key=self.config.utt_id_key)
                    include_sample = sample_in_split(
                        sample_id,
                        self.config.split,
                        self.split_config,
                        shard_name=shard_path.name,
                    )
                if include_sample:
                    yield decode_prediction_webdataset_sample(
                        key=key,
                        audio_bytes=sample["audio"],
                        metadata_bytes=sample["json"],
                        feature_extractor=self.feature_extractor,
                        utt_id_key=self.config.utt_id_key,
                        metadata=metadata,
                    )
                pending.pop(key, None)


def _build_prediction_loader(config: PredictionConfig) -> DataLoader:
    has_manifest = config.manifest_path is not None
    has_webdataset = config.webdataset_root is not None
    if has_manifest == has_webdataset:
        raise ValueError("Exactly one of manifest_path or webdataset_root must be provided for prediction.")

    if has_manifest:
        dataset = PredictionManifestDataset(str(config.manifest_path))
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=PredictionCollator(),
        )

    dataset = PredictionWebDataset(
        str(config.webdataset_root),
        config=WebDatasetConfig(
            shuffle_shards=False,
            split=config.webdataset_split,
            eval_ratio=config.webdataset_eval_ratio,
            hash_seed=config.webdataset_hash_seed,
            split_by=config.webdataset_split_by,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=PredictionCollator(),
    )


def _build_labeled_prediction_loader(config: PredictionConfig) -> DataLoader:
    has_manifest = config.manifest_path is not None
    has_webdataset = config.webdataset_root is not None
    if has_manifest == has_webdataset:
        raise ValueError("Exactly one of manifest_path or webdataset_root must be provided for prediction.")

    if has_manifest:
        dataset = ASRManifestDataset(str(config.manifest_path))
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=LabeledPredictionCollator(),
        )

    dataset = WebDatasetASRIterableDataset(
        str(config.webdataset_root),
        config=WebDatasetConfig(
            shuffle_shards=False,
            split=config.webdataset_split,
            eval_ratio=config.webdataset_eval_ratio,
            hash_seed=config.webdataset_hash_seed,
            split_by=config.webdataset_split_by,
            partition_by_rank=False,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=LabeledPredictionCollator(),
    )


def _load_prediction_model(
    config: PredictionConfig,
    *,
    device: torch.device,
) -> tuple[RWKVCTCModel, torch.dtype | None]:
    feature_dtype = torch.bfloat16 if device.type == "cuda" else None
    model = RWKVCTCModel(config.model_config)
    if feature_dtype is not None:
        model = model.to(device=device, dtype=feature_dtype)
    else:
        model = model.to(device)
    load_checkpoint(config.checkpoint_path, model=model, map_location=device.type)
    model.eval()
    return model, feature_dtype


def ctc_prefix_beam_search(
    frame_log_probs: Tensor,
    *,
    blank_id: int,
    beam_size: int,
    token_prune_topk: int | None = None,
    length_bonus: float = 0.0,
    insertion_bonus: float = 0.0,
) -> list[CTCPrefixBeamHypothesis]:
    if frame_log_probs.dim() != 2:
        raise ValueError(f"Expected [T, V] log-probs, got shape {tuple(frame_log_probs.shape)}")
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")

    log_probs = frame_log_probs.detach().to(dtype=torch.float32, device="cpu")
    num_tokens = int(log_probs.size(-1))
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, _LOG_ZERO)}

    for time_idx in range(log_probs.size(0)):
        frame = log_probs[time_idx]
        if token_prune_topk is not None and 0 < token_prune_topk < num_tokens:
            topk = min(token_prune_topk, num_tokens)
            topk_values, topk_indices = torch.topk(frame, k=topk)
            candidates = [(int(idx), float(value)) for idx, value in zip(topk_indices.tolist(), topk_values.tolist(), strict=True)]
            if blank_id not in {token for token, _ in candidates}:
                candidates.append((blank_id, float(frame[blank_id].item())))
        else:
            candidates = [(token_id, float(frame[token_id].item())) for token_id in range(num_tokens)]

        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}
        for prefix, (prefix_blank, prefix_non_blank) in beams.items():
            prefix_total = _log_addexp(prefix_blank, prefix_non_blank)
            for token_id, token_logp in candidates:
                if token_id == blank_id:
                    next_blank, next_non_blank = next_beams.get(prefix, (_LOG_ZERO, _LOG_ZERO))
                    next_blank = _log_addexp(next_blank, prefix_total + token_logp)
                    next_beams[prefix] = (next_blank, next_non_blank)
                    continue

                last_token = prefix[-1] if prefix else None
                extended_prefix = prefix + (token_id,)
                if token_id == last_token:
                    same_blank, same_non_blank = next_beams.get(prefix, (_LOG_ZERO, _LOG_ZERO))
                    same_non_blank = _log_addexp(same_non_blank, prefix_non_blank + token_logp)
                    next_beams[prefix] = (same_blank, same_non_blank)

                    ext_blank, ext_non_blank = next_beams.get(extended_prefix, (_LOG_ZERO, _LOG_ZERO))
                    ext_non_blank = _log_addexp(ext_non_blank, prefix_blank + token_logp)
                    next_beams[extended_prefix] = (ext_blank, ext_non_blank)
                    continue

                ext_blank, ext_non_blank = next_beams.get(extended_prefix, (_LOG_ZERO, _LOG_ZERO))
                ext_non_blank = _log_addexp(ext_non_blank, prefix_total + token_logp)
                next_beams[extended_prefix] = (ext_blank, ext_non_blank)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _hypothesis_sort_score(
                    item[1][0],
                    item[1][1],
                    token_count=len(item[0]),
                    length_bonus=length_bonus,
                    insertion_bonus=insertion_bonus,
                ),
                reverse=True,
            )[:beam_size]
        )

    return [
        CTCPrefixBeamHypothesis(
            token_ids=prefix,
            score=_hypothesis_sort_score(
                prefix_blank,
                prefix_non_blank,
                token_count=len(prefix),
                length_bonus=length_bonus,
                insertion_bonus=insertion_bonus,
            ),
            blank_score=prefix_blank,
            non_blank_score=prefix_non_blank,
        )
        for prefix, (prefix_blank, prefix_non_blank) in sorted(
            beams.items(),
            key=lambda item: _hypothesis_sort_score(
                item[1][0],
                item[1][1],
                token_count=len(item[0]),
                length_bonus=length_bonus,
                insertion_bonus=insertion_bonus,
            ),
            reverse=True,
        )
    ]


def batched_ctc_prefix_beam_search(
    logits: Tensor,
    lengths: Tensor | None,
    *,
    blank_id: int,
    beam_size: int,
    token_prune_topk: int | None = None,
    length_bonus: float = 0.0,
    insertion_bonus: float = 0.0,
) -> list[list[CTCPrefixBeamHypothesis]]:
    if logits.dim() != 3:
        raise ValueError(f"Expected [B, T, V] logits, got shape {tuple(logits.shape)}")

    log_probs = logits.detach().float().log_softmax(dim=-1).cpu()
    if lengths is None:
        lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
    else:
        lengths = lengths.detach().to(dtype=torch.long, device="cpu")

    results: list[list[CTCPrefixBeamHypothesis]] = []
    for batch_idx in range(logits.size(0)):
        length = int(lengths[batch_idx].item())
        results.append(
            ctc_prefix_beam_search(
                log_probs[batch_idx, :length],
                blank_id=blank_id,
                beam_size=beam_size,
                token_prune_topk=token_prune_topk,
                length_bonus=length_bonus,
                insertion_bonus=insertion_bonus,
            )
        )
    return results


def ctc_forced_align(
    log_probs: Tensor,
    token_ids: list[int],
    *,
    blank_id: int,
) -> list[tuple[int, int]]:
    if log_probs.dim() != 2:
        raise ValueError(f"Expected [T, V] log-probs, got shape {tuple(log_probs.shape)}")
    if not token_ids:
        return []

    time_steps = int(log_probs.size(0))
    if time_steps == 0:
        raise ValueError("Cannot align an empty log-prob sequence.")

    extended = [blank_id]
    for token_id in token_ids:
        extended.extend([int(token_id), blank_id])
    num_states = len(extended)
    neg_inf = float("-inf")

    scores = [[neg_inf] * num_states for _ in range(time_steps)]
    backpointers = [[-1] * num_states for _ in range(time_steps)]

    scores[0][0] = float(log_probs[0, blank_id].item())
    if num_states > 1:
        scores[0][1] = float(log_probs[0, extended[1]].item())

    for time_idx in range(1, time_steps):
        for state_idx, token_id in enumerate(extended):
            candidates = [(scores[time_idx - 1][state_idx], state_idx)]
            if state_idx > 0:
                candidates.append((scores[time_idx - 1][state_idx - 1], state_idx - 1))
            if (
                state_idx > 1
                and token_id != blank_id
                and token_id != extended[state_idx - 2]
            ):
                candidates.append((scores[time_idx - 1][state_idx - 2], state_idx - 2))

            best_score, best_prev = max(candidates, key=lambda item: item[0])
            if best_score == neg_inf:
                continue
            scores[time_idx][state_idx] = best_score + float(log_probs[time_idx, token_id].item())
            backpointers[time_idx][state_idx] = best_prev

    end_candidates = [(scores[time_steps - 1][num_states - 1], num_states - 1)]
    if num_states > 1:
        end_candidates.append((scores[time_steps - 1][num_states - 2], num_states - 2))
    best_final_score, best_final_state = max(end_candidates, key=lambda item: item[0])
    if best_final_score == neg_inf:
        raise ValueError("Unable to compute a valid CTC alignment for the provided token sequence.")

    path_states = [best_final_state] * time_steps
    state = best_final_state
    for time_idx in range(time_steps - 1, 0, -1):
        state = backpointers[time_idx][state]
        if state < 0:
            raise ValueError("CTC alignment backtracking failed due to an incomplete path.")
        path_states[time_idx - 1] = state

    spans: list[tuple[int, int]] = []
    for token_index in range(len(token_ids)):
        state_index = 2 * token_index + 1
        positions = [time_idx for time_idx, state in enumerate(path_states) if state == state_index]
        if not positions:
            raise ValueError(f"Token at index {token_index} received no CTC alignment span.")
        spans.append((positions[0], positions[-1]))
    return spans


def _frontend_frame_span(
    start_encoder_t: int,
    end_encoder_t: int,
    *,
    frontend_type: str,
) -> tuple[int, int]:
    if frontend_type == "conv2d6":
        return 6 * start_encoder_t, 6 * end_encoder_t + 10
    if frontend_type == "linear":
        return start_encoder_t, end_encoder_t
    raise ValueError(f"Unsupported frontend_type for timestamp projection: {frontend_type}")


def _frontend_alignment_config(frontend_type: str) -> tuple[int, int]:
    if frontend_type == "conv2d6":
        return 6, 10
    if frontend_type == "linear":
        return 1, 0
    raise ValueError(f"Unsupported frontend_type for timestamp projection: {frontend_type}")


def build_token_alignments(
    log_probs: Tensor,
    token_ids: list[int],
    *,
    blank_id: int,
    frontend_type: str,
    frame_shift_ms: float,
    decode_fn: Any | None = None,
) -> list[CTCTokenAlignment]:
    spans = ctc_forced_align(log_probs, token_ids, blank_id=blank_id)
    alignments: list[CTCTokenAlignment] = []
    for token_id, (start_encoder_t, end_encoder_t) in zip(token_ids, spans, strict=True):
        start_frame, end_frame = _frontend_frame_span(
            start_encoder_t,
            end_encoder_t,
            frontend_type=frontend_type,
        )
        token_text = str(decode_fn([token_id])) if callable(decode_fn) else None
        alignments.append(
            CTCTokenAlignment(
                token_id=int(token_id),
                token_text=token_text,
                start_encoder_t=int(start_encoder_t),
                end_encoder_t=int(end_encoder_t),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                start_ms=float(start_frame * frame_shift_ms),
                end_ms=float((end_frame + 1) * frame_shift_ms),
            )
        )
    return alignments


def predict_ctc(config: PredictionConfig) -> list[CTCPrediction]:
    device = torch.device(config.device)
    model, feature_dtype = _load_prediction_model(config, device=device)

    tokenizer = build_text_tokenizer(
        config.tokenizer_type,
        model_path=config.tokenizer_model_path,
        language=config.tokenizer_language,
        task=config.tokenizer_task,
    )
    decode_fn = getattr(tokenizer, "decode", None)

    loader = _build_prediction_loader(config)
    predictions: list[CTCPrediction] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, feature_dtype=feature_dtype)
            mask = build_inference_direction_mask(model.config.num_layers, mode=config.mode, device=batch.features.device)
            logits, logit_lengths, _ = model(
                batch.features,
                batch.feature_lengths,
                direction_mask=mask,
            )
            log_probs = logits.detach().float().log_softmax(dim=-1).cpu()
            if logit_lengths is None:
                lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            else:
                lengths = logit_lengths.detach().to(dtype=torch.long, device="cpu")

            for batch_idx, utt_id in enumerate(batch.utt_ids):
                length = int(lengths[batch_idx].item())
                hypotheses = ctc_prefix_beam_search(
                    log_probs[batch_idx, :length],
                    blank_id=model.config.blank_id,
                    beam_size=config.beam_size,
                    token_prune_topk=config.token_prune_topk,
                    length_bonus=config.length_bonus,
                    insertion_bonus=config.insertion_bonus,
                )
                best = hypotheses[0] if hypotheses else CTCPrefixBeamHypothesis((), 0.0, 0.0, _LOG_ZERO)
                token_ids = [int(token_id) for token_id in best.token_ids]
                text = str(decode_fn(token_ids)) if callable(decode_fn) else None
                debug = None
                if config.save_debug_lengths:
                    debug = _build_decode_debug(
                        log_probs[batch_idx, :length],
                        blank_id=model.config.blank_id,
                        feature_length=int(batch.feature_lengths[batch_idx].item()),
                        logit_length=length,
                        pred_token_count=len(token_ids),
                        ref_token_count=None,
                    )
                alignments = build_token_alignments(
                    log_probs[batch_idx, :length],
                    token_ids,
                    blank_id=model.config.blank_id,
                    frontend_type=model.config.frontend_type,
                    frame_shift_ms=config.frame_shift_ms,
                    decode_fn=decode_fn,
                )
                predictions.append(
                    CTCPrediction(
                        utt_id=str(utt_id),
                        token_ids=token_ids,
                        text=text,
                        score=float(best.score),
                        mode=config.mode,
                        alignments=alignments,
                        debug=debug,
                    )
                )
    return predictions


def export_ctc_logits(
    config: PredictionConfig,
    output_dir: str | Path,
    *,
    max_batches: int | None = None,
) -> ExportedLogitsIndex:
    from safetensors.torch import save_file

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be >= 1 when provided.")

    device = torch.device(config.device)
    model, feature_dtype = _load_prediction_model(config, device=device)
    loader = _build_prediction_loader(config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts: list[ExportedLogitsPart] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            batch = batch.to(device, feature_dtype=feature_dtype)
            mask = build_inference_direction_mask(
                model.config.num_layers,
                mode=config.mode,
                device=batch.features.device,
            )
            logits, logit_lengths, _ = model(
                batch.features,
                batch.feature_lengths,
                direction_mask=mask,
            )
            logits_cpu = logits.detach().float().cpu().contiguous()
            if logit_lengths is None:
                lengths_cpu = torch.full(
                    (logits_cpu.size(0),),
                    logits_cpu.size(1),
                    dtype=torch.int32,
                )
            else:
                lengths_cpu = logit_lengths.detach().to(dtype=torch.int32, device="cpu").contiguous()

            part_stem = f"part-{batch_index:05d}"
            tensors_path = output_dir / f"{part_stem}.safetensors"
            utt_ids_path = output_dir / f"{part_stem}.utt_ids.txt"
            save_file(
                {
                    "logits": logits_cpu,
                    "lengths": lengths_cpu,
                },
                str(tensors_path),
                metadata={
                    "checkpoint_path": str(config.checkpoint_path),
                    "mode": config.mode,
                    "blank_id": str(model.config.blank_id),
                    "frontend_type": model.config.frontend_type,
                    "subsampling_rate": str(_frontend_alignment_config(model.config.frontend_type)[0]),
                    "right_context": str(_frontend_alignment_config(model.config.frontend_type)[1]),
                    "frame_shift_ms": str(config.frame_shift_ms),
                },
            )
            utt_ids_path.write_text(
                "".join(f"{utt_id}\n" for utt_id in batch.utt_ids),
                encoding="utf-8",
            )
            parts.append(
                ExportedLogitsPart(
                    part_index=batch_index,
                    tensors_path=str(tensors_path),
                    utt_ids_path=str(utt_ids_path),
                    num_samples=len(batch.utt_ids),
                    max_time=int(logits_cpu.size(1)),
                    vocab_size=int(logits_cpu.size(2)),
                )
            )

    subsampling_rate, right_context = _frontend_alignment_config(model.config.frontend_type)
    export_index = ExportedLogitsIndex(
        checkpoint_path=str(config.checkpoint_path),
        mode=config.mode,
        blank_id=int(model.config.blank_id),
        frontend_type=model.config.frontend_type,
        subsampling_rate=subsampling_rate,
        right_context=right_context,
        frame_shift_ms=float(config.frame_shift_ms),
        logits_key="logits",
        lengths_key="lengths",
        parts=parts,
    )
    index_path = output_dir / "export_index.json"
    index_path.write_text(
        json.dumps(asdict(export_index), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return export_index


def predict_ctc_labeled(
    config: PredictionConfig,
    *,
    limit: int | None = None,
) -> list[CTCLabeledPrediction]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided.")

    device = torch.device(config.device)
    model, feature_dtype = _load_prediction_model(config, device=device)
    tokenizer = build_text_tokenizer(
        config.tokenizer_type,
        model_path=config.tokenizer_model_path,
        language=config.tokenizer_language,
        task=config.tokenizer_task,
    )
    decode_fn = getattr(tokenizer, "decode", None)
    loader = _build_labeled_prediction_loader(config)

    predictions: list[CTCLabeledPrediction] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, feature_dtype=feature_dtype)
            mask = build_inference_direction_mask(
                model.config.num_layers,
                mode=config.mode,
                device=batch.features.device,
            )
            logits, logit_lengths, _ = model(
                batch.features,
                batch.feature_lengths,
                direction_mask=mask,
            )
            log_probs = logits.detach().float().log_softmax(dim=-1).cpu()
            if logit_lengths is None:
                lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            else:
                lengths = logit_lengths.detach().to(dtype=torch.long, device="cpu")

            target_offset = 0
            for batch_idx, utt_id in enumerate(batch.utt_ids):
                length = int(lengths[batch_idx].item())
                hypotheses = ctc_prefix_beam_search(
                    log_probs[batch_idx, :length],
                    blank_id=model.config.blank_id,
                    beam_size=config.beam_size,
                    token_prune_topk=config.token_prune_topk,
                    length_bonus=config.length_bonus,
                    insertion_bonus=config.insertion_bonus,
                )
                best = hypotheses[0] if hypotheses else CTCPrefixBeamHypothesis((), 0.0, 0.0, _LOG_ZERO)
                pred_token_ids = [int(token_id) for token_id in best.token_ids]

                target_length = int(batch.target_lengths[batch_idx].item())
                ref_token_ids = batch.targets[target_offset : target_offset + target_length].tolist()
                target_offset += target_length

                pred_text = str(decode_fn(pred_token_ids)) if callable(decode_fn) else None
                ref_text = batch.texts[batch_idx]
                if ref_text is None and callable(decode_fn):
                    ref_text = str(decode_fn(ref_token_ids))

                debug = None
                if config.save_debug_lengths:
                    debug = _build_decode_debug(
                        log_probs[batch_idx, :length],
                        blank_id=model.config.blank_id,
                        feature_length=int(batch.feature_lengths[batch_idx].item()),
                        logit_length=length,
                        pred_token_count=len(pred_token_ids),
                        ref_token_count=target_length,
                    )
                alignments = build_token_alignments(
                    log_probs[batch_idx, :length],
                    pred_token_ids,
                    blank_id=model.config.blank_id,
                    frontend_type=model.config.frontend_type,
                    frame_shift_ms=config.frame_shift_ms,
                    decode_fn=decode_fn,
                )
                predictions.append(
                    CTCLabeledPrediction(
                        utt_id=str(utt_id),
                        pred_token_ids=pred_token_ids,
                        ref_token_ids=[int(token_id) for token_id in ref_token_ids],
                        pred_text=pred_text,
                        ref_text=ref_text,
                        score=float(best.score),
                        mode=config.mode,
                        alignments=alignments,
                        debug=debug,
                    )
                )
                if limit is not None and len(predictions) >= limit:
                    return predictions
    return predictions


def write_labeled_predictions_jsonl(path: str | Path, predictions: list[CTCLabeledPrediction]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(
                    {
                        "utt_id": prediction.utt_id,
                        "pred_token_ids": prediction.pred_token_ids,
                        "ref_token_ids": prediction.ref_token_ids,
                        "pred_text": prediction.pred_text,
                        "ref_text": prediction.ref_text,
                        "score": prediction.score,
                        "mode": prediction.mode,
                        "alignments": [asdict(alignment) for alignment in prediction.alignments],
                        "debug": None if prediction.debug is None else asdict(prediction.debug),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output_path


def write_predictions_jsonl(path: str | Path, predictions: list[CTCPrediction]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(
                    {
                        "utt_id": prediction.utt_id,
                        "token_ids": prediction.token_ids,
                        "text": prediction.text,
                        "score": prediction.score,
                        "mode": prediction.mode,
                        "alignments": [asdict(alignment) for alignment in prediction.alignments],
                        "debug": None if prediction.debug is None else asdict(prediction.debug),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output_path
