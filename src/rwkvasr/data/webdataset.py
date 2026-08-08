from __future__ import annotations

import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .manifest import (
    FeatureCollator,
    TokenizerLike,
    WenetFbankFeatureExtractor,
    build_text_tokenizer,
    combine_decoder_prompt_templates,
    infer_decoder_language_from_text,
    load_ctc_draft_cache,
    maybe_dropout_ctc_draft_text,
    maybe_append_eos_token_ids,
    maybe_flip_decoder_prompt_language_label,
    render_decoder_language_template,
    resolve_ctc_draft_text,
)
from .text_normalization import normalize_asr_text
from .webdataset_common import AUDIO_SUFFIXES
from .webdataset_index import StableHashSplitConfig, resolve_sample_id, sample_in_split, shard_in_split


@dataclass(frozen=True)
class WebDatasetConfig:
    shard_pattern: str = "*.tar"
    shuffle_shards: bool = True
    seed: int = 42
    text_key: str = "text"
    utt_id_key: str = "sid"
    token_ids_key: str = "token_ids"
    split: str = "all"
    eval_ratio: float = 0.0
    hash_seed: int = 0
    split_by: str = "shard_name"
    partition_by_rank: bool = True
    length_index_path: str | None = None
    use_length_bucketing: bool = False
    length_bucket_drop_last: bool = True
    length_bucket_frame_budget: int | None = None
    decoded_batch_prefetch: int = 2
    max_open_shards_per_worker: int = 8
    bucket_source_interleave: bool = False
    append_eos: bool = False
    text_normalization: str = "none"
    decoder_append_eos: bool = False
    decoder_text_normalization: str = "none"
    decoder_prompt_before_audio: str = ""
    decoder_prompt_before_audio_use_language: bool = False
    decoder_ctc_draft_cache_path: str | None = None
    decoder_ctc_draft_prompt_template: str = ""
    decoder_ctc_draft_text_key: str = "pred_text"
    decoder_ctc_draft_missing_policy: str = "empty"
    decoder_ctc_draft_dropout_prob: float = 0.0
    decoder_ctc_draft_language_mismatch_dropout_prob: float = 0.0
    decoder_ctc_draft_dropout_seed: int = 0
    ctc_label_override_cache_path: str | None = None
    ctc_label_override_text_key: str = "pred_text"
    allow_missing_targets: bool = False
    decoder_target_prefix: str = ""
    decoder_target_prefix_use_language: bool = False
    decoder_language_confirmation_en: str = "This is English text."
    decoder_language_confirmation_zh: str = "这是中文文字。"
    decoder_prompt_language_label_noise_prob: float = 0.0
    decoder_prompt_language_label_noise_seed: int = 0
    skip_decode_errors: bool = False


def preload_decoder_ctc_draft_cache(config: WebDatasetConfig) -> None:
    cache_path = config.decoder_ctc_draft_cache_path
    if cache_path:
        load_ctc_draft_cache(
            str(cache_path),
            text_key=str(config.decoder_ctc_draft_text_key or "pred_text"),
        )
    override_cache_path = config.ctc_label_override_cache_path
    if override_cache_path:
        load_ctc_draft_cache(
            str(override_cache_path),
            text_key=str(config.ctc_label_override_text_key or "pred_text"),
        )


def resolve_ctc_label_override_text(
    *,
    cache_path: str | None,
    sample_id: str,
    metadata: dict[str, Any] | None = None,
    text_key: str = "pred_text",
) -> str | None:
    if not cache_path:
        return None
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
    return None


def log_webdataset_decode_skip(
    *,
    key: str | None,
    shard_name: str | None,
    audio_member: str | None,
    json_member: str | None,
    exc: Exception,
) -> None:
    rank = os.environ.get("RANK")
    worker = get_worker_info()
    context = [
        f"rank={rank}" if rank is not None else None,
        f"worker={worker.id}" if worker is not None else None,
        f"shard={shard_name}" if shard_name else None,
        f"key={key}" if key else None,
        f"audio={audio_member}" if audio_member else None,
        f"json={json_member}" if json_member else None,
    ]
    context_text = " ".join(part for part in context if part is not None)
    print(
        "[rwkvasr-webdataset] skipped corrupt sample "
        f"{context_text}: {type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )


def _decode_audio_bytes_with_ffmpeg(audio_bytes: bytes, *, key: str) -> tuple[torch.Tensor, int]:
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(f"ffmpeg is unavailable for WebDataset sample key={key}")

    sample_rate = 16000
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    env = os.environ.copy()
    loader_paths = ["/usr/lib/x86_64-linux-gnu/blas", "/usr/lib/x86_64-linux-gnu/lapack"]
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = ":".join([*loader_paths, existing] if existing else loader_paths)
    result = subprocess.run(
        command,
        input=audio_bytes,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode WebDataset sample key={key}: {message}")
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg decoded no samples for WebDataset sample key={key}")
    return torch.from_numpy(audio).unsqueeze(0), sample_rate


def load_webdataset_audio_bytes(audio_bytes: bytes, *, key: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    try:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        return torch.from_numpy(audio_array).transpose(0, 1), int(sample_rate)
    except Exception as soundfile_exc:
        try:
            waveform, sample_rate = _decode_audio_bytes_with_ffmpeg(audio_bytes, key=key)
            print(
                f"[rwkvasr-webdataset] decoded via ffmpeg fallback key={key}",
                file=sys.stderr,
                flush=True,
            )
            return waveform, sample_rate
        except Exception as ffmpeg_exc:
            raise RuntimeError(
                "Could not decode WebDataset audio "
                f"key={key}: soundfile={type(soundfile_exc).__name__}: {soundfile_exc}; "
                f"ffmpeg={type(ffmpeg_exc).__name__}: {ffmpeg_exc}"
            ) from ffmpeg_exc


def decode_webdataset_sample(
    *,
    key: str,
    audio_bytes: bytes,
    metadata_bytes: bytes,
    tokenizer: TokenizerLike | None,
    decoder_tokenizer: TokenizerLike | None = None,
    feature_extractor: WenetFbankFeatureExtractor,
    text_key: str,
    utt_id_key: str,
    token_ids_key: str,
    append_eos: bool,
    decoder_append_eos: bool = False,
    text_normalization: str = "none",
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
    ctc_label_override_cache_path: str | None = None,
    ctc_label_override_text_key: str = "pred_text",
    allow_missing_targets: bool = False,
    decoder_target_prefix: str = "",
    decoder_target_prefix_use_language: bool = False,
    decoder_language_confirmation_en: str = "This is English text.",
    decoder_language_confirmation_zh: str = "这是中文文字。",
    decoder_prompt_language_label_noise_prob: float = 0.0,
    decoder_prompt_language_label_noise_seed: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or json.loads(metadata_bytes.decode("utf-8"))
    token_ids = metadata.get(token_ids_key)
    raw_text = metadata.get(text_key)
    ctc_raw_text = raw_text
    language = metadata.get("language")
    utt_id = str(metadata.get(utt_id_key) or key)
    ctc_label_override_text = resolve_ctc_label_override_text(
        cache_path=ctc_label_override_cache_path,
        sample_id=utt_id,
        metadata=metadata,
        text_key=ctc_label_override_text_key,
    )
    if ctc_label_override_text is not None:
        ctc_raw_text = ctc_label_override_text
        token_ids = None
        metadata = {**metadata, "_ctc_label_override_text": ctc_label_override_text}
    if allow_missing_targets and decoder_tokenizer is None:
        if append_eos:
            raise ValueError("allow_missing_targets=True requires append_eos=False.")
        ctc_raw_text = None
        token_ids = []
    text = ctc_raw_text
    if text is not None:
        text = normalize_asr_text(
            str(text),
            language=str(language) if language is not None else None,
            mode=text_normalization,
        )
    if token_ids is None:
        if text is None:
            if allow_missing_targets:
                token_ids = []
            else:
                raise ValueError("WebDataset sample needs token_ids or text with a tokenizer.")
        if tokenizer is None:
            tokenizer = build_text_tokenizer("whisper_multilingual")
        if token_ids is None:
            token_ids = tokenizer.encode(text)
    token_ids = maybe_append_eos_token_ids(
        token_ids,
        append_eos=append_eos,
        tokenizer=tokenizer,
    )
    decoder_token_ids = None
    decoder_prompt_before_audio_token_ids = None
    if decoder_tokenizer is not None:
        if raw_text is None:
            raise ValueError("WebDataset sample needs text when decoder_tokenizer is set.")
        prompt_template = combine_decoder_prompt_templates(
            decoder_prompt_before_audio,
            decoder_ctc_draft_prompt_template,
        )
        if prompt_template and (
            decoder_prompt_before_audio_use_language or decoder_ctc_draft_prompt_template
        ):
            prompt_language = maybe_flip_decoder_prompt_language_label(
                str(language) if language is not None else None,
                sample_id=utt_id,
                probability=decoder_prompt_language_label_noise_prob,
                seed=decoder_prompt_language_label_noise_seed,
            )
            ctc_draft = resolve_ctc_draft_text(
                cache_path=decoder_ctc_draft_cache_path,
                sample_id=utt_id,
                metadata=metadata,
                text_key=decoder_ctc_draft_text_key,
                missing_policy=decoder_ctc_draft_missing_policy,
            )
            ctc_draft = maybe_dropout_ctc_draft_text(
                ctc_draft,
                sample_id=utt_id,
                reference_text=str(raw_text),
                reference_language=str(language) if language is not None else None,
                dropout_prob=decoder_ctc_draft_dropout_prob,
                language_mismatch_dropout_prob=decoder_ctc_draft_language_mismatch_dropout_prob,
                seed=decoder_ctc_draft_dropout_seed,
            )
            prompt_text = render_decoder_language_template(
                prompt_template,
                language=prompt_language,
                english_confirmation=decoder_language_confirmation_en,
                chinese_confirmation=decoder_language_confirmation_zh,
                ctc_draft=ctc_draft,
            )
            decoder_prompt_before_audio_token_ids = decoder_tokenizer.encode(prompt_text)
        decoder_text = normalize_asr_text(
            str(raw_text),
            language=str(language) if language is not None else None,
            mode=decoder_text_normalization,
        )
        if decoder_target_prefix:
            target_prefix_language = infer_decoder_language_from_text(
                str(raw_text),
                fallback=str(language) if language is not None else None,
            )
            if decoder_target_prefix_use_language:
                decoder_text = (
                    render_decoder_language_template(
                        decoder_target_prefix,
                        language=target_prefix_language,
                        english_confirmation=decoder_language_confirmation_en,
                        chinese_confirmation=decoder_language_confirmation_zh,
                    )
                    + decoder_text
                )
            else:
                decoder_text = decoder_target_prefix + decoder_text
        decoder_token_ids = decoder_tokenizer.encode(decoder_text)
        decoder_token_ids = maybe_append_eos_token_ids(
            decoder_token_ids,
            append_eos=decoder_append_eos,
            tokenizer=decoder_tokenizer,
        )

    waveform, sample_rate = load_webdataset_audio_bytes(audio_bytes, key=key)
    features = feature_extractor(waveform, sample_rate).float()
    targets = torch.tensor([int(token) for token in token_ids], dtype=torch.long)

    sample = {
        "utt_id": utt_id,
        "features": features,
        "feature_length": features.size(0),
        "targets": targets,
        "target_length": targets.numel(),
        "text": text,
        "language": language,
        "metadata": metadata,
    }
    if decoder_token_ids is not None:
        decoder_targets = torch.tensor([int(token) for token in decoder_token_ids], dtype=torch.long)
        sample["decoder_targets"] = decoder_targets
        sample["decoder_target_length"] = decoder_targets.numel()
    if decoder_prompt_before_audio_token_ids is not None:
        decoder_prompt_before_audio_targets = torch.tensor(
            [int(token) for token in decoder_prompt_before_audio_token_ids],
            dtype=torch.long,
        )
        sample["decoder_prompt_before_audio"] = decoder_prompt_before_audio_targets
        sample["decoder_prompt_before_audio_length"] = decoder_prompt_before_audio_targets.numel()
    return sample


class WebDatasetASRIterableDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        shard_root: str | Path,
        *,
        tokenizer: TokenizerLike | None = None,
        decoder_tokenizer: TokenizerLike | None = None,
        feature_extractor: WenetFbankFeatureExtractor | None = None,
        config: WebDatasetConfig | None = None,
    ):
        super().__init__()
        self.shard_root = Path(shard_root)
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.config = config or WebDatasetConfig()
        preload_decoder_ctc_draft_cache(self.config)
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

        if self.config.partition_by_rank:
            rank = int(os.environ.get("RANK", "0"))
            world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
            shards = shards[rank::world_size]

            worker = get_worker_info()
            if worker is not None:
                shards = shards[worker.id :: worker.num_workers]
        return shards

    def _iter_shard(self, shard_path: Path) -> Iterable[dict[str, Any]]:
        pending: dict[str, dict[str, Any]] = {}
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
                    sample["json_member"] = member.name
                else:
                    sample["audio"] = extracted.read()
                    sample["audio_member"] = member.name
                if "audio" in sample and "json" in sample:
                    try:
                        metadata = json.loads(sample["json"].decode("utf-8"))
                    except Exception as exc:
                        if self.config.skip_decode_errors:
                            log_webdataset_decode_skip(
                                key=key,
                                shard_name=shard_path.name,
                                audio_member=sample.get("audio_member"),
                                json_member=sample.get("json_member"),
                                exc=exc,
                            )
                            pending.pop(key, None)
                            continue
                        raise
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
                        try:
                            yield self._decode_sample(key, sample["audio"], sample["json"], metadata=metadata)
                        except Exception as exc:
                            if not self.config.skip_decode_errors:
                                raise
                            log_webdataset_decode_skip(
                                key=key,
                                shard_name=shard_path.name,
                                audio_member=sample.get("audio_member"),
                                json_member=sample.get("json_member"),
                                exc=exc,
                            )
                    pending.pop(key, None)

    def _decode_sample(
        self,
        key: str,
        audio_bytes: bytes,
        metadata_bytes: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return decode_webdataset_sample(
            key=key,
            audio_bytes=audio_bytes,
            metadata_bytes=metadata_bytes,
            tokenizer=self.tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            feature_extractor=self.feature_extractor,
            text_key=self.config.text_key,
            utt_id_key=self.config.utt_id_key,
            token_ids_key=self.config.token_ids_key,
            append_eos=self.config.append_eos,
            decoder_append_eos=self.config.decoder_append_eos,
            text_normalization=self.config.text_normalization,
            decoder_text_normalization=self.config.decoder_text_normalization,
            decoder_prompt_before_audio=self.config.decoder_prompt_before_audio,
            decoder_prompt_before_audio_use_language=self.config.decoder_prompt_before_audio_use_language,
            decoder_ctc_draft_cache_path=self.config.decoder_ctc_draft_cache_path,
            decoder_ctc_draft_prompt_template=self.config.decoder_ctc_draft_prompt_template,
            decoder_ctc_draft_text_key=self.config.decoder_ctc_draft_text_key,
            decoder_ctc_draft_missing_policy=self.config.decoder_ctc_draft_missing_policy,
            decoder_ctc_draft_dropout_prob=self.config.decoder_ctc_draft_dropout_prob,
            decoder_ctc_draft_language_mismatch_dropout_prob=(
                self.config.decoder_ctc_draft_language_mismatch_dropout_prob
            ),
            decoder_ctc_draft_dropout_seed=self.config.decoder_ctc_draft_dropout_seed,
                ctc_label_override_cache_path=self.config.ctc_label_override_cache_path,
                ctc_label_override_text_key=self.config.ctc_label_override_text_key,
                allow_missing_targets=self.config.allow_missing_targets,
                decoder_target_prefix=self.config.decoder_target_prefix,
            decoder_target_prefix_use_language=self.config.decoder_target_prefix_use_language,
            decoder_language_confirmation_en=self.config.decoder_language_confirmation_en,
            decoder_language_confirmation_zh=self.config.decoder_language_confirmation_zh,
            decoder_prompt_language_label_noise_prob=self.config.decoder_prompt_language_label_noise_prob,
            decoder_prompt_language_label_noise_seed=self.config.decoder_prompt_language_label_noise_seed,
            metadata=metadata,
        )

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for shard_path in self._partition_shards():
            yield from self._iter_shard(shard_path)


def build_webdataset_dataloader(
    shard_root: str | Path,
    *,
    tokenizer: TokenizerLike | None = None,
    decoder_tokenizer: TokenizerLike | None = None,
    feature_extractor: WenetFbankFeatureExtractor | None = None,
    config: WebDatasetConfig | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    dataset = WebDatasetASRIterableDataset(
        shard_root,
        tokenizer=tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        feature_extractor=feature_extractor,
        config=config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=FeatureCollator(),
    )
