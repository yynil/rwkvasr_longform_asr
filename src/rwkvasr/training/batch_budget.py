from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch

from .ctc_task import CTCBatch


@dataclass(frozen=True)
class BatchTokenStats:
    batch_size: int
    max_audio_frames: int
    sum_audio_frames: int
    padded_audio_tokens: int
    text_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class BudgetedBatchResult:
    batch: CTCBatch
    skipped_samples: int
    dropped_tail_samples: int


def ctc_batch_token_stats(batch: CTCBatch) -> BatchTokenStats:
    batch_size = int(batch.features.size(0))
    if batch_size == 0:
        return BatchTokenStats(
            batch_size=0,
            max_audio_frames=0,
            sum_audio_frames=0,
            padded_audio_tokens=0,
            text_tokens=0,
            total_tokens=0,
        )
    feature_lengths = batch.feature_lengths.to(dtype=torch.long)
    target_lengths = batch.target_lengths.to(dtype=torch.long)
    max_audio_frames = int(feature_lengths.max().item())
    sum_audio_frames = int(feature_lengths.sum().item())
    padded_audio_tokens = batch_size * max_audio_frames
    text_tokens = int(target_lengths.sum().item())
    return BatchTokenStats(
        batch_size=batch_size,
        max_audio_frames=max_audio_frames,
        sum_audio_frames=sum_audio_frames,
        padded_audio_tokens=padded_audio_tokens,
        text_tokens=text_tokens,
        total_tokens=padded_audio_tokens + text_tokens,
    )


def estimate_token_budget_from_memory(
    *,
    observed_tokens: int,
    observed_peak_reserved_bytes: int,
    target_memory_gib: float,
) -> int:
    if observed_tokens <= 0 or observed_peak_reserved_bytes <= 0 or target_memory_gib <= 0:
        return 0
    target_bytes = int(target_memory_gib * (1024**3))
    return max(1, int(observed_tokens * target_bytes / observed_peak_reserved_bytes))


def split_ctc_batch_by_token_budget(
    batch: CTCBatch,
    *,
    token_budget: int,
    skip_oversized_samples: bool,
) -> tuple[list[CTCBatch], int]:
    if token_budget <= 0:
        return [batch], 0

    feature_lengths = batch.feature_lengths.to(dtype=torch.long)
    target_lengths = batch.target_lengths.to(dtype=torch.long)

    target_offset = 0
    sample_targets: list[torch.Tensor] = []
    for target_len in target_lengths.tolist():
        sample_targets.append(batch.targets[target_offset : target_offset + target_len])
        target_offset += target_len

    emitted_batches: list[CTCBatch] = []
    skipped_samples = 0
    pending_features: list[torch.Tensor] = []
    pending_feature_lengths: list[int] = []
    pending_targets: list[torch.Tensor] = []
    pending_target_lengths: list[int] = []

    def _pending_total_tokens(next_feature_len: int, next_target_len: int) -> int:
        next_batch_size = len(pending_features) + 1
        next_max_frames = max([next_feature_len, *pending_feature_lengths], default=next_feature_len)
        return next_batch_size * next_max_frames + sum(pending_target_lengths) + next_target_len

    def _emit_pending() -> None:
        if not pending_features:
            return
        max_frames = max(pending_feature_lengths)
        feat_dim = int(pending_features[0].size(-1))
        dtype = pending_features[0].dtype
        device = pending_features[0].device
        batch_features = torch.zeros(len(pending_features), max_frames, feat_dim, dtype=dtype, device=device)
        batch_feature_lengths = torch.tensor(pending_feature_lengths, dtype=torch.long, device=device)
        batch_target_lengths = torch.tensor(pending_target_lengths, dtype=torch.long, device=device)
        total_targets = sum(pending_target_lengths)
        batch_targets = torch.zeros(total_targets, dtype=torch.long, device=device)

        offset = 0
        for sample_idx, (features, feature_len, targets, target_len) in enumerate(
            zip(
                pending_features,
                pending_feature_lengths,
                pending_targets,
                pending_target_lengths,
                strict=True,
            )
        ):
            batch_features[sample_idx, :feature_len] = features[:feature_len]
            batch_targets[offset : offset + target_len] = targets[:target_len]
            offset += target_len

        emitted_batches.append(
            CTCBatch(
                features=batch_features,
                feature_lengths=batch_feature_lengths,
                targets=batch_targets,
                target_lengths=batch_target_lengths,
            )
        )
        pending_features.clear()
        pending_feature_lengths.clear()
        pending_targets.clear()
        pending_target_lengths.clear()

    for sample_idx in range(batch.features.size(0)):
        feature_len = int(feature_lengths[sample_idx].item())
        target_len = int(target_lengths[sample_idx].item())
        sample_features = batch.features[sample_idx, :feature_len]
        sample_target = sample_targets[sample_idx]
        sample_tokens = feature_len + target_len

        if sample_tokens > token_budget and skip_oversized_samples:
            skipped_samples += 1
            continue

        if pending_features and _pending_total_tokens(feature_len, target_len) > token_budget:
            _emit_pending()

        if sample_tokens > token_budget and not pending_features:
            pending_features.append(sample_features)
            pending_feature_lengths.append(feature_len)
            pending_targets.append(sample_target)
            pending_target_lengths.append(target_len)
            _emit_pending()
            continue

        pending_features.append(sample_features)
        pending_feature_lengths.append(feature_len)
        pending_targets.append(sample_target)
        pending_target_lengths.append(target_len)

    _emit_pending()
    return emitted_batches, skipped_samples


def _iter_ctc_samples(batch: CTCBatch) -> list[tuple[torch.Tensor, int, torch.Tensor, int]]:
    feature_lengths = batch.feature_lengths.to(dtype=torch.long)
    target_lengths = batch.target_lengths.to(dtype=torch.long)
    target_offset = 0
    samples: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
    for sample_idx in range(batch.features.size(0)):
        feature_len = int(feature_lengths[sample_idx].item())
        target_len = int(target_lengths[sample_idx].item())
        target = batch.targets[target_offset : target_offset + target_len]
        target_offset += target_len
        samples.append((batch.features[sample_idx, :feature_len], feature_len, target, target_len))
    return samples


def _build_ctc_batch_from_samples(
    samples: list[tuple[torch.Tensor, int, torch.Tensor, int]],
) -> CTCBatch:
    if not samples:
        raise ValueError("samples must not be empty")
    max_frames = max(feature_len for _, feature_len, _, _ in samples)
    feat_dim = int(samples[0][0].size(-1))
    dtype = samples[0][0].dtype
    device = samples[0][0].device
    batch_features = torch.zeros(len(samples), max_frames, feat_dim, dtype=dtype, device=device)
    batch_feature_lengths = torch.tensor([feature_len for _, feature_len, _, _ in samples], dtype=torch.long, device=device)
    batch_target_lengths = torch.tensor([target_len for _, _, _, target_len in samples], dtype=torch.long, device=device)
    total_targets = int(batch_target_lengths.sum().item())
    batch_targets = torch.zeros(total_targets, dtype=torch.long, device=device)

    offset = 0
    for sample_idx, (features, feature_len, targets, target_len) in enumerate(samples):
        batch_features[sample_idx, :feature_len] = features[:feature_len]
        batch_targets[offset : offset + target_len] = targets[:target_len]
        offset += target_len

    return CTCBatch(
        features=batch_features,
        feature_lengths=batch_feature_lengths,
        targets=batch_targets,
        target_lengths=batch_target_lengths,
    )


def iter_budgeted_ctc_batches(
    candidate_batches: Iterable[CTCBatch],
    *,
    token_budget: int | None,
    max_batch_size: int,
    skip_oversized_samples: bool,
) -> Iterator[BudgetedBatchResult]:
    pending_samples: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
    skipped_samples = 0

    def _pending_total_tokens(next_feature_len: int, next_target_len: int) -> int:
        next_batch_size = len(pending_samples) + 1
        next_max_frames = max([next_feature_len, *[feature_len for _, feature_len, _, _ in pending_samples]], default=next_feature_len)
        next_text_tokens = sum(target_len for _, _, _, target_len in pending_samples) + next_target_len
        return next_batch_size * next_max_frames + next_text_tokens

    def _emit_pending() -> BudgetedBatchResult | None:
        nonlocal pending_samples, skipped_samples
        if not pending_samples:
            return None
        batch = _build_ctc_batch_from_samples(pending_samples)
        result = BudgetedBatchResult(batch=batch, skipped_samples=skipped_samples, dropped_tail_samples=0)
        pending_samples = []
        skipped_samples = 0
        return result

    for candidate_batch in candidate_batches:
        for sample in _iter_ctc_samples(candidate_batch):
            _, feature_len, _, target_len = sample
            sample_tokens = feature_len + target_len
            if token_budget and sample_tokens > token_budget and skip_oversized_samples:
                skipped_samples += 1
                continue

            over_max_batch = len(pending_samples) >= max_batch_size
            over_token_budget = bool(token_budget) and pending_samples and _pending_total_tokens(feature_len, target_len) > token_budget
            if over_max_batch or over_token_budget:
                result = _emit_pending()
                if result is not None:
                    yield result

            pending_samples.append(sample)

            if token_budget and sample_tokens > token_budget and not skip_oversized_samples:
                result = _emit_pending()
                if result is not None:
                    yield result

    result = _emit_pending()
    if result is not None:
        yield result


def select_ctc_batch_prefix_by_token_budget(
    batch: CTCBatch,
    *,
    token_budget: int | None,
    skip_oversized_samples: bool,
) -> BudgetedBatchResult | None:
    if not token_budget or token_budget <= 0:
        return BudgetedBatchResult(batch=batch, skipped_samples=0, dropped_tail_samples=0)

    selected_samples: list[tuple[torch.Tensor, int, torch.Tensor, int]] = []
    skipped_samples = 0
    dropped_tail_samples = 0

    for sample_idx, sample in enumerate(_iter_ctc_samples(batch)):
        _, feature_len, _, target_len = sample
        sample_tokens = feature_len + target_len

        if sample_tokens > token_budget and skip_oversized_samples:
            skipped_samples += 1
            continue

        candidate_samples = [*selected_samples, sample]
        candidate_batch = _build_ctc_batch_from_samples(candidate_samples)
        candidate_tokens = ctc_batch_token_stats(candidate_batch).total_tokens
        if selected_samples and candidate_tokens > token_budget:
            dropped_tail_samples = batch.features.size(0) - sample_idx
            break

        selected_samples.append(sample)
        if candidate_tokens > token_budget:
            dropped_tail_samples = batch.features.size(0) - sample_idx - 1
            break

    if not selected_samples:
        return None

    return BudgetedBatchResult(
        batch=_build_ctc_batch_from_samples(selected_samples),
        skipped_samples=skipped_samples,
        dropped_tail_samples=dropped_tail_samples,
    )
