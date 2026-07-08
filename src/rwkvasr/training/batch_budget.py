from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch

from .ctc_task import CTCBatch


@dataclass(frozen=True)
class BatchTokenStats:
    batch_size: int
    max_audio_frames: int
    max_text_tokens: int
    sum_audio_frames: int
    padded_audio_tokens: int
    text_tokens: int
    padded_text_tokens: int
    budget_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class BudgetedBatchResult:
    batch: CTCBatch
    skipped_samples: int
    dropped_tail_samples: int


CTCSample = tuple[
    torch.Tensor,
    int,
    torch.Tensor,
    int,
    torch.Tensor | None,
    int | None,
    torch.Tensor | None,
    int | None,
    str | None,
    dict[str, Any] | None,
]


def ctc_batch_token_stats(batch: CTCBatch) -> BatchTokenStats:
    batch_size = int(batch.features.size(0))
    if batch_size == 0:
        return BatchTokenStats(
            batch_size=0,
            max_audio_frames=0,
            max_text_tokens=0,
            sum_audio_frames=0,
            padded_audio_tokens=0,
            text_tokens=0,
            padded_text_tokens=0,
            budget_tokens=0,
            total_tokens=0,
        )
    feature_lengths = batch.feature_lengths.to(dtype=torch.long)
    target_lengths = _budget_target_lengths(batch)
    max_audio_frames = int(feature_lengths.max().item())
    max_text_tokens = int(target_lengths.max().item())
    sum_audio_frames = int(feature_lengths.sum().item())
    padded_audio_tokens = batch_size * max_audio_frames
    text_tokens = int(target_lengths.sum().item())
    padded_text_tokens = batch_size * max_text_tokens
    return BatchTokenStats(
        batch_size=batch_size,
        max_audio_frames=max_audio_frames,
        max_text_tokens=max_text_tokens,
        sum_audio_frames=sum_audio_frames,
        padded_audio_tokens=padded_audio_tokens,
        text_tokens=text_tokens,
        padded_text_tokens=padded_text_tokens,
        budget_tokens=padded_audio_tokens + padded_text_tokens,
        total_tokens=padded_audio_tokens + text_tokens,
    )


def _budget_target_lengths(batch: CTCBatch) -> torch.Tensor:
    prompt_lengths = getattr(batch, "decoder_prompt_before_audio_lengths", None)
    if batch.decoder_target_lengths is not None:
        lengths = batch.decoder_target_lengths.to(dtype=torch.long)
    else:
        lengths = batch.target_lengths.to(dtype=torch.long)
    if prompt_lengths is not None:
        lengths = lengths + prompt_lengths.to(dtype=torch.long)
    return lengths


def _packed_samples(targets: torch.Tensor, target_lengths: torch.Tensor) -> list[torch.Tensor]:
    offset = 0
    samples: list[torch.Tensor] = []
    for target_len in target_lengths.to(dtype=torch.long).tolist():
        samples.append(targets[offset : offset + target_len])
        offset += target_len
    return samples


def effective_batch_token_budget(
    stats: BatchTokenStats,
    *,
    use_padded_text_tokens: bool = False,
    text_tokens_per_sample_extra: int = 0,
) -> int:
    if not use_padded_text_tokens:
        return stats.total_tokens
    return stats.budget_tokens + stats.batch_size * max(0, int(text_tokens_per_sample_extra))


def effective_padded_text_token_budget(
    stats: BatchTokenStats,
    *,
    text_tokens_per_sample_extra: int = 0,
) -> int:
    return stats.batch_size * (stats.max_text_tokens + max(0, int(text_tokens_per_sample_extra)))


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
    use_padded_text_tokens: bool = False,
    text_tokens_per_sample_extra: int = 0,
) -> tuple[list[CTCBatch], int]:
    if token_budget <= 0:
        return [batch], 0

    feature_lengths = batch.feature_lengths.to(dtype=torch.long)
    target_lengths = batch.target_lengths.to(dtype=torch.long)
    budget_target_lengths = _budget_target_lengths(batch)
    sample_targets = _packed_samples(batch.targets, target_lengths)
    sample_decoder_targets = (
        _packed_samples(batch.decoder_targets, batch.decoder_target_lengths)
        if batch.decoder_targets is not None and batch.decoder_target_lengths is not None
        else [None] * len(sample_targets)
    )
    sample_decoder_prompts = (
        _packed_samples(batch.decoder_prompt_before_audio, batch.decoder_prompt_before_audio_lengths)
        if batch.decoder_prompt_before_audio is not None
        and batch.decoder_prompt_before_audio_lengths is not None
        else [None] * len(sample_targets)
    )

    emitted_batches: list[CTCBatch] = []
    skipped_samples = 0
    pending_features: list[torch.Tensor] = []
    pending_feature_lengths: list[int] = []
    pending_targets: list[torch.Tensor] = []
    pending_target_lengths: list[int] = []
    pending_decoder_targets: list[torch.Tensor | None] = []
    pending_decoder_target_lengths: list[int | None] = []
    pending_decoder_prompts: list[torch.Tensor | None] = []
    pending_decoder_prompt_lengths: list[int | None] = []
    pending_utt_ids: list[str | None] = []
    pending_teacher_audio_rows: list[dict[str, Any] | None] = []

    def _pending_total_tokens(next_feature_len: int, next_budget_target_len: int) -> int:
        next_batch_size = len(pending_features) + 1
        next_max_frames = max([next_feature_len, *pending_feature_lengths], default=next_feature_len)
        if use_padded_text_tokens:
            pending_budget_lengths = [
                (int(length) if length is not None else int(target_len))
                + (int(prompt_len) if prompt_len is not None else 0)
                for length, target_len, prompt_len in zip(
                    pending_decoder_target_lengths,
                    pending_target_lengths,
                    pending_decoder_prompt_lengths,
                    strict=True,
                )
            ]
            next_max_text = max([next_budget_target_len, *pending_budget_lengths], default=next_budget_target_len)
            return next_batch_size * next_max_frames + next_batch_size * (next_max_text + text_tokens_per_sample_extra)
        pending_budget_sum = sum(
            (int(length) if length is not None else int(target_len))
            + (int(prompt_len) if prompt_len is not None else 0)
            for length, target_len, prompt_len in zip(
                pending_decoder_target_lengths,
                pending_target_lengths,
                pending_decoder_prompt_lengths,
                strict=True,
            )
        )
        return next_batch_size * next_max_frames + pending_budget_sum + next_budget_target_len

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
        has_decoder_targets = any(target is not None for target in pending_decoder_targets)
        batch_decoder_targets = None
        batch_decoder_target_lengths = None
        batch_decoder_prompt_before_audio = None
        batch_decoder_prompt_before_audio_lengths = None
        if has_decoder_targets:
            decoder_lengths = [
                int(length) if length is not None else int(target_len)
                for length, target_len in zip(
                    pending_decoder_target_lengths,
                    pending_target_lengths,
                    strict=True,
                )
            ]
            batch_decoder_target_lengths = torch.tensor(decoder_lengths, dtype=torch.long, device=device)
            batch_decoder_targets = torch.zeros(sum(decoder_lengths), dtype=torch.long, device=device)
        has_decoder_prompts = any(prompt is not None for prompt in pending_decoder_prompts)
        if has_decoder_prompts:
            prompt_lengths = [
                int(length) if length is not None else 0
                for length in pending_decoder_prompt_lengths
            ]
            batch_decoder_prompt_before_audio_lengths = torch.tensor(
                prompt_lengths,
                dtype=torch.long,
                device=device,
            )
            batch_decoder_prompt_before_audio = torch.zeros(
                sum(prompt_lengths),
                dtype=torch.long,
                device=device,
            )

        offset = 0
        decoder_offset = 0
        decoder_prompt_offset = 0
        for sample_idx, (
            features,
            feature_len,
            targets,
            target_len,
            decoder_targets,
            decoder_target_len,
            decoder_prompt,
            decoder_prompt_len,
            _teacher_audio_row,
        ) in enumerate(
            zip(
                pending_features,
                pending_feature_lengths,
                pending_targets,
                pending_target_lengths,
                pending_decoder_targets,
                pending_decoder_target_lengths,
                pending_decoder_prompts,
                pending_decoder_prompt_lengths,
                pending_teacher_audio_rows,
                strict=True,
            )
        ):
            batch_features[sample_idx, :feature_len] = features[:feature_len]
            batch_targets[offset : offset + target_len] = targets[:target_len]
            offset += target_len
            if batch_decoder_targets is not None and batch_decoder_target_lengths is not None:
                current_decoder_targets = decoder_targets if decoder_targets is not None else targets
                current_decoder_len = int(decoder_target_len) if decoder_target_len is not None else int(target_len)
                batch_decoder_targets[decoder_offset : decoder_offset + current_decoder_len] = current_decoder_targets[
                    :current_decoder_len
                ]
                decoder_offset += current_decoder_len
            if (
                batch_decoder_prompt_before_audio is not None
                and batch_decoder_prompt_before_audio_lengths is not None
                and decoder_prompt is not None
            ):
                current_prompt_len = int(decoder_prompt_len) if decoder_prompt_len is not None else 0
                batch_decoder_prompt_before_audio[
                    decoder_prompt_offset : decoder_prompt_offset + current_prompt_len
                ] = decoder_prompt[:current_prompt_len]
                decoder_prompt_offset += current_prompt_len

        emitted_batches.append(
            CTCBatch(
                features=batch_features,
                feature_lengths=batch_feature_lengths,
                targets=batch_targets,
                target_lengths=batch_target_lengths,
                decoder_targets=batch_decoder_targets,
                decoder_target_lengths=batch_decoder_target_lengths,
                decoder_prompt_before_audio=batch_decoder_prompt_before_audio,
                decoder_prompt_before_audio_lengths=batch_decoder_prompt_before_audio_lengths,
                utt_ids=[str(utt_id) for utt_id in pending_utt_ids] if any(utt_id is not None for utt_id in pending_utt_ids) else None,
                ctc_teacher_audio_rows=(
                    list(pending_teacher_audio_rows)
                    if any(row is not None for row in pending_teacher_audio_rows)
                    else None
                ),
            )
        )
        pending_features.clear()
        pending_feature_lengths.clear()
        pending_targets.clear()
        pending_target_lengths.clear()
        pending_decoder_targets.clear()
        pending_decoder_target_lengths.clear()
        pending_decoder_prompts.clear()
        pending_decoder_prompt_lengths.clear()
        pending_utt_ids.clear()
        pending_teacher_audio_rows.clear()

    for sample_idx in range(batch.features.size(0)):
        feature_len = int(feature_lengths[sample_idx].item())
        target_len = int(target_lengths[sample_idx].item())
        budget_target_len = int(budget_target_lengths[sample_idx].item())
        sample_features = batch.features[sample_idx, :feature_len]
        sample_target = sample_targets[sample_idx]
        sample_decoder_target = sample_decoder_targets[sample_idx]
        sample_decoder_prompt = sample_decoder_prompts[sample_idx]
        sample_decoder_target_len = (
            int(batch.decoder_target_lengths[sample_idx].item())
            if sample_decoder_target is not None and batch.decoder_target_lengths is not None
            else None
        )
        sample_decoder_prompt_len = (
            int(batch.decoder_prompt_before_audio_lengths[sample_idx].item())
            if sample_decoder_prompt is not None and batch.decoder_prompt_before_audio_lengths is not None
            else None
        )
        batch_utt_ids = getattr(batch, "utt_ids", None)
        sample_utt_id = (
            str(batch_utt_ids[sample_idx])
            if batch_utt_ids is not None and sample_idx < len(batch_utt_ids)
            else None
        )
        batch_teacher_audio_rows = getattr(batch, "ctc_teacher_audio_rows", None)
        sample_teacher_audio_row = (
            batch_teacher_audio_rows[sample_idx]
            if batch_teacher_audio_rows is not None and sample_idx < len(batch_teacher_audio_rows)
            else None
        )
        sample_tokens = feature_len + (
            budget_target_len + text_tokens_per_sample_extra if use_padded_text_tokens else budget_target_len
        )

        if token_budget and sample_tokens > token_budget and skip_oversized_samples:
            skipped_samples += 1
            continue

        if pending_features and _pending_total_tokens(feature_len, budget_target_len) > token_budget:
            _emit_pending()

        if token_budget and sample_tokens > token_budget and not pending_features:
            pending_features.append(sample_features)
            pending_feature_lengths.append(feature_len)
            pending_targets.append(sample_target)
            pending_target_lengths.append(target_len)
            pending_decoder_targets.append(sample_decoder_target)
            pending_decoder_target_lengths.append(sample_decoder_target_len)
            pending_decoder_prompts.append(sample_decoder_prompt)
            pending_decoder_prompt_lengths.append(sample_decoder_prompt_len)
            pending_utt_ids.append(sample_utt_id)
            pending_teacher_audio_rows.append(sample_teacher_audio_row)
            _emit_pending()
            continue

        pending_features.append(sample_features)
        pending_feature_lengths.append(feature_len)
        pending_targets.append(sample_target)
        pending_target_lengths.append(target_len)
        pending_decoder_targets.append(sample_decoder_target)
        pending_decoder_target_lengths.append(sample_decoder_target_len)
        pending_decoder_prompts.append(sample_decoder_prompt)
        pending_decoder_prompt_lengths.append(sample_decoder_prompt_len)
        pending_utt_ids.append(sample_utt_id)
        pending_teacher_audio_rows.append(sample_teacher_audio_row)

    _emit_pending()
    return emitted_batches, skipped_samples


def _iter_ctc_samples(
    batch: CTCBatch,
) -> list[CTCSample]:
    feature_lengths = batch.feature_lengths.to(dtype=torch.long)
    target_lengths = batch.target_lengths.to(dtype=torch.long)
    sample_targets = _packed_samples(batch.targets, target_lengths)
    sample_decoder_targets = (
        _packed_samples(batch.decoder_targets, batch.decoder_target_lengths)
        if batch.decoder_targets is not None and batch.decoder_target_lengths is not None
        else [None] * len(sample_targets)
    )
    sample_decoder_prompts = (
        _packed_samples(batch.decoder_prompt_before_audio, batch.decoder_prompt_before_audio_lengths)
        if batch.decoder_prompt_before_audio is not None
        and batch.decoder_prompt_before_audio_lengths is not None
        else [None] * len(sample_targets)
    )
    samples: list[CTCSample] = []
    batch_utt_ids = getattr(batch, "utt_ids", None)
    batch_teacher_audio_rows = getattr(batch, "ctc_teacher_audio_rows", None)
    for sample_idx in range(batch.features.size(0)):
        feature_len = int(feature_lengths[sample_idx].item())
        target_len = int(target_lengths[sample_idx].item())
        decoder_target = sample_decoder_targets[sample_idx]
        decoder_target_len = (
            int(batch.decoder_target_lengths[sample_idx].item())
            if decoder_target is not None and batch.decoder_target_lengths is not None
            else None
        )
        decoder_prompt = sample_decoder_prompts[sample_idx]
        decoder_prompt_len = (
            int(batch.decoder_prompt_before_audio_lengths[sample_idx].item())
            if decoder_prompt is not None and batch.decoder_prompt_before_audio_lengths is not None
            else None
        )
        sample_utt_id = (
            str(batch_utt_ids[sample_idx])
            if batch_utt_ids is not None and sample_idx < len(batch_utt_ids)
            else None
        )
        sample_teacher_audio_row = (
            batch_teacher_audio_rows[sample_idx]
            if batch_teacher_audio_rows is not None and sample_idx < len(batch_teacher_audio_rows)
            else None
        )
        samples.append(
            (
                batch.features[sample_idx, :feature_len],
                feature_len,
                sample_targets[sample_idx],
                target_len,
                decoder_target,
                decoder_target_len,
                decoder_prompt,
                decoder_prompt_len,
                sample_utt_id,
                sample_teacher_audio_row,
            )
        )
    return samples


def _build_ctc_batch_from_samples(
    samples: list[CTCSample],
) -> CTCBatch:
    if not samples:
        raise ValueError("samples must not be empty")
    max_frames = max(feature_len for _, feature_len, _, _, _, _, _, _, _, _ in samples)
    feat_dim = int(samples[0][0].size(-1))
    dtype = samples[0][0].dtype
    device = samples[0][0].device
    batch_features = torch.zeros(len(samples), max_frames, feat_dim, dtype=dtype, device=device)
    batch_feature_lengths = torch.tensor([feature_len for _, feature_len, _, _, _, _, _, _, _, _ in samples], dtype=torch.long, device=device)
    batch_target_lengths = torch.tensor([target_len for _, _, _, target_len, _, _, _, _, _, _ in samples], dtype=torch.long, device=device)
    total_targets = int(batch_target_lengths.sum().item())
    batch_targets = torch.zeros(total_targets, dtype=torch.long, device=device)
    has_decoder_targets = any(decoder_target is not None for _, _, _, _, decoder_target, _, _, _, _, _ in samples)
    batch_decoder_target_lengths = None
    batch_decoder_targets = None
    if has_decoder_targets:
        decoder_lengths = [
            int(decoder_target_len) if decoder_target_len is not None else int(target_len)
            for _, _, _, target_len, _, decoder_target_len, _, _, _, _ in samples
        ]
        batch_decoder_target_lengths = torch.tensor(decoder_lengths, dtype=torch.long, device=device)
        batch_decoder_targets = torch.zeros(sum(decoder_lengths), dtype=torch.long, device=device)
    has_decoder_prompts = any(decoder_prompt is not None for _, _, _, _, _, _, decoder_prompt, _, _, _ in samples)
    batch_decoder_prompt_before_audio = None
    batch_decoder_prompt_before_audio_lengths = None
    if has_decoder_prompts:
        prompt_lengths = [
            int(decoder_prompt_len) if decoder_prompt_len is not None else 0
            for _, _, _, _, _, _, _, decoder_prompt_len, _, _ in samples
        ]
        batch_decoder_prompt_before_audio_lengths = torch.tensor(
            prompt_lengths,
            dtype=torch.long,
            device=device,
        )
        batch_decoder_prompt_before_audio = torch.zeros(sum(prompt_lengths), dtype=torch.long, device=device)

    offset = 0
    decoder_offset = 0
    decoder_prompt_offset = 0
    for sample_idx, (
        features,
        feature_len,
        targets,
        target_len,
        decoder_targets,
        decoder_target_len,
        decoder_prompt,
        decoder_prompt_len,
        _utt_id,
        _teacher_audio_row,
    ) in enumerate(samples):
        batch_features[sample_idx, :feature_len] = features[:feature_len]
        batch_targets[offset : offset + target_len] = targets[:target_len]
        offset += target_len
        if batch_decoder_targets is not None and batch_decoder_target_lengths is not None:
            current_decoder_targets = decoder_targets if decoder_targets is not None else targets
            current_decoder_len = int(decoder_target_len) if decoder_target_len is not None else int(target_len)
            batch_decoder_targets[decoder_offset : decoder_offset + current_decoder_len] = current_decoder_targets[
                :current_decoder_len
            ]
            decoder_offset += current_decoder_len
        if (
            batch_decoder_prompt_before_audio is not None
            and batch_decoder_prompt_before_audio_lengths is not None
            and decoder_prompt is not None
        ):
            current_prompt_len = int(decoder_prompt_len) if decoder_prompt_len is not None else 0
            batch_decoder_prompt_before_audio[
                decoder_prompt_offset : decoder_prompt_offset + current_prompt_len
            ] = decoder_prompt[:current_prompt_len]
            decoder_prompt_offset += current_prompt_len

    return CTCBatch(
        features=batch_features,
        feature_lengths=batch_feature_lengths,
        targets=batch_targets,
        target_lengths=batch_target_lengths,
        decoder_targets=batch_decoder_targets,
        decoder_target_lengths=batch_decoder_target_lengths,
        decoder_prompt_before_audio=batch_decoder_prompt_before_audio,
        decoder_prompt_before_audio_lengths=batch_decoder_prompt_before_audio_lengths,
        utt_ids=[str(sample[-2]) for sample in samples] if any(sample[-2] is not None for sample in samples) else None,
        ctc_teacher_audio_rows=(
            [sample[-1] for sample in samples]
            if any(sample[-1] is not None for sample in samples)
            else None
        ),
    )


def iter_budgeted_ctc_batches(
    candidate_batches: Iterable[CTCBatch],
    *,
    token_budget: int | None,
    max_batch_size: int,
    skip_oversized_samples: bool,
    use_padded_text_tokens: bool = False,
    text_tokens_per_sample_extra: int = 0,
) -> Iterator[BudgetedBatchResult]:
    pending_samples: list[CTCSample] = []
    skipped_samples = 0

    def _pending_total_tokens(next_feature_len: int, next_target_len: int) -> int:
        next_batch_size = len(pending_samples) + 1
        next_max_frames = max([next_feature_len, *[feature_len for _, feature_len, _, _, _, _, _, _, _, _ in pending_samples]], default=next_feature_len)
        if use_padded_text_tokens:
            next_max_text = max(
                [
                    next_target_len,
                    *[
                        (int(decoder_target_len) if decoder_target_len is not None else int(target_len))
                        + (int(decoder_prompt_len) if decoder_prompt_len is not None else 0)
                        for _, _, _, target_len, _, decoder_target_len, _, decoder_prompt_len, _, _ in pending_samples
                    ],
                ],
                default=next_target_len,
            )
            return next_batch_size * next_max_frames + next_batch_size * (next_max_text + text_tokens_per_sample_extra)
        next_text_tokens = (
            sum(
                (int(decoder_target_len) if decoder_target_len is not None else int(target_len))
                + (int(decoder_prompt_len) if decoder_prompt_len is not None else 0)
                for _, _, _, target_len, _, decoder_target_len, _, decoder_prompt_len, _, _ in pending_samples
            )
            + next_target_len
        )
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
            _, feature_len, _, target_len, _, decoder_target_len, _, decoder_prompt_len, _, _ = sample
            budget_target_len = (
                (int(decoder_target_len) if decoder_target_len is not None else int(target_len))
                + (int(decoder_prompt_len) if decoder_prompt_len is not None else 0)
            )
            sample_tokens = feature_len + (
                budget_target_len + text_tokens_per_sample_extra if use_padded_text_tokens else budget_target_len
            )
            if token_budget and sample_tokens > token_budget and skip_oversized_samples:
                skipped_samples += 1
                continue

            over_max_batch = len(pending_samples) >= max_batch_size
            over_token_budget = bool(token_budget) and pending_samples and _pending_total_tokens(feature_len, budget_target_len) > token_budget
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
    use_padded_text_tokens: bool = False,
    text_tokens_per_sample_extra: int = 0,
    padded_text_token_budget: int | None = None,
) -> BudgetedBatchResult | None:
    if not token_budget or token_budget <= 0:
        if padded_text_token_budget is None or padded_text_token_budget <= 0:
            return BudgetedBatchResult(batch=batch, skipped_samples=0, dropped_tail_samples=0)

    selected_samples: list[CTCSample] = []
    skipped_samples = 0
    dropped_tail_samples = 0

    for sample_idx, sample in enumerate(_iter_ctc_samples(batch)):
        _, feature_len, _, target_len, _, decoder_target_len, _, decoder_prompt_len, _, _ = sample
        budget_target_len = (
            (int(decoder_target_len) if decoder_target_len is not None else int(target_len))
            + (int(decoder_prompt_len) if decoder_prompt_len is not None else 0)
        )
        sample_tokens = feature_len + (
            budget_target_len + text_tokens_per_sample_extra if use_padded_text_tokens else budget_target_len
        )
        sample_padded_text_tokens = budget_target_len + max(0, int(text_tokens_per_sample_extra))

        if token_budget and sample_tokens > token_budget and skip_oversized_samples:
            skipped_samples += 1
            continue
        if padded_text_token_budget and sample_padded_text_tokens > padded_text_token_budget and skip_oversized_samples:
            skipped_samples += 1
            continue

        candidate_samples = [*selected_samples, sample]
        candidate_batch = _build_ctc_batch_from_samples(candidate_samples)
        candidate_stats = ctc_batch_token_stats(candidate_batch)
        candidate_tokens = effective_batch_token_budget(
            candidate_stats,
            use_padded_text_tokens=use_padded_text_tokens,
            text_tokens_per_sample_extra=text_tokens_per_sample_extra,
        )
        candidate_padded_text_tokens = effective_padded_text_token_budget(
            candidate_stats,
            text_tokens_per_sample_extra=text_tokens_per_sample_extra,
        )
        over_total_budget = bool(token_budget) and candidate_tokens > token_budget
        over_text_budget = bool(padded_text_token_budget) and candidate_padded_text_tokens > padded_text_token_budget
        if selected_samples and (over_total_budget or over_text_budget):
            dropped_tail_samples = batch.features.size(0) - sample_idx
            break

        selected_samples.append(sample)
        if over_total_budget or over_text_budget:
            dropped_tail_samples = batch.features.size(0) - sample_idx - 1
            break

    if not selected_samples:
        return None

    return BudgetedBatchResult(
        batch=_build_ctc_batch_from_samples(selected_samples),
        skipped_samples=skipped_samples,
        dropped_tail_samples=dropped_tail_samples,
    )
