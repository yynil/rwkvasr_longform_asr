import torch

from rwkvasr.training import ctc_batch_token_stats, estimate_token_budget_from_memory
from rwkvasr.training.batch_budget import (
    iter_budgeted_ctc_batches,
    select_ctc_batch_prefix_by_token_budget,
    split_ctc_batch_by_token_budget,
)
from rwkvasr.training.ctc_task import CTCBatch


def _batch() -> CTCBatch:
    features = torch.zeros(3, 8, 80)
    feature_lengths = torch.tensor([8, 5, 3], dtype=torch.long)
    targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    target_lengths = torch.tensor([3, 2, 1], dtype=torch.long)
    return CTCBatch(
        features=features,
        feature_lengths=feature_lengths,
        targets=targets,
        target_lengths=target_lengths,
    )


def test_ctc_batch_token_stats_uses_padded_audio_plus_text() -> None:
    stats = ctc_batch_token_stats(_batch())

    assert stats.batch_size == 3
    assert stats.max_audio_frames == 8
    assert stats.padded_audio_tokens == 24
    assert stats.text_tokens == 6
    assert stats.total_tokens == 30


def test_estimate_token_budget_from_memory_scales_linearly() -> None:
    budget = estimate_token_budget_from_memory(
        observed_tokens=3000,
        observed_peak_reserved_bytes=6 * (1024**3),
        target_memory_gib=22.0,
    )

    assert budget == 11000


def test_split_ctc_batch_by_token_budget_splits_and_preserves_targets() -> None:
    sub_batches, skipped = split_ctc_batch_by_token_budget(
        _batch(),
        token_budget=18,
        skip_oversized_samples=False,
    )

    assert skipped == 0
    assert len(sub_batches) == 2
    assert [sub.features.size(0) for sub in sub_batches] == [1, 2]
    assert [sub.target_lengths.tolist() for sub in sub_batches] == [[3], [2, 1]]


def test_split_ctc_batch_by_token_budget_can_skip_oversized_samples() -> None:
    sub_batches, skipped = split_ctc_batch_by_token_budget(
        _batch(),
        token_budget=6,
        skip_oversized_samples=True,
    )

    assert skipped == 2
    assert len(sub_batches) == 1
    assert sub_batches[0].features.size(0) == 1
    assert sub_batches[0].feature_lengths.tolist() == [3]


def test_iter_budgeted_ctc_batches_yields_one_budgeted_batch_per_next() -> None:
    results = list(
        iter_budgeted_ctc_batches(
            [_batch()],
            token_budget=18,
            max_batch_size=3,
            skip_oversized_samples=False,
        )
    )

    assert len(results) == 2
    assert [result.batch.features.size(0) for result in results] == [1, 2]
    assert all(ctc_batch_token_stats(result.batch).total_tokens <= 18 for result in results)


def test_select_ctc_batch_prefix_by_token_budget_shrinks_batch_without_replaying_tail() -> None:
    result = select_ctc_batch_prefix_by_token_budget(
        _batch(),
        token_budget=18,
        skip_oversized_samples=False,
    )

    assert result is not None
    assert result.batch.features.size(0) == 1
    assert result.dropped_tail_samples == 2
    assert result.skipped_samples == 0
    assert ctc_batch_token_stats(result.batch).total_tokens <= 18
