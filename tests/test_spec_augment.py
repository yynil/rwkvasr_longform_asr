import torch

from rwkvasr.training.spec_augment import apply_spec_augment


def test_spec_augment_keeps_shape_and_does_not_mutate_input() -> None:
    features = torch.ones(2, 8, 4)
    augmented = apply_spec_augment(
        features,
        torch.tensor([8, 5]),
        time_masks=1,
        time_width=3,
        freq_masks=1,
        freq_width=2,
    )
    assert augmented.shape == features.shape
    assert torch.all(features == 1)


def test_spec_augment_disabled_returns_same_tensor() -> None:
    features = torch.ones(1, 4, 3)
    augmented = apply_spec_augment(features, time_masks=0, freq_masks=0)
    assert augmented is features


def test_spec_augment_respects_feature_lengths() -> None:
    torch.manual_seed(0)
    features = torch.ones(1, 6, 3)
    augmented = apply_spec_augment(
        features,
        torch.tensor([3]),
        time_masks=0,
        freq_masks=4,
        freq_width=3,
    )
    assert torch.all(augmented[:, 3:, :] == 1)
