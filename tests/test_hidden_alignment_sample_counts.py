import pytest
import torch

from rwkvasr.training.deepspeed_loop import (
    _ctc_teacher_decoder_hidden_loss,
    _ctc_teacher_layer_hidden_loss,
)


def _teacher_record():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]])
    return {
        "encoder_layer_hiddens": {
            str(layer): {name: hidden for name in ("mixer", "ffn", "block")}
            for layer in range(2)
        },
        "ctc_decoder_hiddens": {
            name: hidden for name in ("input", *(f"layer_{i}" for i in range(5)))
        },
        "topk_token_ids": torch.tensor([[1], [0], [2]]),
        "topk_log_probs": torch.zeros(3, 1),
        "project_blank_id": 0,
        "project_ignored_token_ids": [],
    }


def _hidden_result(kind, student, lengths, ids, records, *, balance="all", policy="error"):
    options = dict(
        normalized_mse_weight=1.0,
        cosine_weight=0.25,
        energy_mse_weight=0.25,
        log_rms_weight=0.1,
        raw_mse_weight=0.5,
        frame_tolerance=0,
        missing_policy=policy,
        frame_balance_mode=balance,
        blank_id=0,
    )
    if kind == "layer":
        return _ctc_teacher_layer_hidden_loss(
            {i: {name: student for name in ("mixer", "ffn", "block")} for i in range(2)},
            lengths,
            ids,
            records,
            layer_ids=(0, 1),
            component_weights={"mixer": 0.25, "ffn": 0.25, "block": 1.0},
            **options,
        )
    return _ctc_teacher_decoder_hidden_loss(
        {name: student for name in ("input", *(f"layer_{i}" for i in range(5)))},
        lengths,
        ids,
        records,
        **options,
    )


@pytest.mark.parametrize("kind", ["layer", "decoder"])
@pytest.mark.parametrize("balance", ["all", "teacher_top1_balanced"])
def test_hidden_matches_count_duplicate_positions_without_changing_loss_or_gradients(kind, balance):
    student = (torch.arange(18, dtype=torch.float32).reshape(3, 3, 2) / 10 + 1).requires_grad_()
    lengths = torch.tensor([3, 3, 0])
    record = _teacher_record()
    duplicate = _hidden_result(
        kind, student, lengths, ("same", "same", "same"), {"same": record}, balance=balance
    )
    unique = _hidden_result(
        kind, student, lengths, ("a", "b", "c"), dict.fromkeys(("a", "b", "c"), record),
        balance=balance,
    )
    assert duplicate.matched_samples == unique.matched_samples == 2
    assert duplicate.missing_samples == unique.missing_samples == 0
    assert duplicate.events == unique.events == 12
    assert duplicate.max_frame_delta == unique.max_frame_delta == 0
    assert torch.equal(duplicate.loss, unique.loss)
    actual_grad = torch.autograd.grad(duplicate.loss, student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(unique.loss, student)[0]
    assert torch.equal(actual_grad, expected_grad)
    assert torch.count_nonzero(actual_grad[:2]) > 0
    assert torch.count_nonzero(actual_grad[2]) == 0


@pytest.mark.parametrize("kind", ["layer", "decoder"])
@pytest.mark.parametrize("missing", ["record", "hiddens", "top1"])
def test_hidden_missing_counts_repeated_positions_and_preserves_error_policy(kind, missing):
    student = torch.ones(4, 3, 2, requires_grad=True)
    lengths = torch.tensor([3, 3, 3, 3])
    record = _teacher_record()
    bad_record = dict(record)
    balance = "teacher_top1_balanced" if missing == "top1" else "all"
    if missing == "hiddens":
        bad_record.pop("encoder_layer_hiddens" if kind == "layer" else "ctc_decoder_hiddens")
    elif missing == "top1":
        bad_record.pop("topk_token_ids")
    records = {"good": record}
    if missing != "record":
        records["bad"] = bad_record
    ids = ("good", "good", "bad", "bad")
    result = _hidden_result(kind, student, lengths, ids, records, balance=balance, policy="skip")
    assert result.matched_samples == 2
    assert result.missing_samples == 2
    assert result.events == 12
    result.loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[:2]) > 0
    assert torch.count_nonzero(student.grad[2:]) == 0
    with pytest.raises((KeyError, ValueError)):
        _hidden_result(kind, student, lengths, ids, records, balance=balance, policy="error")
