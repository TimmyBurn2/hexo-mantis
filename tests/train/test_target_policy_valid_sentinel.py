"""⊕ WP12-R Phase T (TARGET INTEGRITY) — S2c PYTHON LEG: the `policy_valid` zero-row
sentinel end-to-end contract on the dense-ls route (DESIGN_T §3.5, flip-set row 10;
Q1 RATIFIED with conditions, REVIEW_DESIGN_T §3).

Post-§3.5, a grid-ls cluster row whose window sees zero visit mass records as the
ALL-ZERO row — the pipeline's pre-existing value-only sentinel. These oracles pin the
CONSUMER half of that ratification, on the exact predicate + loss the trainer runs
(`trainer/core.py:397` `policy_valid = policies_t.sum(dim=1) > 1e-6`;
`losses.py:54-67` indexes numerator AND denominator by the mask):

 1. zero rows are excluded from numerator AND denominator — NOT a dilutant (the
    verified dense/ragged asymmetry Q1 rides);
 2. a FABRICATED UNIFORM row (the pre-fix 2c object) passes the mask and CHANGES the
    loss at full weight — the measured reason the uniform fill must be unconstructible
    (this stays true post-fix; it is the harm pin, not a defect pin).

Deviation recorded in ORACLE_NOTES_T.md: this leg drives the trainer's OWN predicate +
loss functions on locally-built batches rather than a full `train_step_from_tensors`
run — the predicate and the mask-indexed loss are the end-to-end-relevant behaviors,
and both are exercised verbatim.

PRE-FIX status at HEAD: GREEN (consumer-side pins; the Rust zero-row producer oracle
`target_gridls_projection.rs::s2c_zero_visible_window_yields_the_zero_row_not_uniform`
is the RED half). Killers: M-E (Rust half); these legs red under a consumer-side mask
removal (the mutation class LAW-07 requires them to be sensitive to is exercised by
deleting the mask indexing — covered by the loss-difference assert in test 2).
"""
from __future__ import annotations

import torch

from mantis.train.losses import compute_policy_loss

_STRIDE = 362


def _log_policy(b: int) -> torch.Tensor:
    torch.manual_seed(20260731)
    return torch.log_softmax(torch.randn(b, _STRIDE), dim=1)


def _valid_rows(n: int) -> torch.Tensor:
    torch.manual_seed(97)
    t = torch.rand(n, _STRIDE)
    return t / t.sum(dim=1, keepdim=True)


def test_zero_rows_leave_numerator_and_denominator() -> None:
    valid = _valid_rows(3)
    zeros = torch.zeros(2, _STRIDE)
    batch = torch.cat([valid, zeros], dim=0)

    # the trainer's exact predicate (core.py:397)
    policy_valid = batch.sum(dim=1) > 1e-6
    assert policy_valid.tolist() == [True, True, True, False, False], (
        "the zero row must read policy_valid=False — flip-set row 10's Python leg"
    )

    logp5 = _log_policy(5)
    with_zeros = compute_policy_loss(logp5, batch, policy_valid, torch.device("cpu"))
    without = compute_policy_loss(logp5[:3], valid, policy_valid[:3], torch.device("cpu"))
    assert torch.isfinite(with_zeros) and float(without) > 0.0
    assert abs(float(with_zeros) - float(without)) <= 1e-7, (
        "appending policy_valid-masked zero rows changed the policy loss — the dense "
        "mask must exclude them from numerator AND denominator (losses.py:62-67); "
        "dilution here would void the Q1 ratification grounds"
    )


def test_fabricated_uniform_rows_pass_the_mask_and_move_the_loss() -> None:
    logp5 = _log_policy(5)
    # Valid rows = one-hots on each row's argmax (a well-trained target: LOW CE);
    # fabricated rows = uniform over 362 (CE ~= the mean -logp: HIGH) — the loss shift
    # a fabricated row inflicts at full weight is then unambiguous.
    valid = torch.zeros(3, _STRIDE)
    valid[torch.arange(3), logp5[:3].argmax(dim=1)] = 1.0
    uniform = torch.full((2, _STRIDE), 1.0 / _STRIDE)
    batch = torch.cat([valid, uniform], dim=0)

    policy_valid = batch.sum(dim=1) > 1e-6
    assert policy_valid.all(), (
        "a fabricated uniform row passes policy_valid (sum == 1) — it trains at FULL "
        "weight, which is why §3.5 makes it unconstructible rather than masked"
    )

    with_uniform = compute_policy_loss(logp5, batch, policy_valid, torch.device("cpu"))
    without = compute_policy_loss(logp5[:3], valid, policy_valid[:3], torch.device("cpu"))
    assert float(with_uniform) - float(without) > 0.1, (
        "the fabricated uniform rows did not move the loss — the harm pin is vacuous "
        f"(with {float(with_uniform)}, without {float(without)})"
    )
