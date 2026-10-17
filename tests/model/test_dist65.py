"""O1 — dist65 two-hot encode/decode primitives.

Gates the codec against a frozen old-side golden (`value_probes/dist65_golden.json`,
COPIED from old-side capture #1) + the mathematical two-hot identity. fp32 pinned.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from mantis.model.dist65 import (
    N_VALUE_BINS,
    VALUE_SUPPORT,
    binned_value_loss,
    decode_binned_value,
    end_bin_value_loss,
    scalar_to_two_hot,
)

_GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes" / "dist65_golden.json"
_BIN_WIDTH = 2.0 / 64.0  # 0.03125


def _golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def test_constants_pinned() -> None:
    assert N_VALUE_BINS == 65
    assert torch.equal(VALUE_SUPPORT, torch.linspace(-1.0, 1.0, 65))
    assert VALUE_SUPPORT.dtype == torch.float32


def test_scalar_to_two_hot_byte_matches_golden() -> None:
    g = _golden()
    assert g["meta"]["N_VALUE_BINS"] == 65
    assert len(g["scalar_to_two_hot"]) >= 16
    for item in g["scalar_to_two_hot"]:
        z = torch.tensor([item["z"]], dtype=torch.float32)
        got = scalar_to_two_hot(z).reshape(-1).numpy().astype(np.float32)
        want = np.asarray(item["two_hot"], dtype=np.float32)
        assert np.array_equal(got, want), f"z={item['z']}: two_hot byte-mismatch"
        assert abs(float(got.sum()) - 1.0) < 1e-6


def test_decode_support_bins_roundtrip_identity() -> None:
    """decode(one-hot·30 at each support bin) returns that support value (< 1e-6)."""
    g = _golden()
    for item in g["decode_support_bins"]:
        b = item["bin"]
        one_hot = torch.zeros(1, N_VALUE_BINS, dtype=torch.float32)
        one_hot[0, b] = 30.0
        decoded = float(decode_binned_value(one_hot).item())
        assert abs(decoded - item["support_value"]) < 1e-6
        assert abs(decoded - float(VALUE_SUPPORT[b])) < 1e-6


def test_decode_log_two_hot_recovers_z() -> None:
    """decode(log(scalar_to_two_hot(z))) recovers z within one bin-width, and matches
    the golden decoded value within 1e-6 (the 'logit-form' = log(two_hot))."""
    g = _golden()
    for item in g["decode_twohot_as_logits"]:
        z = item["z"]
        two_hot = scalar_to_two_hot(torch.tensor([z], dtype=torch.float32))
        logits = torch.log(two_hot.clamp_min(0.0) + 0.0)  # log(0)->-inf on the zero bins; ok
        # guard the -inf: softmax handles it, but replace to avoid nan in edge float
        logits = torch.where(torch.isinf(logits), torch.full_like(logits, -1e30), logits)
        decoded = float(decode_binned_value(logits).item())
        assert abs(decoded - z) <= _BIN_WIDTH
        assert abs(decoded - item["decoded"]) < 1e-6


def test_binned_value_loss_mask_semantics() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, N_VALUE_BINS)
    outcome = torch.tensor([0.5, -0.5, 0.25, -1.0])

    # No mask == mean over all rows.
    full = binned_value_loss(logits, outcome)
    # value_mask==0 rows excluded from numerator AND denominator.
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    masked = binned_value_loss(logits, outcome, value_mask=mask)
    # Compute the reference: mean of per-row loss over kept rows {0, 2}.
    import torch.nn.functional as F
    target = scalar_to_two_hot(outcome)
    logp = F.log_softmax(logits.to(torch.float32), dim=-1)
    per_row = -(target * logp).sum(dim=-1)
    ref_kept = per_row[[0, 2]].mean()
    assert torch.allclose(masked, ref_kept)
    assert not torch.allclose(masked, full)  # different denominator

    # All-masked -> zeros(()).
    allzero = binned_value_loss(logits, outcome, value_mask=torch.zeros(4))
    assert allzero.shape == torch.Size([])
    assert float(allzero) == 0.0


def test_fp16_input_decodes_via_internal_fp32() -> None:
    """A fp16 bin-logit input still decodes (the internal `.to(float32)`)."""
    one_hot = torch.zeros(1, N_VALUE_BINS, dtype=torch.float16)
    one_hot[0, 48] = 30.0
    decoded = decode_binned_value(one_hot)
    assert decoded.dtype == torch.float32
    assert abs(float(decoded.item()) - float(VALUE_SUPPORT[48])) < 1e-3


def test_the_end_bin_loss_at_plus_and_minus_one_is_the_one_hot_z_loss_bit_for_bit() -> None:
    """At v = z = ±1 the two-point law IS one-hot z: the same rows over the same denominator give the same bits."""
    logits = torch.randn(6, N_VALUE_BINS, generator=torch.Generator().manual_seed(3))
    z = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    assert end_bin_value_loss(logits, z, 6.0).item() == binned_value_loss(logits, z, denominator=6.0).item()


def test_the_end_bin_target_has_no_interior_mass_and_a_planted_two_hot_reds() -> None:
    """d(loss)/d(logit) = (softmax − target)/n reads the target back: none inside, the win probability at the ends; PLANTED BREAK: two-hot reds."""
    logits = torch.randn(5, N_VALUE_BINS, generator=torch.Generator().manual_seed(4), requires_grad=True)
    v = torch.tensor([0.5, -0.25, 0.0, 0.9, -0.6])

    def target_of(loss_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]) -> torch.Tensor:
        logits.grad = None
        loss_fn(logits, v, 5.0).backward()
        assert logits.grad is not None
        return torch.softmax(logits.detach(), dim=-1) - 5.0 * logits.grad

    t = target_of(end_bin_value_loss)
    assert torch.allclose(t[:, 1:-1], torch.zeros_like(t[:, 1:-1]), atol=1e-6)
    assert torch.allclose(t[:, -1], (1.0 + v) / 2, atol=1e-6) and torch.allclose(t[:, 0], (1.0 - v) / 2, atol=1e-6)

    def two_hot(lg: torch.Tensor, val: torch.Tensor, den: float) -> torch.Tensor:
        return -(scalar_to_two_hot(val) * torch.log_softmax(lg, dim=-1)).sum() / den
    planted = target_of(two_hot)
    assert not torch.allclose(planted[:, 1:-1], torch.zeros_like(planted[:, 1:-1]), atol=1e-6)
