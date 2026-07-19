"""O1 — dist65 two-hot encode/decode primitives.

Gates the codec against a frozen old-side golden (`value_probes/dist65_golden.json`,
COPIED from old-side capture #1) + the mathematical two-hot identity. fp32 pinned.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mantis.model.dist65 import (
    N_VALUE_BINS,
    VALUE_SUPPORT,
    binned_value_loss,
    decode_binned_value,
    scalar_to_two_hot,
)

_GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "value_probes" / "dist65_golden.json"
_BIN_WIDTH = 2.0 / 64.0  # 0.03125


def _golden() -> dict:
    return json.loads(_GOLDEN.read_text())


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
