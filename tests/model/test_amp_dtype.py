"""O6 — bf16-graph LAW-06 regime-parity.

The unconditional graph→bf16 code pin is the fully-WP9 protection (F-11: fp16 GINE
sum-aggregation overflows 65504 → NaN). No config value may flip graph off bf16.
The "suite default == production graph config" sub-leg is DEFERRED to WP8/WP10
(the minted production config is unported here) — a noted seam, not a WP9 failure.
"""
from __future__ import annotations

import pytest
import torch

from mantis.model.amp import amp_dtype_for


@pytest.mark.parametrize("cfg", [None, {}, {"amp_dtype": "fp16"}, {"amp_dtype": "bf16"}])
def test_graph_is_bf16_unconditionally(cfg) -> None:
    assert amp_dtype_for("graph", cfg) is torch.bfloat16


def test_grid_regime_parity() -> None:
    assert amp_dtype_for("grid", {"amp_dtype": "fp16"}) is torch.float16
    assert amp_dtype_for("grid", {"amp_dtype": "bf16"}) is torch.bfloat16
    assert amp_dtype_for("grid", None) is torch.float16   # default fp16
    assert amp_dtype_for("grid", {}) is torch.float16


def test_grid_invalid_amp_dtype_raises() -> None:
    with pytest.raises(ValueError):
        amp_dtype_for("grid", {"amp_dtype": "garbage"})
