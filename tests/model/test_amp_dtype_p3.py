"""WPSC Phase 3 SC-B3 — single amp authority, new 2-arg `amp_dtype_for(representation,
declared_amp_dtype)` signature (R30b; DESIGN_P3.md §4.3/§4.4). Replaces the OLD 1-arg-dict
`amp_dtype_for(representation, config_dict)` shape `tests/model/test_amp_dtype.py` pins
today.

DEVIATION FROM PREREG PATH (logged in ORACLE_NOTES_P3.md): PREREG names this a REWRITE of
the existing `tests/model/test_amp_dtype.py`. ORACLE-WRITE's writable surface is NEW files
only — the rewritten content lives here (`_p3` suffix), matching the ORACLE_NOTES_P2.md
house convention; the OLD file is untouched and stays green at HEAD (still exercising the
old signature that today's `amp_dtype_for` still has). IMPL replaces the old file's content
with this one's at port time and deletes this staging copy.

RED at HEAD (`507c23b`): `amp_dtype_for` still takes `(representation, config: dict |
None)`, not `(representation, declared_amp_dtype: str)` — every call below either raises
TypeError-shaped or resolves through the wrong (dict-reading) code path.

RED-TEAM amp probe (PREREG_P3.md lens #2): `test_graph_is_bf16_unconditionally` feeds the
EXACT value `configs/run5.yaml`/`configs/smoke_gnn.yaml` mint today on a graph run
(`declared_amp_dtype="fp16"`) and proves the merged single authority still resolves bf16.
"""
from __future__ import annotations

import pytest
import torch

from mantis.model.amp import amp_dtype_for


@pytest.mark.parametrize("declared_amp_dtype", ["fp16", "bf16"])
def test_graph_is_bf16_unconditionally(declared_amp_dtype: str) -> None:
    assert amp_dtype_for("graph", declared_amp_dtype) is torch.bfloat16


def test_grid_regime_parity() -> None:
    assert amp_dtype_for("grid", "fp16") is torch.float16
    assert amp_dtype_for("grid", "bf16") is torch.bfloat16


def test_grid_invalid_amp_dtype_raises() -> None:
    with pytest.raises(ValueError):
        amp_dtype_for("grid", "garbage")
