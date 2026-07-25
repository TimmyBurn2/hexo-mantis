"""O6 — bf16-graph LAW-06 regime-parity + single amp authority (R30b).

WPSC Phase 3 SC-B3 REWRITE (DESIGN_P3.md §4.3/§4.4; ported from the staged
`tests/model/test_amp_dtype_p3.py` oracle, ORACLE_NOTES_P3.md row 7): new 2-arg
`amp_dtype_for(representation, declared_amp_dtype)` signature — the OLD 1-arg-dict
`amp_dtype_for(representation, config_dict)` shape is retired. The unconditional
graph->bf16 code pin is the LAW-06 protection (F-11: fp16 GINE sum-aggregation overflows
65504 -> NaN); no declared config value may flip graph off bf16.

RED-TEAM amp probe: `test_graph_is_bf16_unconditionally` feeds the EXACT value
`configs/run5.yaml`/`configs/smoke_gnn.yaml` mint today on a graph run
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
