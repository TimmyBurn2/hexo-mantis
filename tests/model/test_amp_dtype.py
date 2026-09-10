"""O6 — bf16-graph LAW-06 regime-parity + single amp authority (R30b).

The signature has narrowed twice. WPSC Phase 3 SC-B3 replaced the 1-arg-dict
`amp_dtype_for(representation, config_dict)` with an explicit
`amp_dtype_for(representation, declared_amp_dtype)`; R346(f) then deleted `train.amp_dtype`
outright, so there is no declared value left and the signature is
`amp_dtype_for(representation)`.

The unconditional graph->bf16 code pin is the LAW-06 protection (F-11: fp16 GINE
sum-aggregation overflows 65504 -> NaN). It used to have to survive a declared value that
disagreed; now there is no second spelling of the question, and what is pinned is that the
answer comes off the representation alone and that an unknown representation is an ERROR.
"""
from __future__ import annotations

import pytest
import torch

from mantis.model.amp import amp_dtype_for


def test_graph_is_bf16_unconditionally() -> None:
    assert amp_dtype_for("graph") is torch.bfloat16


def test_an_unknown_representation_raises() -> None:
    with pytest.raises(ValueError):
        amp_dtype_for("hexcanvas")
