"""O6 — bf16-graph regime-parity + single amp authority.

The signature has narrowed twice. Phase 3 SC-B3 replaced the 1-arg-dict
`amp_dtype_for(representation, config_dict)` with an explicit
`amp_dtype_for(representation, declared_amp_dtype)`; `train.amp_dtype` was then deleted
outright, so there is no declared value left and the signature is
`amp_dtype_for(representation)`.

The unconditional graph->bf16 code pin is the bf16 protection (fp16 GINE
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
