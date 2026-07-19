"""Representation-aware autocast dtype resolver — the LAW-06 bf16-graph pin.

`amp_dtype_for` is the ONE resolver both the graph training step and the
graph inference seam consult. The graph branch returns bf16 UNCONDITIONALLY
(not config-tunable): `_GINEConv`'s sum-aggregation accumulates one ReLU'd
message per incoming edge into each destination node, and on production-scale
late-game graphs that sum tips past fp16's 65504 ceiling → inf → LayerNorm →
NaN (LAW-06 / F-11). bf16 keeps fp32's 8-bit exponent at 2-byte width, so the
overflow class cannot recur — and pinning it in CODE means a dropped or stale
config override can never flip graph back to fp16.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def amp_dtype_for(
    representation: str, config: Mapping[str, Any] | None = None
) -> torch.dtype:
    """Autocast dtype for a representation kind.

    GRAPH: `torch.bfloat16`, unconditionally — the LAW-06 pin (not config-tunable).
    GRID:  delegates to the `amp_dtype` config knob (default `"fp16"`); `"fp16"` →
           float16, `"bf16"` → bfloat16, anything else → ValueError.
    """
    if representation == "graph":
        return torch.bfloat16
    raw = str((config or {}).get("amp_dtype", "fp16")).lower()
    if raw in ("fp16", "float16", "half"):
        return torch.float16
    if raw in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(
        f"amp_dtype must be 'fp16' or 'bf16', got {raw!r}."
    )
