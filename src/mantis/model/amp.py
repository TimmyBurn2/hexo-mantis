"""Representation-aware autocast dtype resolver — the LAW-06 bf16-graph pin.

R30b: `resolve_amp_dtype` (mantis.config.resolve.amp) is now THE single decision authority
(string-level, torch-free); `amp_dtype_for` is a thin wrapper that maps its string token to a
real `torch.dtype`. No raw dict read, no duplicated "fp16"/"bf16" literal set, no default —
`train.amp_dtype` and `train.fp16` are DELETED (R346(f)), so the pin is the only authority.

The graph branch returns bf16 UNCONDITIONALLY (not config-tunable): `_GINEConv`'s sum-
aggregation accumulates one ReLU'd message per incoming edge into each destination node, and
on production-scale late-game graphs that sum tips past fp16's 65504 ceiling -> inf ->
LayerNorm -> NaN (LAW-06 / F-11). bf16 keeps fp32's 8-bit exponent at 2-byte width, so the
overflow class cannot recur — and pinning it in CODE means a dropped or stale config override
can never flip graph back to fp16.
"""
from __future__ import annotations

import torch

from mantis.config.resolve.amp import resolve_amp_dtype

_STRING_TO_TORCH: dict[str, torch.dtype] = {"fp16": torch.float16, "bf16": torch.bfloat16}


def amp_dtype_for(representation: str) -> torch.dtype:
    """Autocast dtype for a representation kind (R30b: ONE authority).

    Args:
        representation: the identity representation the run declares.

    Returns:
        `torch.bfloat16`, the LAW-06 pin.

    Raises:
        ValueError: from `resolve_amp_dtype`, on an unknown representation.
    """
    return _STRING_TO_TORCH[resolve_amp_dtype(representation)]
