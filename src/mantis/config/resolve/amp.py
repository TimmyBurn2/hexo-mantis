"""AMP autocast dtype resolver (NEW; LAW-06 / F-11).

Returns a STRING dtype token ("bf16" / "fp16"), NEVER a torch.dtype — mantis.config imports
only encoding + util (DAG), so it must not pull torch. The model (WP9) maps the token → a real
torch.dtype. graph → "bf16" is pinned in code (LAW-06): fp16 GINE sum-aggregation overflowed to
NaN on production-scale self-play graphs (F-11); it is not config-tunable.
"""
from __future__ import annotations


def resolve_amp_dtype(representation: str) -> str:
    """Resolve the autocast dtype token for a representation.

    "graph" → "bf16" (pinned constant, LAW-06); "grid" → "fp16" (historical grid token,
    resolution-rule constant). Unknown representation → ValueError (no silent fallback).
    """
    if representation == "graph":
        return "bf16"
    if representation == "grid":
        return "fp16"
    raise ValueError(
        f"unknown representation {representation!r}; expected 'grid' or 'graph'"
    )
