"""AMP autocast dtype resolver: THE single amp authority.

Returns a STRING dtype token, NEVER a torch.dtype — mantis.config imports only encoding +
util (DAG), so it must not pull torch. `mantis.model.amp.amp_dtype_for` is a thin
torch-mapping wrapper over THIS function — this is the one decision site. bf16 is pinned in
code: fp16 GINE sum-aggregation overflowed to NaN on production-scale self-play graphs.
It is not config-tunable, and no second spelling of the question exists anywhere.
"""
from __future__ import annotations

#: The one autocast dtype token this project uses.
AMP_DTYPE: str = "bf16"


def resolve_amp_dtype(representation: str) -> str:
    """Resolve the autocast dtype token for a representation.

    Args:
        representation: the identity representation the run declares.

    Returns:
        The pinned dtype token.

    Raises:
        ValueError: the representation is not one this project knows — an unknown
            representation must never resolve a dtype by falling through.
    """
    if representation == "graph":
        return AMP_DTYPE
    raise ValueError(
        f"unknown representation {representation!r}; expected 'graph'"
    )
