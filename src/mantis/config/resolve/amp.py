"""AMP autocast dtype resolver (R30b: THE single amp authority).

Returns a STRING dtype token ("bf16" / "fp16"), NEVER a torch.dtype — mantis.config imports
only encoding + util (DAG), so it must not pull torch. `mantis.model.amp.amp_dtype_for` is a
thin torch-mapping wrapper over THIS function — this is the one decision site. graph -> "bf16"
is pinned in code (LAW-06): fp16 GINE sum-aggregation overflowed to NaN on production-scale
self-play graphs (F-11); it is not config-tunable, so `declared_amp_dtype` is ignored on the
graph branch.
"""
from __future__ import annotations


def resolve_amp_dtype(representation: str, declared_amp_dtype: str | None = None) -> str:
    """Resolve the autocast dtype token for a representation.

    "graph" -> "bf16" unconditionally (pinned constant, LAW-06); ``declared_amp_dtype`` is
    structurally ignored on this branch. "grid" -> ``declared_amp_dtype``, which must be
    "fp16" or "bf16" (already ``Literal["fp16","bf16"]``-bounded by the schema; re-validated
    here since this function has non-schema callers too) — no silent default.

    ``declared_amp_dtype``'s default of ``None`` exists ONLY for one frozen Phase-2 oracle
    call shape (`tests/config/test_regime_parity_p2.py::test_o10_amp_is_bf16_on_graph_unchanged`,
    a 1-positional-arg call, always representation="graph" — sha256-pinned in
    ORACLE_NOTES_P2.md, must not be edited). The default is INERT on the graph branch
    (structurally ignored) and loud on the grid branch: omitting it there raises, same as
    any other invalid value — R1 is unaffected for every real (2-arg) caller.
    """
    if representation == "graph":
        return "bf16"
    if representation == "grid":
        if declared_amp_dtype not in ("fp16", "bf16"):
            raise ValueError(
                f"amp_dtype must be 'fp16' or 'bf16', got {declared_amp_dtype!r}."
            )
        return declared_amp_dtype
    raise ValueError(
        f"unknown representation {representation!r}; expected 'grid' or 'graph'"
    )
