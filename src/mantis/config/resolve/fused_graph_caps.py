"""`resolve_fused_graph_caps` — THE one read path for `inference.fused_graph_caps`.

Read HERE and nowhere else: `InferenceServer.__init__` calls it ONCE, EAGERLY, inside the GRAPH
branch and stores the frozen spec, while the grid branch never calls it because a dense batch is
already bounded by `inference_batch_size`. Eager, where `resolve_microbatch_caps` is lazy,
because `__init__` already branches on the representation and a mis-minted run then fails in the
first second.

ABSENCE IS A NAMED RAISE, NEVER A DEFAULT, and the raise names the LEVEL that is missing, since
seven levels are seven different edits. `null` is not an off state either: schema-VALID so the
repo ships a complete config, runtime-REFUSED so an uncalibrated config cannot construct its
server, and a SUBCLASS because the two refusals send an operator to two different places. There
is no `.get(...)`, no `or`-default and no `except` here, enforced by an `ast` census — a cap
that silently becomes unbounded is worse than no cap, because it reports as present.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mantis.config.resolve.arch_scope import refuse_outside_its_arch

_SECTION, _FIELD = "inference", "fused_graph_caps"
_KEY = f"{_SECTION}.{_FIELD}"
_MEMBERS = ("max_fused_edges", "max_fused_nodes")
#: The entry point that PRODUCES the value, named in the refusal so an operator is not left to
#: guess where a measured cap comes from.
_CALIBRATE = "uv run python -m mantis.diagnostics.fusion_calibrate"


class MissingFusedGraphCapsError(ValueError):
    """The graph inference forward's memory caps are not declared, at some named level.

    A configuration ERROR: raised only on the GRAPH route and never caught, so it travels the
    seam as a run-fatal construction failure.
    """


class UncalibratedFusedGraphCapsError(MissingFusedGraphCapsError):
    """A member is the `null` placeholder: the cap exists as a key with no measured value.

    A SUBCLASS, so a caller handling the general absence does not miss it; a DISTINCT type,
    because it carries a remedy the general absence does not.
    """


@dataclass(frozen=True)
class FusedGraphCapsSpec:
    """The resolved per-fused-forward bound: `max_fused_edges` and `max_fused_nodes`, together.

    FROZEN because this crosses a process seam, where a rebind in the child would be invisible
    to the parent that measured the budget. BOTH members, because the fitted cost model is
    `peak ~ a + b*E + c*N` and the builder's two dummy edges per real node force `E >= 2(N-1)`,
    so an edge-only bound admits an N term LARGER than the E term it bounds.
    """

    max_fused_edges: int
    max_fused_nodes: int


def resolve_fused_graph_caps(full_config: Any) -> FusedGraphCapsSpec:
    """Return the declared fused-graph-inference caps; the WRONG ARCH is refused first and by
    name, then absence naming the level, then the `null` placeholder.
    Raises:
        ArchScopedKeyOutsideItsArchError: the config declares a representation other than
            `graph`.
        MissingFusedGraphCapsError: the block, or one of its members, is not declared.
        UncalibratedFusedGraphCapsError: a member is the `null` placeholder.
    """
    refuse_outside_its_arch(full_config, _SECTION, _FIELD)
    if not isinstance(full_config, Mapping):
        raise MissingFusedGraphCapsError(
            f"{_KEY}: the config is not a mapping ({type(full_config).__name__}), so no "
            "`inference` section can be read — the graph inference forward then has no memory "
            "bound, and an unbounded fused forward is the defect this block exists to make "
            "unconstructible"
        )
    if "inference" not in full_config:
        raise MissingFusedGraphCapsError(
            f"{_KEY}: the config has no `inference` section. Absent is an ERROR, never a "
            "default (LAW-11): a cap that silently became absent-and-unbounded would still "
            "report as present. Mint the block."
        )
    inference_section = full_config["inference"]
    if not isinstance(inference_section, Mapping):
        raise MissingFusedGraphCapsError(
            f"{_KEY}: the `inference` section is not a mapping "
            f"({type(inference_section).__name__}); `fused_graph_caps` cannot be read from it"
        )
    if "fused_graph_caps" not in inference_section:
        raise MissingFusedGraphCapsError(
            f"{_KEY}: `inference` carries no `fused_graph_caps` block. The block is REQUIRED "
            "by the schema, so a config that reaches here without it was not built through "
            "the one loader — there is no code-side default to fall back to (R1). A caller "
            "with no `RunConfig` at all threads the resolved spec instead of inventing one "
            "here (D-1)."
        )
    block = inference_section["fused_graph_caps"]
    if not isinstance(block, Mapping):
        raise MissingFusedGraphCapsError(
            f"{_KEY}: the `fused_graph_caps` block is not a mapping "
            f"({type(block).__name__}); both members must arrive together or not at all"
        )
    for member in _MEMBERS:
        if member not in block:
            raise MissingFusedGraphCapsError(
                f"{_KEY}.{member} is absent. The two members are sized TOGETHER from one fit "
                "against one budget and arrive together — one member alone bounds neither "
                "term of `peak ~ a + b*E + c*N`."
            )
    for member in _MEMBERS:
        if block[member] is None:
            raise UncalibratedFusedGraphCapsError(
                f"{_KEY}.{member} is null — the R119 PLACEHOLDER, not an off state. `null` is "
                "schema-valid so the repo ships a complete config, and refused here so an "
                "uncalibrated production config cannot construct its graph inference server. "
                f"Measure the value on the box:\n"
                f"    {_CALIBRATE} --config <this config> --budget-bytes <B>\n"
                "then mint what it reports (never a hand-picked number):\n"
                "    uv run python tools/mint_config.py --template <t> --out <this config> "
                f"--force --set {_KEY}.{_MEMBERS[0]}=<E> --set {_KEY}.{_MEMBERS[1]}=<N>\n"
                "Both members are minted in ONE act: they are sized from one fit against one "
                "budget, so a half-minted block is a state the calibration cannot produce."
            )
    return FusedGraphCapsSpec(
        max_fused_edges=int(block[_MEMBERS[0]]),
        max_fused_nodes=int(block[_MEMBERS[1]]),
    )


__all__ = [
    "FusedGraphCapsSpec",
    "MissingFusedGraphCapsError",
    "UncalibratedFusedGraphCapsError",
    "resolve_fused_graph_caps",
]
