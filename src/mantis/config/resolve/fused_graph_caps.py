"""`resolve_fused_graph_caps` — THE one read path for `inference.fused_graph_caps`
(F-816-10, R276(f), CARD-RUN5-GPU-OOM's inference-side sibling).

`inference.fused_graph_caps` is read HERE and nowhere else. `InferenceServer.__init__` calls
this ONCE, EAGERLY, inside the GRAPH branch and stores the frozen spec; `_run_graph_loop`
hands that spec to `plan_fused_forwards` and never reads a config again. The grid branch never
invokes this function at all — the dense batch is a fixed-shape tensor already bounded by
`inference_batch_size`, so there is no unbounded quantity there for a cap to bound.

WHY EAGER, WHERE `resolve_microbatch_caps` IS LAZY. The microbatch resolver is handed to the
dispatcher as a PROVIDER because Python evaluates every argument before the call, so resolving
at the call site would read `train` on BOTH representations and a grid `full_config` may carry
no `train` section at all. Here `__init__` ALREADY branches on the representation, so the
resolution is naturally route-scoped and there is nothing to buy by deferring it — while
failing a mis-minted run in the first second, instead of three hours in, is the whole value of
the placeholder posture below.

ABSENCE IS A NAMED RAISE, NEVER A DEFAULT (LAW-11, R1). `MissingFusedGraphCapsError` names the
LEVEL that is missing — `full_config` not a mapping, no `inference` section, `inference` not a
mapping, no `fused_graph_caps`, the block not a mapping, or a member absent. Seven levels are
seven different edits, so one "the caps are absent" message would be a refusal an operator
cannot act on.

`null` IS NOT AN OFF STATE, and `UncalibratedFusedGraphCapsError` is the subclass that says so.
It is R119's placeholder: schema-VALID (gate 7 stays green and the repo ships a complete
config) and runtime-REFUSED (a graph run on an uncalibrated production config cannot construct
its inference server). "You never minted this" and "your config is malformed" send an operator
to two different places, so they are two exception types — related by subclassing, because an
uncalibrated cap IS a special case of an unusable one and a caller handling the general absence
must not miss the placeholder.

THERE IS NO `.get(...)`, NO `or`-DEFAULT AND NO `except` ON THIS PATH, and that refusal is
ruled rather than stylistic (F2-ABORT-5(i), transferred verbatim from `resolve/microbatch.py`).
A defaulting read on the input to a memory-safety cap is the silent-fallback class: **a cap
that silently becomes absent-and-unbounded is worse than no cap, because it reports as
present** — the phantom-gate shape R4/LAW-07 exist to kill. An `ast` census over this module
enforces all three.

RUN-SCOPED CONSTANTS (R85/R119): both members are sized together from ONE measured fit against
ONE budget at the box sitting, and are never hand-edited in a minted file.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_KEY = "inference.fused_graph_caps"
_MEMBERS = ("max_fused_edges", "max_fused_nodes")
#: The entry point that PRODUCES the value, named in the refusal so the operator is not left
#: to guess where a measured cap comes from (R69: a number without its producing mechanism is
#: struck, and this is the mechanism).
_CALIBRATE = "uv run python -m mantis.diagnostics.fusion_calibrate"


class MissingFusedGraphCapsError(ValueError):
    """The graph inference forward's memory caps are not declared, at some named level.

    A `ValueError` for the same reason `MissingMicrobatchCapsError` is one: an absent
    memory-bound key is a configuration ERROR, not a condition to recover from. Raised only on
    the GRAPH route, and never caught anywhere — it travels the R276 seam as a run-fatal
    construction failure.
    """


class UncalibratedFusedGraphCapsError(MissingFusedGraphCapsError):
    """A member is the `null` placeholder: the cap exists as a key and has no measured value.

    A SUBCLASS, deliberately. An uncalibrated cap is a special case of an unusable one, so a
    caller that handles the general absence must not miss the placeholder; and it is a
    DISTINCT type, because "you have not calibrated this yet" carries a remedy the general
    absence does not — the calibration entry point and the mint line that fixes it.
    """


@dataclass(frozen=True)
class FusedGraphCapsSpec:
    """The resolved per-fused-forward bound: `max_fused_edges` and `max_fused_nodes`, together.

    FROZEN because a resolved run-scoped constant a consumer could rebind is a second authority
    with extra steps — and this one crosses a process seam (`RoundSpec`), where a rebind in the
    child would be invisible to the parent that measured the budget.

    A frozen dataclass beside the resolver rather than the pydantic block itself, for
    `MicrobatchCapsSpec`'s reason: nothing in `mantis.selfplay` or `mantis.eval` should have to
    import a schema class to consume a resolved value.

    BOTH MEMBERS, because the cost model the calibration fits is `peak ~ a + b*E + c*N` and a
    bound on bytes needs both terms bounded. E dominates at the production ratio and does not
    dominate structurally: the builder's two dummy edges per real node force `E >= 2(N-1)`, so
    an edge-only bound admits an N term LARGER than the E term it bounds.
    """

    max_fused_edges: int
    max_fused_nodes: int


def resolve_fused_graph_caps(full_config: Any) -> FusedGraphCapsSpec:
    """Return the declared fused-graph-inference caps. Absence raises, naming the level."""
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
