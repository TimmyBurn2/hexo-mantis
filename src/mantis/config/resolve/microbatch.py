"""`resolve_microbatch_caps` — THE one read path for `train.microbatch_caps`
(WP12-R dispatch 6 phase F2, CARD-RUN5-GPU-OOM, R179).

`train.microbatch_caps` is read HERE and nowhere else. `StepCoordinator._microbatch_caps`
memoises this call and hands the dispatcher the BOUND METHOD; `run_declared_train_step` passes
that callable to `_graph_step` alone and the graph arm invokes it once. The grid arm is not
given the provider at all, so a grid run structurally cannot reach this function — which is
why a grid `full_config` that carries no `train` section stays loadable, and why the four
FROZEN grid coordinators that construct exactly such a config keep working.

WHY THE PROVIDER RATHER THAN THE VALUE. Python evaluates every argument before the call, so
resolving at the call site would read `full_config["train"]` on BOTH representations — the
defect this indirection exists to prevent (DESIGN_DFIX §3.11.1). "Eager for the router, lazy
for the routed" is the rule: `spec` is resolved eagerly because it DECIDES the route; the caps
are meaningful only on one branch OF that decision.

ABSENCE IS A NAMED RAISE, NEVER A DEFAULT (LAW-11, R1). `MissingMicrobatchCapsError` names the
level that is missing — `full_config` not a mapping, no `train` section, `train` not a mapping,
no `microbatch_caps`, `microbatch_caps` not a mapping, or a member absent. This is the
`resolve_from_config` / `MissingEncodingError` posture the same call already relies on for the
encoding (`train/coordinator/dispatch.py:45-52`).

THERE IS NO `.get(...)` ON THIS PATH, and that refusal is ruled rather than stylistic
(F2-ABORT-5(i)). A defaulting read on the input to a memory-safety cap is the silent-fallback
class: **a cap that silently becomes absent-and-unbounded is worse than no cap, because it
reports as present** — the phantom-gate shape R4/LAW-07 exist to kill.

RUN-SCOPED CONSTANTS (R85/R179): both members are sized together from ONE measured cost model
at mint prereg and are never hand-edited in a minted file.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_KEY = "train.microbatch_caps"


class MissingMicrobatchCapsError(ValueError):
    """The graph training step's memory caps are not declared, at some named level.

    A `ValueError` for the same reason `MissingEncodingError` is one: an absent identity-class
    key is a configuration ERROR, not a condition to recover from. Raised only on the GRAPH
    route (the grid arm never invokes the provider), and never caught by `_graph_step`.
    """


@dataclass(frozen=True)
class MicrobatchCapsSpec:
    """The resolved per-micro-batch bound: `max_edges` and `max_nodes`, together.

    A frozen dataclass beside the resolver rather than the pydantic block itself, for
    `DrawRateAbortSpec`'s reason: `train/coordinator/` is the DAG-clean seam layer and nothing
    in `mantis.train` should have to import a schema class to consume a resolved value.

    BOTH MEMBERS, because the cost model the sizing pass fitted is `peak ~ a + b*E + c*N` and
    a bound on bytes needs both terms bounded. E dominates at the production ratio
    (E/N ~ 26.8) and does not dominate structurally: a micro-batch of many low-degree graphs
    passes an edge-only bound and can be arbitrarily large in N.
    """

    max_edges: int
    max_nodes: int


def resolve_microbatch_caps(full_config: Any) -> MicrobatchCapsSpec:
    """Return the declared graph micro-batch caps. Absence raises, naming the level."""
    if not isinstance(full_config, Mapping):
        raise MissingMicrobatchCapsError(
            f"{_KEY}: the config is not a mapping ({type(full_config).__name__}), so no "
            "`train` section can be read — the graph training step has no memory bound and "
            "an unbounded graph step is the defect this block exists to make unconstructible"
        )
    if "train" not in full_config:
        raise MissingMicrobatchCapsError(
            f"{_KEY}: the config has no `train` section. Absent is an ERROR, never a default "
            "(LAW-11): a cap that silently became absent-and-unbounded would still report as "
            "present. Mint the block."
        )
    train_section = full_config["train"]
    if not isinstance(train_section, Mapping):
        raise MissingMicrobatchCapsError(
            f"{_KEY}: the `train` section is not a mapping "
            f"({type(train_section).__name__}); `microbatch_caps` cannot be read from it"
        )
    if "microbatch_caps" not in train_section:
        raise MissingMicrobatchCapsError(
            f"{_KEY}: `train` carries no `microbatch_caps` block. The block is REQUIRED by "
            "the schema, so a config that reaches here without it was not built through the "
            "one loader — there is no code-side default to fall back to (R1)."
        )
    block = train_section["microbatch_caps"]
    if not isinstance(block, Mapping):
        raise MissingMicrobatchCapsError(
            f"{_KEY}: the block is not a mapping ({type(block).__name__}); both members must "
            "arrive together or not at all"
        )
    for member in ("max_edges", "max_nodes"):
        if member not in block:
            raise MissingMicrobatchCapsError(
                f"{_KEY}.{member} is absent. The two members are sized TOGETHER from one fit "
                "against one budget and arrive together — one member alone bounds neither "
                "term of `peak ~ a + b*E + c*N`."
            )
    return MicrobatchCapsSpec(max_edges=int(block["max_edges"]),
                              max_nodes=int(block["max_nodes"]))


__all__ = ["MicrobatchCapsSpec", "MissingMicrobatchCapsError", "resolve_microbatch_caps"]
