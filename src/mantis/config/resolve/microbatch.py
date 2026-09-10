"""`resolve_microbatch_caps` — THE one read path for `train.microbatch_caps`.

The coordinator memoises this call and hands the dispatcher the BOUND METHOD, which reaches the
graph arm alone, so a grid run structurally cannot reach this function and a grid config carrying
no `train` section stays loadable. A provider rather than a value because Python evaluates every
argument before the call. The arch-scope refusal comes FIRST and by name; absence is then a named
raise, never a default, and there is no `.get(...)` on this path — a cap that silently becomes
absent-and-unbounded is worse than no cap, because it reports as present.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mantis.config.resolve.arch_scope import refuse_outside_its_arch

_SECTION, _FIELD = "train", "microbatch_caps"
_KEY = f"{_SECTION}.{_FIELD}"


class MissingMicrobatchCapsError(ValueError):
    """The graph training step's memory caps are not declared, at some named level — a
    configuration ERROR rather than a condition to recover from, raised only on the GRAPH
    route and never caught."""


@dataclass(frozen=True)
class MicrobatchCapsSpec:
    """The resolved per-micro-batch bound: `max_edges` and `max_nodes`, together.

    A frozen dataclass beside the resolver, so nothing in `mantis.train` imports a schema class
    to consume a resolved value. BOTH members, because the fitted cost model is
    `peak ~ a + b*E + c*N`: E dominates at the production ratio (E/N ~ 26.8) but not
    structurally, and many low-degree graphs pass an edge-only bound at arbitrary N.
    """

    max_edges: int
    max_nodes: int


def resolve_microbatch_caps(full_config: Any) -> MicrobatchCapsSpec:
    """Return the declared graph micro-batch caps.

    A config of the WRONG ARCH is refused FIRST and by name; after that, absence raises,
    naming the level.

    Raises:
        ArchScopedKeyOutsideItsArchError: the config declares a representation other than
            `graph`. Refused ahead of every presence check on purpose: a resolver that only
            refuses ABSENCE is green by accident on a grid config, and turns red the moment
            anyone re-adds the block.
        MissingMicrobatchCapsError: the block, or one of its members, is not declared.
    """
    refuse_outside_its_arch(full_config, _SECTION, _FIELD)
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
