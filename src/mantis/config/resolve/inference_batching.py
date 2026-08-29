"""`resolve_inference_batching` — THE one read path for the graph collector's two batching
knobs, `inference.inference_batch_size` and `inference.inference_max_wait_ms`.

WHY THIS MODULE EXISTS (PERF-TRANCHE-1 G-2, ledger F-2). `LocalInferenceEngine` hand-builds
its `InferenceServer` from a dict literal because it is the ONE graph-server construction
site with no `RunConfig` to resolve against — and that literal carried
`inference_batch_size: 64` and `inference_max_wait_ms: 10` as HARDCODED numbers. The literal
already states the argument against exactly this, for the cap it does thread: *"a cap
written here would be a SECOND authority over one byte budget, on the one construction path
with no config to be the first."* The identical argument covers these two, and the ledger
measured what they cost when they are wrong for the route: at the single-stream deploy head,
supply 8 against a collector threshold of 32 put **1.76 of the eval path's 5.30 ms/sim —
33 %** into the collector's own deadline, set by a code literal, on the one path LAW-15 reads
a promotion bar off.

ABSENCE IS A NAMED RAISE, NEVER A DEFAULT (LAW-11, R1), and the levels are named separately
for `resolve_fused_graph_caps`' reason: a missing `inference` section and a missing member
are two different edits, so one message would be a refusal an operator cannot act on.

There is no `.get(...)`, no `or`-default and no `except` on this path — a defaulting read
here is precisely the defect the module closes.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_KEY = "inference"
_MEMBERS = ("inference_batch_size", "inference_max_wait_ms")


class MissingInferenceBatchingError(ValueError):
    """A batching knob is absent from the config. Names the missing LEVEL."""


@dataclass(frozen=True)
class InferenceBatchingSpec:
    """The resolved graph-collector batching geometry: pop width and pop deadline.

    FROZEN for `FusedGraphCapsSpec`'s reason — a resolved run-scoped constant a consumer
    could rebind is a second authority with extra steps, and this one crosses the eval
    process seam on `RoundSpec`, where a rebind in the child is invisible to the parent.

    BOTH MEMBERS, because they are one geometry: the width sets the collector's saturation
    threshold and the deadline sets what a pop pays when that threshold is not reached. A
    caller given one and left to invent the other is the hardcode this module removes.
    """

    inference_batch_size: int
    inference_max_wait_ms: int


def resolve_inference_batching(full_config: Any) -> InferenceBatchingSpec:
    """Return the declared graph-collector batching geometry.

    Args:
        full_config: the whole validated config mapping (`RunConfig.model_dump()`).

    Returns:
        The frozen `InferenceBatchingSpec`.

    Raises:
        MissingInferenceBatchingError: the config is not a mapping, carries no `inference`
            section, that section is not a mapping, or either member is absent.
    """
    if not isinstance(full_config, Mapping):
        raise MissingInferenceBatchingError(
            f"{_KEY}: the config is not a mapping ({type(full_config).__name__}), so no "
            "`inference` section can be read — the collector's batching geometry would then "
            "come from a code literal, which is the defect this resolver closes"
        )
    if _KEY not in full_config:
        raise MissingInferenceBatchingError(
            f"{_KEY}: the config has no `inference` section. Absent is an ERROR, never a "
            "default (LAW-11): a batching knob that silently fell back to a literal would "
            "still report as configured."
        )
    section = full_config[_KEY]
    if not isinstance(section, Mapping):
        raise MissingInferenceBatchingError(
            f"{_KEY}: the `inference` section is not a mapping "
            f"({type(section).__name__}); the batching knobs cannot be read from it"
        )
    for member in _MEMBERS:
        if member not in section:
            raise MissingInferenceBatchingError(
                f"{_KEY}.{member} is absent. The member is REQUIRED by the schema, so a "
                "config that reaches here without it was not built through the one loader — "
                "there is no code-side default to fall back to (R1). A caller with no "
                "`RunConfig` at all threads the resolved spec instead of inventing one here."
            )
    return InferenceBatchingSpec(
        inference_batch_size=int(section[_MEMBERS[0]]),
        inference_max_wait_ms=int(section[_MEMBERS[1]]),
    )


__all__ = [
    "InferenceBatchingSpec",
    "MissingInferenceBatchingError",
    "resolve_inference_batching",
]
