"""THE one read path for the graph collector's two batching knobs.

`LocalInferenceEngine` is the ONE graph-server construction site with no `RunConfig` to resolve
against, and its dict literal carried both knobs as HARDCODED numbers — a second authority
over a geometry the config already owns. Measured cost of the wrong values for a route: supply
8 against a collector threshold of 32 spent 1.76 of the eval path's 5.30 ms/sim inside the
collector's own deadline.

ABSENCE IS A NAMED RAISE, NEVER A DEFAULT, and the levels are named separately because a
missing `inference` section and a missing member are two different edits. There is no
`.get(...)`, no `or`-default and no `except` on this path.
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

    FROZEN because it crosses the eval process seam on `RoundSpec`, where a rebind in the child
    is invisible to the parent. BOTH MEMBERS, because they are one geometry: the width sets the
    saturation threshold and the deadline what a pop pays when it is not reached.
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
