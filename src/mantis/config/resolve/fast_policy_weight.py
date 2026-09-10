"""THE one read path for `train.fast_policy_weight`.

The declared POLICY weight a fast-arm row (`is_full_search == 0`) carries; a minted `0.0`
reproduces the binary gate it replaced.

A resolver rather than a `TrainHParams` field: it is consumed on the GRAPH route only, through
a zero-arg PROVIDER. Python evaluates every argument before the call, so resolving at the
dispatcher's call site would read `full_config["train"]` on BOTH representations, and the grid
coordinators construct a `full_config` with no `train` section at all.

ABSENCE IS A NAMED RAISE, NEVER A DEFAULT: a `.get(..., 0.0)` here would silently disarm an
armed ablation and report as present.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SECTION, _FIELD = "train", "fast_policy_weight"
_KEY = f"{_SECTION}.{_FIELD}"


class MissingFastPolicyWeightError(ValueError):
    """`train.fast_policy_weight` is absent. Names the level that is missing."""


def resolve_fast_policy_weight(full_config: Any) -> float:
    """The POLICY weight a fast-arm graph row carries.

    Args:
        full_config: the whole validated config mapping (`RunConfig.model_dump()`).

    Returns:
        The minted `train.fast_policy_weight`, as a float.

    Raises:
        MissingFastPolicyWeightError: `full_config` is not a mapping, has no `train`
            section, that section is not a mapping, or it carries no `fast_policy_weight`.
    """
    if not isinstance(full_config, Mapping):
        raise MissingFastPolicyWeightError(
            f"{_KEY}: the config is not a mapping ({type(full_config).__name__}), so the "
            "weight cannot be read; the schema is the sole default authority and there is "
            "no code-side fallback (R1)"
        )
    section = full_config.get(_SECTION)
    if not isinstance(section, Mapping):
        raise MissingFastPolicyWeightError(
            f"{_KEY}: the config has no `{_SECTION}` mapping, so the weight cannot be read"
        )
    if _FIELD not in section:
        raise MissingFastPolicyWeightError(
            f"{_KEY}: the `{_SECTION}` section carries no `{_FIELD}`. A default here would "
            "silently disarm the fast arm's policy ablation while reporting as present"
        )
    return float(section[_FIELD])
