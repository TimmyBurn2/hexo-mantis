"""`resolve_fast_policy_weight` — THE one read path for `train.fast_policy_weight` (R347(b)).

WHAT THE KEY IS. Fast-arm rows (`is_full_search == 0`) have always trained VALUE; their
POLICY was gated off entirely, by a binary mask that was neither declared nor readable in a
config. This key is the declared WEIGHT that replaces the gate, and its minted `0.0`
reproduces the gate exactly.

WHY A RESOLVER AND NOT A `TrainHParams` FIELD. It is consumed on the GRAPH training route
only, by `train/coordinator/dispatch.py::_build_graph_parts`, which is handed a zero-arg
PROVIDER rather than a value — the same shape and the same grounds as
`resolve_microbatch_caps` and `resolve_sample_threads`. Python evaluates every argument
before the call, so resolving at the dispatcher's call site would read `full_config["train"]`
on BOTH representations, and the four frozen grid coordinators construct a `full_config` with
no `train` section at all. A `TrainHParams` field would ALSO work for the training step and
would then be a second authority over the same leaf for the pretrain and held-out routes,
which reach the dispatcher directly.

ABSENCE IS A NAMED RAISE, NEVER A DEFAULT (LAW-11, R1). The schema is the sole default
authority; a `.get(..., 0.0)` here would silently disarm an armed ablation and report as
present, which is the phantom-input class R4/LAW-07 exist to kill.
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
