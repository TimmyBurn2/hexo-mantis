"""`resolve_decided_tails` — THE read path for `selfplay.decided_tails` (v63): the armed block's value weight, or `None` for the explicit OFF."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SECTION, _FIELD = "selfplay", "decided_tails"
_KEY = f"{_SECTION}.{_FIELD}"


class DecidedTailsShapeError(ValueError):
    """`selfplay.decided_tails` is absent, or neither `null` nor exactly `{value_weight: w}` (the v60 bool included)."""


def resolve_decided_tails(full_config: Any) -> float | None:
    """The armed tails' value weight, or `None` when the config states `null`; Raises: DecidedTailsShapeError — no `selfplay` section, the key absent, or any other shape."""
    if not isinstance(full_config, Mapping) or not isinstance(full_config.get(_SECTION), Mapping):
        raise DecidedTailsShapeError(f"{_KEY}: the config carries no `{_SECTION}` section (R1/LAW-11)")
    section = full_config[_SECTION]
    if _FIELD not in section:
        raise DecidedTailsShapeError(f"{_KEY} is absent; it is REQUIRED (`null` is the explicit OFF)")
    block = section[_FIELD]
    if block is None:
        return None
    if not isinstance(block, Mapping) or set(block) != {"value_weight"}:
        raise DecidedTailsShapeError(f"{_KEY} must be null or exactly {{value_weight: w}}; got {block!r} (a v60 bool "
                                     "is read only off a stamp, through mantis.config.retired)")
    weight = block["value_weight"]
    if isinstance(weight, bool) or not isinstance(weight, int | float):
        raise DecidedTailsShapeError(f"{_KEY}.value_weight must be a number; got {weight!r}")
    return float(weight)


__all__ = ["DecidedTailsShapeError", "resolve_decided_tails"]
