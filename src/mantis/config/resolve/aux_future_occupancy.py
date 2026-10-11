"""`resolve_aux_future_occupancy` — THE read path for `model.aux_future_occupancy` (v59): the weight, or `None` for the explicit OFF."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_SECTION, _FIELD = "model", "aux_future_occupancy"
_KEY = f"{_SECTION}.{_FIELD}"


class MissingAuxFutureOccupancyError(ValueError):
    """`model.aux_future_occupancy` is absent (not `null`), or carries no `w`."""


def resolve_aux_future_occupancy(full_config: Any) -> float | None:
    """The head's weight, or `None` when the config states `null`; Raises: MissingAuxFutureOccupancyError — no `model` section, the key absent, or no `w`."""
    if not isinstance(full_config, Mapping) or not isinstance(full_config.get(_SECTION), Mapping):
        raise MissingAuxFutureOccupancyError(f"{_KEY}: the config carries no `{_SECTION}` section (R1/LAW-11)")
    section = full_config[_SECTION]
    if _FIELD not in section:
        raise MissingAuxFutureOccupancyError(f"{_KEY} is absent; it is REQUIRED (`null` is the explicit OFF)")
    block = section[_FIELD]
    if block is None:
        return None
    if not isinstance(block, Mapping) or "w" not in block:
        raise MissingAuxFutureOccupancyError(f"{_KEY} must be null or carry `w`; got {block!r}")
    return float(block["w"])


__all__ = ["MissingAuxFutureOccupancyError", "resolve_aux_future_occupancy"]
