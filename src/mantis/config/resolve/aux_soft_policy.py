"""`resolve_aux_soft_policy` — THE read path for `model.aux_soft_policy` (v36): temperature and weight, or `None` for the explicit OFF; an absent key is an error, never OFF."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SECTION, _FIELD = "model", "aux_soft_policy"
_KEY = f"{_SECTION}.{_FIELD}"
_MEMBERS: tuple[str, ...] = ("target_temperature", "weight")


class MissingAuxSoftPolicyError(ValueError):
    """`model.aux_soft_policy` is absent (not `null`) at some named level, or a member is short."""


@dataclass(frozen=True)
class AuxSoftPolicySpec:
    """The armed rows: the temperature on the explicit entries and the CE's weight in the step's loss."""

    temperature: float
    weight: float


def resolve_aux_soft_policy(full_config: Any) -> AuxSoftPolicySpec | None:
    """Return the armed rows, or `None` when the config states `null`; Raises: MissingAuxSoftPolicyError — no `model` section, the key absent, or a member short."""
    if not isinstance(full_config, Mapping) or not isinstance(full_config.get(_SECTION), Mapping):
        raise MissingAuxSoftPolicyError(
            f"{_KEY}: the config carries no `{_SECTION}` section, so the head's posture cannot be "
            "read — a code-side OFF would report as configured (R1/LAW-11)"
        )
    section = full_config[_SECTION]
    if _FIELD not in section:
        raise MissingAuxSoftPolicyError(
            f"{_KEY} is absent. The key is REQUIRED (`null` is the explicit OFF), so a config "
            "reaching here without it was not built through the one loader"
        )
    block = section[_FIELD]
    if block is None:
        return None
    if not isinstance(block, Mapping) or any(m not in block for m in _MEMBERS):
        raise MissingAuxSoftPolicyError(f"{_KEY} must be null or carry {list(_MEMBERS)}; got {block!r}")
    return AuxSoftPolicySpec(temperature=float(block["target_temperature"]), weight=float(block["weight"]))


__all__ = ["AuxSoftPolicySpec", "MissingAuxSoftPolicyError", "resolve_aux_soft_policy"]
