"""`resolve_heldout_gap` — THE read path for `train.heldout_gap` (v37): the held-out witness's rows, or `None` for the explicit OFF; absence of the key is an error, never OFF."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SECTION, _FIELD = "train", "heldout_gap"
_KEY = f"{_SECTION}.{_FIELD}"
_MEMBERS: tuple[str, ...] = ("ring", "ring_sha256", "batches", "seed", "interval")


class MissingHeldoutGapError(ValueError):
    """`train.heldout_gap` is absent (not `null`) at some named level, or a member is short."""


@dataclass(frozen=True)
class HeldoutGapSpec:
    """The armed rows: the frozen slice's ring and its sha, how many production samples, the sampler seed, the read cadence in steps."""

    ring: str
    ring_sha256: str
    batches: int
    seed: int
    interval: int


def resolve_heldout_gap(full_config: Any) -> HeldoutGapSpec | None:
    """Return the armed rows, or `None` when the config states `null`; Raises: MissingHeldoutGapError — no `train` section, the key absent, or a member short."""
    if not isinstance(full_config, Mapping) or not isinstance(full_config.get(_SECTION), Mapping):
        raise MissingHeldoutGapError(
            f"{_KEY}: the config carries no `{_SECTION}` section, so the witness's posture cannot "
            "be read — a code-side OFF would report as configured (R1/LAW-11)"
        )
    section = full_config[_SECTION]
    if _FIELD not in section:
        raise MissingHeldoutGapError(
            f"{_KEY} is absent. The key is REQUIRED (`null` is the explicit OFF), so a config "
            "reaching here without it was not built through the one loader"
        )
    block = section[_FIELD]
    if block is None:
        return None
    if not isinstance(block, Mapping) or any(m not in block for m in _MEMBERS):
        raise MissingHeldoutGapError(f"{_KEY} must be null or carry {list(_MEMBERS)}; got {block!r}")
    return HeldoutGapSpec(ring=str(block["ring"]), ring_sha256=str(block["ring_sha256"]),
                          batches=int(block["batches"]), seed=int(block["seed"]),
                          interval=int(block["interval"]))


__all__ = ["HeldoutGapSpec", "MissingHeldoutGapError", "resolve_heldout_gap"]
