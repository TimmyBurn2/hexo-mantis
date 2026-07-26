"""`resolve_actor_sync_cadence` — K1's ONE read path (WP-UNFREEZE, DESIGN_U §5/§9).

`train.actor_sync_cadence_steps` is read HERE and nowhere else; the composition root
(`mantis.run.compose_run`) threads the resolved value into `ActorSync.maybe_sync`.
The schema bound (`Field(ge=1)`) is the sole authority: no representable "off" value
exists (R49 at the type level), so this resolver carries no disable sentinel and no
code-side default (R1) — a missing key never reaches here (pydantic rejects it at
load, naming the key).
"""
from __future__ import annotations

from typing import Any


def resolve_actor_sync_cadence(train_section: Any) -> int:
    """Return the validated actor-sync cadence in coordinator training steps."""
    return int(train_section.actor_sync_cadence_steps)


__all__ = ["resolve_actor_sync_cadence"]
