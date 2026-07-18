"""Legal-move radius curriculum resolver (PORT of frozen resolve/radius.py).

The schedule-scan rule (last entry whose ``step`` <= query step wins) + the offline HARD-ERROR
gate. ``resolve_eval_radius`` (the eval_board delegation) is NOT ported — it relocates to
WP-eval (config → eval is a DAG violation).

Schedule entries are Mappings ({"step", "radius"}); a loaded config's list[RadiusStage] is
passed as ``[stage.model_dump() for stage in schedule]``.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class OfflineRadiusUnresolvableError(ValueError):
    """An offline instrument could not resolve a radius: no baked schedule AND no --radius-stage.

    Pre-CONFRES this fell through to the registry default silently — a per-stage book read at the
    wrong radius biases the measurement. HARD-ERROR instead.
    """


def resolve_radius_from_schedule(
    schedule: Sequence[Mapping[str, Any]] | None,
    step: int,
) -> int | None:
    """Curriculum-current radius at ``step``. None when no schedule (caller keeps registry radius).

    Entries ordered by ``step``; the last entry whose ``step`` is <= the query step wins.
    """
    if not schedule:
        return None
    current: int | None = None
    for entry in schedule:
        if step >= entry["step"]:
            current = entry["radius"]
    return current


def require_offline_radius(
    resolved: int | None,
    radius_stage_override: int | None,
    *,
    ckpt_label: str = "<checkpoint>",
) -> int:
    """Offline HARD-ERROR gate: return a concrete radius or raise.

    Precedence: an explicit ``--radius-stage`` override wins; else the schedule-resolved radius;
    if BOTH are None → OfflineRadiusUnresolvableError naming the checkpoint + the fix.
    """
    if radius_stage_override is not None:
        return int(radius_stage_override)
    if resolved is not None:
        return int(resolved)
    raise OfflineRadiusUnresolvableError(
        f"cannot resolve legal_move_radius for {ckpt_label}: no baked "
        "legal_move_radius_schedule and no --radius-stage supplied. A per-stage book read at "
        "the wrong radius biases the measurement; refusing to silently fall back to the registry "
        "default. Fix: preserve the curriculum stage in the checkpoint, or pass --radius-stage <int>."
    )
