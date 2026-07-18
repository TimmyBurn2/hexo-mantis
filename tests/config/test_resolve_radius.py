"""O3 — radius schedule scan + offline gate (resolve/radius).

Vectors from frozen hexo_rl/config/resolve/radius.py. The eval_board delegation
(resolve_eval_radius) is NOT ported (relocates to WP-eval; config->eval is a DAG violation).
"""
import pytest

from mantis.config.resolve.radius import (
    OfflineRadiusUnresolvableError,
    require_offline_radius,
    resolve_radius_from_schedule,
)

_SCHED = [
    {"step": 0, "radius": 4},
    {"step": 200_000, "radius": 5},
    {"step": 400_000, "radius": 6},
]


def test_schedule_scan_last_step_le_query_wins():
    assert resolve_radius_from_schedule(_SCHED, 0) == 4
    assert resolve_radius_from_schedule(_SCHED, 199_999) == 4
    assert resolve_radius_from_schedule(_SCHED, 200_000) == 5
    assert resolve_radius_from_schedule(_SCHED, 500_000) == 6


def test_no_schedule_returns_none():
    assert resolve_radius_from_schedule(None, 100) is None
    assert resolve_radius_from_schedule([], 100) is None


def test_offline_override_wins():
    assert require_offline_radius(None, 4) == 4


def test_offline_resolved_wins_when_no_override():
    assert require_offline_radius(5, None) == 5


def test_offline_unresolvable_raises_naming_ckpt_and_flag():
    with pytest.raises(OfflineRadiusUnresolvableError) as exc:
        require_offline_radius(None, None, ckpt_label="strip.pt")
    msg = str(exc.value)
    assert "strip.pt" in msg and "--radius-stage" in msg


def test_offline_error_is_valueerror():
    assert issubclass(OfflineRadiusUnresolvableError, ValueError)
