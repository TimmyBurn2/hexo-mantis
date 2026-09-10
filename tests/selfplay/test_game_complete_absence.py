"""The `game_complete` and `monitor_gates` payloads distinguish a measured 0 from an absence.

Three places where they did not: the six investigation metrics read 0 when the lever was off or
the game recorded no moves; an unrecognised winner code resolved to a measured DRAW; and
`_watchdog_counters` returned `{}` for both "armed, nothing failed" and "no watchdog wired".

The `data_loss_counters` sibling is deliberately untouched: its registry always exists and is
always counting, so an empty snapshot from it is a true "nothing was lost".
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

from mantis._engine import DEFAULT_CLUSTER_THRESHOLD
from mantis.selfplay.instrumentation import PoolInstrumentation

_MOVES = [(0, 0), (1, 0), (2, 1), (3, 1)]


def _run(instr: PoolInstrumentation, *, moves: list[tuple[int, int]]) -> tuple:
    return instr.on_game_complete(
        threading.Lock(), 1, moves, 0, 0, 0, 0, 1, 0,
    )


def test_the_investigation_metrics_are_absent_when_the_lever_is_OFF() -> None:
    """With the lever off, the six metrics are absent rather than six zeros."""
    ext_c, ext_t, ext_f, _p90, ll, ll_frac, n_comp = _run(
        PoolInstrumentation(log_investigation_metrics=False, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD), moves=_MOVES
    )
    assert (ext_c, ext_t, ext_f) == (None, None, None)
    assert (ll, ll_frac, n_comp) == (None, None, None)


def test_the_investigation_metrics_are_absent_when_the_game_recorded_no_moves() -> None:
    """The other gate on the same block: a game with no moves measures nothing."""
    ext_c, ext_t, ext_f, _p90, ll, ll_frac, n_comp = _run(
        PoolInstrumentation(log_investigation_metrics=True, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD), moves=[]
    )
    assert (ext_c, ext_t, ext_f) == (None, None, None)
    assert (ll, ll_frac, n_comp) == (None, None, None)


def test_a_MEASURED_zero_still_reads_as_zero() -> None:
    """Control: two adjacent stones extend nothing, so that real 0 must survive."""
    adjacent = [(0, 0), (1, 0)]
    ext_c, ext_t, ext_f, _p90, _ll, _ll_frac, n_comp = _run(
        PoolInstrumentation(log_investigation_metrics=True, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD), moves=adjacent
    )
    assert ext_c == 0 and ext_c is not None
    assert ext_t == 2, "the denominator is measured, so the fraction is meaningful"
    assert ext_f == 0.0 and ext_f is not None
    assert n_comp is not None


def test_an_unrecognised_winner_code_is_absent_not_a_draw() -> None:
    """An unrecognised winner code must not resolve to one of the three real outcomes."""
    import mantis.selfplay.pool_drain as pd

    source = pd.__loader__.get_source("mantis.selfplay.pool_drain")
    assert '{0: -1, 1: 0, 2: 1}.get(winner_code)' in source, (
        "the winner map grew a fallback again: an unrecognised code must not resolve to one "
        "of the three real outcomes"
    )
    assert '.get(winner_code, -1)' not in source


def test_the_captured_drain_golden_no_longer_freezes_either_fabrication() -> None:
    """The captured drain golden carries both absences and the measured values beside them."""
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    golden = json.loads(
        (repo / "tests" / "fixtures" / "selfplay" / "drain" / "drain_goldens.json")
        .read_text(encoding="utf-8")
    )
    events = [e for e in golden["variants"]["dense_5s_crossed"]["events"]
              if e["event"] == "game_complete"]
    assert events[5]["winner"] is None, "winner_code 3 is captured as a measured draw again"
    assert events[3]["colony_extension_stone_count"] is None
    assert events[3]["n_components"] is None
    # The games that DID measure still carry their numbers.
    assert events[0]["colony_extension_stone_count"] == 12
    assert events[1]["colony_extension_stone_count"] == 0, (
        "a measured zero in the very same capture — this is why absence cannot share its value"
    )


class _Coord:
    """Stand-in carrying only the attribute `_watchdog_counters` reads."""

    def __init__(self, watchdog: Any) -> None:
        self.heartbeat_watchdog = watchdog

    def counters(self) -> Any:
        from mantis.train.coordinator.step import StepCoordinator

        return StepCoordinator._watchdog_counters(self)


def test_no_watchdog_wired_reports_absence_not_a_clean_bill() -> None:
    """No watchdog wired reports absence, not an empty clean bill."""
    assert _Coord(None).counters() is None
    assert _Coord(object()).counters() is None, "a watchdog with no counters is still absent"


def test_an_ARMED_watchdog_with_nothing_to_report_still_reports_an_empty_mapping() -> None:
    """Control: an empty snapshot from a LIVE watchdog is a real measurement."""
    from mantis.monitor.best_effort import BestEffortCounters

    live = type("W", (), {"counters": BestEffortCounters()})()
    assert _Coord(live).counters() == {}
    live.counters.increment("mirror_failed")
    assert _Coord(live).counters() == {"mirror_failed": 1}


def test_the_data_loss_counters_sibling_is_deliberately_UNCHANGED() -> None:
    """The asymmetry is a decision: `PIPELINE_COUNTERS` always exists and is always counting,
    so `{}` from it is a true "nothing was lost" — and a registry no producer feeds would be
    the opposite, an always-empty mapping reading as a measurement."""
    from mantis.data import loss_counters

    assert loss_counters.PIPELINE_COUNTERS.snapshot() is not None
    assert isinstance(loss_counters.PIPELINE_COUNTERS.snapshot(), dict)
    assert not hasattr(loss_counters, "REPLAY_COUNTERS"), (
        "the deleted registry came back without its producers — an always-empty snapshot "
        "reading as 'nothing was lost' is precisely the absence C05 is about"
    )
