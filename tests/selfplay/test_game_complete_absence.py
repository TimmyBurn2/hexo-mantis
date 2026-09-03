"""AUDIT-1 F-28 rows C04 and C05 — the `game_complete` payload distinguishes 0 from absent.

TWO FABRICATIONS IN ONE PAYLOAD.

* **The six investigation metrics** (`colony_extension_stone_count` / `_total` / `_fraction`,
  `longest_line_fraction`, `n_components`) were `0`/`0.0` whenever
  `log_investigation_metrics` was off OR the game recorded no moves. A zero longest line and
  a zero component count are legitimate MEASUREMENTS for other games in the same run, so the
  lever-off case and the measured-zero case were one observable.
* **The winner map.** `{0: -1, 1: 0, 2: 1}.get(winner_code, -1)` sent an UNRECOGNISED code to
  `-1`, i.e. reported it as a measured DRAW — while the log line emitted from the same block
  printed `winner=unknown` off `_WINNER_NAMES[...] if winner_code < 3 else "unknown"`. Two
  readings of one game, and the ONE channel carried the wrong one. The captured drain golden
  had frozen exactly this: game 6 of `dense_5s_crossed` carries `winner_code = 3`.

C05 is the same class in `monitor_gates`: `_watchdog_counters` returned `{}` for both "the
watchdog is armed and nothing has failed" and "there is no watchdog wired at all". Its
sibling `data_loss_counters` is NOT this defect and is deliberately left alone — its
`BestEffortCounters` registry always exists and is always counting, so an empty snapshot from
it is a true "nothing was lost".
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


# ── C04: the six investigation metrics ────────────────────────────────────────────────

def test_the_investigation_metrics_are_absent_when_the_lever_is_OFF() -> None:
    """THE PIN. Six zeros before the repair."""
    ext_c, ext_t, ext_f, _p90, ll, ll_frac, n_comp = _run(
        PoolInstrumentation(log_investigation_metrics=False, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD), moves=_MOVES
    )
    assert (ext_c, ext_t, ext_f) == (None, None, None)
    assert (ll, ll_frac, n_comp) == (None, None, None)


def test_the_investigation_metrics_are_absent_when_the_game_recorded_no_moves() -> None:
    """The other gate on the same block — and the one the drain golden had frozen."""
    ext_c, ext_t, ext_f, _p90, ll, ll_frac, n_comp = _run(
        PoolInstrumentation(log_investigation_metrics=True, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD), moves=[]
    )
    assert (ext_c, ext_t, ext_f) == (None, None, None)
    assert (ll, ll_frac, n_comp) == (None, None, None)


def test_a_MEASURED_zero_still_reads_as_zero() -> None:
    """The load-bearing control. Two adjacent stones extend nothing, so the colony count is a
    real 0 — and that number must survive, or the repair has replaced one collision with
    another."""
    adjacent = [(0, 0), (1, 0)]
    ext_c, ext_t, ext_f, _p90, _ll, _ll_frac, n_comp = _run(
        PoolInstrumentation(log_investigation_metrics=True, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD), moves=adjacent
    )
    assert ext_c == 0 and ext_c is not None
    assert ext_t == 2, "the denominator is measured, so the fraction is meaningful"
    assert ext_f == 0.0 and ext_f is not None
    assert n_comp is not None


# ── C04: the undecodable winner ───────────────────────────────────────────────────────

def test_an_unrecognised_winner_code_is_absent_not_a_draw() -> None:
    """THE PIN. `winner_code = 3` reached the event as `-1` — a measured draw — while the log
    line beside it said `unknown`."""
    import mantis.selfplay.pool_drain as pd

    source = pd.__loader__.get_source("mantis.selfplay.pool_drain")
    assert '{0: -1, 1: 0, 2: 1}.get(winner_code)' in source, (
        "the winner map grew a fallback again: an unrecognised code must not resolve to one "
        "of the three real outcomes"
    )
    assert '.get(winner_code, -1)' not in source


def test_the_captured_drain_golden_no_longer_freezes_either_fabrication() -> None:
    """The fixture is the third witness, and it had both cells in it."""
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
    # and the games that DID measure still carry their numbers
    assert events[0]["colony_extension_stone_count"] == 12
    assert events[1]["colony_extension_stone_count"] == 0, (
        "a measured zero in the very same capture — this is why absence cannot share its value"
    )


# ── C05: the watchdog counters ────────────────────────────────────────────────────────

class _Coord:
    """`StepCoordinator._watchdog_counters` invoked against a stand-in carrying only the one
    attribute it reads."""

    def __init__(self, watchdog: Any) -> None:
        self.heartbeat_watchdog = watchdog

    def counters(self) -> Any:
        from mantis.train.coordinator.step import StepCoordinator

        return StepCoordinator._watchdog_counters(self)


def test_no_watchdog_wired_reports_absence_not_a_clean_bill() -> None:
    """THE PIN. `{}` used to mean both "armed, nothing failed" and "no fire path exists"."""
    assert _Coord(None).counters() is None
    assert _Coord(object()).counters() is None, "a watchdog with no counters is still absent"


def test_an_ARMED_watchdog_with_nothing_to_report_still_reports_an_empty_mapping() -> None:
    """The control: an empty snapshot from a LIVE watchdog is a real, good measurement."""
    from mantis.monitor.best_effort import BestEffortCounters

    live = type("W", (), {"counters": BestEffortCounters()})()
    assert _Coord(live).counters() == {}
    live.counters.increment("mirror_failed")
    assert _Coord(live).counters() == {"mirror_failed": 1}


def test_the_data_loss_counters_sibling_is_deliberately_UNCHANGED() -> None:
    """Stated so the asymmetry is a decision, not an oversight. `REPLAY_COUNTERS` is a
    module-level registry that always exists and is always counting, so `{}` from it is a
    true "nothing was lost" — not the absence C05 is about."""
    from mantis.data import loss_counters

    assert loss_counters.REPLAY_COUNTERS.snapshot() is not None
    assert isinstance(loss_counters.REPLAY_COUNTERS.snapshot(), dict)
