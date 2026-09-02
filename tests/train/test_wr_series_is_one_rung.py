"""AUDIT-1 F-14 — the sealbot-WR trajectory ring is a series over ONE rung.

THE DEFECT. `eval/rounds.py::_first_sealbot_wr` returns the WR of the FIRST sealbot rung in
ladder order with `games > 0`, and DROPS the name. Once `sealbot_d5` saturates,
`LadderState.allocate_games` gives it 0 games off-cadence (run5 mints
`calibration_every_k_rounds: 4`, `calibration_games: 8`), so `wr_sealbot` alternates between a
d5 reading and a d6 reading — and an 8-game calibration number sits in the same ring as
32-game ones. `StepCoordinator.on_eval_round_complete` appended `(step, float(wr))` and
`sealbot_wr_trajectory_alert` then tested `wr < peak * ratio` across the lot.

A drop from one opponent's win rate to a HARDER opponent's is not a collapse. That is the
false positive `sealbot_wr_warn` carried and the one the `wr_hard_abort_enabled` capability
would have STOPPED THE RUN on.

TWO HALVES, AND WHERE THEY LIVE. `mantis.eval.rounds` is a FROZEN producer under R118/A-1
(PREREG_A §8 abort 8), so the identity is published by the PIPELINE — which re-walks the same
ladder to NAME what the frozen rule picked and raises if the two disagree (R104,
agreement-or-raise) — and consumed by the coordinator, which restarts the series when the
reporting rung changes. Clearing is the conservative direction: a trigger needing N
observations simply waits N more rounds.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mantis.eval.errors import ResultContractError
from mantis.train.coordinator.step import GATE_NAMES
from mantis.eval.pipeline import EvalPipeline


class _Ladder:
    """A two-rung sealbot ladder in the order run5 mints it, plus a non-sealbot rung that
    must never be selected."""

    rungs = (
        SimpleNamespace(name="strixbot_1", bot="strixbot"),
        SimpleNamespace(name="sealbot_d5", bot="sealbot"),
        SimpleNamespace(name="sealbot_d6", bot="sealbot"),
    )


class _Pipe:
    def __init__(self) -> None:
        self._eval_cfg = SimpleNamespace(ladder=_Ladder())

    def name(self, rungs_raw: dict[str, Any], wr: float | None) -> tuple[Any, Any]:
        return EvalPipeline._name_the_sealbot_rung(self, rungs_raw, wr, round_id="r1")


# ── the pipeline half: naming what the frozen producer picked ─────────────────────────

def test_the_named_rung_is_the_FIRST_sealbot_rung_with_games() -> None:
    pipe = _Pipe()
    rungs = {"strixbot_1": {"games": 10, "wr": 0.9},
             "sealbot_d5": {"games": 32, "wr": 0.6},
             "sealbot_d6": {"games": 32, "wr": 0.4}}
    assert pipe.name(rungs, 0.6) == ("sealbot_d5", 32)


def test_a_SATURATED_first_rung_hands_the_reading_to_the_next_one() -> None:
    """THE MECHANISM. d5 got 0 games this round, so `wr_sealbot` is d6's — a different
    opponent, reported under the same field name."""
    pipe = _Pipe()
    rungs = {"sealbot_d5": {"games": 0, "wr": None},
             "sealbot_d6": {"games": 8, "wr": 0.4}}
    assert pipe.name(rungs, 0.4) == ("sealbot_d6", 8)


def test_a_round_with_no_sealbot_games_names_nothing() -> None:
    assert _Pipe().name({"sealbot_d5": {"games": 0, "wr": None}}, None) == (None, None)


def test_a_DISAGREEMENT_with_the_frozen_producer_RAISES() -> None:
    """R104. The selection rule stays in the frozen `mantis.eval.rounds`; this walk only
    names what it picked. If they ever disagree, a ring labelled by the wrong rung is worse
    than one with no label, so the round is refused rather than mislabelled."""
    pipe = _Pipe()
    rungs = {"sealbot_d5": {"games": 32, "wr": 0.6}}
    with pytest.raises(ResultContractError, match="drifted"):
        pipe.name(rungs, 0.4)


def test_a_non_sealbot_rung_is_never_named() -> None:
    """The control: `strixbot_1` is first in ladder order and has games."""
    pipe = _Pipe()
    rungs = {"strixbot_1": {"games": 40, "wr": 0.7},
             "sealbot_d6": {"games": 8, "wr": 0.4}}
    assert pipe.name(rungs, 0.4) == ("sealbot_d6", 8)


# ── the coordinator half: the ring restarts when the rung changes ─────────────────────

class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


def _coordinator(sink: _Sink) -> Any:
    """The real `StepCoordinator.on_eval_round_complete`, against a stand-in carrying only the
    attributes it reads on the healthy path."""
    from mantis.monitor.config import MonitorConfig

    return SimpleNamespace(
        _wr_history=[], _wr_history_rung=None, _sink=sink,
        monitor_cfg=MonitorConfig(),
        # The shape the production constructor builds, derived from the real GATE_NAMES so a
        # new counter costs no edit here.
        _gate_stats={name: {"checks": 0, "fires": 0, "skips": 0, "warns": 0}
                     for name in GATE_NAMES},
        _train_step=0,
    )


def _feed(coord: Any, *, step: int, wr: float, rung: str) -> None:
    from mantis.train.coordinator.step import StepCoordinator

    StepCoordinator.on_eval_round_complete(coord, {
        "wr_sealbot": wr, "wr_sealbot_rung": rung, "wr_sealbot_games": 32,
        "eval_broken_reason": None,
    })
    coord._train_step = step


def test_a_run_of_rounds_on_ONE_rung_builds_one_series() -> None:
    """The control, first: nothing about the repair may shorten a legitimate series."""
    sink = _Sink()
    coord = _coordinator(sink)
    for i in range(4):
        _feed(coord, step=i, wr=0.6 - i * 0.01, rung="sealbot_d5")
    assert len(coord._wr_history) == 4
    assert coord._wr_history_rung == "sealbot_d5"
    assert not sink.named("sealbot_wr_series_restarted")


def test_the_series_RESTARTS_when_the_reporting_rung_changes() -> None:
    """THE PIN. Before this, d5's 0.60 and d6's 0.40 were consecutive points in one series
    and the trajectory rules read the gap as a collapse."""
    sink = _Sink()
    coord = _coordinator(sink)
    _feed(coord, step=0, wr=0.60, rung="sealbot_d5")
    _feed(coord, step=1, wr=0.62, rung="sealbot_d5")
    _feed(coord, step=2, wr=0.40, rung="sealbot_d6")

    assert coord._wr_history == [(0, 0.40)] or len(coord._wr_history) == 1, coord._wr_history
    assert coord._wr_history_rung == "sealbot_d6"
    restarts = sink.named("sealbot_wr_series_restarted")
    assert len(restarts) == 1, restarts
    assert restarts[0]["from_rung"] == "sealbot_d5"
    assert restarts[0]["to_rung"] == "sealbot_d6"
    assert restarts[0]["discarded"] == 2, (
        "the event must say how much evidence the change threw away, or a reader cannot tell "
        "a one-round blip from a whole series lost"
    )


def test_the_FIRST_round_of_a_run_restarts_nothing_and_says_nothing() -> None:
    """`_wr_history_rung` starts `None`, and an empty ring is not a series that was lost."""
    sink = _Sink()
    coord = _coordinator(sink)
    _feed(coord, step=0, wr=0.6, rung="sealbot_d5")
    assert len(coord._wr_history) == 1
    assert not sink.named("sealbot_wr_series_restarted")


def test_the_ring_length_is_published_beside_the_rung_it_is_a_series_over() -> None:
    """`monitor_gates.wr_history_len` alone is a number over an unnamed population."""
    from mantis.train.coordinator.step import StepCoordinator

    assert "wr_history_rung" in StepCoordinator._emit_monitor_gates.__code__.co_consts \
        or "wr_history_rung" in str(StepCoordinator._emit_monitor_gates.__code__.co_consts)
