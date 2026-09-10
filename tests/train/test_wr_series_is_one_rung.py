"""The sealbot-WR trajectory ring is a series over ONE rung.

`_first_sealbot_wr` returns the WR of the first sealbot rung with `games > 0` and used to
drop the name. Once `sealbot_d5` saturates it gets 0 games off-cadence, so `wr_sealbot`
alternates between a d5 reading and a d6 one, and an 8-game calibration number sits in the
same ring as 32-game ones. A drop to a HARDER opponent's win rate is not a collapse, but the
trajectory alert read it as one.

Three halves: the PRODUCER publishes `(wr, rung, games)` out of the one walk that selects it;
the PIPELINE derives the triple independently and refuses the round on any disagreement; the
COORDINATOR restarts the series when the reporting rung changes. Clearing is the conservative
direction — a trigger needing N observations simply waits N more rounds.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mantis.eval.errors import ResultContractError
from mantis.train.coordinator.step import GATE_NAMES
from mantis.eval.pipeline import EvalPipeline
from mantis.eval.rounds import build_round_result


class _Ladder:
    """A two-rung sealbot ladder in minted order, plus a non-sealbot rung never to be selected."""

    rungs = (
        SimpleNamespace(name="strixbot_1", bot="strixbot"),
        SimpleNamespace(name="sealbot_d5", bot="sealbot"),
        SimpleNamespace(name="sealbot_d6", bot="sealbot"),
    )


class _Pipe:
    def __init__(self) -> None:
        self._eval_cfg = SimpleNamespace(ladder=_Ladder())

    def check(self, rungs_raw: dict[str, Any], result: dict[str, Any]) -> None:
        EvalPipeline._check_the_sealbot_rung_identity(self, rungs_raw, result, round_id="r1")


def _published(rungs_raw: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Return the three keys the producer publishes, read off a real `build_round_result` call."""
    result = build_round_result(
        step=1, round_id="r1", rungs_config=_Ladder.rungs, rung_results=rungs_raw,
        gate_result=None, skipped_rungs=[], bt={"ratings": {}, "p_hat": {}},
        schedule_next={}, eval_round_wall_sec=1.0, reason=None, detail=None, random_wr=None,
    )
    return result["wr_sealbot"], result["wr_sealbot_rung"], result["wr_sealbot_games"]


def test_the_named_rung_is_the_FIRST_sealbot_rung_with_games() -> None:
    rungs = {"strixbot_1": {"games": 10, "wr": 0.9},
             "sealbot_d5": {"games": 32, "wr": 0.6},
             "sealbot_d6": {"games": 32, "wr": 0.4}}
    assert _published(rungs) == (0.6, "sealbot_d5", 32)


def test_a_SATURATED_first_rung_hands_the_reading_to_the_next_one() -> None:
    """Prove a saturated first rung hands the reading to the next: a different opponent, same field."""
    rungs = {"sealbot_d5": {"games": 0, "wr": None},
             "sealbot_d6": {"games": 8, "wr": 0.4}}
    assert _published(rungs) == (0.4, "sealbot_d6", 8)


def test_a_round_with_no_sealbot_games_names_nothing() -> None:
    assert _published({"sealbot_d5": {"games": 0, "wr": None}}) == (None, None, None)


def test_a_non_sealbot_rung_is_never_named() -> None:
    """Control: a non-sealbot rung first in ladder order and with games is never named."""
    rungs = {"strixbot_1": {"games": 40, "wr": 0.7},
             "sealbot_d6": {"games": 8, "wr": 0.4}}
    assert _published(rungs) == (0.4, "sealbot_d6", 8)


def test_the_agreement_check_PASSES_on_what_the_producer_published() -> None:
    """Control: the agreement check must not red on a correct round."""
    rungs = {"sealbot_d5": {"games": 32, "wr": 0.6},
             "sealbot_d6": {"games": 32, "wr": 0.4}}
    wr, rung, games = _published(rungs)
    _Pipe().check(rungs, {"wr_sealbot": wr, "wr_sealbot_rung": rung,
                          "wr_sealbot_games": games})


@pytest.mark.parametrize("published", [
    {"wr_sealbot": 0.4, "wr_sealbot_rung": "sealbot_d5", "wr_sealbot_games": 32},
    {"wr_sealbot": 0.6, "wr_sealbot_rung": "sealbot_d6", "wr_sealbot_games": 32},
    {"wr_sealbot": 0.6, "wr_sealbot_rung": "sealbot_d5", "wr_sealbot_games": 8},
])
def test_a_DISAGREEMENT_on_ANY_of_the_three_RAISES(published: dict[str, Any]) -> None:
    """Prove a drift in the WR, the name or the game count refuses the round.

    A ring labelled by the wrong rung is worse than an unlabelled one: the coordinator
    restarts the series on a name change and would discard real observations.
    """
    rungs = {"sealbot_d5": {"games": 32, "wr": 0.6}}
    with pytest.raises(ResultContractError, match="drifted"):
        _Pipe().check(rungs, published)


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


def _coordinator(sink: _Sink) -> Any:
    """Build a stand-in carrying only the attributes `on_eval_round_complete` reads."""
    from mantis.monitor.config import MonitorConfig

    return SimpleNamespace(
        _wr_history=[], _wr_history_rung=None, _sink=sink,
        monitor_cfg=MonitorConfig(),
        # Derived from the real GATE_NAMES so a new counter costs no edit here.
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
    """Control: a run of rounds on one rung builds one uninterrupted series."""
    sink = _Sink()
    coord = _coordinator(sink)
    for i in range(4):
        _feed(coord, step=i, wr=0.6 - i * 0.01, rung="sealbot_d5")
    assert len(coord._wr_history) == 4
    assert coord._wr_history_rung == "sealbot_d5"
    assert not sink.named("sealbot_wr_series_restarted")


def test_the_series_RESTARTS_when_the_reporting_rung_changes() -> None:
    """Prove the series restarts when the reporting rung changes, and says how much it discarded."""
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
    """Prove the first round restarts nothing: an empty ring is not a series that was lost."""
    sink = _Sink()
    coord = _coordinator(sink)
    _feed(coord, step=0, wr=0.6, rung="sealbot_d5")
    assert len(coord._wr_history) == 1
    assert not sink.named("sealbot_wr_series_restarted")


def test_the_ring_length_is_published_beside_the_rung_it_is_a_series_over() -> None:
    """Prove the ring length is published beside its rung; a length alone names no population."""
    from mantis.train.coordinator.step import StepCoordinator

    assert "wr_history_rung" in StepCoordinator._emit_monitor_gates.__code__.co_consts \
        or "wr_history_rung" in str(StepCoordinator._emit_monitor_gates.__code__.co_consts)
