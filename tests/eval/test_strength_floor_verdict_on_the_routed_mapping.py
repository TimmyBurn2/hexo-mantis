"""A round the strength floor REFUSED is a THIRD thing on the routed mapping and the stream.

Not broken, not healthy: deliberately not played. The verdict is produced by the worker, read
by `_emit_posture_events` onto `eval_strength_floor`, and carried by `_success_result` onto the
mapping the promotion seam reads — three surfaces that must agree. PRESENCE IS THE ARMING
EVIDENCE: a disarmed round carries no `strength_floor` key and reads exactly as it did before
the floor existed. (The coordinator gate that once NAMED the refusal on its own skip channel
left with the sealbot rung, R362(c); the mapping and the stream are the surfaces that remain.)
"""
from __future__ import annotations

from typing import Any

import pytest

from mantis.eval.errors import EvalBrokenReason
from mantis.eval.floor_gate import evaluate_strength_floor
from mantis.eval.rounds import build_round_result

pytest.importorskip("torch")


class _Rec:
    """The two `GameRecord` fields `probe_measurements` reads, and nothing else, so it cannot
    drift by carrying a stale field."""

    def __init__(self, *, terminal: str, winner: int | None) -> None:
        self.terminal, self.winner = terminal, winner


def _verdict_payload(*, decisive: int, games: int) -> dict[str, Any]:
    """A REAL floor verdict from the production rule, never a hand-written dict; only the
    probe's RECORDS are planted, since no off-box checkpoint gives a controlled decisive rate."""
    from mantis.arena.adjudicate import TERMINAL_PLY_CAP, TERMINAL_WIN
    from mantis.config.resolve.eval_posture import StrengthFloorSpec

    records = [_Rec(terminal=TERMINAL_WIN, winner=1) for _ in range(decisive)]
    records += [_Rec(terminal=TERMINAL_PLY_CAP, winner=None) for _ in range(games - decisive)]
    spec = StrengthFloorSpec(probe_games=games, min_decisive_rate=0.25, min_winrate=0.0)
    return evaluate_strength_floor(records, spec).as_payload()


def _round(*, floor: dict[str, Any] | None, reason: EvalBrokenReason | None = None,
           step: int = 5000) -> dict[str, Any]:
    """A round result built by the PRODUCTION producer, floor payload threaded as it is live."""
    return build_round_result(
        step=step, round_id=f"r000001_{step}", gate_result=None, eval_round_wall_sec=340.6,
        reason=reason, detail=None, random_wr=None, strength_floor=floor,
    )


def test_the_producer_carries_the_verdict_onto_the_routed_mapping() -> None:
    """`build_round_result` puts the floor verdict on the routed mapping, and a refused round is
    NOT broken — that is the whole distinction."""
    floor = _verdict_payload(decisive=0, games=4)
    result = _round(floor=floor)
    assert result["strength_floor"] == floor
    assert result["strength_floor"]["passed"] is False
    assert result["eval_broken_reason"] is None
    assert result["promoted"] is False, "the refused round never played the gate block"


def test_the_disarmed_posture_puts_NO_floor_key_on_the_routed_mapping() -> None:
    """Presence IS the arming evidence: a disarmed round's mapping is byte-identical to a
    pre-floor one."""
    assert "strength_floor" not in _round(floor=None)


def test_the_three_cases_are_PAIRWISE_DISTINCT_on_the_mapping() -> None:
    """Refused, healthy-without-a-gate and broken read as three different mappings, asserted as
    one fact rather than inferred from three rows passing."""
    refused = _round(floor=_verdict_payload(decisive=0, games=4))
    healthy = _round(floor=_verdict_payload(decisive=4, games=4))
    broken = _round(floor=None, reason=next(iter(EvalBrokenReason)))
    keys = (
        (refused["strength_floor"]["passed"], refused["eval_broken_reason"]),
        (healthy["strength_floor"]["passed"], healthy["eval_broken_reason"]),
        (broken.get("strength_floor"), broken["eval_broken_reason"]),
    )
    assert len(set(keys)) == 3, keys


@pytest.mark.parametrize("reason", tuple(EvalBrokenReason), ids=[r.value for r in EvalBrokenReason])
def test_a_broken_round_keeps_its_floor_payload_beside_the_stronger_fact(reason: EvalBrokenReason) -> None:
    """A broken round may still carry a floor payload from before the break; the mapping carries
    both, and "this round could not run" is the fact a consumer must read first."""
    result = _round(floor=_verdict_payload(decisive=0, games=4), reason=reason)
    assert result["eval_broken_reason"] is reason
    assert result["strength_floor"]["passed"] is False
    assert result["promoted"] is False


class _FakePipeline:
    """`EvalPipeline._success_result` lifted off the class, so the code exercised is production."""

    def __init__(self, sink) -> None:
        self._sink = sink
        self._round_counter = 0
        self._floor_checked_total = 0
        self._floor_skipped_total = 0

    def _emit_posture_events(self, inflight, raw) -> None:
        """The PRODUCTION method: the event channel and the routed mapping read the SAME `raw`
        key, so a payload that stops arriving silences both rather than staling one."""
        from mantis.eval.pipeline import EvalPipeline

        EvalPipeline._emit_posture_events(self, inflight, raw)


def _round_complete_event(sink_events: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [e for e in sink_events if e.get("event") == "eval_round_complete"]
    assert len(matches) == 1, matches
    return matches[0]


def _drive_success_result(raw: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The PRODUCTION `_success_result` against the lifted stand-in; returns (result, events)."""
    from mantis.eval.pipeline import EvalPipeline

    class _Sink:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def emit(self, payload: dict) -> None:
            self.events.append(dict(payload))

    sink = _Sink()
    fake = _FakePipeline(sink)
    result = EvalPipeline._success_result(
        fake, {"round_id": "r000001_5000", "step": 5000, "round_idx": 1}, raw, wall_sec=12.5,
    )
    return result, sink.events


@pytest.mark.parametrize("armed", [True, False], ids=["armed", "disarmed"])
def test_the_PIPELINE_carries_the_workers_floor_payload_onto_the_routed_mapping(
    armed: bool,
) -> None:
    """THE SEAM ROW. `_success_result` is the only place the child's verdict can enter the
    routed mapping, and deleting that keyword left every other row here green."""
    floor = _verdict_payload(decisive=0, games=4) if armed else None
    raw: dict[str, Any] = {"rungs": {}, "gate": None, "random": {"games": 0, "wr": None},
                           "skipped_rungs": []}
    if floor is not None:
        raw["strength_floor"] = floor
    result, events = _drive_success_result(raw)
    floor_events = [e for e in events if e.get("event") == "eval_strength_floor"]
    if armed:
        assert result["strength_floor"] == floor, (
            "the worker measured a floor verdict and the routed mapping dropped it"
        )
        assert len(floor_events) == 1 and floor_events[0]["passed"] is False
        assert _round_complete_event(events)["promoted"] is None, "no gate ran: no decision"
    else:
        assert "strength_floor" not in result
        assert floor_events == []


# `_success_result` sums random + gate games, all zero on a refused floor, so
# `eval_round_complete.games_total` read 0 for a round that had just played `probe_games` real
# games. The 0 was COMPUTED, so the sentinel banning a literal `0` at the call site never saw it.

def test_a_floor_refused_round_reports_the_games_its_probe_PLAYED() -> None:
    """Sixteen probe games, every other phase empty: `games_total` is 16, not the 0 it read."""
    floor = _verdict_payload(decisive=0, games=16)
    raw: dict[str, Any] = {
        "rungs": {}, "gate": None, "random": {"games": 0, "wr": None},
        "skipped_rungs": [], "strength_floor": floor,
    }
    _result, events = _drive_success_result(raw)
    complete = _round_complete_event(events)
    assert complete["games_total"] == 16, complete


def test_the_two_events_AGREE_about_the_probe_games() -> None:
    """`eval_round_complete` and `eval_strength_floor` must agree on the probe's game count."""
    floor = _verdict_payload(decisive=0, games=16)
    raw: dict[str, Any] = {
        "rungs": {}, "gate": None, "random": {"games": 0, "wr": None},
        "skipped_rungs": [], "strength_floor": floor,
    }
    _result, events = _drive_success_result(raw)
    floor_events = [e for e in events if e.get("event") == "eval_strength_floor"]
    assert len(floor_events) == 1, events
    assert _round_complete_event(events)["games_total"] == floor_events[0]["games"] == 16


def test_a_PASSING_floor_adds_its_probe_games_to_the_rounds_total() -> None:
    """The probe's games count on the passing path too, or `games_total` means two things."""
    floor = _verdict_payload(decisive=4, games=4)
    assert floor["passed"] is True
    raw: dict[str, Any] = {
        "rungs": {}, "gate": None, "random": {"games": 6, "wr": 0.5},
        "skipped_rungs": [], "strength_floor": floor,
    }
    _result, events = _drive_success_result(raw)
    assert _round_complete_event(events)["games_total"] == 6 + 4


def test_the_DISARMED_round_total_is_byte_identical_to_the_pre_floor_sum() -> None:
    """With no floor key the round total must be exactly what it always was."""
    raw: dict[str, Any] = {
        "rungs": {}, "gate": None, "random": {"games": 6, "wr": 0.5}, "skipped_rungs": [],
    }
    _result, events = _drive_success_result(raw)
    assert _round_complete_event(events)["games_total"] == 6
