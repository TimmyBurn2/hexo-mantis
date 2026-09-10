# >300 justify: producer, seam and consumer are ONE claim here — the worker's verdict reached
# the gate through NOTHING, so a file split at any of the three joins would assert a key it
# also invented and pass over the gap.
"""A round the strength floor REFUSED is NAMED at LAW-15's gate.

A refused round is a THIRD thing and was reported as the first: not broken, not
healthy-but-metric-less, but deliberately not played, so the gate emitted `wr_sealbot_absent`.
The producer half is here because THE ROUTE IS THE DEFECT — the verdict was produced by the
worker, read by `_emit_posture_events`, and reached the gate through nothing. PRESENCE IS THE
ARMING EVIDENCE: every committed config mints `eval.strength_floor: null`, so a disarmed round
carries no key and this gate behaves as it did before the floor existed.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.eval.errors import EvalBrokenReason
from mantis.eval.floor_gate import evaluate_strength_floor
from mantis.eval.rounds import build_round_result

pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[2]


def _make_coordinator():
    """A minimal coordinator, harness PRIVATE to this file: `tests` is not a package (R5), so a
    cross-test import resolves under one pytest invocation and raises under another. The config
    is DERIVED from the production builder, so a new coordinator knob costs no edit here."""
    from mantis.config.loader import load_config
    from mantis.config.resolve.coordinator import resolve_coordinator_knobs
    from mantis.config.resolve.drain import resolve_drain_caps
    from mantis.monitor.config import MonitorConfig
    from mantis.run import _step_coordinator_config
    from mantis.train.coordinator.step import StepCoordinator
    from mantis.train.lifecycle.signals import ShutdownState

    dev = load_config(_REPO / "configs" / "dev_example.yaml")

    class _Sink:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, payload: dict) -> None:
            self.events.append(payload)

        def named(self, name: str) -> list[dict]:
            return [e for e in self.events if e.get("event") == name]

    class _Pool:
        games_completed = 0
        avg_game_length = 0.0

        def runner_stats(self):
            return SimpleNamespace()

    class _Trainer:
        step = 0
        model = object()

        def train_step_from_tensors(self, *a, **k) -> dict:
            return {}

        def save_checkpoint(self, loss_info) -> None: ...

    class _Buffer:
        size = 1000
        capacity = 100_000

        def resize(self, n: int) -> None: ...

        def save_to_path(self, p) -> None: ...

    sink = _Sink()
    config = dataclasses.replace(
        _step_coordinator_config(
            stop_step=10 ** 9, draw_rate_abort=None,
            drain_caps=resolve_drain_caps(dev.monitor),
            gate_interval=dev.monitor.gate_interval,
            knobs=resolve_coordinator_knobs(dev.train),
        ),
        eval_interval=1, log_interval=1, gate_interval=1, min_buf_size=10,
    )
    coord = StepCoordinator(
        trainer=_Trainer(), buffer=_Buffer(), pretrained_buffer=None, recent_buffer=None,
        pool=_Pool(), eval_pipeline=None, subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        shutdown=ShutdownState(), eval_model=object(), bufs=None,
        config=config, full_config={}, train_cfg={}, mixing_cfg={},
        sink=sink, heartbeat=None, monitor_cfg=MonitorConfig(),
    )
    return SimpleNamespace(coord=coord, sink=sink)


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
        step=step, round_id=f"r000001_{step}", rungs_config=[], rung_results={},
        gate_result=None, skipped_rungs=[], bt={}, schedule_next={},
        eval_round_wall_sec=340.6, reason=reason, detail=None, random_wr=None,
        strength_floor=floor,
    )


def _skip_event(result: dict[str, Any]) -> dict[str, Any]:
    harness = _make_coordinator()
    harness.coord.on_eval_round_complete(result)
    events = harness.sink.named("sealbot_wr_gate_skipped")
    assert len(events) == 1, f"exactly one skip event per delivered round; got {events}"
    return dict(events[0])


def test_the_producer_carries_the_verdict_the_gate_reads() -> None:
    """`build_round_result` must put the floor verdict on the routed mapping — before this, no
    branch on `strength_floor` in `step.py` could ever have fired."""
    floor = _verdict_payload(decisive=0, games=4)
    result = _round(floor=floor)
    assert result["strength_floor"] == floor
    assert result["strength_floor"]["passed"] is False
    assert result["eval_broken_reason"] is None, (
        "a floor-refused round is NOT broken — that is the whole distinction, and if the "
        "producer marked it broken the third reason would be unreachable"
    )
    assert result["wr_sealbot"] is None, (
        "the refused round never played the gate block, which is why it arrives at this gate "
        "with no number at all"
    )
    assert result["promoted"] is False


def test_the_disarmed_posture_puts_NO_floor_key_on_the_routed_mapping() -> None:
    """Presence IS the arming evidence: a disarmed round's mapping is byte-identical to a
    pre-floor one."""
    assert "strength_floor" not in _round(floor=None)


class _FakePipeline:
    """`EvalPipeline._success_result` lifted off the class, ladder/config collaborators stubbed
    so the code exercised is production."""

    class _Ladder:
        rungs: tuple = ()
        bt_prior_games = 1.0
        # Threaded into the ONE CI authority the assessment reuses; real values, not stubs.
        bootstrap_resamples = 200
        bootstrap_ci_level = 0.95
        bootstrap_seed = 0

    class _State:
        def status(self, rung: str) -> str:
            """`_success_result` stamps each rung's REAL ladder status, so this must answer."""
            return "active"

        def record_round(self, *a, **k) -> None: ...

        def save(self, *a, **k) -> None: ...

        def allocate_games(self, *a, **k) -> dict:
            return {}

    def __init__(self, sink) -> None:
        self._sink = sink
        self._eval_cfg = SimpleNamespace(ladder=self._Ladder())
        self._ladder_state_path = Path("/nonexistent/ladder.json")
        self._last_p_hat: dict = {}
        # `_finalize_round` also drives the external-channel assessment, so the stand-in
        # carries the REAL method: a stub would keep these rows green with the producer unrun.
        from functools import partial

        from mantis.eval.pipeline import EvalPipeline as _EP

        self._external_history: list = []
        self._degradation_flags = 0
        self._round_counter = 0
        self._assess_external_channel = partial(_EP._assess_external_channel, self)
        self._floor_checked_total = 0
        self._floor_skipped_total = 0
        self._state = self._State()

    def _ensure_ladder_state(self):
        return self._state

    def _current_p_hat(self) -> dict:
        return self._last_p_hat

    def _check_the_sealbot_rung_identity(self, rungs_raw, result, *, round_id):
        """The PRODUCTION method — it walks the `rungs` this stand-in supplies."""
        from mantis.eval.pipeline import EvalPipeline

        return EvalPipeline._check_the_sealbot_rung_identity(
            self, rungs_raw, result, round_id=round_id)

    def _emit_posture_events(self, inflight, raw) -> None:
        """The PRODUCTION method: the event channel and the routed mapping read the SAME `raw`
        key, so a payload that stops arriving silences both rather than staling one."""
        from mantis.eval.pipeline import EvalPipeline

        EvalPipeline._emit_posture_events(self, inflight, raw)


@pytest.mark.parametrize("armed", [True, False], ids=["armed", "disarmed"])
def test_the_PIPELINE_carries_the_workers_floor_payload_onto_the_routed_mapping(
    armed: bool,
) -> None:
    """THE SEAM ROW. `_success_result` is the only place the child's verdict can enter the
    mapping the gate reads, and deleting that keyword left every other row here green."""
    from mantis.eval.pipeline import EvalPipeline

    class _Sink:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, payload: dict) -> None:
            self.events.append(dict(payload))

    floor = _verdict_payload(decisive=0, games=4) if armed else None
    raw: dict[str, Any] = {"rungs": {}, "gate": None, "random": {"games": 0, "wr": None},
                           "skipped_rungs": []}
    if floor is not None:
        raw["strength_floor"] = floor
    fake = _FakePipeline(_Sink())
    result = EvalPipeline._success_result(
        fake, {"round_id": "r000001_5000", "step": 5000, "round_idx": 1}, raw, wall_sec=340.6,
    )
    if armed:
        assert result["strength_floor"] == floor, (
            "the worker measured a floor verdict and the routed mapping dropped it — the "
            "gate can then never name a refusal, which is the pre-R324(d) state"
        )
        assert _skip_event(result)["reason"] == "strength_floor_refused"
    else:
        assert "strength_floor" not in result
        assert _skip_event(result)["reason"] == "wr_sealbot_absent"


def test_a_floor_refused_round_is_NAMED_at_the_gate() -> None:
    floor = _verdict_payload(decisive=0, games=4)
    event = _skip_event(_round(floor=floor))
    assert event["reason"] == "strength_floor_refused", event
    assert event["eval_broken_reason"] is None, event
    assert event["strength_floor_failed_bars"] == floor["failed_bars"], event
    assert event["strength_floor_failed_bars"], (
        "a refusal that names no failed bar is a verdict with no evidence — the payload "
        "already carries them and this is what puts them on the gate's own channel"
    )
    assert event["skipped_total"] == 1, event


def test_a_floor_that_PASSED_is_not_a_refusal() -> None:
    """A round whose probe PASSED but produced no sealbot number keeps its own reason."""
    floor = _verdict_payload(decisive=4, games=4)
    assert floor["passed"] is True
    event = _skip_event(_round(floor=floor))
    assert event["reason"] == "wr_sealbot_absent", event
    assert event["strength_floor_failed_bars"] is None, event


def test_the_disarmed_round_still_says_wr_sealbot_absent() -> None:
    """THE LOAD-BEARING ROW: without it, a branch naming every metric-less round
    `strength_floor_refused` would satisfy every row above while re-opening the hole."""
    event = _skip_event(_round(floor=None))
    assert event["reason"] == "wr_sealbot_absent", event
    assert event["strength_floor_failed_bars"] is None, event


@pytest.mark.parametrize("reason", tuple(EvalBrokenReason), ids=[r.value for r in EvalBrokenReason])
def test_broken_OUTRANKS_refused_and_the_precedence_is_pinned(reason: EvalBrokenReason) -> None:
    """A broken round may still carry a floor payload from before the break, and "this round
    could not run" is the stronger fact; pinned so the ordering is a decision."""
    event = _skip_event(_round(floor=_verdict_payload(decisive=0, games=4), reason=reason))
    assert event["reason"] == "eval_round_broken", event
    assert event["eval_broken_reason"] is reason, event


def test_the_three_reasons_are_PAIRWISE_DISTINCT() -> None:
    """The whole point, asserted as one fact rather than inferred from three rows passing."""
    refused = _skip_event(_round(floor=_verdict_payload(decisive=0, games=4)))["reason"]
    absent = _skip_event(_round(floor=None))["reason"]
    broken = _skip_event(_round(floor=None, reason=next(iter(EvalBrokenReason))))["reason"]
    assert len({refused, absent, broken}) == 3, (refused, absent, broken)


# `_success_result` sums rung + random + gate games, all zero on a refused floor, so
# `eval_round_complete.games_total` read 0 for a round that had just played `probe_games` real
# games. The 0 was COMPUTED, so the sentinel banning a literal `0` at the call site never saw it.

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
    assert complete["games_total"] != 0, (
        "a refused round that played games reported none — the RECAL §8.1 misread, live"
    )


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
        "rungs": {"sealbot_d5": {"games": 8, "wr": 0.5, "wr_ci_lower": 0.2}},
        "gate": None, "random": {"games": 6, "wr": 0.5},
        "skipped_rungs": [], "strength_floor": floor,
    }
    _result, events = _drive_success_result(raw)
    assert _round_complete_event(events)["games_total"] == 8 + 6 + 4


def test_the_DISARMED_round_total_is_byte_identical_to_the_pre_floor_sum() -> None:
    """With no floor key the round total must be exactly what it always was."""
    raw: dict[str, Any] = {
        "rungs": {"sealbot_d5": {"games": 8, "wr": 0.5, "wr_ci_lower": 0.2}},
        "gate": None, "random": {"games": 6, "wr": 0.5}, "skipped_rungs": [],
    }
    _result, events = _drive_success_result(raw)
    assert _round_complete_event(events)["games_total"] == 14
