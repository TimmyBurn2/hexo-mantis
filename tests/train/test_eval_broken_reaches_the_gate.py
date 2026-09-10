"""A round that BROKE must reach the promotion gate AS a broken round.

A round that COULD NOT RUN is not a round that ran without a number, but both used to emit
`sealbot_wr_gate_skipped` with `reason: "wr_sealbot_absent"` — byte-identical on the stream, so
nothing distinguished "the bar was not met" from "the bar was never measured". The
healthy-but-metric-less arm stays unchanged, or the fix is a blanket rename. The producer half
is driven here too, since the gate can only read a key the round result carries.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.eval.errors import EvalBrokenReason
from mantis.eval.rounds import build_round_result

pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[2]


def _make_coordinator():
    """Build a minimal coordinator, harness PRIVATE to this file: `tests` is not a package, so
    a sibling-module import resolves under one pytest invocation and not another. The config is
    DERIVED from the production builder, so a new knob costs this file no edit."""
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

#: Every reason the taxonomy spells: a guard written around one reason would be silent on the
#: next.
_REASONS = tuple(EvalBrokenReason)


def _broken_round(reason: EvalBrokenReason, *, step: int = 5000) -> dict[str, object]:
    """Build a REAL broken round result through the production producer — a hand-built mapping
    would let this file pass while the producer stopped emitting the key."""
    return build_round_result(
        step=step, round_id=f"r000001_{step}", rungs_config=[], rung_results={},
        gate_result=None, skipped_rungs=[], bt={}, schedule_next={},
        eval_round_wall_sec=3_720.0, reason=reason, detail=None, random_wr=None,
    )


def _skip_event(result) -> dict[str, object]:
    harness = _make_coordinator()
    harness.coord.on_eval_round_complete(result)
    events = harness.sink.named("sealbot_wr_gate_skipped")
    assert len(events) == 1, f"exactly one skip event per delivered round; got {events}"
    return dict(events[0])


@pytest.mark.parametrize("reason", _REASONS, ids=[r.value for r in _REASONS])
def test_the_producer_carries_the_reason_the_gate_reads(reason: EvalBrokenReason) -> None:
    """`build_round_result` puts the typed reason on the mapping, for every member."""
    result = _broken_round(reason)
    assert result["eval_broken_reason"] is reason
    assert result["wr_sealbot"] is None, (
        "a broken round has no sealbot number — which is exactly why it used to be "
        "indistinguishable from a healthy round that had none either"
    )
    assert result["promoted"] is False, (
        "a broken round must never be promotable; `apply_gate_decision` also refuses it, and "
        "this is the producer-side half of that guarantee"
    )


@pytest.mark.parametrize("reason", _REASONS, ids=[r.value for r in _REASONS])
def test_a_broken_round_reaches_the_gate_and_is_NAMED_there(reason: EvalBrokenReason) -> None:
    """The gate is entered, the skip is counted, and the event says the round BROKE — over the
    whole taxonomy, so a guard written around one reason cannot satisfy it."""
    event = _skip_event(_broken_round(reason))
    assert event["reason"] == "eval_round_broken", (
        "a round that could not run must not be reported with the same reason string as a "
        f"healthy round carrying no sealbot number; got {event['reason']!r}"
    )
    assert event["eval_broken_reason"] is reason, (
        "the typed reason travels onto the gate's own event, so a consumer reads one key "
        f"rather than inferring the case from a string; got {event.get('eval_broken_reason')!r}"
    )
    assert event["skipped_total"] == 1


def test_the_gate_is_ENTERED_by_a_broken_round_and_not_skipped_before_it() -> None:
    """`checks` increments for a broken round: the gate ran and declined rather than being
    walked around, which would leave the failure invisible to the per-gate counters."""
    harness = _make_coordinator()
    before = dict(harness.coord._gate_stats["sealbot_wr_abort"])
    harness.coord.on_eval_round_complete(_broken_round(EvalBrokenReason.JOIN_TIMEOUT))
    after = dict(harness.coord._gate_stats["sealbot_wr_abort"])
    assert after["checks"] == before["checks"] + 1, "the gate must be entered by a broken round"
    assert after["skips"] == before["skips"] + 1, "and the skip must be counted (LAW-18)"
    assert after["fires"] == before["fires"], "a broken round must never FIRE the abort"


def test_a_HEALTHY_round_with_no_sealbot_number_still_says_wr_sealbot_absent() -> None:
    """A healthy metric-less round is unchanged; without this row the fix could be a blanket
    rename that destroys the very distinction it exists to draw."""
    clean = build_round_result(
        step=5000, round_id="r000002_5000", rungs_config=[], rung_results={},
        gate_result=None, skipped_rungs=[], bt={}, schedule_next={},
        eval_round_wall_sec=12.0, reason=None, detail=None, random_wr=None,
    )
    assert clean["eval_broken_reason"] is None
    event = _skip_event(clean)
    assert event["reason"] == "wr_sealbot_absent", (
        "a healthy round that carried no sealbot number is unchanged by this fix; got "
        f"{event['reason']!r}"
    )
    assert event["eval_broken_reason"] is None, (
        "the key is present on EVERY skip so a consumer reads one key rather than two shapes"
    )


def test_the_two_cases_are_DISTINGUISHABLE_on_the_stream_PLANTED_BREAK() -> None:
    """The two cases are distinguishable on the stream. PLANTED BREAK: stop the consumer
    reading `eval_broken_reason` and they collapse back into byte-identical events."""
    broken = _skip_event(_broken_round(EvalBrokenReason.JOIN_TIMEOUT))
    clean = build_round_result(
        step=5000, round_id="r000003_5000", rungs_config=[], rung_results={},
        gate_result=None, skipped_rungs=[], bt={}, schedule_next={},
        eval_round_wall_sec=12.0, reason=None, detail=None, random_wr=None,
    )
    healthy = _skip_event(clean)
    assert broken["reason"] != healthy["reason"], (
        "a broken round and a healthy metric-less round must not be the same observable on the "
        f"gate's own stream; both said {broken['reason']!r}"
    )


def test_a_mapping_WITHOUT_the_key_does_not_kill_the_poller() -> None:
    """A mapping without the key reads as clean here, unlike on the promotion path: a
    `KeyError` here would propagate out of the eval poller thread and convert a visible skip
    into a silent hang."""
    event = _skip_event({"step": 5000, "wr_sealbot": None})
    assert event["reason"] == "wr_sealbot_absent"
    assert event["eval_broken_reason"] is None


def test_every_reason_the_taxonomy_spells_is_covered_by_this_file() -> None:
    """The rows above are parametrised over the WHOLE taxonomy, read at run time. Set
    equality, never a count: a cardinality check is one rename away from meaningless."""
    assert set(_REASONS) == set(EvalBrokenReason), (
        "the reasons this file drives must be the taxonomy itself, read at run time"
    )


def test_the_round_budget_the_resit_measured_against_is_a_LIVE_config_fact() -> None:
    """The round budget the re-sit measured against is re-derived at HEAD, never quoted."""
    from pathlib import Path

    from mantis.config.loader import load_config

    repo = Path(__file__).resolve().parents[2]
    timeout = load_config(repo / "configs" / "run6.yaml").eval.round_timeout_sec
    assert timeout == 3600.0, (
        "the re-sit measured a ~62-minute round against a 3600 s budget; if run5's bound has "
        f"moved, F-RESIT-14's arithmetic needs re-deriving rather than re-quoting. Got {timeout}"
    )
