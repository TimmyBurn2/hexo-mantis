"""F-RESIT-14's gate hole: a round that BROKE must reach LAW-15's gate AS a broken round.

**THE HOLE, measured at the 2026-08-27 re-calibration re-sit.** Every in-run eval round of the
90-minute validation burst ended `eval_broken` — the round-progress budget
(`eval.round_timeout_sec`) escalating through `_escalate_and_finalize`. So the promotion gate
never fired for the life of the burst. What `step.py::on_eval_round_complete` emitted each time
was `sealbot_wr_gate_skipped` with `reason: "wr_sealbot_absent"` — **byte-identical to what a
perfectly healthy round that happened to carry no sealbot number emits.**

A round that COULD NOT RUN is not a round that ran without a number. Read off the event stream,
the two were the same fact, and a promotion bar reported as merely un-evidenced when the evidence
PATH has failed is LAW-15's gate with a door in it: nothing in the stream distinguishes "the bar
was not met" from "the bar was never measured".

**What this file pins.** The gate is ENTERED on a broken round (it always was — `checks`
increments before any branch), and it now NAMES the failure: `reason: "eval_round_broken"` with
the typed reason carried on the payload. The healthy-but-metric-less arm is unchanged, which is
the half that makes the first assertion mean something — a blanket rename would satisfy "broken
rounds say something different" while destroying the distinction it exists to draw.

**Why the producer half is here too.** The gate can only read `eval_broken_reason` if the round
result carries it, and the two live in different modules. `build_round_result` is driven directly
so this file witnesses the WHOLE path — producer to consumer — rather than asserting a key it
also invents (R69: a measurement travels with its mechanism).

**NOT claimed here, and it is the larger half of F-RESIT-14.** Why the round exceeded its budget
at all is a WORKLOAD fact, not a code one — the box carried no bootstrap checkpoint, so the gate
block played two randomly-initialised players and no game ended early. Fixing that is a config or
prereg decision and is the architect's; this file closes the reporting hole, which is the part
that is engine-side and was silently wrong regardless of why any given round broke.
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
    """A minimal coordinator, harness PRIVATE to this file — the house convention.

    Not imported from a sibling test module: `tests` is not a package (R5 — single collection
    root, no package named `tests` below it, zero `sys.path` mutation), so a
    `from tests.train... import` resolves under one pytest invocation and raises
    `ModuleNotFoundError` under another. This file was written that way first and the count
    gate's `uv run pytest --collect-only` caught it while a direct `python -m pytest` had not —
    which is precisely the unreproducible-collection failure R5 exists to close.

    The config is DERIVED from the production builder (`mantis.run._step_coordinator_config`)
    rather than hand-written as a kwarg census, so a new coordinator knob costs this file no
    edit — the same discipline the frozen `tests/eval/test_wr_sealbot_handshake.py` states."""
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

#: Every reason the taxonomy spells. The gate must name ALL of them, not the one that happened to
#: be measured — a guard written around `join_timeout` alone would be silent on the next reason,
#: and `ROUND_COMPLETION_ERROR` is the one that fires when the pipeline's own catch-all trips.
_REASONS = tuple(EvalBrokenReason)


def _broken_round(reason: EvalBrokenReason, *, step: int = 5000) -> dict[str, object]:
    """A REAL broken round result, built by the production producer.

    Driven through `build_round_result` rather than hand-written: the gate reads
    `eval_broken_reason`, and a hand-built mapping would let this file pass while the producer
    stopped emitting the key the consumer depends on."""
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


# ── the producer half ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("reason", _REASONS, ids=[r.value for r in _REASONS])
def test_the_producer_carries_the_reason_the_gate_reads(reason: EvalBrokenReason) -> None:
    """`build_round_result` puts the typed reason on the mapping, for every member.

    The gate's whole ability to tell a broken round from a quiet one rests on this key being
    present and truthful on the real payload."""
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


# ── the consumer half: the timeout path reaching the gate ────────────────────────────────
@pytest.mark.parametrize("reason", _REASONS, ids=[r.value for r in _REASONS])
def test_a_broken_round_reaches_the_gate_and_is_NAMED_there(reason: EvalBrokenReason) -> None:
    """The gate is entered, the skip is counted, and the event says the round BROKE.

    `join_timeout` is the member the re-sit measured — the round-progress budget escalating —
    and it is parametrized alongside the rest so this row cannot be satisfied by a guard written
    around one reason."""
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
    """`checks` increments for a broken round: the gate ran and declined, it was not bypassed.

    This is the literal reading of "a timeout exit must FIRE the gate, never walk around it" —
    a round that never reached the gate would leave `checks` untouched and the failure invisible
    to the per-gate counters `monitor_gates` publishes."""
    harness = _make_coordinator()
    before = dict(harness.coord._gate_stats["sealbot_wr_abort"])
    harness.coord.on_eval_round_complete(_broken_round(EvalBrokenReason.JOIN_TIMEOUT))
    after = dict(harness.coord._gate_stats["sealbot_wr_abort"])
    assert after["checks"] == before["checks"] + 1, "the gate must be entered by a broken round"
    assert after["skips"] == before["skips"] + 1, "and the skip must be counted (LAW-18)"
    assert after["fires"] == before["fires"], "a broken round must never FIRE the abort"


# ── the half that makes the first one mean something ─────────────────────────────────────
def test_a_HEALTHY_round_with_no_sealbot_number_still_says_wr_sealbot_absent() -> None:
    """The distinction, from the other side. Without this row the fix could be a blanket rename
    — every skip newly called `eval_round_broken` — which would satisfy "broken rounds say
    something different" while destroying the very distinction the change exists to draw."""
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
    """The planted break, stated as the property rather than as a mutation.

    Before the fix these two events were byte-identical in `reason`, which is precisely why the
    re-sit's burst could end every round broken with nothing in the stream saying so. If the
    consumer stops reading `eval_broken_reason` the two collapse back together and this row is
    the one that reds — no other assertion in this file compares them."""
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
    """`.get`, not a subscript — and the asymmetry with `apply_gate_decision` is deliberate.

    There a subscript is right: on the PROMOTION path an absent reason must never read as clean
    (R152/LAW-11). Here a `KeyError` would propagate out of the eval poller thread, which is the
    F1 failure mode the whole pipeline is built against — a visible skip converted into a silent
    hang. The terminal and hand-built routes legitimately deliver mappings that predate the key,
    and the FROZEN `tests/eval/test_wr_sealbot_handshake.py` drives exactly such a mapping."""
    event = _skip_event({"step": 5000, "wr_sealbot": None})
    assert event["reason"] == "wr_sealbot_absent"
    assert event["eval_broken_reason"] is None


def test_every_reason_the_taxonomy_spells_is_covered_by_this_file() -> None:
    """The parametrisations above must be over the WHOLE taxonomy, not a snapshot of it.

    A member added later — the round-progress timeout wants one of its own (see this file's
    closing note) — must arrive already covered rather than silently outside the rows that
    claim to cover every reason. Set equality, never a count: a cardinality check is a rename
    away from meaningless, which is the derive-or-delete lesson this repo keeps re-learning."""
    assert set(_REASONS) == set(EvalBrokenReason), (
        "the reasons this file drives must be the taxonomy itself, read at run time"
    )


def test_the_round_budget_the_resit_measured_against_is_a_LIVE_config_fact() -> None:
    """F-RESIT-14's arithmetic — a round killed at ~62 min against `eval.round_timeout_sec` —
    rests on a bound this row re-derives at HEAD rather than quoting from a document."""
    from pathlib import Path

    from mantis.config.loader import load_config

    repo = Path(__file__).resolve().parents[2]
    timeout = load_config(repo / "configs" / "run5.yaml").eval.round_timeout_sec
    assert timeout == 3600.0, (
        "the re-sit measured a ~62-minute round against a 3600 s budget; if run5's bound has "
        f"moved, F-RESIT-14's arithmetic needs re-deriving rather than re-quoting. Got {timeout}"
    )
