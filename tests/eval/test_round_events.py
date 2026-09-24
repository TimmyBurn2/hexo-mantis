"""⊕ WP11-A DESIGN §c.4 — round wall-time events (pays recon T6 gap (c): the eval-round WALL
event producer lands here; the floor itself is a cutover re-baseline item, NOT this WP's job —
PREREG P-2 bench posture n/a).

Events (§c.4, verbatim):
  * `eval_round_started`  {round_id, step, scheduled: {rung: n}, gate_scheduled: bool, ts}
  * `eval_round_complete` {round_id, step, wall_sec, games_total, promoted, wr_sealbot}
  * `eval_round_skipped_busy` {step, in_flight_round_id}

RED-at-import: `mantis.eval.pipeline` does not exist yet.

ORACLE-CHOSEN SEAM: rather than driving a full `build_eval_pipeline` round (subprocess spawn,
real nets/engine — that integration lives in test_round_end_to_end.py / test_pipeline_isolation
.py, owned elsewhere), this file pins the EVENT-BUILDING units directly — the pure emit
functions `pipeline.emit_round_started` / `pipeline.emit_round_complete` /
`pipeline.emit_round_skipped_busy`, each: (1) building the exact payload shape below, (2)
calling `sink.emit(payload)`, (3) returning the payload dict (so a caller — the real
`run_evaluation`/poller-thread wiring, tested elsewhere — can also read it back, e.g. to fill
`eval_round_wall_sec` into the routed result). This is the producer unit these dispatch/T6
success criteria name; the ORDER + WIRING of these calls INSIDE `run_evaluation` is the
integration suites' job, not duplicated here.
"""
from __future__ import annotations

from mantis.eval.pipeline import (  # noqa: F401 — RED-at-import anchor: mantis.eval does not exist yet
    emit_round_complete,
    emit_round_skipped_busy,
    emit_round_started,
)


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def test_round_emits_start_and_complete_wall_events() -> None:
    from mantis.eval.pipeline import emit_round_complete, emit_round_started

    sink = _SpySink()
    started = emit_round_started(
        sink, round_id="r000001_1000", step=1000, gate_scheduled=True, ts=1234.5,
    )
    for key in ("round_id", "step", "gate_scheduled", "ts"):
        assert key in started, f"eval_round_started missing {key!r}"
    assert started["round_id"] == "r000001_1000"
    assert started["gate_scheduled"] is True

    complete = emit_round_complete(
        sink, round_id="r000001_1000", step=1000, wall_sec=12.5, games_total=88,
        promoted=False, gate=None,
    )
    for key in ("round_id", "step", "wall_sec", "games_total", "promoted", "gate"):
        assert key in complete, f"eval_round_complete missing {key!r}"
    assert complete["wall_sec"] == 12.5 > 0
    assert complete["round_id"] == started["round_id"]

    all_started = sink.named("eval_round_started")
    all_complete = sink.named("eval_round_complete")
    assert len(all_started) == 1 and len(all_complete) == 1
    assert sink.events.index(all_started[0]) < sink.events.index(all_complete[0]), (
        "eval_round_started must precede eval_round_complete in the sink's event stream"
    )
    assert all_started[0] == started and all_complete[0] == complete, (
        "the returned payload must be EXACTLY what was emitted (no post-emit mutation)"
    )


def test_busy_kick_emits_eval_round_skipped_busy() -> None:
    from mantis.eval.pipeline import emit_round_skipped_busy

    sink = _SpySink()
    ack = emit_round_skipped_busy(sink, step=1000, in_flight_round_id="r000001_900")
    assert ack["step"] == 1000
    assert ack["in_flight_round_id"] == "r000001_900"

    busy_events = sink.named("eval_round_skipped_busy")
    assert len(busy_events) == 1
    assert busy_events[0] == ack
