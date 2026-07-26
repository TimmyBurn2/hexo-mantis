"""⊕ WPUF Phase U ORACLE — O-U1: continuous, unconditional actor sync (DESIGN_U §1.1/§8).

RED-at-import until IMPL lands `mantis.train.actor_sync.ActorSync` (the sync engine).
Byte-frozen through IMPL; an oracle contradicting DESIGN is adjudicated, never satisfied.

The headline law (R49): the actor's weights track the learner on a step-modulo cadence,
with NO gate, promotion, deploy or eval object anywhere in the loop. There is deliberately
no gate object in ANY harness below — "the gate fails forever" is vacuously true here,
which per DESIGN §0 is also HEAD's production reality on the compose_run path.

Every drive is a direct `maybe_sync` call — zero threads, zero joins, zero sleeps.
"""
from __future__ import annotations

import pytest

from mantis.train.actor_sync import ActorSync  # RED-at-import anchor (module does not exist yet)

_EVENT_KEYS = {
    "event", "step", "actor_ckpt_step", "lag_steps_pre_sync",
    "cadence_steps", "sync_count", "duration_ms",
}


class _Learner:
    """Mutable learner-step cell; `step_fn` reads it live."""

    def __init__(self, step: int = 0) -> None:
        self.step = step


class _SyncTargetSpy:
    """ActorSyncTarget-shaped spy: records both pushes, optionally rigged to raise."""

    def __init__(self, *, raise_on: str | None = None) -> None:
        self.sync_payloads: list = []
        self.step_calls: list[int] = []
        self._raise_on = raise_on

    def sync_inference_weights(self, state_dict) -> None:
        if self._raise_on == "sync_inference_weights":
            raise RuntimeError("rigged: weight push failed")
        self.sync_payloads.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        if self._raise_on == "update_checkpoint_step":
            raise RuntimeError("rigged: step push failed")
        self.step_calls.append(int(step))


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


def _engine(*, target, learner, cadence_steps, sink=None, state_dict=None):
    sd = state_dict if state_dict is not None else {"w": 0}
    return ActorSync(
        target=target,
        state_dict_fn=lambda: sd,
        step_fn=lambda: int(learner.step),
        cadence_steps=cadence_steps,
        sink=sink if sink is not None else _SpySink(),
        run_id="oracle_u1",
    )


def test_first_maybe_sync_call_syncs_unconditionally() -> None:
    """DESIGN §1.1 cadence semantics: the FIRST call pushes even off-boundary, so the
    invariant is established without assuming boot parity between pool and trainer."""
    target = _SyncTargetSpy()
    learner = _Learner(step=3)
    engine = _engine(target=target, learner=learner, cadence_steps=8)

    synced = engine.maybe_sync(3)  # 3 % 8 != 0 — still must push

    assert synced is True, "the first maybe_sync call must sync unconditionally"
    assert len(target.sync_payloads) == 1, "exactly one weight push on the first call"
    assert target.step_calls == [3], "the pushed step is the driving train_step"
    assert engine.actor_ckpt_step() == 3, "actor_ckpt_step must record the synced step"


def test_actor_tracks_learner_within_cadence_when_gate_never_passes() -> None:
    """O-U1 headline: 50 learner steps, cadence 8, no gate/eval object in existence —
    the actor's recorded step never lags the learner by the cadence bound or more, and
    the target received a push at exactly the first call plus every modulo boundary."""
    cadence = 8
    target = _SyncTargetSpy()
    learner = _Learner(step=0)
    engine = _engine(target=target, learner=learner, cadence_steps=cadence)

    for k in range(1, 51):
        learner.step = k
        synced = engine.maybe_sync(k)
        lag = learner.step - engine.actor_ckpt_step()
        assert lag <= cadence, f"step {k}: lag {lag} exceeds the cadence bound {cadence}"
        if k % cadence == 0:
            assert lag < cadence, f"boundary step {k}: lag {lag} not inside the bound"
            assert synced is True, f"boundary step {k} must sync"

    assert target.step_calls == [1, 8, 16, 24, 32, 40, 48], (
        "pushes must land at exactly the unconditional first call + every cadence boundary"
    )


@pytest.mark.parametrize("failing_method",
                         ["sync_inference_weights", "update_checkpoint_step"])
def test_actor_step_advances_only_after_successful_push(failing_method: str) -> None:
    """Producer honesty (DESIGN §3, R4): `_actor_step` is written ONLY after both target
    calls return. A raising target leaves the recorded step un-advanced — the lag
    invariant then reports the truth — and the exception PROPAGATES (LAW-14: no swallow)."""
    target = _SyncTargetSpy(raise_on=failing_method)
    learner = _Learner(step=0)
    engine = _engine(target=target, learner=learner, cadence_steps=4)
    before = engine.actor_ckpt_step()

    learner.step = 4
    with pytest.raises(RuntimeError):
        engine.maybe_sync(4)

    assert engine.actor_ckpt_step() == before, (
        "a failed push must never advance actor_ckpt_step — the recorded step would lie"
    )
    if failing_method == "sync_inference_weights":
        assert target.step_calls == [], (
            "the step push must not fire when the weight push already failed (order: "
            "weights, then step, then record — DESIGN §1.1)"
        )


def test_sync_pushes_state_dict_and_step_together() -> None:
    """One sync = BOTH target methods: the state_dict by identity from `state_dict_fn`,
    the step from the driving train_step."""
    sd = {"layer.weight": 1}
    target = _SyncTargetSpy()
    learner = _Learner(step=4)
    engine = _engine(target=target, learner=learner, cadence_steps=4, state_dict=sd)

    engine.maybe_sync(4)

    assert len(target.sync_payloads) == 1 and len(target.step_calls) == 1, (
        "one sync must invoke both sync_inference_weights and update_checkpoint_step"
    )
    assert target.sync_payloads[0] is sd, (
        "the pushed state_dict must be the exact object state_dict_fn returned"
    )
    assert target.step_calls == [4]


def test_actor_sync_event_carries_lever_fire_rate_fields() -> None:
    """LAW-18: the cadence is a lever under test — every sync emits an `actor_sync`
    event with exactly the DESIGN §5 payload keys, `sync_count` monotonic."""
    sink = _SpySink()
    target = _SyncTargetSpy()
    learner = _Learner(step=0)
    engine = _engine(target=target, learner=learner, cadence_steps=1, sink=sink)

    for k in (1, 2):
        learner.step = k
        engine.maybe_sync(k)

    events = sink.named("actor_sync")
    assert len(events) == 2, "every successful sync must emit exactly one actor_sync event"
    for event in events:
        assert set(event) == _EVENT_KEYS, (
            f"actor_sync payload must carry exactly the §5 keys; got {sorted(event)}"
        )
        assert event["cadence_steps"] == 1
    assert [e["sync_count"] for e in events] == [1, 2], "sync_count must be monotonic"
    assert [e["step"] for e in events] == [1, 2]
    assert [e["actor_ckpt_step"] for e in events] == [1, 2]
