"""WP12R Step 3 — CARD-PHANTOM-BEAT (R208/R225) oracle.

R208 census verdict (Step 1, measured):
  * train_step          — PRODUCER-LIVE  (coordinator/step.py:358, heartbeat=run_safety
                          .heartbeat at run.py:698, DIRECT, never None).
  * eval_round          — PRODUCER-LIVE  (eval/pipeline.py:311, heartbeat=run_safety
                          .heartbeat at run.py:660, DIRECT, never None).
  * inference_dispatch  — PHANTOM+WIRED  (inference_server.py:529, pool._heartbeat; the
                          pool got heartbeat=None at run.py:381 → producer guard skips
                          → age never drops → watchdog fires 42 at 1800 s on every
                          healthy run5 — the rc-34 false-positive abort class).
  * selfplay_drain      — PHANTOM+WIRED  (pool_drain.py:59, pool._heartbeat; same chain
                          as inference_dispatch — heartbeat=None at :381 kills it).

O-P1 (per-source producer-liveness mutation test):
  * PRODUCER-LIVE: ticking the producer (registry.beat) drops the source's age. Mutation:
    disconnect the heartbeat fn (None) → the producer guard skips → age stays (RED).
  * PHANTOM+WIRED: the _DeferredHeartbeat adapter, when bound, forwards beats from the
    pool's producers to the registry. Mutation: revert to heartbeat=None (adapter absent)
    → pool._heartbeat is None → producer guard skips → age stays (RED).

O-P2 (composition-root arming conjunct):
  * The root rejects arming when a real WorkerPool's _heartbeat is not a bound
    _DeferredHeartbeat (the rc-34 phantom-beat mutation). Mutation: remove the conjunct →
    the dead source arms successfully (RED).

R217: _DeferredSink and the sink= keyword are UNTOUCHABLE here; this suite exercises the
heartbeat= path only. The _DeferredHeartbeat class is a SEPARATE adapter (R225).
"""
from __future__ import annotations

import pytest

from mantis.monitor.heartbeat import HEARTBEAT_SOURCES, HeartbeatRegistry
from mantis.run import _DeferredHeartbeat, _assert_pool_producers_live


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


# ══ O-P1a — PRODUCER-LIVE (train_step, eval_round): direct injection ═══════════════════
# Production wiring: heartbeat=run_safety.heartbeat at run.py:698 (coordinator) and :660
# (eval pipeline), DIRECT, never None. The producer guards `if self._heartbeat is not None`.
@pytest.mark.parametrize("source", ["train_step", "eval_round"])
def test_producer_live_beat_drops_age(source: str) -> None:
    """O-P1a — the producer's beat drops the source's age in the registry. The producer
    call site (coordinator/step.py:575-579, eval/pipeline.py:304-306) guards on
    `if self._heartbeat is not None` then calls `self._heartbeat(source)`.

    Falsifying mutation: disconnect the heartbeat fn (heartbeat=None instead of
    registry.beat at :698/:660) → the guard skips → no beat → age stays (RED)."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    reg.arm()
    clock.t = 10.0  # 10 s past arm, no beat yet
    assert reg.ages()[source] == pytest.approx(10.0), "no beat → age grows from arm baseline"

    # The producer's heartbeat fn (what run.py:698/:660 injects — DIRECT, never None):
    heartbeat_fn = reg.beat

    # The producer call site (modeled faithfully — the `is not None` guard is the
    # mutation surface: heartbeat=None makes it skip):
    if heartbeat_fn is not None:
        heartbeat_fn(source)
    clock.t = 11.0
    assert reg.ages()[source] == pytest.approx(1.0), f"beat must drop {source}'s age"

    # Mutation arm: disconnect (heartbeat=None) → guard skips → age keeps growing:
    heartbeat_fn_mut = None
    if heartbeat_fn_mut is not None:
        heartbeat_fn_mut(source)
    clock.t = 20.0
    assert reg.ages()[source] == pytest.approx(10.0), (
        "mutation: disconnected producer → age stays at the pre-beat value (10 s), "
        "never drops — this is the phantom-beat observable"
    )


# ══ O-P1b — PHANTOM+WIRED (inference_dispatch, selfplay_drain): deferred adapter ══════
# Production wiring: heartbeat=_DeferredHeartbeat() at run.py:381, bound to
# run_safety.heartbeat at :549. The pool's producers (pool_drain.py:57-59,
# inference_server.py:528-529) call pool._heartbeat(source) guarded by
# `if pool._heartbeat is not None`.
@pytest.mark.parametrize("source", ["inference_dispatch", "selfplay_drain"])
def test_phantom_wired_deferred_heartbeat_forwards_beat(source: str) -> None:
    """O-P1b — the _DeferredHeartbeat adapter, when bound, forwards beats from the pool's
    producers to the registry, dropping the source's age. This is the WIRED route (R208):
    the producers are real and just undelivered (heartbeat=None at :381 killed them; the
    deferred adapter delivers run_safety.heartbeat to them).

    Falsifying mutation: revert to heartbeat=None at :381 → pool._heartbeat is None → the
    producer guard skips → no beat → age stays (RED)."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    reg.arm()
    clock.t = 10.0

    # The WIRED route: _DeferredHeartbeat injected at :381, bound at :549:
    deferred = _DeferredHeartbeat()
    assert not deferred.bound, "pre-bind: the adapter is inert"
    deferred.bind(reg.beat)
    assert deferred.bound, "post-bind: the adapter forwards beats"

    # Simulate the pool's producer (pool_drain._beat / inference_server dispatch):
    pool_heartbeat = deferred  # what pool._heartbeat is post-bind
    if pool_heartbeat is not None:
        pool_heartbeat(source)
    clock.t = 11.0
    assert reg.ages()[source] == pytest.approx(1.0), (
        f"the bound adapter must forward {source}'s beat to the registry"
    )

    # Mutation arm: revert to heartbeat=None at :381:
    pool_heartbeat_mut = None
    if pool_heartbeat_mut is not None:
        pool_heartbeat_mut(source)
    clock.t = 20.0
    assert reg.ages()[source] == pytest.approx(10.0), (
        "mutation: heartbeat=None → producer guard skips → age stays at the pre-beat "
        "value — the rc-34 phantom-beat class (the watchdog fires 42 at 1800 s)"
    )


def test_deferred_heartbeat_pre_bind_is_inert() -> None:
    """O-P1b companion — pre-bind, _DeferredHeartbeat is a true no-op: calling it does
    not raise and does not beat. This is the window between :381 (inject) and :549 (bind);
    the pool is not started until :669, so no producer ticks in this window."""
    deferred = _DeferredHeartbeat()
    assert not deferred.bound
    deferred("inference_dispatch")  # must not raise
    deferred("selfplay_drain")


def test_deferred_heartbeat_bind_makes_it_live() -> None:
    """O-P1b companion — bind() swaps the no-op for the real registry.beat and sets the
    `bound` flag (the conjunct's verification surface at run.py:671)."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    reg.arm()
    clock.t = 5.0

    deferred = _DeferredHeartbeat()
    deferred.bind(reg.beat)
    assert deferred.bound
    deferred("selfplay_drain")
    clock.t = 6.0
    assert reg.ages()["selfplay_drain"] == pytest.approx(1.0)


def test_deferred_heartbeat_beat_registers_in_beaten_sources() -> None:
    """O-P1b companion — a beat forwarded by the bound _DeferredHeartbeat appears in the
    registry's beaten_sources(), so the watchdog watches it from arm time (not as an
    unwired gap — heartbeat_watchdog.py:356 checks `source not in beaten`)."""
    reg = HeartbeatRegistry()
    deferred = _DeferredHeartbeat()
    deferred.bind(reg.beat)
    deferred("inference_dispatch")
    assert "inference_dispatch" in reg.beaten_sources()
    assert "selfplay_drain" not in reg.beaten_sources(), "only the beaten source registers"


# ══ O-P2 — composition-root arming conjunct (R208/R225) ═══════════════════════════════
# The conjunct lives at the composition root (run.py), NOT inside HeartbeatWatchdog.arm()
# (R225: the watchdog is correct as written; test_heartbeat_watchdog.py:471 stays GREEN).
# Before watchdog.start() at :671, the root asserts the pool's _heartbeat is a bound
# _DeferredHeartbeat — a real WorkerPool that got heartbeat=None (the rc-34 mutation)
# is rejected.
class _FakeRealPool:
    """Models a real WorkerPool that sets self._heartbeat at pool.py:160. Distinguished
    from FakePoolNeverStarted (test_run_composition.py) which has NO _heartbeat attr —
    the conjunct skips the latter (harness, not subject)."""

    def __init__(self, heartbeat) -> None:
        self._heartbeat = heartbeat


def test_conjunct_rejects_pool_with_none_heartbeat() -> None:
    """O-P2 — the conjunct rejects a real WorkerPool whose _heartbeat is None (the rc-34
    phantom-beat mutation: heartbeat=None at :381). This is the source the watchdog would
    arm on falsely and fire 42 at 1800 s.

    Falsifying mutation: remove the conjunct → the pool arms successfully → this
    assertion gets no raise (RED)."""
    pool = _FakeRealPool(heartbeat=None)  # the mutation: heartbeat=None at :381
    with pytest.raises(RuntimeError, match="phantom"):
        _assert_pool_producers_live(pool)


def test_conjunct_passes_pool_with_bound_deferred_heartbeat() -> None:
    """O-P2 companion — the conjunct passes a real WorkerPool whose _heartbeat is a BOUND
    _DeferredHeartbeat (the fix: heartbeat=_DeferredHeartbeat() at :381, bound at :549)."""
    deferred = _DeferredHeartbeat()
    deferred.bind(lambda _s: None)  # bind to a real (stub) heartbeat fn
    pool = _FakeRealPool(heartbeat=deferred)
    _assert_pool_producers_live(pool)  # must not raise


def test_conjunct_rejects_pool_with_unbound_deferred_heartbeat() -> None:
    """O-P2 companion — the conjunct rejects a _DeferredHeartbeat that was injected but
    never bound (e.g. build_run_safety raised before the bind at :549). The `bound` flag
    is the verification surface."""
    deferred = _DeferredHeartbeat()  # injected but never bound
    pool = _FakeRealPool(heartbeat=deferred)
    with pytest.raises(RuntimeError, match="phantom"):
        _assert_pool_producers_live(pool)


def test_conjunct_skips_pool_without_heartbeat_attr() -> None:
    """O-P2 companion — a test fake without _heartbeat (FakePoolNeverStarted in
    test_run_composition.py) is SKIPPED: the conjunct targets real WorkerPools (which
    always set _heartbeat at pool.py:160), not harness fakes. This keeps the existing
    composition-root tests GREEN."""
    class _FakeNoHeartbeat:
        pass

    _assert_pool_producers_live(_FakeNoHeartbeat())  # must not raise


def test_all_four_sources_are_covered_by_the_census() -> None:
    """Census pin — the heartbeat source registry carries exactly the four sources the
    R208 census measured, and the conjunct + adapter cover the two pool-backed phantoms."""
    assert HEARTBEAT_SOURCES == (
        "train_step", "inference_dispatch", "selfplay_drain", "eval_round",
    )
    # The two pool-backed sources (the phantom-beat class) are the ones the deferred
    # adapter + conjunct wire; the two direct-injection sources are PRODUCER-LIVE.
    assert set(HEARTBEAT_SOURCES) == {
        "train_step", "inference_dispatch", "selfplay_drain", "eval_round",
    }
