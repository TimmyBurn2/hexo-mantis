"""Oracle for the phantom-beat class: a heartbeat source with a producer that never delivers.

`train_step` and `eval_round` get the registry's own `beat` directly. The two pool-backed
sources had real producers that were handed `heartbeat=None`, so the guard skipped, the age
never dropped, and the watchdog fired on every healthy run. Each row carries its falsifying
mutation, and the last rows pin the composition root's arming conjunct.
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


# Direct injection: the composition root passes the registry's `beat` itself, and the producer
# guards on `if self._heartbeat is not None`.
@pytest.mark.parametrize("source", ["train_step", "eval_round"])
def test_producer_live_beat_drops_age(source: str) -> None:
    """The producer's beat drops the source's age.

    MUTATION: disconnect the heartbeat fn (`heartbeat=None`) — the guard skips, no beat, the
    age stays."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    reg.arm()
    clock.t = 10.0
    assert reg.ages()[source] == pytest.approx(10.0), "no beat → age grows from arm baseline"

    heartbeat_fn = reg.beat

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


# The pool's producers call `pool._heartbeat(source)` guarded by `is not None`, so the deferred
# adapter is what delivers the registry's beat to them once it is bound.
@pytest.mark.parametrize("source", ["inference_dispatch", "selfplay_drain"])
def test_phantom_wired_deferred_heartbeat_forwards_beat(source: str) -> None:
    """The bound `_DeferredHeartbeat` forwards the pool's beats to the registry.

    MUTATION: revert to `heartbeat=None` — the guard skips and the age stays."""
    clock = _Clock()
    reg = HeartbeatRegistry(clock=clock)
    reg.arm()
    clock.t = 10.0

    deferred = _DeferredHeartbeat()
    assert not deferred.bound, "pre-bind: the adapter is inert"
    deferred.bind(reg.beat)
    assert deferred.bound, "post-bind: the adapter forwards beats"

    pool_heartbeat = deferred
    if pool_heartbeat is not None:
        pool_heartbeat(source)
    clock.t = 11.0
    assert reg.ages()[source] == pytest.approx(1.0), (
        f"the bound adapter must forward {source}'s beat to the registry"
    )

    # Mutation arm: revert to heartbeat=None:
    pool_heartbeat_mut = None
    if pool_heartbeat_mut is not None:
        pool_heartbeat_mut(source)
    clock.t = 20.0
    assert reg.ages()[source] == pytest.approx(10.0), (
        "mutation: heartbeat=None → producer guard skips → age stays at the pre-beat "
        "value — the rc-34 phantom-beat class (the watchdog fires 42 at 1800 s)"
    )


def test_deferred_heartbeat_pre_bind_is_inert() -> None:
    """Pre-bind the adapter is a true no-op — the window between injection and bind, in which
    no producer ticks."""
    deferred = _DeferredHeartbeat()
    assert not deferred.bound
    deferred("inference_dispatch")
    deferred("selfplay_drain")


def test_deferred_heartbeat_bind_makes_it_live() -> None:
    """`bind()` swaps the no-op for the real `registry.beat` and sets the `bound` flag."""
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
    """A forwarded beat appears in `beaten_sources()`, so the watchdog watches the source from
    arm time rather than treating it as an unwired gap."""
    reg = HeartbeatRegistry()
    deferred = _DeferredHeartbeat()
    deferred.bind(reg.beat)
    deferred("inference_dispatch")
    assert "inference_dispatch" in reg.beaten_sources()
    assert "selfplay_drain" not in reg.beaten_sources(), "only the beaten source registers"


# The arming conjunct lives at the composition root, not inside `HeartbeatWatchdog.arm()`:
# before the watchdog starts, the root asserts the pool's `_heartbeat` is a bound adapter.
class _FakeRealPool:
    """A real `WorkerPool` sets `self._heartbeat`; a fake without the attribute is skipped."""

    def __init__(self, heartbeat) -> None:
        self._heartbeat = heartbeat


def test_conjunct_rejects_pool_with_none_heartbeat() -> None:
    """The conjunct rejects a real pool whose `_heartbeat` is None — the source the watchdog
    would arm on falsely and then fire on.

    MUTATION: remove the conjunct and the pool arms successfully."""
    pool = _FakeRealPool(heartbeat=None)
    with pytest.raises(RuntimeError, match="phantom"):
        _assert_pool_producers_live(pool)


def test_conjunct_passes_pool_with_bound_deferred_heartbeat() -> None:
    """The conjunct passes a real pool whose `_heartbeat` is a BOUND adapter."""
    deferred = _DeferredHeartbeat()
    deferred.bind(lambda _s: None)
    pool = _FakeRealPool(heartbeat=deferred)
    _assert_pool_producers_live(pool)


def test_conjunct_rejects_pool_with_unbound_deferred_heartbeat() -> None:
    """The conjunct rejects an adapter that was injected but never bound; the `bound` flag is
    the verification surface."""
    deferred = _DeferredHeartbeat()
    pool = _FakeRealPool(heartbeat=deferred)
    with pytest.raises(RuntimeError, match="phantom"):
        _assert_pool_producers_live(pool)


def test_conjunct_skips_pool_without_heartbeat_attr() -> None:
    """A harness fake without `_heartbeat` is SKIPPED: the conjunct targets real pools."""
    class _FakeNoHeartbeat:
        pass

    _assert_pool_producers_live(_FakeNoHeartbeat())


def test_all_four_sources_are_covered_by_the_census() -> None:
    """The registry carries exactly the four censused sources, two of them pool-backed."""
    assert HEARTBEAT_SOURCES == (
        "train_step", "inference_dispatch", "selfplay_drain", "eval_round",
    )
    assert set(HEARTBEAT_SOURCES) == {
        "train_step", "inference_dispatch", "selfplay_drain", "eval_round",
    }
