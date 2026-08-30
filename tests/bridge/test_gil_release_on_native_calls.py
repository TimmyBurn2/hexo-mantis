"""PERF-TRANCHE-1 A5/B2 — the two long native calls run with the GIL RELEASED.

The PERF-BASELINE ledger measured self-play and training in separate drives, so it could
not see what happens when both run in one process. The PERF-TRANCHE-1 M-1 joint drive did:
`HexgBuffer.sample_graph_batch` held the GIL for **99.9 %** of every ring-sample window and
the in-process inference-server thread served **zero** graphs across 16.85 s of them,
against 79.9 requests/s outside. Twelve self-play workers keep building leaves through that
stall; nothing answers them.

WHAT THESE TESTS CAN AND CANNOT SEE. CPython exposes no "is the GIL held" predicate, so the
only oracle available from Python is whether another Python thread makes progress while the
native call runs. That is a TIMING observation, and it is framed to have an enormous margin
rather than a tight one: a GIL-holding call lets the observer thread advance **not at all**,
so the assertion is "advanced substantially", not "advanced by N". The authoritative witness
for these fixes is the joint drive on the box; these tests are the standing regression guard
that keeps the release from being quietly removed.
"""
from __future__ import annotations

import threading
import time

import pytest

from mantis import _engine

# A ring whose sample is long enough that a GIL hold would be unmistakable, and short
# enough to belong in the default tier. Measured, not guessed: 24 stones x 32 sampled
# graphs runs on the order of a tenth of a second.
_STONES = 24
_RECORDS = 256
_CAPACITY = 512
_BATCH = 32
#: The observer must advance at least this many times during the native call. A held GIL
#: yields 0; a released one yields orders of magnitude more than this floor.
_MIN_OBSERVER_TICKS = 50


def _mk_ring() -> object:
    hb = _engine.HexgBuffer(_CAPACITY, "gnn_axis_v1", 128)
    for i in range(_RECORDS):
        stones = [(q, (q % 3) - 1, 1 if q % 2 == 0 else -1) for q in range(_STONES)]
        hb.push_graph_position(
            stones, [(-1, 0, 0.6), (_STONES, 0, 0.4)], 1, 30, 2 + (i % 50),
            True, 1.0, True, 10 + i,
        )
    return hb


class _Observer(threading.Thread):
    """A second Python thread that only counts. It cannot advance while the GIL is held."""

    def __init__(self) -> None:
        # NOT `_stop`: that name shadows `threading.Thread._stop`, which the runtime calls.
        super().__init__(daemon=True, name="gil-observer")
        self.ticks = 0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.ticks += 1
            time.sleep(0)  # yield: give the GIL up so this measures availability, not greed

    def stop(self) -> None:
        self._halt.set()


def _ticks_during(call) -> tuple[int, float]:
    """`(observer ticks, elapsed ms)` across one native call."""
    obs = _Observer()
    obs.start()
    time.sleep(0.02)  # let the observer reach steady state before the call
    before = obs.ticks
    t0 = time.perf_counter()
    call()
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    ticks = obs.ticks - before
    obs.stop()
    obs.join(timeout=2.0)
    return ticks, elapsed_ms


def test_sample_graph_batch_releases_the_gil() -> None:
    """B2 — the trainer's longest call is GIL-free.

    Unreleased, this call is a total serve stall for its whole duration; it is 1 386 ms of a
    2 769 ms step at run5 shape, so the stall is not a tail case.
    """
    hb = _mk_ring()
    hb.sample_graph_batch(_BATCH)  # warm: first call pays one-time build costs
    ticks, elapsed_ms = _ticks_during(lambda: hb.sample_graph_batch(_BATCH))
    if elapsed_ms < 20.0:
        pytest.skip(
            f"sample took {elapsed_ms:.1f} ms on this host — too short for the observer to "
            "resolve; the release is witnessed by the joint drive, not by a race"
        )
    assert ticks >= _MIN_OBSERVER_TICKS, (
        f"a second Python thread advanced {ticks} times during a {elapsed_ms:.1f} ms "
        "sample_graph_batch. A GIL-holding call yields 0 — this reads as the GIL being "
        "held across the ring sample (B2 regressed)"
    )


def test_next_graph_batch_fuse_releases_the_gil() -> None:
    """A5 — the inference-path fuse is GIL-free, like the pop it follows.

    The mock producer is the only Python-reachable way to put graphs on this queue, so the
    batch is sized until the fuse window is resolvable rather than left at a handful.
    """
    spec = _engine.RegistrySpec.from_registry("gnn_axis_v1")
    batcher = _engine.InferenceBatcher(encoding_spec=spec)
    # 1024 mock leaves, not a handful: the fuse is the window under test and a batch too
    # small to resolve would leave this test SKIPPING on a fast host — a green that means
    # nothing, which is the whole defect this file exists to prevent. Sized by measurement,
    # not by guess: at 256 the pop+fuse ran 16.7 ms on an idle host, inside the 20 ms
    # resolution floor below; at 1024 it runs ~43 ms, clear of it with room for a host
    # faster than this one.
    n_mock = 1024
    batcher.spawn_mock_graph_games(n_mock)
    for _ in range(400):
        if batcher.has_pending_graph_requests():
            break
        time.sleep(0.005)
    time.sleep(0.3)  # let the spawned submitters land their whole batch on the queue
    ticks, elapsed_ms = _ticks_during(lambda: batcher.next_graph_batch(n_mock, 200))
    batcher.close()
    if elapsed_ms < 20.0:
        pytest.skip(
            f"pop+fuse took {elapsed_ms:.1f} ms — too short to resolve; the pop's own "
            "`py.detach` already dominates this window"
        )
    assert ticks >= _MIN_OBSERVER_TICKS, (
        f"a second Python thread advanced {ticks} times during a {elapsed_ms:.1f} ms "
        "next_graph_batch (A5 regressed)"
    )


# ── The half the two tests above could not see ────────────────────────────────────────────
# `_Observer` only counts; it never TOUCHES the ring. So it witnessed the GIL release while
# staying blind to what the release exposed: `sample_graph_batch` took pyo3's `PyRefMut` for
# its whole body and held it across the GIL-free window, so any other Python thread that
# touched the same buffer was refused with `RuntimeError: Already mutably borrowed`. The sole
# self-play producer does exactly that — `pool_drain.run_stats_loop` reads `.size` — so the
# guard logged `selfplay_producer_died` and the run ended. Measured on dev at 2026-08-30:
# one successful `.size` read, then the refusal. These tests are the LAW-07 producer for the
# repair: exclusion moved to a mutex with every pymethod on `&self`, so a contender WAITS.


class _RingToucher(threading.Thread):
    """A second thread that READS the ring, which is what the observer above never did."""

    def __init__(self, ring: object) -> None:
        super().__init__(daemon=True, name="ring-toucher")
        self._ring = ring
        self._halt = threading.Event()
        self.reads = 0
        self.error: BaseException | None = None

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                _ = self._ring.size
                _ = self._ring.capacity
                self.reads += 1
            except BaseException as exc:  # noqa: BLE001 - the refusal is the whole subject
                self.error = exc
                return
            time.sleep(0)

    def stop(self) -> None:
        self._halt.set()


def _read_ring_during_samples(ring: object, n_samples: int) -> _RingToucher:
    toucher = _RingToucher(ring)
    toucher.start()
    time.sleep(0.02)
    for _ in range(n_samples):
        ring.sample_graph_batch(_BATCH)
    toucher.stop()
    toucher.join(timeout=5.0)
    return toucher


def test_a_concurrent_reader_is_NOT_refused_while_the_ring_is_sampled() -> None:
    """The regression guard. Before the repair this raised on the reader's second read."""
    ring = _mk_ring()
    toucher = _read_ring_during_samples(ring, n_samples=12)
    assert toucher.error is None, (
        "a second Python thread reading the ring during sample_graph_batch was refused "
        f"({type(toucher.error).__name__}: {toucher.error}). 'Already mutably borrowed' here "
        "means a pymethod is holding pyo3's PyRefMut across the GIL release again; that "
        "refusal kills the sole self-play producer, which reads .size on its stats loop"
    )
    assert toucher.reads > 0, "the toucher never read the ring; the guard proves nothing"


def test_the_concurrent_reader_actually_OVERLAPS_the_sample_window() -> None:
    """Vacuity control for the guard above.

    If the reader finished before the first sample started, a refusal could not have fired
    and a green would mean nothing. This pins that reads land while sampling is in flight.
    """
    ring = _mk_ring()
    toucher = _read_ring_during_samples(ring, n_samples=12)
    assert toucher.error is None
    assert toucher.reads > _MIN_OBSERVER_TICKS, (
        f"only {toucher.reads} ring reads landed across 12 samples — too few to have "
        "overlapped the sample windows, so the refusal guard above is not being exercised"
    )


def test_the_ring_reports_no_poisoned_lock_recoveries() -> None:
    """LAW-18: the mutex that replaced the borrow is observable, not silent. A non-zero count
    means a panic happened under the guard and the seam kept going."""
    ring = _mk_ring()
    _read_ring_during_samples(ring, n_samples=4)
    assert ring.lock_recoveries == 0
