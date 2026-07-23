"""Suite C remainder + G-18 — drain arms a ONE-iteration harness cannot see.

IMPL-written (non-⊕) per DESIGN §b / PREREG §3. The ⊕ Suite C golden
(`test_pool_drain_parity.py`) runs the real drain body for exactly ONE iteration with a
recording sink, which is the right shape for byte-parity but is structurally blind to two
things:

  * **C-09** — whether the 5-second `system_stats` cadence carries its "last emitted"
    marker ACROSS iterations. The marker is a local of the loop; an implementation that
    hoists it onto the pool and forgets to update it emits every tick, and a
    one-iteration harness sees exactly one emission either way. The clock here is chosen
    so a forgotten update produces a SECOND emission.
  * **C-10** — whether the no-sink default path runs at all. The ⊕ harness always injects
    a recording sink (C-08 covers only `heartbeat=None`), so nothing exercises the drop
    branch that every pool built without a monitor will take.

**G-18** rides along because it consumes the same replayed drain script: the instrumentation
silently DROPS terminal-reason codes outside the known four, so `terminal_reason_counts()`
under-counts. That is OLD TRUTH, pinned deliberately — it reads exactly like a bug, and
"fixing" it is a parity violation until a monitoring work package owns the change.

The harness is self-contained (it does not import the ⊕ module) so a change there cannot
silently reshape these arms.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

import pytest

from mantis.selfplay import pool_drain
from mantis.selfplay.instrumentation import PoolInstrumentation
from mantis.selfplay.pool import WorkerPool

# ONE iteration consumes two clock reads (the drain-interval read and the buffer-cadence
# read); the loop takes one more before it starts. Two iterations therefore need five.
# 1005.0 → first buffer read at 1006.0 crosses 5 s from the 1000.0 start and RESETS the
# marker; the second iteration's 1008.0 is only 2 s past the reset, so it must NOT emit —
# but it IS 8 s past the start, so a marker that never got updated emits twice.
CLOCK_TWO_ITERATIONS = (1000.0, 1005.0, 1006.0, 1007.0, 1008.0)


class _Buffer:
    def __init__(self) -> None:
        self.size = 1234
        self.capacity = 5678
        self.dense_calls: list[tuple] = []

    def push_dense_many(self, *args: Any, **kwargs: Any) -> None:
        self.dense_calls.append((args, kwargs))


class _Recorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def maybe_record(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


class _Runner:
    def __init__(self, rows, games, counters, positions_generated) -> None:
        self._rows = rows
        self._games = games
        for key, value in counters.items():
            setattr(self, key, value)
        self.positions_generated = positions_generated

    def collect_data(self):
        return self._rows

    def drain_game_results(self):
        return list(self._games)


class _NShotStop:
    """`is_set()` → False for the first `n` calls, then True: the loop runs n iterations."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls > self.n


class _Clock:
    def __init__(self, sequence) -> None:
        self.sequence = list(sequence)
        self.i = 0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        value = self.sequence[min(self.i, len(self.sequence) - 1)]
        self.i += 1
        return value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class _Pool:
    """The drain body's read surface, plus the real methods G-18 needs bound to it.

    Binding the REAL `WorkerPool` methods to a stub carrying only their inputs is the same
    instrument the old-side capture used, so the numbers below come out of production code
    rather than a re-implementation.
    """

    terminal_reason_counts = WorkerPool.terminal_reason_counts
    buffer_composition = WorkerPool.buffer_composition


def _games_from_golden(golden: dict[str, Any]) -> list[tuple]:
    games = []
    for row in golden["_constants"]["games_batch"]:
        plies, winner_code, moves, *rest = row
        games.append((plies, winner_code, [tuple(m) for m in moves], *rest))
    return games


def _build_pool(golden, rows, *, sink, iterations: int, clock) -> _Pool:
    consts = golden["_constants"]
    pool = _Pool()
    pool._stop_event = _NShotStop(iterations)
    pool._is_graph = False
    pool._runner = _Runner(rows, _games_from_golden(golden),
                           consts["runner_counters"],
                           consts["runner_positions_generated"])
    pool.replay_buffer = _Buffer()
    pool._lock = threading.Lock()
    pool.positions_pushed = 0
    pool.self_play_positions_pushed = 0
    pool._feat_len = consts["feat_len"]
    pool._chain_len = consts["chain_len"]
    pool._trunk_size = consts["trunk_size"]
    pool.recent_buffer = None
    pool._last_drain_time = clock[0]
    pool._last_pos_generated = consts["last_pos_generated_before"]
    pool._effective_sims_per_move = consts["effective_sims_per_move"]
    pool._total_sims = 0
    pool._sims_per_sec = 0.0
    pool._game_lengths = deque(maxlen=200)
    pool._avg_game_length = 0.0
    pool._instrumentation = PoolInstrumentation(log_investigation_metrics=True)
    pool._recorder = _Recorder()
    pool._sink = sink
    pool._heartbeat = None
    pool.games_completed = 0
    pool.x_wins = 0
    pool.o_wins = 0
    pool.draws = 0
    pool.config = {"training": {"draw_value": -0.5, "ply_cap_value": -0.7}}
    return pool


@pytest.fixture
def run_drain(monkeypatch, drain_goldens, collect_data_input):
    """Factory → the stub pool after `iterations` completed drain loops."""
    def run(*, sink, iterations=1, clock=CLOCK_TWO_ITERATIONS, dense_n=4) -> _Pool:
        rows = tuple(a[:dense_n] for a in collect_data_input)
        pool = _build_pool(drain_goldens, rows, sink=sink, iterations=iterations,
                           clock=clock)
        monkeypatch.setattr(pool_drain, "time", _Clock(clock))
        pool_drain.run_stats_loop(pool)
        return pool

    return run


# ═══ C-09 — the system_stats marker survives across iterations ════════════════════
def test_system_stats_cadence_two_iterations(run_drain) -> None:
    """C-09 — PASS iff TWO drain iterations whose clock crosses the 5 s boundary exactly
    once (relative to the RESET marker) emit exactly ONE `system_stats`.

    The marker is a local of the loop and is reset on every emission. If an implementation
    hoists it onto the pool and never updates it, the second iteration measures 8 s from
    the loop start instead of 2 s from the reset, and emits again — at 10 Hz that is an
    event-stream flood that a monitor cannot see past, and no one-iteration harness can
    detect it. FAIL in the other direction (zero emissions) means the buffer panel goes
    permanently stale between training iterations."""
    sink = _Sink()
    run_drain(sink=sink, iterations=2)

    assert len(sink.named("system_stats")) == 1, (
        "exactly one system_stats across two iterations — a second means the 5 s marker "
        "is not being reset; zero means the cadence never fires"
    )
    assert len(sink.named("game_complete")) == 12, (
        "both iterations must still drain their six games — the cadence pin must not be "
        "satisfied by the loop simply doing less work"
    )


def test_system_stats_marker_is_not_pool_state(run_drain) -> None:
    """C-09 (mechanism arm) — PASS iff no `_last_buf_emit`-shaped attribute is left on the
    pool after the loop. The cadence marker must live in the loop frame; an attribute here
    is the hoist C-09 exists to catch, in a form a reviewer can grep for."""
    pool = run_drain(sink=_Sink(), iterations=2)
    leaked = [name for name in vars(pool) if "buf_emit" in name]
    assert not leaked, f"cadence marker leaked onto the pool: {leaked}"


# ═══ C-10 — the no-sink default path ═════════════════════════════════════════════
def test_drain_with_no_sink(run_drain) -> None:
    """C-10 — PASS iff a full drain iteration with `sink=None` (the constructor default)
    completes with no error and does all of its real work: the push happens, the counters
    move, the recorder still sees every game.

    Every pool built without a monitor takes this branch, and the ⊕ suite never does.
    FAIL = the drop branch raises `AttributeError` on the first drained game, killing the
    sole producer of training data on any run without an event sink."""
    pool = run_drain(sink=None, iterations=1)

    assert len(pool.replay_buffer.dense_calls) == 1
    assert pool.positions_pushed == 4
    assert pool.games_completed == 6
    assert len(pool._recorder.records) == 6, (
        "dropping events must not drop the recorder — they are separate seams"
    )


def test_no_sink_and_recording_sink_agree_on_everything_else(run_drain) -> None:
    """C-10 (neutrality arm) — PASS iff injecting a sink changes NOTHING except that the
    events are observed: same pushes, same counters, same sims bill. FAIL = the emit path
    has a side effect on drain state, so a monitored run and an unmonitored run produce
    different training data."""
    with_sink = run_drain(sink=_Sink(), iterations=1)
    without = run_drain(sink=None, iterations=1)

    for attr in ("positions_pushed", "self_play_positions_pushed", "games_completed",
                 "x_wins", "o_wins", "draws", "_total_sims", "_sims_per_sec",
                 "_avg_game_length"):
        assert getattr(with_sink, attr) == getattr(without, attr), attr
    assert list(with_sink._game_lengths) == list(without._game_lengths)


# ═══ G-18 — the deliberately-pinned unknown-reason under-count ═══════════════════
def test_terminal_reason_counts_drops_unknown_codes(run_drain, drain_goldens) -> None:
    """G-18 — PASS iff, after the captured 6-game drain script (one game carrying the
    unrecognised terminal-reason code 7), `terminal_reason_counts()` reports the captured
    four-key histogram totalling **5**, not 6.

    This is OLD TRUTH pinned on purpose. The counter accepts any code but the reader
    surfaces only the four known ones, so an unknown code is counted internally and never
    reported. It reads exactly like a bug and someone will eventually "fix" it — that fix
    is a parity violation until a monitoring redesign owns it, and this row is the thing
    that makes the fix visible instead of silent."""
    pool = run_drain(sink=_Sink(), iterations=1)
    expected = drain_goldens["variants"]["dense_5s_crossed"]["instrumentation_after"][
        "terminal_reason_counts"]

    counts = pool.terminal_reason_counts()
    assert counts == expected
    assert sum(counts.values()) == 5, (
        "six games were drained; the game with reason code 7 is silently dropped"
    )
    assert set(counts) == {"six_in_a_row", "colony", "ply_cap", "other_draw"}


def test_buffer_composition_inherits_the_under_count(run_drain, drain_goldens) -> None:
    """G-18 (propagation arm) — PASS iff `buffer_composition()["n_games_observed"]`
    inherits the under-count (5 for 6 drained games) and the four terminal fractions are
    normalised by that same under-count.

    The fractions are therefore inflated relative to games actually played whenever an
    unknown code appears. Pinned, not corrected: a monitor consuming this field is reading
    the same number the old run emitted."""
    pool = run_drain(sink=_Sink(), iterations=1)
    composition = pool.buffer_composition()

    assert composition["n_games_observed"] == 5
    counts = pool.terminal_reason_counts()
    assert composition["six_terminal_fraction"] == counts["six_in_a_row"] / 5
    assert composition["colony_terminal_fraction"] == counts["colony"] / 5
    assert composition["cap_terminal_fraction"] == counts["ply_cap"] / 5
    assert composition["other_draw_fraction"] == counts["other_draw"] / 5
