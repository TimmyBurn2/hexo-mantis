"""Scripted drain/push GOLDENS.

>300 justify (R8): one scripted harness — stub pool, recording collaborators, patched clock —
feeds every row; splitting it would duplicate the harness and let the two copies drift.

The REAL drain body runs against a stub pool carrying exactly the attribute surface it reads,
every collaborator records, the clock is a scripted sequence, and the stop event releases after
ONE iteration. The graph variants were re-captured from this harness with every arm-independent
field verified equal to the old-side dense capture they replace.
"""
from __future__ import annotations

import json
import re
import threading
from collections import deque
from typing import Any

import numpy as np
import pytest

from mantis._engine import DEFAULT_CLUSTER_THRESHOLD
from mantis.selfplay import pool_drain
from mantis.selfplay.instrumentation import PoolInstrumentation

GAME_ID_RE = re.compile(r"[0-9a-f]{32}")

CLOCK_CROSSED = (1000.0, 1002.0, 1006.5)
CLOCK_NOT_CROSSED = (1000.0, 1002.0, 1003.25)

#: How each graph golden variant was driven.
_VARIANT_RUNS: dict[str, dict[str, Any]] = {
    "graph": {},
    "graph_5s_not_crossed": {"clock": CLOCK_NOT_CROSSED},
    "graph_zero_rows": {"graph_n": 0},
}


class RecordingBuffer:
    """Records every graph push and every id allocation; it has no dense push to call."""

    def __init__(self, size: int = 1234, capacity: int = 5678) -> None:
        self.size = size
        self.capacity = capacity
        self.graph_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.next_game_id_calls = 0

    def push_graph_position(self, *args: Any, **kwargs: Any) -> None:
        self.graph_calls.append((args, dict(kwargs)))

    def next_game_id(self) -> int:
        """The ring's own id allocator, which the push calls once per GAME. Recorded so the CALL
        COUNT can be asserted: allocating once per row would tag every position as its own game,
        and no assertion on the ids alone would catch that."""
        self.next_game_id_calls += 1
        return 900 + self.next_game_id_calls


class RecordingRecentBuffer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def push(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, dict(kwargs)))


class RecordingRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.steps: list[int] = []
        self.stopped = 0

    def maybe_record(self, **kwargs: Any) -> None:
        self.records.append(dict(kwargs))

    def set_step(self, step: int) -> None:
        self.steps.append(step)

    def stop(self) -> None:
        self.stopped += 1


class RecordingSink:
    """Structural `EventSink`: a single `emit(event: Mapping)` method (DV-4)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


class RecordingHeartbeat:
    """Structural `HeartbeatFn`: `Callable[[str], None]` (§c.5)."""

    def __init__(self) -> None:
        self.sources: list[str] = []

    def __call__(self, source: str) -> None:
        self.sources.append(source)


class ScriptedRunner:
    """Answers only the graph drain; a dense `collect_data()` call would raise AttributeError."""

    def __init__(self, graph_rows, games, counters, positions_generated) -> None:
        self._graph_rows = graph_rows
        self._games = games
        self.calls: list[str] = []
        for key, value in counters.items():
            setattr(self, key, value)
        self.positions_generated = positions_generated

    def collect_graph_data(self):
        self.calls.append("collect_graph_data")
        return list(self._graph_rows)

    def drain_game_results(self):
        self.calls.append("drain_game_results")
        return list(self._games)


class OneShotStop:
    """`is_set()` → False exactly once, then True: the loop runs ONE iteration."""

    def __init__(self) -> None:
        self.n = 0

    def is_set(self) -> bool:
        self.n += 1
        return self.n > 1


class ScriptedTime:
    """Stands in for the `time` module inside `pool_drain` (capture harness, verbatim)."""

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


class ScriptedPool:
    """Exactly the attribute surface the drain body reads (capture `PoolStub`, plus the
    three DV-4/§c.5 injection seams)."""


def _games_from_golden(golden: dict[str, Any]) -> list[tuple]:
    """The scripted `drain_game_results()` 10-tuples (moves back to tuple-of-tuples)."""
    games = []
    for row in golden["_constants"]["games_batch"]:
        plies, winner_code, moves, *rest = row
        games.append((plies, winner_code, [tuple(m) for m in moves], *rest))
    return games


def _build_pool(golden, graph_rows, *, clock, recent_buffer=True, sink=None,
                heartbeat=None) -> ScriptedPool:
    consts = golden["_constants"]
    pool = ScriptedPool()
    pool._stop_event = OneShotStop()
    pool._runner = ScriptedRunner(graph_rows, _games_from_golden(golden),
                                  consts["runner_counters"], consts["runner_positions_generated"])
    pool.replay_buffer = RecordingBuffer()
    pool._lock = threading.Lock()
    pool.positions_pushed = 0
    pool.self_play_positions_pushed = 0
    pool.graph_rows_pushed = 0
    pool.alpha_full_rows = 0
    pool.alpha_full_rows_emitted = 0
    pool.recent_buffer = RecordingRecentBuffer() if recent_buffer else None
    pool._last_drain_time = clock[0]
    pool._last_pos_generated = consts["last_pos_generated_before"]
    pool._effective_sims_per_move = consts["effective_sims_per_move"]
    pool._total_sims = 0
    pool._sims_per_sec = 0.0
    pool._game_lengths = deque(maxlen=200)
    pool._avg_game_length = 0.0
    pool._instrumentation = PoolInstrumentation(log_investigation_metrics=True, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD)
    pool._recorder = RecordingRecorder()
    pool._sink = sink if sink is not None else RecordingSink()
    pool._heartbeat = heartbeat
    pool.games_completed = 0
    pool.x_wins = 0
    pool.o_wins = 0
    pool.draws = 0
    return pool


@pytest.fixture
def run_drain(monkeypatch, drain_goldens, graph_rows_input):
    """Factory → (pool, scripted_clock) after ONE `run_stats_loop` iteration."""
    def run(*, clock=CLOCK_CROSSED, recent_buffer=True, graph_n=3, sink=None, heartbeat=None):
        pool = _build_pool(drain_goldens, graph_rows_input[:graph_n], clock=clock,
                           recent_buffer=recent_buffer, sink=sink, heartbeat=heartbeat)
        scripted = ScriptedTime(clock)
        monkeypatch.setattr(pool_drain, "time", scripted)
        pool_drain.run_stats_loop(pool)
        return pool, scripted

    return run


def _variant(drain_goldens: dict[str, Any], name: str) -> dict[str, Any]:
    return drain_goldens["variants"][name]


def _assert_array(actual: Any, expected: np.ndarray, label: str) -> None:
    assert isinstance(actual, np.ndarray), f"{label}: expected ndarray, got {type(actual)}"
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    assert np.array_equal(actual, expected), f"{label}: bytes differ from the captured push"


def _assert_graph_push_bytes(pool: ScriptedPool, graph_pushed: dict[str, np.ndarray]) -> None:
    """Every pushed row's two arrays are byte-identical to the capture."""
    assert len(pool.replay_buffer.graph_calls) == 3
    for i, (args, _kwargs) in enumerate(pool.replay_buffer.graph_calls):
        _assert_array(args[0], graph_pushed[f"push_graph_position_{i}_arg0"], f"row{i}.arg0")
        _assert_array(args[1], graph_pushed[f"push_graph_position_{i}_arg1"], f"row{i}.arg1")


def test_graph_drain_push_rows(run_drain, drain_goldens, graph_pushed, graph_rows_input):
    """Each graph row's fields are forwarded UNCHANGED and its runner game id is translated.

    What is pinned is the TRANSLATION — one allocation per GAME, not per row — because that is the
    property an implementation can get wrong while still passing "the ids are not -1".
    """
    pool, _ = run_drain()
    golden = _variant(drain_goldens, "graph")

    assert pool._runner.calls == golden["runner_call_order"] == [
        "collect_graph_data", "drain_game_results"]
    assert len(pool.replay_buffer.graph_calls) == len(golden["buffer_calls"]) == 3

    for i, (args, kwargs) in enumerate(pool.replay_buffer.graph_calls):
        expected_row = graph_rows_input[i]
        assert len(args) == len(expected_row) - 2, (
            f"row {i}: arity changed — the tail mass rides by keyword and the trailing game id "
            "is consumed, so neither may be forwarded positionally"
        )
        assert kwargs["tail_mass"] == expected_row[-2], (
            f"row {i}: the row's tail mass (R347(a)'s alpha) must reach the push; got {kwargs}"
        )
        _assert_array(args[0], graph_pushed[f"push_graph_position_{i}_arg0"], f"row{i}.arg0")
        _assert_array(args[1], graph_pushed[f"push_graph_position_{i}_arg1"], f"row{i}.arg1")
        assert args[2] == expected_row[2] and args[3] == expected_row[3]
        want_args = golden["buffer_calls"][i]["args"]
        assert [args[2], args[3]] == [want_args[2]["value"], want_args[3]["value"]], f"row {i}"
        assert args[0] is expected_row[0] and args[1] is expected_row[1], (
            f"row {i}: arrays were re-materialized — the push path must forward, not copy"
        )

    pushed_ids = [kw["game_id"] for _a, kw in pool.replay_buffer.graph_calls]
    assert pushed_ids == [c["kwargs"]["game_id"]["value"] for c in golden["buffer_calls"]]
    assert pushed_ids[0] == pushed_ids[1], (
        f"rows 0 and 1 are the same runner game and got different ids: {pushed_ids}"
    )
    assert pushed_ids[2] != pushed_ids[0], f"two games collapsed to one id: {pushed_ids}"
    assert (pool.replay_buffer.next_game_id_calls
            == golden["counters_after"]["next_game_id_calls"] == 2), (
        f"the buffer allocated {pool.replay_buffer.next_game_id_calls} ids for 2 games — "
        "allocation is per ROW, which tags every position as its own game"
    )


def test_graph_zero_rows_pushes_nothing(run_drain, drain_goldens):
    """`collect_graph_data()` returning no rows produces NO push and allocates NO id: a
    "helpful" empty push would write nothing yet still spend ring ids every idle tick."""
    pool, _ = run_drain(graph_n=0)
    counters = _variant(drain_goldens, "graph_zero_rows")["counters_after"]

    assert pool.replay_buffer.graph_calls == [], "no rows must not call the graph push"
    assert pool.replay_buffer.next_game_id_calls == counters["next_game_id_calls"] == 0
    assert pool.positions_pushed == counters["positions_pushed"] == 0
    assert pool.self_play_positions_pushed == counters["self_play_positions_pushed"] == 0
    assert pool.graph_rows_pushed == counters["graph_rows_pushed"] == 0


def test_game_complete_payload_golden(run_drain, drain_goldens):
    """The six emitted `game_complete` payloads equal the capture on ALL 19 keys except the uuid.

    A failure is the event contract drifting — including a dropped, added, or reordered key.

    THE CAPTURE WAS UPDATED DELIBERATELY IN TWO CELLS, each a fabrication it had frozen. One game
    carries a winner code no map entry covers and was captured as a measured DRAW while the log
    line beside it printed `winner=unknown`. Another has an EMPTY move history, so nothing computed
    its colony or longest-line metrics and all five were captured as zeros — and a zero longest
    line is a legitimate measurement for other games in this very capture, which is why the absent
    case cannot share its value. Both are now null.
    """
    pool, _ = run_drain()
    golden = _variant(drain_goldens, "graph")
    expected_events = [e for e in golden["events"] if e["event"] == "game_complete"]
    actual_events = pool._sink.named("game_complete")

    assert len(actual_events) == len(expected_events) == 6
    for i, (actual, expected) in enumerate(
            zip(actual_events, expected_events, strict=True)):
        game_id = actual.pop("game_id", None)
        assert isinstance(game_id, str) and GAME_ID_RE.fullmatch(game_id), (
            f"event {i}: game_id must be a 32-char lowercase hex uuid4 hex, got {game_id!r}"
        )
        assert set(actual) == set(expected), (
            f"event {i}: key set drift — missing {set(expected) - set(actual)}, "
            f"extra {set(actual) - set(expected)}"
        )
        for key, want in expected.items():
            assert actual[key] == want, f"event {i}: {key} = {actual[key]!r} != {want!r}"

    # The dedupe property the capture verified: byte-identical move sequences collide.
    hashes = [e["game_id_byte_hash"] for e in expected_events]
    assert hashes[2] == hashes[5], (
        "capture pins games 2 and 5 (identical move sequences) to the SAME byte hash — the "
        "trajectory-hash dedupe that LAW-04's effective-n depends on"
    )


def test_sims_per_sec_billing(run_drain, drain_goldens):
    """The per-MOVE bill reproduces the capture exactly (240 moves × 111 effective sims over 2.0 s
    ⇒ 13 320.0), rather than the falsified per-GAME undercount that understated throughput ~100×."""
    pool, scripted = run_drain()
    counters = _variant(drain_goldens, "graph")["counters_after"]

    assert pool._total_sims == counters["_total_sims"] == 26640
    assert pool._sims_per_sec == counters["_sims_per_sec"] == 13320.0
    assert pool._last_pos_generated == counters["_last_pos_generated"] == 340
    assert pool._last_drain_time == counters["_last_drain_time"] == 1002.0
    assert scripted.sleeps == _variant(drain_goldens, "graph")["sleeps"] == [0.1]


@pytest.mark.parametrize("variant", sorted(_VARIANT_RUNS))
def test_counters_mirror_runner(run_drain, drain_goldens, variant):
    """The public counters mirror the runner exactly, with the captured pushed-row counts, per-game
    lengths and average. A failure desyncs the numbers the monitor and the training loop read."""
    pool, _ = run_drain(**_VARIANT_RUNS[variant])
    counters = _variant(drain_goldens, variant)["counters_after"]

    assert pool.games_completed == counters["games_completed"] == 6
    assert pool.x_wins == counters["x_wins"] == 2
    assert pool.o_wins == counters["o_wins"] == 1
    assert pool.draws == counters["draws"] == 3
    assert pool.positions_pushed == counters["positions_pushed"]
    assert pool.self_play_positions_pushed == counters["self_play_positions_pushed"]
    assert pool.graph_rows_pushed == counters["graph_rows_pushed"]
    assert list(pool._game_lengths) == counters["_game_lengths"] == [6, 12, 125, 15, 8, 5]
    assert pool._avg_game_length == counters["_avg_game_length"] == 28.5


def test_graph_drain_leaves_the_recent_buffer_alone(run_drain, drain_goldens, graph_pushed):
    """Graph recency flows in-engine, so the drain pushes nothing to a Python recent buffer, and a
    pool with none leaves the replay push byte-identical and raises nothing."""
    pool, _ = run_drain()
    assert pool.recent_buffer.calls == _variant(drain_goldens, "graph")["recent_buffer_calls"] == []
    _assert_graph_push_bytes(pool, graph_pushed)

    absent, _ = run_drain(recent_buffer=False)
    assert absent.recent_buffer is None
    _assert_graph_push_bytes(absent, graph_pushed)


def test_system_stats_cadence(run_drain, drain_goldens):
    """A clock crossing the 5 s boundary emits exactly one `system_stats` payload after the six
    `game_complete` events, and one that does not cross emits none. The order comparison filters to
    the golden's tracked types, since the stream now also carries lifecycle events."""
    _GOLDEN_TYPES = {"game_complete", "system_stats"}
    crossed_golden = _variant(drain_goldens, "graph")
    pool, _ = run_drain(clock=CLOCK_CROSSED)
    golden_tracked = [e for e in pool._sink.events if e["event"] in _GOLDEN_TYPES]
    assert [e["event"] for e in golden_tracked] == crossed_golden["event_order"]
    stats = pool._sink.named("system_stats")
    assert len(stats) == 1
    expected_stats = [e for e in crossed_golden["events"] if e["event"] == "system_stats"][0]
    assert stats[0] == expected_stats == {"event": "system_stats", "buffer_size": 1234,
                                          "buffer_capacity": 5678}

    not_crossed_golden = _variant(drain_goldens, "graph_5s_not_crossed")
    pool2, _ = run_drain(clock=CLOCK_NOT_CROSSED)
    golden_tracked2 = [e for e in pool2._sink.events if e["event"] in _GOLDEN_TYPES]
    assert [e["event"] for e in golden_tracked2] == not_crossed_golden["event_order"]
    assert pool2._sink.named("system_stats") == [], (
        "system_stats must not fire before the 5 s boundary"
    )


def test_heartbeat_emission_at_drain(run_drain):
    """An injected `HeartbeatFn` receives exactly ONE call per loop iteration, and the default
    `None` produces zero side effects — same events, pushes and counters. Behaviour-neutrality is
    the verdict, not a nicety: this emission point exists for a watchdog not wired here."""
    beat = RecordingHeartbeat()
    with_beat, _ = run_drain(heartbeat=beat)
    assert beat.sources == ["selfplay_drain"], (
        f"expected exactly one 'selfplay_drain' beat per iteration, got {beat.sources}"
    )

    without_beat, _ = run_drain(heartbeat=None)
    assert ([e["event"] for e in with_beat._sink.events]
            == [e["event"] for e in without_beat._sink.events])
    assert (len(with_beat.replay_buffer.graph_calls)
            == len(without_beat.replay_buffer.graph_calls) == 3)
    for i, ((a_args, a_kw), (b_args, b_kw)) in enumerate(zip(
            with_beat.replay_buffer.graph_calls, without_beat.replay_buffer.graph_calls,
            strict=True)):
        assert all(np.array_equal(x, y) for x, y in zip(a_args[:2], b_args[:2], strict=True)), (
            f"injecting a heartbeat changed pushed row {i} — not behaviour-neutral"
        )
        assert a_args[2:] == b_args[2:] and a_kw == b_kw, f"row {i}: scalars or ids changed"
    assert with_beat._total_sims == without_beat._total_sims
    assert with_beat.games_completed == without_beat.games_completed


#: Every kwarg `maybe_record` is called with, asserted as a SET so drift reds in BOTH directions —
#: a dropped field and a smuggled one. Only the golden-valued subset is compared by value:
#: `game_id` is a fresh uuid4 per game and can never be a golden.
_RECORDER_KWARGS = {
    "game_id", "moves", "winner_code", "plies", "worker_id", "terminal_reason",
    "game_id_byte_hash", "served_sims", "move_arms", "search_stats",
}


def test_recorder_receives_every_drained_game(run_drain, drain_goldens):
    """`recorder.maybe_record` is called once per drained game with the captured fields.

    The golden's `game_length` was RENAMED to `plies` carrying the same values: this call site
    always passed plies while the drain computes a real game length as `(plies + 1) // 2`, so the
    golden was recording plies under a name that means something else (LAW-03).
    """
    pool, _ = run_drain()
    expected = _variant(drain_goldens, "graph")["recorder_calls"]

    assert len(pool._recorder.records) == len(expected) == 6
    for i, (actual, want) in enumerate(
            zip(pool._recorder.records, expected, strict=True)):
        assert set(actual) == _RECORDER_KWARGS, f"recorder call {i}: kwarg set drift"
        assert set(want) <= set(actual), f"recorder call {i}: golden field missing"
        assert [list(m) for m in actual["moves"]] == want["moves"], f"recorder call {i}: moves"
        assert actual["winner_code"] == want["winner_code"]
        assert actual["plies"] == want["plies"]
        # R353(d): the runner's per-move arms reach the recorder one per move, untouched.
        assert len(actual["move_arms"]) == len(actual["moves"]), f"recorder call {i}: arms"
        # R355(d): the tenth field reaches the recorder untouched — `None` on an un-sampled game.
        assert json.loads(json.dumps(actual["search_stats"])) == want["search_stats"], (
            f"recorder call {i}: search_stats"
        )
