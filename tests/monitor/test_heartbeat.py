"""O-13 (registry + file-codec facets) — heartbeat registry monotonicity, atomic file
codec, clock-injection immunity.

RED-at-import until IMPL writes `mantis.monitor.heartbeat`. Asserts the §c.4 public API:
`HEARTBEAT_SOURCES`, `WATCHDOG_STALL_EXIT_CODE=42`, `PERSIST_FATAL_EXIT_CODE=43`,
`HeartbeatRegistry(*, clock, sources)` with `.beat/.arm/.ages`, and the file codec
`write_heartbeat_file(path, *, seq, pid, ages, wall_ts)` / `read_heartbeat_file(path)`.

Facets covered here (the supervisor-side seq/mtime/pid staleness edges are O-13(ii/iii) in
test_supervisor.py):
  * registry `beat` on an unknown source ⇒ ValueError; `arm` grace-resets all sources;
  * ages() use the INJECTED monotonic clock only — a wall-clock jump changes nothing (O-13 i);
  * the file codec round-trips seq/pid, is tolerant of an absent OR torn file (→ None), and
    a re-write with a higher seq reads back the higher seq (monotone progression carrier).
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest

from mantis.monitor.heartbeat import (
    HEARTBEAT_SOURCES,
    PERSIST_FATAL_EXIT_CODE,
    WATCHDOG_STALL_EXIT_CODE,
    HeartbeatRegistry,
    read_heartbeat_file,
    write_heartbeat_file,
)


def test_exit_code_constants_are_pinned() -> None:
    """The supervisor-readable authority: 42 = stall/livelock, 43 = persist-fatal."""
    assert WATCHDOG_STALL_EXIT_CODE == 42
    assert PERSIST_FATAL_EXIT_CODE == 43


def test_heartbeat_sources_name_pins() -> None:
    """The pipeline stages are name-pinned (the watchdog + manifest key on these).
    WP11-A adds "eval_round" as the 4th source (the eval pipeline's poller thread)."""
    assert HEARTBEAT_SOURCES == (
        "train_step", "inference_dispatch", "selfplay_drain", "eval_round",
    )


# ── registry ──────────────────────────────────────────────────────────────────────────
def _clock():
    box = [0.0]

    def c() -> float:
        return box[0]

    c.box = box  # type: ignore[attr-defined]
    return c


def test_arm_zeroes_all_ages() -> None:
    clock = _clock()
    reg = HeartbeatRegistry(clock=clock)
    clock.box[0] = 100.0
    reg.arm()
    ages = reg.ages()
    assert set(ages) == set(HEARTBEAT_SOURCES)
    assert all(a == 0.0 for a in ages.values()), "arm() grace-resets every source to age 0"


def test_beat_resets_only_that_source() -> None:
    clock = _clock()
    reg = HeartbeatRegistry(clock=clock)
    reg.arm()                       # t=0 baseline
    clock.box[0] = 50.0
    reg.beat("train_step")          # only train_step refreshed at t=50
    clock.box[0] = 60.0
    ages = reg.ages()
    assert ages["train_step"] == pytest.approx(10.0)          # 60 - 50
    assert ages["inference_dispatch"] == pytest.approx(60.0)  # 60 - 0
    assert ages["selfplay_drain"] == pytest.approx(60.0)


def test_beat_unknown_source_raises() -> None:
    """An unknown source is a wiring bug → ValueError, never a silently-dropped beat."""
    reg = HeartbeatRegistry(clock=_clock())
    reg.arm()
    with pytest.raises(ValueError):
        reg.beat("not_a_pipeline_stage")


def test_ages_use_injected_clock_not_wall_clock(monkeypatch) -> None:
    """O-13(i) — staleness is measured on the INJECTED monotonic clock; a wall-clock jump
    (time.time / a system clock change) never moves an age. Bites NTP-skew false fires."""
    clock = _clock()
    reg = HeartbeatRegistry(clock=clock)
    reg.arm()
    import time as _time

    monkeypatch.setattr(_time, "time", lambda: 10 ** 9)  # violent wall-clock jump
    assert all(a == 0.0 for a in reg.ages().values()), (
        "a wall-clock jump must not change any age — only the injected clock advances it"
    )
    clock.box[0] = 5.0
    assert all(a == pytest.approx(5.0) for a in reg.ages().values())


# ── file codec ────────────────────────────────────────────────────────────────────────
def test_file_codec_round_trips_seq_and_pid(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    write_heartbeat_file(path, seq=7, pid=4321, ages={"train_step": 1.5}, wall_ts=12.0)
    state = read_heartbeat_file(path)
    assert state is not None
    assert state.seq == 7
    assert state.pid == 4321


def test_file_codec_higher_seq_is_read_back(tmp_path: Path) -> None:
    """A re-write with a higher seq reads back the higher seq (the monotone progression the
    supervisor keys liveness on)."""
    path = tmp_path / "heartbeat.json"
    write_heartbeat_file(path, seq=1, pid=1, ages={}, wall_ts=0.0)
    write_heartbeat_file(path, seq=2, pid=1, ages={}, wall_ts=1.0)
    state = read_heartbeat_file(path)
    assert state is not None and state.seq == 2


def test_read_absent_file_is_none(tmp_path: Path) -> None:
    """A missing heartbeat file is tolerated (→ None) — the reader never raises (R6)."""
    assert read_heartbeat_file(tmp_path / "does_not_exist.json") is None


def test_read_torn_file_is_none(tmp_path: Path) -> None:
    """A torn/garbage file (partial write on an exotic FS) reads as None, never an exception —
    the supervisor treats that as no-progress, which only ever errs toward a relaunch (R6)."""
    path = tmp_path / "heartbeat.json"
    path.write_text("{ this is not valid json")
    assert read_heartbeat_file(path) is None


def test_write_is_atomic_no_partial_left_behind(tmp_path: Path) -> None:
    """The write goes tmp → os.replace, so a reader never sees a half-written file and no
    stray tmp sibling is left behind after a successful write."""
    path = tmp_path / "heartbeat.json"
    write_heartbeat_file(path, seq=3, pid=9, ages={"selfplay_drain": 0.1}, wall_ts=2.0)
    siblings = [p.name for p in tmp_path.iterdir()]
    assert siblings == ["heartbeat.json"], f"no tmp sibling may survive, found {siblings}"
    assert read_heartbeat_file(path) is not None


# ══ RED-TEAM F4 — the reader NEVER raises, on the whole hostile corpus ════════════════
_HOSTILE_FILES = {
    "empty": "",
    "whitespace": "   \n",
    "truncated_json": '{"seq": 3, "pid":',
    "not_json": "this is not json at all",
    "json_list": "[1, 2, 3]",
    "json_scalar": "7",
    "json_string": '"hello"',
    "no_seq": '{"pid": 10}',
    "no_pid": '{"seq": 10}',
    "seq_list": '{"seq": [1], "pid": 2}',
    "seq_string": '{"seq": "7", "pid": 2}',
    "seq_bool": '{"seq": true, "pid": 2}',
    "seq_negative": '{"seq": -5, "pid": 2}',
    "seq_infinity": '{"seq": Infinity, "pid": 2}',
    "seq_neg_infinity": '{"seq": -Infinity, "pid": 2}',
    "seq_nan": '{"seq": NaN, "pid": 2}',
    "seq_1e400": '{"seq": 1e400, "pid": 2}',
    "seq_huge_int": '{"seq": %d, "pid": 2}' % (2 ** 200),
    "pid_infinity": '{"seq": 1, "pid": Infinity}',
    "pid_nan": '{"seq": 1, "pid": NaN}',
    "ages_string": '{"seq": 1, "pid": 2, "ages": "nope"}',
    "ages_list": '{"seq": 1, "pid": 2, "ages": [1, 2]}',
    "ages_nan_value": '{"seq": 1, "pid": 2, "ages": {"train_step": NaN}}',
    "wall_ts_infinity": '{"seq": 1, "pid": 2, "wall_ts": Infinity}',
    "junk_1mb": "x" * 1_000_000,
}


@pytest.mark.parametrize("name", sorted(_HOSTILE_FILES))
def test_read_heartbeat_file_never_raises_on_hostile_content(tmp_path: Path, name: str) -> None:
    """RED-TEAM F4 — `read_heartbeat_file` contracts to NEVER raise; the red team broke that
    with `Infinity` / `1e400` / `pid: Infinity` (`int(inf)` → `OverflowError`), which kills the
    supervisor's unguarded poll loop and leaves the child running unsupervised.

    Every hostile file must yield either `None` (no progress observable — the safe side, which
    only ever errs toward a relaunch) or a WELL-FORMED state with finite, non-negative,
    in-range counters. Nothing may propagate."""
    path = tmp_path / "hb.json"
    path.write_text(_HOSTILE_FILES[name])
    state = read_heartbeat_file(path)          # must not raise, whatever the content
    if state is not None:
        assert isinstance(state.seq, int) and state.seq >= 0
        assert isinstance(state.pid, int) and state.pid >= 0
        assert math.isfinite(state.wall_ts)
        assert all(math.isfinite(v) for v in state.ages.values())


def test_read_heartbeat_file_rejects_non_finite_counters(tmp_path: Path) -> None:
    """F4 — the specific three the red team used must read as NO STATE, never as progress:
    a forged `Infinity` seq must not look like the largest possible advance."""
    for body in ('{"seq": Infinity, "pid": 1}', '{"seq": 1e400, "pid": 1}',
                 '{"seq": 1, "pid": Infinity}'):
        path = tmp_path / "hb.json"
        path.write_text(body)
        assert read_heartbeat_file(path) is None, body


def test_write_heartbeat_file_uses_a_unique_tmp_sibling(tmp_path: Path) -> None:
    """RED-TEAM F17 — two writers on ONE path must not race each other's temp file. A fixed
    `<name>.tmp` produced `FileNotFoundError` out of `os.replace` on ~30% of writes (invisible
    inside the watchdog, because the mirror is best_effort-wrapped)."""
    path = tmp_path / "hb.json"
    errors: list[BaseException] = []

    def _write(pid: int) -> None:
        try:
            for seq in range(1, 40):
                write_heartbeat_file(path, seq=seq, pid=pid, ages={"train_step": 0.0},
                                     wall_ts=1.0)
        except BaseException as exc:            # noqa: BLE001 — reported, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(pid,)) for pid in (101, 102, 103)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == [], f"concurrent writers must not break each other: {errors[:3]}"
    assert read_heartbeat_file(path) is not None
    assert list(tmp_path.glob("*.tmp")) == [], "no tmp sibling may survive"
