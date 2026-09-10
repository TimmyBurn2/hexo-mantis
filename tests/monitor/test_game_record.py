# >300 justify (R8): the store's producer tests over one format — writer, reader, shard
# lifecycle, failure posture and record builders are one contract read from both ends.
"""GAME-RECORD-1's producer tests: the shard closes and re-opens byte-exact, a planted truncated
shard is skipped rather than fatal, the SEGMENT half of the shard key keeps a resume from
appending into a stopped process's file, a lost record disables and counts rather than raising
quietly, and an undecodable outcome is never recorded as a measured draw."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.monitor.game_record import (
    GAME_RECORD_CONTRACT,
    GameRecordWriter,
    iter_run_games,
    read_shard,
    seat_result,
    selfplay_record,
)
from mantis.monitor.game_recorder import GameRecorder
from mantis.monitor.sink import RunIdError


def _game(i: int) -> dict[str, object]:
    return selfplay_record(
        game_id=f"g{i:04d}", run_id="testrun", step=1000 + i,
        moves=[(q, q % 3 - 1) for q in range(6 + i % 5)],
        result="p1" if i % 2 else "draw", plies=6 + i % 5,
        termination="six_in_a_row" if i % 2 else "ply_cap",
        worker_id=i % 4, seed=20260719, served_sims=50,
        game_id_byte_hash=f"{i:040x}",
    )


def test_a_closed_shard_reopens_byte_exact(tmp_path: Path) -> None:
    """Every record written comes back, in order, field for field — not "the right number of
    lines". MUTATION THAT REDS IT: buffer the writes and drop the `close()` fsync."""
    writer = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    written = [_game(i) for i in range(25)]
    for record in written:
        writer.write(record)
    path = writer.shard_path
    writer.close()

    read_back, skipped = read_shard(path)
    assert skipped == 0, "a shard this process closed cleanly has no partial line"
    assert read_back == written, "the shard did not re-open byte-exact"
    assert writer.games_written == 25


def test_a_planted_truncated_shard_is_SKIPPED_not_fatal(tmp_path: Path) -> None:
    """A live shard's last line is routinely partial: the record before the tear survives, the
    torn one is COUNTED, nothing raises. A bare skip cannot tell one tear from a garbage file."""
    writer = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    for i in range(5):
        writer.write(_game(i))
    path = writer.shard_path
    writer.close()

    text = path.read_text(encoding="utf-8")
    path.write_text(text[: -len(text.splitlines()[-1]) // 2], encoding="utf-8")

    records, skipped = read_shard(path)
    assert skipped == 1, f"the torn line must be counted, not swallowed; skipped={skipped}"
    assert len(records) == 4, "every intact record before the tear must survive"
    assert [r["game_id"] for r in records] == [f"g{i:04d}" for i in range(4)]


def test_two_writers_in_one_run_and_one_hour_claim_DIFFERENT_shards(tmp_path: Path) -> None:
    """A resume inside the same wall-clock hour is a second process with the same run id, and a
    shard keyed on `(run, hour)` alone would append into the stopped process's file. MUTATION
    THAT REDS IT: drop `seg` from `shard_filename`, or open with `"a"`."""
    first = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    second = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    assert first.shard_path != second.shard_path, (
        "the second writer claimed the first writer's shard — a resume would append into a "
        "stopped process's file"
    )
    first.write(_game(1))
    second.write(_game(2))
    first.close()
    second.close()

    assert [r["game_id"] for r in iter_run_games(tmp_path, "testrun")] == ["g0001", "g0002"], (
        "both shards must be readable as ONE run, in segment order"
    )


def test_the_index_names_only_shards_whose_bytes_are_down(tmp_path: Path) -> None:
    """The index is written AFTER the fsync and carries a count the reader can check: an index
    naming a shard not yet on disk is a promise a reader will act on."""
    writer = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    for i in range(7):
        writer.write(_game(i))
    assert not writer.index_path.exists(), "an OPEN shard is not indexed"
    writer.close()

    rows = [json.loads(line) for line in writer.index_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["games"] == 7
    assert row["contract"] == GAME_RECORD_CONTRACT
    shard = writer.shard_path.parent / row["shard"]
    assert shard.exists() and shard.stat().st_size == row["bytes"]
    assert len(read_shard(shard)[0]) == row["games"], "the index count must match the file"


def test_the_shard_rotates_when_the_utc_hour_turns(tmp_path: Path, monkeypatch) -> None:
    """The hour rotation, driven by a moved clock rather than by waiting.
    MUTATION THAT REDS IT: rotate on record COUNT instead of on the hour."""
    import mantis.monitor.game_record as module

    hours = iter(["2026090810", "2026090810", "2026090811"])
    monkeypatch.setattr(module, "_utc_hour", lambda when=None: next(hours))
    writer = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    writer.write(_game(1))
    first_shard = writer.shard_path
    writer.write(_game(2))
    second_shard = writer.shard_path
    writer.close()

    assert first_shard != second_shard, "the hour turned and the shard did not"
    assert "2026090810" in first_shard.name and "2026090811" in second_shard.name
    assert len(read_shard(first_shard)[0]) == 1
    assert len(read_shard(second_shard)[0]) == 1


def test_a_write_failure_DISABLES_and_COUNTS_and_never_raises(tmp_path: Path) -> None:
    """This writer's loss costs VISIBILITY, not correctness, so it disables itself and counts
    rather than raising or passing; it stays disabled, or one full volume is one error per game.
    MUTATION THAT REDS IT: `except OSError: pass`."""
    writer = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    writer.write(_game(1))
    writer._handle.close()          # the shape an OSError on write leaves behind

    writer.write(_game(2))          # must not raise
    assert writer.persist_errors_total == 1, "the failure must be counted"
    assert writer.games_written == 1, "a failed write must not be billed as a written game"

    writer.write(_game(3))
    assert writer.persist_errors_total == 1, (
        "a disabled store must stop trying, or one full volume becomes one error per game"
    )


def test_an_unsafe_run_id_is_refused_at_the_writer(tmp_path: Path) -> None:
    """The filename law is enforced at THIS boundary too: `games_<run_id>_seg…` has the same
    escape and never-advancing-index failures as the event sink's."""
    for bad in ("", "../escape", "with/slash", " padded "):
        with pytest.raises(RunIdError):
            GameRecordWriter(record_dir=tmp_path, run_id=bad)


def test_the_recorder_writes_one_record_per_game_with_the_actor_step(tmp_path: Path) -> None:
    """`set_step` carries the ACTOR step and `step_kind` says so, rather than leaving a reader to
    assume the learner's. MUTATION THAT REDS IT: stamp `step_kind: "learner"`, or record a game
    drained before `set_step` as step 0 instead of -1."""
    recorder = GameRecorder(record_dir=tmp_path, run_id="testrun", seed=20260719)
    recorder.maybe_record(
        game_id="pre-sync", moves=[(0, 0)], winner_code=0, plies=1, worker_id=0,
        terminal_reason="ply_cap", game_id_byte_hash="0" * 40, served_sims=50)
    recorder.set_step(3000)
    recorder.maybe_record(
        game_id="post-sync", moves=[(0, 0), (1, 0)], winner_code=1, plies=2, worker_id=2,
        terminal_reason="six_in_a_row", game_id_byte_hash="1" * 40, served_sims=50)
    recorder.stop()

    records = list(iter_run_games(tmp_path, "testrun"))
    assert [r["game_id"] for r in records] == ["pre-sync", "post-sync"]
    assert records[0]["step"] == -1, (
        "a game drained before the first actor sync belongs to NO step; 0 would name a step "
        "that really did play games"
    )
    assert records[1]["step"] == 3000
    assert {r["step_kind"] for r in records} == {"actor"}
    assert records[1]["worker_id"] == 2 and records[1]["served_sims"] == 50
    assert records[1]["seed"] == 20260719
    assert "colors" not in records[0], (
        "self-play has no candidate seat; a tautological colors dict would read as information"
    )
    assert "search_stats" not in records[0], (
        "the self-play stats producer does not exist at HEAD — the field must be ABSENT, "
        "never an empty list a reader would take for a measured nothing"
    )


@pytest.mark.parametrize(
    ("winner_code", "expected"),
    [(0, "draw"), (1, "p1"), (2, "p2"), (3, "unknown"), (255, "unknown")],
)
def test_an_undecodable_winner_code_is_unknown_and_NEVER_a_draw(
    tmp_path: Path, winner_code: int, expected: str
) -> None:
    """An undecodable code is `unknown`, never the `draw` a `dict.get(code, "draw")` would land
    on. MUTATION THAT REDS IT: `_SEAT_BY_WINNER_CODE.get(code, "draw")`."""
    recorder = GameRecorder(record_dir=tmp_path, run_id="testrun", seed=1)
    recorder.maybe_record(
        game_id="g", moves=[(0, 0)], winner_code=winner_code, plies=1, worker_id=0,
        terminal_reason="unknown", game_id_byte_hash="0" * 40, served_sims=1)
    recorder.stop()
    assert next(iter(iter_run_games(tmp_path, "testrun")))["result"] == expected


def test_seat_result_maps_every_arena_outcome_through_the_candidate_colour() -> None:
    """The eval winner is CANDIDATE-relative and a board is drawn in SEATS, so an inverted map
    renders every game the candidate played as black back to front."""
    assert seat_result("candidate", 1) == "p1"
    assert seat_result("candidate", -1) == "p2"
    assert seat_result("opponent", 1) == "p2"
    assert seat_result("opponent", -1) == "p1"
    assert seat_result("draw", 1) == "draw"
    assert seat_result("draw", -1) == "draw"


def test_the_production_pool_is_BUILT_with_a_real_recorder() -> None:
    """The recorder producer must exist at step 0, pinned structurally: `compose_run`'s fakes
    cannot reach `build_run_collaborators`, so AST over it requires the `WorkerPool(...)` call to
    bind `recorder=` to a name a `GameRecorder(...)` in the same function produced — substring
    search passes on a comment mention or on `recorder=None`. MUTATION THAT REDS IT: drop the
    `recorder=` kwarg, or build a `NullRecorder`."""
    import ast
    import inspect

    import mantis.run as run_module

    tree = ast.parse(inspect.getsource(run_module))
    builder = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_run_collaborators"
    )

    pool_calls = [
        node for node in ast.walk(builder)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "WorkerPool"
    ]
    assert len(pool_calls) == 1, f"expected one WorkerPool construction, found {len(pool_calls)}"
    bound = {kw.arg: kw.value for kw in pool_calls[0].keywords}
    assert "recorder" in bound, "the pool is built with no recorder — no game is ever written"
    assert isinstance(bound["recorder"], ast.Name), (
        "the recorder must be a name bound in this function, not an inline expression a "
        "reader cannot trace to its constructor"
    )
    recorder_name = bound["recorder"].id

    constructed = {
        target.id
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "GameRecorder"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert recorder_name in constructed, (
        f"`recorder={recorder_name}` is not bound to a GameRecorder(...) in this function"
    )


def test_an_hour_rotation_KEEPS_the_writers_own_segment(tmp_path: Path, monkeypatch) -> None:
    """The segment names the WRITER and the hour names the window, so a rotation that re-scanned
    for `max(segment) + 1` would spread one trainer over three segment numbers a reader reads as
    three processes. MUTATION THAT REDS IT: drop `segment=self._segment` from the claim."""
    import mantis.monitor.game_record as module

    # A settable clock, not a fixed sequence: the number of `_utc_hour` calls is incidental.
    now = {"hour": "2026090810"}
    monkeypatch.setattr(module, "_utc_hour", lambda when=None: now["hour"])

    trainer = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    trainer.write(_game(1))
    child = GameRecordWriter(record_dir=tmp_path, run_id="testrun")   # a round's child
    child.write(_game(2))
    child.close()
    now["hour"] = "2026090811"                                       # the hour turns
    trainer.write(_game(3))
    trainer.close()

    trainer_shards = sorted(
        p.name for p in tmp_path.iterdir() if p.name.startswith("games_testrun_seg0001_")
    )
    assert len(trainer_shards) == 2, (
        f"the trainer's two shards must share ONE segment; found {trainer_shards} beside "
        f"{sorted(p.name for p in tmp_path.iterdir())}"
    )
    assert trainer_shards[0].endswith("_2026090810.jsonl")
    assert trainer_shards[1].endswith("_2026090811.jsonl")
    assert (tmp_path / "games_testrun_seg0002_2026090810.jsonl").exists(), (
        "the round child must still get its OWN segment"
    )
    assert [r["game_id"] for r in iter_run_games(tmp_path, "testrun")] == [
        "g0001", "g0003", "g0002",
    ], "shard order is (segment, hour): the writer first, in its own time order"


def test_a_shard_torn_MID_CHARACTER_is_still_only_one_skipped_line(tmp_path: Path) -> None:
    """A line torn MID-CHARACTER is still only one skipped line: a text-mode reader raises
    `UnicodeDecodeError` while ITERATING, losing every intact record before the tear too.
    MUTATION THAT REDS IT: open the shard in text mode."""
    writer = GameRecordWriter(record_dir=tmp_path, run_id="testrun")
    for i in range(3):
        writer.write(_game(i))
    path = writer.shard_path
    writer.close()

    with path.open("ab") as handle:
        handle.write('{"game_id": "torn", "note": "é'.encode()[:-1] + b"\n")

    records, skipped = read_shard(path)
    assert skipped == 1, f"the torn line and only the torn line; skipped={skipped}"
    assert len(records) == 3, "every intact record before the tear must survive"
