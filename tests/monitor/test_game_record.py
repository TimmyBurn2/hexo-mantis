# >300 justify (R8): the store's producer tests over one format. Writer, reader, shard
# lifecycle, failure posture and the record builders are one contract read from both ends;
# a reader test that lived apart from the writer test it reads back would be testing a
# format nobody wrote.
"""⊕ R344(b) — GAME-RECORD-1's producer tests.

The ruling names two of these itself: *"the producer test asserts the shard closes and
re-opens byte-exact and that a planted truncated shard is skipped, not fatal"*. The rest are
the failures this store would otherwise have to discover in a 25 001-step run:

* the SEGMENT half of the shard key. A shard keyed on `(run, hour)` alone lets a resume
  landing in the same wall-clock hour append to the stopped process's file — the exact TOCTOU
  `monitor/sink.py` earned its `O_CREAT|O_EXCL` claim over. Two writers, one run, one hour
  must produce two shards;
* the FAILURE POSTURE. A lost game record must not end a healthy run, and it must not be
  silent either — so a write failure counts and disables rather than raising or continuing
  quietly;
* the WINNER MAP, because an undecodable outcome recorded as a measured draw is AUDIT-1
  F-28/C04 in a new file.
"""
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
    """R344(b)'s first named producer assertion. Every record written comes back, in order,
    field for field — not "the right number of lines".

    MUTATION THAT REDS IT: buffer the writes and drop the `close()` fsync."""
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
    """R344(b)'s second named producer assertion, planted rather than hoped for.

    A live shard's last line is routinely partial — the reader is running while the writer
    is running. The record before the tear must survive, the torn one must be COUNTED, and
    nothing may raise. Counting is the half that matters: a bare skip cannot tell "one torn
    tail" from "this file is garbage"."""
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
    """The segment half of the key, which the ruling's `(run, hour)` wording leaves out.

    A resume inside the same wall-clock hour is exactly this: a second process, same run id,
    same hour. If the shard were keyed on `(run, hour)` it would append to the stopped
    process's file — the never-append law `monitor/sink.py` states absolutely.

    MUTATION THAT REDS IT: drop `seg` from `shard_filename`, or open with `"a"` instead of
    `O_CREAT|O_EXCL`."""
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
    """The index is written AFTER the fsync, and it carries the count the reader can check.

    An index naming a shard that is not yet on disk is worse than no index: it is a promise a
    reader will act on."""
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
    """The hour rotation R344(b) asks for, driven by a moved clock rather than by waiting.

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
    """LAW-14 says persistence failures are run-fatal; R319(e)(ii) ruled the opposite for a
    writer whose loss costs VISIBILITY and not correctness, and this is that second kind.

    So the posture is neither `raise` nor `pass`: the store disables itself, the failure is
    counted, and `persist_errors_total` is the observable that makes a lost record loud
    (LAW-18). It also stays disabled — retrying once per game into a full volume is how a
    disk-space failure becomes a log-flood failure.

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
    """The filename law is enforced at THIS boundary too, not only at the event sink's —
    `games_<run_id>_seg…` has the same escape and same never-advancing-index failures."""
    for bad in ("", "../escape", "with/slash", " padded "):
        with pytest.raises(RunIdError):
            GameRecordWriter(record_dir=tmp_path, run_id=bad)


# --------------------------------------------------------------------------------------- #
# The recorder — the seam's first concrete implementation
# --------------------------------------------------------------------------------------- #
def test_the_recorder_writes_one_record_per_game_with_the_actor_step(tmp_path: Path) -> None:
    """`set_step` carries the ACTOR step (the weights that played the game), and the record
    says so in `step_kind` rather than leaving a reader to assume it is the learner's step.

    MUTATION THAT REDS IT: stamp `step_kind: "learner"`, or record `self._step` before
    `set_step` has been called as `0` instead of `-1`."""
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
    """AUDIT-1 F-28/C04, in this file's own vocabulary: `pool_drain` already refuses to map
    an unrecognised code onto a real outcome, and the record must not undo that by falling
    to `draw` — the value it would land on if the map were a `dict.get(code, "draw")`.

    MUTATION THAT REDS IT: `_SEAT_BY_WINNER_CODE.get(code, "draw")`."""
    recorder = GameRecorder(record_dir=tmp_path, run_id="testrun", seed=1)
    recorder.maybe_record(
        game_id="g", moves=[(0, 0)], winner_code=winner_code, plies=1, worker_id=0,
        terminal_reason="unknown", game_id_byte_hash="0" * 40, served_sims=1)
    recorder.stop()
    assert next(iter(iter_run_games(tmp_path, "testrun")))["result"] == expected


def test_seat_result_maps_every_arena_outcome_through_the_candidate_colour() -> None:
    """The eval channel's winner is CANDIDATE-relative and a board is drawn in SEATS. Getting
    this inversion wrong renders every game the candidate played as black back to front.

    Hand-checkable: candidate moved first (colour 1) and won -> p1; candidate moved second
    and won -> p2; and each with the opponent winning instead."""
    assert seat_result("candidate", 1) == "p1"
    assert seat_result("candidate", -1) == "p2"
    assert seat_result("opponent", 1) == "p2"
    assert seat_result("opponent", -1) == "p1"
    assert seat_result("draw", 1) == "draw"
    assert seat_result("draw", -1) == "draw"


def test_the_production_pool_is_BUILT_with_a_real_recorder() -> None:
    """R344(b) — *"the producer must exist at step 0"*, pinned at the composition root.

    `compose_run`'s fakes cannot reach this: the recorder is built inside
    `build_run_collaborators`, which constructs a real trainer. So the assertion is STRUCTURAL
    over that function's AST — the `WorkerPool(...)` call must bind `recorder=`, and the name
    it binds must be one a `GameRecorder(...)` call in the same function produced.

    It is AST and not a substring search on purpose: `"GameRecorder" in source` passes on a
    file that only mentions it in a comment, and `"recorder=" in source` passes on
    `recorder=None`.

    MUTATION THAT REDS IT: drop the `recorder=` kwarg, or build a `NullRecorder` instead —
    either of which leaves a run that plays 25 000 games and writes none of them.
    """
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
    """A run writes from two processes into one directory — the trainer continuously, each
    eval round's child for its own lifetime — so the segment must name the WRITER and the
    hour must name the window.

    The defect this pins: if rotation re-scanned for `max(segment) + 1`, a trainer whose hour
    turned after a round child had claimed segment 2 would take segment 3 for itself, and its
    own shards would carry three segment numbers across three hours while a reader took each
    for a different process.

    MUTATION THAT REDS IT: drop `segment=self._segment` from the rotation's claim."""
    import mantis.monitor.game_record as module

    # A settable clock rather than a fixed sequence: the number of `_utc_hour` calls is an
    # implementation detail, and a test that had to predict it would red on a refactor that
    # changed nothing observable.
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
