"""An eval round says how far it got, and never reports a default as a measurement.

>300 justify (R8): every row pins ONE surface, the round's self-report, whose two defects were
the same defect at two call sites. A hardcoded `games_total=0` is indistinguishable from a round
that played zero games, so the sentinel is `None`. Observability must not become a new failure
mode: a write failure never breaks a round, a read failure never raises, and escalation
semantics are unchanged — checked structurally, since branching on progress leaves rows green.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mantis.eval.pipeline import emit_round_complete, read_progress
from mantis.eval.worker import _RoundProgress

_PIPELINE = Path(__import__("mantis.eval.pipeline", fromlist=["x"]).__file__)


class _Sink:
    """The event-sink shape `_emit` calls — `.emit(payload)`, not a bare callable."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, payload: dict) -> None:
        self.events.append(payload)


def _game(plies: int = 40, *, winner: str = "draw", terminal: str = "ply_cap",
          candidate_color: int = 1, margin: int | None = None):
    """Build a duck-typed GameRecord stand-in carrying every field the writer reads;
    `margin=None` models both the disarmed posture and a game that never reached the cap."""
    adjudication = None if margin is None else SimpleNamespace(margin=margin)
    return SimpleNamespace(
        plies=plies, moves=[(1, 2)], trajectory_hash="deadbeef",
        winner=winner, terminal=terminal, colors={"candidate": candidate_color,
                                                  "opponent": -candidate_color},
        adjudication=adjudication,
    )


def test_a_broken_round_reports_games_total_None_not_a_countable_zero() -> None:
    """A broken round reports `games_total=None` — `0` is a number a reader will believe, and
    the exact value is asserted because "falsy" admits the `0`."""
    sink = _Sink()
    emit_round_complete(sink, round_id="r1", step=25, wall_sec=3600.0,
                        games_total=None, promoted=False, wr_sealbot=None)
    payload = sink.events[-1]
    assert payload["games_total"] is None, (
        f"a broken round must report games_total=None, got {payload['games_total']!r}. A 0 here "
        f"is a hardcoded default that reads as a measurement — the §8.1 retraction's own cause."
    )


def test_a_successful_round_still_reports_its_real_count() -> None:
    """The sentinel must not cost the success path its number."""
    sink = _Sink()
    emit_round_complete(sink, round_id="r1", step=25, wall_sec=12.5,
                        games_total=88, promoted=True, wr_sealbot=0.61)
    assert sink.events[-1]["games_total"] == 88


def test_the_broken_call_site_passes_None_and_no_literal_zero() -> None:
    """The pipeline's own broken call site passes `None` and no literal zero."""
    tree = ast.parse(_PIPELINE.read_text(encoding="utf-8"))
    passed: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "emit_round_complete":
            continue
        for kw in node.keywords:
            if kw.arg == "games_total":
                passed.append(ast.unparse(kw.value))
    assert "0" not in passed, (
        f"a literal 0 is passed as games_total somewhere in pipeline.py: {passed!r}. The broken "
        f"path must pass None; a 0 is the default-as-measurement defect returning."
    )
    assert "None" in passed, f"no call site passes None; got {passed!r}"


def test_the_child_writes_one_row_per_game_with_counters_only(tmp_path: Path) -> None:
    """The child writes one row per game, safe by construction: counters and a timestamp."""
    path = tmp_path / "r1_progress.txt"
    progress = _RoundProgress(path)
    screen = progress.sink("gate_screen")
    screen(_game(plies=40))
    screen(_game(plies=52))
    progress.sink("rung")(_game(plies=31))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["game_index"] for r in rows] == [1, 2, 3], (
        f"the game index must run monotonically ACROSS phases so a reader can say 'game 3 of N' "
        f"for the whole round; got {rows}"
    )
    assert [r["phase"] for r in rows] == ["gate_screen", "gate_screen", "rung"]
    assert [r["plies"] for r in rows] == [40, 52, 31]
    for row in rows:
        assert set(row) == {"game_index", "phase", "plies", "t_wall",
                            "terminal", "winner", "candidate_color", "margin"}, (
            f"progress rows carry counters, LABELS and a timestamp ONLY — no moves, no "
            f"positions, no trajectory hash, so there is nothing for the redaction pass to "
            f"catch. The set is asserted EXACTLY: a row that grew a `moves` or a "
            f"`trajectory_hash` would satisfy any subset check. Got {row}"
        )


def test_each_row_carries_the_outcome_facts_the_margin_distribution_needs(
    tmp_path: Path,
) -> None:
    """Each row carries the four facts a margin histogram and a seat split need — the
    adjudicator's tally reads the decisive RATE and nothing about its shape."""
    path = tmp_path / "r1_progress.txt"
    sink = _RoundProgress(path).sink("gate_screen")
    sink(_game(plies=128, winner="candidate", candidate_color=1, margin=2))
    sink(_game(plies=128, winner="opponent", candidate_color=-1, margin=-3))
    sink(_game(plies=128, winner="draw", candidate_color=1, margin=0))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["margin"] for r in rows] == [2, -3, 0], (
        f"the SIGNED candidate-minus-opponent margin must survive to the row — its sign is the "
        f"seat-neutrality evidence and its size is the histogram. Got {rows}"
    )
    assert [r["winner"] for r in rows] == ["candidate", "opponent", "draw"]
    assert [r["candidate_color"] for r in rows] == [1, -1, 1], (
        "the seat split is computed from this field; without it the two legs of a colour pair "
        "are indistinguishable in the progress file"
    )
    assert [r["terminal"] for r in rows] == ["ply_cap"] * 3


def test_a_ZERO_margin_and_an_ABSENT_one_do_not_collide(tmp_path: Path) -> None:
    """A measured `margin: 0` and an absent `margin: null` do not collide: `0` is the two
    sides EXACTLY LEVEL, the residual-draw bin, while `null` is no measurement at all."""
    path = tmp_path / "r1_progress.txt"
    sink = _RoundProgress(path).sink("gate_screen")
    sink(_game(plies=128, winner="draw", margin=0))
    sink(_game(plies=128, winner="draw", margin=None))
    sink(_game(plies=61, winner="candidate", terminal="win", margin=None))

    margins = [json.loads(line)["margin"] for line in
               path.read_text(encoding="utf-8").splitlines()]
    assert margins == [0, None, None], (
        f"expected [0, None, None] — a measured tie, a disarmed posture, and a game that never "
        f"reached the cap. Got {margins}; a 0 in position 2 or 3 is the default-as-measurement "
        f"defect, and it would inflate the tie bin the min_margin decision is read from."
    )


def test_the_disarmed_posture_still_writes_a_complete_row(tmp_path: Path) -> None:
    """The disarmed row is the COMMON case and still carries `terminal`, `winner` and seat."""
    path = tmp_path / "r1_progress.txt"
    _RoundProgress(path).sink("gate_confirm")(
        _game(plies=128, winner="draw", terminal="ply_cap", candidate_color=-1, margin=None)
    )
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["terminal"] == "ply_cap" and row["winner"] == "draw"
    assert row["candidate_color"] == -1 and row["margin"] is None


def test_a_record_missing_the_new_fields_degrades_and_never_raises(tmp_path: Path) -> None:
    """An unrecognised record shape writes nulls rather than raising: the writer must not
    become a way to kill a round."""
    path = tmp_path / "r1_progress.txt"
    _RoundProgress(path).sink("rung")(SimpleNamespace(plies=17))
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["plies"] == 17
    assert row["terminal"] is None and row["winner"] is None
    assert row["candidate_color"] is None and row["margin"] is None


def test_the_sink_reads_the_REAL_GameRecord_shape_not_only_the_stand_in() -> None:
    """The writer's field names are the REAL `GameRecord`'s, which the stand-in cannot see."""
    from dataclasses import fields as dataclass_fields

    from mantis.arena.match import GameRecord

    names = {f.name for f in dataclass_fields(GameRecord)}
    assert {"terminal", "winner", "plies", "colors", "adjudication"} <= names, (
        f"the progress writer reads these off a GameRecord; GameRecord now has {sorted(names)}. "
        f"A rename here writes silent nulls into every progress row."
    )


def test_the_verdict_margin_the_writer_reads_is_the_ADJUDICATORS_OWN() -> None:
    """The row's `margin` is pinned to the real adjudicator, so a sign flip or a unit change
    in `PlyCapVerdict` reds here."""
    from mantis.arena.adjudicate import PlyCapVerdict

    verdict = PlyCapVerdict(winner="opponent", criterion="longest_run_margin", margin=-2)
    record = SimpleNamespace(plies=128, winner="opponent", terminal="ply_cap",
                             colors={"candidate": 1, "opponent": -1}, adjudication=verdict)
    captured: list[dict] = []

    class _Capture(_RoundProgress):
        def _write(self, row: dict) -> None:
            captured.append(row)

    _Capture("unused").sink("gate_screen")(record)
    assert captured[0]["margin"] == -2 and captured[0]["winner"] == "opponent"


def test_the_parent_reads_the_LAST_row_back(tmp_path: Path) -> None:
    """The parent reads the LAST row back — how far the round got is the newest row."""
    path = tmp_path / "r1_progress.txt"
    progress = _RoundProgress(path)
    sink = progress.sink("gate_screen")
    for _ in range(5):
        sink(_game())
    assert read_progress(SimpleNamespace(progress_path=str(path)))["game_index"] == 5


def test_a_broken_round_now_CARRIES_how_far_it_got(tmp_path: Path) -> None:
    """A broken round reports `games_total=None` AND the progress it made."""
    path = tmp_path / "r1_progress.txt"
    progress = _RoundProgress(path)
    sink = progress.sink("gate_screen")
    for _ in range(7):
        sink(_game())

    events = _Sink()
    emit_round_complete(events, round_id="r1", step=25, wall_sec=3600.0,
                        games_total=None, promoted=False, wr_sealbot=None,
                        progress=read_progress(SimpleNamespace(progress_path=str(path))))
    payload = events.events[-1]
    assert payload["games_total"] is None
    assert payload["progress"]["game_index"] == 7, (
        "a killed round must report the progress it made. Reporting nothing is what made the "
        "§8.1 error possible; reporting a count would repeat it."
    )


@pytest.mark.parametrize("bad", ["", "   ", "not json\n", '{"a": 1}\nnot json\n'])
def test_an_unreadable_or_partial_progress_file_returns_None_and_never_raises(
    tmp_path: Path, bad: str,
) -> None:
    """A half-written final line is the NORMAL state when the writer was killed mid-round —
    precisely when this is read — so it degrades rather than raising."""
    path = tmp_path / "p.txt"
    path.write_text(bad, encoding="utf-8")
    result = read_progress(SimpleNamespace(progress_path=str(path)))
    assert result is None or isinstance(result, dict)


def test_a_missing_file_and_a_spec_without_the_field_both_return_None(tmp_path: Path) -> None:
    assert read_progress(SimpleNamespace(progress_path=str(tmp_path / "nope.txt"))) is None
    assert read_progress(SimpleNamespace()) is None


def test_a_progress_WRITE_failure_disables_itself_and_never_breaks_the_round(
    tmp_path: Path, capsys,
) -> None:
    """A write failure disables progress, reports LOUD on stderr once, and never breaks the
    round — raising would let a diagnostic line kill a healthy round."""
    blocked = tmp_path / "afile"
    blocked.write_text("i am a file, not a directory", encoding="utf-8")
    progress = _RoundProgress(blocked / "sub" / "p.txt")   # parent mkdir must fail
    sink = progress.sink("gate_screen")
    sink(_game())     # must not raise
    sink(_game())     # must not raise, and must not re-report
    assert capsys.readouterr().err.count("progress writes DISABLED") == 1, (
        "the failure must be reported exactly once — silent would hide it, per-game would "
        "flood a round's stderr with the same line"
    )


def test_progress_is_REPORTING_ONLY_and_no_escalation_branches_on_it() -> None:
    """`read_progress` is consumed only as an ARGUMENT — never tested, compared or branched
    on, because a poller that branched on it would make a reporting field a policy input."""
    tree = ast.parse(_PIPELINE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.Compare, ast.BoolOp, ast.IfExp)):
            if "read_progress" in ast.unparse(node):
                offenders.append(ast.unparse(node)[:120])
    assert not offenders, (
        f"`read_progress` is being branched on in pipeline.py: {offenders!r}. It is REPORTING "
        f"ONLY this sitting — escalation semantics are explicitly unchanged (R319(e)(ii)). "
        f"Making it a policy input is a separate, ruled decision."
    )
