"""R319(e) — an eval round says how far it got, and never reports a default as a measurement.

**THE TWO DEFECTS, both measured at RECAL-SITTING-3 and both in the same blind spot.**

(i) `emit_round_complete` was called with a HARDCODED `games_total=0` on every broken path,
while the success path computed the real sum. The two are indistinguishable to any reader, so
`games_total: 0` means *"no result file was written"* and reads as *"zero games were played"*.
The sitting published the second having measured only the first, retracted it in §8.1, and the
retraction is the reason this file exists. **A default must not be readable as a measurement.**

(ii) `RoundSpec.progress_path` was constructed, threaded across the spawn seam and declared on
the dataclass — with NO writer and NO reader anywhere in `src/`. A round was observable only as
*started* and *finished/killed*. That is what made (i) available to get wrong: with no progress
signal, `games_total` was the only number in the room, and it was a lie. **A declared field with
no consumer does not remain in the tree.**

**WHAT THIS FILE PINS, and the mutation each row survives without:**
  - `None`, not `0`, on the broken path — the whole of (i); a row asserting merely "not equal to
    the success value" would pass on any other plausible-looking integer;
  - the child WRITES per-game rows, and they carry counters/timestamps ONLY — no moves, no
    positions, no trajectory hash, so the redaction discipline holds by construction rather than
    by filtering;
  - the parent READS the last row back, on the broken path especially — the case the whole
    feature is for;
  - a progress-write failure NEVER breaks a round, and a progress-read failure NEVER raises:
    observability must not become a new failure mode;
  - **escalation semantics are UNCHANGED** (R319(e)(ii)) — checked structurally, because a
    future edit that branched on progress would silently turn a reporting field into a policy
    input, which is exactly the class this sitting keeps finding.
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
    """The event-sink shape `_emit` actually calls (`.emit(payload)`), not a bare callable."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, payload: dict) -> None:
        self.events.append(payload)


def _game(plies: int = 40):
    """A duck-typed GameRecord stand-in: the writer reads `.plies` and nothing else."""
    return SimpleNamespace(plies=plies, moves=[(1, 2)], trajectory_hash="deadbeef")


# ── (e)(i) the sentinel ──────────────────────────────────────────────────────────────────
def test_a_broken_round_reports_games_total_None_not_a_countable_zero() -> None:
    """THE RETRACTED DEFECT, pinned. `0` is a number a reader will average, compare and
    believe; `None` is not. This asserts the exact value, because "some falsy thing" would be
    satisfied by the very `0` that caused the error."""
    sink = _Sink()
    emit_round_complete(sink, round_id="r1", step=25, wall_sec=3600.0,
                        games_total=None, promoted=False, wr_sealbot=None)
    payload = sink.events[-1]
    assert payload["games_total"] is None, (
        f"a broken round must report games_total=None, got {payload['games_total']!r}. A 0 here "
        f"is a hardcoded default that reads as a measurement — the §8.1 retraction's own cause."
    )


def test_a_successful_round_still_reports_its_real_count() -> None:
    """The sentinel must not cost the success path its number, or the fix would be a
    regression wearing a fix's clothes."""
    sink = _Sink()
    emit_round_complete(sink, round_id="r1", step=25, wall_sec=12.5,
                        games_total=88, promoted=True, wr_sealbot=0.61)
    assert sink.events[-1]["games_total"] == 88


def test_the_broken_call_site_passes_None_and_no_literal_zero() -> None:
    """STRUCTURAL (R296(f)). The behavioural row above passes whatever the caller hands it;
    this one pins what the PIPELINE's own broken path actually hands it, so re-introducing the
    literal reds even if every other row still passes."""
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


# ── (e)(ii) progress is written, read, and harmless when it fails ────────────────────────
def test_the_child_writes_one_row_per_game_with_counters_only(tmp_path: Path) -> None:
    """The writer exists at all (it did not before), and what it writes is safe by
    construction: counters and a timestamp, never a move list or a position."""
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
        assert set(row) == {"game_index", "phase", "plies", "t_wall"}, (
            f"progress rows carry counters and a timestamp ONLY — no moves, no positions, no "
            f"trajectory hash, so there is nothing for the redaction pass to catch. Got {row}"
        )


def test_the_parent_reads_the_LAST_row_back(tmp_path: Path) -> None:
    """The round-complete payload needs how far it got, which is the newest row."""
    path = tmp_path / "r1_progress.txt"
    progress = _RoundProgress(path)
    sink = progress.sink("gate_screen")
    for _ in range(5):
        sink(_game())
    assert read_progress(SimpleNamespace(progress_path=str(path)))["game_index"] == 5


def test_a_broken_round_now_CARRIES_how_far_it_got(tmp_path: Path) -> None:
    """The whole point, end to end: the case that produced two blind 3600 s drives.
    `games_total` is None AND the payload still says game 7 was reached."""
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
    """A half-written final line is the NORMAL state of a file being appended to when the
    writer is killed mid-round — precisely when this is read. It must degrade, not raise."""
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
    """OBSERVABILITY MUST NOT BECOME A NEW FAILURE MODE. Deliberately NOT LAW-14's
    persistence-is-fatal posture: losing this file costs visibility, while raising would let a
    diagnostic line kill a round that was otherwise healthy. It fails LOUD on stderr, once."""
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


# ── the boundary R319(e)(ii) draws: reporting, NEVER policy ──────────────────────────────
def test_progress_is_REPORTING_ONLY_and_no_escalation_branches_on_it() -> None:
    """R319(e)(ii): *escalation semantics UNCHANGED this sitting*. Checked structurally,
    because the failure mode is silent: a future edit adding `if progress[...]` to the poller
    or the escalation path would turn a reporting field into a policy input, and every
    behavioural test here would still pass. The rule is that `read_progress` may only be
    consumed as an ARGUMENT — never tested, compared, or branched on."""
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
