"""Item 5 pins — promotion integrity: the right incumbent, both colour legs, one finalise.

Three ways the promotion bar could be wrong while every round still completed, reported a
win rate, and promoted. None of them raise; all three are silent.

  (a) WRONG INCUMBENT. `resolve_anchor` defaulted `best_model_path` to a CWD-relative
      `checkpoints/best_model.pt`, and `train/loop.py` passed nothing — while the promotion
      WRITE side got the run's real `<out-dir>/checkpoints/best_model.pt`. Read and write
      named different files for any run not launched from the repo root.
  (b) A DROPPED COLOUR LEG. `play_paired_match` plays every opening twice with the colours
      swapped, but `trajectory_hash` is a sha256 over the MOVE LIST alone. When the two
      legs' move sequences coincide — routine under argmax/temp-0 from a fixed opening,
      the exact regime LAW-04 exists for — they hashed identically and LAW-04's dedupe threw
      one away. The kept leg supplies the outcome, so the WR skews to whichever arrived
      first, on half the eff_n: LAW-04's remedy corrupting the LAW-15 bar it feeds.
  (c) A DOUBLE FINALISE. The poll loop and the drain both read `self._inflight` and both
      finalise it; `self._inflight` is not cleared until the end. One round's games could be
      appended, persisted and gated twice.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mantis.eval.aggregate import aggregate_rung
from mantis.train.anchor import canonical_anchor_path, resolve_anchor


# ── (a) the anchor read and the promotion write name the same file ─────────────────────


def test_resolve_anchor_refuses_to_invent_an_anchor_path() -> None:
    """No CWD fallback. A wrong incumbent is silently wrong, so absence must be loud."""
    with pytest.raises(ValueError, match="best_model_path"):
        resolve_anchor(trainer=SimpleNamespace(), eval_pipeline=object())


def test_anchor_path_is_derived_from_the_runs_checkpoint_dir(tmp_path: Path) -> None:
    got = canonical_anchor_path(tmp_path / "checkpoints")
    assert got == tmp_path / "checkpoints" / "best_model.pt"
    assert got.is_absolute(), (
        "a relative anchor path is the defect: the run evaluates against one file and "
        "promotes into another"
    )


def test_the_run_root_reads_and_writes_the_anchor_through_one_helper() -> None:
    """Source census: `run.py` must not spell the anchor filename twice.

    The read path (`run_training_loop(best_model_path=)`) and the write path
    (`DeployTagHooks(best_model_path=)`) are DIFFERENT call sites. Two literals is how they
    drifted apart in the first place, so the pin is that neither writes one.
    """
    text = (Path(__file__).resolve().parents[2] / "src" / "mantis" / "run.py").read_text(
        encoding="utf-8")
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if '"best_model.pt"' in ln], (
        "run.py spells the anchor filename literally; both the anchor READ and the "
        "promotion WRITE must go through canonical_anchor_path()"
    )
    assert sum(1 for ln in code if "canonical_anchor_path(" in ln) >= 2, (
        "both the read side (run_training_loop) and the write side (DeployTagHooks) must "
        "derive the anchor path from the same helper"
    )


def test_the_training_loop_forwards_the_anchor_path() -> None:
    """`train/loop.py` must PASS it — passing nothing is what selected the old default."""
    seen: dict[str, Any] = {}

    def _spy(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(best_model=None, best_model_step=None,
                               best_model_path=Path("x"), representation="graph")

    from mantis.train.loop import run_training_loop

    run_training_loop(
        trainer=SimpleNamespace(step=0, save_checkpoint=lambda _l: None),
        shutdown_state=SimpleNamespace(running=False, shutdown_save=False, abort_rule=None),
        eval_pipeline=object(), resolve_anchor=_spy, max_steps=0,
        best_model_path="/runs/r5/checkpoints/best_model.pt",
    )
    assert seen.get("best_model_path") == "/runs/r5/checkpoints/best_model.pt", (
        f"the loop did not forward best_model_path to resolve_anchor; got {seen!r}"
    )


# ── (b) both colour legs survive LAW-04 dedupe ─────────────────────────────────────────


def _leg(colour: int, winner: str) -> dict[str, Any]:
    """One leg of a colour pair: IDENTICAL trajectory, opposite seat and opposite result.

    This is not a contrived record — it is what a deterministic (argmax / temp-0) paired
    match from a fixed opening produces whenever the two legs' move sequences coincide.
    """
    return {
        "p1": "cand", "p2": "opponent", "winner": winner,
        "trajectory_hash": "IDENTICAL-ACROSS-BOTH-LEGS",
        "candidate_color": colour,
        "regime_key": "r",
    }


def test_a_colour_pair_counts_as_two_distinct_games() -> None:
    """LAW-04 dedupes COPIES; the two legs of a colour pair are not copies."""
    got = aggregate_rung([_leg(1, "p1"), _leg(-1, "p2")])
    assert got.eff_n == 2, (
        "the two legs of a colour pair collapsed into one distinct game. The surviving "
        f"leg's outcome then supplies the pair's value — a biased WR on half the eff_n. "
        f"eff_n={got.eff_n}"
    )
    assert got.wr == pytest.approx(0.5), (
        "with one win and one loss the draw-aware WR is 0.5; a value of 1.0 or 0.0 means "
        "one leg was dropped and the other spoke for the pair"
    )


def test_genuine_copies_still_collapse() -> None:
    """The mutation half. Without it, the fix above could be 'never dedupe anything'.

    Mechanism: two records identical in trajectory AND seat are true copies — the case
    LAW-04 exists for (a deterministic regime replaying one game inflates a CI by
    sqrt(copies)). They must still count once.
    """
    got = aggregate_rung([_leg(1, "p1"), _leg(1, "p1")])
    assert got.eff_n == 1, (
        f"true copies stopped collapsing — LAW-04's dedupe is disabled, not fixed "
        f"(eff_n={got.eff_n})"
    )


def test_legacy_records_without_a_seat_are_unaffected() -> None:
    """Additive: a record that never carried a seat cannot start colliding because of one."""
    a = {"p1": "cand", "p2": "opponent", "winner": "p1", "moves": [[0, 0]], "regime_key": "r"}
    b = dict(a)
    assert aggregate_rung([a, b]).eff_n == 1


def test_the_eval_worker_actually_emits_the_seat() -> None:
    """LAW-07: the dedupe key is only fixed if a live producer supplies the field.

    Without this, `_traj_key` reads `candidate_color`, every real record omits it, and the
    key qualifier is a constant `None` — the fix would be inert on the production path and
    green in every unit test that builds records by hand.
    """
    from mantis.eval.worker import _agg_record

    record = _agg_record(SimpleNamespace(
        winner="candidate", moves=((0, 0), (1, 1)),
        regime_key=SimpleNamespace(canonical=lambda: "r"),
        trajectory_hash="h", colors={"candidate": -1, "opponent": 1},
    ))
    assert record["candidate_color"] == -1, (
        "the eval worker drops the seat, so every production record keys the same as its "
        "colour twin and the LAW-04 fix never fires in a real round"
    )


# ── (c) a round finalises exactly once ─────────────────────────────────────────────────


def _pipeline(monkeypatch: pytest.MonkeyPatch, reads: list[str]) -> Any:
    """A real `EvalPipeline` with only the two collaborators a finalise reaches stubbed.

    `__new__` without `__init__` on purpose: the guard is pure state + lock, and booting a
    full pipeline would drag in a worker process, a ladder file and a spool dir — none of
    which the once-only latch touches. `_read_worker_result` and `unregister_child` are the
    only things the real method calls out to on the success path, and `reads` records that
    the ROUND BODY ran, which is what a double finalise duplicates.
    """
    import mantis.train.lifecycle.signals as signals
    from mantis.eval.pipeline import EvalPipeline

    pipe = EvalPipeline.__new__(EvalPipeline)
    pipe._lock = threading.Lock()
    pipe._inflight = None
    pipe._mailbox = []
    pipe._sink = None
    pipe._double_finalize_suppressed = 0
    pipe._clock = lambda: 0.0

    monkeypatch.setattr(signals, "unregister_child", lambda _p: None)
    monkeypatch.setattr(
        EvalPipeline, "_read_worker_result",
        lambda self, inflight, **_kw: (reads.append(inflight["round_id"]),
                                       {"round_id": inflight["round_id"], "ok": True})[1])
    return pipe


def _inflight(round_id: str) -> dict[str, Any]:
    return {"round_id": round_id, "step": 7, "proc": SimpleNamespace(exitcode=0), "t0": 0.0}


def test_a_second_finalise_of_the_same_round_is_suppressed_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two routes can hold the same inflight dict; only the first may take effect.

    Drives the REAL `EvalPipeline._finalize_round` twice on one dict — the situation the
    poll loop and a concurrent drain actually produce.
    """
    from mantis.eval.pipeline import EvalPipeline

    reads: list[str] = []
    pipe = _pipeline(monkeypatch, reads)
    inflight = _inflight("r1")

    first = EvalPipeline._finalize_round(pipe, inflight)
    second = EvalPipeline._finalize_round(pipe, inflight)

    assert reads == ["r1"], (
        f"the round body ran {len(reads)} times — a double finalise re-reads the worker "
        "result, re-appends to the mailbox and re-runs the gate on one round's games"
    )
    assert first == {"round_id": "r1", "ok": True}
    assert second == first, "the suppressed call must return the first finalise's result"
    assert len(pipe._mailbox) == 1, (
        "the round's result was appended to the mailbox twice — one round's games would be "
        "gated as two"
    )
    assert pipe._double_finalize_suppressed == 1, (
        "the suppression was not COUNTED; a silent suppression hides that two routes are "
        "racing (LAW-18)"
    )


def test_the_guard_is_per_round_not_per_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation self-test for the latch's LOCATION.

    Mechanism: a `self`-level flag would have to be reset between rounds, and a missed reset
    silently disables every later finalise. Two DIFFERENT rounds must both run their body.
    """
    from mantis.eval.pipeline import EvalPipeline

    reads: list[str] = []
    pipe = _pipeline(monkeypatch, reads)
    for round_id in ("r1", "r2"):
        EvalPipeline._finalize_round(pipe, _inflight(round_id))

    assert reads == ["r1", "r2"], (
        f"a second, DIFFERENT round was suppressed — the latch is not per-round; got {reads}"
    )
    assert pipe._double_finalize_suppressed == 0, "no suppression should have been counted"
