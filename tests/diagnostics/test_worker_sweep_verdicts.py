"""Rung verdicts, the OOM extension-stop, the ladder walk, and the rounds that were not measured.

>300 justify (R8): ONE SUBJECT — what a rung's verdict IS, and what the ladder does with it.
The verdict rows and the `walk_ladder` rows cannot be separated without one of them losing its
meaning: the OOM rows are simultaneously a statement about a rung (it fails) and about the ladder
(only the EXTENSION stops), the thread-bound row is about which rungs are RUN, and the
termination row exists because a verdict token was added without the walk's predicate being
updated. Splitting by "verdict" and "walk" would put each half of every R309(f) clause in a
different file, and the clauses are what this suite is about. The `_round`/`_rung` builders are
shared for `0bb4381`'s reason: a fixture and the rows that read it move together or the rows
quietly stop testing what their names say.

THE DISCIPLINE THIS SUITE PINS is `mantis.diagnostics.eval_child_memory`'s, carried onto a
second instrument: a verdict comes from a STATED STOPPING RULE applied to a SERIES, REFUSED is
never a verdict, and a round nobody could measure is listed and named rather than dropped.

The 2026-08-22 sitting is why. `eval_child` looked converged at 41 samples and at 709 and was
2.98x larger the first time a round was allowed to complete (`RECAL_EXIT_2026-08-22.md` §11b).
The lesson generalises past that one term: *a term measured by watching until it looks flat is
not a bound*, and a sweep that picked a worker count off a rung whose memory was still climbing
would be the same defect with a different subject.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.diagnostics import worker_sweep as ws

_PLAN = Path(__file__).resolve().parents[2] / "tools" / "worker_sweep_plan.toml"
_MIB = 1024 ** 2


@pytest.fixture()
def plan() -> ws.SweepPlan:
    return ws.load_plan(_PLAN)


def _round(index: int, *, peak_mib: float | None, warmup: bool = False,
           moves: int = 1000, games: int = 3) -> ws.RoundReading:
    return ws.RoundReading(
        index=index, warmup=warmup, wall_sec=120.0, games=games, moves=moves,
        available=peak_mib is not None,
        sampled_peak_bytes=None if peak_mib is None else int(peak_mib * _MIB),
        allocator_peak_bytes=None if peak_mib is None else int(peak_mib * _MIB * 0.9),
        card_samples=0 if peak_mib is None else 120,
    )


def _rung(n_workers: int, peaks: list[float | None], *, moves: int = 1000,
          plan: ws.SweepPlan) -> ws.RungResult:
    rounds = [_round(0, peak_mib=peaks[0], warmup=True, moves=moves)]
    rounds += [_round(i + 1, peak_mib=p, moves=moves) for i, p in enumerate(peaks[1:])]
    verdict, refusal = ws._verdict_for(tuple(rounds), plan)
    return ws.RungResult(n_workers=n_workers, verdict=verdict, rounds=tuple(rounds),
                         refusal=refusal, produced_by="test")


# ══ the stopping rule is the IMPORTED one ════════════════════════════════════════════════
def test_the_stopping_rule_is_the_eval_child_instrument_s_own_function() -> None:
    """One stopping rule in this tree, one place it can be wrong. If this ever stops being
    true, two instruments can disagree about what "plateau" means on the same series."""
    from mantis.diagnostics import eval_child_memory

    assert ws.classify is eval_child_memory.classify
    assert ws.PLATEAU == eval_child_memory.PLATEAU
    assert ws.GROWING == eval_child_memory.GROWING
    assert ws.RC_REFUSED == eval_child_memory.RC_REFUSED


def test_a_flat_series_plateaus(plan: ws.SweepPlan) -> None:
    rung = _rung(4, [900.0, 1000.0, 1002.0, 1001.0, 1003.0, 1000.0], plan=plan)
    assert rung.verdict == ws.PLATEAU


# ══ P2 — a GROWING rung is excluded from the knee set ════════════════════════════════════
def test_a_rising_series_verdicts_growing_and_is_excluded_from_the_knee_set(
    plan: ws.SweepPlan,
) -> None:
    """The rise is INSIDE the trailing window on purpose: growth outside it is history, and a
    series that ever rose could otherwise never converge."""
    growing = _rung(8, [900.0, 1000.0, 1000.0, 1400.0, 1900.0, 2600.0], moves=4000, plan=plan)
    flat = _rung(2, [900.0, 1000.0, 1000.0, 1001.0, 1000.0, 1002.0], moves=1000, plan=plan)
    assert growing.verdict == ws.GROWING
    rows = [r.as_dict(plan.metric) for r in (flat, growing)]
    selection = ws.select_knee(rows, knee_pct=plan.knee_pct, metric=plan.metric, noise_floor_rel_std=0.0)
    assert [p["n_workers"] for p in selection["passing"]] == [2]
    assert selection["picked"] == 2, (
        "the GROWING rung is the faster one; if it can be picked, the memory verdict is "
        "decorative and the knee rule ranks on throughput alone"
    )


# ══ REFUSED is never a verdict ═══════════════════════════════════════════════════════════
def test_a_host_with_no_counters_refuses_rather_than_reporting_a_plateau(
    plan: ws.SweepPlan,
) -> None:
    """`0 rounds, plateau` is the reading this whole family exists to refuse."""
    rung = _rung(4, [None, None, None, None, None, None], plan=plan)
    assert rung.verdict == ws.REFUSED
    assert rung.refusal and "measured rounds" in rung.refusal


def test_unmeasured_rounds_are_listed_and_counted_not_dropped(plan: ws.SweepPlan) -> None:
    """Silently dropping them would bias the series without saying so, and "we had no counters
    that round" is itself a finding about the drive."""
    rung = _rung(4, [900.0, 1000.0, None, 1001.0, None, 1002.0], plan=plan)
    row = rung.as_dict(plan.metric)
    assert row["rounds_total"] == 6
    assert row["rounds_measured"] == 3
    assert row["rounds_unmeasured"] == 2  # the warm-up round is not in the scored set
    assert len(row["rounds"]) == 6
    assert [r["available"] for r in row["rounds"]] == [True, True, False, True, False, True]


# ══ the two sinks ════════════════════════════════════════════════════════════════════════
def test_the_larger_of_the_two_sinks_governs() -> None:
    """The box block's standing rule, which needs BOTH numbers to be applied at all."""
    reading = ws.RoundReading(index=0, warmup=False, wall_sec=60.0, games=1, moves=10,
                              available=True, sampled_peak_bytes=9 * _MIB,
                              allocator_peak_bytes=4 * _MIB, card_samples=60)
    assert reading.governing_peak_bytes == 9 * _MIB
    inverted = ws.RoundReading(index=0, warmup=False, wall_sec=60.0, games=1, moves=10,
                               available=True, sampled_peak_bytes=4 * _MIB,
                               allocator_peak_bytes=9 * _MIB, card_samples=60)
    assert inverted.governing_peak_bytes == 9 * _MIB


# ══ P3 — a synthetic OOM fails its rung and stops the EXTENSION, as the register writes it ══
def test_an_oom_fails_its_own_rung_and_stops_only_the_extension(plan: ws.SweepPlan) -> None:
    """R309(f) VERBATIM: *"an OOM at a rung is data that fails the rung and stops the ladder's
    EXTENSION, never a sitting failure."* The base rungs above the OOM are STILL WALKED.

    An earlier cut stopped the whole ladder, on the plausible physical argument that a higher
    rung OOMs harder. Plausible is not the test: this is a pre-registered operator row closed
    with "no post-hoc movement", so the widening is FILED as an adjudication (F-WS-2) and the
    clause is implemented as written until it comes back. This row is what would red if someone
    took it quietly."""
    calls: list[int] = []

    def runner(n_workers: int) -> ws.RungResult:
        calls.append(n_workers)
        if n_workers == 8:
            return ws.RungResult(n_workers=n_workers, verdict=ws.OOM, rounds=(),
                                 refusal="synthetic OOM", produced_by="test")
        return _rung(n_workers, [900.0, 1000.0, 1000.0, 1001.0, 1000.0, 1002.0], plan=plan)

    results, stopped = ws.walk_ladder(plan, runner=runner, label="test")
    assert calls == [2, 4, 8, 12, 14], (
        "the base ladder must be walked WHOLE — R309(f) stops the EXTENSION at an OOM, not the "
        f"ladder; rungs actually run: {calls}"
    )
    verdicts = {r.n_workers: r.verdict for r in results}
    assert verdicts[8] == ws.OOM
    assert verdicts[12] == verdicts[14] == ws.PLATEAU
    assert "EXTENSION" in stopped and "8" in stopped
    rows = [r.as_dict(plan.metric) for r in results]
    selection = ws.select_knee(rows, knee_pct=plan.knee_pct, metric=plan.metric, noise_floor_rel_std=0.0)
    assert selection["picked"] == 2
    assert ws.rc_for({"rungs": rows, "selection": selection}) == 0, (
        "the sweep SURVIVES the OOM — an OOM is data that fails a rung, never a sitting failure"
    )


def test_an_oom_anywhere_closes_the_extension_even_if_the_top_rung_passed(
    plan: ws.SweepPlan,
) -> None:
    def runner(n_workers: int) -> ws.RungResult:
        if n_workers == 4:
            return ws.RungResult(n_workers=n_workers, verdict=ws.OOM, rounds=(),
                                 refusal="synthetic OOM", produced_by="test")
        # every later rung faster than the last, so ONLY the OOM can stop the extension
        return _rung(n_workers, [900.0, 1000.0, 1000.0, 1001.0, 1000.0, 1002.0],
                     moves=1000 * n_workers, plan=plan)

    results, stopped = ws.walk_ladder(plan, runner=runner, label="test")
    assert [r.n_workers for r in results] == [2, 4, 8, 12, 14], "no extension rung may be run"
    assert "EXTENSION" in stopped


# ══ D-11 — extension starts above the HIGHEST rung run, never above the best passing one ══
def test_extension_never_proposes_a_rung_inside_the_base_bracket(plan: ws.SweepPlan) -> None:
    """With 2, 4, 8 PLATEAU and 12, 14 GROWING, "best passing + step" would propose 10 — a rung
    inside the base bracket, below two rungs already measured as failing, and one neither
    R309(f) ("extension past 14") nor R309(g) (the base ladder) names at all."""
    def runner(n_workers: int) -> ws.RungResult:
        peaks = ([900.0, 1000.0, 1000.0, 1001.0, 1000.0, 1002.0] if n_workers <= 8
                 else [900.0, 1000.0, 1000.0, 1400.0, 1900.0, 2600.0])
        return _rung(n_workers, peaks, moves=1000 * n_workers, plan=plan)

    results, stopped = ws.walk_ladder(plan, runner=runner, label="test")
    assert [r.n_workers for r in results] == [2, 4, 8, 12, 14]
    assert "did not PASS" in stopped
    assert all(r.n_workers % 2 == 0 and r.n_workers in plan.rungs for r in results), (
        f"an extension rung inside the base bracket was generated: "
        f"{[r.n_workers for r in results]}"
    )


def test_extension_walks_past_the_top_of_the_ladder_while_gains_persist(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ws, "thread_bound", lambda: (64, "os.cpu_count()"))
    def runner(n_workers: int) -> ws.RungResult:
        # 20% per rung while <= 18, then flat: the gain floor is what must stop it.
        moves = 1000 * min(n_workers, 18)
        return _rung(n_workers, [900.0, 1000.0, 1000.0, 1001.0, 1000.0, 1002.0],
                     moves=moves, plan=plan)

    results, stopped = ws.walk_ladder(plan, runner=runner, label="test")
    ran = [r.n_workers for r in results]
    assert ran[:5] == [2, 4, 8, 12, 14]
    assert ran[5:] and min(ran[5:]) > 14, f"extension must go PAST 14; got {ran}"
    assert "gains no longer persist" in stopped


def test_a_real_cuda_oom_inside_a_rung_is_caught_and_becomes_that_rung_s_verdict(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The synthetic break is raised where a real one lands — inside the rung's construction,
    which is where `F-R302-1` actually OOM'd (the GNN training forward, 3.3 s after the first
    game). If the guard did not cover the BUILD, that OOM would escape as a crash and the
    ladder would be lost rather than truncated."""
    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise torch.OutOfMemoryError("CUDA out of memory (synthetic)")

    monkeypatch.setattr(ws, "build_sweep_pool", explode)
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: False)
    with (tmp_path / "sweep.log").open("w", encoding="utf-8") as handle:
        result = ws.drive_rung(object(), plan, n_workers=8, device=torch.device("cpu"),
                               label="test", out=handle)
    assert result.verdict == ws.OOM
    assert "synthetic" in (result.refusal or "")


# ══ a rung that produced nothing is refused, not reported as zero ════════════════════════
def test_a_rung_that_generated_no_moves_is_refused_with_its_sampling_limit(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A round shorter than the work it samples reports 0, and 0 from a healthy rung is a
    sampling limit, not a measurement. Reporting it as throughput would rank a working rung
    last."""
    class _Stats:
        games_completed = 0
        positions_generated = 0

    class _Pool:
        _producer_exc = None
        # R317(c)(i): drive_rung hashes `pool.model` right after the build; a mock pool needs one.
        model = type("_NoParams", (), {"state_dict": lambda self: {}})()

        def start(self) -> None: ...
        def stop(self) -> None: ...
        def check_producer_health(self) -> None: ...

    monkeypatch.setattr(ws, "build_sweep_pool", lambda *a, **k: _Pool())
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: False)
    monkeypatch.setattr(ws, "runner_stats", lambda _p: _Stats())
    log = tmp_path / "sweep.log"
    with log.open("w", encoding="utf-8") as handle:
        result = ws.drive_rung(object(), plan, n_workers=2, device=torch.device("cpu"),
                               label="test", out=handle, sleep=lambda _s: None)
    assert result.verdict == ws.REFUSED
    assert "NO moves" in (result.refusal or "")
    assert "sampling limit" in (result.refusal or "")


# ══ the ladder respects the box, and says which bound stopped it ═════════════════════════
def test_the_base_bracket_is_walked_WHOLE_even_on_a_box_with_fewer_threads(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R309(f) attaches the thread bound to the EXTENSION; R309(g) fixes the base ladder. An
    earlier cut skipped base rungs above the bound, which on an 8-vCPU instance silently reduced
    a pre-registered bracket to `2, 4, 8` — an executor narrowing a pre-registered clause on a
    plausible physical argument, which is the shape the design review sent back once already."""
    monkeypatch.setattr(ws, "thread_bound", lambda: (4, "os.sched_getaffinity(0)"))
    ran: list[int] = []

    def runner(n: int) -> ws.RungResult:
        ran.append(n)
        return _rung(n, [900.0, 1000.0, 1000.0, 1001.0, 1000.0, 1002.0], moves=1000 * n,
                     plan=plan)

    results, stopped = ws.walk_ladder(plan, runner=runner, label="test")
    assert ran == [2, 4, 8, 12, 14], f"the pre-registered bracket was truncated: {ran}"
    assert [r.n_workers for r in results] == [2, 4, 8, 12, 14]
    assert "thread bound 4" in stopped and "sched_getaffinity" in stopped, (
        "the EXTENSION must still be bounded by the measured thread count, and say so"
    )


@pytest.mark.parametrize("verdict", ["RUNG_ERROR", "PRODUCER_DEAD"])
def test_the_ladder_TERMINATES_when_an_extension_rung_fails(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch, verdict: str,
) -> None:
    """THE DEFECT THIS CLOSES DID NOT TERMINATE. The extension chose the highest rung whose
    verdict was in an ENUMERATED set; a later verdict token was added and not added to that set,
    so the same extension rung was proposed and re-driven forever — a fourteen-minute pool build
    and teardown per iteration, unbounded, on a rented box, with no report ever written.

    The predicate is now "the last rung run", which needs no enumeration to stay correct. This
    row is parametrized over the two tokens that broke it and would break again on a third."""
    monkeypatch.setattr(ws, "thread_bound", lambda: (64, "os.cpu_count()"))
    ran: list[int] = []

    def runner(n: int) -> ws.RungResult:
        ran.append(n)
        assert len(ran) <= 12, f"walk_ladder did not terminate: {ran}"
        if n > 14:
            return ws.RungResult(n_workers=n, verdict=getattr(ws, verdict), rounds=(),
                                 refusal="synthetic", produced_by="t")
        return _rung(n, [900.0, 1000.0, 1000.0, 1001.0, 1000.0, 1002.0],
                     moves=1000 * min(n, 18), plan=plan)

    results, stopped = ws.walk_ladder(plan, runner=runner, label="test")
    assert ran == [2, 4, 8, 12, 14, 16], f"the extension re-drove a rung: {ran}"
    assert results[-1].verdict == getattr(ws, verdict)
    assert "did not PASS" in stopped and getattr(ws, verdict) in stopped


def test_the_stated_stop_reason_is_true_when_every_base_rung_fails(
    plan: ws.SweepPlan,
) -> None:
    """The one field a reader consults for WHY the ladder ended used to assert a thread bound
    that did not apply — five rungs run on a 16-thread box, reported as "no rung was run: every
    base rung is above the measured thread bound 16". A ladder that states the wrong why is
    worse than one that states none."""
    results, stopped = ws.walk_ladder(
        plan,
        runner=lambda n: ws.RungResult(n_workers=n, verdict=ws.RUNG_ERROR, rounds=(),
                                       refusal="synthetic", produced_by="t"),
        label="test")
    assert [r.n_workers for r in results] == [2, 4, 8, 12, 14]
    assert "thread bound" not in stopped
    assert "did not PASS" in stopped and ws.RUNG_ERROR in stopped


def test_the_thread_bound_is_measured_and_names_which_call_answered() -> None:
    bound, source = ws.thread_bound()
    assert bound >= 1
    assert source in ("os.sched_getaffinity(0)", "os.cpu_count()")


# ══ the plan's own toml is the one the suite reasons about ═══════════════════════════════
def test_the_plan_fixture_is_the_committed_file_and_not_a_copy() -> None:
    assert tomllib.loads(_PLAN.read_text(encoding="utf-8"))["provenance"]["prereg_ruling"] \
        == "R309(f)"
