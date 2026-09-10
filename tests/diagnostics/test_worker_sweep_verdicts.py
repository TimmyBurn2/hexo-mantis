"""Pin rung verdicts, the OOM extension-stop, the ladder walk, and the unmeasured rounds.

>300 justify (R8): an OOM row is simultaneously about a rung (it fails) and about the ladder (only
the EXTENSION stops), so splitting by "verdict" and "walk" would put each half of every clause in
a different file. A verdict comes from a STATED stopping rule on a SERIES, REFUSED is never a
verdict, and an unmeasurable round is listed rather than dropped.
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


def test_the_stopping_rule_is_the_eval_child_instrument_s_own_function() -> None:
    """Prove the stopping rule is the imported one, so no two copies can disagree."""
    from mantis.diagnostics import eval_child_memory

    assert ws.classify is eval_child_memory.classify
    assert ws.PLATEAU == eval_child_memory.PLATEAU
    assert ws.GROWING == eval_child_memory.GROWING
    assert ws.RC_REFUSED == eval_child_memory.RC_REFUSED


def test_a_flat_series_plateaus(plan: ws.SweepPlan) -> None:
    rung = _rung(4, [900.0, 1000.0, 1002.0, 1001.0, 1003.0, 1000.0], plan=plan)
    assert rung.verdict == ws.PLATEAU


def test_a_rising_series_verdicts_growing_and_is_excluded_from_the_knee_set(
    plan: ws.SweepPlan,
) -> None:
    """Prove a rising series verdicts GROWING and is excluded from the knee set; the rise is
    inside the trailing window, or a series that ever rose could never converge."""
    growing = _rung(8, [900.0, 1000.0, 1000.0, 1400.0, 1900.0, 2600.0], moves=4000, plan=plan)
    flat = _rung(2, [900.0, 1000.0, 1000.0, 1001.0, 1000.0, 1002.0], moves=1000, plan=plan)
    assert growing.verdict == ws.GROWING
    rows = [r.as_dict(plan.metric) for r in (flat, growing)]
    selection = ws.select_knee(rows, knee_pct=plan.knee_pct, metric=plan.metric)
    assert [p["n_workers"] for p in selection["passing"]] == [2]
    assert selection["picked"] == 2, (
        "the GROWING rung is the faster one; if it can be picked, the memory verdict is "
        "decorative and the knee rule ranks on throughput alone"
    )


def test_a_host_with_no_counters_refuses_rather_than_reporting_a_plateau(
    plan: ws.SweepPlan,
) -> None:
    """Prove a host with no counters refuses rather than reporting `0 rounds, plateau`."""
    rung = _rung(4, [None, None, None, None, None, None], plan=plan)
    assert rung.verdict == ws.REFUSED
    assert rung.refusal and "measured rounds" in rung.refusal


def test_unmeasured_rounds_are_listed_and_counted_not_dropped(plan: ws.SweepPlan) -> None:
    """Prove unmeasured rounds are listed and counted, not silently dropped from the series."""
    rung = _rung(4, [900.0, 1000.0, None, 1001.0, None, 1002.0], plan=plan)
    row = rung.as_dict(plan.metric)
    assert row["rounds_total"] == 6
    assert row["rounds_measured"] == 3
    assert row["rounds_unmeasured"] == 2  # the warm-up round is not in the scored set
    assert len(row["rounds"]) == 6
    assert [r["available"] for r in row["rounds"]] == [True, True, False, True, False, True]


def test_the_larger_of_the_two_sinks_governs() -> None:
    """Prove the larger of the two sinks governs, which needs both numbers recorded."""
    reading = ws.RoundReading(index=0, warmup=False, wall_sec=60.0, games=1, moves=10,
                              available=True, sampled_peak_bytes=9 * _MIB,
                              allocator_peak_bytes=4 * _MIB, card_samples=60)
    assert reading.governing_peak_bytes == 9 * _MIB
    inverted = ws.RoundReading(index=0, warmup=False, wall_sec=60.0, games=1, moves=10,
                               available=True, sampled_peak_bytes=4 * _MIB,
                               allocator_peak_bytes=9 * _MIB, card_samples=60)
    assert inverted.governing_peak_bytes == 9 * _MIB


def test_an_oom_fails_its_own_rung_and_stops_only_the_extension(plan: ws.SweepPlan) -> None:
    """Prove an OOM fails its own rung and stops only the EXTENSION; base rungs above it are
    still walked, whatever the plausible physical argument for stopping the whole ladder."""
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
    selection = ws.select_knee(rows, knee_pct=plan.knee_pct, metric=plan.metric)
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


def test_extension_never_proposes_a_rung_inside_the_base_bracket(plan: ws.SweepPlan) -> None:
    """Prove the extension starts above the HIGHEST rung run: "best passing + step" would propose
    a rung inside the base bracket, below two already measured as failing."""
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
    """Prove a real CUDA OOM inside a rung's CONSTRUCTION becomes that rung's verdict."""
    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise torch.OutOfMemoryError("CUDA out of memory (synthetic)")

    monkeypatch.setattr(ws, "build_sweep_pool", explode)
    monkeypatch.setattr(ws, "cuda_counters_available", lambda _d: False)
    with (tmp_path / "sweep.log").open("w", encoding="utf-8") as handle:
        result = ws.drive_rung(object(), plan, n_workers=8, device=torch.device("cpu"),
                               label="test", out=handle)
    assert result.verdict == ws.OOM
    assert "synthetic" in (result.refusal or "")


def test_a_rung_that_generated_no_moves_is_refused_with_its_sampling_limit(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Prove a rung that generated no moves is refused with its sampling limit named, since
    reporting the 0 as throughput would rank a working rung last."""
    class _Stats:
        games_completed = 0
        positions_generated = 0

    class _Pool:
        _producer_exc = None
        # drive_rung hashes `pool.model` right after the build, so a mock pool needs one.
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


def test_the_base_bracket_is_walked_WHOLE_even_on_a_box_with_fewer_threads(
    plan: ws.SweepPlan, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the base bracket is walked whole on a box with fewer threads: the bound attaches to
    the EXTENSION only, and skipping base rungs silently truncates a pre-registered bracket."""
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
    """Prove the ladder terminates when an extension rung fails; the predicate is "the last rung
    run", since an enumerated verdict set left a new token out and re-drove one rung forever."""
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
    """Prove the stated stop reason is true when every base rung fails."""
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


def test_the_plan_fixture_is_the_committed_file_and_not_a_copy() -> None:
    assert tomllib.loads(_PLAN.read_text(encoding="utf-8"))["provenance"]["prereg_ruling"] \
        == "R309(f)"
