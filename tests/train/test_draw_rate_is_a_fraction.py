"""F-816-2 — `iteration_complete.draw_rate` is a FRACTION, and the pool owns its denominator.

MEASURED ON THE BURN'S OWN STREAM, not inferred: at steps 2, 3, 5, 6 and 7 the emitted
`draw_rate` read 1.3333, 1.5, 1.125, 1.2222 and 1.0909 — values a fraction cannot take. It
settled to exactly 1.000 once the counts grew, which is the signature of a ratio whose
denominator lags its numerator rather than of a mis-scaled statistic.

THE MECHANISM, derived from the code rather than guessed. The payload built the share by
hand as `pool.draws / games_played`:

  * the NUMERATOR `pool.draws` is read live inside the event builder;
  * the DENOMINATOR `games_played` is `StepCoordinator._games_played`, a snapshot taken near
    the top of `step()` and frozen for the rest of it while the feeder thread keeps draining
    finished games into the pool.

On a run where every game is a draw the numerator equals the pool's CURRENT
`games_completed` and the denominator is an EARLIER one, so the ratio is
`games_completed(t2) / games_completed(t1) >= 1`, decaying toward 1 as the counts grow.
That reproduces all five observed values exactly (4/3, 3/2, 9/8, 11/9, 12/11). It is the
same straddle class R218 rider 1 removed between the `target_integrity` and cluster blocks,
one payload over.

A CORRECTION TO THE QUEUED ROW, stated plainly. F-816-2 reads "and it feeds an armed
abort". Measured: it does not. `draw_rate_collapse` reads
`pooled_draw_rate(pool.pooled_draw_counts(), N_pool_min=…)` — `Sum(draws)/Sum(completed)`
over per-worker windows of 0/1 values, which cannot exceed 1 by construction
(`instrumentation.py::pooled_draw_counts`). The two statistics share a name and nothing
else. The defect is real and worth fixing before run5 — a metric whose definition is wrong
is wrong wherever it is read — but the abort was never exposed to it, and the row's harm
claim should not be carried forward as measured.

THE FIX: `WorkerPool.draw_rate`, computed under the pool's own lock against the same
`games_completed` the two win rates use, exactly like `x_winrate` / `o_winrate` — which
never had this defect for exactly that reason. The three outcome shares now share a
denominator and sum to 1.

MUTATION THAT REDS THIS FILE (M-DR-1): restore `pool.draws / games_played` in
`events.py`. The first row below then emits 4/3 and fails on `<= 1.0`.
"""
from __future__ import annotations

from typing import Any

from mantis.selfplay.pool_hooks import RunnerStats
from mantis.train.events import emit_iteration_complete_event


def _rstats() -> RunnerStats:
    """A REAL snapshot — the payload reads several of its fields, and a hand-shaped double
    would paper over a rename in `pool_hooks`."""
    return RunnerStats(
        games_completed=0, positions_generated=0, x_wins=0, o_wins=0, draws=0,
        model_version=0, mcts_quiescence_fires=0, mcts_mean_depth=5.0,
        mcts_mean_root_concentration=0.1, cluster_value_std_mean=None,
        cluster_policy_disagreement_mean=None, cluster_variance_sample_count=0,
    )


class _Sink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))


class _StraddlingPool:
    """A pool that DRAINS between the coordinator's snapshot and the emit — the exact
    condition the burn ran under.

    `games_completed` and `draws` both advance to `now`; the caller passes the STALE
    `games_played` it snapshotted earlier. A 100%-draw run is the worst case and the one
    that was actually observed, so it is what this double models.
    """

    gumbel_mcts = False
    avg_game_length = 128.0
    sims_per_sec = 100.0
    batch_fill_pct = 0.9
    inference_batch_timing = None

    def __init__(self, *, completed_now: int) -> None:
        self._completed = completed_now
        self._draws = completed_now  # every game a draw (F-816-6)
        self.draws = completed_now  # raw count, present only so M-DR-1 is runnable
        self.recent_move_histories: list = []

    @property
    def x_winrate(self) -> float:
        return 0.0

    @property
    def o_winrate(self) -> float:
        return 0.0

    @property
    def draw_rate(self) -> float:
        # The pool's own lock-consistent read: one denominator, taken with the numerator.
        return self._draws / self._completed if self._completed else 0.0

    def runner_stats(self) -> Any:
        return _rstats()


class _Buffer:
    size = 1000
    capacity = 100_000


def _emit(*, completed_now: int, games_played_snapshot: int) -> dict:
    sink = _Sink()
    emit_iteration_complete_event(
        7, 0.0, games_played_snapshot, 0, _StraddlingPool(completed_now=completed_now),
        _Buffer(), {}, {}, 100_000, lambda: 10.0, lambda: 5.0, {}, _rstats(), sink,
    )
    events = [e for e in sink.events if e["event"] == "iteration_complete"]
    assert len(events) == 1, f"expected one iteration_complete, got {len(events)}"
    return events[0]


def test_the_emitted_draw_rate_cannot_exceed_one_when_the_pool_drains_mid_step() -> None:
    """M-DR-1 — the reproduction, with the burn's own numbers.

    Snapshot at 3 completed games, pool at 4 by emit time: the old form emits 4/3 = 1.3333,
    which is step 2 of the burn stream verbatim."""
    payload = _emit(completed_now=4, games_played_snapshot=3)
    assert payload["draw_rate"] <= 1.0, (
        f"draw_rate emitted {payload['draw_rate']} — a fraction above 1. The share is being "
        "divided by the coordinator's stale game count instead of the pool's own "
        "denominator (F-816-2)"
    )
    assert payload["draw_rate"] == 1.0, (
        "a run where every completed game is a draw has a draw share of exactly 1.0; got "
        f"{payload['draw_rate']}"
    )


def test_the_denominator_is_the_pools_and_not_the_coordinators_snapshot() -> None:
    """The stale denominator must not influence the value AT ALL.

    Same pool, three wildly different snapshots. A value that moves with the snapshot is
    still reading the coordinator's number, whether or not it happens to stay under 1."""
    values = {
        snap: _emit(completed_now=8, games_played_snapshot=snap)["draw_rate"]
        for snap in (1, 8, 10_000)
    }
    assert len(set(values.values())) == 1, (
        f"the emitted share moved with the coordinator's snapshot: {values} — the pool owns "
        "this denominator, and a payload that mixes the two is reporting a ratio of two "
        "different clocks"
    )


def test_the_three_outcome_shares_share_a_denominator() -> None:
    """Why the fix is the pool property and not a clamp.

    `win_rate_p0` and `win_rate_p1` were always computed by the pool against
    `games_completed`; only the draw share was assembled by the caller. Printing three
    shares side by side where one has a different denominator makes them incommensurable —
    and a `min(1.0, …)` would have kept the payload plausible while leaving that true."""
    payload = _emit(completed_now=8, games_played_snapshot=3)
    total = payload["win_rate_p0"] + payload["win_rate_p1"] + payload["draw_rate"]
    assert abs(total - 1.0) < 1e-9, (
        f"the three outcome shares sum to {total}, not 1 — they are not shares of the same "
        "quantity"
    )
