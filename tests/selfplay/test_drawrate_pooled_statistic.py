"""ORACLE — the draw-rate abort's STATISTIC, and the two permanent regression oracles.

Every drive is on the REAL `PoolInstrumentation`, the window constant is asserted against the
deque itself and never a literal, and nothing here writes a file.

The statistic is `Sum(draws) / Sum(completed)` over the UNION of worker windows, replacing an
unweighted mean over a *filtered* set that was measured firing at a true pool draw rate of
0.0319 and staying silent at 0.968; both counterexamples are preserved below as permanent
regression oracles, and being 32-worker rigs they pin a STATISTIC, not run5's `n_workers: 1`
posture. Below `N_pool_min` there is NO OBSERVATION — `None`, never a healthy `0.0` appended to
the abort history as a real measurement. And `Sum(completed)` cannot exceed
`DRAW_RATE_WINDOW * n_workers`, so a bar above that ceiling makes the gate structurally unable
to observe while auditing ARMED; the schema half lives in `test_drawrate_schema_range.py`.

>300 justify (R8): four measured rigs plus the transport and no-default census in ONE file,
because they share one construction helper and one real subject — splitting them would fork
`_play` into copies free to drift in exactly the direction the statistic moves.
"""
from __future__ import annotations

import inspect
import threading

from mantis._engine import DEFAULT_CLUSTER_THRESHOLD
from mantis.monitor.rules import check_draw_rate_collapse
from mantis.selfplay.instrumentation import (  # RED anchor (R92) — `pooled_draw_counts` is new
    _DRAW_RATE_WINDOW,
    PoolInstrumentation,
)
from mantis.selfplay.pool import WorkerPool
from mantis.train.coordinator.config import (  # RED anchor (R92) — replaces recent_pool_draw_rate
    WorkerPoolLike,
    pooled_draw_rate,
)

#: run5's pre-registered terms and the coordinator's own `consec`, written here so the two
#: mandated oracles fire the REAL rule against the REAL numbers rather than a convenient pair.
RUN5_THRESHOLD = 0.25
RUN5_MIN_STEP = 25000
RUN5_N_POOL_MIN = 50
CONSEC = 3


def _instr() -> tuple[PoolInstrumentation, threading.Lock]:
    return PoolInstrumentation(log_investigation_metrics=False, cluster_threshold=DEFAULT_CLUSTER_THRESHOLD), threading.Lock()


def _play(instr, lock, *, worker_id: int, games: int, draws: int) -> None:
    """`draws` drawn games then `games - draws` decisive ones, on `worker_id`. A draw is
    `winner_code == 0`, spanning terminal reasons `2 = ply_cap` AND `3 = other_draw`: the
    "ply-cap truncations only" characterisation is of the HEALTHY regime, and the metric is
    wider."""
    for index in range(games):
        drawn = index < draws
        instr.on_game_complete(lock, 0 if drawn else 1, [], worker_id,
                               2 if drawn else 0, 0, 0, 1, 0)


def _fires(rate: float | None, *, at_step: int = RUN5_MIN_STEP) -> bool:
    """Does the REAL rule abort on a sustained `rate`? `None` (no observation) never reaches
    the history at all, which is the whole point of the type split — so it cannot fire."""
    if rate is None:
        return False
    history = [rate] * CONSEC
    return check_draw_rate_collapse(history, at_step, threshold=RUN5_THRESHOLD,
                                    consec=CONSEC, min_step=RUN5_MIN_STEP) is not None


# MANDATED: true pooled rate 0.968 MUST FIRE
def test_a_true_pool_draw_rate_of_0968_fires_the_abort() -> None:
    """PERMANENT REGRESSION ORACLE, from the measured counterexample.

    RIG: 32 workers — 31 drawing 100% at 49 completed games each, one healthy worker at 50
    decisive games. True pooled draw rate = 1519 / 1569 = 0.9681325685149776, on which the
    shipped statistic reported 0.0: an unweighted mean over the *included* set, where the ONE
    worker past `min_samples=50` was the only healthy one and the 31 collapsing workers were
    excluded into invisibility. The count-weighted rate cannot exclude anyone.
    """
    instr, lock = _instr()
    for worker in range(1, 32):
        _play(instr, lock, worker_id=worker, games=49, draws=49)
    _play(instr, lock, worker_id=0, games=50, draws=0)

    draws, completed = instr.pooled_draw_counts(lock)
    assert (draws, completed) == (1519, 1569), (
        f"harness precondition (DR's block B): 31x49 drawn + 1x50 decisive = 1519 draws in "
        f"1569 completed games; got {(draws, completed)}"
    )
    rate = pooled_draw_rate((draws, completed), N_pool_min=RUN5_N_POOL_MIN)
    assert rate == 1519 / 1569, (
        f"the pooled statistic is Sum(draws)/Sum(completed) over the UNION of worker windows, "
        f"which is {1519 / 1569!r} here; got {rate!r}"
    )
    assert _fires(rate) is True, (
        f"a true pool draw rate of {rate} is nearly four times run5's pre-registered threshold "
        f"of {RUN5_THRESHOLD} and the abort MUST fire. The old statistic reported 0.0 on this "
        "exact input (RECHECK_D finding DR-3, false negative) — this oracle is permanent by "
        "operator ruling R92 and is not to be re-pointed"
    )


# MANDATED: true pooled rate 0.0319 must stay SILENT
def test_a_true_pool_draw_rate_of_00319_stays_silent() -> None:
    """PERMANENT REGRESSION ORACLE, the other half of the counterexample pair.

    RIG: 32 workers — one at 50 completed games all drawn, 31 healthy at 49 decisive games each.
    True pooled draw rate = 50 / 1569 = 0.03186743148502231, an order of magnitude BELOW the
    threshold, on which the shipped statistic reported 1.0 and fired: the single fully-drawn
    worker was the only one past `min_samples=50`, and 31 healthy workers could not dilute it.
    """
    instr, lock = _instr()
    _play(instr, lock, worker_id=0, games=50, draws=50)
    for worker in range(1, 32):
        _play(instr, lock, worker_id=worker, games=49, draws=0)

    draws, completed = instr.pooled_draw_counts(lock)
    assert (draws, completed) == (50, 1569), (
        f"harness precondition (DR's block A): 1x50 drawn + 31x49 decisive = 50 draws in 1569 "
        f"completed games; got {(draws, completed)}"
    )
    rate = pooled_draw_rate((draws, completed), N_pool_min=RUN5_N_POOL_MIN)
    assert rate == 50 / 1569, (
        f"count-weighting is the point: one fully-drawn worker among 32 contributes 50 of "
        f"1569 games, not 1/32 of the mean; got {rate!r}"
    )
    assert _fires(rate) is False, (
        f"a true pool draw rate of {rate} is well below the pre-registered threshold of "
        f"{RUN5_THRESHOLD} and the abort MUST stay silent. The old statistic reported 1.0 on "
        "this exact input (RECHECK_D finding DR-3, false positive) — this oracle is permanent "
        "by operator ruling R92 and is not to be re-pointed"
    )


# the fabricated healthy 0.0 is unrepresentable
def test_total_collapse_below_the_old_bar_reports_collapse_and_never_a_healthy_zero() -> None:
    """The rig verbatim: 32 workers x 49 DRAWN games — total collapse, nobody past the retired
    per-worker bar of 50. That returned an empty per-worker map whose pooled rate was `0.0`,
    APPENDED to the abort history as a real measurement. Both halves of the type answer are
    asserted: with the evidence in hand the statistic reports 1.0 and fires, and with it
    WITHHELD the answer is `None`."""
    instr, lock = _instr()
    for worker in range(32):
        _play(instr, lock, worker_id=worker, games=49, draws=49)

    draws, completed = instr.pooled_draw_counts(lock)
    assert (draws, completed) == (1568, 1568), f"harness precondition; got {(draws, completed)}"
    assert pooled_draw_rate((draws, completed), N_pool_min=RUN5_N_POOL_MIN) == 1.0, (
        "total collapse must report 1.0. The old estimator reported 0.0 here because every "
        "worker sat one game under the per-worker bar (DR-4)"
    )
    assert _fires(pooled_draw_rate((draws, completed), N_pool_min=RUN5_N_POOL_MIN)) is True

    starved, starved_lock = _instr()
    _play(starved, starved_lock, worker_id=0, games=3, draws=3)
    assert pooled_draw_rate(starved.pooled_draw_counts(starved_lock),
                            N_pool_min=RUN5_N_POOL_MIN) is None, (
        "three drawn games is not evidence of a collapse; below `N_pool_min` the answer is "
        "NO OBSERVATION (`None`), never 1.0 and never the fail-safe 0.0. R92: insufficient "
        "evidence is never a healthy reading, and it is never a firing one either"
    )
    assert pooled_draw_rate((0, 0), N_pool_min=RUN5_N_POOL_MIN) is None, (
        "zero completed games is the same answer — and it is the STALL family's jurisdiction "
        "(R92), not this gate's. A 0/0 that reached the division would be a ZeroDivisionError "
        "inside a hard-abort gate"
    )


# the ceiling `N_pool_min` must respect, measured
def test_the_pooled_evidence_ceiling_is_the_window_times_the_worker_count() -> None:
    """The fourth "armed in the config, absent in effect" axis.

    `Sum(completed)` is `Sum_w len(dq_w)` and each deque's `maxlen` IS `_DRAW_RATE_WINDOW`, so an
    `N_pool_min` above `_DRAW_RATE_WINDOW * n_workers` is a condition no history can satisfy: no
    observation for the entire run while the audit reports the row ARMED. The window is asserted
    AGAINST THE DEQUE ITSELF — a named constant that drifted from the container it names would
    re-open the dead zone with every assertion here still green.
    """
    for n_workers in (1, 2, 8, 32):
        instr, lock = _instr()
        for worker in range(n_workers):
            _play(instr, lock, worker_id=worker, games=500, draws=250)

        deque_maxlen = instr._per_worker_draws[0].maxlen
        assert _DRAW_RATE_WINDOW == deque_maxlen, (
            f"`_DRAW_RATE_WINDOW` ({_DRAW_RATE_WINDOW}) must BE the per-worker deque's maxlen "
            f"({deque_maxlen}) — two constants that merely agree today are two authorities (R1)"
        )
        pooled_draws, completed = instr.pooled_draw_counts(lock)
        assert completed == _DRAW_RATE_WINDOW * n_workers, (
            f"n_workers={n_workers}: 500 completed games EACH must still leave the pooled "
            f"evidence at the ring's own ceiling {_DRAW_RATE_WINDOW * n_workers}; got "
            f"{completed}. If this ever exceeds the ceiling the schema bound below it is wrong"
        )
        assert pooled_draw_rate((pooled_draws, completed), N_pool_min=completed + 1) is None, (
            f"n_workers={n_workers}: a bar ONE game above the ceiling can never be met, so the "
            "gate observes nothing for the whole run while auditing ARMED — 'armed in the "
            "config, absent in effect' on its fourth axis"
        )
        assert pooled_draw_rate((pooled_draws, completed), N_pool_min=completed) is not None, (
            "…and AT the ceiling the bar is satisfiable, which is what makes 50 a live value "
            "on this tree's one-worker configs rather than a dead one"
        )


# transport: the bar has NO default, and the counts are not transposable
def test_N_pool_min_has_no_default_on_the_one_layer_that_takes_it() -> None:
    """One authority for the evidence bar, and no default. The retired design threaded the
    inclusion bar config to pool to estimator, so THREE signatures could hold a second authority
    over it; the metric is unconditional now and the bar is an evidence-sufficiency rule on the
    abort decision, so exactly ONE signature takes it, keyword-only. The pool-side surfaces are
    pinned in the OTHER direction, because a telemetry object that knew the abort's evidence
    rule would be the second authority this removed."""
    params = inspect.signature(pooled_draw_rate).parameters
    assert "N_pool_min" in params, (
        "`pooled_draw_rate` must TAKE the bar. A function that decided it would be a second "
        "authority over `train.draw_rate_abort.N_pool_min` (R1)"
    )
    assert params["N_pool_min"].default is inspect.Parameter.empty, (
        f"`N_pool_min` carries a default ({params['N_pool_min'].default!r}) — that default is "
        "a second authority over the operator's own pre-registered value, and its most "
        "natural values (0, 1) are the hair-trigger ADJ-14 filed"
    )
    assert params["N_pool_min"].kind is inspect.Parameter.KEYWORD_ONLY, (
        "`N_pool_min` must be keyword-only, so no positional call site can supply it in the "
        "slot the counts occupy"
    )

    for where, func in {
        "PoolInstrumentation.pooled_draw_counts": PoolInstrumentation.pooled_draw_counts,
        "WorkerPool.pooled_draw_counts": WorkerPool.pooled_draw_counts,
        "WorkerPoolLike.pooled_draw_counts": WorkerPoolLike.pooled_draw_counts,
    }.items():
        assert "N_pool_min" not in inspect.signature(func).parameters, (
            f"{where}: the pool reports RAW counts. R92 moved the evidence bar to the abort "
            "decision; a bar back here is the config authority leaking into telemetry"
        )


def test_the_counts_are_draws_then_completed_and_a_transposition_is_visible() -> None:
    """The bare `tuple[int, int]` is transposable at both ends, so the order is a pin rather
    than a comment: `(1, 4)` is one draw in four games = 0.25, where the transposed reading is
    `4.0`, outside the metric's own range and above every legal threshold. The two mandated
    oracles pin the same fact end to end; this arm states it locally."""
    assert pooled_draw_rate((1, 4), N_pool_min=4) == 0.25, (
        "counts are (draws, completed): one draw in four games is 0.25, not 4.0"
    )
    assert pooled_draw_rate((0, 50), N_pool_min=50) == 0.0, (
        "a MEASURED zero — 50 completed games, none drawn — is a real healthy observation and "
        "must be a float. R92 removes the FABRICATED zero, not the measured one"
    )
    assert pooled_draw_rate((50, 50), N_pool_min=50) == 1.0, (
        "1.0 is the metric's maximum at every worker count, which is what makes `le=1` a bound "
        "on the metric rather than a policy"
    )
