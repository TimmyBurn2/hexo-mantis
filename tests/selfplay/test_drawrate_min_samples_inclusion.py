"""⊕ WPAX Phase D ORACLE — O-D9: the min-sample inclusion rule (DESIGN_D §4.5; R80's first
half, ADJ-14's actual mechanism).

RED-at-import until IMPL lands `_DRAW_RATE_WINDOW` — the deque's `maxlen` given a name, so
the schema's `min_samples: Field(le=_DRAW_RATE_WINDOW)` ceiling and the window it bounds are
ONE authority rather than two constants that agree today.

**The defect this is the only witness to.** `instrumentation.py:365-371` includes any worker
with `len(dq) > 0` — ONE game — so a single drawn game per worker saturates the pool mean at
`1.0`, the maximum the metric can take, at or above EVERY legal threshold, at every worker
count. The `0.0`-on-empty guard (`coordinator/config.py:143-144`) covers the interval before
the first game and nothing after. REVIEW-design escalated ADJ-14 on exactly this point: the
hazard is the INCLUSION RULE, not the step count, so a `min_step` guard alone would not have
closed it — which is why R80 orders BOTH guards.

Not caught by anything else in the delta: O-D2 drives `compose_run` and never reaches the
estimator; O-D8 uses a fake pool and never reaches it either; O-D10 observes the `min_samples`
value at the call site but not what the estimator DOES with it.

The second subject is the 51-counterexample R85 preserved by name. `len(dq)` is bounded by
the deque's `maxlen`, so any `min_samples` above it is permanently unsatisfiable: at 51, ten
thousand consecutive DRAWN games report a pool rate of `0.0` and the abort can never fire.
That is MF-1's class on a second axis — "armed in the config, absent in effect" — and R71's
class-fix law is why the bound is `le=_DRAW_RATE_WINDOW` on the schema field rather than a
comment. The schema half is `tests/config/test_drawrate_schema_range.py`; this file is the
BEHAVIOUR the bound exists to describe, so the counterexample is a test and not a note.

R7 / gate 6: nothing here writes anything; every drive is in-memory.
"""
from __future__ import annotations

import inspect
import threading

import pytest

from mantis.selfplay.instrumentation import (  # RED anchor — `_DRAW_RATE_WINDOW` is new (R80)
    _DRAW_RATE_WINDOW,
    PoolInstrumentation,
)
from mantis.selfplay.pool import WorkerPool
from mantis.train.coordinator.config import WorkerPoolLike, recent_pool_draw_rate


def _instr() -> tuple[PoolInstrumentation, threading.Lock]:
    return PoolInstrumentation(log_investigation_metrics=False), threading.Lock()


def _play(instr, lock, *, worker_id: int, games: int, draws: int) -> None:
    """`draws` drawn games then `games - draws` decisive ones, on `worker_id`.

    A draw is `winner_code == 0` (`instrumentation.py:322`), which spans terminal reasons
    `2 = ply_cap` AND `3 = other_draw` — R82's "ply-cap truncations only" characterises the
    HEALTHY regime, while the metric itself is wider (SF-3's correction).
    """
    for index in range(games):
        drawn = index < draws
        instr.on_game_complete(lock, 0 if drawn else 1, [], worker_id,
                               2 if drawn else 0, 0, 0, 1, 0, 5)


def test_the_min_sample_inclusion_rule_closes_the_one_game_saturation() -> None:
    """The ADJ-14 hazard, driven at 1, 2 and 8 workers because the saturation is independent
    of worker count: the pool statistic is an UNWEIGHTED mean of per-worker rates, so one
    drawn game per worker gives `mean(1.0, 1.0, …) == 1.0` however many workers there are.
    Averaging cannot dilute it, which is why "more workers" is not a mitigation.

    Both arms of the changed predicate (`len(dq) >= min_samples`) are driven at every worker
    count: below the bar the worker is EXCLUDED and the pool reports the fail-safe `0.0`; at
    or above it the worker is INCLUDED and the pool reports the true window rate. A rule that
    excluded everything always would satisfy the first arm alone.
    """
    for n_workers in (1, 2, 8):
        instr, lock = _instr()
        for worker in range(n_workers):
            _play(instr, lock, worker_id=worker, games=1, draws=1)

        rates = instr.per_worker_draw_rates(lock, min_samples=_DRAW_RATE_WINDOW)
        assert rates == {}, (
            f"n_workers={n_workers}: one drawn game per worker must include NO worker. Under "
            f"the shipped `len(dq) > 0` rule this returned {{w: 1.0 …}} and the pool mean "
            "saturated at 1.0 — at or above every legal threshold, on the second game of a "
            f"healthy run; got {rates}"
        )
        assert recent_pool_draw_rate(rates) == 0.0, (
            "…and the pool statistic must therefore be the fail-safe 0.0, below every legal "
            "threshold. This is the direction `rules.py:163-164` records as the cheaper "
            "error: a missed abort costs less than a self-correcting dip aborting a "
            "RECOVERING run"
        )

        for worker in range(n_workers):
            _play(instr, lock, worker_id=worker, games=_DRAW_RATE_WINDOW - 1, draws=0)
        included = instr.per_worker_draw_rates(lock, min_samples=_DRAW_RATE_WINDOW)
        assert set(included) == set(range(n_workers)), (
            f"n_workers={n_workers}: at exactly `min_samples` completed games every worker "
            f"must be INCLUDED — a rule that excludes forever is as inert as one that "
            f"includes at one game; got {included}"
        )
        assert all(rate == 1 / _DRAW_RATE_WINDOW for rate in included.values()), (
            "…and the reported rate must be the TRUE window rate (1 draw in "
            f"{_DRAW_RATE_WINDOW}), not a saturated or rounded stand-in; got {included}"
        )


def test_the_inclusion_bar_is_the_min_samples_ARGUMENT_and_not_a_constant() -> None:
    """Transport. The arm above is satisfied by an estimator that hardcoded
    `len(dq) >= _DRAW_RATE_WINDOW`, which would be right for run5 and wrong for every other
    value an operator may pre-register. Driven across the range with the boundary on both
    sides, so the bar has to BE the argument.
    """
    for bar in (1, 2, 7, 25, _DRAW_RATE_WINDOW):
        instr, lock = _instr()
        _play(instr, lock, worker_id=0, games=bar - 1, draws=bar - 1)
        assert instr.per_worker_draw_rates(lock, min_samples=bar) == {}, (
            f"min_samples={bar}: a worker with {bar - 1} games is BELOW the bar and must be "
            "excluded — off-by-one here is the whole defect, one game further along"
        )
        _play(instr, lock, worker_id=0, games=1, draws=1)
        assert instr.per_worker_draw_rates(lock, min_samples=bar) == {0: 1.0}, (
            f"min_samples={bar}: the SAME worker one game later is at the bar and must be "
            "included, reporting the true rate of a fully-drawn window"
        )


def test_the_51_counterexample_the_le_bound_exists_for() -> None:
    """R85 preserved this by name, so it is a test rather than a comment.

    `len(dq)` is bounded above by the deque's own `maxlen`, so `min_samples > maxlen` is a
    condition no history can satisfy. Under TOTAL collapse — every one of ten thousand games
    a draw, the most extreme input the metric admits — a `min_samples` of 51 reports `0.0`
    and the abort CANNOT FIRE while auditing ARMED. That is the same defect class as a
    threshold of 35.0 (MF-1), on a different axis, which is why `le=_DRAW_RATE_WINDOW` is a
    type bound and `50` is the most conservative value that can still fire.

    The window constant is asserted AGAINST THE DEQUE ITSELF, never against the literal 50: a
    named constant that drifted from the container it names would re-open the dead zone with
    every assertion in this file still green.
    """
    instr, lock = _instr()
    _play(instr, lock, worker_id=0, games=10_000, draws=10_000)

    deque_maxlen = instr._per_worker_draws[0].maxlen
    assert _DRAW_RATE_WINDOW == deque_maxlen, (
        f"`_DRAW_RATE_WINDOW` ({_DRAW_RATE_WINDOW}) must BE the per-worker deque's maxlen "
        f"({deque_maxlen}); it is the schema's `min_samples` ceiling, and two constants that "
        "merely agree today are two authorities (R1)"
    )
    assert len(instr._per_worker_draws[0]) == deque_maxlen, (
        "harness precondition: 10 000 games must have saturated the ring, so the only reason "
        "a worker can be excluded below is the inclusion bar"
    )

    at_bound = instr.per_worker_draw_rates(lock, min_samples=_DRAW_RATE_WINDOW)
    assert recent_pool_draw_rate(at_bound) == 1.0, (
        f"at min_samples={_DRAW_RATE_WINDOW} total collapse must report a pool rate of 1.0 "
        "and therefore FIRE against any legal threshold"
    )
    for over in (_DRAW_RATE_WINDOW + 1, 2 * _DRAW_RATE_WINDOW):
        beyond = instr.per_worker_draw_rates(lock, min_samples=over)
        assert recent_pool_draw_rate(beyond) == 0.0, (
            f"min_samples={over} is above the deque's own bound, so under TOTAL collapse the "
            f"pool rate is {recent_pool_draw_rate(beyond)} and the abort can never fire — "
            "armed in the config, absent in effect. The schema's `le` bound is what makes "
            "this unrepresentable rather than merely undocumented"
        )


def test_min_samples_has_NO_default_at_any_layer_on_the_path() -> None:
    """R1, and MF-2's lesson applied one seam over. A defaulted `min_samples` on the pool
    method or the Protocol would re-create the ADJ-14 defect the moment any caller omitted
    it — and the value that default would take is EXACTLY the defect (`1`, i.e. `len(dq) > 0`
    spelled differently). The config block is the sole authority for the bar, so no layer on
    the path between it and the deque may hold a second one.

    All three layers are asserted because they fail on different mutations: the estimator is
    where the rule lives, the pool method is what `StepCoordinator` actually calls, and the
    Protocol is what makes a fake that omits the parameter a type error rather than a silent
    pass.
    """
    surfaces = {
        "PoolInstrumentation.per_worker_draw_rates (the rule itself)":
            PoolInstrumentation.per_worker_draw_rates,
        "WorkerPool.per_worker_draw_rates (what the coordinator calls)":
            WorkerPool.per_worker_draw_rates,
        "WorkerPoolLike.per_worker_draw_rates (the injected-seam Protocol)":
            WorkerPoolLike.per_worker_draw_rates,
    }
    for where, func in surfaces.items():
        params = inspect.signature(func).parameters
        assert "min_samples" in params, (
            f"{where}: `min_samples` must appear in the signature. The inclusion bar is "
            "config-authored (train.draw_rate_abort.min_samples); a layer that does not take "
            "it is a layer that decided it"
        )
        assert params["min_samples"].default is inspect.Parameter.empty, (
            f"{where}: `min_samples` carries a default "
            f"({params['min_samples'].default!r}) — that default is a second authority over "
            "the operator's own pre-registered value (R1), and its most natural value is the "
            "defect ADJ-14 filed"
        )
        assert params["min_samples"].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{where}: `min_samples` must be keyword-only, so no positional call site can "
            "supply it by accident in the slot the lock used to occupy"
        )

    instr, lock = _instr()
    with pytest.raises(TypeError):
        instr.per_worker_draw_rates(lock)  # type: ignore[call-arg]
