"""Derived-mean value parity across the FFI, and the zero-count reading.

The bridge-derived means reconstruct the fixed-point formula from the seam raw atomics:
`accum / (count x 1_000_000.0)`, with `mcts_mean_root_concentration` in f32 and the others
in f64.

The Rust unit tests in `runner.rs` are the authoritative oracle: they seed `(accum, count)`
directly and pin the divisor, the f32/f64 split and the zero-count arm. Seeding the raw
atomics from Python is not reachable — the accumulators are worker-thread-private and no
bridge setter exists — so this leg asserts what a fresh runner reports across the real FFI.

A mean over zero samples reads `None`, never a measured zero. The two MCTS means keep a
zero-guard because `mcts_stat_count` advances once per search, so its zero is a run that has
not moved yet rather than an absent instrument.
"""
import math

from mantis import _engine

#: Means whose zero-count arm still returns 0.0.
ZERO_GUARDED_MEANS = [
    "mcts_mean_depth",
    "mcts_mean_root_concentration",
]
DERIVED_MEANS = list(ZERO_GUARDED_MEANS)


def _fresh_runner():
    cfg = _engine.SelfPlayRunnerConfig(n_workers=1, encoding_name="gnn_axis_v1")
    return _engine.SelfPlayRunner(cfg)


def test_every_derived_mean_is_present_and_finite():
    runner = _fresh_runner()
    for name in DERIVED_MEANS:
        val = getattr(runner, name)
        assert val is None or isinstance(val, float), f"{name}: unexpected {type(val)}"
        assert val is None or math.isfinite(val), f"{name} not finite: {val}"


def test_mcts_means_keep_the_zero_guard_on_an_empty_runner():
    """Prove the MCTS means keep their zero-guard: `mcts_stat_count`'s zero is transient."""
    runner = _fresh_runner()
    for name in ZERO_GUARDED_MEANS:
        assert getattr(runner, name) == 0.0, f"{name} zero-guard changed unannounced"


def test_raw_count_getters_present():
    """Prove the raw atomic count getters are plain loads that read 0 on a fresh runner."""
    runner = _fresh_runner()
    assert runner.mcts_quiescence_fires == 0
    assert runner.games_completed == 0
    assert runner.get_win_stats() == (0, 0, 0)


def test_max_sims_per_search_is_on_the_surface_and_truthful_at_zero():
    """Prove the served-sims counter reaches Python and reads a truthful zero, not the budget.

    The exact-budget assertion lives in `crates/mantis-selfplay/tests/served_sims_exact.rs`,
    where it can be driven.
    """
    runner = _fresh_runner()
    assert runner.max_sims_per_search == 0
