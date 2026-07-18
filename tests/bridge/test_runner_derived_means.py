"""Derived-mean value-parity (O19, review gap 4).

The 4 bridge-derived means reconstruct the frozen fixed-point formula from the
SEAM RAW atomics: `accum / (count x 1_000_000.0)`, `count == 0 -> 0.0`, with
`mcts_mean_root_concentration` in f32 arithmetic and the other three in f64
(DESIGN §c.6).

The Rust unit test `derived_means_match_fixed_point_formula` (runner.rs) is the
AUTHORITATIVE O19 oracle: it seeds `(accum, count)` states directly and pins the
divisor + zero-guard + f32/f64 split. Seeding the raw atomics from Python is NOT
reachable (the accumulators are worker-thread-private; no bridge setter exists),
so the Python leg asserts the four getters are PRESENT, finite, and correctly
zero-guarded on a fresh (empty) runner — the `count == 0 -> 0.0` branch. Documented
per DESIGN §b: the Rust unit test carries the seeded-value parity.
"""
import math

from mantis import _engine

DERIVED_MEANS = [
    "mcts_mean_depth",
    "mcts_mean_root_concentration",
    "cluster_value_std_mean",
    "cluster_policy_disagreement_mean",
]


def _fresh_runner():
    cfg = _engine.SelfPlayRunnerConfig(n_workers=1, encoding_name="v6")
    return _engine.SelfPlayRunner(cfg)


def test_four_derived_means_present_finite_and_zero_guarded():
    runner = _fresh_runner()
    for name in DERIVED_MEANS:
        val = getattr(runner, name)
        assert isinstance(val, float)
        assert math.isfinite(val), f"{name} not finite: {val}"
        # Fresh runner: count == 0 -> the zero-guard returns exactly 0.0 (no NaN /
        # div-by-zero from an empty accumulator).
        assert val == 0.0, f"{name} zero-guard failed on empty runner: {val}"


def test_raw_count_getters_present():
    """The raw atomic count getters the means derive from are plain loads and
    read 0 on a fresh runner."""
    runner = _fresh_runner()
    assert runner.mcts_quiescence_fires == 0
    assert runner.cluster_variance_sample_count == 0
    assert runner.games_completed == 0
    assert runner.get_win_stats() == (0, 0, 0)
