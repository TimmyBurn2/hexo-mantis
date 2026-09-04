"""Derived-mean value-parity (O19, review gap 4) + the R249 zero-count reading.

The 4 bridge-derived means reconstruct the frozen fixed-point formula from the
SEAM RAW atomics: `accum / (count x 1_000_000.0)`, with
`mcts_mean_root_concentration` in f32 arithmetic and the other three in f64
(DESIGN §c.6).

The Rust unit tests in `runner.rs` are the AUTHORITATIVE O19 oracle: they seed
`(accum, count)` states directly and pin the divisor, the f32/f64 split and the
zero-count arm (`zero_count_derived_mean_is_none_never_zero`). Seeding the raw
atomics from Python is NOT reachable (the accumulators are worker-thread-private;
no bridge setter exists), so the Python leg asserts what a FRESH (empty) runner
reports across the real FFI boundary. Documented per DESIGN §b: the Rust unit
tests carry the seeded-value parity.

ADJ-D32 / R249 (this file's assertions CHANGED with the fix): the two cluster means
used to be asserted `== 0.0` on an empty runner, and that assertion was a
restatement of the defect — a mean over zero samples read as a measured zero, which
on the graph arm is the permanent state. They now read `None`. The two MCTS means
keep the zero-guard: `mcts_stat_count` advances once per search on both arms, so
its zero is a run that has not moved yet, not an absent instrument.
"""
import math

from mantis import _engine

#: Means whose zero-count arm still returns 0.0 (see the module docstring).
ZERO_GUARDED_MEANS = [
    "mcts_mean_depth",
    "mcts_mean_root_concentration",
]
#: Means that report `None` when `cluster_variance_sample_count` is 0 (R249).
CLUSTER_MEANS = [
    "cluster_value_std_mean",
    "cluster_policy_disagreement_mean",
]
DERIVED_MEANS = ZERO_GUARDED_MEANS + CLUSTER_MEANS


def _fresh_runner():
    cfg = _engine.SelfPlayRunnerConfig(n_workers=1, encoding_name="v6")
    return _engine.SelfPlayRunner(cfg)


def test_four_derived_means_present_and_finite():
    runner = _fresh_runner()
    for name in DERIVED_MEANS:
        val = getattr(runner, name)
        assert val is None or isinstance(val, float), f"{name}: unexpected {type(val)}"
        assert val is None or math.isfinite(val), f"{name} not finite: {val}"


def test_mcts_means_keep_the_zero_guard_on_an_empty_runner():
    """`mcts_stat_count == 0` -> 0.0. Scoped deliberately (ADJ-D32 mandate is the cluster
    pair): this counter advances once per search on BOTH arms, so its zero is transient."""
    runner = _fresh_runner()
    for name in ZERO_GUARDED_MEANS:
        assert getattr(runner, name) == 0.0, f"{name} zero-guard changed unannounced"


def test_cluster_means_read_none_not_zero_at_zero_samples():
    """R249 across the REAL FFI boundary — the producer test for the fix.

    A fresh runner has `cluster_variance_sample_count == 0`, and that is exactly the state
    run5's graph arm never leaves: the variance atomics are unreachable there. The getters
    must report `None` (no measurement), which is what lets the event builder drop the
    fields instead of publishing the 0.0 that made `iteration_complete` lie for a whole run.

    FALSIFYING MUTATION: restore `derived_mean_f64`'s zero-count `0.0` -> this test RED.
    """
    runner = _fresh_runner()
    assert runner.cluster_variance_sample_count == 0
    for name in CLUSTER_MEANS:
        assert getattr(runner, name) is None, (
            f"{name} on a zero-sample runner must be None, got "
            f"{getattr(runner, name)!r} — a mean over zero samples is not a measurement"
        )


def test_raw_count_getters_present():
    """The raw atomic count getters the means derive from are plain loads and
    read 0 on a fresh runner."""
    runner = _fresh_runner()
    assert runner.mcts_quiescence_fires == 0
    assert runner.cluster_variance_sample_count == 0
    assert runner.games_completed == 0
    assert runner.get_win_stats() == (0, 0, 0)


def test_max_sims_per_search_is_on_the_surface_and_truthful_at_zero():
    """R335(c)/LAW-18 — the served-sims lever reports its own rate in-run.

    The clamp that made a search stop at exactly `n_simulations` is worthless as evidence if
    the only place it can be read is a Rust test: the ledger's `53.46 sims/move` line was
    measured in a run, and re-measuring it must not need a diagnostic rig branch. This pins
    that the counter reaches Python at all, and that its zero is TRUTHFUL — no search has
    completed on a fresh runner, so the honest reading is 0 and not the budget.

    The exact-budget assertion lives where it can be driven:
    `crates/mantis-selfplay/tests/served_sims_exact.rs`.
    """
    runner = _fresh_runner()
    assert runner.max_sims_per_search == 0
