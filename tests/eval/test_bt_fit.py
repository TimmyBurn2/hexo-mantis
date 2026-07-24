"""⊕ WP11-A — vectorized numpy Bradley-Terry fit (mantis.eval.bt, design §a.3/§c).

RED-at-import until IMPL writes `mantis.eval.bt`. NO scipy (D-4): the frozen old-side
`hexo_rl/eval/bradley_terry.py` L-BFGS-B stack is NOT ported — `fit_bt` is a numpy MM
(minorization-maximization) fit with a symmetric pseudo-count prior (`prior_games`)
against the anchor entity (index 0), so all-wins / disconnected / zero-game inputs stay
finite instead of diverging to +/-inf (the untreated BT degenerate case).

`wins[i, j]` = number of times entity i beat entity j (head-to-head win COUNT, draws
pre-split 0.5/0.5 by the caller before this layer — bt.py itself is margin-blind on a
plain win-count matrix). Ratings are shift-invariant (BT has one degree of freedom); every
assertion below anchors at entity 0 (`ratings - ratings[0]`) before comparing.
"""
from __future__ import annotations

import time

import numpy as np

from mantis.eval.bt import fit_bt


def test_bt_recovers_known_ratings_on_synthetic_data() -> None:
    # True strengths s = [0, 1, 2] (log-odds units). True head-to-head win prob
    # P(i beats j) = sigmoid(s_i - s_j) (standard BT generative model). A large,
    # deterministic (expected-value, not sampled) win matrix removes sampling noise so
    # the recovered order + magnitude must match the generator, not merely its sign.
    true_s = np.array([0.0, 1.0, 2.0])
    n_games = 2000
    wins = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            p = 1.0 / (1.0 + np.exp(-(true_s[i] - true_s[j])))
            wins[i, j] = n_games * p
    ratings = fit_bt(wins, prior_games=1.0)
    ratings = np.asarray(ratings) - np.asarray(ratings)[0]
    assert ratings[2] > ratings[1] > ratings[0]
    assert abs(ratings[1] - true_s[1]) < 0.2
    assert abs(ratings[2] - true_s[2]) < 0.2


def test_all_wins_stays_finite_via_prior() -> None:
    # Entity 0 beats entity 1 every one of 100 games, ZERO reverse games — a fully
    # separated pair. Without a prior this is the untreated BT degenerate case (the
    # MLE strength gap diverges to +inf). `prior_games` (a symmetric pseudo-count vs the
    # anchor, entity 0) must keep the fit finite.
    wins = np.array([[0.0, 100.0], [0.0, 0.0]])
    ratings = np.asarray(fit_bt(wins, prior_games=1.0))
    assert np.all(np.isfinite(ratings))
    assert ratings[0] > ratings[1]


def test_zero_games_returns_prior_ratings_not_nan() -> None:
    # No data anywhere: every entity must tie at the prior (all ratings equal), never NaN.
    wins = np.zeros((4, 4))
    ratings = np.asarray(fit_bt(wins, prior_games=1.0))
    assert not np.any(np.isnan(ratings))
    assert np.all(np.isfinite(ratings))
    assert np.allclose(ratings, ratings[0])


def test_single_rung_fit_is_well_defined() -> None:
    wins = np.array([[0.0, 5.0], [3.0, 0.0]])
    ratings = np.asarray(fit_bt(wins, prior_games=1.0))
    assert ratings.shape == (2,)
    assert np.all(np.isfinite(ratings))
    assert ratings[0] > ratings[1]  # entity 0 won more of the head-to-head


def test_disconnected_rungs_stay_finite_and_flagged() -> None:
    # Two clusters that never play each other directly: {0, 1} play only within
    # themselves, {2, 3} play only within themselves. A raw (unregularized) BT fit over a
    # disconnected graph has NO unique solution (the between-cluster gap is unconstrained
    # by data). `bt_prior_games` connects EVERY entity to the anchor (index 0) with a
    # symmetric pseudo-count, so the fit stays finite and well-defined even though the
    # RAW win data alone would leave entities 2/3 unconstrained relative to 0/1 — that
    # finiteness (no divergence, no NaN) IS the "flagged" guarantee this test pins: a
    # disconnected component never silently produces +/-inf or NaN ratings.
    wins = np.zeros((4, 4))
    wins[0, 1], wins[1, 0] = 30.0, 10.0
    wins[2, 3], wins[3, 2] = 8.0, 22.0
    ratings = np.asarray(fit_bt(wins, prior_games=1.0))
    assert ratings.shape == (4,)
    assert np.all(np.isfinite(ratings))
    assert not np.any(np.isnan(ratings))
    # within-cluster order must still reflect the local win data
    assert ratings[0] > ratings[1]
    assert ratings[3] > ratings[2]


def test_bt_and_bootstrap_are_vectorized_existence_proof() -> None:
    # Existence proof of vectorization (no per-game Python loop), NOT a bench (P-2:
    # bench posture n/a). 100k synthetic games + a 1000-resample bootstrap + one BT fit
    # must complete in well under 5s wall on CPU.
    from mantis.eval.aggregate import pair_bootstrap_wr_ci

    rng = np.random.default_rng(1234)
    n = 100_000
    outcomes = rng.binomial(1, 0.55, size=n).astype(np.float64)

    t0 = time.perf_counter()
    lo, hi = pair_bootstrap_wr_ci(outcomes, resamples=1000, ci_level=0.95, seed=1234)
    total_wins_a = float(outcomes.sum())
    wins = np.array([[0.0, total_wins_a], [n - total_wins_a, 0.0]])
    ratings = fit_bt(wins, prior_games=1.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"BT+bootstrap over 100k games took {elapsed:.2f}s (>= 5s bound)"
    assert lo is not None and 0.0 <= lo <= hi <= 1.0
    assert np.all(np.isfinite(np.asarray(ratings)))
