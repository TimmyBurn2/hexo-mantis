"""Vectorized numpy Bradley-Terry fit (design §a.3 bt.py; D-4).

NO scipy: the frozen old-side `hexo_rl/eval/bradley_terry.py` L-BFGS-B stack is not
carried (scipy is an undeclared optional dep). `fit_bt` is a numpy MM
(minorization-maximization) fit with a symmetric pseudo-count prior against the anchor
entity (index 0), so all-wins / disconnected / zero-game inputs stay finite instead of
diverging to +/-inf — the untreated BT degenerate case. ONE global fit across candidate +
best + all rungs; no per-game Python loop anywhere (the input is an aggregated win-count
matrix, and the MM update itself is fully vectorized over entities).
"""
from __future__ import annotations

import numpy as np

_DEFAULT_N_ITER = 200
_DEFAULT_TOL = 1e-10


def fit_bt(
    wins: np.ndarray,
    prior_games: float,
    *,
    n_iter: int = _DEFAULT_N_ITER,
    tol: float = _DEFAULT_TOL,
) -> np.ndarray:
    """MM Bradley-Terry fit. `wins[i, j]` = win COUNT of i over j (margin-blind; draws are
    pre-split 0.5/0.5 by the caller before this layer). Returns log-scale ratings, one per
    entity, shift-invariant (anchor at `ratings[0] == 0.0` by construction).
    """
    w = np.asarray(wins, dtype=np.float64)
    n = w.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    # Symmetric pseudo-count prior vs the anchor (index 0): connects EVERY entity to the
    # anchor, so an all-wins pair or a disconnected component stays finite and well-defined
    # instead of diverging (the untreated BT degenerate case).
    w = w.copy()
    if prior_games > 0:
        w[0, 1:] += prior_games
        w[1:, 0] += prior_games

    total_wins = w.sum(axis=1)          # W_i — total wins by i over everyone
    pair_games = w + w.T                # symmetric total games contested between i and j
    np.fill_diagonal(pair_games, 0.0)

    gamma = np.ones(n, dtype=np.float64)
    for _ in range(n_iter):
        denom = (pair_games / (gamma[:, None] + gamma[None, :] + 1e-300)).sum(axis=1)
        new_gamma = np.where(denom > 0.0, total_wins / np.where(denom > 0.0, denom, 1.0), gamma)
        new_gamma = new_gamma / new_gamma[0]
        if np.max(np.abs(new_gamma - gamma)) < tol:
            gamma = new_gamma
            break
        gamma = new_gamma

    return np.log(np.clip(gamma, 1e-300, None))


def predict_p(ratings: np.ndarray, i: int, j: int) -> float:
    """P(entity i beats entity j) under the fitted BT ratings: `sigmoid(r_i - r_j)`."""
    diff = float(np.asarray(ratings)[i] - np.asarray(ratings)[j])
    return float(1.0 / (1.0 + np.exp(-diff)))


__all__ = ["fit_bt", "predict_p"]
