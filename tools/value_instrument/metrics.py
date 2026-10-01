"""Per-draw arithmetic: one temperature fitted on each game fold scores the other (calibration-free CE); raw CE ranks nothing."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

PLY_BANDS = {"plies_0_10": (0, 10), "plies_11_40": (11, 40), "plies_41_up": (41, 10**6)}
MIN_LINE = 0.005
EFFECT = 0.01
_PHI = NormalDist().cdf


def softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + e^x), overflow-free."""
    return np.logaddexp(0.0, x)


def fit_beta(u: np.ndarray, y: np.ndarray) -> float:
    """The beta >= 0 minimising mean softplus(-beta·y·u) (temperature 1/beta, no intercept), by bisection; 0 at no skill."""
    m = y * u

    def grad(b: float) -> float:
        return float(np.mean(-m / (1.0 + np.exp(np.clip(b * m, -700, 700)))))
    if len(m) == 0 or grad(0.0) >= 0.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while grad(hi) < 0.0:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if grad(mid) < 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    """P(score of a +1 draw > score of a -1 draw), ties counted half; None when a class is absent."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s), np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = y > 0
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def game_folds(game_ids: np.ndarray, seed: int, universe: np.ndarray) -> np.ndarray:
    """Fold 0/1 per entry of `game_ids`: the universe's games (the whole ring's) halved by a seeded permutation. Raises: ValueError when a game is outside the universe."""
    games = np.unique(universe)
    at = np.searchsorted(games, game_ids)
    if np.any(at >= len(games)) or np.any(games[np.minimum(at, len(games) - 1)] != game_ids):
        raise ValueError("game_folds: a draw's game is not in the ring's games")
    perm = np.random.default_rng(seed).permutation(len(games))
    fold_of = np.empty(len(games), np.int8)
    fold_of[perm] = (np.arange(len(games)) >= len(games) // 2).astype(np.int8)
    return fold_of[at]


def crossfit_losses(u: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Each draw's binary log loss under the temperature fitted on the OTHER fold."""
    f0, f1 = fold == 0, fold == 1
    b0, b1 = fit_beta(u[f0], y[f0]), fit_beta(u[f1], y[f1])
    return np.where(f1, softplus(-b0 * y * u), softplus(-b1 * y * u))


def _temperature(beta: float) -> float | None:
    return float(1.0 / beta) if beta > 0 else None


def block(u: np.ndarray, y: np.ndarray, raw: np.ndarray, fold: np.ndarray) -> dict[str, float | int | None]:
    """The instrument's rows for one set of draws (T > 1 overconfident, None = no skill); `{"n": 0}` when empty."""
    if len(y) == 0:
        return {"n": 0}
    f1 = fold == 1
    loss = crossfit_losses(u, y, fold)
    p = [float(np.clip(np.mean(y[f] > 0), 1e-6, 1 - 1e-6)) if f.any() else 0.5 for f in (fold == 0, f1)]
    p_other = np.where(f1, p[0], p[1])
    const = np.where(y > 0, -np.log(p_other), -np.log(1.0 - p_other))
    beta = fit_beta(u, y)
    return {"n": int(len(y)), "cf_ce": float(loss.mean()), "cf_ce_f1": float(loss[f1].mean()) if f1.any() else None,
            "beta": beta, "temperature": _temperature(beta), "auc": auc(u, y), "auc_f1": auc(u[f1], y[f1]),
            "right_way": float(np.mean(np.where(u == 0, 0.5, np.sign(u) == y))), "raw_ce": float(np.mean(raw)),
            "constant_ce": float(const.mean()), "uncal_binary_ce": float(np.mean(softplus(-y * u)))}


def side(u: np.ndarray, y: np.ndarray, raw: np.ndarray, fold: np.ndarray, ply: np.ndarray) -> dict[str, dict]:
    """`block` overall and per ply band, each band fitting its own temperatures."""
    out = {"overall": block(u, y, raw, fold)}
    for band, (lo, hi) in PLY_BANDS.items():
        sel = (ply >= lo) & (ply <= hi)
        out[band] = block(u[sel], y[sel], raw[sel], fold[sel])
    return out


def game_se(d: np.ndarray, game_ids: np.ndarray, seed: int, resamples: int = 2000) -> dict[str, float | int]:
    """The draw-weighted mean of a paired per-draw difference `d` and its bootstrap SE over games."""
    games, inv = np.unique(game_ids, return_inverse=True)
    s = np.bincount(inv, weights=d)
    n = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    est = np.empty(resamples)
    for i in range(resamples):
        w = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games)).astype(np.float64)
        est[i] = (w * s).sum() / (w * n).sum()
    return {"diff": float(s.sum() / n.sum()), "se_game": float(est.std()), "games": int(len(games)),
            "resamples": resamples, "boot_seed": seed}


def line_and_power(sd_seed: float, se_game: float) -> dict[str, float]:
    """The line max(MIN_LINE, 2·SD) and the power at EFFECT, with SD = √(seed SD² + game SE²)."""
    sd = math.sqrt(sd_seed ** 2 + se_game ** 2)
    line = max(MIN_LINE, 2.0 * sd)
    return {"sd": sd, "line": line, "power": _PHI((EFFECT - line) / sd) if sd > 0 else 1.0,
            "false_pass": _PHI(-line / sd) if sd > 0 else 0.0}


__all__ = ["EFFECT", "MIN_LINE", "PLY_BANDS", "auc", "block", "crossfit_losses", "fit_beta", "game_folds", "game_se",
           "line_and_power", "side", "softplus"]
