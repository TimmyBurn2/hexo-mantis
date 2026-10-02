"""Per-draw arithmetic: one temperature fitted on each game fold scores the other (calibration-free CE); raw CE ranks nothing."""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

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


def _base_rate(y: np.ndarray) -> float:
    return float(np.clip(np.mean(y > 0), 1e-6, 1 - 1e-6)) if len(y) else 0.5


def fit_platt(u: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """The (a, c) minimising mean softplus(-y·(a·u + c)), by damped Newton; (0, the clipped base-rate logit) when a class is absent. Raises: ValueError on a non-finite u or y (numpy's LinAlgError, a subclass, should the solve fail)."""
    if not (np.isfinite(u).all() and np.isfinite(y).all()):
        raise ValueError("fit_platt: a non-finite logit or outcome")
    p = _base_rate(y)
    w = np.array([0.0, math.log(p / (1.0 - p))])
    t = (y > 0).astype(np.float64)
    if len(t) == 0 or t.min() == t.max():
        return 0.0, float(w[1])
    x = np.stack([u, np.ones_like(u)], axis=1).astype(np.float64)

    def loss(at: np.ndarray) -> float:
        return float(np.mean(softplus(-y * (x @ at))))
    cur = loss(w)
    for _ in range(iters):
        q = 0.5 * (1.0 + np.tanh(0.5 * (x @ w)))
        g = x.T @ (q - t) / len(t)
        step = np.linalg.lstsq((x.T * (q * (1.0 - q))) @ x / len(t), g, rcond=None)[0]
        if float(g @ step) < 1e-18:
            break
        s, new = 1.0, loss(w - step)
        while new > cur and s > 1e-6:
            s *= 0.5
            new = loss(w - s * step)
        if new > cur:
            break
        w, cur = w - s * step, new
    return float(w[0]), float(w[1])


def crossfit_platt_losses(u: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Each draw's binary log loss under the slope and intercept fitted on the OTHER fold. Raises: ValueError on non-finite input."""
    f0, f1 = fold == 0, fold == 1
    (a0, c0), (a1, c1) = fit_platt(u[f0], y[f0]), fit_platt(u[f1], y[f1])
    return np.where(f1, softplus(-y * (a0 * u + c0)), softplus(-y * (a1 * u + c1)))


def calibrated(v: np.ndarray, temperature: float | None) -> np.ndarray:
    """tanh(atanh(v) / T): a value at the fitted temperature, odd and the identity at T = 1; zeros at T None (no skill)."""
    x = np.asarray(v, np.float64)
    if temperature is None:
        return np.zeros_like(x)
    return np.tanh(np.arctanh(np.clip(x, -1 + 1e-7, 1 - 1e-7)) / temperature)


def _temperature(beta: float) -> float | None:
    return float(1.0 / beta) if beta > 0 else None


def block(u: np.ndarray, y: np.ndarray, raw: np.ndarray, fold: np.ndarray) -> dict[str, float | int | None]:
    """The instrument's rows for one set of draws (T > 1 overconfident, None = no skill); `{"n": 0}` when empty. Raises: ValueError on a non-finite u."""
    if len(y) == 0:
        return {"n": 0}
    f1 = fold == 1
    loss, platt = crossfit_losses(u, y, fold), crossfit_platt_losses(u, y, fold)
    p = [_base_rate(y[f]) for f in (fold == 0, f1)]
    p_other = np.where(f1, p[0], p[1])
    const = np.where(y > 0, -np.log(p_other), -np.log(1.0 - p_other))
    beta = fit_beta(u, y)
    return {"n": int(len(y)), "cf_ce": float(loss.mean()), "cf_ce_f1": float(loss[f1].mean()) if f1.any() else None,
            "platt_ce": float(platt.mean()), "platt_ce_f1": float(platt[f1].mean()) if f1.any() else None,
            "beta": beta, "temperature": _temperature(beta), "auc": auc(u, y), "auc_f1": auc(u[f1], y[f1]),
            "right_way": float(np.mean(np.where(u == 0, 0.5, np.sign(u) == y))), "raw_ce": float(np.mean(raw)),
            "constant_ce": float(const.mean()), "uncal_binary_ce": float(np.mean(softplus(-y * u)))}


def band_masks(ply: np.ndarray) -> dict[str, np.ndarray]:
    """Each PLY_BANDS band's draws, both bounds inclusive."""
    return {band: (ply >= lo) & (ply <= hi) for band, (lo, hi) in PLY_BANDS.items()}


def side(u: np.ndarray, y: np.ndarray, raw: np.ndarray, fold: np.ndarray, ply: np.ndarray) -> dict[str, dict]:
    """`block` overall and per ply band, each band fitting its own temperatures. Raises: ValueError on a non-finite u."""
    out = {"overall": block(u, y, raw, fold)}
    for band, sel in band_masks(ply).items():
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


def line_and_power(sd_seed: float, se_game: float, effect: float = EFFECT) -> dict[str, float]:
    """The line max(MIN_LINE, 2·SD) and the power at `effect`, with SD = √(seed SD² + game SE²)."""
    sd = math.sqrt(sd_seed ** 2 + se_game ** 2)
    line = max(MIN_LINE, 2.0 * sd)
    return {"sd": sd, "line": line, "power": _PHI((effect - line) / sd) if sd > 0 else 1.0,
            "false_pass": _PHI(-line / sd) if sd > 0 else 0.0}


def verdicts(diff: float, line: float, effect: float = EFFECT) -> dict[str, Any]:
    """The interval diff ± line read as a detection (arm better), a worse arm, and a TOST null within ±effect."""
    lo, hi = diff - line, diff + line
    return {"ci": [lo, hi], "detection": hi <= 0.0, "effect_of_record": hi if hi <= 0.0 else None, "worse": lo >= 0.0,
            "tost_null": lo >= -effect and hi <= effect, "effect": effect}


__all__ = ["EFFECT", "MIN_LINE", "PLY_BANDS", "auc", "band_masks", "block", "calibrated", "crossfit_losses", "crossfit_platt_losses",
           "fit_beta", "fit_platt", "game_folds", "game_se", "line_and_power", "side", "softplus", "verdicts"]
