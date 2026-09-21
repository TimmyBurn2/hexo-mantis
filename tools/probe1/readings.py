"""The reductions: E3 calibration (reading 2), P-B1's KLs (reading 5) and C3-1's gap table (reading 3) from the per-row / per-batch arrays."""
from __future__ import annotations

from typing import Any

import numpy as np


def _q(a: np.ndarray, p: float) -> float | None:
    return None if a.size == 0 else float(np.quantile(a, p))


def _bootstrap_ratio(a: np.ndarray, b: np.ndarray, *, seed: int, n: int = 2000) -> tuple[float, float] | None:
    """The 95 % percentile interval of mean(a)/mean(b) over row resamples; None when either side is empty."""
    if a.size == 0 or b.size == 0:
        return None
    rng = np.random.default_rng(seed)
    ratios = [rng.choice(a, a.size).mean() / max(rng.choice(b, b.size).mean(), 1e-12) for _ in range(n)]
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def _bucket(ev: np.ndarray, z: np.ndarray, ce: np.ndarray) -> dict[str, Any]:
    err = np.abs(ev - z)
    edges = np.linspace(-1.0, 1.0, 11)
    which = np.clip(np.digitize(ev, edges) - 1, 0, 9)
    reliability = [{"ev_lo": float(edges[k]), "ev_hi": float(edges[k + 1]), "n": int((which == k).sum()),
                    "ev_mean": (float(ev[which == k].mean()) if (which == k).any() else None),
                    "z_mean": (float(z[which == k].mean()) if (which == k).any() else None)} for k in range(10)]
    return {"n": int(ev.size), "mae": (float(err.mean()) if err.size else None), "rmse": (float(np.sqrt((err ** 2).mean())) if err.size else None),
            "bias_ev_minus_z": (float((ev - z).mean()) if err.size else None), "value_ce_mean": (float(ce.mean()) if ce.size else None),
            "value_ce_median": _q(ce, 0.5), "z_mean": (float(z.mean()) if z.size else None),
            "draw_share": (float((np.abs(z) < 0.999).mean()) if z.size else None), "reliability": reliability}


def calibration(rows: dict[str, np.ndarray], net_hash: str, *, seed: int) -> dict[str, Any]:
    """E3: E[v] vs realised z on value-valid rows by mr = 1 / 2, with the MAE and value-CE ratios mr1/mr2 and bootstrap intervals."""
    g = lambda k: rows[f"{net_hash}:{k}"]  # noqa: E731
    valid = g("valid")
    ev, z, ce, mr = g("ev")[valid], g("z")[valid], g("value_ce")[valid], g("mr")[valid]
    out: dict[str, Any] = {"n_valid": int(valid.sum()), "n_rows": int(valid.size), "pooled": _bucket(ev, z, ce), "by_moves_remaining": {}}
    for k in (1, 2):
        out["by_moves_remaining"][str(k)] = _bucket(ev[mr == k], z[mr == k], ce[mr == k])
    e1, e2 = np.abs(ev[mr == 1] - z[mr == 1]), np.abs(ev[mr == 2] - z[mr == 2])
    c1, c2 = ce[mr == 1], ce[mr == 2]
    out["mae_ratio_mr1_over_mr2"] = (float(e1.mean() / e2.mean()) if e1.size and e2.size else None)
    out["mae_ratio_ci95"] = _bootstrap_ratio(e1, e2, seed=seed)
    out["value_ce_ratio_mr1_over_mr2"] = (float(c1.mean() / c2.mean()) if c1.size and c2.size else None)
    out["value_ce_ratio_ci95"] = _bootstrap_ratio(c1, c2, seed=seed + 1)
    return out


def kl_summary(rows: dict[str, np.ndarray], net_hash: str) -> dict[str, Any]:
    """P-B1 over full-arm rows: KL(p‖t) (the packet's), KL(t‖p) = CE − H, KL(t‖t^(1/4)), and the unsupported-prior-mass share (KL(p‖t) is clamped there)."""
    g = lambda k: rows[f"{net_hash}:{k}"]  # noqa: E731
    full, mr = g("full"), g("mr")

    def _stats(key: str, mask: np.ndarray) -> dict[str, Any]:
        a = g(key)[mask]
        return {"n": int(a.size), "median": _q(a, 0.5), "mean": (float(a.mean()) if a.size else None), "p25": _q(a, 0.25),
                "p75": _q(a, 0.75), "p90": _q(a, 0.9)}

    out: dict[str, Any] = {"n_full_arm": int(full.sum()), "n_rows": int(full.size)}
    for key in ("kl_prior_target", "kl_target_prior", "kl_target_soft", "policy_ce"):
        out[key] = {"all": _stats(key, full), "mr1": _stats(key, full & (mr == 1)), "mr2": _stats(key, full & (mr == 2))}
    unsupported = g("unsupported_mass")[full]
    out["unsupported_prior_mass"] = {"share_rows_gt_1e-3": (float((unsupported > 1e-3).mean()) if unsupported.size else None),
                                     "mean": (float(unsupported.mean()) if unsupported.size else None)}
    supported = full & (g("unsupported_mass") <= 1e-3)
    out["kl_prior_target_supported_rows_only"] = _stats("kl_prior_target", supported)
    out["alpha_zero_share_full_arm"] = (float((g("alpha")[full] <= 0.0).mean()) if full.any() else None)
    return out


def gap_table(per_ring: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """C3-1 per net: its loss on its OWN ring (train) and on the NEXT checkpoint's ring (held-out), from `per_ring` rows of `{ring_step, losses}`."""
    by_step: dict[int, dict[str, Any]] = {int(r["ring_step"]): r for r in per_ring}
    steps = sorted(by_step)
    table = []
    for j, step in enumerate(steps):
        own = next((v for v in by_step[step]["losses"].values() if int(v["step"]) == step), None)
        nxt = by_step[steps[j + 1]] if j + 1 < len(steps) else None
        held = next((v for v in nxt["losses"].values() if int(v["step"]) == step), None) if nxt else None
        row: dict[str, Any] = {"net_step": step, "train_ring": step, "heldout_ring": (steps[j + 1] if nxt else None),
                               "train_policy": (own["policy"] if own else None), "train_value": (own["value"] if own else None),
                               "heldout_policy": (held["policy"] if held else None), "heldout_value": (held["value"] if held else None)}
        if own and held:
            row["gap_policy"] = held["policy"] - own["policy"]
            row["gap_value"] = held["value"] - own["value"]
            row["gap_total"] = row["gap_policy"] + row["gap_value"]
        table.append(row)
    return table


__all__ = ["calibration", "gap_table", "kl_summary"]
