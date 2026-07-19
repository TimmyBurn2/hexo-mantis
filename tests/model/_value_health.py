"""Vendored value-health metric math + a synthetic min-pool scoring path (O2).

Ported from the old `scripts/e1/validate_ckpt.py` + its `scripts.headswap.metrics`
(`auc`, `false_pessimism`) / `scripts.valprobe.value_health` (`compute_ece`) helpers,
so the oracle imports NO `scripts.*` (a sys.path hazard the old code carried). The
metric definitions (M1–M4) are frozen; the scoring path is exercised on a synthetic
net in-test (real-anchor end-to-end scoring DEFERS to WP11).
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch


class ScoreRow(TypedDict):
    """Per-position deploy-decoded score: min-pool decoded scalar + (dist arm) tail-mass."""
    v: float
    tail_mass: float | None

# Frozen constants (old `validate_ckpt.py` / `headswap/targets.py`).
LOSS_TAIL_BIN = 16          # inclusive upper bin index for the dist-arm tail sum
FALSE_PESS_THRESHOLD = -0.5
ECE_N_BINS = 10

# SHA256 of the FROZEN default probe files (halt if a default-path file differs).
PROBE_SHA256 = "7899fa136ac083f0a428f5f6fa4c89918f1ba82c85618e8c7369a19506a9adb6"
NEGATIVES_SHA256 = "8faa6af74a7640f869cc3b1c4cb058b62660a052c5381e8ab7ad740a38cafef3"

# Stable row schema (order fixed).
ROW_KEYS: tuple[str, ...] = (
    "step", "arm", "ckpt_sha", "encoding", "mean_v_on_losses", "ece",
    "tail_mass_auc", "decoded_auc", "false_pessimism",
    "recognition_lag_mean_v_on_losses", "recognition_lag_note", "n_loss", "n_safe",
)


# ── vendored metric helpers ──────────────────────────────────────────────────


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC of `scores` predicting label==1 via the rank-sum identity (midranks for
    ties). Invariant to any strictly-monotone transform. nan if either class empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = midrank
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def false_pessimism(neg_scores: Sequence[float], threshold: float = FALSE_PESS_THRESHOLD) -> float:
    """Fraction of SAFE negatives scored pessimistic (decoded v <= threshold)."""
    s = np.asarray(neg_scores, dtype=float)
    if len(s) == 0:
        return float("nan")
    return float((s <= threshold).mean())


def compute_ece(v_vals: list[float], outcomes: list[float], n_bins: int = ECE_N_BINS) -> float:
    """Expected Calibration Error (equal-width bins on P_win=(v+1)/2). v_vals in
    [-1,1] head-perspective; outcomes ∈ {+1,-1}."""
    if not v_vals:
        return float("nan")
    p_wins = [(v + 1.0) / 2.0 for v in v_vals]
    y_wins = [(o + 1.0) / 2.0 for o in outcomes]
    n = len(p_wins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [j for j in range(n) if lo <= p_wins[j] < hi]
        if i == n_bins - 1:
            in_bin = [j for j in range(n) if lo <= p_wins[j] <= hi]
        if not in_bin:
            continue
        bin_n = len(in_bin)
        avg_conf = np.mean([p_wins[j] for j in in_bin])
        avg_acc = np.mean([y_wins[j] for j in in_bin])
        ece += (bin_n / n) * abs(avg_conf - avg_acc)
    return float(ece)


# ── M1–M4 (frozen defs) ──────────────────────────────────────────────────────


def compute_metrics(
    arm: str,
    loss_scores: list[ScoreRow],
    safe_scores: list[ScoreRow],
) -> dict[str, float | None]:
    """M1–M4 from per-position {v, tail_mass} score dicts."""
    loss_v = [s["v"] for s in loss_scores]
    safe_v = [s["v"] for s in safe_scores]

    m1 = float(np.mean(loss_v)) if loss_v else float("nan")

    all_v = loss_v + safe_v
    all_outcomes = [-1.0] * len(loss_v) + [1.0] * len(safe_v)
    ece = compute_ece(all_v, all_outcomes)

    labels = [1] * len(loss_v) + [0] * len(safe_v)
    if arm == "dist65":
        tail: list[float] = []
        for s in loss_scores + safe_scores:
            tm = s["tail_mass"]
            assert tm is not None, "dist65 arm must populate tail_mass"
            tail.append(tm)
        tail_mass_auc = auc(tail, labels)
        decoded_auc = None
    else:
        neg_v = [-v for v in all_v]
        decoded_auc = auc(neg_v, labels)
        tail_mass_auc = None

    fp = false_pessimism(safe_v, threshold=FALSE_PESS_THRESHOLD)
    return {
        "mean_v_on_losses": m1,
        "ece": ece,
        "tail_mass_auc": tail_mass_auc,
        "decoded_auc": decoded_auc,
        "false_pessimism": fp,
    }


RECOGNITION_LAG_NOTE = "harness not wired; use mean_v_on_losses"


def metrics_from_decoded(
    arm: str, loss_v, safe_v, loss_tail=None, safe_tail=None
) -> dict[str, float | None]:
    """M1–M4 directly from the frozen decoded-v arrays (the O2a golden path)."""
    loss_scores: list[ScoreRow] = [
        {"v": float(v), "tail_mass": (None if loss_tail is None else float(loss_tail[i]))}
        for i, v in enumerate(loss_v)
    ]
    safe_scores: list[ScoreRow] = [
        {"v": float(v), "tail_mass": (None if safe_tail is None else float(safe_tail[i]))}
        for i, v in enumerate(safe_v)
    ]
    return compute_metrics(arm, loss_scores, safe_scores)


# ── SHA guard (O2c) ──────────────────────────────────────────────────────────


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_probe_shas(probe_path: str, negatives_path: str, *, is_default: bool) -> None:
    """When `is_default` (the frozen default paths are in use), assert both files
    match the pinned SHAs; a mismatch raises RuntimeError. Custom paths SKIP the
    guard (matching the old `_verify_probe_shas` semantics)."""
    if not is_default:
        return
    ph = _file_sha256(probe_path)
    if ph != PROBE_SHA256:
        raise RuntimeError(
            f"probe SHA mismatch — HALTING. file={probe_path} got={ph} want={PROBE_SHA256}"
        )
    nh = _file_sha256(negatives_path)
    if nh != NEGATIVES_SHA256:
        raise RuntimeError(
            f"negatives SHA mismatch — HALTING. file={negatives_path} "
            f"got={nh} want={NEGATIVES_SHA256}"
        )


# ── synthetic-net scoring (O2b) ──────────────────────────────────────────────


def assert_arm_matches(model, arm: str) -> None:
    """A dist65 net scored as scalar (or vice versa) is a hard error — never a
    silent mis-decode (mirrors the old gated-loader arm check)."""
    if arm not in ("scalar", "dist65"):
        raise ValueError(f"unknown arm {arm!r}; expected 'scalar' or 'dist65'")
    if model.value_head_type != arm:
        raise ValueError(
            f"arm/checkpoint value_head_type mismatch: declared arm={arm!r} but "
            f"model has value_head_type={model.value_head_type!r}."
        )


@torch.inference_mode()
def score_cluster_tensor(model, arm: str, x_kchw: torch.Tensor) -> ScoreRow:
    """Deploy-decoded, min-pooled score for ONE board's (K, C, H, W) cluster tensor:
    forward the full net, argmin the decoded scalar over K clusters (== infer_batch's
    min-pool); for the dist arm, take that cluster's tail-mass P(v<=-0.5)."""
    log_policy, value, v_aux = model(x_kchw)
    v_k = value.squeeze(-1).float()                     # (K,)
    k_star = int(torch.argmin(v_k).item())
    if arm == "dist65":
        probs = torch.softmax(v_aux[k_star].float(), dim=-1)
        tail = float(probs[: LOSS_TAIL_BIN + 1].sum().item())
        return {"v": float(v_k[k_star].item()), "tail_mass": tail}
    return {"v": float(v_k[k_star].item()), "tail_mass": None}


def build_row(arm: str, ckpt_sha: str, encoding: str, metrics: dict[str, float | None],
              n_loss: int, n_safe: int, step: int | None = None) -> dict[str, object]:
    """Assemble the stable ROW_KEYS row (rounded to 8dp), matching the old schema."""
    def _r(x: float | None) -> float | None:
        if x is None:
            return None
        if np.isnan(x) or np.isinf(x):
            return None
        return round(float(x), 8)

    row = {
        "step": step,
        "arm": arm,
        "ckpt_sha": ckpt_sha,
        "encoding": encoding,
        "mean_v_on_losses": _r(metrics["mean_v_on_losses"]),
        "ece": _r(metrics["ece"]),
        "tail_mass_auc": _r(metrics["tail_mass_auc"]),
        "decoded_auc": _r(metrics["decoded_auc"]),
        "false_pessimism": _r(metrics["false_pessimism"]),
        "recognition_lag_mean_v_on_losses": None,
        "recognition_lag_note": RECOGNITION_LAG_NOTE,
        "n_loss": n_loss,
        "n_safe": n_safe,
    }
    return {k: row[k] for k in ROW_KEYS}
