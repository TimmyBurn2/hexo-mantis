"""`read` a stamped checkpoint on a held-out ring (and a train ring, for the gap); `compare` two sets of reads, `lagged` two saves of one run; `exams` a position set."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mantis.model import build_net
from mantis.train.checkpoints import deploy_state, load_checkpoint

from .draws import BATCH, TraceMismatchError
from .draws import read as read_draws
from .metrics import (
    EFFECT,
    band_masks,
    calibrated,
    crossfit_losses,
    game_folds,
    game_se,
    line_and_power,
    side,
    verdicts,
)

# The seeds, batch size and batch count define "the same rows": change one and no earlier read compares.
HELDOUT_SEED = 20260929
TRAIN_SEED = 20261003
FOLD_SEED = 20261002
BATCHES = 160
BOOT_SEED = 20261001
_KEEP = ("u", "raw_ce", "z", "valid", "game_id", "ply", "slot", "fold")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _outcomes(z: np.ndarray) -> np.ndarray:
    """The value-valid draws' z as ±1. Raises: ValueError on any other outcome (a draw row or a mixed target)."""
    if not np.all(np.abs(z) == 1.0):
        raise ValueError(f"value-valid draws carry {int((np.abs(z) != 1.0).sum())} outcomes other than ±1")
    return z.astype(np.float64)


def read(a: argparse.Namespace) -> dict[str, Any]:
    """Read one stamped checkpoint's served net and write its rows and metrics. Raises: ValueError (no arch, a non-±1 outcome, a malformed ring), TraceMismatchError, CheckpointStampError, RuntimeError (a state that does not fit), OSError."""
    ck = load_checkpoint(a.ckpt)
    if ck.metadata.arch is None:
        raise ValueError(f"{a.ckpt}: the stamp carries no architecture to build the net from")
    model = build_net(ck.metadata.arch)
    state, label = deploy_state(ck)
    model.load_state_dict(state)  # the SERVED copy (EMA when the stamp has one): the net that plays is the net read
    device = torch.device(a.device)
    model.to(device)
    config = dict(ck.config)
    body: dict[str, Any] = {"ckpt": str(a.ckpt), "ckpt_sha256": _sha256(a.ckpt), "weights": label, "device": a.device,
                            "batch": BATCH, "batches": a.batches, "fold_seed": a.fold_seed, "precision": "fp32"}
    rows: dict[str, Any] = {}
    for name, ring_path, seed in (("heldout", a.heldout, a.seed), ("train", a.train, a.train_seed)):
        if ring_path is None:
            continue
        r = read_draws(model, config, ring_path, seed=seed, batches=a.batches, device=device, threads=a.threads,
                       dump_dir=a.out.parent / "collate")
        r["fold"] = game_folds(r["game_id"], a.fold_seed, r["games"])
        v = r["valid"]
        body[name] = {"ring": str(ring_path), "ring_sha256": _sha256(ring_path), "seed": seed,
                      "policy_ce": float(r["policy_ce"][r["full"]].mean()) if r["full"].any() else None,
                      **side(r["u"][v], _outcomes(r["z"][v]), r["raw_ce"][v], r["fold"][v], r["ply"][v])}
        rows.update({f"{name}__{k}": r[k] for k in _KEEP})
    if "train" in body:
        body["gap"] = {"cf_ce": body["heldout"]["overall"]["cf_ce"] - body["train"]["overall"]["cf_ce"],
                       "raw_ce": body["heldout"]["overall"]["raw_ce"] - body["train"]["overall"]["raw_ce"]}
    np.savez(a.out.with_suffix(".rows.npz"), **rows)
    a.out.write_text(json.dumps(body, indent=1, allow_nan=False), encoding="utf-8")
    return body


def _loaded(path: Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"rows": dict(np.load(Path(path).with_suffix(".rows.npz"))), "ring_sha256": body["heldout"]["ring_sha256"]}


def _same_draws(reads: list[dict[str, Any]], cmd: str) -> None:
    """Raises: ValueError when the reads were not taken on the same held-out draws and folds."""
    first = reads[0]["rows"]
    for x in reads[1:]:
        if x["ring_sha256"] != reads[0]["ring_sha256"] or any(
                not np.array_equal(x["rows"][f"heldout__{k}"], first[f"heldout__{k}"]) for k in ("slot", "valid", "fold")):
            raise ValueError(f"{cmd}: the reads were not taken on the same held-out draws and folds")


def _paired(arm: list[Path], base: list[Path], sigma: float | None, source: str, effect: float, cmd: str) -> dict[str, Any]:
    """Overall and per ply band: `sigma` rules overall when given, a band's own groups' spread rules it when it has one. Raises: ValueError (an --effect <= 0, other draws, no seed spread), OSError, KeyError."""
    if not effect > 0.0:
        raise ValueError(f"{cmd}: --effect must be positive, got {effect}")
    reads = [_loaded(p) for p in [*arm, *base]]
    _same_draws(reads, cmd)
    first = reads[0]["rows"]
    v = first["heldout__valid"]
    gid, fold, ply = (first[f"heldout__{k}"][v] for k in ("game_id", "fold", "ply"))
    y = _outcomes(first["heldout__z"][v])
    u = [x["rows"]["heldout__u"][v] for x in reads]

    def contrast(sel: np.ndarray, prefer_groups: bool) -> dict[str, Any]:
        if not sel.any():
            return {"n": 0}
        loss = [crossfit_losses(x[sel], y[sel], fold[sel]) for x in u]
        arm_loss, base_loss = loss[:len(arm)], loss[len(arm):]
        groups = [[float(x.mean()) for x in arm_loss], [float(x.mean()) for x in base_loss]]
        df = sum(len(g) - 1 for g in groups)
        if df > 0 and (prefer_groups or sigma is None):
            sig, src = math.sqrt(sum(float(np.sum((np.asarray(g) - np.mean(g)) ** 2)) for g in groups) / df), "groups"
        elif sigma is None:
            raise ValueError(f"{cmd}: one read per side has no seed spread; pass a family's pooled --sigma")
        else:
            sig, src = sigma, source
        sd_seed = sig * math.sqrt(1.0 / len(arm_loss) + 1.0 / len(base_loss))
        g = game_se(np.mean(arm_loss, axis=0) - np.mean(base_loss, axis=0), gid[sel], BOOT_SEED)
        lp = line_and_power(sd_seed, g["se_game"], effect)
        vd = verdicts(g["diff"], lp["line"], effect)
        return {"n": int(sel.sum()), **g, "sigma_seed": sig, "sigma_df": df, "sigma_source": src, "sd_seed": sd_seed,
                **lp, "beats": vd["detection"], "powered": lp["power"] >= 0.8, **vd}
    return {**contrast(np.ones(len(y), bool), False),
            "bands": {band: contrast(sel, True) for band, sel in band_masks(ply).items()}}


def compare(a: argparse.Namespace) -> dict[str, Any]:
    """The arm reads against the base reads on the same held-out draws. Raises: ValueError (different draws or folds, no seed spread and no --sigma, an --effect <= 0), OSError, KeyError."""
    return _paired(a.arm, a.base, a.sigma, "--sigma", a.effect, "compare")


def lagged(a: argparse.Namespace) -> dict[str, Any]:
    """Current minus lagged, two saves of one run on the same held-out draws: no seed spread, so game noise only. Raises: ValueError (different draws or folds, an --effect <= 0), OSError, KeyError."""
    return _paired([a.current], [a.lagged], 0.0, "none (one run): game noise only", a.effect, "lagged")


def _se(x: np.ndarray) -> float | None:
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None


def exams(a: argparse.Namespace) -> dict[str, Any]:
    """A JSONL field's per-position values, raw and calibrated at the read's held-out temperature. Raises: ValueError (no temperature, no rows, a non-finite or missing value, a malformed line, the read's JSON), OSError."""
    body = json.loads(Path(a.read).read_text(encoding="utf-8"))
    t = ((body.get("heldout") or {}).get("overall") or {}).get("temperature")
    if t is None:
        raise ValueError(f"exams: {a.read} carries no held-out temperature (a read with no skill calibrates nothing)")
    vals: list[float] = []
    for i, line in enumerate(Path(a.rows).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"exams: line {i} of {a.rows} is not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"exams: line {i} of {a.rows} is not a JSON object")
        if a.net is not None and row.get("net") != a.net:
            continue
        x = row.get(a.field)
        if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
            raise ValueError(f"exams: line {i} of {a.rows}: {a.field!r} is {x!r}, not a finite number")
        vals.append(float(x))
    if not vals:
        raise ValueError(f"exams: no rows in {a.rows}" + (f" for net {a.net!r}" if a.net is not None else ""))
    raw = np.asarray(vals)
    cal = calibrated(raw, float(t))
    out = {"n": len(raw), "raw_mean": float(raw.mean()), "calibrated_mean": float(cal.mean()), "se_raw": _se(raw),
           "se_calibrated": _se(cal), "temperature": float(t)}
    return out if a.floor is None else {**out, "floor": a.floor, "holds": out["calibrated_mean"] >= a.floor}


def main(argv: list[str] | None = None) -> int:
    """The command line; a refusal prints its reason and returns 2."""
    ap = argparse.ArgumentParser(prog="value_instrument")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="read a stamped checkpoint's value head on a held-out (and a train) ring")
    r.add_argument("--ckpt", type=Path, required=True)
    r.add_argument("--heldout", type=Path, required=True)
    r.add_argument("--train", type=Path, default=None)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--seed", type=int, default=HELDOUT_SEED)
    r.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    r.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    r.add_argument("--batches", type=int, default=BATCHES)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c = sub.add_parser("compare", help="an arm's reads against a base's, on the same held-out draws")
    c.add_argument("--arm", type=Path, nargs="+", required=True)
    c.add_argument("--base", type=Path, nargs="+", required=True)
    c.add_argument("--sigma", type=float, default=None, help="a family's pooled seed SD (default: the two groups')")
    c.add_argument("--effect", type=float, default=EFFECT)
    lg = sub.add_parser("lagged", help="two saves of one run on the same held-out draws, game noise only")
    lg.add_argument("--current", type=Path, required=True)
    lg.add_argument("--lagged", type=Path, required=True)
    lg.add_argument("--effect", type=float, default=EFFECT)
    e = sub.add_parser("exams", help="a position set's values calibrated at a read's held-out temperature")
    e.add_argument("--read", type=Path, required=True)
    e.add_argument("--rows", type=Path, required=True)
    e.add_argument("--field", required=True)
    e.add_argument("--net", default=None)
    e.add_argument("--floor", type=float, default=None)
    a = ap.parse_args(argv)
    try:
        out = {"read": read, "compare": compare, "lagged": lagged, "exams": exams}[a.cmd](a)
    except (ValueError, TraceMismatchError) as exc:
        print(f"value_instrument: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out if a.cmd != "read" else {k: out[k] for k in out if k in ("heldout", "gap")}, indent=1,
                     allow_nan=False))
    return 0


__all__ = ["compare", "exams", "lagged", "main", "read"]
