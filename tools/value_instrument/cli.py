"""`read` a stamped checkpoint on a held-out ring (and a train ring, for the gap); `compare` two sets of reads."""
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
from .metrics import crossfit_losses, game_folds, game_se, line_and_power, side

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


def compare(a: argparse.Namespace) -> dict[str, Any]:
    """The arm reads against the base reads on the same held-out draws. Raises: ValueError (different draws or folds, or no seed spread and no --sigma), OSError, KeyError."""
    reads = [_loaded(p) for p in [*a.arm, *a.base]]
    first = reads[0]["rows"]
    for x in reads[1:]:
        if x["ring_sha256"] != reads[0]["ring_sha256"] or any(
                not np.array_equal(x["rows"][f"heldout__{k}"], first[f"heldout__{k}"]) for k in ("slot", "valid", "fold")):
            raise ValueError("compare: the reads were not taken on the same held-out draws and folds")
    v = first["heldout__valid"]
    gid, fold = first["heldout__game_id"][v], first["heldout__fold"][v]
    y = _outcomes(first["heldout__z"][v])
    loss = [crossfit_losses(x["rows"]["heldout__u"][v], y, fold) for x in reads]
    arm, base = loss[:len(a.arm)], loss[len(a.arm):]
    groups = [[float(x.mean()) for x in arm], [float(x.mean()) for x in base]]
    df = sum(len(g) - 1 for g in groups)
    if a.sigma is None and df == 0:
        raise ValueError("compare: one read per side has no seed spread; pass a family's pooled --sigma")
    sigma = a.sigma if a.sigma is not None else math.sqrt(
        sum(float(np.sum((np.asarray(g) - np.mean(g)) ** 2)) for g in groups) / df)
    sd_seed = sigma * math.sqrt(1.0 / len(arm) + 1.0 / len(base))
    g = game_se(np.mean(arm, axis=0) - np.mean(base, axis=0), gid, BOOT_SEED)
    lp = line_and_power(sd_seed, g["se_game"])
    return {**g, "sigma_seed": sigma, "sigma_df": df, "sd_seed": sd_seed, **lp, "beats": g["diff"] <= -lp["line"],
            "powered": lp["power"] >= 0.8}


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
    a = ap.parse_args(argv)
    try:
        out = read(a) if a.cmd == "read" else compare(a)
    except (ValueError, TraceMismatchError) as exc:
        print(f"value_instrument: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out if a.cmd == "compare" else {k: out[k] for k in out if k in ("heldout", "gap")}, indent=1,
                     allow_nan=False))
    return 0


__all__ = ["compare", "main", "read"]
