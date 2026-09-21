"""`python tools/probe1.py <reading> …` — one subcommand per PROBE-1 reading, JSON out, every row naming its inputs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mantis.diagnostics.ring_reader import load_ring
from mantis.util.git import head_sha, is_dirty

from .nets import load_net, open_ring, read_ring
from .proofs import proof_rate
from .readings import calibration, gap_table, kl_summary
from .rings import decompose_rings, spread_positions
from .spread import spread_series
from .swa import average_checkpoints

DEFAULT_SEED = 20260921
RING_SUFFIX = ".ring.bin"


def _log(msg: str) -> None:
    print(f"[probe1 {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _host() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    return {"head": head_sha(repo), "dirty": is_dirty(repo), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "utc": time.strftime("%FT%TZ", time.gmtime())}


def _write(out: Path, body: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"host": _host(), **body}, indent=1), encoding="utf-8")
    _log(f"wrote {out}")


def _ring_of(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.name + RING_SUFFIX)


def cmd_decompose(a: argparse.Namespace) -> int:
    _write(a.out, {"reading": "1 decomposer", "rings": decompose_rings([Path(p) for p in a.rings])})
    return 0


def cmd_netread(a: argparse.Namespace) -> int:
    net = load_net(a.checkpoint)
    ring = a.ring or _ring_of(a.checkpoint)
    buffer, n = open_ring(ring, seed=a.seed)
    read = read_ring([net], buffer, batches=a.batches, batch_size=a.batch_size, threads=a.threads, rows=True, log=_log)
    rows = read.pop("rows")
    npz = a.out.with_suffix(".rows.npz")
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **{k.split(":", 1)[1]: v for k, v in rows.items()})
    _write(a.out, {"reading": "2 calibration + 5 kl", "checkpoint": a.checkpoint.name, "net_hash": net.net_hash, "step": net.step,
                   "ring": ring.name, "ring_rows": n, "seed": a.seed, "sampled": read,
                   "calibration": calibration(rows, net.net_hash, seed=a.seed), "kl": kl_summary(rows, net.net_hash),
                   "rows_file": npz.name})
    return 0


def cmd_gap(a: argparse.Namespace) -> int:
    ckpts = sorted(p for p in a.checkpoints.glob(f"{a.run_id}_*.ckpt") if not p.name.endswith(".bak"))
    nets = [load_net(p) for p in ckpts]
    _log(f"{len(nets)} nets: {[n.step for n in nets]}")
    per_ring = []
    for j, net in enumerate(nets):
        ring = _ring_of(net.path)
        if not ring.is_file():
            _log(f"{net.path.name}: no ring beside it, skipped")
            continue
        readers = [net] + ([nets[j - 1]] if j > 0 else [])
        buffer, n = open_ring(ring, seed=a.seed)
        _log(f"ring {net.step} ({n} rows) read by {[r.step for r in readers]}")
        read = read_ring(readers, buffer, batches=a.batches, batch_size=a.batch_size, threads=a.threads, rows=False, log=_log)
        per_ring.append({"ring": ring.name, "ring_step": net.step, "ring_rows": n, **read})
        _write(a.out, {"reading": "3 gap (partial)", "seed": a.seed, "per_ring": per_ring, "table": gap_table(per_ring)})
    _write(a.out, {"reading": "3 gap", "seed": a.seed, "batches": a.batches, "batch_size": a.batch_size, "per_ring": per_ring,
                   "table": gap_table(per_ring)})
    return 0


def cmd_proofs(a: argparse.Namespace) -> int:
    ring = load_ring(a.ring)
    body: dict[str, Any] = {"reading": "4 proofs", "ring": a.ring.name, "ring_rows": ring.header.size, "seed": a.seed, "arms": []}
    for depth in a.depths:
        _log(f"depth {depth} plies, budget {a.budget}, rows {a.rows}")
        body["arms"].append(proof_rate(ring, rows=a.rows, seed=a.seed, depth=depth, node_budget=a.budget))
        _write(a.out, body)
    return 0


def cmd_spread(a: argparse.Namespace) -> int:
    ring = load_ring(a.ring)
    positions = spread_positions(ring, seed=a.seed, n=a.positions)
    _log(f"{len(positions)} positions from {a.ring.name}, plies {[p['ply'] for p in positions]}")
    series = spread_series(a.checkpoints, positions, threads=a.threads, log=_log)
    _write(a.out, {"reading": "6 spread series", "ring": a.ring.name, "seed": a.seed, "positions": positions, "series": series})
    return 0


def cmd_swa(a: argparse.Namespace) -> int:
    record = average_checkpoints([Path(p) for p in a.checkpoints], config_path=a.config, run_id=a.run_id, out_dir=a.out_dir)
    _write(a.out, {"reading": "7 ema cell (the averaged net; the cell is the box's)", **record})
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The five dev-side readings and reading 7's averaged net; the EMA CELL itself is the box's."""
    ap = argparse.ArgumentParser(prog="python tools/probe1.py", description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--threads", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decompose", help="reading 1: the one-hot share with tail-only rows excluded, by moves_remaining")
    p.add_argument("--rings", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_decompose)
    p = sub.add_parser("netread", help="readings 2 + 5: one net on one ring, per-row E[v]/z and the KLs")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ring", type=Path, default=None, help="default: the ring beside the checkpoint")
    p.add_argument("--batches", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_netread)
    p = sub.add_parser("gap", help="reading 3: every net on its own ring and on the next checkpoint's ring")
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--run-id", default="run8")
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_gap)
    p = sub.add_parser("proofs", help="reading 4: the tactics solver over ring roots")
    p.add_argument("--ring", type=Path, required=True)
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--depths", type=int, nargs="+", default=[6])
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_proofs)
    p = sub.add_parser("spread", help="reading 6: the symmetry spread over fixed positions per checkpoint")
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--ring", type=Path, required=True, help="the ring the positions are drawn from")
    p.add_argument("--positions", type=int, default=24)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_spread)
    p = sub.add_parser("swa", help="reading 7's net: the uniform weight-average of stamped checkpoints, written through the one writer")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--config", type=Path, required=True, help="the run's minted config the stamp carries")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(fn=cmd_swa)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Run one reading; 0 on success, 2 on a refusal (printed to stderr)."""
    a = build_parser().parse_args(argv)
    torch.set_num_threads(int(a.threads))
    try:
        return int(a.fn(a))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"probe1: REFUSED: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
