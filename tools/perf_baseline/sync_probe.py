"""PERF-BASELINE — is the forward's wall its own work, or a pipeline drain?

`py-spy` puts 40 % of the whole serving thread inside ONE line: `_node_offsets_to_batch_vec`
(`model/gnn.py`), which is `torch.repeat_interleave(arange(B), counts)` with `counts` a CUDA
tensor. That form has to report a data-dependent output length to the host, so it is a
host-device SYNC in the middle of the forward — the same mechanism R284(b)/P-MASK removed one
call earlier when it replaced `emb[bool_mask]` with `index_select`.

This probe separates the two readings a 40 % share is consistent with:
  (a) the call is EXPENSIVE — its own cost is large; or
  (b) the call is a BARRIER — it is merely where the trunk's already-launched GPU work is
      charged, and the wall would appear somewhere else if it moved.
Measured on an IDLE stream (nothing queued) and on a LOADED stream (the trunk just launched).
"""
from __future__ import annotations

import argparse
import glob
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cuda_ms, git_sha, stats, write_json  # noqa: E402
from silicon_floor import build_batch, load_single_graphs  # noqa: E402

from mantis.config.loader import load_config  # noqa: E402
from mantis.diagnostics.worker_sweep import build_sweep_net  # noqa: E402
from mantis.model.gnn import _node_offsets_to_batch_vec  # noqa: E402
from mantis.selfplay.graph_collate import stone_mask_from_batch  # noqa: E402
from mantis.selfplay.hparams import resolve_pool_encoding  # noqa: E402


def wall_ms(fn: Any, n: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run5.yaml")
    ap.add_argument("--wire-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="8,17,40")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    config = load_config(args.config)
    raw = config.model_dump()
    device = torch.device("cuda")
    spec = resolve_pool_encoding(raw, arch=None).registry_spec
    from mantis.model import amp_dtype_for, arch_from_spec_and_config

    arch = arch_from_spec_and_config(spec, raw)
    model = build_sweep_net(config, arch, device)
    model.eval()
    amp = amp_dtype_for("graph", raw["train"]["amp_dtype"])
    graphs = load_single_graphs(args.wire_dir, 2000)
    if not graphs:
        print(f"REFUSING: no captured wires under {args.wire_dir}")
        return 2
    rng = random.Random(args.seed)
    record: dict[str, Any] = {"sha": git_sha(), "wire_dir": args.wire_dir, "sizes": {}}

    for size in [int(s) for s in args.sizes.split(",")]:
        picked = [graphs[rng.randrange(len(graphs))] for _ in range(size)]
        batch = build_batch(picked, spec, str(device))
        mask = stone_mask_from_batch(batch)
        n_edges = int(batch.edge_index.shape[1])
        n_nodes = int(batch.x.shape[0])

        def trunk(b=batch) -> Any:
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp):
                return model.representation(b.x, b.edge_index, b.edge_attr)

        def full(b=batch, m=mask) -> None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp):
                model.forward_batch(b.x, b.edge_index, b.edge_attr,
                                    b.legal_node_gather, m, b.node_offsets)

        def idle_call(b=batch) -> None:
            torch.cuda.synchronize()
            _node_offsets_to_batch_vec(b.node_offsets)

        def loaded_call(b=batch) -> None:
            torch.cuda.synchronize()
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp):
                model.representation(b.x, b.edge_index, b.edge_attr)
            _node_offsets_to_batch_vec(b.node_offsets)

        # The barrier's own cost, with nothing queued behind it.
        idle = wall_ms(idle_call, args.reps, 5)
        # Idle-call wall includes its own leading synchronize; subtract a bare synchronize.
        bare_sync = wall_ms(lambda: torch.cuda.synchronize(), args.reps, 5)
        loaded = wall_ms(loaded_call, args.reps, 5)
        trunk_ev = cuda_ms(lambda: trunk(), args.reps, 5)
        full_ev = cuda_ms(lambda: full(), args.reps, 5)
        full_wall = wall_ms(full, args.reps, 5)

        record["sizes"][str(size)] = {
            "graphs": size, "nodes": n_nodes, "edges": n_edges,
            "barrier_idle_ms": stats(idle),
            "bare_synchronize_ms": stats(bare_sync),
            "trunk_plus_barrier_loaded_ms": stats(loaded),
            "trunk_cuda_event_ms": stats(trunk_ev),
            "forward_cuda_event_ms": stats(full_ev),
            "forward_wall_ms": stats(full_wall),
            "ns_per_edge_forward": stats(full_ev)["median"] * 1e6 / n_edges,
        }
        r = record["sizes"][str(size)]
        print(f"size {size:3d} ({n_edges:8d} edges): forward event={r['forward_cuda_event_ms']['median']:8.3f} "
              f"wall={r['forward_wall_ms']['median']:8.3f} | trunk event={r['trunk_cuda_event_ms']['median']:8.3f} "
              f"| barrier idle={r['barrier_idle_ms']['median']:7.4f} loaded={r['trunk_plus_barrier_loaded_ms']['median']:8.3f} "
              f"| {r['ns_per_edge_forward']:.2f} ns/edge")
        del batch, mask
        torch.cuda.empty_cache()

    write_json(args.out, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
