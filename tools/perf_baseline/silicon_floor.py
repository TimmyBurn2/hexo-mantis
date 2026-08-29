"""PERF-BASELINE §2 A + the top-kernel table — what the silicon actually charges.

Replays REAL captured wire payloads (production radius, real halo sizes) through the bare
`GnnNet.forward_batch` at batch 1 / 8 / 64 / 256, bf16 as production serves, warmed and
steady-state. Also emits the torch-profiler kernel table for the same forward.
"""
from __future__ import annotations

import argparse
import glob
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cuda_ms, git_sha, stats, write_json  # noqa: E402

from mantis.config.loader import load_config  # noqa: E402
from mantis.diagnostics.worker_sweep import build_sweep_net  # noqa: E402
from mantis.selfplay.graph_collate import (  # noqa: E402
    GraphWirePayload,
    collate_graph_batch,
    reset_semantic_canary,
    segment_softmax,
    stone_mask_from_batch,
)
from mantis.selfplay.graph_wire_split import slice_graph_wire  # noqa: E402


def merge_wires(parts: list[GraphWirePayload]) -> GraphWirePayload:
    """Concatenate single-graph wires into one block-diagonal wire.

    The exact inverse of `slice_graph_wire`, including its flat-`edge_index` trap: the wire's
    `edge_index` is `2E` flat with all sources first and all destinations second, so a merge
    must rebuild BOTH halves rather than concatenating whole arrays.
    """
    node_base = 0
    srcs, dsts, feats, coords, attrs = [], [], [], [], []
    gathers, slots, checks, stones, centers, players = [], [], [], [], [], []
    node_off, edge_off, legal_off = [0], [0], [0]
    for p in parts:
        no = np.asarray(p.node_offsets, dtype=np.int64)
        eo = np.asarray(p.edge_offsets, dtype=np.int64)
        lo = np.asarray(p.legal_offsets, dtype=np.int64)
        n_tot, e_tot = int(no[-1]), int(eo[-1])
        ei = np.asarray(p.edge_index, dtype=np.int64)
        srcs.append(ei[:e_tot] + node_base)
        dsts.append(ei[e_tot:2 * e_tot] + node_base)
        feats.append(np.asarray(p.node_feat))
        coords.append(np.asarray(p.node_coords))
        attrs.append(np.asarray(p.edge_attr))
        gathers.append(np.asarray(p.legal_node_gather, dtype=np.int64) + node_base)
        slots.append(np.asarray(p.policy_dst_slot))
        checks.append(np.asarray(p.n_nodes_checksum))
        stones.append(np.asarray(p.n_stones))
        centers.append(np.asarray(p.window_center))
        players.append(np.asarray(p.current_player))
        for k in range(1, len(no)):
            node_off.append(node_off[-1] + int(no[k]) - int(no[k - 1]))
            edge_off.append(edge_off[-1] + int(eo[k]) - int(eo[k - 1]))
            legal_off.append(legal_off[-1] + int(lo[k]) - int(lo[k - 1]))
        node_base += n_tot
    return GraphWirePayload(
        contract_version=int(parts[0].contract_version),
        builder_impl=int(parts[0].builder_impl),
        n_graphs=sum(p.n_graphs for p in parts),
        node_feat=np.concatenate(feats),
        node_coords=np.concatenate(coords),
        edge_index=np.concatenate(srcs + dsts),
        edge_attr=np.concatenate(attrs),
        node_offsets=np.asarray(node_off, dtype=np.int64),
        edge_offsets=np.asarray(edge_off, dtype=np.int64),
        legal_offsets=np.asarray(legal_off, dtype=np.int64),
        legal_node_gather=np.concatenate(gathers),
        policy_dst_slot=np.concatenate(slots),
        n_nodes_checksum=np.concatenate(checks),
        n_stones=np.concatenate(stones),
        window_center=np.concatenate(centers),
        current_player=np.concatenate(players),
    )


def load_single_graphs(wire_dir: str, cap: int) -> list[GraphWirePayload]:
    out: list[GraphWirePayload] = []
    for path in sorted(glob.glob(f"{wire_dir}/wire_*.pkl")):
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 — our own capture, this box only
        for g in range(payload.n_graphs):
            out.append(slice_graph_wire(payload, g, g + 1))
            if len(out) >= cap:
                return out
    return out


def build_batch(graphs: list[GraphWirePayload], spec: Any, device: str) -> Any:
    reset_semantic_canary()
    return collate_graph_batch(
        merge_wires(graphs), expected_version=1, trunk_size=spec.trunk_size,
        win_length=spec.win_length, node_feat_dim=spec.node_feat_dim,
        edge_feat_dim=spec.edge_feat_dim, device=device, semantic="off", canary_period=0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run5.yaml")
    ap.add_argument("--wire-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="1,8,64,256")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--graph-cap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--profile-size", type=int, default=8)
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda")
    raw = config.model_dump()
    from mantis.config.resolve.pool_encoding import resolve_pool_encoding
    from mantis.model import amp_dtype_for, arch_from_spec_and_config

    resolved = resolve_pool_encoding(raw, arch=None)
    spec = resolved.registry_spec
    arch = arch_from_spec_and_config(spec, raw)
    model = build_sweep_net(config, arch, device)
    model.eval()
    amp = amp_dtype_for("graph", raw["train"]["amp_dtype"])
    n_params = sum(p.numel() for p in model.parameters())

    graphs = load_single_graphs(args.wire_dir, args.graph_cap)
    if not graphs:
        print(f"REFUSING: no captured wires under {args.wire_dir}")
        return 2
    nodes = [int(np.asarray(g.node_offsets)[-1]) for g in graphs]
    edges = [int(np.asarray(g.edge_offsets)[-1]) for g in graphs]
    rng = random.Random(args.seed)

    record: dict[str, Any] = {
        "sha": git_sha(),
        "wire_dir": args.wire_dir,
        "n_params": n_params,
        "amp_dtype": str(amp),
        "graph_population": {
            "n_graphs_sampled": len(graphs),
            "nodes": stats([float(n) for n in nodes]),
            "edges": stats([float(e) for e in edges]),
        },
        "sizes": {},
    }

    for size in [int(s) for s in args.sizes.split(",")]:
        try:
            picked = [graphs[rng.randrange(len(graphs))] for _ in range(size)]
            batch = build_batch(picked, spec, str(device))
            mask = stone_mask_from_batch(batch)

            def fwd(b=batch, m=mask) -> None:
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=amp, enabled=True
                ):
                    model.forward_batch(b.x, b.edge_index, b.edge_attr,
                                        b.legal_node_gather, m, b.node_offsets)

            def fwd_full(b=batch, m=mask) -> None:
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=amp, enabled=True
                ):
                    logits, value, _ = model.forward_batch(
                        b.x, b.edge_index, b.edge_attr, b.legal_node_gather, m,
                        b.node_offsets)
                probs = segment_softmax(logits.float(), b.legal_offsets)
                probs.detach().cpu().numpy()
                value.detach().float().cpu().numpy()

            ms = cuda_ms(fwd, args.reps, warmup=10)
            ms_full = cuda_ms(fwd_full, args.reps, warmup=5)
            s = stats(ms)
            record["sizes"][str(size)] = {
                "batch_nodes": int(np.asarray(batch.node_offsets.cpu())[-1]),
                "batch_edges": int(batch.edge_index.shape[1]),
                "forward_ms": s,
                "forward_plus_softmax_d2h_ms": stats(ms_full),
                "ms_per_position": s["median"] / size,
            }
            print(f"size {size:4d}: forward {s['median']:.3f} ms  "
                  f"= {s['median']/size:.4f} ms/position")
            del batch, mask
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError as exc:
            record["sizes"][str(size)] = {"oom": str(exc)}
            print(f"size {size}: OOM")
            torch.cuda.empty_cache()

    # ── top-kernel table for one representative batch ────────────────────────────
    size = args.profile_size
    picked = [graphs[rng.randrange(len(graphs))] for _ in range(size)]
    batch = build_batch(picked, spec, str(device))
    mask = stone_mask_from_batch(batch)
    for _ in range(10):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp):
            model.forward_batch(batch.x, batch.edge_index, batch.edge_attr,
                                batch.legal_node_gather, mask, batch.node_offsets)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp):
                model.forward_batch(batch.x, batch.edge_index, batch.edge_attr,
                                    batch.legal_node_gather, mask, batch.node_offsets)
        torch.cuda.synchronize()
    table = prof.key_averages().table(
        sort_by="self_device_time_total", row_limit=20, max_name_column_width=70)
    record["kernel_table"] = {
        "batch_graphs": size,
        "iterations": 20,
        "table": table,
        "rows": [
            {
                "name": e.key,
                "count": e.count,
                "self_device_us": getattr(e, "self_device_time_total", 0.0),
                "device_us": getattr(e, "device_time_total", 0.0),
                "self_cpu_us": e.self_cpu_time_total,
                "cpu_us": e.cpu_time_total,
            }
            for e in sorted(
                prof.key_averages(),
                key=lambda x: -getattr(x, "self_device_time_total", 0.0),
            )[:25]
        ],
    }
    print(table)
    write_json(args.out, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
