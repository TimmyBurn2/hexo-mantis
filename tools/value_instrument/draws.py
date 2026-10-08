"""A net read on a ring's seeded draws through the trainer's batch preparation, each draw traced to its ring slot."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from mantis._engine import HexgBuffer
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.diagnostics.ring_reader import Ring, load_ring
from mantis.model.dist65 import VALUE_SUPPORT, scalar_to_two_hot
from mantis.train.coordinator.dispatch import _build_graph_parts, resolve_step_spec
from mantis.train.losses import rebuild_sparse_target, segment_softmax, segment_sum
from mantis.util.loadpkg import load_tools_package

load_tools_package("probe1")
open_ring = importlib.import_module("probe1.nets").open_ring  # the ring at the reader's encoding: one implementation

BATCH = 256
TRACE_SCALE = float(1 << 17)
_SUPPORT = VALUE_SUPPORT.to(torch.float64)


class TraceMismatchError(RuntimeError):
    """A read's draws disagree with the tracer's slots."""


def push_ring(buf: Any, ring: Ring, outcome: np.ndarray | None = None) -> int:
    """Push every row of `ring` in slot order, `outcome` replacing the ring's when given. Raises: ValueError (the engine's) on a malformed row."""
    for i in range(len(ring.game_id)):
        buf.push_graph_position(
            [(int(s["q"]), int(s["r"]), int(s["p"])) for s in ring.row_stones(i)],
            [(int(v["q"]), int(v["r"]), float(v["prob"])) for v in ring.row_visits(i)],
            int(ring.current_player[i]), int(ring.moves_remaining[i]), int(ring.ply_index[i]),
            bool(ring.is_full_search[i]), float(ring.outcome[i] if outcome is None else outcome[i]),
            bool(ring.value_valid[i]), int(ring.game_length[i]), int(ring.game_id[i]), float(ring.tail_mass[i]))
    return len(ring.game_id)


def trace(ring: Ring, *, seed: int, batches: int, threads: int, encoding: str) -> np.ndarray:
    """Each draw's ring slot: the rows pushed with outcome = slot / 2^17 (exact in f32), sampled at `seed` in an `encoding` buffer; the sampler reads no outcome. Raises: ValueError when the ring is too large to code."""
    n = len(ring.game_id)
    if n >= (1 << 24):
        raise ValueError(f"the tracer cannot code {n} rows in f32")
    buf = HexgBuffer(max(ring.header.size, 8), encoding, ring.header.max_visits)
    push_ring(buf, ring, outcome=np.arange(n) / TRACE_SCALE)
    buf.seed_sampler(seed)
    out = []
    for _ in range(batches):
        _wire, targets = buf.sample_graph_batch(BATCH, augment=False, n_threads=threads)
        out.append(np.rint(np.asarray(targets.outcomes, dtype=np.float64) * TRACE_SCALE).astype(np.int64))
    return np.concatenate(out)


@torch.no_grad()
def _rows(model: Any, inputs: Any) -> dict[str, np.ndarray]:
    logits, _v, bins = model.forward_batch(inputs.x, inputs.edge_index, inputs.edge_attr, inputs.legal_index,
                                           inputs.stone_mask, node_offsets=inputs.node_offsets)
    lo = inputs.legal_offsets
    p = segment_softmax(logits.to(torch.float32), lo)
    t = rebuild_sparse_target(inputs.policy_target.to(torch.float32), p, inputs.explicit_mask, inputs.tail_mass, lo)
    z = inputs.outcomes.reshape(-1).to(torch.float32)
    logbins = torch.log_softmax(bins.to(torch.float64), dim=-1).cpu()
    pb = logbins.exp()
    # 1 ± E summed over the bins that carry them, so neither cancels near |E| = 1.
    one_plus, one_minus = (pb * (1.0 + _SUPPORT)).sum(dim=-1), (pb * (1.0 - _SUPPORT)).sum(dim=-1)
    return {"u": (torch.log(one_plus) - torch.log(one_minus)).numpy(), "e": (one_plus - 1.0).numpy(),
            "z": z.cpu().numpy(), "valid": inputs.value_valid.reshape(-1).cpu().numpy().astype(bool),
            "full": inputs.policy_row_weight.reshape(-1).cpu().numpy() > 0,
            "tail": inputs.tail_mass.reshape(-1).cpu().numpy(),
            "raw_ce": (-(scalar_to_two_hot(z).cpu().to(torch.float64) * logbins).sum(dim=-1)).numpy(),
            "policy_ce": segment_sum(-(t * torch.log(p.clamp_min(1e-12))), lo).cpu().numpy().astype(np.float64)}


def read(model: Any, config: dict[str, Any], ring_path: Path, *, seed: int, batches: int, device: torch.device,
         threads: int, dump_dir: Path) -> dict[str, np.ndarray]:
    """`model` read in eval mode, FP32, per draw, with its slot, game and ply; `games` is the ring's. Raises: TraceMismatchError when a draw's z, value-valid, tail mass or full-search flag (one way: alpha = 1 drops it) differs from its slot's; ValueError and OSError from the ring."""
    ring = load_ring(ring_path)
    encoding = str(config["identity"]["encoding"])  # the net's own: graphs are built at the reader's encoding
    slot = trace(ring, seed=seed, batches=batches, threads=threads, encoding=encoding)
    buf, _rows_loaded = open_ring(ring_path, seed=seed, encoding=encoding)
    model.eval()
    stub = SimpleNamespace(device=device, checkpoint_dir=str(dump_dir))
    spec = resolve_step_spec(config)
    cols: dict[str, list[np.ndarray]] = {}
    for _ in range(batches):
        built = _build_graph_parts(stub, buf, spec, batch_size=BATCH, augment=False,
                                   caps_provider=lambda: resolve_microbatch_caps(config),
                                   sample_threads_provider=lambda: threads, value_mask=None)
        for make in built["parts"]:
            for k, v in _rows(model, make()).items():
                cols.setdefault(k, []).append(v)
    r = {k: np.concatenate(v) for k, v in cols.items()}
    bad = {"z": int((ring.outcome[slot] != r["z"]).sum()), "valid": int((ring.value_valid[slot] != r["valid"]).sum()),
           "tail": int((ring.tail_mass[slot] != r["tail"]).sum()),
           "full": int((r["full"] & ~ring.is_full_search[slot].astype(bool)).sum())}
    if any(bad.values()):
        raise TraceMismatchError(f"the tracer disagrees with the read: {bad}")
    return {**r, "slot": slot, "game_id": ring.game_id[slot].astype(np.int64), "ply": ring.ply_index[slot].astype(np.int32),
            "games": np.unique(ring.game_id).astype(np.int64)}


__all__ = ["BATCH", "TraceMismatchError", "push_ring", "read", "trace"]
