"""The net-on-ring readings through the PRODUCTION sample + collate: per-row E[v] / z / KLs (readings 2, 5) and the batch losses (3)."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mantis._engine import HexgBuffer
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.diagnostics.ring_reader import _read_header
from mantis.model import build_net
from mantis.model.dist65 import binned_value_loss, decode_binned_value, scalar_to_two_hot
from mantis.model.identity import net_param_hash
from mantis.train.checkpoints import load_checkpoint
from mantis.train.coordinator.dispatch import _build_graph_parts
from mantis.train.losses import (
    ragged_policy_ce,
    rebuild_sparse_target,
    segment_ids,
    segment_softmax,
    segment_sum,
)

SOFT_POLICY_TEMPERATURE = 4.0  # P-B1's target^(1/4), KataGo's T


@dataclass
class Net:
    """A stamped checkpoint as an eval-mode net on CPU, with the facts its readings cite."""

    path: Path
    step: int
    net_hash: str
    model: torch.nn.Module
    config: dict[str, Any]
    spec: Any


def load_net(path: Path) -> Net:
    """The checkpoint-stamp loader, the stamp's arch rebuilt, eval mode; raises `RuntimeError` when the stamp resolves no arch."""
    from mantis.encoding import lookup

    ck = load_checkpoint(path)
    if ck.metadata.arch is None:
        raise RuntimeError(f"{path.name}: the stamp resolves no arch, so the net cannot be rebuilt")
    model = build_net(ck.metadata.arch)
    model.load_state_dict(ck.model_state)  # the LEARNER's weights: the probe reads the trained net, not a deploy view
    model.eval()
    return Net(path=path, step=int(ck.metadata.step), net_hash=net_param_hash(model), model=model,
               config=dict(ck.config), spec=lookup(str(ck.metadata.encoding_name)))


def open_ring(path: Path, *, seed: int) -> tuple[HexgBuffer, int]:
    """The ring file loaded into the engine's own buffer at its own geometry; `(buffer, rows)`."""
    header, _ = _read_header(memoryview(path.read_bytes()[:4096]))
    buffer = HexgBuffer(max(header.size, 8), header.encoding, header.max_visits)
    loaded = int(buffer.load_from_path(str(path)))
    if loaded < 1:
        raise ValueError(f"{path.name}: loaded no records")
    buffer.seed_sampler(seed)
    return buffer, loaded


class _Stub:
    device = torch.device("cpu")


def sample_parts(buffer: HexgBuffer, config: dict[str, Any], spec: Any, *, batch_size: int,
                 threads: int) -> dict[str, Any]:
    """ONE production sample (no augmentation, no recency) as the trainer's lazy micro-batch parts + denominators."""
    caps = resolve_microbatch_caps(config)
    return _build_graph_parts(
        _Stub(), buffer, spec, batch_size=batch_size, augment=False, recency_weight=0.0,
        caps_provider=lambda: caps, sample_threads_provider=lambda: threads,
        fast_policy_weight_provider=lambda: float(config["train"]["fast_policy_weight"]))


@torch.no_grad()
def read_rows(model: torch.nn.Module, inputs: Any) -> dict[str, np.ndarray]:
    """Per graph: E[v], z, valid, mr, full-arm, α, policy CE, KL(p‖t), KL(t‖p), KL(t‖soft t), unsupported mass — the trainer's target, detached."""
    logits, _v, bins = model.forward_batch(  # pyright: ignore[reportCallIssue]
        inputs.x, inputs.edge_index, inputs.edge_attr, inputs.legal_index, inputs.stone_mask, node_offsets=inputs.node_offsets)
    logits = logits.to(torch.float32)
    lo = inputs.legal_offsets
    b = int(lo.shape[0]) - 1
    seg = segment_ids(lo, total=int(logits.shape[0]))
    p = segment_softmax(logits, lo)
    t = rebuild_sparse_target(inputs.policy_target, p, inputs.explicit_mask, inputs.tail_mass, lo)
    logp, logt = torch.log(p.clamp_min(1e-12)), torch.log(t.clamp_min(1e-12))
    # The WHOLE-SET soft form PROBE-1 measured (the artefact the explicit-only trainer form replaced) — kept as read.
    soft = torch.exp(logt / SOFT_POLICY_TEMPERATURE) * (t > 0).to(t.dtype)
    soft = soft / segment_sum(soft, seg, b).clamp_min(1e-12)[seg]
    unsupported = segment_sum(p * ((t <= 0) & (p > 1e-6)).to(p.dtype), seg, b)
    ev = decode_binned_value(bins).reshape(-1)
    z = inputs.outcomes.reshape(-1).to(torch.float32)
    logbins = torch.log_softmax(bins.to(torch.float32), dim=-1)
    value_ce = -(scalar_to_two_hot(z) * logbins).sum(dim=-1)
    mr = torch.round(inputs.x[inputs.node_offsets[:-1].to(torch.long), 3] * 2.0)
    return {
        "ev": ev.numpy(), "z": z.numpy(), "valid": inputs.value_valid.reshape(-1).numpy().astype(bool),
        "mr": mr.numpy().astype(np.int64), "full": (inputs.policy_row_weight.reshape(-1) > 0).numpy(),
        "alpha": inputs.tail_mass.reshape(-1).numpy(), "value_ce": value_ce.numpy(),
        "policy_ce": segment_sum(-(t * logp), seg, b).numpy(),
        "kl_prior_target": segment_sum(p * (logp - logt), seg, b).numpy(),
        "kl_target_prior": segment_sum(t * (logt - logp), seg, b).numpy(),
        "kl_target_soft": segment_sum(t * (logt - torch.log(soft.clamp_min(1e-12))), seg, b).numpy(),
        "unsupported_mass": unsupported.numpy(),
    }


@torch.no_grad()
def batch_losses(model: torch.nn.Module, inputs: Any, *, policy_denominator: float,
                 value_denominator: float) -> tuple[float, float]:
    """This part's (policy CE, value CE) exactly as `Trainer.eval_step_from_graph_batch` sums them."""
    logits, _v, bins = model.forward_batch(  # pyright: ignore[reportCallIssue]
        inputs.x, inputs.edge_index, inputs.edge_attr, inputs.legal_index, inputs.stone_mask, node_offsets=inputs.node_offsets)
    pol = ragged_policy_ce(logits, inputs.policy_target, inputs.legal_offsets, full_search_mask=inputs.policy_row_weight,
                           explicit_mask=inputs.explicit_mask, tail_mass=inputs.tail_mass, denominator=policy_denominator)
    val = binned_value_loss(bins, inputs.outcomes, value_mask=inputs.value_valid, denominator=value_denominator)
    return float(pol.item()), float(val.item())


def read_ring(nets: list[Net], buffer: HexgBuffer, *, batches: int, batch_size: int, threads: int,
              rows: bool, log: Callable[[str], None] = lambda _s: None) -> dict[str, Any]:
    """`batches` production samples, EVERY net read on the SAME parts: per-net mean batch losses (paired) and, when `rows`, the per-row arrays."""
    cfg, spec = nets[0].config, nets[0].spec
    losses: dict[str, list[tuple[float, float]]] = {n.net_hash: [] for n in nets}
    cols: dict[str, list[np.ndarray]] = {}
    t0 = time.perf_counter()
    for k in range(batches):
        parts = sample_parts(buffer, cfg, spec, batch_size=batch_size, threads=threads)
        pd, vd = parts["policy_denominator"], parts["value_denominator"]
        sums = {n.net_hash: [0.0, 0.0] for n in nets}
        for make in parts["parts"]:
            inputs = make()
            for net in nets:
                pol, val = batch_losses(net.model, inputs, policy_denominator=pd, value_denominator=vd)
                sums[net.net_hash][0] += pol
                sums[net.net_hash][1] += val
                if rows:
                    for key, arr in read_rows(net.model, inputs).items():
                        cols.setdefault(f"{net.net_hash}:{key}", []).append(arr)
        for net in nets:
            losses[net.net_hash].append((sums[net.net_hash][0], sums[net.net_hash][1]))
        log(f"batch {k + 1}/{batches} {time.perf_counter() - t0:.0f} s")
    out: dict[str, Any] = {"batches": batches, "batch_size": batch_size, "losses": {}}
    for net in nets:
        arr = np.asarray(losses[net.net_hash])
        out["losses"][net.net_hash] = {"step": net.step, "policy": float(arr[:, 0].mean()), "value": float(arr[:, 1].mean()),
                                       "policy_sd": float(arr[:, 0].std(ddof=1)) if batches > 1 else None,
                                       "value_sd": float(arr[:, 1].std(ddof=1)) if batches > 1 else None}
    if rows:
        out["rows"] = {key: np.concatenate(chunks) for key, chunks in cols.items()}
    return out


__all__ = ["Net", "batch_losses", "load_net", "open_ring", "read_ring", "read_rows", "sample_parts"]
