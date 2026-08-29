# Exceeds the 300-line soft cap (R8): one harness for the micro-batch legs, and its
# pieces are load-bearing on each other — the fixed-pair buffer exists so two arms see
# ONE sample, and the caps helpers derive their bounds from that same wire. Split up,
# an arm could silently be compared against a different draw.
"""Shared rig for the WP12-R F2 micro-batch oracles (OF2-*, DESIGN_DFIX §5.2).

Not a test module (leading underscore, no `test_` prefix): pytest does not collect it, and
the three F2 suites import it the way `tests/model/test_gine_bf16_drift.py:120` imports
`_bf16_parity`. It exists so the graph harness — a real `HexgBuffer`, a tiny real `GnnNet`
`Trainer`, and the wire-replay buffer the two-arm parity legs need — has ONE definition:
three copies of a harness that drift while all three stay green is the fork-and-drift failure
`tests/train/test_periodic_checkpoint.py:5-8` argues against, and R5 bars importing it from a
sibling TEST module, which is why it lives here rather than in one of the suites.

WHAT IS REAL AND WHAT IS NOT (disclosed once, for all three suites): real everywhere are the
buffer, the wire, `collate_graph_batch`, the partition, the losses, the optimizer, the
scheduler, the event sink protocol and the filesystem. Fake: the ARCH (a tiny `GnnNet`,
hidden=16/num_layers=1) and, where a suite says so, the SINK (a recording spy). Nothing here
fakes the caps, the split, or the normalisation.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mantis._engine import HexgBuffer
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.train.trainer.core import Trainer, TrainHParams

GRAPH_ENCODING = "gnn_axis_v1"
GSPEC = lookup(GRAPH_ENCODING)
SEED = 20260803


@contextlib.contextmanager
def deterministic_algorithms():
    """**TEST SCOPE ONLY.** `torch.use_deterministic_algorithms(True)` for the block, then
    the ambient setting restored exactly (including `CUBLAS_WORKSPACE_CONFIG` back to ABSENT).

    Copied — not imported — from `tests/model/_bf16_parity.py:95-127`, the established
    pattern; R5 bars a cross-directory test import and duplicating fifteen stdlib lines is
    cheaper than inventing a second mechanism.

    **PRODUCTION KEEPS ITS KERNELS (R191).** Nothing in `src/mantis/` calls this. Every leg
    that uses it carries `deterministic_mode` in its own NAME, so no reader can conclude a
    production run is deterministic. R191 is why F2's exact legs run here at all: the median
    statistic F1 used reads exactly 0.0 against a defect confined to <=50% of graphs, and a
    micro-batch split produces precisely that minority-subset shape — so F2 asserts EXACT
    equality and per-graph identities and inherits no statistic from F1.
    """
    was_enabled = torch.are_deterministic_algorithms_enabled()
    had_cublas = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not had_cublas:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled)
        if had_cublas:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


# ── buffers ──────────────────────────────────────────────────────────────────────────────
def uniform_graph_buffer(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real `HexgBuffer` whose records are IDENTICAL in shape, so every sampled graph has
    the same edge and node count.

    That uniformity is load-bearing for the cadence legs: with a constant per-graph edge
    count `c` and `B` graphs, `max_edges = c * (B // M)` yields EXACTLY `M` micro-batches for
    any `M` dividing `B`. A ragged fixture would make the requested M a search rather than an
    identity, and OF2-4/OF2-5 sweep M as an exact quantity."""
    hb = HexgBuffer(capacity, GRAPH_ENCODING, 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
        policy = [(2, 0, 0.6), (1, 1, 0.4)]
        outcome = 1.0 if i % 2 == 0 else -1.0
        hb.push_graph_position(stones, policy, 1, 30, 2 + i, True, outcome, True, 10 + i)
    return hb


def ragged_graph_buffer(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """A real `HexgBuffer` whose records differ in stone count, so per-graph (N, E) varies —
    the fixture the slice-fidelity and over-cap legs want."""
    hb = HexgBuffer(capacity, GRAPH_ENCODING, 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, 1, -1), (1, 2, 1)][: 2 + (i % 4)]
        policy = [(3, 0, 0.6), (1, 1, 0.4)]
        outcome = 1.0 if i % 2 == 0 else -1.0
        hb.push_graph_position(stones, policy, 1, 30, 2 + i, True, outcome, True, 10 + i)
    return hb


class ReplayWireBuffer:
    """A buffer double that samples the REAL buffer ONCE and then returns that same
    `(payload, targets)` pair on every later call.

    The two-arm parity legs (OF2-3b/c/d) compare an M=1 step against an M=k step and the
    comparison is only meaningful if BOTH arms see the same graphs; `sample_graph_batch`
    draws randomly through the Rust RNG, which `torch.manual_seed` does not reach. Nothing
    else is faked — the arrays, the targets and every downstream call are the real ones.

    THE PAIR HOLDS A PAYLOAD, NOT THE PYCLASS, and that is required rather than tidy since
    PERF-TRANCHE-1 A2: `GraphWire.take()` now MOVES its buffers into numpy, so a wire can be
    read exactly once and handing the same pyclass to two arms raises `WireAlreadyConsumed`
    on the second. Reading it into a `GraphWirePayload` here — through the same
    `graph_wire_from_rust` the dispatcher uses — makes the pair repeatable, and the
    dispatcher then reads the payload through the duck-typed getter path. Both arms still
    see one sample of the real buffer, which is the whole point of the double.
    """

    def __init__(self, real: HexgBuffer, batch_size: int, augment: bool = False) -> None:
        from mantis.selfplay.graph_collate import graph_wire_from_rust

        self._real = real
        self.size = real.size
        self.capacity = real.capacity
        self.calls = 0
        wire, targets = real.sample_graph_batch(batch_size, augment=augment, recent_frac=0.0)
        self._pair = (graph_wire_from_rust(wire), targets)

    def sample_graph_batch(self, batch_size: int, augment: bool = False,
                           recent_frac: float = 0.0, n_threads: int = 1):
        # `n_threads` is B1's rebuild width. The double accepts it because the dispatcher
        # now passes it on every graph step; it has no rebuild of its own to widen.
        self.calls += 1
        return self._pair

    @property
    def wire(self):
        return self._pair[0]

    @property
    def targets(self):
        return self._pair[1]


# ── per-graph counts, read off the wire ──────────────────────────────────────────────────
def per_graph_counts(wire: Any) -> tuple[np.ndarray, np.ndarray]:
    """`(edge_counts, node_counts)` per graph, from the wire's CSR offsets."""
    eo = np.asarray(wire.edge_offsets)
    no = np.asarray(wire.node_offsets)
    return np.diff(eo), np.diff(no)


def caps_for_exactly(wire: Any, m: int) -> tuple[int, int]:
    """`(max_edges, max_nodes)` that split THIS wire into exactly `m` micro-batches.

    Derived, never guessed: with a uniform fixture every `ec[i]` is equal, so the edge member
    is `c * (B // m)` and the node member is set past the whole batch so the split is
    edge-driven and `m` is an identity rather than a coincidence."""
    ec, nc = per_graph_counts(wire)
    b = len(ec)
    if b % m != 0 or len(set(ec.tolist())) != 1:
        raise AssertionError(
            f"caps_for_exactly needs a uniform fixture and m | B; got B={b}, m={m}, "
            f"distinct edge counts {sorted(set(ec.tolist()))}"
        )
    return int(ec[0]) * (b // m), int(nc.sum()) + 1


def non_binding_caps(wire: Any) -> tuple[int, int]:
    """Caps that cannot bind on this wire — both members set past the whole batch."""
    ec, nc = per_graph_counts(wire)
    return int(ec.sum()) + 1, int(nc.sum()) + 1


# ── trainer ──────────────────────────────────────────────────────────────────────────────
def tiny_graph_arch() -> GnnArch:
    return GnnArch(in_dim=GSPEC.node_feat_dim, edge_dim=GSPEC.edge_feat_dim, hidden=16,
                   num_layers=1, policy_hidden=16, value_hidden=16)


def graph_hparams(**over: Any) -> TrainHParams:
    base: dict[str, Any] = dict(
        lr=1e-3, weight_decay=1e-4, grad_clip=1.0, fp16=False, lr_schedule="cosine",
        total_steps=1_000_000, scheduler_t_max=None, eta_min=5e-4, min_lr=None,
        checkpoint_interval=0, completed_q_values=False, policy_prune_frac=0.0,
        entropy_reg_weight=0.0, aux_opp_reply_weight=0.0, uncertainty_weight=0.0,
        ownership_weight=0.0, threat_weight=0.0, aux_chain_weight=0.0, ply_index_weight=0.0,
        threat_pos_weight=1.0, value_target="pure_outcome_z",
        policy_target="raw_visit_distribution", draw_reward=-0.5, ply_cap_value=-0.5,
    )
    base.update(over)
    return TrainHParams(**base)


def minted_config(name: str) -> dict[str, Any]:
    """A REAL minted `configs/*.yaml`, loaded through the real loader and dumped to a dict.

    Not a hand-built stub: `Trainer.save_checkpoint` schema-validates its config on write
    (`train/checkpoints.py`, repo_design §6), so any leg that lets the periodic-checkpoint
    seam fire needs a complete `RunConfig`. Using the shipped config also means the CAPS the
    trainer's config carries are the minted ones — which is a useful accident: the caps the
    step actually uses arrive through `caps_provider`, never through the trainer's config, so
    a leg that binds the caps and still sees the minted values in the checkpoint is evidence
    the two paths are separate."""
    from mantis.config.loader import load_config
    repo = Path(__file__).resolve().parents[2]
    return load_config(repo / "configs" / name).model_dump()


def graph_config() -> dict[str, Any]:
    """`configs/dev_example.yaml` — graph, `gnn_axis_v1`, complete and schema-valid."""
    return minted_config("dev_example.yaml")


def tiny_graph_trainer(tmp_path: Path, *, sink: Any = None, seed: int = SEED,
                       **hp_over: Any) -> Trainer:
    torch.manual_seed(seed)
    arch = tiny_graph_arch()
    return Trainer(build_net(arch), graph_config(), arch=arch,
                   checkpoint_dir=Path(tmp_path) / "ckpt", device=torch.device("cpu"),
                   train_hparams=graph_hparams(**hp_over), sink=sink)


def ema_graph_trainer(tmp_path: Path, *, sink: Any = None, seed: int = SEED,
                      update_every: int = 1, **hp_over: Any) -> Trainer:
    """A tiny graph `Trainer` with EMA ACTUALLY ENABLED.

    `tiny_graph_trainer` above builds from `configs/dev_example.yaml`, which declares no `ema`
    block, so `resolve_ema_config` returns `enabled=False` and `ema_model is None` — measured.
    That means the EMA branch of the graph step's tail never executes in a default-fixture row,
    and an EMA update moved into the accumulation loop would be invisible to it. This builder
    exists so OF2-4 can assert the EMA count the row's registered PASS column names.

    `ema` is a runtime key read by `mantis.train.ema.resolve_ema_config` off the trainer's
    config dict, not a `RunConfig` field — so a trainer built this way must not write a
    checkpoint (the writer schema-validates its config). Every leg using it sets
    `checkpoint_interval=0`."""
    torch.manual_seed(seed)
    arch = tiny_graph_arch()
    config = graph_config()
    config["ema"] = {"enabled": True, "decay": 0.9, "update_every": update_every}
    hp_over.setdefault("checkpoint_interval", 0)
    return Trainer(build_net(arch), config, arch=arch,
                   checkpoint_dir=Path(tmp_path) / "ckpt_ema", device=torch.device("cpu"),
                   train_hparams=graph_hparams(**hp_over), sink=sink)


class SpySink:
    """Records every emitted event mapping (the structural `EventSink` protocol)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


class OptimizerSpy:
    """Counts `zero_grad` / `step` on the REAL optimizer object by wrapping its bound
    methods in place — a spy on the object the trainer actually drives, not a stub that
    replaces it (MB-7's kill surface: `optimizer.step()` moved inside the loop)."""

    def __init__(self, optimizer: Any) -> None:
        self.zero_grads = 0
        self.steps = 0
        self._optimizer = optimizer
        self._real_zero = optimizer.zero_grad
        self._real_step = optimizer.step
        optimizer.zero_grad = self._zero_grad
        optimizer.step = self._step

    def _zero_grad(self, *a: Any, **kw: Any) -> Any:
        self.zero_grads += 1
        return self._real_zero(*a, **kw)

    def _step(self, *a: Any, **kw: Any) -> Any:
        self.steps += 1
        return self._real_step(*a, **kw)


class SchedulerSpy:
    def __init__(self, scheduler: Any) -> None:
        self.steps = 0
        self._real = scheduler.step
        scheduler.step = self._step

    def _step(self, *a: Any, **kw: Any) -> Any:
        self.steps += 1
        return self._real(*a, **kw)


def grad_vector(model: torch.nn.Module) -> torch.Tensor:
    """The flattened parameter-gradient vector (zeros for a parameter with no grad)."""
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in model.parameters()
    ])


def param_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])
