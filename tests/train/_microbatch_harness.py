"""Shared rig for the micro-batch oracles: one buffer, one trainer, one wire-replay double.

Real: the buffer, the wire, `collate_graph_batch`, the partition, the losses, the optimizer,
the scheduler, the sink protocol and the filesystem. Fake: the ARCH and, where a suite says so,
the SINK. Nothing here fakes the caps, the split, or the normalisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from _determinism import deterministic_algorithms  # re-export: H.deterministic_algorithms stays the microbatch suites' entry
from mantis._engine import HexgBuffer
from mantis.config.resolve.microbatch import MicrobatchCapsSpec
from mantis.encoding import lookup
from mantis.model import GnnArch, GnnArchV2SoftPolicy, build_net, gnn_widths_block
from mantis.train.coordinator.dispatch import run_declared_train_step
from mantis.train.trainer.core import Trainer, TrainHParams

GRAPH_ENCODING = "gnn_axis_v1"
GSPEC = lookup(GRAPH_ENCODING)
SEED = 20260803


def uniform_graph_buffer(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """Build a real `HexgBuffer` whose records are IDENTICAL in shape, so a constant per-graph
    edge count makes `max_edges = c * (B // M)` yield exactly `M` micro-batches."""
    hb = HexgBuffer(capacity, GRAPH_ENCODING, 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]
        policy = [(2, 0, 0.6), (1, 1, 0.4)]
        outcome = 1.0 if i % 2 == 0 else -1.0
        hb.push_graph_position(stones, policy, 1, 30, 2 + i, True, outcome, True, 10 + i)
    return hb


def ragged_graph_buffer(n_records: int = 8, capacity: int = 64) -> HexgBuffer:
    """Build a real `HexgBuffer` whose records differ in stone count, so (N, E) varies."""
    hb = HexgBuffer(capacity, GRAPH_ENCODING, 128)
    for i in range(n_records):
        stones = [(0, 0, 1), (1, 0, -1), (0, 1, 1), (2, 1, -1), (1, 2, 1)][: 2 + (i % 4)]
        policy = [(3, 0, 0.6), (1, 1, 0.4)]
        outcome = 1.0 if i % 2 == 0 else -1.0
        hb.push_graph_position(stones, policy, 1, 30, 2 + i, True, outcome, True, 10 + i)
    return hb


class ReplayWireBuffer:
    """A buffer double that samples the REAL buffer ONCE and replays that pair thereafter.

    The parity legs need both arms to see the same graphs, and `sample_graph_batch` draws
    through the Rust RNG, which `torch.manual_seed` does not reach. The pair holds a PAYLOAD,
    not the pyclass: `GraphWire.take()` MOVES its buffers, so a wire reads exactly once.
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
        # `n_threads` is the rebuild width: accepted because the dispatcher passes it on
        # every graph step, ignored because the double has no rebuild of its own.
        self.calls += 1
        return self._pair

    @property
    def wire(self):
        return self._pair[0]

    @property
    def targets(self):
        return self._pair[1]


def per_graph_counts(wire: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return `(edge_counts, node_counts)` per graph, from the wire's CSR offsets."""
    eo = np.asarray(wire.edge_offsets)
    no = np.asarray(wire.node_offsets)
    return np.diff(eo), np.diff(no)


def caps_for_exactly(wire: Any, m: int) -> tuple[int, int]:
    """Return the caps that split THIS wire into exactly `m` micro-batches — the node member
    is set past the whole batch, so the split is edge-driven and `m` is an identity."""
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


def graph_step(trainer: Any, buffer: Any) -> dict[str, float]:
    """One real graph training step through the PRODUCTION dispatch: `dispatch._graph_step`
    is what the coordinator calls, so a guard driven here sits on the path that runs."""
    from mantis.train.coordinator.dispatch import _graph_step as production_graph_step

    wire, _targets = buffer.sample_graph_batch(4, augment=False, recent_frac=0.0)
    max_edges, max_nodes = non_binding_caps(wire)
    return production_graph_step(
        trainer, buffer, GSPEC,
        batch_size=4, augment=False, recency_weight=0.0,
        caps_provider=lambda: MicrobatchCapsSpec(max_edges=max_edges, max_nodes=max_nodes),
        sample_threads_provider=lambda: 1,
        fast_policy_weight_provider=lambda: 0.0,
    )


def step_once(trainer: Any, buf: HexgBuffer, *, replay_n: int = 8) -> dict[str, float]:
    """One declared train step over a fixed `replay_n`-row wire, with caps that cannot bind."""
    replay = ReplayWireBuffer(buf, replay_n)
    return run_declared_train_step(
        trainer, replay, GSPEC, batch_size=replay_n, augment=False, recency_weight=0.0,
        caps_provider=lambda: MicrobatchCapsSpec(*non_binding_caps(replay.wire)),
        sample_threads_provider=lambda: 1, fast_policy_weight_provider=lambda: 0.0)


def tiny_graph_arch() -> GnnArch:
    return GnnArch(in_dim=GSPEC.node_feat_dim, edge_dim=GSPEC.edge_feat_dim, hidden=16,
                   num_layers=1, policy_hidden=16, value_hidden=16)


def graph_hparams(**over: Any) -> TrainHParams:
    base: dict[str, Any] = dict(
        lr=1e-3, weight_decay=1e-4, grad_clip=1.0, lr_schedule="cosine",
        total_steps=1_000_000, scheduler_t_max=None, eta_min=5e-4,
        checkpoint_interval=0, policy_loss_warmup_steps=0, aux_soft_policy=None,
    )
    base.update(over)
    return TrainHParams(**base)


def minted_config(name: str) -> dict[str, Any]:
    """Load a REAL minted `configs/*.yaml` through the real loader and dump it to a dict.

    `save_checkpoint` schema-validates its config on write, so any leg that lets the
    periodic-checkpoint seam fire needs a complete `RunConfig`.
    """
    from mantis.config.loader import load_config
    repo = Path(__file__).resolve().parents[2]
    return load_config(repo / "configs" / name).model_dump()


def graph_config() -> dict[str, Any]:
    """`configs/dev_example.yaml` — graph, `gnn_axis_v1`, complete and schema-valid — with `model.gnn` at the tiny arch's own widths (v35: the writer refuses a stamp claiming another shape)."""
    config = minted_config("dev_example.yaml")
    arch = tiny_graph_arch()
    config["model"]["gnn"] = gnn_widths_block(arch)
    return config


def tiny_graph_trainer(tmp_path: Path, *, sink: Any = None, seed: int = SEED,
                       **hp_over: Any) -> Trainer:
    torch.manual_seed(seed)
    arch = tiny_graph_arch()
    return Trainer(build_net(arch), graph_config(), arch=arch,
                   checkpoint_dir=Path(tmp_path) / "ckpt", device=torch.device("cpu"),
                   train_hparams=graph_hparams(**hp_over), sink=sink)


def tiny_soft_policy_arch() -> GnnArchV2SoftPolicy:
    a = tiny_graph_arch()
    return GnnArchV2SoftPolicy(in_dim=a.in_dim, edge_dim=a.edge_dim, hidden=a.hidden, num_layers=a.num_layers,
                               policy_hidden=a.policy_hidden, value_hidden=a.value_hidden)


def soft_policy_graph_trainer(tmp_path: Path, *, sink: Any = None, seed: int = SEED,
                              target_temperature: float = 4.0, weight: float = 4.0,
                              **hp_over: Any) -> Trainer:
    """A tiny `GnnNetV2SoftPolicy` trainer with `model.aux_soft_policy` ARMED."""
    torch.manual_seed(seed)
    arch = tiny_soft_policy_arch()
    config = graph_config()
    config["identity"]["arch_kind"] = "GnnArchV2SoftPolicy"
    config["model"]["aux_soft_policy"] = {"target_temperature": target_temperature, "weight": weight}
    hp_over.setdefault("checkpoint_interval", 0)
    hp_over.setdefault("aux_soft_policy", (target_temperature, weight))
    return Trainer(build_net(arch), config, arch=arch, checkpoint_dir=Path(tmp_path) / "ckpt_soft",
                   device=torch.device("cpu"), train_hparams=graph_hparams(**hp_over), sink=sink)


def ema_graph_trainer(tmp_path: Path, *, sink: Any = None, seed: int = SEED,
                      update_every: int = 1, **hp_over: Any) -> Trainer:
    """Build a tiny graph `Trainer` with EMA ACTUALLY ENABLED — the shipped config mints it
    false, so the EMA branch never executes on a default fixture."""
    torch.manual_seed(seed)
    arch = tiny_graph_arch()
    config = graph_config()
    config["train"]["ema"] = {"enabled": True, "decay": 0.9, "update_every": update_every}
    hp_over.setdefault("checkpoint_interval", 0)
    return Trainer(build_net(arch), config, arch=arch,
                   checkpoint_dir=Path(tmp_path) / "ckpt_ema", device=torch.device("cpu"),
                   train_hparams=graph_hparams(**hp_over), sink=sink)


class SpySink:
    """Record every emitted event mapping, as the structural `EventSink` protocol."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == name]


class OptimizerSpy:
    """Count `zero_grad` / `step` by wrapping the REAL optimizer's bound methods in place —
    a spy on the object the trainer drives, not a stub that replaces it."""

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
    """Return the flattened parameter-gradient vector, zeros where there is no grad."""
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in model.parameters()
    ])


def param_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])
