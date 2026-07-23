"""Loop subsystem boot — inference/eval model builds + run-safety subsystems (WP10 §a.4/§c.6).

The old `training/lifecycle.py` was a NAME-COLLISION trap: it is subsystem-BOOT (model builds +
monitor/probe/dashboard construction), NOT the run-safety `lifecycle` repo_design §11 defines.
It lands HERE as `subsystems.py`, freeing the `lifecycle` name for the §11 subsystem.

Two ratified changes from the port:
  * `InfModelArch` carries the DECLARED `representation` (off the WP9 arch dataclass), and
    `build_inference_model` / `build_eval_model` construct via `build_net(arch)` — NOT a
    `model_representation(module)` sniff (WP9-deleted, §c.4/§c.6). No shape-inference.
  * The DISPLAY-boot half (WebDashboard, `register_renderer`/`register_jsonl_sink`, TB
    `MetricsWriter`, `EarlyGameProbe`, `ValueProbe`) DEFERS→WP13 — `build_subsystems` returns
    only the run-safety subsystems (disk guard + the optional GPU monitor + the injected
    `EventSink`); the trainer's `EventSink` is the seam WP13 wires the real sink onto.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from mantis.model import ModelArch, amp_dtype_for, build_net
from mantis.train.lifecycle.disk_guard import DiskGuard

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class InfModelArch:
    """The declared inference/eval arch — the SOLE construction source at sync/eval (§c.6).

    `arch` is the WP9 `CnnArch|GnnArch` dataclass; `representation` is the DECLARED
    discriminant read off it (never sniffed off a live module). `spec` carries the resolved
    registry spec for a graph warmup. `build_eval_model` reconstructs the IDENTICAL net from
    just this object."""

    arch: ModelArch
    representation: str
    spec: Any = None


def build_inference_model(trainer: Any, device: torch.device) -> tuple[torch.nn.Module, InfModelArch]:
    """Build the inference-model instance the self-play server owns, from the trainer's DECLARED
    arch (`trainer.arch`, a WP9 dataclass) via `build_net`. Loads `trainer.inference_state_dict()`
    (EMA weights when EMA is on; identical to the raw model at step 0). No shape-inference."""
    arch = trainer.arch
    inf_model = build_net(arch).to(device)
    inf_model.load_state_dict(trainer.inference_state_dict())
    inf_model.eval()
    representation = getattr(arch, "representation", "grid")
    spec = None
    try:
        from mantis.encoding import resolve_from_config
        spec = resolve_from_config(dict(trainer.config))
    except Exception:  # noqa: BLE001 — spec is only consulted for a CUDA graph warmup
        spec = None
    return inf_model, InfModelArch(arch=arch, representation=representation, spec=spec)


def build_eval_model(arch: InfModelArch, device: torch.device) -> torch.nn.Module:
    """Reconstruct the IDENTICAL net (grid or graph) from just `arch` via `build_net` (§c.6)."""
    eval_model = build_net(arch.arch).to(device)
    eval_model.eval()
    return eval_model


def cuda_warmup(inf_model: torch.nn.Module, device: torch.device, arch: InfModelArch) -> None:
    """Warm CUDA kernels with a dummy forward so the first worker inference returns immediately.

    CUDA-only (no-op on CPU). Representation-aware: the grid path feeds a dummy (1, C, B, B)
    tensor; the graph path runs `forward_batch` on a minimal synthetic 1-graph batch under bf16
    autocast (LAW-06 pin). Warmup dtype matches the production loop so the RIGHT kernels compile.
    """
    if device.type != "cuda":
        return
    representation = arch.representation
    _t = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda",
                                         dtype=amp_dtype_for(representation, None)):
        if representation == "graph":
            x, ei, ea, legal_mask, stone_mask, node_offsets = _synthetic_graph_warmup(arch, device)
            inf_model.forward_batch(x, ei, ea, legal_mask, stone_mask, node_offsets=node_offsets)
        else:
            base = getattr(inf_model, "_orig_mod", inf_model)
            ch = int(getattr(base, "in_channels", 8))
            board = int(getattr(base, "board_size", 19))
            inf_model(torch.zeros(1, ch, board, board, device=device))
    torch.cuda.synchronize()
    _LOG.info("cuda_warmup_done elapsed_sec=%.1f", time.time() - _t)


def _synthetic_graph_warmup(arch: InfModelArch, device: torch.device):
    """A minimal 2-node / 2-edge synthetic graph batch for the graph CUDA warmup. Self-contained
    (no self-play collate dependency — that lives in WP6's selfplay surface)."""
    in_dim = int(getattr(arch.arch, "in_dim", 11))
    edge_dim = int(getattr(arch.arch, "edge_dim", 5))
    x = torch.zeros(2, in_dim, device=device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    edge_attr = torch.zeros(2, edge_dim, device=device)
    legal_mask = torch.tensor([False, True], dtype=torch.bool, device=device)
    stone_mask = torch.tensor([True, False], dtype=torch.bool, device=device)
    node_offsets = torch.tensor([0, 2], dtype=torch.long, device=device)
    return x, edge_index, edge_attr, legal_mask, stone_mask, node_offsets


@dataclass
class LoopSubsystems:
    """The run-safety subsystem bundle (WP10). Display subsystems (probes / TB / dashboards)
    DEFER→WP13; the `sink` is the seam WP13 wires the real event sink onto."""

    disk_guard: DiskGuard
    sink: Any
    gpu_monitor: Any = None
    composition_interval: int = 0
    instrumentation_enabled: bool = False
    axis_baseline: dict = field(default_factory=dict)

    def teardown(self) -> None:
        stop = getattr(self.gpu_monitor, "stop", None)
        if stop is not None:
            stop()
        self.disk_guard.stop()


def build_subsystems(
    *,
    checkpoint_dir: str | Path,
    config: dict[str, Any],
    sink: Any,
    gpu_monitor: Any = None,
) -> LoopSubsystems:
    """Build + start the run-safety subsystems: the disk guard (emits through the injected
    `EventSink`, SIGTERMs below fail_gb) and (optionally) an injected GPU monitor. The display
    subsystems DEFER→WP13."""
    dg_cfg = config.get("disk_guard", {}) if isinstance(config.get("disk_guard"), dict) else {}
    disk_guard = DiskGuard(
        watch_path=checkpoint_dir,
        interval_sec=float(dg_cfg.get("interval_sec", 60.0)),
        warn_gb=float(dg_cfg.get("warn_gb", 10.0)),
        fail_gb=float(dg_cfg.get("fail_gb", 5.0)),
        keep_all=bool(dg_cfg.get("keep_all", False)),
        sink=sink,
    )
    if gpu_monitor is not None and hasattr(gpu_monitor, "start"):
        gpu_monitor.start()
    return LoopSubsystems(disk_guard=disk_guard, sink=sink, gpu_monitor=gpu_monitor)
