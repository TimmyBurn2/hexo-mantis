"""Loop subsystem boot — inference/eval model builds + run-safety subsystems (WP10 §a.4/§c.6).

The old `training/lifecycle.py` was a NAME-COLLISION trap: it is subsystem-BOOT (model builds +
monitor/probe/dashboard construction), NOT the run-safety `lifecycle` repo_design §11 defines.
It lands HERE as `subsystems.py`, freeing the `lifecycle` name for the §11 subsystem.

Two ratified changes from the port:
  * `InfModelArch` carries the DECLARED `representation` (off the WP9 arch dataclass), and
    `build_inference_model` / `build_eval_model` construct via `build_net(arch)` — NOT a
    `model_representation(module)` sniff (WP9-deleted, §c.4/§c.6). No shape-inference.
  * The DISPLAY-boot half (WebDashboard, `register_renderer`/`register_jsonl_sink`, TB
    `MetricsWriter`, `EarlyGameProbe`, `ValueProbe`) is DEFER/ARCH — `build_subsystems`
    returns only the run-safety subsystems (disk guard + the optional GPU monitor + the
    injected `EventSink`).

WP13-A adds `build_run_safety` — the composition root for the run-safety triple (the REAL
JSONL sink, the heartbeat registry, the INDEPENDENT watchdog thread). This is one of the
THREE declared `train → mantis.monitor` import sites (census-pinned, O-19); everywhere else
the sink stays INJECTED, never imported.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import torch

import mantis.train.checkpoints as _checkpoints
from mantis.model import ModelArch, amp_dtype_for, build_net
from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import HEARTBEAT_SOURCES, HeartbeatRegistry
from mantis.monitor.sink import JsonlEventSink
from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.train.lifecycle.heartbeat_watchdog import ActorLagSpec, HeartbeatWatchdog
from mantis.train.lifecycle.watchdog import watchdog_snapshot_path

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class InfModelArch:
    """The declared inference/eval arch — the SOLE construction source at sync/eval (§c.6).

    `arch` is the WP9 `CnnArch|GnnArch` dataclass; `representation` is the DECLARED
    discriminant read off it (never sniffed off a live module). `amp_dtype` is the DECLARED
    `train.amp_dtype` string (R30b) — carried, no default (R1), so `cuda_warmup` matches the
    production loop's autocast dtype instead of guessing. `spec` carries the resolved
    registry spec for a graph warmup. `build_eval_model` reconstructs the IDENTICAL net from
    just this object."""

    arch: ModelArch
    representation: str
    amp_dtype: str
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
    amp_dtype = trainer.config["train"]["amp_dtype"]
    spec = None
    try:
        from mantis.encoding import resolve_from_config
        spec = resolve_from_config(dict(trainer.config))
    except Exception:  # noqa: BLE001 — spec is only consulted for a CUDA graph warmup
        spec = None
    return inf_model, InfModelArch(
        arch=arch, representation=representation, amp_dtype=amp_dtype, spec=spec
    )


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
                                         dtype=amp_dtype_for(representation, arch.amp_dtype)):
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


# ── WP13-A run-safety composition root ───────────────────────────────────────────────
class RunSafety(NamedTuple):
    """The run-safety triple. Tuple-shaped (§a.2 `-> (sink, registry, watchdog)`) with
    names, so a caller can unpack it or read `run_safety.heartbeat`."""

    sink: JsonlEventSink
    registry: HeartbeatRegistry
    watchdog: HeartbeatWatchdog

    @property
    def heartbeat(self) -> Callable[[str], None]:
        """THE `HeartbeatFn` — pass it to `WorkerPool(heartbeat=…)`,
        `InferenceServer(heartbeat=…)` and `StepCoordinator(heartbeat=…)`."""
        return self.registry.beat


def build_run_safety(
    *,
    log_dir: str | Path,
    run_id: str,
    buffer: Any,
    buffer_persist_path: str | Path,
    wired_sources: Sequence[str],
    actor_ckpt_step_fn: Callable[[], int],
    learner_step_fn: Callable[[], int],
    monitor_cfg: MonitorConfig,
    heartbeat_file: str | Path | None = None,
    exit_fn: Callable[[int], None] = os._exit,
) -> RunSafety:
    """Build the REAL event sink + heartbeat registry + independent watchdog (unstarted).

    Wiring contract:
      * the sink replaces `NullEventSink` everywhere the run injects one (trainer,
        coordinator, disk guard, pool) — ONE JSONL segment per process start (§11 log
        identity: a file never spans two run segments);
      * `run_safety.heartbeat` is handed to the pool / inference server / coordinator so
        all three pipeline stages beat into ONE registry;
      * `counters_fn` reads the persist counters as LIVE module/instance ATTRIBUTES —
        `_checkpoints.persist_errors_total`, never `from … import persist_errors_total`,
        which binds the int at import and reads a frozen 0 forever after `global … += 1`
        (O-28). BOTH sources are fatal via the watchdog (LAW-14);
      * the fire-time snapshot targets `watchdog_snapshot_path(canonical)` — the DISTINCT
        `.watchdog` path, so an abnormal-exit save can never truncate the resume buffer;
      * `wired_sources` is REQUIRED and has NO default: the caller must DECLARE which
        pipeline stages it actually handed `run_safety.heartbeat` to. A declared stage is
        watched from arm time; an undeclared one gets a loud `heartbeat_source_unwired`
        event instead of a stall abort. Without the declaration, one forgotten `heartbeat=`
        kwarg makes a healthy run fire 42 and the supervisor relaunch into the same missing
        wiring until the budget is gone (RED-TEAM F3) — so this may never be inferred;
      * `actor_ckpt_step_fn` / `learner_step_fn` are REQUIRED with NO defaults (the E32
        posture, WP-UNFREEZE §4.3): a default here would silently unwire the actor-lag
        check. They feed `ActorLagSpec` together with the monitor config's
        `actor_lag_threshold_steps` / `actor_lag_abort_enabled`, and are read LIVE at
        poll time;
      * `monitor_cfg` is REQUIRED with NO default, for the SAME reason and by the same
        posture (WPAX RED-TEAM F-2). It used to default to `None` and fall back to a bare
        `MonitorConfig()`, whose `actor_lag_abort_enabled` is `False` — so a caller that
        merely FORGOT the kwarg got a silently DISARMED hard abort while its config said
        armed. That is the ADJ-07 shape one function downstream of the `compose_run`
        `monitor_cfg=` parameter WPAX S-2 deleted, and it is the arm that actually
        constructs `ActorLagSpec`. Three of `ActorLagSpec`'s four inputs are now
        required-with-no-default; the fourth is derived from this one.

    The watchdog is returned UNSTARTED: the caller starts it only after the pool is up
    (an unstarted pool must never be torn down by a fire), and passes it to the
    coordinator so `close_out` can disarm staleness first (O-27).
    """
    cfg = monitor_cfg
    sink = JsonlEventSink(log_dir=Path(log_dir), run_id=run_id)
    registry = HeartbeatRegistry(sources=HEARTBEAT_SOURCES)
    snapshot_target = watchdog_snapshot_path(Path(buffer_persist_path))

    def _save_snapshot() -> None:
        saver = getattr(buffer, "save_to_path", None)
        if saver is not None:
            saver(str(snapshot_target))

    def _persist_errors_total() -> int:
        # LIVE module-attribute read (O-28): the counter is re-read on EVERY poll. The
        # ignore is only for the type checker, which cannot infer a `global`-mutated
        # module counter — never a licence to bind the value (a `from … import` here
        # would read a frozen 0 forever and silently exempt checkpoint persist failures).
        checkpoint_errors = int(
            _checkpoints.persist_errors_total  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        )
        return checkpoint_errors + int(sink.persist_errors_total)

    watchdog = HeartbeatWatchdog(
        registry=registry,
        deadlines={
            "train_step": cfg.heartbeat_deadline_train_step_sec,
            "inference_dispatch": cfg.heartbeat_deadline_inference_dispatch_sec,
            "selfplay_drain": cfg.heartbeat_deadline_selfplay_drain_sec,
            "eval_round": cfg.heartbeat_deadline_eval_round_sec,
        },
        sink=sink,
        counters_fn=_persist_errors_total,
        heartbeat_file=(Path(heartbeat_file) if heartbeat_file is not None
                        else Path(log_dir) / f"heartbeat_{run_id}.json"),
        file_interval_sec=cfg.heartbeat_file_interval_sec,
        poll_interval_sec=cfg.heartbeat_poll_interval_sec,
        close_out_deadline_sec=cfg.heartbeat_close_out_deadline_sec,
        snapshot_timeout_sec=cfg.heartbeat_fire_effect_timeout_sec,
        wired_sources=list(wired_sources),
        actor_lag=ActorLagSpec(
            learner_step_fn=learner_step_fn,
            actor_ckpt_step_fn=actor_ckpt_step_fn,
            threshold_steps=cfg.actor_lag_threshold_steps,
            abort_enabled=cfg.actor_lag_abort_enabled,
        ),
        save_snapshot=_save_snapshot,
        exit_fn=exit_fn,
    )
    _LOG.info("run_safety_built sink=%s heartbeat_file=%s", sink.path, watchdog.heartbeat_file)
    return RunSafety(sink=sink, registry=registry, watchdog=watchdog)
