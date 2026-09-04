"""Loop subsystem boot — inference/eval model builds + run-safety subsystems (WP10 §a.4/§c.6).

The old `training/lifecycle.py` was a NAME-COLLISION trap: it is subsystem-BOOT (model builds +
monitor/probe/dashboard construction), NOT the run-safety `lifecycle` repo_design §11 defines.
It lands HERE as `subsystems.py`, freeing the `lifecycle` name for the §11 subsystem.

The DISPLAY-boot half (WebDashboard, `register_renderer`/`register_jsonl_sink`, TB
`MetricsWriter`, `EarlyGameProbe`, `ValueProbe`) is DEFER/ARCH.

WPMAIN (R116/R121(b)) DELETED `build_subsystems` and `LoopSubsystems`. Both had ZERO callers
and zero test references, and the disk guard they returned had therefore never been
constructed in any run — LAW-16's third leg was dead, with its `60/10/5` arriving as
`config.get("disk_guard", {}).get(...)` code-side defaults over a key that existed in no
schema and no config. The guard is now composed by `mantis.run.compose_run` from the minted
`monitor.disk_guard` block through `mantis.config.resolve.disk_guard.resolve_disk_guard`.
This module's surviving subject is `build_run_safety`.

WP13-A adds `build_run_safety` — the composition root for the run-safety triple (the REAL
JSONL sink, the heartbeat registry, the INDEPENDENT watchdog thread). This is one of the
THREE declared `train → mantis.monitor` import sites (census-pinned, O-19); everywhere else
the sink stays INJECTED, never imported.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import mantis.train.checkpoints as _checkpoints
from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import HEARTBEAT_SOURCES, HeartbeatRegistry
from mantis.monitor.sink import JsonlEventSink
from mantis.train.lifecycle.heartbeat_watchdog import (
    ActorLagSpec,
    HeartbeatWatchdog,
    MonitorLivenessSpec,
)
from mantis.train.lifecycle.watchdog import watchdog_snapshot_path

_LOG = logging.getLogger(__name__)

# ══ THE MODEL-BUILD HALF IS DELETED (AUDIT-1 F-47) ══════════════════════════════════════
# `InfModelArch`, `build_inference_model`, `build_eval_model`, `cuda_warmup` and
# `_synthetic_graph_warmup` stood here with ZERO references anywhere in `src/`, `tests/` or
# `tools/`. The header above already recorded that WPMAIN deleted `build_subsystems` — the one
# caller this half ever had — and left the builders behind; this is the rest of that deletion.
#
# TWO THINGS WENT WITH THEM, and both are the reason the row mattered more than its size:
#   * `cuda_warmup` read `getattr(arch.arch, "in_dim", 11)` and `("edge_dim", 5)` — literal
#     defaults on IDENTITY quantities, read off the LIVE module, which is the sniff
#     `repo_design` §3 bans and AUDIT-1 F-46 counts.
#   * The consumer-registry citation for `train.amp_dtype` NAMED `cuda_warmup` as a read site.
#     A string naming a dead symbol satisfied the LAW-08 bijection, so the key's entry was
#     evidence of nothing. It now names `InferenceServer._warmup_compile_path`, which reads it.
#
# This module's surviving subject is `build_run_safety`, as the header says.


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
    monitor_liveness: Sequence[MonitorLivenessSpec] = (),
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
      * `monitor_liveness` (AUDIT-1 F-11 / R334(b)) names the MONITORS whose own counters
        the watchdog reports the liveness of — a monitor thread that swallows its errors is
        invisible on every observable it publishes, because what it stops publishing IS the
        evidence. It carries a `()` default, DELIBERATELY unlike the three neighbours above,
        and the distinction is the failure mode rather than the convenience: an omitted
        `wired_sources` / `actor_ckpt_step_fn` / `monitor_cfg` makes a healthy run fire or
        silently disarms a live abort, while an omitted monitor costs an observable and no
        run. It is not silent either way — `arm()` emits `monitor_liveness_unwired` on an
        empty tuple, and production's wiring is pinned structurally;
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
        monitor_liveness=monitor_liveness,
        save_snapshot=_save_snapshot,
        exit_fn=exit_fn,
    )
    _LOG.info("run_safety_built sink=%s heartbeat_file=%s", sink.path, watchdog.heartbeat_file)
    return RunSafety(sink=sink, registry=registry, watchdog=watchdog)
