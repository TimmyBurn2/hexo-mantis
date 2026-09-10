"""Loop subsystem boot — the run-safety composition root.

`build_run_safety` builds the REAL JSONL sink, the heartbeat registry and the INDEPENDENT
watchdog thread. This is one of the THREE declared `train → mantis.monitor` import sites
(census-pinned); everywhere else the sink stays INJECTED, never imported.
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


class RunSafety(NamedTuple):
    """The run-safety triple, tuple-shaped but named, so a caller can unpack it or read
    `run_safety.heartbeat`."""

    sink: JsonlEventSink
    registry: HeartbeatRegistry
    watchdog: HeartbeatWatchdog

    @property
    def heartbeat(self) -> Callable[[str], None]:
        """THE `HeartbeatFn` — pass it to `WorkerPool`, `InferenceServer` and
        `StepCoordinator`."""
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
      * the sink replaces `NullEventSink` everywhere the run injects one — ONE JSONL segment
        per process start, so a file never spans two run segments;
      * `run_safety.heartbeat` is handed to the pool / inference server / coordinator, so all
        three pipeline stages beat into ONE registry;
      * `counters_fn` reads the persist counters as LIVE module attributes, never a
        `from … import` that binds the int at import and reads a frozen 0 forever;
      * the fire-time snapshot targets the DISTINCT `.watchdog` path, so an abnormal-exit save
        can never truncate the resume buffer;
      * `wired_sources`, `actor_ckpt_step_fn`, `learner_step_fn` and `monitor_cfg` are REQUIRED
        with NO default: an omitted one either makes a healthy run fire 42 and relaunch into
        the same missing wiring, or silently disarms an abort the config says is armed. None
        of them may be inferred;
      * `monitor_liveness` carries a `()` default deliberately — an omitted monitor costs an
        observable and no run, and `arm()` still emits `monitor_liveness_unwired` on an empty
        tuple.

    The watchdog is returned UNSTARTED: the caller starts it only after the pool is up (an
    unstarted pool must never be torn down by a fire), and passes it to the coordinator so
    `close_out` can disarm staleness first.
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
        # LIVE module-attribute read, re-read on EVERY poll: a `from … import` here would bind
        # a frozen 0 and silently exempt checkpoint persist failures. The ignore is for the
        # type checker only, never a licence to bind the value.
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
