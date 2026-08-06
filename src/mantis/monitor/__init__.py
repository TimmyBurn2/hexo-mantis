"""HEADLESS ONLY: event emit, producer manifest, alert rules — no display code, ever.

The public seams of the monitor package (WP13-A §a.1). Import-light and torch-free by law
(§2 L94, census-pinned O-18/O-19): stdlib + `yaml` + at most `mantis.util`/`mantis.encoding`
— nothing here may import `mantis.train`, `mantis.selfplay` or `mantis.eval`, so the
out-of-process supervisor loads in milliseconds on a box whose GPU has wedged.
"""
from __future__ import annotations

from mantis.monitor.best_effort import BestEffortCounters, best_effort
from mantis.monitor.config import MonitorConfig
from mantis.monitor.heartbeat import (
    HEARTBEAT_SOURCES,
    PERSIST_FATAL_EXIT_CODE,
    WATCHDOG_STALL_EXIT_CODE,
    HeartbeatFileState,
    HeartbeatRegistry,
    read_heartbeat_file,
    write_heartbeat_file,
)
from mantis.monitor.logging_setup import configure_logging
from mantis.monitor.manifest import (
    DEFAULT_MANIFEST_PATH,
    ManifestError,
    load_manifest,
    verify_manifest,
)
from mantis.monitor.rules import (
    WARN_RULE_NAMES,
    check_draw_rate_collapse,
    check_entropy_collapse,
    check_grad_norm_spike,
    check_loss_increase_window,
    check_nonfinite_loss,
    check_sealbot_wr_hard_abort,
    check_selfplay_entropy_collapse,
    emit_training_step_alerts,
    sealbot_wr_trajectory_alert,
)
from mantis.monitor.sink import JsonlEventSink

__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "HEARTBEAT_SOURCES",
    "PERSIST_FATAL_EXIT_CODE",
    "WARN_RULE_NAMES",
    "WATCHDOG_STALL_EXIT_CODE",
    "BestEffortCounters",
    "HeartbeatFileState",
    "HeartbeatRegistry",
    "JsonlEventSink",
    "ManifestError",
    "MonitorConfig",
    "best_effort",
    "check_draw_rate_collapse",
    "check_entropy_collapse",
    "check_grad_norm_spike",
    "check_nonfinite_loss",
    "check_loss_increase_window",
    "check_sealbot_wr_hard_abort",
    "check_selfplay_entropy_collapse",
    "configure_logging",
    "emit_training_step_alerts",
    "load_manifest",
    "read_heartbeat_file",
    "sealbot_wr_trajectory_alert",
    "verify_manifest",
    "write_heartbeat_file",
]
