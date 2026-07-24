"""Run-safety lifecycle subsystem (repo_design §11).

The §11 subsystem boundary: SIGINT/SIGTERM save-then-exit, the always-armed self-play
stall watchdog, the INDEPENDENT heartbeat watchdog (WP13-A L-B), and the disk guard.
Distinct from the OLD `training/lifecycle.py` (subsystem BOOT / model builds + display),
which relocates to `train/subsystems.py` (WP10 §a.2 name-collision note).

The two watchdogs are complements, not duplicates: `StallWatchdog` watches games-PROGRESS
from inside the loop (it catches a live-but-unproductive run), `HeartbeatWatchdog` watches
per-source liveness from its own thread (it catches a wedged one). Both end in `os._exit`.
"""
from __future__ import annotations

from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.train.lifecycle.heartbeat_watchdog import HeartbeatWatchdog
from mantis.train.lifecycle.signals import ShutdownState, install_signal_handlers
from mantis.train.lifecycle.watchdog import (
    DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC,
    SELFPLAY_STALL_EXIT_CODE,
    StallWatchdog,
    watchdog_snapshot_path,
)

__all__ = [
    "DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC",
    "SELFPLAY_STALL_EXIT_CODE",
    "DiskGuard",
    "HeartbeatWatchdog",
    "ShutdownState",
    "StallWatchdog",
    "install_signal_handlers",
    "watchdog_snapshot_path",
]
