"""Run-safety lifecycle subsystem (repo_design §11).

The §11 subsystem boundary: SIGINT/SIGTERM save-then-exit, the always-armed self-play
stall watchdog, and the disk guard. Distinct from the OLD `training/lifecycle.py`
(subsystem BOOT / model builds + display), which relocates to `train/subsystems.py`
(WP10 §a.2 name-collision note).
"""
from __future__ import annotations

from mantis.train.lifecycle.disk_guard import DiskGuard
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
    "ShutdownState",
    "StallWatchdog",
    "install_signal_handlers",
    "watchdog_snapshot_path",
]
