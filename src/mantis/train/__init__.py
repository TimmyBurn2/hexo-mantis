"""Trainer, step coordinator, lifecycle, pretrain, checkpoint IO (the ONE loader).

Slice 1 (WP10) lands: the ONE checkpoint loader + envelope-v2 writer (`checkpoints`), the
resume-precedence layer (`orchestrator`), the injected emit seam (`emit`), the run-safety
lifecycle subsystem (`lifecycle`), and the training-side event builders (`events`,
`axis_distribution`, `aux_decode`). The trainer/coordinator/loop/anchor/warmstart/pretrain
land in later slices.

Only the torch-free public seams are re-exported at package import (the emit Protocol + the
§11 lifecycle names); the checkpoint IO is reached via `mantis.train.checkpoints` so
`import mantis.train` stays cheap.
"""
from __future__ import annotations

from mantis.train.emit import EventSink, NullEventSink, emit_via
from mantis.train.lifecycle import (
    DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC,
    SELFPLAY_STALL_EXIT_CODE,
    DiskGuard,
    ShutdownState,
    StallWatchdog,
    install_signal_handlers,
    watchdog_snapshot_path,
)

__all__ = [
    "DEFAULT_SELFPLAY_STALL_TIMEOUT_SEC",
    "SELFPLAY_STALL_EXIT_CODE",
    "DiskGuard",
    "EventSink",
    "NullEventSink",
    "ShutdownState",
    "StallWatchdog",
    "emit_via",
    "install_signal_handlers",
    "watchdog_snapshot_path",
]
