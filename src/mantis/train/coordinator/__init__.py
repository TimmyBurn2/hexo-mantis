"""Step-coordinator package (WP10 §a.4 split of the old 13-class step_coordinator.py).

`config` = collaborator Protocols + `StepCoordinatorConfig` + `StepOutcome` + clock/tracemalloc
defaults; `step` = `StepCoordinator.step()` + `run_until_stopped`; `drain` = terminal-eval flush
+ `close_out`. `bot_refresh.py` is a DEFINITE KILL — NOT created.
"""
from __future__ import annotations

from mantis.train.coordinator.config import (
    ClockLike,
    EvalPipelineLike,
    GpuMonitorLike,
    GraphRouteBufferLike,
    GridRouteBufferLike,
    RealClock,
    RealTracemalloc,
    RecentBufferLike,
    ReplayBufferLike,
    StepCoordinatorConfig,
    StepOutcome,
    TracemallocLike,
    TrainerLike,
    WorkerPoolLike,
    pooled_draw_rate,
    promotion_capable_rounds,
)
from mantis.train.coordinator.step import StepCoordinator

__all__ = [
    "ClockLike",
    "EvalPipelineLike",
    "GpuMonitorLike",
    "GraphRouteBufferLike",
    "GridRouteBufferLike",
    "RealClock",
    "RealTracemalloc",
    "RecentBufferLike",
    "ReplayBufferLike",
    "StepCoordinator",
    "StepCoordinatorConfig",
    "StepOutcome",
    "TracemallocLike",
    "TrainerLike",
    "WorkerPoolLike",
    "pooled_draw_rate",
    "promotion_capable_rounds",
]
