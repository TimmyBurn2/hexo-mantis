"""Step-coordinator collaborator Protocols + config + outcome (WP10 §a.4 split — `config` slice).

The 13-class god-module `training/step_coordinator.py` splits by responsibility (collaborator
protocol): this file is the DAG-clean seam layer — the injected-collaborator Protocols (no torch
import), `StepCoordinatorConfig`, `StepOutcome`, and the `RealClock`/`RealTracemalloc` defaults.
`step.py` holds `StepCoordinator.step()`; `drain.py` holds the terminal-eval flush + close_out.

KILL severances (must not re-enter): the `bot_refresh` subprocess family (`bot_corpus_refresh_*`
config fields) is a DEFINITE KILL (0 config consumers, §e/§f) — those fields are dropped; the
`bot_refresh.py` slice is NOT created. `EvalPipelineLike` keeps eval an INJECTED seam (no
`train → eval` import).
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# ── Protocols (no torch import; the DAG-clean injected seams) ───────────────────────────
@runtime_checkable
class TrainerLike(Protocol):
    step: int
    model: Any

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]: ...
    def save_checkpoint(self, loss_info: dict[str, float] | None) -> Any: ...


@runtime_checkable
class ReplayBufferLike(Protocol):
    size: int
    capacity: int

    def resize(self, new_capacity: int) -> None: ...
    def save_to_path(self, path: str) -> None: ...


@runtime_checkable
class RecentBufferLike(Protocol):
    def push(self, *args: Any, **kwargs: Any) -> None: ...
    def sample(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class WorkerPoolLike(Protocol):
    games_completed: int
    n_workers: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def buffer_composition(self) -> dict[str, Any]: ...
    def per_worker_draw_rates(self) -> dict[int, float]: ...
    def current_stride5_p90(self) -> int: ...
    def check_producer_health(self) -> None: ...
    def update_checkpoint_step(self, step: int) -> None: ...


@runtime_checkable
class EvalPipelineLike(Protocol):
    """The injected eval seam — the ONLY way the coordinator reaches eval (no `train → eval`
    import). WP11 supplies the concrete pipeline."""

    def run_evaluation(
        self,
        model: Any,
        step: int,
        best: Any | None,
        *,
        full_config: dict[str, Any],
        best_model_step: int | None,
        ignore_stride: bool = False,
    ) -> dict[str, Any]: ...


@runtime_checkable
class GpuMonitorLike(Protocol):
    gpu_util_pct: float


@runtime_checkable
class ClockLike(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class TracemallocLike(Protocol):
    def start(self, max_frames: int = 25) -> None: ...
    def stop(self) -> None: ...
    def get_traced_memory(self) -> tuple[int, int]: ...
    def take_snapshot(self) -> Any: ...
    def reset_peak(self) -> None: ...


# ── Default implementations ─────────────────────────────────────────────────────────────
class RealClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class RealTracemalloc:
    def start(self, max_frames: int = 25) -> None:
        tracemalloc.start(max_frames)

    def stop(self) -> None:
        tracemalloc.stop()

    def get_traced_memory(self) -> tuple[int, int]:
        return tracemalloc.get_traced_memory()

    def take_snapshot(self) -> Any:
        return tracemalloc.take_snapshot()

    def reset_peak(self) -> None:
        tracemalloc.reset_peak()


# ── close-out / abort-gate defaults (named, no literals — §D-LOOPFIX W1) ────────────────
DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC: float = 900.0
DEFAULT_FINAL_EVAL_DRAIN_SAFETY_FACTOR: float = 3.0
DEFAULT_FINAL_EVAL_DRAIN_HARD_CAP_SEC: float = 14400.0
DEFAULT_TERMINAL_EVAL_HARD_CAP_SEC: float = 14400.0


def promotion_capable_rounds(stop_step: int | None, eval_interval: int, best_stride: int) -> list[int]:
    """The round indices in a bounded run that are promotion-capable (best_checkpoint opponent
    fires → a gate decision can land). Surfaced at launch so a near-empty decision cadence is
    LOUD, not silent (§D-LOOPFIX W1)."""
    if stop_step is None or eval_interval <= 0:
        return []
    n_rounds = stop_step // eval_interval
    stride = max(int(best_stride), 1)
    return [r for r in range(1, n_rounds + 1) if r % stride == 0]


def recent_pool_draw_rate(per_worker_rates: "dict[int, float]") -> float:
    """Pool-wide recent self-play draw rate (unweighted mean over workers with a game). 0.0
    when no worker has a game yet, so the draw-rate hard-abort gate can't fire on empty signal."""
    if not per_worker_rates:
        return 0.0
    return sum(per_worker_rates.values()) / len(per_worker_rates)


@dataclass(frozen=True)
class StepCoordinatorConfig:
    """Per-step coordinator knobs. `bot_corpus_refresh_*` (the KILLED bot_refresh subprocess
    family) is DROPPED; `bot_batch_share`/`bot_corpus_path` are batch-mixing knobs, kept."""

    eval_interval: int
    log_interval: int
    checkpoint_interval: int
    composition_interval: int
    value_probe_interval: int
    min_buf_size: int
    capacity: int
    buffer_schedule: tuple[dict[str, Any], ...]
    training_steps_per_game: float
    max_train_burst: int
    batch_size: int
    augment: bool
    recency_weight: float
    mixing_initial_w: float
    mixing_min_w: float
    mixing_decay_steps: float
    soft_ew_threshold: float
    soft_ew_min_pts: int
    hard_gn_threshold: float
    hard_gn_min_steps: int
    instrumentation_enabled: bool
    stop_step: int | None
    final_eval_drain_timeout_sec: float
    eval_final_drain_safety_factor: float = DEFAULT_FINAL_EVAL_DRAIN_SAFETY_FACTOR
    eval_final_drain_hard_cap_sec: float = DEFAULT_FINAL_EVAL_DRAIN_HARD_CAP_SEC
    terminal_eval_enabled: bool = True
    terminal_eval_hard_cap_sec: float = DEFAULT_TERMINAL_EVAL_HARD_CAP_SEC
    # §CANARY-VAL stride-5 spam hard-abort (threshold <= 0 disables).
    stride5_p90_threshold: float = 30.0
    stride5_p90_consec: int = 3
    # §D-GOLONG sustained draw-rate hard-abort (threshold <= 0 disables; default OFF).
    draw_rate_threshold: float = 0.0
    draw_rate_consec: int = 3
    draw_rate_min_step: int = 0
    # §178 bot-corpus batch slot (mixing knobs — NOT the killed refresh hook).
    bot_batch_share: float = 0.0
    bot_corpus_path: str = ""
    # Self-play stall watchdog (2026-07-11 run2 eval-boundary wedge; <= 0 disables).
    selfplay_stall_timeout_sec: float = 1800.0


@dataclass(frozen=True)
class StepOutcome:
    """The decision record one `step()` returns (every decision made on that iter)."""

    train_step: int
    games_played: int
    in_warmup: bool
    waiting_for_games: bool
    steps_run: int
    last_loss_info: dict[str, float] | None
    buffer_resized: int | None
    checkpoint_saved: bool
    axis_emitted: bool
    eval_kicked_off: bool
    eval_skipped_busy: bool
    eval_drained: bool
    promoted_step: int | None
    soft_abort_fired: bool
    hard_abort_fired: bool
    consec_high_gn: int
    instrumentation_emitted: list[str]
    pool_overflow_delta: int
    games_per_hour: float
