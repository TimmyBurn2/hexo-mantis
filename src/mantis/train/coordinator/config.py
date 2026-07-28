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
class DrawRateAbortLike(Protocol):
    """The RESOLVED draw-rate abort terms, as this seam layer sees them (WPAX Phase D).

    A local Protocol for the same reason every other injected collaborator here has one:
    this file is the DAG-clean seam layer, so it describes the shape it consumes rather
    than importing the concrete `mantis.config.resolve.draw_rate.DrawRateAbortSpec`.
    `None` in the field's type is the EXPLICIT disarmed posture — never an absent value.
    """

    threshold: float
    min_step: int
    N_pool_min: int


@runtime_checkable
class WorkerPoolLike(Protocol):
    games_completed: int
    n_workers: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def buffer_composition(self) -> dict[str, Any]: ...
    def pooled_draw_counts(self) -> tuple[int, int]: ...
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


def pooled_draw_rate(counts: tuple[int, int], *, N_pool_min: int) -> float | None:
    """The draw-rate abort's gated statistic (WPMINT Phase DS, operator ruling R92).

    `counts` is `(draws, completed)` — IN THAT ORDER — summed over the UNION of the pool's
    per-worker draw windows (`WorkerPool.pooled_draw_counts`). The statistic is the POOLED
    COUNT-WEIGHTED rate `draws / completed`, a fraction in [0, 1].

    Returns **`None` = NO OBSERVATION** when `completed < N_pool_min`, and a `float`
    otherwise — including a genuine `0.0`, which is a real healthy measurement and belongs
    in the abort history. R92 answers ADJ-19 by TYPE, not by value: below the evidence bar
    the gate reports nothing, and a healthy-looking `0.0` synthesised from no evidence
    becomes unrepresentable rather than merely unlikely. Zero-completion starvation lands in
    the same `None` arm and is explicitly the STALL family's jurisdiction (R92), not this
    gate's.

    WHAT IT REPLACES, and why by ruling rather than by preference. `recent_pool_draw_rate`
    took an UNWEIGHTED MEAN over the workers past a per-worker inclusion bar — neither a
    pool rate nor a worker rate. WPMINT Phase DR measured it (RECHECK_D DR-3/DR-4):

    * 32 workers, one at 50 games all drawn, 31 healthy at 49 → included set = {that one},
      mean = 1.0, **FIRED** at a true pool draw rate of 0.0319;
    * the inverse — 31 workers drawing 100% at 49 games, 1 healthy at 50 → mean = 0.0,
      **SILENT** at a true pool draw rate of 0.968;
    * total collapse with nobody past the bar → empty map → `0.0`, appended to the abort
      history as a real healthy measurement.

    Count-weighting kills the first two (no worker can carry the pool, none can be excluded
    into invisibility — there is no inclusion bar left), and the `None` kills the third.
    Both counterexamples are PERMANENT regression oracles by R92
    (`tests/selfplay/test_drawrate_pooled_statistic.py`).

    `N_pool_min` is keyword-only and has NO default: it is
    `train.draw_rate_abort.N_pool_min`, and this is the ONE signature on the path that takes
    it, so a default here would be a second authority over the operator's pre-registered
    value (R1). Its top end is bounded by `schema/core.py` against
    `DRAW_RATE_WINDOW * selfplay.n_workers` — a bar above that ceiling would make this
    function return `None` for the whole run while the abort audited ARMED.
    """
    draws, completed = counts
    if completed < N_pool_min:
        return None
    return draws / completed


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
    # WPAX Phase D (R65 + R80): NO default, and it sits HERE — beside `stop_step` — because
    # these are now precisely the two facts the CONFIG authors on this dataclass. `None` is
    # EXPLICITLY OFF (`train.draw_rate_abort: null`), never an inherited posture: a literal
    # the caller always replaces is still a second default authority (R1), which is what
    # `draw_rate_threshold: float = 0.0` was.
    draw_rate_abort: "DrawRateAbortLike | None"
    final_eval_drain_timeout_sec: float
    eval_final_drain_safety_factor: float = DEFAULT_FINAL_EVAL_DRAIN_SAFETY_FACTOR
    eval_final_drain_hard_cap_sec: float = DEFAULT_FINAL_EVAL_DRAIN_HARD_CAP_SEC
    terminal_eval_enabled: bool = True
    terminal_eval_hard_cap_sec: float = DEFAULT_TERMINAL_EVAL_HARD_CAP_SEC
    # §D-GOLONG sustained draw-rate hard-abort. `threshold` and `min_step` moved to the
    # config (`train.draw_rate_abort`, above) at WPAX Phase D — R80 names exactly three
    # keys and `consec` is not one of them, so it stays a code-side default owned by
    # CARD-COORD-KNOBS (R78). WPMINT Phase DS re-read that boundary rather than assuming
    # it: R92's prereg row NAMES `consec=3` among the values that "stand", which
    # pre-registers a constant without making it a config key, and R80's assignment of it
    # to CARD-COORD-KNOBS is untouched. It stays here, and Phase K still owns it.
    # Safe rather than merely bounded: with `N_pool_min` closing the one-drawn-game route
    # (schema/train.py `_one_drawn_game_cannot_fire_the_abort`) and `min_step` closing the
    # early-run route, `consec` is not load-bearing for the ADJ-14 hazard.
    # DISCLOSED (WPMINT DR-8): `consec` counts consecutive CHECKS at a stride of
    # `log_interval` train steps, so at the shipped log_interval=1000 a `consec` of 3 is
    # 3000 sustained steps and run5's earliest possible fire is step 27000, not 25000.
    draw_rate_consec: int = 3
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
