# >300 justify (R8): the DAG-clean seam layer — the injected-collaborator Protocols, the config
# dataclass they are typed against, and the outcome record. Splitting it would put a Protocol and
# the dataclass that consumes it on opposite sides of an import for no gain, and
# `pooled_draw_rate` sits here because `DrawRateAbortLike` is the shape it is bounded by.
"""Step-coordinator collaborator Protocols + config + outcome.

The injected-collaborator Protocols (no torch import), `StepCoordinatorConfig`, `StepOutcome` and
the `RealClock` default. `step.py` holds `StepCoordinator.step()`; `drain.py` holds the
terminal-eval flush and close_out. `EvalPipelineLike` keeps eval an INJECTED seam, so there is no
`train -> eval` import.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainerLike(Protocol):
    """The DECLARED coordinator-to-trainer seam, pinned both ways by the conformance gate so an
    undeclared call site reds it. `train_step` is DEAD: the seam is the two TYPED entry points,
    dispatched off the declared representation — never a buffer sniff."""

    step: int
    model: Any
    device: Any
    #: The run's checkpoint directory, read by the stall watchdog's snapshot-path derivation; it
    #: used to be a CWD-relative code-side default.
    checkpoint_dir: Any
    #: The resume-bundle publisher the coordinator INSTALLS on the trainer — declared because an
    #: undeclared write is the same hidden coupling as an undeclared read.
    bundle_publisher: Any

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]: ...
    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]: ...
    #: The FORWARD-ONLY sibling, declared here because `dispatch.py` reaches it through the same
    #: holder.
    def eval_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]: ...
    def save_checkpoint(self, loss_info: dict[str, float] | None) -> Any: ...


@runtime_checkable
class ReplayBufferLike(Protocol):
    size: int
    capacity: int
    # The symmetry-draw counters LEFT with their reader, their producer and the dense buffer that
    # carried them: a `runtime_checkable` protocol still demanding them would refuse every ring.

    def resize(self, new_capacity: int) -> None: ...
    def save_to_path(self, path: str) -> None: ...
    #: The last sampled batch's rows-per-game and age quantiles, on the SHARED protocol because
    #: it is a fact about a ring, not about which sampler it carries.
    def last_batch_composition(self) -> dict[str, int]: ...


@runtime_checkable
class GraphRouteBufferLike(Protocol):
    """The graph route-key, deliberately ONE member and deliberately NOT folded into
    `ReplayBufferLike`: each engine buffer carries exactly one sampler, and the OTHER route's
    absence is the `RepresentationRouteError` mismatch signal that a shared protocol claiming
    both members would erase."""

    def sample_graph_batch(self, batch_size: int, *, augment: bool, recent_frac: float) -> Any: ...


@runtime_checkable
class GridRouteBufferLike(Protocol):
    """The grid route-key — `GraphRouteBufferLike`'s dense twin; same grounds, same fence."""

    def sample_batch_with_pos(self, n: int, augment: bool) -> Any: ...


@runtime_checkable
class RecentBufferLike(Protocol):
    """Completed against the concrete recorder: `size` and `save_to_path` both existed on it and
    neither was declared."""

    size: int

    def push(self, *args: Any, **kwargs: Any) -> None: ...
    def sample(self, *args: Any, **kwargs: Any) -> Any: ...
    def save_to_path(self, path: str) -> int: ...


@runtime_checkable
class DrawRateAbortLike(Protocol):
    """The RESOLVED draw-rate abort terms as this seam layer sees them — a local Protocol,
    because this file describes the shapes it consumes rather than importing the concretes.
    `None` is the EXPLICIT disarmed posture, and the members are read-only properties because the
    concrete spec is a FROZEN dataclass a writable declaration would reject."""

    @property
    def threshold(self) -> float: ...
    @property
    def min_step(self) -> int: ...
    @property
    def N_pool_min(self) -> int: ...
    @property
    def consec(self) -> int: ...


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
    # The coordinator READS the runner snapshot itself to build the target-integrity block. It is
    # the one member shared with `PoolTelemetryLike`, declared here because the conformance gate
    # measures `step.py`'s pool accesses against THIS protocol. `Any` keeps the DAG one-way.
    def runner_stats(self) -> Any: ...


@runtime_checkable
class EvalPipelineLike(Protocol):
    """The injected eval seam — the ONLY way the coordinator reaches eval. `poll_completed`,
    `drain_pending` and `apply_gate_decision` were called-and-undeclared; declaring them changes
    NO runtime posture, since `drain.py` keeps its getattr guards."""

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

    def poll_completed(self) -> dict[str, Any] | list[Any] | None: ...
    def drain_pending(self) -> dict[str, Any] | list[Any] | None: ...
    def apply_gate_decision(self, result: Any) -> int | None: ...


@runtime_checkable
class ClockLike(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


# The four drain/terminal-eval cap constants that stood here are GONE: `monitor.drain.*` is the
# authority, and one of the four had already rotted into a dead twin of a bare literal with no
# reader at all.


def promotion_capable_rounds(stop_step: int | None, eval_interval: int, best_stride: int) -> list[int]:
    """Return the round indices in a bounded run that are promotion-capable. Surfaced at launch
    so a near-empty decision cadence is LOUD, not silent."""
    if stop_step is None or eval_interval <= 0:
        return []
    n_rounds = stop_step // eval_interval
    stride = max(int(best_stride), 1)
    return [r for r in range(1, n_rounds + 1) if r % stride == 0]


def pooled_draw_rate(counts: tuple[int, int], *, N_pool_min: int) -> float | None:
    """Return the draw-rate abort's gated statistic: the POOLED COUNT-WEIGHTED rate
    `draws / completed` over the UNION of the pool's per-worker windows.

    Returns **`None` = NO OBSERVATION** below `N_pool_min`, and a `float` otherwise — including a
    genuine `0.0`, which is a real healthy measurement. The bar is answered by TYPE, not by
    value, so a healthy-looking zero synthesised from no evidence is unrepresentable.

    It replaces an UNWEIGHTED MEAN over workers past a per-worker inclusion bar, which was
    neither a pool rate nor a worker rate: measured, one worker at 50 games all drawn against 31
    healthy at 49 FIRED at a true pool rate of 0.0319, the inverse stayed SILENT at 0.968, and a
    total collapse appended `0.0` to the abort history as a healthy reading.

    `N_pool_min` is keyword-only with NO default, this being the ONE signature that takes it, and
    its top end is schema-bounded — a bar above that ceiling returns `None` for the whole run
    while the abort audits ARMED.
    """
    draws, completed = counts
    if completed < N_pool_min:
        return None
    return draws / completed


@dataclass(frozen=True)
class StepCoordinatorConfig:
    """Per-step coordinator knobs. EVERY field is CONFIG-AUTHORED and NONE carries a default.

    Fields are DELETED rather than authored when they have no reader in `src/`: a config key with
    no live consumer is the violation this dataclass exists to close. A default here would be a
    second authority a caller silently inherits, so construction fails rather than assuming.
    """

    eval_interval: int
    #: The NARRATION cadence: the `training_step` payload, the four WARN rules and the axis
    #: distribution. It decides nothing about arming — see `gate_interval` below.
    log_interval: int
    #: The ARMING cadence: the stride at which the LIVE hard-abort gates run and the
    #: `monitor_gates` summary is published. A separate field and not a reuse of `log_interval`
    #: because the defect is precisely that the two were one knob: at a minted `log_interval:
    #: 1000` no draw-rate abort could fire before training step 1000.
    gate_interval: int
    min_buf_size: int
    capacity: int
    buffer_schedule: tuple[dict[str, Any], ...]
    training_steps_per_game: float
    max_train_burst: int
    batch_size: int
    augment: bool
    recency_weight: float
    hard_gn_threshold: float
    hard_gn_min_steps: int
    stop_step: int | None
    # NO default, and it sits beside `stop_step` because these are the two facts the CONFIG
    # authors on this dataclass. `None` is EXPLICITLY OFF, never an inherited posture: a literal
    # the caller always replaces is still a second default authority.
    draw_rate_abort: DrawRateAbortLike | None
    # The four drain/terminal-eval caps are CONFIG-AUTHORED and lose their code-side defaults for
    # `stop_step`'s reason: `monitor.drain.*` had been minted and validated while the resolver
    # popped the block and threw it away, so the defaults were what the run actually used.
    final_eval_drain_timeout_sec: float
    eval_final_drain_safety_factor: float
    eval_final_drain_hard_cap_sec: float
    terminal_eval_hard_cap_sec: float
    # The last three terminal defaults are GONE; each was a second authority that would have
    # survived the schema key beside it, and `selfplay_stall_timeout_sec` sat beside a watchdog
    # whose own contract lets `<= 0` disable the fire while still emitting the arm-log.
    terminal_eval_enabled: bool
    # Self-play stall watchdog (2026-07-11 run2 eval-boundary wedge).
    selfplay_stall_timeout_sec: float


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
    # `games_per_hour` was a field here too, built as a hard `0.0` with NO reader anywhere — a
    # second, always-zero authority for a fact already published from a measured source.
