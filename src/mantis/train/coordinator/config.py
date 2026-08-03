# >300 justify (R8), stated at this file's MEASURED size of 382 lines (re-measured at
# WP12-R Phase O, which declares `runner_stats` on `WorkerPoolLike` — the coordinator now
# reads the runner snapshot itself for `iteration_complete`'s target-integrity block). It crossed the cap at
# WPMINT Phase K-B, which DELETED six fields and added no code beyond moving `draw_rate_consec`
# onto `DrawRateAbortLike` as `consec` — the growth is entirely the `StepCoordinatorConfig`
# docstring recording WHY six fields are gone and why no field may carry a default; WPCLEAN
# Phase PC (R106) then completed the protocol declarations against their concretes. This module
# is the DAG-clean seam layer: the injected-collaborator Protocols, the config dataclass they are
# typed against, and the outcome record. Splitting it would put a Protocol and the dataclass that
# consumes it on opposite sides of an import for no gain, and `pooled_draw_rate` sits here
# because `DrawRateAbortLike` is the shape it is bounded by. Roughly two fifths is that rationale.
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
    """The DECLARED coordinator↔trainer seam (WPTS Phase T, R102 class-kill).

    The members here are exactly what the coordinator-side call sites use (`step.py`,
    `coordinator/dispatch.py`, `loop.py`), pinned both ways by
    `tests/train/test_trainer_seam_conformance.py` — an undeclared call site on this seam
    reds that gate. `train_step` is DEAD (TD-1): the seam is the two TYPED entry points,
    dispatched by `coordinator/dispatch.py` off the declared representation — never a
    buffer sniff, and never an untyped adapter joining a buffer to "whichever" path.
    """

    step: int
    model: Any
    device: Any

    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]: ...
    def train_step_from_graph_batch(self, **kwargs: Any) -> dict[str, float]: ...
    def save_checkpoint(self, loss_info: dict[str, float] | None) -> Any: ...


@runtime_checkable
class ReplayBufferLike(Protocol):
    size: int
    capacity: int

    def resize(self, new_capacity: int) -> None: ...
    def save_to_path(self, path: str) -> None: ...


@runtime_checkable
class GraphRouteBufferLike(Protocol):
    """The graph route-key (WPCLEAN Phase PC, R106 — declaring dispatch.py's probe).

    Deliberately ONE member and deliberately NOT folded into `ReplayBufferLike`: each engine
    buffer carries exactly one sampler, and the OTHER route's absence is the
    `RepresentationRouteError` mismatch signal (`coordinator/dispatch.py`, R102). A shared
    protocol claiming both members would erase the very asymmetry the typed route keys on.
    """

    def sample_graph_batch(self, batch_size: int, *, augment: bool, recent_frac: float) -> Any: ...


@runtime_checkable
class GridRouteBufferLike(Protocol):
    """The grid route-key — `GraphRouteBufferLike`'s dense twin; same grounds, same fence."""

    def sample_batch_with_pos(self, n: int, augment: bool) -> Any: ...


@runtime_checkable
class RecentBufferLike(Protocol):
    """Completed against `train/recency_buffer.py` (WPCLEAN Phase PC re-census): `size` is
    read by `coordinator/dispatch.py`'s grid arm and `save_to_path` by
    `train/buffer_persist.py`'s best-effort snapshot — both existed on the concrete,
    neither was declared (the C-2b-adjacent drift class R106 rules on)."""

    size: int

    def push(self, *args: Any, **kwargs: Any) -> None: ...
    def sample(self, *args: Any, **kwargs: Any) -> Any: ...
    def save_to_path(self, path: str) -> int: ...


@runtime_checkable
class DrawRateAbortLike(Protocol):
    """The RESOLVED draw-rate abort terms, as this seam layer sees them (WPAX Phase D).

    A local Protocol for the same reason every other injected collaborator here has one:
    this file is the DAG-clean seam layer, so it describes the shape it consumes rather
    than importing the concrete `mantis.config.resolve.draw_rate.DrawRateAbortSpec`.
    `None` in the field's type is the EXPLICIT disarmed posture — never an absent value.

    Read-only properties, not plain attributes: the concrete spec is a FROZEN dataclass,
    and this seam only ever reads the four terms — a writable declaration here would
    reject the frozen concrete.
    """

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
    # WP12-R Phase O (R164): the coordinator now READS the runner snapshot itself, to build
    # the `target_integrity` fire-rate block `iteration_complete` carries. It is the one
    # member this protocol shares with `mantis.train.events.PoolTelemetryLike` — declared
    # here because the conformance gate (`tests/train/test_trainer_seam_conformance.py`)
    # measures `step.py`'s pool accesses against THIS protocol, and a called-and-undeclared
    # member is the TD-1 class R106 exists to kill. `Any` keeps the no-`train → selfplay`
    # edge, exactly as `PoolTelemetryLike` types it.
    def runner_stats(self) -> Any: ...


@runtime_checkable
class EvalPipelineLike(Protocol):
    """The injected eval seam — the ONLY way the coordinator reaches eval (no `train → eval`
    import). WP11 supplies the concrete pipeline.

    Completed against the concrete (WPCLEAN Phase PC, R106 / CENSUS_C C-10/C-11/C-16):
    `poll_completed` (step.py's every-iteration mailbox read), `drain_pending` (the
    teardown flush) and `apply_gate_decision` (the promotion applier) were
    called-and-undeclared. Declaring them changes NO runtime posture: drain.py keeps its
    getattr guards — an absent `drain_pending` still no-ops the flush and an absent
    `apply_gate_decision` still logs `eval_promotion_unapplied` LOUD — the conformance gate
    (`tests/train/test_trainer_seam_conformance.py`) is what now reds an undeclared call.
    """

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


# ── close-out drain caps: DELETED, not moved (WPMINT Phase K-A, R93) ────────────────────
# The four `DEFAULT_FINAL_EVAL_DRAIN_*` / `DEFAULT_TERMINAL_EVAL_HARD_CAP_SEC` constants
# that stood here are GONE. `monitor.drain.*` is the authority (`DrainCapsConfig` ->
# `mantis.config.resolve.drain.resolve_drain_caps`), and a named constant beside an
# authored key is the duplicated-default class R1 exists to kill — one of the four
# (`DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC`) had already rotted into a dead twin of a bare
# `900.0` literal in `run.py` with no reader at all. Re-adding one is a regression.


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
    """Per-step coordinator knobs. EVERY field is CONFIG-AUTHORED and NONE carries a default
    (WPMINT Phase K-B, `CARD-COORD-KNOBS` / R78 as clarified by R80).

    `bot_corpus_refresh_*` (the KILLED bot_refresh subprocess family) was DROPPED at WP10.
    Phase K-B deletes six more — `composition_interval`, `value_probe_interval`,
    `soft_ew_threshold`, `soft_ew_min_pts`, `instrumentation_enabled`, `bot_corpus_path` —
    which had NO reader anywhere in `src/` (re-verified at HEAD by grep AND by recording every
    attribute read on a live instance across the whole test tier). They are deleted rather
    than authored because a config key with no live consumer is an R1/LAW-08 violation, so
    typing them into the schema would have created the defect the card exists to close
    (adjudication call K-a). `bot_batch_share` survives and is authored — it is read, at
    `step.py::_run_training_step` — even though its sibling path knob is gone; that asymmetry
    is disclosed on `train.bot_batch_share` itself.

    `draw_rate_consec` is gone too, in the other direction: it MOVED, into
    `train.draw_rate_abort.consec` and thence onto `DrawRateAbortLike.consec`, because a term
    of a DISARMED abort is not a fact (R80's "the terms travel together").

    `checkpoint_interval` is DELETED by R178(a) (R116/LAW-08). It was the REPLAY-BUFFER save
    cadence — never the trainer's, which is `TrainHParams.checkpoint_interval` and is
    untouched — and its only reader was `step.py`'s D4 `_try_save_buffer` arm, which WP12-R
    Phase CS (F-CS-2) measured production-dead on every leg. The arm and the config key
    `train.buffer_save_interval` that fed it are deleted with the field; buffer persistence
    returns only as ONE design under CARD-RESUME (R178(c), post-mint).

    NO FIELD HAS A DEFAULT, and that is the invariant, not a coincidence: with the schema
    authoritative a default here is a second authority a caller silently inherits, which is
    exactly what `draw_rate_threshold: float = 0.0` was and what the drain caps' four
    `DEFAULT_*` constants were. Construction fails rather than assuming anything.
    """

    eval_interval: int
    log_interval: int
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
    hard_gn_threshold: float
    hard_gn_min_steps: int
    stop_step: int | None
    # WPAX Phase D (R65 + R80): NO default, and it sits HERE — beside `stop_step` — because
    # these are now precisely the two facts the CONFIG authors on this dataclass. `None` is
    # EXPLICITLY OFF (`train.draw_rate_abort: null`), never an inherited posture: a literal
    # the caller always replaces is still a second default authority (R1), which is what
    # `draw_rate_threshold: float = 0.0` was.
    draw_rate_abort: DrawRateAbortLike | None
    # WPMINT Phase K-A (R93, the DR-11 finding): the four drain/terminal-eval caps join
    # `stop_step`/`draw_rate_abort` as CONFIG-AUTHORED facts, and so lose their code-side
    # defaults for the same reason those two have none. `monitor.drain.*` had been minted,
    # schema-validated and registry-claimed since SC-A3 while `resolve_monitor_config`
    # popped the block and threw it away — the three defaults below (and a fourth, dead,
    # `DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC`) were what the run actually used. A default
    # here is a second authority over the same number, so there is none: the value arrives
    # through `mantis.config.resolve.drain.resolve_drain_caps` or construction fails.
    final_eval_drain_timeout_sec: float
    eval_final_drain_safety_factor: float
    eval_final_drain_hard_cap_sec: float
    terminal_eval_hard_cap_sec: float
    # WPMINT Phase K-B: the last three terminal defaults on this dataclass are GONE. Each was
    # a second authority that would have survived the schema key beside it —
    # `terminal_eval_enabled`'s was the LAST of three (Phase K-A retired the
    # `getattr(cfg, "terminal_eval_enabled", True)` shadow in `drain.py`), and
    # `selfplay_stall_timeout_sec = 1800.0` sat beside a watchdog LAW-16 calls always-armed
    # while `watchdog.py`'s own contract lets `<= 0` disable the fire AND still emit the
    # arm-log. `train.selfplay_stall_timeout_sec`'s `gt=0` is what makes that posture
    # unwritable; the watchdog keeps its arm for direct constructions.
    terminal_eval_enabled: bool
    # §178 bot-corpus batch slot (a mixing knob — NOT the killed refresh hook). Its sibling
    # `bot_corpus_path` was one of the six DEAD fields deleted by this phase.
    bot_batch_share: float
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
    games_per_hour: float
