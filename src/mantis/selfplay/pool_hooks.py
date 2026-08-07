# >300 justify (R8): one boundary, seen from both directions. The injected collaborators the
# pool writes OUT to (EventSink / RecorderLike / HeartbeatFn), the CALLEE surface the train-side
# sync engine calls IN to (ActorSyncTarget), and the typed read-only snapshots that cross that
# same boundary carrying values are the pool's edge contract; splitting them would put a shape
# and the reader that fills it in separate files, where a dropped or re-typed field stops being
# a one-diff read. It crossed the cap when ADJ-D32 gave the two cluster means their
# `float | None` zero-sample semantics — precisely the kind of edge-contract change that has to
# be visible beside the Protocol the value flows through.
"""The pool's outward hook surface: injection Protocols + read-only snapshots.

This is the "promotion-hooks" quarter of the pool split. Two kinds of thing live here and
they share one reason to exist — they are the pool's EDGES:

  * the injected collaborators (`EventSink`, `RecorderLike`, `HeartbeatFn`) the pool writes
    OUT to, each with an explicit no-op default, and `ActorSyncTarget`, the surface the
    train-side sync engine calls IN to. Actor sync is a CALLEE surface: `WorkerPool`
    satisfies `ActorSyncTarget`, and nothing in this package imports `mantis.train` or
    `mantis.eval` — the caller reaches the pool by injection on its own side, so the
    import DAG stays one-way.
  * the typed read-only snapshots (`RunnerStats`, `InferenceStats`) that replaced ad-hoc
    reaches into the private runner / inference-server attributes, plus the small
    forwarders that sit on the same seam (weight sync, recorder step, radius override,
    batch-fill math).

Free functions taking the pool instance, so `pool.py` imports this module and never the
reverse.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Injection Protocols — every one has an explicit no-op default at the pool ctor.
# ---------------------------------------------------------------------------


class EventSink(Protocol):
    """Where `game_complete` / `system_stats` payloads go.

    Structural (duck-typed) on purpose: self-play defines its own shape rather than
    importing the trainer's, so there is no `selfplay → train` edge. The default is
    `None` — events are dropped, exactly as the WP-owned emit convention elsewhere.
    """

    def emit(self, event: Mapping[str, Any]) -> None: ...


#: Heartbeat sink. The argument is the SOURCE string ("selfplay_drain",
#: "inference_dispatch"); the consumer owns timestamps and staleness. Emission is
#: behaviour-neutral: with the default `None` no call site does anything, and no
#: watchdog thread, file or timer is created by this package.
HeartbeatFn = Callable[[str], None]


class RecorderLike(Protocol):
    """The replay-recorder seam. A concrete recorder is a display-surface concern and
    does not live here; the pool talks to this shape and defaults to `NullRecorder`."""

    def set_step(self, step: int) -> None: ...

    def maybe_record(
        self, *, moves: list[tuple[int, int]], winner_code: int, game_length: int
    ) -> None: ...

    def latest_replay_path(self) -> Path | None: ...

    def stop(self) -> None: ...


@runtime_checkable
class ActorSyncTarget(Protocol):
    """What the TRAIN-side sync engine (`mantis.train.actor_sync.ActorSync`) calls INTO
    the pool, on a fixed step cadence, unconditionally (WP-UNFREEZE, R49). Nothing else
    participates in the decision to sync: no gate, promotion, or eval code may ever
    hold a reference shaped like this.

    `runtime_checkable` so the surface conformance test can assert
    `isinstance(pool, ActorSyncTarget)` — that assertion is this Protocol's live
    consumer, and a Protocol nothing checks is the dead surface LAW-08 exists to
    prevent.
    """

    def sync_inference_weights(self, state_dict: dict[str, Any]) -> None: ...

    def update_checkpoint_step(self, step: int) -> None: ...


class NullRecorder:
    """The no-op `RecorderLike` default: records nothing, has no latest replay.

    Not a silent failure — it is the declared default for a seam whose concrete
    implementation is a display surface that does not exist in this tree. A pool built
    without a recorder therefore reports `latest_replay_path() is None`.
    """

    def set_step(self, step: int) -> None:
        return None

    def maybe_record(
        self, *, moves: list[tuple[int, int]], winner_code: int, game_length: int
    ) -> None:
        return None

    def latest_replay_path(self) -> Path | None:
        return None

    def stop(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Typed snapshots (read-only; no computation, no behaviour of their own)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerStats:
    """Snapshot of the Rust `SelfPlayRunner` counters / scalars."""

    games_completed: int
    positions_generated: int
    x_wins: int
    o_wins: int
    draws: int
    model_version: int
    mcts_quiescence_fires: int
    mcts_mean_depth: float
    mcts_mean_root_concentration: float
    # ADJ-D32 / R249: `None` = no samples, NOT "measured zero". The bridge getters
    # return `None` at `cluster_variance_sample_count == 0`, and this snapshot carries
    # that through unchanged — the count beside them says why. Coercing to 0.0 here
    # would restore the fabrication one layer up from where it was removed.
    cluster_value_std_mean: float | None
    cluster_policy_disagreement_mean: float | None
    cluster_variance_sample_count: int
    # In-run solver fire-rate counters (cumulative since pool start). The `getattr`
    # defaults below cover engine builds that pre-date an individual counter (all 0).
    solver_moves_eligible: int = 0
    solver_win_proven: int = 0
    solver_injected: int = 0
    solver_injected_offwindow: int = 0
    solver_budget_exhausted: int = 0
    solver_moves_eligible_seeded: int = 0
    solver_injected_seeded: int = 0
    seeded_games_started: int = 0
    # WP12-R Phase T target-integrity counters (LAW-18, DESIGN_T §3.6): an idle
    # lever stays VISIBLE at 0 (the chain_loss_with_fire_rate posture).
    export_offwindow_mass_moves: int = 0
    gridls_zero_policy_rows: int = 0
    target_integrity_defects: int = 0
    # Item 10(b) / R250: the DENSE record path's K histogram, bucket `i` counting
    # recorded positions with `K == i + 1` and the LAST bucket guarding every K outside
    # that range. `None` = NO PRODUCER, and it is the wheel-compat default for the same
    # reason R249 gave the cluster means theirs: an engine build without the getter has
    # measured nothing, and an all-zero tuple would read as "K was never anything",
    # which is a distribution. On a graph run the buckets are genuinely all zero and the
    # EMITTER drops the field — absence is decided there, on `is_graph_run`, not here.
    k_cluster_histogram: tuple[int, ...] | None = None
    # R256/ADJ-D37: proven forced wins swallowed by the LS coverage gate while the
    # injecting lever was armed. `None` = NO PRODUCER (an engine build predating the
    # getter), the k_cluster_histogram wheel-compat posture — NOT the Phase-T zeros:
    # an old wheel has measured nothing, and a fabricated produced-0 here would be
    # the exact reading R256 exists to kill. On dense runs the EMITTER omits the
    # field; absence is decided there, on `is_graph_run`, not here.
    uncovered_forced_win: int | None = None
    # Worker threads that died by panic. Reads 0 in a healthy run; non-zero means
    # self-play HALTED on a worker death rather than merely slowing down, which is the
    # distinction the old silent-swallow made impossible to draw.
    worker_panics: int = 0
    # Vestigial `None`-valued slot: the legacy 4-field encoding-spec getter retired with
    # the runner field it mirrored. Kept so external callers that construct `RunnerStats`
    # by kwarg do not break; the live spec cross-check reads `pool.encoding_spec`.
    runner_encoding: Any = None


@dataclass(frozen=True)
class InferenceStats:
    """Snapshot of `InferenceServer` counters + the bound encoding spec."""

    forward_count: int
    total_requests: int
    encoding_spec: Any


def _optional_mean(value: Any) -> float | None:
    """Carry a derived mean through as `float`, or `None` when the producer has no
    samples (R249). `float(None)` raises and a `getattr(..., 0.0)` default would
    manufacture the very zero ADJ-D32 removed, so the absence is preserved explicitly."""
    return None if value is None else float(value)


def _k_histogram(value: Any) -> tuple[int, ...] | None:
    """Freeze the bridge's K-histogram list into a tuple, or carry `None` through when
    the engine build has no such getter (item 10(b)).

    Frozen because `RunnerStats` is a frozen dataclass and a list field would hand every
    consumer a mutable alias of one snapshot. `None` is preserved rather than defaulted to
    zeros for the R249 reason: zeros here are a DISTRIBUTION claim, and an absent producer
    has not measured one. An EMPTY list from a live getter is likewise not zeros — it is
    kept as an empty tuple, so the emitter can tell "no buckets" from "no producer"."""
    return None if value is None else tuple(int(v) for v in value)


def runner_stats(pool: Any) -> RunnerStats:
    """Snapshot the runner's counters / scalars.

    Defaults via `getattr` cover engine builds that pre-date an individual counter —
    they reproduce the legacy per-field `getattr(runner, name, 0.0)` reaches this
    dataclass replaced, so a counter added later cannot break an older wheel. The two
    cluster means default to `None`, not 0.0: their absence is a REAL state (zero
    samples), so the wheel-compat default and the live zero-count reading agree.
    """
    r = pool._runner
    return RunnerStats(
        games_completed=int(getattr(r, "games_completed", 0)),
        positions_generated=int(getattr(r, "positions_generated", 0)),
        x_wins=int(getattr(r, "x_wins", 0)),
        o_wins=int(getattr(r, "o_wins", 0)),
        draws=int(getattr(r, "draws", 0)),
        model_version=int(getattr(r, "model_version", 0)),
        mcts_quiescence_fires=int(getattr(r, "mcts_quiescence_fires", 0)),
        mcts_mean_depth=float(getattr(r, "mcts_mean_depth", 0.0)),
        mcts_mean_root_concentration=float(
            getattr(r, "mcts_mean_root_concentration", 0.0)
        ),
        cluster_value_std_mean=_optional_mean(
            getattr(r, "cluster_value_std_mean", None)
        ),
        cluster_policy_disagreement_mean=_optional_mean(
            getattr(r, "cluster_policy_disagreement_mean", None)
        ),
        cluster_variance_sample_count=int(
            getattr(r, "cluster_variance_sample_count", 0)
        ),
        solver_moves_eligible=int(getattr(r, "solver_moves_eligible", 0)),
        solver_win_proven=int(getattr(r, "solver_win_proven", 0)),
        solver_injected=int(getattr(r, "solver_injected", 0)),
        solver_injected_offwindow=int(getattr(r, "solver_injected_offwindow", 0)),
        solver_budget_exhausted=int(getattr(r, "solver_budget_exhausted", 0)),
        solver_moves_eligible_seeded=int(
            getattr(r, "solver_moves_eligible_seeded", 0)
        ),
        solver_injected_seeded=int(getattr(r, "solver_injected_seeded", 0)),
        seeded_games_started=int(getattr(r, "seeded_games_started", 0)),
        export_offwindow_mass_moves=int(getattr(r, "export_offwindow_mass_moves", 0)),
        gridls_zero_policy_rows=int(getattr(r, "gridls_zero_policy_rows", 0)),
        target_integrity_defects=int(getattr(r, "target_integrity_defects", 0)),
        k_cluster_histogram=_k_histogram(getattr(r, "k_cluster_histogram", None)),
        uncovered_forced_win=(
            None
            if getattr(r, "uncovered_forced_win", None) is None
            else int(r.uncovered_forced_win)
        ),
        worker_panics=int(getattr(r, "worker_panics", 0)),
    )


def inference_stats(pool: Any) -> InferenceStats:
    """Snapshot the inference server's counters + bound encoding spec."""
    s = pool._inference_server
    return InferenceStats(
        forward_count=int(getattr(s, "_forward_count", 0)),
        total_requests=int(getattr(s, "_total_requests", 0)),
        encoding_spec=getattr(s, "encoding_spec", None),
    )


def batch_fill_pct(pool: Any) -> float:
    """Mean batch occupancy as a percentage of the configured batch size, capped at 100.

    Zero forwards ⇒ 0.0 rather than a division by zero, so the metric is defined from the
    first monitor read onward.
    """
    srv = pool._inference_server
    fwd = getattr(srv, "_forward_count", 0)
    reqs = getattr(srv, "_total_requests", 0)
    bs = getattr(srv, "_batch_size", 1)
    if fwd == 0:
        return 0.0
    return min((reqs / (fwd * max(bs, 1))) * 100.0, 100.0)


def inference_batch_timing(pool: Any) -> dict[str, Any]:
    """The inference server's batching instrument, as an `iteration_complete` block.

    The DISTRIBUTION behind `batch_fill_pct`'s single ratio, plus the two waits that ratio
    cannot see: the collector wait (`queue_wait`) and the collate cost. `batch_fill_pct`
    is a mean over the whole run and cannot distinguish "always 1 request per forward"
    from "sometimes 64, sometimes 0"; this block carries the min/max and the histogram
    that can, and it carries the deadline (`max_wait_ms`) each wait is measured against.
    """
    return pool._inference_server.batch_timing_snapshot()


# ---------------------------------------------------------------------------
# Forwarders on the same seam (mutating actions, so not on the snapshots)
# ---------------------------------------------------------------------------


def sync_inference_weights(pool: Any, state_dict: dict[str, Any]) -> None:
    """Forward a promoted state_dict to the bound inference server.

    The promotion path used to reach into the private inference server directly. This is
    a mutating action, not a stat, so it gets its own forwarder rather than living on the
    snapshot dataclasses.
    """
    pool._inference_server.load_state_dict_safe(state_dict)


def update_checkpoint_step(pool: Any, step: int) -> None:
    """Forward the current training step to the game recorder."""
    pool._recorder.set_step(step)


def latest_replay_path(pool: Any) -> Path | None:
    """Most recent recorded self-play replay file, or `None`.

    `None` under the default `NullRecorder` — the concrete recorder is an injected
    collaborator this package does not build.
    """
    return pool._recorder.latest_replay_path()


__all__ = [
    "EventSink",
    "HeartbeatFn",
    "InferenceStats",
    "NullRecorder",
    "ActorSyncTarget",
    "RecorderLike",
    "RunnerStats",
    "batch_fill_pct",
    "inference_batch_timing",
    "inference_stats",
    "latest_replay_path",
    "runner_stats",
    "sync_inference_weights",
    "update_checkpoint_step",
]
