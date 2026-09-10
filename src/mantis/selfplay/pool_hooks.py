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
    does not live here; the pool talks to this shape and defaults to `NullRecorder`.

    R344(b) FILLED IT — `mantis.monitor.game_recorder.GameRecorder` is the first concrete
    implementation, and the signature widened to carry what a GAME RECORD needs and a replay
    file did not: which worker played it, how it ended, its id and its dedupe hash, and the
    sims each move was served. `game_length` became `plies` in the same act because that is
    what the one call site always passed (`game_length=plies`) — the old name described a
    quantity the drain computes separately as `(plies + 1) // 2`, so a reader of this
    Protocol was being told the wrong unit (LAW-03).
    """

    def set_step(self, step: int) -> None: ...

    def maybe_record(
        self,
        *,
        game_id: str,
        moves: list[tuple[int, int]],
        winner_code: int,
        plies: int,
        worker_id: int,
        terminal_reason: str,
        game_id_byte_hash: str,
        served_sims: int,
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
        self,
        *,
        game_id: str,
        moves: list[tuple[int, int]],
        winner_code: int,
        plies: int,
        worker_id: int,
        terminal_reason: str,
        game_id_byte_hash: str,
        served_sims: int,
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
    # WP12-R Phase T target-integrity counters (LAW-18, DESIGN_T §3.6): an idle
    # lever stays VISIBLE at 0 (the chain_loss_with_fire_rate posture).
    export_offwindow_mass_moves: int = 0
    target_integrity_defects: int = 0
    # R275(b) SEAM conjunct — leaf inferences that FAILED on an open queue and halted the
    # run. Same family and same posture as `target_integrity_defects`: run-fatal, so it
    # reads 0 in every run that survives to emit, and that permanent zero is the posture,
    # not an unproduced field. `int = 0` rather than the `None` wheel-compat default,
    # deliberately: the value is a COUNT that a live engine always supplies, and the
    # `_snapshot_counter` docstring's "the None arm cannot fire in production" claim
    # depends on every counter in that block being declared this way.
    inference_failures_total: int = 0
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


def runner_stats(pool: Any) -> RunnerStats:
    """Snapshot the runner's counters / scalars.

    Defaults via `getattr` cover engine builds that pre-date an individual counter —
    they reproduce the legacy per-field `getattr(runner, name, 0.0)` reaches this
    dataclass replaced, so a counter added later cannot break an older wheel.

    FIFTEEN FIELDS LEFT WITH R346(f) and are not merely unread here: the engine exposes no
    getter for any of them, so every one would have snapshotted its wheel-compat default
    forever — the two cluster means and their sample count, the K histogram, the
    uncovered-forced-win count, the seven solver counters, `seeded_games_started`, and
    `gridls_zero_policy_rows`. A snapshot field whose producer is gone publishes a
    fabricated reading, which is the phantom-input class LAW-07 refuses; the last of them
    was still riding the LAW-18 target-integrity channel as a permanent 0.
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
        export_offwindow_mass_moves=int(getattr(r, "export_offwindow_mass_moves", 0)),
        target_integrity_defects=int(getattr(r, "target_integrity_defects", 0)),
        inference_failures_total=int(getattr(r, "inference_failures_total", 0)),
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
