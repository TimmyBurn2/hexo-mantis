# >300 justify (R8): one boundary, seen from both directions. The injected collaborators the pool
# writes OUT to, the CALLEE surface the sync engine calls IN to, and the typed snapshots that
# cross the same boundary are one edge contract; splitting them would put a shape and the reader
# that fills it in separate files, where a dropped or re-typed field stops being a one-diff read.
"""The pool's outward hook surface: injection Protocols + read-only snapshots.

The injected collaborators the pool writes OUT to, each with an explicit no-op default, plus
`ActorSyncTarget`, the CALLEE surface the train-side sync engine calls IN to — so nothing here
imports `mantis.train` or `mantis.eval` and the import DAG stays one-way. Plus the typed
snapshots that replaced ad-hoc reaches into private attributes, and the forwarders on that seam.

Free functions taking the pool instance, so `pool.py` imports this module and never the reverse.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Injection Protocols — every one has an explicit no-op default at the pool ctor.


class EventSink(Protocol):
    """Where `game_complete` / `system_stats` payloads go. Structural on purpose: self-play
    defines its own shape rather than importing the trainer's, so there is no `selfplay -> train`
    edge, and the default `None` drops events."""

    def emit(self, event: Mapping[str, Any]) -> None: ...


#: Heartbeat sink; the argument is the SOURCE string and the consumer owns timestamps and
#: staleness. Behaviour-neutral — the default `None` creates no thread, file or timer.
HeartbeatFn = Callable[[str], None]


class RecorderLike(Protocol):
    """The replay-recorder seam: a concrete recorder is a display-surface concern, so the pool
    talks to this shape and defaults to `NullRecorder`. The signature carries what a GAME RECORD
    needs and a replay file did not, and `game_length` became `plies` because the old name
    described a quantity the drain computes separately and told a reader the wrong unit."""

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
    """What the TRAIN-side sync engine calls INTO the pool, on a fixed step cadence,
    unconditionally; nothing else participates in the decision to sync. `runtime_checkable` so
    the conformance test can assert `isinstance(pool, ActorSyncTarget)` — that assertion is this
    Protocol's live consumer."""

    def sync_inference_weights(self, state_dict: dict[str, Any]) -> None: ...

    def update_checkpoint_step(self, step: int) -> None: ...


class NullRecorder:
    """The no-op `RecorderLike` default: the declared default for a seam whose concrete
    implementation is a display surface that does not exist in this tree."""

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


# Typed snapshots: read-only, no computation and no behaviour of their own.


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
    # Target-integrity counters: an idle lever stays VISIBLE at 0.
    export_offwindow_mass_moves: int = 0
    target_integrity_defects: int = 0
    # Leaf inferences that FAILED on an open queue and halted the run: run-fatal, so its
    # permanent 0 is the posture rather than an unproduced field. `int = 0` rather than the
    # `None` wheel-compat default, because the value is a COUNT a live engine always supplies.
    inference_failures_total: int = 0
    # Worker threads that died by panic: non-zero means self-play HALTED rather than slowed.
    worker_panics: int = 0
    # Vestigial `None` slot, kept so kwarg constructions do not break; the live spec cross-check
    # reads `pool.encoding_spec`.
    runner_encoding: Any = None


@dataclass(frozen=True)
class InferenceStats:
    """Snapshot of `InferenceServer` counters + the bound encoding spec."""

    forward_count: int
    total_requests: int
    encoding_spec: Any


def runner_stats(pool: Any) -> RunnerStats:
    """Snapshot the runner's counters and scalars.

    Defaults via `getattr` cover engine builds pre-dating an individual counter. Fifteen fields
    left with the arms they measured: the engine exposes no getter for any of them, so each would
    have snapshotted its wheel-compat default forever, publishing a fabricated reading.
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
    """Mean batch occupancy as a percentage of the configured batch size, capped at 100; zero
    forwards give 0.0, so the metric is defined from the first monitor read onward."""
    srv = pool._inference_server
    fwd = getattr(srv, "_forward_count", 0)
    reqs = getattr(srv, "_total_requests", 0)
    bs = getattr(srv, "_batch_size", 1)
    if fwd == 0:
        return 0.0
    return min((reqs / (fwd * max(bs, 1))) * 100.0, 100.0)


def inference_batch_timing(pool: Any) -> dict[str, Any]:
    """The inference server's batching instrument, as an `iteration_complete` block: the
    DISTRIBUTION behind `batch_fill_pct`'s single ratio plus the collector wait and collate cost
    it cannot see. A mean over the whole run cannot distinguish "always 1 request per forward"
    from "sometimes 64, sometimes 0"."""
    return pool._inference_server.batch_timing_snapshot()


# Forwarders on the same seam: mutating actions, so not on the snapshots.


def sync_inference_weights(pool: Any, state_dict: dict[str, Any]) -> None:
    """Forward a promoted state_dict to the bound inference server — a mutating action rather
    than a stat, so it gets its own forwarder."""
    pool._inference_server.load_state_dict_safe(state_dict)


def update_checkpoint_step(pool: Any, step: int) -> None:
    """Forward the current training step to the game recorder."""
    pool._recorder.set_step(step)


def latest_replay_path(pool: Any) -> Path | None:
    """Most recent recorded self-play replay file, or `None` under the default `NullRecorder`."""
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
