"""The self-play worker pool: construction, thread lifecycle, and the read surface.

>300 justify: ONE object. The trainer duck-types the pool as a single collaborator and reads
about twenty members off it, so the class cannot be split without breaking that contract; what
IS split is the implementation, into free functions over the pool instance. Concurrency is
Rust-owned, and the Python-side stats feeder is the SOLE producer of training data, so its death
is fatal and reported through `check_producer_health`. Every knob is resolved ONCE at
construction, except `search_kind` and `buffer_composition`, which re-read the LIVE config.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from mantis._engine import DEFAULT_CLUSTER_THRESHOLD, SelfPlayRunner
from mantis.config.resolve.edge_geometry_check import resolve_edge_geometry_check
from mantis.config.resolve.search import resolve_search_kind
from mantis.selfplay.buffers import ReplayFacade
from mantis.selfplay.hparams import (
    SelfPlayHParams,
    build_runner_config,
    is_graph_representation,
    resolve_pool_encoding,
)
from mantis.selfplay.inference_server import InferenceServer
from mantis.selfplay.instrumentation import PoolInstrumentation
from mantis.selfplay.pool_drain import run_stats_loop
from mantis.selfplay.pool_hooks import (
    EventSink,
    HeartbeatFn,
    InferenceStats,
    NullRecorder,
    RecorderLike,
    RunnerStats,
)
from mantis.selfplay.pool_hooks import batch_fill_pct as _batch_fill_pct
from mantis.selfplay.pool_hooks import (
    inference_batch_timing as _inference_batch_timing,
)
from mantis.selfplay.pool_hooks import inference_stats as _inference_stats
from mantis.selfplay.pool_hooks import latest_replay_path as _latest_replay_path
from mantis.selfplay.pool_hooks import runner_stats as _runner_stats
from mantis.selfplay.pool_hooks import (
    sync_inference_weights as _sync_inference_weights,
)
from mantis.selfplay.pool_hooks import update_checkpoint_step as _update_checkpoint_step
from mantis.selfplay.pool_push import buffer_composition as _buffer_composition

_LOG = logging.getLogger(__name__)


def _collate_dump_target(config: Any) -> tuple[str, Any]:
    """Where a self-play graph-contract failure is dumped, and its context. The directory is
    DERIVED from the run's own checkpoint directory, exactly as the trainer's dump is, so all
    three paths' dumps land under one run record with no second path authority. The context is
    a CALLABLE, because what is interesting is read at the moment of the fire."""
    try:
        ckpt_dir = config["train"]["checkpoint_dir"]
    except (KeyError, TypeError):
        ckpt_dir = "checkpoints"
    dump_dir = str(Path(ckpt_dir).parent / "collate_dumps")

    def _context() -> dict[str, Any]:
        return {"path": "selfplay", "phase": "selfplay_inference", "concurrency": 1}

    return dump_dir, _context


class WorkerPool:
    """Runs concurrent self-play games on Rust-owned worker threads."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: torch.device,
        replay_buffer: Any,
        n_workers: int | None = None,
        *,
        arch: Any,
        sink: EventSink | None = None,
        recorder: RecorderLike | None = None,
        heartbeat: HeartbeatFn | None = None,
    ) -> None:
        """Build the runner, the inference server and the drain state.

        `arch` is REQUIRED (pass `None` to state that there is nothing to cross-check): the
        resolved encoding's canvas geometry is checked against the arch's DECLARED `board_size`,
        so a mis-paired arch and config fail before any Rust runner exists. `sink`/`recorder`/
        `heartbeat` are injected with no-op defaults — declared seams, not silent failures.
        """
        self.model = model
        self.config = config
        self.device = device

        # Encoding resolve FIRST — an unregistered or unsupported encoding must fail
        # before any knob validation, matching the frozen construction order.
        resolved = resolve_pool_encoding(config, arch=arch)
        spec = resolved.registry_spec
        self.encoding_spec = spec
        # Gates the drain branch between the dense bulk-push path and the graph
        # per-row path. Closed match on the spec's representation — no dense default.
        self._is_graph: bool = is_graph_representation(spec)

        hp = SelfPlayHParams.from_config(config, n_workers)
        self.n_workers = hp.n_workers
        self.n_simulations = hp.n_simulations
        self.c_puct = hp.c_puct
        self.fpu_reduction = hp.fpu_reduction
        self.quiescence_enabled = hp.quiescence_enabled
        self.quiescence_blend_2 = hp.quiescence_blend_2
        self._effective_sims_per_move = hp.effective_sims_per_move

        # The pool takes a RAW engine buffer and wraps it: the facade resolves the kind from
        # the SAME spec the drain dispatches on and cross-checks the handle, so a graph buffer
        # under a grid encoding dies here rather than producing corrupt training data.
        self.replay_buffer = ReplayFacade(spec, replay_buffer)

        sp_config, dims = build_runner_config(
            hp,
            spec_dims=resolved,
            encoding_name=resolved.encoding_name,
        )
        self._runner = SelfPlayRunner(sp_config)
        self._inference_server = InferenceServer(
            model, device, config,
            batcher=self._runner.batcher,
            encoding_spec=spec,
            heartbeat=heartbeat,
            sink=sink,
            # 1-in-1 on the self-play path too, for the WHOLE run. The batch-size-derived
            # 1-in-64 that stood here rested on "a class that has only ever fired on eval", and
            # the class has since fired on the training path — at 1-in-64 a corrupted batch had
            # 63 chances in 64 of passing through untouched.
            collate_check_period=1,
            collate_dump=_collate_dump_target(config),
            # R347(e): where check 14 runs, read through its one resolver, never a literal.
            edge_geometry_check=resolve_edge_geometry_check(config),
        )

        self._stop_event = threading.Event()
        self._stats_thread: threading.Thread | None = None
        # Set if the sole-producer feeder daemon dies on an exception;
        # `check_producer_health` re-raises it so the trainer fails fast.
        self._producer_exc: BaseException | None = None

        self._lock = threading.Lock()
        self.games_completed = 0
        self.positions_pushed = 0
        self.self_play_positions_pushed = 0
        self.x_wins = 0
        self.o_wins = 0
        self.draws = 0
        # None = NOT MEASURED. A drain has to observe a positive `positions_generated` delta
        # over a positive interval before there is a rate at all; a starting 0.0 was published
        # as "the search is doing nothing" for every iteration before the first drain.
        self._sims_per_sec: float | None = None
        self._last_drain_time: float = time.monotonic()
        # Last-seen runner `positions_generated`; the per-drain delta is what the
        # sims/sec bill multiplies by the effective per-move sim count.
        self._last_pos_generated: int = 0
        self._total_sims: int = 0
        self._game_lengths: deque[int] = deque(maxlen=200)
        # None = NOT MEASURED, for the same reason as `_sims_per_sec`: no game has finished
        # yet, so there is no mean length. A 0.0 reads as "games are ending instantly".
        self._avg_game_length: float | None = None

        # Injected collaborators (all optional, all defaulting to inert).
        self._sink: EventSink | None = sink
        self._heartbeat: HeartbeatFn | None = heartbeat
        self._recorder: RecorderLike = (
            recorder if recorder is not None else NullRecorder()
        )

        # Optional recent buffer for recency-weighted sampling. Set by the training loop
        # after construction; None = disabled.
        self.recent_buffer: Any | None = None

        self._board_size = resolved.board_size   # canvas geometry
        self._trunk_size = resolved.trunk_size   # per-cluster NN-input geometry
        self._feat_len = dims.feat_len
        self._chain_len = dims.chain_len
        self._pol_len = dims.pol_len

        self._log_investigation_metrics = hp.log_investigation_metrics
        self._instrumentation = PoolInstrumentation(
            log_investigation_metrics=hp.log_investigation_metrics,
            # The n_components bound is per-ENCODING and was a module literal 5. It is
            # resolved from the same spec the drain dispatches on.
            cluster_threshold=(
                spec.cluster_threshold if spec.cluster_threshold is not None
                else DEFAULT_CLUSTER_THRESHOLD
            ),
        )

    # ── read surface ────────────────────────────────────────────────────────────
    @property
    def batch_fill_pct(self) -> float:
        return _batch_fill_pct(self)

    @property
    def inference_batch_timing(self) -> dict[str, Any]:
        """The inference server's batching instrument (queue wait / collate / occupancy) — the
        companion to `batch_fill_pct`, which says how full the batches were on average where
        this says what the distribution was and what each batch waited for."""
        return _inference_batch_timing(self)

    @property
    def x_winrate(self) -> float:
        with self._lock:
            total = self.games_completed
            return (self.x_wins / total) if total > 0 else 0.0

    @property
    def o_winrate(self) -> float:
        with self._lock:
            total = self.games_completed
            return (self.o_wins / total) if total > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """The pool's draw SHARE — `draws / games_completed`, a fraction in [0, 1].

        `iteration_complete` used to build this by hand, pairing a LIVE numerator with a
        coordinator snapshot frozen while the feeder kept draining; on the shakedown burn that
        emitted 1.3333, 1.5 and 1.125 — values a fraction cannot take. Reading both counters
        under `_lock` off the SAME update is what makes it a fraction. NOT the armed abort's
        statistic, which is a `Sum/Sum` that cannot exceed 1; this property is telemetry.
        """
        with self._lock:
            total = self.games_completed
            return (self.draws / total) if total > 0 else 0.0

    @property
    def sims_per_sec(self) -> float | None:
        """Simulations per second over the last drain interval, or `None` before the first
        interval that measured one (`docs/contracts/event_manifest.md`'s unproduced-field
        convention)."""
        return self._sims_per_sec

    @property
    def search_kind(self) -> str:
        """The run's `search.kind`, as its config spelling, read from the LIVE config rather than
        the frozen ctor-time hparams: the PUCT-only diagnostics are meaningless under Gumbel-root
        sampling, so the emitter must see a config flipped after construction.

        Raises:
            MissingSearchKindError: no `search.kind`, or one this build does not implement. NOT
                defaulted: a pool that cannot say which search it ran must not answer "puct".
        """
        return resolve_search_kind(self.config)

    @property
    def avg_game_length(self) -> float | None:
        """Mean completed-game length, or `None` before any game has completed."""
        return self._avg_game_length

    @property
    def recent_move_histories(self) -> list[list[tuple[int, int]]]:
        """Snapshot of the last ≤100 self-play move histories (thread-safe copy)."""
        return self._instrumentation.recent_move_histories(self._lock)

    def runner_stats(self) -> RunnerStats:
        """Read-only snapshot of the Rust runner's counters / scalars."""
        return _runner_stats(self)

    def inference_stats(self) -> InferenceStats:
        """Read-only snapshot of the inference server's counters + bound spec."""
        return _inference_stats(self)

    def current_stride5_p90(self) -> int:
        """Rolling P90 of stride5_run over the last ≤50 games."""
        return self._instrumentation.current_stride5_p90(self._lock)

    def pooled_draw_counts(self) -> tuple[int, int]:
        """`(Sum(draws), Sum(completed))` over the union of the per-worker draw windows. Raw
        counts, no parameters: the evidence bar is applied at the abort DECISION, so this path
        holds no second default over the operator's pre-registered value."""
        return self._instrumentation.pooled_draw_counts(self._lock)

    def terminal_reason_counts(self) -> dict[str, int]:
        """Cumulative terminal-reason counts since pool start, for the four KNOWN reason codes
        only — a code outside that set is counted internally but never surfaced, so the total
        under-counts games whenever an unknown code appears. Frozen behaviour, pinned
        deliberately, and `buffer_composition` inherits the same under-count."""
        return self._instrumentation.terminal_reason_counts(self._lock)

    def model_version_summary(self) -> dict[str, Any]:
        """Distribution stats over per-game model-version ranges."""
        return self._instrumentation.model_version_summary(self._lock)

    def buffer_composition(self) -> dict[str, float]:
        """Composition snapshot of the live replay buffer."""
        return _buffer_composition(self)

    # ── actor-sync / recorder seam ───────────────────────────────────────────────
    def sync_inference_weights(self, state_dict: dict[str, Any]) -> None:
        """Forward a promoted state_dict to the bound inference server."""
        _sync_inference_weights(self, state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        """Forward the current training step to the game recorder."""
        _update_checkpoint_step(self, step)

    def latest_replay_path(self) -> Path | None:
        """Most recent recorded self-play replay file, or `None` under the default."""
        return _latest_replay_path(self)

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def check_producer_health(self) -> None:
        """Fail-fast hook the trainer calls every step. The buffer feeder is the SOLE producer,
        so a death by exception re-raises LOUD rather than letting training run on a stale
        buffer; a clean `stop()` leaves `_producer_exc` as None, so there is no false abort."""
        if self._producer_exc is not None:
            raise RuntimeError(
                "self-play buffer feeder died — training cannot continue on a "
                "stale buffer (see the selfplay_producer_died log for the cause)"
            ) from self._producer_exc
        # R347(e): a check-14 failure found after its batch was served is run-fatal on the
        # NEXT step whether or not another pop ever arrives to carry it to the runner's latch.
        server = getattr(self, "_inference_server", None)
        deferred = getattr(server, "deferred_contract_failure", None)
        if deferred is not None:
            raise RuntimeError(
                "graph wire contract failed on a served batch (checker thread, F-816-37 "
                "dump written) — the run halts rather than train on it"
            ) from deferred
        # `guard_worker` catches a worker panic, counts it and halts the runner — and nothing in
        # Python read the count, so the failure presented as a healthy pool draining nothing until
        # the 1800 s stall timeout noticed. `worker_panics` and NOT `is_running()`, because
        # `running` also goes false on a CLEAN stop; `getattr` because a runner double or an older
        # wheel may not expose it, and an absent instrument must read as no measurement. It is a
        # METHOD on the engine runner and a plain int FIELD on `RunnerStats`, so accepting either
        # shape is not laxity — picking one would make this check pass vacuously against half its
        # callers.
        raw: Any = getattr(self._runner, "worker_panics", 0)
        panics = int(raw() if callable(raw) else raw)  # pyright: ignore[reportArgumentType]
        if panics > 0:
            raise RuntimeError(
                f"{panics} self-play worker thread(s) died by panic — the runner has halted "
                "and the replay buffer is no longer being fed. Training on a ring nothing "
                "fills produces a curve that looks like convergence; the run stops here "
                "instead (see the worker's own panic message on stderr)."
            )

    def _stats_loop(self) -> None:
        """Guard wrapper around the drain loop — see :meth:`check_producer_health`. An
        unguarded raise kills the daemon silently and leaves training on a stale buffer, so the
        exception is logged LOUD, stored, and re-raised with its cause attached."""
        try:
            run_stats_loop(self)
        except Exception as exc:  # noqa: BLE001 — sole-producer watchdog
            self._producer_exc = exc
            _LOG.error("selfplay_producer_died", exc_info=True)

    def start(self) -> None:
        """Start the inference server, the Rust runner and the feeder thread.

        Idempotent while the runner is already running.
        """
        if self._runner.is_running():
            return

        self._stop_event.clear()
        self.model.eval()

        # Lifecycle events go through the injected selfplay-local `EventSink`; no sink means
        # dropped, which is the declared default rather than a silent failure.
        if self._sink is not None:
            self._sink.emit({
                "event": "runner_started",
                "n_workers": self.n_workers,
                "encoding": self.encoding_spec.name,
            })

        self._inference_server.start()
        self._runner.start()

        # The Rust spawn loop (`crates/mantis-selfplay/src/runner/spawn.rs:67-79`) is
        # synchronous, so by the time `_runner.start()` returns the N threads are spawned.
        if self._sink is not None:
            self._sink.emit({
                "event": "workers_spawned",
                "n_workers": self.n_workers,
            })

        self._stats_thread = threading.Thread(
            target=self._stats_loop,
            daemon=True,
            name="selfplay-stats",
        )
        self._stats_thread.start()

        _LOG.info("worker_pool_started: n_workers=%s", self.n_workers)

    def stop(self) -> None:
        """Stop the runner, the inference server, the feeder thread and the recorder."""
        self._stop_event.set()
        self._runner.stop()
        self._inference_server.stop()
        self._inference_server.join(timeout=5.0)

        if self._stats_thread is not None:
            self._stats_thread.join(timeout=5.0)
            self._stats_thread = None

        self._recorder.stop()


__all__ = ["WorkerPool"]
