"""The self-play worker pool: construction, thread lifecycle, and the read surface.

>300 justify: ONE object. The trainer duck-types the pool as a single collaborator and
reads ~20 members off it, so the class cannot be split without breaking that contract; what
IS split is the implementation — the drain loop (`pool_drain`), the buffer push arms
(`pool_push`) and the hook/snapshot surface (`pool_hooks`) are free functions over the pool
instance, and this module holds only the constructor, `start`/`stop`, and thin delegators.
The delegator block is long because the surface is wide, not because it does much.

Concurrency is Rust-owned: worker threads live inside the runner. Python contributes two
threads — the inference server and the stats feeder — and the feeder is the SOLE producer
of training data, so its death is fatal and reported through `check_producer_health`.

Every knob is resolved ONCE at construction through `hparams` (there is no config read in
the hot loop), with two deliberate exceptions that re-read the LIVE config because the old
behaviour did and callers depend on it: the `gumbel_mcts` property and
`buffer_composition`'s independent draw/ply-cap resolution.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from mantis._engine import SelfPlayRunner
from mantis.selfplay.buffers import ReplayFacade
from mantis.selfplay.hparams import (
    SelfPlayHParams,
    _load_seed_corpus,
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
from mantis.selfplay.pool_hooks import inference_stats as _inference_stats
from mantis.selfplay.pool_hooks import latest_replay_path as _latest_replay_path
from mantis.selfplay.pool_hooks import runner_stats as _runner_stats
from mantis.selfplay.pool_hooks import (
    sync_inference_weights as _sync_inference_weights,
)
from mantis.selfplay.pool_hooks import update_checkpoint_step as _update_checkpoint_step
from mantis.selfplay.pool_push import buffer_composition as _buffer_composition

_LOG = logging.getLogger(__name__)


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

        `arch` is REQUIRED (pass `None` to state explicitly that there is nothing to
        cross-check): the resolved encoding's canvas geometry is checked against the
        arch's DECLARED `board_size`, so a mis-paired arch and config fail before any
        Rust runner exists. Nothing is sniffed off the live module — a graph arch
        declares no board size and passes vacuously, which is the frozen behaviour.

        `sink` / `recorder` / `heartbeat` are injected collaborators with no-op defaults:
        events are dropped, nothing is recorded, no heartbeat fires. Each default is a
        declared seam, not a silent failure.
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

        # The pool takes a RAW engine buffer and wraps it: the facade resolves the kind
        # from the SAME spec the drain dispatches on and cross-checks the handle, so a
        # graph buffer under a grid encoding (or the inverse) dies here rather than
        # producing corrupt training data. Everything downstream pushes through
        # `self.replay_buffer`, so the guard cannot be bypassed.
        self.replay_buffer = ReplayFacade(spec, replay_buffer)

        seed_prefixes = _load_seed_corpus(hp.seed_corpus_path, hp.seed_fraction)
        sp_config, dims = build_runner_config(
            hp,
            spec_dims=resolved,
            encoding_name=resolved.encoding_name,
            seed_prefixes=seed_prefixes,
        )
        self._runner = SelfPlayRunner(sp_config)
        self._inference_server = InferenceServer(
            model, device, config,
            batcher=self._runner.batcher,
            encoding_spec=spec,
            heartbeat=heartbeat,
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
        self._sims_per_sec: float = 0.0
        self._last_drain_time: float = time.monotonic()
        # Last-seen runner `positions_generated`; the per-drain delta is what the
        # sims/sec bill multiplies by the effective per-move sim count.
        self._last_pos_generated: int = 0
        self._total_sims: int = 0
        self._game_lengths: deque[int] = deque(maxlen=200)
        self._avg_game_length: float = 0.0

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
        self._instrumentation_enabled = hp.instrumentation_enabled
        self._instrumentation = PoolInstrumentation(
            log_investigation_metrics=hp.log_investigation_metrics,
        )

    # ── read surface ────────────────────────────────────────────────────────────
    @property
    def batch_fill_pct(self) -> float:
        return _batch_fill_pct(self)

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
    def sims_per_sec(self) -> float:
        return self._sims_per_sec

    @property
    def gumbel_mcts(self) -> bool:
        """Whether Gumbel-root MCTS is active.

        Read from the LIVE config, not from the frozen ctor-time hparams: the PUCT-only
        diagnostics are descent-rule-specific and meaningless under Gumbel-root sampling,
        so the event emitter suppresses them when this is True — and it must see a config
        flipped after construction.
        """
        sp = self.config.get("selfplay", self.config)
        return bool(sp.get("gumbel_mcts", False))

    @property
    def avg_game_length(self) -> float:
        return self._avg_game_length

    @property
    def recent_move_histories(self) -> list[list[tuple[int, int]]]:
        """Snapshot of the last ≤100 self-play move histories (thread-safe copy)."""
        return self._instrumentation.recent_move_histories(self._lock)

    @property
    def instrumentation_enabled(self) -> bool:
        return self._instrumentation_enabled

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
        """`(Sum(draws), Sum(completed))` over the union of the per-worker draw windows.

        WPMINT Phase DS (R92): raw counts, no parameters. The evidence bar
        (`train.draw_rate_abort.N_pool_min`) is applied at the abort DECISION, not here —
        this path carries no config authority at all now, which is one fewer layer that
        could hold a second default over the operator's pre-registered value (R1).
        """
        return self._instrumentation.pooled_draw_counts(self._lock)

    def terminal_reason_counts(self) -> dict[str, int]:
        """Cumulative terminal-reason counts since pool start.

        Reports the four KNOWN reason codes only. A code outside that set is counted
        internally but never surfaced here, so the total under-counts games whenever an
        unknown code appears. That is the frozen behaviour and it is pinned deliberately:
        `buffer_composition`'s `n_games_observed` inherits the same under-count.
        """
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
        """Fail-fast hook the trainer calls every step.

        The buffer feeder is the SOLE producer draining Rust self-play results into the
        replay buffer. If it died on an exception, re-raise LOUD so training aborts
        instead of silently running on a stale buffer. A clean `stop()` leaves
        `_producer_exc` as None — no false abort.
        """
        if self._producer_exc is not None:
            raise RuntimeError(
                "self-play buffer feeder died — training cannot continue on a "
                "stale buffer (see the selfplay_producer_died log for the cause)"
            ) from self._producer_exc

    def _stats_loop(self) -> None:
        """Guard wrapper around the drain loop — see :meth:`check_producer_health`.

        The feeder is the sole producer; an unguarded raise kills the daemon silently and
        leaves training on a stale buffer. Catch, log LOUD at error level, and flag
        producer death so the trainer fails fast on its next step. Nothing is swallowed:
        the exception is stored and re-raised with its cause attached.
        """
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

        self._inference_server.start()
        self._runner.start()

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
