"""StepCoordinator.step() — the per-step outer-loop core (WP10 §a.4 split — `step` slice).

>300 justify: `step()` + `run_until_stopped()` reproduce one outer iteration of the old
`loop.py::_run_loop` closure (warmup tick / waiting-for-games tick / training burst) — one
cohesive control-flow unit, kept together. Behaviour-exact on the reachable seams, routed
through the injected collaborators (`config.py` Protocols); the stall watchdog is driven via
the Slice-1 `lifecycle.watchdog.StallWatchdog.tick(...)` (the slice-2 wiring the DESIGN calls
for). The terminal-eval flush + close_out live in `drain.py`.

Severances (must not re-enter): the `bot_refresh` subprocess family (`_tick_bot_refresh`,
force-refresh sentinel) is a DEFINITE KILL — NOT ported. The `track_b_*` snapshot/attribution
call-sites (F-22..F-33 KILL) are severed. The display/instrumentation/tracemalloc half (perf
probes, dashboard renderers, value-probe cadence) DEFERS→WP13; events route through the
injected `EventSink`. The stride5-spam / draw-rate sustained-collapse hard-abort gates DEFER→WP13
(they consume monitoring-aggregated pool signals + the `monitors.*` config block that WP13 owns);
the cheap loss-info hard-GN abort is kept (run-safety, reads the trainer's own grad_norm).
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable

from mantis.train.buffer_persist import try_save_buffer as _try_save_buffer
from mantis.train.coordinator.config import (
    ClockLike,
    RealClock,
    StepCoordinatorConfig,
    StepOutcome,
)
from mantis.train.emit import emit_via
from mantis.train.lifecycle.watchdog import StallWatchdog, watchdog_snapshot_path
from mantis.train.mixing import _compute_pretrained_weight, _steps_budget

_LOG = logging.getLogger(__name__)


class StepCoordinator:
    """Owns the per-step mutable state extracted from the old `loop.py::_run_loop`.

    One ``step()`` equals one outer iteration; it returns a :class:`StepOutcome` describing
    every decision. The caller (`run_training_loop`) installs signal handlers against the same
    ``ShutdownState`` this coordinator holds and calls ``close_out()`` (drain.py) at teardown.
    """

    def __init__(
        self,
        *,
        trainer: Any,
        buffer: Any,
        pretrained_buffer: Any | None,
        recent_buffer: Any | None,
        pool: Any,
        eval_pipeline: Any | None,
        subsystems: Any,
        anchor_state: Any,
        shutdown: Any,
        eval_model: Any,
        bufs: Any,
        config: StepCoordinatorConfig,
        full_config: dict[str, Any] | None = None,
        train_cfg: dict[str, Any] | None = None,
        mixing_cfg: dict[str, Any] | None = None,
        batch_size_cfg: int | None = None,
        iterations: int | None = None,
        run_id: str | None = None,
        clock: ClockLike | None = None,
        sink: Any = None,
        bot_buffer: Any | None = None,
        exit_fn: Callable[[int], None] = os._exit,
    ) -> None:
        self.trainer = trainer
        self.buffer = buffer
        self.pretrained_buffer = pretrained_buffer
        self.bot_buffer = bot_buffer
        self.recent_buffer = recent_buffer
        self.pool = pool
        self.eval_pipeline = eval_pipeline
        self.subsystems = subsystems
        self.anchor_state = anchor_state
        self.shutdown = shutdown
        self.eval_model = eval_model
        self.bufs = bufs
        self.config = config
        self.full_config = full_config or {}
        self.train_cfg = train_cfg or {}
        self.mixing_cfg = mixing_cfg or {}
        self.batch_size_cfg = batch_size_cfg
        self.iterations = iterations
        self.run_id = run_id
        self._clock = clock or RealClock()
        self._sink = sink
        self._exit_fn = exit_fn

        # Per-step mutable bookkeeping.
        self._train_step = int(getattr(trainer, "step", 0))
        self._games_played = 0
        self.last_train_game_count = 0
        self._schedule_idx = 0
        self.last_warmup_log = 0.0
        self._last_loss_info: dict[str, float] | None = None
        self._initial_policy_loss: float | None = None
        self._consec_high_gn = 0
        self._eval_round_last_step = -1

        # Self-play stall watchdog — always armed (context law, LAW-16). Driven via
        # `.tick(...)` from step(); fires → best-effort snapshot to a DISTINCT path + exit.
        bp = self.mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin")
        self._watchdog = StallWatchdog(
            timeout_sec=config.selfplay_stall_timeout_sec,
            clock=self._clock.now,
            sink=sink,
            exit_fn=exit_fn,
            save_snapshot=lambda: self._snapshot_buffer(watchdog_snapshot_path(bp)),
        )
        self._watchdog.arm(getattr(pool, "games_completed", 0))

    # ── watchdog snapshot (distinct path; never the canonical resume buffer) ──────────────
    def _snapshot_buffer(self, path: Any) -> None:
        saver = getattr(self.buffer, "save_to_path", None)
        if saver is not None:
            saver(str(path))

    # ── outcome builder ───────────────────────────────────────────────────────────────────
    def _build_outcome(self, **kw: Any) -> StepOutcome:
        return StepOutcome(
            train_step=self._train_step,
            games_played=self._games_played,
            consec_high_gn=self._consec_high_gn,
            last_loss_info=self._last_loss_info,
            games_per_hour=0.0,
            **kw,
        )

    def stop(self, reason: str) -> None:
        _LOG.info("stop_requested reason=%s", reason)
        self.shutdown.running = False

    def run_until_stopped(self) -> None:
        """Production entry — drive ``step()`` until shutdown (mirrors the old
        ``while _shutdown.running`` loop)."""
        while self.shutdown.running:
            self.step()

    # ── one outer iteration ─────────────────────────────────────────────────────────────
    def step(self) -> StepOutcome:
        """Run exactly one outer iteration; return a :class:`StepOutcome`."""
        cfg = self.config

        # F02 fail-fast: the self-play feeder is the sole producer — abort loudly if it died.
        health = getattr(self.pool, "check_producer_health", None)
        if health is not None:
            health()

        base = dict(
            steps_run=0, buffer_resized=None, checkpoint_saved=False, axis_emitted=False,
            eval_kicked_off=False, eval_skipped_busy=False, eval_drained=False,
            promoted_step=None, soft_abort_fired=False, hard_abort_fired=False,
            instrumentation_emitted=[], pool_overflow_delta=0,
        )

        # O2: iteration-limit reached.
        if cfg.stop_step is not None and self._train_step >= cfg.stop_step:
            self.shutdown.running = False
            return self._build_outcome(in_warmup=False, waiting_for_games=False, **base)

        # O3: shutdown-save (signal-handler flag) — save + buffer save, then stop.
        if self.shutdown.shutdown_save:
            self.trainer.save_checkpoint(self._last_loss_info or None)
            _try_save_buffer(self.buffer, self.mixing_cfg, "shutdown_signal", self.recent_buffer)
            self.shutdown.running = False
            return self._build_outcome(in_warmup=False, waiting_for_games=False,
                                       **{**base, "checkpoint_saved": True})

        self._games_played = int(getattr(self.pool, "games_completed", 0))
        # Stall watchdog — driven via tick(...) (slice-2 wiring; behaviour byte-identical).
        self._watchdog.tick(self._games_played, self._clock.now())

        # O4: warmup — buffer below the training floor.
        if self.buffer.size < cfg.min_buf_size:
            if (self._clock.now() - self.last_warmup_log) >= 5.0:
                emit_via(self._sink, {"event": "system_stats", "buffer_size": self.buffer.size,
                                      "buffer_capacity": cfg.capacity})
                self.last_warmup_log = self._clock.now()
            self._clock.sleep(0.5)
            return self._build_outcome(in_warmup=True, waiting_for_games=False, **base)

        # O5: no new games since the last burst.
        new_games = self._games_played - self.last_train_game_count
        if new_games <= 0:
            self._clock.sleep(0.1)
            return self._build_outcome(in_warmup=False, waiting_for_games=True, **base)

        # O6: compute the training-step budget + advance bookkeeping.
        steps_budget = _steps_budget(new_games, cfg.training_steps_per_game, cfg.max_train_burst)
        self.last_train_game_count = self._games_played

        loss_info: dict[str, float] = {}
        buffer_resized: int | None = None
        checkpoint_saved = False
        hard_abort_fired = False

        for _ in range(steps_budget):
            if cfg.stop_step is not None and self._train_step >= cfg.stop_step:
                break
            # D1: buffer growth schedule.
            while (self._schedule_idx < len(cfg.buffer_schedule)
                   and self._train_step >= cfg.buffer_schedule[self._schedule_idx]["step"]):
                new_cap = cfg.buffer_schedule[self._schedule_idx]["capacity"]
                if new_cap > self.buffer.capacity:
                    self.buffer.resize(new_cap)
                    buffer_resized = new_cap
                self._schedule_idx += 1

            # D2: training step — mixed (corpus + selfplay) when a pretrained buffer is present,
            # else a straight self-play step. Both route through the injected trainer.
            loss_info = self._run_training_step(cfg)
            self._train_step = self.trainer.step
            if self._initial_policy_loss is None and "policy_loss" in loss_info:
                self._initial_policy_loss = float(loss_info["policy_loss"])
            self._last_loss_info = loss_info

            # D3: hard-abort on sustained gradient norm (run-safety; reads the trainer's own gn).
            step_gn = float(loss_info.get("grad_norm", 0.0))
            if math.isfinite(step_gn) and step_gn > cfg.hard_gn_threshold:
                self._consec_high_gn += 1
                if self._consec_high_gn >= cfg.hard_gn_min_steps:
                    _LOG.error("hard_abort_grad_norm step=%s consec=%s gn=%.4f",
                               self._train_step, self._consec_high_gn, step_gn)
                    self.shutdown.running = False
                    hard_abort_fired = True
            else:
                self._consec_high_gn = 0

            # D4: checkpoint-cadence buffer save (the trainer saves its own ckpt inside the step).
            if cfg.checkpoint_interval > 0 and self._train_step > 0 \
                    and self._train_step % cfg.checkpoint_interval == 0:
                _try_save_buffer(self.buffer, self.mixing_cfg, "checkpoint_interval",
                                 self.recent_buffer)
                checkpoint_saved = True

        eval_kicked_off = self._maybe_kick_eval(cfg)
        return self._build_outcome(
            in_warmup=False, waiting_for_games=False,
            **{**base, "steps_run": steps_budget, "buffer_resized": buffer_resized,
               "checkpoint_saved": checkpoint_saved, "eval_kicked_off": eval_kicked_off,
               "hard_abort_fired": hard_abort_fired},
        )

    # ── training-step dispatch (mixed vs straight self-play) ──────────────────────────────
    def _run_training_step(self, cfg: StepCoordinatorConfig) -> dict[str, float]:
        batch_size = int(self.train_cfg.get("batch_size", self.full_config.get("batch_size", 256)))
        if (self.pretrained_buffer is not None and self.pretrained_buffer.size > 0
                and self.buffer.size > 0):
            from mantis.train.batch_assembly import assemble_mixed_batch  # lazy (heavy deps)

            w_pre = _compute_pretrained_weight(self._train_step, cfg.mixing_initial_w,
                                               cfg.mixing_min_w, cfg.mixing_decay_steps)
            n_bot = (round(cfg.bot_batch_share * batch_size)
                     if (self.bot_buffer is not None and self.bot_buffer.size > 0) else 0)
            n_pre = max(1, int(math.ceil(w_pre * (batch_size - n_bot))))
            n_self = batch_size - n_pre - n_bot
            batch = assemble_mixed_batch(
                self.pretrained_buffer, self.buffer, self.recent_buffer,
                n_pre, n_self, batch_size, self.batch_size_cfg, cfg.recency_weight,
                self.bufs, self._train_step, augment=cfg.augment,
                bot_buffer=self.bot_buffer, n_bot=n_bot,
            )
            return self.trainer.train_step_from_tensors(
                batch.states, batch.policies, batch.outcomes,
                chain_planes=batch.chain_planes, ownership_targets=batch.ownership,
                threat_targets=batch.winning_line, is_full_search=batch.is_full_search,
                n_pretrain=n_pre + n_bot, n_recent=batch.n_recent_actual,
                position_indices=batch.position_indices,
                value_target_valid=batch.value_target_valid,
            )
        # straight self-play step (the injected buffer owns sampling; a graph HexgBuffer routes
        # to the trainer's graph path inside train_step).
        return self.trainer.train_step(self.buffer, augment=cfg.augment,
                                       recent_buffer=self.recent_buffer)

    # ── eval kickoff at the boundary (via the INJECTED EvalPipelineLike; no train→eval) ───
    def _maybe_kick_eval(self, cfg: StepCoordinatorConfig) -> bool:
        if self.eval_pipeline is None or cfg.eval_interval <= 0:
            return False
        round_idx = self._train_step // cfg.eval_interval
        if round_idx <= 0 or round_idx == self._eval_round_last_step:
            return False
        if self._train_step % cfg.eval_interval != 0:
            return False
        self._eval_round_last_step = round_idx
        best = getattr(self.anchor_state, "best_model", None)
        best_step = getattr(self.anchor_state, "best_model_step", None)
        self.eval_pipeline.run_evaluation(
            self.eval_model, self._train_step, best,
            full_config=self.full_config, best_model_step=best_step,
        )
        return True

    # ── close-out / terminal-eval flush (delegates to drain.py; §a.4 `drain` slice) ───────
    def flush_pending_eval(self) -> Any:
        from mantis.train.coordinator import drain
        return drain.flush_pending_eval(self)

    def run_terminal_eval(self) -> Any:
        from mantis.train.coordinator import drain
        return drain.run_terminal_eval(self)

    def close_out(self, on_drained: "Callable[[], None] | None" = None) -> None:
        from mantis.train.coordinator import drain
        drain.close_out(self, on_drained=on_drained)
