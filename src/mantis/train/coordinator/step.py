"""StepCoordinator.step() — the per-step outer-loop core (WP10 §a.4 split — `step` slice).

>300 justify: `step()` + `run_until_stopped()` reproduce one outer iteration of the old
`loop.py::_run_loop` closure (warmup tick / waiting-for-games tick / training burst) — one
cohesive control-flow unit, kept together. Behaviour-exact on the reachable seams, routed
through the injected collaborators (`config.py` Protocols); the stall watchdog is driven via
the Slice-1 `lifecycle.watchdog.StallWatchdog.tick(...)` (the slice-2 wiring the DESIGN calls
for). The terminal-eval flush + close_out live in `drain.py`. WP13-A adds the run-safety
instrumentation half to the SAME control-flow unit (it is one loop, not two): the
`train_step` heartbeat beats, the log_interval event emission + the 4 WARN rules, the
draw-rate hard-abort gate, the LAW-18 `monitor_gates` summary, and the async eval-RESULT
consumer `on_eval_round_complete` (warn-only sealbot by default, operator G-3) — all in this
file so the loop's decision trail stays readable end to end. (The stride5-spam gate was
REMOVED at close-out, operator directive B.)

Severances (must not re-enter): the `bot_refresh` subprocess family (`_tick_bot_refresh`,
force-refresh sentinel) is a DEFINITE KILL — NOT ported. The `track_b_*` snapshot/attribution
call-sites (F-22..F-33 KILL) are severed. The DISPLAY half (dashboard renderers, TB writer),
the perf/tracemalloc probes and the value-probe cadence stay DEFER/ARCH; events route through
the injected `EventSink`. The probe-as-gate class (early-game probe, value-spread canary) is
FALSIFIED (F-27/F-30) and never becomes a run gate here.

L-B/L-A discipline: `step()` NEVER blocks on eval and NEVER reads the eval KICK return for
WR — a blocking drain in the step path is the run3 wedge class the independent heartbeat
watchdog exists to kill. Completed eval ROUNDS reach the sealbot-WR gate only through
`on_eval_round_complete` (WP11-A routes them mid-run; `drain.py` routes them at teardown).
"""
from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping
from typing import Any, cast

from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import (
    check_draw_rate_collapse,
    check_sealbot_wr_hard_abort,
    emit_training_step_alerts,
    sealbot_wr_trajectory_alert,
)
from mantis.train.buffer_persist import try_save_buffer as _try_save_buffer
from mantis.train.coordinator.config import (
    ClockLike,
    RealClock,
    StepCoordinatorConfig,
    StepOutcome,
    pooled_draw_rate,
)
from mantis.train.coordinator.dispatch import (
    RepresentationRouteError,
    resolve_step_spec,
    run_declared_train_step,
)
from mantis.train.emit import NullEventSink, emit_via
from mantis.train.events import emit_axis_distribution, emit_training_events
from mantis.train.lifecycle.watchdog import StallWatchdog, watchdog_snapshot_path
from mantis.train.mixing import _compute_pretrained_weight, _steps_budget

_LOG = logging.getLogger(__name__)

#: The gate keys carried by the LAW-18 `monitor_gates` summary (checks/fires/skips/warns).
#: The KEPT WP10 grad-norm abort is in the list so the one hard-abort that is unconditionally
#: ACTIVE at landing is visible in the ONE channel like the WP13-A gates. `sealbot_wr_abort`
#: ships WARN-ONLY (operator G-3) and `draw_rate_collapse` is armed BY THE CONFIG
#: (`train.draw_rate_abort`; `null` is the explicit off posture — WPAX Phase D, R65/R80) —
#: both named here so their inert/warn posture is readable, never silent. (`stride5_spam` was
#: REMOVED at close-out, operator directive B.)
GATE_NAMES: tuple[str, ...] = (
    "draw_rate_collapse", "sealbot_wr_abort", "grad_norm_hard_abort",
)

#: Depth of the sealbot-WR ring (old `step_coordinator.py` parity: `pop(0)` past 5).
WR_HISTORY_DEPTH = 5
#: Depth of the pool-signal rings the two live-producer gates slide over.
_GATE_HISTORY_DEPTH = 32


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
        heartbeat: Callable[[str], None] | None = None,
        monitor_cfg: Any = None,
        heartbeat_watchdog: Any = None,
        actor_sync: Any = None,
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
        # WP13-A run-safety seams: the heartbeat fn (the watchdog's `train_step` source),
        # the monitor thresholds and the independent watchdog `close_out` disarms.
        self._heartbeat = heartbeat
        self.monitor_cfg = monitor_cfg if monitor_cfg is not None else MonitorConfig()
        self.heartbeat_watchdog = heartbeat_watchdog
        # WP-UNFREEZE: the continuous actor-sync engine (mantis.train.actor_sync).
        # None is a unit-test affordance ONLY (like `eval_pipeline=None`); production
        # wiring is unconditional at the ONE composition root, pinned by
        # tests/train/test_actor_sync_isolation.py.
        self.actor_sync = actor_sync

        # Per-step mutable bookkeeping.
        self._train_step = int(getattr(trainer, "step", 0))
        self._games_played = 0
        self.last_train_game_count = 0
        self._schedule_idx = 0
        self.last_warmup_log = 0.0
        self._last_loss_info: dict[str, float] | None = None
        # WPTS Phase T: the resolved encoding spec (lazy, once) — the straight arm's typed
        # route dispatches off spec.representation, resolved from the DECLARED config this
        # coordinator already holds, through THE one resolver (never a buffer sniff).
        self._resolved_step_spec: Any | None = None
        self._initial_policy_loss: float | None = None
        self._consec_high_gn = 0
        self._eval_round_last_step = -1

        # WP13-A gate state — every ring is caller-owned (the rules are stateless).
        self._wr_history: list[tuple[int, float]] = []
        self._draw_rate_history: list[float] = []
        self._loss_window: list[float] = []
        self._last_iter_games = 0
        self._run_started = self._clock.now()
        # `warns` is carried alongside checks/fires/skips so the warn-only sealbot posture
        # (operator G-3) is visible per-gate in every `monitor_gates` event, not silent.
        self._gate_stats: dict[str, dict[str, int]] = {
            name: {"checks": 0, "fires": 0, "skips": 0, "warns": 0} for name in GATE_NAMES
        }

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

        # L-B: the outer loop is alive. Beaten at ENTRY (so every early-return branch —
        # warmup, waiting-for-games — still proves liveness) and once per burst iteration.
        self._beat("train_step")

        # F02 fail-fast: the self-play feeder is the sole producer — abort loudly if it died.
        health = getattr(self.pool, "check_producer_health", None)
        if health is not None:
            health()

        # WP11-A: non-blocking eval-result poll at the TOP of every iteration — main-thread
        # routing through drain._route_eval_result, on every branch (warmup/waiting-for-
        # games included), never a blocking call (never `drain_pending()`; §c.4b).
        eval_drained = self._poll_eval_results()

        base = dict(
            steps_run=0, buffer_resized=None, checkpoint_saved=False, axis_emitted=False,
            eval_kicked_off=False, eval_skipped_busy=False, eval_drained=eval_drained,
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
        axis_emitted = False

        for _ in range(steps_budget):
            self._beat("train_step")
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
            # D2b (WP-UNFREEZE): continuous actor weight sync — per inner step, the
            # house cadence pattern (D4, log_interval); `_train_step` advances by
            # exactly 1 per burst iteration so a modulo boundary can never be skipped.
            if self.actor_sync is not None:
                self.actor_sync.maybe_sync(self._train_step)
            if self._initial_policy_loss is None and "policy_loss" in loss_info:
                self._initial_policy_loss = float(loss_info["policy_loss"])
            self._last_loss_info = loss_info

            # D3: hard-abort on sustained gradient norm (run-safety; reads the trainer's own gn).
            # WP13-A routes the FIRE through the shared `_fire_hard_abort` contract so this
            # gate is visible in the ONE channel (a `hard_abort` event + `monitor_gates`)
            # like every other gate; the DECISION (threshold, consecutive count, reset) is
            # byte-identical to WP10.
            self._gate_stats["grad_norm_hard_abort"]["checks"] += 1
            step_gn = float(loss_info.get("grad_norm", 0.0))
            if math.isfinite(step_gn) and step_gn > cfg.hard_gn_threshold:
                self._consec_high_gn += 1
                if self._consec_high_gn >= cfg.hard_gn_min_steps:
                    _LOG.error("hard_abort_grad_norm step=%s consec=%s gn=%.4f",
                               self._train_step, self._consec_high_gn, step_gn)
                    hard_abort_fired = self._fire_hard_abort(
                        "grad_norm_hard_abort",
                        f"HARD-ABORT (grad-norm): grad_norm {step_gn:.4f} > "
                        f"{cfg.hard_gn_threshold:.4f} for {self._consec_high_gn} consecutive "
                        f"training steps — optimizer instability",
                    ) or hard_abort_fired
            else:
                self._consec_high_gn = 0

            # D4: checkpoint-cadence buffer save (the trainer saves its own ckpt inside the step).
            if cfg.checkpoint_interval > 0 and self._train_step > 0 \
                    and self._train_step % cfg.checkpoint_interval == 0:
                _try_save_buffer(self.buffer, self.mixing_cfg, "checkpoint_interval",
                                 self.recent_buffer)
                checkpoint_saved = True

            # WP13-A: the log_interval boundary is tested PER TRAINING STEP (old-side parity,
            # `step_coordinator.py:1370/1383`). Testing it once per burst would skip every
            # boundary the post-burst step does not land exactly on, thinning BOTH the LAW-18
            # emission stream and the sampling cadence of the draw-rate gate by ~the mean
            # burst — the gate's `consec` window would silently stretch by that factor.
            axis_step, gate_fired = self._run_log_interval(cfg, loss_info)
            axis_emitted = axis_emitted or axis_step
            hard_abort_fired = hard_abort_fired or gate_fired

        # The eval KICK return is NEVER consumed for WR and `step()` adds NO blocking call:
        # completed rounds reach `on_eval_round_complete` via the async drain (§c.4b).
        # `_maybe_kick_eval` consumes the ACK only for `eval_skipped_busy` (WP13-A P-06 pin).
        eval_kicked_off, eval_skipped_busy = self._maybe_kick_eval(cfg)
        return self._build_outcome(
            in_warmup=False, waiting_for_games=False,
            **{**base, "steps_run": steps_budget, "buffer_resized": buffer_resized,
               "checkpoint_saved": checkpoint_saved, "eval_kicked_off": eval_kicked_off,
               "eval_skipped_busy": eval_skipped_busy,
               "hard_abort_fired": hard_abort_fired, "axis_emitted": axis_emitted},
        )

    # ── L-B heartbeat ─────────────────────────────────────────────────────────────────
    def _beat(self, source: str) -> None:
        """Beat one heartbeat source. An unknown source raises inside the registry — a
        wiring bug must be loud, never a silently-dropped beat the watchdog can't see."""
        if self._heartbeat is not None:
            self._heartbeat(source)

    # ── WP13-A: log_interval instrumentation + the hard-abort gates ───────────────────
    def _run_log_interval(
        self, cfg: StepCoordinatorConfig, loss_info: dict[str, float]
    ) -> tuple[bool, bool]:
        """At the `log_interval` boundary: emit the run's payload events, run the 4 WARN
        rules on them, run the two LIVE-producer hard-abort gates, and publish the LAW-18
        `monitor_gates` summary. Returns ``(axis_emitted, hard_abort_fired)``."""
        if not loss_info or cfg.log_interval <= 0 or self._train_step % cfg.log_interval != 0:
            return False, False
        sink = self._sink if self._sink is not None else NullEventSink()

        payload = self._emit_training_events(cfg, loss_info, sink)
        emit_training_step_alerts(payload, self.monitor_cfg, self._loss_window, sink=sink)
        keep = max(2 * int(self.monitor_cfg.alert_loss_increase_window) + 2, 8)
        del self._loss_window[:-keep]

        axis = emit_axis_distribution(
            self._train_step, self.pool, self.monitor_cfg,
            getattr(self.subsystems, "axis_baseline", None) or {},
            getattr(self.subsystems, "tb_writer", None), sink,
        )
        fired = self._run_hard_abort_gates(cfg)
        self._emit_monitor_gates(cfg, sink)
        return axis is not None, fired

    def _emit_training_events(
        self, cfg: StepCoordinatorConfig, loss_info: dict[str, float], sink: Any
    ) -> dict[str, Any]:
        """Build + emit `training_step` and `iteration_complete` through the injected sink
        (the WP10 builders; payload shapes unchanged) and return the `training_step`
        payload the WARN rules read."""
        w_pre = 0.0
        if self.pretrained_buffer is not None:
            w_pre = _compute_pretrained_weight(self._train_step, cfg.mixing_initial_w,
                                               cfg.mixing_min_w, cfg.mixing_decay_steps)
        payload = emit_training_events(
            self._train_step, loss_info, w_pre, self._games_played, self._last_iter_games,
            self.pool, self.buffer, getattr(self.subsystems, "gpu_monitor", None),
            self.full_config, self.full_config.get("mcts", {}), cfg.capacity,
            # `quiescence_fires_per_step` has NO producer new-side (the solver-delta half is
            # DEFER/ARCH): the field travels as None = NOT MEASURED. A constant 0 would read
            # as a real measurement ("quiescence never fires") — a miniature F-10.
            self._games_per_hour, None, sink,
        )
        self._last_iter_games = self._games_played
        return payload

    def _games_per_hour(self) -> float:
        elapsed = self._clock.now() - self._run_started
        return (self._games_played / elapsed) * 3600.0 if elapsed > 0 else 0.0

    def _run_hard_abort_gates(self, cfg: StepCoordinatorConfig) -> bool:
        """The DEFER→WP13 draw-rate gate, keyed on the LIVE pool producer.

        draw-rate reads `pooled_draw_rate(pool.pooled_draw_counts(), N_pool_min=…)` — the
        POOLED COUNT-WEIGHTED rate over the union of worker windows (R92), with the evidence
        bar config-authored — never the NaN draw-target phantom at `pool_push.py:135`, whose
        very TOKEN is grep-banned here (O-15). A missing producer AND insufficient evidence
        are both SKIP-counted (LAW-18), never silently read as a healthy signal. (The
        stride5-spam gate was REMOVED at close-out, operator directive B.)
        """
        counts_fn = getattr(self.pool, "pooled_draw_counts", None)
        spec = cfg.draw_rate_abort
        # WPAX Phase D: `is not None`, NOT `> 0`. Under the type change `draw_rate_abort` is
        # `None` on every disarmed run, and `None > 0` would raise TypeError here once per
        # step() — so testing for absence is required BY the type change, not a tidy-up.
        # Both absences (EXPLICIT-off `train.draw_rate_abort: null`, and a producer that has
        # not landed) route through `_sample` with a `None` producer, which is what SKIP-
        # counts them (LAW-18) — `_sample` owns that counter and always has. WPMINT DR-1 /
        # R72: the earlier shape guarded the live path with `if draw and spec is not None`
        # plus an `elif draw:` skip arm. `_sample` returns False whenever its producer is
        # None and the producer is None exactly when `spec is None`, so `draw` implied
        # `spec is not None`: the conjunct had NO flip-set and the `elif` arm was provably
        # unreachable (its LAW-18 comment was measured false). The early return keeps `spec`
        # narrowed for the type checker without a conjunct that no input can flip.
        if spec is None or counts_fn is None:
            self._sample("draw_rate_collapse", self._draw_rate_history, None)
            return False
        # WPMINT Phase DS (R92): `_sample`'s return IS branched on now, and the conjunct DR-1
        # ruled out has become live. Pre-R92 the producer could only yield a float, so past
        # the early return `_sample` returned True unconditionally and branching on it would
        # have been a second no-flip-set conjunct (R72). `pooled_draw_rate` returns `None`
        # when `Sum(completed) < N_pool_min` — INSUFFICIENT EVIDENCE, which R92 makes a NO
        # OBSERVATION rather than a fabricated healthy 0.0 (DR-4) — so the False arm has a
        # real flip-set: an armed spec, a live producer, and a pool below the bar. Its
        # witness is `test_drawrate_gate_branch_flipset.py`'s drive B5.
        #
        # The return is required, not tidy: `consec` counts consecutive OBSERVATIONS, so
        # running the rule when nothing was appended would re-decide the abort on a stale
        # tail. `_sample` has already skip-counted this case (LAW-18, ONE skip site).
        if not self._sample(
            "draw_rate_collapse", self._draw_rate_history,
            lambda: pooled_draw_rate(counts_fn(), N_pool_min=spec.N_pool_min),
        ):
            return False
        message = check_draw_rate_collapse(self._draw_rate_history, self._train_step,
                                           threshold=spec.threshold,
                                           consec=spec.consec,
                                           min_step=spec.min_step)
        return self._fire_hard_abort("draw_rate_collapse", message)

    def _sample(self, gate: str, history: list[float], producer: Any) -> bool:
        """Append one LIVE producer sample to ``history``; False (+skip) when there is no
        observation to append.

        TWO absences, ONE counter (LAW-18). The producer itself may be absent — a disarmed
        gate, or a producer that has not landed — and a LIVE producer may return `None`,
        which under R92 means INSUFFICIENT EVIDENCE (`Sum(completed) < N_pool_min`). Both are
        skips and neither appends: an unobserved interval must never enter an abort history
        as a number, in either direction. R72: both arms have flip-sets, driven in
        `test_drawrate_gate_branch_flipset.py` (B1/B2 for the first, B5 for the second).
        """
        self._gate_stats[gate]["checks"] += 1
        if producer is None:
            self._gate_stats[gate]["skips"] += 1
            return False
        value = producer()
        if value is None:
            self._gate_stats[gate]["skips"] += 1
            return False
        history.append(float(value))
        del history[:-_GATE_HISTORY_DEPTH]
        return True

    def _fire_hard_abort(self, rule: str, message: str | None, step: int | None = None) -> bool:
        """The ONE fire contract every WP13-A gate shares: stop the run + one `hard_abort`
        event naming the rule and carrying the rule's own message.

        A gate that resolves AFTER the run already stopped (the teardown-routed eval result)
        records a DISTINCT `hard_abort_after_stop` event: the trail stays complete, but a
        stopped run is never reported as a second abort decision.

        WPMINT Phase X (CARD-ABORT-EXIT / R84): the fire also records the RULE NAME on
        `ShutdownState.abort_rule`, beside the `running = False` it was already writing. That
        is the whole of R84's "supervisor-distinguishable from a clean run" on this side — the
        three clean stops (`stop()`, O2 iteration-limit, O3 shutdown-save) leave the field
        `None`. The NAME, not a code: this method must not import `mantis.config.armed_aborts`,
        and it is shared by rules that have no authored exit code, so the rule -> code
        resolution (`armed_aborts.exit_code_for_abort`) belongs at the process boundary, not
        here. The assignment is deliberately paired with `running = False` — it must be
        impossible to stop the run on a fired rule without recording which rule it was.
        """
        if message is None:
            return False
        at_step = self._train_step if step is None else int(step)
        sink = self._sink
        if not bool(getattr(self.shutdown, "running", True)):
            emit_via(sink, {"event": "hard_abort_after_stop", "rule": rule,
                            "message": message, "step": at_step})
            _LOG.warning("hard_abort_after_stop rule=%s step=%s message=%s", rule, at_step, message)
            return False
        _LOG.error("hard_abort rule=%s step=%s message=%s", rule, at_step, message)
        emit_via(sink, {"event": "hard_abort", "rule": rule, "message": message, "step": at_step})
        self.shutdown.running = False
        self.shutdown.abort_rule = rule
        if rule in self._gate_stats:
            self._gate_stats[rule]["fires"] += 1
        return True

    def _emit_monitor_gates(self, cfg: StepCoordinatorConfig, sink: Any) -> None:
        """LAW-18 in-run visibility: every gate publishes its own checks/fires/skips AND
        its live threshold, so an inert gate (explicitly disarmed, or a producer that has
        not landed yet) is READABLE in the event stream instead of silently dead.

        WPAX Phase D: `draw_rate_threshold` keeps its event-contract NAME and is now read
        off the resolved block. `null` is the EXPLICIT off posture (`train.draw_rate_abort:
        null`) — it used to be `0.0`, a number in the middle of the range an operator picks
        from, which is precisely the spelling R79 removed."""
        spec = cfg.draw_rate_abort
        emit_via(sink, {
            "event": "monitor_gates",
            "step": self._train_step,
            "gates": {name: dict(stats) for name, stats in self._gate_stats.items()},
            "draw_rate_threshold": None if spec is None else spec.threshold,
            "sealbot_wr_hard_abort_enabled": bool(self.monitor_cfg.wr_hard_abort_enabled),
            "sealbot_wr_result_producer_pending": None,  # WP11-A: producer landed (eval.rounds's build_round_result)
            "wr_history_len": len(self._wr_history),
            # The watchdog's best-effort counters get a LIVE in-run consumer here (LAW-08 /
            # LAW-18): a degraded fire-path effect (a failed mirror, a timed-out snapshot)
            # is readable in the ONE channel while the run is alive, not only in the
            # `heartbeat_watchdog_fire_complete` event emitted moments before `os._exit`.
            "watchdog_best_effort": self._watchdog_counters(),
        })

    def _watchdog_counters(self) -> dict[str, int]:
        counters = getattr(self.heartbeat_watchdog, "counters", None)
        snapshot = getattr(counters, "snapshot", None)
        if not callable(snapshot):
            return {}
        # `WatchdogCounters.snapshot()` contract: a str→int mapping (duck-typed here —
        # the watchdog is an injected Any collaborator).
        return dict(cast("Mapping[str, int]", snapshot()))

    # ── the async eval-RESULT seam — THE sealbot-WR consumer (§c.4b, MUST-1) ──────────
    def on_eval_round_complete(self, result: Mapping[str, Any]) -> None:
        """Consume ONE completed (drained) eval-round result dict.

        The new-side twin of old `step_coordinator.py` L1168-1200, which read the async
        `_pending_eval_result` — NEVER the eval kick return. Callers today: `drain.py`'s
        `flush_pending_eval` / `run_terminal_eval`; mid-run, WP11-A's non-blocking drain
        runtime MUST route every completed round here (Appendix B handshake).

        `wr_sealbot` absent/None ⇒ ONE `sealbot_wr_gate_skipped` event + skip counter
        (LAW-18: an inert gate is loud, never silently dead).

        Disposition (operator G-3): a sustained-collapse trajectory HARD-ABORTS only when
        `monitor_cfg.wr_hard_abort_enabled` is True; the shipped default is False = WARN-ONLY,
        which emits a VISIBLE `sealbot_wr_warn` carrying the same de-diagnosed trajectory fact
        and does NOT set `shutdown.running=False`. Warn-only is never silent — a warn-only
        gate that emitted nothing would be the silently-disabled class (R1/LAW-18).
        """
        payload: Mapping[str, Any] = result or {}
        stats = self._gate_stats["sealbot_wr_abort"]
        stats["checks"] += 1
        # `or self._train_step` would rewrite a legitimate step 0 (falsy) to the current
        # train step and mis-stamp the WR ring (RED-TEAM F12): test for absence, not truth.
        raw_step = payload.get("step")
        step = self._train_step if raw_step is None else int(raw_step)
        wr = payload.get("wr_sealbot")
        if wr is None:
            stats["skips"] += 1
            emit_via(self._sink, {
                "event": "sealbot_wr_gate_skipped",
                "step": step,
                "reason": "wr_sealbot_absent",
                "skipped_total": stats["skips"],
                "pending_producer": None,  # WP11-A: producer landed (eval.rounds's build_round_result)
            })
            return
        self._wr_history.append((step, float(wr)))
        del self._wr_history[:-WR_HISTORY_DEPTH]
        alert = sealbot_wr_trajectory_alert(self._wr_history, step, self.monitor_cfg)
        if alert is None:
            return
        hard = check_sealbot_wr_hard_abort(self._wr_history, step, self.monitor_cfg)
        if hard is not None:                         # wr_hard_abort_enabled=True → hard-abort
            self._fire_hard_abort("sealbot_wr_abort", hard, step=step)
        else:                                        # default warn-only (operator G-3)
            stats["warns"] += 1
            _LOG.warning("sealbot_wr_warn step=%s message=%s", step, alert)
            emit_via(self._sink, {
                "event": "sealbot_wr_warn",
                "step": step,
                "message": alert,
                "warn_total": stats["warns"],
                "pending_producer": None,  # WP11-A: producer landed (eval.rounds's build_round_result)
            })

    # ── training-step dispatch (mixed vs straight self-play) ──────────────────────────────
    def _run_training_step(self, cfg: StepCoordinatorConfig) -> dict[str, float]:
        # WPMINT Phase K-B CLOSED the R1 violation Phase K-A disclosed here. This line read
        # `int(self.train_cfg.get("batch_size", self.full_config.get("batch_size", 256)))`,
        # and K-A MEASURED that on the production path both lookups miss — `compose_run`
        # passes `train_cfg={}` and a `full_config` whose top-level keys are the RunConfig
        # SECTIONS — so the batch size was unconditionally the literal `256` while
        # `StepCoordinatorConfig.batch_size` (the builder's `8`) sat beside it unread. It is
        # now `train.batch_size`, minted at 256 so the number is unchanged and only its
        # AUTHOR moved (`mantis.config.resolve.coordinator.resolve_coordinator_knobs`). The
        # dict lookups are deleted rather than kept as a fallback: a fallback is the second
        # authority, and `train_cfg` is the legacy flat-hparams path this root does not use.
        batch_size = cfg.batch_size
        if (self.pretrained_buffer is not None and self.pretrained_buffer.size > 0
                and self.buffer.size > 0):
            # WPTS Phase T (CENSUS_C C-2b): the mixed feed is DENSE-ONLY by construction
            # (`sample_batch_with_pos`, `buffer.encoding`, (N,C,H,W) shapes). Entering it
            # under a graph declaration must die at the ROUTE with a named error, not
            # mid-assembly on an AttributeError. No graph-mixed route exists — none is
            # carded and none has a producer.
            spec = self._step_spec()
            if spec.representation != "grid":
                raise RepresentationRouteError(
                    f"the mixed (corpus + selfplay) arm is a dense-only feed; declared "
                    f"representation {spec.representation!r} has no mixed-batch route"
                )
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
        # Straight self-play step (WPTS Phase T / TD-1 / R102): the DECLARED dispatcher
        # routes off the resolved representation to the trainer's TYPED entry points
        # (`train_step_from_graph_batch` / `train_step_from_tensors`). `train_step` is dead.
        return run_declared_train_step(
            self.trainer, self.buffer, self._step_spec(),
            batch_size=batch_size, augment=cfg.augment,
            recency_weight=cfg.recency_weight, recent_buffer=self.recent_buffer,
        )

    def _step_spec(self) -> Any:
        """The resolved encoding spec, lazily resolved ONCE from the declared config this
        coordinator holds (`full_config`), through THE one resolver. An undeclared encoding
        raises `MissingEncodingError` — the LAW-11 posture, never a default arm."""
        if self._resolved_step_spec is None:
            self._resolved_step_spec = resolve_step_spec(self.full_config)
        return self._resolved_step_spec

    # ── eval kickoff at the boundary (via the INJECTED EvalPipelineLike; no train→eval) ───
    def _maybe_kick_eval(self, cfg: StepCoordinatorConfig) -> tuple[bool, bool]:
        """Returns `(eval_kicked_off, eval_skipped_busy)`. The kick ACK is consumed ONLY
        for `eval_skipped_busy` (`ack.get("kicked") is False`) — NEVER for WR (P-06)."""
        if self.eval_pipeline is None or cfg.eval_interval <= 0:
            return False, False
        round_idx = self._train_step // cfg.eval_interval
        if round_idx <= 0 or round_idx == self._eval_round_last_step:
            return False, False
        if self._train_step % cfg.eval_interval != 0:
            return False, False
        self._eval_round_last_step = round_idx
        best = getattr(self.anchor_state, "best_model", None)
        best_step = getattr(self.anchor_state, "best_model_step", None)
        ack = self.eval_pipeline.run_evaluation(
            self.eval_model, self._train_step, best,
            full_config=self.full_config, best_model_step=best_step,
        )
        eval_skipped_busy = bool(ack.get("kicked") is False)
        eval_kicked_off = bool(ack.get("kicked") is True)
        return eval_kicked_off, eval_skipped_busy

    # ── the async eval-result POLL at the top of step() (main-thread, never blocking) ─────
    def _poll_eval_results(self) -> bool:
        if self.eval_pipeline is None:
            return False
        result = self.eval_pipeline.poll_completed()
        if result is None:
            return False
        from mantis.train.coordinator import drain
        drain._route_eval_result(self, result)
        return True

    # ── close-out / terminal-eval flush (delegates to drain.py; §a.4 `drain` slice) ───────
    def flush_pending_eval(self) -> Any:
        from mantis.train.coordinator import drain
        return drain.flush_pending_eval(self)

    def run_terminal_eval(self) -> Any:
        from mantis.train.coordinator import drain
        return drain.run_terminal_eval(self)

    def close_out(self, on_drained: Callable[[], None] | None = None) -> None:
        from mantis.train.coordinator import drain
        drain.close_out(self, on_drained=on_drained)
