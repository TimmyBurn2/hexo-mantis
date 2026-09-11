"""StepCoordinator.step() — the per-step outer-loop core.

>300 justify: one outer iteration (warmup tick / waiting-for-games tick / training burst)
plus its run-safety instrumentation is ONE control-flow unit. The clean-completion save has
to sit on the arm that ACTS on the completion predicate, so splitting the file would put the
write in one place and the decision authorizing it in another.

`step()` never blocks on eval and never reads the eval kick return; completed rounds reach
the sealbot-WR gate only through `on_eval_round_complete`.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import mantis.monitor.rules as _rules  # module-attribute counter reads
import mantis.train.buffer_persist as _buffer_persist
import mantis.train.bundle as _bundle
import mantis.train.bundle_receipts as _bundle_receipts
import mantis.train.resume_state as _resume_state
from mantis.config.resolve.fast_policy_weight import resolve_fast_policy_weight
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.config.resolve.sample_threads import resolve_sample_threads
from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import (
    WR_PEAK_WINDOW_EVALS,
    check_draw_rate_collapse,
    check_sealbot_wr_hard_abort,
    emit_training_step_alerts,
    sealbot_wr_trajectory_alert,
)
from mantis.train.coordinator.config import (
    ClockLike,
    RealClock,
    StepCoordinatorConfig,
    StepOutcome,
    pooled_draw_rate,
)
from mantis.train.coordinator.dispatch import (
    resolve_step_spec,
    run_declared_train_step,
)
from mantis.train.emit import NullEventSink, emit_via
from mantis.train.events import (
    emit_axis_distribution,
    emit_iteration_complete_event,
    emit_training_step_event,
)
from mantis.train.lifecycle.watchdog import StallWatchdog, watchdog_snapshot_path
from mantis.train.mixing import _steps_budget

#: How many COMPLETE resume bundles the run keeps. Two, not one: pruning to one means an
#: unreadable new bundle leaves nothing behind it. Not a config key.
_BUNDLES_RETAINED = 2

_LOG = logging.getLogger(__name__)


def _anchor_sha256(anchor_state: Any) -> str | None:
    """The live anchor's PARAMETER identity, or None when there is no anchor file to hash.

    In `checkpoint_state_sha256`'s denomination, the one `resolve_anchor` compares the launch
    pin against; hashed from the STORED weights, because a resumed run reloads the disk artifact.
    """
    path = getattr(anchor_state, "best_model_path", None)
    if path is None or not Path(path).exists():
        return None
    from mantis.train.anchor import checkpoint_state_sha256

    return checkpoint_state_sha256(Path(path))

#: The gate keys carried by the `monitor_gates` summary (checks/fires/skips/warns).
#: `sealbot_wr_abort` ships warn-only and `draw_rate_collapse` is armed by the config, so both
#: are named here to keep an inert posture readable rather than silent.
GATE_NAMES: tuple[str, ...] = (
    "draw_rate_collapse", "sealbot_wr_abort", "grad_norm_hard_abort",
)

#: The target-integrity counters plus the RECORDED-POSITION counter their fire rate is taken
#: over, all read off the one `RunnerStats` snapshot. `inference_failures_total` rides here
#: because it is the seam conjunct of the same defect class and is only useful read beside them.
_TARGET_INTEGRITY_COUNTERS: tuple[str, ...] = (
    "export_offwindow_mass_moves", "target_integrity_defects", "inference_failures_total",
)
_POSITIONS_COUNTER = "positions_generated"

# Neither the sealbot-WR ring nor the draw-rate ring has a depth constant: a literal clipped
# every schema-legal `consec` above it into unfireable-in-effect, so capacity is derived at the
# point of use. Plain `#` — this documents an ABSENCE and must attach to no assignment.


def _snapshot_counter(rstats: Any, name: str) -> int | None:
    """Read ONE cumulative counter off the runner snapshot, or `None` when it is absent.

    The `None` arm cannot fire in production — `RunnerStats` declares every counter — and is
    kept only because injected telemetry stand-ins drive it.
    """
    value = getattr(rstats, name, None)
    return None if value is None else int(value)


def _fire_rate(delta: int | None, positions_delta: int | None) -> float | None:
    """Fires per RECORDED POSITION over the interval; `None` when there is nothing to divide by.

    No `max(1, ...)` guard: with no position recorded there is NO rate, and `0.0` would claim a
    measurement nobody took. A negative delta passes through, since the atomics are monotonic
    and a decrease is a wiring bug.
    """
    if delta is None or positions_delta is None or positions_delta == 0:
        return None
    return delta / positions_delta


class StepCoordinator:
    """Own the per-step mutable state of the outer training loop; one ``step()`` is one outer
    iteration returning a :class:`StepOutcome`."""

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
        monitor_cfg: MonitorConfig,
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
        # Run-safety seams: the heartbeat fn, the monitor thresholds, the independent watchdog.
        self._heartbeat = heartbeat
        # REQUIRED, no fallback: a default here substitutes dataclass literals for whatever the
        # operator minted. `MonitorConfig` is no longer constructible from this module.
        self.monitor_cfg = monitor_cfg
        self.heartbeat_watchdog = heartbeat_watchdog
        # The clean-completion latch, PUBLIC because `train/loop.py`'s post-loop guard is its
        # one consumer. Set AFTER the leg-3 write, and carrying no set-once guard: exactly-once
        # is a property of the driver, not of a branch only a test can reach.
        self.clean_stop_saved = False
        # None is a unit-test affordance ONLY; production wiring is unconditional at the one
        # composition root.
        self.actor_sync = actor_sync
        # Every PERIODIC checkpoint becomes a full resume bundle: the cadence stays the
        # trainer's, the ring and sidecar come from here because the trainer holds neither, and
        # one publisher keeps the stop legs from drifting. `trainer=None` is a test affordance.
        if self.trainer is not None:
            self.trainer.bundle_publisher = (
                lambda path, _step: self.persist_resume_state(path)
            )

        # Per-step mutable bookkeeping.
        self._train_step = int(getattr(trainer, "step", 0))
        self._games_played = 0
        self.last_train_game_count = 0
        self._schedule_idx = 0
        self.last_warmup_log = 0.0
        self._last_loss_info: dict[str, float] | None = None
        # The resolved encoding spec (lazy, once): the straight arm dispatches off
        # spec.representation, resolved from the declared config through the one resolver.
        self._resolved_step_spec: Any | None = None
        #: The ring rebuild's thread budget, derived once (see `_sample_threads`).
        #: `None` = not yet derived, never "no threads".
        self._resolved_sample_threads: int | None = None
        #: The memo behind `_microbatch_caps`, the graph-only cap thunk.
        self._resolved_caps: Any | None = None
        self._initial_policy_loss: float | None = None
        self._consec_high_gn = 0
        self._eval_round_last_step = -1

        # Gate state — every ring is caller-owned (the rules are stateless).
        self._wr_history: list[tuple[int, float]] = []
        # Which sealbot rung the entries in `_wr_history` came from; `None` before the first
        # round, and whenever the round reported no sealbot rung at all.
        self._wr_history_rung: str | None = None
        self._draw_rate_history: list[float] = []
        self._loss_window: list[float] = []
        self._last_iter_games = 0
        # The previous `iteration_complete` boundary's counter readings, so the payload can
        # publish an INTERVAL delta beside the cumulative total. Seeded at 0 (pool start).
        self._last_target_counters: dict[str, int] = dict.fromkeys(
            (*_TARGET_INTEGRITY_COUNTERS, _POSITIONS_COUNTER), 0,
        )
        # The TERMINAL round's outcome, latched set-once by `drain._record_terminal_outcome`
        # and read by the composition root.
        self._terminal_eval_reason: str | None = None
        self._run_started = self._clock.now()
        # `warns` rides beside checks/fires/skips so the warn-only sealbot posture is visible
        # per-gate in every `monitor_gates` event, not silent.
        self._gate_stats: dict[str, dict[str, int]] = {
            name: {"checks": 0, "fires": 0, "skips": 0, "warns": 0} for name in GATE_NAMES
        }

        # Self-play stall watchdog — always armed; fires to a DISTINCT snapshot path + exit. No
        # code-side default for that path: it is derived from the trainer's own checkpoint dir
        # and resolved at FIRE time, so construction needs no trainer attribute.
        def _snapshot_target() -> Path:
            bp = self.mixing_cfg.get("buffer_persist_path")
            if bp is None:
                bp = _buffer_persist.canonical_buffer_path(self.trainer.checkpoint_dir)
            return watchdog_snapshot_path(Path(bp))

        self._watchdog = StallWatchdog(
            timeout_sec=config.selfplay_stall_timeout_sec,
            clock=self._clock.now,
            sink=sink,
            exit_fn=exit_fn,
            save_snapshot=lambda: self._snapshot_buffer(_snapshot_target()),
            # The stall abort saves WEIGHTS too, routed through the trainer's stamped save path
            # so the artifact is a real envelope-v2 checkpoint, not a bare state_dict.
            save_model=lambda: self.trainer.save_checkpoint(self._last_loss_info or None),
        )
        self._watchdog.arm(getattr(pool, "games_completed", 0))

    def _snapshot_buffer(self, path: Any) -> None:
        saver = getattr(self.buffer, "save_to_path", None)
        if saver is not None:
            saver(str(path))

    def _disk_critical(self) -> bool:
        """Has the disk guard already fired? `False` when none is wired — the safe direction,
        since an absent guard means no disk abort is in flight."""
        guard = getattr(self.subsystems, "disk_guard", None)
        return bool(guard is not None and getattr(guard, "critical_fired", False))

    def persist_resume_state(self, checkpoint_path: Any) -> Any:
        """Persist the ring and write the sidecar that makes `checkpoint_path` resumable.

        Called from BOTH save legs — this coordinator's O3 arm and `loop.py`'s `_final_save()`,
        the one that completes a stop when `step()` does not return. Safe to call twice: both
        writes are idempotent and the last caller wins with the truest step.

        The resumable-stop guard lives here so the two legs cannot drift. The one abort that
        reaches them with `abort_rule` unrecorded is the disk guard's, and persisting a large
        ring on a full disk would deepen the condition that fired.

        Distinct from `_snapshot_buffer`, the watchdog's best-effort `.watchdog` snapshot: this
        one IS the resume buffer, and a failure stops the run loudly rather than leaving a
        resume that silently refills from empty.

        Returns the sidecar path, or `None` when the stop is not a resumable one.

        Raises:
            OSError: the ring or the sidecar could not be written.
            AttributeError: the buffer cannot persist itself — a wiring error, never a state.
        """
        if self.shutdown.abort_rule is not None or self._disk_critical():
            return None
        # The ring is named for ITS OWN checkpoint rather than one canonical path every save
        # overwrote — that made "keep the previous bundle" impossible in principle.
        ring_path = _bundle.ring_path_for(checkpoint_path)
        # THE STEP COMES FROM THE CHECKPOINT, not `self._train_step`: the counter is refreshed
        # after `_run_training_step` returns while the periodic seam fires inside it, so at a
        # periodic publication it is one behind. The counter is the unparsable-name fallback.
        bundle_step = _bundle.step_of(checkpoint_path)
        if bundle_step is None:
            bundle_step = int(self._train_step)
        pipeline = self.eval_pipeline
        state = _resume_state.ResumeState(
            version=_resume_state.SIDECAR_VERSION,
            run_id=str(self.full_config.get("run_id", "")),
            step=bundle_step,
            checkpoint_filename=Path(checkpoint_path).name,
            # The counters with no HEAD mechanism. Read through the pipeline's own accessor: a
            # second authority for "which round is next" is what `gate.stride` cannot survive.
            round_counter=(0 if pipeline is None else int(pipeline.round_counter)),
            last_p_hat=({} if pipeline is None else dict(pipeline.last_p_hat)),
            anchor_sha256=_anchor_sha256(self.anchor_state),
            # A placeholder the publisher replaces: the ring's hash is unknown until written.
            ring=None,
            rng=_resume_state.capture_rng_streams(),
        )

        def _write_ring(path: Path) -> None:
            self.buffer.save_to_path(str(path))

        def _write_sidecar(path: Path) -> None:
            ring = _resume_state.RingRef(
                path=str(ring_path), sha256=_bundle.sha256_file(ring_path),
                positions=int(self.buffer.size),
            )
            _resume_state.write_resume_state(
                dataclasses.replace(state, ring=ring), checkpoint_path,
            )
            del path  # the sidecar's own path authority is `sidecar_path_for`

        manifest = _bundle.publish_bundle(
            checkpoint_path=checkpoint_path,
            run_id=str(self.full_config.get("run_id", "")),
            step=bundle_step,
            write_ring=_write_ring, ring_path=ring_path,
            write_sidecar=_write_sidecar,
            sidecar_path=_resume_state.sidecar_path_for(checkpoint_path),
        )
        # Retention runs AFTER the manifest commits, never before: pruning first would leave a
        # window in which the run holds fewer complete bundles than its own policy promises.
        pruned = _bundle.prune_bundles(self.trainer.checkpoint_dir, keep=_BUNDLES_RETAINED)
        side = _resume_state.sidecar_path_for(checkpoint_path)
        loaded = _bundle.read_manifest(manifest)
        _LOG.info(
            "resume_bundle_published manifest=%s ring=%s positions=%d round_counter=%d pruned=%d",
            manifest.name, ring_path.name, int(self.buffer.size), state.round_counter,
            len(pruned),
        )
        emit_via(self._sink, {
            "event": "resume_state_persisted", "step": state.step, "sidecar": str(side),
            "ring_positions": int(self.buffer.size),
            "ring_sha256": "" if loaded.ring is None else loaded.ring.sha256,
            "round_counter": state.round_counter,
            "manifest": str(manifest), "pruned_members": len(pruned),
            # R349(b): every retained complete bundle (this one included) without receipts on all
            # its files — the mirror's lag; two is the dashboard's warning, nothing halts.
            "unreceipted_bundles": _bundle_receipts.unreceipted_bundle_steps(
                self.trainer.checkpoint_dir),
        })
        return side

    def _build_outcome(self, **kw: Any) -> StepOutcome:
        return StepOutcome(
            train_step=self._train_step,
            games_played=self._games_played,
            consec_high_gn=self._consec_high_gn,
            last_loss_info=self._last_loss_info,
            **kw,
        )

    def stop(self, reason: str) -> None:
        _LOG.info("stop_requested reason=%s", reason)
        self.shutdown.running = False

    @property
    def terminal_eval_reason(self) -> str | None:
        """The TERMINAL eval round's typed reason, or `None` for a clean terminal battery.

        A `str` and never the reason enum, because the train package may not import the eval
        package; the enum re-parses it at the boundary, where an unknown spelling is loud.
        """
        return self._terminal_eval_reason

    def record_terminal_eval_reason(self, reason: str | None) -> None:
        """Record the terminal round's outcome. **FIRST NON-`None` WINS.**

        Not set-once, as measured: a first call with `None` (a clean round) latches nothing, so
        a later call can still write. One writer in `src/`, so the difference is unpinned.
        """
        if self._terminal_eval_reason is not None:
            return
        self._terminal_eval_reason = reason

    def step(self) -> StepOutcome:
        """Run exactly one outer iteration; return a :class:`StepOutcome`."""
        cfg = self.config

        # The outer loop is alive. Beaten at ENTRY so every early-return branch still proves
        # liveness, and once per burst iteration.
        self._beat("train_step")

        # Fail-fast: the self-play feeder is the sole producer — abort loudly if it died.
        health = getattr(self.pool, "check_producer_health", None)
        if health is not None:
            health()

        # Non-blocking eval-result poll at the TOP of every iteration, on every branch
        # (warmup and waiting-for-games included); never a blocking drain.
        eval_drained = self._poll_eval_results()

        base = dict(
            steps_run=0, buffer_resized=None, checkpoint_saved=False, axis_emitted=False,
            eval_kicked_off=False, eval_skipped_busy=False, eval_drained=eval_drained,
            promoted_step=None, soft_abort_fired=False, hard_abort_fired=False,
            instrumentation_emitted=[], pool_overflow_delta=0,
        )

        # O2: iteration-limit reached — CLEAN COMPLETION, and the third save leg. The one
        # OUTER-loop site that ACTS on the completion predicate (the inner burst-break only
        # ends the burst), so the save must sit here; it runs before `running = False`, and
        # clean-vs-aborted is carried by `ShutdownState.abort_rule`, never by a file.
        if cfg.stop_step is not None and self._train_step >= cfg.stop_step:
            self._clean_stop_save(cfg)
            self.shutdown.running = False
            return self._build_outcome(in_warmup=False, waiting_for_games=False,
                                       **{**base, "checkpoint_saved": True})

        # O3: shutdown-save — checkpoint, ring and sidecar, then stop. `persist_resume_state`
        # is run-fatal: a stop that cannot record its ring has not stopped resumably.
        if self.shutdown.shutdown_save:
            ckpt = self.trainer.save_checkpoint(self._last_loss_info or None)
            self.persist_resume_state(ckpt)
            self.shutdown.running = False
            return self._build_outcome(in_warmup=False, waiting_for_games=False,
                                       **{**base, "checkpoint_saved": True})

        self._games_played = int(getattr(self.pool, "games_completed", 0))
        # Stall watchdog — driven via tick(...).
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
        # Burst accumulators: the eval kick runs PER TRAINING STEP inside the burst, so its two
        # outcomes are OR-folded across the burst like the flags above.
        eval_kicked_off = False
        eval_skipped_busy = False

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

            # D2: training step — mixed when a pretrained buffer is present, else straight
            # self-play. Both route through the injected trainer.
            loss_info = self._run_training_step(cfg)
            self._train_step = self.trainer.step
            # D2b: continuous actor weight sync, per inner step. `_train_step` advances by
            # exactly 1 per burst iteration so a modulo boundary can never be skipped.
            if self.actor_sync is not None:
                self.actor_sync.maybe_sync(self._train_step)
            if self._initial_policy_loss is None and "policy_loss" in loss_info:
                self._initial_policy_loss = float(loss_info["policy_loss"])
            self._last_loss_info = loss_info

            # D3: hard-abort on sustained gradient norm. The FIRE routes through the shared
            # `_fire_hard_abort` contract so this gate is visible in the one channel.
            self._gate_stats["grad_norm_hard_abort"]["checks"] += 1
            step_gn = float(loss_info.get("grad_norm", 0.0))
            # NaN/inf is EXCLUDED from this abort — a KNOWN GAP, not an oversight. This exact
            # comparison is a SOURCE PIN in `config/armed_aborts.py`'s `grad_norm_hard_abort`
            # row: re-adjudicate the row rather than editing the line. The non-finite guard is
            # in `clip_and_step`, which refuses the step, so resetting the counter is correct.
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

            # There is no checkpoint-cadence buffer save on this leg: `checkpoint_saved` stays
            # `False` for the whole burst path, and the O2/O3 legs above announce a real write.

            # The cadence boundaries are tested PER TRAINING STEP: once per burst would skip
            # every boundary the post-burst step misses, stretching the draw-rate gate's
            # `consec` window. Two knobs — narration (`log`) and arming (`gate`).
            axis_emitted = self._run_log_interval(cfg, loss_info) or axis_emitted
            hard_abort_fired = self._run_gate_interval(cfg) or hard_abort_fired

            # Kicked INSIDE the burst: run once after the whole burst, a burst that stepped over
            # the exact multiple never satisfied the modulo and the round was SILENTLY SKIPPED.
            # The kick return is never consumed for WR; rounds arrive via the drain.
            kicked_step, skipped_step = self._maybe_kick_eval(cfg)
            eval_kicked_off = eval_kicked_off or kicked_step
            eval_skipped_busy = eval_skipped_busy or skipped_step

        # `iteration_complete` emits at the O6 burst return, per coordinator step, INDEPENDENT
        # of `log_interval`. Not called on the O2/O3 early returns, which are not burst returns.
        self._emit_iteration_complete(cfg)
        return self._build_outcome(
            in_warmup=False, waiting_for_games=False,
            **{**base, "steps_run": steps_budget, "buffer_resized": buffer_resized,
               "checkpoint_saved": checkpoint_saved, "eval_kicked_off": eval_kicked_off,
               "eval_skipped_busy": eval_skipped_busy,
               "hard_abort_fired": hard_abort_fired, "axis_emitted": axis_emitted},
        )

    def _clean_stop_save(self, cfg: StepCoordinatorConfig) -> None:
        """The CLEAN-COMPLETION save — the third save leg, beside the trainer's periodic cadence
        and the signal-driven `shutdown_save`.

        Its artefact means "the run FINISHED", not "a resumption point" or "a rescue of
        interrupted work". It calls the SAME `trainer.save_checkpoint` the other legs call, so
        the product rides the one stamp path, and a failure is NOT caught — the persist-fatal
        watchdog already owns that exit code.

        The latch and the event land AFTER the write: an event named for a save is a claim it
        happened, and a pre-set latch would suppress leg 2 on a run whose leg-3 write died.
        """
        path = self.trainer.save_checkpoint(self._last_loss_info or None)
        self.clean_stop_saved = True
        emit_via(self._sink, {
            "event": "clean_stop_save",
            "step": self._train_step,
            "stop_step": cfg.stop_step,
            "path": None if path is None else str(path),
        })

    def _beat(self, source: str) -> None:
        """Beat one heartbeat source; an unknown source raises in the registry, because a
        wiring bug must be loud rather than a beat the watchdog silently never sees."""
        if self._heartbeat is not None:
            self._heartbeat(source)

    def _run_log_interval(
        self, cfg: StepCoordinatorConfig, loss_info: dict[str, float]
    ) -> bool:
        """Emit the run's NARRATION at the `log_interval` boundary — the `training_step`
        payload, the WARN rules over it and the axis distribution. Returns ``axis_emitted``.

        Arming moved out to `_run_gate_interval`; only logging is `log_interval`-gated. The
        `loss_info` guard belongs here: every payload built here is made of the loss dict.
        """
        if not loss_info or cfg.log_interval <= 0 or self._train_step % cfg.log_interval != 0:
            return False
        sink = self._sink if self._sink is not None else NullEventSink()

        payload = self._emit_training_step(cfg, loss_info, sink)
        emit_training_step_alerts(payload, self.monitor_cfg, self._loss_window, sink=sink)
        keep = max(2 * int(self.monitor_cfg.alert_loss_increase_window) + 2, 8)
        del self._loss_window[:-keep]

        axis = emit_axis_distribution(
            self._train_step, self.pool, self.monitor_cfg,
            getattr(self.subsystems, "axis_baseline", None) or {},
            getattr(self.subsystems, "tb_writer", None), sink,
        )
        return axis is not None

    def _run_gate_interval(self, cfg: StepCoordinatorConfig) -> bool:
        """Run the live-producer hard-abort gates and publish the `monitor_gates` summary at the
        `monitor.gate_interval` boundary. Returns ``hard_abort_fired``.

        Split off `log_interval` so arming never rides the narration cadence: at a minted
        `log_interval: 1000` no draw-rate observation could be taken before training step 1000.

        NO `loss_info` condition, deliberately: these gates' producer is the POOL, not the
        trainer. `consec` counts OBSERVATIONS taken at a stride of AT LEAST `gate_interval`
        steps, so `consec * gate_interval` bounds a fire's span from below.

        The `<= 0` arm is unreachable from any minted config; it exists so a direct construction
        cannot raise `ZeroDivisionError` inside the burst.
        """
        if cfg.gate_interval <= 0 or self._train_step % cfg.gate_interval != 0:
            return False
        sink = self._sink if self._sink is not None else NullEventSink()
        fired = self._run_hard_abort_gates(cfg)
        self._emit_monitor_gates(cfg, sink)
        return fired

    def _emit_training_step(
        self, cfg: StepCoordinatorConfig, loss_info: dict[str, float], sink: Any
    ) -> dict[str, Any]:
        """Build and emit the `training_step` event through the injected sink and return its
        payload (the WARN rules read it). Stays `log_interval`-gated: it is narration."""
        payload = emit_training_step_event(
            self._train_step, loss_info,
            # `quiescence_fires_per_step` has no producer here: the field travels as None =
            # NOT MEASURED, because a constant 0 would read as "quiescence never fires".
            None, sink,
        )
        return payload

    def _emit_iteration_complete(self, cfg: StepCoordinatorConfig) -> None:
        """Build and emit `iteration_complete` at the O6 training-burst return, per coordinator
        step and INDEPENDENT of `log_interval`.

        Passes the `RunnerStats` snapshot from `_target_integrity_report` into the builder, so
        ONE atomic snapshot serves the whole payload and the two blocks cannot straddle a game
        boundary. Not called on the O2/O3 early returns.
        """
        sink = self._sink if self._sink is not None else NullEventSink()
        w_pre = 0.0
        rstats_report, rstats = self._target_integrity_report()
        emit_iteration_complete_event(
            self._train_step, w_pre, self._games_played, self._last_iter_games,
            self.pool, self.buffer, self.full_config, self.full_config.get("mcts", {}),
            cfg.capacity, self._games_per_hour, self._steps_per_hour,
            rstats_report, rstats, sink,
        )
        self._last_iter_games = self._games_played

    def _target_integrity_report(self) -> tuple[dict[str, Any], Any]:
        """The target-integrity counters as an `iteration_complete` block, and the `RunnerStats`
        snapshot they were built from.

        Each counter publishes its cumulative `total`, its INTERVAL `delta` and a `per_position`
        rate over the denominator published beside it. An idle lever stays VISIBLE at 0, which
        keeps a permanently-0 `target_integrity_defects` distinguishable from a field with no
        producer. Costs ONE `pool.runner_stats()` FFI crossing per emit, reused by the builder.
        """
        rstats = self.pool.runner_stats()
        positions = _snapshot_counter(rstats, _POSITIONS_COUNTER)
        positions_delta = (None if positions is None
                           else positions - self._last_target_counters[_POSITIONS_COUNTER])
        report: dict[str, Any] = {"positions_delta": positions_delta}
        for name in _TARGET_INTEGRITY_COUNTERS:
            total = _snapshot_counter(rstats, name)
            delta = None if total is None else total - self._last_target_counters[name]
            report[name] = {"total": total, "delta": delta,
                            "per_position": _fire_rate(delta, positions_delta)}
            if total is not None:
                self._last_target_counters[name] = total
        if positions is not None:
            self._last_target_counters[_POSITIONS_COUNTER] = positions
        return report, rstats

    def _games_per_hour(self) -> float | None:
        """Games per hour over the run clock, or `None` before the clock has advanced — a rate
        over zero elapsed time is an ABSENT measurement, not a rate of zero."""
        elapsed = self._clock.now() - self._run_started
        return (self._games_played / elapsed) * 3600.0 if elapsed > 0 else None

    def _steps_per_hour(self) -> float | None:
        """Train steps per hour over the same clock as `_games_per_hour`, published beside it;
        `None` before the clock has advanced, for that method's reason."""
        elapsed = self._clock.now() - self._run_started
        return (self._train_step / elapsed) * 3600.0 if elapsed > 0 else None

    def _run_hard_abort_gates(self, cfg: StepCoordinatorConfig) -> bool:
        """The draw-rate hard-abort gate, keyed on the LIVE pool producer.

        Run at the `monitor.gate_interval` boundary, which is this gate's ATTEMPT stride, while
        `consec` counts OBSERVATIONS a boundary may fail to supply and a failure does not reset.
        Reads the POOLED count-weighted rate over the union of worker windows with a
        config-authored evidence bar; a missing producer and insufficient evidence are both
        SKIP-counted, never read as a healthy signal.
        """
        counts_fn = getattr(self.pool, "pooled_draw_counts", None)
        spec = cfg.draw_rate_abort
        # `is not None`, NOT `> 0`: `draw_rate_abort` is `None` on every disarmed run and
        # `None > 0` raises. Both absences route through `_sample` with a `None` producer,
        # which is the one site that SKIP-counts them.
        if spec is None or counts_fn is None:
            self._sample("draw_rate_collapse", self._draw_rate_history, None)
            return False
        # Branching on the return is required, not tidy: `pooled_draw_rate` returns `None` below
        # `N_pool_min` (insufficient evidence, not a fabricated healthy 0.0), and `consec` counts
        # OBSERVATIONS, so running the rule with nothing appended re-decides on a stale tail.
        if not self._sample(
            "draw_rate_collapse", self._draw_rate_history,
            lambda: pooled_draw_rate(counts_fn(), N_pool_min=spec.N_pool_min),
        ):
            return False
        # The ring's capacity IS the minted `consec`, derived from the one authority
        # `check_draw_rate_collapse` also gates on, so no schema-legal `consec` is unfireable.
        # The trim sits AFTER a True `_sample` return, the one place `spec` is narrowed.
        del self._draw_rate_history[:-spec.consec]
        message = check_draw_rate_collapse(self._draw_rate_history, self._train_step,
                                           threshold=spec.threshold,
                                           consec=spec.consec,
                                           min_step=spec.min_step)
        return self._fire_hard_abort("draw_rate_collapse", message)

    def _sample(self, gate: str, history: list[float], producer: Any) -> bool:
        """Append one LIVE producer sample to ``history``; False (+skip) when there is no
        observation to append.

        An observation is ATTEMPTED once per `_run_gate_interval` boundary, not taken. A
        boundary that yields none neither appends NOR resets, so `consec` counts consecutive
        ENTRIES IN `history` and `consec * gate_interval` is a LOWER BOUND on the span a fire
        covers, never an equality.

        Two absences, one skip counter — an absent producer, and a live producer returning
        `None` for insufficient evidence. Neither appends: an unobserved interval must never
        enter an abort history as a number.

        THE CALLER OWNS THE RING'S CAPACITY. A True return means exactly one append; this
        function must never clip the ring, because a constant here silently capped every
        history that slid through it.
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
        return True

    def _fire_hard_abort(self, rule: str, message: str | None, step: int | None = None) -> bool:
        """Stop the run and emit one `hard_abort` event naming the rule and its message.

        A gate resolving AFTER the run stopped records a DISTINCT `hard_abort_after_stop`, so
        the trail stays complete without reporting a second abort decision.

        The fire also records the RULE NAME on `ShutdownState.abort_rule`, paired with
        `running = False` so a fired rule can never go unrecorded. The name, not a code: rule →
        exit-code resolution belongs at the process boundary.
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
        self.shutdown.record_abort(rule)
        if rule in self._gate_stats:
            self._gate_stats[rule]["fires"] += 1
        return True

    def _emit_monitor_gates(self, cfg: StepCoordinatorConfig, sink: Any) -> None:
        """Publish every gate's checks/fires/skips and its live threshold, so an inert gate is
        READABLE in the event stream instead of silently dead. `draw_rate_threshold` is `None`
        on the explicit off posture; it used to be `0.0`, a number in the operator's own range."""
        spec = cfg.draw_rate_abort
        emit_via(sink, {
            "event": "monitor_gates",
            "step": self._train_step,
            "gates": {name: dict(stats) for name, stats in self._gate_stats.items()},
            "draw_rate_threshold": None if spec is None else spec.threshold,
            "sealbot_wr_hard_abort_enabled": bool(self.monitor_cfg.wr_hard_abort_enabled),
            "sealbot_wr_result_producer_pending": None,  # producer landed (eval.rounds)
            "wr_history_len": len(self._wr_history),
            # The ring's length means nothing without the rung it is a series OVER.
            "wr_history_rung": self._wr_history_rung,
            # The watchdog's best-effort counters get a live in-run consumer here: a degraded
            # fire path is readable while the run is alive, not only moments before `os._exit`.
            "watchdog_best_effort": self._watchdog_counters(),
            # Per-WARN-rule count of steps at which the rule could not run for want of its
            # input, else "never fires" and "healthy" are one observable. Module-attribute read.
            "warn_rule_skipped_absent": dict(_rules.WARN_RULE_SKIPS),
            # The best-effort buffer-save swallows, counted and read live. A module-attribute
            # read, never a from-import of the int.
            "buffer_save_errors_total": int(_buffer_persist.buffer_save_errors_total),
        })

    def _watchdog_counters(self) -> dict[str, int] | None:
        """The watchdog's best-effort counters, or `None` when NO WATCHDOG IS WIRED — an empty
        mapping from a live watchdog is a real measurement and absence is not."""
        counters = getattr(self.heartbeat_watchdog, "counters", None)
        snapshot = getattr(counters, "snapshot", None)
        if not callable(snapshot):
            return None
        # `snapshot()` returns a str-to-int mapping, duck-typed: the watchdog is injected Any.
        return dict(cast("Mapping[str, int]", snapshot()))

    def on_eval_round_complete(self, result: Mapping[str, Any]) -> None:
        """Consume ONE completed (drained) eval-round result dict.

        Callers: `drain.py`'s `flush_pending_eval` / `run_terminal_eval`, and the mid-run
        non-blocking drain, which MUST route every completed round here.

        `wr_sealbot` absent/None gives ONE `sealbot_wr_gate_skipped` event plus a skip counter
        carrying one of three reasons — `eval_round_broken`, `strength_floor_refused`,
        `wr_sealbot_absent` — in that precedence, so the three causes of "this gate has no
        number" are never one observable. That path appends nothing and must not trim.

        THIS METHOD OWNS THE RING'S CAPACITY and derives it from the two minted consec keys and
        rule B's peak window; a literal made every larger consec unfireable, and dropping the
        window floor would widen rule B's peak with the ring.

        A collapse trajectory HARD-ABORTS only when `wr_hard_abort_enabled` is True; the shipped
        default is warn-only, which still emits a visible `sealbot_wr_warn`.
        """
        payload: Mapping[str, Any] = result or {}
        stats = self._gate_stats["sealbot_wr_abort"]
        stats["checks"] += 1
        # `or self._train_step` would rewrite a legitimate step 0 (falsy) to the current train
        # step and mis-stamp the WR ring: test for absence, not truth.
        raw_step = payload.get("step")
        step = self._train_step if raw_step is None else int(raw_step)
        # A round that BROKE and a healthy round carrying no sealbot number used to reach this
        # gate as the SAME observable, and the promotion bar must not read as un-evidenced when
        # the evidence PATH failed. `.get` and NOT a subscript, unlike `apply_gate_decision`:
        # the key legitimately postdates some routes, and a `KeyError` kills the poller thread.
        broken = payload.get("eval_broken_reason")
        # A round the strength floor REFUSED is neither broken nor healthy-but-metric-less.
        # PRESENCE of the key is the arming evidence: a disarmed round carries none.
        floor = payload.get("strength_floor")
        floor_refused = False
        # `None` on every arm but a refusal, so a consumer reads one key rather than inferring
        # the case from a string.
        floor_failed_bars: list[Any] | None = None
        if isinstance(floor, Mapping) and floor.get("passed") is False:
            floor_refused = True
            floor_failed_bars = list(floor.get("failed_bars") or [])
        wr = payload.get("wr_sealbot")
        if wr is None:
            stats["skips"] += 1
            # Precedence, stated rather than incidental: broken outranks refused. A round that
            # broke may carry a floor payload from before the break.
            if broken is not None:
                reason = "eval_round_broken"
            elif floor_refused:
                reason = "strength_floor_refused"
            else:
                reason = "wr_sealbot_absent"
            emit_via(self._sink, {
                "event": "sealbot_wr_gate_skipped",
                "step": step,
                "reason": reason,
                # Present on EVERY skip, `None` on the healthy-but-metric-less arm.
                "eval_broken_reason": broken,
                # The bars that failed, or `None` when the floor did not refuse this round.
                "strength_floor_failed_bars": floor_failed_bars,
                "skipped_total": stats["skips"],
                "pending_producer": None,  # producer landed (eval.rounds)
            })
            return
        # The ring is PER-RUNG. `wr_sealbot` is the first sealbot rung with games this round, so
        # a saturated rung going off-cadence makes consecutive rounds report different rungs, and
        # a drop to a HARDER opponent is not a collapse. A rung change restarts the series and
        # says so; carrying the rung in the tuple would reshape a ring two rules read.
        rung = payload.get("wr_sealbot_rung")
        if rung != self._wr_history_rung:
            if self._wr_history:
                emit_via(self._sink, {
                    "event": "sealbot_wr_series_restarted",
                    "step": step,
                    "from_rung": self._wr_history_rung,
                    "to_rung": rung,
                    "discarded": len(self._wr_history),
                    "reason": "the trajectory rules compare a rung against ITSELF; a series "
                              "spanning two opponents is not one series",
                })
                _LOG.info("sealbot_wr_series_restarted step=%s from=%s to=%s discarded=%s",
                          step, self._wr_history_rung, rung, len(self._wr_history))
            self._wr_history.clear()
            self._wr_history_rung = rung
        self._wr_history.append((step, float(wr)))
        # The capacity is DERIVED from everything that reads it — the two minted consec keys and
        # rule B's peak window — so no schema-legal consec is unfireable. `max` also keeps it
        # positive, since `[:-0]` deletes NOTHING and a 0 capacity would unbound the ring.
        capacity = max(WR_PEAK_WINDOW_EVALS,
                       int(self.monitor_cfg.wr_collapse_consecutive_evals),
                       int(self.monitor_cfg.wr_rolling_consecutive_evals))
        del self._wr_history[:-capacity]
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

    def _run_training_step(self, cfg: StepCoordinatorConfig) -> dict[str, float]:
        # `train.batch_size`, minted at 256. This was a dict lookup whose two levels both miss
        # on the production path, so the batch size was unconditionally a literal fallback.
        batch_size = cfg.batch_size
        # The DECLARED dispatcher routes off the resolved representation to the trainer's typed
        # entry points (`train_step_from_graph_batch` / `train_step_from_tensors`).
        return run_declared_train_step(
            self.trainer, self.buffer, self._step_spec(),
            batch_size=batch_size, augment=cfg.augment,
            recency_weight=cfg.recency_weight, recent_buffer=self.recent_buffer,
            caps_provider=self._microbatch_caps,
            sample_threads_provider=self._sample_threads,
            fast_policy_weight_provider=self._fast_policy_weight,
        )

    def _fast_policy_weight(self) -> float:
        """The graph route's fast-arm policy weight, resolved lazily.

        Not memoised, unlike the caps: it is one dict lookup and a float, and a memo would be
        a second place the value lives.
        """
        return resolve_fast_policy_weight(self.full_config)

    def _step_spec(self) -> Any:
        """The resolved encoding spec, lazily resolved ONCE from the declared config through
        the one resolver. An undeclared encoding raises `MissingEncodingError`, never a
        default arm."""
        if self._resolved_step_spec is None:
            self._resolved_step_spec = resolve_step_spec(self.full_config)
        return self._resolved_step_spec

    def _sample_threads(self) -> int:
        """The ring rebuild's thread budget, derived ONCE from this coordinator's config; cached
        for `_step_spec`'s reason, not for speed, since its inputs are fixed for the run."""
        if self._resolved_sample_threads is None:
            self._resolved_sample_threads = resolve_sample_threads(self.full_config)
        return self._resolved_sample_threads

    def _microbatch_caps(self) -> Any:
        """The resolved graph micro-batch caps, lazily resolved ONCE. Absence raises by name.

        Passed to the dispatcher as a CALLABLE — the bound method, not a call — and invoked by
        the GRAPH arm only. Python evaluates every argument before the call, so calling it here
        would resolve `full_config["train"]` on BOTH representations, and a graph-only knob must
        not make a grid config unloadable. Memoised like `_step_spec`, but lazy where that one is
        eager: `_step_spec` DECIDES the route, these caps matter on one branch of it.
        """
        if self._resolved_caps is None:
            self._resolved_caps = resolve_microbatch_caps(self.full_config)
        return self._resolved_caps

    def _maybe_kick_eval(self, cfg: StepCoordinatorConfig) -> tuple[bool, bool]:
        """Return `(eval_kicked_off, eval_skipped_busy)`. The kick ACK is consumed ONLY for
        `eval_skipped_busy` (`ack.get("kicked") is False`) — never for WR."""
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

    def _poll_eval_results(self) -> bool:
        if self.eval_pipeline is None:
            return False
        result = self.eval_pipeline.poll_completed()
        if result is None:
            return False
        from mantis.train.coordinator import drain
        drain._route_eval_result(self, result)
        return True

    # close-out / terminal-eval flush (delegated to drain.py)
    def flush_pending_eval(self) -> Any:
        from mantis.train.coordinator import drain
        return drain.flush_pending_eval(self)

    def run_terminal_eval(self) -> Any:
        from mantis.train.coordinator import drain
        return drain.run_terminal_eval(self)

    def close_out(
        self, on_drained: Callable[[], None] | None = None, *, resumable_stop: bool = False,
    ) -> None:
        from mantis.train.coordinator import drain
        drain.close_out(self, on_drained=on_drained, resumable_stop=resumable_stop)
