"""StepCoordinator.step() — the per-step outer-loop core (WP10 §a.4 split — `step` slice).

>300 justify: `step()` reproduces one outer iteration of the old
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
REMOVED at close-out, operator directive B.) WP12-R Phase CS adds the THIRD save leg
(R137/CARD-CLEANSTOP-SAVE) to the SAME unit: the O2 iteration-limit arm is the one
OUTER-loop site that ACTS on the clean-completion predicate by ending the run — the inner
burst-break evaluates the character-identical expression but only ends the burst, leaving
`running` True so the outer arm fires on the next `step()` — so the save that makes a
finished run's product exist has to sit on this arm; splitting it out would put the write in
one file and the decision that authorizes it in another.

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
from pathlib import Path
from typing import Any, cast

import mantis.train.buffer_persist as _buffer_persist
from mantis.config.resolve.microbatch import resolve_microbatch_caps
from mantis.monitor.config import MonitorConfig
from mantis.monitor.rules import (
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
    RepresentationRouteError,
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

#: The three WP12-R Phase-T target-integrity counters (LAW-18 / R164), in the order
#: `IMPL_NOTES_T §3.6` names them, and the RECORDED-POSITION counter their fire rate is
#: taken over. Both live on the `RunnerStats` snapshot `mantis.train.events` already reads
#: once per `iteration_complete`, so nothing here opens a second reader of the pool.
_TARGET_INTEGRITY_COUNTERS: tuple[str, ...] = (
    "export_offwindow_mass_moves", "gridls_zero_policy_rows", "target_integrity_defects",
)
_POSITIONS_COUNTER = "positions_generated"

#: Depth of the sealbot-WR ring (old `step_coordinator.py` parity: `pop(0)` past 5).
WR_HISTORY_DEPTH = 5
#: Depth of the pool-signal rings the two live-producer gates slide over.
_GATE_HISTORY_DEPTH = 32


def _snapshot_counter(rstats: Any, name: str) -> int | None:
    """Read ONE cumulative counter off the runner snapshot, or `None` if the snapshot does
    not carry it.

    **THE `None` ARM CANNOT FIRE IN PRODUCTION, and saying so is the point** (RED-TEAM F-02
    — graded instance FOUR of this phase's weak axis, and the first instance in shipped code
    rather than in an instrument). An earlier version of this docstring spent eleven lines on
    what `None` MEANS in the event stream without checking what can REACH it. Measured:
    `pool_hooks.RunnerStats` declares all three counters `int = 0` and `runner_stats()`
    supplies every field explicitly, so a real snapshot ALWAYS carries the attribute —
    `getattr(rstats, name, None)` never returns `None` on any production path, and no oracle
    drives that arm (O-23's `None` is reached through the OTHER condition, a zero
    `positions_delta`). The precedent the old text cited to justify the arm — `pool_hooks`'s
    own per-field `getattr` defaults — is the very mechanism that makes it unreachable.

    So the arm's TRUE purpose, stated honestly: it keeps the **17 injected telemetry
    stand-ins** drivable — one of them a SEALED oracle
    (`tests/train/test_terminal_eval_rc.py`'s five-attribute `_RunnerStats`, which O-05 node
    1 drives through `coord.step()` at `log_interval=1`). It is a test-driven branch in
    production code. It is NOT deleted, because deleting it reds those 17 files and the full
    mutation battery has already been driven against these bytes; it is recorded as
    `Q-O-DEAD-NOT-MEASURED-ARM`.

    What the arm does NOT do, so the next reader does not re-derive it wrong: it does not
    protect against a counter losing its producer. A renamed bridge getter is swallowed one
    layer BELOW this function by `pool_hooks`'s `getattr(r, name, 0)`, which publishes a
    fabricated `0` — `Q-O-BRIDGE-GETTER-NAMES`, measured at the Rust level by RED-TEAM §2.
    If the value ever DOES arrive as `None`, `event_manifest.md`'s convention governs it
    (NOT MEASURED, never a fabricated value) — that contract is correct; only its
    reachability was overstated.
    """
    value = getattr(rstats, name, None)
    return None if value is None else int(value)


def _fire_rate(delta: int | None, positions_delta: int | None) -> float | None:
    """Fires per RECORDED POSITION over the interval (LAW-03: the unit is stated, and the
    denominator is published beside the rate so no consumer has to guess it).

    `None` when there is nothing to divide by. `delta / max(1, positions_delta)` is the
    tempting divide-by-zero guard and it is exactly what this must not do: with no position
    recorded there is NO rate, and publishing `0.0` would tell a reader "this lever did not
    fire per position", which is a claim nobody measured. A NEGATIVE delta is passed through
    as measured — the atomics are monotonic, so a decrease is a wiring bug and a `max(0, …)`
    would hide it behind a plausible reading (the `actor_lag_negative` precedent).
    """
    if delta is None or positions_delta is None or positions_delta == 0:
        return None
    return delta / positions_delta


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
        # R137/CARD-CLEANSTOP-SAVE: the clean-completion latch. A plain bool, PUBLIC because
        # `train/loop.py`'s post-loop guard is its one consumer (`_clean_stop_already_saved`)
        # and a private name would make that read look like an intrusion. Set AFTER the leg-3
        # write (never before — see `_clean_stop_save`), and carrying NO set-once guard: the
        # leg has no internal latch because exactly-once is a property of the DRIVER, not of
        # a branch only a test can reach. Deliberately NOT the first-non-None shape
        # `record_terminal_eval_reason` uses.
        self.clean_stop_saved = False
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
        #: WP12-R F2: the memo behind `_microbatch_caps`, the graph-only cap thunk.
        self._resolved_caps: Any | None = None
        self._initial_policy_loss: float | None = None
        self._consec_high_gn = 0
        self._eval_round_last_step = -1

        # WP13-A gate state — every ring is caller-owned (the rules are stateless).
        self._wr_history: list[tuple[int, float]] = []
        self._draw_rate_history: list[float] = []
        self._loss_window: list[float] = []
        self._last_iter_games = 0
        # WP12-R Phase O (R164): the previous `iteration_complete` boundary's counter
        # readings, so the payload can publish an INTERVAL delta beside the cumulative
        # total. Seeded at 0 (pool start), the same baseline `_last_iter_games` uses.
        self._last_target_counters: dict[str, int] = dict.fromkeys(
            (*_TARGET_INTEGRITY_COUNTERS, _POSITIONS_COUNTER), 0,
        )
        # WP12-R Phase O (R152/R133): the TERMINAL round's outcome, latched set-once by
        # `drain._record_terminal_outcome` and read by the composition root.
        self._terminal_eval_reason: str | None = None
        self._run_started = self._clock.now()
        # `warns` is carried alongside checks/fires/skips so the warn-only sealbot posture
        # (operator G-3) is visible per-gate in every `monitor_gates` event, not silent.
        self._gate_stats: dict[str, dict[str, int]] = {
            name: {"checks": 0, "fires": 0, "skips": 0, "warns": 0} for name in GATE_NAMES
        }

        # Self-play stall watchdog — always armed (context law, LAW-16). Driven via
        # `.tick(...)` from step(); fires → best-effort snapshot to a DISTINCT path + exit.
        # NO code-side default (R1). This read used to be
        # `mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin")`, and the
        # production root passes `mixing_cfg={}` — so the default ALWAYS won, and it is
        # CWD-relative: a run launched from outside the repo root wrote its stall snapshot
        # into some unrelated `./checkpoints/`, not into its own `--out-dir`. Derived instead
        # from the trainer's own checkpoint directory, which is the object that actually
        # knows where this run's artifacts live (R98, derive at point of use).
        # Resolved at FIRE time, not here: the path is only needed if the watchdog actually
        # fires, and deferring keeps construction free of any trainer-attribute requirement.
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
            # Item 4(b): the stall abort saves WEIGHTS, not just positions. Routed through
            # the trainer's own stamped save path so the artifact is a real envelope-v2
            # checkpoint (LAW-12), not a bare state_dict nothing can load.
            save_model=lambda: self.trainer.save_checkpoint(self._last_loss_info or None),
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

    # ── the terminal-eval outcome latch (WP12-R Phase O, R152/R133) ───────────────────
    @property
    def terminal_eval_reason(self) -> str | None:
        """The TERMINAL eval round's typed reason, or `None` for a clean (or not-yet-run)
        terminal battery. Read once, by the composition root, after `close_out` returns.

        A `str` and never the reason ENUM: the train package may not import the eval
        package at all (repo_design §2, census-tested — which is why this docstring names no
        dotted path into it). The authority is NOT weakened by the crossing: the string is
        the enum member's own value, produced by the enum on the eval side and re-parsed by
        the enum at the process boundary, where an unregistered spelling is a loud
        `ValueError`. This layer transports the fact and never authors it.
        """
        return self._terminal_eval_reason

    def record_terminal_eval_reason(self, reason: str | None) -> None:
        """Record the terminal round's outcome. **FIRST NON-`None` WINS.**

        The semantic is stated as MEASURED, not as intended (RED-TEAM F-04 corrected an
        earlier version of this docstring, which said "SET-ONCE, first call wins" — that is
        NOT what the guard below does). The guard is `if self._terminal_eval_reason is not
        None: return`, so:

        * a first call with a REASON latches it, and every later call is refused —
          driven: `record("killed")` then `record("exit_nonzero")` → `"killed"`;
        * a first call with `None` (a CLEAN terminal round) latches nothing, so a later
          call CAN still write — driven: `record(None)` then `record("killed")` →
          `"killed"`, where a true set-once would have kept `None`.

        The stated purpose — "a later resolution must not re-label the outcome that stopped
        the run" — is therefore delivered **only for a non-`None` first call**. That is
        arguably the right rule here and it is why `ShutdownState.record_abort`'s shape was
        copied rather than its literal contract: `record_abort`'s value is always non-`None`,
        so for it the two semantics coincide, while `None` is a LEGAL clean-round value here.

        **The difference is inert today and NOT pinned, which is why it is written down.**
        There is exactly ONE writer in all of `src/` (`drain._record_terminal_outcome`,
        reachable only from `drain.run_terminal_eval`) and it is called at most once per run,
        so no production path calls this twice — but **no test in `tests/` calls it twice
        either**: O-07 censuses call SITES, not invocations. An inert difference nobody has
        pinned is how the next phase's regression gets in. `Q-O-LATCH-SET-ONCE-UNPINNED`.

        The one-writer property is what keeps R133's mid-run/terminal split structural — a
        mid-run broken round cannot reach this method at all, so it stays non-fatal by
        construction rather than by a conditional somebody can get wrong later.
        """
        if self._terminal_eval_reason is not None:
            return
        self._terminal_eval_reason = reason

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

        # O2: iteration-limit reached — CLEAN COMPLETION, and the THIRD save leg
        # (R137/CARD-CLEANSTOP-SAVE). This is the one OUTER-loop site that ACTS on the
        # clean-completion predicate — it is not the only site that EVALUATES it: the inner
        # burst-break runs the character-identical expression once per burst iteration, but
        # only `break`s the burst and leaves `running` True, so `step()` returns normally and
        # THIS arm fires on the next call. Acting on it is what makes this the only place the
        # save can sit, and it is why the leg lives here and not in
        # `drain.close_out` (it runs from a `finally`, i.e. on aborted exits too, and now
        # also latches the terminal-eval reason) nor after `loop.py`'s `while` (the loop
        # cannot tell WHY it exited, and `abort_rule` is still None there for the rc-48
        # class, which `run.py` records ~80 lines later).
        #
        # The save runs BEFORE `running = False` so the run's product is on disk before the
        # driver is told to stop. It does NOT make the filesystem a cleanliness proxy: the
        # disk guard (rc 47) and the terminal-eval reason (rc 48) are both recorded in
        # `run.py`'s teardown, strictly AFTER `close_out`, so a leg-3 artefact legally
        # coexists with a non-zero rc. Clean-vs-aborted is carried by
        # `ShutdownState.abort_rule` — which this leg neither reads nor writes — and by the
        # rc it resolves to, never by a file's presence.
        if cfg.stop_step is not None and self._train_step >= cfg.stop_step:
            self._clean_stop_save(cfg)
            self.shutdown.running = False
            return self._build_outcome(in_warmup=False, waiting_for_games=False,
                                       **{**base, "checkpoint_saved": True})

        # O3: shutdown-save (signal-handler flag) — save the checkpoint, then stop. The
        # `_try_save_buffer(..., "shutdown_signal", ...)` call that stood here is DELETED by
        # R178(a) (R116/LAW-08): F-CS-2 measured it a no-op on the production path
        # (`mixing_cfg={}`, nothing in `src/` sets `buffer_persist`), so removing it changes
        # no production behaviour and stops the signal leg claiming a save it never made.
        if self.shutdown.shutdown_save:
            self.trainer.save_checkpoint(self._last_loss_info or None)
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
        # Burst accumulators (item 7): the eval kick now runs PER TRAINING STEP inside the
        # burst, so its two outcomes are OR-folded across the burst exactly like the flags
        # above rather than being the single post-burst call's return.
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
            # NaN/inf is EXCLUDED from this abort, and that is a KNOWN GAP, not an oversight
            # — see ADJ-D13. Item 6 changed this line to
            # `if not math.isfinite(step_gn) or step_gn > cfg.hard_gn_threshold:` and it was
            # REVERTED for two reasons, both operator-owned:
            #   1. R56 — this exact comparison is a SOURCE PIN in
            #      `config/armed_aborts.py`'s `grad_norm_hard_abort` row. The preflight's own
            #      failure text is "re-adjudicate the row rather than editing the pin".
            #   2. The row is DEFERRED and knowingly disarmed: `train.hard_gn_threshold` is
            #      the unauthored 1e9 that no finite norm reaches. Making non-finite fire
            #      REGARDLESS of the threshold would partially ARM a row the manifest says
            #      run5 mints disarmed "knowingly and in writing" — an armed-value change.
            # The other two item-6 paths (the trainer's non-finite guard, and the alert rules)
            # DID land, so a NaN is caught and reported; only this backstop stays gated.
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

            # D4 IS DELETED (R178(a) / R116 / LAW-08). It was the checkpoint-cadence
            # replay-BUFFER save, gated on `cfg.checkpoint_interval` (fed by the now-deleted
            # `train.buffer_save_interval`), and F-CS-2 measured its `_try_save_buffer` call
            # a no-op on every production leg. Every committed config minted the interval at
            # `0`, so the arm never fired even before the helper's own early return — the
            # deletion moves no production behaviour. `checkpoint_saved` therefore stays
            # `False` for the whole burst path; the O2/O3 legs above still report `True`,
            # which is where a real checkpoint write is announced.

            # WP13-A: the log_interval boundary is tested PER TRAINING STEP (old-side parity,
            # `step_coordinator.py:1370/1383`). Testing it once per burst would skip every
            # boundary the post-burst step does not land exactly on, thinning BOTH the LAW-18
            # emission stream and the sampling cadence of the draw-rate gate by ~the mean
            # burst — the gate's `consec` window would silently stretch by that factor.
            axis_step, gate_fired = self._run_log_interval(cfg, loss_info)
            axis_emitted = axis_emitted or axis_step
            hard_abort_fired = hard_abort_fired or gate_fired

            # INSIDE the burst (item 7). `_maybe_kick_eval` tests
            # `self._train_step % cfg.eval_interval != 0`, and it used to run ONCE after the
            # whole burst — so with `max_train_burst > 1` a burst that steps over the exact
            # multiple (e.g. interval 500, burst 3, landing on 499 → 502) never satisfied the
            # modulo and the eval round was SILENTLY SKIPPED. Not delayed: skipped, because
            # `_eval_round_last_step` is keyed on the round index. Long runs could go many
            # intervals without an eval while the config said otherwise. Tested per training
            # step, the exact boundary is always hit.
            #
            # The kick return is NEVER consumed for WR and adds NO blocking call: completed
            # rounds reach `on_eval_round_complete` via the async drain (§c.4b).
            # `_maybe_kick_eval` consumes the ACK only for `eval_skipped_busy` (P-06 pin).
            kicked_step, skipped_step = self._maybe_kick_eval(cfg)
            eval_kicked_off = eval_kicked_off or kicked_step
            eval_skipped_busy = eval_skipped_busy or skipped_step

        # WP12R Step 3 narration (R210): `iteration_complete` emits at the O6 training-burst
        # return, per coordinator step, INDEPENDENT of `log_interval`. `training_step`
        # alerting stays `log_interval`-gated (above, in `_run_log_interval`); only
        # `iteration_complete` (the per-iteration counter) was decoupled — "games_total is
        # a per-iteration counter, not a training-logging event" (R210). NOT called on O2/O3
        # early returns (clean-completion / shutdown-save): those are not training-burst
        # returns (`loss_info` would be {}, `games_this_iter` degenerate).
        self._emit_iteration_complete(cfg)
        return self._build_outcome(
            in_warmup=False, waiting_for_games=False,
            **{**base, "steps_run": steps_budget, "buffer_resized": buffer_resized,
               "checkpoint_saved": checkpoint_saved, "eval_kicked_off": eval_kicked_off,
               "eval_skipped_busy": eval_skipped_busy,
               "hard_abort_fired": hard_abort_fired, "axis_emitted": axis_emitted},
        )

    # ── R137 leg 3: the clean-completion save ─────────────────────────────────────────
    def _clean_stop_save(self, cfg: StepCoordinatorConfig) -> None:
        """The CLEAN-COMPLETION save — the THIRD taxonomy leg (R137/CARD-CLEANSTOP-SAVE),
        beside the trainer's periodic cadence and the signal-driven `shutdown_save`.

        It is its OWN semantic, not a differently-triggered `shutdown_save`: leg 1's artefact
        means "a resumption point — the run continues past it", leg 2's means "a rescue of
        interrupted work — the run did not finish", and leg 3's means "the run FINISHED" —
        the terminal weights the terminal eval and the deploy tag are about. A leg that were
        merely shutdown_save fired differently would label a completed run as interrupted.

        LAW-12: this is the SAME `trainer.save_checkpoint` entry legs 1 and 2 call, so the
        run's product rides the ONE stamp path — `checkpoints.save_checkpoint(kind="full")`
        -> `_write_v2_payload`, config validated before any file exists, stamp built once and
        immutable, filename carrying run-id + content hash. No second write surface, no
        re-stamp, no added loader.

        LAW-14: a failure is NOT caught. `_write_v2_payload` counts it on
        `checkpoints.persist_errors_total` and re-raises; that counter is the persist-fatal
        watchdog's REGISTERED input (`monitor/producer_manifest.yaml`, id `persist_fatal`),
        and the watchdog is NOT disarmed during close-out. Catching here would be the
        silent-except LAW-14 forbids AND a second authority for a storage fault's exit code
        beside the registered chain that already owns `PERSIST_FATAL_EXIT_CODE`. This card
        authors no new exit code.

        The latch and the event both land AFTER the write, deliberately UNLIKE `loop.py`'s
        `_final_save`, which emits `shutdown_save` BEFORE its own: an event named for a save
        is a CLAIM the save happened, so a pre-emit puts a false record in the stream of
        every failed final save, and a pre-set latch would suppress leg 2 on a run whose
        leg-3 write died — turning one lost save into two. `loop.py`'s ordering is a queued
        row, not something silently mirrored here.

        The event publishes the WRITER'S returned path, never a directory string re-derived
        here, which could name a file that does not exist. `step` and `stop_step` are carried
        as two fields because they are two different facts: a resumed run past its cap fires
        this arm with `_train_step > stop_step` (LAW-18: "did the run's final save happen,
        and to what file" must be answerable from the ONE channel).
        """
        path = self.trainer.save_checkpoint(self._last_loss_info or None)
        self.clean_stop_saved = True
        emit_via(self._sink, {
            "event": "clean_stop_save",
            "step": self._train_step,
            "stop_step": cfg.stop_step,
            "path": None if path is None else str(path),
        })

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
        """At the `log_interval` boundary: emit the `training_step` payload, run the 4 WARN
        rules on it, run the two LIVE-producer hard-abort gates, and publish the LAW-18
        `monitor_gates` summary. Returns ``(axis_emitted, hard_abort_fired)``.

        WP12R Step 3 narration (R210): `iteration_complete` is NO LONGER emitted here. It
        moved to `_emit_iteration_complete` at the O6 training-burst return, per coordinator
        step, INDEPENDENT of `log_interval`. R210: "training_step alerting stays gated" —
        the `training_step` event, the WARN rules, `emit_axis_distribution`, the hard-abort
        gates and `monitor_gates` all STAY `log_interval`-gated here. Only
        `iteration_complete` (the per-iteration counter, not a training-logging event) was
        decoupled."""
        if not loss_info or cfg.log_interval <= 0 or self._train_step % cfg.log_interval != 0:
            return False, False
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
        fired = self._run_hard_abort_gates(cfg)
        self._emit_monitor_gates(cfg, sink)
        return axis is not None, fired

    def _emit_training_step(
        self, cfg: StepCoordinatorConfig, loss_info: dict[str, float], sink: Any
    ) -> dict[str, Any]:
        """Build + emit the `training_step` event (WP13-A §c.4) through the injected sink
        and return its payload (the 4 WARN rules read it). Split out of the old
        `_emit_training_events` (WP12R Step 3 narration, R210): `iteration_complete` moved to
        `_emit_iteration_complete` at the O6 return; this half STAYS `log_interval`-gated."""
        payload = emit_training_step_event(
            self._train_step, loss_info,
            # `quiescence_fires_per_step` has NO producer new-side (the solver-delta half is
            # DEFER/ARCH): the field travels as None = NOT MEASURED. A constant 0 would read
            # as a real measurement ("quiescence never fires") — a miniature F-10.
            None, sink,
        )
        return payload

    def _emit_iteration_complete(self, cfg: StepCoordinatorConfig) -> None:
        """Build + emit `iteration_complete` (the per-iteration counter payload) at the O6
        training-burst return, per coordinator step, INDEPENDENT of `log_interval` (WP12R
        Step 3 narration, R210). Carries `games_total`, `games_this_iter`, `buffer_size`,
        `corpus_selfplay_frac`, `batch_fill_pct`, the `target_integrity` block, plus
        `mcts_mean_depth` and the regime-gated cluster stats.

        R218 rider 1 (`Q-O-TWO-POOL-READS` collapse): takes the `RunnerStats` snapshot from
        `_target_integrity_report` and passes it INTO `emit_iteration_complete_event` as the
        `rstats` kwarg, so the builder does NOT make its own `pool.runner_stats()` call. ONE
        atomic snapshot per emit — the straddle between the `target_integrity` block and the
        `mcts_mean_depth`/cluster block is ELIMINATED (a semantic change, more correct than a
        no-op). See `_target_integrity_report`'s docstring for the cost disclosure.

        Updates `self._last_iter_games` so an "iteration" = one coordinator `step()` = one
        burst (the `games_this_iter` denominator). NOT called on O2/O3 early returns
        (clean-completion / shutdown-save) — those are not training-burst returns.
        """
        sink = self._sink if self._sink is not None else NullEventSink()
        w_pre = 0.0
        if self.pretrained_buffer is not None:
            w_pre = _compute_pretrained_weight(self._train_step, cfg.mixing_initial_w,
                                               cfg.mixing_min_w, cfg.mixing_decay_steps)
        rstats_report, rstats = self._target_integrity_report()
        emit_iteration_complete_event(
            self._train_step, w_pre, self._games_played, self._last_iter_games,
            self.pool, self.buffer, self.full_config, self.full_config.get("mcts", {}),
            cfg.capacity, self._games_per_hour, self._steps_per_hour,
            rstats_report, rstats, sink,
        )
        self._last_iter_games = self._games_played

    def _target_integrity_report(self) -> tuple[dict[str, Any], Any]:
        """The three Phase-T target-integrity counters as an `iteration_complete` block
        (WP12-R Phase O, R164 / LAW-18), and the `RunnerStats` snapshot they were built from.

        `PREREG_T §0b` names `export_offwindow_mass_moves` as THE in-run witness attributing
        the expected game-shape drift — and at HEAD that counter was readable only by a test
        calling `runner_stats(pool)`. A witness a live run cannot read is not a witness, and
        LAW-18's text is explicit that a post-hoc offline probe cannot distinguish "starved"
        from "ineffective". So each counter publishes its cumulative `total`, its INTERVAL
        `delta` and a `per_position` rate over the `positions_delta` denominator published
        beside it. An idle lever stays VISIBLE at 0 (the `chain_loss_with_fire_rate`
        posture): nothing is omitted for being zero, which is what keeps a permanently-0
        `target_integrity_defects` — its latch is run-fatal, so it reads 0 in every run that
        survives to emit — distinguishable from a field with no producer.

        Returns `(report, rstats)`: the report dict travels into `iteration_complete`'s
        `target_integrity` block; `rstats` (the SAME snapshot) travels into
        `emit_iteration_complete_event`'s `rstats` kwarg so the builder does NOT make its own
        `pool.runner_stats()` call (R218 rider 1, `Q-O-TWO-POOL-READS` collapse).

        COST, stated as MEASURED (WP12R Step 3 narration, R210 + R218 rider 1): this method
        makes ONE `pool.runner_stats()` FFI crossing per `iteration_complete` emit. After
        R210's decoupling `iteration_complete` emits per coordinator step (per burst), so this
        runs once per burst, NOT once per `log_interval` boundary. R218 rider 1 COLLAPSED the
        two reads (this one + `events.py:297`'s) into ONE — the builder no longer takes its
        own snapshot. One FFI crossing (~20 atomic loads) per burst, plus three subtractions
        and three divisions. At run5's `log_interval=1000` this ran on 1 step in 1000; per
        burst it runs on every `step()` return.

        SEMANTIC CHANGE (R218 rider 1): the collapse ELIMINATES the straddle. Before it,
        `target_integrity` came from THIS snapshot and `mcts_mean_depth`/cluster stats came
        from a second `runner_stats()` call inside `emit_iteration_complete_event`, taken
        microseconds apart with workers live — the two could straddle a game boundary (a
        counter increment between the reads). After the collapse, both blocks operate on the
        SAME atomic snapshot. This is more correct (guaranteed-consistent), not a no-op.
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

    def _games_per_hour(self) -> float:
        elapsed = self._clock.now() - self._run_started
        return (self._games_played / elapsed) * 3600.0 if elapsed > 0 else 0.0

    def _steps_per_hour(self) -> float:
        """R29 gap metric (b), the twin of `_games_per_hour` over the SAME clock: train
        steps per hour from the coordinator's own step counter. Published beside (a) in
        `iteration_complete` — the cutover floor's live emitter (WPBOX CB-3)."""
        elapsed = self._clock.now() - self._run_started
        return (self._train_step / elapsed) * 3600.0 if elapsed > 0 else 0.0

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
        here. The record is deliberately paired with `running = False` — it must be
        impossible to stop the run on a fired rule without recording which rule it was.

        WPMAIN RT-2/R132: the bare assignment became `ShutdownState.record_abort`, which is now
        the ONE writer of that field. Nothing about this path's behaviour moves — the state is
        `None` here on every reachable call (the `hard_abort_after_stop` arm above returns
        before it) so first-fire-wins records exactly what the assignment recorded. What
        changed is that the disk-guard leg gained a second fire path, and the set-once
        invariant this method's docstring already claimed is enforced by the carrier rather
        than by two call sites agreeing to be careful.
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
            # R-BUFFER-PERSIST-COUNTER (WPCLEAN Phase RES): the best-effort buffer-save
            # swallows are counted and read HERE, live — a module-attribute read, never a
            # from-import of the int (the subsystems.py counter-binding rule).
            "buffer_save_errors_total": int(_buffer_persist.buffer_save_errors_total),
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
            caps_provider=self._microbatch_caps,
        )

    def _step_spec(self) -> Any:
        """The resolved encoding spec, lazily resolved ONCE from the declared config this
        coordinator holds (`full_config`), through THE one resolver. An undeclared encoding
        raises `MissingEncodingError` — the LAW-11 posture, never a default arm."""
        if self._resolved_step_spec is None:
            self._resolved_step_spec = resolve_step_spec(self.full_config)
        return self._resolved_step_spec

    def _microbatch_caps(self) -> Any:
        """The resolved graph micro-batch caps, lazily resolved ONCE from the declared config
        this coordinator holds, through THE one resolver. Absence raises by name.

        Passed to the dispatcher as a CALLABLE — the BOUND METHOD, not a call — and invoked by
        the GRAPH arm only, so a grid run never reads `train`. That asymmetry is the whole
        point and it is not decoration: Python evaluates every argument before the call, so
        `caps_provider=self._microbatch_caps()` would resolve `full_config["train"]` on BOTH
        representations, and FOUR FROZEN test files build a `StepCoordinator` whose
        `full_config` is `{"identity": {...}}` with no `train` key at all. A graph-only knob
        must not make a grid config unloadable (WP12-R F2, DESIGN_DFIX §3.11.1).

        Memoised, mirroring `_step_spec` above — the burst calls this once per coordinator,
        not once per step. It mirrors `_step_spec` in MEMOISATION and deliberately NOT in call
        site: `_step_spec()` is invoked unconditionally because it DECIDES the route, while
        these caps are meaningful only on one branch OF that decision. Eager for the router,
        lazy for the routed.
        """
        if self._resolved_caps is None:
            self._resolved_caps = resolve_microbatch_caps(self.full_config)
        return self._resolved_caps

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
