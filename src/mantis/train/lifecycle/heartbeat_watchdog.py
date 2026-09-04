"""The INDEPENDENT heartbeat watchdog thread — L-B (repo_design §11 amendment; WP13-A §c.4).

The run3 lesson: the WP10 `StallWatchdog` is `tick()`-driven from the MAIN loop, so a
thread wedged INSIDE an eval call can never fire it — a 45 h livelock looked exactly like
a busy run. This watchdog owns its OWN daemon thread and reads heartbeat STALENESS, so a
wedge anywhere in the pipeline trips it with ZERO main-thread cooperation.

Three fire classes, two exit codes (the supervisor's restart key):
  * per-source staleness      → `WATCHDOG_STALL_EXIT_CODE` (42, transient → relaunch);
  * close-out overrun         → `WATCHDOG_STALL_EXIT_CODE` (42, same class);
  * `counters_fn() > 0`       → `PERSIST_FATAL_EXIT_CODE` (43, a storage fault → loud stop).

`disarm_staleness()` (called FIRST by `drain.close_out`) SWAPS the per-source deadlines for
ONE bounded close-out deadline; it does not switch staleness off. Rationale: close-out waits
are legally up to 14400 s ≫ the 1800 s per-source deadline, and during teardown the sources
legitimately stop beating (the pool is stopped, no training step runs), so per-source ages
would false-fire on every clean finish — but leaving teardown UNBOUNDED left both levels
blind, because the file mirror keeps advancing `seq` and the supervisor therefore reads a
wedged child as healthy. A single generous-but-finite teardown budget keeps a clean finish
quiet and still kills an unbounded teardown wedge. The persist-fatal fire and the
heartbeat-file mirror are NEVER disarmed.

DOCUMENTED LIMIT (O-14) — the GIL. This thread is the backstop for a wedged pipeline
thread, not for the interpreter itself: when the GIL is held inside non-yielding native
code, the watchdog thread is starved too, its poll loop stops, and the heartbeat file's
`seq` freezes. That frozen `seq` is precisely what the out-of-process SUPERVISOR
(`python -m mantis.monitor.supervise`) keys on — it kills and relaunches the child. The
in-process fire is level one; the supervisor is level two. Neither alone is sufficient.

`counters_fn` MUST read module ATTRIBUTES live (`checkpoints.persist_errors_total`), never
`from … import persist_errors_total` — that binds the int at import and reads a frozen 0
forever after `global … += 1` (O-28 pins the difference with a post-construction bump).

>300 justify: ONE unit — the watchdog thread — whose arming, three fire predicates, bounded
fire path and file mirror share the same mutable state (`_fired`, `_staleness_armed`,
`_close_out_started`, `_seq`, the counter registry) and must be read together to reason
about "can this process still die?". Splitting the predicates from `_fire` would put the
decision and its consequence in different files; the alternative to length here is a
distributed state machine, which is exactly the shape that hid the run3 livelock.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mantis.monitor.best_effort import BestEffortCounters, best_effort
from mantis.monitor.heartbeat import (
    ACTOR_LAG_EXIT_CODE,
    PERSIST_FATAL_EXIT_CODE,
    WATCHDOG_STALL_EXIT_CODE,
    write_heartbeat_file,
)

_LOG = logging.getLogger(__name__)

_THREAD_NAME = "heartbeat-watchdog"

#: Teardown budget armed by `disarm_staleness()` in place of the per-source deadlines. It
#: matches the drain/terminal-eval hard caps the close-out path is SUPPOSED to enforce
#: (`StepCoordinatorConfig.eval_final_drain_hard_cap_sec` / `terminal_eval_hard_cap_sec`,
#: 14400 s) — generous enough that no legitimate close-out is killed, finite so an
#: unbounded teardown wedge cannot leave BOTH watchdog levels blind. See the
#: R-DRAIN-HARDCAP-CONSUMERS debt: those config fields have no consumer, so this deadline
#: is currently the ONLY bound on a teardown wedge.
DEFAULT_CLOSE_OUT_DEADLINE_SEC: float = 14400.0

#: Hard time budget for ONE optional effect inside the fire path (snapshot, sink close).
#: `best_effort` catches exceptions, NOT hangs — without a bound, a wedged filesystem
#: suspends the fire before `exit_fn` and the process never dies.
DEFAULT_FIRE_EFFECT_TIMEOUT_SEC: float = 30.0

#: How many of a monitor's OWN poll intervals may pass with no completed check before the
#: monitor is called stalled (AUDIT-1 F-11 / R334(b)).
#:
#: WHY THIS EXISTS. A monitor thread that swallows its own errors — which every one of them
#: must, or a transient `statvfs` kills the run — is indistinguishable from a healthy one on
#: every observable it publishes, because the thing it stops publishing is the evidence. The
#: disk guard is the measured case: `check_once` raising on every tick emits no `disk_free`,
#: so absence-of-alert reads as "plenty of space", and the rc-47 abort stays armed in the
#: config for the whole run while the volume fills.
#:
#: WHY 3, and the grounds are that it separates SLOW from DEAD. One missed tick is a busy
#: volume; three consecutive missed ticks is a mechanism, and the guard's own counters say
#: which — `errors_total` advancing with `checks_total` frozen is a raising `check_once`,
#: both frozen is a thread that is gone. The bound is in the monitor's OWN interval, so a
#: 60 s guard is judged at 180 s and a 5 s one at 15 s; nothing here assumes a period.
#:
#: WHY IT IS NOT A CONFIG KEY, and this is `EARLIEST_FIRE_FRACTION`'s argument verbatim
#: (`mantis.config.armed_aborts`): a config that could set its own liveness deadline could
#: disable the check that says its monitors are dead, and the disarm this constant exists to
#: refuse would be re-spellable as a large number nobody read. It is also NOT a fire — see
#: `_check_monitor_liveness` — so it decides an OBSERVABLE, never a process exit.
MONITOR_STALL_INTERVALS: int = 3


@dataclass(frozen=True)
class MonitorSample:
    """One monitor's liveness reading: what it has completed, what it has swallowed, and the
    period both are denominated in. All three come from the monitor itself."""

    checks_total: int
    errors_total: int
    interval_sec: float


@dataclass(frozen=True)
class MonitorLivenessSpec:
    """A monitor this watchdog reports the liveness of (AUDIT-1 F-11 / R334(b)).

    `sample_fn` returns `None` while the monitor does not exist yet, which is a REAL state
    and not an error: the composition root starts this watchdog BEFORE it builds the disk
    guard (the pinned order is pool → watchdog, and the guard follows), so the first polls of
    every run legitimately have nothing to read. `None` is therefore "not known yet" and is
    silent; a monitor that appears and then stops advancing is the reportable thing.

    Read LIVE at poll time, never at build time — the O-28 discipline `ActorLagSpec` states
    one class up, and for the same reason: a value captured at construction would report the
    monitor's birth forever.
    """

    name: str
    sample_fn: Callable[[], MonitorSample | None]


@dataclass(frozen=True)
class ActorLagSpec:
    """The actor-lag invariant's inputs (WP-UNFREEZE §4): `learner_step_fn() −
    actor_ckpt_step_fn() > threshold_steps` → escalation when `abort_enabled`, else ONE
    loud event per exceedance episode. The QUANTITIES are step-clock; the SAMPLING rides
    the watchdog's existing seconds poll — deliberately NOT a fifth `HEARTBEAT_SOURCES`
    entry (a step-delta threshold in a seconds-deadline dict is a type lie)."""

    learner_step_fn: Callable[[], int]     # lambda: int(trainer.step)
    actor_ckpt_step_fn: Callable[[], int]  # actor_sync.actor_ckpt_step
    threshold_steps: int                   # N; from monitor.actor_lag_threshold_steps
    abort_enabled: bool                    # from monitor.actor_lag_abort_enabled


class HeartbeatWatchdog:
    """Poll `registry` staleness + the persist counters on an independent daemon thread.

    Every collaborator is injected (clock / sink / snapshot / exit_fn) so the whole fire
    path is deterministically testable: `arm()` and `poll_once()` are factored OUT of
    `start()`, so a fake clock drives exact staleness units with no sleeps.
    """

    def __init__(
        self,
        *,
        registry: Any,
        deadlines: Mapping[str, float],
        sink: Any,
        counters_fn: Callable[[], int],
        heartbeat_file: Path | str,
        file_interval_sec: float,
        poll_interval_sec: float,
        clock: Callable[[], float] = time.monotonic,
        save_snapshot: Callable[[], None],
        exit_fn: Callable[[int], None] = os._exit,
        close_out_deadline_sec: float = DEFAULT_CLOSE_OUT_DEADLINE_SEC,
        snapshot_timeout_sec: float = DEFAULT_FIRE_EFFECT_TIMEOUT_SEC,
        wired_sources: Sequence[str] | None = None,
        actor_lag: ActorLagSpec | None = None,
        monitor_liveness: Sequence[MonitorLivenessSpec] = (),
    ) -> None:
        self._registry = registry
        self._deadlines = dict(deadlines)
        self._sink = sink
        self._counters_fn = counters_fn
        self._heartbeat_file = Path(heartbeat_file)
        self._file_interval = float(file_interval_sec)
        self._poll_interval = float(poll_interval_sec)
        self._clock = clock
        self._save_snapshot = save_snapshot
        self._exit_fn = exit_fn
        self._close_out_deadline = float(close_out_deadline_sec)
        self._snapshot_timeout = float(snapshot_timeout_sec)
        # AUDIT-1 F-11 / R334(b). The default is `()` and NOT the required-with-no-default
        # posture `wired_sources` takes, because the two failure modes are different: an
        # undeclared heartbeat source makes a HEALTHY run fire 42, so it may never be
        # inferred; an undeclared monitor makes this watchdog quiet about a monitor it was
        # never told about, which costs an observable and no run. It is also not silent —
        # `arm()` emits `monitor_liveness_unwired` when the tuple is empty, and production's
        # wiring is pinned structurally rather than by anyone remembering the kwarg.
        self._monitor_liveness = tuple(monitor_liveness)
        #: name -> (clock at the last OBSERVED advance, checks_total at that advance).
        self._monitor_seen: dict[str, tuple[float, int]] = {}
        self._monitor_stalled: set[str] = set()
        self._last_monitor_sample: float | None = None

        sources = getattr(registry, "sources", None)
        self._sources: tuple[str, ...] = tuple(sources) if sources else tuple(self._deadlines)
        # A registry source with NO deadline entry is a WIRING BUG, not a default: silently
        # reading it as 0.0 (= disabled) would blind the watchdog to a whole pipeline stage
        # while it keeps mirroring a fresh `seq`, so the supervisor backstop would not fire
        # either. Fail LOUD at construction instead (R1: no code-side defaults).
        missing = [source for source in self._sources if source not in self._deadlines]
        if missing:
            raise ValueError(
                f"HeartbeatWatchdog: no deadline for heartbeat source(s) {missing}; "
                f"every registry source needs an explicit deadline "
                f"(use <= 0 to disable that source's staleness fire)"
            )
        # The composition root DECLARES which sources it actually handed `registry.beat` to.
        # A declared source is staleness-eligible from arm time (so a stage that dies before
        # its FIRST beat is still caught — the wedge coverage O-10/O-11 pin); an UNDECLARED
        # source that has never beaten is a WIRING GAP, not a wedge, and gets a loud
        # `heartbeat_source_unwired` event instead of a 42 that would kill a healthy run and
        # send the supervisor relaunching into the same missing kwarg until the budget dies.
        # Default None = "every registry source is wired", the conservative reading.
        self._wired: frozenset[str] = (
            frozenset(self._sources) if wired_sources is None else frozenset(wired_sources)
        )
        unknown = sorted(self._wired - set(self._sources))
        if unknown:
            raise ValueError(
                f"HeartbeatWatchdog: wired_sources names unknown heartbeat source(s) "
                f"{unknown}; known sources: {list(self._sources)}"
            )
        # None = no lag surveillance (direct-ctor tests / non-run contexts) — LOUD as
        # "absent" in the arm event, never silent.
        self._actor_lag = actor_lag
        self._lag_exceeded_latched = False
        self._lag_negative_reported = False
        self._counters = BestEffortCounters()
        self._staleness_armed = True
        self._close_out_started: float | None = None
        self._unwired_warned: set[str] = set()
        self._fired = False
        self._seq = 0
        self._last_file_write: float | None = None
        # LAW-18 (WPAX P / TD-6): the lag SAMPLE's own gate. Derived from `_file_interval`
        # above — no new ctor parameter, so one config fact never enters this constructor
        # twice under two names (LAW-08) and no construction site changes.
        self._last_lag_sample: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── public surface ────────────────────────────────────────────────────────────────
    @property
    def fired(self) -> bool:
        return self._fired

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def heartbeat_file(self) -> Path:
        return self._heartbeat_file

    def arm(self) -> None:
        """Grace-reset the registry and emit `heartbeat_watchdog_armed`. Spawns NO thread.

        The arm-log is UNCONDITIONAL and names every source with its deadline — including a
        source whose ``deadline <= 0`` disables its fire (the WP10 visibility law: a
        disabled or misconfigured watchdog must be loud, never silent).

        ``enabled`` is per-source AND honest: it reads ``False`` for a source that can NOT
        fire — a non-positive deadline, or an UNDECLARED source nothing has beaten (which is
        a wiring gap, not a watched stage). ``unwired_sources`` names exactly the latter and
        ``awaiting_first_beat`` names declared sources that have not beaten yet, so an
        incomplete wiring is readable at arm time instead of surfacing later as a false 42.
        """
        arm = getattr(self._registry, "arm", None)
        if arm is not None:
            arm()
        beaten = self._beaten_sources()
        watched = {s for s in self._sources if s in self._wired or s in beaten}
        self._emit({
            "event": "heartbeat_watchdog_armed",
            "sources": list(self._sources),
            "deadlines": {s: float(self._deadlines[s]) for s in self._sources},
            "enabled": {s: float(self._deadlines[s]) > 0.0 and s in watched
                        for s in self._sources},
            "wired_sources": sorted(self._wired),
            "unwired_sources": [s for s in self._sources if s not in watched],
            "awaiting_first_beat": [s for s in self._sources if s in watched and s not in beaten],
            # WP-UNFREEZE §4.3 visibility: a disabled or unwired lag check is loud at
            # arm time, never silent.
            "actor_lag": (
                {"armed": bool(self._actor_lag.abort_enabled),
                 "threshold_steps": int(self._actor_lag.threshold_steps)}
                if self._actor_lag is not None else "absent"
            ),
            # AUDIT-1 F-11 / R334(b): the monitors whose liveness this watchdog reports, and
            # the empty case named rather than absent — an unwired monitor is a gap somebody
            # must be able to see at arm time, exactly as an unwired heartbeat source is.
            "monitor_liveness": ([m.name for m in self._monitor_liveness]
                                 or "monitor_liveness_unwired"),
            "monitor_stall_intervals": MONITOR_STALL_INTERVALS,
            "poll_interval_sec": self._poll_interval,
            "file_interval_sec": self._file_interval,
            "close_out_deadline_sec": self._close_out_deadline,
            "snapshot_timeout_sec": self._snapshot_timeout,
            "heartbeat_file": str(self._heartbeat_file),
        })

    def start(self) -> None:
        """`arm()` then spawn the daemon poll thread (never joined at exit)."""
        self.arm()
        thread = threading.Thread(target=self._run, name=_THREAD_NAME, daemon=True)
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Stop the poll thread (test/teardown only — a fire never returns in production)."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def disarm_staleness(self) -> None:
        """Clean-shutdown entry: SWAP the per-source deadlines for ONE close-out deadline.

        Not an off switch. Teardown legitimately stops every heartbeat (the pool is stopped,
        no training step runs) and legitimately takes far longer than a per-source deadline —
        but an UNBOUNDED teardown left both watchdog levels blind, because the file mirror
        keeps advancing `seq` and the supervisor therefore reads a wedged child as healthy.
        Persist-fatal and the file mirror are untouched.
        """
        if not self._staleness_armed:
            return
        self._staleness_armed = False
        self._close_out_started = float(self._clock())
        self._emit({"event": "heartbeat_watchdog_staleness_disarmed",
                    "reason": "close_out", "sources": list(self._sources),
                    "close_out_deadline_sec": self._close_out_deadline})

    def poll_once(self) -> None:
        """ONE poll cycle: persist-fatal → actor-lag → staleness → file mirror (that order).

        Persist first: a storage fault is already fatal and its diagnosis is unambiguous,
        so it must not be masked by a staleness fire it probably caused. The actor-lag
        check runs iff staleness is armed (WP-UNFREEZE §4.2): during close-out training
        has stopped, both step counters freeze, and a teardown must never die to a stale
        lag reading.
        """
        if self._fired:
            return
        count = int(self._counters_fn())
        if count > 0:
            self._fire(PERSIST_FATAL_EXIT_CODE, reason="persist_fatal",
                       detail={"persist_errors_total": count})
            return
        if self._staleness_armed:
            # AUDIT-1 F-11 / R334(b). Gated on `_staleness_armed`, INSIDE the armed branch,
            # for a reason this leg learned by breaking the tier with the other placement.
            #
            # This check was first written to run on EVERY poll — armed or not — on the
            # argument that a monitor's death is not a pipeline stall and close-out does not
            # excuse it. That argument is about WHEN the reading is interesting. It ignored
            # WHO takes the reading: this watchdog's OWN thread, which emits through the sink
            # it also polices. A run tearing down closes its sink; an emit after that is a
            # failed write; `JsonlEventSink` COUNTS a failed write rather than raising
            # (LAW-14) — and the counter it increments is `counters_fn`, which is the FIRST
            # thing `poll_once` reads and which it answers with `os._exit(43)`. **A periodic
            # diagnostic emitted from the watchdog thread is self-fatal**, and it was measured
            # so: `monitor_liveness_sample` into a closed sink, then rc 43 on the next poll,
            # taking the whole test tier down from a leaked watchdog three files later.
            #
            # `_check_actor_lag` has always sat behind this same gate and
            # `test_no_sample_during_close_out_and_none_without_a_spec` says why in as many
            # words — the emission "inherits the two structural gates instead of becoming a
            # third, independently-wrong one". This is that third one, corrected.
            self._check_monitor_liveness()
            if self._check_actor_lag():
                return
            if self._check_source_staleness():
                return
        elif self._check_close_out_overrun():
            return
        self._mirror_file()

    def _check_monitor_liveness(self) -> None:
        """Report a monitor whose own counter has stopped advancing. NEVER fires (R334(b)).

        The shape is deliberate and it is the ruling's, not a preference. Making the disk
        guard a fifth `HEARTBEAT_SOURCES` member would put a monitor thread on an instrument
        whose stall code is **42, the TRANSIENT class the supervisor RELAUNCHES on** — so a
        guard raising every tick would stall-abort and be relaunched into the same broken
        state, a crash loop into a filling volume, on the leg whose whole purpose is stopping
        a run before the volume fills. It would also need a fifth REQUIRED schema key and a
        re-mint of all seven configs. This reads the counters the guard already publishes and
        says so out loud.

        LAW-18: the reading is emitted on a healthy run too, bounded by the same file
        interval `_mirror_file` and the lag sample use, so an observer can tell a live
        reading from a frozen one. The stall event itself is LATCHED per monitor and clears
        when the counter advances again, so a long outage is one event and a recovery is
        visible.
        """
        if not self._monitor_liveness:
            return
        now = float(self._clock())
        readings: list[dict[str, Any]] = []
        for spec in self._monitor_liveness:
            sample = spec.sample_fn()
            if sample is None:
                # Not built yet. A real state, not an error: the root starts this watchdog
                # before it constructs the guard.
                continue
            seen = self._monitor_seen.get(spec.name)
            if seen is None or sample.checks_total > seen[1]:
                self._monitor_seen[spec.name] = (now, int(sample.checks_total))
                if spec.name in self._monitor_stalled:
                    self._monitor_stalled.discard(spec.name)
                    _LOG.warning("monitor_recovered name=%s checks_total=%s",
                                 spec.name, sample.checks_total)
                    self._emit({"event": "monitor_recovered", "monitor": spec.name,
                                "checks_total": int(sample.checks_total),
                                "errors_total": int(sample.errors_total)})
                seen = self._monitor_seen[spec.name]
            age = now - seen[0]
            deadline = MONITOR_STALL_INTERVALS * float(sample.interval_sec)
            readings.append({"monitor": spec.name, "checks_total": int(sample.checks_total),
                             "errors_total": int(sample.errors_total),
                             "last_advance_age_sec": round(age, 3),
                             "stall_deadline_sec": round(deadline, 3)})
            if deadline > 0.0 and age > deadline and spec.name not in self._monitor_stalled:
                self._monitor_stalled.add(spec.name)
                _LOG.error(
                    "monitor_stalled name=%s checks_total=%s errors_total=%s "
                    "last_advance_age_sec=%.1f deadline_sec=%.1f — this monitor has "
                    "published nothing for %d of its own intervals; the abort it feeds is "
                    "armed in the config and absent in effect",
                    spec.name, sample.checks_total, sample.errors_total, age, deadline,
                    MONITOR_STALL_INTERVALS,
                )
                self._emit({"event": "monitor_stalled", "monitor": spec.name,
                            "checks_total": int(sample.checks_total),
                            "errors_total": int(sample.errors_total),
                            "last_advance_age_sec": round(age, 3),
                            "stall_deadline_sec": round(deadline, 3),
                            "stall_intervals": MONITOR_STALL_INTERVALS})
        if not readings:
            return
        if (self._last_monitor_sample is None
                or (now - self._last_monitor_sample) >= self._file_interval):
            self._last_monitor_sample = now
            self._emit({"event": "monitor_liveness_sample", "seq": self._seq,
                        "monitors": readings})

    def _check_actor_lag(self) -> bool:
        """The WP-UNFREEZE lag invariant. Returns True when a fire was issued.

        Both callables are read LIVE on every poll (the O-28 discipline — a value
        captured at ctor/arm would read a frozen delta forever). Disarmed exceedance is
        ONE loud event per episode (latched; the latch resets once lag re-enters the
        threshold). A negative lag is a wiring bug being reported honestly, never a fire.
        """
        spec = self._actor_lag
        if spec is None:
            return False
        learner_step = int(spec.learner_step_fn())
        actor_step = int(spec.actor_ckpt_step_fn())
        lag = learner_step - actor_step
        detail = {"learner_step": learner_step, "actor_ckpt_step": actor_step,
                  "lag_steps": lag, "threshold_steps": int(spec.threshold_steps)}
        # LAW-18: the lag invariant is a lever under test, so it logs its own reading
        # in-run, not only when it fires. Before this, a healthy run emitted NOTHING from
        # this check (`actor_lag_negative` / `actor_lag_exceeded` are the only two arms), so
        # no observer could tell a live reading from a frozen 0 — the exact discrimination
        # the mint preflight (R61 assertion (b)) has to make. It is the SAME `detail` dict
        # the fire path uses, so a sample can never disagree with the reading that fires.
        # Emitted BEFORE the `lag < 0` arm on purpose: b5a reads `lag_steps < 0` off the
        # SAMPLE, and a sample placed after that arm is silenced on exactly the wiring
        # defect it exists to expose. Bounded by the interval ALREADY in this object — the
        # same one `_mirror_file` uses (one rule, two consumers) — so a 5 s poll cannot
        # flood the segment.
        now = float(self._clock())
        if (self._last_lag_sample is None
                or (now - self._last_lag_sample) >= self._file_interval):
            self._last_lag_sample = now
            self._emit({"event": "actor_lag_sample", "seq": self._seq, **detail})
        if lag < 0:
            if not self._lag_negative_reported:
                self._lag_negative_reported = True
                _LOG.error("actor_lag_negative learner_step=%s actor_ckpt_step=%s",
                           learner_step, actor_step)
                self._emit({"event": "actor_lag_negative", **detail})
            return False
        if lag > spec.threshold_steps:
            if spec.abort_enabled:
                self._fire(ACTOR_LAG_EXIT_CODE, reason="actor_lag_exceeded", detail=detail)
                return True
            if not self._lag_exceeded_latched:
                self._lag_exceeded_latched = True
                _LOG.error("actor_lag_exceeded (disarmed) lag_steps=%s threshold_steps=%s",
                           lag, spec.threshold_steps)
                self._emit({"event": "actor_lag_exceeded", "armed": False, **detail})
            return False
        self._lag_exceeded_latched = False
        return False

    def _check_source_staleness(self) -> bool:
        """Per-source staleness. Returns True when a fire was issued."""
        ages = self._registry.ages()
        beaten = self._beaten_sources()
        for source in self._sources:
            # PER-SOURCE off switch: `deadline <= 0` disables THAT source's fire and
            # nothing else — the other sources stay armed (DESIGN §c.4, PREREG P-16,
            # repo_design §11 "per-source"). A global kill-switch must never be an
            # emergent property of one zeroed field.
            deadline = float(self._deadlines[source])
            if deadline <= 0.0:
                continue
            # UNDECLARED-and-never-beaten is a wiring gap, not a wedge (see `_wired`): it
            # gets a loud non-fatal signal, never a 42 that would kill a healthy run.
            if source not in self._wired and source not in beaten:
                self._warn_unwired(source, float(ages.get(source, 0.0)), deadline)
                continue
            age = float(ages.get(source, 0.0))
            if age >= deadline:
                self._fire(
                    WATCHDOG_STALL_EXIT_CODE,
                    reason=f"heartbeat_stale:{source}",
                    detail={"stale_source": source, "age_sec": round(age, 3),
                            "deadline_sec": deadline},
                )
                return True
        return False

    def _check_close_out_overrun(self) -> bool:
        """The teardown budget armed by `disarm_staleness()`. True when a fire was issued."""
        if self._close_out_started is None or self._close_out_deadline <= 0.0:
            return False
        elapsed = float(self._clock()) - self._close_out_started
        if elapsed >= self._close_out_deadline:
            self._fire(
                WATCHDOG_STALL_EXIT_CODE,
                reason="close_out_timeout",
                detail={"elapsed_sec": round(elapsed, 3),
                        "close_out_deadline_sec": self._close_out_deadline},
            )
            return True
        return False

    def _beaten_sources(self) -> frozenset[str]:
        """Sources that have been beaten at least once. A registry without the capability
        (a duck-typed stub) is treated as all-beaten — the conservative reading, since the
        never-beaten carve-out may only ever SUPPRESS a fire on a source we know is unwired.
        """
        beaten = getattr(self._registry, "beaten_sources", None)
        if not callable(beaten):
            return frozenset(self._sources)
        return frozenset(cast("Iterable[str]", beaten()))

    def _warn_unwired(self, source: str, age: float, deadline: float) -> None:
        """Emit ONCE per source: this source has never beaten past its own deadline, so the
        wiring is incomplete. Loud and actionable, but NOT a fire (killing a healthy run
        because a `heartbeat=` kwarg was forgotten is the worse failure)."""
        if age < deadline or source in self._unwired_warned:
            return
        self._unwired_warned.add(source)
        _LOG.error(
            "heartbeat_source_never_beaten source=%s deadline_sec=%.1f — this pipeline stage "
            "is NOT wired to the heartbeat registry; the watchdog is blind to it",
            source, deadline,
        )
        self._emit({
            "event": "heartbeat_source_unwired",
            "source": source,
            "deadline_sec": deadline,
            "detail": "never beaten since arm; staleness fire suppressed for this source",
        })

    # ── internals ─────────────────────────────────────────────────────────────────────
    def _run(self) -> None:
        """The poll loop. An unexpected exception HERE fires 42 `watchdog_error` — a dead
        watchdog must never be silent (a silent one is indistinguishable from a healthy
        run right up to the moment it was needed)."""
        while not self._stop.is_set() and not self._fired:
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 — a dead watchdog is never silent
                _LOG.exception("heartbeat_watchdog_error")
                self._fire(WATCHDOG_STALL_EXIT_CODE, reason="watchdog_error",
                           detail={"exc": repr(exc)})
                return
            self._stop.wait(max(self._poll_interval, 0.0))

    def _mirror_file(self) -> None:
        """Publish the freshest ages to the heartbeat FILE with a monotonic ``seq``.

        The `seq` is the supervisor's liveness signal: it advances iff THIS thread still
        runs. The write is best-effort — a transient FS failure must not kill a healthy
        run; a persistent one freezes `seq`, which is the supervisor's cue.
        """
        now = float(self._clock())
        if self._last_file_write is not None and self._file_interval > 0.0 \
                and (now - self._last_file_write) < self._file_interval:
            return
        self._last_file_write = now
        self._seq += 1
        ages = self._registry.ages()
        best_effort(
            "watchdog_file_mirror",
            lambda: write_heartbeat_file(self._heartbeat_file, seq=self._seq, pid=os.getpid(),
                                         ages=ages, wall_ts=time.time()),
            counters=self._counters,
        )

    def _fire(self, code: int, *, reason: str, detail: Mapping[str, Any] | None = None) -> None:
        """LOUD event → BOUNDED snapshot → outcome event → BOUNDED sink close → exit.

        Guarantee, stated exactly (restated at WPCLEAN Phase RES — the WP13-A N3 row): the
        fire runs ENTIRELY on the watchdog thread (O-10/O-14 — the wedged main thread is
        never asked to cooperate) and reaches ``exit_fn`` even when an optional effect
        RAISES (`best_effort` counts it). The HANG bound is narrower and holds for exactly
        TWO of the four effects: the snapshot and the sink close run under `_bounded`'s hard
        `snapshot_timeout_sec` budget on their own worker threads; the two `_emit` calls are
        exception-safe but NOT time-bounded — a sink whose ``emit`` wedges (hung log
        filesystem) still suspends the fire. That residual is deliberate here (bounding the
        emits is a fire-path timing change carded with N1's cross-check, not a docstring
        fix) and its backstop is external: a suspended fire stops the heartbeat-file writer
        with it, which is exactly the staleness the process-level supervisor acts on (N3's
        own row: "supervisor backstop covers").

        Order note: the `.watchdog` snapshot now runs BEFORE the sink close so its OUTCOME can
        be recorded in the ONE channel (`heartbeat_watchdog_fire_complete`). Previously a
        failed or skipped snapshot left no JSONL trace at all — only a stderr WARN moments
        before `os._exit`. The sink is line-buffered, so nothing already emitted is at risk.
        """
        if self._fired:
            return
        self._fired = True
        payload: dict[str, Any] = {
            "event": "heartbeat_watchdog_fired",
            "reason": reason,
            "code": int(code),
            "ages": self._safe_ages(),
            "staleness_armed": self._staleness_armed,
            "seq": self._seq,
        }
        if detail:
            payload.update(dict(detail))
        _LOG.error("heartbeat_watchdog_fired reason=%s code=%s payload=%s", reason, code, payload)
        self._emit(payload)

        snapshot_ok = self._bounded("watchdog_snapshot", self._save_snapshot)
        self._emit({
            "event": "heartbeat_watchdog_fire_complete",
            "reason": reason,
            "code": int(code),
            "snapshot_ok": snapshot_ok,
            "best_effort_counters": self._counters.snapshot(),
        })
        close = getattr(self._sink, "close", None)
        if close is not None:
            self._bounded("watchdog_sink_close", close)
        self._exit_fn(int(code))

    def _bounded(self, label: str, fn: Callable[[], Any]) -> bool:
        """Run ONE optional fire-path effect under a hard time budget on its own thread.

        Returns True iff it completed without raising. A TIMEOUT is counted under
        ``<label>_timeout`` and abandoned (the worker is a daemon): the fire must reach
        ``exit_fn`` even on a wedged filesystem, so waiting forever is not an option.
        """
        outcome: list[bool] = []

        def _run() -> None:
            ok, _ = best_effort(label, fn, counters=self._counters)
            outcome.append(ok)

        worker = threading.Thread(target=_run, name=f"{_THREAD_NAME}-{label}", daemon=True)
        worker.start()
        worker.join(timeout=max(self._snapshot_timeout, 0.0))
        if worker.is_alive():
            self._counters.increment(f"{label}_timeout")
            _LOG.error("watchdog_fire_effect_timeout label=%s budget_sec=%.1f — proceeding to "
                       "exit; the effect is abandoned", label, self._snapshot_timeout)
            return False
        return bool(outcome and outcome[0])

    @property
    def counters(self) -> BestEffortCounters:
        """The fire path's best-effort counter registry (published in
        `heartbeat_watchdog_fire_complete`, readable by a LAW-18 summary)."""
        return self._counters

    def _safe_ages(self) -> dict[str, float]:
        ok, ages = best_effort("watchdog_ages", self._registry.ages, counters=self._counters)
        return {k: round(float(v), 3) for k, v in ages.items()} if ok and ages else {}

    def _emit(self, event: Mapping[str, Any]) -> None:
        sink = self._sink
        if sink is None:
            return
        best_effort("watchdog_emit", lambda: sink.emit(dict(event)), counters=self._counters)
