# >300 justify (R8): the ONE composition authority — collaborator builder, launcher and the
# signal/watchdog/disk-guard legs; split, a SECOND site could build a collaborator set or a
# `StepCoordinatorConfig`, which the one-authority tests forbid. Its bulk is per-decision rationale.
"""mantis.run — the run composition root AND the run launcher (design §a.4/§c.6).

The ONE module importing both `mantis.train` and `mantis.eval` at module top level; nothing
imports it, so it is a source-only DAG node. `python -m mantis.run --config <path>
--out-dir <path>` loads the config through the one loader, composes the run, drives the live
loop, and maps a fired hard abort to a process rc through the same `exit_code_for_abort` the
mint preflight's child reads. The preflight child calls the SAME `build_run_collaborators` and
`compose_run`, so no divergent boot path exists. `compose_run` is INJECTION-FIRST, but no
parameter may carry a CONFIG FACT: `eval_enabled`, `run_id` and the device are read from the
validated config. The actor-sync engine is built UNCONDITIONALLY here.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch

from mantis._engine import HexgBuffer
from mantis.config.armed_aborts import (
    DISK_GUARD_LIVENESS_PROBE,
    DISK_SPACE_ABORT_RULE,
    TERMINAL_EVAL_BROKEN_ABORT_RULE,
    ArmingSurfaceMissingError,
    ProducerProbeMissingError,
    audit_arming_live,
    exit_code_for_abort,
)
from mantis.config.emit import resolve_config, write_resolved_config
from mantis.config.loader import config_identity_sha256, load_config
from mantis.config.preflight_stamp import require_preflight_stamp
from mantis.config.resolve.actor_sync import resolve_actor_sync_cadence
from mantis.config.resolve.allocator_posture import (
    assert_allocator_posture as _assert_allocator_posture,
)
from mantis.config.resolve.allocator_posture import (
    declared_allocator_posture as _declared_allocator_posture,
)
from mantis.config.resolve.allocator_posture import (
    governs_device as _posture_governs_device,
)
from mantis.config.resolve.bootstrap import resolve_bootstrap
from mantis.config.resolve.composition import require_run_config, revalidate_run_config
from mantis.config.resolve.coordinator import CoordinatorKnobsSpec, resolve_coordinator_knobs
from mantis.config.resolve.disk_guard import resolve_disk_guard
from mantis.config.resolve.drain import DrainCapsSpec, resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec, resolve_draw_rate_abort
from mantis.config.resolve.fused_graph_caps import resolve_fused_graph_caps
from mantis.config.resolve.heldout_gap import resolve_heldout_gap
from mantis.config.resolve.inference_batching import resolve_inference_batching
from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
from mantis.config.resolve.monitor import resolve_monitor_config
from mantis.config.resolve.ply_cap import PlyCapAbortSpec, resolve_ply_cap_abort
from mantis.config.resolve.policy_loss_trough import (
    PolicyLossTroughAbortSpec,
    resolve_policy_loss_trough_abort,
)
from mantis.config.resolve.run_length import resolve_max_train_steps
from mantis.config.resolve.search import resolve_deploy_search_kind
from mantis.config.schema import RunConfig
from mantis.config.schema.core import derived_visit_capacity
from mantis.eval.errors import EvalBrokenReason
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.monitor.game_recorder import GameRecorder
from mantis.monitor.logging_setup import configure_logging
from mantis.selfplay.pool import WorkerPool
from mantis.train.actor_sync import ActorSync
from mantis.train.anchor import (
    _guarded_load_state_dict,
    canonical_anchor_path,
    save_best_model_atomic,
)
from mantis.train.buffer_persist import canonical_buffer_path
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.dispatch import RepresentationRouteError
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.emit import NullEventSink, emit_via
from mantis.train.heldout import HeldoutSlice
from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.train.lifecycle.heartbeat_watchdog import (
    MonitorLivenessSpec,
    MonitorSample,
)
from mantis.train.lifecycle.signals import (
    ParentDeathDecision,
    ShutdownState,
    arm_parent_death_if_supervised,
    install_signal_handlers,
    last_parent_death_decision,
)
from mantis.train.loop import run_training_loop
from mantis.train.orchestrator import init_trainer
from mantis.train.resume_state import (
    RingIdentityError,
    load_resume_state,
    restore_rng_streams,
    sidecar_path_for,
    verify_ring,
)
from mantis.train.subsystems import build_run_safety
from mantis.train.warmstart import resolve_bc_warm_start
from mantis.util.determinism import seed_everything

#: The 3 pipeline stages every run wires unconditionally; "eval_round" joins them iff an
#: eval pipeline is actually built (the caller DECLARES what it handed `heartbeat=` to).
_BASE_WIRED_SOURCES: tuple[str, ...] = ("train_step", "inference_dispatch", "selfplay_drain")

#: `DiskGuard.keep_all` gets NO config key: it is a PRUNING knob the safety thresholds
#: deliberately ignore, and passing it explicitly here is the disclosure of that.
_DISK_GUARD_KEEP_ALL = False


class UnregisteredAbortExitError(RuntimeError):
    """A hard-abort rule FIRED and the manifest authors no exit code for it."""


class RunHandles(NamedTuple):
    """What `compose_run` hands back — enough for a caller to inspect or drive further."""

    coordinator: Any
    run_safety: Any
    eval_pipeline: Any | None
    shutdown: ShutdownState


class _DeferredSink:
    """Late-binding sink adapter: satisfies both `EventSink` Protocols via `emit(Mapping)`."""

    def __init__(self) -> None:
        self._inner: Any = NullEventSink()

    def bind(self, real: Any) -> None:
        self._inner = real

    def emit(self, event: Mapping[str, Any]) -> None:
        self._inner.emit(event)


class _DeferredHeartbeat:
    """Late-binding heartbeat adapter for the pool, bound in `compose_run`.

    Heartbeat-ONLY, a separate class from `_DeferredSink`. The `bound` flag is the composition
    root's verification surface: a pool that got `heartbeat=None` has dead producers, so the
    watchdog arms on sources nothing feeds and fires rc 42 at 1800 s on every healthy run.
    """

    def __init__(self) -> None:
        self._inner: Callable[[str], None] = lambda _source: None
        self.bound: bool = False

    def bind(self, real: Callable[[str], None]) -> None:
        self._inner = real
        self.bound = True

    def __call__(self, source: str) -> None:
        self._inner(source)


def _assert_pool_producers_live(pool: Any) -> None:
    """Refuse to arm the watchdog unless the pool's `_DeferredHeartbeat` is bound.

    `_BASE_WIRED_SOURCES` declares `inference_dispatch` + `selfplay_drain` wired
    unconditionally and their producers beat via `pool._heartbeat`, so a pool built with
    `heartbeat=None` arms the watchdog on phantom sources. A fake without one is SKIPPED.
    """
    if not hasattr(pool, "_heartbeat"):
        return  # test fake — no producer surface; skip (harness, not subject)
    hb = pool._heartbeat
    if isinstance(hb, _DeferredHeartbeat) and hb.bound:
        return  # the WIRED route: bound adapter forwards beats to the registry
    raise RuntimeError(
        "R208 producer-liveness conjunct: _BASE_WIRED_SOURCES declares pool-backed "
        "sources (inference_dispatch, selfplay_drain) but the pool's heartbeat is not a "
        f"bound _DeferredHeartbeat (got {hb!r}); the watchdog would arm on phantom "
        "producers — the rc-34 false-positive abort class (LAW-07). Either the pool was "
        "constructed with heartbeat=None (the pre-R208 defect) or the bind at compose_run "
        "never ran (build_run_safety raised before it)."
    )


class RunCollaborators(NamedTuple):
    """The three injected collaborators plus the ONE in-boot derivation of the output dirs.

    One derivation survives OUTSIDE the boot, in the preflight PARENT, so renaming the child's
    directory silently stops that parent's stale-segment guard guarding.
    """

    trainer: Any
    pool: Any
    buffer: Any
    log_dir: Path
    checkpoint_dir: Path
    #: The resume sidecar this boot restored FROM (None when fresh), carried, never re-read:
    #: a second read would be a second authority for what this boot resumed from.
    resume_state: Any = None


@contextmanager
def _seam(name: str) -> Iterator[None]:
    """NAME a composition seam without catching anything.

    Annotates the in-flight exception with a PEP 678 note and re-raises it unchanged, so the
    preflight's rc-32 sniff still reads the original final line. Seamed: the four builder
    steps and the eleven composer steps. NOT seamed, deliberately: the two config validators
    and the `Path`/`mkdir` lines (their own named refusals), and `run_training_loop`/`close_out`
    (the DRIVE — a seam note there would mislabel a step failure as a boot wall).
    """
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 — annotate-and-re-raise; nothing is caught
        exc.add_note(f"composition seam: {name}")
        raise


def _stop_pool_if_start_attempted(pool: Any, *, start_attempted: bool) -> Callable[[], None]:
    """Stop the pool iff `start()` was CALLED, not iff it returned.

    An unstarted pool's `InferenceServer.join(timeout=5.0)` raises on a never-started thread.
    "Called" and not "returned" because `start()` is three sub-starts: a raise in #2 or #3 left
    a half-started pool that a set-after flag reported as never started, and its workers leaked.
    """
    def _stop() -> None:
        if start_attempted:
            pool.stop()
    return _stop


def _select_buffer(config: Any, capacity: int) -> Any:
    """Select the replay buffer off `config.identity.representation`; an unknown or absent
    representation RAISES — never sniffed off a live module, never defaulted.

    This raise is the boot's one identity-key refusal, and it lives under `src/` so CI gate 11 can
    see it. `RepresentationRouteError` is REUSED from the train-step route — one error family
    per axis — and carries no `rc`: a `src/` exception carrying a CI tool's exit code is the
    layering defect this closes. (That tool error class is named by description and never
    spelled, because `tests/test_run_buffer_route.py` scans this function's source.)

    BOTH ARMS SEED THE RING'S SAMPLER FROM `config.seed` here, not at the caller: the sampler
    is a Rust `StdRng` seeded from OS entropy at construction, so without this two launches of
    one config drew different batch sequences. It does not resume the pre-stop stream.
    """
    representation = config.identity.representation
    if representation == "graph":
        buffer = HexgBuffer(capacity, config.identity.encoding, derived_visit_capacity(config))
        buffer.seed_sampler(config.seed)
        return buffer
    raise RepresentationRouteError(
        f"identity.representation {representation!r} selects no buffer — an absent or "
        "unknown representation is an ERROR, never a default (LAW-11)"
    )


def _restore_resume_state(buffer: Any, checkpoint_path: str) -> Any:
    """Load the sidecar beside `checkpoint_path`, verify the ring, and reload it into `buffer`.

    Raises:
        ResumeStateError: the sidecar exists but is malformed, or names another checkpoint.
        RingIdentityError: the persisted ring hashes differently than recorded, or reloads a
            different number of positions."""
    # The SIDECAR'S PRESENCE alone tells a resume from a warm start (`--resume-from` means both);
    # its absence is announced, since a CONTINUATION must never refill from empty quietly.
    if not sidecar_path_for(checkpoint_path).exists():
        _LOG.warning(
            "resume_state_absent checkpoint=%s — no sidecar beside this checkpoint, so it is "
            "read as a WARM START, not a continuation: the replay ring starts EMPTY and the "
            "round counter starts at zero. If this was meant to continue a stopped run, that "
            "run did not write a sidecar and its ring is gone", checkpoint_path,
        )
        return None
    state = load_resume_state(checkpoint_path)
    if state.ring is None:
        _LOG.warning(
            "resume_state_no_ring checkpoint=%s — the sidecar records no persisted ring, so "
            "this resume starts from an empty replay buffer", checkpoint_path,
        )
        return state
    verify_ring(state.ring)
    loaded = int(buffer.load_from_path(state.ring.path))
    if loaded != state.ring.positions:
        raise RingIdentityError(
            f"persisted ring {state.ring.path} reloaded {loaded} positions but the resume "
            f"sidecar recorded {state.ring.positions} — the file verified by hash and then "
            "disagreed on its own contents, which means the sidecar does not describe it"
        )
    restored = restore_rng_streams(state.rng)
    _LOG.info(
        "resume_state_restored step=%d ring_positions=%d round_counter=%d rng=%s",
        state.step, loaded, state.round_counter, ",".join(sorted(restored)),
    )
    return state


def build_run_collaborators(
    *, config: RunConfig, out_dir: str | Path, checkpoint_path: str | None = None,
) -> RunCollaborators:
    """Build the three injected collaborators and derive the run's output directories.

    Lifted verbatim in sequence from the preflight child's own boot, so every composition step
    both callers take is one of this module's two functions.

    NO `device` PARAMETER: the device is the config fact `config.train.device`. What must be
    unrepresentable is a caller pointing the boot at a different device than the config
    declares — how a `--device cpu` preflight false-cleared a cuda-minted run's GPU wall.
    `checkpoint_path` is a LAUNCH fact, so it is a parameter and not a schema key: two runs
    from one minted config, one fresh and one resumed, are the same config.
    """
    # The ONE determinism boot site: seed before any RNG-consuming object exists.
    seed_everything(config.seed)

    out_dir = Path(out_dir)
    log_dir = out_dir / "logs"
    checkpoint_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # The run records its COMPLETE resolved config, including every leaf the shipped file
    # left to a schema default, BEFORE anything can wedge. Persistence-fatal.
    with _seam("resolved_config record"):
        write_resolved_config(config, out_dir)

    device = torch.device(config.train.device)
    # The allocator-posture assertion precedes the first CUDA allocation: caps run under the other
    # regime differed by 3.62 GiB of card high-water (measured 2026-08-22); non-cuda is exempt.
    _assert_allocator_posture(config.model_dump(), device_type=device.type)
    with _seam("init_trainer"):
        # The SAME late-binding sink adapter the WorkerPool gets; pre-bind emissions (resume-time
        # events fired in this builder) still drop, per the adapter's semantics.
        trainer = init_trainer(config=config.model_dump(), checkpoint_dir=str(checkpoint_dir),
                               device=device, sink=_DeferredSink(),
                               checkpoint_path=checkpoint_path)
    capacity = int(resolve_coordinator_knobs(config.train).capacity)
    with _seam("_select_buffer"):
        buffer = _select_buffer(config, capacity)
    # THE RING IS A RESUME INPUT, loaded before `WorkerPool` can push into it; on a resume a
    # failure PROPAGATES — it must not quietly become a fresh run.
    resume_state = None
    if checkpoint_path is not None:
        with _seam("restore_resume_state"):
            resume_state = _restore_resume_state(buffer, checkpoint_path)
    # The self-play recorder, built before the pool so every game from step 0 is written and an
    # un-openable store is a loud STARTUP failure.
    with _seam("GameRecorder"):
        recorder = GameRecorder(record_dir=log_dir / "games", run_id=config.run_id,
                                seed=config.seed)
    with _seam("WorkerPool"):
        # Unchanged debt: the pool still builds only via the legacy hparams dict path
        # elsewhere, so it is handed `config.model_dump()`.
        pool = WorkerPool(model=trainer.model, config=config.model_dump(), device=device,
                          replay_buffer=buffer, arch=trainer.arch, sink=_DeferredSink(),
                          recorder=recorder, heartbeat=_DeferredHeartbeat())
    return RunCollaborators(trainer=trainer, pool=pool, buffer=buffer, log_dir=log_dir,
                            checkpoint_dir=checkpoint_dir, resume_state=resume_state)


def _step_coordinator_config(
    *,
    stop_step: int,
    draw_rate_abort: DrawRateAbortSpec | None,
    policy_loss_trough_abort: PolicyLossTroughAbortSpec | None,
    ply_cap_abort: PlyCapAbortSpec | None,
    drain_caps: DrainCapsSpec,
    gate_interval: int,
    knobs: CoordinatorKnobsSpec,
) -> StepCoordinatorConfig:
    """Assemble `StepCoordinatorConfig` from RESOLVED CONFIG FACTS ONLY — zero literals.

    The config-authored values are PARAMETERS **with no default of their own**: a parameter
    default would merely MIGRATE the authority from the dataclass field to this signature,
    leaving every `dataclasses.fields()` assertion green while a caller that omits the argument
    silently inherits a posture. `gate_interval` in particular has deliberately NO fallback to
    `log_interval` — the arming cadence having been an inherited property of it IS the defect.
    """
    return StepCoordinatorConfig(
        eval_interval=knobs.eval_interval,
        log_interval=knobs.log_interval,
        gate_interval=gate_interval,
        min_buf_size=knobs.min_buf_size,
        capacity=knobs.capacity,
        buffer_schedule=knobs.buffer_schedule,
        training_steps_per_game=knobs.training_steps_per_game,
        max_train_burst=knobs.max_train_burst,
        batch_size=knobs.batch_size,
        augment=knobs.augment,
        recency_weight=knobs.recency_weight,
        hard_gn_threshold=knobs.hard_gn_threshold,
        hard_gn_min_steps=knobs.hard_gn_min_steps,
        stop_step=stop_step,
        draw_rate_abort=draw_rate_abort,
        policy_loss_trough_abort=policy_loss_trough_abort,
        ply_cap_abort=ply_cap_abort,
        final_eval_drain_timeout_sec=drain_caps.final_eval_drain_timeout_sec,
        eval_final_drain_safety_factor=drain_caps.eval_final_drain_safety_factor,
        eval_final_drain_hard_cap_sec=drain_caps.eval_final_drain_hard_cap_sec,
        terminal_eval_hard_cap_sec=drain_caps.terminal_eval_hard_cap_sec,
        terminal_eval_enabled=knobs.terminal_eval_enabled,
        selfplay_stall_timeout_sec=knobs.selfplay_stall_timeout_sec,
    )


def _parent_death_event(decision: ParentDeathDecision | None) -> dict[str, Any] | None:
    """The in-run payload for the parent-death arming decision — or None if there is none.

    Built from the GATE'S OWN LATCH and never re-derived, because the gate runs at `main`'s
    first statement. `None` in means `None` out: a None latch means the gate never ran, and
    manufacturing an `armed=false` there would be the emitter inventing its own producer.
    """
    if decision is None:
        return None
    return {
        "event": "parent_death_signal_armed",
        "armed": decision.armed,
        "reason": decision.reason,
        "supervisor_pid": decision.supervisor_pid,
        "ppid": decision.ppid,
        "chain_depth": decision.chain_depth,
        "signal": decision.signal_name,
        # The sibling watchdogs' own field name for "this lever CAN fire".
        "enabled": decision.armed,
    }


#: This root's logger; `main` installs THE one mantis handler so it has somewhere to go.
_LOG = logging.getLogger(__name__)


def _emit_live_arming_audit(config: Any, sink: Any, *,
                            probes: Mapping[str, Callable[[], bool]]) -> None:
    """Publish the LIVE arming verdict for this run.

    Args:
        config: the validated `RunConfig` this run was composed from.
        sink: the run's event sink.
        probes: liveness answers by probe name, read at call time.

    Raises:
        Nothing. `ProducerProbeMissingError` and `ArmingSurfaceMissingError` are caught and
        emitted as the audit's own failure — a teardown-time diagnostic must never end the run."""
    try:
        result = audit_arming_live(config, probes=probes)
    except (ProducerProbeMissingError, ArmingSurfaceMissingError) as exc:
        _LOG.error("armed_abort_live_audit_failed: %s", exc)
        sink.emit({"event": "armed_abort_live_audit_failed",
                   "error_class": type(exc).__name__, "detail": str(exc)[:300]})
        return
    disarmed = [row.name for row in result.disarmed]
    if disarmed:
        _LOG.error(
            "armed_abort_live_audit: %s REQUIRED abort row(s) were NOT live this run: %s "
            "— the config armed them and the producer did not run, so the run carried the "
            "protection in writing and not in fact",
            len(disarmed), ", ".join(disarmed),
        )
    sink.emit({"event": "armed_abort_live_audit",
               "required": len(result.required),
               "disarmed": disarmed,
               "probes": sorted(probes)})


class BurstBoundError(ValueError):
    """A preflight burst bound outside 1..`train.max_train_steps`."""


def _resolve_stop_step(config: RunConfig, burst_stop_step: int | None) -> int:
    """The coordinator's stop step: `train.max_train_steps`, or the PREFLIGHT's bound below it.

    Raises:
        BurstBoundError: the bound is below 1 or above the minted run length.
    """
    ceiling = resolve_max_train_steps(config.train)
    if burst_stop_step is None:
        return ceiling
    bound = int(burst_stop_step)
    if bound < 1 or bound > ceiling:
        raise BurstBoundError(
            f"burst_stop_step {bound} is not inside 1..train.max_train_steps ({ceiling}): a "
            "preflight burst runs a PREFIX of the minted run, never past it and never nothing")
    return bound


def compose_run(
    *,
    config: RunConfig | Any,
    trainer: Any,
    pool: Any,
    buffer: Any,
    log_dir: str | Path,
    checkpoint_dir: str | Path,
    resume_state: Any = None,
    burst_stop_step: int | None = None,
) -> RunHandles:
    """The run composition root (§c.6).

    Injection-first: every COLLABORATOR arrives via a kwarg, never built here — no parameter
    may carry a CONFIG FACT (the census pins the list; `burst_stop_step` is the preflight's bound,
    a prefix of the run with no launcher route). MAIN-THREAD CALL: `signal.signal` needs it.

    Raises:
        BurstBoundError: `burst_stop_step` is outside the minted run length.
    """
    config = require_run_config(config, caller="compose_run")
    # The gate above checks the CLASS and stays first; this re-runs the CROSS-FIELD validators a
    # `model_copy(update=…)` skips (one such copy drove a 20-step run with a single actor sync).
    config = revalidate_run_config(config, caller="compose_run")

    # Signal handling, HOISTED: the loop installs only on a branch this root never takes. A signal
    # mid-compose only SETS the state; composition completes and the loop's entry arm saves.
    shutdown = ShutdownState()
    install_signal_handlers(shutdown)

    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)
    # The root making the dir it was HANDED usable: the disk guard stats `watch_path` and its poll
    # thread swallows errors, so a missing dir buys a guard that publishes NOTHING.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with _seam("resolve_monitor_config"):
        monitor_cfg = resolve_monitor_config(config.monitor)
    run_id = config.run_id

    wired_sources: list[str] = list(_BASE_WIRED_SOURCES)
    if config.eval_enabled:
        wired_sources.append("eval_round")

    # HOISTED: `_disk_guard_sample` closes over it and the watchdog (started before the guard)
    # polls it from its own thread, where an unassigned local would raise `NameError`.
    disk_guard = None

    def _disk_guard_sample() -> MonitorSample | None:
        """Read the disk guard's own counters LIVE at poll time."""
        guard = disk_guard
        if guard is None:
            return None
        return MonitorSample(checks_total=guard.checks_total,
                             errors_total=guard.errors_total,
                             interval_sec=guard.interval_sec)

    # The lag-watchdog callables read `actor_sync` LIVE at poll time (it is assigned below, before
    # any start): ActorSync comes after this call because its sink IS `run_safety.sink`.
    with _seam("build_run_safety"):
        run_safety = build_run_safety(
            log_dir=log_dir, run_id=run_id, buffer=buffer,
            buffer_persist_path=canonical_buffer_path(checkpoint_dir),
            wired_sources=wired_sources, monitor_cfg=monitor_cfg,
            actor_ckpt_step_fn=lambda: actor_sync.actor_ckpt_step(),
            learner_step_fn=lambda: int(trainer.step),
            monitor_liveness=(MonitorLivenessSpec(name="disk_guard",
                                                  sample_fn=_disk_guard_sample),),
        )

    # Bind the pool's `_DeferredSink` before `pool.start()` so the first emit is delivered; `bind()`
    # claims nothing, so it sits outside the teardown ladder. `isinstance` spares bare fake pools.
    deferred = getattr(pool, "_sink", None)
    if isinstance(deferred, _DeferredSink):
        deferred.bind(run_safety.sink)

    # The TRAINER's `_DeferredSink`: a SEPARATE bind for a SEPARATE adapter, so neither unbinds the
    # other; its first emission is inside `run_training_loop`. `isinstance` spares bare fakes.
    deferred_tr = getattr(trainer, "_sink", None)
    if isinstance(deferred_tr, _DeferredSink):
        deferred_tr.bind(run_safety.sink)

    # The pool's `_DeferredHeartbeat`, bound before `pool.start()` so beats forward from the first
    # tick; a separate bind from the sink's. `isinstance` spares bare fake pools.
    deferred_hb = getattr(pool, "_heartbeat", None)
    if isinstance(deferred_hb, _DeferredHeartbeat):
        deferred_hb.bind(run_safety.heartbeat)

    pool_start_attempted = False
    coordinator = None
    eval_pipeline = None
    resolved_anchor = SimpleNamespace(best_model=None, best_model_step=None)
    # TEARDOWN LADDER: nothing is left half-alive after a failed compose and the ORIGINAL failure
    # propagates (a teardown failure chains). It opens at the SINK, which opens the JSONL segment.
    # `coordinator is None` is the discriminator: once it exists `close_out` owns drain, save, stop.
    try:
        with _seam("run_boot_identity emit"):
            # The booted process publishes ITS OWN post-revalidation config identity first; the
            # preflight parent hashes its own the same way, so a child on another file is NAMED.
            emit_via(run_safety.sink, {
                "event": "run_boot_identity",
                "run_id": run_id,
                "config_sha256": config_identity_sha256(config),
            })
        with _seam("resolved_config emit"):
            # After the identity witness (which lands FIRST in case the boot wedges), once per
            # segment; the payload IS `to_event_payload(resolve_config(config))`, never hand-built.
            emit_via(run_safety.sink, resolve_config(config).to_event_payload())

        with _seam("parent_death_signal_armed emit"):
            # The arming lever announces itself in the run's OWN stream. Payload
            # and None-guard both live in `_parent_death_event`.
            parent_death_event = _parent_death_event(last_parent_death_decision())
            if parent_death_event is not None:
                emit_via(run_safety.sink, parent_death_event)

        # Built UNCONDITIONALLY: the actor's weights come from the learner on a cadence and NEVER
        # from a gate decision, so no config or eval state may make actor sync conditional.
        with _seam("ActorSync"):
            actor_sync = ActorSync(
                target=pool,
                state_dict_fn=trainer.actor_state_dict,
                step_fn=lambda: int(trainer.step),
                cadence_steps=resolve_actor_sync_cadence(config.train),
                sink=run_safety.sink,
                run_id=run_id,
            )

        # Built FIRST, `DrainCaps` LIFTED from its 4 fields, every term PASSED IN through its own
        # resolver: a literal always overwritten afterwards is still a second default authority.
        with _seam("_step_coordinator_config"):
            step_coordinator_cfg = _step_coordinator_config(
                stop_step=_resolve_stop_step(config, burst_stop_step),
                draw_rate_abort=resolve_draw_rate_abort(config.train),
                policy_loss_trough_abort=resolve_policy_loss_trough_abort(config.train),
                ply_cap_abort=resolve_ply_cap_abort(config.train),
                drain_caps=resolve_drain_caps(config.monitor),
                # The ARMING cadence; never `knobs.log_interval`, which would blind armed aborts
                # for the first `log_interval` steps.
                gate_interval=config.monitor.gate_interval,
                knobs=resolve_coordinator_knobs(config.train),
            )

        if config.eval_enabled:
            with _seam("build_eval_pipeline"):
                eval_pipeline = build_eval_pipeline(
                    eval_cfg=config.eval,
                    coordinator_cfg_caps=DrainCaps(
                        final_eval_drain_timeout_sec=step_coordinator_cfg.final_eval_drain_timeout_sec,
                        eval_final_drain_safety_factor=step_coordinator_cfg.eval_final_drain_safety_factor,
                        eval_final_drain_hard_cap_sec=step_coordinator_cfg.eval_final_drain_hard_cap_sec,
                        terminal_eval_hard_cap_sec=step_coordinator_cfg.terminal_eval_hard_cap_sec,
                    ),
                    encoding=config.identity.encoding,
                    # The fused-forward memory bound, carried to every round's `RoundSpec`.
                    fused_graph_caps=resolve_fused_graph_caps(config.model_dump()),
                    # The collector's pop width and deadline; the child has no `RunConfig`.
                    inference_batching=resolve_inference_batching(config.model_dump()),
                    # The SAME `selfplay.leaf_batch_size` the targets were searched at; another
                    # deploy-head width is a train/deploy mismatch.
                    leaf_batch_size=config.selfplay.leaf_batch_size,
                    # The eval leaf-graph build's WIDTH; the derivation reserves the pool's and
                    # the server's threads, so it takes only what the run has not promised.
                    leaf_build_threads=resolve_leaf_build_threads(config.model_dump()),
                    # The child, a SECOND allocator in its own process, asserts the posture FOR
                    # ITSELF; this seam only threads the token. `None` on the not-cuda arm.
                    allocator_posture=(
                        _declared_allocator_posture(config.model_dump())
                        if _posture_governs_device(config.eval.worker_device)
                        else None
                    ),
                    # The RUN's ply cap, carried to every round: the eval child has no RunConfig.
                    max_plies=config.eval.max_plies,
                    # The deploy head's sigma terms are the RUN's minted keys, not the
                    # player's signature defaults.
                    c_visit=config.selfplay.c_visit, c_scale=config.selfplay.c_scale,
                    q_rescale=config.selfplay.q_rescale,
                    # The deploy head searches with `deploy.search.kind` — matched to what
                    # will be deployed, not to the training search.
                    search_kind=resolve_deploy_search_kind(config),
                    gumbel_m=config.selfplay.gumbel_m,
                    run_id=run_id, spool_dir=log_dir / "eval_spool",
                    # The SAME directory the self-play recorder writes into, named once
                    # here, so one run's four channels land in one store.
                    game_record_dir=log_dir / "games",
                    promotion=DeployTagHooks(
                        anchor_state=resolved_anchor,
                        best_model_path=canonical_anchor_path(checkpoint_dir), run_id=run_id,
                        encoding=config.identity.encoding,
                        save_anchor=save_best_model_atomic,
                        guarded_load=_guarded_load_state_dict,
                    ),
                    sink=run_safety.sink, heartbeat=run_safety.heartbeat,
                )

        # Restored before `pool.start()`: promotion is gated on `round_idx % gate.stride`, so a
        # counter restarting at zero would realign a resumed run's promotion cadence.
        if resume_state is not None and eval_pipeline is not None:
            with _seam("restore_round_state"):
                eval_pipeline.restore_round_state(round_counter=resume_state.round_counter)

        # ORDER PINNED: pool, THEN watchdog. The flag is set BEFORE the call: `start()` is three
        # sub-starts, and a raise inside leaves a HALF-started pool a set-after flag misses.
        with _seam("pool.start"):
            pool_start_attempted = True
            pool.start()
        # A bound `_DeferredHeartbeat` is asserted HERE, before the watchdog arms: with
        # `heartbeat=None` it would fire rc 42 at 1800 s on every healthy run.
        with _seam("producer_liveness_conjunct"):
            _assert_pool_producers_live(pool)
        with _seam("watchdog.start"):
            run_safety.watchdog.start()

        # The disk guard, from the config family's ONE resolver; its critical arm SIGTERMs the
        # process, landing on the handlers installed above.
        with _seam("DiskGuard"):
            guard_spec = resolve_disk_guard(config.monitor)
            disk_guard = DiskGuard(
                watch_path=checkpoint_dir, interval_sec=guard_spec.interval_sec,
                warn_gb=guard_spec.warn_gb, fail_gb=guard_spec.fail_gb,
                keep_all=_DISK_GUARD_KEEP_ALL, sink=run_safety.sink,
            )
            disk_guard.start()

        # The held-out witness: the frozen slice is opened HERE, before the coordinator
        # can step, so a missing or mis-hashed ring is a loud STARTUP failure; `null` opens nothing.
        with _seam("HeldoutSlice"):
            heldout_spec = resolve_heldout_gap(config.model_dump())
            heldout = None
            if heldout_spec is not None:
                heldout = HeldoutSlice.open(
                    heldout_spec, encoding=config.identity.encoding,
                    visit_capacity=derived_visit_capacity(config),
                    capacity=int(resolve_coordinator_knobs(config.train).capacity))
                _LOG.info("heldout_slice_opened ring=%s rows=%s batches=%s interval=%s",
                          heldout.ring_path, heldout.rows, heldout_spec.batches, heldout_spec.interval)

        with _seam("StepCoordinator"):
            coordinator = StepCoordinator(
                trainer=trainer, buffer=buffer,
                pool=pool, eval_pipeline=eval_pipeline,
                # The guard latches `critical_fired` BEFORE it signals, so this is how the O3 arm
                # sees the one abort that reaches it with `abort_rule` still unrecorded.
                subsystems=SimpleNamespace(gpu_monitor=None, disk_guard=disk_guard),
                anchor_state=resolved_anchor, shutdown=shutdown,
                eval_model=trainer.deploy_module(),
                config=step_coordinator_cfg, full_config=config.model_dump(),
                run_id=run_id,
                sink=run_safety.sink, heartbeat=run_safety.heartbeat, monitor_cfg=monitor_cfg,
                heartbeat_watchdog=run_safety.watchdog, actor_sync=actor_sync, heldout=heldout,
            )
        if resume_state is not None:
            # The last KICKED round travels with the ring: a resume at an exact eval_interval
            # multiple kicks that boundary's round iff the stop did not (B-3).
            coordinator.restore_eval_round_state(resume_state.eval_round_last_step)
            # The abort windows and guard counters the stop held, so an armed abort's earliest
            # fire does not move by a window on every resume (B-7).
            coordinator.restore_guard_state(resume_state.guards)

        # NOTHING is swallowed: the loop's failure propagates, `close_out` runs in a `finally`.
        # The launch pin DERIVES from `identity.warm_start` (one source); an absent row is `None`.
        declared_warm_start = resolve_bc_warm_start(config.model_dump())
        # THE PIN HOLDS AT STEP 0 ONLY: a fresh launch asserts the WARM-START hash, but a promotion
        # moves a resumed run's anchor legitimately, so a resume asserts what THE STOP RECORDED.
        resumed_anchor_sha = getattr(resume_state, "anchor_sha256", None)
        expected_anchor = (
            resumed_anchor_sha if resumed_anchor_sha is not None
            else (None if declared_warm_start is None else declared_warm_start.net_hash)
        )
        try:
            run_training_loop(trainer=trainer, shutdown_state=shutdown,
                              eval_pipeline=eval_pipeline, coordinator=coordinator,
                              anchor_state=resolved_anchor, sink=run_safety.sink,
                              best_model_path=canonical_anchor_path(checkpoint_dir),
                              expected_anchor_sha256=expected_anchor)
        finally:
            # THE RESUMABLE-STOP DECISION: the only scope holding `shutdown_save`, the guard's
            # LATCHED `critical_fired` and `abort_rule` (blind to a disk abort, recorded later).
            # Anything short of all three saying "operator stop" runs the terminal battery.
            resumable_stop = bool(
                shutdown.shutdown_save
                and shutdown.abort_rule is None
                and not (disk_guard is not None and disk_guard.critical_fired)
            )
            coordinator.close_out(
                on_drained=_stop_pool_if_start_attempted(
                    pool, start_attempted=pool_start_attempted),
                resumable_stop=resumable_stop)
    finally:
        # THE LIVE ARMING AUDIT (gate 12 sees only a config number); it REPORTS, never refuses. At
        # the TOP of the ladder, the only point where the sink is open on BOTH paths; `checks_total`
        # only RISES, so reading it before the join can under-report, never over-report.
        _emit_live_arming_audit(
            config, run_safety.sink,
            probes={DISK_GUARD_LIVENESS_PROBE:
                    lambda: disk_guard is not None and disk_guard.checks_total > 0},
        )
        if coordinator is None:
            # PARTIAL COMPOSITION: `close_out` never ran, so the pool and run-safety threads stop
            # HERE. The arm restriction is DEBT (test fakes lack `close`/`stop`); meanwhile the sink
            # is line-buffered, the watchdog a daemon, and both production callers exit right after.
            _stop_pool_if_start_attempted(pool, start_attempted=pool_start_attempted)()
            run_safety.watchdog.stop()
            run_safety.sink.close()
        # The eval poller and any spawn child need teardown on BOTH paths: `close_out` never joins
        # the poller, and on the PARTIAL path it never ran.
        if eval_pipeline is not None:
            eval_pipeline.stop()
        # The disk guard is this root's on BOTH paths — `close_out` has never heard of it, and
        # a daemon guard that outlives its run will SIGTERM a process no longer running one.
        if disk_guard is not None:
            disk_guard.stop()
            # A fired guard's handler never writes `abort_rule`, so it is recorded HERE (the name is
            # a MANIFEST row's, which `mantis.train` may not import), AFTER `stop()` joined the
            # thread so the latch is final; `record_abort` is set-once, so first fire wins.
            if disk_guard.critical_fired:
                shutdown.record_abort(DISK_SPACE_ABORT_RULE)
        # THE TERMINAL-EVAL SEAM: `ShutdownState` alone carries the outcome to `main`. Read AFTER
        # the disk guard so a disk-full run reports the ROOT CAUSE (47); the bare
        # `EvalBrokenReason(raw)` makes an unknown spelling crossing JSON raise instead of rc 0.
        if coordinator is not None:
            terminal_reason = coordinator.terminal_eval_reason
            if terminal_reason is not None:
                EvalBrokenReason(terminal_reason)
                shutdown.record_abort(TERMINAL_EVAL_BROKEN_ABORT_RULE)

    return RunHandles(coordinator=coordinator, run_safety=run_safety, eval_pipeline=eval_pipeline,
                      shutdown=shutdown)


def launch_run(
    *, config: RunConfig, out_dir: str | Path, checkpoint_path: str | None = None,
) -> RunHandles:
    """THE launch path: build the collaborators, compose the run. Nothing else."""
    collaborators = build_run_collaborators(
        config=config, out_dir=out_dir, checkpoint_path=checkpoint_path)
    return compose_run(config=config, trainer=collaborators.trainer, pool=collaborators.pool,
                       buffer=collaborators.buffer, log_dir=collaborators.log_dir,
                       checkpoint_dir=collaborators.checkpoint_dir,
                       resume_state=collaborators.resume_state)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m mantis.run --config <path> --out-dir <path>` — the production launcher.

    `--config` and `--out-dir` are required with NO `default=`: a defaulted `--out-dir` is a run
    input the code decides, and every run that forgets the flag writes into one shared directory.
    The OPTIONAL flags, `--resume-from` and `--inherit-preflight`, default to `None` —
    no action, not a value — and are flags, not schema keys, because each names a property of
    THIS invocation. NO `--device` flag and no eval switch: `config.train.device` and
    `config.eval_enabled` are the only routes. The preflight's in-repo `--out-dir` refusal is not
    mirrored here; that guard exists because a CI gate must not dirty the tree it gates.

    rc policy goes through the SAME `exit_code_for_abort` the preflight child reads. Three rules
    reach it with an authored code: `draw_rate_collapse` (46), `disk_space_exhausted` (47) off the
    guard's latch, `terminal_eval_broken` (48) off the coordinator's set-once latch, read AFTER
    the disk-guard read so first-fire-wins keeps the root cause. A signal this process did NOT
    send itself still resolves to 0. rc 1 is CPython's rc for any exception leaving `main`, not an
    AUTHORED code: `UnregisteredAbortExitError` and a composition wall look alike to a supervisor.

    Raises:
        PreflightStampRefusal: the loaded config has no passing preflight stamp for this tree
            and inherits none; a named subclass says which fact is missing.
        UnregisteredAbortExitError: an abort fired whose rule authors no exit code.
    """
    # THE FIRST STATEMENT: closes the window where the supervisor dies while the run comes up. A
    # NO-OP unless a supervisor stamped its pid in the env (else unattended runs and pytest would
    # be armed); never at import time.
    arm_parent_death_if_supervised()
    parser = argparse.ArgumentParser(
        prog="python -m mantis.run",
        description="Launch a run from a minted config (the ONE composition authority).",
    )
    parser.add_argument("--config", required=True, help="path to the minted run config")
    parser.add_argument("--out-dir", required=True,
                        help="run artifacts root; logs/ and checkpoints/ are derived from it")
    parser.add_argument(
        "--resume-from", default=None,
        help="checkpoint to resume the trainer from; omit for a fresh run")
    parser.add_argument(
        "--inherit-preflight", type=Path, default=None,
        help="a shakedown twin's run config: the twin (its rows but run_id) launches on that "
             "run's vested preflight stamp instead of one of its own (R360(c))")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    # THE ONE mantis stderr handler, installed before the launch so the boot path's own
    # diagnostics are covered.
    configure_logging()

    # Fails a stale `--resume-from` AT LAUNCH, before `torch.load` and before any run is built.
    bootstrap = resolve_bootstrap(args.resume_from)
    config = load_config(args.config)
    # A run cannot skip the manual preflight by not running it; the stamp is keyed by
    # the config's identity hash and bound to the tree that preflighted it.
    stamp = require_preflight_stamp(config, tree_root=Path(__file__).resolve().parent,
                                    inherit_from=args.inherit_preflight)
    _LOG.info("preflight_stamp_accepted config_sha256=%s tree_sha=%s preflight_utc=%s%s",
              stamp["config_sha256"], stamp["tree_sha"], stamp["preflight_utc"],
              f" {stamp['inherited']}" if "inherited" in stamp else "")
    handles = launch_run(config=config, out_dir=args.out_dir, checkpoint_path=bootstrap.path)
    rule = handles.shutdown.abort_rule
    if rule is None:
        return 0
    code = exit_code_for_abort(rule)
    if code is None:
        raise UnregisteredAbortExitError(
            f"the run's hard-abort rule {rule!r} FIRED and stopped the run, but "
            "`mantis.config.armed_aborts.MANIFEST` authors no exit code for it. Reported as "
            "a named failure rather than as rc 0: an aborted run is not a clean one. No code "
            "is invented here — R84 declined to author one for a rule nobody pre-registered, "
            "and doing it at the launcher would be that same class one layer up"
        )
    return int(code)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "RunCollaborators",
    "RunHandles",
    "UnregisteredAbortExitError",
    "build_run_collaborators",
    "compose_run",
    "launch_run",
    "main",
]
