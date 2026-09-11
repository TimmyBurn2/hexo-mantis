# >300 justify (R8). This module is the ONE composition authority: the collaborator builder,
# the launcher and LAW-16's three legs (signals, watchdog, disk guard) live here together,
# and it is the only module importing both `mantis.train` and `mantis.eval` at top level.
# Splitting any layer out would create a SECOND site where a collaborator set or a
# `StepCoordinatorConfig` can be built, which is the two-surfaces shape the one-authority
# tests forbid; the per-decision rationale is the file's bulk and is what R8's clause protects.
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
from mantis.config.resolve.inference_batching import resolve_inference_batching
from mantis.config.resolve.leaf_build_threads import resolve_leaf_build_threads
from mantis.config.resolve.monitor import resolve_monitor_config
from mantis.config.resolve.run_length import resolve_max_train_steps
from mantis.config.resolve.search import resolve_search_kind
from mantis.config.schema import RunConfig
from mantis.eval.errors import EvalBrokenReason
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.monitor.config import MonitorConfig
from mantis.monitor.game_recorder import GameRecorder
from mantis.monitor.logging_setup import configure_logging
from mantis.selfplay.pool import WorkerPool
from mantis.train.actor_sync import ActorSync
from mantis.train.anchor import canonical_anchor_path
from mantis.train.buffer_persist import canonical_buffer_path
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.dispatch import RepresentationRouteError
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.emit import NullEventSink, emit_via
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
    #: The resume sidecar this boot restored FROM, or None on a fresh run. It rides the
    #: tuple rather than being re-read in `compose_run`: a second read of the same sidecar
    #: is a second authority for what this boot resumed from.
    resume_state: Any = None


@contextmanager
def _seam(name: str) -> Iterator[None]:
    """NAME a composition seam without catching anything (LAW-14).

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


def _resolve_monitor_cfg(config: RunConfig) -> MonitorConfig:
    """Read the monitor section through its ONE resolver, requiring it to be present.

    The retired absent-section arm returned a bare `MonitorConfig()`, which carries
    `actor_lag_abort_enabled=False` and so silently disarmed a hard abort the config arms.
    """
    return resolve_monitor_config(config.monitor)


def _resolve_actor_sync_cadence_steps(config: RunConfig) -> int:
    """Read `train.actor_sync_cadence_steps` through its ONE resolver, requiring the section.

    The retired smoke arm substituted cadence 1 for any config without a train section — a
    test-only value on a production axis.
    """
    return resolve_actor_sync_cadence(config.train)


def _select_buffer(config: Any, capacity: int) -> Any:
    """Select the replay buffer off `config.identity.representation`; an unknown or absent
    representation RAISES (LAW-11) — never sniffed off a live module, never defaulted.

    This raise is the boot's one LAW-11 refusal, and it lives under `src/` so CI gate 11 can
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
        # Lazy with a stated reason: `mantis._engine` is not an edge on the design's `run`
        # row, and the extension module is the one import this root must not make
        # unconditional.
        from mantis._engine import HexgBuffer, derived_hexg_visit_capacity

        # The ring's visit-slot geometry is DERIVED at composition from the config's sims
        # regime, through the same Rust authority the schema validator ran at load — so it
        # cannot raise on a validated config, and no literal can reappear on this path.
        sp = config.selfplay
        pc = sp.playout_cap
        visit_capacity = derived_hexg_visit_capacity(
            n_simulations=sp.mcts.n_simulations,
            standard_sims=pc.standard_sims,
            fast_prob=pc.fast_prob,
            fast_sims=pc.fast_sims,
            full_search_prob=pc.full_search_prob,
            n_sims_quick=pc.n_sims_quick,
            n_sims_full=pc.n_sims_full,
            leaf_batch_size=sp.leaf_batch_size,
            gumbel_m=sp.gumbel_m,
            search_kind=config.search.kind,
        )
        buffer = HexgBuffer(capacity, config.identity.encoding, visit_capacity)
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
    # THE SIDECAR'S PRESENCE IS WHAT DISCRIMINATES A RESUME FROM A WARM START, and nothing
    # else at HEAD can: `--resume-from` carries both meanings, and refusing a sidecar-less
    # checkpoint would refuse a BC warm-start launch. The absence is announced, never inferred:
    # the one thing that must not happen quietly is a CONTINUATION refilling from empty.
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
    # left to a schema default, BEFORE anything can wedge. Persistence-fatal by LAW-14.
    with _seam("resolved_config record"):
        write_resolved_config(config, out_dir)

    device = torch.device(config.train.device)
    # The allocator-posture assertion, BEFORE the first CUDA allocation and therefore before
    # `init_trainer`: a cap fitted under one allocator regime and run under the other is a
    # partition measured for a machine state this process is not in, and that difference
    # measured 3.62 GiB of card high-water on 2026-08-22. Non-cuda runs are not enforced.
    _assert_allocator_posture(config.model_dump(), device_type=device.type)
    with _seam("init_trainer"):
        # The trainer gets the SAME late-binding adapter shape the WorkerPool below gets;
        # `sink=None` here meant every trainer-side emission was authored, unit-tested and
        # DROPPED in every production run. Pre-bind emissions (resume-time events fire in
        # this builder) still drop, which is the adapter's documented semantics.
        trainer = init_trainer(config=config.model_dump(), checkpoint_dir=str(checkpoint_dir),
                               device=device, sink=_DeferredSink(),
                               checkpoint_path=checkpoint_path)
    capacity = int(resolve_coordinator_knobs(config.train).capacity)
    with _seam("_select_buffer"):
        buffer = _select_buffer(config, capacity)
    # THE RING IS A RESUME INPUT, NOT A BUFFER THAT REFILLS FROM EMPTY. Placed immediately
    # after construction and before `WorkerPool` can push, because a load into a ring
    # self-play has already written is a load into a ring whose contents nobody declared. On
    # a resume a failure PROPAGATES (LAW-14) — it must not quietly become a fresh run.
    resume_state = None
    if checkpoint_path is not None:
        with _seam("restore_resume_state"):
            resume_state = _restore_resume_state(buffer, checkpoint_path)
    # The game record's self-play producer, constructed before the pool that feeds it, so
    # from step 0 every self-play game is written. Constructed here rather than lazily on the
    # first game because an un-openable store must be a loud STARTUP failure.
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
        final_eval_drain_timeout_sec=drain_caps.final_eval_drain_timeout_sec,
        eval_final_drain_safety_factor=drain_caps.eval_final_drain_safety_factor,
        eval_final_drain_hard_cap_sec=drain_caps.eval_final_drain_hard_cap_sec,
        terminal_eval_hard_cap_sec=drain_caps.terminal_eval_hard_cap_sec,
        terminal_eval_enabled=knobs.terminal_eval_enabled,
        selfplay_stall_timeout_sec=knobs.selfplay_stall_timeout_sec,
    )


def _parent_death_event(decision: ParentDeathDecision | None) -> dict[str, Any] | None:
    """The LAW-18 payload for the parent-death arming decision — or None if there is none.

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
    # The gate above answers "is this the class?"; this answers "is this a config the loader
    # would accept?". `model_copy(update=…)` builds a genuine RunConfig whose CROSS-FIELD
    # validators never re-ran, and one such copy drove a 20-step run with a single actor sync.
    # A second statement, because the gate must remain compose_run's first and the two rules
    # have different contracts: the gate is identity-preserving, this is not.
    config = revalidate_run_config(config, caller="compose_run")

    # LAW-16 leg 1, HOISTED to the top of the composition: `install_signal_handlers` used to
    # fire only on `run_training_loop`'s self-construct branch, which this root never takes, so
    # save-then-exit was dead in every composed run (probed at 19/19). "Signal-covered" here
    # means the STATE IS SET, not that save-then-exit fires mid-compose: nothing between this
    # line and `run_training_loop` polls the state, so a signal in that window lets composition
    # COMPLETE and the save happens at the loop's entry-set arm.
    shutdown = ShutdownState()
    install_signal_handlers(shutdown)

    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)
    # Not a second derivation (that lives in `build_run_collaborators`) — the root making the
    # directory it was HANDED usable. Load-bearing for LAW-16 leg 3: the disk guard stats
    # `watch_path` and its poll thread swallows its own errors, so a checkpoint dir that does
    # not exist yet buys a guard that runs, logs and publishes NOTHING.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with _seam("_resolve_monitor_cfg"):
        monitor_cfg = _resolve_monitor_cfg(config)
    run_id = config.run_id

    wired_sources: list[str] = list(_BASE_WIRED_SOURCES)
    if config.eval_enabled:
        wired_sources.append("eval_round")

    # HOISTED above `build_run_safety` because `_disk_guard_sample` closes over this name and
    # the watchdog polls it from its own thread. The pinned order is pool -> watchdog -> guard,
    # so a closure over an unassigned local would raise `NameError` on that thread instead of
    # reading "not yet".
    disk_guard = None

    def _disk_guard_sample() -> MonitorSample | None:
        """Read the disk guard's own counters LIVE at poll time."""
        guard = disk_guard
        if guard is None:
            return None
        return MonitorSample(checks_total=guard.checks_total,
                             errors_total=guard.errors_total,
                             interval_sec=guard.interval_sec)

    # The lag-watchdog callables are read LIVE at poll time, never at build time —
    # `actor_sync` is assigned immediately below, before anything can start. The design's
    # "ActorSync first" ordering is inverted here because the engine's LAW-18 sink IS
    # `run_safety.sink`, which only exists after this call.
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

    # Bind the pool's `_DeferredSink` to the real sink now that it exists; `pool.start()`
    # runs below, so events are delivered from the first emit. `bind()` claims no resource,
    # so it sits OUTSIDE the teardown ladder. Sink-only — `heartbeat=` is untouched here. The
    # `isinstance` guard keeps test harnesses injecting a bare fake pool working.
    deferred = getattr(pool, "_sink", None)
    if isinstance(deferred, _DeferredSink):
        deferred.bind(run_safety.sink)

    # Bind the TRAINER's `_DeferredSink` the same way — a SEPARATE bind for a SEPARATE
    # adapter, so neither can silently unbind the other's surface. The first trainer-side
    # emission fires inside `run_training_loop`, after this bind. The `isinstance` guard keeps
    # test harnesses injecting a bare fake trainer working.
    deferred_tr = getattr(trainer, "_sink", None)
    if isinstance(deferred_tr, _DeferredSink):
        deferred_tr.bind(run_safety.sink)

    # Bind the pool's `_DeferredHeartbeat` now that the real fn exists; `pool.start()` runs
    # below, so beats are forwarded from the first tick. Heartbeat-only, a SEPARATE bind from
    # the sink's. The `isinstance` guard keeps bare fake pools in test harnesses working.
    deferred_hb = getattr(pool, "_heartbeat", None)
    if isinstance(deferred_hb, _DeferredHeartbeat):
        deferred_hb.bind(run_safety.heartbeat)

    pool_start_attempted = False
    coordinator = None
    eval_pipeline = None
    resolved_anchor = SimpleNamespace(best_model=None, best_model_step=None)
    # TEARDOWN LADDER. By the time `StepCoordinator` is constructed the pool is started and
    # the watchdog and disk-guard threads are running; a bare propagation would leave all
    # three alive after a failed compose. The contract: nothing is half-alive, and the
    # ORIGINAL failure propagates (nothing here catches; a teardown failure chains).
    #
    # THE LADDER OPENS AT THE SINK, not at `pool.start()`: `build_run_safety` OPENS the run's
    # JSONL segment file, and the steps that used to sit between it and the old `try:` were
    # outside both the ladder and any seam — so an eval-pipeline wall reached the process
    # boundary with the segment file still open and no seam name for the rc-32/33 classifier.
    #
    # `coordinator is None` is the discriminator, not a second flag: once it exists,
    # `close_out` owns the drain, the buffer save and the guarded pool stop.
    try:
        with _seam("run_boot_identity emit"):
            # The booted process publishes ITS OWN post-revalidation config identity, first
            # thing after the sink exists. One authority on both sides: the mint preflight's
            # parent hashes the config IT loaded with the same function and compares, so a
            # child that read a different file is a NAMED failure instead of invisible.
            emit_via(run_safety.sink, {
                "event": "run_boot_identity",
                "run_id": run_id,
                "config_sha256": config_identity_sha256(config),
            })
        with _seam("resolved_config emit"):
            # The run's own resolved posture, immediately after the identity witness — which
            # must land FIRST because it has to exist even if the boot later wedges. Exactly
            # once per segment, and the payload IS `to_event_payload(resolve_config(config))`:
            # a hand-assembled copy would be a second authority for it.
            emit_via(run_safety.sink, resolve_config(config).to_event_payload())

        with _seam("parent_death_signal_armed emit"):
            # LAW-18: the arming lever announces itself in the run's OWN stream. Payload
            # and None-guard both live in `_parent_death_event`.
            parent_death_event = _parent_death_event(last_parent_death_decision())
            if parent_death_event is not None:
                emit_via(run_safety.sink, parent_death_event)

        # The continuous-sync engine is built UNCONDITIONALLY — no config or eval state may
        # make actor sync conditional. The actor's weights come from the learner on a cadence
        # and NEVER from a gate decision.
        with _seam("ActorSync"):
            actor_sync = ActorSync(
                target=pool,
                state_dict_fn=trainer.inference_state_dict,
                step_fn=lambda: int(trainer.step),
                cadence_steps=_resolve_actor_sync_cadence_steps(config),
                sink=run_safety.sink,
                run_id=run_id,
            )

        # The StepCoordinatorConfig is built FIRST and DrainCaps is LIFTED from its own 4
        # fields, never a second independently-hardcoded set of literals. `stop_step`,
        # `draw_rate_abort`, `drain_caps` and `knobs` are PASSED IN through their own
        # resolvers rather than replaced afterwards: a literal that is always overwritten is
        # still a second default authority.
        with _seam("_step_coordinator_config"):
            step_coordinator_cfg = _step_coordinator_config(
                stop_step=_resolve_stop_step(config, burst_stop_step),
                draw_rate_abort=resolve_draw_rate_abort(config.train),
                drain_caps=resolve_drain_caps(config.monitor),
                # The ARMING cadence, named directly off the validated monitor section. It
                # is NOT `knobs.log_interval` and must never become it: that identity IS the
                # defect (armed aborts with a blind first `log_interval` steps).
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
                    # The fused-forward memory bound, resolved through its ONE read path
                    # here and carried to every round's `RoundSpec`. GRAPH-only: the dense
                    # batch is already bounded by `inference_batch_size`, and reading the
                    # block on a grid run would make a graph-only key a grid dependency.
                    fused_graph_caps=(
                        resolve_fused_graph_caps(config.model_dump())
                        if config.identity.representation == "graph"
                        else None
                    ),
                    # The collector's pop width and pop deadline, resolved through their ONE
                    # read path here and carried to the child, which used to write both as
                    # literals. Graph-only: a grid round builds no graph server.
                    inference_batching=(
                        resolve_inference_batching(config.model_dump())
                        if config.identity.representation == "graph"
                        else None
                    ),
                    # The deploy head's MCTS leaf-batch width, read from the SAME
                    # `selfplay.leaf_batch_size` the self-play worker reads. The net's targets
                    # are generated by leaf-batched search, so a deploy head searching at a
                    # different width is a train/deploy mismatch.
                    leaf_batch_size=config.selfplay.leaf_batch_size,
                    # The eval leaf-graph build's WIDTH, derived through its ONE read path
                    # and carried to the child, which has no `RunConfig`. Graph-only. A
                    # profile put 95.3 % of the eval game loop inside the serial build this
                    # widens; the derivation RESERVES `selfplay.n_workers` threads plus the
                    # server thread, so it takes only what the run has not promised.
                    leaf_build_threads=(
                        resolve_leaf_build_threads(config.model_dump())
                        if config.identity.representation == "graph"
                        else 1
                    ),
                    # The eval child is a SECOND allocator on the same card, in its own
                    # process with its own CUDA context, so it asserts the posture FOR ITSELF
                    # at round start. THREADED here, not judged here: this seam does not know
                    # the child's device and the child does. `None` on the not-cuda arm.
                    allocator_posture=(
                        _declared_allocator_posture(config.model_dump())
                        if _posture_governs_device(config.eval.worker_device)
                        else None
                    ),
                    # The run's declared autocast dtype, resolved ONCE here and carried to
                    # every round — the eval child has no RunConfig, and its dense forward had
                    # no dtype at all. The arena's ply cap is likewise the RUN's, not a module
                    # constant that was a copy of a copy of this key.
                    max_plies=config.selfplay.max_game_moves,
                    # The deploy head's sigma terms are the RUN's minted keys, not the
                    # player's signature defaults.
                    c_visit=config.selfplay.c_visit, c_scale=config.selfplay.c_scale,
                    # The deploy head searches with the RUN'S OWN KIND, through the SAME
                    # resolver `SelfPlayHParams.from_config` reads. LAW-15's deploy-matched
                    # bar is a construction here, not a coincidence between two call sites.
                    search_kind=resolve_search_kind(config),
                    gumbel_m=config.selfplay.gumbel_m,
                    run_id=run_id, spool_dir=log_dir / "eval_spool",
                    # The SAME directory the self-play recorder writes into, named once
                    # here, so one run's four channels land in one store.
                    game_record_dir=log_dir / "games",
                    ladder_state_path=log_dir / "eval_ladder_state.json",
                    promotion=DeployTagHooks(
                        anchor_state=resolved_anchor,
                        best_model_path=canonical_anchor_path(checkpoint_dir), run_id=run_id,
                        encoding=config.identity.encoding,
                        save_anchor=_lazy_save_anchor, guarded_load=_lazy_guarded_load,
                    ),
                    sink=run_safety.sink, heartbeat=run_safety.heartbeat,
                )

        # The round counter and `p_hat` a stopped process left behind, restored before
        # `pool.start()` so no round is kicked against a counter that restarted at zero:
        # `_build_round_spec` gates the promotion channel on `round_idx % gate.stride`, so
        # without this a resumed run's promotion cadence realigns to the restart.
        if resume_state is not None and eval_pipeline is not None:
            with _seam("restore_round_state"):
                eval_pipeline.restore_round_state(
                    round_counter=resume_state.round_counter,
                    last_p_hat=resume_state.last_p_hat,
                )

        # ORDER PINNED: pool starts, THEN the watchdog. The flag is set BEFORE the call, not
        # after: `WorkerPool.start()` is three sub-starts, so a raise inside it leaves a
        # HALF-started pool that a set-after flag reports as never started.
        with _seam("pool.start"):
            pool_start_attempted = True
            pool.start()
        # Before the watchdog arms, assert the pool's heartbeat is a bound
        # `_DeferredHeartbeat`: a pool that got `heartbeat=None` has dead producers, so the
        # watchdog would arm on phantom sources and fire rc 42 at 1800 s on every healthy run.
        # The conjunct lives HERE, not inside `HeartbeatWatchdog.arm()`.
        with _seam("producer_liveness_conjunct"):
            _assert_pool_producers_live(pool)
        with _seam("watchdog.start"):
            run_safety.watchdog.start()

        # LAW-16 leg 3. The guard's only prior construction site had ZERO callers and read
        # `dict.get` defaults over a key no schema carried; the root constructs it now from
        # the config family's ONE resolver. Its critical arm SIGTERMs the process, which lands
        # on the handlers installed above.
        with _seam("DiskGuard"):
            guard_spec = resolve_disk_guard(config.monitor)
            disk_guard = DiskGuard(
                watch_path=checkpoint_dir, interval_sec=guard_spec.interval_sec,
                warn_gb=guard_spec.warn_gb, fail_gb=guard_spec.fail_gb,
                keep_all=_DISK_GUARD_KEEP_ALL, sink=run_safety.sink,
            )
            disk_guard.start()

        # RESERVED, NOT DEAD — the mixed-batch / pretrained-buffer path. `pretrained_buffer`,
        # `recent_buffer` and `bufs` are None here, so `train/batch_assembly.py` contributes
        # nothing while its config keys and resolver stamps stay live: a WIRING gap, not dead
        # code. **This path MUST NOT be deleted** — it is the corpus-mix candidate mechanism
        # of the bootstrap prereg row, one of the two arms the mint is still choosing between.
        with _seam("StepCoordinator"):
            coordinator = StepCoordinator(
                trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
                pool=pool, eval_pipeline=eval_pipeline,
                # `disk_guard` rides the EXISTING subsystems carrier: the O3 arm persists
                # the ring, and the one abort reaching O3 with `abort_rule` still unrecorded
                # is the disk guard's — it latches `critical_fired` BEFORE it signals, so this
                # is the only term that can see it in time.
                subsystems=SimpleNamespace(gpu_monitor=None, disk_guard=disk_guard),
                anchor_state=resolved_anchor, shutdown=shutdown,
                eval_model=getattr(trainer, "model", None), bufs=None,
                config=step_coordinator_cfg, full_config=config.model_dump(),
                train_cfg={}, mixing_cfg={}, run_id=run_id,
                sink=run_safety.sink, heartbeat=run_safety.heartbeat, monitor_cfg=monitor_cfg,
                heartbeat_watchdog=run_safety.watchdog, actor_sync=actor_sync,
            )

        # NOTHING is swallowed. The old blanket `except Exception -> log -> return` swallowed
        # actor-SYNC failures into an exit-0 return — a run that looks launched and never
        # syncs. The loop's failure propagates and `close_out` still runs in a `finally`.
        # The launch pin DERIVES from `identity.warm_start` — ONE source, no hand-synced twin;
        # an absent row is `None`.
        declared_warm_start = resolve_bc_warm_start(config.model_dump())
        # THE PIN HOLDS AT STEP 0 ONLY, and that qualifier is load-bearing: the config pin is
        # the WARM-START artifact's hash, the anchor a FRESH launch must have, but a RESUMED
        # run's anchor has legitimately moved (a promotion rewrites `best_model.pt`) and
        # asserting the step-0 pin against it refused a resume after ONE promotion.
        # So the pin's SOURCE follows the launch mode and neither mode is unpinned: a resume
        # asserts the anchor THE STOP RECORDED, carried on the sidecar.
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
            # THE RESUMABLE-STOP DECISION, taken HERE because this is the only scope holding
            # all three terms: `shutdown_save`, the guard's LATCHED `critical_fired`, and
            # `abort_rule`. The teardown below records the disk rule AFTER this call, so
            # `abort_rule` alone cannot see a disk abort. Anything short of all three saying
            # "operator stop" runs the terminal battery.
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
        # THE LIVE ARMING AUDIT, and this is its ONE production consumer. CI gate 12 audits
        # the `disk_space_exhausted` row against a CONFIG NUMBER with no process to ask, so it
        # cannot tell a guard that ran from one whose `check_once` raised on every tick. This
        # REPORTS and does not refuse; the refusal half belongs at the mint preflight.
        #
        # It sits at the TOP of the ladder because `run_safety.sink.close()` runs on the
        # PARTIAL path and `disk_guard.stop()` after it — an emit there hits a closed file,
        # which `JsonlEventSink` counts as a persistence failure and the heartbeat watchdog
        # turns into `os._exit(43)`. This is the only point where the sink is open on BOTH
        # paths. The counters are read before the thread is joined, but `checks_total` only
        # ever RISES, so the reading can under-report and never over-report.
        _emit_live_arming_audit(
            config, run_safety.sink,
            probes={DISK_GUARD_LIVENESS_PROBE:
                    lambda: disk_guard is not None and disk_guard.checks_total > 0},
        )
        if coordinator is None:
            # PARTIAL COMPOSITION: `close_out` never ran and never will, so this is the only
            # place the run-safety threads and the pool get stopped.
            #
            # Why the arm restriction: `close_out` owns neither call — repo-wide,
            # `watchdog.stop()` and `sink.close()` have EXACTLY ONE call site each in all of
            # `src/`, the two lines below, so on the COMPLETED path both are left to process
            # exit. The forcing cause is DEBT, not principle: seven off-list suites stand in
            # `SimpleNamespace` sinks and watchdogs implementing no `close`/`stop`, so an
            # unconditional teardown here reds them; completing that protocol against
            # concretes is the condition under which these two lines move out of the `if`.
            # Bounded meanwhile: the sink is line-buffered, the watchdog thread is a daemon,
            # and both production callers exit the process immediately after this returns.
            _stop_pool_if_start_attempted(pool, start_attempted=pool_start_attempted)()
            run_safety.watchdog.stop()
            run_safety.sink.close()
        # The eval pipeline's poller thread and any in-flight spawn child need explicit
        # teardown on BOTH paths: `close_out` drains the in-flight round on the completed path
        # but never joins the poller, and on the PARTIAL path it never ran at all.
        if eval_pipeline is not None:
            eval_pipeline.stop()
        # The disk guard is this root's on BOTH paths — `close_out` has never heard of it, and
        # a daemon guard that outlives its run will SIGTERM a process no longer running one.
        if disk_guard is not None:
            disk_guard.stop()
            # THE SEAM: a guard that fired stopped this run, and until this line said so the
            # run exited 0 — the handler writes `shutdown_save`/`running` and never
            # `abort_rule`, so a supervisor relaunched into the same full volume. Named HERE
            # and not in the guard because the name is a MANIFEST row's and `mantis.train` may
            # not import that module. The read happens AFTER `stop()` joined the guard thread,
            # so the latch is final; `record_abort` is set-once, so first fire wins.
            if disk_guard.critical_fired:
                shutdown.record_abort(DISK_SPACE_ABORT_RULE)
        # THE TERMINAL-EVAL SEAM. `close_out` computed the terminal round's result and then
        # DISCARDED it one frame below `ShutdownState` — the only object that can carry an
        # outcome to `main` — so a run whose terminal battery was killed or could not persist
        # its ladder state exited 0 and the supervisor recorded a clean finish.
        #
        # This read sits AFTER the disk-guard read so first-fire-wins keeps the ROOT CAUSE: a
        # disk-full run whose terminal eval then breaks BECAUSE the volume is full reports 47.
        #
        # The bare `EvalBrokenReason(raw)` is the RUNTIME half of the unrepresentability claim
        # — a `# type: ignore` slips past the typed chokepoints, and the reason crosses a JSON
        # boundary where types do not travel. A spelling no member spells is a loud
        # `ValueError` here rather than a silent rc 0.
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


def _lazy_save_anchor(*args: Any, **kwargs: Any) -> None:
    from mantis.train.anchor import save_best_model_atomic

    save_best_model_atomic(*args, **kwargs)


def _lazy_guarded_load(model: Any, state_dict: Any) -> None:
    from mantis.train.anchor import _guarded_load_state_dict

    _guarded_load_state_dict(model, state_dict)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m mantis.run --config <path> --out-dir <path>` — the production launcher.

    `--config` and `--out-dir` are required and NEITHER has a `default=`: a defaulted
    `--out-dir` is a run input the code decides, and every run that forgets the flag then
    writes into one shared directory. `--resume-from` is the one OPTIONAL flag, and its
    `default=None` selects no action rather than picking a value; it is a flag rather than a
    schema key because a resume target is a property of THIS invocation. There is NO `--device`
    flag and no eval switch — `config.train.device` and `config.eval_enabled` are the only
    routes. The preflight's in-repo `--out-dir` refusal is deliberately not mirrored here: that
    guard exists because a CI gate must not dirty the tree it gates.

    rc policy goes through the SAME `exit_code_for_abort` the preflight child reads. Three
    rules reach it with an authored code: `draw_rate_collapse` (46), `disk_space_exhausted`
    (47) off the guard's latch, and `terminal_eval_broken` (48) off the coordinator's set-once
    latch, read AFTER the disk-guard read so first-fire-wins keeps the root cause. A signal
    this process did NOT send itself still resolves to 0. rc 1 is not an AUTHORED code but
    CPython's rc for any exception leaving `main`, so `UnregisteredAbortExitError` and any
    composition wall are indistinguishable to a supervisor; no code is invented for either.

    Raises:
        PreflightStampRefusal: the loaded config has no passing preflight stamp for this tree
            (R348(c)); one of its three named subclasses says which fact is missing.
        UnregisteredAbortExitError: an abort fired whose rule authors no exit code.
    """
    # THE FIRST STATEMENT, before argparse and before any collaborator exists: the window this
    # closes is exactly "the supervisor was killed while the run was still coming up". A NO-OP
    # unless a mantis supervisor stamped its pid in the environment — an unconditional arm
    # would SIGKILL every unattended run the moment its launching shell exited, and would arm
    # the pytest process through this function's in-process callers. Never at import time.
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
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    # THE ONE mantis stderr handler, installed at the process entry. Until this call
    # `configure_logging` had ZERO callers, so Python's lastResort dropped every `logger.info`
    # and printed WARNING+ unformatted. Installed BEFORE the launch so the boot path's own
    # diagnostics are covered.
    configure_logging()

    # `resolve_bootstrap` exists to fail a stale `--resume-from` AT LAUNCH, before
    # `torch.load`, and had zero callers — so a mistyped path surfaced as a torch error deep
    # inside `init_trainer`'s resume branch, after the composition root had built a run.
    bootstrap = resolve_bootstrap(args.resume_from)
    config = load_config(args.config)
    # R348(c): a run cannot skip the manual preflight by not running it; the stamp is keyed by
    # the config's identity hash and bound to the tree that preflighted it.
    stamp = require_preflight_stamp(config, tree_root=Path(__file__).resolve().parent)
    _LOG.info("preflight_stamp_accepted config_sha256=%s tree_sha=%s preflight_utc=%s",
              stamp["config_sha256"], stamp["tree_sha"], stamp["preflight_utc"])
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
