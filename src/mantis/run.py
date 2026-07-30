# >300 justify (R8), stated at this file's MEASURED size of 734 lines. WPMAIN made this
# module the ONE composition authority in fact and not only in name: the collaborator
# builder (`build_run_collaborators` + `_select_buffer`), the launcher (`launch_run`,
# `main`, `UnregisteredAbortExitError`) and LAW-16's three legs (signals, watchdog, disk
# guard) all landed here, lifted out of a CI GATE (`tools/ci_gates/preflight_mint.py`),
# which is the one-authority violation CARD-RUN-MAIN existed to end. The original R8
# argument extends verbatim to that layer and is the reason a split is argued AGAINST, not
# merely skipped: this module is the only one importing both `mantis.train` and `mantis.eval`
# at top level (§a.4/§c.6), `build_run_collaborators` is the only module-external consumer of
# `resolve_coordinator_knobs(...).capacity`, and moving the builder out would either put a
# `mantis.run -> sibling` import in the one place the DAG forbids new edges or create a
# SECOND place a `StepCoordinatorConfig` / a collaborator set can be built — which is
# precisely the two-surfaces shape `tests/test_run_one_authority.py`,
# `tests/config/test_drawrate_arming_authority.py` and `test_coordinator_knobs_wiring.py`
# exist to forbid. The honest alternative (a `mantis.launch` sibling plus a repo_design §2
# amendment) is recorded as REJECTED for that reason. Executable content stays a minority of
# the file; the rest is the per-decision rationale R8's clause protects (R64, MF-1/MF-2,
# S-4/Phase D/K-A/K-B, R120/R121/R122/R125/R126). The previous header claimed 335 lines
# against a 349-line file; SF-7 — a justification that is not true is worse than none — so
# the number above is re-measured by `wc -l`, never transcribed.
"""mantis.run — the run composition root AND the run launcher (design §a.4/§c.6).

TOP-LEVEL module, ABOVE both `mantis.train` and `mantis.eval` — the ONE module that
imports both at module top level (no lazy-import loophole); nothing imports `mantis.run`,
so it is a source-only DAG node and the §2 "train -> all above except eval" ban stays
verbatim (census-tested: tests/test_run_composition.py::
test_no_train_module_imports_eval_even_lazily).

`python -m mantis.run --config <path> --out-dir <path>` is the entry point (CLAUDE.md
`python -m mantis.*` law). It is a REAL launcher: it loads the config through the one
loader, builds the collaborators, composes the run, drives the live loop and maps a fired
hard abort to a process rc through `mantis.config.armed_aborts.exit_code_for_abort` — the
same resolver the mint preflight's child reads. Until WPMAIN it validated the config,
printed a readiness line and returned 0 — which is what made "the preflight boots what run5
boots" a claim with no producer on either side. That readiness print is DELETED, not
demoted: a boot record that is a stdout line nobody parses is not a boot record, and the
run's own event stream (`run_boot_identity`, `resolved_config`) is the record now.

The composition is TWO shared functions plus one composed entry, and that shape is forced:
`build_run_collaborators` builds the collaborator set, `compose_run` composes and drives,
`launch_run` is exactly the pass-through between them. The preflight child calls the SAME
two functions with its two sanctioned instruments wrapped AROUND them (a config-level burst
override before, a read-only resumed-trainer refusal between) — so there is no opaque
`boot()` with a preflight hook smuggled inside it, which would be the divergence seam the
whole card is about.

`compose_run` stays INJECTION-FIRST: every collaborator (trainer/pool/buffer) arrives via a
kwarg and is never built inside it (R-10) — the builder layer sits AROUND it, so the
fakes-testable seam survives. But no parameter may carry a CONFIG FACT: `eval_enabled`
(R120), `run_id` (R123) and the device (R126) are all read from the validated config, and
their parameters are DELETED rather than merely stripped of defaults — a required parameter
is a forcing route with the default removed, not a closed one. The pool still builds only
via the legacy hparams dict path elsewhere (R-SELFPLAYCONFIG-SCHEMA, unchanged debt, now
cited from the builder). WP-UNFREEZE lives here: this root builds the continuous actor-sync
engine (`mantis.train.actor_sync.ActorSync`) UNCONDITIONALLY and wires the actor-lag
watchdog callables (`actor_ckpt_step` / learner step) into `build_run_safety`.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch

from mantis.config.armed_aborts import exit_code_for_abort
from mantis.config.emit import resolve_config
from mantis.config.loader import config_identity_sha256, load_config
from mantis.config.resolve.actor_sync import resolve_actor_sync_cadence
from mantis.config.resolve.composition import require_run_config, revalidate_run_config
from mantis.config.resolve.coordinator import CoordinatorKnobsSpec, resolve_coordinator_knobs
from mantis.config.resolve.disk_guard import resolve_disk_guard
from mantis.config.resolve.drain import DrainCapsSpec, resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec, resolve_draw_rate_abort
from mantis.config.resolve.monitor import resolve_monitor_config
from mantis.config.resolve.run_length import resolve_max_train_steps
from mantis.config.schema import RunConfig
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.monitor.config import MonitorConfig
from mantis.selfplay.pool import WorkerPool
from mantis.train.actor_sync import ActorSync
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.dispatch import RepresentationRouteError
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.determinism import seed_everything
from mantis.train.emit import emit_via
from mantis.train.lifecycle.disk_guard import DiskGuard
from mantis.train.lifecycle.signals import ShutdownState, install_signal_handlers
from mantis.train.loop import run_training_loop
from mantis.train.orchestrator import init_trainer
from mantis.train.subsystems import build_run_safety

#: The 3 pipeline stages every run wires unconditionally; "eval_round" joins them iff an
#: eval pipeline is actually built (the caller DECLARES what it handed `heartbeat=` to).
_BASE_WIRED_SOURCES: tuple[str, ...] = ("train_step", "inference_dispatch", "selfplay_drain")

#: `DiskGuard.keep_all` gets NO config key (R122): it is a PRUNING knob the safety
#: thresholds deliberately ignore (`disk_guard.py`'s own verbatim invariant), and its only
#: consumer is the "thresholds ignore it" pin. Passing it explicitly here rather than
#: leaving it to a constructor default is the MF-2 posture — a parameter default is a
#: MIGRATED authority, not an absent one — and this constant is the disclosure.
_DISK_GUARD_KEEP_ALL = False


class UnregisteredAbortExitError(RuntimeError):
    """A hard-abort rule FIRED and the manifest authors no exit code for it.

    The launcher's three outcomes are the child's three, by design (D-6): `abort_rule is
    None` is the ONLY thing that means a clean run; a rule with an authored code exits with
    the code the manifest row carries, resolved and never written here; a rule with NO
    authored code is a NAMED failure. `grad_norm_hard_abort` and `sealbot_wr_abort` share
    `_fire_hard_abort` and neither is pre-registered — R84 declined to invent codes for
    them, and inventing one here would be that same class one layer up. Reporting an
    aborted run as rc 0 is strictly worse: the supervisor above relaunches into the wall.
    """


class RunHandles(NamedTuple):
    """What `compose_run` hands back — enough for a caller to inspect or drive further."""

    coordinator: Any
    run_safety: Any
    eval_pipeline: Any | None
    shutdown: ShutdownState


class RunCollaborators(NamedTuple):
    """What `build_run_collaborators` hands back: the three injected collaborators plus the
    ONE derivation of the run's two output directories INSIDE THE BOOT (D-8).

    Stated at its measured scope, because the wider claim is false. Before WPMAIN the
    launcher and the preflight child each derived `logs/` and `checkpoints/` themselves;
    both now read them off this tuple, so no BOOT can derive them differently. What survives
    is one derivation OUTSIDE the boot: `tools/ci_gates/preflight_mint.py:1000` writes its
    own `log_dir = out_dir / "logs"` in the preflight PARENT and uses it for the
    `PreflightOutDirReusedError` stale-segment refusal and for `_read_segment`. Repo-wide the
    expression has exactly two producers — `run.py:269` and that line.

    Not folded in here, and the residue is named rather than papered over (REVIEW-impl F-3,
    queued as Q-D8-PARENT-DERIVATION): the parent half must stay importable WITHOUT torch,
    which is why the child imports `mantis.run` function-locally at all (DESIGN §1.4) — this
    module imports `torch` at line 68. Sharing the derivation therefore means a new
    torch-free module and a new exported symbol, i.e. a live-consumer row and a DAG row, for
    a two-line expression; that is materially wider than a truth-correction and is not this
    condition's scope. The consequence a future reader must know: rename the child's
    directory and the parent's reuse guard silently stops guarding while the segment read
    goes empty."""

    trainer: Any
    pool: Any
    buffer: Any
    log_dir: Path
    checkpoint_dir: Path


@contextmanager
def _seam(name: str) -> Iterator[None]:
    """NAME a composition seam without catching anything (R64/LAW-14).

    A collaborator wall stays BARE: no wrapping class, no except arm that decides what the
    failure means, because a wall is a TREE DEFECT and must look like one. What a bare wall
    lacks is WHERE it happened, so this annotates the in-flight exception with a PEP 678
    note and re-raises it unchanged — same type, same traceback, nothing swallowed. The
    preflight's rc-32 sniff is unaffected: notes append BELOW the traceback whose final
    exception line still carries the `object has no attribute` text the classifier reads.
    """
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 — annotate-and-re-raise; nothing is caught
        exc.add_note(f"composition seam: {name}")
        raise


def _stop_pool_if_started(pool: Any, *, pool_started: bool) -> Callable[[], None]:
    """The item-11 closure (§c.7): `pool.stop()` only if THIS run's own `pool.start()`
    actually fired. An unstarted pool's `InferenceServer.join(timeout=5.0)` raises on a
    never-started thread (pool.py:335) — calling `.stop()` unconditionally on a
    never-started pool is the real hazard this guard exists to close."""
    def _stop() -> None:
        if pool_started:
            pool.stop()
    return _stop


def _resolve_monitor_cfg(config: RunConfig) -> MonitorConfig:
    """WPAX S-1/S-2: a plain typed section read through the monitor section's ONE resolver.

    This used to be a member of the duck-typed config-section family, whose absent-section
    arm returned a bare `MonitorConfig()` — and a bare one carries
    `actor_lag_abort_enabled=False`, so it silently DISARMED the hard abort `configs/run5.yaml`
    ships armed (ADJ-07). `compose_run`'s gate makes the section typed and present, so there
    is no absent arm left to take."""
    return resolve_monitor_config(config.monitor)


def _resolve_actor_sync_cadence_steps(config: RunConfig) -> int:
    """The train-section twin of `_resolve_monitor_cfg`: `train.actor_sync_cadence_steps`
    through its ONE resolver (K1). Its retired smoke arm substituted cadence 1 for any
    config object without a train section — a test-only value on a production axis."""
    return resolve_actor_sync_cadence(config.train)


def _select_buffer(config: Any, capacity: int) -> Any:
    """Select the replay buffer off `config.identity.representation` — never sniffed off a
    live module, never defaulted (LAW-11). An unknown or absent representation RAISES.

    LIFTED out of `tools/ci_gates/preflight_mint.py::_build_buffer` (D-1/D-2). The move is
    not tidiness: the raise below is the boot's one LAW-11 refusal, and while it sat in
    `tools/` CI gate 11 could not see it at all (`silent_encoding_gate.py`'s
    `SCAN_ROOTS = ("src", "crates")`). It is now inside the scan, and measured quiet there —
    gate 11's patterns all require a REGISTERED-ENCODING literal in a default position, and
    this function contains no encoding literal at all: it passes `config.identity.encoding`
    affirmatively. The name loses the tool-side `_build_` residue with the move (R73).

    `RepresentationRouteError` is REUSED, not invented: the SAME axis already raises it for
    the train-step route (`mantis.train.coordinator.dispatch`, whose own docstring cites
    "the `_build_buffer` posture" by name). One error family per axis. It carries no `rc` and
    correctly should not — a `src/` exception carrying a CI tool's exit code is the layering
    defect this WP ends, and R125 ruled the child-seam mapping of this error onto the CI
    tool's rc-10 config-error class REJECTED rather than kept alive to feed a test (R116).
    (The tool class is named by DESCRIPTION and not spelled, because the oracle for this
    rule scans this very docstring — see `tests/test_run_buffer_route.py`.)

    RIDER, recorded here so the next reader finds it in-tree (R125): the third arm is
    UNREACHABLE from a validated `RunConfig` today — `Literal["grid","graph"]` plus the
    registry cross-check make an unknown representation unrepresentable. LAW-11 makes
    widening the representation enum a deliberate design act, and whoever widens it re-opens
    child-seam routing in that same design; until then this error propagates through the
    preflight child as an UNCAUGHT loud failure with a full named traceback (LAW-14), never
    a silent arm.
    """
    representation = config.identity.representation
    if representation == "graph":
        # Lazy, with a STATED reason (repo_design §87): `mantis._engine` is not an edge on
        # §2's `run` row, and the extension module is the one import this root must not
        # make unconditional.
        from mantis._engine import HexgBuffer

        return HexgBuffer(capacity, config.identity.encoding)
    if representation == "grid":
        from mantis._engine import ReplayBuffer

        return ReplayBuffer(capacity, config.identity.encoding)
    raise RepresentationRouteError(
        f"identity.representation {representation!r} selects no buffer — an absent or "
        "unknown representation is an ERROR, never a dense default (LAW-11)"
    )


def build_run_collaborators(*, config: RunConfig, out_dir: str | Path) -> RunCollaborators:
    """Build the three injected collaborators and derive the run's output directories.

    Lifted VERBATIM IN SEQUENCE from the preflight child's own boot (D-1): seed, derive
    dirs, trainer, capacity, buffer, pool. The child now calls this function instead of
    re-implementing it, which is the whole of success criterion 2 — every composition step
    both callers take is one of these two functions.

    NO `device` PARAMETER (R126/MF-1). The device is a CONFIG FACT — `config.train.device`,
    typed and required — and `torch.device(...)` is applied ONCE here and threaded to both
    consumers. `init_trainer` and `WorkerPool` keep their own `device` constructor
    parameters: those are collaborator threading BELOW the composition surface, not
    config-fact carriers. What must be unrepresentable is a CALLER that can point the boot
    at a different device than the config declares — which is exactly how a `--device cpu`
    preflight false-cleared a cuda-minted run's 16 GiB GPU wall (CARD-RUN5-GPU-OOM).

    `checkpoint_path` is deliberately never passed to `init_trainer`: resume threading is
    the owed S-2 work and is untouched here (resume fires only on an explicit
    `checkpoint_path`, `orchestrator.py:97,113`), which is also what makes the preflight's
    §4.2 resumed-trainer refusal a meaningful read.

    `capacity` comes from `resolve_coordinator_knobs(config.train).capacity` — the sanctioned
    one-authority read of the 19 coordinator knobs. The throwaway `StepCoordinatorConfig`
    the child used to build just to read `.capacity` DIES with this (D-3): the real one is
    built exactly once, inside `compose_run`, so no second construction site exists.
    """
    # R30a — the ONE determinism boot site: seed before any RNG-consuming object exists.
    # Both callers used to seed at their own site; there is one site now.
    seed_everything(config.seed)

    out_dir = Path(out_dir)
    log_dir = out_dir / "logs"
    checkpoint_dir = out_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.train.device)
    with _seam("init_trainer"):
        trainer = init_trainer(config=config.model_dump(), checkpoint_dir=str(checkpoint_dir),
                               device=device, sink=None)
    capacity = int(resolve_coordinator_knobs(config.train).capacity)
    with _seam("_select_buffer"):
        buffer = _select_buffer(config, capacity)
    with _seam("WorkerPool"):
        # R-SELFPLAYCONFIG-SCHEMA (unchanged debt, now cited from the builder rather than
        # from an injection-first disclaimer): the pool still builds only via the legacy
        # hparams dict path elsewhere, so it is handed `config.model_dump()`.
        pool = WorkerPool(model=trainer.model, config=config.model_dump(), device=device,
                          replay_buffer=buffer, arch=trainer.arch, sink=None, heartbeat=None)
    return RunCollaborators(trainer=trainer, pool=pool, buffer=buffer, log_dir=log_dir,
                            checkpoint_dir=checkpoint_dir)


def _step_coordinator_config(
    *,
    stop_step: int,
    draw_rate_abort: DrawRateAbortSpec | None,
    drain_caps: DrainCapsSpec,
    knobs: CoordinatorKnobsSpec,
) -> StepCoordinatorConfig:
    """Assemble `StepCoordinatorConfig` from RESOLVED CONFIG FACTS ONLY — zero literals
    (WPMINT Phase K-B closes `CARD-COORD-KNOBS`, R78 as clarified by R80).

    This function's own docstring used to open "Smoke-grade defaults … for the ~22 knobs
    R-TRAINCONFIG-SCHEMA / CARD-COORD-KNOBS (R78) still owns", and the literal below carried
    them: `eval_interval`, `log_interval`, `batch_size`, `hard_gn_threshold`,
    `selfplay_stall_timeout_sec` and fourteen more decided what every run WAS from a number
    no config could see and no mint record published. R78 named the deadline (pre-run5-mint);
    `knobs` is it. Six further fields had no reader at all and are DELETED rather than
    authored (call K-a) — see `mantis.config.resolve.coordinator`.

    The CONFIG-AUTHORED values are PARAMETERS **with no default of their own**. That is not
    style: a literal the caller always replaces is a second default authority (R1), and so
    is a parameter default — the authority would merely MIGRATE from the dataclass field to
    this signature, leaving every `dataclasses.fields()` assertion green while a caller that
    omits the argument silently inherits a posture (MF-2 Attack B). `tests/config/
    test_drawrate_arming_authority.py` pins `stop_step`/`draw_rate_abort`'s
    `Parameter.empty` for exactly that reason (R83),
    `tests/config/test_drain_caps_wiring.py` pins `drain_caps`' and
    `tests/config/test_coordinator_knobs_wiring.py` pins `knobs`'; the renamed function is
    the name-truth half (R73): it no longer DEFAULTS the facts the config authors.

    WPMINT Phase K-A (R93): `drain_caps` was the third such fact. The `900.0` that used to
    sit in the literal below, and the three `StepCoordinatorConfig` terminal defaults beside
    it, were the run's REAL drain caps while the minted, schema-validated,
    registry-claimed `monitor.drain.*` block was popped and discarded by
    `resolve_monitor_config` (the DR-11 finding). The four values now arrive whole, through
    `resolve_drain_caps`, or this call raises.
    """
    return StepCoordinatorConfig(
        eval_interval=knobs.eval_interval,
        log_interval=knobs.log_interval,
        checkpoint_interval=knobs.checkpoint_interval,
        min_buf_size=knobs.min_buf_size,
        capacity=knobs.capacity,
        buffer_schedule=knobs.buffer_schedule,
        training_steps_per_game=knobs.training_steps_per_game,
        max_train_burst=knobs.max_train_burst,
        batch_size=knobs.batch_size,
        augment=knobs.augment,
        recency_weight=knobs.recency_weight,
        mixing_initial_w=knobs.mixing_initial_w,
        mixing_min_w=knobs.mixing_min_w,
        mixing_decay_steps=knobs.mixing_decay_steps,
        hard_gn_threshold=knobs.hard_gn_threshold,
        hard_gn_min_steps=knobs.hard_gn_min_steps,
        stop_step=stop_step,
        draw_rate_abort=draw_rate_abort,
        final_eval_drain_timeout_sec=drain_caps.final_eval_drain_timeout_sec,
        eval_final_drain_safety_factor=drain_caps.eval_final_drain_safety_factor,
        eval_final_drain_hard_cap_sec=drain_caps.eval_final_drain_hard_cap_sec,
        terminal_eval_hard_cap_sec=drain_caps.terminal_eval_hard_cap_sec,
        terminal_eval_enabled=knobs.terminal_eval_enabled,
        bot_batch_share=knobs.bot_batch_share,
        selfplay_stall_timeout_sec=knobs.selfplay_stall_timeout_sec,
    )


def compose_run(
    *,
    config: RunConfig | Any,
    trainer: Any,
    pool: Any,
    buffer: Any,
    log_dir: str | Path,
    checkpoint_dir: str | Path,
) -> RunHandles:
    """The run composition root (§c.6). Injection-first: every COLLABORATOR arrives via a
    kwarg, never built here (R-10) — but no parameter may carry a CONFIG FACT: the gate on
    the first line below is the ONE authority for what this root may be composed from, and
    the parameter list is pinned by a signature census so a re-add cannot be silent (WPAX
    S-1/S-2, MF-1).

    `eval_enabled` (R120) and `run_id` (R123) were the last two parameters carrying config
    facts and are DELETED, not merely stripped of defaults. R64's "the preflight may never
    force False" is only structurally unrepresentable once there is no route to force
    anything through; and a caller-supplied `run_id != config.run_id` would split the JSONL
    segment identity from the config identity, which is the F-B1 class `run_boot_identity`
    exists to kill.

    MAIN-THREAD CALL, stated because it is a real precondition: `signal.signal` raises off
    the main thread, and this root installs LAW-16's handlers. Both production callers and
    every in-process pytest drive are on the main thread.
    """
    config = require_run_config(config, caller="compose_run")
    # RED-TEAM F-3: the gate above answers "is this the class?"; this answers "is this a
    # config the loader would accept?". `model_copy(update=…)` builds a genuine RunConfig
    # whose CROSS-FIELD validators never re-ran, and one such copy drove a 20-step run with
    # a single actor sync — run3's frozen actor — past the gate. Re-validating the dump
    # closes that route, `model_construct`, and post-gate mutation together. It stays a
    # SECOND statement because the gate must remain compose_run's first (pinned) and because
    # the two rules have different contracts: the gate is identity-preserving, this is not.
    config = revalidate_run_config(config, caller="compose_run")

    # LAW-16 leg 1 (F-1-SIGNALS). HOISTED to the top of the composition, before anything
    # starts. `install_signal_handlers` used to fire ONLY on `run_training_loop`'s
    # self-construct branch — and this root always injects its own state, so the branch never
    # ran: a probe over 19 real composed drives found SIGINT at `default_int_handler` and
    # SIGTERM at `SIG_DFL` on ALL 19. Save-then-exit was dead in every composed run,
    # including the preflight child's.
    #
    # THE WINDOW IS DEFINED, NOT GLOSSED: "signal-covered" during composition means the
    # STATE IS SET, not that save-then-exit fires mid-compose. Nothing between this install
    # and `run_training_loop` polls the state, so a signal in that window lets composition
    # COMPLETE (eval pipeline, `pool.start()`, watchdog, disk guard, coordinator) and the
    # save happens at the loop's entry-set arm, after which `close_out` drains and the
    # teardown ladder below leaves nothing half-alive. A pre-`pool.start()` `if not
    # shutdown.running:` bail-out was argued against and NOT added: a rarely-exercised extra
    # branch in the one composer is worse than a bounded, pinned window. A signal arriving
    # BEFORE this line (during `build_run_collaborators`) takes the default disposition —
    # defined and clean, because nothing has started yet.
    shutdown = ShutdownState()
    install_signal_handlers(shutdown)

    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)
    # The DERIVATION of these two paths lives once in the BOOT, in `build_run_collaborators`
    # (D-8, at the scope `RunCollaborators` states it and no wider) — this
    # is not a second derivation, it is the root making the directory it was HANDED usable.
    # It is load-bearing for LAW-16 leg 3: the disk guard stats `watch_path`, and its poll
    # thread swallows its own errors by design (a monitor thread must not kill the run), so a
    # checkpoint dir that does not exist yet buys a guard that runs, logs, and publishes
    # NOTHING — armed in the composition, absent in effect. `build_run_safety` already
    # creates `log_dir` for the same reason; this is its twin.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor_cfg = _resolve_monitor_cfg(config)
    run_id = config.run_id

    wired_sources: list[str] = list(_BASE_WIRED_SOURCES)
    if config.eval_enabled:
        wired_sources.append("eval_round")

    # WP-UNFREEZE §4.3: the lag-watchdog callables are read LIVE at poll time, never at
    # build time — `actor_sync` is assigned immediately below, before anything can start
    # (this root owns both the assignment and `watchdog.start()`). DESIGN §4.3's
    # "ActorSync first" ordering is inverted here because the engine's LAW-18 sink IS
    # `run_safety.sink`, which only exists after this call.
    with _seam("build_run_safety"):
        run_safety = build_run_safety(
            log_dir=log_dir, run_id=run_id, buffer=buffer,
            buffer_persist_path=checkpoint_dir / "replay_buffer.bin",
            wired_sources=wired_sources, monitor_cfg=monitor_cfg,
            actor_ckpt_step_fn=lambda: actor_sync.actor_ckpt_step(),
            learner_step_fn=lambda: int(trainer.step),
        )

    # F-B1 closure (WPCLEAN Phase RES): the booted process publishes ITS OWN post-revalidation
    # config identity into the run's event stream, first thing after the sink exists — before
    # anything can wedge. One authority (`config_identity_sha256`) on both sides: the mint
    # preflight's parent hashes the config IT loaded with the same function and compares, so
    # a child that read a different file is a NAMED preflight failure instead of invisible.
    emit_via(run_safety.sink, {
        "event": "run_boot_identity",
        "run_id": run_id,
        "config_sha256": config_identity_sha256(config),
    })
    # §5.4 (LAW-08): `resolve_config` / `to_event_payload` had ZERO production call sites —
    # a resolved-config surface with no emitter, i.e. a payload nobody had ever published.
    # The run now publishes its own resolved posture into its own stream, immediately after
    # the identity witness (which must land FIRST: it is the F-B1 closure and has to exist
    # even if the boot later wedges). Exactly once per segment, and the payload IS
    # `to_event_payload(resolve_config(config))` — a hand-assembled copy would be a second
    # authority for the run's resolved posture. Producer-manifest row: `resolved_config`.
    emit_via(run_safety.sink, resolve_config(config).to_event_payload())

    # WP-UNFREEZE (R49): the continuous-sync engine is built UNCONDITIONALLY — no config
    # or eval state may make actor sync conditional (pinned by
    # tests/train/test_actor_sync_isolation.py). The actor's weights come from the
    # learner on a cadence and NEVER from a gate decision.
    actor_sync = ActorSync(
        target=pool,
        state_dict_fn=trainer.inference_state_dict,
        step_fn=lambda: int(trainer.step),
        cadence_steps=_resolve_actor_sync_cadence_steps(config),
        sink=run_safety.sink,
        run_id=run_id,
    )

    # M-4: the StepCoordinatorConfig instance is built FIRST — DrainCaps is LIFTED from
    # its own 4 fields, never a second, independently-hardcoded set of literals. The two
    # used to duplicate config.py's own defaults (900.0/3.0/14400.0/14400.0) by
    # coincidence; a future default change there would have silently diverged the two
    # (R1: duplicated default authority).
    # WPAX S-4 + Phase D + WPMINT Phase K-A/K-B: `stop_step` (train.max_train_steps),
    # `draw_rate_abort` (train.draw_rate_abort), `drain_caps` (monitor.drain) and `knobs`
    # (the 19 `train.*` step-coordinator keys) are the facts the CONFIG authors, and they are
    # PASSED IN through their own resolvers rather than replaced afterwards — a
    # `dataclass_replace` over a defaulted object requires a complete object first, i.e. a
    # literal, and a literal that is always overwritten is still a second default authority
    # (R1). With `knobs` there are no unauthored knobs left: `_step_coordinator_config` holds
    # zero literals and R78's card is closed.
    step_coordinator_cfg = _step_coordinator_config(
        stop_step=resolve_max_train_steps(config.train),
        draw_rate_abort=resolve_draw_rate_abort(config.train),
        drain_caps=resolve_drain_caps(config.monitor),
        knobs=resolve_coordinator_knobs(config.train),
    )

    resolved_anchor = SimpleNamespace(best_model=None, best_model_step=None)
    eval_pipeline = None
    if config.eval_enabled:
        eval_pipeline = build_eval_pipeline(
            eval_cfg=config.eval,
            coordinator_cfg_caps=DrainCaps(
                final_eval_drain_timeout_sec=step_coordinator_cfg.final_eval_drain_timeout_sec,
                eval_final_drain_safety_factor=step_coordinator_cfg.eval_final_drain_safety_factor,
                eval_final_drain_hard_cap_sec=step_coordinator_cfg.eval_final_drain_hard_cap_sec,
                terminal_eval_hard_cap_sec=step_coordinator_cfg.terminal_eval_hard_cap_sec,
            ),
            encoding=config.identity.encoding,
            run_id=run_id, spool_dir=log_dir / "eval_spool",
            ladder_state_path=log_dir / "eval_ladder_state.json",
            promotion=DeployTagHooks(
                anchor_state=resolved_anchor,
                best_model_path=checkpoint_dir / "best_model.pt", run_id=run_id,
                encoding=config.identity.encoding,
                save_anchor=_lazy_save_anchor, guarded_load=_lazy_guarded_load,
            ),
            sink=run_safety.sink, heartbeat=run_safety.heartbeat,
        )

    pool_started = False
    coordinator = None
    disk_guard = None
    # TEARDOWN LADDER (§8, the pre-registered RED-TEAM lens: builder N succeeds, builder N+1
    # raises). By the time `StepCoordinator` is constructed the pool is started, the watchdog
    # thread is polling and the disk-guard thread is running; if a raise merely propagated,
    # all three would survive the failed compose — worker processes, a watchdog whose
    # `exit_fn` is `os._exit`, and a daemon guard that will SIGTERM a process no longer
    # running a run. The contract is: no worker process and no non-daemon thread survives a
    # failed compose, nothing is half-alive, and the failure that propagates is the ORIGINAL
    # (nothing here catches anything — the `finally` runs and the exception continues; a
    # teardown failure would chain as its `__context__`).
    #
    # `coordinator is None` is the discriminator, not a second flag: once the coordinator
    # exists, `close_out` is the epilogue that owns the drain, the buffer save and the
    # guarded pool stop, and re-stopping the pool after it would be a second authority for
    # the same teardown. Before it exists, nothing else will ever stop what this root
    # started, so this ladder does.
    try:
        # ORDER PINNED (subsystems.py:213-215 contract): pool starts, THEN the watchdog.
        pool.start()
        pool_started = True
        run_safety.watchdog.start()

        # LAW-16 leg 3 (F-2-DISKGUARD). At HEAD the guard was constructed at exactly one
        # site — `build_subsystems`, which had ZERO callers — from `dict.get` defaults over
        # a key no schema carried. R121(b) mandates the root construct it; R1 forbids the
        # values being literals; R122 grants the config family and its ONE resolver, so the
        # root reads no `config.monitor.disk_guard` attribute of its own. Its critical arm
        # SIGTERMs the process, which now lands on the handlers installed above — F-1 and
        # F-2 were coupled defects and they close together.
        guard_spec = resolve_disk_guard(config.monitor)
        disk_guard = DiskGuard(
            watch_path=checkpoint_dir, interval_sec=guard_spec.interval_sec,
            warn_gb=guard_spec.warn_gb, fail_gb=guard_spec.fail_gb,
            keep_all=_DISK_GUARD_KEEP_ALL, sink=run_safety.sink,
        )
        disk_guard.start()

        with _seam("StepCoordinator"):
            coordinator = StepCoordinator(
                trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
                pool=pool, eval_pipeline=eval_pipeline,
                subsystems=SimpleNamespace(gpu_monitor=None),
                anchor_state=resolved_anchor, shutdown=shutdown,
                eval_model=getattr(trainer, "model", None), bufs=None,
                config=step_coordinator_cfg, full_config=config.model_dump(),
                train_cfg={}, mixing_cfg={}, run_id=run_id,
                sink=run_safety.sink, heartbeat=run_safety.heartbeat, monitor_cfg=monitor_cfg,
                heartbeat_watchdog=run_safety.watchdog, actor_sync=actor_sync,
            )

        # WPAX S-5: NOTHING is swallowed. The old blanket `except Exception -> log -> return`
        # existed so a fakes harness could not crash this root; that is the same defect as the
        # smoke resolver arm (accommodating test doubles in production code), and it also
        # swallowed actor-SYNC failures into an exit-0 return — a run that looks launched and
        # never syncs, which is run3's silent freeze with the backstop routed around it.
        # Fail-loud law wins: the loop's failure propagates, and `close_out` still runs in a
        # `finally` so the buffer save and the guarded pool stop are not lost. If `close_out`
        # also raises, Python chains the loop failure as its `__context__`.
        try:
            run_training_loop(trainer=trainer, shutdown_state=shutdown,
                              eval_pipeline=eval_pipeline, coordinator=coordinator,
                              anchor_state=resolved_anchor, sink=run_safety.sink)
        finally:
            coordinator.close_out(
                on_drained=_stop_pool_if_started(pool, pool_started=pool_started))
    finally:
        if coordinator is None:
            # PARTIAL COMPOSITION. `close_out` — the epilogue that owns the drain, the
            # buffer save, the staleness DISARM and the guarded pool stop — never ran and
            # never will, so this is the only place the run-safety threads and the pool get
            # stopped.
            #
            # Why the arm restriction, stated TRUE (REVIEW-impl F-2 measured the previous
            # sentence here FALSE and SF-7 makes a false justification worse than none):
            # `close_out` (`train/coordinator/drain.py:140-171`) does NOT own either call.
            # It disarms staleness, flushes the eval, runs `on_drained` and runs the terminal
            # eval — it never touches the watchdog thread and never touches the sink.
            # Repo-wide, `watchdog.stop()` and `sink.close()` have EXACTLY ONE call site each
            # in all of `src/`, and it is the two lines below. So on the COMPLETED path
            # neither runs and nothing else runs them: both are left to process exit, which
            # is bounded but is not a teardown anybody owns.
            #
            # The forcing cause is not a principle, it is DEBT: seven off-list suites stand
            # in `SimpleNamespace` sinks and watchdogs that implement no `close`/`stop`, so an
            # unconditional teardown here reds them. R131 countersigned the deviation and
            # REFUSED to accept it as shape — "production code contorting around
            # under-implemented test fakes is the tail wagging the dog" — and routed the fix
            # to CARD-PROTOCOL-COMPLETE (R106): complete the sink/watchdog protocol against
            # concretes, THEN lift this restriction so teardown runs unconditionally. That
            # card is the condition under which these two lines move out of the `if`; nothing
            # here is a reason to close it as a no-op.
            #
            # Bounded, meanwhile, and measured rather than assumed: the sink is line-buffered
            # (`monitor/sink.py:120`), the watchdog thread is a daemon
            # (`train/lifecycle/heartbeat_watchdog.py:234-239`), and both production callers
            # exit the process immediately after this returns.
            _stop_pool_if_started(pool, pool_started=pool_started)()
            run_safety.watchdog.stop()
            run_safety.sink.close()
        # The disk guard is this root's on BOTH paths — `close_out` has never heard of it
        # (it is composed here for the first time in this WP), and a daemon guard that
        # outlives its run will SIGTERM a process that is no longer running one.
        if disk_guard is not None:
            disk_guard.stop()

    return RunHandles(coordinator=coordinator, run_safety=run_safety, eval_pipeline=eval_pipeline,
                      shutdown=shutdown)


def launch_run(*, config: RunConfig, out_dir: str | Path) -> RunHandles:
    """THE launch path: build the collaborators, compose the run. Nothing else.

    The body is EXACTLY those two calls, pass-through, and it is censused as such (O-A2): a
    third step here — a config transform, a device coercion, an `if resume:` branch — or a
    DIFFERENT config object handed to the composer than the collaborators were built from
    would be a divergent boot path wearing the one-authority name, and both mutations are
    behaviourally invisible on a green tier.
    """
    collaborators = build_run_collaborators(config=config, out_dir=out_dir)
    return compose_run(config=config, trainer=collaborators.trainer, pool=collaborators.pool,
                       buffer=collaborators.buffer, log_dir=collaborators.log_dir,
                       checkpoint_dir=collaborators.checkpoint_dir)


def _lazy_save_anchor(*args: Any, **kwargs: Any) -> None:
    from mantis.train.anchor import save_best_model_atomic

    save_best_model_atomic(*args, **kwargs)


def _lazy_guarded_load(model: Any, state_dict: Any) -> None:
    from mantis.train.anchor import _guarded_load_state_dict

    _guarded_load_state_dict(model, state_dict)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m mantis.run --config <path> --out-dir <path>` — the production launcher.

    R1 posture at the CLI boundary: BOTH inputs are required and NEITHER has a `default=`.
    A defaulted `--out-dir` in particular is a run input the code decides, and every run
    that forgets the flag then writes into one shared directory — which is how two runs'
    checkpoints end up in one lineage. A usage error is argparse's own rc 2.

    There is NO `--device` flag (R126): the device is `config.train.device`, so preflighting
    or launching run5 uses run5's own minted device and no invocation can point either
    caller somewhere else. There is no eval switch either (R64/O-10): `config.eval_enabled`
    is the only route.

    There is deliberately no in-repo `--out-dir` refusal here, unlike the preflight's
    `_checked_out_dir`: that guard exists because a CI GATE must not dirty the tree it
    gates. A production launch writing untracked `logs/`/`checkpoints/` is what R7 already
    anticipates ("never tracked", not "never written"), so the refusal stays preflight-only
    policy, parent-side in the tool.

    rc policy, through THE resolver (D-6 — the `repo_design.md` OWED paragraph, discharged):
    the launcher reads the SAME `exit_code_for_abort` the preflight child's `_abort_rc`
    reads, so the abort-to-rc mapping has one authority and is never re-derived.
    """
    parser = argparse.ArgumentParser(
        prog="python -m mantis.run",
        description="Launch a run from a minted config (the ONE composition authority).",
    )
    parser.add_argument("--config", required=True, help="path to the minted run config")
    parser.add_argument("--out-dir", required=True,
                        help="run artifacts root; logs/ and checkpoints/ are derived from it")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    handles = launch_run(config=load_config(args.config), out_dir=args.out_dir)
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
