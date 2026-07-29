# >300 justify (R8), stated at this file's MEASURED size of 335 lines. It crossed the cap at
# WPMINT Phase K-B, and the whole delta is `_step_coordinator_config`'s: with the 19 coordinator
# knobs authored, its body became a 25-field assembly from four resolved specs and its docstring
# had to record what the literals it replaced were doing. That assembly cannot move: this module
# is the ONE composition root (§a.4/§c.6), it is the only module importing both `mantis.train`
# and `mantis.eval` at top level, and splitting the builder out would either put a `mantis.run
# -> sibling` import in the one place the DAG forbids new edges or create a second place a
# `StepCoordinatorConfig` can be built — which is exactly the authority migration
# `tests/config/test_drawrate_arming_authority.py` and `test_coordinator_knobs_wiring.py` exist
# to forbid. The executable content is ~170 lines; the rest is the per-decision rationale (R64,
# MF-1/MF-2, S-4/Phase D/K-A/K-B) that made those authority defects findable.
"""mantis.run — the run composition root (design §a.4/§c.6; MUST-FIX 4: RELOCATE).

TOP-LEVEL module, ABOVE both `mantis.train` and `mantis.eval` — the ONE module that
imports both at module top level (no lazy-import loophole); nothing imports `mantis.run`,
so it is a source-only DAG node and the §2 "train -> all above except eval" ban stays
verbatim (census-tested: tests/test_run_composition.py::
test_no_train_module_imports_eval_even_lazily).

`python -m mantis.run <config.yaml>` is the entry point (CLAUDE.md `python -m mantis.*`
law). Smoke-grade until WP-SCHEMA-CLOSE (R-SELFPLAYCONFIG-SCHEMA: the pool still builds
only via the legacy hparams dict path elsewhere) — `compose_run` is therefore
injection-first: every collaborator (trainer/pool/buffer) is handed in, never built here,
so it stays fakes-testable. WP-UNFREEZE lives here: this root builds the continuous
actor-sync engine (`mantis.train.actor_sync.ActorSync`) UNCONDITIONALLY and wires the
actor-lag watchdog callables (`actor_ckpt_step` / learner step) into `build_run_safety`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Sequence

from mantis.config.resolve.actor_sync import resolve_actor_sync_cadence
from mantis.config.resolve.composition import require_run_config, revalidate_run_config
from mantis.config.resolve.coordinator import CoordinatorKnobsSpec, resolve_coordinator_knobs
from mantis.config.resolve.drain import DrainCapsSpec, resolve_drain_caps
from mantis.config.resolve.draw_rate import DrawRateAbortSpec, resolve_draw_rate_abort
from mantis.config.resolve.monitor import resolve_monitor_config
from mantis.config.resolve.run_length import resolve_max_train_steps
from mantis.config.schema import RunConfig
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.monitor.config import MonitorConfig
from mantis.train.actor_sync import ActorSync
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from mantis.train.loop import run_training_loop
from mantis.train.subsystems import build_run_safety

#: The 3 pipeline stages every run wires unconditionally; "eval_round" joins them iff an
#: eval pipeline is actually built (the caller DECLARES what it handed `heartbeat=` to).
_BASE_WIRED_SOURCES: tuple[str, ...] = ("train_step", "inference_dispatch", "selfplay_drain")


class RunHandles(NamedTuple):
    """What `compose_run` hands back — enough for a caller to inspect or drive further."""

    coordinator: Any
    run_safety: Any
    eval_pipeline: "Any | None"
    shutdown: ShutdownState


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


def _step_coordinator_config(
    *,
    stop_step: int,
    draw_rate_abort: "DrawRateAbortSpec | None",
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
    log_dir: "str | Path",
    checkpoint_dir: "str | Path",
    eval_enabled: bool = True,
    run_id: str = "run",
) -> RunHandles:
    """The run composition root (§c.6). Injection-first: every COLLABORATOR arrives via a
    kwarg, never built here (R-10) — but no parameter may carry a CONFIG FACT: the gate on
    the first line below is the ONE authority for what this root may be composed from, and
    the parameter list is pinned by a signature census so a re-add cannot be silent (WPAX
    S-1/S-2, MF-1)."""
    config = require_run_config(config, caller="compose_run")
    # RED-TEAM F-3: the gate above answers "is this the class?"; this answers "is this a
    # config the loader would accept?". `model_copy(update=…)` builds a genuine RunConfig
    # whose CROSS-FIELD validators never re-ran, and one such copy drove a 20-step run with
    # a single actor sync — run3's frozen actor — past the gate. Re-validating the dump
    # closes that route, `model_construct`, and post-gate mutation together. It stays a
    # SECOND statement because the gate must remain compose_run's first (pinned) and because
    # the two rules have different contracts: the gate is identity-preserving, this is not.
    config = revalidate_run_config(config, caller="compose_run")
    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)
    monitor_cfg = _resolve_monitor_cfg(config)

    wired_sources: list[str] = list(_BASE_WIRED_SOURCES)
    if eval_enabled:
        wired_sources.append("eval_round")

    # WP-UNFREEZE §4.3: the lag-watchdog callables are read LIVE at poll time, never at
    # build time — `actor_sync` is assigned immediately below, before anything can start
    # (this root owns both the assignment and `watchdog.start()`). DESIGN §4.3's
    # "ActorSync first" ordering is inverted here because the engine's LAW-18 sink IS
    # `run_safety.sink`, which only exists after this call.
    run_safety = build_run_safety(
        log_dir=log_dir, run_id=run_id, buffer=buffer,
        buffer_persist_path=checkpoint_dir / "replay_buffer.bin",
        wired_sources=wired_sources, monitor_cfg=monitor_cfg,
        actor_ckpt_step_fn=lambda: actor_sync.actor_ckpt_step(),
        learner_step_fn=lambda: int(trainer.step),
    )

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
    if eval_enabled:
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

    # ORDER PINNED (subsystems.py:213-215 contract): pool starts, THEN the watchdog.
    pool.start()
    pool_started = True
    run_safety.watchdog.start()

    shutdown = ShutdownState()
    coordinator = StepCoordinator(
        trainer=trainer, buffer=buffer, pretrained_buffer=None, recent_buffer=None,
        pool=pool, eval_pipeline=eval_pipeline, subsystems=SimpleNamespace(gpu_monitor=None),
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
        run_training_loop(trainer=trainer, shutdown_state=shutdown, eval_pipeline=eval_pipeline,
                          coordinator=coordinator, anchor_state=resolved_anchor,
                          sink=run_safety.sink)
    finally:
        coordinator.close_out(on_drained=_stop_pool_if_started(pool, pool_started=pool_started))

    return RunHandles(coordinator=coordinator, run_safety=run_safety, eval_pipeline=eval_pipeline,
                      shutdown=shutdown)


def _lazy_save_anchor(*args: Any, **kwargs: Any) -> None:
    from mantis.train.anchor import save_best_model_atomic

    save_best_model_atomic(*args, **kwargs)


def _lazy_guarded_load(model: Any, state_dict: Any) -> None:
    from mantis.train.anchor import _guarded_load_state_dict

    _guarded_load_state_dict(model, state_dict)


def main(argv: "Sequence[str] | None" = None) -> int:
    """`python -m mantis.run <config.yaml>` — smoke-grade until WP-SCHEMA-CLOSE (R-10):
    the pool/trainer/buffer build-out is NOT this WP's property (hparams.py:9-15 legacy
    dict path); this entry point loads + validates the config and reports readiness."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m mantis.run <config.yaml>", file=sys.stderr)
        return 2
    from mantis.config.loader import load_config
    from mantis.train.determinism import seed_everything

    cfg = load_config(argv[0])
    # R30a — the ONE determinism boot site: seed before any RNG-consuming object exists.
    # This entry point does not yet build one (smoke-grade, see module docstring); a future
    # WP that wires main() to launch a real run inherits the already-seeded state for free.
    seed_everything(cfg.seed)
    print(f"config OK: run_id={cfg.run_id} encoding={cfg.identity.encoding}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["RunHandles", "compose_run", "main"]
