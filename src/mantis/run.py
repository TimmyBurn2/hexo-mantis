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

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple, Sequence

from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.monitor.config import MonitorConfig
from mantis.train.actor_sync import ActorSync
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState
from mantis.train.loop import run_training_loop
from mantis.train.subsystems import build_run_safety

_LOG = logging.getLogger(__name__)

#: The 3 pipeline stages every run wires unconditionally; "eval_round" joins them iff an
#: eval pipeline is actually built (the caller DECLARES what it handed `heartbeat=` to).
_BASE_WIRED_SOURCES: tuple[str, ...] = ("train_step", "inference_dispatch", "selfplay_drain")

#: The fakes-path actor-sync cadence (R-10 justification, same as
#: `_default_step_coordinator_config`): a config object with no `.train` attribute (every
#: existing fakes test) gets cadence 1 — the zero-staleness, MOST-synced posture, never a
#: quietly frozen one. A real `RunConfig` missing the key never reaches this branch:
#: pydantic rejects it at load, naming the key.
_SMOKE_ACTOR_SYNC_CADENCE_STEPS: int = 1


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


def _resolve_monitor_cfg(config: Any) -> MonitorConfig:
    """The monitor-section twin of the `eval_cfg=getattr(config, "eval", None)` idiom two
    lines below this function's call site in `compose_run` (STOP CANDIDATE 5, DESIGN_P3.md
    §5.0). `config` may be a real `RunConfig` (`.monitor: MonitorSchemaConfig`, production)
    or a fakes-test `SimpleNamespace()` with no `.monitor` attribute at all (every existing
    `compose_run` test) — `getattr(..., None)` tolerates both, unlike a direct
    `config.monitor` read."""
    section = getattr(config, "monitor", None)
    if section is None:
        return MonitorConfig()
    from mantis.config.resolve.monitor import resolve_monitor_config
    return resolve_monitor_config(section)


def _resolve_actor_sync_cadence_steps(config: Any) -> int:
    """The train-section twin of `_resolve_monitor_cfg`: a real `RunConfig` resolves
    `train.actor_sync_cadence_steps` through its ONE resolver (K1); a fakes-test config
    with no `.train` attribute gets `_SMOKE_ACTOR_SYNC_CADENCE_STEPS`."""
    section = getattr(config, "train", None)
    if section is None:
        return _SMOKE_ACTOR_SYNC_CADENCE_STEPS
    from mantis.config.resolve.actor_sync import resolve_actor_sync_cadence
    return resolve_actor_sync_cadence(section)


def _default_step_coordinator_config() -> StepCoordinatorConfig:
    """Smoke-grade defaults (R-10: injection-first, pre-WP-SCHEMA-CLOSE) — no config-key
    reads here; the run5 mint threads real values through this seam once the schema
    extension lands. `stop_step=0` makes a single `step()` call terminal by construction,
    matching the composition root's own smoke-grade posture."""
    return StepCoordinatorConfig(
        eval_interval=1000, log_interval=1000, checkpoint_interval=0, composition_interval=0,
        value_probe_interval=0, min_buf_size=1, capacity=100_000, buffer_schedule=(),
        training_steps_per_game=1.0, max_train_burst=1, batch_size=8, augment=False,
        recency_weight=0.0, mixing_initial_w=0.0, mixing_min_w=0.0, mixing_decay_steps=1.0,
        soft_ew_threshold=0.0, soft_ew_min_pts=0, hard_gn_threshold=1e9, hard_gn_min_steps=3,
        instrumentation_enabled=False, stop_step=0, final_eval_drain_timeout_sec=900.0,
    )


def compose_run(
    *,
    config: Any,
    trainer: Any,
    pool: Any,
    buffer: Any,
    log_dir: "str | Path",
    checkpoint_dir: "str | Path",
    monitor_cfg: "MonitorConfig | None" = None,
    eval_enabled: bool = True,
    run_id: str = "run",
) -> RunHandles:
    """The run composition root (§c.6). Injection-first: every collaborator arrives via
    a kwarg, never built here (R-10)."""
    log_dir = Path(log_dir)
    checkpoint_dir = Path(checkpoint_dir)
    monitor_cfg = monitor_cfg if monitor_cfg is not None else _resolve_monitor_cfg(config)

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
    # its own 4 fields (config.py:176-180), never a second, independently-hardcoded set of
    # literals. The two used to duplicate config.py's own defaults (900.0/3.0/14400.0/
    # 14400.0) by coincidence; a future default change there would have silently
    # diverged the two (R1: duplicated default authority).
    step_coordinator_cfg = _default_step_coordinator_config()

    resolved_anchor = SimpleNamespace(best_model=None, best_model_step=None)
    eval_pipeline = None
    if eval_enabled:
        eval_pipeline = build_eval_pipeline(
            eval_cfg=getattr(config, "eval", None),
            coordinator_cfg_caps=DrainCaps(
                final_eval_drain_timeout_sec=step_coordinator_cfg.final_eval_drain_timeout_sec,
                eval_final_drain_safety_factor=step_coordinator_cfg.eval_final_drain_safety_factor,
                eval_final_drain_hard_cap_sec=step_coordinator_cfg.eval_final_drain_hard_cap_sec,
                terminal_eval_hard_cap_sec=step_coordinator_cfg.terminal_eval_hard_cap_sec,
            ),
            encoding=getattr(getattr(config, "identity", None), "encoding", "unknown"),
            run_id=run_id, spool_dir=log_dir / "eval_spool",
            ladder_state_path=log_dir / "eval_ladder_state.json",
            promotion=DeployTagHooks(
                anchor_state=resolved_anchor,
                best_model_path=checkpoint_dir / "best_model.pt", run_id=run_id,
                encoding=getattr(getattr(config, "identity", None), "encoding", "unknown"),
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
        config=step_coordinator_cfg, full_config=(config if isinstance(config, dict) else {}),
        train_cfg={}, mixing_cfg={}, run_id=run_id,
        sink=run_safety.sink, heartbeat=run_safety.heartbeat, monitor_cfg=monitor_cfg,
        heartbeat_watchdog=run_safety.watchdog, actor_sync=actor_sync,
    )

    # Drive the loop + epilogue. Defensive: a placeholder/fake collaborator (no real
    # `.arch`, no real checkpoint machinery) must never crash the COMPOSITION root itself
    # — a real production run's collaborators satisfy every downstream contract; a
    # fakes-testable smoke harness's do not, and this root's job is to WIRE, not to assert
    # the drive succeeded end to end.
    try:
        run_training_loop(trainer=trainer, shutdown_state=shutdown, eval_pipeline=eval_pipeline,
                          coordinator=coordinator, anchor_state=resolved_anchor,
                          sink=run_safety.sink)
    except Exception:  # noqa: BLE001 — see docstring note above
        _LOG.exception("run_training_loop_raised run_id=%s", run_id)
    try:
        coordinator.close_out(on_drained=_stop_pool_if_started(pool, pool_started=pool_started))
    except Exception:  # noqa: BLE001 — see docstring note above
        _LOG.exception("close_out_raised run_id=%s", run_id)

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
