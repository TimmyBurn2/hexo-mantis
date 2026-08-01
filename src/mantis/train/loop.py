"""The launch loop — subsystem boot + save-then-exit wiring (WP10 §a.4/§c.8 IMPROVE).

`run_training_loop` is the injected-collaborator rewrite of the old `training/loop.py`
`run_training_loop` (which coupled directly to the InferenceServer / WorkerPool / eval).
Per §c.8 it takes an INJECTED `trainer` (TrainerLike) + optional `shutdown_state`, drives
the per-step loop (via an injected `coordinator` or `step_fn`), polls `ShutdownState`
between steps, and — observing `shutdown_save` even if already set at ENTRY — writes the
FINAL envelope-v2 checkpoint via `trainer.save_checkpoint` before returning (T-LC-04),
unless the coordinator's own clean-completion leg already wrote it (R137, the post-loop
guard `_clean_stop_already_saved`).

`resolve_anchor` is an INJECTED callable (default None), lazily bound to
`train.anchor.resolve_anchor` (Slice 3) ONLY inside the anchor seed/persist branch, which
fires only when an `eval_pipeline` is injected (absent in a WP10-only launch). There is NO
top-level `loop → anchor` import, so the DAG stays acyclic and the O-SMOKE Slice-2 gate is
reachable without Slice 3.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from mantis.train.emit import emit_via
from mantis.train.lifecycle.signals import ShutdownState, install_signal_handlers

_LOG = logging.getLogger(__name__)


def _clean_stop_already_saved(coordinator: Any) -> bool:
    """True iff the coordinator already wrote the run's FINAL checkpoint on its
    clean-completion leg (R137/CARD-CLEANSTOP-SAVE — the O2 arm of `coordinator/step.py`).

    ``None`` — the ``step_fn`` and bare-loop drives — answers False: there was no leg 3 to
    have fired, so leg 2 keeps its rescue write exactly as before.

    A coordinator that is PRESENT but publishes no ``clean_stop_saved`` is a WIRING BUG and
    RAISES, never a silent ``False``. A silent False re-opens the very window this guard
    exists to close — a signal landing INSIDE leg 3's own multi-second full-envelope
    ``torch.save`` — and it re-opens it invisibly: the run would write two final artefacts
    at one step and nothing would say so. They are two DISTINCT files, not one idempotent
    write, because the filename carries a content hash over a microsecond-resolution
    ``created_utc``, so there is no filename-idempotence to fall back on. This is
    ``close_out``'s posture on ``disarm_staleness`` verbatim (`coordinator/drain.py`): a
    duck-typed object missing the member is a wiring bug, and a wiring bug must not degrade
    into "no guard" without anybody noticing.
    """
    if coordinator is None:
        return False
    saved = getattr(coordinator, "clean_stop_saved", None)
    if saved is None:
        raise TypeError(
            f"run_training_loop: coordinator ({type(coordinator).__name__}) publishes no "
            "`clean_stop_saved` — the loop cannot tell whether the clean-completion leg "
            "already wrote the FINAL checkpoint, and a second `_final_save()` would write a "
            "duplicate FINAL artefact at the same step (R137/CARD-CLEANSTOP-SAVE)"
        )
    return bool(saved)


def run_training_loop(
    *,
    trainer: Any,
    shutdown_state: ShutdownState | None = None,
    resolve_anchor: Callable[..., Any] | None = None,
    eval_pipeline: Any = None,
    coordinator: Any = None,
    step_fn: Callable[[], Any] | None = None,
    max_steps: int | None = None,
    anchor_state: Any = None,
    sink: Any = None,
    loss_info: dict | None = None,
) -> ShutdownState:
    """Drive the training loop under the §c.8 injection contract; return the final state.

    - ``trainer`` (TrainerLike) is REQUIRED and INJECTED.
    - ``shutdown_state`` is used AS INJECTED when supplied; ONLY when it is ``None`` does the
      loop build a fresh ``ShutdownState`` AND install signal handlers (the self-construct
      branch — a caller that injects a state owns its own handler policy).
    - Between steps the loop polls ``shutdown_state``; on ``shutdown_save`` being set —
      INCLUDING already-set at ENTRY (a 0-step loop over an injected
      ``ShutdownState(running=False, shutdown_save=True)``) — it calls
      ``trainer.save_checkpoint(...)`` once before returning (the FINAL v2 save; T-LC-04) —
      EXCEPT when the injected coordinator's clean-completion leg has already written the
      run's final checkpoint, which ``_clean_stop_already_saved`` latches out (R137).
    - ``resolve_anchor`` is lazily bound inside the eval_pipeline seed/persist branch only.
    """
    owns_state = shutdown_state is None
    if owns_state:
        shutdown_state = ShutdownState()
        install_signal_handlers(shutdown_state)

    # Anchor seed/persist branch — only when an eval pipeline is injected (WP11 wires it);
    # absent in a WP10-only launch. resolve_anchor is bound lazily here, never at module top.
    if eval_pipeline is not None:
        if resolve_anchor is None:
            from mantis.train.anchor import resolve_anchor as resolve_anchor  # lazy (Slice 3)
        resolved = resolve_anchor(
            trainer=trainer, eval_pipeline=eval_pipeline, anchor_state=anchor_state, sink=sink)
        if anchor_state is None:
            anchor_state = resolved
        else:
            # PUBLISH onto the caller's object, do not rebind a local.
            #
            # The composition root hands ONE anchor object to `DeployTagHooks` and to
            # `StepCoordinator` before this loop starts. Rebinding `anchor_state` here
            # updated nothing either of them could see, so `best_model` stayed None
            # forever, `eval/pipeline.py`'s `run_gate = (best is not None) and …` was
            # permanently False, and no round could ever promote. That is the WP-SP /
            # WP11-A port parity gap (WPUF-2 chunk U-w): the actor never synced and
            # nothing was ever blessed.
            for field in ("best_model", "best_model_step", "best_model_path", "representation"):
                if hasattr(resolved, field):
                    setattr(anchor_state, field, getattr(resolved, field))

    saved = False

    def _final_save() -> None:
        nonlocal saved
        if saved:
            return
        emit_via(sink, {"event": "shutdown_save", "step": getattr(trainer, "step", None)})
        trainer.save_checkpoint(loss_info)
        saved = True

    # Observe shutdown_save even if already set at entry (0-step shutdown; T-LC-04).
    if shutdown_state.shutdown_save:
        _final_save()
        return shutdown_state

    steps = 0
    while shutdown_state.running:
        if max_steps is not None and steps >= max_steps:
            break
        if coordinator is not None:
            coordinator.step()
        elif step_fn is not None:
            step_fn()
        else:
            break  # nothing to drive — a bare loop just polls shutdown state
        steps += 1
        if shutdown_state.shutdown_save:
            break

    # The POST-LOOP arm only. The ENTRY arm above is deliberately unguarded: at entry the
    # coordinator has not run, so leg 3 cannot have fired and there is nothing to latch out.
    # Here it can have: a signal landing inside leg 3's own write leaves `shutdown_save` set
    # on a run that has ALREADY written its final artefact, and firing `_final_save()` on top
    # of it would write a duplicate FINAL checkpoint at the same step (R137).
    if shutdown_state.shutdown_save and not _clean_stop_already_saved(coordinator):
        _final_save()
    return shutdown_state
