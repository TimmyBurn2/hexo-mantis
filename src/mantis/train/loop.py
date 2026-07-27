"""The launch loop — subsystem boot + save-then-exit wiring (WP10 §a.4/§c.8 IMPROVE).

`run_training_loop` is the injected-collaborator rewrite of the old `training/loop.py`
`run_training_loop` (which coupled directly to the InferenceServer / WorkerPool / eval).
Per §c.8 it takes an INJECTED `trainer` (TrainerLike) + optional `shutdown_state`, drives
the per-step loop (via an injected `coordinator` or `step_fn`), polls `ShutdownState`
between steps, and — observing `shutdown_save` even if already set at ENTRY — writes the
FINAL envelope-v2 checkpoint via `trainer.save_checkpoint` before returning (T-LC-04).

`resolve_anchor` is an INJECTED callable (default None), lazily bound to
`train.anchor.resolve_anchor` (Slice 3) ONLY inside the anchor seed/persist branch, which
fires only when an `eval_pipeline` is injected (absent in a WP10-only launch). There is NO
top-level `loop → anchor` import, so the DAG stays acyclic and the O-SMOKE Slice-2 gate is
reachable without Slice 3.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from mantis.train.emit import emit_via
from mantis.train.lifecycle.signals import ShutdownState, install_signal_handlers

_LOG = logging.getLogger(__name__)


def run_training_loop(
    *,
    trainer: Any,
    shutdown_state: Optional[ShutdownState] = None,
    resolve_anchor: Optional[Callable[..., Any]] = None,
    eval_pipeline: Any = None,
    coordinator: Any = None,
    step_fn: Optional[Callable[[], Any]] = None,
    max_steps: Optional[int] = None,
    anchor_state: Any = None,
    sink: Any = None,
    loss_info: Optional[dict] = None,
) -> ShutdownState:
    """Drive the training loop under the §c.8 injection contract; return the final state.

    - ``trainer`` (TrainerLike) is REQUIRED and INJECTED.
    - ``shutdown_state`` is used AS INJECTED when supplied; ONLY when it is ``None`` does the
      loop build a fresh ``ShutdownState`` AND install signal handlers (the self-construct
      branch — a caller that injects a state owns its own handler policy).
    - Between steps the loop polls ``shutdown_state``; on ``shutdown_save`` being set —
      INCLUDING already-set at ENTRY (a 0-step loop over an injected
      ``ShutdownState(running=False, shutdown_save=True)``) — it calls
      ``trainer.save_checkpoint(...)`` once before returning (the FINAL v2 save; T-LC-04).
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

    if shutdown_state.shutdown_save:
        _final_save()
    return shutdown_state
