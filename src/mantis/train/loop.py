"""The launch loop: subsystem boot and save-then-exit wiring over injected collaborators.

`resolve_anchor` is lazily bound inside the anchor branch only, so there is no top-level
`loop -> anchor` import and the DAG stays acyclic.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mantis.train.emit import emit_via
from mantis.train.lifecycle.signals import ShutdownState, install_signal_handlers

_LOG = logging.getLogger(__name__)


def _clean_stop_already_saved(coordinator: Any) -> bool:
    """Report whether the coordinator already wrote the run's FINAL checkpoint.

    ``None`` answers False: there was no clean-completion leg to have fired. A coordinator that
    is PRESENT but publishes no ``clean_stop_saved`` RAISES rather than answering a silent False,
    which would invisibly re-open the window this guard closes — the two final artefacts are
    DISTINCT files, since the filename carries a content hash over a microsecond timestamp.
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
    best_model_path: str | Path | None = None,
    expected_anchor_sha256: str | None = None,
) -> ShutdownState:
    """Drive the training loop over injected collaborators and return the final shutdown state.

    ``shutdown_state`` is used AS INJECTED when supplied; only when it is ``None`` does the loop
    build one AND install signal handlers, since a caller that injects a state owns its own
    handler policy. ``shutdown_save`` is observed even if already set at ENTRY, and the final
    save is skipped only when the coordinator's clean-completion leg already wrote it.
    """
    owns_state = shutdown_state is None
    if owns_state:
        shutdown_state = ShutdownState()
        install_signal_handlers(shutdown_state)

    # Bound lazily here, never at module top, so the import DAG stays acyclic.
    if eval_pipeline is not None:
        if resolve_anchor is None:
            from mantis.train.anchor import resolve_anchor as resolve_anchor  # lazy
        # `best_model_path` is FORWARDED, never defaulted here: a default falls back to a
        # CWD-relative anchor while the promotion hook writes the run's real one.
        resolved = resolve_anchor(
            trainer=trainer, eval_pipeline=eval_pipeline, anchor_state=anchor_state, sink=sink,
            best_model_path=best_model_path, expected_anchor_sha256=expected_anchor_sha256)
        if anchor_state is None:
            anchor_state = resolved
        else:
            # PUBLISH onto the caller's object rather than rebinding a local: the composition
            # root already handed this ONE anchor object to the hooks and the coordinator.
            for field in ("best_model", "best_model_step", "best_model_path", "representation"):
                if hasattr(resolved, field):
                    setattr(anchor_state, field, getattr(resolved, field))

    saved = False

    def _final_save() -> None:
        nonlocal saved
        if saved:
            return
        emit_via(sink, {"event": "shutdown_save", "step": getattr(trainer, "step", None)})
        checkpoint_path = trainer.save_checkpoint(loss_info)
        saved = True
        # The sidecar rides this leg too: a stop DURING a step never returns from `step()`, so
        # only this leg runs and a checkpoint without one resumes from an empty ring.
        persist = getattr(coordinator, "persist_resume_state", None)
        if persist is not None:
            persist(checkpoint_path)

    # Observe shutdown_save even if already set at entry, for a 0-step shutdown.
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

    # Guarded only here: at entry the coordinator has not run, so it cannot have saved. A signal
    # landing inside its own write would otherwise duplicate the FINAL checkpoint at one step.
    if shutdown_state.shutdown_save and not _clean_stop_already_saved(coordinator):
        _final_save()
    return shutdown_state
