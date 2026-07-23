"""Terminal-eval flush + close_out (WP10 §a.4 split — `drain` slice).

The run-lifecycle epilogue: training has STOPPED. Eval is reached ONLY through the injected
`EvalPipelineLike` (no `train → eval` import; DAG-clean). These are free functions taking the
coordinator instance so `drain.py` never imports `step.py` (acyclic); `StepCoordinator` exposes
thin `flush_pending_eval` / `run_terminal_eval` / `close_out` methods that delegate here.

The async eval-THREAD drain + promotion-stamping runtime (old `eval_drain.drain_pending_eval` /
`promote_anchor`) DEFERS→WP11 — in a WP10-only launch `eval_pipeline is None`, so both functions
no-op. When WP11 injects a concrete pipeline, `flush_pending_eval` joins the in-flight round and
`run_terminal_eval` runs the terminal full-battery eval on the final checkpoint.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from mantis.train.emit import emit_via

_LOG = logging.getLogger(__name__)


def flush_pending_eval(coord: Any) -> Any:
    """Drain a possibly-promoted final eval before teardown (D-012). No-op when no eval
    pipeline is injected. The pool is still UP here so a drained promotion can sync into
    self-play inference (WP11 wires the promotion sync)."""
    pipeline = getattr(coord, "eval_pipeline", None)
    if pipeline is None:
        return None
    drain = getattr(pipeline, "drain_pending", None)
    if drain is None:
        return None
    _LOG.info("flush_pending_eval step=%s", getattr(coord, "_train_step", None))
    emit_via(getattr(coord, "_sink", None),
             {"event": "flush_pending_eval", "step": getattr(coord, "_train_step", None)})
    return drain()


def run_terminal_eval(coord: Any) -> Any:
    """Terminal full-battery eval on the FINAL checkpoint (stride ignored). No-op when no eval
    pipeline is injected or `terminal_eval_enabled` is False."""
    pipeline = getattr(coord, "eval_pipeline", None)
    cfg = coord.config
    if pipeline is None or not getattr(cfg, "terminal_eval_enabled", True):
        return None
    best = getattr(coord.anchor_state, "best_model", None)
    best_step = getattr(coord.anchor_state, "best_model_step", None)
    _LOG.info("terminal_eval step=%s", getattr(coord, "_train_step", None))
    emit_via(getattr(coord, "_sink", None),
             {"event": "terminal_eval", "step": getattr(coord, "_train_step", None)})
    return pipeline.run_evaluation(
        coord.eval_model, coord._train_step, best,
        full_config=coord.full_config, best_model_step=best_step, ignore_stride=True,
    )


def close_out(coord: Any, on_drained: "Callable[[], None] | None" = None) -> None:
    """The run epilogue (§D-LOOPFIX W1): (1) DRAIN the in-flight eval (pool still UP so a
    drained promotion can sync), (2) ``on_drained()`` (the caller passes ``pool.stop`` so the
    terminal eval runs on an UNLOADED GPU), (3) TERMINAL full-battery eval on the final ckpt."""
    flush_pending_eval(coord)
    if on_drained is not None:
        on_drained()
    run_terminal_eval(coord)
