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
from typing import Any, Callable, Mapping, Sequence, cast

from mantis.train.emit import emit_via

_LOG = logging.getLogger(__name__)


def _route_eval_result(coord: Any, result: Any, *, sync_inference: bool = True) -> Any:
    """Route completed eval-round result(s) through the coordinator's async eval-RESULT seam
    (`on_eval_round_complete` — THE sealbot-WR consumer, §c.4b), then apply any promotion
    decision (WP11-A `_apply_promotion`).

    Shapes handled EXPLICITLY (RED-TEAM F7 — the sealbot gate's only feed path must never go
    quiet the way F-10 did):
      * ``None``            — no pending round; the normal no-op.
      * a ``Mapping``       — ONE completed round (the pre-WP11-A teardown shape).
      * a list/tuple        — a BATCH of completed rounds (the N-2 handshake shape WP11-A's
                              drain may plausibly return): every Mapping element is routed,
                              one handler call per round.
      * anything else       — LOUD: an `eval_result_unroutable` event + an ERROR log. It is
                              deliberately not a raise: a raise here escapes into `close_out`
                              and skips `on_drained` (`pool.stop`) and the terminal eval
                              (RED-TEAM F10), which is a worse failure than a recorded drop.
    """
    handler = getattr(coord, "on_eval_round_complete", None)
    if result is None:
        return result
    if handler is None:
        _unroutable(coord, result, "coordinator has no on_eval_round_complete handler")
        return result
    if isinstance(result, Mapping):
        handler(cast("Mapping[str, Any]", result))
        _apply_promotion(coord, result, sync_inference=sync_inference)
        return result
    if isinstance(result, (list, tuple)):
        rounds = cast("Sequence[Any]", result)
        for item in rounds:
            if isinstance(item, Mapping):
                handler(cast("Mapping[str, Any]", item))
                _apply_promotion(coord, item, sync_inference=sync_inference)
            else:
                _unroutable(coord, item, "batch element is not a result mapping")
        return result
    _unroutable(coord, result, "unsupported eval-result shape")
    return result


def _apply_promotion(coord: Any, result: Any, *, sync_inference: bool) -> None:
    """WP11-A: apply a promoted round's gate decision through the injected pipeline's
    `apply_gate_decision` (mantis/eval/promote.py's ONE call site). A promoted result with
    NO promotion surface (no pipeline, or one missing the method) is recorded LOUD — the
    same posture as `_unroutable` — never silently dropped."""
    if not isinstance(result, Mapping) or not result.get("promoted"):
        return
    pipeline = getattr(coord, "eval_pipeline", None)
    apply_fn = getattr(pipeline, "apply_gate_decision", None)
    if apply_fn is None:
        step = result.get("step")
        _LOG.error("eval_promotion_unapplied step=%s reason=no_promotion_surface", step)
        emit_via(getattr(coord, "_sink", None), {
            "event": "eval_promotion_unapplied", "step": step, "reason": "no_promotion_surface",
        })
        return
    apply_fn(result, sync_inference=sync_inference)


def _unroutable(coord: Any, result: Any, reason: str) -> None:
    """A result the sealbot seam cannot consume is RECORDED, never dropped in silence."""
    _LOG.error("eval_result_unroutable reason=%s type=%s", reason, type(result).__name__)
    emit_via(getattr(coord, "_sink", None), {
        "event": "eval_result_unroutable",
        "reason": reason,
        "result_type": type(result).__name__,
        "step": getattr(coord, "_train_step", None),
    })


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
    # "pool still UP" contract (drain.py:73-74,105-134): a drained promotion may sync.
    return _route_eval_result(coord, drain(), sync_inference=True)


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
    # Terminal promotion: pool already stopped (step_coordinator.py:1705-1710 parity) —
    # never sync inference weights into a torn-down pool.
    return _route_eval_result(coord, pipeline.run_evaluation(
        coord.eval_model, coord._train_step, best,
        full_config=coord.full_config, best_model_step=best_step, ignore_stride=True,
    ), sync_inference=False)


def close_out(coord: Any, on_drained: "Callable[[], None] | None" = None) -> None:
    """The run epilogue (§D-LOOPFIX W1): (0) DISARM the heartbeat watchdog's staleness fire,
    (1) DRAIN the in-flight eval (pool still UP so a drained promotion can sync),
    (2) ``on_drained()`` (the caller passes ``pool.stop`` so the terminal eval runs on an
    UNLOADED GPU), (3) TERMINAL full-battery eval on the final ckpt.

    Step (0) is the FIRST action and it is load-bearing (O-27): the close-out waits below
    are legally up to 14400 s, an order of magnitude past the 1800 s staleness deadline, so
    a disarm that lands after `flush_pending_eval` turns every clean finish with a long
    terminal eval into a false-42 supervisor RELAUNCH STORM. Only staleness is disarmed —
    the persist-fatal fire and the heartbeat-file `seq` stay live through the whole epilogue.
    """
    watchdog = getattr(coord, "heartbeat_watchdog", None)
    if watchdog is not None:
        disarm = getattr(watchdog, "disarm_staleness", None)
        if disarm is None:
            # FAIL LOUD (RED-TEAM F8): silently skipping the disarm because a duck-typed
            # object lacks the method is the exact false-42 relaunch storm MUST-2 exists to
            # prevent. A wrong object here is a wiring bug, and a wiring bug must not
            # degrade into "no watchdog" without anybody noticing.
            raise TypeError(
                f"close_out: coord.heartbeat_watchdog ({type(watchdog).__name__}) has no "
                "disarm_staleness(); a close-out that cannot disarm staleness would "
                "false-fire 42 on every clean finish with a long terminal eval"
            )
        disarm()
    flush_pending_eval(coord)
    if on_drained is not None:
        on_drained()
    run_terminal_eval(coord)
