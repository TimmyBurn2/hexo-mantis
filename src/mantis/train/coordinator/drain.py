"""Terminal-eval flush + close_out (WP10 §a.4 split — `drain` slice).

The run-lifecycle epilogue: training has STOPPED. Eval is reached ONLY through the injected
`EvalPipelineLike` (no `train → eval` import; DAG-clean). These are free functions taking the
coordinator instance so `drain.py` never imports `step.py` (acyclic); `StepCoordinator` exposes
thin `flush_pending_eval` / `run_terminal_eval` / `close_out` methods that delegate here.

The promotion runtime is LIVE (WP11-A wired it): completed rounds route through
`_apply_promotion` into the pipeline's `apply_gate_decision`. Since WP-UNFREEZE (R49) a gate
decision moves ONLY the deploy tag (anchor + best_model.pt) — actor weights sync continuously
in `mantis.train.actor_sync` — so every route calls the SAME single-signature applier and the
pool's lifecycle state is irrelevant to a gate decision. With no pipeline injected
(`eval_pipeline is None`) both flush functions no-op.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence, cast

from mantis.train.emit import emit_via

_LOG = logging.getLogger(__name__)


def _route_eval_result(coord: Any, result: Any) -> Any:
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
        _apply_promotion(coord, result)
        return result
    if isinstance(result, (list, tuple)):
        rounds = cast("Sequence[Any]", result)
        for item in rounds:
            if isinstance(item, Mapping):
                handler(cast("Mapping[str, Any]", item))
                _apply_promotion(coord, item)
            else:
                _unroutable(coord, item, "batch element is not a result mapping")
        return result
    _unroutable(coord, result, "unsupported eval-result shape")
    return result


def _apply_promotion(coord: Any, result: Any) -> None:
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
    apply_fn(result)


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
    pipeline is injected. A drained promotion moves only the deploy tag (WP-UNFREEZE,
    R49), so this route and the terminal route call the SAME single-signature applier."""
    pipeline = getattr(coord, "eval_pipeline", None)
    if pipeline is None:
        return None
    drain = getattr(pipeline, "drain_pending", None)
    if drain is None:
        return None
    _LOG.info("flush_pending_eval step=%s", getattr(coord, "_train_step", None))
    emit_via(getattr(coord, "_sink", None),
             {"event": "flush_pending_eval", "step": getattr(coord, "_train_step", None)})
    return _route_eval_result(coord, drain())


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
    # Terminal promotion: the pool is already stopped here, and that is FINE — a gate
    # decision is pool-independent on every route (WP-UNFREEZE, R49): it writes the
    # deploy tag only, so the mid-run/terminal asymmetry the old sync flag encoded is gone.
    return _route_eval_result(coord, pipeline.run_evaluation(
        coord.eval_model, coord._train_step, best,
        full_config=coord.full_config, best_model_step=best_step, ignore_stride=True,
    ))


def close_out(coord: Any, on_drained: "Callable[[], None] | None" = None) -> None:
    """The run epilogue (§D-LOOPFIX W1): (0) DISARM the heartbeat watchdog's staleness fire,
    (1) DRAIN the in-flight eval, (2) ``on_drained()`` (the caller passes ``pool.stop`` so
    the terminal eval runs on an UNLOADED GPU), (3) TERMINAL full-battery eval on the final
    ckpt. The drain-before-stop order survives for drain-BOUNDING reasons alone (the flush
    joins the in-flight round under its budget); gate decisions themselves are
    pool-independent on every route (WP-UNFREEZE, R49).

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
