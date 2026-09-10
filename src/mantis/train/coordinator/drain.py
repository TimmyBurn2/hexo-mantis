"""Terminal-eval flush and close_out — the run-lifecycle epilogue, training already stopped.

Eval is reached ONLY through the injected `EvalPipelineLike`, and these are free functions
taking the coordinator instance so `drain.py` never imports `step.py` (acyclic). With no
pipeline injected both flush functions no-op.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from mantis.train.emit import emit_via

_LOG = logging.getLogger(__name__)


def _route_eval_result(coord: Any, result: Any) -> Any:
    """Route completed eval-round result(s) through `on_eval_round_complete`, then apply any
    promotion decision.

    A `Mapping` is ONE round, a list/tuple a BATCH, and any other shape is recorded loud rather
    than raised: a raise here escapes into `close_out` and skips `on_drained` (`pool.stop`) and
    the terminal eval, which is worse than a recorded drop.
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
    """Apply a promoted round's gate decision through the injected pipeline; a promoted result
    with NO promotion surface is recorded loud, never silently dropped."""
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
    """Drain a possibly-promoted final eval before teardown; no-op when no eval pipeline is
    injected."""
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


def run_terminal_eval(coord: Any, *, resumable_stop: bool = False) -> Any:
    """Terminal full-battery eval on the FINAL checkpoint (stride ignored). No-op when no eval
    pipeline is injected or `terminal_eval_enabled` is False."""
    pipeline = getattr(coord, "eval_pipeline", None)
    cfg = coord.config
    # Read as a plain attribute, never `getattr(cfg, …, True)`: a fallback here would be a
    # second default authority beside the schema field, and an absent key must raise.
    if pipeline is None or not cfg.terminal_eval_enabled:
        return None
    # A resumable stop is PASSED IN, never re-derived here: the disk guard stops a run by
    # SIGTERMing its own process and its abort rule is recorded after `close_out`, so no state
    # visible here separates an operator stop from an abort. The default is False.
    if resumable_stop:
        _LOG.info(
            "terminal_eval_skipped_on_signal_stop step=%s — this stop is an interruption, not "
            "a terminus; the terminal battery runs when the run ENDS",
            getattr(coord, "_train_step", None),
        )
        emit_via(getattr(coord, "_sink", None), {
            "event": "terminal_eval_skipped", "reason": "signal_stop",
            "step": getattr(coord, "_train_step", None),
        })
        return None
    best = getattr(coord.anchor_state, "best_model", None)
    best_step = getattr(coord.anchor_state, "best_model_step", None)
    _LOG.info("terminal_eval step=%s", getattr(coord, "_train_step", None))
    emit_via(getattr(coord, "_sink", None),
             {"event": "terminal_eval", "step": getattr(coord, "_train_step", None)})
    # The pool is already stopped here, and that is fine: a gate decision writes the deploy tag
    # only, so it is pool-independent on every route.
    result = _route_eval_result(coord, pipeline.run_evaluation(
        coord.eval_model, coord._train_step, best,
        full_config=coord.full_config, best_model_step=best_step, ignore_stride=True,
    ))
    _record_terminal_outcome(coord, result)
    return result


def _record_terminal_outcome(coord: Any, result: Any) -> None:
    """Latch the TERMINAL round's own reason on the coordinator.

    THE one writer, reachable only from `run_terminal_eval` — that is what keeps the mid-run
    /terminal split structural rather than conditional. The reason is read in ONE expression off
    the routed mapping and travels as the enum member's own `str` value, because the train
    package may not import the eval package.

    Raises:
        TypeError: the routed result is not a mapping, or the coordinator has no set-once
            writer; both are wiring bugs, and degrading either would report a broken run as
            rc 0.
    """
    if not isinstance(result, Mapping):
        raise TypeError(
            f"run_terminal_eval: the terminal round routed a {type(result).__name__}, "
            "which carries no `eval_broken_reason` — the terminal seam cannot read it, and "
            "a terminal outcome that goes unrecorded reports a broken run as rc 0"
        )
    if getattr(coord, "record_terminal_eval_reason", None) is None:
        raise TypeError(
            f"run_terminal_eval: coord ({type(coord).__name__}) has no "
            "record_terminal_eval_reason(); without it a broken terminal round cannot "
            "reach the process exit code and the run reports success (R133/LAW-15)"
        )
    # Written as the attribute call and not through a local: the one-writer census walks the AST
    # for a call to this name, and a bound local would hide a second writer from it.
    coord.record_terminal_eval_reason(result["eval_broken_reason"])


def close_out(
    coord: Any, on_drained: Callable[[], None] | None = None, *, resumable_stop: bool = False,
) -> None:
    """Run the epilogue: disarm the watchdog's staleness fire, drain the in-flight eval, call
    `on_drained` (the caller passes `pool.stop`), then run the terminal eval on an unloaded GPU.

    The disarm is FIRST and load-bearing: the close-out waits are legally up to 14400 s against
    a 1800 s staleness deadline, so a disarm landing later turns every clean finish with a long
    terminal eval into a false-42 relaunch storm. Only staleness is disarmed — the persist-fatal
    fire and the heartbeat `seq` stay live through the whole epilogue.

    Raises:
        TypeError: `coord.heartbeat_watchdog` carries no `disarm_staleness()`.
    """
    watchdog = getattr(coord, "heartbeat_watchdog", None)
    if watchdog is not None:
        disarm = getattr(watchdog, "disarm_staleness", None)
        if disarm is None:
            # Fail loud: a duck-typed object without the method is a wiring bug, and letting it
            # degrade into "no watchdog" is the false-42 relaunch storm this disarm prevents.
            raise TypeError(
                f"close_out: coord.heartbeat_watchdog ({type(watchdog).__name__}) has no "
                "disarm_staleness(); a close-out that cannot disarm staleness would "
                "false-fire 42 on every clean finish with a long terminal eval"
            )
        disarm()
    flush_pending_eval(coord)
    if on_drained is not None:
        on_drained()
    run_terminal_eval(coord, resumable_stop=resumable_stop)
