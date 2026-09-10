"""The eval round must not be SKIPPED by a multi-step burst.

`_maybe_kick_eval` tests `self._train_step % cfg.eval_interval != 0`. Called ONCE after a whole
burst, a `max_train_burst > 1` run steps OVER the exact multiple (interval 5, burst 3, step
4 -> 7) and the round is not delayed but LOST, because `_eval_round_last_step` is keyed on the
round index and nothing retries it. Calling INSIDE the burst tests the boundary per training
step, so every exact multiple is hit.
"""
from __future__ import annotations

import ast
from pathlib import Path

_STEP_PY = (Path(__file__).resolve().parents[2]
            / "src" / "mantis" / "train" / "coordinator" / "step.py")


def _step_method() -> ast.FunctionDef:
    tree = ast.parse(_STEP_PY.read_text(encoding="utf-8"))
    coordinator = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.ClassDef) and n.name == "StepCoordinator")
    return next(n for n in coordinator.body
                if isinstance(n, ast.FunctionDef) and n.name == "step")


def _burst_loop(fn: ast.FunctionDef) -> ast.For:
    """The training-burst `for` loop — the one whose body calls `_run_log_interval`."""
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                     and c.func.attr == "_run_log_interval" for c in ast.walk(n))]
    assert len(loops) == 1, (
        f"expected exactly one training-burst loop in step(); found {len(loops)}. The "
        "instrument below identifies the burst by that call, so a second one makes it "
        "ambiguous rather than wrong — fix the instrument, do not relax it."
    )
    return loops[0]


def test_the_eval_kick_is_called_inside_the_burst_loop() -> None:
    """Structural, because at `max_train_burst == 1` the old placement and the new one are the
    same program: the bug exists only when a burst spans the multiple, and the fix is exactly
    which side of the loop the call sits on.

    MUTATION THAT REDS IT: move `self._maybe_kick_eval(cfg)` back below the loop.
    """
    fn = _step_method()
    loop = _burst_loop(fn)

    def _kick_calls(node: ast.AST) -> list[ast.Call]:
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_maybe_kick_eval"]

    assert _kick_calls(loop), (
        "`_maybe_kick_eval` is not called inside the training-burst loop, so a burst that "
        "steps over the exact eval_interval multiple silently loses that round"
    )
    assert len(_kick_calls(fn)) == 1, (
        "`_maybe_kick_eval` must be called exactly ONCE in step() — a second call site "
        "double-kicks or reintroduces the post-burst placement alongside the fixed one; "
        f"found {len(_kick_calls(fn))}"
    )


def test_the_kick_outcomes_are_or_folded_across_the_burst() -> None:
    """Per-step kicking makes the two outcomes burst accumulators, not a single return: without
    the fold, an early-step kick is overwritten by a later step's `(False, False)` and a round
    that DID fire is reported as not fired."""
    fn = _step_method()
    src = ast.unparse(_burst_loop(fn))
    assert "eval_kicked_off = eval_kicked_off or" in src, (
        "eval_kicked_off is rebound rather than OR-folded across the burst"
    )
    assert "eval_skipped_busy = eval_skipped_busy or" in src

    outer = ast.unparse(fn)
    assert "eval_kicked_off = False" in outer, (
        "the accumulator is never initialised before the loop"
    )
    assert "eval_skipped_busy = False" in outer


def test_the_kick_still_guards_on_the_interval_modulo() -> None:
    """The fix must not become 'kick every step'.

    Moving the call inside the loop without keeping `self._train_step % cfg.eval_interval != 0`
    would burn a round index per step and turn the eval cadence into 'continuous'. The modulo is
    what makes per-step TESTING correct rather than per-step firing.
    """
    tree = ast.parse(_STEP_PY.read_text(encoding="utf-8"))
    kick = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_maybe_kick_eval")
    src = ast.unparse(kick)
    assert "self._train_step % cfg.eval_interval != 0" in src, (
        "the interval modulo guard is gone — the kick would fire every training step"
    )
    assert "self._eval_round_last_step" in src, (
        "the once-per-round-index latch is gone — one boundary could kick twice"
    )
