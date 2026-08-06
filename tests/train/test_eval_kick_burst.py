"""Item 7 (eval-kick half) — the eval round must not be SKIPPED by a multi-step burst.

THE DEFECT. `_maybe_kick_eval` tests `self._train_step % cfg.eval_interval != 0`, and it was
called ONCE after the whole training burst. With `max_train_burst > 1` a burst steps over the
exact multiple — interval 5, burst 3, `_train_step` going 4 → 7 — and the modulo is never
satisfied on the step the coordinator happens to observe. The round is not delayed, it is
LOST: `_eval_round_last_step` is keyed on the round index, so nothing retries it. A long run
could go many intervals without an eval while the config said otherwise, and the only visible
symptom is an eval cadence quietly slower than the one that was minted.

Moving the call INSIDE the burst tests the boundary per training step, so every exact
multiple is hit.

SCOPE. Item 7's other half — decoupling `monitor_gates` and the four other gated families
from `train.log_interval` — is NOT implemented and is NOT pinned here. It conflicts with
R210, which authorized decoupling `iteration_complete` ONLY and ruled in terms that
`training_step alerting stays gated`. See ADJ-D12 in the adjudication queue.
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
    """Structural, because the defect is invisible behaviourally at `max_train_burst == 1`.

    At burst 1 the old placement and the new one are the same program — every step is a
    boundary candidate — so a behavioural test at the default burst passes either way. The
    bug only exists when a burst spans the multiple, and the fix is exactly "which side of
    the loop the call sits on". That is what is asserted.

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
    """Per-step kicking makes the two outcomes burst accumulators, not a single return.

    Without the fold, a kick on an early step of the burst would be overwritten by a
    later step's `(False, False)` and the `eval_kicked_off` the outcome reports would be
    wrong — a round that DID fire, reported as not fired.
    """
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

    Mechanism: moving the call inside the loop without keeping
    `self._train_step % cfg.eval_interval != 0` would kick an eval round on EVERY training
    step — which the busy-ack would mostly reject, but would also burn a round index per
    step and turn the eval cadence into 'continuous'. The modulo is what makes per-step
    testing correct rather than per-step firing.
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
