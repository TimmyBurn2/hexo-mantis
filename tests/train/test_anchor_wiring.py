"""The resolved anchor reaches the objects that read it (WPUF-2 chunk U-w, R55).

The port defect: `mantis.run` hands ONE anchor object to `PromotionHooks` and to
`StepCoordinator` before the training loop starts, then `run_training_loop` resolved the
real anchor into a **local that was never read again**. Nothing either holder could see
ever changed, so `best_model` stayed `None`, `eval/pipeline.py`'s
`run_gate = (best is not None) and …` was permanently False, no round could promote, and
`promote.py` returned early before reaching `sync_inference_weights`.

The consequence was larger than the freeze WP-UNFREEZE was written to remove: the actor
never synced AT ALL (not the ~39% of run3), and nothing was ever deploy-blessed either.

No existing test pinned this, which is exactly how it survived the WP-SP zero-behavior port
and WP11-A's wiring — see the R50 change-list re-verification in the dispatch log.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from mantis.train.loop import run_training_loop

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Trainer:
    step = 0
    model = None


def _run_once(anchor_state, resolved):
    """Drive one bounded loop iteration with an injected `resolve_anchor`."""
    return run_training_loop(
        trainer=_Trainer(),
        eval_pipeline=SimpleNamespace(),          # non-None: arms the anchor branch
        resolve_anchor=lambda **_kw: resolved,
        anchor_state=anchor_state,
        step_fn=lambda: None,
        max_steps=0,
        coordinator=None,
    )


def test_resolved_anchor_is_published_onto_the_callers_object():
    """The pin. Rebinding a local here is the defect; the caller must observe the result."""
    shared = SimpleNamespace(best_model=None, best_model_step=None)
    resolved = SimpleNamespace(
        best_model="THE-MODEL", best_model_step=1234,
        best_model_path=Path("/tmp/best_model.pt"), representation="graph",
    )

    _run_once(shared, resolved)

    assert shared.best_model == "THE-MODEL", (
        "the resolved anchor did not reach the caller's object — this is the U-w defect"
    )
    assert shared.best_model_step == 1234
    assert shared.best_model_path == Path("/tmp/best_model.pt")
    assert shared.representation == "graph"


def test_composition_root_actually_threads_the_anchor_into_the_loop():
    """`run.py` must PASS it — publication is useless if the loop never receives it.

    Asserted structurally rather than by booting a real run: `mantis.run.main` IS a real
    launcher since WPMAIN, but a full boot is an integration-tier cost this assertion does
    not need — the AST census below reads exactly the keyword this test is about, and the
    live end-to-end path is covered by the launcher's own boot oracle.
    """
    tree = ast.parse((REPO_ROOT / "src" / "mantis" / "run.py").read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None)) == "run_training_loop"
    ]
    assert calls, "run_training_loop is no longer called from the composition root"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "anchor_state" in kwargs, (
            "run.py calls run_training_loop without anchor_state — the resolved anchor "
            "cannot reach PromotionHooks or StepCoordinator (U-w defect reopened)"
        )
