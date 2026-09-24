"""Actor sync runs in PRODUCTION posture (`eval_enabled=True`), unconditionally.

A structural pin can only ban the shapes it enumerates: an `ast.IfExp` ternary walked
through the `ast.If`-only pin, so the behavioral test below observes the consequence.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import mantis.run
from _drivable import DrivablePoolStub, DrivableTrainerStub, fake_run_safety, with_deltas

_STOP_STEP = 4


#: Captured at import so the patch below can delegate without re-entering itself.
_PRODUCTION_BUILDER = mantis.run._step_coordinator_config


#: The production builder bounded so no eval round runs; config-authored values pass through.
_bounded_config = with_deltas(_PRODUCTION_BUILDER, terminal_eval_enabled=False, eval_interval=1000,
                              log_interval=1, stop_step=_STOP_STEP)


def _install_harness(monkeypatch):
    """Replace the three collaborators this test is not about; the sync path stays real."""
    import mantis.train.anchor as _anchor

    monkeypatch.setattr(mantis.run, "build_run_safety", fake_run_safety)
    monkeypatch.setattr(mantis.run, "_step_coordinator_config", _bounded_config)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(
            best_model=None, best_model_step=None,
            best_model_path=None, representation="graph",
        ),
    )


def test_actor_syncs_with_eval_enabled_the_posture_a_real_run_uses(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
):
    """THE production-posture pin: sync observed by consequence rather than syntax."""
    pool, trainer = DrivablePoolStub(game_per_read=True), DrivableTrainerStub()
    _install_harness(monkeypatch)

    handles = mantis.run.compose_run(
        config=smoke_run_config(
            train={"actor_sync_cadence_steps": 1, "max_train_steps": _STOP_STEP,
                   "batch_size": 8},
            monitor={"actor_lag_threshold_steps": _STOP_STEP - 1},
            # The posture is a CONFIG fact; no parameter can carry it.
            eval_enabled=True),
        trainer=trainer, pool=pool, buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
    )

    assert handles.eval_pipeline is not None, (
        "harness precondition: eval_enabled=True must actually build the deploy side"
    )
    assert trainer.step >= 1, "harness precondition: at least one real training step ran"
    assert pool.sync_payloads, (
        "the actor never synced with eval_enabled=True. Sync must be UNCONDITIONAL "
        "(R49) — if it happens only when the deploy side is absent, that is run3's "
        "freeze in production and a healthy-looking suite"
    )
    assert pool.step_calls, "no actor checkpoint step was ever recorded"
    assert trainer.step - pool.step_calls[-1] <= 1, (
        f"actor_ckpt_step {pool.step_calls[-1]} must track learner_step {trainer.step}"
    )


def test_sync_volume_does_not_depend_on_whether_the_deploy_side_exists(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer
):
    """Both postures must sync. A difference between them IS the coupling R49 forbids."""
    results = {}
    for label, eval_enabled in (("no_eval", False), ("with_eval", True)):
        pool, trainer = DrivablePoolStub(game_per_read=True), DrivableTrainerStub()
        _install_harness(monkeypatch)
        mantis.run.compose_run(
            # The loop variable is a CONFIG delta, so one config is built per posture
            # inside the loop; both drive semantics are byte-preserved.
            config=smoke_run_config(
                train={"actor_sync_cadence_steps": 1, "max_train_steps": _STOP_STEP,
                       "batch_size": 8},
                monitor={"actor_lag_threshold_steps": _STOP_STEP - 1},
                eval_enabled=eval_enabled),
            trainer=trainer, pool=pool, buffer=mk_graph_buffer(n_records=32),
            log_dir=str(tmp_path / label), checkpoint_dir=str(tmp_path / label / "ckpt"),
        )
        results[label] = len(pool.sync_payloads)

    assert results["no_eval"] > 0 and results["with_eval"] > 0, results
    assert results["no_eval"] == results["with_eval"], (
        f"sync count differs by deploy-side presence: {results} — the two seams are "
        "coupled, which is exactly what R49 forbids"
    )


def _actor_sync_assignment() -> ast.Assign:
    src = inspect.getsource(mantis.run)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "actor_sync" for t in node.targets
        ):
            return node
    raise AssertionError("no `actor_sync = ...` assignment found in mantis.run")


def test_actor_sync_construction_is_conditional_in_no_form_at_all():
    """Bans conditional nodes of ANY kind, because the frozen pin enumerated and a
    ternary walked through it."""
    node = _actor_sync_assignment()
    conditional = [
        n for n in ast.walk(node.value)
        if isinstance(n, (ast.IfExp, ast.BoolOp))
    ]
    assert not conditional, (
        f"`actor_sync = ...` is built through a conditional expression "
        f"({[type(n).__name__ for n in conditional]}). Construction must be "
        f"unconditional (R49) in every syntactic form, not merely outside an `if` block."
    )
    assert isinstance(node.value, ast.Call), (
        f"expected a direct ActorSync(...) call, got {type(node.value).__name__}"
    )


def test_the_composition_root_still_documents_the_unconditionality_requirement():
    """Cheap guard on the comment that tells the next reader why this matters."""
    src = Path(mantis.run.__file__).read_text(encoding="utf-8")
    assert "UNCONDITIONALLY" in src
