"""`eval_enabled` is the CONFIG's fact, with no code-side default and no forcing route.

The parameter goes rather than merely losing its default: a required parameter is a forcing
route with the default removed, and "may never force False" is unrepresentable only when there
is no route to force anything through. Fakes: the eval side is stubbed on the True arm; the
subject — which branch the config selects — is driven through the real `build_run_safety`.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import mantis.run as mantis_run
from mantis.run import compose_run
from _drivable import DrivablePoolStub, DrivableTrainerStub

_REPO = Path(__file__).resolve().parents[1]
_RUN_PY = _REPO / "src" / "mantis" / "run.py"
_TOOL_PY = _REPO / "tools" / "ci_gates" / "preflight_mint.py"

_DRIVE_STEPS = 3


def _drive(tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request, *,
           eval_enabled: bool):
    """Compose one run whose ONLY delta is the config's `eval_enabled` value, returning the
    `wired_sources` declaration `build_run_safety` received. The finalizer closes the real sink
    because the drive leaves a daemon watchdog thread holding it reachable."""
    declared: dict[str, list[str]] = {}
    real_build = mantis_run.build_run_safety

    def _recording_build(**kwargs):
        declared["wired_sources"] = list(kwargs.get("wired_sources", []))
        return real_build(**kwargs)

    monkeypatch.setattr(mantis_run, "build_run_safety", _recording_build)

    production_builder = mantis_run._step_coordinator_config

    def _no_terminal_eval(**kwargs):
        return dataclasses.replace(production_builder(**kwargs), terminal_eval_enabled=False)

    monkeypatch.setattr(mantis_run, "_step_coordinator_config", _no_terminal_eval)
    monkeypatch.setattr(mantis_run, "build_eval_pipeline", lambda **_kw: SimpleNamespace(
        run_evaluation=lambda *a, **k: {"kicked": False, "reason": None},
        poll_completed=lambda: None, drain_pending=lambda: None,
        apply_gate_decision=lambda *a, **k: None, stop=lambda: None,
    ))
    import mantis.train.anchor as anchor

    monkeypatch.setattr(anchor, "resolve_anchor", lambda **_kw: SimpleNamespace(
        best_model=None, best_model_step=None, best_model_path=None, representation="graph"))

    # `dev_example.yaml`, not the armed smoke: an armed `draw_rate_abort.min_step` above this
    # drive's `max_train_steps` is refused by the schema's reachability validator.
    config = smoke_run_config(
        "dev_example.yaml", eval_enabled=eval_enabled,
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS, "batch_size": 8},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1},
    )
    handles = compose_run(
        config=config, trainer=DrivableTrainerStub(), pool=DrivablePoolStub(game_per_read=True), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    request.addfinalizer(handles.run_safety.sink.close)
    return handles, declared["wired_sources"]


def test_the_config_key_alone_decides_whether_the_eval_pipeline_is_built(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """Flipping the key in the CONFIG flips the consumer — two composed runs, one delta.

    The `wired_sources` half must move with it: a run that builds the pipeline without
    DECLARING the stage gets a loud `heartbeat_source_unwired` instead of stall coverage.
    """
    on_handles, on_wired = _drive(tmp_path / "on", monkeypatch, smoke_run_config,
                                  mk_graph_buffer, request, eval_enabled=True)
    assert on_handles.eval_pipeline is not None, (
        "`eval_enabled: true` must build the pipeline — run5 mints True, and a promotion "
        "bar with eval off is unrepresentable as a decision (LAW-15/R120)"
    )
    assert "eval_round" in on_wired, f"…and DECLARE the stage watched; got {on_wired}"

    off_handles, off_wired = _drive(tmp_path / "off", monkeypatch, smoke_run_config,
                                    mk_graph_buffer, request, eval_enabled=False)
    assert off_handles.eval_pipeline is None, (
        "`eval_enabled: false` must build no pipeline; got "
        f"{off_handles.eval_pipeline!r}"
    )
    assert "eval_round" not in off_wired, (
        f"…and must not declare a stage nothing beats into; got {off_wired}"
    )


def test_no_cli_switch_on_either_caller_can_reach_the_eval_posture() -> None:
    """No CLI switch on either caller declares an eval-flavoured option — `--no-eval` would be
    a run input the CLI decides over a fact the minted config authors."""
    for path in (_RUN_PY, _TOOL_PY):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        declared = {arg.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        offenders = [option for option in declared if "eval" in option.lower()]
        assert not offenders, (
            f"{path.relative_to(_REPO)} declares {offenders}: no CLI switch may reach the "
            "eval posture (O-10's ban, R64)"
        )
