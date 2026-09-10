"""`eval_enabled` is the CONFIG's fact, with no code-side default and no forcing route.

The parameter goes rather than merely losing its default: a required parameter is a forcing
route with the default removed, and "may never force False" is unrepresentable only when there
is no route to force anything through. Fakes: the eval side is stubbed on the True arm; the
subject — which branch the config selects — is driven through the real `build_run_safety`.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mantis.run as mantis_run
from mantis.config.schema import RunConfig
from mantis.run import compose_run

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS_DIR = _REPO / "configs"
_RUN_PY = _REPO / "src" / "mantis" / "run.py"
_TOOL_PY = _REPO / "tools" / "ci_gates" / "preflight_mint.py"

_DRIVE_STEPS = 3


class _Pool:
    search_kind = "gumbel"
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draw_rate = 0.05  # the third outcome share
    draws = 1
    sims_per_sec = 100.0
    batch_fill_pct = 0.9

    class _RunnerStats:
        mcts_mean_depth = 5.0
        mcts_mean_root_concentration = 0.1
        cluster_value_std_mean = 0.0
        cluster_policy_disagreement_mean = 0.0
        cluster_variance_sample_count = 0

    def __init__(self) -> None:
        self._games = 0
        self._started = False
        self.recent_move_histories: list = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        if not self._started:
            raise RuntimeError("cannot join thread before it is started")

    def check_producer_health(self) -> None:
        return None

    def pooled_draw_counts(self) -> tuple[int, int]:
        return (0, 0)

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return self._RunnerStats()

    def sync_inference_weights(self, state_dict) -> None:
        return None

    def update_checkpoint_step(self, step: int) -> None:
        return None


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.device = "cpu"

    def train_step_from_tensors(self, *args, **kwargs) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def train_step_from_graph_batch(self, **kwargs) -> dict[str, float]:
        return self.train_step_from_tensors()

    def inference_state_dict(self) -> dict:
        return {}

    def save_checkpoint(self, loss_info) -> None:
        return None


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
        config=config, trainer=_Trainer(), pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
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


def test_no_parameter_can_force_the_eval_posture_or_the_run_identity() -> None:
    """Neither `eval_enabled` nor `run_id` is a parameter of the composition root: a required
    parameter still lets the preflight child pass `False` while the run passes `True`, and a
    caller-supplied `run_id` splits the segment identity from the config identity."""
    parameters = list(inspect.signature(compose_run).parameters)
    for banned in ("eval_enabled", "run_id"):
        assert banned not in parameters, (
            f"{banned} is a CONFIG FACT and may not be a parameter of the composition root "
            f"(R120/R123, MF-1); got {parameters}"
        )


def test_the_key_is_required_with_no_code_side_default() -> None:
    """The key is required on `RunConfig`, with no code-side default. Killer:
    `eval_enabled: bool = True` — harmless-looking, but a config that forgets the key then
    declares nothing while the run evaluates."""
    assert "eval_enabled" in RunConfig.model_fields, (
        "the key is TOP-LEVEL (`schema/core.py`, after `seed`) because it is a "
        "root-composition fact spanning the eval and monitor wired-sources, not an "
        "eval-section tuning knob"
    )
    assert RunConfig.model_fields["eval_enabled"].is_required(), (
        "RunConfig.eval_enabled has a code-side default — R120's first clause"
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


@pytest.mark.parametrize("name", sorted(p.name for p in _CONFIGS_DIR.glob("*.yaml")))
def test_every_minted_config_declares_the_key_explicitly(name: str, smoke_run_config) -> None:
    """Every minted config declares the key explicitly. The axis is globbed off `configs/`,
    since an enumeration can both omit a new file and outlive a deleted one."""
    assert smoke_run_config(name).eval_enabled is True, (
        f"{name} must declare eval_enabled explicitly; today's effective posture is the "
        "code default True everywhere, so True is a zero-behaviour mint (§6)"
    )


def test_the_minted_axis_is_not_empty() -> None:
    """An axis of zero params would make the row above a green no-op."""
    assert sorted(p.name for p in _CONFIGS_DIR.glob("*.yaml")), "configs/ globbed to nothing"
