"""⊕ WPMAIN ORACLE — `eval_enabled` is the CONFIG's fact, not a parameter (R120, O-E1).

RED-at-import until IMPL lands the `RunConfig.eval_enabled` schema field: every drive below
builds its config through the ONE loader, so the key must exist before any of them runs.

What this file exists to stop, measured at `b482243`:

`eval_enabled` is a `compose_run` PARAMETER with a code-side default `True` (`run.py:172`),
and the preflight child hardcodes the literal `True` (`preflight_mint.py:917-922`). So the
dispatch's own R64 clause — "eval_enabled per the config's own value" — named a value that
did not exist, and R64's "the preflight may never force False" was enforced by a comment.
R120 promotes it: the code-side default dies, every minted config carries the key, and the
one composition root is its live consumer.

Why the PARAMETER has to go rather than merely lose its default (§1.5): a required parameter
is a forcing route with the default removed, not a closed one. "May never force False" is
only structurally unrepresentable when there is no route to force anything through.

Fakes, disclosed: the eval SIDE is faked on the `eval_enabled=True` arm — `build_eval_pipeline`
returns a drivable stand-in and the anchor resolver is stubbed, because `run_training_loop`
seeds the anchor from `trainer.model` and reads `.arch` off it. That is the same harness, for
the same stated reason, as `tests/test_run_composition.py`'s eval-side drives and
`tests/train/test_actor_sync_real_config.py::_drive`. The SUBJECT — which branch the config
selects — is not faked: `build_run_safety` is the real builder and the `wired_sources`
declaration read below is the one it actually received.
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
_RUN_PY = _REPO / "src" / "mantis" / "run.py"
_TOOL_PY = _REPO / "tools" / "ci_gates" / "preflight_mint.py"

_DRIVE_STEPS = 3


class _Pool:
    gumbel_mcts = True
    avg_game_length = 20.0
    x_winrate = 0.5
    o_winrate = 0.45
    draw_rate = 0.05  # F-816-2: the third outcome share.
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
    """One composed run whose ONLY delta is the config's `eval_enabled` value.

    Returns `(handles, wired_sources)` — the declaration `build_run_safety` actually
    received, which is what decides whether the eval stage is watched by the stall watchdog.

    N4 (dispatcher-ownable backlog): the completed drive leaves `run_safety.watchdog`'s
    daemon thread running (`close_out` never touches it either, `run.py:899-920` — LAW-16
    debt CARD-PROTOCOL-COMPLETE, bounded in production because both real callers exit the
    process right after `compose_run` returns). A live watchdog thread keeps the sink object
    reachable for the rest of this pytest session, so nothing ever garbage-collects its way
    to closing the fd. `request.addfinalizer` closes the REAL sink deterministically
    (idempotent, `sink.py:205-206`) instead of relying on that reachability."""
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

    config = smoke_run_config(
        "smoke_gnn.yaml", eval_enabled=eval_enabled,
        train={"actor_sync_cadence_steps": 1, "max_train_steps": _DRIVE_STEPS, "batch_size": 8},
        monitor={"actor_lag_threshold_steps": _DRIVE_STEPS - 1},
    )
    handles = compose_run(
        config=config, trainer=_Trainer(), pool=_Pool(), buffer=mk_graph_buffer(n_records=32),
        log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    request.addfinalizer(handles.run_safety.sink.close)
    return handles, declared["wired_sources"]


# ══ O-E1 — the mutation oracle ════════════════════════════════════════════════════════
def test_the_config_key_alone_decides_whether_the_eval_pipeline_is_built(
    tmp_path, monkeypatch, smoke_run_config, mk_graph_buffer, request
) -> None:
    """O-E1, the liveness half (LAW-08/LAW-07): flip the key in the CONFIG, observe the
    consumer. Two composed runs, one delta.

    MUTATION THAT REDS IT: read anything but `config.eval_enabled` in the two branches
    (`run.py:194` and `:256`) — a surviving literal `True`, or a value carried in from a
    caller. The `wired_sources` half matters on its own: a run that builds the pipeline but
    fails to DECLARE the stage gets a loud `heartbeat_source_unwired` instead of stall
    coverage, so the two must move together."""
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
    """O-E1, the structural half — R64's "may never force False", made unrepresentable
    rather than asserted; and R123's same doctrine applied to `run_id`.

    MUTATION THAT REDS IT: re-add either parameter, with or without a default. A required
    parameter still lets the preflight child pass `False` while run5 passes `True` — the
    posture divergence "one composition authority" is supposed to close — and a
    caller-supplied `run_id != config.run_id` splits the JSONL segment identity from the
    config identity, which is the F-B1 class `run_boot_identity` exists to kill.

    The frozen 6-tuple census (`test_run_strict_composition.py:423`) is the R50-sanctioned
    flip site for the exact tuple; this is the narrower, independently-held claim that these
    two names in particular never come back."""
    parameters = list(inspect.signature(compose_run).parameters)
    for banned in ("eval_enabled", "run_id"):
        assert banned not in parameters, (
            f"{banned} is a CONFIG FACT and may not be a parameter of the composition root "
            f"(R120/R123, MF-1); got {parameters}"
        )


def test_the_key_is_required_with_no_code_side_default() -> None:
    """O-E1, the R120 clause "the code-side default `True` at the composition root dies".

    MUTATION THAT REDS IT: `eval_enabled: bool = True` on `RunConfig`. That reads harmless —
    True is today's effective posture — but it moves the authority from the minted config
    back into the code, and a config that forgets the key then declares nothing while the
    run evaluates. R1: a default lives only in a schema field, and this field has none."""
    assert "eval_enabled" in RunConfig.model_fields, (
        "the key is TOP-LEVEL (`schema/core.py`, after `seed`) because it is a "
        "root-composition fact spanning the eval and monitor wired-sources, not an "
        "eval-section tuning knob"
    )
    assert RunConfig.model_fields["eval_enabled"].is_required(), (
        "RunConfig.eval_enabled has a code-side default — R120's first clause"
    )


def test_no_cli_switch_on_either_caller_can_reach_the_eval_posture() -> None:
    """O-E1's no-route census — O-10's surviving half at its new scope.

    O-10 pinned `eval_enabled=True` as an unconditional LITERAL in the tool and banned any
    eval-flavoured CLI option. The literal half is superseded by "no route at all" (the
    child passes nothing, O-A4); the CLI ban is the half that must SURVIVE, and it now
    covers the launcher too — which never had it.

    MUTATION THAT REDS IT: `--no-eval` on either parser. It would be a run input the CLI
    decides, over a fact the minted config authors (R1), and it would re-open exactly the
    escape R64 bans."""
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


@pytest.mark.parametrize("name", ["dev_example.yaml", "run5.yaml", "shakedown_20260807.yaml",
                                  "smoke_gnn.yaml", "smoke_preflight_armed.yaml",
                                  "smoke_radius_curriculum.yaml", "sustained_kcluster.yaml"])
def test_every_minted_config_declares_the_key_explicitly(name: str, smoke_run_config) -> None:
    """O-E1's R1-completeness arm — every minted config carries the key EXPLICITLY.

    The seven are enumerated here rather than read from `tests/conftest.py:52-53`'s
    `MINTED_CONFIGS`, which lists five: it omits `smoke_preflight_armed.yaml`. That gap is
    pre-existing and is recorded, not fixed here (F-12) — but an oracle that inherited it
    would leave the one config the preflight actually boots uncovered.
    `shakedown_20260807.yaml` joins at F-P2B (R259): the armed-abort manifest's terminal-eval
    residual leans on every committed config minting the key True, production configs first.

    MUTATION THAT REDS IT: re-mint five of six. The schema makes that a load-time failure,
    which is the point: this arm is what turns "the key is required" into "and every shipped
    config has it", including run5, whose value R120 fixes at True."""
    assert smoke_run_config(name).eval_enabled is True, (
        f"{name} must declare eval_enabled explicitly; today's effective posture is the "
        "code default True everywhere, so True is a zero-behaviour mint (§6)"
    )
