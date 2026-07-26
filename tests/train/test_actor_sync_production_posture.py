"""Actor sync runs in PRODUCTION posture (`eval_enabled=True`), unconditionally.

RED-TEAM F-1. The frozen §2.4 pin sweeps only `ast.If`, so constructing the engine with a
*conditional expression* — `ActorSync(...) if not eval_enabled else None`, an `ast.IfExp` —
evaded it. Worse, every behavioral sync-through-`compose_run` test ran
`eval_enabled=False`, so **no test exercised the posture a real run uses**. RED-TEAM applied
that two-line regression to a repo copy and **all 106 guard tests passed** while a driven
`compose_run(eval_enabled=True)` produced zero weight pushes ever and a lag reading frozen
at 0 — run3's silent freeze, resurrected, with the exit-45 gate blinded.

The lesson, and why the behavioral test below is the real fix: **a structural pin can only
ban the shapes someone thought to enumerate.** `ast.If` was enumerated; `ast.IfExp` was not.
A behavioral assertion in production posture cannot be evaded by changing the shape of the
construction, because it observes the consequence rather than the syntax.

The structural test here is kept as the cheap early-warning half, and is deliberately
written to reject conditionality in ANY form rather than to enumerate node types.

NOT frozen: written after ORACLE-WRITE in response to a RED-TEAM finding.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mantis.run
from mantis.train.coordinator.config import StepCoordinatorConfig

_STOP_STEP = 4


class _RunnerStats:
    mcts_mean_depth = 5.0
    mcts_mean_root_concentration = 0.1
    cluster_value_std_mean = 0.0
    cluster_policy_disagreement_mean = 0.0
    cluster_variance_sample_count = 0


class _SyncRecordingPool:
    def __init__(self) -> None:
        self._games = 0
        self.gumbel_mcts = True
        self.avg_game_length = 20.0
        self.x_winrate = 0.5
        self.o_winrate = 0.45
        self.draws = 1
        self.sims_per_sec = 100.0
        self.batch_fill_pct = 0.9
        self.recent_move_histories: list = []
        self.sync_payloads: list = []
        self.step_calls: list[int] = []

    @property
    def games_completed(self) -> int:
        self._games += 1
        return self._games

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def check_producer_health(self) -> None: ...
    def per_worker_draw_rates(self) -> dict[int, float]:
        return {}

    def current_stride5_p90(self) -> int:
        return 1

    def runner_stats(self) -> Any:
        return _RunnerStats()

    def sync_inference_weights(self, state_dict) -> None:
        self.sync_payloads.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        self.step_calls.append(int(step))


class _Trainer:
    def __init__(self) -> None:
        self.step = 0
        self.model = object()
        self.inference_sd = {"w": "SENTINEL"}

    def train_step(self, buffer, augment=False, recent_buffer=None) -> dict[str, float]:
        self.step += 1
        return {"loss": 1.0, "policy_loss": 0.6, "value_loss": 0.4, "grad_norm": 0.1,
                "policy_entropy": 2.0, "value_accuracy": 0.5, "lr": 1e-3,
                "opp_reply_loss": 0.0, "loss_total": 1.0}

    def inference_state_dict(self) -> dict:
        return self.inference_sd

    def save_checkpoint(self, loss_info) -> None: ...


class _Buffer:
    size = 1000
    capacity = 100_000

    def resize(self, n: int) -> None: ...
    def save_to_path(self, p) -> None: ...


def _bounded_config() -> StepCoordinatorConfig:
    # The deploy side must be CONSTRUCTED — that is the posture under test — but must not
    # RUN a round: `eval_interval` beyond `stop_step` suppresses the periodic kick and
    # `terminal_eval_enabled=False` suppresses the close-out one. Executing a round would
    # demand a snapshot-able model with a real `.arch`, which is a different test's
    # subject. What is asserted here is that sync happens while the deploy machinery
    # exists, not anything about round execution.
    return StepCoordinatorConfig(
        terminal_eval_enabled=False,
        eval_interval=1000, log_interval=1, checkpoint_interval=0, composition_interval=0,
        value_probe_interval=0, min_buf_size=1, capacity=100_000, buffer_schedule=(),
        training_steps_per_game=1.0, max_train_burst=1, batch_size=8, augment=False,
        recency_weight=0.0, mixing_initial_w=0.0, mixing_min_w=0.0, mixing_decay_steps=1.0,
        soft_ew_threshold=0.0, soft_ew_min_pts=0, hard_gn_threshold=1e9, hard_gn_min_steps=3,
        instrumentation_enabled=False, stop_step=_STOP_STEP,
        final_eval_drain_timeout_sec=900.0,
    )


def _fake_run_safety(**_kwargs):
    return SimpleNamespace(
        sink=SimpleNamespace(emit=lambda e: None),
        registry=SimpleNamespace(beat=lambda s: None),
        watchdog=SimpleNamespace(start=lambda: None, disarm_staleness=lambda: None),
        heartbeat=lambda s: None,
    )


def _install_harness(monkeypatch):
    """Replace the three collaborators this test is not about.

    With `eval_enabled=True` the composition root exercises more of the production path
    than any previous test did, so it reaches real machinery that needs a real
    `torch.nn.Module` with a declared `.arch`: `resolve_anchor` builds a net from
    `trainer.arch`. Building one is a different test's subject (`test_anchor_wiring.py`
    owns anchor publication), so it is stubbed here. The SYNC path is left entirely real —
    stubbing it would defeat the point.
    """
    import mantis.train.anchor as _anchor

    monkeypatch.setattr(mantis.run, "build_run_safety", _fake_run_safety)
    monkeypatch.setattr(mantis.run, "_default_step_coordinator_config", _bounded_config)
    monkeypatch.setattr(
        _anchor, "resolve_anchor",
        lambda **_kw: SimpleNamespace(
            best_model=None, best_model_step=None,
            best_model_path=None, representation="grid",
        ),
    )


def test_actor_syncs_with_eval_enabled_the_posture_a_real_run_uses(tmp_path, monkeypatch):
    """THE F-1 pin. Production posture, observed by consequence rather than syntax.

    `eval_enabled=True` is what run5 launches with. Before this test, every behavioral
    sync assertion ran with the deploy side absent, so a regression that disabled sync
    exactly when the deploy side EXISTS was invisible to the entire suite.
    """
    pool, trainer = _SyncRecordingPool(), _Trainer()
    _install_harness(monkeypatch)

    handles = mantis.run.compose_run(
        config=SimpleNamespace(), trainer=trainer, pool=pool, buffer=_Buffer(),
        log_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
        eval_enabled=True,
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


def test_sync_volume_does_not_depend_on_whether_the_deploy_side_exists(tmp_path, monkeypatch):
    """Both postures must sync. A difference between them IS the coupling R49 forbids."""
    results = {}
    for label, eval_enabled in (("no_eval", False), ("with_eval", True)):
        pool, trainer = _SyncRecordingPool(), _Trainer()
        _install_harness(monkeypatch)
        mantis.run.compose_run(
            config=SimpleNamespace(), trainer=trainer, pool=pool, buffer=_Buffer(),
            log_dir=str(tmp_path / label), checkpoint_dir=str(tmp_path / label / "ckpt"),
            eval_enabled=eval_enabled,
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
    """Complements the frozen `ast.If`-only pin, which a ternary walked straight through.

    Written as "the assigned value contains NO conditional node of any kind" rather than
    as a list of banned node types, because the frozen pin's failure was precisely that it
    enumerated. `ast.IfExp` was the tenth resurrection route; the eleventh would be
    whatever else an enumeration forgets.
    """
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
