"""⊕ WPUF Phase U ORACLE — O-U2 (both directions of actor⊥deploy independence) + O-U6
(the eval deploy-matched rung reads the DeployTag, never the actor's live weights).
DESIGN_U §8.

RED-at-import until IMPL lands `DeployTagHooks` (the renamed, `promotion_target`-less
`PromotionHooks`, DESIGN §1.1) and `mantis.train.actor_sync.ActorSync`.

Per DESIGN §0/GAPS-7 the gate-PASS direction cannot be proven on production wiring at
HEAD (no production path yields `promoted=True`), so it is driven exactly the way the
existing routing suites drive it: fabricated results through `drain._route_eval_result`
into a REAL `apply_gate_decision`. All drives are direct calls — zero threads, zero
joins, zero sleeps.

>300 justify (R8): one seam family (gate↔actor independence) whose four oracles share
one spy/hooks/coordinator harness; splitting harness from assertions would duplicate it.
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mantis.config.loader import load_config
from mantis.config.resolve.drain import resolve_drain_caps
from mantis.eval.promote import DeployTagHooks, apply_gate_decision  # RED-at-import anchor
from mantis.monitor.config import MonitorConfig
from mantis.run import _step_coordinator_config
from mantis.train.actor_sync import ActorSync
from mantis.train.coordinator import drain
from mantis.train.coordinator.config import StepCoordinatorConfig
from mantis.train.coordinator.step import StepCoordinator
from mantis.train.lifecycle.signals import ShutdownState

#: WPMINT Phase K-A stage 0: the four drain caps are `monitor.drain.*` (R93/DR-11) — read
#: from a MINTED config, never restated here.
_DRAIN_CAPS = resolve_drain_caps(
    load_config(Path(__file__).resolve().parents[2] / "configs" / "dev_example.yaml").monitor)


# ── shared spies ──────────────────────────────────────────────────────────────────────
class _CallSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> None:
        self.calls.append((args, dict(kwargs)))


class _ActorTargetSpy:
    """ActorSyncTarget-shaped; held by the HARNESS, never by any deploy-side object."""

    def __init__(self) -> None:
        self.sync_payloads: list = []
        self.step_calls: list[int] = []

    def sync_inference_weights(self, state_dict) -> None:
        self.sync_payloads.append(state_dict)

    def update_checkpoint_step(self, step: int) -> None:
        self.step_calls.append(int(step))


class _DeployOnlyPipeline:
    """Delegates to the REAL single-signature applier. Deliberately accepts NO
    `sync_inference` keyword: a drain that still threads one fails here (R49)."""

    def __init__(self, hooks: Any) -> None:
        self._hooks = hooks
        self.apply_calls = 0

    def apply_gate_decision(self, result):
        self.apply_calls += 1
        return apply_gate_decision(self._hooks, result)


def _hooks(tmp_path, anchor) -> tuple[Any, _CallSpy, _CallSpy]:
    save_anchor = _CallSpy()
    guarded_load = _CallSpy()
    hooks = DeployTagHooks(
        anchor_state=anchor, best_model_path=tmp_path / "best_model.pt",
        run_id="oracle_u2", encoding="gnn_axis_v1",
        save_anchor=save_anchor, guarded_load=guarded_load,
    )
    return hooks, save_anchor, guarded_load


def _routing_coord(pipeline) -> SimpleNamespace:
    return SimpleNamespace(
        on_eval_round_complete=lambda result: None, eval_pipeline=pipeline,
        _sink=None, _train_step=0,
    )


# ── O-U2: both directions ─────────────────────────────────────────────────────────────
def test_gate_pass_advances_deploy_tag_and_never_touches_actor(tmp_path) -> None:
    """Gate PASS moves ONLY the deploy tag: anchor advanced + save_anchor once, while an
    actor-target spy (held elsewhere in the harness) records ZERO calls in the window."""
    anchor = SimpleNamespace(best_model=object(), best_model_step=None)
    hooks, save_anchor, guarded_load = _hooks(tmp_path, anchor)
    actor_target = _ActorTargetSpy()  # in scope, reachable — but never wired to the gate
    pipeline = _DeployOnlyPipeline(hooks)
    coord = _routing_coord(pipeline)

    result = {"step": 7, "promoted": True, "eval_broken": False}
    drain._route_eval_result(coord, result)

    assert pipeline.apply_calls == 1, "the routed promotion must reach apply_gate_decision"
    assert anchor.best_model_step == 7, "a gate pass must advance the DeployTag step"
    assert len(save_anchor.calls) == 1, "a gate pass must persist the anchor exactly once"
    assert len(guarded_load.calls) == 1, (
        "the evaluated snapshot bytes must be loaded into the anchor (F-12/LAW-12)"
    )
    assert actor_target.sync_payloads == [] and actor_target.step_calls == [], (
        "a gate decision must never reach an actor surface (R49: the deploy side has no "
        "attribute through which to touch a pool)"
    )


def test_gate_fail_freezes_deploy_tag_while_sync_continues(tmp_path) -> None:
    """Gate FAIL forever: routed `promoted=False` results leave the DeployTag byte-frozen
    while `ActorSync.maybe_sync`, driven in the same harness, keeps pushing weights."""
    anchor = SimpleNamespace(best_model=object(), best_model_step=None)
    hooks, save_anchor, _ = _hooks(tmp_path, anchor)
    pipeline = _DeployOnlyPipeline(hooks)
    coord = _routing_coord(pipeline)

    target = _ActorTargetSpy()
    learner = SimpleNamespace(step=0)
    engine = ActorSync(
        target=target, state_dict_fn=lambda: {"w": 0},
        step_fn=lambda: int(learner.step), cadence_steps=1,
        sink=SimpleNamespace(emit=lambda e: None), run_id="oracle_u2",
    )

    sync_counts: list[int] = []
    for k in (1, 2, 3):
        drain._route_eval_result(
            coord, {"step": k, "promoted": False, "eval_broken": False})
        learner.step = k
        engine.maybe_sync(k)
        sync_counts.append(len(target.sync_payloads))

    assert sync_counts == [1, 2, 3], (
        f"sync must continue strictly unimpaired while the gate fails: {sync_counts}"
    )
    assert anchor.best_model_step is None, "a failing gate must freeze the DeployTag"
    assert save_anchor.calls == [], "a failing gate must never persist an anchor"
    # And the applier's own guard, driven directly, is a no-op on a failed round:
    assert apply_gate_decision(hooks, {"promoted": False, "eval_broken": False}) is None


def test_apply_gate_decision_has_no_sync_parameter() -> None:
    """R49 signature pin: the `sync_inference` keyword — the parameter that made
    sync-on-gate representable — no longer exists on the ONE applier."""
    params = inspect.signature(apply_gate_decision).parameters
    assert "sync_inference" not in params, (
        "apply_gate_decision must not carry a sync_inference parameter (R49: the old "
        "mode must be structurally unrepresentable)"
    )


# ── O-U6: regime_key honesty survives the split ──────────────────────────────────────
class _KickSpyPipeline:
    def __init__(self) -> None:
        self.received: list[dict] = []

    def run_evaluation(self, model, step, best, *, full_config, best_model_step,
                       ignore_stride=False) -> dict:
        self.received.append({
            "model": model, "step": step, "best": best,
            "full_config": full_config, "best_model_step": best_model_step,
        })
        return {"kicked": True, "round_id": "r0", "step": step, "reason": None}


class _AttrReadRecorder:
    """Records every attribute read; `weights` yields the actor-side sentinel."""

    def __init__(self, weights) -> None:
        object.__setattr__(self, "reads", [])
        object.__setattr__(self, "_weights", weights)

    def __getattr__(self, name: str):
        self.reads.append(name)
        if name == "weights":
            return self._weights
        return self._weights  # any read is already a violation; recorded either way


def _kick_config() -> StepCoordinatorConfig:
    """DERIVED from the production builder (WPMINT Phase K-A stage 0) — this file's deltas
    only. `None` is the EXPLICIT disarmed draw-rate posture; the four drain caps come from
    a MINTED `monitor.drain` block (R93/DR-11)."""
    return dataclasses.replace(
        _step_coordinator_config(stop_step=10**9, draw_rate_abort=None,
                                 drain_caps=_DRAIN_CAPS),
        eval_interval=4, log_interval=0,
    )


def test_eval_kick_fields_deploy_tag_model_not_actor_weights(tmp_path) -> None:
    """The deploy-matched incumbent fielded at an eval boundary is the TAG
    (`anchor_state.best_model`, by identity) — never anything read off the actor/pool.
    Zero pool attribute reads during the kick window (regime_key honesty)."""
    sentinel_tag = object()      # sentinel A — the DeployTag model
    sentinel_actor = object()    # sentinel B — the actor's live weights
    pipeline = _KickSpyPipeline()
    coord = StepCoordinator(
        trainer=SimpleNamespace(step=4, model=object(),
                                save_checkpoint=lambda info: None),
        buffer=SimpleNamespace(size=1000, capacity=100_000,
                               save_to_path=lambda p: None, resize=lambda n: None),
        pretrained_buffer=None, recent_buffer=None,
        pool=SimpleNamespace(games_completed=0), eval_pipeline=pipeline,
        subsystems=SimpleNamespace(gpu_monitor=None),
        anchor_state=SimpleNamespace(best_model=sentinel_tag, best_model_step=42),
        shutdown=ShutdownState(), eval_model=object(), bufs=None,
        config=_kick_config(), full_config={}, train_cfg={}, mixing_cfg={},
        sink=None, monitor_cfg=MonitorConfig(),
    )
    recorder = _AttrReadRecorder(sentinel_actor)
    coord.pool = recorder          # any pool read during the kick is recorded
    coord._train_step = 4          # the eval_interval=4 boundary
    recorder.reads.clear()

    kicked, busy = coord._maybe_kick_eval(coord.config)

    assert kicked is True and busy is False
    assert len(pipeline.received) == 1
    got = pipeline.received[0]
    assert got["best"] is sentinel_tag, (
        "the eval kick must field the DeployTag model (identity), not a copy and not "
        "anything actor-side"
    )
    assert got["best_model_step"] == 42
    assert recorder.reads == [], (
        f"the kick must read NOTHING off the pool/actor: {recorder.reads}"
    )
    assert all(v is not sentinel_actor for v in got.values()), (
        "the actor's live weights must never reach run_evaluation"
    )
