"""The eval_broken isolation law: every subprocess join is timeout-bounded and every
timeout, crash or garbage result escalates to a named `eval_broken` event AND a routed
broken result — never a hang, never a silent skip.

No real OS subprocess or worker module is spawned: a `_FakeProcess` whose `alive`/`exitcode`
the test flips is injected through a monkeypatched stdlib `multiprocessing.get_context`, and
a `FakeClock` (the `clock=` constructor kwarg) drives the hung-past-timeout scenario without
a real sleep. `drain_pending()` is the synchronous, budget-bounded join point, so calling it
directly exercises the escalation without waiting on the background poller's tick. Every
such call is wrapped in `_bounded()`, a thread-join watchdog, so an implementation bug that
hangs cannot hang this suite.

>300 justify: five eval_broken scenarios (killed, hung, garbage-json, missing-file,
never-promotes-never-skips) share one fake-process/fake-context/fake-clock harness and one
minimal-config builder; splitting them would duplicate the harness and let the escalation
reason taxonomy drift across files.
"""
from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.config.schema import EvalConfig, GateConfig
from mantis.eval.errors import ResultContractError
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks, apply_gate_decision
from mantis.eval.rounds import partial_gate_path, write_partial_gate
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net

_GSPEC = lookup("gnn_axis_v1")


def _tiny_model() -> torch.nn.Module:
    arch = GnnArch(in_dim=int(_GSPEC.node_feat_dim), edge_dim=int(_GSPEC.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    return net


def _eval_cfg(**overrides: Any) -> EvalConfig:
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625, sequential=None,
    )
    defaults = dict(
        random_model_sims=96, max_plies=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=0.3, worker_kill_grace_sec=0.2, gate=gate,
        ply_cap_adjudication=None, strength_floor=None,
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)


def _promotion_hooks(tmp_path: Path) -> DeployTagHooks:
    from types import SimpleNamespace

    return DeployTagHooks(
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        best_model_path=tmp_path / "best_model.pt",
        run_id="oracle_test_run",
        encoding="v6_live2_ls",
        save_anchor=lambda *a, **k: None,
        guarded_load=lambda *a, **k: None,
    )


def _pipeline_kwargs(tmp_path: Path, *, eval_cfg: EvalConfig | None = None, **overrides: Any) -> dict:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(exist_ok=True)
    kwargs = dict(
        eval_cfg=eval_cfg if eval_cfg is not None else _eval_cfg(),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=2.0,
            eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=2.0,
            terminal_eval_hard_cap_sec=2.0,
        ),
        encoding="v6_live2_ls",
        max_plies=128,
        c_visit=50.0, c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16,
        run_id="oracle_test_run",
        spool_dir=spool_dir, game_record_dir=str(spool_dir) + "_games",
        promotion=_promotion_hooks(tmp_path),
        # `None` is the no-fused-forward arm; the parameter is required, so it is
        # written out rather than omitted.
        fused_graph_caps=None,
        inference_batching=None,
    )
    kwargs.update(overrides)
    return kwargs


class _SpySink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        self.events.append(dict(event))

    def named(self, name: str) -> list[dict]:
        return [e for e in self.events if e.get("event") == name]


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeProcess:
    """Stand in for a spawned worker whose `alive`/`exitcode` the test flips directly."""

    def __init__(self, *, target=None, args=(), kwargs=None, daemon=None) -> None:
        self._target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.pid = 4242
        self.alive = False
        self.exitcode: int | None = None
        self.terminated = False
        self.killed = False

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False
        if self.exitcode is None:
            self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        if self.exitcode is None:
            self.exitcode = -9


class _FakeCtx:
    def __init__(self) -> None:
        self.process_calls: list[dict] = []
        self.last_process: _FakeProcess | None = None

    def Process(self, *, target=None, args=(), kwargs=None, daemon=None) -> _FakeProcess:
        self.process_calls.append({"target": target, "args": args, "kwargs": kwargs})
        proc = _FakeProcess(target=target, args=args, kwargs=kwargs, daemon=daemon)
        self.last_process = proc
        return proc


@pytest.fixture()
def fake_mp(monkeypatch):
    ctx = _FakeCtx()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name=None: ctx)
    return ctx


def _bounded(fn, *, timeout: float):
    """Run `fn` on a daemon thread and fail loudly if it does not return within `timeout`."""
    box: dict[str, Any] = {}

    def _run() -> None:
        box["value"] = fn()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"operation exceeded the {timeout}s test-level hard bound (must never hang)")
    return box.get("value")


def _result_path_from_ctx(ctx: _FakeCtx) -> Path:
    """Recover the result-sidecar path from the args the pipeline passed the faked Process."""
    assert ctx.process_calls, "no subprocess was ever requested"
    args = ctx.process_calls[-1]["args"] or ()
    candidates = [Path(a) for a in args if isinstance(a, (str, Path)) and str(a).endswith(".json")]
    result_candidates = [p for p in candidates if "result" in p.name] or candidates[1:2] or candidates[:1]
    assert result_candidates, f"could not infer a result-file path from spawn args: {args}"
    return result_candidates[0]


def test_killed_worker_yields_eval_broken_and_clean_drain(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert ack["kicked"] is True
        proc = fake_mp.last_process
        assert proc is not None
        # SIGKILL mid-round: the process is gone, with a signal-style exitcode.
        proc.alive = False
        proc.exitcode = -9

        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        assert result is not None
        assert result["eval_broken_reason"] is not None
        assert result.get("promoted") is False
        assert result.get("wr_sealbot") is None
        assert result.get("step") == 1000

        broken_events = sink.named("eval_broken")
        assert broken_events, "no eval_broken event emitted"
        ev = broken_events[-1]
        assert ev.get("reason") in ("exit_nonzero", "killed")
        assert "exit_code" in ev
        assert "phase" in ev
    finally:
        pipeline.stop()


def test_a_resumable_stop_abandons_the_live_round_at_once_without_the_drain_budget(fake_mp, tmp_path) -> None:
    """CARD-STOP-DRAIN-VS-GRACE: a LIVE round is terminated at once, finalised as killed, nothing in flight."""
    sink = _SpySink()
    clock = FakeClock(0.0)
    cfg = _eval_cfg(round_timeout_sec=3600.0, worker_kill_grace_sec=0.1)
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, eval_cfg=cfg, sink=sink, clock=clock), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert ack["kicked"] is True
        proc = fake_mp.last_process
        assert proc is not None and proc.alive is True

        result = _bounded(lambda: pipeline.abandon_pending(), timeout=5.0)
        assert result is not None and result["eval_broken_reason"] == "abandoned"
        assert proc.terminated is True
        assert sink.named("eval_round_abandoned")[-1]["reason"] == "resumable_stop"
        assert pipeline.abandon_pending() is None, "nothing may remain in flight after an abandon"
    finally:
        pipeline.stop()


def test_hung_worker_join_timeout_escalates_terminate_then_kill(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    clock = FakeClock(0.0)
    cfg = _eval_cfg(round_timeout_sec=0.2, worker_kill_grace_sec=0.1)
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, eval_cfg=cfg, sink=sink, clock=clock), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert ack["kicked"] is True
        proc = fake_mp.last_process
        assert proc is not None
        assert proc.alive is True  # started, never told to die -> a genuine hang

        clock.advance(1000.0)  # far past round_timeout_sec + worker_kill_grace_sec

        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        assert result is not None
        assert result["eval_broken_reason"] is not None
        assert proc.terminated is True
        assert proc.killed is True   # terminate() alone did not clear it -> escalated to kill()

        broken_events = sink.named("eval_broken")
        assert broken_events and broken_events[-1].get("reason") == "join_timeout"
    finally:
        pipeline.stop()


def test_garbage_sidecar_json_is_eval_broken_not_a_crash(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        result_path = _result_path_from_ctx(fake_mp)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{not valid json::: ")

        proc = fake_mp.last_process
        proc.alive = False
        proc.exitcode = 0  # the child exited "cleanly" but wrote garbage

        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        assert result is not None
        assert result["eval_broken_reason"] is not None
        assert result.get("promoted") is False

        broken_events = sink.named("eval_broken")
        assert broken_events and broken_events[-1].get("reason") == "result_invalid"
    finally:
        pipeline.stop()


def test_missing_result_file_is_eval_broken(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        proc = fake_mp.last_process
        proc.alive = False
        proc.exitcode = 0  # clean exit, but NO result file was ever written

        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        assert result is not None
        assert result["eval_broken_reason"] is not None
        assert result.get("promoted") is False

        broken_events = sink.named("eval_broken")
        assert broken_events and broken_events[-1].get("reason") == "result_missing"
    finally:
        pipeline.stop()


def test_eval_broken_never_promotes_and_never_silently_skips(fake_mp, tmp_path) -> None:
    # Both must hold together: a result with no event is silent, an event with no result
    # is dropped, and each alone is a partial failure.
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        proc = fake_mp.last_process
        proc.alive = False
        proc.exitcode = -9

        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        got_routed_result = result is not None and result.get("promoted") is False
        got_event = bool(sink.named("eval_broken"))
        assert got_routed_result, "a broken round must still route a result with promoted=False"
        assert got_event, "a broken round must still emit eval_broken (never silent)"
        # With NO partial gate sidecar; a round that finished its gate phase is the A-3 tests'.
        assert result["gate_verdict_partial"] is False
        assert sink.named("eval_broken")[-1]["partial_gate"] is False
    finally:
        pipeline.stop()


def _gate_verdict(promoted: bool) -> dict:
    return {"wr_screen": 0.7, "wr_confirm": 0.66, "n_screen": 80, "n_confirm": 128, "n_pooled": 208,
            "escalated": True, "elo_ci_lower_boot": 40.0, "low_power": False, "eff_n": 100.0,
            "reason": "", "deploy_matched": True, "promoted": promoted, "rule": "screen_confirm",
            "llr": None, "pairs_played": None, "stopped": None}


def _kill_after_partial(fake_mp, pipeline, *, partial_step: int, promoted: bool) -> str:
    pipeline.run_evaluation(_tiny_model(), 3000, None, full_config={}, best_model_step=None)
    proc = fake_mp.last_process
    result_path = pipeline._inflight["spec"].result_path
    write_partial_gate(result_path, step=partial_step, gate_result=_gate_verdict(promoted))
    proc.alive = False
    proc.exitcode = -9  # killed at the bound, after the gate phase
    return result_path


def test_a_round_killed_after_its_gate_phase_promotes_off_the_partial_verdict(fake_mp, tmp_path) -> None:
    """A-3 (R355(e)): a round killed after its gate phase is broken for the ladder and still promotes."""
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        result_path = _kill_after_partial(fake_mp, pipeline, partial_step=3000, promoted=True)
        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        assert result is not None
        assert result["eval_broken_reason"] is not None, "the ROUND is still broken (no rungs)"
        assert result["gate_verdict_partial"] is True
        assert result["promoted"] is True and result["promoted_step"] == 3000
        assert result["gate"]["promoted"] is True
        assert sink.named("eval_broken")[-1]["partial_gate"] is True
        assert sink.named("eval_round_complete")[-1]["promoted"] is True
        assert not partial_gate_path(result_path).exists(), "consumed with the round"
    finally:
        pipeline.stop()


def test_a_partial_verdict_that_did_not_promote_does_not(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        _kill_after_partial(fake_mp, pipeline, partial_step=3000, promoted=False)
        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        assert result["gate_verdict_partial"] is True and result["promoted"] is False
    finally:
        pipeline.stop()


def test_a_partial_from_another_step_is_ignored(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        _kill_after_partial(fake_mp, pipeline, partial_step=2000, promoted=True)
        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        assert result["gate_verdict_partial"] is False and result["promoted"] is False
        assert sink.named("eval_broken")[-1]["partial_gate"] is False
    finally:
        pipeline.stop()


def test_apply_gate_decision_honours_a_partial_verdict_and_refuses_a_broken_round_without_one(tmp_path) -> None:
    loads: list = []
    hooks = _promotion_hooks(tmp_path)
    hooks = DeployTagHooks(anchor_state=hooks.anchor_state, best_model_path=hooks.best_model_path,
                           run_id=hooks.run_id, encoding=hooks.encoding,
                           save_anchor=lambda *a, **k: None,
                           guarded_load=lambda *a, **k: loads.append(a))
    partial = {"eval_broken_reason": "killed", "gate_verdict_partial": True, "promoted": True,
               "step": 3000}
    assert apply_gate_decision(hooks, partial) == 3000 and len(loads) == 1
    broken = {"eval_broken_reason": "killed", "gate_verdict_partial": False, "promoted": False,
              "step": 3000}
    assert apply_gate_decision(hooks, broken) is None and len(loads) == 1


def test_a_partial_left_by_an_earlier_process_cannot_promote_a_new_round(fake_mp, tmp_path) -> None:
    """A watchdog exit finalises nothing and the relaunch restores the same round id: the ghost must go."""
    sink = _SpySink()
    kwargs = _pipeline_kwargs(tmp_path, sink=sink)
    pipeline = build_eval_pipeline(**kwargs, leaf_batch_size=1)
    try:
        pipeline.run_evaluation(_tiny_model(), 3000, None, full_config={}, best_model_step=None)
        result_path = pipeline._inflight["spec"].result_path
    finally:
        pipeline.stop()
    write_partial_gate(result_path, step=3000, gate_result=_gate_verdict(True))  # the ghost
    pipeline2 = build_eval_pipeline(**kwargs, leaf_batch_size=1)
    try:
        assert not partial_gate_path(result_path).exists(), "the constructor sweep takes it"
        write_partial_gate(result_path, step=3000, gate_result=_gate_verdict(True))
        pipeline2.run_evaluation(_tiny_model(), 3000, None, full_config={}, best_model_step=None)
        spec = pipeline2._inflight["spec"]
        assert spec.result_path == result_path, "the same round id recurs after a restore"
        assert not partial_gate_path(result_path).exists(), "the spawn takes it too"
        proc = fake_mp.last_process
        proc.alive = False
        proc.exitcode = -9
        result = _bounded(lambda: pipeline2.drain_pending(), timeout=5.0)
        assert result["gate_verdict_partial"] is False and result["promoted"] is False
    finally:
        pipeline2.stop()
