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

from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung
from mantis.eval.errors import ResultContractError
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
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
    rungs = [
        LadderRung(name="sealbot_d5", bot="sealbot", variant="d5", depth=5, opponent_sims=None,
                   opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=32),
        LadderRung(name="random_floor_rung", bot="random", variant="raw", depth=None,
                   opponent_sims=None, opening_book="book_v1_s20260625_p4",
                   deploy_matched=True, games_max=32),
    ]
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625,
    )
    ladder = LadderConfig(
        rungs=rungs, round_games=64, min_games_per_active_rung=4,
        graduation_wr_lower_ci=0.75, graduation_consec_rounds=3, activation_wr_lower_ci=0.65,
        calibration_every_k_rounds=4, calibration_games=8, bootstrap_resamples=1000,
        bootstrap_ci_level=0.95, bt_prior_games=1.0, bootstrap_seed=1234,
    )
    defaults = dict(
        random_model_sims=96, sealbot_model_sims=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=0.3, worker_kill_grace_sec=0.2, gate=gate, ladder=ladder,
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
        c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16,
        run_id="oracle_test_run",
        spool_dir=spool_dir, game_record_dir=str(spool_dir) + "_games",
        ladder_state_path=tmp_path / "ladder_state.json",
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
    finally:
        pipeline.stop()
