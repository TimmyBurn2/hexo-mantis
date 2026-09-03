"""⊕ WP11-A — eval_broken isolation law (mantis.eval.pipeline; design §a.3/§c.3/§c.4).

RED-at-import until IMPL writes `mantis.eval.pipeline`. Isolation law 2 (non-negotiable):
every subprocess join is timeout-bounded; a timeout/crash/garbage-result ALWAYS escalates
to a named `eval_broken` event + a routed broken result — never a hang, never a silent
skip.

These tests deliberately do NOT spawn a real OS subprocess or the real `mantis.eval.worker`
module (that would make every scenario here torch-import-latency-dependent and racy on a
signal/pickling boundary). Instead they inject:
  * a `_FakeProcess` (mutable `alive`/`exitcode` the test flips directly — simulating "the
    child died", "the child is still alive", "the child exited 0") via a monkeypatched
    STDLIB `multiprocessing.get_context`, exactly as `test_pipeline_isolation.py` does
    (same rationale; see that file's docstring for why patching the shared module object
    is robust regardless of pipeline.py's own import style);
  * a `FakeClock` (the `clock=` constructor kwarg `build_eval_pipeline` already exposes,
    §c.3) so a "hung past round_timeout_sec" scenario is driven by advancing a fake
    monotonic clock, never a real sleep — deterministic and instant.

`drain_pending()` is the synchronous, budget-bounded join point (§c.3: "budget = min(...)")
— calling it directly after arranging the fake process's state is the natural way to
exercise the escalation logic without waiting on the pipeline's own background poller
thread's real-time tick interval. Every call is ALSO wrapped in `_bounded()` (a
thread-`join(timeout)` test-level hard watchdog) so a real implementation bug that hangs
cannot hang THIS TEST SUITE — belt and suspenders over the isolation law's own bound.

>300 justify: five eval_broken scenarios (killed, hung, garbage-json, missing-file,
never-promotes-never-skips) sharing one fake-process/fake-context/fake-clock harness and
one minimal-config builder — splitting them would duplicate that harness five times and
let the escalation-reason taxonomy (join_timeout/exit_nonzero/killed/result_missing/
result_invalid) drift out of sync across files, which is exactly the "quiet failure path"
class this suite exists to close.
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
from mantis.model import CnnArch, build_net


# ── shared fixtures (self-contained; duplicated from test_pipeline_isolation.py by design
#    — see WP11A_dispatch/ORACLE_NOTES: two independent oracle-write agents wrote the
#    tests/eval/ suites in parallel and neither adds a shared conftest.py to avoid a
#    collision) ─────────────────────────────────────────────────────────────────────────
def _tiny_model() -> torch.nn.Module:
    arch = CnnArch(board_size=5, in_channels=4, filters=8, res_blocks=1)
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
        random_model_sims=96, sealbot_model_sims=128, kraken_model_sims=128,
        strix_model_sims=128, random_floor_games=4, worker_device="cpu",
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
        amp_dtype="bf16",
        max_plies=128,
        c_visit=50.0, c_scale=1.0,
        run_id="oracle_test_run",
        spool_dir=spool_dir,
        ladder_state_path=tmp_path / "ladder_state.json",
        promotion=_promotion_hooks(tmp_path),
        # F-816-10 D-1: the pipeline resolves the fused-forward memory bound ONCE in
        # the parent and carries it to every `RoundSpec`, because the eval child is a
        # SECOND allocator on the same card that no in-process bound can see. `None`
        # is the GRID arm — these fixtures run `v6_live2_ls`, which has no fused graph
        # forward to bound — and it is written out rather than omitted (the parameter
        # is required for that reason).
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
    """Mutable stand-in for a spawned worker process: the test flips `alive`/`exitcode`
    directly to simulate death-by-signal, a clean exit, or an indefinite hang."""

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
    """Test-level hard watchdog: run `fn` on a daemon thread, fail loudly if it does not
    return within `timeout` — this test suite must never hang, even under an IMPL bug."""
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
    """Best-effort recovery of the result-sidecar path the pipeline told its worker to
    write to, from the args it passed the (faked) Process constructor — the worker CLI is
    `python -m mantis.eval.worker <spec.json> <result.json>` (design §a.3), so the second
    positional-ish path-like arg is the result file."""
    assert ctx.process_calls, "no subprocess was ever requested"
    args = ctx.process_calls[-1]["args"] or ()
    candidates = [Path(a) for a in args if isinstance(a, (str, Path)) and str(a).endswith(".json")]
    result_candidates = [p for p in candidates if "result" in p.name] or candidates[1:2] or candidates[:1]
    assert result_candidates, f"could not infer a result-file path from spawn args: {args}"
    return result_candidates[0]


# ── scenarios ────────────────────────────────────────────────────────────────────────────
def test_killed_worker_yields_eval_broken_and_clean_drain(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert ack["kicked"] is True
        proc = fake_mp.last_process
        assert proc is not None
        # simulate SIGKILL mid-round: the process is simply gone, negative signal-style exitcode.
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
    # Both must hold together — a routed promoted=False result WITH no event (silent), or
    # an event WITH no routed result (dropped), are each a partial failure this test rejects.
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
