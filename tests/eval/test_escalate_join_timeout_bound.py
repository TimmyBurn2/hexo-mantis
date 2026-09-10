# >300 justify (R8): ONE finding — the `_bounded_join_timeout` layer-2 guard driven on both
# paths that reach it — and over half the file is the self-contained fixture copy the eval-suite
# house convention requires, including the fake process that reproduces CPython's join(inf).
"""Layer 2 of the non-finite `worker_kill_grace_sec` fix: `_bounded_join_timeout`.

A pre-fix `worker_kill_grace_sec=float("inf")` reached `_escalate_and_finalize` from the poller's
tick — outside `_finalize_round`'s catch-all — and called `Process.join(inf)`, which raises an
UNCAUGHT `OverflowError` inside `selectors.select()` and kills the poller thread silently. Layer
1 makes such a value unreachable through a config load; this suite proves that one arriving by a
bug that bypasses validation (simulated with `model_copy(update=...)`, which does not re-run
field validators) still ends in a delivered `eval_broken` result with the poller alive.
"""
from __future__ import annotations

import math
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung
from mantis.eval.pipeline import (
    DrainCaps,
    _bounded_join_timeout,
    _JOIN_TIMEOUT_CEILING_SEC,
    build_eval_pipeline,
)
from mantis.eval.promote import DeployTagHooks
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net

_GSPEC = lookup("gnn_axis_v1")


# Fixtures are a private copy rather than a shared conftest — the eval-suite house convention.
def _tiny_model():
    import torch

    arch = GnnArch(in_dim=int(_GSPEC.node_feat_dim), edge_dim=int(_GSPEC.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    return net


def _eval_cfg(**overrides: Any) -> EvalConfig:
    rungs = [
        LadderRung(name="sealbot_d5", bot="sealbot", variant="d5", depth=5, opponent_sims=None,
                   opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=32),
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
        round_timeout_sec=0.05, worker_kill_grace_sec=0.05, gate=gate, ladder=ladder,
        ply_cap_adjudication=None, strength_floor=None,
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)


def _cfg_with_bypassed_worker_kill_grace_sec(value: float) -> EvalConfig:
    """The one supported way to build a schema-shaped but schema-INVALID `EvalConfig` for
    injection testing: `model_copy(update=...)` does not re-run field validators. It simulates a
    future code path that mutates an `EvalConfig` without going through config-load validation,
    which is the residual risk layer 2 covers."""
    return _eval_cfg().model_copy(update={"worker_kill_grace_sec": value})


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


def _pipeline_kwargs(tmp_path: Path, *, eval_cfg: "EvalConfig | None" = None, **overrides: Any) -> dict:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(exist_ok=True)
    kwargs = dict(
        eval_cfg=eval_cfg if eval_cfg is not None else _eval_cfg(),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=2.0, eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=2.0, terminal_eval_hard_cap_sec=2.0,
        ),
        encoding="v6_live2_ls",
        max_plies=128,
        c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, run_id="oracle_test_run", spool_dir=spool_dir, game_record_dir=str(spool_dir) + "_games",
        ladder_state_path=tmp_path / "ladder_state.json", promotion=_promotion_hooks(tmp_path),
        # The pipeline resolves the fused-forward memory bound ONCE in the parent, because the
        # eval child is a SECOND allocator no in-process bound can see; `None` is the GRID arm.
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


class _RealisticFakeProcess:
    """Unlike the frozen suites' no-op `_FakeProcess`, this one reproduces the real
    `multiprocessing.Process` behaviour under test: `.join(timeout)` raises `OverflowError` for a
    non-finite timeout. Every timeout it is called with is recorded in `join_calls`, so a test
    can assert the value that reached `.join()` was bounded BEFORE the call rather than that no
    exception happened to propagate."""

    def __init__(self, *, target=None, args=(), kwargs=None, daemon=None) -> None:
        self._target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.pid = 4242
        self.alive = False
        self.exitcode: "int | None" = None
        self.terminated = False
        self.killed = False
        self.join_calls: "list[float | None]" = []

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: "float | None" = None) -> None:
        self.join_calls.append(timeout)
        if timeout is not None and not math.isfinite(timeout):
            raise OverflowError("cannot convert float infinity to integer")
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
        self.last_process: "_RealisticFakeProcess | None" = None

    def Process(self, *, target=None, args=(), kwargs=None, daemon=None) -> _RealisticFakeProcess:
        proc = _RealisticFakeProcess(target=target, args=args, kwargs=kwargs, daemon=daemon)
        self.last_process = proc
        return proc


@pytest.fixture()
def fake_mp(monkeypatch):
    ctx = _FakeCtx()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name=None: ctx)
    return ctx


def _bounded(fn, *, timeout: float):
    """Test-level hard watchdog: this test must never hang even if the fix regresses."""
    box: dict[str, Any] = {}

    def _run() -> None:
        box["value"] = fn()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"operation exceeded the {timeout}s test-level hard bound (must never hang)")
    return box.get("value")


@pytest.mark.parametrize(
    "raw,expected",
    [
        (float("inf"), _JOIN_TIMEOUT_CEILING_SEC),
        (float("-inf"), _JOIN_TIMEOUT_CEILING_SEC),
        (float("nan"), _JOIN_TIMEOUT_CEILING_SEC),
        (-5.0, 0.0),
        (0.0, 0.0),
        (10.0, 10.0),
        (_JOIN_TIMEOUT_CEILING_SEC, _JOIN_TIMEOUT_CEILING_SEC),
        (_JOIN_TIMEOUT_CEILING_SEC * 10.0, _JOIN_TIMEOUT_CEILING_SEC),
    ],
    ids=["inf", "-inf", "nan", "negative", "zero", "normal", "at_ceiling", "far_above_ceiling"],
)
def test_bounded_join_timeout_never_raises_and_stays_finite(raw: float, expected: float) -> None:
    result = _bounded_join_timeout(raw)
    assert math.isfinite(result), f"_bounded_join_timeout({raw!r}) must always return finite, got {result!r}"
    assert 0.0 <= result <= _JOIN_TIMEOUT_CEILING_SEC
    assert result == expected


# Integration: the REAL poller's own tick invokes `_escalate_and_finalize` directly — the
# reproduction path, entirely outside `_finalize_round`'s catch-all.
def test_escalate_and_finalize_survives_non_finite_worker_kill_grace_sec(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    clock = FakeClock(0.0)
    bad_cfg = _cfg_with_bypassed_worker_kill_grace_sec(float("inf"))
    assert not math.isfinite(bad_cfg.worker_kill_grace_sec)  # confirm the injection landed
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, eval_cfg=bad_cfg, sink=sink, clock=clock), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert ack["kicked"] is True
        proc = fake_mp.last_process
        assert proc is not None
        assert proc.alive is True  # started, never told to die -> a genuine hang

        # Push the fake clock past `round_timeout_sec`: the REAL poller's next tick sees the
        # elapsed time and calls `_escalate_and_finalize` ON ITS OWN, rather than by hand.
        clock.advance(1000.0)

        def _wait_for_result():
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                result = pipeline.poll_completed()
                if result is not None:
                    return result
                time.sleep(0.01)
            return None

        result = _bounded(_wait_for_result, timeout=6.0)

        # 1. escalation completed and delivered a result, never a dead thread.
        assert result is not None, (
            "escalation with a non-finite worker_kill_grace_sec must still deliver a "
            "result via the mailbox, never hang the poller forever"
        )
        assert result["eval_broken_reason"] is not None
        assert result.get("promoted") is False

        # 2. the escalation ran, and every timeout that reached `.join()` was bounded BEFORE it.
        assert proc.terminated is True
        assert proc.killed is True
        assert proc.join_calls, "escalate_and_finalize must join() the process at least once"
        for called_timeout in proc.join_calls:
            if called_timeout is not None:
                assert math.isfinite(called_timeout), (
                    f"a non-finite timeout ({called_timeout!r}) reached Process.join() -- "
                    "the layer-2 clamp must run BEFORE every join() call"
                )

        # 3. a named eval_broken event was emitted (never silent).
        broken = sink.named("eval_broken")
        assert broken, "no eval_broken event emitted for the non-finite-grace escalation"

        # The poller THREAD is still alive, the exact invariant the finding found broken.
        assert pipeline._poller.is_alive(), (  # noqa: SLF001 -- intentional internal check
            "the eval-pipeline-poller thread must survive a non-finite worker_kill_grace_sec"
        )

        # 5. the pipeline is still usable afterwards.
        assert pipeline.poll_completed() is None
    finally:
        pipeline.stop()


# The drain/teardown path shares the identical `proc.join(...)` call shape and is reachable by
# the same crash, so it is verified here too.
def test_drain_pending_survives_non_finite_worker_kill_grace_sec(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    bad_cfg = _cfg_with_bypassed_worker_kill_grace_sec(float("inf"))
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, eval_cfg=bad_cfg, sink=sink), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        proc = fake_mp.last_process
        assert proc is not None
        assert proc.alive is True  # a genuine hang: drain_pending must terminate/kill it

        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)

        assert result is not None, "drain_pending() must never hang on a non-finite grace period"
        assert result["eval_broken_reason"] is not None
        assert proc.terminated is True
        assert proc.killed is True
        for called_timeout in proc.join_calls:
            if called_timeout is not None:
                assert math.isfinite(called_timeout)

        broken = sink.named("eval_broken")
        assert broken, "no eval_broken event emitted for the non-finite-grace drain"
    finally:
        pipeline.stop()
