# >300 justify (R8): ONE finding (RED-TEAM-2 F-RT2-1) — the _bounded_join_timeout layer-2 guard driven on both paths that reach it (escalate_and_finalize, drain_pending); over half the file is the self-contained fixture copy the eval-suite house convention requires (no shared conftest), incl. the _RealisticFakeProcess reproducing CPython's join(inf) OverflowError.
"""RED-TEAM-2 F-RT2-1 (BLOCKER) FIX, layer 2 — `_bounded_join_timeout` structural guard
(mantis-migration/wp/WP11A/REDTEAM_2.md Finding F-RT2-1).

RED-TEAM-2 built a REAL `EvalPipeline` + a REAL OS subprocess and confirmed empirically:
a schema-valid (pre-fix) `worker_kill_grace_sec=float("inf")` reaches `_escalate_and_
finalize` (pipeline.py, invoked directly from the background poller's `_poll_loop` tick,
entirely OUTSIDE `_finalize_round`'s F1 layer-2 catch-all) and calls a real
`multiprocessing.Process.join(float("inf"))`, which raises an UNCAUGHT `OverflowError`
deep inside `selectors.select()` — killing the poller thread silently, exactly F1's
original failure mode, via a code path F1's own fix never covered.

Layer 1 (config/schema.py, `tests/config/test_eval_schema_bounds.py`) makes a non-finite
`worker_kill_grace_sec` unreachable through a config load. THIS suite proves layer 2
(pipeline.py's `_bounded_join_timeout`, defense-in-depth): even if a non-finite value
reaches the runtime path by some future bug that bypasses schema validation entirely
(simulated here via `EvalConfig.model_copy(update=...)`, which — unlike normal
construction — does NOT re-run field validators, the one supported way to hand-construct
a schema-shaped-but-invalid config for exactly this kind of injection test), escalation
still completes: a delivered `eval_broken` result, the poller thread alive, never a hang.

Does NOT spawn a real OS subprocess (same rationale as the frozen `test_eval_broken.py`/
`test_round_completion_error.py`): a `_RealisticFakeProcess` reproduces the ONE behavior
under test — `multiprocessing.Process.join` raises `OverflowError` for a non-finite
timeout (mirrors CPython's real `selectors.select()` -> `math.ceil(timeout * 1e3)` crash,
confirmed by RED-TEAM-2 against a real subprocess) — while staying deterministic and
instant everywhere else. This is a NEW file (frozen-oracle discipline unaffected; does not
edit `tests/eval/test_eval_broken.py` or `tests/eval/test_round_completion_error.py`).
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
from mantis.model import CnnArch, build_net


# ── shared fixtures (self-contained; house convention — see test_round_completion_error.py
#    / test_eval_broken.py's own docstrings for why each oracle-write agent keeps a private
#    copy rather than a shared conftest.py) ──────────────────────────────────────────────
def _tiny_model():
    import torch

    arch = CnnArch(board_size=5, in_channels=4, filters=8, res_blocks=1)
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
        random_model_sims=96, sealbot_model_sims=128, kraken_model_sims=128,
        strix_model_sims=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=0.05, worker_kill_grace_sec=0.05, gate=gate, ladder=ladder,
        ply_cap_adjudication=None, strength_floor=None,
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)


def _cfg_with_bypassed_worker_kill_grace_sec(value: float) -> EvalConfig:
    """The one supported way to construct a schema-shaped-but-schema-INVALID `EvalConfig`
    for injection testing: `model_copy(update=...)` does not re-run field validators
    (unlike normal construction/`model_validate`, which would reject a non-finite value at
    the schema boundary — see `test_eval_schema_bounds.py`). This simulates "a future code
    path constructs/mutates an EvalConfig without going through config-load validation",
    exactly the residual-risk scenario layer 2 exists to cover."""
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
        amp_dtype="bf16",
        max_plies=128,
        c_visit=50.0, c_scale=1.0, run_id="oracle_test_run", spool_dir=spool_dir,
        ladder_state_path=tmp_path / "ladder_state.json", promotion=_promotion_hooks(tmp_path),
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


class _RealisticFakeProcess:
    """Unlike the frozen suites' `_FakeProcess` (whose `.join()` is an unconditional
    no-op), THIS fake reproduces the one real-`multiprocessing.Process` behavior under
    test: `.join(timeout)` raises `OverflowError` for a non-finite timeout — mirroring
    CPython's real `selectors.select()` -> `math.ceil(timeout * 1e3)` crash that
    RED-TEAM-2 confirmed empirically against a real OS subprocess. Every timeout it is
    ever called with is recorded in `join_calls`, so a test can assert the value that
    actually reached `.join()` was bounded BEFORE the call, not merely that no exception
    happened to propagate."""

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
    """Test-level hard watchdog (house convention): this test must itself never hang even
    if the fix under test regresses."""
    box: dict[str, Any] = {}

    def _run() -> None:
        box["value"] = fn()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"operation exceeded the {timeout}s test-level hard bound (must never hang)")
    return box.get("value")


# ── unit-level: the helper itself, every non-finite/negative/huge-finite input ──────────
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


# ── integration: the REAL background poller's own tick invokes _escalate_and_finalize
#    directly (RED-TEAM-2's exact code path -- entirely outside _finalize_round's F1
#    layer-2 catch-all) with a non-finite worker_kill_grace_sec that bypassed schema ──────
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

        # Push the fake clock far past round_timeout_sec (0.05s): the REAL background
        # poller's next real-time tick (every _POLL_TICK_SEC=0.02s) will see
        # `elapsed > round_timeout_sec` and call `_escalate_and_finalize` ON ITS OWN --
        # RED-TEAM-2's exact reproduction path, not a hand-invoked method call.
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

        # 1. escalation completed and delivered a result -- never a dead thread/hang.
        assert result is not None, (
            "escalation with a non-finite worker_kill_grace_sec must still deliver a "
            "result via the mailbox, never hang the poller forever"
        )
        assert result["eval_broken_reason"] is not None
        assert result.get("promoted") is False

        # 2. the escalation actually ran (terminate -> join -> kill -> join), and every
        #    timeout that reached the fake process's .join() was bounded BEFORE the call
        #    -- never inf itself (proves the clamp happens ahead of the join, not that an
        #    exception merely failed to propagate for some unrelated reason).
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

        # 4. the poller THREAD itself is still alive -- the exact invariant F-RT2-1 found
        #    broken (RED-TEAM-2: "poller thread confirmed dead... zero result delivered").
        assert pipeline._poller.is_alive(), (  # noqa: SLF001 -- intentional internal check
            "the eval-pipeline-poller thread must survive a non-finite worker_kill_grace_sec"
        )

        # 5. the pipeline is still usable afterwards.
        assert pipeline.poll_completed() is None
    finally:
        pipeline.stop()


# ── the drain/teardown path shares the identical proc.join(max(worker_kill_grace_sec, 0.0))
#    call shape (RED-TEAM-2 explicitly named `drain_or_kill`/`drain_pending()` as reachable
#    by the same crash, "not independently re-verified... for time") -- verified here ──────
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
