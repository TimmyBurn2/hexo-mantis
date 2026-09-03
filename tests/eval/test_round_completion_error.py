# Exceeds the 300-line soft cap (R8): one defect, three routes. The poller thread, the
# drain and the never-silent assertion are the same failure observed where it can reach
# the run from, and a route moved to another file stops sharing the harness that proves
# the other two see the same exception.
"""RED-TEAM-FIX WP11-A F1 (BLOCKER), layer 2 — the poller/round-completion thread must be
exception-proof (mantis-migration/wp/WP11A/RED_TEAM.md Finding F1; isolation law 2: "a
timeout/crash/garbage-result ALWAYS escalates to a named `eval_broken` event + a routed
broken result — never a hang, never a silent skip").

Pre-fix: an uncaught `KeyError` (or any other exception) inside `EvalPipeline._success_
result` (reached via `_finalize_round` -> `_read_worker_result`) propagated straight out
of the daemon `eval-pipeline-poller` thread. Python's default thread excepthook printed a
traceback and the THREAD DIED: `_finalize_round`'s shared tail (`self._inflight = None;
self._mailbox.append(result)`) never ran, so the in-flight round was never cleared and
never delivered (`poll_completed()` returns `None` forever) and the poller's own `eval_
round` heartbeat — its SOLE source — stopped beating, silent until the run-safety
watchdog's staleness deadline eventually fires (up to `heartbeat_deadline_eval_round_sec`,
mint default 1800s).

This suite does NOT spawn a real OS subprocess (same rationale as the frozen `test_eval_
broken.py`/`test_pipeline_isolation.py`: deterministic, no torch-import-latency race) --
it injects a fake process whose exit looks clean, then monkeypatches the ONE completion-
path method (`_read_worker_result`) to raise, exercising `_finalize_round`'s new catch-all
exactly where RED_TEAM's real KeyError chain would have hit it. This is a NEW file (does
not edit the frozen `tests/eval/test_eval_broken.py`).
"""
from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.model import CnnArch, build_net


# ── shared fixtures (self-contained; deliberately duplicated from test_eval_broken.py's
#    own duplication of test_pipeline_isolation.py's harness -- house convention, see
#    those files' docstrings for why: independent oracle-write agents each keep a private
#    copy rather than a shared conftest.py) ─────────────────────────────────────────────
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
        round_timeout_sec=5.0, worker_kill_grace_sec=0.2, gate=gate, ladder=ladder,
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
        max_plies=128, run_id="oracle_test_run", spool_dir=spool_dir,
        ladder_state_path=tmp_path / "ladder_state.json", promotion=_promotion_hooks(tmp_path),
        # F-816-10 D-1: the GRID arm (`v6_live2_ls`), stated because there is no default.
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


class _FakeProcess:
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

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: "float | None" = None) -> None:
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
        self.last_process: "_FakeProcess | None" = None

    def Process(self, *, target=None, args=(), kwargs=None, daemon=None) -> _FakeProcess:
        proc = _FakeProcess(target=target, args=args, kwargs=kwargs, daemon=daemon)
        self.last_process = proc
        return proc


@pytest.fixture()
def fake_mp(monkeypatch):
    ctx = _FakeCtx()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name=None: ctx)
    return ctx


def _bounded(fn, *, timeout: float):
    """Test-level hard watchdog (mirrors test_eval_broken.py's `_bounded`): this test must
    itself never hang even if the fix under test regresses."""
    box: dict[str, Any] = {}

    def _run() -> None:
        box["value"] = fn()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"operation exceeded the {timeout}s test-level hard bound (must never hang)")
    return box.get("value")


# ── the injected exception: mirrors RED_TEAM's real KeyError, but via monkeypatch (per
#    this fix pass's explicit oracle instruction) rather than a real natural-activation
#    round, keeping the scenario deterministic and independent of worker/torch timing ────
class _InjectedCompletionError(RuntimeError):
    """A stand-in for the real `KeyError('sealbot_d5')` RED_TEAM reproduced deep inside
    `_success_result` -> `allocate_games` -- any exception class must be handled alike."""


def _inject_completion_crash(pipeline) -> None:
    def _boom(inflight, *, exit_code, wall_sec):
        raise _InjectedCompletionError("simulated round-completion crash (RED-TEAM F1 shape)")

    pipeline._read_worker_result = _boom  # noqa: SLF001 -- intentional, test-only injection


# ── scenario 1: the REAL background poller thread notices the crash on its own tick ─────
def test_poller_thread_survives_an_uncaught_exception_in_round_completion(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert ack["kicked"] is True
        _inject_completion_crash(pipeline)

        proc = fake_mp.last_process
        assert proc is not None
        proc.alive = False
        proc.exitcode = 0  # "clean exit" -> the poller's tick routes to _read_worker_result

        def _wait_for_result():
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                result = pipeline.poll_completed()
                if result is not None:
                    return result
                time.sleep(0.01)
            return None

        result = _bounded(_wait_for_result, timeout=6.0)

        # 1. a routed result WAS delivered (never a silent hang / dropped round).
        assert result is not None, "poll_completed() must eventually deliver a result, never hang forever"
        assert result["eval_broken_reason"] == "round_completion_error"
        assert result.get("promoted") is False
        assert result.get("wr_sealbot") is None
        assert "_InjectedCompletionError" in (result["eval_broken_detail"] or "")

        # 2. a named eval_broken event WAS emitted (never silent).
        broken = sink.named("eval_broken")
        assert broken, "no eval_broken event emitted for the injected completion crash"
        ev = broken[-1]
        assert ev.get("reason") == "round_completion_error"
        assert "_InjectedCompletionError" in ev.get("detail", "") or "simulated round-completion crash" in ev.get("detail", "")

        # 3. the poller THREAD itself is still alive -- it must never die silently.
        assert pipeline._poller.is_alive(), "the eval-pipeline-poller thread must survive an uncaught exception"

        # 4. the pipeline is still usable afterwards (in-flight cleared, not stuck forever).
        assert pipeline.poll_completed() is None  # mailbox drained, nothing left pending
    finally:
        pipeline.stop()


# ── scenario 2: drain_pending() (the synchronous teardown/mid-run join point) is equally
#    exception-proof -- the RED_TEAM severity amplifier explicitly calls out a crash
#    reachable via drain_pending/close_out, not only the poller's own tick ─────────────────
def test_drain_pending_survives_an_uncaught_exception_in_round_completion(fake_mp, tmp_path) -> None:
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        _inject_completion_crash(pipeline)

        proc = fake_mp.last_process
        proc.alive = False
        proc.exitcode = 0

        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)

        assert result is not None
        assert result["eval_broken_reason"] is not None
        assert result.get("promoted") is False

        broken = sink.named("eval_broken")
        assert broken and broken[-1].get("reason") == "round_completion_error"
        assert "exception_class" in broken[-1]
        assert broken[-1]["exception_class"] == "_InjectedCompletionError"
    finally:
        pipeline.stop()


def test_round_completion_error_never_silent_never_dropped(fake_mp, tmp_path) -> None:
    """Combined belt-and-suspenders assertion (mirrors the frozen suite's own `test_eval_
    broken_never_promotes_and_never_silently_skips`): a routed result WITH no event, or an
    event WITH no routed result, are each rejected."""
    sink = _SpySink()
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path, sink=sink), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        _inject_completion_crash(pipeline)
        proc = fake_mp.last_process
        proc.alive = False
        proc.exitcode = 0

        result = _bounded(lambda: pipeline.drain_pending(), timeout=5.0)
        got_routed_result = result is not None and result.get("promoted") is False
        got_event = bool(sink.named("eval_broken"))
        assert got_routed_result, "a round-completion crash must still route a result with promoted=False"
        assert got_event, "a round-completion crash must still emit eval_broken (never silent)"
    finally:
        pipeline.stop()
