"""⊕ WP11-A — eval pipeline isolation laws (mantis.eval.pipeline; design §a.3/§c.3).

RED-at-import until IMPL writes `mantis.eval.pipeline` (+ `mantis.eval.promote`,
`mantis.eval.snapshot`). ORACLE-FIRST: the top-level imports raise ModuleNotFoundError
before any port code exists.

Isolation law 1 (non-negotiable, run3 45h livelock heritage): eval inference is NEVER
in-process. `build_eval_pipeline` has NO `device`/`model` constructor kwargs — the type
surface itself cannot express an in-process CUDA path. The worker subprocess is spawned
under a `multiprocessing.get_context("spawn")` context (own CUDA context); every
`.join(` call in pipeline.py carries a `timeout=`. These tests patch the STDLIB
`multiprocessing.get_context` (not a `mantis.eval.pipeline`-qualified name — the patch
targets the shared module object, so it works regardless of how pipeline.py imports the
name, as long as it does the ordinary `import multiprocessing; multiprocessing.get_context(...)`
attribute-lookup idiom) with a `_FakeProcess` that never really spawns an OS process, so
kick-latency assertions are deterministic and fast regardless of what a REAL worker would
do (torch import, model load, ...).

IMPL API pin introduced by this oracle (design leaves this as a convention, not
pseudocode): the model object passed to `run_evaluation` carries its declared `arch`
dataclass as a plain `.arch` attribute — mirrors the established `trainer.arch` /
`InfModelArch` convention already in this tree (`mantis.train.subsystems.
build_inference_model`; `HexTacToeNet`/`GnnNet` themselves do NOT store `.arch`, so
something upstream of the model must carry it, and `.arch` on the model instance is this
suite's concrete choice for that seam). If IMPL threads arch through differently (e.g. an
explicit `write_model_snapshot(model, path, arch=...)` kwarg), that is a narrow interface
mismatch to flag at IMPL, not a design contradiction — no test here asserts on the
snapshot's internal payload shape, only on where the file lands and that it carries no
checkpoint-envelope keys.

>300 justify: one isolation-law seam (kick/ack, no-module-retained, spawn-context,
join-boundedness, snapshot-vs-checkpoint) sharing one fake-process/fake-context harness
and one minimal-config builder — splitting by behavior would duplicate that harness
across files and let the isolation-law halves drift out of sync with each other, which is
the exact run3-livelock-class risk this suite exists to pin.
"""
from __future__ import annotations

import ast
import multiprocessing
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from mantis.config.schema import EvalConfig, GateConfig, LadderConfig, LadderRung
from mantis.eval.pipeline import DrainCaps, build_eval_pipeline
from mantis.eval.promote import DeployTagHooks
from mantis.model import CnnArch, build_net

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"
_SRC_EVAL = _SRC / "eval"
_SRC_ARENA = _SRC / "arena"
_TORCH_FREE_EVAL_MODULES = ("pipeline.py", "ladder.py", "bt.py", "aggregate.py", "rounds.py", "errors.py")


# ── shared fixtures (self-contained; no conftest.py — avoids a collision with the
#    sibling oracle-write agent's files landing in the same tests/eval/ directory) ──────
def _tiny_model() -> torch.nn.Module:
    arch = CnnArch(board_size=5, in_channels=4, filters=8, res_blocks=1)
    net = build_net(arch)
    net.arch = arch  # see module docstring: the declared-arch-travels-with-the-model convention
    return net


def _eval_cfg() -> EvalConfig:
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
    return EvalConfig(
        random_model_sims=96, sealbot_model_sims=128, kraken_model_sims=128,
        strix_model_sims=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=5.0, worker_kill_grace_sec=1.0, gate=gate, ladder=ladder,
        ply_cap_adjudication=None, strength_floor=None,
    )


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


def _pipeline_kwargs(tmp_path: Path, **overrides: Any) -> dict:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(exist_ok=True)
    kwargs = dict(
        eval_cfg=_eval_cfg(),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=5.0,
            eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=5.0,
            terminal_eval_hard_cap_sec=5.0,
        ),
        encoding="v6_live2_ls",
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


class _FakeProcess:
    """Stands in for `multiprocessing.context.Process`: `.start()` never actually spawns
    or runs any target — models "a stub worker" without incurring real subprocess/torch
    import overhead, so kick-latency assertions are deterministic."""

    def __init__(self, *, target=None, args=(), kwargs=None, daemon=None) -> None:
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon
        self.pid = 4242
        self.exitcode: int | None = None
        self._alive = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self._alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self._alive = False
        self.exitcode = -9


class _FakeCtx:
    def __init__(self) -> None:
        self.process_calls: list[dict] = []

    def Process(self, *, target=None, args=(), kwargs=None, daemon=None) -> _FakeProcess:
        self.process_calls.append({"target": target, "args": args, "kwargs": kwargs})
        return _FakeProcess(target=target, args=args, kwargs=kwargs, daemon=daemon)


@pytest.fixture()
def fake_mp(monkeypatch):
    requested: dict = {}
    ctx = _FakeCtx()

    def _fake_get_context(name: str | None = None):
        requested["name"] = name
        return ctx

    monkeypatch.setattr(multiprocessing, "get_context", _fake_get_context)
    return requested, ctx


# ── kick / ack plumbing ──────────────────────────────────────────────────────────────
def test_kick_returns_ack_immediately_and_never_blocks(fake_mp, tmp_path) -> None:
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        t0 = time.perf_counter()
        ack = pipeline.run_evaluation(
            _tiny_model(), 1000, None, full_config={}, best_model_step=None
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1, f"kick took {elapsed:.3f}s (must be a non-blocking ack, <100ms)"
        assert ack["kicked"] is True
        assert {"kicked", "round_id", "step", "reason"} <= set(ack)
        assert "wr_sealbot" not in ack   # P-06 heritage: the kick ack NEVER carries WR
    finally:
        pipeline.stop()


def test_builder_refuses_device_and_model_arguments(tmp_path) -> None:
    with pytest.raises(TypeError):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path), device="cpu", leaf_batch_size=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path), model=_tiny_model(), leaf_batch_size=1)  # type: ignore[call-arg]


def test_pipeline_retains_no_module_after_kick(fake_mp, tmp_path) -> None:
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(
            _tiny_model(), 1000, None, full_config={}, best_model_step=None
        )
        assert ack["kicked"] is True
        for name in dir(pipeline):
            if name.startswith("__"):
                continue
            try:
                value = getattr(pipeline, name)
            except Exception:
                continue
            assert not isinstance(value, torch.nn.Module), (
                f"pipeline.{name} retains a live torch.nn.Module after kick "
                "(snapshot-and-drop violated)"
            )
    finally:
        pipeline.stop()


def test_worker_spawned_with_spawn_context(fake_mp, tmp_path) -> None:
    requested, ctx = fake_mp
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert requested.get("name") == "spawn"
        assert ctx.process_calls, "no subprocess was ever requested via the spawn context"
    finally:
        pipeline.stop()


def test_snapshots_are_not_checkpoints(fake_mp, tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    kwargs = _pipeline_kwargs(tmp_path)
    pipeline = build_eval_pipeline(**kwargs, leaf_batch_size=1)
    try:
        pipeline.run_evaluation(_tiny_model(), 1000, None, full_config={}, best_model_step=None)
        spool_dir = Path(kwargs["spool_dir"])
        snapshot_files = [p for p in spool_dir.rglob("*") if p.is_file()]
        assert snapshot_files, "no snapshot file was written under spool_dir during kick"
        for p in snapshot_files:
            assert checkpoint_dir.resolve() not in p.resolve().parents
            payload = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(payload, dict):
                # a checkpoint ENVELOPE (WP10) carries provenance keys a spool snapshot
                # must never carry — the LAW-12 one-loader carve-out this test pins.
                assert "envelope_version" not in payload
                assert "checkpoint_stamp" not in payload
    finally:
        pipeline.stop()


# ── source-level census (isolation laws structural pins) ────────────────────────────────
def test_parent_side_eval_modules_have_no_inference_surface() -> None:
    banned: list[str] = []
    files = sorted(_SRC_EVAL.glob("*.py")) + sorted(_SRC_ARENA.glob("*.py"))
    for path in files:
        if path.name == "worker.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("cuda", "autocast"):
                banned.append(f"{path.name}:{node.lineno} .{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "forward"
            ):
                banned.append(f"{path.name}:{node.lineno} forward(")
    assert not banned, f"inference surface found on the parent side: {banned}"

    for name in _TORCH_FREE_EVAL_MODULES:
        path = _SRC_EVAL / name
        tree = ast.parse(path.read_text(), filename=str(path))
        torch_imports = [
            node
            for node in ast.walk(tree)
            if (isinstance(node, ast.Import) and any(a.name.split(".")[0] == "torch" for a in node.names))
            or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch")
        ]
        assert not torch_imports, (
            f"{name} imports torch — only snapshot.py may (write-side torch.save/load only)"
        )


def test_every_join_is_timeout_bounded() -> None:
    source = (_SRC_EVAL / "pipeline.py").read_text()
    tree = ast.parse(source, filename="pipeline.py")
    bare_joins: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            has_timeout_kw = any(kw.arg == "timeout" for kw in node.keywords)
            has_positional = len(node.args) >= 1
            if not (has_timeout_kw or has_positional):
                bare_joins.append(node.lineno)
    assert not bare_joins, f"pipeline.py has bare .join() calls with no timeout at lines {bare_joins}"
