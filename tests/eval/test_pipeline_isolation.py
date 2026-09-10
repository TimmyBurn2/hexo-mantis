"""Eval pipeline isolation laws.

Isolation law 1: eval inference is NEVER in-process. `build_eval_pipeline` has NO
`device`/`model` constructor kwargs, the worker spawns under a `multiprocessing.get_context
("spawn")` context, and every `.join(` call in pipeline.py carries a `timeout=`. These tests
patch the STDLIB `multiprocessing.get_context` — the shared module object, so the patch holds
whichever way pipeline.py imports the name — with a `_FakeProcess` that never spawns, so
kick-latency assertions are deterministic.

The model passed to `run_evaluation` carries its declared `arch` dataclass as a plain `.arch`
attribute. Nothing here asserts on the snapshot's payload shape, only on where the file lands
and that it carries no checkpoint-envelope keys.

>300 justify: one isolation-law seam (kick/ack, no-module-retained, spawn-context,
join-boundedness, snapshot-vs-checkpoint) sharing one fake-process/fake-context harness and one
minimal-config builder — splitting by behavior would duplicate that harness and let the
isolation-law halves drift out of sync.
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
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net

_GSPEC = lookup("gnn_axis_v1")

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"
_SRC_EVAL = _SRC / "eval"
_SRC_ARENA = _SRC / "arena"
_TORCH_FREE_EVAL_MODULES = ("pipeline.py", "ladder.py", "bt.py", "aggregate.py", "rounds.py", "errors.py")


def _tiny_model() -> torch.nn.Module:
    arch = GnnArch(in_dim=int(_GSPEC.node_feat_dim), edge_dim=int(_GSPEC.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch  # the declared arch travels with the model
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
        random_model_sims=96, sealbot_model_sims=128, random_floor_games=4, worker_device="cpu",
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
        max_plies=128,
        c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16,
        run_id="oracle_test_run",
        spool_dir=spool_dir, game_record_dir=str(spool_dir) + "_games",
        ladder_state_path=tmp_path / "ladder_state.json",
        promotion=_promotion_hooks(tmp_path),
        # The parent resolves the fused-forward memory bound ONCE and carries it to every
        # `RoundSpec`; `None` is the GRID arm, which `v6_live2_ls` here takes.
        fused_graph_caps=None,
        inference_batching=None,
    )
    kwargs.update(overrides)
    return kwargs


class _FakeProcess:
    """Stands in for `multiprocessing.context.Process`: `.start()` spawns and runs nothing."""

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
        assert "wr_sealbot" not in ack   # the kick ack NEVER carries WR
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
                # a checkpoint ENVELOPE carries provenance keys a spool snapshot must not.
                assert "envelope_version" not in payload
                assert "checkpoint_stamp" not in payload
    finally:
        pipeline.stop()


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
