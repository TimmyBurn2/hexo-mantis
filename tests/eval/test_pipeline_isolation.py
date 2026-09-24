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
"""
from __future__ import annotations

import ast
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from _pipeline_harness import FakeCtx, eval_config, pipeline_kwargs, tiny_model

from mantis.encoding import lookup, normalize_encoding_name
from mantis.eval.pipeline import build_eval_pipeline
from mantis.eval.rounds import RoundSpec
from mantis.config.schema import EvalConfig

_SRC = Path(__file__).resolve().parents[2] / "src" / "mantis"
_SRC_EVAL = _SRC / "eval"
_SRC_ARENA = _SRC / "arena"
_TORCH_FREE_EVAL_MODULES = ("pipeline.py", "aggregate.py", "rounds.py", "errors.py")


def _eval_cfg() -> EvalConfig:
    return eval_config(worker_kill_grace_sec=1.0)


def _pipeline_kwargs(tmp_path: Path, **overrides: Any) -> dict:
    return pipeline_kwargs(
        tmp_path, eval_cfg=_eval_cfg(), drain_caps_sec=5.0, **overrides
    )


@pytest.fixture()
def fake_mp(monkeypatch):
    requested: dict = {}
    ctx = FakeCtx()

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
            tiny_model(), 1000, None, full_config={}, best_model_step=None
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1, f"kick took {elapsed:.3f}s (must be a non-blocking ack, <100ms)"
        assert ack["kicked"] is True
        assert {"kicked", "round_id", "step", "reason"} <= set(ack)
    finally:
        pipeline.stop()


def test_builder_refuses_device_and_model_arguments(tmp_path) -> None:
    with pytest.raises(TypeError):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path), device="cpu", leaf_batch_size=1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        build_eval_pipeline(**_pipeline_kwargs(tmp_path), model=tiny_model(), leaf_batch_size=1)  # type: ignore[call-arg]


def test_pipeline_retains_no_module_after_kick(fake_mp, tmp_path) -> None:
    pipeline = build_eval_pipeline(**_pipeline_kwargs(tmp_path), leaf_batch_size=1)
    try:
        ack = pipeline.run_evaluation(
            tiny_model(), 1000, None, full_config={}, best_model_step=None
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
        pipeline.run_evaluation(tiny_model(), 1000, None, full_config={}, best_model_step=None)
        assert requested.get("name") == "spawn"
        assert ctx.process_calls, "no subprocess was ever requested via the spawn context"
        # The parent never resolves the encoding; the child's `lookup` on this spec is its validator.
        spec_path = Path(ctx.process_calls[0]["args"][0])
        carried = RoundSpec.from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
        assert carried.encoding == "gnn_axis_v1"
        assert lookup(normalize_encoding_name(carried.encoding)).representation == "graph"
        assert carried.fused_graph_caps is not None and carried.inference_batching is not None
    finally:
        pipeline.stop()


def test_snapshots_are_not_checkpoints(fake_mp, tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    kwargs = _pipeline_kwargs(tmp_path)
    pipeline = build_eval_pipeline(**kwargs, leaf_batch_size=1)
    try:
        pipeline.run_evaluation(tiny_model(), 1000, None, full_config={}, best_model_step=None)
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    source = (_SRC_EVAL / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="pipeline.py")
    bare_joins: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            has_timeout_kw = any(kw.arg == "timeout" for kw in node.keywords)
            has_positional = len(node.args) >= 1
            if not (has_timeout_kw or has_positional):
                bare_joins.append(node.lineno)
    assert not bare_joins, f"pipeline.py has bare .join() calls with no timeout at lines {bare_joins}"
