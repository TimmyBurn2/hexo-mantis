"""The eval suites' shared fake-subprocess rig: tiny net, config builders, fake process tree.

Every member drives the REAL `build_eval_pipeline` against a fake `multiprocessing` context;
the knobs a suite exists to set (round timeout, kill grace, drain caps, run id) stay at each
caller's site as overrides, never baked into a second copy of the rig.
"""
from __future__ import annotations

import math
import multiprocessing
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mantis._engine import Board
from _fused_caps import CAPS
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.config.schema import EvalConfig, GateConfig
from mantis.encoding import lookup
from mantis.eval.pipeline import DrainCaps
from mantis.eval.promote import DeployTagHooks
from mantis.model import GnnArch, build_net
from mantis.selfplay.inference_local import LocalInferenceEngine

GSPEC = lookup("gnn_axis_v1")
#: `LocalInferenceEngine` builds its `InferenceServer` config with no `RunConfig`, so the
#: fused-forward memory bound is a REQUIRED keyword threaded as a spec.


def tiny_model() -> torch.nn.Module:
    arch = GnnArch(in_dim=int(GSPEC.node_feat_dim), edge_dim=int(GSPEC.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    return net


def eval_config(**overrides: Any) -> EvalConfig:
    gate = GateConfig(
        stride=1, screen_games=80, confirm_games=128, promotion_winrate=0.55,
        screen_confirm_lo=0.44, deploy_sims=150, opening_book="book_v1_s20260625_p4",
        bootstrap_resamples=1000, min_distinct_per_pair=10, seed_base=20260625, sequential=None,
    )
    defaults = dict(
        random_model_sims=96, max_plies=128, random_floor_games=4, worker_device="cpu",
        round_timeout_sec=5.0, worker_kill_grace_sec=0.2, gate=gate,
        ply_cap_adjudication=None, strength_floor=None,
    )
    defaults.update(overrides)
    return EvalConfig(**defaults)


def promotion_hooks(tmp_path: Path, *, run_id: str = "oracle_test_run") -> DeployTagHooks:
    return DeployTagHooks(
        anchor_state=SimpleNamespace(best_model=None, best_model_step=None),
        best_model_path=tmp_path / "best_model.pt",
        run_id=run_id,
        encoding="gnn_axis_v1",
        save_anchor=lambda *a, **k: None,
        guarded_load=lambda *a, **k: None,
    )


def pipeline_kwargs(
    tmp_path: Path,
    *,
    eval_cfg: EvalConfig | None = None,
    run_id: str = "oracle_test_run",
    drain_caps_sec: float = 2.0,
    **overrides: Any,
) -> dict:
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        eval_cfg=eval_cfg if eval_cfg is not None else eval_config(),
        coordinator_cfg_caps=DrainCaps(
            final_eval_drain_timeout_sec=drain_caps_sec,
            eval_final_drain_safety_factor=1.0,
            eval_final_drain_hard_cap_sec=drain_caps_sec,
            terminal_eval_hard_cap_sec=drain_caps_sec,
        ),
        encoding="gnn_axis_v1",
        max_plies=128,
        c_visit=50.0, c_scale=1.0, q_rescale=True, search_kind="puct", gumbel_m=16,
        run_id=run_id,
        spool_dir=spool_dir, game_record_dir=str(spool_dir) + "_games",
        promotion=promotion_hooks(tmp_path, run_id=run_id),
        fused_graph_caps=CAPS,
        inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10),
    )
    kwargs.update(overrides)
    return kwargs


def seeded_net(seed: int):
    spec = lookup(GSPEC.name)
    torch.manual_seed(seed)
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)
    net = build_net(arch)
    net.arch = arch
    net.eval()
    return net


def caps_for(enc_name: str):
    """The fused-forward memory bound this encoding's route needs, derived from the encoding:
    the graph route resolves the bound EAGERLY when its `InferenceServer` is constructed, and a
    non-graph route never reads it."""
    if lookup(enc_name).representation != "graph":
        return None
    return CAPS


def rule_logit(i: int) -> float:
    """The fixture's `logit_rule`, over the BUILDER's per-graph legal-node index."""
    return ((i * 37) % 101) / 20.0


class RuleNet(torch.nn.Module):
    """`GnnNet.forward_batch`'s contract with a deterministic policy head — the ONE stand-in.

    Cross-language byte-parity needs determinism; the graph loop, `collate_graph_batch`,
    `segment_softmax`, `assemble_ls_from_gnn_probs` and the expand are all production. The legal
    rows of each graph are contiguous and in builder order, which is the order `legal_offsets`
    segments and `assemble` zips against.
    """

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        n_graphs = int(node_offsets.shape[0]) - 1
        logits: list[float] = []
        for g in range(n_graphs):
            lo, hi = int(node_offsets[g]), int(node_offsets[g + 1])
            # `legal_index` is the wire's `legal_node_gather`: ROWS of legal nodes (strictly
            # ascending, hence unique), so counting entries in `[lo, hi)` equals summing a mask's bits.
            n_legal = int(((legal_index >= lo) & (legal_index < hi)).sum().item())
            logits.extend(rule_logit(i) for i in range(n_legal))
        return (
            torch.tensor(logits, dtype=torch.float32),
            torch.zeros((n_graphs, 1), dtype=torch.float32),
            torch.zeros((n_graphs, 65), dtype=torch.float32),
        )


@pytest.fixture()
def graph_engine():
    """A REAL `LocalInferenceEngine` on the graph spec, driving the production graph seam."""
    spec = GSPEC
    net = RuleNet()
    net.eval()
    engine = LocalInferenceEngine(net, torch.device("cpu"), encoding_spec=spec,
                                  fused_graph_caps=CAPS,
                                  inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8,
                                  )
    try:
        yield engine, spec
    finally:
        engine.close()


def board_from(pos: dict) -> Board:
    """Replay the recorded move sequence — the identical construction the Rust leg performs."""
    board = Board.with_encoding_name(GSPEC.name)
    flat = pos["moves"]
    for i in range(0, len(flat), 2):
        board.apply_move(flat[i], flat[i + 1])
    return board


class FakeProcess:
    """A spawn-context child stand-in whose `alive`/`exitcode` the test drives directly.

    `join()` reproduces the real `multiprocessing.Process` behaviour on a non-finite timeout
    (an `OverflowError`) and records every timeout it was called with, so a suite can assert
    the value that reached `.join()` was bounded BEFORE the call.
    """

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
        self.join_calls: list[float | None] = []

    def start(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        if timeout is not None and not math.isfinite(timeout):
            raise OverflowError("cannot convert float infinity to integer")

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


class FakeCtx:
    def __init__(self) -> None:
        self.last_process: FakeProcess | None = None
        self.process_calls: list[dict] = []
        #: the `name` `multiprocessing.get_context` was called with, for a suite asserting
        #: WHICH context string reached it (the check keys off the name, not the object).
        self.requested_name: str | None = None

    def Process(self, *, target=None, args=(), kwargs=None, daemon=None) -> FakeProcess:
        proc = FakeProcess(target=target, args=args, kwargs=kwargs, daemon=daemon)
        self.process_calls.append({"target": target, "args": args, "kwargs": kwargs})
        self.last_process = proc
        return proc


@pytest.fixture()
def fake_mp(monkeypatch: pytest.MonkeyPatch) -> FakeCtx:
    ctx = FakeCtx()

    def _get_context(name: str | None = None) -> FakeCtx:
        ctx.requested_name = name
        return ctx

    monkeypatch.setattr(multiprocessing, "get_context", _get_context)
    return ctx


def bounded(fn, *, timeout: float):
    """Test-level hard watchdog: the call must never hang even if the fix under test regresses."""
    box: dict[str, Any] = {}

    def _run() -> None:
        box["value"] = fn()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"operation exceeded the {timeout}s test-level hard bound (must never hang)")
    return box.get("value")


class _InjectedCompletionError(RuntimeError):
    """A stand-in for any uncaught exception deep inside the round-completion path."""
