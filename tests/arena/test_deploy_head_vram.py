"""Oracle test for VRAM cache release at move boundaries (Phase V, VERDICT-A).

Producer test (LAW-07): verifies `DeployHeadPlayer.select_move` calls
`torch.cuda.empty_cache()` exactly once after each move's MCTS completes, on
BOTH the grid (infer_fn) and graph (expand_fn) arms. Mutation: remove the
`_release_cuda_cache()` call from `select_move` → `test_*_fires_per_move`
reds; hoist it into the per-simulation loop → `== 1` assertion reds.

The V-0 measurement (CARD_VRAM_ACCUMULATION) proved the CUDA caching allocator
accumulates variable-size graph batches monotonically (0 → 8450 MiB reserved
in ~15 min, allocated flat at 25–42 MiB). The fix is an unconditional
`empty_cache` at the move boundary — this test pins its presence and count
on both decode arms.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from mantis._engine import Board  # noqa: E402
from mantis.arena.deploy_head import DeployHeadPlayer  # noqa: E402


def _stub_infer_fn(_board):
    return [0.0] * 362, 0.0


def _stub_expand_fn(tree, leaves):
    policies = [[0.0] * 362 for _ in leaves]
    values = [0.0] * len(leaves)
    tree.expand_and_backup(policies, values)


def test_grid_arm_empty_cache_fires_once_per_move(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls: list = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))

    player = DeployHeadPlayer(infer_fn=_stub_infer_fn, n_sims=2)
    player.new_game()
    board = Board.with_encoding_name("v6_live2_ls")
    player.select_move(board)

    assert len(calls) == 1, (
        "DeployHeadPlayer.select_move must call torch.cuda.empty_cache() exactly "
        f"once per move on the grid (infer_fn) arm — got {len(calls)} calls. "
        "More than 1 means the call was hoisted into the per-simulation loop "
        "(sync cost multiplied by n_sims); 0 means the fix is absent."
    )


def test_graph_arm_empty_cache_fires_once_per_move(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls: list = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))

    player = DeployHeadPlayer(expand_fn=_stub_expand_fn, n_sims=2)
    player.new_game()
    board = Board.with_encoding_name("v6_live2_ls")
    player.select_move(board)

    assert len(calls) == 1, (
        "DeployHeadPlayer.select_move must call torch.cuda.empty_cache() exactly "
        f"once per move on the graph (expand_fn) arm — got {len(calls)} calls. "
        "The graph arm is the VERDICT-A trigger (variable-size graph batches); "
        "a call missing here is the production regression this test exists to catch."
    )


def test_empty_cache_skipped_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls: list = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))

    player = DeployHeadPlayer(infer_fn=_stub_infer_fn, n_sims=2)
    player.new_game()
    board = Board.with_encoding_name("v6_live2_ls")
    player.select_move(board)

    assert len(calls) == 0, (
        "empty_cache must NOT fire when CUDA is unavailable — the is_available() "
        "guard makes the CPU path a no-op (R231 fallback: eval.worker_device: cpu)."
    )
