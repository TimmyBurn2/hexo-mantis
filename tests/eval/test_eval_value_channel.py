"""Prove the eval head's VALUE channel crosses the graph decode seam and moves its choice.

Killer: zeroing the decoded values in `mantis.eval.worker._graph_expand_fn`, which reds none of
the frozen parity bank because every one of its production-entrance oracles runs at `n_sims=1`,
where the value channel is provably inert, and its stub net's value head is a constant.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis._engine import Board
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.selfplay.inference_local import LocalInferenceEngine

_ENC = "gnn_axis_v1"
#: The fused-forward memory bound is a required keyword with no `RunConfig` to resolve it
#: from. Sized non-binding by construction: nothing here exercises a split.
_CAPS = FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)
_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "eval_selfplay_parity" / "dispersed_r6_v1.json"
)

#: 32 rather than production's 150: measured, n_sims=1 yields one distinct root q at 4/4
#: positions and n_sims=32 yields 2-4, the killing mutation reds both flips at every n_sims
#: >= 2, and 150 costs ~13.5 s against ~3.0 s here for no added detection.
_SIMS = 32

#: With every child at `visits=0, q=0` PUCT's Q term is identically 0, so the move reduces to
#: `argmax prior` and no value the net produced can reach it.
_INERT_SIMS = 1


def _positions() -> list[dict]:
    """Re-nest the committed dispersed fixture from its flat form."""
    fx = json.loads(_FIXTURE.read_text())
    out = []
    for i in range(fx["n_positions"]):
        prefix = f"p{i}_"
        out.append({k[len(prefix):]: v for k, v in fx.items() if k.startswith(prefix)})
    return out


def _board(pos: dict) -> Board:
    board = Board.with_encoding_name(_ENC)
    flat = pos["moves"]
    for i in range(0, len(flat), 2):
        board.apply_move(flat[i], flat[i + 1])
    return board


class _ValueVisibleNet(torch.nn.Module):
    """Serve `forward_batch`'s contract with the bank's policy rule and a position-dependent
    value head, so `sign` flips every value without touching a single prior."""

    def __init__(self, sign: float) -> None:
        super().__init__()
        self.sign = float(sign)

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        n_graphs = int(node_offsets.shape[0]) - 1
        logits: list[float] = []
        values: list[float] = []
        for g in range(n_graphs):
            lo, hi = int(node_offsets[g]), int(node_offsets[g + 1])
            # `legal_index` carries the ROWS of the legal nodes, not a dense mask; the gather
            # is strictly ascending, so counting entries in `[lo, hi)` equals summing a mask.
            n_legal = int(((legal_index >= lo) & (legal_index < hi)).sum().item())
            logits.extend(((i * 37) % 101) / 20.0 for i in range(n_legal))
            key = (hi - lo) * 31 + n_legal
            values.append(self.sign * (((key * 7919) % 2001) - 1000) / 1000.0)
        return (
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(values, dtype=torch.float32).reshape(n_graphs, 1),
            torch.zeros((n_graphs, 65), dtype=torch.float32),
        )


def _engine(sign: float) -> LocalInferenceEngine:
    net = _ValueVisibleNet(sign)
    net.eval()
    return LocalInferenceEngine(net, torch.device("cpu"), encoding_spec=lookup(_ENC),
                                fused_graph_caps=_CAPS,
                                inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8, )


@pytest.fixture
def value_visible_engines():
    """Yield two real engines differing only in the sign of the value head."""
    spec = lookup(_ENC)
    positive, negative = _engine(1.0), _engine(-1.0)
    try:
        yield positive, negative, spec
    finally:
        positive.close()
        negative.close()


def _search(engine, spec, board, n_sims):
    """Drive the production entrance and return the played move and root child q values."""
    player = worker.build_candidate_player(engine, n_sims, spec=spec, leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, gumbel_seed=0)
    player.new_game()
    move = player.select_move(board)
    tree = player._tree
    assert tree is not None, "the deploy head must retain its tree after select_move"
    return move, [float(q) for _coord, _idx, _prior, _visits, q in tree.get_root_children_info()]


def test_value_channel_reaches_the_tree_at_production_sims(value_visible_engines) -> None:
    """Prove per-leaf values cross the FFI and back up, not just the policy half.

    A non-degenerate root-child q vector cannot be true otherwise.
    """
    positive, _negative, spec = value_visible_engines
    for pos in _positions():
        _move, qs = _search(positive, spec, _board(pos), _SIMS)
        distinct = {round(q, 9) for q in qs}
        assert len(distinct) >= 2, (
            f"{pos['id']}: every root child reads q={distinct} after {_SIMS} sims — the "
            f"value channel never reached the tree"
        )


def test_flipping_the_value_head_moves_the_heads_choice(value_visible_engines) -> None:
    """Prove the value channel is load-bearing on the move played, not merely present.

    The two engines share a byte-identical policy head, so a changed move is attributable to
    the value channel and to nothing else.
    """
    positive, negative, spec = value_visible_engines
    agreed = []
    for pos in _positions():
        board = _board(pos)
        move_pos, _q = _search(positive, spec, board, _SIMS)
        move_neg, _q = _search(negative, spec, _board(pos), _SIMS)
        if move_pos == move_neg:
            agreed.append((pos["id"], move_pos))
    assert not agreed, (
        f"sign-flipping the value head left the played move unchanged at {len(agreed)} of 4 "
        f"positions {agreed} — the value channel is not load-bearing on the head's choice"
    )


def test_the_value_channel_is_inert_at_one_simulation(value_visible_engines) -> None:
    """Pin the frozen bank's blindness: at one simulation the value channel cannot move the
    played move, so a value oracle parameterized there covers nothing.

    A control, not a flip — the killing mutation leaves this green by design. It reds instead if
    the head's PUCT drive ever grows a value-sensitive term live at one simulation.
    """
    positive, negative, spec = value_visible_engines
    for pos in _positions():
        board = _board(pos)
        move_pos, qs = _search(positive, spec, board, _INERT_SIMS)
        move_neg, _q = _search(negative, spec, _board(pos), _INERT_SIMS)
        assert {round(q, 9) for q in qs} == {0.0}, (
            f"{pos['id']}: root q is not uniformly 0 after one simulation"
        )
        assert move_pos == move_neg, (
            f"{pos['id']}: the played move changed with the value sign at one simulation"
        )
