"""The head answers OUTSIDE the 361-cell window from the RUNG seat, not just the gate seat.

The rung builds its player at `_model_sims_for_kind(spec, bot)` — a DIFFERENT sims authority
from `gate.deploy_sims` — so a window confinement re-appearing at this seat would leave the
frozen gate-seat oracle green while every sealbot rung number measured that asymmetry instead
of strength.

The stub net is the ONE stand-in and is duplicated rather than imported, since cross-test
imports are barred; everything else on the path is production.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis._engine import Board
from mantis.bots.random_bot import RandomBot
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.selfplay.inference_local import LocalInferenceEngine

_ENC = "gnn_axis_v1"
#: `LocalInferenceEngine` takes the fused-forward memory bound as a REQUIRED keyword. The pair
#: here is non-binding by construction: nothing in this file exercises a split.
_CAPS = FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)
_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "eval_selfplay_parity" / "dispersed_r6_v1.json"
)

#: `policy_logit_count` is 362 and `Board.to_flat` returns a sentinel above the window, so
#: `to_flat(q, r) >= 361` is exactly "off-window" — the same test the Rust leg applies.
_OFF_WINDOW_FLAT = 361

#: The two dispersed positions with the largest off-window child sets (134 and 145 measured
#: from the fixture); fewer options would make an absent off-window move ambiguous.
_POSITIONS = (2, 3)


def _rule_logit(i: int) -> float:
    return ((i * 37) % 101) / 20.0


class _RuleNet(torch.nn.Module):
    """`GnnNet.forward_batch`'s contract with a deterministic policy head."""

    def forward_batch(self, x, edge_index, edge_attr, legal_index, stone_mask, node_offsets):
        n_graphs = int(node_offsets.shape[0]) - 1
        logits: list[float] = []
        for g in range(n_graphs):
            lo, hi = int(node_offsets[g]), int(node_offsets[g + 1])
            # `legal_index` gathers the ROWS of the legal nodes, not a dense mask; the gather is
            # strictly ascending, so counting entries in `[lo, hi)` equals summing mask bits.
            n_legal = int(((legal_index >= lo) & (legal_index < hi)).sum().item())
            logits.extend(_rule_logit(i) for i in range(n_legal))
        return (
            torch.tensor(logits, dtype=torch.float32),
            torch.zeros((n_graphs, 1), dtype=torch.float32),
            torch.zeros((n_graphs, 65), dtype=torch.float32),
        )


@pytest.fixture
def graph_engine():
    spec = lookup(_ENC)
    net = _RuleNet()
    net.eval()
    engine = LocalInferenceEngine(net, torch.device("cpu"), encoding_spec=spec,
                                  fused_graph_caps=_CAPS,
                                  inference_batching=InferenceBatchingSpec(inference_batch_size=64, inference_max_wait_ms=10), max_in_flight=8, )
    try:
        yield engine, spec
    finally:
        engine.close()


def _position(index: int) -> dict:
    """Re-nest the FLAT fixture (`p0_*`, `p1_*`, ...); a missing key raises, never defaults."""
    fx = json.loads(_FIXTURE.read_text())
    prefix = f"p{index}_"
    return {k[len(prefix):]: v for k, v in fx.items() if k.startswith(prefix)}


def _board(pos: dict) -> Board:
    board = Board.with_encoding_name(_ENC)
    flat = pos["moves"]
    for i in range(0, len(flat), 2):
        board.apply_move(flat[i], flat[i + 1])
    return board


def _rung_round_spec() -> SimpleNamespace:
    """The per-kind sims fields `_model_sims_for_kind` reads off a `RoundSpec`.

    The subject is which AUTHORITY the rung seat reads, not search depth, so `gate.deploy_sims`
    carries a DIFFERENT value: a seat reading it would be visible rather than coincidental.
    """
    return SimpleNamespace(
        sealbot_model_sims=1, random_model_sims=4,
        gate=SimpleNamespace(deploy_sims=150),
    )


@pytest.mark.parametrize("position_index", _POSITIONS)
def test_rung_seat_head_plays_an_off_window_move_against_a_full_legal_set_opponent(
    graph_engine, position_index: int
) -> None:
    """Build the head as `_play_rung_block` does and prove it answers outside the window."""
    engine, spec = graph_engine
    round_spec = _rung_round_spec()
    rung_sims = worker._model_sims_for_kind(round_spec, "sealbot")
    assert rung_sims == round_spec.sealbot_model_sims, (
        "the rung seat must read the PER-KIND sims authority (M-3); reading "
        f"gate.deploy_sims here would stamp a regime the rung did not play — got {rung_sims}"
    )

    pos = _position(position_index)
    board = _board(pos)
    head_seat = int(board.current_player)
    player = worker.build_candidate_player(engine, rung_sims, spec=spec, leaf_batch_size=1, c_visit=50.0, c_scale=1.0, search_kind="puct", gumbel_m=16, gumbel_seed=0)
    player.new_game()
    bot = RandomBot(seed=20260802)

    off_window_head_moves: list[tuple[int, int]] = []
    for _ply in range(8):
        if board.winner() is not None or not board.legal_moves():
            break
        if int(board.current_player) == head_seat:
            move = player.select_move(board)
            if board.to_flat(*move) >= _OFF_WINDOW_FLAT:
                off_window_head_moves.append(move)
        else:
            move = bot.select_move(board)
        board.apply_move(*move)

    assert off_window_head_moves, (
        f"{pos['id']}: the head played no off-window move in 8 plies from the RUNG seat, "
        f"with {pos['expected_off_window_children']} off-window children available. A "
        f"window-confined head cannot answer an opponent that samples the full legal set, "
        f"and every sealbot rung number would be measuring that asymmetry instead of strength."
    )
