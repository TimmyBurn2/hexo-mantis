"""⊕ WP12-R Phase A / O-A9 arm (b) (DESIGN_A §1.8/§5, PREREG_A §1) — the rung seat.

R138's third oracle already exists, is green and is FROZEN:
`tests/eval/test_eval_selfplay_child_parity.py::test_head_plays_an_off_window_move_against_
random_bot` (`:404`). Phase A does not re-write it — DESIGN_A §6.2 makes that a design
constraint, not a preference, because the file is frozen at
`ORACLE_FREEZE_EVALDECODE.sha256:2`. Phase A's obligation is narrower: re-run it and record
the result, and extend the claim to the seat R147 turns into a production rung.

**Why the rung seat is not the seat the frozen row measures.** The frozen row builds the
player directly at `n_sims=1`. `_play_rung_block` (`worker.py:240-262`) builds it at
`_model_sims_for_kind(spec, rung_job.bot)` — a DIFFERENT sims authority, wired by M-3
precisely so the rung plays at the per-kind value and never at `gate.deploy_sims`. A window
confinement that survived at the gate seat and re-appeared at the rung seat would leave the
frozen row green and every sealbot number wrong: the head would answer only inside the
361-cell window against an opponent (SealBot, and RandomBot at the floor) that samples the
full legal set. This is the seat where the ladder asymmetry would actually be paid for.

MUTATION (M-A15, transient): make `_build_candidate_player` return the grid arm for a graph
spec. **The kill arrives as an ERROR, not as the named assertion** — R144's known chain
(`GnnNet` has no `forward`) makes `select_move` raise before the assertion is reached — so
the cell is labelled `[reached, error-mode]` and NOT "RED via `assert off_window_moves`".
PREREG_A §3 registers an assertion-mode alternative if IMPL wants one.

The stub net is the ONE stand-in: `LocalInferenceEngine`, the graph decode, the expand and
`DeployHeadPlayer` are all production, and the net is a stand-in only because the assertion
needs determinism. It is duplicated from the frozen file rather than imported — R5 bars
cross-test imports, and §6.2 requires this arm to land in a NEW file.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mantis._engine import Board
from mantis.bots.random_bot import RandomBot
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.encoding import lookup
from mantis.eval import worker
from mantis.selfplay.inference_local import LocalInferenceEngine

_ENC = "gnn_axis_v1"
#: F-816-10 D-1: `LocalInferenceEngine` takes the fused-forward memory bound as a REQUIRED
#: keyword — it hand-builds its `InferenceServer` config with no `RunConfig`, so the spec is
#: THREADED from a parent resolver and never hardcoded at the site. Non-binding by
#: construction here: nothing in this file exercises a split.
_CAPS = FusedGraphCapsSpec(max_fused_edges=57149441, max_fused_nodes=1785921)
_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "eval_selfplay_parity" / "dispersed_r6_v1.json"
)

#: `policy_logit_count` is 362 and `Board.to_flat` returns a sentinel above the window, so
#: `to_flat(q, r) >= 361` is exactly "off-window" — the same test the Rust leg applies.
_OFF_WINDOW_FLAT = 361

#: The two dispersed positions with the largest off-window child sets (measured from the
#: fixture at ORACLE-WRITE: 134 and 145 expected off-window children). Chosen because a
#: position with few off-window options makes an absent off-window move ambiguous between
#: "the head is confined" and "the head preferred an in-window cell".
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
            # `legal_index` is the wire's `legal_node_gather` (R284 P-MASK): the ROWS of the
            # legal nodes, not a dense mask. Counting index entries that fall in this graph's
            # `[lo, hi)` row range is the same count as summing the mask's bits over it, for
            # every payload the contract admits — the gather is strictly ascending, hence
            # unique (wire check 13). The stub's OUTPUT is unchanged; nothing it asserts moves.
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
                                  fused_graph_caps=_CAPS)
    try:
        yield engine, spec
    finally:
        engine.close()


def _position(index: int) -> dict:
    """Re-nest the FLAT fixture (`p0_*`, `p1_*`, ...). A missing key is a KeyError, never a
    default — the fixture either carries the position or it does not."""
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
    """The four per-kind sims fields `_model_sims_for_kind` reads off a `RoundSpec`.

    `sealbot_model_sims` is **1** here and not run5's 128: the subject is which AUTHORITY
    the rung seat reads, not how deep it searches, and 128 sims per half-ply over 8 plies is
    a benchmark, not an oracle. `gate.deploy_sims` is set to a DIFFERENT value so a seat
    that silently read the gate's authority would be visible rather than coincidental.
    """
    return SimpleNamespace(
        sealbot_model_sims=1, kraken_model_sims=2, strix_model_sims=3, random_model_sims=4,
        gate=SimpleNamespace(deploy_sims=150),
    )


@pytest.mark.parametrize("position_index", _POSITIONS)
def test_rung_seat_head_plays_an_off_window_move_against_a_full_legal_set_opponent(
    graph_engine, position_index: int
) -> None:
    """O-A9 arm (b). The head is built exactly as `_play_rung_block` builds it — through
    `_model_sims_for_kind`, not at a hand-picked `n_sims` — and must answer OUTSIDE the
    361-cell window against an opponent that samples the full legal set."""
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
    player = worker._build_candidate_player(engine, rung_sims, spec=spec, leaf_batch_size=1)
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
