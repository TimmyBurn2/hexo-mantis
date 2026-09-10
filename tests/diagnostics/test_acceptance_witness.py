"""The BC acceptance witness's two repairs, each beside the mutation arm that reproduces the defect.

Defect 1: the control arm must be a seeded fixed baseline, not a fresh `build_net` draw.
Defect 2: stone colour comes from the engine, never from ply parity — player 1 takes the
first stone and colours then alternate in PAIRS.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mantis._engine import Board
from mantis.arena.books import Opening
from mantis.bots.resolve import resolve_bot
from mantis.config.resolve.eval_posture import StrengthFloorSpec
from mantis.diagnostics.acceptance_witness import (
    ArmSpec,
    WitnessArmError,
    measure_arm,
    play_arm,
    record_runs,
    replay_board,
    seeded_net,
    witness_regime,
)
from mantis.encoding import lookup
from mantis.eval.worker import build_candidate_player
from mantis.config.resolve.fused_graph_caps import FusedGraphCapsSpec
from mantis.config.resolve.inference_batching import InferenceBatchingSpec
from mantis.model import GnnArch, build_net
from mantis.model.identity import net_param_hash
from mantis.selfplay.inference_local import LocalInferenceEngine

#: A finished, REACHABLE game: player 1 completes a six along `(1, 0)` at ply 11. Cells follow
#: the engine's compound-turn order, which is what the ply-parity expression gets wrong.
_WIN_LINE: list[tuple[int, int]] = [
    (0, 0),          # ply 0      -> player  1
    (0, 4), (1, 4),  # plies 1,2  -> player -1 (adjacent: the loser's longest run is 2)
    (1, 0), (2, 0),  # plies 3,4  -> player  1
    (4, 4), (6, 4),  # plies 5,6  -> player -1
    (3, 0), (4, 0),  # plies 7,8  -> player  1
    (8, 4), (10, 4), # plies 9,10 -> player -1
    (5, 0),          # ply 11     -> player  1, and the six is complete
]
_ENCODING = "gnn_axis_v1"
_AXES = ((1, 0), (0, 1), (1, -1))


class _Record:
    """Hold the two `GameRecord` fields `record_runs` reads, and nothing else."""

    def __init__(self, moves: list[tuple[int, int]], candidate_color: int) -> None:
        self.moves = tuple(moves)
        self.colors = {"candidate": candidate_color, "opponent": -candidate_color}


def _parity_longest_run(moves: list[tuple[int, int]], player: int) -> int:
    """Reconstruct runs the original witness's way (ply parity), kept as the mutation arm."""
    stones = {m: (1 if (i // 2) % 2 == 0 else -1) for i, m in enumerate(moves)}
    own = {cell for cell, colour in stones.items() if colour == player}
    best = 0
    for q, r in own:
        for dq, dr in _AXES:
            if (q - dq, r - dr) in own:
                continue
            length, cell = 0, (q, r)
            while cell in own:
                length += 1
                cell = (cell[0] + dq, cell[1] + dr)
            best = max(best, length)
    return best


def _tiny_arch() -> GnnArch:
    # Registry-true `_ENCODING` dims, minimal width/depth for speed.
    spec = lookup(_ENCODING)
    return GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim),
                   hidden=8, num_layers=1, policy_hidden=8, value_hidden=8)


def _derived_opening() -> Opening:
    """Return a four-ply opening derived at `_ENCODING`'s own geometry, not drawn from the book.

    Book openings can need a replay radius the encoding under test does not have; deriving
    keeps the pairing legal at whatever radius the encoding declares.
    """
    board = Board.with_encoding_name(_ENCODING)
    moves: list[tuple[int, int]] = []
    for ply in range(4):
        legal = sorted(board.legal_moves())
        move = legal[(7 + ply * 3) % len(legal)]
        board.apply_move(*move)
        moves.append(move)
    return Opening(opening_id="derived-at-encoding-radius", moves=moves)


def _readout(seed: int) -> dict:
    """Run one full witness pass for a seeded control arm, on CPU."""
    spec = lookup(_ENCODING)
    device = torch.device("cpu")
    engine = LocalInferenceEngine(
        seeded_net(_tiny_arch(), seed=seed).to(device).eval(), device, encoding_spec=spec,
        fused_graph_caps=FusedGraphCapsSpec(max_fused_edges=57149441,
                                           max_fused_nodes=1785921),
        inference_batching=InferenceBatchingSpec(inference_batch_size=64,
                                                inference_max_wait_ms=10),
        max_in_flight=8, )
    try:
        openings = [
            Opening(opening_id="planted-win", moves=list(_WIN_LINE)),
            _derived_opening(),
        ]
        records = play_arm(
            build_candidate_player(engine, 2, spec=spec, leaf_batch_size=1, c_visit=50.0,
                                   c_scale=1.0, search_kind="puct", gumbel_m=16,
                                   gumbel_seed=seed),
            resolve_bot("random", depth=None, opponent_sims=2)(seed=7),
            openings,
            regime_key=witness_regime(encoding_name=_ENCODING, model_sims=2,
                                      opening_book="book_v1_s20260625_p4"),
            encoding_name=_ENCODING, max_plies=128, games=4,
        )
        return measure_arm(records, encoding_name=_ENCODING, floor=None)
    finally:
        engine.close()


def test_the_seeded_control_is_the_same_net_twice() -> None:
    arch = _tiny_arch()
    assert net_param_hash(seeded_net(arch, seed=1234)) == \
        net_param_hash(seeded_net(arch, seed=1234))


def test_an_unseeded_build_is_a_fresh_draw_which_is_the_defect() -> None:
    """Prove an unseeded `build_net` draws fresh weights: the mutation arm for the seeded control."""
    arch = _tiny_arch()
    assert net_param_hash(build_net(arch)) != net_param_hash(build_net(arch))


def test_two_seeds_build_two_different_baselines() -> None:
    arch = _tiny_arch()
    assert net_param_hash(seeded_net(arch, seed=1234)) != \
        net_param_hash(seeded_net(arch, seed=9999))


def test_a_known_six_in_a_row_reads_longest_run_six() -> None:
    board = replay_board(_WIN_LINE, encoding_name=_ENCODING)
    assert board.check_win() and board.winner() == 1
    assert record_runs(_Record(_WIN_LINE, candidate_color=1),
                       encoding_name=_ENCODING) == (6, 2)


def test_the_seat_follows_the_record_not_the_board() -> None:
    """Prove the seat follows the record: same board, other seat, the pair swaps."""
    assert record_runs(_Record(_WIN_LINE, candidate_color=-1),
                       encoding_name=_ENCODING) == (2, 6)


def test_ply_parity_colouring_reads_the_same_six_as_one() -> None:
    """Measure the original defect: the six-cell winning line reads as ONE under ply parity."""
    assert _parity_longest_run(_WIN_LINE, 1) == 1
    assert _parity_longest_run(_WIN_LINE, -1) == 1


def test_a_board_with_no_stones_reads_zero() -> None:
    assert record_runs(_Record([], candidate_color=1), encoding_name=_ENCODING) == (0, 0)


def test_a_seeded_control_reproduces_its_decisive_count_across_two_runs() -> None:
    """Prove two passes of the same seeded control agree on every measured quantity.

    The decisive count is non-vacuous (the planted opening's two legs are won) and the
    trajectory hashes are asserted beside it, since counts can agree by coincidence.
    """
    first, second = _readout(1234), _readout(1234)
    assert first["decisive_games"] == second["decisive_games"] == 2
    assert first["trajectory_hashes"] == second["trajectory_hashes"]
    assert first == second


def test_a_different_seed_plays_different_games() -> None:
    """Prove a different seed plays different games: the mutation arm for the agreement above."""
    assert _readout(1234)["trajectory_hashes"] != _readout(9999)["trajectory_hashes"]


def test_the_readout_reports_the_winner_and_loser_runs_from_the_engine() -> None:
    measured = _readout(1234)
    assert measured["winner_runs"] == [6, 6], "both legs of the planted opening are won by a six"
    assert max(measured["loser_runs"]) < 6, "the losing side of a won game owns no six"
    assert measured["longest_run_max"] == 6


def test_an_unarmed_floor_reports_no_verdict_rather_than_a_pass() -> None:
    assert measure_arm([], encoding_name=_ENCODING, floor=None)["floor_verdict"] is None


def test_an_armed_floor_reports_its_verdict_and_its_bars() -> None:
    records = [_Record(_WIN_LINE, 1)]
    for rec in records:
        rec.terminal, rec.winner, rec.plies = "win", "candidate", len(_WIN_LINE)
        rec.trajectory_hash = "planted"
    verdict = measure_arm(
        records, encoding_name=_ENCODING,
        floor=StrengthFloorSpec(probe_games=1, min_decisive_rate=0.25, min_winrate=0.0),
    )["floor_verdict"]
    assert verdict["passed"] is True
    assert verdict["decisive_rate"] == 1.0


def test_the_control_arm_is_named_not_inferred_from_a_missing_path() -> None:
    assert ArmSpec.parse("control=CONTROL").checkpoint is None
    assert ArmSpec.parse("bc=/x/y.pt").checkpoint == Path("/x/y.pt")


@pytest.mark.parametrize("raw", ["control", "=CONTROL", ""])
def test_a_malformed_arm_is_refused(raw: str) -> None:
    with pytest.raises(WitnessArmError):
        ArmSpec.parse(raw)


@pytest.mark.parametrize(("encoding", "radius"), [("gnn_axis_v1", 6), ("gnn_axis_r8", 8)])
def test_the_witness_board_carries_the_declared_geometry(encoding: str, radius: int) -> None:
    """Prove `replay_board` builds at the named encoding's geometry, asserted by radius not name."""
    assert replay_board([], encoding_name=encoding).legal_move_radius() == radius
