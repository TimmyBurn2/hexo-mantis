"""The BC acceptance witness's two repaired halves, each with the arm that made it necessary.

Sitting 6's first witness was an ad-hoc box script. It reported a PASS, and two defects rode
inside the pass without appearing in its output. This suite pins the repairs and, beside each
one, the mutation that reproduces the original defect — a repair asserted only in its fixed
form passes equally against code that never fixed anything (LAW-07's whole complaint).

DEFECT 1, the control arm was a DRAW. `build_net` was called with whatever global RNG state
the process happened to hold, so the baseline the BC arm was compared against was resampled
every invocation. Pinned by `net_param_hash` equality across two builds, with the unseeded
pair asserted UNEQUAL beside it.

DEFECT 2, stone colour was re-derived from PLY PARITY. The script assigned
`1 if (i // 2) % 2 == 0 else -1`; the engine gives player 1 the first stone and then alternates
in PAIRS (LAW-03). The planted line below is the same one
`tests/arena/test_ply_cap_adjudication.py` uses for its win-on-the-cap case: player 1 owns a
genuine six-in-a-row. The engine reads that as 6; the parity expression reads it as 1, and the
suite asserts BOTH so the size of the error is on the record rather than described.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mantis.arena.books import Opening, paired_openings
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
from mantis.model import CnnArch, build_net
from mantis.model.identity import net_param_hash
from mantis.selfplay.inference_local import LocalInferenceEngine

#: A finished game: player 1 completes a six-in-a-row along `(1, 0)` at ply 11. Cells follow
#: the ENGINE's compound-turn order (ply 0 to player 1, then pairs), which is exactly the order
#: the ply-parity expression gets wrong.
_WIN_LINE: list[tuple[int, int]] = [
    (0, 0),          # ply 0      -> player  1
    (9, 9), (0, 9),  # plies 1,2  -> player -1
    (1, 0), (2, 0),  # plies 3,4  -> player  1
    (2, 9), (4, 9),  # plies 5,6  -> player -1
    (3, 0), (4, 0),  # plies 7,8  -> player  1
    (6, 9), (8, 9),  # plies 9,10 -> player -1
    (5, 0),          # ply 11     -> player  1, and the six is complete
]
_ENCODING = "v6"
_AXES = ((1, 0), (0, 1), (1, -1))


class _Record:
    """The two `GameRecord` fields `record_runs` reads, and nothing else."""

    def __init__(self, moves: list[tuple[int, int]], candidate_color: int) -> None:
        self.moves = tuple(moves)
        self.colors = {"candidate": candidate_color, "opponent": -candidate_color}


def _parity_longest_run(moves: list[tuple[int, int]], player: int) -> int:
    """The ORIGINAL witness's reconstruction, verbatim, kept as the mutation arm."""
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


def _tiny_arch() -> CnnArch:
    # Registry-true "v6" dims (board_size=19, n_planes=8), minimal width/depth for speed —
    # the same fixture shape `tests/eval/test_round_end_to_end.py` uses.
    return CnnArch(board_size=19, in_channels=8, filters=8, res_blocks=1)


def _readout(seed: int) -> dict:
    """One full witness pass for a SEEDED control arm, on CPU."""
    spec = lookup(_ENCODING)
    device = torch.device("cpu")
    engine = LocalInferenceEngine(
        seeded_net(_tiny_arch(), seed=seed).to(device).eval(), device, encoding_spec=spec,
        fused_graph_caps=None, inference_batching=None, max_in_flight=1, amp_dtype="bf16",
    )
    try:
        openings = [
            Opening(opening_id="planted-win", moves=list(_WIN_LINE)),
            *paired_openings("book_v1_s20260625_p4", n_pairs=1, seed=7),
        ]
        records = play_arm(
            build_candidate_player(engine, 2, spec=spec, leaf_batch_size=1),
            resolve_bot("random", depth=None, opponent_sims=2)(seed=7),
            openings,
            regime_key=witness_regime(encoding_name=_ENCODING, model_sims=2,
                                      opening_book="book_v1_s20260625_p4"),
            encoding_name=_ENCODING, max_plies=128, games=4,
        )
        return measure_arm(records, encoding_name=_ENCODING, floor=None)
    finally:
        engine.close()


# --------------------------------------------------------------------------------------
# DEFECT 1 — the control must be a fixed baseline, not a fresh draw.
# --------------------------------------------------------------------------------------

def test_the_seeded_control_is_the_same_net_twice() -> None:
    arch = _tiny_arch()
    assert net_param_hash(seeded_net(arch, seed=1234)) == \
        net_param_hash(seeded_net(arch, seed=1234))


def test_an_unseeded_build_is_a_fresh_draw_which_is_the_defect() -> None:
    """The mutation arm: without the seam the two builds differ, so the arm the BC number is
    compared against moves on its own. Without this assertion the test above would pass
    against a `build_net` that happened to be deterministic for an unrelated reason."""
    arch = _tiny_arch()
    assert net_param_hash(build_net(arch)) != net_param_hash(build_net(arch))


def test_two_seeds_build_two_different_baselines() -> None:
    arch = _tiny_arch()
    assert net_param_hash(seeded_net(arch, seed=1234)) != \
        net_param_hash(seeded_net(arch, seed=9999))


# --------------------------------------------------------------------------------------
# DEFECT 2 — the longest run is the ENGINE's, and the colours are the engine's.
# --------------------------------------------------------------------------------------

def test_a_known_six_in_a_row_reads_longest_run_six() -> None:
    board = replay_board(_WIN_LINE, encoding_name=_ENCODING)
    assert board.check_win() and board.winner() == 1
    assert record_runs(_Record(_WIN_LINE, candidate_color=1),
                       encoding_name=_ENCODING) == (6, 2)


def test_the_seat_follows_the_record_not_the_board() -> None:
    """Same board, other seat: the pair swaps rather than the numbers changing."""
    assert record_runs(_Record(_WIN_LINE, candidate_color=-1),
                       encoding_name=_ENCODING) == (2, 6)


def test_ply_parity_colouring_reads_the_same_six_as_one() -> None:
    """The mutation arm, and the measurement of how wrong the original was: the winning line
    is six cells and the parity reconstruction finds ONE, for either side. That is why the
    first witness could report twenty wins — each requiring a six — with a longest run of 4."""
    assert _parity_longest_run(_WIN_LINE, 1) == 1
    assert _parity_longest_run(_WIN_LINE, -1) == 1


def test_a_board_with_no_stones_reads_zero() -> None:
    assert record_runs(_Record([], candidate_color=1), encoding_name=_ENCODING) == (0, 0)


# --------------------------------------------------------------------------------------
# The two repairs together: the readout a witness run produces.
# --------------------------------------------------------------------------------------

def test_a_seeded_control_reproduces_its_decisive_count_across_two_runs() -> None:
    """Two independent passes of the same seeded control agree on every measured quantity.

    The decisive count is asserted explicitly AND is non-vacuous: the planted opening is
    already won when the arena replays it, so both of its colour legs terminate `win` while
    the book opening's two legs run to the ply cap at fresh-init strength (the all-draw shape
    F-R-P2B-5 measured). Equality of the trajectory hashes is asserted beside it, because a
    decisive count alone can agree by coincidence and a move-for-move replay cannot.
    """
    first, second = _readout(1234), _readout(1234)
    assert first["decisive_games"] == second["decisive_games"] == 2
    assert first["trajectory_hashes"] == second["trajectory_hashes"]
    assert first == second


def test_a_different_seed_plays_different_games() -> None:
    """The mutation arm for the pass above: without the seeding, that agreement is what an
    uncontrolled draw would have to reproduce by luck."""
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


# --------------------------------------------------------------------------------------
# The arm vocabulary.
# --------------------------------------------------------------------------------------

def test_the_control_arm_is_named_not_inferred_from_a_missing_path() -> None:
    assert ArmSpec.parse("control=CONTROL").checkpoint is None
    assert ArmSpec.parse("bc=/x/y.pt").checkpoint == Path("/x/y.pt")


@pytest.mark.parametrize("raw", ["control", "=CONTROL", ""])
def test_a_malformed_arm_is_refused(raw: str) -> None:
    with pytest.raises(WitnessArmError):
        ArmSpec.parse(raw)


@pytest.mark.parametrize(("encoding", "radius"), [("gnn_axis_v1", 6), ("gnn_axis_r8", 8)])
def test_the_witness_board_carries_the_declared_geometry(encoding: str, radius: int) -> None:
    """`replay_board` names the encoding at every construction, and the encoding is what fixes
    the geometry — a board built under another one is another board, which is the whole content
    of the r8 identity change. The two run6-lineage rows are asserted by their radii so this
    reads as a geometry check rather than a name check."""
    assert replay_board([], encoding_name=encoding).legal_move_radius() == radius
