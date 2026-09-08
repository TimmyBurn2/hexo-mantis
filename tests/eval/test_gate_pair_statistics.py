"""R345(b)(4) — the gate's CI must resample opening PAIRS, not games.

THE ERROR, PRECISELY. `pair_bootstrap_wr_ci` is named for pairs and resamples
`_distinct_outcomes`, which is one value per distinct GAME. `_traj_key` deliberately qualifies
by seat so the two colour-swapped legs of one opening are two independent entries — and that
was the right fix for the defect it addressed (identical move lists collapsing into one
record, which halved `eff_n` and biased the WR). What it also did was make the bootstrap treat
two legs of one opening as two independent draws.

They are not independent. Both legs start from the SAME position; an opening that is winning
for whoever moves first contributes a win and a loss almost deterministically, and an opening
the candidate simply understands better contributes two wins. The variance the bootstrap needs
lives BETWEEN openings, and resampling within them understates it. The direction is the bad
one: a CI narrower than the truth, on the lower bound `gate_promotion_decision` reads as
`ci_lo_boot > 0.0`. That is a promotion bar that clears more often than its stated confidence.

THE UNIT IS THE OPENING. Both legs of one opening are averaged into one value, and the
bootstrap resamples those. `eff_n` follows the same unit, because an effective-n counted in
games while the CI is computed over pairs would be two answers to one question (LAW-04's own
subject).

AND THE ROUNDS MUST NOT REPLAY ONE ANOTHER. Every round drew its openings at a CONSTANT seed
(`spec.gate.seed_base`), so round 40 played the same games as round 1. Correlated looks make
the promotion series far less informative than its game count suggests, and the degradation
flag reads a series that is mostly one sample repeated.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mantis.arena.books import round_openings
from mantis.eval.aggregate import (
    aggregate_gate,
    pair_bootstrap_wr_ci,
    pair_units,
)


class _GateCfg:
    bootstrap_resamples = 400
    seed_base = 7
    promotion_winrate = 0.55
    min_distinct_per_pair = 1
    screen_confirm_lo = 0.0


def _rec(opening: str, seat: int, winner: str, moves: list[tuple[int, int]]) -> dict[str, Any]:
    return {
        "p1": "cand", "p2": "anchor", "winner": winner, "moves": moves,
        "opening_id": opening, "candidate_color": seat, "regime_key": "rk",
    }


def _split_pair(opening: str, first: str, second: str) -> list[dict[str, Any]]:
    """One opening's two colour-swapped legs, with distinct move lists."""
    return [
        _rec(opening, 1, first, [(0, 0), (1, 0)]),
        _rec(opening, -1, second, [(0, 0), (0, 1)]),
    ]


# ── the unit ────────────────────────────────────────────────────────────────────────────
def test_the_two_legs_of_an_opening_are_one_unit() -> None:
    """A split pair is ONE value at 0.5, not two values at 1.0 and 0.0."""
    records = _split_pair("op0", "p1", "p2")
    units = pair_units(records)
    assert len(units) == 1, f"an opening produced {len(units)} units, not one"
    assert units[0] == pytest.approx(0.5)


def test_a_swept_opening_is_one_unit_at_one() -> None:
    """The unit averages, it does not collapse to a first-seen leg."""
    assert pair_units(_split_pair("op0", "p1", "p1")) == pytest.approx([1.0])


def test_distinct_openings_are_distinct_units() -> None:
    records = _split_pair("op0", "p1", "p2") + _split_pair("op1", "p1", "p1")
    assert sorted(pair_units(records)) == pytest.approx([0.5, 1.0])


def test_an_unpaired_leg_is_still_a_unit() -> None:
    """A round that lost one leg to a forfeit must not lose the opening entirely."""
    assert pair_units([_rec("op0", 1, "p1", [(0, 0)])]) == pytest.approx([1.0])


def test_a_record_with_no_opening_id_is_its_own_unit() -> None:
    """Legacy records carry no `opening_id`; they must not all collapse into one unit.

    Pairing on a missing key would make an entire legacy round ONE observation, which would
    read as a catastrophic loss of power rather than as the absent field it is.
    """
    records = [
        {"p1": "c", "p2": "a", "winner": "p1", "moves": [(0, 0)], "candidate_color": 1},
        {"p1": "c", "p2": "a", "winner": "p2", "moves": [(1, 1)], "candidate_color": 1},
    ]
    assert len(pair_units(records)) == 2


# ── the CI ──────────────────────────────────────────────────────────────────────────────
def test_the_pair_bootstrap_is_wider_than_the_game_bootstrap_on_correlated_legs() -> None:
    """The measurement that makes this leg worth doing, not an assertion about the code.

    Twenty openings, each swept one way or the other — the maximally correlated case, and the
    realistic one for a candidate that is simply better or worse on a given position. Counted
    as games the sample looks like 40 near-coin-flips; counted as pairs it is 20 observations
    at 0 or 1. The game-level interval is the narrower one, and it is narrower than the truth.
    """
    records: list[dict[str, Any]] = []
    for i in range(20):
        swept = "p1" if i % 2 == 0 else "p2"
        records.extend(_split_pair(f"op{i}", swept, swept))

    game_level = np.array([1.0 if r["winner"] == "p1" else 0.0 for r in records])
    pair_level = np.asarray(pair_units(records))

    g_lo, g_hi = pair_bootstrap_wr_ci(game_level, resamples=2000, ci_level=0.95, seed=3)
    p_lo, p_hi = pair_bootstrap_wr_ci(pair_level, resamples=2000, ci_level=0.95, seed=3)
    assert g_lo is not None and g_hi is not None and p_lo is not None and p_hi is not None
    assert (p_hi - p_lo) > (g_hi - g_lo), (
        f"the pair-level interval ({p_hi - p_lo:.4f}) is not wider than the game-level one "
        f"({g_hi - g_lo:.4f}) — the correlated legs were still being counted as independent"
    )


def test_the_gate_ci_and_eff_n_are_both_over_pairs() -> None:
    """Two answers to one question is the defect; the unit must be the same on both."""
    records: list[dict[str, Any]] = []
    for i in range(12):
        records.extend(_split_pair(f"op{i}", "p1", "p2"))

    result = aggregate_gate(records, [], _GateCfg())

    assert result.eff_n == 12, (
        f"eff_n is {result.eff_n} — counted in games (24) rather than in the pairs the CI "
        "is computed over"
    )


def test_the_gate_result_retains_wins_losses_and_draws() -> None:
    """A promotion decision with no W/L/D beside it cannot be read after the fact."""
    records = _split_pair("op0", "p1", "p2") + _split_pair("op1", "p1", "draw")
    result = aggregate_gate(records, [], _GateCfg())
    assert (result.wins, result.losses, result.draws) == (2, 1, 1), (
        f"W/L/D reads {(result.wins, result.losses, result.draws)} over "
        f"{result.n_pooled} pooled games"
    )
    assert result.wins + result.losses + result.draws == result.n_pooled


def test_a_perfect_sweep_still_promotes() -> None:
    """Mutation half: a CI so wide nothing clears it is not a fix, it is a broken gate."""
    records: list[dict[str, Any]] = []
    for i in range(30):
        records.extend(_split_pair(f"op{i}", "p1", "p1"))
    result = aggregate_gate(records, [], _GateCfg())
    assert result.promoted, (
        "a candidate that won every game on thirty distinct openings did not promote — the "
        "pair-level interval is not an interval, it is a refusal"
    )


# ── the opening subsets ─────────────────────────────────────────────────────────────────
def test_consecutive_rounds_draw_disjoint_openings() -> None:
    """The property the constant seed did not have."""
    a = round_openings("book_v1_s20260625_p4", n_pairs=24, seed_base=20260625, round_index=0)
    b = round_openings("book_v1_s20260625_p4", n_pairs=24, seed_base=20260625, round_index=1)
    assert len(a) == len(b) == 24
    assert not ({o.opening_id for o in a} & {o.opening_id for o in b}), (
        "rounds 0 and 1 share openings — the promotion series repeats its own sample"
    )


def test_the_subset_is_derived_from_the_seed_and_the_round_index() -> None:
    """Same (seed_base, round_index) → same openings; either one moves → they move."""
    args = dict(n_pairs=8, seed_base=20260625)
    first = [o.opening_id for o in round_openings("book_v1_s20260625_p4", round_index=3, **args)]
    again = [o.opening_id for o in round_openings("book_v1_s20260625_p4", round_index=3, **args)]
    other_round = [
        o.opening_id for o in round_openings("book_v1_s20260625_p4", round_index=4, **args)
    ]
    other_seed = [
        o.opening_id
        for o in round_openings("book_v1_s20260625_p4", n_pairs=8, seed_base=99, round_index=3)
    ]
    assert first == again, "the same round drew a different subset twice"
    assert first != other_round, "the round index does not move the subset"
    assert first != other_seed, "the seed does not move the subset"


def test_the_confirm_block_does_not_replay_the_screen_block() -> None:
    """The confirm phase must draw its own slice, on its own PERMUTATION.

    Found by measuring, not by reading. The first cut of this leg moved
    `_CONFIRM_SEED_OFFSET` from the seed to the ROUND INDEX, which looks equivalent and is
    not: screen and confirm take windows of DIFFERENT widths (40 and 64 pairs at run6's
    settings) out of the SAME permutation, so an index offset merely shifts where each lands
    and they collide on a schedule. Round 2's confirm block drew ALL FORTY of the screen's
    openings; rounds 1 and 3 drew 24 and 32. With deterministic argmax players a replayed
    opening yields the SAME game, so escalation bought 128 games of wall-clock and no new
    evidence — and `pair_units` correctly merges them, so the pooled `eff_n` silently
    collapses instead of growing.

    A different permutation makes the overlap incidental rather than scheduled. This asserts
    the property (no round is a near-replay), not a particular number.
    """
    screen_pairs, confirm_pairs = 40, 64
    seed_base, offset = 20260625, 7919
    for round_index in range(12):
        screen = {o.opening_id for o in round_openings(
            "book_v1_s20260625_p4", n_pairs=screen_pairs,
            seed_base=seed_base, round_index=round_index)}
        confirm = {o.opening_id for o in round_openings(
            "book_v1_s20260625_p4", n_pairs=confirm_pairs,
            seed_base=seed_base + offset, round_index=round_index)}
        shared = len(screen & confirm)
        assert shared <= screen_pairs // 2, (
            f"round {round_index}: the confirm block re-played {shared} of the screen's "
            f"{screen_pairs} openings — an escalation that mostly repeats the screen is "
            "wall-clock without evidence"
        )


def test_the_window_wraps_rather_than_running_out() -> None:
    """A block outlasts the book: round `n_openings/k` must still get a full subset.

    Wrapping is not the same as repeating the FIRST round: the permutation is over the whole
    book, so a wrapped round is a fresh alignment rather than a replay — but the guarantee
    this leg makes is the disjointness of CONSECUTIVE rounds, and past the wrap it is stated
    as what it is rather than claimed.
    """
    per_round = 24
    late = round_openings(
        "book_v1_s20260625_p4", n_pairs=per_round, seed_base=1, round_index=1000,
    )
    assert len(late) == per_round, "a late round ran out of openings instead of wrapping"
    assert len({o.opening_id for o in late}) == per_round, "a wrapped round repeated an opening"
