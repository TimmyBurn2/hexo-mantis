"""The strength-floor decision rule — both bars, both reported, and the measured defect.

The load-bearing test in this file is
`test_an_all_ply_cap_probe_reads_a_healthy_half_on_the_WIN_RATE_axis_alone`: it reproduces
the shakedown burn's own numbers (`draw_rate` 1.0, every game at the arena's ply cap) and
shows that a win-rate bar ALONE reads such a round as a perfectly healthy 0.5. That is the
reason `min_decisive_rate` exists as a separate term rather than as a tightening of
`min_winrate`, and it is measured here rather than argued.

The rule is a pure function over records, so this suite needs no GPU, no book and no
subprocess — which is the whole point of `floor_gate.py` being a module of its own.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from mantis.arena.adjudicate import TERMINAL_PLY_CAP, TERMINAL_WIN
from mantis.config.resolve.eval_posture import StrengthFloorSpec
from mantis.eval.floor_gate import evaluate_strength_floor, probe_measurements


@dataclass(frozen=True)
class _Rec:
    """The two `GameRecord` fields the floor rule reads, and nothing else."""

    winner: str
    terminal: str


def _capped(n: int) -> list[_Rec]:
    return [_Rec(winner="draw", terminal=TERMINAL_PLY_CAP) for _ in range(n)]


def _won(n: int) -> list[_Rec]:
    return [_Rec(winner="candidate", terminal=TERMINAL_WIN) for _ in range(n)]


def _lost(n: int) -> list[_Rec]:
    return [_Rec(winner="opponent", terminal=TERMINAL_WIN) for _ in range(n)]


def _spec(*, decisive: float, wr: float, games: int = 4) -> StrengthFloorSpec:
    return StrengthFloorSpec(
        probe_games=games, min_decisive_rate=decisive, min_winrate=wr
    )


# ── the measurement ────────────────────────────────────────────────────────────────────
def test_probe_measurements_count_decisiveness_from_the_recorded_terminal() -> None:
    """A capped game is NOT decisive; a won or lost game is. The count comes off `terminal`,
    which is why the arena records it — `(winner, plies)` cannot distinguish a win found on
    the cap ply from the cap itself."""
    games, decisive, wins, draws = probe_measurements(_won(2) + _lost(1) + _capped(1))
    assert games == 4
    assert decisive == 3
    assert draws == 1
    assert wins == pytest.approx(2 + 0.5)


def test_the_win_rate_is_draw_aware_like_every_other_win_rate_in_the_package() -> None:
    """Half a win per draw — the same convention `worker.py::_draw_aware_wr` and
    `aggregate.py` use, so the floor's number is comparable with the ones beside it."""
    verdict = evaluate_strength_floor(_capped(4), _spec(decisive=0.0, wr=0.0))
    assert verdict.winrate == pytest.approx(0.5)


# ── the measured defect this bar exists for ────────────────────────────────────────────
def test_an_all_ply_cap_probe_reads_a_healthy_half_on_the_WIN_RATE_axis_alone() -> None:
    """The shakedown burn's shape, reproduced: every game a ply-cap draw (`draw_rate` 1.0).

    A win-rate bar alone PASSES it — 0.5 clears any bar at or below 0.5 — while the round
    contains no information at all. The decisiveness bar is what refuses it, and the two
    assertions below are the same probe read through the two bars.
    """
    probe = _capped(4)
    wr_only = evaluate_strength_floor(probe, _spec(decisive=0.0, wr=0.5))
    assert wr_only.passed, (
        "the premise: at draw_rate 1.0 a win-rate bar of 0.5 is met exactly, which is why a "
        "win-rate-only floor cannot see this failure"
    )
    with_decisive = evaluate_strength_floor(probe, _spec(decisive=0.5, wr=0.5))
    assert not with_decisive.passed
    assert with_decisive.failed_bars == ("decisive_rate",)
    assert with_decisive.decisive_rate == pytest.approx(0.0)


# ── the decision rule ──────────────────────────────────────────────────────────────────
def test_both_bars_are_reported_even_when_both_fail() -> None:
    """A verdict that stopped at the first failing bar would hide the other axis from an
    operator re-tuning the floor. Both must appear."""
    verdict = evaluate_strength_floor(_capped(2) + _lost(2), _spec(decisive=0.9, wr=0.9))
    assert not verdict.passed
    assert set(verdict.failed_bars) == {"decisive_rate", "winrate"}


def test_a_probe_that_clears_both_bars_passes_with_no_failed_bar() -> None:
    verdict = evaluate_strength_floor(_won(3) + _capped(1), _spec(decisive=0.5, wr=0.5))
    assert verdict.passed
    assert verdict.failed_bars == ()
    assert verdict.decisive_rate == pytest.approx(0.75)
    assert verdict.winrate == pytest.approx(3.5 / 4)


def test_an_empty_probe_fails_loudly_rather_than_dividing_by_zero() -> None:
    """Zero games is zero evidence, and a floor that PASSED on no evidence is the
    phantom-gate class LAW-07 exists to prevent. `probe_games >= 1` makes this unreachable
    through a validated config, so the arm is defence in depth."""
    verdict = evaluate_strength_floor([], _spec(decisive=0.0, wr=0.0))
    assert not verdict.passed
    assert "no_probe_games" in verdict.failed_bars
    assert verdict.decisive_rate == 0.0 and verdict.winrate == 0.0


def test_a_zero_winrate_bar_is_an_EXPLICIT_posture_not_a_disabled_lever() -> None:
    """`min_winrate: 0.0` makes that conjunct vacuous BY OPERATOR STATEMENT, in the config,
    where it is readable — as opposed to an absent key saying it silently. The rule must
    honour it rather than treat 0.0 as "unset"."""
    verdict = evaluate_strength_floor(_lost(4), _spec(decisive=1.0, wr=0.0))
    assert verdict.winrate == pytest.approx(0.0)
    assert "winrate" not in verdict.failed_bars
    assert verdict.passed, "the decisiveness bar is met and the win-rate bar was set vacuous"


def test_the_payload_carries_every_number_the_decision_used() -> None:
    """LAW-18's complaint about a bare flag: a `False` with no measurement beside it cannot
    distinguish a starved probe from a failing candidate. The emitted payload must carry the
    bars it was judged against, not only the verdict."""
    spec = _spec(decisive=0.6, wr=0.4)
    payload = evaluate_strength_floor(_won(1) + _capped(3), spec).as_payload()
    assert set(payload) == {
        "passed", "games", "decisive_games", "decisive_rate", "wins", "draws", "winrate",
        "min_decisive_rate", "min_winrate", "failed_bars",
    }
    assert payload["min_decisive_rate"] == spec.min_decisive_rate
    assert payload["min_winrate"] == spec.min_winrate
