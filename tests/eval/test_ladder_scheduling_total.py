"""`allocate_games` must be TOTAL over every ACTIVE rung.

An unconditional `bt_probs[name]` lookup raised an uncaught `KeyError` for an active rung absent
from the round's freshly-fit probabilities — activated this same round, loud-skipped while a
sibling played, or otherwise active with zero games — and crashed the pipeline's poller thread.

A missing rung falls back to `UNINFORMATIVE_P_HAT = 0.5`, the exact no-information point of the
`p*(1-p)` weighting, so an unplayed active rung gets the maximum-information scheduling weight
rather than an arbitrary default.
"""
from __future__ import annotations

import math

import pytest

from mantis.config.schema import LadderConfig, LadderRung
from mantis.eval.ladder import UNINFORMATIVE_P_HAT, LadderState


def _rung(name: str, games_max: int = 1_000_000) -> LadderRung:
    return LadderRung(
        name=name, bot="random", variant="raw", depth=None, opponent_sims=None,
        opening_book="book_v1_s20260625_p4", deploy_matched=True, games_max=games_max,
    )


def _cfg(rungs, **overrides) -> LadderConfig:
    defaults = dict(
        round_games=100, min_games_per_active_rung=0, graduation_wr_lower_ci=0.75,
        graduation_consec_rounds=3, activation_wr_lower_ci=0.01, calibration_every_k_rounds=4,
        calibration_games=8, bootstrap_resamples=200, bootstrap_ci_level=0.95,
        bt_prior_games=1.0, bootstrap_seed=1234,
    )
    defaults.update(overrides)
    return LadderConfig(rungs=list(rungs), **defaults)


def test_uninformative_p_hat_is_the_no_information_maximum() -> None:
    """`p*(1-p)` is maximized at exactly `p=0.5`, strictly above any other value in (0,1)."""
    assert UNINFORMATIVE_P_HAT == 0.5
    peak = UNINFORMATIVE_P_HAT * (1.0 - UNINFORMATIVE_P_HAT)
    for p in (0.01, 0.1, 0.3, 0.7, 0.9, 0.99):
        assert p * (1.0 - p) < peak


def test_allocate_games_is_total_over_active_rungs_missing_from_bt_probs() -> None:
    """An active rung absent from `bt_probs` must not KeyError: the allocation stays finite and
    total over every active rung."""
    rungs = [_rung("resolvable_stub"), _rung("sealbot_d5")]
    cfg = _cfg(rungs)
    state = LadderState.initial(cfg)
    # resolvable_stub clears the very low activation threshold, so sealbot_d5 activates in the
    # SAME record_round call — the natural-activation reproduction.
    state.record_round(1, {"resolvable_stub": {"games": 20, "wr": 0.9, "ci_lo": 0.8}})
    assert state.status("sealbot_d5") == "active"

    # p_hat covers ONLY the rung that played this round, as the pipeline sets it: from THIS
    # round's freshly-fit values, over the rungs present in the worker's raw result.
    p_hat = {"resolvable_stub": 0.75}

    alloc = state.allocate_games(2, p_hat)  # must not raise

    assert set(alloc) == {"resolvable_stub", "sealbot_d5"}, "schedule_next must be TOTAL"
    for name, n in alloc.items():
        assert isinstance(n, int)
        assert math.isfinite(n)
        assert n >= 0
    assert sum(alloc.values()) == cfg.round_games  # no min-floor clamp active here (floor=0)


def test_unplayed_active_rung_gets_max_information_weight() -> None:
    """The unplayed rung's fallback weight is the MAXIMUM possible `p*(1-p)`, so it must take a
    larger share than a played rung whose measured p_hat is farther from 0.5."""
    rungs = [_rung("resolvable_stub"), _rung("sealbot_d5")]
    cfg = _cfg(rungs, round_games=1000, min_games_per_active_rung=0)
    state = LadderState.initial(cfg)
    state.record_round(1, {"resolvable_stub": {"games": 20, "wr": 0.9, "ci_lo": 0.8}})
    assert state.status("sealbot_d5") == "active"

    # weight(unplayed) = 0.25 exceeds weight(resolvable_stub) = 0.75*0.25 = 0.1875, so the
    # unplayed rung must get the LARGER share.
    p_hat = {"resolvable_stub": 0.75}
    alloc = state.allocate_games(2, p_hat)
    assert alloc["sealbot_d5"] > alloc["resolvable_stub"]


@pytest.mark.parametrize("missing_rung_p", [0.0, 1.0])
def test_degenerate_missing_and_present_weights_still_total_no_crash(missing_rung_p) -> None:
    """A missing rung and a zero-weight degenerate rung must compose without crashing or
    starving the round to all-zero."""
    rungs = [_rung("a"), _rung("b")]
    cfg = _cfg(rungs, round_games=20, min_games_per_active_rung=0)
    state = LadderState.initial(cfg)
    state.record_round(1, {"a": {"games": 10, "wr": 1.0 if missing_rung_p else 0.0, "ci_lo": 0.99}})
    assert state.status("b") == "active"
    p_hat = {"a": missing_rung_p}  # weight(a) = p*(1-p) = 0; "b" missing entirely
    alloc = state.allocate_games(2, p_hat)
    assert set(alloc) == {"a", "b"}
    assert sum(alloc.values()) == cfg.round_games
    assert alloc["b"] > 0, "the missing rung (uninformative fallback) must not be starved"
