"""RED-TEAM-FIX WP11-A F1 (BLOCKER), layer 1 — `allocate_games` must be TOTAL over every
ACTIVE rung (mantis-migration/wp/WP11A/RED_TEAM.md Finding F1).

Pre-fix: `allocate_games` did an unconditional `bt_probs[name]` dict lookup per active rung.
An active rung absent from the round's freshly-fit `bt_probs` (activated this same round,
loud-skipped while a sibling played, or otherwise zero-games-this-round-but-active) raised
an uncaught `KeyError` — reproduced two ways in RED_TEAM.md (forced-active state tamper AND
fully natural STATE §5 activation), both crashing the pipeline's background poller thread
(see `tests/eval/test_round_completion_error.py` for the layer-2 structural fix and its own
oracle).

This is a NEW file (does not edit the frozen `tests/eval/test_ladder_scheduling.py`).

Fix (layer 1, `mantis/eval/ladder.py`): a rung missing from `bt_probs` falls back to
`UNINFORMATIVE_P_HAT = 0.5` — the exact no-information point of the STATE §5 `p*(1-p)`
information-weighting formula (maximized at `p=0.5`), so an unplayed active rung gets the
maximum-information scheduling weight, not an arbitrary default.
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
    """The constant itself: `p*(1-p)` is maximized at exactly `p=0.5` (derivative
    `1-2p=0`), and that maximum (0.25) is strictly greater than any other value in (0,1)."""
    assert UNINFORMATIVE_P_HAT == 0.5
    peak = UNINFORMATIVE_P_HAT * (1.0 - UNINFORMATIVE_P_HAT)
    for p in (0.01, 0.1, 0.3, 0.7, 0.9, 0.99):
        assert p * (1.0 - p) < peak


def test_allocate_games_is_total_over_active_rungs_missing_from_bt_probs() -> None:
    """The RED_TEAM Finding F1 shape, reproduced directly at the `LadderState` unit level:
    an active rung absent from `bt_probs` must not KeyError -- `allocate_games` must return
    a finite, total allocation covering every active rung."""
    rungs = [_rung("resolvable_stub"), _rung("sealbot_d5")]
    cfg = _cfg(rungs)
    state = LadderState.initial(cfg)
    # round 1: resolvable_stub plays and clears the (very low) activation threshold ->
    # sealbot_d5 activates in the SAME record_round call (natural-activation reproduction,
    # mirroring RED_TEAM.md's "(b) Natural-activation reproduction").
    state.record_round(1, {"resolvable_stub": {"games": 20, "wr": 0.9, "ci_lo": 0.8}})
    assert state.status("sealbot_d5") == "active"

    # p_hat covers ONLY the rung that played this round -- mirrors
    # EvalPipeline._current_p_hat() / self._last_p_hat, which is set from THIS round's
    # freshly-fit p_hat only (rung_entities = rungs present in the worker's raw result).
    p_hat = {"resolvable_stub": 0.75}

    alloc = state.allocate_games(2, p_hat)  # must not raise

    assert set(alloc) == {"resolvable_stub", "sealbot_d5"}, "schedule_next must be TOTAL"
    for name, n in alloc.items():
        assert isinstance(n, int)
        assert math.isfinite(n)
        assert n >= 0
    assert sum(alloc.values()) == cfg.round_games  # no min-floor clamp active here (floor=0)


def test_unplayed_active_rung_gets_max_information_weight() -> None:
    """The unplayed rung's fallback weight (`UNINFORMATIVE_P_HAT`, weight 0.25) is the
    MAXIMUM possible p*(1-p) weight -- strictly greater than a played rung's weight unless
    that played rung's own measured p_hat also happens to be exactly 0.5. Verified via the
    actual allocation: the unplayed rung must receive a share at least as large as its
    proportional weight would predict, and strictly larger than a played rung whose
    measured p_hat is farther from 0.5 (i.e. more information already extracted)."""
    rungs = [_rung("resolvable_stub"), _rung("sealbot_d5")]
    cfg = _cfg(rungs, round_games=1000, min_games_per_active_rung=0)
    state = LadderState.initial(cfg)
    state.record_round(1, {"resolvable_stub": {"games": 20, "wr": 0.9, "ci_lo": 0.8}})
    assert state.status("sealbot_d5") == "active"

    # resolvable_stub's measured p_hat (0.75) is farther from 0.5 than the unplayed rung's
    # fallback (exactly 0.5) -> the unplayed rung's weight (0.25) exceeds the played rung's
    # weight (0.75*0.25=0.1875) -> the unplayed rung must get the LARGER share.
    p_hat = {"resolvable_stub": 0.75}
    alloc = state.allocate_games(2, p_hat)
    assert alloc["sealbot_d5"] > alloc["resolvable_stub"]


@pytest.mark.parametrize("missing_rung_p", [0.0, 1.0])
def test_degenerate_missing_and_present_weights_still_total_no_crash(missing_rung_p) -> None:
    """Combine a missing-from-bt_probs rung with a degenerate all-p=0/1 PRESENT rung (weight
    0) -- the `total_weight > 0` uniform-fallback guard (pre-existing) and the F1 total-over-
    active fix must compose without crashing or starving the round to all-zero."""
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
