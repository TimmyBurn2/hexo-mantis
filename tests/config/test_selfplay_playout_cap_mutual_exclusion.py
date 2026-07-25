"""SC-A2 oracle — `PlayoutCapConfig._mutual_exclusion` validator (DESIGN_P2.md §3 /
PREREG_P2.md suite #5). Ports `hparams.py:285-298`'s two frozen hard errors onto the
schema seam, PLUS the Phase-2 "PCR quick>full" RED-TEAM-lens check (REV1 / MUST-FIX #4,
`WPSC_dispatch.md:177-179`): `n_sims_quick > n_sims_full` (both positive) is now REJECTED
at schema-load when `full_search_prob > 0`. The fuller `V-PCR` semantics (R40:
`full_sims==quick_sims` no-op rejection, degenerate `full_fraction` bounds) remain
Phase 3/SC-B4 and are NOT asserted here.

RED-at-import until IMPL lands `PlayoutCapConfig`.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.schema import PlayoutCapConfig

BASE: dict = {
    "fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0, "full_search_prob": 0.0,
    "n_sims_quick": 0, "n_sims_full": 0, "zoi_enabled": False, "zoi_lookback": 16,
    "zoi_margin": 5, "temperature_threshold_compound_moves": 0, "temp_min": 0.5,
}


def _payload(**over: object) -> dict:
    out = dict(BASE)
    out.update(over)
    return out


def test_fast_prob_and_full_search_prob_both_positive_raises():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        PlayoutCapConfig.model_validate(_payload(fast_prob=0.5, full_search_prob=0.5))


def test_full_search_prob_positive_with_zero_n_sims_quick_raises():
    with pytest.raises(ValidationError, match="n_sims_quick"):
        PlayoutCapConfig.model_validate(
            _payload(full_search_prob=0.5, n_sims_quick=0, n_sims_full=200)
        )


def test_full_search_prob_positive_with_zero_n_sims_full_raises():
    with pytest.raises(ValidationError, match="n_sims_full"):
        PlayoutCapConfig.model_validate(
            _payload(full_search_prob=0.5, n_sims_quick=40, n_sims_full=0)
        )


def test_quick_greater_than_full_with_full_search_prob_positive_raises():
    # REV1 MUST-FIX #4 — the "PCR quick>full" RED-TEAM-lens case, moved into Phase 2.
    with pytest.raises(ValidationError, match="n_sims_quick must be <= n_sims_full"):
        PlayoutCapConfig.model_validate(
            _payload(full_search_prob=0.5, n_sims_quick=250, n_sims_full=200)
        )


def test_quick_equal_to_full_constructs_cleanly():
    # boundary/negative-control: equal (not >) is legal.
    cfg = PlayoutCapConfig.model_validate(
        _payload(full_search_prob=0.5, n_sims_quick=200, n_sims_full=200)
    )
    assert cfg.n_sims_quick == cfg.n_sims_full == 200


def test_quick_greater_than_full_with_full_search_prob_zero_constructs_cleanly():
    # the quick>full check only fires when full_search_prob > 0 (matching the other two
    # mutual-exclusion checks' gating and today's zero-behavior-change mint).
    cfg = PlayoutCapConfig.model_validate(
        _payload(full_search_prob=0.0, n_sims_quick=250, n_sims_full=200)
    )
    assert cfg.n_sims_quick == 250 and cfg.n_sims_full == 200


def test_valid_move_level_cap_regime_constructs_cleanly():
    # positive control: a well-formed full_search_prob regime with quick <= full.
    cfg = PlayoutCapConfig.model_validate(
        _payload(full_search_prob=0.3, n_sims_quick=40, n_sims_full=250)
    )
    assert cfg.full_search_prob == 0.3


def test_fast_prob_alone_constructs_cleanly():
    # contrast arm: fast_prob>0 with full_search_prob==0 never triggers ANY of the three
    # mutual-exclusion checks (they all require full_search_prob>0).
    cfg = PlayoutCapConfig.model_validate(_payload(fast_prob=0.3))
    assert cfg.fast_prob == 0.3
