"""`PlayoutCapConfig._mutual_exclusion` — what the schema refuses at load.

Gated on `full_search_prob > 0`: `fast_prob` and `full_search_prob` both positive; either
sims preset zero; `n_sims_quick > n_sims_full`.

Gated instead on BOTH presets being set, independently of `full_search_prob`:
`n_sims_quick == n_sims_full` (a no-op randomization) and a degenerate `full_search_prob`
outside the open interval (0, 1).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantis.config.schema import PlayoutCapConfig

BASE: dict = {
    "fast_sims": 50, "fast_prob": 0.0, "standard_sims": 0, "full_search_prob": 0.0,
    "n_sims_quick": 0, "n_sims_full": 0,
    "temperature_threshold_compound_moves": 0, "temp_min": 0.5,
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
    with pytest.raises(ValidationError, match="n_sims_quick must be <= n_sims_full"):
        PlayoutCapConfig.model_validate(
            _payload(full_search_prob=0.5, n_sims_quick=250, n_sims_full=200)
        )


def test_quick_equal_to_full_now_raises_no_op():
    # The ordering check does NOT fire here (equal is not `>`); the no-op check does.
    with pytest.raises(ValidationError, match="no-op"):
        PlayoutCapConfig.model_validate(
            _payload(full_search_prob=0.5, n_sims_quick=200, n_sims_full=200)
        )


def test_quick_greater_than_full_with_full_search_prob_zero_now_raises_degenerate():
    # The ordering check needs `full_search_prob > 0`, so what fires here is the
    # degenerate-probability check: 0 is outside the open interval (0, 1).
    with pytest.raises(ValidationError, match="0, 1"):
        PlayoutCapConfig.model_validate(
            _payload(full_search_prob=0.0, n_sims_quick=250, n_sims_full=200)
        )


def test_valid_move_level_cap_regime_constructs_cleanly():
    # positive control: quick < full and full_search_prob strictly in (0, 1)
    cfg = PlayoutCapConfig.model_validate(
        _payload(full_search_prob=0.3, n_sims_quick=40, n_sims_full=250)
    )
    assert cfg.full_search_prob == 0.3


def test_fast_prob_alone_constructs_cleanly():
    # contrast arm: `full_search_prob == 0` clears the first three checks and both presets at
    # 0 clears the other gate.
    cfg = PlayoutCapConfig.model_validate(_payload(fast_prob=0.3))
    assert cfg.fast_prob == 0.3


def test_equal_quick_and_full_sims_raises():
    with pytest.raises(ValidationError, match="no-op"):
        PlayoutCapConfig.model_validate(
            _payload(n_sims_quick=100, n_sims_full=100, full_search_prob=0.5)
        )


def test_degenerate_full_search_prob_zero_raises():
    with pytest.raises(ValidationError, match="0, 1"):
        PlayoutCapConfig.model_validate(
            _payload(n_sims_quick=75, n_sims_full=600, full_search_prob=0.0)
        )


def test_degenerate_full_search_prob_one_raises():
    with pytest.raises(ValidationError, match="0, 1"):
        PlayoutCapConfig.model_validate(
            _payload(n_sims_quick=75, n_sims_full=600, full_search_prob=1.0)
        )


def test_valid_differing_sims_and_mid_probability_constructs_cleanly():
    cfg = PlayoutCapConfig.model_validate(
        _payload(n_sims_quick=75, n_sims_full=600, full_search_prob=0.5)
    )
    assert cfg.n_sims_quick == 75
    assert cfg.n_sims_full == 600


def test_all_zero_minted_shape_is_unaffected_negative_control():
    """The gate's `and` needs BOTH sims > 0, so the all-zero minted shape stays constructible."""
    cfg = PlayoutCapConfig.model_validate(
        _payload(n_sims_quick=0, n_sims_full=0, full_search_prob=0.0)
    )
    assert cfg.n_sims_quick == 0
    assert cfg.n_sims_full == 0
