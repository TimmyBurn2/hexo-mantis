"""SC-A2 oracle — `PlayoutCapConfig._mutual_exclusion` validator (DESIGN_P2.md §3 /
PREREG_P2.md suite #5). Ports `hparams.py:285-298`'s two frozen hard errors onto the
schema seam, PLUS the Phase-2 "PCR quick>full" RED-TEAM-lens check (REV1 / MUST-FIX #4,
`WPSC_dispatch.md:177-179`): `n_sims_quick > n_sims_full` (both positive) is now REJECTED
at schema-load when `full_search_prob > 0`.

WPSC Phase 3 SC-B4 (R40, DESIGN_P3.md §5.2) folds in the fuller V-PCR semantics this
file's own Phase-2 docstring deferred: `n_sims_quick == n_sims_full` (no-op) and a
degenerate `full_search_prob` (<=0 or >=1) — gated on "both presets are set"
(`n_sims_quick>0 and n_sims_full>0`), independently of `full_search_prob`. This is a
BROADER gate than the two Phase-2 checks above (which only fire when `full_search_prob >
0`) — it forces two Phase-2 "constructs cleanly" boundary tests to flip to "now raises"
(both renamed below, in place, per ORACLE_NOTES_P3.md row 11's port instruction to fold the
staged `tests/config/test_selfplay_playout_cap_v_pcr_p3.py` oracle into this file):
`test_quick_equal_to_full_constructs_cleanly` (quick==full, both>0 — now the literal no-op
case) and `test_quick_greater_than_full_with_full_search_prob_zero_constructs_cleanly`
(both>0, full_search_prob==0 — now the literal degenerate-probability case, for a DIFFERENT
reason than the old ordering check's gating). Neither rename loses coverage: the narrow
Phase-2 claim each test made ("the OLD check does/doesn't fire under this exact gating") is
preserved as an inline comment; the OUTER assertion (successful construction) was the part
that stopped being true once V-PCR's broader, independent gate landed.
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


def test_quick_equal_to_full_now_raises_no_op():
    # WPSC Phase 3 SC-B4 (was `test_quick_equal_to_full_constructs_cleanly`, Phase 2): the
    # OLD ordering check (`n_sims_quick must be <= n_sims_full`) still does NOT fire here —
    # equal is not `>`. But V-PCR's independently-gated no-op check (both>0, quick==full)
    # now does. The Phase-2 claim ("equal, not >, is legal") is retired by the new, broader
    # gate this file's own docstring predicted ("the fuller V-PCR semantics... remain Phase
    # 3/SC-B4... NOT asserted here").
    with pytest.raises(ValidationError, match="no-op"):
        PlayoutCapConfig.model_validate(
            _payload(full_search_prob=0.5, n_sims_quick=200, n_sims_full=200)
        )


def test_quick_greater_than_full_with_full_search_prob_zero_now_raises_degenerate():
    # WPSC Phase 3 SC-B4 (was `..._constructs_cleanly`, Phase 2): the OLD ordering check
    # still does NOT fire here (it requires full_search_prob>0) — but V-PCR's degenerate-
    # probability check is gated independently of full_search_prob (both n_sims>0 is
    # sufficient), so full_search_prob==0.0 with both presets set now raises for a
    # DIFFERENT reason (0 is outside the open interval (0, 1)), not the ordering check.
    with pytest.raises(ValidationError, match="0, 1"):
        PlayoutCapConfig.model_validate(
            _payload(full_search_prob=0.0, n_sims_quick=250, n_sims_full=200)
        )


def test_valid_move_level_cap_regime_constructs_cleanly():
    # positive control: a well-formed full_search_prob regime with quick <= full AND
    # quick != full AND full_search_prob strictly in (0, 1) — passes every check, old and
    # new (V-PCR SC-B4).
    cfg = PlayoutCapConfig.model_validate(
        _payload(full_search_prob=0.3, n_sims_quick=40, n_sims_full=250)
    )
    assert cfg.full_search_prob == 0.3


def test_fast_prob_alone_constructs_cleanly():
    # contrast arm: fast_prob>0 with full_search_prob==0 never triggers ANY of the three
    # Phase-2 mutual-exclusion checks (they all require full_search_prob>0), and n_sims_
    # quick/n_sims_full are both 0 (BASE default) so V-PCR's "both presets set" gate never
    # fires either.
    cfg = PlayoutCapConfig.model_validate(_payload(fast_prob=0.3))
    assert cfg.fast_prob == 0.3


# ── V-PCR (R40, WPSC Phase 3 SC-B4, DESIGN_P3.md §5.2) ────────────────────────────────────
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
    """The new gate's `and` requires BOTH sims > 0 — the all-zero disabled shape every
    current config + template mints must stay constructible."""
    cfg = PlayoutCapConfig.model_validate(
        _payload(n_sims_quick=0, n_sims_full=0, full_search_prob=0.0)
    )
    assert cfg.n_sims_quick == 0
    assert cfg.n_sims_full == 0
