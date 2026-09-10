"""Opponent-ladder scheduling and CI-hysteresis graduation (mantis.eval.ladder).

Every threshold and cadence is read off the `LadderConfig` fixture rather than hardcoded,
so literal drift in the implementation shows up in the source-grep oracle in
tests/eval/test_ladder_config_schema.py, not here.

>300 justify: one state machine (activation/graduation/scheduling/persistence) under one
shared fixture ladder; splitting it by behavior would duplicate the LadderConfig/rung
fixtures and let the scheduling and hysteresis halves drift apart.
"""
from __future__ import annotations

import os
import stat

import pytest

from mantis.config.schema import LadderConfig, LadderRung
from mantis.eval.errors import LadderStateError
from mantis.eval.ladder import LadderState


def _rung(name: str, games_max: int = 1_000_000) -> LadderRung:
    return LadderRung(
        name=name,
        bot="random",
        variant="raw",
        depth=None,
        opponent_sims=None,
        opening_book="book_v1_s20260625_p4",
        deploy_matched=True,
        games_max=games_max,
    )


def _cfg(rungs, **overrides) -> LadderConfig:
    defaults = dict(
        round_games=100,
        min_games_per_active_rung=4,
        graduation_wr_lower_ci=0.75,
        graduation_consec_rounds=3,
        activation_wr_lower_ci=0.65,
        calibration_every_k_rounds=4,
        calibration_games=8,
        bootstrap_resamples=200,
        bootstrap_ci_level=0.95,
        bt_prior_games=1.0,
        bootstrap_seed=1234,
    )
    defaults.update(overrides)
    return LadderConfig(rungs=list(rungs), **defaults)


def _round(games: int, wr: float | None, ci_lo: float | None) -> dict:
    return {"games": games, "wr": wr, "ci_lo": ci_lo}


def test_allocation_proportional_to_p_hat_variance() -> None:
    # w = p(1-p) = [0.25, 0.09, 0.21]; 100*w/sum(w) floors to [45, 16, 38] with remainder 1,
    # whose largest fractional part is rung "a" -> [46, 16, 38].
    cfg = _cfg([_rung("a"), _rung("b"), _rung("c")], round_games=100, min_games_per_active_rung=0)
    state = LadderState.initial(cfg)
    state.record_round(0, {"b": _round(10, 0.7, 0.66), "c": _round(10, 0.7, 0.66)})
    assert state.status("a") == "active"
    assert state.status("b") == "active"
    assert state.status("c") == "active"

    alloc = state.allocate_games(1, {"a": 0.5, "b": 0.9, "c": 0.3})
    assert alloc == {"a": 46, "b": 16, "c": 38}


def test_min_games_floor_applies_to_active_rungs() -> None:
    # Skew that would starve "b" to 0 under pure proportional allocation.
    cfg = _cfg([_rung("a"), _rung("b")], round_games=10, min_games_per_active_rung=4)
    state = LadderState.initial(cfg)
    state.record_round(0, {"b": _round(10, 0.99, 0.66)})
    assert state.status("b") == "active"

    alloc = state.allocate_games(1, {"a": 0.5, "b": 0.99})
    assert alloc["a"] >= 4
    assert alloc["b"] >= 4


def test_single_active_rung_takes_all_round_games() -> None:
    cfg = _cfg([_rung("a")], round_games=50)
    state = LadderState.initial(cfg)
    alloc = state.allocate_games(0, {"a": 0.5})
    assert alloc == {"a": 50}


def test_allocation_clamps_at_rung_games_max() -> None:
    # Clamped excess is not redistributed: the total undershoots round_games deterministically.
    cfg = _cfg([_rung("a", games_max=5), _rung("b", games_max=1_000_000)], round_games=50)
    state = LadderState.initial(cfg)
    state.record_round(0, {"b": _round(10, 0.7, 0.66)})
    assert state.status("b") == "active"

    alloc = state.allocate_games(1, {"a": 0.5, "b": 0.5})
    assert alloc["a"] == 5
    assert alloc["b"] == 25          # unclamped proportional share, NOT topped up to 45
    assert sum(alloc.values()) < 50  # excess not redistributed


def test_rung_activates_when_predecessor_lower_ci_reaches_threshold() -> None:
    cfg = _cfg([_rung("a"), _rung("b")], activation_wr_lower_ci=0.65)
    state = LadderState.initial(cfg)
    assert state.status("a") == "active"
    assert state.status("b") == "dormant"

    state.record_round(0, {"a": _round(20, 0.6, 0.64)})   # below threshold -> stays dormant
    assert state.status("b") == "dormant"

    state.record_round(1, {"a": _round(20, 0.7, 0.65)})   # at threshold -> activates
    assert state.status("b") == "active"

    # Sticky: a later drop in "a"'s ci_lo does not de-activate "b".
    state.record_round(2, {"a": _round(20, 0.5, 0.40)})
    assert state.status("b") == "active"


def test_graduation_requires_three_consecutive_measured_qualifying_rounds() -> None:
    cfg = _cfg([_rung("a")], graduation_wr_lower_ci=0.75, graduation_consec_rounds=3)
    state = LadderState.initial(cfg)
    for i in range(2):
        state.record_round(i, {"a": _round(10, 0.9, 0.80)})
        assert state.status("a") == "active"
        assert state.consec("a") == i + 1
    state.record_round(2, {"a": _round(10, 0.9, 0.80)})
    assert state.consec("a") == 3
    assert state.status("a") == "saturated"


def test_flapping_around_threshold_resets_consecutive_counter() -> None:
    cfg = _cfg([_rung("a")], graduation_wr_lower_ci=0.75, graduation_consec_rounds=3)
    state = LadderState.initial(cfg)
    sequence = [0.76, 0.74, 0.76, 0.76, 0.76]
    for i, ci_lo in enumerate(sequence):
        state.record_round(i, {"a": _round(10, ci_lo + 0.05, ci_lo)})
        if i < len(sequence) - 1:
            assert state.status("a") == "active", f"round {i} graduated too early"
    assert state.status("a") == "saturated"   # only after the final three (indices 2,3,4)
    assert state.consec("a") == 3


def test_zero_game_round_holds_streak_without_advancing() -> None:
    # A zero-game round is transparent: it neither advances nor resets the streak, so
    # [qualify, ZERO, qualify, qualify] reads 1, 1, 2, 3.
    cfg = _cfg([_rung("a")], graduation_wr_lower_ci=0.75, graduation_consec_rounds=3)
    state = LadderState.initial(cfg)
    state.record_round(0, {"a": _round(10, 0.9, 0.80)})
    assert state.consec("a") == 1
    state.record_round(1, {})                              # rung absent -> zero games, HELD
    assert state.consec("a") == 1
    assert state.status("a") == "active"
    state.record_round(2, {"a": _round(10, 0.9, 0.80)})
    assert state.consec("a") == 2
    state.record_round(3, {"a": _round(10, 0.9, 0.80)})
    assert state.consec("a") == 3
    assert state.status("a") == "saturated"

    # contrast arm: [qualify, sub-threshold(measured), qualify] -> counter 1, 0, 1
    cfg2 = _cfg([_rung("x")], graduation_wr_lower_ci=0.75, graduation_consec_rounds=3)
    state2 = LadderState.initial(cfg2)
    state2.record_round(0, {"x": _round(10, 0.9, 0.80)})
    assert state2.consec("x") == 1
    state2.record_round(1, {"x": _round(10, 0.5, 0.40)})   # measured, sub-threshold -> RESET
    assert state2.consec("x") == 0
    state2.record_round(2, {"x": _round(10, 0.9, 0.80)})
    assert state2.consec("x") == 1


def test_zero_games_round_leaves_state_unchanged_and_loud() -> None:
    cfg = _cfg([_rung("a")])
    state = LadderState.initial(cfg)
    state.record_round(0, {"a": _round(10, 0.9, 0.80)})
    consec_before = state.consec("a")
    status_before = state.status("a")

    class _Sink:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, event) -> None:
            self.events.append(dict(event))

    sink = _Sink()
    state.record_round(1, {"a": _round(0, None, None)}, sink=sink)
    assert state.consec("a") == consec_before   # unchanged (HELD, not reset/advanced)
    assert state.status("a") == status_before
    assert len(sink.events) >= 1                # loud: something was emitted


def test_saturated_rung_drops_to_calibration_cadence_and_never_retires() -> None:
    cfg = _cfg(
        [_rung("a")],
        graduation_wr_lower_ci=0.75,
        graduation_consec_rounds=3,
        calibration_every_k_rounds=4,
        calibration_games=5,
    )
    state = LadderState.initial(cfg)
    for i in range(3):
        state.record_round(i, {"a": _round(10, 0.9, 0.80)})
    assert state.status("a") == "saturated"

    hits = []
    for round_idx in range(3, 20):
        alloc = state.allocate_games(round_idx, {"a": 0.9})
        hits.append(alloc.get("a", 0))
        # calibration WR still flows into history even though saturated (never retired).
        if alloc.get("a", 0) > 0:
            state.record_round(round_idx, {"a": _round(alloc["a"], 0.9, 0.80)})
        assert state.status("a") == "saturated"   # terminal: never de-saturates

    nonzero = [h for h in hits if h > 0]
    assert nonzero, "a saturated rung must still play calibration games periodically"
    assert all(h == 5 for h in nonzero)
    assert any(h == 0 for h in hits), "off-cadence rounds must allocate zero, not every round"


def test_ladder_state_roundtrips_json(tmp_path) -> None:
    cfg = _cfg([_rung("a"), _rung("b")])
    state = LadderState.initial(cfg)
    state.record_round(0, {"a": _round(10, 0.9, 0.80), "b": _round(5, 0.5, 0.40)})
    path = tmp_path / "ladder_state.json"
    state.save(path)
    assert path.exists()

    reloaded = LadderState.load(path)
    assert reloaded.status("a") == state.status("a")
    assert reloaded.status("b") == state.status("b")
    assert reloaded.consec("a") == state.consec("a")
    assert reloaded.consec("b") == state.consec("b")


def test_ladder_state_persist_failure_raises(tmp_path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permission bits")
    cfg = _cfg([_rung("a")])
    state = LadderState.initial(cfg)
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    os.chmod(ro_dir, stat.S_IREAD | stat.S_IEXEC)
    try:
        with pytest.raises(LadderStateError):
            state.save(ro_dir / "ladder_state.json")
    finally:
        os.chmod(ro_dir, stat.S_IRWXU)
