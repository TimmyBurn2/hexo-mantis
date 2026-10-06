"""The run monitor's rules: the two-read exam floors, the bands, the gap rule and the rates, each with its planted breaks."""
from __future__ import annotations

import importlib
from typing import Any

import pytest

from mantis.util.loadpkg import load_tools_package

load_tools_package("run_monitor")
rules = importlib.import_module("run_monitor.rules")


def _exam(mean: float, floor: float) -> dict[str, Any]:
    return {"calibrated_mean": mean, "floor": floor, "holds": mean >= floor}


_GOOD = {"T4_V": _exam(0.29, 0.154), "DEF_V_att": _exam(0.21, 0.100)}


def _verdict(check: Any, exams: dict[str, Any], misses: list[str], armed: list[str], *, floors_live: bool = True,
             bands_live: bool = True) -> dict[str, Any]:
    return check(exams, {"misses": misses}, armed=armed, floors_live=floors_live, bands_live=bands_live,
                 two_read_bands=frozenset(), armed_bands=[])


_LOW = {**_GOOD, "DEF_V_att": _exam(0.0995, 0.100)}
_LOW_ROW = "DEF_V_att calibrated 0.0995 below the floor 0.1"


def _assert_the_two_read_rule(check: Any) -> None:
    """One miss arms, the same floor's next miss fires, a pass or the other floor disarms; bands halt; pre-live rows report."""
    assert _verdict(check, _GOOD, [], [])["fired"] == []
    assert check(_GOOD, {"misses": [], "not_measured": "the ring was gone"}, armed=[], floors_live=True,
                 bands_live=True, two_read_bands=frozenset(), armed_bands=[])["fired"] == []
    first = _verdict(check, _LOW, [], [])
    assert first["fired"] == [] and first["armed"] == ["DEF_V_att"] and first["reported"] == [_LOW_ROW]
    assert _verdict(check, _LOW, [], ["DEF_V_att"])["fired"] == [f"{_LOW_ROW}, its second miss in a row"]
    assert _verdict(check, _GOOD, [], ["DEF_V_att"])["armed"] == [], "a pass disarms"
    assert _verdict(check, _LOW, [], ["T4_V"])["fired"] == [], "another floor's miss is not this floor's second"
    assert _verdict(check, _GOOD, ["cap_rate: 0.07 not lt 0.05"], [])["fired"] == ["ring band cap_rate: 0.07 not lt 0.05"]
    early = _verdict(check, _GOOD, ["cap_rate: 0.07 not lt 0.05"], [], bands_live=False)
    assert early["fired"] == [] and early["reported"] == ["ring band cap_rate: 0.07 not lt 0.05"]
    before = _verdict(check, _LOW, [], [], floors_live=False)
    assert before["fired"] == [] and before["armed"] == [] and before["floors_live"] is False
    assert _verdict(check, _GOOD, [], [], floors_live=False)["floors_live"] is True, "the first pass makes floors live"
    unread = {**_LOW, "T4_V": {"floor": 0.154, "calibrated_mean": None, "holds": None}}
    assert _verdict(check, unread, [], ["T4_V"])["armed"] == ["DEF_V_att", "T4_V"], "an unread floor keeps its arm"
    assert _verdict(check, {**_GOOD, "T4_V": unread["T4_V"]}, [], [], floors_live=False)["floors_live"] is False


_BROKEN = {
    "exams ignored": lambda e, b, **k: rules.verdict({}, b, **k),
    "bands ignored": lambda e, b, **k: rules.verdict(e, {"misses": []}, **k),
    "one read halts": lambda e, b, **k: rules.verdict(e, b, **{**k, "armed": sorted(e)}),
    "a pass never disarms": lambda e, b, **k: {**rules.verdict(e, b, **k), "armed": sorted(
        set(rules.verdict(e, b, **k)["armed"]) | set(k["armed"]))},
    "floors live from the start": lambda e, b, **k: rules.verdict(e, b, **{**k, "floors_live": True}),
    "bands live from the start": lambda e, b, **k: rules.verdict(e, b, **{**k, "bands_live": True}),
    "an unread floor disarms": lambda e, b, **k: {**rules.verdict(e, b, **k), "armed": sorted(
        set(rules.verdict(e, b, **k)["armed"]) - {x for x, row in e.items() if row["holds"] is None})},
}


@pytest.mark.parametrize("planted", sorted(_BROKEN))
def test_the_two_read_rule_holds_and_each_planted_break_reds(planted: str) -> None:
    """PLANTED BREAKS: each checker that drops a half of the rule misses a row the rule decides."""
    _assert_the_two_read_rule(rules.verdict)
    with pytest.raises(AssertionError):
        _assert_the_two_read_rule(_BROKEN[planted])


_CAP = "cap_rate: 0.11 not lt 0.1"
_ONE_HOT = "one_hot_share_full: 0.5 not lt 0.47"


def _bands(check: Any, misses: list[str], armed_bands: list[str], *, bands_live: bool = True,
           unread: bool = False) -> dict[str, Any]:
    bands: dict[str, Any] = {"misses": misses, **({"not_measured": "the ring audit failed"} if unread else {})}
    return check(_GOOD, bands, armed=[], floors_live=True, bands_live=bands_live,
                 two_read_bands=frozenset({"cap_rate"}), armed_bands=armed_bands)


def _assert_the_two_read_bands(check: Any) -> None:
    """A two-read band's miss arms, its next miss fires, a pass or an unread audit decide as a floor's; others fire at once."""
    first = _bands(check, [_CAP], [])
    assert first["fired"] == [] and first["armed_bands"] == ["cap_rate"] and first["reported"] == [f"ring band {_CAP}"]
    assert _bands(check, [_CAP], ["cap_rate"])["fired"] == [f"ring band {_CAP}, its second miss in a row"]
    assert _bands(check, [], ["cap_rate"])["armed_bands"] == [], "a pass disarms"
    assert _bands(check, [], ["cap_rate"], unread=True)["armed_bands"] == ["cap_rate"], "an unread audit keeps the arm"
    assert _bands(check, [_ONE_HOT], [])["fired"] == [f"ring band {_ONE_HOT}"], "a one-read band fires at once"
    early = _bands(check, [_CAP], ["cap_rate"], bands_live=False)
    assert early["fired"] == [] and early["armed_bands"] == [] and early["reported"] == [f"ring band {_CAP}"]


_BROKEN_BANDS = {
    "a two-read band fires at once": lambda e, b, **k: rules.verdict(e, b, **{**k, "two_read_bands": frozenset()}),
    "a pass never disarms": lambda e, b, **k: {**rules.verdict(e, b, **k), "armed_bands": sorted(
        set(rules.verdict(e, b, **k)["armed_bands"]) | set(k["armed_bands"]))},
    "an unread audit disarms": lambda e, b, **k: rules.verdict(e, {"misses": b["misses"]}, **k),
    "every band reads twice": lambda e, b, **k: rules.verdict(e, b, **{**k, "two_read_bands": frozenset(
        rules.band_key(m) for m in b["misses"]) | k["two_read_bands"]}),
}


@pytest.mark.parametrize("planted", sorted(_BROKEN_BANDS))
def test_a_two_read_band_halts_on_its_second_miss_and_each_planted_break_reds(planted: str) -> None:
    """PLANTED BREAKS: each checker that drops a half of the band rule misses a row the rule decides."""
    _assert_the_two_read_bands(rules.verdict)
    with pytest.raises(AssertionError):
        _assert_the_two_read_bands(_BROKEN_BANDS[planted])


def test_a_band_trend_is_the_change_since_the_last_read_save_and_a_missing_reading_is_no_zero() -> None:
    keys = frozenset({"cap_rate", "draw_share"})
    out = rules.band_trends({"cap_rate": 0.0545, "draw_share": 0.0}, {"cap_rate": 0.052, "draw_share": 0.0}, 69000, keys)
    assert out["cap_rate"]["per_save"] == pytest.approx(0.0025) and out["cap_rate"]["previous_step"] == 69000
    assert out["draw_share"]["per_save"] == 0.0
    first = rules.band_trends({"cap_rate": 0.0545}, None, None, keys)
    assert "per_save" not in first["cap_rate"] and first["cap_rate"]["value"] == 0.0545
    assert "NOT MEASURED" in first["draw_share"]["note"] and first["draw_share"]["value"] is None
    assert "per_save" not in rules.band_trends({"cap_rate": float("nan")}, {"cap_rate": 0.05}, 3000, keys)["cap_rate"]


def _assert_the_gap_rule(rule: Any) -> None:
    assert rule(0.06, [], 3000, 0.05) == {"gap": 0.06, "line": 0.05, "over": [3000], "fired": False}
    assert rule(0.06, [3000], 6000, 0.05)["fired"] is True
    assert rule(0.05, [3000], 6000, 0.05)["over"] == [], "at the line is not above it"
    assert rule(None, [3000], 6000, 0.05)["over"] == [3000], "an unread gap leaves the run"


def test_the_gap_rule_fires_at_two_saves_in_a_row_and_its_planted_break_reds() -> None:
    """PLANTED BREAK: a rule that fires on one save above the line reds."""
    _assert_the_gap_rule(rules.gap_rule)
    with pytest.raises(AssertionError):
        _assert_the_gap_rule(lambda gap, over, step, line: {**rules.gap_rule(gap, over, step, line),
                                                            "fired": gap is not None and gap > line})


def test_the_rates_read_the_idle_windows_apart_and_a_missing_counter_is_no_zero() -> None:
    rows = [{"ts": 0.0, "games_total": 0, "positions_produced_total": 0, "step": 0},
            {"ts": 1800.0, "games_total": 100, "positions_produced_total": 9000, "step": 240},
            {"ts": 3600.0, "games_total": 120, "positions_produced_total": 10800, "step": 288}]
    out = rules.rates(rows, 0.0, 3600.0, busy=[(1900.0, 3500.0)])
    assert out["games_per_h"] == pytest.approx(120.0) and out["idle_games_per_h"] == pytest.approx(200.0)
    assert out["idle_h"] == pytest.approx(0.5) and out["steps_per_h"] == pytest.approx(288.0)
    assert "NOT MEASURED" in rules.rates(rows[:1], 0.0, 3600.0, busy=[])["note"]
    assert rules.counter_row({"event": "iteration_complete", "ts": 1.0, "games_total": 3, "step": 9}) is None
