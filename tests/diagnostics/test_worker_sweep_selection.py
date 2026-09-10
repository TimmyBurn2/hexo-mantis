# >300 justify (R8): the knee rule's ONE oracle — the band, the pick, the refusals, the exit
# codes, the re-derivation from a written report and the per-rung noise term. A row that moves
# the pick and a row that refuses to pick must read the same fixture and the same arithmetic.
"""The knee rule, its arithmetic, and the exit codes a sitting gates on.

The rule is fixed before any number exists: the SMALLEST rung within 95 percent of the best
PASSING rung's throughput — smallest, not fastest, because a knee is about not paying for
workers that buy nothing, and passing, because a GROWING memory verdict is out however fast it
was. `--select-only` re-derives a pick from a written report THROUGH THE SAME PURE FUNCTION, so
a sitting record carries the derivation rather than the answer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.diagnostics import worker_sweep as ws


def _row(n_workers: int, value: float, verdict: str = ws.PLATEAU, *,
         rel_se: float | None = 0.0) -> dict:
    """A rung row as `RungResult.as_dict` writes it; `rel_se` is that rung's OWN noise."""
    def spread(v: float) -> dict:
        return {"min": v, "median": v, "max": v, "mean": v, "rel_se": rel_se, "n_rounds": 5}
    return {"n_workers": n_workers, "verdict": verdict, "moves_per_min": value,
            "games_per_min": value / 40.0, "refusal": None, "produced_by": "test",
            "rounds_total": 6, "rounds_measured": 5, "rounds_unmeasured": 0,
            "wall_sec": 720.0, "ranking_metric": "moves_per_min",
            "moves_per_min_spread": spread(value), "games_per_min_spread": spread(value / 40.0),
            "rung_peak_bytes": 1024 ** 3, "rounds": []}


def test_the_knee_picks_the_smallest_rung_within_the_band_not_the_fastest() -> None:
    rows = [_row(2, 30.0), _row(4, 39.9), _row(8, 41.2), _row(12, 41.5)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert selection["best"]["n_workers"] == 12
    assert selection["threshold"] == pytest.approx(41.5 * 0.95)
    assert [p["n_workers"] for p in selection["within"]] == [4, 8, 12]
    assert selection["picked"] == 4, (
        "a knee rule that returned the best rung would be an argmax with extra steps"
    )


def test_a_rung_just_below_the_band_is_excluded() -> None:
    rows = [_row(2, 94.9), _row(4, 100.0)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert selection["picked"] == 4
    assert [p["n_workers"] for p in selection["within"]] == [4]


def test_a_rung_exactly_on_the_band_is_included() -> None:
    """WITHIN 95 percent: excluding equality would make the printed threshold unused."""
    rows = [_row(2, 95.0), _row(4, 100.0)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert selection["picked"] == 2


def test_only_plateau_rungs_are_in_the_passing_set() -> None:
    rows = [_row(2, 30.0), _row(4, 90.0, ws.GROWING), _row(8, 100.0, ws.OOM),
            _row(12, 31.0, ws.REFUSED), _row(14, 29.0, ws.RUNG_ERROR)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert [p["n_workers"] for p in selection["passing"]] == [2]
    assert selection["picked"] == 2


def test_no_passing_rung_yields_no_pick_and_says_why() -> None:
    rows = [_row(2, 30.0, ws.GROWING), _row(4, 40.0, ws.OOM)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert selection["picked"] is None
    assert "PLATEAU" in selection["reason"]


def test_the_ranking_metric_is_PINNED_and_cannot_be_swapped_by_a_caller() -> None:
    """Only the RANKING metric is single-valued, and it is pinned in SOURCE: with `games_per_min`
    and rounds shorter than a game, every rung ranked 0.000 and the knee picked the smallest rung
    at rc 0 while the moves column said the top rung was 3.7x faster."""
    rows = [_row(2, 30.0), _row(4, 100.0)]
    assert ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")["picked"] == 4
    with pytest.raises(ValueError, match="pre-registered"):
        ws.select_knee(rows, knee_pct=95.0, metric="games_per_min")


def test_the_knee_percent_is_taken_from_SOURCE_in_both_modes() -> None:
    """A report is a file, and `--select-only` used to re-derive the pick from the REPORT's own
    `knee_pct`: editing one integer moved the pick, at rc 0, in the tool's own arithmetic."""
    rows = [_row(2, 900.0), _row(4, 1400.0), _row(8, 1450.0)]
    assert ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")["picked"] == 4
    for edited in (60.0, 100.0, 94.9):
        with pytest.raises(ValueError, match="R309"):
            ws.select_knee(rows, knee_pct=edited, metric="moves_per_min")


def test_a_rung_row_that_smuggles_n_workers_1_is_refused_not_picked() -> None:
    """A hand-written three-key dict used to yield the one pick value the rule REJECTS, at rc 0."""
    with pytest.raises(ValueError, match="REJECTS"):
        ws.select_knee([_row(1, 5.0)], knee_pct=95.0, metric="moves_per_min")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_ranking_value_is_refused(value: float) -> None:
    """`json.loads` accepts `NaN`/`Infinity`: a NaN row vanished from the set, an inf row won."""
    with pytest.raises(ValueError, match="NaN"):
        ws.select_knee([_row(2, 30.0), _row(4, value)], knee_pct=95.0, metric="moves_per_min")


def test_an_identically_zero_ranking_column_is_refused_not_picked_from() -> None:
    with pytest.raises(ValueError, match="cannot order the ladder"):
        ws.select_knee([_row(2, 0.0), _row(4, 0.0)], knee_pct=95.0, metric="moves_per_min")


def test_an_empty_report_says_so_instead_of_talking_about_memory() -> None:
    """`rungs: []` used to print a statement about MEMORY for a document with no rungs at all."""
    selection = ws.select_knee([], knee_pct=95.0, metric="moves_per_min")
    assert selection["picked"] is None
    assert "NO RUNGS" in selection["reason"]


def test_the_selection_block_names_the_rungs_that_did_not_pass() -> None:
    """A reader of the quoted arithmetic alone would otherwise see `PICK = 2` and no OOM."""
    rows = [_row(2, 30.0), _row(4, 40.0, ws.GROWING), _row(8, 50.0, ws.OOM)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert set(selection["notes"]) == {ws.GROWING, ws.OOM} - {ws.GROWING} | {ws.OOM}
    assert ws.OOM in selection["notes"]


def test_rc_is_zero_when_a_pick_was_made() -> None:
    rows = [_row(2, 30.0)]
    report = {"rungs": rows, "selection": ws.select_knee(rows, knee_pct=95.0,
                                                         metric="moves_per_min")}
    assert ws.rc_for(report) == 0


def test_rc_is_one_when_rungs_were_measurable_and_none_passed() -> None:
    rows = [_row(2, 30.0, ws.GROWING), _row(4, 40.0, ws.OOM)]
    report = {"rungs": rows, "selection": ws.select_knee(rows, knee_pct=95.0,
                                                         metric="moves_per_min")}
    assert ws.rc_for(report) == 1


def test_rc_is_refused_when_nothing_was_measurable_at_all() -> None:
    """A host with no CUDA counters cannot answer the question; rc 1 would blame the card."""
    rows = [_row(2, 30.0, ws.REFUSED), _row(4, 40.0, ws.REFUSED)]
    report = {"rungs": rows, "selection": ws.select_knee(rows, knee_pct=95.0,
                                                         metric="moves_per_min")}
    assert ws.rc_for(report) == ws.RC_REFUSED == 2


def test_the_selection_screen_prints_every_input_the_rule_ran_on(capsys) -> None:
    rows = [_row(2, 30.0), _row(4, 39.9), _row(8, 41.2)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    ws.render_selection(selection, __import__("sys").stdout)
    text = capsys.readouterr().out
    for needle in ("knee_pct=95", "passing rungs", "best passing", "threshold =",
                   "R330(d) noise term", "per-rung rel-SE", "at or above adjusted threshold", "PICK = 4"):
        assert needle in text, f"the selection screen omits {needle!r} — a sitting record that "\
            "carries the answer without the arithmetic cannot be checked by its reader"


def test_select_only_re_derives_the_same_pick_from_a_written_report(tmp_path: Path,
                                                                    capsys) -> None:
    rows = [_row(2, 30.0), _row(4, 39.9), _row(8, 41.2)]
    report = {
        "tool": ws.TOOL, "rungs": rows,
        "plan": {"knee_pct": 95.0, "metric": "moves_per_min"},
        "provenance": {"produced_by": "run5@abc1234"},
        "selection": ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min"),
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert ws.main(["--select-only", str(path)]) == 0
    text = capsys.readouterr().out
    assert "PICK = 4" in text
    assert "run5@abc1234" in text, "a re-derived pick must still name what produced the numbers"


def test_select_only_refuses_to_be_given_inputs_it_does_not_read(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")
    assert ws.main(["--select-only", str(path), "--config", "configs/run6.yaml"]) \
        == ws.RC_REFUSED


def test_select_only_refuses_a_report_it_cannot_read(tmp_path: Path) -> None:
    path = tmp_path / "not-a-report.json"
    path.write_text("{\"rungs\": []}", encoding="utf-8")
    assert ws.main(["--select-only", str(path)]) == ws.RC_REFUSED


def test_the_driver_refuses_to_default_either_of_its_two_inputs() -> None:
    """A config this tool picked would measure a program nobody asked about."""
    assert ws.main([]) == ws.RC_REFUSED
    assert ws.main(["--config", "configs/run6.yaml"]) == ws.RC_REFUSED
    assert ws.main(["--plan", "tools/worker_sweep_plan.toml"]) == ws.RC_REFUSED


@pytest.mark.parametrize("verdict", [ws.RUNG_ERROR, ws.PRODUCER_DEAD])
def test_a_ladder_of_unmeasurable_rungs_is_rc_2_not_rc_1(verdict: str) -> None:
    """A rung that ERRORED or whose feeder DIED measured no memory, so it is not evidence that
    the card failed; rc 1 claims every measurable rung was GROWING or OOM."""
    rows = [_row(2, 30.0, verdict), _row(4, 40.0, verdict)]
    report = {"rungs": rows,
              "selection": ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")}
    assert ws.rc_for(report) == ws.RC_REFUSED
    assert verdict in report["selection"]["notes"]


def test_the_notes_print_on_the_NO_PICK_path_too(capsys) -> None:
    """The notes must print on the NO-PICK path: the line sat BELOW `render_selection`'s early
    return, unreachable in exactly the run where a reader most needs to know WHY."""
    rows = [_row(2, 30.0, ws.GROWING), _row(4, 40.0, ws.OOM)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert selection["picked"] is None
    ws.render_selection(selection, __import__("sys").stdout)
    text = capsys.readouterr().out
    assert "PICK = none" in text
    assert ws.OOM in text and "what the ladder DID return" in text


def test_the_widening_uses_the_MAX_rel_se_over_the_passing_set_not_the_best_rungs_or_a_scalar():
    """PLANTED BREAK. Rung 4 — the best — is quiet; rung 2 sits just below the 95 % band, noisy
    at 2 %. A scalar floor, the best rung's own noise, or a MIN over the set all widen by 0 and
    leave the pick at 4; the MAX over the candidate set widens `3 × 0.02 × 100 = 6`, the
    threshold falls 95 → 89, and the pick moves to 2."""
    rows = [_row(2, 91.0, rel_se=0.02), _row(4, 100.0, rel_se=0.0)]
    sel = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert sel["noise_rel_se_max"] == 0.02 and sel["noise_source_rung"] == 2
    assert sel["noise_adjustment"] == pytest.approx(6.0)
    assert sel["adjusted_threshold"] == pytest.approx(89.0)
    assert sel["picked"] == 2, sel


def test_a_noisy_rung_that_did_not_pass_is_not_a_candidate_and_does_not_widen():
    """The candidate set is the PASSING set, so a GROWING rung's noise widens nothing."""
    rows = [_row(2, 91.0, rel_se=0.0), _row(4, 100.0, rel_se=0.0),
            _row(8, 120.0, ws.GROWING, rel_se=0.5)]
    sel = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert sel["noise_rel_se_max"] == 0.0 and sel["noise_adjustment"] == 0.0
    assert sel["picked"] == 4 and "8" not in sel["per_rung_rel_se"]


def test_a_rung_without_its_own_measured_rel_se_is_refused_never_defaulted():
    """No scalar to pass and no default to fall to: a `rel_se: None` spread and a row with no
    spread block at all are both refused BY RUNG, with the metric named."""
    with pytest.raises(ValueError, match=r"rung 2 carries no measured rel_se for 'moves_per_min'"):
        ws.select_knee([_row(2, 30.0, rel_se=None)], knee_pct=95.0, metric="moves_per_min")
    legacy = _row(4, 30.0)
    del legacy["moves_per_min_spread"]
    with pytest.raises(ValueError, match="rung 4 carries no measured rel_se"):
        ws.select_knee([_row(2, 30.0), legacy], knee_pct=95.0, metric="moves_per_min")


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), True])
def test_a_non_finite_negative_or_boolean_rel_se_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="rel_se"):
        ws.select_knee([_row(2, 30.0, rel_se=bad)], knee_pct=95.0, metric="moves_per_min")


def test_the_selection_block_carries_every_rungs_rel_se_so_the_widening_is_re_derivable():
    """The block carries every passing rung's rel-SE, which was the max, and the adjustment."""
    rows = [_row(2, 30.0, rel_se=0.01), _row(4, 39.9, rel_se=0.03), _row(8, 41.2, rel_se=0.02)]
    sel = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert sel["per_rung_rel_se"] == {"2": 0.01, "4": 0.03, "8": 0.02}
    assert sel["noise_source_rung"] == 4
    assert sel["noise_adjustment"] == pytest.approx(3 * 0.03 * 41.2)
    assert sel["picked"] == 4  # 39.9 clears 39.14 - 3.708; 30.0 does not


def test_rung_result_derives_rel_se_from_its_own_scored_rounds() -> None:
    """The producer half: `RungResult.spread` states the rung's noise over its SCORED rounds,
    `None` when a single round cannot, and `select_knee` refuses that `None` as not-zero."""
    rates = [100, 110, 90, 105, 95]  # moves over a 60 s wall → moves_per_min == moves
    rounds = tuple(
        ws.RoundReading(index=i, warmup=(i == 0), wall_sec=60.0, games=1, moves=m, available=True,
                        sampled_peak_bytes=1, allocator_peak_bytes=1, card_samples=1)
        for i, m in enumerate([999, *rates]))
    rung = ws.RungResult(n_workers=4, verdict=ws.PLATEAU, rounds=rounds, refusal=None,
                         produced_by="t")
    spread = rung.spread("moves_per_min")
    assert spread["n_rounds"] == 5 and spread["mean"] == 100.0
    # sample std of the five rates is sqrt(62.5); / sqrt(5) / 100
    assert spread["rel_se"] == pytest.approx((62.5 ** 0.5) / (5 ** 0.5) / 100.0)
    one = ws.RungResult(n_workers=4, verdict=ws.PLATEAU, rounds=rounds[:2], refusal=None,
                        produced_by="t")
    assert one.spread("moves_per_min")["rel_se"] is None
    with pytest.raises(ValueError, match="rel_se"):
        ws.select_knee([one.as_dict("moves_per_min")], knee_pct=95.0, metric="moves_per_min")
    # and a full rung's own row selects through the same path the report writes
    sel = ws.select_knee([rung.as_dict("moves_per_min")], knee_pct=95.0, metric="moves_per_min")
    assert sel["per_rung_rel_se"] == {"4": pytest.approx(spread["rel_se"])}
