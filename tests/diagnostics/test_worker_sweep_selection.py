"""P7/P8 — the knee rule, its arithmetic, and the exit codes a sitting gates on.

R309(f) fixes the selection rule before any number exists: *the smallest rung within 95 percent
of the best PASSING rung's throughput*. Two words in that sentence carry the whole rule and each
has its own row here — **smallest** (not fastest: the point of a knee is to stop paying for
workers that buy nothing) and **PASSING** (a rung with a GROWING memory verdict is not in the
set, however fast it was).

The arithmetic is printed with its inputs, and `--select-only` re-derives a pick from a written
report THROUGH THE SAME PURE FUNCTION. That is what makes the pick checkable by someone who was
not at the box: the sitting record carries the derivation, not the answer.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mantis.diagnostics import worker_sweep as ws


def _row(n_workers: int, value: float, verdict: str = ws.PLATEAU) -> dict:
    return {"n_workers": n_workers, "verdict": verdict, "moves_per_min": value,
            "games_per_min": value / 40.0, "refusal": None, "produced_by": "test",
            "rounds_total": 6, "rounds_measured": 5, "rounds_unmeasured": 0,
            "wall_sec": 720.0, "ranking_metric": "moves_per_min",
            "rung_peak_bytes": 1024 ** 3, "rounds": []}


# ══ P7 — SMALLEST within the band, not fastest ═══════════════════════════════════════════
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
    """The rule says WITHIN 95 percent. A boundary that excluded the equality case would make
    the printed threshold a number the rule does not actually use."""
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
    """Both figures are recorded for every rung; only the RANKING is single-valued — and it is
    pinned in SOURCE, not in the plan.

    THE DEFECT THIS CLOSES, measured: with `metric = "games_per_min"` and rounds shorter than a
    game, every rung ranks at 0.000, the knee picks the SMALLEST rung at rc 0, and the ladder-stop
    line reads "gains no longer persist" — while the moves column says the top rung is 3.7x
    faster. One token in a plan file made the pick arbitrary with every check green. A
    pre-registration with a measured basis (DESIGN amendment A1) and no enforcement is a
    preference."""
    rows = [_row(2, 30.0), _row(4, 100.0)]
    assert ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")["picked"] == 4
    with pytest.raises(ValueError, match="pre-registered"):
        ws.select_knee(rows, knee_pct=95.0, metric="games_per_min")


def test_the_knee_percent_is_taken_from_SOURCE_in_both_modes() -> None:
    """A report is a file, and a file can be edited between the drive that wrote it and the
    reader that quotes it. `--select-only` used to re-derive the pick from the REPORT's own
    `plan.knee_pct`: editing one integer printed `PICK = 2` (knee 60) or `PICK = 8` (knee 100)
    at rc 0, in the tool's own arithmetic. That is the hole the plan loader closed, re-opened in
    the mode a sitting record quotes."""
    rows = [_row(2, 900.0), _row(4, 1400.0), _row(8, 1450.0)]
    assert ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")["picked"] == 4
    for edited in (60.0, 100.0, 94.9):
        with pytest.raises(ValueError, match="R309"):
            ws.select_knee(rows, knee_pct=edited, metric="moves_per_min")


def test_a_rung_row_that_smuggles_n_workers_1_is_refused_not_picked() -> None:
    """A three-key hand-written dict used to yield `PICK = 1` — the one value R309(f) REJECTS —
    at rc 0, printed with the ruling's own arithmetic."""
    with pytest.raises(ValueError, match="REJECTS"):
        ws.select_knee([_row(1, 5.0)], knee_pct=95.0, metric="moves_per_min")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_ranking_value_is_refused(value: float) -> None:
    """`json.loads` accepts `NaN` and `Infinity`. A NaN row used to vanish from the knee set with
    no refusal; an inf row used to capture the pick."""
    with pytest.raises(ValueError, match="NaN"):
        ws.select_knee([_row(2, 30.0), _row(4, value)], knee_pct=95.0, metric="moves_per_min")


def test_an_identically_zero_ranking_column_is_refused_not_picked_from() -> None:
    with pytest.raises(ValueError, match="cannot order the ladder"):
        ws.select_knee([_row(2, 0.0), _row(4, 0.0)], knee_pct=95.0, metric="moves_per_min")


def test_an_empty_report_says_so_instead_of_talking_about_memory() -> None:
    """`rungs: []` used to print "no rung PASSED (a PLATEAU memory verdict is required)" — a
    statement about MEMORY for a document that contains no rungs at all."""
    selection = ws.select_knee([], knee_pct=95.0, metric="moves_per_min")
    assert selection["picked"] is None
    assert "NO RUNGS" in selection["reason"]


def test_the_selection_block_names_the_rungs_that_did_not_pass() -> None:
    """With an OOM at rung 8 and passes elsewhere, a reader of the QUOTED ARITHMETIC alone would
    otherwise see `PICK = 2` with no mention of the OOM."""
    rows = [_row(2, 30.0), _row(4, 40.0, ws.GROWING), _row(8, 50.0, ws.OOM)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert set(selection["notes"]) == {ws.GROWING, ws.OOM} - {ws.GROWING} | {ws.OOM}
    assert ws.OOM in selection["notes"]


# ══ P8 — the exit codes ══════════════════════════════════════════════════════════════════
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
    """A host with no CUDA counters cannot answer the question that was asked. `2` says that;
    `1` would say the card failed, which is a different and false claim."""
    rows = [_row(2, 30.0, ws.REFUSED), _row(4, 40.0, ws.REFUSED)]
    report = {"rungs": rows, "selection": ws.select_knee(rows, knee_pct=95.0,
                                                         metric="moves_per_min")}
    assert ws.rc_for(report) == ws.RC_REFUSED == 2


# ══ the arithmetic is printed, and it is re-derivable off-box ════════════════════════════
def test_the_selection_screen_prints_every_input_the_rule_ran_on(capsys) -> None:
    rows = [_row(2, 30.0), _row(4, 39.9), _row(8, 41.2)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    ws.render_selection(selection, __import__("sys").stdout)
    text = capsys.readouterr().out
    for needle in ("knee_pct=95", "passing rungs", "best passing", "threshold =",
                   "at or above threshold", "PICK = 4"):
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
    assert ws.main(["--select-only", str(path), "--config", "configs/run5.yaml"]) \
        == ws.RC_REFUSED


def test_select_only_refuses_a_report_it_cannot_read(tmp_path: Path) -> None:
    path = tmp_path / "not-a-report.json"
    path.write_text("{\"rungs\": []}", encoding="utf-8")
    assert ws.main(["--select-only", str(path)]) == ws.RC_REFUSED


def test_the_driver_refuses_to_default_either_of_its_two_inputs() -> None:
    """A config this tool picked would measure a program nobody asked about; a plan it picked
    would be a pre-registration nobody wrote."""
    assert ws.main([]) == ws.RC_REFUSED
    assert ws.main(["--config", "configs/run5.yaml"]) == ws.RC_REFUSED
    assert ws.main(["--plan", "tools/worker_sweep_plan.toml"]) == ws.RC_REFUSED


@pytest.mark.parametrize("verdict", [ws.RUNG_ERROR, ws.PRODUCER_DEAD])
def test_a_ladder_of_unmeasurable_rungs_is_rc_2_not_rc_1(verdict: str) -> None:
    """A rung that ERRORED or whose feeder DIED did not measure memory, so it is not evidence
    that the card failed. rc 1 says "every measurable rung was GROWING or OOM"; the block's
    Phase W posture branches on that distinction, and rc 1 would send a sitting to the wrong arm."""
    rows = [_row(2, 30.0, verdict), _row(4, 40.0, verdict)]
    report = {"rungs": rows,
              "selection": ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")}
    assert ws.rc_for(report) == ws.RC_REFUSED
    assert verdict in report["selection"]["notes"]


def test_the_notes_print_on_the_NO_PICK_path_too(capsys) -> None:
    """The line was added so a reader who quotes the arithmetic alone sees that the ladder had a
    failing rung — and it sat BELOW `render_selection`'s early return, i.e. unreachable in
    exactly the run where the reader most needs it (no pick, and the question is WHY)."""
    rows = [_row(2, 30.0, ws.GROWING), _row(4, 40.0, ws.OOM)]
    selection = ws.select_knee(rows, knee_pct=95.0, metric="moves_per_min")
    assert selection["picked"] is None
    ws.render_selection(selection, __import__("sys").stdout)
    text = capsys.readouterr().out
    assert "PICK = none" in text
    assert ws.OOM in text and "what the ladder DID return" in text
