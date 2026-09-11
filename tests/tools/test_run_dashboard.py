"""The run dashboard reports what the record holds, and says so when the record holds nothing.

"Absent is not zero" applies to pixels: a panel with no producer at HEAD is drawn as a stated
gap, never as a zero. Every arm below is a way a report can quietly become fiction — a page
rendered from an empty file, a chart drawn from no series, a zero that is really an absence,
a panel banked without saying what would fill it.

It is tested despite gating nothing because it will be read at sittings, and a report is an
instrument.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = REPO_ROOT / "tools" / "run_dashboard.py"


def _load():
    """Load the tool by path, since `tools/` is not a package and CWD is not on `sys.path`."""
    spec = importlib.util.spec_from_file_location("run_dashboard", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: `@dataclass` resolves a class's module through `sys.modules`,
    # so a module executed while absent from it raises inside `dataclasses`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dash = _load()


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_an_eventless_record_REFUSES_rather_than_rendering_a_clean_page(tmp_path: Path):
    """Prove an eventless record refuses: a page built from nothing looks like a healthy one."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n   \nnot json at all\n", encoding="utf-8")
    with pytest.raises(dash.EmptyRunRecord):
        dash.load_record(empty)


def test_a_banked_panel_names_the_producer_that_would_fill_it(tmp_path: Path):
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    assert page.count("BANKED — no producer at HEAD") == len(dash.BANKED_PANELS)
    for name, why in dash.BANKED_PANELS.items():
        assert "NO PRODUCER" in why, f"{name}'s bank does not say what is missing"
        assert "Filling this panel needs" in why, (
            f"{name}'s bank does not name what would fill it — a gap with no owner is a "
            "complaint, not a finding"
        )


def test_an_undeclared_banked_panel_is_refused():
    """Prove an undeclared banked panel is refused, so no gap appears without a stated reason."""
    with pytest.raises(dash.UnknownPanel):
        dash.banked_block("a panel nobody declared")


def test_no_chart_is_drawn_from_a_record_with_no_series(tmp_path: Path):
    """Prove no chart is drawn from a record with no series."""
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    assert "<figure>" not in page


def test_every_zero_on_a_measurement_free_page_is_labelled_an_absence(tmp_path: Path):
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    naked = [row for row in re.findall(r"<tr>.*?</tr>", page)
             if "<td>0</td>" in row and "ABSENT" not in row]
    assert naked == [], f"zero(s) drawn without an absence label: {naked[:3]}"


def test_a_real_series_IS_drawn_and_carries_its_own_min_max_last(tmp_path: Path):
    """Control: a real series is drawn and carries its own min/max/last."""
    rows = [{"event": "iteration_complete", "step": i, "games_per_hour": 10.0 + i,
             "sims_per_sec": 100.0 * i} for i in range(1, 6)]
    page = dash.render(dash.load_record(_write(tmp_path, rows)), "t")
    assert "<figure>" in page and "<polyline" in page
    assert "min 11" in page and "max 15" in page, (
        "a drawn series must state its own min/max/last beside it — a line with no numbers is "
        "a shape, and a reader cannot check a shape against anything"
    )


def test_a_broken_round_is_not_read_as_a_round_that_played_nothing(tmp_path: Path):
    """Prove a broken round renders as broken: `games_total: null` means killed, not zero games."""
    rows = [{"event": "eval_round_complete", "round_id": "r1", "step": 1, "wall_sec": 3.0,
             "games_total": None, "promoted": None, "wr_sealbot": None}]
    page = dash.render(dash.load_record(_write(tmp_path, rows)), "t")
    assert "None" in page, "the null must survive to the page as a null"
    assert "BROKEN round" in page, "the page must say what a null games_total means"


def test_the_absence_of_an_event_is_reported_as_an_absence_not_a_zero(tmp_path: Path):
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    assert "is an ABSENCE" in page
    assert "monitor_gates" in page, "an absent panel must still name the event it reads"


def test_the_page_is_self_contained_no_network_no_script(tmp_path: Path):
    """Prove the page is self-contained: a page that fetches can render differently tomorrow."""
    rows = [{"event": "iteration_complete", "step": i, "games_per_hour": float(i)}
            for i in range(1, 4)]
    page = dash.render(dash.load_record(_write(tmp_path, rows)), "t")
    for forbidden in ("<script", "http://", "https://", "<iframe", "src=\"//"):
        assert forbidden not in page, f"the page references {forbidden!r} — it is not self-contained"


def test_the_tools_self_test_passes():
    """Prove the tool's own self-test passes, driven from the suite so a broken control cannot ship."""
    assert dash.self_test() == 0


def test_the_panel_roster_matches_the_ruling(tmp_path: Path):
    """Prove the page carries one section per declared panel, count asserted beside the titles.

    Titles alone would pass while an extra panel rode along unnamed.
    """
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    for title in ("Throughput", "Average sims/move", "Memory shares vs minted caps",
                  "Training losses", "Held-out loss", "Gate outcomes and floor refusals",
                  "Strength vs external rungs, with CIs", "Determinism hash", "Health",
                  "F-816-37 firings", "Mirror receipts", "alpha = 1.0 rows per 1,000"):
        assert f"<h2>{title}</h2>" in page, f"the ruling's {title!r} panel is not on the page"
    assert len(dash.PANELS) == 12


def test_the_mirror_panel_warns_at_two_unreceipted_bundles_and_not_at_one(tmp_path: Path):
    """R349(b): two missing intervals is a dashboard WARNING, one is the bundle just published."""
    warn = dash.render(dash.load_record(_write(tmp_path, [
        {"event": "resume_state_persisted", "step": 2000, "unreceipted_bundles": [1000, 2000]},
    ])), "t")
    assert "WARNING" in warn and "missed at least 2 intervals" in warn
    calm = dash.render(dash.load_record(_write(tmp_path, [
        {"event": "resume_state_persisted", "step": 2000, "unreceipted_bundles": [2000]},
    ])), "t")
    assert "WARNING" not in calm


def test_the_alpha_panel_draws_the_rate_and_an_absence_for_an_unproduced_block(tmp_path: Path):
    """R349(c): the per-1,000 count from step 0; `None` is no producer, never a zero."""
    page = dash.render(dash.load_record(_write(tmp_path, [
        {"event": "iteration_complete", "step": 0,
         "gumbel_alpha_full": {"rows": 0, "graph_rows": 512, "per_1000": 0.0}},
        {"event": "iteration_complete", "step": 10,
         "gumbel_alpha_full": {"rows": 1, "graph_rows": 8000, "per_1000": 0.125}},
    ])), "t")
    assert "0.125" in page and "<figure>" in page
    absent = dash.render(dash.load_record(_write(tmp_path, [
        {"event": "iteration_complete", "step": 10, "gumbel_alpha_full": None},
    ])), "t")
    assert "iteration_complete.gumbel_alpha_full" in absent and "0.125" not in absent


def test_the_firings_panel_draws_an_absence_when_it_was_given_no_record_dir(tmp_path: Path):
    """Prove the firings panel draws an absence with no record dir: a run with no firings and a
    run nobody pointed at the dumps must not render the same."""
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    assert "ABSENT — no run-record directory" in page
    assert "0 firings" not in page, "an unread firing count must never render as a zero"


def test_the_page_carries_no_absolute_home_path(tmp_path: Path):
    """Prove the page carries no absolute path: it names its source record by filename only."""
    events = _write(tmp_path, [{"event": "run_boot_identity", "run_id": "x"}])
    page = dash.render(dash.load_record(events), "t")
    assert str(tmp_path) not in page, "the page prints its source's absolute path"
    assert events.name in page, "…but it must still say WHICH record it read"
