"""The run dashboard reports what the record holds, and says so when the record holds nothing.

R333(d): *"a panel with no producer at HEAD is a banked finding, not an invented number —
'absent is not zero' applies to pixels."* This is that rule as a test. Every arm below is a way
a report can quietly become fiction: a page rendered from an empty file, a chart drawn from no
series, a zero that is really an absence, a panel banked without saying what would fill it.

WHY IT IS TESTED AT ALL, given it renders nothing anyone gates on: because it will be READ at
sittings, and the whole P1 packet exists because instruments that read zero for "absent" sent
the operator wrong numbers on artifacts they were reading at the time (F-01's false entropy
alert, F-05's `games_total` on a floor-refused round). A report is an instrument.
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
    """Load the tool by path — `tools/` is not a package (no `__init__.py`), so `from tools.…`
    is unresolvable under `uv run pytest` (a console script, which does not prepend CWD). This
    is the same loader `tests/tools/test_contract_doc_gate.py` uses and for the same reason."""
    spec = importlib.util.spec_from_file_location("run_dashboard", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves a class's module through `sys.modules`, so a
    # module executed while absent from it raises inside `dataclasses`, not in the tool.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dash = _load()


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_an_eventless_record_REFUSES_rather_than_rendering_a_clean_page(tmp_path: Path):
    """The load-bearing refusal. A page built from nothing looks exactly like a page built from
    a healthy run, and that is the one mistake a report must not be able to make."""
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
    """The banked set is DATA. A panel cannot be drawn as a gap without joining the table that
    says why, or the page could grow gaps nobody has to justify."""
    with pytest.raises(dash.UnknownPanel):
        dash.banked_block("a panel nobody declared")


def test_no_chart_is_drawn_from_a_record_with_no_series(tmp_path: Path):
    """A chart from nothing is the invented number in its most convincing form."""
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    assert "<figure>" not in page


def test_every_zero_on_a_measurement_free_page_is_labelled_an_absence(tmp_path: Path):
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    naked = [row for row in re.findall(r"<tr>.*?</tr>", page)
             if "<td>0</td>" in row and "ABSENT" not in row]
    assert naked == [], f"zero(s) drawn without an absence label: {naked[:3]}"


def test_a_real_series_IS_drawn_and_carries_its_own_min_max_last(tmp_path: Path):
    """The negative control. A report that refuses to draw anything is not honest, it is inert."""
    rows = [{"event": "iteration_complete", "step": i, "games_per_hour": 10.0 + i,
             "sims_per_sec": 100.0 * i} for i in range(1, 6)]
    page = dash.render(dash.load_record(_write(tmp_path, rows)), "t")
    assert "<figure>" in page and "<polyline" in page
    assert "min 11" in page and "max 15" in page, (
        "a drawn series must state its own min/max/last beside it — a line with no numbers is "
        "a shape, and a reader cannot check a shape against anything"
    )


def test_a_broken_round_is_not_read_as_a_round_that_played_nothing(tmp_path: Path):
    """AUDIT-1 F-28/B04 and R319(e)(i), carried into the page: `games_total: null` means the
    round was KILLED before it could report and `promoted: null` means no decision was taken.
    Rendering either as 0/false is the exact misreading RECAL-SITTING-3 §8.1 paid for."""
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
    """No server, no CDN, no JS — the whole point of 'file-based'. A page that fetches is a
    page that can render differently tomorrow, or not at all offline."""
    rows = [{"event": "iteration_complete", "step": i, "games_per_hour": float(i)}
            for i in range(1, 4)]
    page = dash.render(dash.load_record(_write(tmp_path, rows)), "t")
    for forbidden in ("<script", "http://", "https://", "<iframe", "src=\"//"):
        assert forbidden not in page, f"the page references {forbidden!r} — it is not self-contained"


def test_the_tools_self_test_passes():
    """The tool's own controls, driven from the suite so a broken control cannot ship green."""
    assert dash.self_test() == 0


def test_the_panel_roster_matches_the_ruling(tmp_path: Path):
    """R333(d) names the panels. The page must carry one section per panel, so a panel silently
    dropped is caught rather than simply not appearing."""
    page = dash.render(dash.load_record(_write(tmp_path, [{"event": "run_boot_identity"}])), "t")
    for title in ("Throughput", "Average sims/move", "Memory shares vs minted caps",
                  "Training losses", "Held-out loss", "Gate outcomes and floor refusals",
                  "Strength vs external rungs, with CIs", "Determinism hash", "Health"):
        assert f"<h2>{title}</h2>" in page, f"the ruling's {title!r} panel is not on the page"
    assert len(dash.PANELS) == 9


def test_the_page_carries_no_absolute_home_path(tmp_path: Path):
    """A rendered page is an artifact someone attaches to a record. CI gate 17 (rule 7) exists
    to keep absolute home paths out of a public repo, and a report must not be the thing that
    smuggles one in — the source is shown by NAME."""
    events = _write(tmp_path, [{"event": "run_boot_identity", "run_id": "x"}])
    page = dash.render(dash.load_record(events), "t")
    assert str(tmp_path) not in page, "the page prints its source's absolute path"
    assert events.name in page, "…but it must still say WHICH record it read"
