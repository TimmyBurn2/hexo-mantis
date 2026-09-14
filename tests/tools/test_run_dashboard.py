# >300 justify (R8): the page roster, the absence rules and the frozen command line are one
# contract; a test file per tier could pass while the tiers disagreed about a panel.
"""The run dashboard reports what the record holds, and says so when the record holds nothing:
a panel with no producer is a stated gap, a zero is labelled, a broken round is drawn as broken,
and the health badge is never green while an input is unread."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIM = REPO_ROOT / "tools" / "run_dashboard.py"


@pytest.fixture(scope="module")
def html(dashboard):
    return importlib.import_module("dashboard.html")


@pytest.fixture(scope="module")
def reader(dashboard):
    return importlib.import_module("dashboard.reader")


@pytest.fixture(scope="module")
def tier3(dashboard):
    return importlib.import_module("dashboard.tier3")


@pytest.fixture(scope="module")
def shim():
    spec = importlib.util.spec_from_file_location("run_dashboard", _SHIM)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, rows: list[dict], name: str = "events.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _page(html, reader, tmp_path: Path, rows: list[dict], ladder: dict | None = None,
          record_dir: Path | None = None) -> str:
    ladder_path = None
    if ladder is not None:
        ladder_path = tmp_path / "eval_ladder_state.json"
        ladder_path.write_text(json.dumps(ladder), encoding="utf-8")
    return html.render(reader.load_record(_write(tmp_path, rows), ladder_path, record_dir), "t")


def _round(idx: int, wr, games_total=56, promoted=None, lo=None, hi=None) -> dict:
    return {"event": "eval_round_complete", "round_id": f"r{idx:06d}_{idx * 1000}",
            "step": idx * 1000, "wall_sec": 600.0, "games_total": games_total,
            "promoted": promoted, "wr_sealbot": wr, "wr_sealbot_ci_lower": lo,
            "wr_sealbot_ci_upper": hi, "ts": 1000.0 + idx}


def _ladder(history: list[tuple[int, int, float]]) -> dict:
    return {"sealbot_d5": {"name": "sealbot_d5", "status": "active", "consec": 0,
                           "history": [{"round_idx": i, "games": g, "wr": wr, "ci_lo": None}
                                       for i, g, wr in history]}}


BOOT = [{"event": "run_boot_identity", "run_id": "x", "config_sha256": "abc", "ts": 1.0}]


def _hero_values(page: str) -> list[str]:
    hero = re.search(r'<section class="hero">(.*?)</section>', page, re.S).group(1)
    return re.findall(r'<span class="value">(.*?)</span>', hero, re.S)


def test_an_eventless_record_REFUSES_at_the_cli_rather_than_writing_a_clean_page(shim, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n   \nnot json at all\n", encoding="utf-8")
    out = tmp_path / "out.html"
    assert shim.main(["--events", str(empty), "--out", str(out)]) != 0
    assert not out.exists()


def test_the_dropped_panels_name_the_producer_that_would_fill_them(html, reader, tier3, tmp_path):
    page = _page(html, reader, tmp_path, BOOT)
    assert page.count("NO PRODUCER") >= len(tier3.DROPPED)
    for name, why in tier3.DROPPED.items():
        assert "NO PRODUCER" in why and "Filling this panel needs" in why, (
            f"{name}'s note does not name what would fill it — a gap with no owner is a "
            "complaint, not a finding")


def test_an_undeclared_dropped_panel_is_refused(tier3):
    with pytest.raises(tier3.UnknownPanel):
        tier3.dropped_note("a panel nobody declared")


def test_no_chart_is_drawn_from_a_record_with_no_series(html, reader, tmp_path):
    assert "<svg" not in _page(html, reader, tmp_path, BOOT)


def test_every_zero_on_a_measurement_free_page_is_labelled_an_absence(html, reader, tmp_path):
    page = _page(html, reader, tmp_path, BOOT)
    naked = [row for row in re.findall(r"<tr>.*?</tr>", page, re.S)
             if "<td>0</td>" in row and "absent" not in row.lower()]
    assert naked == [], f"zero(s) drawn without an absence label: {naked[:3]}"


def test_a_real_series_IS_drawn_and_carries_its_own_min_max_last(html, reader, tmp_path):
    rows = [{"event": "iteration_complete", "step": i, "games_per_hour": 10.0 + i,
             "sims_per_sec": 100.0 * i} for i in range(1, 6)]
    page = _page(html, reader, tmp_path, rows)
    assert "<svg" in page and "<polyline" in page
    assert "min 11" in page and "max 15" in page and "last 15" in page


def test_a_broken_round_renders_the_hollow_marker_and_the_legend_entry(html, reader, tmp_path):
    rows = [_round(1, 0.2), _round(2, None, games_total=None, promoted=False), _round(3, 0.25)]
    page = _page(html, reader, tmp_path, rows, _ladder([(1, 32, 0.2), (3, 32, 0.25)]))
    assert 'class="marker broken"' in page
    assert "broken round" in page and "games_total: null" in page
    assert page.count('class="marker broken"') == 1


def test_promotion_markers_are_drawn_and_no_decision_is_not_a_false(html, reader, tmp_path):
    rows = [_round(1, 0.2), _round(2, 0.3, promoted=True), _round(3, 0.1, promoted=False)]
    page = _page(html, reader, tmp_path, rows, _ladder([(1, 32, 0.2), (2, 32, 0.3), (3, 32, 0.1)]))
    assert page.count('class="marker promoted"') == 1
    assert page.count('class="marker rejected"') == 1


def test_the_absence_of_an_event_is_reported_as_an_absence_not_a_zero(html, reader, tmp_path):
    page = _page(html, reader, tmp_path, BOOT)
    assert "absence" in page.lower()
    assert "monitor_gates" in page, "an absent panel must still name the event it reads"


def test_the_page_is_self_contained_with_only_inline_script(html, reader, tmp_path):
    rows = [{"event": "iteration_complete", "step": i, "games_per_hour": float(i)}
            for i in range(1, 4)]
    page = _page(html, reader, tmp_path, rows)
    for forbidden in ("http://", "https://", "<iframe", 'src="', "@import", "url(", "<link"):
        assert forbidden not in page, f"the page references {forbidden!r}"
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    assert len(scripts) == 1 and len(scripts[0].encode("utf-8")) <= 6 * 1024


def test_the_panel_roster_matches_the_design_contract(html, reader, tmp_path):
    page = _page(html, reader, tmp_path, BOOT)
    hero = re.search(r'<section class="hero">(.*?)</section>', page, re.S).group(1)
    assert hero.count('<div class="cell') == 7
    tier2 = re.findall(r'<section class="panel tier2"[^>]*>\s*<h2>(.*?)</h2>', page)
    assert tier2 == ["Strength ladder", "Losses", "Self-play quality", "Data economy",
                     "Health timeline"]
    tier3 = set(re.findall(r'<details class="tier3"[^>]*>\s*<summary>(.*?)</summary>', page))
    assert {"Provenance", "Event inventory", "Memory shares vs minted caps", "Determinism hash",
            "Lifecycle", "Mirror receipts", "Eval child device memory", "alpha = 1.0 rows per 1,000",
            "Gate outcomes and floor refusals", "F-816-37 firings", "Rounds",
            "Learning details", "Absence notes"} <= tier3


def test_every_panel_carries_its_reads_line(html, reader, tmp_path):
    page = _page(html, reader, tmp_path, BOOT)
    panels = len(re.findall(r'<section class="panel tier2"', page))
    details = len(re.findall(r'<details class="tier3"', page))
    assert page.count('<p class="reads">') == panels + details


def test_the_mirror_warning_fires_at_two_unreceipted_bundles_and_not_at_one(html, reader, tmp_path):
    warn = _page(html, reader, tmp_path, [
        {"event": "resume_state_persisted", "step": 2000, "unreceipted_bundles": [1000, 2000]}])
    assert "missed at least 2 intervals" in warn
    calm = _page(html, reader, tmp_path, [
        {"event": "resume_state_persisted", "step": 2000, "unreceipted_bundles": [2000]}])
    assert "missed at least" not in calm


def test_the_alpha_panel_draws_the_rate_and_an_absence_for_an_unproduced_block(html, reader, tmp_path):
    page = _page(html, reader, tmp_path, [
        {"event": "iteration_complete", "step": 0,
         "gumbel_alpha_full": {"rows": 0, "graph_rows": 512, "per_1000": 0.0}},
        {"event": "iteration_complete", "step": 10,
         "gumbel_alpha_full": {"rows": 1, "graph_rows": 8000, "per_1000": 0.125}}])
    assert "0.125" in page and "<svg" in page
    absent = _page(html, reader, tmp_path, [
        {"event": "iteration_complete", "step": 10, "gumbel_alpha_full": None}])
    assert "iteration_complete.gumbel_alpha_full" in absent and "0.125" not in absent


def test_the_firings_input_is_unmeasured_without_a_record_dir(html, reader, tmp_path):
    page = _page(html, reader, tmp_path, BOOT)
    assert "collate_dumps/ and the logs were not read" in page
    assert "0 firings" not in page and "0 in-wire" not in page


def test_the_page_carries_no_absolute_home_path(html, reader, tmp_path):
    events = _write(tmp_path, BOOT)
    page = html.render(reader.load_record(events), "t")
    assert str(tmp_path) not in page
    assert events.name in page


def test_a_record_without_game_complete_renders_quality_gaps_and_an_unmeasured_badge(
        html, reader, tmp_path):
    rows = BOOT + [{"event": "trainer_step", "step": i, "loss": 1.0 / i, "policy_loss": 0.5,
                    "value_loss": 0.2, "grad_norm": 3.0, "lr": 1e-3} for i in range(1, 50)]
    page = _page(html, reader, tmp_path, rows)
    quality = re.search(r'<section class="panel tier2" id="quality">(.*?)</section>', page, re.S)
    assert quality is not None
    assert "<svg" not in quality.group(1)
    assert quality.group(1).count("not measured — see notes") >= 4
    assert 'class="cell badge unmeasured"' in page and "unmeasured for" in page


def test_the_hero_strength_cell_carries_wr_games_wilson_and_elo(html, reader, tmp_path):
    rows = [_round(1, 0.1875, lo=0.0625, hi=0.34375)]
    page = _page(html, reader, tmp_path, rows, _ladder([(1, 32, 0.1875)]))
    values = _hero_values(page)
    assert values[0].startswith("18.75"), values[0]
    assert "32 games" in page and "8.9" in page and "35.3" in page, "the Wilson bounds"
    assert "−255" in page, "the Elo equivalent of 18.75 % is −254.7"
    assert "sealbot_d5" in page


def test_the_strength_panel_says_every_sealbot_reading_is_provisional(html, reader, tmp_path):
    """R353(b): the strength panel says every sealbot reading is PROVISIONAL, with rounds and without."""
    for rows in (BOOT, [_round(1, 0.5)]):
        page = _page(html, reader, tmp_path, rows, _ladder([(1, 32, 0.5)]))
        panel = re.search(r'<section class="panel tier2" id="ladder">(.*?)</section>', page, re.S)
        assert panel is not None
        note = re.search(r'<p class="note">(.*?)</p>', panel.group(1), re.S)
        assert note is not None, "the strength panel carries no note"
        assert "PROVISIONAL" in note.group(1)
        assert "CARD-SEALBOT-TT-SEAT" in note.group(1) and "R353" in note.group(1)


def test_the_hero_trend_cell_names_the_rounds_in_its_window(html, reader, tmp_path):
    rows = [_round(i, wr) for i, wr in enumerate([0.2, 0.25, 0.3, 0.35, 0.4], start=1)]
    page = _page(html, reader, tmp_path, rows,
                 _ladder([(i, 32, wr) for i, wr in enumerate([0.2, 0.25, 0.3, 0.35, 0.4], 1)]))
    assert "5 rounds in window" in page
    assert "Elo / 1k steps" in page


def test_the_promotion_cell_has_three_states(html, reader, tmp_path):
    def cell(rows):
        return _hero_values(_page(html, reader, tmp_path, rows))[2]
    assert cell([_round(1, 0.2, promoted=True)]) == "promoted"
    assert cell([_round(1, 0.2, promoted=False)]) == "not promoted"
    assert cell([_round(1, 0.2)]) == "no decision taken"
    assert cell(BOOT) == "no round"


def test_tier_three_is_collapsed_and_a_warning_hoists_into_tier_one(html, reader, tmp_path):
    rows = BOOT + [{"event": "training_alert", "rule": "loss_increase_window", "step": 4,
                    "message": "loss increased 3 consecutive steps"}]
    page = _page(html, reader, tmp_path, rows)
    assert "<details open" not in page
    badge = re.search(r'<div class="cell badge[^"]*">(.*?)</div>', page, re.S).group(1)
    assert "training alerts" in badge and "loss_increase_window" in badge


def test_every_tier_one_number_survives_with_the_script_stripped(html, reader, tmp_path):
    rows = BOOT + [_round(1, 0.2, promoted=True)] + [
        {"event": "trainer_step", "step": i, "loss": 1.0, "policy_loss": 0.5, "value_loss": 0.25,
         "grad_norm": 2.0, "lr": 1e-3, "ts": 2.0 + i} for i in range(1, 4)] + [
        {"event": "iteration_complete", "step": 3, "games_total": 9, "games_per_hour": 1200.0,
         "steps_per_hour": 1100.0, "ts": 6.0}]
    page = _page(html, reader, tmp_path, rows, _ladder([(1, 32, 0.2)]))
    values = _hero_values(page)
    assert len(values) == 7 and all(v.strip() for v in values)
    stripped = re.sub(r"<script>.*?</script>", "", page, flags=re.S)
    for value in values:
        assert value in stripped


def test_the_grad_norm_rule_is_a_stated_gap_when_the_threshold_is_not_in_the_record(
        html, reader, tmp_path):
    rows = [{"event": "trainer_step", "step": i, "grad_norm": float(i), "loss": 1.0,
             "policy_loss": 0.5, "value_loss": 0.5, "lr": 1e-3} for i in range(1, 5)]
    page = _page(html, reader, tmp_path, rows)
    assert 'class="rule"' not in page
    assert "monitor.alert_grad_norm_max" in page


def test_the_cli_shim_keeps_the_frozen_flags_and_writes_the_file(shim, tmp_path):
    events = _write(tmp_path, BOOT)
    ladder = tmp_path / "eval_ladder_state.json"
    ladder.write_text(json.dumps(_ladder([])), encoding="utf-8")
    out = tmp_path / "out.html"
    rc = shim.main(["--events", str(events), "--ladder-state", str(ladder), "--record-dir",
                    str(tmp_path), "--out", str(out), "--title", "run x"])
    assert rc == 0 and out.exists()
    assert "<title>run x</title>" in out.read_text(encoding="utf-8")
    assert shim.MIRROR_LAG_WARN_BUNDLES == 2


def test_the_shakedown_record_renders_under_the_size_cap(shim, tmp_path):
    events = os.environ.get("MANTIS_DASH_FIXTURE_EVENTS")
    if not events or not Path(events).is_file():
        pytest.skip("MANTIS_DASH_FIXTURE_EVENTS is unset or absent — the 173 MB run6 record "
                    "is never a fixture in git (R7); set the variable to run this on the box")
    ladder = Path(events).with_name("eval_ladder_state.json")
    out = tmp_path / "run.html"
    argv = ["--events", events, "--out", str(out), "--record-dir", str(Path(events).parent)]
    if ladder.is_file():
        argv += ["--ladder-state", str(ladder)]
    assert shim.main(argv) == 0
    assert out.stat().st_size <= shim.SIZE_CAP_BYTES, out.stat().st_size


def test_the_grad_norm_rule_is_drawn_when_the_ceiling_rides_resolved_config(html, reader, tmp_path):
    rows = [{"event": "resolved_config",
             "knobs": {"monitor.alert_grad_norm_max": {"value": 10.0, "source": "file"}}}] + [
        {"event": "trainer_step", "step": i, "grad_norm": float(i), "loss": 1.0,
         "policy_loss": 0.5, "value_loss": 0.5, "lr": 1e-3} for i in range(1, 5)]
    page = _page(html, reader, tmp_path, rows)
    assert page.count('class="rule"') == 1
    assert "monitor.alert_grad_norm_max = 10" in page


def test_the_quality_panel_draws_the_halts_windowed_cap_rate_from_game_zero(html, reader, tmp_path):
    """R352(c): the fifth multiple is the ply-cap share over the halt's own window."""
    rows = BOOT + [{"event": "monitor_gates", "step": 1, "gates": {}, "ply_cap_abort_rate": 0.5,
                    "ply_cap_window_games": 4}]
    rows += [{"event": "game_complete", "winner": 0, "moves": 30, "terminal_reason": "six_in_a_row"}
             for _ in range(4)]
    rows += [{"event": "game_complete", "winner": -1, "moves": 256, "terminal_reason": "ply_cap"}
             for _ in range(4)]
    page = _page(html, reader, tmp_path, rows)
    quality = re.search(r'<section class="panel tier2" id="quality">(.*?)</section>', page, re.S)
    assert quality is not None
    body = quality.group(1)
    assert "ply-cap share, 4-game window" in body, "the window is the record's armed one"
    assert "halt rate 0.5" in body
    assert "peak 1.00" in body and "game 8" in body


def test_an_unarmed_record_draws_the_windowed_cap_rate_at_the_minted_window_and_says_so(
        html, reader, tmp_path):
    rows = BOOT + [{"event": "game_complete", "winner": 0, "moves": 30, "terminal_reason": "six_in_a_row"}
                   for _ in range(10)]
    page = _page(html, reader, tmp_path, rows)
    body = re.search(r'<section class="panel tier2" id="quality">(.*?)</section>', page, re.S).group(1)
    assert "ply-cap share, 600-game window" in body
    assert "not armed" in body
    assert "not measured — see notes" in body, "below the window is a stated gap, never a zero"
    assert "10 games, fewer than the 600-game window" in page, "the statement lives in the notes"
