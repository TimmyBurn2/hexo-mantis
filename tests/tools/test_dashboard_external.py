"""R356(d): the strix sidecars draw as a series per unit with CIs and regime, the gap is a number, and the throughput panel carries a unit on every label."""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def external(dashboard):
    return importlib.import_module("dashboard.external")


@pytest.fixture(scope="module")
def throughput(dashboard):
    return importlib.import_module("dashboard.throughput")


@pytest.fixture(scope="module")
def html(dashboard):
    return importlib.import_module("dashboard.html")


@pytest.fixture(scope="module")
def reader(dashboard):
    return importlib.import_module("dashboard.reader")


def _sidecar(step: int, wr: float, *, unit: str = "equal_work", ours: int = 256, strix: int = 256,
             regime: str = "CONTENDED", trigger: str = "cadence") -> dict:
    return {"schema_version": 1, "run_id": "run8", "checkpoint": f"run8_{step:08d}_abcd1234.ckpt",
            "checkpoint_sha256": "c" * 64, "step": step, "net_hash": "n" * 64, "unit": unit,
            "ours": {"search_kind": "puct", "sims": ours}, "strix": {"commit": "5a771e57",
            "checkpoint": "checkpoint_00237000.pt", "checkpoint_sha256": "3" * 64, "sims": strix},
            "trigger": trigger, "regime": regime, "regime_evidence": {"heartbeat_age_sec": 12.0},
            "games": 288, "eff_n": 288, "pairs": 144, "wins": round(288 * wr), "losses": 288 - round(288 * wr),
            "draws": 0, "wr": wr, "wr_ci_lower": wr - 0.04, "wr_ci_upper": wr + 0.045,
            "median_plies": 43.0, "sec_per_game": 15.0, "wall_sec": 4320.0}


def _write_sidecars(root: Path, cells: list[dict]) -> Path:
    ck = root / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    for c in cells:
        suffix = "strix256" if c["ours"]["sims"] == 256 else "strix512"
        (ck / f"{c['checkpoint']}.{suffix}.json").write_text(json.dumps(c), encoding="utf-8")
    return ck


def _page(html, reader, tmp_path: Path, rows: list[dict], external_points: Path | None) -> str:
    events = tmp_path / "events.jsonl"
    events.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return html.render(reader.load_record(events, None, None, external_points), "t")


_BOOT = [{"event": "run_boot_identity", "ts": 1.0, "run_id": "run8"}]


def test_sidecars_load_by_unit_and_a_failed_or_broken_one_is_skipped_and_named(external, tmp_path: Path) -> None:
    ck = _write_sidecars(tmp_path, [_sidecar(15000, 0.15), _sidecar(30000, 0.20),
                                    _sidecar(30000, 0.14, unit="as_shipped", ours=512, strix=128)])
    (ck / "run8_00045000_ffffffff.ckpt.strix256.failed.json").write_text("{}", encoding="utf-8")
    (ck / "run8_00060000_ffffffff.ckpt.strix256.json").write_text("{\"step\": 60000}", encoding="utf-8")
    points, note = external.load_external_points(ck)
    assert [(p.step, p.unit) for p in points] == [(30000, "as_shipped"), (15000, "equal_work"), (30000, "equal_work")]
    units = external.series_by_unit(points)
    assert list(units) == ["as_shipped: ours PUCT-512 vs strix 128 sims",
                           "equal_work: ours PUCT-256 vs strix 256 sims"]
    assert "3 sidecar(s) read" in note and "failed cell, not a receipt" in note
    assert "run8_00060000_ffffffff.ckpt.strix256.json (missing step, wr or the unit's sims)" in note


def test_the_gap_to_strix_is_stated_as_a_number(external, tmp_path: Path) -> None:
    point = external.parse_sidecar(Path("x"), _sidecar(42000, 0.142))
    text = external.gap_statement(point)
    assert "WR 14.2\u202f% vs strix" in text and "35.8 pp below parity" in text and "Elo" in text
    assert "CONTENDED" in text and "n = 288" in text


def test_the_panel_draws_one_series_per_unit_with_whiskers_regime_and_axis_labels(
        html, reader, tmp_path: Path) -> None:
    ck = _write_sidecars(tmp_path, [_sidecar(15000, 0.15), _sidecar(30000, 0.20, regime="IDLE"),
                                    _sidecar(30000, 0.14, unit="as_shipped", ours=512, strix=128)])
    page = _page(html, reader, tmp_path, _BOOT, ck)
    panel = re.search(r'<section class="panel tier2" id="external">(.*?)</section>', page, re.S).group(1)
    assert panel.count("<polyline") == 1, "two equal-work points make a line; one as-shipped point does not"
    assert panel.count('class="whisker') == 3
    assert 'class="marker point' in panel and 'class="marker idle' in panel
    assert "equal_work: ours PUCT-256 vs strix 256 sims" in panel
    assert "as_shipped: ours PUCT-512 vs strix 128 sims" in panel
    assert "y = WR vs strix in the unit the legend names" in panel and "x = step" in panel
    assert "30.0 pp below parity" in panel and "36.0 pp below parity" in panel
    assert "not measured" not in panel


def test_without_sidecars_the_panel_is_a_stated_gap_naming_the_producer(html, reader, tmp_path: Path) -> None:
    page = _page(html, reader, tmp_path, _BOOT, None)
    panel = re.search(r'<section class="panel tier2" id="external">(.*?)</section>', page, re.S).group(1)
    assert "No external point in this record" in panel and "<svg" not in panel
    assert "tools/strix_follower.py sidecars" in page
    assert "no --external-points given" in page


def test_plies_per_hour_is_the_columns_plies_over_its_own_wall(throughput) -> None:
    # Eight games over 2.0 h, 520 plies in all; `width` cannot split them below the minimum.
    games = [{"ts": 3600.0 * h, "moves": m} for h, m in
             ((0.0, 40), (0.2, 60), (0.3, 50), (0.5, 50), (1.0, 80), (1.4, 80), (1.6, 80), (2.0, 80))]
    series = throughput.plies_per_hour(games, width=2)
    assert [(x, round(y, 1)) for x, y in series] == [(8.0, 520.0 / 2.0)], (
        "eight games make ONE window at the eight-game minimum: 520 plies over 2.0 h")
    wide = [{"ts": 3600.0 * i / 8.0, "moves": 50} for i in range(64)]
    assert len(throughput.plies_per_hour(wide, width=640)) == 8, "64 games → 8 windows of 8"


def test_a_column_with_no_wall_span_is_skipped_not_divided(throughput) -> None:
    games = [{"ts": 10.0, "moves": 30}, {"ts": 10.0, "moves": 30}]
    assert throughput.plies_per_hour(games, width=1) == []


def test_the_throughput_panel_carries_a_unit_on_every_label(html, reader, tmp_path: Path) -> None:
    rows = list(_BOOT)
    rows += [{"event": "iteration_complete", "step": i * 100, "ts": 1.0 + i * 60.0,
              "games_per_hour": 900.0 + i, "positions_per_hour": 30000.0 + i,
              "sims_per_sec": 3190.0, "steps_per_hour": 980.0} for i in range(1, 6)]
    rows += [{"event": "game_complete", "ts": 1.0 + i * 3.6, "moves": 60 + (i % 7), "winner": i % 2,
              "terminal_reason": "six_in_a_row"} for i in range(1, 400)]
    page = _page(html, reader, tmp_path, rows, None)
    panel = re.search(r'<section class="panel tier2" id="throughput">(.*?)</section>', page, re.S).group(1)
    labels = re.findall(r"<h3>(.*?)</h3>", panel)
    assert labels == ["games / h",
                      "plies / h (Σ game_complete.moves per window ÷ the window&#x27;s wall)",
                      "turns / h (positions_per_hour: compound turns, ≈ 2 plies each)",
                      "leaves / s (sims_per_sec: positions × effective sims per move, billed)",
                      "steps / h"]
    assert panel.count("<polyline") == 5
    assert "x-labels" in panel and "step" in panel and "game" in panel
    economy = re.search(r'<section class="panel tier2" id="economy">(.*?)</section>', page, re.S).group(1)
    assert "games / h" not in economy, "the rates moved; the economy panel keeps the buffer"
