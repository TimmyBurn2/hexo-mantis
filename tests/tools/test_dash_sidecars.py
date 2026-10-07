"""The dash strength reader: one series per unit with its own parent, the rule marked, the going-forward read, the ladder."""
from __future__ import annotations

import importlib
import json
import math

import pytest

from _dash_record import SIX, TACTICS, sidecar, write_config

PARENT = "p0_00045000_abcd1234"


@pytest.fixture(scope="module")
def sc(dash):
    return importlib.import_module("dash.readers.sidecars")


@pytest.fixture(scope="module")
def ladder(dash):
    return importlib.import_module("dash.readers.ladder")


@pytest.fixture(scope="module")
def record(dash):
    return importlib.import_module("dash.readers.record")


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _six455(nodes: int) -> dict:
    return {"unit": f"six455_{nodes}", "six": {**SIX, "generation": 455, "nodes": nodes, "net_sha256": "g455"}}


def test_every_unit_is_its_own_series_never_joined(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.60)
    sidecar(tmp_path, "r1", 6000, 0.62)
    sidecar(tmp_path, "r1", 9000, 0.90, tactics={**TACTICS, "arm": "off"}, suffix="six30_16.off")
    sidecar(tmp_path, "r1", 9000, 0.40, suffix="six455_128.full", **_six455(128))
    cells, _ = sc.load([tmp_path])
    by_name = {r.name: [c.step for c in r.line] for r in sc.rulers(cells, "r1", None, None)[0]}
    assert by_name == {"six30_16.full": [3000, 6000], "six30_16.off": [9000], "six455_128.full": [9000]}


@pytest.mark.parametrize("full_dir", ["a_full", "z_full"])
def test_a_checkpoint_read_twice_on_one_unit_keeps_the_reading_over_more_games(sc, tmp_path, full_dir):
    sidecar(tmp_path / full_dir, "r1", 36000, 0.19, suffix="six455_128.full", **_six455(128))
    sidecar(tmp_path / "m_screen", "r1", 36000, 0.18, n=128, suffix="six455_128.full", **_six455(128))
    sidecar(tmp_path / "n_smoke", "r1", 36000, 0.0, n=2, suffix="six455_128.full", **_six455(128))
    cells, _ = sc.load([tmp_path])
    assert [(c.step, c.wr, c.n) for c in cells] == [(36000, 0.19, 576)]


def test_six_forfeits_are_left_out_of_the_reading_not_counted_as_our_wins(sc, tmp_path):
    findings = {"count": 58, "first": ["six_forfeit_finding failed ply 6: bestmove none"]}
    sidecar(tmp_path, "r1", 24000, 0.3715, six_findings=findings, wins=214, draws=0, losses=362)
    (cell,), _ = sc.load([tmp_path])
    assert cell.forfeits == 58 and cell.n == 518
    assert cell.wr == pytest.approx(156 / 518) and cell.lo == pytest.approx(0.262, abs=0.002) and cell.hi == pytest.approx(0.343, abs=0.002)


def test_a_forfeit_correction_scores_a_draw_as_half_a_game(sc, tmp_path):
    sidecar(tmp_path, "r1", 24000, 0.105, n=100, six_findings={"count": 3}, wins=10, draws=1, losses=89)
    (cell,), _ = sc.load([tmp_path])
    assert cell.n == 97 and cell.wr == pytest.approx(7.5 / 97)


@pytest.mark.parametrize(("over", "why"), [({"six_findings": {"count": 576}}, "every one of its 576 games is a Six forfeit"),
                                           ({"six_findings": {"count": 5}, "games": 600}, "counted over 600 games")])
def test_a_cell_its_forfeits_cannot_be_read_out_of_is_refused_by_name(sc, tmp_path, over, why):
    sidecar(tmp_path, "r1", 24000, 0.5, **over)
    cells, skipped = sc.load([tmp_path])
    assert cells == [] and any(why in note for note in skipped)


def test_each_ruler_has_its_own_parent_on_the_same_unit(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.70)
    sidecar(tmp_path, "r1", 3000, 0.40, suffix="six455_128.full", **_six455(128))
    sidecar(tmp_path / "parent", "p0", 45000, 0.59)
    sidecar(tmp_path / "parent", "p0", 45000, 0.30, suffix="six455_128.full", **_six455(128))
    cells, _ = sc.load([tmp_path])
    parents = {r.name: r.parent.wr for r in sc.rulers(cells, "r1", PARENT, None)[0]}
    assert parents == {"six30_16.full": 0.59, "six455_128.full": 0.30}


def test_a_parent_read_in_another_unit_never_anchors_the_line(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.70)
    sidecar(tmp_path, "p0", 45000, 0.59, ours={"search_kind": "puct", "sims": 512})
    cells, _ = sc.load([tmp_path])
    (only,), _ = sc.rulers(cells, "r1", PARENT, "six30_16")
    assert only.parent is None and only.going_forward is None


def test_the_rule_is_named_by_unit_field_or_full_name_and_listed_first(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.40, suffix="six455_128.full", **_six455(128))
    sidecar(tmp_path, "r1", 3000, 0.70)
    sidecar(tmp_path, "r1", 3000, 0.12, family="strix")
    cells, _ = sc.load([tmp_path])
    for rule in ("six30_16", "six30_16.full"):
        listed, _ = sc.rulers(cells, "r1", None, rule)
        assert listed[0].name == "six30_16.full" and listed[0].rule and not any(r.rule for r in listed[1:])
    assert [r.family for r in sc.rulers(cells, "r1", None, None)[0]] == ["six", "six", "strix"]


def test_the_going_forward_read_is_the_mean_logit_of_the_last_four_cells_over_the_parents(sc, tmp_path):
    for step, wr in ((3000, 0.50), (6000, 0.60), (9000, 0.62), (12000, 0.64), (15000, 0.66)):
        sidecar(tmp_path, "r1", step, wr)
    sidecar(tmp_path, "p0", 45000, 0.59)
    cells, _ = sc.load([tmp_path])
    mean, used = sc.rulers(cells, "r1", PARENT, "six30_16")[0][0].going_forward
    expected = sum(_logit(p) for p in (0.60, 0.62, 0.64, 0.66)) / 4 - _logit(0.59)
    assert used == 4 and mean == pytest.approx(expected)


def test_fewer_than_four_cells_read_with_those_available(sc, tmp_path):
    sidecar(tmp_path, "r1", 32201, 0.759)
    sidecar(tmp_path, "p0", 45000, 0.594)
    cells, _ = sc.load([tmp_path])
    mean, used = sc.rulers(cells, "r1", PARENT, "six30_16")[0][0].going_forward
    assert used == 1 and mean == pytest.approx(_logit(0.759) - _logit(0.594))


def test_a_sidecar_without_distinct_games_is_refused_not_read_on_raw_games(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.70, eff_n=None)
    cells, skipped = sc.load([tmp_path])
    assert cells == [] and any("distinct games" in s for s in skipped)


def test_a_failed_cell_a_broken_sidecar_and_a_missing_directory_are_named(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.70)
    (tmp_path / "r1_00006000_abcd1234.ckpt.six30_16.failed.json").write_text("{}", encoding="utf-8")
    (tmp_path / "r1_00009000_abcd1234.ckpt.six30_16.json").write_text("{nope", encoding="utf-8")
    cells, skipped = sc.load([tmp_path, tmp_path / "gone"])
    assert len(cells) == 1 and any("failed" in s for s in skipped) and any("Error" in s for s in skipped)
    assert any(s == "a --cells directory named gone does not exist" for s in skipped)


def test_one_sidecar_copied_into_two_directories_counts_once_and_another_run_never_joins(sc, tmp_path):
    a = sidecar(tmp_path / "a", "r1", 3000, 0.70)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / a.name).write_text(a.read_text(encoding="utf-8"), encoding="utf-8")
    sidecar(tmp_path / "b", "r2", 6000, 0.30)
    cells, _ = sc.load([tmp_path / "a", tmp_path / "b"])
    assert len(cells) == 2 and [c.run_id for r in sc.rulers(cells, "r1", None, None)[0] for c in r.line] == ["r1"]


def test_a_bridge_is_one_checkpoint_read_on_both_rungs(sc, tmp_path):
    sidecar(tmp_path, "r1", 48000, 0.72, suffix="six455_128.full", **_six455(128))
    sidecar(tmp_path, "r1", 48000, 0.55, suffix="six455_256.full", **_six455(256))
    sidecar(tmp_path, "r1", 51000, 0.74, suffix="six455_128.full", **_six455(128))
    cells, _ = sc.load([tmp_path])
    ((a, b),) = sc.bridges(cells, "r1", ("six455_128", "six455_256"))
    assert a.step == b.step == 48000 and (a.wr, b.wr) == (0.72, 0.55)


def test_the_ladder_state_is_read_and_its_absence_is_a_stated_gap(ladder, tmp_path):
    path = tmp_path / "ladder_state.json"
    path.write_text(json.dumps({"current_unit": "six455_256", "streak": 0,
                                "history": [{"step": 48000, "unit": "six455_128", "wr": 0.72, "lo": 0.70, "hi": 0.75}],
                                "changes": [{"step": 48000, "from": "six455_128", "to": "six455_256"}, {"bad": 1}]}),
                    encoding="utf-8")
    read = ladder.read(path)
    assert read.current == "six455_256" and read.streak == 0 and read.cells == 1
    assert read.note == "read, but 1 change row(s) lack step, from or to"
    assert [(c.step, c.frm, c.to) for c in read.changes] == [(48000, "six455_128", "six455_256")]
    assert ladder.read(tmp_path / "absent.json").note == "the ladder file is not there yet"
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert "did not read" in ladder.read(tmp_path / "bad.json").note and ladder.read(None) is None


def test_the_parent_stem_is_read_from_the_identity_warm_start(record, tmp_path):
    write_config(tmp_path, "r1", PARENT)
    assert record.parent_stem(tmp_path)[0] == PARENT
    write_config(tmp_path / "fresh", "r2", None)
    stem, note = record.parent_stem(tmp_path / "fresh")
    assert stem is None and "no parent" in note


def test_a_rule_naming_two_units_marks_neither_and_the_names_are_told_apart(sc, tmp_path):
    sidecar(tmp_path, "r1", 36000, 0.76)
    sidecar(tmp_path, "r1", 39000, 0.50, six={**SIX, "commit": "newpin"})
    cells, _ = sc.load([tmp_path])
    listed, matches = sc.rulers(cells, "r1", None, "six30_16")
    assert matches == 2 and not any(r.rule for r in listed)
    assert len({r.name for r in listed}) == 2 and all(r.name.startswith("six30_16.full #") for r in listed)


def test_a_rule_naming_no_unit_yet_marks_none(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.40, suffix="six455_128.full", **_six455(128))
    cells, _ = sc.load([tmp_path])
    listed, matches = sc.rulers(cells, "r1", None, "six30_16")
    assert matches == 0 and not any(r.rule for r in listed)


def test_a_reading_at_zero_is_moved_half_a_game_in_never_to_minus_fourteen(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.0, n=288, wr_ci_lower=0.0, wr_ci_upper=0.0)
    (cell,) = sc.load([tmp_path])[0]
    assert cell.logit == pytest.approx(_logit(0.5 / 288)) and cell.logit > -7 and cell.logit_half_width == 0.0


def test_malformed_ladder_fields_are_named_never_raised(ladder, tmp_path):
    path = tmp_path / "ladder_state.json"
    path.write_text(json.dumps({"current_unit": 7, "changes": 5, "history": "x"}), encoding="utf-8")
    read = ladder.read(path)
    assert read.changes == () and read.current is None and "changes is not a list" in read.note
    path.write_text(json.dumps({"changes": [{"step": 1, "from_unit": "a", "to": "b"}]}), encoding="utf-8")
    assert ladder.read(path).note == "read, but 1 change row(s) lack step, from or to"


def test_every_follower_unit_writes_a_sidecar_the_dash_reads(sc) -> None:
    """A census over the follower's units, so a new unit cannot drop out of the display unseen."""
    import fnmatch
    from pathlib import Path

    from _toolpath import load_module_by_path

    follower = load_module_by_path("strix_follower_globs", Path(__file__).resolve().parents[2] / "tools/strix_follower.py")
    for unit in follower.UNITS:
        name = follower.sidecar_path(Path("/x/r_00001000_abcd1234.ckpt"), unit, "full").name
        assert any(fnmatch.fnmatch(name, g) for g in sc.GLOBS), unit


def test_strix_on_another_device_is_another_series(sc, tmp_path):
    cpu = sidecar(tmp_path, "r1", 3000, 0.4, family="strix", suffix="strix256_arena.cpu")
    gpu = sidecar(tmp_path, "r1", 6000, 0.4, family="strix", suffix="strix256_arena.gpu",
                  strix={**json.loads(cpu.read_text(encoding="utf-8"))["strix"], "device": "cuda"})
    a, b = (sc.parse(p, json.loads(p.read_text(encoding="utf-8"))) for p in (cpu, gpu))
    assert a is not None and b is not None and a.unit != b.unit


def _ladder16() -> dict:
    return {"unit": "ladder455_n16", "ours": {"search_kind": "puct", "sims": 128}, "opening_book": "arena_p5",
            "opening_book_sha256": "364c70c7", "six": {**SIX, "generation": 455, "nodes": 16, "net_sha256": "g455"}}


@pytest.mark.parametrize(("text", "current", "switches"), [
    ("six30_16", "six30_16", []),
    ("six30_16,ladder455_n16@177000,ladder455_n128@200000", "ladder455_n128",
     [(177000, "six30_16", "ladder455_n16"), (200000, "ladder455_n16", "ladder455_n128")])])
def test_a_rule_reads_as_its_unit_now_and_the_switches_before_it(sc, text, current, switches):
    now, moved = sc.parse_rule(text)
    assert now == current and [(c.step, c.frm, c.to) for c in moved] == switches


@pytest.mark.parametrize("text", ["", "a@5", "a,b", "a,b@x", "a,b@0", "a,b@200,c@100", "a,a@100", "a,@100", "a,,b@9"])
def test_a_malformed_rule_is_refused_by_name(sc, text):
    with pytest.raises(ValueError, match="rule"):
        sc.parse_rule(text)


def test_a_rule_switch_marks_the_former_rule_until_its_step_and_lists_it_second(sc, tmp_path):
    sidecar(tmp_path, "r1", 168000, 0.78)
    sidecar(tmp_path, "r1", 168000, 0.23, suffix="six455_128.full", **_six455(128))
    sidecar(tmp_path, "r1", 177000, 0.31, suffix="ladder455_n16.full", **_ladder16())
    cells, _ = sc.load([tmp_path])
    listed, matches = sc.rulers(cells, "r1", None, *sc.parse_rule("six30_16,ladder455_n16@177000"))
    assert matches == 1 and [(r.name, r.rule, r.rule_until) for r in listed] == [
        ("ladder455_n16.full", True, None), ("six30_16.full", False, 177000), ("six455_128.full", False, None)]


def test_two_opening_books_are_two_instruments_and_the_label_names_the_book(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.60)
    sidecar(tmp_path, "r1", 6000, 0.50, opening_book="arena_p5", opening_book_sha256="364c70c7")
    cells, _ = sc.load([tmp_path])
    listed, _ = sc.rulers(cells, "r1", None, None)
    assert len(listed) == 2 and sorted("arena_p5 openings" in r.label for r in listed) == [False, True]
