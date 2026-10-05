"""The dash strength reader: one unit per line, the parent joined from the warm start, other units off the line, the going-forward read."""
from __future__ import annotations

import importlib
import json
import math

import pytest

from _dash_record import TACTICS, sidecar, write_config


@pytest.fixture(scope="module")
def sc(dash):
    return importlib.import_module("dash.readers.sidecars")


@pytest.fixture(scope="module")
def record(dash):
    return importlib.import_module("dash.readers.record")


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def test_a_planted_other_unit_sidecar_stays_off_the_line(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.60)
    sidecar(tmp_path, "r1", 6000, 0.62)
    sidecar(tmp_path, "r1", 9000, 0.90, tactics={**TACTICS, "arm": "off"}, suffix="six30_16.off")
    cells, _ = sc.load([tmp_path])
    got = sc.strength(cells, "six", "r1", None)
    assert [c.step for c in got.line] == [3000, 6000]
    assert [c.step for c in got.other] == [9000]


def test_the_parent_joins_by_its_checkpoint_stem_in_the_lines_unit(sc, tmp_path):
    sidecar(tmp_path / "run", "r1", 3000, 0.70)
    sidecar(tmp_path / "parent", "p0", 45000, 0.59)
    cells, _ = sc.load([tmp_path])
    got = sc.strength(cells, "six", "r1", "p0_00045000_abcd1234")
    assert got.parent is not None and got.parent.wr == 0.59 and got.parent_other is None


def test_a_parent_read_in_another_unit_is_named_but_never_the_band(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.70)
    sidecar(tmp_path, "p0", 45000, 0.59, ours={"search_kind": "puct", "sims": 512})
    cells, _ = sc.load([tmp_path])
    got = sc.strength(cells, "six", "r1", "p0_00045000_abcd1234")
    assert got.parent is None and got.parent_other is not None and got.going_forward is None


def test_the_going_forward_read_is_the_mean_logit_of_the_last_four_cells_over_the_parents(sc, tmp_path):
    for step, wr in ((3000, 0.50), (6000, 0.60), (9000, 0.62), (12000, 0.64), (15000, 0.66)):
        sidecar(tmp_path, "r1", step, wr)
    sidecar(tmp_path, "p0", 45000, 0.59)
    cells, _ = sc.load([tmp_path])
    mean, used = sc.strength(cells, "six", "r1", "p0_00045000_abcd1234").going_forward
    expected = sum(_logit(p) for p in (0.60, 0.62, 0.64, 0.66)) / 4 - _logit(0.59)
    assert used == 4 and mean == pytest.approx(expected)


def test_fewer_than_four_cells_read_with_those_available_and_say_how_many(sc, tmp_path):
    sidecar(tmp_path, "r1", 32201, 0.759)
    sidecar(tmp_path, "p0", 45000, 0.594)
    cells, _ = sc.load([tmp_path])
    mean, used = sc.strength(cells, "six", "r1", "p0_00045000_abcd1234").going_forward
    assert used == 1 and mean == pytest.approx(_logit(0.759) - _logit(0.594))


def test_strix_and_six_are_separate_families(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.70)
    sidecar(tmp_path, "r1", 3000, 0.12, family="strix")
    cells, _ = sc.load([tmp_path])
    assert [c.wr for c in sc.strength(cells, "strix", "r1", None).line] == [0.12]
    assert [c.wr for c in sc.strength(cells, "six", "r1", None).line] == [0.70]


def test_a_failed_cell_and_a_broken_sidecar_are_skipped_by_name(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.70)
    (tmp_path / "r1_00006000_abcd1234.ckpt.six30_16.failed.json").write_text("{}", encoding="utf-8")
    (tmp_path / "r1_00009000_abcd1234.ckpt.six30_16.json").write_text("{nope", encoding="utf-8")
    (tmp_path / "r1_00012000_abcd1234.ckpt.six30_16.json").write_text(json.dumps({"step": 1}), encoding="utf-8")
    cells, skipped = sc.load([tmp_path])
    assert len(cells) == 1
    assert any("failed" in s for s in skipped) and any("ValueError" in s or "JSONDecodeError" in s for s in skipped)
    assert any("no step, win rate or opponent" in s for s in skipped)


def test_one_sidecar_copied_into_two_directories_counts_once(sc, tmp_path):
    a = sidecar(tmp_path / "a", "r1", 3000, 0.70)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / a.name).write_text(a.read_text(encoding="utf-8"), encoding="utf-8")
    cells, _ = sc.load([tmp_path / "a", tmp_path / "b"])
    assert len(cells) == 1


def test_another_runs_cells_never_join_the_line(sc, tmp_path):
    sidecar(tmp_path, "r1", 3000, 0.70)
    sidecar(tmp_path, "r2", 6000, 0.30)
    cells, _ = sc.load([tmp_path])
    assert [c.run_id for c in sc.strength(cells, "six", "r1", None).line] == ["r1"]


def test_the_parent_stem_is_read_from_the_identity_warm_start(record, tmp_path):
    write_config(tmp_path, "r1", "p0_00045000_abcd1234")
    assert record.parent_stem(tmp_path)[0] == "p0_00045000_abcd1234"
    write_config(tmp_path / "fresh", "r2", None)
    stem, note = record.parent_stem(tmp_path / "fresh")
    assert stem is None and "no parent" in note
