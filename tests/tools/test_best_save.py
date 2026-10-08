"""The best-save rule over one run's saves read on rung 16 and S, through the dash's sidecar reader (forfeits out)."""
from __future__ import annotations

from pathlib import Path

import pytest
from _dash_record import sidecar
from _toolpath import load_module_by_path

_REPO = Path(__file__).resolve().parents[2]
B = load_module_by_path("best_save_t", _REPO / "tools" / "best_save.py")
_G455 = {"commit": "c0ffee", "net_sha256": "g455", "generation": 455, "nodes": 16}


def _save(d: Path, run: str, step: int, rung: float, s: float, **six: object) -> None:
    sidecar(d, run, step, rung, suffix="ladder455_n16.full", unit="ladder455_n16", six={**_G455, **six})
    sidecar(d, run, step, s, family="strix", suffix="strix256_arena.full", unit="equal_work_arena")


def test_a_save_first_on_both_instruments_is_the_pick(tmp_path: Path) -> None:
    _save(tmp_path, "r", 100, 0.30, 0.55)
    _save(tmp_path, "r", 200, 0.34, 0.60)
    pick = B.best_save([tmp_path], run_id="r", arm="full")
    assert pick["pick"] == "r_00000200_abcd1234" and pick["rule"] == "first on both"


def test_without_agreement_the_first_on_rung_16_whose_s_holds_the_best_s_is_the_pick(tmp_path: Path) -> None:
    _save(tmp_path, "r", 100, 0.35, 0.50)  # first on rung 16, its S interval [0.46, 0.54] misses the best S 0.63
    _save(tmp_path, "r", 200, 0.33, 0.60)  # second on rung 16, S [0.56, 0.64] holds 0.63
    _save(tmp_path, "r", 300, 0.31, 0.63)  # the S leader
    pick = B.best_save([tmp_path], run_id="r", arm="full")
    assert pick["pick"] == "r_00000200_abcd1234" and pick["rule"] == "first on rung 16 with S within its interval"
    assert [row["step"] for row in pick["rows"]] == [100, 200, 300]
    assert pick["rows"][0]["rung16"]["wr"] == 0.35 and pick["rows"][0]["s"]["wr"] == 0.50


def test_another_runs_cells_beside_the_runs_own_are_never_its_pick(tmp_path: Path) -> None:
    _save(tmp_path, "r", 100, 0.30, 0.55)
    _save(tmp_path, "parent", 45000, 0.90, 0.90)  # the parent's anchor cells live in the same directory
    assert B.best_save([tmp_path], run_id="r", arm="full")["pick"] == "r_00000100_abcd1234"


def test_a_six_forfeit_is_not_counted_as_our_win(tmp_path: Path) -> None:
    _save(tmp_path, "r", 100, 0.36, 0.55)
    _save(tmp_path, "r", 200, 0.34, 0.60)
    sidecar(tmp_path, "r", 100, 0.36, suffix="ladder455_n16.full", unit="ladder455_n16", six=_G455,
            wins=207, draws=0, six_findings={"count": 40})  # 207 - 40 forfeits over 536 real games: 0.31
    pick = B.best_save([tmp_path], run_id="r", arm="full")
    by_step = {row["step"]: row for row in pick["rows"]}
    assert pick["pick"] == "r_00000200_abcd1234" and by_step[100]["rung16"]["wr"] == pytest.approx(167 / 536)


def test_a_save_missing_either_reading_is_named_and_left_out(tmp_path: Path) -> None:
    _save(tmp_path, "r", 100, 0.30, 0.55)
    sidecar(tmp_path, "r", 200, 0.40, suffix="ladder455_n16.full", unit="ladder455_n16", six=_G455)
    pick = B.best_save([tmp_path], run_id="r", arm="full")
    assert pick["pick"] == "r_00000100_abcd1234" and pick["left_out"] == ["r_00000200_abcd1234: no S reading"]


def test_a_unit_whose_cells_span_two_instruments_is_refused(tmp_path: Path) -> None:
    _save(tmp_path, "r", 100, 0.30, 0.55)
    _save(tmp_path, "r", 200, 0.34, 0.60, commit="newpin")
    with pytest.raises(ValueError, match="2 instruments"):
        B.best_save([tmp_path], run_id="r", arm="full")


def test_no_save_read_on_both_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both"):
        B.best_save([tmp_path], run_id="r", arm="full")


def test_a_save_read_only_by_a_screen_is_left_out_as_one(tmp_path: Path) -> None:
    _save(tmp_path, "r", 100, 0.30, 0.55)
    sidecar(tmp_path, "r", 200, 0.60, n=128, suffix="ladder455_n16.screen128.full", unit="ladder455_n16", six=_G455)
    sidecar(tmp_path, "r", 200, 0.70, family="strix", suffix="strix256_arena.full", unit="equal_work_arena")
    pick = B.best_save([tmp_path], run_id="r", arm="full")
    assert pick["pick"] == "r_00000100_abcd1234"
    assert pick["left_out"] == ["r_00000200_abcd1234: rung 16 read by a 128-game screen, not the whole book"]
