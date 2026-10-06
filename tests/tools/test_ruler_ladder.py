"""The ruler ladder's reading: a save's N50 from the two rungs bracketing 50 %, and the screen's rung pair."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from _toolpath import load_module_by_path

_REPO = Path(__file__).resolve().parents[2]
L = load_module_by_path("ruler_ladder_t", _REPO / "tools" / "ruler_ladder.py")


def test_n50_is_where_the_logit_line_between_two_rungs_crosses_zero_in_log2_nodes() -> None:
    assert L.n50(32, 0.6, 64, 0.4) == L.N50(nodes=pytest.approx(2 ** 5.5), extrapolated=False, inverted=False)
    third = L.n50(16, 0.5, 32, 0.25)
    assert third.nodes == pytest.approx(16.0) and not third.extrapolated


def test_a_pair_on_one_side_of_50_extrapolates_and_says_so() -> None:
    above = L.n50(32, 0.8, 64, 0.6)
    assert above.extrapolated and above.nodes > 64
    below = L.n50(16, 0.4, 32, 0.3)
    assert below.extrapolated and below.nodes < 16


def test_an_inverted_pair_has_no_n50() -> None:
    assert L.n50(32, 0.4, 64, 0.45) == L.N50(nodes=None, extrapolated=False, inverted=True)


@pytest.mark.parametrize(("screen", "pair"), [
    ({16: 0.62, 32: 0.51, 64: 0.38, 128: 0.27}, (32, 64)),
    ({16: 0.45, 32: 0.33, 64: 0.2, 128: 0.1}, (16, 32)),
    ({16: 0.7, 32: 0.5, 64: 0.55, 128: 0.3}, (64, 128)),
])
def test_the_pair_is_the_highest_rung_at_or_above_50_and_the_rung_above_it(screen: dict, pair: tuple) -> None:
    assert L.pick_pair(screen) == pair


def test_a_screen_still_at_or_above_50_at_its_top_rung_names_the_rung_to_read_next() -> None:
    with pytest.raises(ValueError, match="256"):
        L.pick_pair({16: 0.8, 32: 0.7, 64: 0.6, 128: 0.52})


def _sidecar(tmp: Path, ckpt: str, nodes: int, wr: float, lo: float, hi: float) -> None:
    body = {"unit": f"ladder455_n{nodes}", "wr": wr, "wr_ci_lower": lo, "wr_ci_upper": hi, "games": 576}
    (tmp / f"{ckpt}.ladder455_n{nodes}.full.json").write_text(json.dumps(body), encoding="utf-8")


def test_a_save_reads_its_two_sidecars_into_n50_with_an_interval_and_the_raw_rates(tmp_path: Path) -> None:
    _sidecar(tmp_path, "r.ckpt", 32, 0.58, 0.54, 0.62)
    _sidecar(tmp_path, "r.ckpt", 64, 0.42, 0.38, 0.46)
    row = L.read_save(tmp_path / "r.ckpt", (32, 64), arm="full")
    assert row["rates"] == {"32": 0.58, "64": 0.42}
    assert row["n50"] == pytest.approx(2 ** (5 + math.log(0.58 / 0.42) / (2 * math.log(0.58 / 0.42))))
    assert row["n50_lo"] < row["n50"] < row["n50_hi"] and not row["extrapolated"]
    assert L.read_save(tmp_path / "r.ckpt", (32, 64), arm="full") == row, "the interval's draw is seeded"


def test_a_missing_rung_is_named() -> None:
    with pytest.raises(FileNotFoundError, match="ladder455_n64"):
        L.read_save(Path("/nonexistent/r.ckpt"), (64, 128), arm="full")
