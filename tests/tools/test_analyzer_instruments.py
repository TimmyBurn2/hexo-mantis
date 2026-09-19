"""The symmetry maps form a group about the first stone; the sweep and the trace state their gaps."""
from __future__ import annotations

import importlib
import itertools
import random
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def instruments(analyzer):
    return importlib.import_module("analyzer.instruments")


@pytest.fixture(scope="module")
def count_engine(analyzer):
    return lambda: _CountEngine(importlib.import_module("analyzer.engines").Child)


def test_the_twelve_maps_are_distinct_and_the_first_is_the_identity(instruments):
    assert len(instruments.MAPS) == 12 and instruments.MAPS[0] == ("identity", 0, False)
    cells = [(1, 2), (-3, 1), (0, 4)]
    images = {tuple(instruments.transform(c, (0, 0), k, refl) for c in cells) for _n, k, refl in instruments.MAPS}
    assert len(images) == 12


def test_every_map_inverts_about_any_centre(instruments):
    rng = random.Random(3)
    for _n, k, refl in instruments.MAPS:
        for _ in range(20):
            centre = (rng.randint(-5, 5), rng.randint(-5, 5))
            cell = (rng.randint(-9, 9), rng.randint(-9, 9))
            image = instruments.transform(cell, centre, k, refl)
            assert instruments.transform(image, centre, k, refl, inverse=True) == cell
            assert instruments.transform(centre, centre, k, refl) == centre


def test_the_maps_are_closed_under_composition(instruments):
    probe = [(1, 0), (2, 1), (-1, 3)]
    table = {tuple(instruments.transform(c, (0, 0), k, r) for c in probe): (k, r) for _n, k, r in instruments.MAPS}
    for (_a, ka, ra), (_b, kb, rb) in itertools.product(instruments.MAPS, repeat=2):
        composed = tuple(instruments.transform(instruments.transform(c, (0, 0), ka, ra), (0, 0), kb, rb) for c in probe)
        assert composed in table


class _CountEngine:
    """value = stone count / 100, argmax = the first legal cell: exactly equivariant, so the sweep reads spread 0."""

    encoding = "gnn_axis_v1"

    def __init__(self, child_cls):
        self.calls = 0
        self._child = child_cls

    def raw_read(self, board):
        self.calls += 1
        legal = board.legal_moves()
        children = [self._child((q, r), 1.0 if i == 0 else 0.0, 0, 0.0) for i, (q, r) in enumerate(sorted(legal))]
        return SimpleNamespace(value=len(board.get_stones()) / 100.0, children=children, ms=1.0)

    def search(self, board, sims):
        return SimpleNamespace(root_value=0.25, argmax=sorted(board.legal_moves())[0], children=[], root_visits=sims,
                               quiescence_fires=0, ms=2.0)


def test_the_sweep_reads_zero_spread_on_an_equivariant_engine_and_names_the_worst_map(instruments, count_engine):
    eng = count_engine()
    out = instruments.sweep(eng, [(0, 0), (1, 0), (0, 1)])
    assert out["n"] == 12 and out["centre"] == [0, 0] and out["spread"] == 0.0
    assert out["value_min"] == out["value_max"] == pytest.approx(0.03)
    assert out["worst"]["map"] in {m[0] for m in instruments.MAPS[1:]}, "the worst map is never the identity"
    assert out["translation"]["value"] == pytest.approx(0.03) and out["rows"][0]["map"] == "identity"
    assert eng.calls == 13


def test_the_sweep_states_the_empty_board_and_a_translation_off_the_opening_window(instruments, count_engine):
    eng = count_engine()
    assert instruments.sweep(eng, []) == {"absent": "an empty board has no first stone to centre on"}
    out = instruments.sweep(eng, [(2, 2)])
    assert "absent" in out["translation"] and "opening window" in out["translation"]["absent"]


def test_the_trace_is_in_p1s_perspective_with_gaps_for_the_unsearched_and_terminal_plies(instruments, count_engine, positions):
    eng = count_engine()
    assert instruments.p1_view(0.4, "p1") == 0.4 and instruments.p1_view(0.4, "p2") == -0.4
    rows = instruments.trace(eng, [(0, 0), (1, 0), (0, 1)], 0)
    assert [r["ply"] for r in rows] == [0, 1, 2, 3]
    assert rows[1]["to_move"] == "p2" and rows[1]["raw"] == pytest.approx(-0.01) and rows[1]["root"] is None
    rows = instruments.trace(eng, [(0, 0), (1, 0)], 4)
    assert rows[2]["to_move"] == "p2", "p2 still holds the turn's second stone at ply 2"
    assert rows[2]["root"] == pytest.approx(-0.25) and rows[2]["ms"] == 2.0
    rows = instruments.trace(eng, positions["SIX"], 0)
    assert rows[-1] == {"ply": 12, "to_move": None, "raw": None, "root": None, "terminal": "p1"}
