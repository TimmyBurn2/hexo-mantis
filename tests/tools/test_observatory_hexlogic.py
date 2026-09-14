"""The observatory's hex facts are the viewer's, copied; this guard reds the day they drift."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def new(observatory):
    return importlib.import_module("observatory.readers.hexlogic")


@pytest.fixture(scope="module")
def old(viewer):
    return importlib.import_module("viewer.hexlogic")


def _six_for_p1() -> list[list[int]]:
    p1 = [[k, 0] for k in range(6)]
    p2 = [[2 * k, 5 + k % 2] for k in range(6)]
    return [p1[0], p2[0], p2[1], p1[1], p1[2], p2[2], p2[3], p1[3], p1[4], p2[4], p2[5], p1[5]]


def test_owner_and_win_line_agree_with_the_viewer(new, old):
    assert [new.owner(p) for p in range(9)] == [old.owner(p) for p in range(9)]
    assert new.win_line(_six_for_p1()) == old.win_line(_six_for_p1()) == [[k, 0] for k in range(6)]
    assert new.win_line(_six_for_p1()[:-1]) is None


def test_the_copy_is_byte_identical_to_the_original_until_phase_6_deletes_it(new, old):
    assert Path(new.__file__).read_text(encoding="utf-8") == Path(old.__file__).read_text(encoding="utf-8")
