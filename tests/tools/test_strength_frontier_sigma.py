"""`tools/strength_frontier.py` cells may set their OWN σ (`c_scale`, `q_rescale`) — R351(b)'s
four cells vary the deploy head's scale on one net; a cell that names neither plays the config's.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from _toolpath import load_module_by_path

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def frontier():
    path = _REPO / "tools" / "strength_frontier.py"
    return load_module_by_path("strength_frontier_under_test", path)


@pytest.fixture(scope="module")
def base(frontier, tmp_path_factory):
    from mantis.config.loader import load_config

    config = load_config(str(_REPO / "configs" / "run6.yaml"))
    work = tmp_path_factory.mktemp("frontier")
    return config, frontier.base_round_spec(config, work_dir=work)


def _cell(**over):
    cell = {"label": "c", "candidate": "bc_full", "opponent": "strix", "strix_sims": 128,
            "search_kind": "gumbel", "sims": 128,
            "games": 4}
    cell.update(over)
    return cell


def test_a_cell_without_sigma_rows_plays_the_config_sigma(frontier, base, tmp_path) -> None:
    config, base_spec = base
    spec = frontier.cell_spec(_cell(), base_spec, cell_dir=tmp_path, config=config)
    assert (spec.c_visit, spec.c_scale, spec.q_rescale) == (
        config.selfplay.c_visit, config.selfplay.c_scale, config.selfplay.q_rescale)


@pytest.mark.parametrize("c_scale,rescale", [(0.1, True), (1.0, False)])
def test_a_cell_sigma_row_replaces_the_config_sigma(frontier, base, tmp_path, c_scale, rescale) -> None:
    config, base_spec = base
    cell = _cell(c_scale=c_scale, q_rescale=rescale)
    spec = frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)
    assert spec.c_visit == config.selfplay.c_visit
    assert (spec.c_scale, spec.q_rescale) == (c_scale, rescale)
