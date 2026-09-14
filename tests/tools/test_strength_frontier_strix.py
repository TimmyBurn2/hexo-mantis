"""`tools/strength_frontier.py` plays a STRIX cell (RUNG-2): the pinned checkpoint at `strix_sims`, ours on `strix_model_sims`."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def frontier():
    path = _REPO / "tools" / "strength_frontier.py"
    spec = importlib.util.spec_from_file_location("strength_frontier_strix_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base(frontier, tmp_path_factory):
    from mantis.config.loader import load_config

    config = load_config(str(_REPO / "configs" / "run7.yaml"))
    return config, frontier.base_round_spec(config, work_dir=tmp_path_factory.mktemp("f"))


def test_a_strix_cell_composes_the_rung_at_the_pinned_checkpoint(frontier, base, tmp_path) -> None:
    config, base_spec = base
    cell = {"label": "strix_A", "candidate": "bc_full", "search_kind": "puct", "sims": 512,
            "opponent": "strix", "strix_sims": 128, "games": 288, "concurrency": 8}
    spec = frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)
    assert len(spec.rung_jobs) == 1
    job = spec.rung_jobs[0]
    assert (job.bot, job.variant, job.depth, job.opponent_sims, job.games, job.deploy_matched) == (
        "strix", "checkpoint_00237000", None, 128, 288, True)
    assert job.opening_book == config.eval.gate.opening_book
    assert spec.strix_model_sims == 512 and spec.search_kind == "puct"
    assert spec.rung_concurrency == 8, "the strix rung's games in flight are the cell's concurrency"
    assert frontier.cell_channel(cell) == "external"


def test_a_strix_cell_without_strix_sims_is_refused(frontier, base, tmp_path) -> None:
    config, base_spec = base
    cell = {"label": "x", "candidate": "bc_full", "search_kind": "puct", "sims": 256,
            "opponent": "strix", "games": 4}
    with pytest.raises(frontier.FrontierCellError, match="strix_sims"):
        frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)
