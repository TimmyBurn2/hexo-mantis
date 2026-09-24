"""tools/strength_frontier.py cell composition: one module for the one tool's three families."""
from __future__ import annotations

from pathlib import Path

import pytest
from _toolpath import load_module_by_path

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def frontier():
    return load_module_by_path("strength_frontier_under_test", _REPO / "tools" / "strength_frontier.py")


@pytest.fixture(scope="module")
def base(frontier, tmp_path_factory):
    from mantis.config.loader import load_config

    config = load_config(str(_REPO / "configs" / "run7.yaml"))
    return config, frontier.base_round_spec(config, work_dir=tmp_path_factory.mktemp("f"))


def _self_cell(**over):
    cell = {"label": "anchor_self_r1", "candidate": "ck.ckpt", "opponent": "ck.ckpt",
            "search_kind": "puct", "sims": 256, "games": 1024, "concurrency": 8}
    cell.update(over)
    return cell


def _cell(**over):
    cell = {"label": "c", "candidate": "bc_full", "opponent": "strix", "strix_sims": 128,
            "search_kind": "gumbel", "sims": 128,
            "games": 4}
    cell.update(over)
    return cell


def test_a_gate_cell_without_book_or_seed_rows_plays_the_configs(frontier, base, tmp_path) -> None:
    config, base_spec = base
    spec = frontier.cell_spec(_self_cell(), base_spec, cell_dir=tmp_path, config=config)
    assert spec.gate.opening_book == config.eval.gate.opening_book
    assert (spec.gate.seed_base, spec.seed_base) == (config.eval.gate.seed_base, config.eval.gate.seed_base)


def test_a_gate_cell_book_and_seed_rows_replace_the_configs(frontier, base, tmp_path) -> None:
    config, base_spec = base
    cell = _self_cell(opening_book="book_v2_pool_s20260915_p4", seed_base=3)
    spec = frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)
    assert spec.gate.opening_book == "book_v2_pool_s20260915_p4"
    assert (spec.gate.seed_base, spec.seed_base) == (3, 3)
    assert spec.gate.run_gate and spec.gate.screen_games == 1024 and spec.gate.deploy_sims == 256
    assert spec.best_snapshot == str(tmp_path / "opponent.pt")


def test_a_strix_rung_cell_book_row_replaces_the_gates_book(frontier, base, tmp_path) -> None:
    config, base_spec = base
    cell = {"label": "bridge_v2", "candidate": "ck.ckpt", "opponent": "strix", "strix_sims": 256,
            "search_kind": "puct", "sims": 256, "games": 256, "opening_book": "book_v2_p4",
            "seed_base": 5}
    spec = frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)
    [job] = spec.rung_jobs
    assert job.opening_book == "book_v2_p4" and job.games == 256
    assert spec.seed_base == 5


def test_a_cell_without_an_opponent_is_refused_by_name(frontier, base, tmp_path) -> None:
    """A cell naming no opponent is refused; the sealbot default went with its rung."""
    config, base_spec = base
    cell = {"label": "no_opp", "candidate": "ck.ckpt", "search_kind": "puct", "sims": 256, "games": 4}
    with pytest.raises(frontier.FrontierCellError, match="names its opponent"):
        frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)


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


def test_a_strix_cell_composes_the_rung_at_the_pinned_checkpoint(frontier, base, tmp_path) -> None:
    config, base_spec = base
    cell = {"label": "strix_A", "candidate": "bc_full", "search_kind": "puct", "sims": 512,
            "opponent": "strix", "strix_sims": 128, "games": 288, "concurrency": 8}
    spec = frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)
    assert len(spec.rung_jobs) == 1
    job = spec.rung_jobs[0]
    assert (job.bot, job.variant, job.opponent_sims, job.games, job.deploy_matched) == (
        "strix", "checkpoint_00237000", 128, 288, True)
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
