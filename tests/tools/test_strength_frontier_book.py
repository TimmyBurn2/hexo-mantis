"""`tools/strength_frontier.py` cells may name their OWN `opening_book` and `seed_base` (BOOK_V2's replays); a cell that names neither plays the config's."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def frontier():
    path = _REPO / "tools" / "strength_frontier.py"
    spec = importlib.util.spec_from_file_location("strength_frontier_book_under_test", path)
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


def _self_cell(**over):
    cell = {"label": "anchor_self_r1", "candidate": "ck.ckpt", "opponent": "ck.ckpt",
            "search_kind": "puct", "sims": 256, "games": 1024, "concurrency": 8}
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
    """R362(c): the old default opponent (`sealbot_d5`) went with the sealbot rung; a cell that
    names none is refused rather than silently played against anything."""
    import pytest

    config, base_spec = base
    cell = {"label": "no_opp", "candidate": "ck.ckpt", "search_kind": "puct", "sims": 256, "games": 4}
    with pytest.raises(frontier.FrontierCellError, match="names its opponent"):
        frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)
