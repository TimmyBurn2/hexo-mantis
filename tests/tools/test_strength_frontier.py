"""tools/strength_frontier.py cell composition: one module for the one tool's three families."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _toolpath import load_module_by_path

from mantis.config.census import production_configs

_REPO = Path(__file__).resolve().parents[2]
_TACTICS: dict[str, Any] = {
    "kind": "strict_turn", "leaf_turns": 3, "leaf_nodes": 256, "root_turns": 8, "root_nodes": 20000,
    "audit": {"turns": 8, "nodes": 2000, "k": 4, "m": 4, "total_nodes": 40000},
}


@pytest.fixture(scope="module")
def frontier():
    return load_module_by_path("strength_frontier_under_test", _REPO / "tools" / "strength_frontier.py")


@pytest.fixture(scope="module", params=production_configs(_REPO), ids=lambda p: p.name)
def base(frontier, tmp_path_factory, request):
    from mantis.config.loader import load_config

    config = load_config(request.param)
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
    assert spec.gate.run_gate and spec.gate.deploy_sims == 256
    seq = spec.gate.sequential
    assert seq["min_pairs"] == seq["max_pairs"] == seq["check_every_pairs"] == 512, (
        "a model cell is a fixed-N match: the GSPRT plays every pair and checks once"
    )
    assert spec.best_snapshot == str(tmp_path / "opponent.pt")


def test_a_model_cell_below_the_gsprts_two_pairs_is_refused_when_built(frontier, base, tmp_path) -> None:
    """The rule's own spec refuses it at cell build, not the eval child mid-round."""
    config, base_spec = base
    with pytest.raises(frontier.FrontierCellError, match="min_pairs"):
        frontier.cell_spec(_self_cell(games=2), base_spec, cell_dir=tmp_path, config=config)


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
    assert spec.rung_model_sims == 512 and spec.search_kind == "puct"
    assert spec.rung_concurrency == 8, "the strix rung's games in flight are the cell's concurrency"
    assert frontier.cell_channel(cell) == "external"


def test_a_strix_cell_without_strix_sims_is_refused(frontier, base, tmp_path) -> None:
    config, base_spec = base
    cell = {"label": "x", "candidate": "bc_full", "search_kind": "puct", "sims": 256,
            "opponent": "strix", "games": 4}
    with pytest.raises(frontier.FrontierCellError, match="strix_sims"):
        frontier.cell_spec(cell, base_spec, cell_dir=tmp_path, config=config)


def test_a_cell_tactics_row_arms_the_candidate_and_without_one_the_configs_block_plays(frontier, base, tmp_path) -> None:
    config, base_spec = base
    assert frontier.cell_spec(_cell(), base_spec, cell_dir=tmp_path, config=config).tactics == base_spec.tactics
    block = tmp_path / "block.json"
    block.write_text(json.dumps(_TACTICS), encoding="utf-8")
    armed = frontier.load_arm("known-bad", block)
    spec = frontier.cell_spec(_cell(tactics=armed), base_spec, cell_dir=tmp_path, config=config)
    assert spec.tactics == armed and armed["audit"]["mode"] == "inverted"
    assert frontier.cell_spec(_cell(tactics=None), base_spec, cell_dir=tmp_path, config=config).tactics is None
    with pytest.raises(frontier.FrontierCellError, match="--arm full: .*none was given"):
        frontier.load_arm("full", None)


def test_the_readout_sums_the_candidates_rows_and_names_the_proofs_a_game_did_not_bear_out(frontier) -> None:
    def game(index: int, result: str, seat: int, **rows: int) -> dict[str, Any]:
        return {"game_index": index, "result": result, "colors": {"candidate": seat}, "candidate_tactics": rows}

    records = [game(0, "p1", 1, root_proofs_found=2, descents=10, stones=5),
               {**game(1, "p1", 2, root_proofs_found=1, descents=6, stones=4), "termination": "six_in_a_row"},
               {**game(2, "draw", 1, root_proofs_found=1, stones=3), "termination": "ply_cap"},
               game(3, "p2", 1, descents=4, stones=2),
               {"game_index": 4, "result": "p1", "colors": {"candidate": 1}}]
    assert frontier.tactics_readout(records) == {
        "rows": {"root_proofs_found": 4, "descents": 20, "stones": 14},
        "proof_games_lost": [{"game_index": 1, "termination": "six_in_a_row"}],
        "proof_games_drawn": [{"game_index": 2, "termination": "ply_cap"}]}


def test_a_ruler_cell_arms_our_head_alone_and_a_snapshot_cell_arms_both(frontier) -> None:
    """The receipt names the sides a block armed: a snapshot opponent plays through the deploy-matched gate pair."""
    assert frontier.armed_sides({"opponent": "strix"}) == frontier.armed_sides({"opponent": "six"}) == "candidate"
    assert frontier.armed_sides({"opponent": "anchor_self"}) == "both"


def test_the_module_hash_moves_with_a_tactics_source_byte_and_with_no_other(frontier, tmp_path) -> None:
    src = tmp_path / "crates" / "mantis-search" / "src"
    (src / "tactics").mkdir(parents=True)
    (src / "mcts").mkdir()
    for rel in ("tactics/mod.rs", "tactics/grid.rs", "mcts/tactics_wiring.rs", "mcts/tactics_root.rs", "mcts/puct.rs"):
        (src / rel).write_text(rel, encoding="utf-8")
    first = frontier.tactics_module_sha256(tmp_path)
    (src / "mcts" / "puct.rs").write_text("another search", encoding="utf-8")
    assert frontier.tactics_module_sha256(tmp_path) == first
    (src / "mcts" / "tactics_new.rs").write_text("a new wiring file", encoding="utf-8")
    second = frontier.tactics_module_sha256(tmp_path)
    assert second != first, "a new mcts/tactics_*.rs joins the hash"
    (src / "tactics" / "grid.rs").write_text("another grid", encoding="utf-8")
    assert frontier.tactics_module_sha256(tmp_path) != second
    for rel in ("mcts/tactics_wiring.rs", "mcts/tactics_root.rs", "mcts/tactics_new.rs"):
        (src / rel).unlink()
    with pytest.raises(frontier.FrontierCellError, match="no tactics sources"):
        frontier.tactics_module_sha256(tmp_path)
    assert len(frontier.tactics_module_sha256(_REPO)) == 64, "the tree's sources are where the hash reads them"


def test_main_arms_every_cell_under_its_own_label_and_a_cells_file_names_no_block(frontier, monkeypatch,
                                                                                    tmp_path) -> None:
    """Reds without the label suffix, which gives two arms' games one directory; a raw block has no way in."""
    from types import SimpleNamespace

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(frontier, "run_cell", lambda cell, **_kw: seen.append(dict(cell)) or {"label": cell["label"],
                                                                                              "rc": 0})
    monkeypatch.setattr(frontier, "format_row", lambda _record: "")
    monkeypatch.setattr(frontier, "base_round_spec", lambda _config, work_dir: SimpleNamespace(allocator_posture=None))
    block, cells, raw = tmp_path / "block.json", tmp_path / "cells.json", tmp_path / "raw.json"
    block.write_text(json.dumps(_TACTICS), encoding="utf-8")
    cells.write_text(json.dumps([_cell(label="a"), _cell(label="b")]), encoding="utf-8")
    raw.write_text(json.dumps([_cell(label="c", tactics=_TACTICS)]), encoding="utf-8")
    config = str(production_configs(_REPO)[0])
    run = ["--config", config, "--work-dir", str(tmp_path / "w")]
    assert frontier.main([*run, "--cells", str(cells), "--arm", "known-bad", "--tactics-block", str(block)]) == 0
    assert [c["label"] for c in seen] == ["a_known-bad", "b_known-bad"]
    assert all(c["tactics_arm"] == "known-bad" and c["tactics"]["audit"]["mode"] == "inverted" for c in seen)
    with pytest.raises(frontier.FrontierCellError, match="one way in"):
        frontier.main([*run, "--cells", str(raw)])
    with pytest.raises(SystemExit):
        frontier.main([*run, "--cells", str(cells), "--tactics-block", str(block)])
    assert len(seen) == 2
