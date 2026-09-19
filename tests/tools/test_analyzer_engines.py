"""Engines: stamped names are discovered, snapshots are a stated gap, and one stamp rebuilds the deploy-matched head."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from mantis._engine import Board


@pytest.fixture(scope="module")
def engines(analyzer):
    return importlib.import_module("analyzer.engines")


@pytest.fixture(scope="module")
def stamp(tmp_path_factory, mint_stamp) -> Path:
    return mint_stamp(tmp_path_factory.mktemp("ckpt"))


@pytest.fixture(scope="module")
def engine(engines, stamp):
    info = engines.discover([stamp.parent])[0]
    eng = engines.MantisEngine(info, device="cpu", threads=2)
    yield eng
    eng.close()


def test_discovery_lists_stamped_names_oldest_first_and_reports_a_snapshot_as_a_stated_gap(engines, tmp_path):
    (tmp_path / "run9_00000300_0badcafe.ckpt").write_bytes(b"")
    (tmp_path / "run9_00000100_deadbeef.ckpt").write_bytes(b"")
    (tmp_path / "best_model.pt").write_bytes(b"")
    (tmp_path / "best_model.pt.provenance.json").write_text(json.dumps({"step": 100, "run_id": "run9"}), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    rows = engines.discover([tmp_path])
    assert [r.id for r in rows] == ["run9_00000100_deadbeef", "run9_00000300_0badcafe", f"{tmp_path.name}/best_model.pt"]
    assert rows[0].kind == engines.MANTIS and (rows[0].run_id, rows[0].step, rows[0].sha8) == ("run9", 100, "deadbeef")
    gap = rows[2]
    assert gap.kind == engines.SNAPSHOT_GAP and "no stamp" in gap.note and "step 100 of run9" in gap.note
    assert gap.as_dict()["kind"] == "snapshot_gap"


def test_discovery_also_looks_one_level_down_in_run_checkpoints_dirs(engines, tmp_path):
    sub = tmp_path / "run9" / "checkpoints"
    sub.mkdir(parents=True)
    (sub / "run9_00000100_deadbeef.ckpt").write_bytes(b"")
    assert [r.id for r in engines.discover([tmp_path])] == ["run9_00000100_deadbeef"]


def test_the_card_states_the_stamps_facts_and_the_seed(engines, engine):
    card = engine.card
    assert card["run_id"] == "an1" and card["step"] == 7 and card["encoding"] == "gnn_axis_v1"
    assert card["search_kind"] == "puct" and card["deploy_sims"] == 150 and card["radius"] == engine.radius
    assert card["device"] == "cpu" and card["seed"] == engines.ANALYZER_GUMBEL_SEED == 0
    assert card["params"] > 0 and len(card["net_hash"]) >= 8


def test_the_raw_read_is_the_nets_value_and_the_decoded_priors(engines, engine):
    board = Board.with_encoding_name(engine.encoding)
    for q, r in [(0, 0), (1, 0), (0, 1)]:
        board.apply_move(q, r)
    raw = engine.raw_read(board)
    _dense, _overflow, net_value, _centre = engine.engine.infer_ls(board.clone())
    assert raw.value == pytest.approx(net_value, abs=1e-6)
    assert len(raw.children) == len(board.legal_moves())
    assert sum(c[2] for c in raw.children) == pytest.approx(1.0, abs=1e-4)
    assert all(c[3] == 0 for c in raw.children), "a raw read visits no child"
    again = engine.raw_read(board)
    assert again.value == raw.value, "the cached tree is re-rooted, not accumulated"


def test_the_search_is_the_heads_own_answer_with_its_counters(engines, engine):
    board = Board.with_encoding_name(engine.encoding)
    for q, r in [(0, 0), (1, 0), (0, 1)]:
        board.apply_move(q, r)
    s = engine.search(board, 8)
    assert -1.0 <= s.root_value <= 1.0 and s.root_visits >= 8 and s.quiescence_fires >= 0
    assert s.argmax in {c[0] for c in s.children} and sum(c[3] for c in s.children) >= 7
    assert s.ms > 0


def test_a_gumbel_engine_repeats_its_answer_on_a_fresh_player(engines, tmp_path, mint_stamp):
    mint_stamp(tmp_path, run_id="gm", deploy_kind="gumbel")
    eng = engines.MantisEngine(engines.discover([tmp_path])[0], device="cpu", threads=2)
    try:
        board = Board.with_encoding_name(eng.encoding)
        for q, r in [(0, 0), (1, 0), (0, 1)]:
            board.apply_move(q, r)
        assert eng.card["search_kind"] == "gumbel"
        a, b = eng.search(board, 8), eng.search(board, 8)
        assert a.argmax == b.argmax and a.root_value == pytest.approx(b.root_value)
    finally:
        eng.close()


def test_a_stamp_missing_a_needed_key_is_refused_by_the_keys_name(engines, stamp, monkeypatch):
    real = engines.load_checkpoint

    def broken(path):
        ck = real(path)
        del ck.config["selfplay"]["gumbel_m"]
        return ck

    monkeypatch.setattr(engines, "load_checkpoint", broken)
    with pytest.raises(engines.EngineLoadError, match="gumbel_m"):
        engines.MantisEngine(engines.discover([stamp.parent])[0], device="cpu", threads=1)
