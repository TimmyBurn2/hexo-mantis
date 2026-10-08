"""The distillation harness: Six's planes and root screen, rows put back in move order by their games, the targets on our legal set, the rings, and the read's rules."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mantis import _engine
from mantis.diagnostics import ring_reader as R
from mantis.util.loadpkg import load_tools_package

ENC = "gnn_axis_r8"
RADIUS = 8
FIXTURES = Path(__file__).resolve().parents[2] / "vendor" / "external" / "six" / "engine" / "tests" / "fixtures" / "planes.txt"


@pytest.fixture(scope="module")
def d() -> Any:
    load_tools_package("distill")
    return {m: importlib.import_module(f"distill.{m}") for m in ("planes", "corpus", "rings", "read", "teacher", "label")}


def test_planes_on_a_hand_position(d):
    """The opener's stone, then the other side's first: the mover is mid-turn, the crop centre the rounded mean."""
    P = d["planes"]
    planes, center = P.six_planes(np.array([[0, 0], [1, 0]]), RADIUS)
    assert center == (1, 0)
    flat = planes.reshape(8, -1)
    own, opp = 12 * 25 + 12, 12 * 25 + 11
    assert np.nonzero(flat[1])[0].tolist() == [own] and np.nonzero(flat[2])[0].tolist() == [opp]
    assert np.nonzero(flat[4])[0].tolist() == [own] and np.nonzero(flat[5])[0].tolist() == [opp]
    assert flat[6].all() and not flat[7].any() and flat[0].all()
    assert flat[3][own] == 0 and flat[3][opp] == 0 and flat[3].sum() > 0
    assert P.crop_center(np.array([[-1, 0], [-2, 0]])) == (-1, 0)


@pytest.mark.skipif(not FIXTURES.is_file(), reason="Six's fixtures come with `make vendor.six`")
def test_planes_equal_sixs_own_fixtures(d):
    lines = FIXTURES.read_text().splitlines()
    checked = 0
    for i, line in enumerate(lines):
        if not line.startswith("position"):
            continue
        w = line.split()
        radius, n = int(w[1]), int(w[2])
        moves = np.array(list(map(int, w[3:3 + 2 * n])), dtype=np.int64).reshape(-1, 2)
        got, center = d["planes"].six_planes(moves, radius)
        assert center == tuple(map(int, lines[i + 1].split()[1:3]))
        for k in range(8):
            bits = np.unpackbits(np.frombuffer(bytes.fromhex(lines[i + 2 + k].split()[2]), np.uint8), bitorder="little")
            assert np.array_equal(got.reshape(8, -1)[k], bits[:625].astype(np.float32)), (line, k)
        checked += 1
    assert checked > 100


def test_root_tactics_screen(d):
    act = d["planes"].root_tactics_act
    opp_four = [(0, 0), (1, 0), (2, 0), (0, 5), (0, 6), (3, 0), (4, 0)]
    assert act(opp_four), "the opponent's open four must force the mover"
    mover_win = [(0, 0), (5, 5), (6, 5), (1, 0), (2, 0), (5, 7), (6, 7), (3, 0), (0, 9), (9, 9), (9, 10)]
    assert act(mover_win), "two stones on a four complete six this turn"
    assert not act(mover_win[:8]), "one stone on a four does not"
    assert not act(opp_four[:3])


def _random_game(seed: int, plies: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    board = _engine.Board.with_encoding_name(ENC)
    moves: list[tuple[int, int]] = []
    for _ in range(plies):
        legal = board.legal_moves()
        q, r = legal[int(rng.integers(len(legal)))]
        board.apply_move(q, r)
        moves.append((q, r))
        if board.winner() is not None:
            break
    return moves


def _world(tmp: Path, d: Any, n_games: int = 6, plies: int = 14) -> tuple[Path, Path, list[list[tuple[int, int]]]]:
    """Game shards and a ring of every position of those games, stones pushed in a shuffled order."""
    P = d["planes"]
    games = [_random_game(100 + g, plies) for g in range(n_games)]
    shard = tmp / "games" / "games_runx_seg0001_2026100800.jsonl"
    shard.parent.mkdir(parents=True)
    with shard.open("w") as fh:
        for g, mv in enumerate(games):
            fh.write(json.dumps({"channel": "selfplay", "run_id": "runx", "game_id": f"g{g}", "step": 900 + g,
                                 "moves": [list(m) for m in mv], "result": "p1" if g % 2 else "p2"}) + "\n")
    buf = _engine.HexgBuffer(1024, ENC, 16)
    rng = np.random.default_rng(7)
    for g, mv in enumerate(games):
        turns = (len(mv) + 1) // 2
        for k in range(len(mv)):
            board = _engine.Board.with_encoding_name(ENC)
            for q, r in mv[:k]:
                board.apply_move(q, r)
            stones = [(q, r, P.player_for_stone(i)) for i, (q, r) in enumerate(mv[:k])]
            rng.shuffle(stones)
            q, r = board.legal_moves()[0]
            buf.push_graph_position(stones, [(q, r, 1.0)], P.player_for_stone(k), P.stones_left_before(k), k,
                                    bool(k % 2), 1.0 if g % 2 else -1.0, True, turns, 1000 + g, 0.0)
    ring = tmp / "ring" / "runx_00001000_abcdef12.ckpt.ring.bin"
    ring.parent.mkdir()
    buf.save_to_path(str(ring))
    return ring, shard.parent, games


def _corpus(tmp: Path, d: Any) -> tuple[dict[str, np.ndarray], list[list[tuple[int, int]]]]:
    ring, gdir, games = _world(tmp, d)
    out, _meta = d["corpus"].build_corpus([ring], sorted(gdir.glob("*.jsonl")), run_id="runx", heldout_rows=20,
                                          split_seed=1, openings_n=4, openings_seed=5, openings_plies=5, step_margin=500)
    return out, games


def test_every_ring_row_finds_its_game_and_ply(tmp_path, d):
    out, games = _corpus(tmp_path, d)
    ring_rows = out["source"] == d["corpus"].SRC_RING
    assert ring_rows.sum() == sum(len(g) for g in games)
    for i in np.nonzero(ring_rows)[0]:
        moves = d["label"].row_moves(out, int(i))
        assert len(moves) == out["ply_index"][i]
        g = int(out["game"][i])
        assert [tuple(m) for m in moves] == games[g][:len(moves)]
    held_games = set(out["game"][out["heldout"]].tolist())
    assert held_games and all(out["heldout"][out["game"] == g].all() for g in held_games), "a split game"
    assert (out["source"] == d["corpus"].SRC_OPENING).sum() == 4
    assert not out["heldout"][out["source"] == d["corpus"].SRC_OPENING].any()


def _labels(out: dict[str, np.ndarray], d: Any, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(out["game"])
    centers = np.array([d["planes"].six_planes(d["label"].row_moves(out, i), RADIUS)[1] for i in range(n)])
    return {"policy": rng.normal(size=(n, 625)).astype(np.float32), "value_logits": rng.normal(size=(n, 2)).astype(np.float32),
            "score": np.zeros(n, np.float32), "center": centers.astype(np.int32)}


def test_the_target_support_is_the_engines_legal_set_in_the_crop(tmp_path, d):
    out, _ = _corpus(tmp_path, d)
    labels = _labels(out, d)
    rows = np.arange(len(out["game"]))
    rec = d["rings"].legal_check(out, labels, rows, radius=RADIUS, encoding=ENC)
    assert rec["rows"] == len(rows)
    empty = d["rings"].legal_support(np.zeros((0, 2), np.int64), (0, 0), RADIUS)
    assert len(empty) == 25


def test_top_targets_and_the_slot_rule(d):
    Rg = d["rings"]
    logits = np.zeros(625, np.float32)
    logits[[10, 20, 30]] = [3.0, 2.0, 1.0]
    idx, p = Rg.top_targets(logits, np.array([5, 10, 20, 30]), 3)
    assert idx.tolist() == [10, 20, 30] and p.sum() < 1.0
    full = Rg.top_targets(logits, np.array([5, 10, 20, 30]), 8)[1]
    assert abs(full.sum() - 1.0) < 1e-12
    peaked = np.zeros((10, 256), np.float32)
    peaked[:, 0] = 1.0
    assert Rg.choose_slots(peaked)[0] == 64
    flat = np.full((10, 256), 1 / 256, np.float32)
    assert Rg.choose_slots(flat)[0] == 256


def test_rings_round_trip_and_sample(tmp_path, d):
    Rg = d["rings"]
    out, _ = _corpus(tmp_path, d)
    labels = _labels(out, d)
    rows = np.arange(len(out["game"]))
    targets = Rg.compute_targets(out, labels, rows, radius=RADIUS, workers=2, chunk=16, log=lambda _s: None)
    values = Rg.teacher_value(labels, rows)
    path = tmp_path / "teacher.ring.bin"
    rec = Rg.write_teacher_ring(path, out, labels, rows, targets, values, 64, encoding=ENC)
    ring = R.load_ring(path)
    assert ring.header.size == len(rows) and rec["slots"] == 64
    seg = np.repeat(np.arange(ring.header.size), ring.n_visits)
    np.testing.assert_allclose(np.bincount(seg, weights=ring.visits["prob"], minlength=ring.header.size), 1.0, atol=1e-5)
    assert (ring.tail_mass == 0).all() and (ring.is_full_search == 1).all()
    np.testing.assert_allclose(ring.outcome, values, atol=1e-6)
    buf = _engine.HexgBuffer(len(rows) + 8, ENC, 64)
    buf.load_from_path(str(path))
    buf.seed_sampler(1)
    buf.sample_graph_batch(8)
    perm = rows[np.random.default_rng(2).permutation(len(rows))]
    kb = Rg.compute_targets(out, labels, rows, radius=RADIUS, source_rows=perm, workers=2, chunk=16, log=lambda _s: None)
    for j in (0, len(rows) // 2, len(rows) - 1):
        support = Rg.legal_support(d["label"].row_moves(out, j), tuple(labels["center"][j]), RADIUS)
        want = Rg.top_targets(labels["policy"][perm[j]], support, 256)[0]
        assert kb["idx"][j, :len(want)].tolist() == want.tolist()
    ring_rows = rows[out["source"] == d["corpus"].SRC_RING]
    Rg.write_control_ring(tmp_path / "control.ring.bin", out, ring_rows, encoding=ENC)
    ctl = R.load_ring(tmp_path / "control.ring.bin")
    np.testing.assert_array_equal(ctl.is_full_search, out["is_full_search"][ring_rows])
    np.testing.assert_allclose(ctl.outcome, out["outcome"][ring_rows])
    with pytest.raises(ValueError, match="ring rows only"):
        Rg.write_control_ring(tmp_path / "bad.ring.bin", out, rows, encoding=ENC)


def _sidecar(directory: Path, ckpt: str, suffix: str, wr: float, half: float = 0.04, forfeits: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    wins = round(wr * 576)
    (directory / f"{ckpt}.{suffix}.full.json").write_text(json.dumps({
        "rc": 0, "games": 576, "eff_n": 576, "wins": wins, "draws": 0, "wr": wins / 576, "wr_ci_lower": wr - half,
        "wr_ci_upper": wr + half, "six_findings": {"count": forfeits}, "checkpoint_sha256": f"sha-{ckpt}",
        "unit": suffix, "started_utc": f"{ckpt}-{suffix}"}))


def _spec(tmp: Path, arms: dict[str, tuple[str, float, float]], ref=(0.32, 0.63), ctl=(0.32, 0.63), kb=0.05) -> dict:
    cells = tmp / "cells"
    _sidecar(cells, "ref.ckpt", "ladder455_n16", ref[0])
    _sidecar(cells, "ref.ckpt", "strix256_arena", ref[1])
    spec: dict[str, Any] = {"reference": ["ref.ckpt"], "control": "ctl", "knownbad": "kb.ckpt", "arms": {}}
    for name, (cls, r16, s) in {**arms, "ctl": ("control", *ctl)}.items():
        saves = [f"{name}_10k.ckpt", f"{name}_15k.ckpt"]
        for ck in saves:
            _sidecar(cells, ck, "ladder455_n16", r16)
            _sidecar(cells, ck, "strix256_arena", s)
        spec["arms"][name] = {"class": cls, "saves": saves}
    _sidecar(cells, "kb.ckpt", "ladder455_n16", kb)
    return spec


def test_the_read_outcomes(tmp_path, d):
    V = d["read"].verdict
    up = (0.40, 0.70)
    flat = (0.32, 0.63)
    got = V(_spec(tmp_path / "a", {"n1": ("narrow", *up), "w1": ("wide", *up)}), [tmp_path / "a" / "cells"])
    assert got["outcome"] == "narrow: n1" and not got["halt"]
    got = V(_spec(tmp_path / "b", {"n1": ("narrow", *flat), "w1": ("wide", *up)}), [tmp_path / "b" / "cells"])
    assert got["outcome"] == "wide: w1"
    got = V(_spec(tmp_path / "c", {"n1": ("narrow", 0.40, 0.63)}), [tmp_path / "c" / "cells"])
    assert got["outcome"] == "none", "one ruler alone does not reach"
    got = V(_spec(tmp_path / "e", {"n1": ("narrow", *up)}, ctl=(0.15, 0.63)), [tmp_path / "e" / "cells"])
    assert got["halt"] and not got["control_near"] and got["outcome"] == "HALT"
    got = V(_spec(tmp_path / "f", {"n1": ("narrow", *up)}, kb=0.25), [tmp_path / "f" / "cells"])
    assert got["halt"] and not got["knownbad_valid"]
    got = V(_spec(tmp_path / "g", {"n1": ("narrow", *up)}, ctl=up), [tmp_path / "g" / "cells"])
    assert got["control_above_line"] and abs(got["arms"]["n1"]["v_control"]["rung16"]["delta"]) < 1e-12


def test_a_copied_receipt_pools_once(tmp_path, d):
    """The same cell mirrored into a second root is one reading of the reference, not two."""
    spec = _spec(tmp_path / "a", {"n1": ("narrow", 0.40, 0.70)})
    copy = tmp_path / "copy"
    copy.mkdir()
    for f in (tmp_path / "a" / "cells").glob("ref.ckpt.*"):
        (copy / f.name).write_text(f.read_text())
    one = d["read"].verdict(spec, [tmp_path / "a" / "cells"])
    two = d["read"].verdict(spec, [tmp_path / "a" / "cells", copy])
    assert one["reference"]["rung16"]["logit"] == two["reference"]["rung16"]["logit"]
    assert one["reference"]["rung16"]["se"] == two["reference"]["rung16"]["se"]


def test_forfeits_leave_our_wins_and_the_interval(tmp_path, d):
    _sidecar(tmp_path, "x.ckpt", "ladder455_n16", 0.5, forfeits=48)
    cell = d["read"].read_cell(tmp_path / "x.ckpt.ladder455_n16.full.json")
    assert cell.wr == pytest.approx((288 - 48) / 528) and cell.games == 528
    assert cell.lo < cell.wr < cell.hi and (cell.lo, cell.hi) != (0.46, 0.54)


def test_the_lineage_tag_and_a_device_the_teacher_refuses(d):
    assert d["teacher"].lineage_tag("f2b5ec2d4d7ec42e", "gen0455") == "six-f2b5ec2-gen455"
    with pytest.raises(d["teacher"].TeacherUnavailable, match="neither"):
        d["teacher"].Teacher(device="mps")


def test_the_rings_command_leaves_the_empty_board_out_and_writes_provenance(tmp_path, d):
    """Teacher rings start at ply 1 (Six never evaluates the empty board); every ring names its lineage beside it."""
    out, _ = _corpus(tmp_path, d)
    labels = _labels(out, d)
    root = tmp_path / "corpus"
    (root / "labels").mkdir(parents=True)
    np.savez(root / "corpus.npz", **dict(out))
    for key, arr in labels.items():
        np.save(root / "labels" / f"{key}.npy", arr)
    (root / "labels" / "labels.json").write_text(json.dumps({"teacher": {"lineage": "six-aaaaaaa-gen1"}}))
    cli = importlib.import_module("distill.cli")
    assert cli.main(["rings", "--corpus", str(root), "--check-rows", "10"]) == 0
    rings = root / "rings"
    labelled = int(((~out["heldout"]) & (out["k"] >= 1)).sum())
    assert R.load_ring(rings / "teacher_train.ring.bin").header.size == labelled
    assert (R.load_ring(rings / "teacher_train.ring.bin").ply_index >= 1).all()
    Rg = d["rings"]
    assert Rg.ring_lineage(rings / "teacher_train.ring.bin") == ("six-aaaaaaa-gen1",)
    assert Rg.ring_lineage(rings / "knownbad_train.ring.bin") == ("six-aaaaaaa-gen1",)
    assert Rg.ring_lineage(rings / "control_train.ring.bin") == ()
    with pytest.raises(FileNotFoundError):
        Rg.ring_lineage(tmp_path / "nowhere.ring.bin")


def _a_graph_config() -> Path:
    """A minted config of the census whose encoding is the harness's (any will do: the arm overrides its shape)."""
    import yaml

    for path in sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.yaml")):
        body = yaml.safe_load(path.read_text())
        if (body.get("identity") or {}).get("encoding") == ENC and (body.get("identity") or {}).get("arch_kind"):
            return path
    raise AssertionError(f"no minted config declares {ENC}")


@pytest.mark.integration
def test_a_tiny_arm_trains_saves_stamped_and_a_warm_arm_inherits(tmp_path, d):
    """The production trainer on a frozen teacher ring: the recipe's cosine, the saves' stamp and lineage, a warm start's inheritance."""
    load_tools_package("distill")
    T = importlib.import_module("distill.train")
    from mantis.train.checkpoints import load_checkpoint

    Rg = d["rings"]
    out, _ = _corpus(tmp_path, d)
    labels = _labels(out, d)
    rows = np.arange(len(out["game"]))
    targets = Rg.compute_targets(out, labels, rows, radius=RADIUS, workers=2, chunk=16, log=lambda _s: None)
    ring = tmp_path / "teacher.ring.bin"
    Rg.write_teacher_ring(ring, out, labels, rows, targets, Rg.teacher_value(labels, rows), 64, encoding=ENC)
    Rg.write_provenance(ring, kind="teacher_train", lineage=["six-aaaaaaa-gen1"], record={})
    base = _a_graph_config()
    recipe = T.Recipe(base_config=base, steps=3, save_at=(2, 3), lr=1e-3, eta_min=1e-4, seed=11)
    arm = T.ArmSpec(run_id="tinyarm", hidden=16, layers=1, ring=ring, warm_start=None, value_mask_p=0.0)
    rec = T.train_arm(arm, recipe, tmp_path / "arm", device="cpu", heldout_ring=ring, heldout_every=1000,
                      heldout_batches=1, sample_threads=1, caps_override=None, log=lambda _s: None)
    assert sorted(rec["saves"]) == [2, 3] and rec["skipped_steps"] == 0
    assert len(list((tmp_path / "arm" / "checkpoints").glob("*.ckpt"))) == 2, "a periodic save beside the named ones"
    ck = load_checkpoint(rec["saves"][3]["path"], declared_encoding=ENC)
    assert ck.metadata.lineage == ("six-aaaaaaa-gen1",)
    assert (ck.metadata.arch.hidden, ck.metadata.arch.num_layers) == (16, 1)
    assert ck.config["train"]["scheduler_t_max"] == 3 and ck.config["identity"]["warm_start"] is None
    warm = T.ArmSpec(run_id="tinywarm", hidden=16, layers=1, ring=ring, warm_start=Path(rec["saves"][3]["path"]),
                     value_mask_p=0.125)
    rec_w = T.train_arm(warm, recipe, tmp_path / "warm", device="cpu", heldout_ring=None, heldout_every=1000,
                        heldout_batches=1, sample_threads=1, caps_override=None, log=lambda _s: None)
    assert rec_w["net_hash_start"] == rec["saves"][3]["net_hash"]
    assert load_checkpoint(rec_w["saves"][2]["path"], declared_encoding=ENC).metadata.lineage == ("six-aaaaaaa-gen1",)
