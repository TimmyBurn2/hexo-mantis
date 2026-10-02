# >300 justify (R8): one oracle for one tool; the fake-read writer and the synthetic reads serve compare, lagged and exams alike.
"""The value instrument: temperature, Platt, scale, ties, folds, bootstrap and line on synthetic draws; trace, read, compare, lagged and exams end to end."""
from __future__ import annotations

import argparse
import importlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mantis import _engine
from mantis.diagnostics import ring_reader as R
from mantis.encoding import lookup
from mantis.model import GnnArch, build_net
from mantis.util.loadpkg import load_tools_package

_ENCODING = "gnn_axis_r8"


@pytest.fixture(scope="module")
def vi() -> Any:
    load_tools_package("value_instrument")
    return tuple(importlib.import_module(f"value_instrument.{m}") for m in ("metrics", "draws", "cli"))


def _draws(n: int, temperature: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose calibrated probability is sigmoid(u / temperature), and outcomes drawn from it."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 2.0, n)
    return u, np.where(rng.random(n) < 1.0 / (1.0 + np.exp(-u / temperature)), 1.0, -1.0)


def test_the_fit_recovers_a_known_overconfidence(vi) -> None:
    m = vi[0]
    u, y = _draws(200_000, temperature=2.5)
    assert 1.0 / m.fit_beta(u, y) == pytest.approx(2.5, rel=0.03)


def test_no_skill_reads_ln2_and_a_balanced_constant(vi) -> None:
    m = vi[0]
    rng = np.random.default_rng(1)
    y = np.where(rng.random(40_000) < 0.5, 1.0, -1.0)
    fold = (np.arange(40_000) % 2).astype(np.int8)
    noise = m.block(rng.normal(0, 3, 40_000), y, np.zeros(40_000), fold)
    assert noise["cf_ce"] == pytest.approx(noise["constant_ce"], abs=0.003)  # temperature only: no intercept
    zero = m.block(np.zeros(40_000), y, np.zeros(40_000), fold)
    assert zero["cf_ce"] == pytest.approx(np.log(2.0), abs=1e-9) and zero["temperature"] is None


def test_the_temperature_absorbs_scale_so_an_overconfident_copy_reads_the_same(vi) -> None:
    m = vi[0]
    u, y = _draws(50_000, temperature=1.0, seed=2)
    fold = (np.arange(len(u)) % 2).astype(np.int8)
    a, b = m.block(u, y, np.zeros(len(u)), fold), m.block(3.0 * u, y, np.zeros(len(u)), fold)
    assert a["cf_ce"] == pytest.approx(b["cf_ce"], abs=1e-9)
    assert b["temperature"] == pytest.approx(3.0 * a["temperature"], rel=1e-6)
    assert b["uncal_binary_ce"] > a["uncal_binary_ce"] + 0.1


def test_auc_counts_ties_half_against_brute_force(vi) -> None:
    m = vi[0]
    rng = np.random.default_rng(3)
    s = rng.integers(0, 5, 60).astype(np.float64)
    y = np.where(rng.random(60) < 0.5, 1.0, -1.0)
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in product(s[y > 0], s[y < 0])])
    assert m.auc(s, y) == pytest.approx(brute, abs=1e-12)


def test_folds_keep_each_game_whole_split_the_universe_in_half_and_refuse_a_stranger(vi) -> None:
    m = vi[0]
    universe = np.repeat(np.arange(101), 7)
    drawn = universe[::3]
    fold = m.game_folds(drawn, 5, universe)
    assert np.array_equal(fold, m.game_folds(universe, 5, universe)[::3])
    whole = m.game_folds(universe, 5, universe)
    assert all(len(set(whole[universe == g].tolist())) == 1 for g in range(101))
    assert int(whole.reshape(101, 7)[:, 0].sum()) == 101 - 101 // 2
    with pytest.raises(ValueError):
        m.game_folds(np.array([500]), 5, universe)


def test_an_empty_band_reads_n0_and_a_one_class_band_has_no_auc(vi) -> None:
    m = vi[0]
    u, y = _draws(2_000, temperature=1.0, seed=4)
    ply = np.full(len(u), 5)
    ply[:1000] = 20
    y = np.where(ply == 20, 1.0, y)
    out = m.side(u, y, np.zeros(len(u)), (np.arange(len(u)) % 2).astype(np.int8), ply)
    assert out["plies_41_up"] == {"n": 0}
    assert out["plies_11_40"]["auc"] is None and out["plies_0_10"]["auc"] is not None
    json.dumps(out, allow_nan=False)


def test_the_game_bootstrap_reads_a_constant_difference_exactly_and_noise_as_spread(vi) -> None:
    m = vi[0]
    gid = np.repeat(np.arange(300), 20)
    flat = m.game_se(np.full(len(gid), 0.01), gid, seed=0, resamples=200)
    assert flat["diff"] == pytest.approx(0.01) and flat["se_game"] == pytest.approx(0.0, abs=1e-12)
    noisy = m.game_se(np.repeat(np.random.default_rng(6).normal(0, 0.1, 300), 20), gid, seed=0, resamples=500)
    assert noisy["se_game"] == pytest.approx(0.1 / np.sqrt(300), rel=0.2)


def test_the_line_has_its_floor_and_power_falls_as_the_spread_grows(vi) -> None:
    m = vi[0]
    tight, loose = m.line_and_power(0.001, 0.001), m.line_and_power(0.004, 0.004)
    assert tight["line"] == m.MIN_LINE and tight["power"] > 0.99
    assert loose["line"] == pytest.approx(2 * np.hypot(0.004, 0.004)) and loose["power"] < 0.5


def _planted(path: Path, n: int = 48) -> R.Ring:
    """n one-visit rows under a 16-visit header, each with a distinct tail mass, three rows per game."""
    buf = _engine.HexgBuffer(64, _ENCODING, 16)
    board = _engine.Board.with_encoding_name(_ENCODING)
    board.apply_move(0, 0)
    for i in range(n):
        tail = 0.01 * (i + 1)
        buf.push_graph_position(list(board.get_stones()), [(1, 0, 1.0 - tail)], int(board.current_player),
                                int(board.moves_remaining), 1, True, 1.0 if i % 2 else -1.0, True, 20, i // 3, tail)
    buf.save_to_path(str(path))
    return R.load_ring(path)


def test_the_tracer_draws_the_slots_the_sampler_draws(vi, tmp_path: Path) -> None:
    d = vi[1]
    ring = _planted(tmp_path / "planted.ring.bin")
    slot = d.trace(ring, seed=11, batches=3, threads=2)
    real = _engine.HexgBuffer(64, _ENCODING, 16)
    real.load_from_path(str(tmp_path / "planted.ring.bin"))
    real.seed_sampler(11)
    got = np.concatenate([np.asarray(real.sample_graph_batch(d.BATCH, augment=False, n_threads=2)[1].tail_mass)
                          for _ in range(3)])
    assert len(slot) == len(got) == 3 * d.BATCH
    np.testing.assert_allclose(ring.tail_mass[slot], got, atol=1e-7)


@pytest.fixture(scope="module")
def net_and_config() -> tuple[torch.nn.Module, dict[str, Any]]:
    """A one-layer, width-8 net in the planted ring's encoding and the two config blocks the read resolves."""
    spec = lookup(_ENCODING)
    assert spec.node_feat_dim is not None and spec.edge_feat_dim is not None
    arch = GnnArch(in_dim=int(spec.node_feat_dim), edge_dim=int(spec.edge_feat_dim), hidden=8, num_layers=1,
                   policy_hidden=8, value_hidden=8)
    cfg = {"identity": {"encoding": _ENCODING}, "train": {"microbatch_caps": {"max_edges": 4_500_000, "max_nodes": 170_000}}}
    return build_net(arch), cfg


def test_read_traces_every_draw_on_a_ring_below_its_visit_cap(vi, net_and_config, tmp_path: Path) -> None:
    d = vi[1]
    model, cfg = net_and_config
    path = tmp_path / "planted.ring.bin"
    ring = _planted(path)
    r = d.read(model, cfg, path, seed=7, batches=1, device=torch.device("cpu"), threads=2, dump_dir=tmp_path)
    assert np.array_equal(r["slot"], d.trace(ring, seed=7, batches=1, threads=2))
    assert np.array_equal(r["z"], ring.outcome[r["slot"]]) and np.isfinite(r["u"]).all()
    assert not model.training


def test_read_refuses_a_draw_its_tracer_misplaces(vi, net_and_config, tmp_path: Path, monkeypatch) -> None:
    d = vi[1]
    model, cfg = net_and_config
    path = tmp_path / "planted.ring.bin"
    _planted(path)
    true_trace = d.trace
    monkeypatch.setattr(d, "trace", lambda ring, **kw: np.roll(true_trace(ring, **kw), 1))
    with pytest.raises(d.TraceMismatchError):
        d.read(model, cfg, path, seed=7, batches=1, device=torch.device("cpu"), threads=2, dump_dir=tmp_path)


def _fake_read(path: Path, u: np.ndarray, *, slot_shift: int = 0, ring_sha: str = "x", ply: np.ndarray | None = None) -> Path:
    n = len(u)
    gid = np.repeat(np.arange(n // 4), 4)
    rows = {"heldout__u": u, "heldout__raw_ce": np.zeros(n), "heldout__z": np.where(np.arange(n) % 2, 1.0, -1.0),
            "heldout__valid": np.ones(n, bool), "heldout__game_id": gid,
            "heldout__ply": np.zeros(n, np.int32) if ply is None else ply,
            "heldout__slot": np.arange(n) + slot_shift, "heldout__fold": (gid % 2).astype(np.int8)}
    np.savez(path.with_suffix(".rows.npz"), **rows)
    path.write_text(json.dumps({"heldout": {"ring_sha256": ring_sha}}), encoding="utf-8")
    return path


def test_compare_refuses_other_draws_and_a_missing_seed_spread_and_reads_a_real_gap(vi, tmp_path: Path) -> None:
    cli = vi[2]
    n = 4000
    z = np.where(np.arange(n) % 2, 1.0, -1.0)
    rng = np.random.default_rng(8)
    good = [_fake_read(tmp_path / f"g{i}.json", 3.0 * z + rng.normal(0, 3, n)) for i in range(3)]
    bad = [_fake_read(tmp_path / f"b{i}.json", rng.normal(0, 3, n)) for i in range(3)]
    ns = lambda arm, base, sigma=None: argparse.Namespace(arm=arm, base=base, sigma=sigma, effect=vi[0].EFFECT)  # noqa: E731
    with pytest.raises(ValueError, match="seed spread"):
        cli.compare(ns(good[:1], bad[:1]))
    with pytest.raises(ValueError, match="same held-out draws"):
        cli.compare(ns(good, [_fake_read(tmp_path / "s.json", rng.normal(0, 3, n), slot_shift=1)]))
    out = cli.compare(ns(good, bad))
    assert out["beats"] and out["sigma_df"] == 4 and 0.0 <= out["power"] <= 1.0
    assert out["diff"] < -out["line"] < 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def test_platt_recovers_a_known_slope_and_intercept_and_falls_back_to_the_base_rate(vi) -> None:
    m = vi[0]
    rng = np.random.default_rng(10)
    u = rng.normal(0.0, 2.0, 200_000)
    y = np.where(rng.random(len(u)) < _sigmoid(0.6 * u + 0.4), 1.0, -1.0)
    a, c = m.fit_platt(u, y)
    assert a == pytest.approx(0.6, abs=0.02) and c == pytest.approx(0.4, abs=0.02)
    assert m.fit_platt(u, np.ones(len(u))) == (0.0, pytest.approx(np.log((1 - 1e-6) / 1e-6)))
    assert m.fit_platt(np.empty(0), np.empty(0)) == (0.0, 0.0)


def test_the_crossfit_platt_ce_beats_the_constant_and_absorbs_the_intercept_a_temperature_cannot(vi) -> None:
    m = vi[0]
    rng = np.random.default_rng(11)
    u = rng.normal(0.0, 2.0, 40_000)
    y = np.where(rng.random(len(u)) < _sigmoid(0.5 * u + 1.0), 1.0, -1.0)
    fold = (np.arange(len(u)) % 2).astype(np.int8)
    out = m.block(u, y, np.zeros(len(u)), fold)
    assert out["platt_ce"] < out["constant_ce"] - 0.05 and out["platt_ce"] < out["cf_ce"] - 0.01
    assert out["platt_ce_f1"] == pytest.approx(float(m.crossfit_platt_losses(u, y, fold)[fold == 1].mean()))
    assert "platt_ce" in m.side(u, y, np.zeros(len(u)), fold, np.full(len(u), 5))["plies_0_10"]


def test_calibrated_is_odd_the_identity_at_one_shrinks_when_overconfident_and_zero_without_skill(vi) -> None:
    m = vi[0]
    v = np.linspace(-0.99, 0.99, 199)
    np.testing.assert_allclose(m.calibrated(v, 1.0), v, rtol=0, atol=1e-12)
    hot = m.calibrated(v, 2.5)
    assert np.all(np.abs(hot) < np.abs(v) + 1e-18) and np.all(np.abs(hot[np.abs(v) > 0.01]) < np.abs(v[np.abs(v) > 0.01]))
    np.testing.assert_allclose(m.calibrated(-v, 2.5), -hot, rtol=0, atol=1e-15)
    assert np.isfinite(m.calibrated(np.array([-1.0, 1.0]), 2.0)).all()
    zero = m.calibrated(v, None)
    assert zero.shape == v.shape and not zero.any()


def _synthetic_reads(tmp_path: Path, n: int = 4000) -> dict[str, list[Path]]:
    """Informative and noise reads, two jittered twins of one function per side, every ply band populated."""
    z = np.where(np.arange(n) % 2, 1.0, -1.0)
    ply = (np.arange(n) % 60).astype(np.int32)
    rng = np.random.default_rng(12)
    shared = 3.0 * z + rng.normal(0, 3, n)
    made = {"good": lambda: 3.0 * z + rng.normal(0, 3, n), "bad": lambda: rng.normal(0, 3, n),
            "twin_a": lambda: shared + rng.normal(0, 0.01, n), "twin_b": lambda: shared + rng.normal(0, 0.01, n)}
    return {k: [_fake_read(tmp_path / f"{k}{i}.json", f(), ply=ply) for i in range(3)] for k, f in made.items()}


def _ns(arm: list[Path], base: list[Path], sigma: float | None = None, effect: float = 0.01) -> argparse.Namespace:
    return argparse.Namespace(arm=arm, base=base, sigma=sigma, effect=effect)


def test_compare_reads_a_detection_a_worse_arm_a_tost_null_and_every_band(vi, tmp_path: Path) -> None:
    m, cli = vi[0], vi[2]
    r = _synthetic_reads(tmp_path)
    out = cli.compare(_ns(r["good"], r["bad"]))
    assert out["detection"] and out["beats"] and not out["worse"] and not out["tost_null"]
    assert out["ci"] == pytest.approx([out["diff"] - out["line"], out["diff"] + out["line"]])
    assert out["effect_of_record"] == pytest.approx(out["diff"] + out["line"]) and out["effect"] == m.EFFECT
    assert out["sigma_source"] == "groups" and set(out["bands"]) == set(m.PLY_BANDS)
    for band in out["bands"].values():
        assert band["n"] > 0 and band["detection"] and band["sigma_source"] == "groups" and len(band["ci"]) == 2
    worse = cli.compare(_ns(r["bad"], r["good"]))
    assert worse["worse"] and not worse["detection"] and worse["effect_of_record"] is None
    null = cli.compare(_ns(r["twin_a"], r["twin_b"]))
    assert null["tost_null"] and not null["detection"] and not null["worse"] and null["effect_of_record"] is None
    assert not cli.compare(_ns(r["twin_a"], r["twin_b"], effect=0.001))["tost_null"]  # the line's floor exceeds it
    pooled = cli.compare(_ns(r["good"], r["bad"], sigma=0.002))
    assert pooled["sigma_source"] == "--sigma" and pooled["sigma_seed"] == 0.002
    assert all(b["sigma_source"] == "groups" for b in pooled["bands"].values())
    single = cli.compare(_ns(r["good"][:1], r["bad"][:1], sigma=0.002))
    assert all(b["sigma_source"] == "--sigma" and b["sigma_seed"] == 0.002 for b in single["bands"].values())
    json.dumps(out, allow_nan=False)


def test_lagged_is_compare_at_zero_sigma_with_its_source_named(vi, tmp_path: Path, capsys) -> None:
    m, cli = vi[0], vi[2]
    r = _synthetic_reads(tmp_path)
    cur, lag = str(r["good"][0]), str(r["bad"][0])
    assert cli.main(["lagged", "--current", cur, "--lagged", lag]) == 0
    got = json.loads(capsys.readouterr().out)
    assert cli.main(["compare", "--arm", cur, "--base", lag, "--sigma", "0"]) == 0
    want = json.loads(capsys.readouterr().out)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("sigma_source", "bands")}  # noqa: E731
    assert strip(got) == strip(want) and all(strip(got["bands"][b]) == strip(want["bands"][b]) for b in m.PLY_BANDS)
    assert {got["sigma_source"], *(b["sigma_source"] for b in got["bands"].values())} == {"none (one run): game noise only"}
    assert got["sd_seed"] == 0.0 and got["line"] == pytest.approx(max(m.MIN_LINE, 2.0 * got["se_game"]))


def test_lagged_refuses_reads_on_other_draws(vi, tmp_path: Path) -> None:
    m, cli = vi[0], vi[2]
    cur = _fake_read(tmp_path / "cur.json", np.random.default_rng(13).normal(0, 3, 400))
    other = _fake_read(tmp_path / "other.json", np.random.default_rng(14).normal(0, 3, 400), slot_shift=1)
    with pytest.raises(ValueError, match="lagged: the reads were not taken on the same held-out draws"):
        cli.lagged(argparse.Namespace(current=cur, lagged=other, effect=m.EFFECT))
    assert cli.main(["lagged", "--current", str(cur), "--lagged", str(other)]) == 2


def test_exams_calibrates_at_the_read_temperature_reads_the_floor_and_refuses(vi, tmp_path: Path) -> None:
    cli = vi[2]
    read, rows = tmp_path / "r.json", tmp_path / "rows.jsonl"
    read.write_text(json.dumps({"heldout": {"overall": {"temperature": 2.0}}}), encoding="utf-8")
    vals = [0.9, 0.5, -0.2, 0.7]
    rows.write_text("\n".join(json.dumps({"net": "b" if i == 3 else "a", "v": x}) for i, x in enumerate(vals)) + "\n\n",
                    encoding="utf-8")
    ns = lambda **kw: argparse.Namespace(**{"read": read, "rows": rows, "field": "v", "net": None, "floor": None, **kw})  # noqa: E731
    want = np.tanh(np.arctanh(np.array(vals[:3])) / 2.0)
    out = cli.exams(ns(net="a", floor=0.2))
    assert out["n"] == 3 and out["temperature"] == 2.0 and out["raw_mean"] == pytest.approx(0.4)
    assert out["calibrated_mean"] == pytest.approx(want.mean()) and out["se_calibrated"] == pytest.approx(want.std() / np.sqrt(3))
    assert out["holds"] and out["floor"] == 0.2
    assert not cli.exams(ns(net="a", floor=0.3))["holds"]  # the raw mean 0.4 would have held
    assert cli.exams(ns())["n"] == 4 and "floor" not in cli.exams(ns())
    with pytest.raises(ValueError, match="no rows"):
        cli.exams(ns(net="c"))
    nan_rows = tmp_path / "nan.jsonl"
    nan_rows.write_text(json.dumps({"v": float("nan")}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a finite number"):
        cli.exams(ns(rows=nan_rows))
    cold = tmp_path / "cold.json"
    cold.write_text(json.dumps({"heldout": {"overall": {"temperature": None}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no held-out temperature"):
        cli.exams(ns(read=cold))
    assert cli.main(["exams", "--read", str(cold), "--rows", str(rows), "--field", "v"]) == 2
