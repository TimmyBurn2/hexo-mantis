"""The value instrument end to end: trace and read on a planted ring; compare, lagged and exams on synthetic reads."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import ModuleType
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


def _planted(path: Path, n: int = 48, wins_in: int = 2) -> R.Ring:
    """n one-visit rows under a 16-visit header, each with a distinct tail mass, three rows per game; one win in `wins_in`."""
    buf = _engine.HexgBuffer(64, _ENCODING, 16)
    board = _engine.Board.with_encoding_name(_ENCODING)
    board.apply_move(0, 0)
    for i in range(n):
        tail = 0.01 * (i + 1)
        buf.push_graph_position(list(board.get_stones()), [(1, 0, 1.0 - tail)], int(board.current_player),
                                int(board.moves_remaining), 1, True, 1.0 if i % wins_in else -1.0, True, 20, i // 3, tail)
    buf.save_to_path(str(path))
    return R.load_ring(path)


def test_the_tracer_draws_the_slots_the_sampler_draws(vi, tmp_path: Path) -> None:
    d = vi[1]
    ring = _planted(tmp_path / "planted.ring.bin")
    slot = d.trace(ring, seed=11, batches=3, threads=2, encoding=_ENCODING)
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
    assert np.array_equal(r["slot"], d.trace(ring, seed=7, batches=1, threads=2, encoding=_ENCODING))
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


def _ns(m: ModuleType, arm: list[Path], base: list[Path], sigma: float | None = None,
        effect: float | None = None) -> argparse.Namespace:
    return argparse.Namespace(arm=arm, base=base, sigma=sigma, effect=m.EFFECT if effect is None else effect)


def test_compare_refuses_other_draws_and_a_missing_seed_spread_and_reads_a_real_gap(vi, tmp_path: Path) -> None:
    m, cli = vi[0], vi[2]
    n = 4000
    z = np.where(np.arange(n) % 2, 1.0, -1.0)
    rng = np.random.default_rng(8)
    good = [_fake_read(tmp_path / f"g{i}.json", 3.0 * z + rng.normal(0, 3, n)) for i in range(3)]
    bad = [_fake_read(tmp_path / f"b{i}.json", rng.normal(0, 3, n)) for i in range(3)]
    with pytest.raises(ValueError, match="seed spread"):
        cli.compare(_ns(m, good[:1], bad[:1]))
    with pytest.raises(ValueError, match="same held-out draws"):
        cli.compare(_ns(m, good, [_fake_read(tmp_path / "s.json", rng.normal(0, 3, n), slot_shift=1)]))
    out = cli.compare(_ns(m, good, bad))
    assert out["beats"] and out["sigma_df"] == 4 and 0.0 <= out["power"] <= 1.0
    assert out["diff"] < -out["line"] < 0.0


def _synthetic_reads(tmp_path: Path, n: int = 4000) -> dict[str, list[Path]]:
    """Informative reads (4x overconfident past ply 40) and noise reads, two jittered twins of one function per side."""
    z = np.where(np.arange(n) % 2, 1.0, -1.0)
    ply = (np.arange(n) % 60).astype(np.int32)
    rng = np.random.default_rng(12)
    shared = 3.0 * z + rng.normal(0, 3, n)
    made = {"good": lambda: np.where(ply > 40, 4.0, 1.0) * (3.0 * z + rng.normal(0, 3, n)), "bad": lambda: rng.normal(0, 3, n),
            "twin_a": lambda: shared + rng.normal(0, 0.01, n), "twin_b": lambda: shared + rng.normal(0, 0.01, n)}
    return {k: [_fake_read(tmp_path / f"{k}{i}.json", f(), ply=ply) for i in range(3)] for k, f in made.items()}


def test_compare_reads_a_detection_a_worse_arm_a_tost_null_and_every_band(vi, tmp_path: Path) -> None:
    m, cli = vi[0], vi[2]
    r = _synthetic_reads(tmp_path)
    out = cli.compare(_ns(m, r["good"], r["bad"]))
    assert out["detection"] and out["beats"] and not out["worse"] and not out["tost_null"]
    assert out["ci"] == pytest.approx([out["diff"] - out["line"], out["diff"] + out["line"]])
    assert out["effect_of_record"] == pytest.approx(out["diff"] + out["line"]) and out["effect"] == m.EFFECT
    assert out["sigma_source"] == "groups" and list(out["bands"]) == list(m.PLY_BANDS)
    assert [b["n"] for b in out["bands"].values()] == [737, 2009, 1254]  # plies 0..59 cycled: both bounds inclusive
    for band in out["bands"].values():
        assert band["detection"] and band["sigma_source"] == "groups" and len(band["ci"]) == 2
    worse = cli.compare(_ns(m, r["bad"], r["good"]))
    assert worse["worse"] and not worse["detection"] and worse["effect_of_record"] is None
    null = cli.compare(_ns(m, r["twin_a"], r["twin_b"]))
    assert null["tost_null"] and not null["detection"] and not null["worse"] and null["effect_of_record"] is None
    assert not cli.compare(_ns(m, r["twin_a"], r["twin_b"], effect=0.001))["tost_null"]  # the line's floor exceeds it
    pooled = cli.compare(_ns(m, r["good"], r["bad"], sigma=0.002))
    assert pooled["sigma_source"] == "--sigma" and pooled["sigma_seed"] == 0.002
    assert all(b["sigma_source"] == "groups" for b in pooled["bands"].values())
    single = cli.compare(_ns(m, r["good"][:1], r["bad"][:1], sigma=0.002))
    assert all(b["sigma_source"] == "--sigma" and b["sigma_seed"] == 0.002 for b in single["bands"].values())
    json.dumps(out, allow_nan=False)


def test_compare_power_moves_with_the_effect_and_a_non_positive_effect_is_refused(vi, tmp_path: Path) -> None:
    m, cli = vi[0], vi[2]
    r = _synthetic_reads(tmp_path)
    near, far = cli.compare(_ns(m, r["good"], r["bad"])), cli.compare(_ns(m, r["good"], r["bad"], effect=0.05))
    assert far["power"] > near["power"] + 0.3 and far["effect"] == 0.05 and far["line"] == near["line"]
    for effect in (0.0, -0.01):
        with pytest.raises(ValueError, match="compare: --effect must be positive"):
            cli.compare(_ns(m, r["good"], r["bad"], effect=effect))


def _rows(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path.with_suffix(".rows.npz")))


def test_each_band_refits_its_own_temperatures(vi, tmp_path: Path) -> None:
    m, cli = vi[0], vi[2]
    r = _synthetic_reads(tmp_path)
    band = cli.compare(_ns(m, r["good"][:1], r["bad"][:1], sigma=0.0))["bands"]["plies_41_up"]
    arm, base = _rows(r["good"][0]), _rows(r["bad"][0])
    y, fold = arm["heldout__z"], arm["heldout__fold"]
    sel = m.band_masks(arm["heldout__ply"])["plies_41_up"]
    refit = (m.crossfit_losses(arm["heldout__u"][sel], y[sel], fold[sel])
             - m.crossfit_losses(base["heldout__u"][sel], y[sel], fold[sel])).mean()
    sliced = (m.crossfit_losses(arm["heldout__u"], y, fold) - m.crossfit_losses(base["heldout__u"], y, fold))[sel].mean()
    assert band["diff"] == pytest.approx(refit, rel=1e-9) and abs(refit - sliced) > 0.01


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


def test_lagged_refuses_reads_on_other_draws_and_a_non_positive_effect(vi, tmp_path: Path) -> None:
    m, cli = vi[0], vi[2]
    cur = _fake_read(tmp_path / "cur.json", np.random.default_rng(13).normal(0, 3, 400))
    other = _fake_read(tmp_path / "other.json", np.random.default_rng(14).normal(0, 3, 400), slot_shift=1)
    with pytest.raises(ValueError, match="lagged: the reads were not taken on the same held-out draws"):
        cli.lagged(argparse.Namespace(current=cur, lagged=other, effect=m.EFFECT))
    assert cli.main(["lagged", "--current", str(cur), "--lagged", str(other)]) == 2
    assert cli.main(["lagged", "--current", str(cur), "--lagged", str(cur), "--effect", "0"]) == 2


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
    assert out["se_raw"] == pytest.approx(np.sqrt(0.62 / 2 / 3))  # squared deviations 0.25 + 0.01 + 0.36, n - 1 = 2
    assert out["calibrated_mean"] == pytest.approx(want.mean())
    assert out["se_calibrated"] == pytest.approx(want.std(ddof=1) / np.sqrt(3))
    assert out["holds"] and out["floor"] == 0.2
    assert not cli.exams(ns(net="a", floor=0.3))["holds"]  # the raw mean 0.4 would have held
    assert cli.exams(ns())["n"] == 4 and "floor" not in cli.exams(ns())
    one = cli.exams(ns(net="b"))
    assert one["n"] == 1 and one["se_raw"] is None and one["se_calibrated"] is None
    with pytest.raises(ValueError, match="no rows"):
        cli.exams(ns(net="c"))
    nan_rows, broken = tmp_path / "nan.jsonl", tmp_path / "broken.jsonl"
    nan_rows.write_text(json.dumps({"v": float("nan")}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a finite number"):
        cli.exams(ns(rows=nan_rows))
    broken.write_text('{"v": 0.1}\n{"v": \n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 of .*broken.jsonl is not JSON"):
        cli.exams(ns(rows=broken))
    cold = tmp_path / "cold.json"
    cold.write_text(json.dumps({"heldout": {"overall": {"temperature": None}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no held-out temperature"):
        cli.exams(ns(read=cold))
    assert cli.main(["exams", "--read", str(cold), "--rows", str(rows), "--field", "v"]) == 2


def test_read_with_a_train_ring_carries_the_gap_the_run_monitor_reads(vi, mint_stamp, tmp_path: Path) -> None:
    """PRODUCER: the run monitor's gap rule reads `gap.cf_ce`, held-out minus train; a renamed key or a flipped sign reds."""
    cli = vi[2]
    ckpt = mint_stamp(tmp_path / "ck", encoding=_ENCODING)
    _planted(tmp_path / "held.ring.bin")
    _planted(tmp_path / "train.ring.bin", n=36, wins_in=4)  # another outcome rate, so the two CEs differ
    body = cli.read(argparse.Namespace(ckpt=ckpt, heldout=tmp_path / "held.ring.bin", train=tmp_path / "train.ring.bin",
                                       out=tmp_path / "read.json", seed=cli.HELDOUT_SEED, train_seed=cli.TRAIN_SEED,
                                       fold_seed=cli.FOLD_SEED, batches=1, threads=2, device="cpu"))
    held, train = body["heldout"]["overall"]["cf_ce"], body["train"]["overall"]["cf_ce"]
    assert held != pytest.approx(train, abs=1e-3) and body["gap"]["cf_ce"] == pytest.approx(held - train), (held, train)
