"""The value instrument's arithmetic on synthetic draws: temperature, Platt, cross-fitting, calibration, scale, ties, folds, bands, bootstrap and line."""
from __future__ import annotations

import importlib
import json
from itertools import product
from types import ModuleType

import numpy as np
import pytest

from mantis.util.loadpkg import load_tools_package


@pytest.fixture(scope="module")
def m() -> ModuleType:
    load_tools_package("value_instrument")
    return importlib.import_module("value_instrument.metrics")


def _draws(n: int, temperature: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose calibrated probability is sigmoid(u / temperature), and outcomes drawn from it."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 2.0, n)
    return u, np.where(rng.random(n) < 1.0 / (1.0 + np.exp(-u / temperature)), 1.0, -1.0)


def test_the_fit_recovers_a_known_overconfidence(m) -> None:
    u, y = _draws(200_000, temperature=2.5)
    assert 1.0 / m.fit_beta(u, y) == pytest.approx(2.5, rel=0.03)


def test_no_skill_reads_ln2_and_a_balanced_constant(m) -> None:
    rng = np.random.default_rng(1)
    y = np.where(rng.random(40_000) < 0.5, 1.0, -1.0)
    fold = (np.arange(40_000) % 2).astype(np.int8)
    noise = m.block(rng.normal(0, 3, 40_000), y, np.zeros(40_000), fold)
    assert noise["cf_ce"] == pytest.approx(noise["constant_ce"], abs=0.003)  # temperature only: no intercept
    zero = m.block(np.zeros(40_000), y, np.zeros(40_000), fold)
    assert zero["cf_ce"] == pytest.approx(np.log(2.0), abs=1e-9) and zero["temperature"] is None


def test_the_temperature_absorbs_scale_so_an_overconfident_copy_reads_the_same(m) -> None:
    u, y = _draws(50_000, temperature=1.0, seed=2)
    fold = (np.arange(len(u)) % 2).astype(np.int8)
    a, b = m.block(u, y, np.zeros(len(u)), fold), m.block(3.0 * u, y, np.zeros(len(u)), fold)
    assert a["cf_ce"] == pytest.approx(b["cf_ce"], abs=1e-9)
    assert b["temperature"] == pytest.approx(3.0 * a["temperature"], rel=1e-6)
    assert b["uncal_binary_ce"] > a["uncal_binary_ce"] + 0.1


def test_auc_counts_ties_half_against_brute_force(m) -> None:
    rng = np.random.default_rng(3)
    s = rng.integers(0, 5, 60).astype(np.float64)
    y = np.where(rng.random(60) < 0.5, 1.0, -1.0)
    brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in product(s[y > 0], s[y < 0])])
    assert m.auc(s, y) == pytest.approx(brute, abs=1e-12)


def test_folds_keep_each_game_whole_split_the_universe_in_half_and_refuse_a_stranger(m) -> None:
    universe = np.repeat(np.arange(101), 7)
    drawn = universe[::3]
    fold = m.game_folds(drawn, 5, universe)
    assert np.array_equal(fold, m.game_folds(universe, 5, universe)[::3])
    whole = m.game_folds(universe, 5, universe)
    assert all(len(set(whole[universe == g].tolist())) == 1 for g in range(101))
    assert int(whole.reshape(101, 7)[:, 0].sum()) == 101 - 101 // 2
    with pytest.raises(ValueError):
        m.game_folds(np.array([500]), 5, universe)


def test_an_empty_band_reads_n0_and_a_one_class_band_has_no_auc(m) -> None:
    u, y = _draws(2_000, temperature=1.0, seed=4)
    ply = np.full(len(u), 5)
    ply[:1000] = 20
    y = np.where(ply == 20, 1.0, y)
    out = m.side(u, y, np.zeros(len(u)), (np.arange(len(u)) % 2).astype(np.int8), ply)
    assert out["plies_41_up"] == {"n": 0}
    assert out["plies_11_40"]["auc"] is None and out["plies_0_10"]["auc"] is not None
    json.dumps(out, allow_nan=False)


def test_the_game_bootstrap_reads_a_constant_difference_exactly_and_noise_as_spread(m) -> None:
    gid = np.repeat(np.arange(300), 20)
    flat = m.game_se(np.full(len(gid), 0.01), gid, seed=0, resamples=200)
    assert flat["diff"] == pytest.approx(0.01) and flat["se_game"] == pytest.approx(0.0, abs=1e-12)
    noisy = m.game_se(np.repeat(np.random.default_rng(6).normal(0, 0.1, 300), 20), gid, seed=0, resamples=500)
    assert noisy["se_game"] == pytest.approx(0.1 / np.sqrt(300), rel=0.2)


def test_the_line_has_its_floor_and_power_falls_as_the_spread_grows(m) -> None:
    tight, loose = m.line_and_power(0.001, 0.001), m.line_and_power(0.004, 0.004)
    assert tight["line"] == m.MIN_LINE and tight["power"] > 0.99
    assert loose["line"] == pytest.approx(2 * np.hypot(0.004, 0.004)) and loose["power"] < 0.5
    assert m.line_and_power(0.004, 0.004, effect=0.05)["power"] > 0.99


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def test_platt_recovers_a_known_slope_and_intercept_and_falls_back_to_the_base_rate(m) -> None:
    rng = np.random.default_rng(10)
    u = rng.normal(0.0, 2.0, 200_000)
    y = np.where(rng.random(len(u)) < _sigmoid(0.6 * u + 0.4), 1.0, -1.0)
    a, c = m.fit_platt(u, y)
    assert a == pytest.approx(0.6, abs=0.02) and c == pytest.approx(0.4, abs=0.02)
    assert m.fit_platt(u, np.ones(len(u))) == (0.0, pytest.approx(np.log((1 - 1e-6) / 1e-6)))
    assert m.fit_platt(np.empty(0), np.empty(0)) == (0.0, 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        m.fit_platt(np.array([0.5, np.inf]), np.array([1.0, -1.0]))


def test_the_crossfit_platt_ce_beats_the_constant_and_absorbs_the_intercept_a_temperature_cannot(m) -> None:
    rng = np.random.default_rng(11)
    u = rng.normal(0.0, 2.0, 40_000)
    y = np.where(rng.random(len(u)) < _sigmoid(0.5 * u + 1.0), 1.0, -1.0)
    fold = (np.arange(len(u)) % 2).astype(np.int8)
    out = m.block(u, y, np.zeros(len(u)), fold)
    assert out["platt_ce"] < out["constant_ce"] - 0.05 and out["platt_ce"] < out["cf_ce"] - 0.01
    assert out["platt_ce_f1"] == pytest.approx(float(m.crossfit_platt_losses(u, y, fold)[fold == 1].mean()))
    assert "platt_ce" in m.side(u, y, np.zeros(len(u)), fold, np.full(len(u), 5))["plies_0_10"]


def test_calibrated_is_odd_the_identity_at_one_shrinks_when_overconfident_and_zero_without_skill(m) -> None:
    v = np.linspace(-0.99, 0.99, 199)
    np.testing.assert_allclose(m.calibrated(v, 1.0), v, rtol=0, atol=1e-12)
    hot = m.calibrated(v, 2.5)
    assert np.all(np.abs(hot) < np.abs(v) + 1e-18) and np.all(np.abs(hot[np.abs(v) > 0.01]) < np.abs(v[np.abs(v) > 0.01]))
    np.testing.assert_allclose(m.calibrated(-v, 2.5), -hot, rtol=0, atol=1e-15)
    assert np.isfinite(m.calibrated(np.array([-1.0, 1.0]), 2.0)).all()
    zero = m.calibrated(v, None)
    assert zero.shape == v.shape and not zero.any()


def test_each_fold_is_scored_by_the_other_folds_fit_so_a_flipped_slope_reads_worse_than_ln2(m) -> None:
    rng = np.random.default_rng(15)
    u = rng.normal(0.0, 2.0, 40_000)
    fold = (np.arange(len(u)) % 2).astype(np.int8)
    y = np.where(rng.random(len(u)) < _sigmoid(np.where(fold == 0, u, -u)), 1.0, -1.0)
    f0, f1 = fold == 0, fold == 1
    (a0, c0), (a1, c1) = m.fit_platt(u[f0], y[f0]), m.fit_platt(u[f1], y[f1])
    platt = m.crossfit_platt_losses(u, y, fold)
    np.testing.assert_allclose(platt[f1], m.softplus(-y[f1] * (a0 * u[f1] + c0)), rtol=1e-12)
    np.testing.assert_allclose(platt[f0], m.softplus(-y[f0] * (a1 * u[f0] + c1)), rtol=1e-12)
    b0, b1 = m.fit_beta(u[f0], y[f0]), m.fit_beta(u[f1], y[f1])
    temp = m.crossfit_losses(u, y, fold)
    np.testing.assert_allclose(temp[f1], m.softplus(-b0 * y[f1] * u[f1]), rtol=1e-12)
    np.testing.assert_allclose(temp[f0], m.softplus(-b1 * y[f0] * u[f0]), rtol=1e-12)
    assert platt.mean() > np.log(2.0) + 0.1 and temp.mean() > np.log(2.0) + 0.05


def test_the_band_masks_hold_both_bounds(m) -> None:
    ply = np.array([0, 10, 11, 40, 41, 10**6])
    masks = m.band_masks(ply)
    assert list(masks) == list(m.PLY_BANDS)
    assert {b: ply[s].tolist() for b, s in masks.items()} == {
        "plies_0_10": [0, 10], "plies_11_40": [11, 40], "plies_41_up": [41, 10**6]}
