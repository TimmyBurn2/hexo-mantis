"""Suite G remainder — G-01 … G-15 (old-suite ports) + G-17 (the DV-10 scipy pin).

IMPL-written (non-⊕). `test_instrumentation.py` is the ⊕ file and carries G-16 ONLY (the
pure-function GOLDENS over the #C3c battery). The dispatcher's slice DAG gated S1 on
A/B/G-16/D-15, S2 on E/F/I and S3 on C/D/H, so G-01 … G-15 and G-17 — required GREEN by
PREREG §3 Suite G — were never assigned to a slice. This file closes that coverage gap.

Source of truth: the frozen old suite `hexo_rl/tests/selfplay/test_instrumentation.py`
(ported near-verbatim — assertions faithful, not re-invented) and the frozen module
`hexo_rl/hexo_rl/selfplay/instrumentation.py`. Every G-01 … G-15 vector is the old test's
own vector; the ported assertion is the old assertion.

Coverage per PREREG §3 Suite G (old-verbatim):
  - windows 100 (recent_move_histories ring) / 50 (per-worker draws + stride5) /
    200 (model-version range archive);
  - the P90 rule (`sorted[max(0, int(n*0.9)-1)]`);
  - the ply→player rule (compound-turn split);
  - cap-at-6 (longest_line capped at `_WIN_LENGTH`);
  - threshold connectivity (`n_components` under `cluster_threshold`);
  - the disabled / empty arms.

G-17 is the DV-10 pin: `spearman_rho_range_vs_draw` degrades to None when scipy cannot be
imported. scipy is deliberately NOT a declared dependency (pan-WP optional-deps item), so
the field is None until scipy is declared. The pin blocks the import deterministically so
it holds regardless of whether the running interpreter happens to have scipy installed.
"""
from __future__ import annotations

import builtins
import importlib.util
import threading

from mantis.selfplay.instrumentation import (
    PoolInstrumentation,
    _compute_colony_extension,
    _compute_longest_line,
    _compute_n_components,
    _compute_stride5_metrics,
)


def _lock() -> threading.Lock:
    return threading.Lock()


def _make_instr(log_metrics: bool = True) -> PoolInstrumentation:
    return PoolInstrumentation(log_investigation_metrics=log_metrics)


def _game_complete(instr, lock, *, winner_code=1, move_history=None, worker_id=0,
                   terminal_reason=0, mv_min=0, mv_max=0, mv_distinct=1, stride5_run=0,
                   cluster_threshold=5):
    return instr.on_game_complete(
        lock, winner_code, move_history or [], worker_id,
        terminal_reason, mv_min, mv_max, mv_distinct, stride5_run,
        cluster_threshold,
    )


# ── G-01 — draws counter (per-worker rolling window 50) ──────────────────────────────
def test_g01_draws_counter_increments_on_draw_terminal() -> None:
    """G-01 — PASS iff a draw (winner_code 0) then a win yields a per-worker draw rate of
    0.5. FAIL = the draw/decisive classification or the per-worker window is broken."""
    instr = _make_instr()
    lk = _lock()
    _game_complete(instr, lk, winner_code=0, worker_id=0)  # draw
    _game_complete(instr, lk, winner_code=1, worker_id=0)  # win
    # WPMINT Phase DS (R92): the estimator reports RAW POOLED COUNTS and no longer takes an
    # inclusion bar (that bar died with the filtered-mean statistic). G-01's subject —
    # "one draw then one win is a draw rate of 0.5" — is unchanged; it is read off the
    # counts instead of off a per-worker map.
    draws, completed = instr.pooled_draw_counts(lk)
    assert (draws, completed) == (1, 2)
    assert abs(draws / completed - 0.5) < 1e-9


# ── G-02 — terminal-reason histogram accumulates ─────────────────────────────────────
def test_g02_terminal_reasons_histogram_accumulates() -> None:
    """G-02 — PASS iff cumulative terminal-reason counts accumulate under the named keys.
    FAIL = a reason code lands in the wrong bucket or the histogram resets."""
    instr = _make_instr()
    lk = _lock()
    _game_complete(instr, lk, terminal_reason=0)  # six
    _game_complete(instr, lk, terminal_reason=0)  # six
    _game_complete(instr, lk, terminal_reason=2)  # cap
    counts = instr.terminal_reason_counts(lk)
    assert counts["six_in_a_row"] == 2
    assert counts["ply_cap"] == 1
    assert counts["colony"] == 0


# ── G-03 — model-version tracking (window 200 archive) ───────────────────────────────
def test_g03_model_version_tracking_threadsafe() -> None:
    """G-03 — PASS iff a single game with mv range [10, 20] yields n == 1 and median_range
    == 10. FAIL = the per-game version-range archive drops or miscomputes the range."""
    instr = _make_instr()
    lk = _lock()
    _game_complete(instr, lk, mv_min=10, mv_max=20, mv_distinct=3, winner_code=1)
    summary = instr.model_version_summary(lk)
    assert summary["n"] == 1
    assert summary["median_range"] == 10


def test_g03b_model_version_archive_caps_at_200() -> None:
    """G-03 (window-200 arm) — the model-version range archive is a maxlen-200 ring: 250
    completed games leave n == 200. Old truth: `_mv_range_history = deque(maxlen=200)`."""
    instr = _make_instr()
    lk = _lock()
    for _ in range(250):
        _game_complete(instr, lk, mv_min=0, mv_max=5, mv_distinct=2, winner_code=1)
    summary = instr.model_version_summary(lk)
    assert summary["n"] == 200


# ── G-04 — recent-move-histories ring buffer (window 100) ────────────────────────────
def test_g04_move_histories_ring_buffer_capacity() -> None:
    """G-04 — PASS iff 110 completed games leave exactly 100 histories (maxlen=100 ring).
    FAIL = the window grows unbounded or drops early."""
    instr = _make_instr()
    lk = _lock()
    moves = [(0, 0), (1, 0)]
    for _ in range(110):
        _game_complete(instr, lk, move_history=moves)
    hist = instr.recent_move_histories(lk)
    assert len(hist) == 100  # maxlen=100 ring buffer


# ── G-05 — stride5 P90 passive calculation (window 50) ───────────────────────────────
def test_g05_stride5_p90_passive_calculation() -> None:
    """G-05 — PASS iff the rolling P90 of stride5_run over [0..9] is 8. The P90 index rule
    is `max(0, int(n*0.9)-1)` = `max(0, int(10*0.9)-1)` = 8 → value 8. FAIL = the
    percentile index rule drifted."""
    instr = _make_instr()
    lk = _lock()
    p90 = 0
    for run in range(10):
        _, _, _, p90, _, _, _ = _game_complete(instr, lk, stride5_run=run)
    # P90 of [0..9] (10 values): index = max(0, int(10*0.9)-1) = 8 → value 8
    assert p90 == 8


def test_g05b_current_stride5_p90_getter_matches_window() -> None:
    """G-05 (getter arm) — `current_stride5_p90` uses the same window+percentile rule and
    returns 0 before any game. Old truth: getter mirrors `on_game_complete`'s P90."""
    instr = _make_instr()
    lk = _lock()
    assert instr.current_stride5_p90(lk) == 0  # no games yet
    for run in range(10):
        _game_complete(instr, lk, stride5_run=run)
    assert instr.current_stride5_p90(lk) == 8


# ── G-06 — colony-extension pure function (ply→player rule) ──────────────────────────
def test_g06_colony_extension_pure_function() -> None:
    """G-06 — PASS iff two stones far apart (P1 at (0,0), P2 at (50,50)) both count as
    colony extension: total == 2, count == 2. FAIL = the ply→player split or the hex
    distance threshold drifted."""
    # P1 at (0,0); P2 far away at (50,50)
    moves = [(0, 0), (50, 50)]
    count, total = _compute_colony_extension(moves)
    assert total == 2
    assert count == 2  # both stones far from any opponent stone


# ── B3a structural metrics: fixtures cross-checked against the old analyzer ───────────
# Ply->player rule: ply0=P1, [1,2]=P2, [3,4]=P1, [5,6]=P2, [7,8]=P1, [9,10]=P2,
# [11]=P1.  Six P1 plies (0,3,4,7,8,11) -> a 6-in-a-row on the r=0 q-axis;
# P2 filler far away so it never interferes.
_SIX_IN_A_ROW_P1 = [
    (0, 0),            # ply0  P1
    (50, 50), (51, 50),  # ply1,2 P2
    (1, 0), (2, 0),    # ply3,4 P1
    (52, 50), (53, 50),  # ply5,6 P2
    (3, 0), (4, 0),    # ply7,8 P1
    (54, 50), (55, 50),  # ply9,10 P2
    (5, 0),            # ply11 P1  -> P1 = (0..5, 0)
]

# P1 = two disjoint pairs: {(0,0),(1,0)} and {(20,0),(21,0)}; gap 19 > thresh 5.
_TWO_CLUSTER_P1 = [
    (0, 0),            # ply0  P1   clusterA
    (50, 50), (51, 50),  # ply1,2 P2
    (1, 0), (20, 0),   # ply3,4 P1  clusterA, clusterB
    (52, 50), (53, 50),  # ply5,6 P2
    (21, 0),           # ply7  P1   clusterB
]


# ── G-07 — longest line: six-in-a-row ────────────────────────────────────────────────
def test_g07_longest_line_six_in_a_row() -> None:
    """G-07 — PASS iff a 6-in-a-row for P1 gives longest_line == 6 and fraction == 1.0
    (6 stones, all on one line). FAIL = the straight-run walk or the fraction drifted."""
    ll, frac = _compute_longest_line(_SIX_IN_A_ROW_P1, 5, 1)
    assert ll == 6
    assert abs(frac - 1.0) < 1e-9


# ── G-08 — longest line CAPPED at 6 (the cap-at-6 rule) ──────────────────────────────
def test_g08_longest_line_capped_at_six() -> None:
    """G-08 — PASS iff 7 collinear P1 stones report longest_line == 6 (capped at
    `_WIN_LENGTH`), fraction == 6/7. FAIL = the cap-at-6 rule is gone — the engine never
    extends a line past a 6-win, so an uncapped value is incomparable to the Rust emit."""
    seven_collinear = [
        (0, 0),            # ply0  P1
        (50, 50), (51, 50),  # ply1,2 P2
        (1, 0), (2, 0),    # ply3,4 P1
        (52, 50), (53, 50),  # ply5,6 P2
        (3, 0), (4, 0),    # ply7,8 P1
        (54, 50), (55, 50),  # ply9,10 P2
        (5, 0), (6, 0),    # ply11,12 P1 -> P1 = (0..6, 0), raw run 7
    ]
    ll, frac = _compute_longest_line(seven_collinear, 5, 1)
    assert ll == 6  # 7 raw -> capped at WIN_LENGTH
    assert abs(frac - 6.0 / 7.0) < 1e-9


# ── G-09 — n_components: two disjoint clusters ───────────────────────────────────────
def test_g09_n_components_two_disjoint_clusters() -> None:
    """G-09 — PASS iff two P1 pairs separated by 19 > threshold(5) give 2 components.
    FAIL = the flood-fill connectivity or the per-player winner split drifted."""
    nc = _compute_n_components(_TWO_CLUSTER_P1, 5, 1)
    assert nc == 2


# ── G-10 — n_components: cluster_threshold honored (threshold connectivity) ───────────
def test_g10_n_components_cluster_threshold_honored() -> None:
    """G-10 — PASS iff the same two pairs (gap 19) give 2 components at threshold 5 but 1
    at threshold 19. FAIL = the connectivity edge no longer reads `cluster_threshold`."""
    assert _compute_n_components(_TWO_CLUSTER_P1, 5, 1) == 2
    assert _compute_n_components(_TWO_CLUSTER_P1, 19, 1) == 1


# ── G-11 — structural metrics: empty arm ─────────────────────────────────────────────
def test_g11_structural_metrics_empty() -> None:
    """G-11 — PASS iff empty move histories give (0, 0.0) longest line and 0 components.
    FAIL = an empty game raises or fabricates structure."""
    assert _compute_longest_line([], 5, 1) == (0, 0.0)
    assert _compute_n_components([], 5, 1) == 0


# ── G-12 — structural metrics via on_game_complete (end-to-end populated) ────────────
def test_g12_structural_metrics_via_on_game_complete() -> None:
    """G-12 — PASS iff the pool path returns the 7-tuple with structural fields populated
    for a 6-in-a-row winner: longest_line == 6, fraction == 1.0, n_components == 1 (all
    six P1 stones one connected line). FAIL = the on_game_complete wiring drops a field."""
    instr = _make_instr(log_metrics=True)
    lk = _lock()
    out = _game_complete(instr, lk, winner_code=1, move_history=_SIX_IN_A_ROW_P1,
                         cluster_threshold=5)
    (_ext_c, _ext_t, _ext_f, _p90, longest_line, ll_frac, n_comp) = out
    assert longest_line == 6
    assert abs(ll_frac - 1.0) < 1e-9
    assert n_comp == 1  # all six P1 stones one connected line


# ── G-13 — structural metrics OFF when log_investigation_metrics disabled ─────────────
def test_g13_structural_metrics_off_when_log_disabled() -> None:
    """G-13 — PASS iff `log_investigation_metrics=False` zeroes the structural fields
    (gate respected). FAIL = the investigation flag no longer gates the expensive emit."""
    instr = _make_instr(log_metrics=False)
    lk = _lock()
    out = _game_complete(instr, lk, winner_code=1, move_history=_SIX_IN_A_ROW_P1,
                         cluster_threshold=5)
    (_ext_c, _ext_t, _ext_f, _p90, longest_line, ll_frac, n_comp) = out
    assert (longest_line, ll_frac, n_comp) == (0, 0.0, 0)


# ── G-14 — stride5 metrics: empty history ────────────────────────────────────────────
def test_g14_stride5_metrics_empty_history() -> None:
    """G-14 — PASS iff an empty history gives (0, 0). FAIL = the detector raises or
    fabricates a run on no stones."""
    assert _compute_stride5_metrics([]) == (0, 0)


# ── G-15 — stride5 metrics: chain along an r-row + adjacent trio ──────────────────────
def test_g15_stride5_metrics_chain_along_r_row() -> None:
    """G-15 — PASS iff four stones on r=0 at q ∈ {3, 8, 13, 18} form a stride-5 chain of
    length 4, with row_max_density 4. FAIL = the stride-5 chain walk drifted."""
    # Four stones on r=0 at q ∈ {3, 8, 13, 18} → stride-5 chain of length 4.
    moves = [(3, 0), (8, 0), (13, 0), (18, 0)]
    stride5_max, row_max = _compute_stride5_metrics(moves)
    assert stride5_max == 4
    assert row_max == 4


def test_g15b_stride5_metrics_no_stride5_pattern() -> None:
    """G-15 (adjacent-trio arm) — three adjacent stones: row_max counts them (3), stride5
    reads each as a degenerate length-1 chain (no stride-5 follow-on). Old truth verbatim."""
    # Adjacent stones — row_max counts them; stride5_max reads each stone as
    # a degenerate "chain of length 1" (no stride-5 follow-on in row).
    moves = [(0, 0), (1, 0), (2, 0)]
    stride5_max, row_max = _compute_stride5_metrics(moves)
    assert stride5_max == 1
    assert row_max == 3


# ── G-17 — scipy-absent degradation (the DV-10 pin) ──────────────────────────────────
def _ten_varied_games(instr: PoolInstrumentation, lock: threading.Lock) -> None:
    """Fill the model-version archive with 10 games of varied range + mixed outcome so the
    `n >= 10` branch that computes the spearman correlation is genuinely reached (a
    constant series would make even a present scipy return NaN, muddying the pin)."""
    for i in range(10):
        _game_complete(instr, lock, mv_min=0, mv_max=i + 1, mv_distinct=i + 1,
                       winner_code=(0 if i % 2 == 0 else 1))


def test_g17_scipy_absent_degrades_to_none(monkeypatch) -> None:
    """G-17 (DV-10) — PASS iff, with the scipy import forced to fail, `model_version_summary`
    over ≥10 games still returns a dict whose `spearman_rho_range_vs_draw` is None (the field
    is PRESENT, degraded to None — not missing, not a raise).

    FAIL = the field is a float (scipy silently became a hard dependency), is absent (the key
    was dropped), or the whole telemetry read raises (the try/except no longer degrades). The
    import is blocked deterministically so the pin holds whether or not the running venv has
    scipy — the frozen contract is 'None until scipy is DECLARED', not 'None while it happens
    to be uninstalled'."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked for the DV-10 degradation pin")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    instr = _make_instr()
    lk = _lock()
    _ten_varied_games(instr, lk)
    summary = instr.model_version_summary(lk)

    assert summary["n"] == 10
    assert "spearman_rho_range_vs_draw" in summary, "the field must be present, not dropped"
    assert summary["spearman_rho_range_vs_draw"] is None, (
        "scipy is not a declared dependency — the lazy import must degrade rho to None"
    )


def test_g17b_rho_none_below_ten_games() -> None:
    """G-17 (n-guard arm) — below 10 games the correlation is not even attempted: rho is
    None. This is the OTHER degradation arm (`if n >= 10`), independent of scipy, so a pass
    here plus the scipy-blocked pass isolates the scipy branch as the thing under test."""
    instr = _make_instr()
    lk = _lock()
    for i in range(9):  # one short of the n >= 10 threshold
        _game_complete(instr, lk, mv_min=0, mv_max=i + 1, mv_distinct=i + 1,
                       winner_code=(0 if i % 2 == 0 else 1))
    summary = instr.model_version_summary(lk)
    assert summary["n"] == 9
    assert summary["spearman_rho_range_vs_draw"] is None


def test_g17c_scipy_is_not_a_declared_dependency() -> None:
    """G-17 (evidence arm) — records the current environment truth behind the pin: scipy is
    not importable in this venv, which is why the degradation is exercised natively. Guarded
    so a future venv that DOES ship scipy skips rather than falsely fails — the deterministic
    pin above (import blocked) is what actually enforces the contract."""
    import pytest
    if importlib.util.find_spec("scipy") is not None:
        pytest.skip("scipy present in this venv — the blocked-import pin (G-17) still holds")
    assert importlib.util.find_spec("scipy") is None
