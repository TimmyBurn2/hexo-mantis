"""⊕ G-16 — instrumentation pure-function GOLDENS (WP-SP).

Written oracle-first against the dispatcher's old-side capture (#C3c, wp/WPSP/CAPTURE_LOG.md)
BEFORE any port code. RED at import until IMPL writes `mantis.selfplay.instrumentation`.

This file carries G-16 ONLY; G-01 … G-15 and G-17 are IMPL-written ports of the old suite.

Battery: 22 move histories = the old test file's vectors verbatim (six-in-a-row, the
7-collinear cap case, the two-cluster case, the stride-5 chains, the adjacent trio) +
degenerate cases (empty, single stone, duplicate cells, negative coords, per-axis lines) +
6 pseudo-random histories from `random.Random(20260723)` at n ∈ {5, 8, 16, 32, 64, 100}.
Each is scored at `cluster_threshold ∈ {5, 19}` × `winner_code ∈ {0, 1, 2}` ⇒ 154 recorded
scalar tuples. Every expected value is a captured number: nothing is recomputed here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mantis.selfplay.instrumentation import (
    _compute_colony_extension,
    _compute_longest_line,
    _compute_n_components,
    _compute_stride5_metrics,
    _split_players,
)

# Module-level load: parametrize ids are resolved at COLLECTION time.
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "selfplay"
_BATTERY: dict[str, Any] = json.loads(
    (_FIXTURES / "instrumentation" / "pure_function_battery.json").read_text()
)
_CASES = sorted(_BATTERY["cases"])
CAPTURE_SEED = _BATTERY["seed"]  # 20260723 — the random-history generator seed, recorded


def _history(case: dict[str, Any]) -> list[tuple[int, int]]:
    return [tuple(move) for move in case["move_history"]]


@pytest.mark.parametrize("label", _CASES)
def test_pure_function_goldens(pure_function_battery, label):
    """G-16 — PASS iff all four pure metric functions (plus `_split_players`, the ply→player
    rule they all share) reproduce the captured scalars EXACTLY for this history, at both
    cluster thresholds and all three winner codes.

    FAIL = a structural game metric drifted. These feed the `game_complete` event and the
    stride-5 / colony / connectivity investigation panels: a drifted metric does not crash
    anything, it just makes every downstream judgement about board structure wrong, silently.
    """
    case = pure_function_battery["cases"][label]
    history = _history(case)

    # ply → player: even plies are p1, odd are p2 (compound-turn aware) — pinned directly
    # rather than implied, because every other metric is computed off this split.
    p1, p2 = _split_players(history)
    assert [list(m) for m in p1] == case["split_players"]["p1"], f"{label}: p1 split"
    assert [list(m) for m in p2] == case["split_players"]["p2"], f"{label}: p2 split"

    stride5_run_max, row_max_density = _compute_stride5_metrics(history)
    assert int(stride5_run_max) == case["stride5_metrics"]["stride5_run_max"], (
        f"{label}: stride5_run_max"
    )
    assert int(row_max_density) == case["stride5_metrics"]["row_max_density"], (
        f"{label}: row_max_density"
    )

    count, total = _compute_colony_extension(history)
    assert int(count) == case["colony_extension"]["count"], f"{label}: colony count"
    assert int(total) == case["colony_extension"]["total"], f"{label}: colony total"

    for cluster_threshold in pure_function_battery["cluster_thresholds"]:
        for winner_code in pure_function_battery["winner_codes"]:
            key = f"ct{cluster_threshold}_wc{winner_code}"

            longest_line, fraction = _compute_longest_line(history, cluster_threshold,
                                                           winner_code)
            assert int(longest_line) == case["longest_line"][key]["longest_line"], (
                f"{label}[{key}]: longest_line"
            )
            assert float(fraction) == case["longest_line"][key]["fraction"], (
                f"{label}[{key}]: longest_line fraction"
            )

            n_components = _compute_n_components(history, cluster_threshold, winner_code)
            assert int(n_components) == case["n_components"][key], (
                f"{label}[{key}]: n_components"
            )


def test_battery_covers_the_captured_shape(pure_function_battery):
    """G-16 (coverage arm) — PASS iff the battery still has its 22 histories scored at both
    thresholds and all three winner codes. FAIL = the fixture was trimmed, which would let
    G-16 pass while covering less than the capture did."""
    assert len(pure_function_battery["cases"]) == 22
    assert pure_function_battery["cluster_thresholds"] == [5, 19]
    assert pure_function_battery["winner_codes"] == [0, 1, 2]
    assert pure_function_battery["seed"] == CAPTURE_SEED == 20260723

    # The six-in-a-row vector must still cap its longest line at 6 (old cap-at-6 rule).
    six = pure_function_battery["cases"]["six_in_a_row_p1"]
    seven = pure_function_battery["cases"]["seven_collinear_p1"]
    assert six["longest_line"]["ct5_wc1"]["longest_line"] == 6
    assert seven["longest_line"]["ct5_wc1"]["longest_line"] == 6, (
        "the 7-collinear case must still cap at 6 — the captured cap-at-6 rule"
    )
