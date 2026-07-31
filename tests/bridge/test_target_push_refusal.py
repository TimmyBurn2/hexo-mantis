"""⊕ WP12-R Phase T (TARGET INTEGRITY) — T-3 loop-2 addition (RED_TEAM_T F-RT-2/F-RT-3;
dispatcher freeze addition FA-3): the FFI-side non-distribution push refusal, PYTHON leg
(the Rust leg is the in-src bank in `crates/mantis-bridge/src/hexg.rs`).

`HexgBuffer.push_graph_position` is the SECOND public graph-record constructor (the
production Python route, pool_drain → pool_push) — R161 unconstructibility is
constructor-quantified, so it refuses non-distribution rows with the SAME typed
semantics as `record_position_graph`: the `TargetIntegrityError` Display (variant name
first) mapped to `ValueError`; `panic="unwind"` untouched. Census grounds for refusing
ALL non-distribution rows on this face: the graph push face has NO legitimate
zero/value-only form (the fast-game zero-policy sentinel is the DENSE recorder's,
runner/record.rs:67-78; graph quick-arm rows carry full mass — the frozen QA oracle
pins it). Duplicate-coord rows stay admitted (caught loud at sample-align); per-entry
NaN/negative refusals pre-date this loop in `push_record_impl`.

Killer: M-Q (bridge refusal removed → the refusal tests here and in the hexg.rs in-src
bank red; the per-entry negative case stays red under M-Q via push_record_impl — its
own pre-existing line). Recorded in PREREG_T AMENDMENT A-9.
"""
from __future__ import annotations

import math

import pytest

from mantis._engine import HexgBuffer

STONES = [(0, 0, 1), (1, 0, -1), (0, 1, 1)]


def _push(hb: HexgBuffer, visits: list[tuple[int, int, float]]) -> None:
    hb.push_graph_position(STONES, visits, 1, 2, 3, True, 0.0, True, 4)


def _refusal(visits: list[tuple[int, int, float]]) -> str:
    hb = HexgBuffer(8, "gnn_axis_v1")
    with pytest.raises(ValueError) as ei:
        _push(hb, visits)
    assert hb.size == 0, "a refused row must never reach the ring"
    return str(ei.value)


def test_half_mass_row_refused_naming_mass_not_unity() -> None:
    msg = _refusal([(2, 0, 0.5)])
    assert "MassNotUnity" in msg and "0.5" in msg, msg


def test_over_unity_row_refused() -> None:
    msg = _refusal([(2, 0, 1.5), (3, 0, 0.5)])
    assert "MassNotUnity" in msg, msg


def test_the_f_rt1_shipped_mass_row_refused_at_the_second_constructor() -> None:
    # The F-RT-1 compound face: pre-loop-2 a Σ=1.5 record (born of sign
    # cancellation) ALSO passed this push face — now unconstructible here too.
    msg = _refusal([(2, 0, 1.5)])
    assert "MassNotUnity" in msg and "1.5" in msg, msg


def test_all_zero_row_refused_naming_empty_target() -> None:
    # No value-only zero form exists on the GRAPH push face (dense-only sentinel).
    msg = _refusal([(2, 0, 0.0), (3, 0, 0.0)])
    assert "EmptyTarget" in msg, msg


def test_empty_visit_list_refused_naming_empty_target() -> None:
    msg = _refusal([])
    assert "EmptyTarget" in msg, msg


def test_nan_and_negative_entries_stay_refused_per_entry() -> None:
    # Pre-existing per-entry lines (push_record_impl) — pinned here so the M-Q
    # mutation cannot silently widen the face while the row checks are gone.
    hb = HexgBuffer(8, "gnn_axis_v1")
    with pytest.raises(ValueError):
        _push(hb, [(2, 0, math.nan), (3, 0, 1.0)])
    with pytest.raises(ValueError):
        _push(hb, [(2, 0, 1.5), (3, 0, -0.5)])
    assert hb.size == 0


def test_unity_and_within_tol_rows_admitted() -> None:
    # F-RT-3 admit-side pin, FFI parity: the ABSOLUTE 1e-4 window is the
    # intended width — exact unity and 1 + 5e-5 both ADMIT.
    hb = HexgBuffer(8, "gnn_axis_v1")
    _push(hb, [(2, 0, 1.0)])
    _push(hb, [(2, 0, 0.6), (3, 0, 0.4 + 5.0e-5)])
    assert hb.size == 2
