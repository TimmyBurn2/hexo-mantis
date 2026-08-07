# >300 not exceeded; single-seam suite (the O1p Python leg + QA over ONE fixture family).
"""⊕ WP12-R Phase T (TARGET INTEGRITY) — O-1 exported-target parity, PYTHON LEG (O1p)
+ the quick-arm parity Python leg (QA). Written at T-2 ORACLE-WRITE, byte-frozen
through IMPL (R138 pattern: same fixture, both sides of the FFI — the Rust leg is
`crates/mantis-selfplay/tests/target_export_parity.rs`).

The fixture pairs are the POST-FIX export (full visit distribution, Σ == 1), minted
from the raw root-child visits of the production call sequence (mint probe preserved
at wp/WP12R/oracle_write_probes/). This leg drives the pairs through the REAL engine
buffer (`HexgBuffer.push_graph_position` → `sample_graph_batch`) and asserts the
Python-side `policy_target` carries the FULL ragged target — exactly the
downstream-drop the DESIGN_T §1 census exists to catch.

Constructions deliberately push fixture-recorded visits (NOT `record_position_graph`)
— out of mutation M-J's reach (PREREG_T §3 M-J green column; T-2 reconciliation).

PRE-FIX status at HEAD: GREEN (stage 3 is provably conserving — DESIGN_T §1.5); these
are carry pins, red-armed by M-A (fixture pairs vs a re-dropping export at the fix
commit) and M-F/M-L class mutations downstream.
Killers (PREREG_T §3): O1p — M-A; QA — M-A (mass leg).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mantis._engine import HexgBuffer

FIXDIR = Path(__file__).resolve().parents[1] / "fixtures" / "eval_selfplay_parity"
FIXTURES = ("target_parity_v1.json", "target_parity_dispersed_v1.json")

PAIR_TOL = 1e-6


def _positions(name: str):
    d = json.loads((FIXDIR / name).read_text())
    assert d["schema"] == 1 and d["encoding"] == "gnn_axis_v1"
    for i in range(int(d["n_positions"])):
        k = f"p{i}_"
        coords = d[k + "expected_coords"]
        mass = d[k + "expected_mass"]
        assert len(coords) == 2 * len(mass), f"{name} p{i}: pair arrays misaligned"
        yield {
            "id": d[k + "id"],
            "stones": [tuple(d[k + "stones"][j : j + 3]) for j in range(0, len(d[k + "stones"]), 3)],
            "current_player": int(d[k + "current_player"]),
            "moves_remaining": int(d[k + "moves_remaining"]),
            "ply_index": int(d[k + "ply_index"]),
            "n_legal": int(d[k + "n_legal"]),
            "pairs": {
                (coords[2 * j], coords[2 * j + 1]): float(mass[j]) for j in range(len(mass))
            },
        }


def _push(hb: HexgBuffer, pos: dict, is_full_search: bool = True) -> None:
    visits = [(q, r, m) for (q, r), m in sorted(pos["pairs"].items())]
    hb.push_graph_position(
        pos["stones"], visits, pos["current_player"], pos["moves_remaining"],
        pos["ply_index"], is_full_search, 0.0, True, 0,
    )


def _sampled_map(hb: HexgBuffer):
    wire, targets = hb.sample_graph_batch(1, False, 0.0)
    pt = np.asarray(targets.policy_target, dtype=np.float64)
    node_coords = np.asarray(wire.node_coords).reshape(-1, 2)
    gather = np.asarray(wire.legal_node_gather)
    offsets = np.asarray(wire.legal_offsets)
    assert offsets.tolist()[0] == 0 and offsets.tolist()[-1] == len(pt)
    got = {}
    for j, row in enumerate(gather):
        m = float(pt[j])
        if m > 0.0:
            q, r = (int(x) for x in node_coords[int(row)])
            got[(q, r)] = m
    return got, targets


def _assert_pairs(pos_id: str, got: dict, want: dict) -> None:
    assert set(got) == set(want), (
        f"{pos_id}: Python-side nonzero coord set != frozen fixture pairs "
        f"(missing {sorted(set(want) - set(got))[:5]}, extra {sorted(set(got) - set(want))[:5]})"
    )
    for c, wm in want.items():
        assert abs(got[c] - wm) <= PAIR_TOL, f"{pos_id}: mass at {c} = {got[c]} != {wm}"
    total = sum(got.values())
    assert abs(total - 1.0) <= 1e-4, f"{pos_id}: Python-side target mass {total} != 1"


# ── O1p: the Python side consumes the FULL ragged target ─────────────────────────────
def test_policy_target_round_trips_the_full_ragged_target() -> None:
    for pos in _positions("target_parity_v1.json"):
        hb = HexgBuffer(2, "gnn_axis_v1", 128)
        _push(hb, pos)
        got, _ = _sampled_map(hb)
        _assert_pairs(pos["id"], got, pos["pairs"])


def test_policy_target_round_trips_the_dispersed_fixture() -> None:
    d = json.loads((FIXDIR / "target_parity_dispersed_v1.json").read_text())
    # Band preconditions (flip-set rows 1-2 as amended at T-2, mirrored on this side of
    # the FFI): p0 in the 193-235 n_legal band; p2 in the >=5000-legal regime.
    assert 193 <= int(d["p0_n_legal"]) <= 235, "p0 left the 193-235 band"
    assert int(d["p2_n_legal"]) >= 5000, "p2 left the >=5000-legal regime"
    for pos in _positions("target_parity_dispersed_v1.json"):
        hb = HexgBuffer(2, "gnn_axis_v1", 128)
        _push(hb, pos)
        got, _ = _sampled_map(hb)
        _assert_pairs(pos["id"], got, pos["pairs"])


def test_fixture_pairs_are_unit_mass_and_fit_the_slot() -> None:
    """Fixture self-check: every position's frozen pairs are a distribution that fits
    MAX_VISITS (=128) — the M-D deliberate-green grounds (<= 57 = 50 + 8 - 1 cells)."""
    budget = 65536
    total_bytes = 0
    for name in FIXTURES:
        total_bytes += (FIXDIR / name).stat().st_size
        for pos in _positions(name):
            s = sum(pos["pairs"].values())
            assert abs(s - 1.0) <= 1e-4, f"{pos['id']}: fixture mass {s} != 1"
            assert 1 <= len(pos["pairs"]) <= 57, (
                f"{pos['id']}: {len(pos['pairs'])} cells outside (0, 57] — the corrected "
                "F-1(a) bound (sims 50 + batch 8 - 1) no longer covers the fixture"
            )
    assert total_bytes <= budget, f"fixture family {total_bytes} B exceeds the {budget} B budget"


# ── QA: the quick-arm row carries full mass; the flag rides independently ────────────
def test_quick_arm_row_carries_full_mass_and_the_flag_rides() -> None:
    pos = next(iter(_positions("target_parity_v1.json")))
    maps = {}
    for flag in (True, False):
        hb = HexgBuffer(2, "gnn_axis_v1", 128)
        _push(hb, pos, is_full_search=flag)
        got, targets = _sampled_map(hb)
        ifs = np.asarray(targets.is_full_search)
        assert ifs.shape == (1,) and bool(ifs[0]) is flag, (
            "is_full_search must ride the record verbatim (loss-gate flag, not an "
            "export input — PROVENANCE_T0 §2)"
        )
        maps[flag] = got
    assert maps[True] == maps[False], "the target must be identical across arms"
    assert abs(sum(maps[False].values()) - 1.0) <= 1e-4, (
        "an is_full_search=false row must still carry FULL mass (flip-set row 6)"
    )
